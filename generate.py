# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

#import os
#os.environ['TRITON_CACHE_DIR'] = './triton_cache'
#os.environ['TRITON_LOG_LEVEL'] = 'debug'
import json
import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch._dynamo.config
import torch._inductor.config

torch._dynamo.config.cache_size_limit = 64
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future


# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from sentencepiece import SentencePieceProcessor

from model import Transformer
from dcformer import DCFormerLlama
from modeling_muddformer import MUDDFormer, MUDDFormerConfig
from ddformer import DynamicDenseFormer, transformer_configs
from tp import maybe_init_dist


def multinomial_sample_one_no_sync(probs_sort): # Does multinomial sampling without a cuda synchronization
    q = torch.empty_like(probs_sort).exponential_(1)
    return torch.argmax(probs_sort / q, dim=-1, keepdim=True).to(dtype=torch.int)

def logits_to_probs(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    logits = logits / max(temperature, 1e-5)

    if top_k is not None:
        v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
        pivot = v.select(-1, -1).unsqueeze(-1)
        logits = torch.where(logits < pivot, -float("Inf"), logits)
    probs = torch.nn.functional.softmax(logits, dim=-1)
    return probs

def sample(logits, temperature: float = 1.0, top_k: Optional[int] = None):
    probs = logits_to_probs(logits[:, -1], temperature, top_k)
    idx_next = multinomial_sample_one_no_sync(probs)
    return idx_next, probs

def prefill(model: nn.Module, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> torch.Tensor:
    # input_pos: [B, S]
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)[0]

def decode_one_token(model: nn.Module, x: torch.Tensor, input_pos: torch.Tensor, **sampling_kwargs) -> Tuple[torch.Tensor, torch.Tensor]:
    # input_pos: [B, 1]
    assert input_pos.shape[-1] == 1
    logits = model(x, input_pos)
    return sample(logits, **sampling_kwargs)

def decode_n_tokens(model: nn.Module, cur_token: torch.Tensor, input_pos: torch.Tensor, num_new_tokens: int, callback=lambda _: _, **sampling_kwargs):
    new_tokens, new_probs = [], []
    for i in range(num_new_tokens):
        with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_mem_efficient=False, enable_math=True): # Actually better for Inductor to codegen attention here
            next_token, next_prob = decode_one_token(
                model, cur_token, input_pos, **sampling_kwargs
            )
        input_pos += 1
        new_tokens.append(next_token.view(-1, 1).clone())
        callback(new_tokens[-1])
        new_probs.append(next_prob.clone())
        cur_token = next_token.view(-1, 1)
    return new_tokens, new_probs


def model_forward(model, x, input_pos):
    return model(x, input_pos)

def speculative_decode(
    model: nn.Module,
    draft_model: nn.Module,
    cur_token: torch.Tensor,
    input_pos: int,
    speculate_k: int,
    **sampling_kwargs
) -> torch.Tensor:
    # draft model inference sequentially
    device = cur_token.device
    orig_input_pos = torch.tensor([input_pos], dtype=torch.int64, device=cur_token.device)
    draft_tokens, draft_probs = decode_n_tokens(draft_model, cur_token.view(1, -1), orig_input_pos.clone(), speculate_k, **sampling_kwargs)

    draft_tokens = torch.cat(draft_tokens)
    # parallel inference on target model using draft tokens
    target_logits = model_forward(
        model,
        torch.cat([cur_token.view(1), draft_tokens]).view(1, -1),
        torch.arange(input_pos, input_pos + speculate_k + 1, device=cur_token.device)
    )
    target_probs = logits_to_probs(target_logits[0], **sampling_kwargs)
    draft_probs = torch.stack(draft_probs)
    # q: target prob, p: draft prob
    # q >= p: always accept draft token
    # q < p: q/p prob to accept draft token
    p = draft_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    q = target_probs[torch.arange(0, speculate_k, device=device), draft_tokens]
    accept_draft_prob = torch.minimum(torch.ones(()), q[:speculate_k]/ p)
    rejected_locations = (torch.rand_like(accept_draft_prob) > accept_draft_prob).nonzero()

    if rejected_locations.shape[0] == 0: # All draft tokens have been accepted
        accept_length = speculate_k + 1
        last_token = multinomial_sample_one_no_sync(target_probs[-1])
        # fill last token into draft model
        model_forward(
            draft_model,
            draft_tokens[-1].view(1, -1),
            orig_input_pos + speculate_k,
        )
        return torch.cat([draft_tokens, last_token])
    else:
        accept_length = rejected_locations[0].item()
        p = draft_probs[accept_length]
        q = target_probs[accept_length]
        new = q - p
        new = torch.where(new > 0, new, 0.0)
        new = new / new.sum()
        next_token = multinomial_sample_one_no_sync(new)
        return torch.cat([draft_tokens[:accept_length], next_token])

@torch.no_grad()
def generate(
    model: nn.Module,
    prompt: torch.Tensor,
    max_new_tokens: int,
    *,
    interactive: bool,
    draft_model: nn.Module,
    speculate_k: Optional[int] = 8,
    callback = lambda x: x,
    max_batch_size=1,
    full_cache=False,
    precision=torch.float16,
    **sampling_kwargs
) -> torch.Tensor:
    """
    Takes a conditioning sequence (prompt) as input and continues to generate as many tokens as requested.
    """

    is_speculative = draft_model is not None
    # create an empty tensor of the expected final shape and fill in the current tokens
    T = prompt.size(1)
    T_new = T + max_new_tokens
    if full_cache:
        max_seq_length = model.config.block_size
    elif interactive:
        max_seq_length = 350
    else:
        max_seq_length = min(T_new, model.config.block_size)

    device, dtype = prompt.device, prompt.dtype
    max_seq_length = max_seq_length + speculate_k + 1 if is_speculative else max_seq_length
    with torch.device(device):
        model.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length, dtype=precision)
        if is_speculative and draft_model is not model:
            draft_model.setup_caches(max_batch_size=max_batch_size, max_seq_length=max_seq_length, dtype=precision)

    # create an empty tensor of the expected final shape and fill in the current tokens
    empty = torch.empty((max_batch_size, T_new), dtype=dtype, device=device)
    empty[:,:T] = prompt
    seq = empty
    input_pos = torch.arange(0, T, device=device)

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    
    next_token = prefill(model, prompt.view(max_batch_size, -1), input_pos, **sampling_kwargs)
    if is_speculative:
        prefill(draft_model, prompt.view(max_batch_size, -1), input_pos, **sampling_kwargs)

    torch.cuda.synchronize()
    prefill_dt = time.perf_counter() - t0
    t0 = time.perf_counter()

    seq[:, T] = next_token[:,0]

    input_pos = torch.tensor([T] , device=device, dtype=torch.int) # max_batch_size.unsqueeze(1)
    accept_counts = [0] * (speculate_k + 1)

    if is_speculative:
        input_pos = input_pos.item()  # for speculative decoding easier to keep on host
        while input_pos < T_new - 1:
            cur_token = next_token.view(())

            next_tokens = speculative_decode(
                model, draft_model, cur_token, input_pos, speculate_k, **sampling_kwargs
            )

            accept_counts[len(next_tokens) - 1] += 1
            num_added = min(T_new - input_pos - 1, len(next_tokens))
            seq[input_pos + 1 : input_pos + num_added + 1] = next_tokens[: num_added]
            for i in next_tokens[: num_added,]:
                callback(i)
            input_pos = input_pos + num_added
            next_token = next_tokens[-1]
    else:
        generated_tokens, _ = decode_n_tokens(model, next_token.view(max_batch_size, -1), input_pos, max_new_tokens - 1, callback=callback, **sampling_kwargs)
        seq[:,T + 1:] = torch.cat(generated_tokens, dim=-1)

    generate_stats = {
        'accept_counts': accept_counts
    }
    return seq, generate_stats, prefill_dt, t0

def encode_tokens(tokenizer, string, bos=True, device='cuda'):
    tokens = tokenizer.encode(string)
    if bos:
        tokens = [tokenizer.bos_id()] + tokens
    return torch.tensor(tokens, dtype=torch.int, device=device)

def _load_model(model_cls, model_name, checkpoint_path, device, precision, use_tp, model_size_str=None, window_size=None, window_type=None, query_wise=False):
    if model_size_str is None: 
        model_size_str = checkpoint_path.parent.name
    kwargs=dict(window_size=window_size, window_type=window_type, query_wise=query_wise)

    if model_cls is MUDDFormer: #
        with open('config_muddformer.json', 'r') as f:
            config = json.loads(f.read())
        size_config = transformer_configs[model_size_str] 
        config.update(size_config)
        config['n_local_heads'] = config['n_head']
        config['intermediate_size'] = None
        config['vocab_size'] = 32000
        # config['use_layer_cache'] = True
        if model_name == 'DenseFormer':
            config.update(dict(dense=True, dynamic_dense=False, sepln=False, dense_type='l'))
        elif model_name == 'DDFormer':
            config.update(dict(dense=True, dynamic_dense=True, sepln=False, dense_type='l'))
        config= MUDDFormerConfig(**config)
        model = MUDDFormer(config)
    elif hasattr(model_cls, 'from_name'):
        model = model_cls.from_name(model_size_str, **kwargs)
    if checkpoint_path is not None:
        if "int8" in str(checkpoint_path):
            print("Using int8 weight-only quantization!")
            from quantize import WeightOnlyInt8QuantHandler
            simple_quantizer = WeightOnlyInt8QuantHandler(model)
            model = simple_quantizer.convert_for_runtime()

        if "int4" in str(checkpoint_path):
            print("Using int4 quantization!")
            path_comps = checkpoint_path.name.split(".")
            assert path_comps[-2].startswith("g")
            groupsize = int(path_comps[-2][1:])
            from quantize import WeightOnlyInt4QuantHandler
            simple_quantizer = WeightOnlyInt4QuantHandler(model, groupsize)
            model = simple_quantizer.convert_for_runtime()

        checkpoint = torch.load(str(checkpoint_path), mmap=True, weights_only=True)
        model.load_state_dict(checkpoint, assign=True, strict=False)

    if use_tp:
        from tp import apply_tp
        print("Applying tensor parallel to model ...")
        apply_tp(model)

    model = model.to(device=device, dtype=precision)
    return model.eval()

B_INST, E_INST = "[INST]", "[/INST]"

def main(
    model_name: str = "DCFormerLlama",
    prompt: str = "Hello, my name is",
    interactive: bool = False,
    num_samples: int = 5,
    max_new_tokens: int = 100,
    top_k: int = 200,
    temperature: float = 0.8,
    checkpoint_path: Path = Path("checkpoints/meta-Transformer/Transformer-2-7b-chat-hf/model.pth"),
    compile: bool = True,
    compile_prefill: bool = False,
    profile: Optional[Path] = None,
    draft_checkpoint_path: Optional[Path] = None,
    speculate_k: int = 5,
    model_size_str: str = "7B",
    fake_prompt: bool = False,
    max_batch_size: int = 1,
    full_cache: bool = False,
    window_size: Optional[int] = None,
    window_type: Optional[str] = None,
    query_wise: bool = False,
) -> None:
    """Generates text samples based on a pre-trained Transformer model and tokenizer.
    """
    global print
    if not checkpoint_path.is_file():
        print('Use untrained model for non-exist model path.')

    tokenizer_path = checkpoint_path.parent / "tokenizer.model"
    assert tokenizer_path.is_file(), tokenizer_path

    rank = maybe_init_dist()
    use_tp = rank is not None
    if use_tp:
        if rank != 0:
            # only print on rank 0
            print = lambda *args, **kwargs: None

    device = 'cuda'
    #precision = torch.bfloat16
    precision = torch.float16
    is_speculative = draft_checkpoint_path is not None
    is_chat = "chat" in str(checkpoint_path)
    if window_type == 'default': window_type = None

    print("Loading model ...")
    t0 = time.time()
    if model_name == "DCFormerLlama":
        model_cls = DCFormerLlama
    elif model_name == "Llama":
        model_cls = Transformer
    elif model_name == "DynamicDenseFormer":
        model_cls = DynamicDenseFormer
    elif model_name in ["MUDDFormer", "DDFormer", "DenseFormer"]:
        model_cls = MUDDFormer

    checkpoint_path = None

    model = _load_model(model_cls, model_name, checkpoint_path, device, precision, use_tp, window_size=window_size,window_type=window_type, query_wise=query_wise, model_size_str=model_size_str)

    if is_speculative:
        draft_model = _load_model(model_cls, model_name,  draft_checkpoint_path, device, precision, use_tp)
    else:
        draft_model = None

    torch.cuda.synchronize()
    print(f"Time to load model: {time.time() - t0:.02f} seconds")

    torch.manual_seed(1234)
    tokenizer = SentencePieceProcessor(model_file=str(tokenizer_path))
    encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)
    if fake_prompt:
        encoded = torch.randint(10000,(max_batch_size, 1024), device=device)
        prompt = tokenizer.decode(encoded[0].tolist())
    prompt_length = encoded.size(1)

    model_size = sum([p.numel() * p.dtype.itemsize for p in itertools.chain(model.parameters(), model.buffers())])
    print('model_size:', model_size/1e9)
    if compile:
        if is_speculative and use_tp:
            torch._inductor.config.triton.cudagraph_trees = False # Bug with cudagraph trees in this case

        if is_speculative:
            global model_forward, logits_to_prob
            model_forward = torch.compile(model_forward, mode="reduce-overhead", fullgraph=True)

        global decode_one_token, prefill
        decode_one_token = torch.compile(decode_one_token, mode="reduce-overhead", fullgraph=True)

        # Uncomment to squeeze more perf out of prefill
        if args.compile_prefill:
            prefill = torch.compile(prefill, fullgraph=True, dynamic=True)


    aggregate_metrics = {
        'tokens_per_sec': [],
        'accept_counts': [],
    }
    start = -1 if compile else 0

    for i in range(start, num_samples):
        torch.cuda.synchronize()
        print(f'num_samples: {i}/{num_samples}')
        if i >= 0 and interactive:
            prompt = input("What is your prompt? ")
            if is_chat:
                prompt = f"{B_INST} {prompt.strip()} {E_INST}"
            encoded = encode_tokens(tokenizer, prompt, bos=True, device=device)

        if interactive and i >= 0:
            buffer = []
            period_id = tokenizer.encode('.')[0]
            done_generating = False
            def callback(x):
                nonlocal done_generating
                if done_generating:
                    return
                buffer.append(tokenizer.decode([period_id] + x.tolist())[1:])
                if x.item() == tokenizer.eos_id():
                    done_generating = True
                if len(buffer) == 4 or done_generating:
                    print(''.join(buffer), end='', flush=True)
                    buffer.clear()
                # print(, end='', flush=True)
        else:
            callback = lambda x : x
        import contextlib
        if (i != num_samples - 1 or not profile) or (use_tp and rank != 0):
            prof = contextlib.nullcontext()
        else:
            torch.profiler._utils._init_for_cuda_graphs()
            prof = torch.profiler.profile()
            print('profiling...')
        with prof:
            y, metrics, prefill_dt, t0 = generate(
                model,
                encoded,
                max_new_tokens,
                draft_model=draft_model,
                max_batch_size=max_batch_size,
                full_cache=full_cache,
                speculate_k=speculate_k,
                interactive=interactive,
                callback=callback,
                temperature=temperature,
                top_k=top_k,
                precision=precision,
            )
            aggregate_metrics['accept_counts'].append(metrics['accept_counts'])
        print('forward done')
        if i == -1:
            print(f"Compilation time: {time.perf_counter() - t0:.2f} seconds")
            continue
        if hasattr(prof, "export_chrome_trace"):
            prof.step()
            if use_tp:
                prof.export_chrome_trace(f"{profile}_rank_{rank}.json")
            else:
                prof.export_chrome_trace(f"{profile}.json")
        torch.cuda.synchronize()
        t = time.perf_counter() - t0 #- prefill_dt

        if not interactive:
            print(y[0].shape, y[0].max())
            print(tokenizer.decode(y[0].tolist()))
        else:
            print()
        tokens_generated = (y.size(1) - prompt_length - 1) * max_batch_size
        tokens_sec = tokens_generated / t
        aggregate_metrics['tokens_per_sec'].append(tokens_sec)
        print(f"Time for prefill inference {i + 1}: {prefill_dt:.05f} sec")
        print(f"Time for inference {i + 1}: {t:.02f} sec total, {tokens_sec:.02f} tokens/sec")
        print(f"Bandwidth achieved: {model_size * tokens_sec / 1e9:.02f} GB/s")
    print("==========")
    if is_speculative:
        counts_aggregated = [sum(i) for i in zip(*aggregate_metrics['accept_counts'])]
        acceptance_probs = [i/sum(counts_aggregated) for i in counts_aggregated]
        print(f"Acceptance probs: {acceptance_probs}")
        print(f"Mean Accepted: {sum([idx * i for idx, i in enumerate(counts_aggregated)])/sum(counts_aggregated)}")

    print(f"Average tokens/sec: {torch.mean(torch.tensor(aggregate_metrics['tokens_per_sec'])).item():.2f}")
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--model_name', type=str, default="DCFormerLlama", help='Model name: Llama or DCFormerLlama')
    parser.add_argument('--model_size', type=str, default="7B", help='Model size: 7B, 13B')
    parser.add_argument('--prompt', type=str, default="Hello, my name is", help='Input prompt.')
    parser.add_argument('--fake_prompt', action='store_true', help='Whether to evaluate inference performance using a fake prompt of 1024 tokens in length.')
    parser.add_argument('--interactive', action='store_true', help='Whether to launch in interactive mode')
    parser.add_argument('--num_samples', type=int, default=5, help='Number of samples.')
    parser.add_argument('--max_new_tokens', type=int, default=200, help='Maximum number of new tokens.')
    parser.add_argument('--max_batch_size', type=int, default=1, help='Maximum number of batch')
    parser.add_argument('--top_k', type=int, default=200, help='Top-k for sampling.')
    parser.add_argument('--temperature', type=float, default=0.8, help='Temperature for sampling.')
    parser.add_argument('--checkpoint_path', type=Path, default=Path("data/model.pth"), help='Model checkpoint path.')
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')
    parser.add_argument('--compile_prefill', action='store_true', help='Whether to compile the prefill (improves prefill perf, but higher compile times)')
    parser.add_argument('--profile', type=Path, default=None, help='Profile path.')
    parser.add_argument('--speculate_k', type=int, default=5, help='Speculative execution depth.')
    parser.add_argument('--draft_checkpoint_path', type=Path, default=None, help='Draft checkpoint path.')
    parser.add_argument('--full_cache', action='store_true', help='Whether to use kv cache with max sequence length.')
    parser.add_argument('--window_size', type=int, default=None, help='Window size of attention block in alternating layers')
    parser.add_argument('--window_type', type=str, default='default', help='Window type, such as LG,LGLL') # local or global
    parser.add_argument('--query_wise', action='store_true', help='Query-wise composition only')

    args = parser.parse_args()
    main(
        args.model_name, args.prompt, args.interactive, args.num_samples, args.max_new_tokens, args.top_k,
        args.temperature, args.checkpoint_path, args.compile, args.compile_prefill, args.profile, args.draft_checkpoint_path, args.speculate_k, args.model_size, args.fake_prompt, args.max_batch_size, args.full_cache,args.window_size, args.window_type,args.query_wise
    )
