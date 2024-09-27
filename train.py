import itertools
import sys
import time
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
import torch._dynamo.config
import torch._inductor.config
from torch.profiler import profile, record_function, ProfilerActivity

torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.triton.unique_kernel_names = True
torch._inductor.config.joint_graph_constant_folding = False # avoid OOM bug for DCFormer-6.9B
#torch._inductor.config.fx_graph_cache = True # Experimental feature to reduce compilation times, will be on by default in future

# support running without installing as a package
wd = Path(__file__).parent.parent.resolve()
sys.path.append(str(wd))

from sentencepiece import SentencePieceProcessor

from dcformer import DCFormerLlama,ModelArgs 
from tp import maybe_init_dist


def train(mod, data):
    input, label = data
    pred = mod(input) # BTV
    loss = torch.nn.CrossEntropyLoss()(pred.view((-1,pred.shape[-1])), label.reshape(-1))
    loss.backward()

def timed(fn):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    result = fn()
    end.record()
    torch.cuda.synchronize()
    return result, start.elapsed_time(end) / 1000

def main(model_size='7B', use_dcmha=False,q_chunk_size=2048, batch_size=1, compile=True, dtype=torch.float16, max_seq_length=2048):
    N_ITERS = 100
    VS = 50257
    device = torch.device('cuda:0')
    
    if model_size == '2p8B':
        n_layer,n_head,dim = 32,32,2560
    elif model_size == '7B':
        n_layer,n_head,dim = 32,32,4096
    config = ModelArgs(n_layer=n_layer,n_head=n_head,dim=dim,block_size=max_seq_length, q_chunk_size=q_chunk_size,use_dcmha=use_dcmha,vocab_size=VS, use_gradient_checkpointing=True, is_training=True) # 6.7B
    print('config', config)

    model = DCFormerLlama(config)
    model = model.to(device=device, dtype=dtype)
    model.setup_caches(max_batch_size=batch_size, max_seq_length=max_seq_length)

    if compile:
        train_opt = torch.compile(train)
    else:
        train_opt = train

    print('start training')
    compile_times = []
    inp = torch.randint(VS, (batch_size,max_seq_length+1)).to(device)
    inp = [inp[:,:-1], inp[:, 1:]]
    for i in range(N_ITERS):
        if i == 50:
            torch.cuda.synchronize()
            t0 = time.perf_counter()
        _, compile_time = timed(lambda: train_opt(model, inp))
        compile_times.append(compile_time)
        print(f"compile train time {i}: {compile_time}")

    torch.cuda.synchronize()
    dt = time.perf_counter()-t0
    print('time for 50 iterations:', dt)
    print("~" * 10)
    compile_med = np.median(compile_times)
    step_time = 50/dt
    
    print('compile mean', compile_med, np.array(compile_times)[50:].mean())
    print(f'training speed: {step_time :.4f} steps/sec')


if  __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Your CLI description.')

    parser.add_argument('--model_size', type=str, default="7B", help='Model size: 2p8B, 7B')
    parser.add_argument('--use_dcmha', action='store_true', help='Whether to use dcmha')
    parser.add_argument('--query_chunk_size', type=int, default=2048, help='query chunk size used in attention calculation')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--compile', action='store_true', help='Whether to compile the model.')

    args = parser.parse_args()
    print('args', args)
    main(args.model_size, args.use_dcmha, args.query_chunk_size, args.batch_size, args.compile)
