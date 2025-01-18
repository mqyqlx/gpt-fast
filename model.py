# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from collections import namedtuple



def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)

@dataclass
class ModelArgs:
    block_size: int = 2048
    vocab_size: int = 32000
    n_layer: int = 32
    n_head: int = 32
    dim: int = 4096
    intermediate_size: int = None
    n_local_heads: int = -1
    head_dim: int = 64
    rope_base: float = 10000
    norm_eps: float = 1e-5
    window_size: Optional[int] = None 
    window_type: Optional[str] = None
    query_wise: bool = False

    def __post_init__(self):
        if self.n_local_heads == -1:
            self.n_local_heads = self.n_head
        if self.intermediate_size is None:
            hidden_dim = 4 * self.dim
            n_hidden = int(2 * hidden_dim / 3)
            self.intermediate_size = find_multiple(n_hidden, 256)
        self.head_dim = self.dim // self.n_head

    @classmethod
    def from_name(cls, name: str, **kwargs):
        if name in transformer_configs:
            kwargs.update(transformer_configs[name])
            return cls(**kwargs)
        # fuzzy search
        config = [config for config in transformer_configs if config in str(name).upper() or config in str(name)]
        assert len(config) == 1, name
        kwargs.update(transformer_configs[config[0]])
        return cls(**kwargs)


transformer_configs = {
    "CodeLlama-7b-Python-hf": dict(block_size=16384, vocab_size=32000, n_layer=32, dim = 4096, rope_base=1000000),
    "0p4B": dict(n_layer=24, n_head=16, dim=1024),
    "1p4Ba": dict(n_layer=24, n_head=16, dim=2048),
    "1p4Bb": dict(n_layer=24, n_head=32, dim=2048),
    "2p8B": dict(n_layer=32, n_head=32, dim=2560),
    "7B": dict(n_layer=32, n_head=32, dim=4096),
    "13B": dict(n_layer=40, n_head=40, dim=5120),
    "33B": dict(n_layer=60, n_head=52, dim=6656),
    "34B": dict(n_layer=48, n_head=64, dim=8192, vocab_size=32000, n_local_heads=8, intermediate_size=22016, rope_base=1000000), # CodeLlama-34B-Python-hf
    "70B": dict(n_layer=80, n_head=64, dim=8192, n_local_heads=8, intermediate_size=28672),
}

class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, window_size=None, dtype=torch.float16):
        super().__init__()
        self.window_size = window_size
        if window_size is None:
            self.seq_length = max_seq_length
        else:
            self.seq_length = min(window_size, max_seq_length)
        cache_shape = (max_batch_size, n_heads, self.seq_length, head_dim)
        self.register_buffer('k_cache', torch.zeros(cache_shape, dtype=dtype))
        self.register_buffer('v_cache', torch.zeros(cache_shape, dtype=dtype))

    def update(self, input_pos, k_val, v_val):
        # input_pos: [S], k_val: [B, H, S, D]
        assert input_pos.shape[0] == k_val.shape[2]
        B,N,S,D = v_val.shape

        k_out = self.k_cache
        v_out = self.v_cache
         
        if self.window_size is None:
            k_out[:, :, input_pos] = k_val
            v_out[:, :, input_pos] = v_val
        elif S==1:
            input_pos = input_pos % self.seq_length
            k_out[:, :, input_pos] = k_val
            v_out[:, :, input_pos] = v_val
        else:
            start = max(0,-self.seq_length)
            input_pos = input_pos[start:] % self.seq_length
            v_out[:, :, input_pos] = v_val[:,:,start:]
            k_out[:, :, input_pos] = k_val[:,:,start:]
        return k_out, v_out

class Transformer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config, lidx) for lidx in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.norm_noscale = RMSnormNoscale(epsilon=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.window_size = config.window_size
        self.max_batch_size = -1
        self.max_seq_length = -1

        self.dense_coeff = None

    def init_dense_coeff(self):
        self.dense_coeff = nn.ParameterList([nn.Parameter(data=torch.randn(1)) for lidx in range(self.config.n_layer)])
        # self.dense_coeff = nn.ParameterList([nn.Parameter(data=torch.tensor([0]* (lidx+1) + [1]).bfloat16()) for lidx in range(self.config.n_layer)])



    def setup_caches(self, max_batch_size, max_seq_length,dtype=torch.float16):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        for b in self.layers:
            b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, window_size=b.attention.window_size, dtype=dtype)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base)
        if self.window_size is None:
            self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))
        else:
            self.causal_mask = torch.stack([make_window_mask(max_seq_length, self.config.window_size), torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool))])

    def forward(self, idx: Tensor, input_pos: Optional[Tensor] = None, return_tensor=True) -> Tensor:
        assert self.freqs_cis is not None, "Caches must be initialized first"
        if input_pos is None:
            input_pos = torch.arange(idx.shape[-1], device=idx.device, dtype=torch.int)
        if self.window_size is None:
            mask = self.causal_mask[None, None, input_pos]
        else:
            mask = self.causal_mask[None, None,:,input_pos]
        freqs_cis = self.freqs_cis[input_pos]
        x = self.tok_embeddings(idx)
        hiddens = [x]
        attn_probs = []
        for i, layer in enumerate(self.layers):
            #layer_mask = mask if self.window_size is None else mask[:,:,i%2]
            if self.window_size is None:
                layer_mask = mask
            elif self.window_size is not None: 
                layer_mask = mask[:,:,1] if layer.attention.window_size is None else mask[:,:,0]
            x, probs = layer(x, input_pos, freqs_cis, layer_mask)
            attn_probs.append(probs)
            hiddens.append(x)
            if self.dense_coeff is not None: x = x * self.dense_coeff[i]  # x_out = (x_out -x_in) * c + x_in
            # if self.dense_coeff is not None: x = sum([self.norm_noscale(hiddens[j]) * self.dense_coeff[i][j] for j in range(i+2)]) # LBTD, L -> BTD
        x = self.norm(x)
        logits = self.output(x)
        if return_tensor:
            return logits
        else:
            CausalLMOutput = namedtuple("CausalLMOutput", ["logits", "hiddens", "attn_probs"])
            return CausalLMOutput(logits=logits, hiddens=hiddens, attn_probs=attn_probs)     

    @classmethod
    def from_name(cls, name: str, **kwargs):
        return cls(ModelArgs.from_name(name, **kwargs))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, lidx) -> None:
        super().__init__()
        self.lidx = lidx
        self.attention = Attention(config, lidx)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor) -> Tensor:
        attn_out = self.attention(self.attention_norm(x), freqs_cis, mask, input_pos)
        h = x + attn_out[0]
        out = h + self.feed_forward(self.ffn_norm(h))
        return out, attn_out[1]


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, lidx):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        self.wo = nn.Linear(config.dim, config.dim, bias=False)
        self.lidx = lidx
        self.kv_cache = None

        self.n_head = config.n_head
        self.head_dim = config.head_dim
        self.scale_factor = 1 / math.sqrt(self.head_dim)
        self.n_local_heads = config.n_local_heads
        self.dim = config.dim
        self.window_types = {
            "LG":[256, None],
            "LGLL":[256, None, 256, 256],
            "LGL6":[256, None, 256, 256, 256, 256, 256, 256],
        }
        # self.window_size = None #if self.lidx % 2 == 1 else config.window_size 
        if config.window_type is None:
            self.window_size = None if self.lidx % 2 == 1 else config.window_size 
        else:
            window_l = self.window_types[config.window_type]
            self.window_size = window_l[self.lidx % len(window_l)] 
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict:
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None) -> Tensor:
        bsz, seqlen, _ = x.shape

        kv_size = self.n_local_heads * self.head_dim
        q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

        q = q.view(bsz, seqlen, self.n_head, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)

        # print(q.shape, freqs_cis.shape)
        q = apply_rotary_emb(q, freqs_cis)
        k = apply_rotary_emb(k, freqs_cis)

        q, k, v = map(lambda x: x.transpose(1, 2), (q, k, v))


        if self.kv_cache is not None:
            if seqlen == 1:
                k, v = self.kv_cache.update(input_pos, k, v)
            else:
                _, _ = self.kv_cache.update(input_pos, k, v)

        # if self.window_size is None:
        #     k_mask = mask
        if seqlen == 1: # one-token generation
            k_mask = mask[:,:,:,:self.kv_cache.seq_length]
        else:# prefill
            k_mask = mask[:,:,:,:k.shape[-2]] 

        logits = q @ k.transpose(-2, -1) * self.scale_factor 
        min_value = torch.finfo(torch.float16).min
        logits = torch.where(k_mask, logits, min_value)
        probs = logits.softmax(-1)
        y = probs @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        y = self.wo(y)
        # probs = None
        return y, probs


class FeedForward(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.w1 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w3 = nn.Linear(config.dim, config.intermediate_size, bias=False)
        self.w2 = nn.Linear(config.intermediate_size, config.dim, bias=False)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def _norm(self, x):
        return x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)

    def forward(self, x: Tensor) -> Tensor:
        output = self._norm(x.float()).type_as(x)
        return output * self.weight


def make_window_mask(t, window_size):
    col_idx = torch.tile(torch.arange(t).unsqueeze(0), [t, 1])
    row_idx = torch.tile(torch.arange(t).unsqueeze(1), [1, t])
    bias_mask = (col_idx + window_size >= row_idx).tril().view(t, t)
    return bias_mask 


def precompute_freqs_cis(
    seq_len: int, n_elem: int, base: int = 10000
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=torch.float16)


class RMSnormNoscale(nn.Module):
    
    def __init__(self, epsilon=1e-6, dim=-1):
        super().__init__()
        self.dim = dim 
        self.epsilon = epsilon

    def forward(self, inputs):
        var = inputs.pow(2).mean(dim=self.dim, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        return normed_inputs 



def apply_rotary_emb(x: Tensor, freqs_cis: Tensor, mode='half') -> Tensor:
    if mode == 'half':
        xshaped = x.float().reshape(*x.shape[:-1], 2,-1).transpose(-1,-2) 
    elif mode == 'alternative':
        xshaped = x.float().reshape(*x.shape[:-1], -1, 2)
    freqs_cis = freqs_cis.view(-1, xshaped.size(1), 1, xshaped.size(3), 2)
    x_out2 = torch.stack(
        [
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 1] * freqs_cis[..., 0] + xshaped[..., 0] * freqs_cis[..., 1],
        ],
        -1,
    )

    x_out2 = x_out2.flatten(3)
    return x_out2.type_as(x)


def match_weight_llama(model, w):
    map_dict={'q_proj':'query', 'k_proj':'key', 'v_proj':'value','wo':'post', 'w1': 'ffn_layer1_gate', 'w3': 'ffn_layer1', 'w2': 'ffn_layer2',
              'weight': 'w'} # 'pre_proj': 'pre_proj', 'post_proj': 'post_proj'
    E, H, D = model.config.dim, model.config.n_head, model.config.head_dim
    N = model.config.vocab_size
    state_dict = {}
    for k, v in model.named_parameters():
        if k == 'tok_embeddings.weight':
            v = w['state.mdl_vars.params.lm.embedding_lookup.emb_var']#[:50257,:]
        elif k == 'norm.weight':
            v = w['state.mdl_vars.params.lm.final_ln.scale']
        elif k == 'output.weight':
            v = w['state.mdl_vars.params.lm.softmax.logits_ffn.linear.w'].T#[:50257,:]  # E,N -> N,E
        else:
            layer = int(k.split('.')[1])
            sub_layer, _layer = 0, layer
            if '.attention.' in k:
                _, _, _, ptype, wtype = k.split('.')
                if ptype == 'wqkv':
                    _q = torch.tensor(w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_{sub_layer}.self_attention.query.w'][_layer]).reshape(E,E) # EHD->EE
                    _k = torch.tensor(w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_{sub_layer}.self_attention.key.w'][_layer]).reshape(E,E) # EHD->EE
                    _v = torch.tensor(w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_{sub_layer}.self_attention.value.w'][_layer]).reshape(E,E) # EHD->EE
                    v = torch.cat([_q, _k, _v],dim=-1).T
                else: # o
                    v = w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_{sub_layer}.self_attention.{map_dict.get(ptype, ptype)}.{map_dict.get(wtype, wtype)}'][_layer].reshape(E,H*D)
            elif 'feed_forward' in k:
                ptype = k.split('.')[3] # w1, w3,w2
                v = w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_{sub_layer}.ff_layer.{map_dict[ptype]}.linear.w'][_layer].T
            elif 'ffn_norm' in k: # mlp layernorm
                v = w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_{sub_layer}.ff_layer.layer_norm.scale'][_layer]
            elif 'attention_norm' in k: # attention layernorm
                v = w[f'state.mdl_vars.params.lm.transformer.repeat.sub.x_layers_{sub_layer}.layer_norm.scale'][_layer]
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=False)
    return model