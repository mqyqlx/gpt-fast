# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

import sys
sys.setrecursionlimit(10000) # https://discuss.pytorch.org/t/using-dataloader-recursionerror-maximum-recursion-depth-exceeded-while-calling-a-python-object/36947

import math
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from einops import rearrange
from collections import namedtuple
from torch.utils.checkpoint import checkpoint
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union



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
    dense_type: str = 'qkvm' # ['l', 'qkvm']; l: layer
    q_chunk_size: int = 128 # TODO
    use_dcmha: bool = False # TODO
    use_gradient_checkpointing: bool = False
    is_training: bool = False
    use_layer_cache: bool = True
    stack_hidden: bool = True



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
    "0p2Ba": dict(n_layer=4, n_head=16, dim=2048),
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

class LayerCache(nn.Module):
    def __init__(self, max_batch_size, num_layers, model_dim, dtype=torch.float16):
        super().__init__()
        cache_shape = (num_layers+1, max_batch_size, 1, model_dim) # LBTD
        self.register_buffer('layer_cache', torch.zeros(cache_shape, dtype=dtype))
    
    def update(self, x, lidx):
        self.layer_cache[lidx] = x
        return self.layer_cache[:lidx+1]

class DynamicDenseBlock(nn.Module):
    def __init__(self, config: ModelArgs, lidx: int) -> None:
        super().__init__()
        self.config = config
        # self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.norm = RMSnormNoscale(epsilon=config.norm_eps)
        l = lidx + 2
        C = len(self.config.dense_type)
        l = l * C
        self.w1 = nn.Linear(config.dim, l, bias=False)
        self.act = nn.GELU() 
        self.w2 = nn.Linear(l, l, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x) 
        dw = self.w2(self.act(self.w1(x))) # BTD->BTL
        C = len(self.config.dense_type)
        dw = rearrange(dw, 'B T (C L) -> C B T L', C=C)
        return dw

class DynamicDenseFormer(nn.Module):
    def __init__(self, config: ModelArgs) -> None:
        super().__init__()
        self.config = config

        self.use_gradient_checkpointing = config.use_gradient_checkpointing 
        self.is_training = config.is_training

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config, lidx) for lidx in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        C = len(self.config.dense_type)
        self.dense_bs = nn.ParameterList([nn.Parameter(data=torch.randn(C, lidx+2)) for lidx in range(config.n_layer)])

        self.dense_coeff = None# nn.ParameterList([nn.Parameter(data=torch.randn(C, lidx+2)) for lidx in range(config.n_layer)])

        self.layer_cache = None
        self.use_layer_cache = False if self.is_training else self.config.use_layer_cache
        # self.stack_hidden = False if self.is_training else self.config.stack_hidden
        self.stack_hidden = self.config.stack_hidden
        # if self.is_training:
        #     assert self.config.stack_hidden == False
        
        self.dynamic = True
        if self.dynamic:
            self.dynamic_dense = nn.ModuleList([DynamicDenseBlock(config, lidx) for lidx in range(config.n_layer)])

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.window_size = config.window_size
        self.max_batch_size = -1
        self.max_seq_length = -1
    
    def init_dense_coeff(self):
        C = len(self.config.dense_type)
        # self.dense_coeff = nn.ParameterList([nn.Parameter(data=torch.randn(C, lidx+2)) for lidx in range(self.config.n_layer)])
        self.dense_coeff = nn.ParameterList([nn.Parameter(data=torch.randn(1)) for lidx in range(self.config.n_layer)])


    def setup_caches(self, max_batch_size, max_seq_length, dtype=torch.float16):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        if not self.config.is_training:
            if self.use_layer_cache:
                self.layer_cache = LayerCache(max_batch_size, self.config.n_layer, self.config.dim)
            for b in self.layers:
                b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, window_size=b.attention.window_size, dtype=dtype)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base).to(self.tok_embeddings.weight.device)
        if self.window_size is None:
            self.causal_mask = torch.tril(torch.ones(self.max_seq_length, self.max_seq_length, dtype=torch.bool, device=self.tok_embeddings.weight.device))
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
        _, seqlen, _ = x.shape
        use_layer_cache = self.use_layer_cache and seqlen == 1
        if use_layer_cache:
            self.layer_cache.update(x, 0)
        else:
            hiddens = [x]
            mixed_hidden = [x]
        dws = []
        attn_probs = []
        for i, layer in enumerate(self.layers):
            #layer_mask = mask if self.window_size is None else mask[:,:,i%2]
            if self.window_size is None:
                layer_mask = mask
            elif self.window_size is not None: 
                layer_mask = mask[:,:,1] if layer.attention.window_size is None else mask[:,:,0]

            if self.use_gradient_checkpointing:
                if i == 0:
                    xq, xk, xv, xm = x, x, x, x
                else:
                    xq, xk, xv, xm = x
                x = checkpoint(layer, xq, xk, xv, xm, input_pos, freqs_cis, layer_mask) # x: CBTD c=4
                # x = layer(xq, xk, xv, xm, input_pos, freqs_cis, layer_mask)
                # x = layer(x, input_pos, freqs_cis, layer_mask)
                probs = None
            else:
                x, probs = layer(x, input_pos, freqs_cis, layer_mask) # x: BTD

            if self.dense_coeff is not None: x = x * self.dense_coeff[i]
            attn_probs.append(probs)
            if use_layer_cache:
                _hidden = self.layer_cache.update(x, i+1) # LBTD
            else:
                hiddens.append(x)
            if self.dynamic:
                dw = self.dynamic_dense[i](x) # BTD -> CBTL
                if not use_layer_cache and self.stack_hidden:
                    _hidden = torch.stack(hiddens)
                dw = dw + self.dense_bs[i][:,None,None,:] # CBTL
                # if self.dense_coeff is not None: dw = dw * self.dense_coeff[i][:,None,None,:]
                dws.append(dw)
                if self.stack_hidden:
                    x = torch.einsum('LBTD, CBTL -> CBTD', _hidden, dw)
                else: 
                    # x = sum([torch.einsum('LBTD, BTL -> BTD', _hidden, dw[cidx] + self.dense_bs[i][cidx,None,None,:]) for cidx in range(4)])
                    x = tuple([sum(dw[cidx,:,:,j,None] * hiddens[j] for j in range(i+1)) for cidx in range(4)]) # BTL, LBTD-> BTD
                    # x = torch.stack(x)
                if self.config.dense_type == 'l':
                    x = x[0]
                # x = torch.einsum('LBTD, BTL -> BTD', self.layer_cache.layer_cache, dw + self.dense_bs[i][None,None]) 
                # x = sum([ (self.dense_bs[i][j] + dw[:,:,j][...,None])* hiddens[j] for j in range(i+1)]) # BTL, LBTD-> BTD
            else:
                x = sum([ self.dense_bs[i][j] * hiddens[j] for j in range(i+1)]) # BTL, LBTD-> BTD
            mixed_hidden.append(x)

        if self.config.dense_type == 'qkvm':
            x = x[1]
        x = self.norm(x)
        logits = self.output(x)
        if return_tensor: 
            return logits
        else:
            CausalLMOutput = namedtuple("CausalLMOutput", ["logits", "dw", "mixed_hidden", "hiddens", "attn_probs"])
            return CausalLMOutput(logits=logits, dw=dws, hiddens=hiddens, mixed_hidden=mixed_hidden, attn_probs=attn_probs)

    @classmethod
    def from_name(cls, name: str, **kwargs):
        return cls(ModelArgs.from_name(name, **kwargs))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelArgs, lidx) -> None:
        super().__init__()
        self.lidx = lidx
        self.config = config
        self.attention = Attention(config, lidx)
        self.feed_forward = FeedForward(config)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        # self.attention_norm = RMSNorm(config.dim, config.norm_eps)
        self.attention_norms = torch.nn.ModuleList([RMSNorm(config.dim, config.norm_eps) for _ in range(3)])

    def forward(self, xq:Tensor, xk:Tensor, xv:Tensor, xm:Tensor, input_pos: Tensor, freqs_cis: Tensor, mask: Tensor, return_probs=False) -> Tensor:
        if self.config.dense_type == 'l' or self.lidx == 0:
            res = xq
            normed_x = self.attention_norms[0](xq)
        elif self.config.dense_type == 'qkvm':
            res = xm # for mlp
            # if self.config.stack_hidden:
            #     normed_x = self.attention_norm(x[:3])
            # else:
            xq, xk, xv = [norm_fn(_x) for norm_fn, _x in zip(self.attention_norms, [xq, xk, xv])]
        attn_out = self.attention(xq, xk, xv, freqs_cis, mask, input_pos)
        h = res + (attn_out[0] if return_probs else attn_out)
        out = h + self.feed_forward(self.ffn_norm(h))
        if return_probs:
            return out, attn_out[1]
        else:
            return out


class Attention(nn.Module):
    def __init__(self, config: ModelArgs, lidx):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.config = config
        if self.config.dense_type == 'l':
            self.wqkv = nn.Linear(config.dim, total_head_dim, bias=False)
        elif self.config.dense_type == 'qkvm':
            self.wq = nn.Linear(config.dim, config.n_head * config.head_dim, bias=False)
            self.wk = nn.Linear(config.dim, config.n_local_heads * config.head_dim, bias=False)
            self.wv = nn.Linear(config.dim, config.n_local_heads * config.head_dim, bias=False)

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
        # self.window_size = config.window_size 
        if config.window_type is None:
            self.window_size = None if self.lidx % 2 == 1 else config.window_size 
        else:
            window_l = self.window_types[config.window_type]
            self.window_size = window_l[self.lidx % len(window_l)] 
        # print(self.lidx, self.window_size)
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict and self.config.dense_type == 'l':
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, xq: Tensor, xk: Tensor, xv: Tensor, freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None, return_probs=False) -> Tensor:
        if self.lidx == 0 or self.config.dense_type == 'l':
            bsz, seqlen, _ = xq.shape
        else:
            # if self.config.stack_hidden:
            #     C, bsz, seqlen, _ = x.shape
            # else:
            C = 3
            bsz, seqlen, _ = xq.shape
        kv_size = self.n_local_heads * self.head_dim

        if self.config.dense_type == 'l':
            q, k, v = self.wqkv(xq).split([self.dim, kv_size, kv_size], dim=-1)

            q = q.view(bsz, seqlen, self.n_head, self.head_dim)
            k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        elif self.config.dense_type == 'qkvm':
            # if self.lidx == 0:
            #     xq, xk, xv = x, x, x
            # else:
            #     xq, xk, xv = x[0], x[1], x[2]
            q = self.wq(xq).view(bsz, seqlen, self.n_head, self.head_dim)
            k = self.wk(xk).view(bsz, seqlen, self.n_local_heads, self.head_dim)
            v = self.wv(xv).view(bsz, seqlen, self.n_local_heads, self.head_dim)
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
        #     print(k_mask.shape)
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
        probs = None
        if return_probs:
            return y, probs
        else:
            return y


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

class RMSnormNoscale(nn.Module):
    
    def __init__(self, epsilon=1e-6, dim=-1):
        super().__init__()
        self.dim = dim 
        self.epsilon = epsilon

    def forward(self, inputs):
        var = inputs.pow(2).mean(dim=self.dim, keepdim=True)
        normed_inputs = inputs * torch.rsqrt(var + self.epsilon)
        return normed_inputs 


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

# tok_embeddings.weight torch.Size([50432, 2048])
# layers.0.attention.wq.weight torch.Size([2048, 2048])
# layers.0.attention.wk.weight torch.Size([2048, 2048])
# layers.0.attention.wv.weight torch.Size([2048, 2048])
# layers.0.attention.wo.weight torch.Size([2048, 2048])
# layers.0.feed_forward.w1.weight torch.Size([5632, 2048])
# layers.0.feed_forward.w3.weight torch.Size([5632, 2048])
# layers.0.feed_forward.w2.weight torch.Size([2048, 5632])
# layers.0.ffn_norm.weight torch.Size([2048])
# layers.0.attention_norm.weight torch.Size([2048])
# norm.weight torch.Size([2048])
# output.weight torch.Size([50432, 2048])
# dense_bs.0 torch.Size([4, 2])
# dynamic_dense.0.w1.weight torch.Size([8, 2048])
# dynamic_dense.0.w2.weight torch.Size([8, 8])

def match_weight_dynamicdense(model, w):
    map_dict={'wq':'query', 'wk':'key', 'wv':'value', 'wo':'post', 'w1': 'ffn_layer1_gate', 'w3': 'ffn_layer1', 'w2': 'ffn_layer2',
              'weight': 'w'}
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
        elif 'dense_bs' in k: # static dense w
            lidx = int(k.split('.')[-1])
            v = w[f'state.mdl_vars.params.lm.transformer.dense_conn_{lidx}']
        elif 'dynamic_dense' in k:
            lidx = int(k.split('.')[1])
            widx = int(k.split('.')[2][-1]) # 1 or 2 in w1, w2
            v = w[f'state.mdl_vars.params.lm.transformer.x_layers_{lidx}.dynamic_dense_conn{widx}_{lidx}'].T
        else:
            assert 'layers' in k
            lidx = int(k.split('.')[1])
            if '.attention.' in k:
                _, _, _, ptype, wtype = k.split('.')
                if ptype in ['wq', 'wk', 'wv', 'wo']:
                    v = w[f'state.mdl_vars.params.lm.transformer.x_layers_{lidx}.self_attention.{map_dict.get(ptype, ptype)}.{map_dict.get(wtype, wtype)}'].reshape(E,E)
                    if ptype != 'wo':
                        v = v.T
            elif 'feed_forward' in k:
                ptype = k.split('.')[3] # w1, w3,w2
                v = w[f'state.mdl_vars.params.lm.transformer.x_layers_{lidx}.ff_layer.{map_dict[ptype]}.linear.w'].T
            elif 'ffn_norm' in k: # mlp layernorm
                v = w[f'state.mdl_vars.params.lm.transformer.x_layers_{lidx}.ff_layer.layer_norm.scale']
            elif 'attention_norm' in k: # attention layernorm
                v = w[f'state.mdl_vars.params.lm.transformer.x_layers_{lidx}.layer_norm.scale']
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=False)
    return model