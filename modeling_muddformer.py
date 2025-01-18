# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from dataclasses import dataclass
from typing import Optional

import math
import torch
import json
import torch.nn as nn
from torch import Tensor
from torch.nn import functional as F
from einops import rearrange
from collections import namedtuple
from torch.utils.checkpoint import checkpoint
from typing import Any, Callable, Dict, Mapping, Optional, Sequence, Tuple, Union

try:
    from .configuration_muddformer import MUDDFormerConfig
except:
    from configuration_muddformer import MUDDFormerConfig

from transformers.modeling_utils import PreTrainedModel



def _rms(x):
  # Note: under pmap .mean() will produce a local mean, not across all hosts.
  return (x**2.).mean()**.5


def find_multiple(n: int, k: int) -> int:
    if n % k == 0:
        return n
    return n + k - (n % k)


class KVCache(nn.Module):
    def __init__(self, max_batch_size, max_seq_length, n_heads, head_dim, window_size=None, dtype=torch.bfloat16):
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
    def __init__(self, max_batch_size, num_layers, model_dim, dtype=torch.bfloat16):
        super().__init__()
        cache_shape = (num_layers+1, max_batch_size, 1, model_dim) # LBTD
        self.register_buffer('layer_cache', torch.zeros(cache_shape, dtype=dtype))
    
    def update(self, x, lidx):
        self.layer_cache[lidx] = x
        return self.layer_cache[:lidx+1]

#class MUDynamicDenseBlock(nn.Module):
#    def __init__(self, config: MUDDFormerConfig, lidx: int, last_layer=False) -> None:
#        super().__init__()
#        self.config = config
#        self.norm = RMSnormNoscale(epsilon=config.norm_eps)
#        l = lidx + 2
#        self.C = len(self.config.dense_type) if not last_layer else 1
#        l = l * self.C
#        self.w1 = nn.Linear(config.dim, l, bias=False)
#        self.act = nn.GELU() 
#        self.w2 = nn.Linear(l, l, bias=False)
#    
#    def forward(self, x: Tensor) -> Tensor:
#        x = self.norm(x) 
#        dw = self.w2(self.act(self.w1(x))) # BTD->BTL
#        dw = rearrange(dw, 'B T (C L) -> C B T L', C=self.C)
#        return dw

class MUDynamicDenseBlock(nn.Module):
    def __init__(self, config: MUDDFormerConfig, lidx: int, last_layer=False, expand_last=True, round64=True) -> None:
        super().__init__()
        self.config = config
        self.norm = RMSnormNoscale(epsilon=config.norm_eps)
        l = lidx + 2
        self.C = len(self.config.dense_type) if not last_layer else 1
        out_dim = l * self.C
        hid_dim = l * self.C 
        if last_layer and expand_last: hid_dim *= 4  
        if round64: hid_dim = (hid_dim// 64 +1) * 64 
        self.w1 = nn.Linear(config.dim, hid_dim, bias=False)
        self.act = nn.GELU() 
        self.w2 = nn.Linear(hid_dim, out_dim, bias=False)
    
    def forward(self, x: Tensor) -> Tensor:
        x = self.norm(x) 
        dw = self.w2(self.act(self.w1(x))) # BTD->BTL
        dw = rearrange(dw, 'B T (C L) -> C B T L', C=self.C)
        return dw
    
    @torch.compile(dynamic=True, fullgraph=True)
    def layer_mix(self, dw, hids, lidx)-> Tensor:
        x = tuple([sum(dw[cidx,:,:,j,None] * hids[j] for j in range(lidx+2)) for cidx in range(self.C)]) # CBTL, LBTD-> CBTD
        return x



class MUDDFormer(nn.Module):
    def __init__(self, config: MUDDFormerConfig) -> None:
        super().__init__()
        self.config = config

        self.use_gradient_checkpointing = config.use_gradient_checkpointing 
        self.is_training = config.is_training

        self.tok_embeddings = nn.Embedding(config.vocab_size, config.dim)
        self.layers = nn.ModuleList(TransformerBlock(config, lidx) for lidx in range(config.n_layer))
        self.norm = RMSNorm(config.dim, eps=config.norm_eps)
        self.output = nn.Linear(config.dim, config.vocab_size, bias=False)
        C = len(self.config.dense_type)
        self.dense_bs = nn.ParameterList([nn.Parameter(data=torch.randn(C if lidx != config.n_layer-1 else 1, lidx+2)) for lidx in range(config.n_layer)])

        self.dense_coeff = None# nn.ParameterList([nn.Parameter(data=torch.randn(C, lidx+2)) for lidx in range(config.n_layer)])

        self.layer_cache = None
        self.use_layer_cache = False if self.is_training else self.config.use_layer_cache
        # self.stack_hidden = False if self.is_training else self.config.stack_hidden
        self.stack_hidden = self.config.stack_hidden
        # if self.is_training:
        #     assert self.config.stack_hidden == False
        
        self.dynamic = self.config.dynamic_dense
        self.dense = self.config.dense
        if self.dynamic:
            self.dynamic_dense = nn.ModuleList([MUDynamicDenseBlock(config, lidx, last_layer=lidx==config.n_layer-1) for lidx in range(config.n_layer)])

        self.freqs_cis: Optional[Tensor] = None
        self.mask_cache: Optional[Tensor] = None
        self.window_size = config.window_size
        self.max_batch_size = -1
        self.max_seq_length = -1
    

    def tie_weights(self): # placeholder
        return 

    def init_dense_coeff(self):
        C = len(self.config.dense_type)
        # self.dense_coeff = nn.ParameterList([nn.Parameter(data=torch.randn(C, lidx+2)) for lidx in range(self.config.n_layer)])
        self.dense_coeff = nn.ParameterList([nn.Parameter(data=torch.randn(1)) for lidx in range(self.config.n_layer)])


    def setup_caches(self, max_batch_size, max_seq_length, dtype=torch.bfloat16):
        if self.max_seq_length >= max_seq_length and self.max_batch_size >= max_batch_size:
            return
        head_dim = self.config.dim // self.config.n_head
        max_seq_length = find_multiple(max_seq_length, 8)
        self.max_seq_length = max_seq_length
        self.max_batch_size = max_batch_size
        if not self.config.is_training:
            if self.use_layer_cache:
                self.layer_cache = LayerCache(max_batch_size, self.config.n_layer, self.config.dim, dtype=dtype)
            for b in self.layers:
                b.attention.kv_cache = KVCache(max_batch_size, max_seq_length, self.config.n_local_heads, head_dim, window_size=b.attention.window_size, dtype=dtype)

        self.freqs_cis = precompute_freqs_cis(self.config.block_size, self.config.dim // self.config.n_head, self.config.rope_base, dtype=dtype).to(self.tok_embeddings.weight.device)
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
            # mixed_hidden = [x]
        dws = []
        attn_probs = []
        for i, layer in enumerate(self.layers):
            #layer_mask = mask if self.window_size is None else mask[:,:,i%2]
            if self.window_size is None:
                layer_mask = mask
            elif self.window_size is not None: 
                layer_mask = mask[:,:,1] if layer.attention.window_size is None else mask[:,:,0]
            probs = None
            if self.use_gradient_checkpointing:
                x = checkpoint(layer, x, input_pos, freqs_cis, layer_mask)
                # x = layer(x, input_pos, freqs_cis, layer_mask)
            else:
                if return_tensor:
                    x = layer(x, input_pos, freqs_cis, layer_mask, return_probs=not return_tensor)
                else:
                    x, probs = layer(x, input_pos, freqs_cis, layer_mask, return_probs=not return_tensor)

            # if self.dense_coeff is not None: x = x * self.dense_coeff[i]
            #attn_probs.append(probs)
            # print(i, x.norm(dim=-1).mean())
            if use_layer_cache:
                _hidden = self.layer_cache.update(x, i+1) # LBTD
            else:
                hiddens.append(x)
                _hidden = torch.stack(hiddens) if self.stack_hidden else hiddens
            if self.dynamic and self.dense:
                dw = self.dynamic_dense[i](x) # BTD -> CBTL
                dw = dw + self.dense_bs[i][:,None,None,:] # CBTL
                # if self.dense_coeff is not None: dw = dw * self.dense_coeff[i][:,None,None,:]
                #dws.append(dw)
                if self.stack_hidden:
                    x = torch.einsum('LBTD, CBTL -> CBTD', _hidden, dw)
                else: 
                    x = self.dynamic_dense[i].layer_mix(dw, _hidden, i)
                    # x = tuple([sum(dw[cidx,:,:,j,None] * _hidden[j] for j in range(i+2)) for cidx in range(self.dynamic_dense[i].C)]) # BTL, LBTD-> BTD
                    # x = torch.stack(x)
                if self.config.dense_type == 'l':
                    x = x[0]
                # x = torch.einsum('LBTD, BTL -> BTD', self.layer_cache.layer_cache, dw + self.dense_bs[i][None,None]) 
                # x = sum([ (self.dense_bs[i][j] + dw[:,:,j][...,None])* hiddens[j] for j in range(i+1)]) # BTL, LBTD-> BTD
            elif self.dense:
                x = sum([ self.dense_bs[i][0,j] * _hidden[j] for j in range(i+1)]) # BTL, LBTD-> BTD
            #mixed_hidden.append(x)

        if self.config.dense_type == 'qkvm' and self.config.dense and self.config.dynamic_dense:
            x = x[0]
        x = self.norm(x)
        logits = self.output(x)
        if return_tensor: 
            return logits
        else:
            CausalLMOutput = namedtuple("CausalLMOutput", ["logits", "dw", "mixed_hidden", "hiddens", "attn_probs"])
            return CausalLMOutput(logits=logits, dw=dws, hiddens=hiddens, mixed_hidden=mixed_hidden, attn_probs=attn_probs)

  
class TransformerBlock(nn.Module):
    def __init__(self, config: MUDDFormerConfig, lidx) -> None:
        super().__init__()
        self.lidx = lidx
        self.config = config
        self.attention = Attention(config, lidx)
        self.feed_forward = FeedForward(config, lidx)
        self.ffn_norm = RMSNorm(config.dim, config.norm_eps)
        if self.config.sepln and self.lidx > 0 :
            self.attention_norms = torch.nn.ModuleList([RMSNorm(config.dim, config.norm_eps) for _ in range(3)])
        else:
            self.attention_norm = RMSNorm(config.dim, config.norm_eps)

    def forward(self, x: Union[Tuple[Tensor], Tensor], input_pos: Tensor, freqs_cis: Tensor, mask: Tensor, return_probs=False) -> Tensor:
        # print('lidx', self.lidx)
        if self.config.dense_type == 'l' or self.lidx == 0 or not self.config.dense:
            res = x
            normed_x = self.attention_norm(x)
        elif self.config.dense_type == 'qkvm':
            res = x[-1] # for mlp
            if self.config.stack_hidden or not self.config.sepln:
                normed_x = self.attention_norm(x[:3])
            else:
                # normed_x = tuple([self.attention_norm(_x) for _x in x[:3]])
                normed_x = tuple([norm_fn(_x) for norm_fn, _x in zip(self.attention_norms, x[:3])])
        attn_out = self.attention(normed_x, freqs_cis, mask, input_pos, return_probs=return_probs)
        #print('attn_out', self.lidx, _rms(attn_out))
        h = res + (attn_out[0] if return_probs else attn_out)
        out = h + self.feed_forward(self.ffn_norm(h))
        #out = h + self.feed_forward(self.ffn_norm(h), hid=h)
        #print('ffn output', _rms(out-h))
        #print('layer output', _rms(out-res))
        if return_probs:
            return out, attn_out[1]
        else:
            return out


class Attention(nn.Module):
    def __init__(self, config: MUDDFormerConfig, lidx):
        super().__init__()
        assert config.dim % config.n_head == 0

        total_head_dim = (config.n_head + 2 * config.n_local_heads) * config.head_dim
        # key, query, value projections for all heads, but in a batch
        self.config = config
        # if self.config.dense_type == 'l' or not self.config.dense:
        if len(self.config.dense_type) <= 1 or not self.config.dense:
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

        self.use_qk_norm = config.use_qk_norm
        if self.use_qk_norm:
            self.q_norm = RMSNorm(self.head_dim, config.norm_eps)
            self.k_norm = RMSNorm(self.head_dim, config.norm_eps)

        # self.window_size = config.window_size 
        if config.window_type is None:
            self.window_size = None if self.lidx % 2 == 1 else config.window_size 
        else:
            window_l = self.window_types[config.window_type]
            self.window_size = window_l[self.lidx % len(window_l)] 
        # print(self.lidx, self.window_size)
        self._register_load_state_dict_pre_hook(self.load_hook)

    def load_hook(self, state_dict, prefix, *args):
        if prefix + "wq.weight" in state_dict and (self.config.dense_type == 'l' or not self.config.dense):
            wq = state_dict.pop(prefix + "wq.weight")
            wk = state_dict.pop(prefix + "wk.weight")
            wv = state_dict.pop(prefix + "wv.weight")
            state_dict[prefix + "wqkv.weight"] = torch.cat([wq, wk, wv])

    def forward(self, x: Union[Tuple[Tensor], Tensor], freqs_cis: Tensor, mask: Tensor, input_pos: Optional[Tensor] = None, return_probs=False) -> Tensor:
        if self.lidx == 0 or self.config.dense_type == 'l' or not self.config.dense:
            bsz, seqlen, _ = x.shape
        else:
            if self.config.stack_hidden:
                C, bsz, seqlen, _ = x.shape
            else:
                C, (bsz, seqlen, _) = len(x), x[0].shape
        kv_size = self.n_local_heads * self.head_dim

        if not self.config.dense or len(self.config.dense_type)<=1 : # dynamic dense 
        # if self.config.dense_type == 'l' or not self.config.dense:
            q, k, v = self.wqkv(x).split([self.dim, kv_size, kv_size], dim=-1)

            q = q.view(bsz, seqlen, self.n_head, self.head_dim)
            k = k.view(bsz, seqlen, self.n_local_heads, self.head_dim)
            v = v.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        elif self.config.dense_type == 'qkvm':
            if self.lidx == 0:
                xq, xk, xv = x, x, x
            else:
                xq, xk, xv = x[0], x[1], x[2]
            q = self.wq(xq).view(bsz, seqlen, self.n_head, self.head_dim)
            k = self.wk(xk).view(bsz, seqlen, self.n_local_heads, self.head_dim)
            v = self.wv(xv).view(bsz, seqlen, self.n_local_heads, self.head_dim)


        #print('attn_q', self.lidx, _rms(q))
        #print('attn_k', self.lidx, _rms(k))
        #print('attn_v', self.lidx, _rms(v))

        if self.use_qk_norm:
            q, k = self.q_norm(q), self.k_norm(k)

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
        dtype = logits.dtype
        min_value = torch.finfo(torch.float32).min
        logits = logits.to(dtype=torch.float32)
        logits = torch.where(k_mask, logits, min_value)
        probs = logits.softmax(-1)
        probs = probs.to(dtype=dtype)
        y = probs @ v

        y = y.transpose(1, 2).contiguous().view(bsz, seqlen, self.dim)

        #print('attn_encoded', self.lidx, _rms(y))

        y = self.wo(y)
        probs = None
        if return_probs:
            return y, probs
        else:
            return y


class FeedForward(nn.Module):
    def __init__(self, config: MUDDFormerConfig, lidx, round128=True, scale_with_layer=True) -> None:
        super().__init__()
        hid_dim = config.intermediate_size
        if config.dynamic_dense and scale_with_layer: # use plus in dynamicdense and muddformer only
            hid_dim = hid_dim * (lidx/(config.n_layer -1) +0.5)
        if config.dynamic_dense and round128:
            hid_dim = round(hid_dim / 128) * 128
        self.w1 = nn.Linear(config.dim, hid_dim, bias=False)
        self.w3 = nn.Linear(config.dim, hid_dim, bias=False)
        self.w2 = nn.Linear(hid_dim, config.dim, bias=False)

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
    seq_len: int, n_elem: int, base: int = 10000, dtype=torch.bfloat16
) -> Tensor:
    freqs = 1.0 / (base ** (torch.arange(0, n_elem, 2)[: (n_elem // 2)].float() / n_elem))
    t = torch.arange(seq_len, device=freqs.device)
    freqs = torch.outer(t, freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)
    cache = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return cache.to(dtype=dtype)


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

def match_weight_muddformer(model, w, strict=False):
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
                elif ptype in ['q_norm', 'k_norm']:
                    v = w[f'state.mdl_vars.params.lm.transformer.x_layers_{lidx}.self_attention.{map_dict.get(ptype, ptype)}.scale']
            elif 'feed_forward' in k:
                ptype = k.split('.')[3] # w1, w3,w2
                v = w[f'state.mdl_vars.params.lm.transformer.x_layers_{lidx}.ff_layer.{map_dict[ptype]}.linear.w'].T
            elif 'ffn_norm' in k: # mlp layernorm
                v = w[f'state.mdl_vars.params.lm.transformer.x_layers_{lidx}.ff_layer.layer_norm.scale']
            elif 'attention_norm' in k: # attention layernorm
                if 'attention_norms' in k:
                    ln_idx = int(k.split('.')[3])
                    v = w[f'state.mdl_vars.params.lm.transformer.x_layers_{lidx}.layer_norms_{ln_idx}.scale']
                else:
                    v = w[f'state.mdl_vars.params.lm.transformer.x_layers_{lidx}.layer_norm.scale']
        state_dict[k] = torch.tensor(v)
    model.load_state_dict(state_dict, strict=strict)
    return model


def main():
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/pythia-6.9b")
    with open('config_muddformer.json', 'r') as f:
        config = json.loads(f.read())
    config['dim'] = 2560 
    config['intermediate_size'] = 6912 
    config['head_dim'] = 80 
    config['n_local_heads'] = 32 
    config['n_head'] = 32 
    config['n_layer'] = 32 
    config=MUDDFormerConfig(**config)
    print(config)
    model = MUDDFormer(config)
    #w = torch.load('/home/lishengping/mengqy/data/PileMUDDLlama3Bv5p_2000.torch.bin')
    w = torch.load('/home/lishengping/mengqy/data/PileMUDDLlama3BPlusOcdbtContv6e_13000.torch.bin')
    for k, v in w.items():
        print(k,v.shape)
    for k,v in model.named_parameters():
        print(k,v.shape, v.mean().item(), v.std().item())
    model = match_weight_muddformer(model, w, strict=True)

    print('load model to GPU')
    device = torch.device('cuda:0')
    _ = model.eval()
    dtype = torch.bfloat16
    _ = model.to(device, dtype=dtype)
    with torch.device(device):
        model.setup_caches(max_batch_size=1, max_seq_length=2048, dtype=dtype)
    _ = model.to(device)

    print('forward')
    prompt ='''
Attention I'm not good at English. If you find a mistake, let me know, please. ## 0. Abstract Interestingly, it's a very interesting phenomenon that global transformation by a the Moon's tide stress seems to be a trigger of occurrence for a disastrous earthquake (M>=5.5). It is found out that some statistically considered research papers geared to past earthquakes, there is no one which show about Lunar Age or Lunar Phase Angle clearly.The one of possibility reason is tidal phase angle on the Earth depends on the lunar age. However, many people can not calculate the tidal phase angle. This report's objective is that many people are able to imagine at time of high potential of earthquake occurrence intuitively by using visualization of **the position of the Moon and the Sun when earthquake was occurred**. ## 1. Introduction Schuster (1987) [1] is first article which discuss the influence of the tide from the moon. Some Studies, such as Tanaka (2012) [2] and Ide et al. (2016) [3], are studied the relationship between the earth tides and earthquakes with targeting specific seismic sources (groups). Tsuruoka and Ohtake (1995) [4] discussed the relationship with the global tides for earthquakes that occurred in the world. However, it is too difficult to calclulate tidal stress and to understand it for non-academia ordinary people. Therefore, I show some figures of relationship between earthquakes and lunar age in order to imagine the timing of earthquake occurrence. ## 2. Data ## 2.1 Data Source I selected the "[Significant Earthquakes, 1965-2016](https://www.kaggle.com/usgs/earthquake-database)" as the earthquake catalog to visualize target which is probably presented by USGS (United States Geological Survey) on Kaggle platform. Some earthquake catalogs (or past event lists) are opend on official sites of public organizations, such as USGS, Meteorological Institute of Japan, Disaster-Reduction Research of Japan. These catalogs are not useful because it is not simple list. Almost earthquakes are caused by plate tectonics at each location, regional characteristics exists. This article target for wide-area seismic activity on the whole earth, regional characteristics are not considered at all. Because it makes hard to discuss the actual mechanism of earthquake occurrence, this may be cause a feeling of strangeness to researcher.
'''
    #input_ids = tokenizer.encode(prompt, return_tensors='pt')
    #labels = input_ids[:,1:]
    #input_ids = input_ids[:,:-1] 

    from test_data import input_ids, labels
    input_ids = input_ids[:1,:]
    labels = labels[:1,:] 
    #from test_data import input_ids, labels
    with torch.no_grad():
        out = model.forward(input_ids.to(device), return_tensor=False)

    loss_func = torch.nn.CrossEntropyLoss(reduction='mean', ignore_index=0)
    logits = out.logits
    loss = loss_func(logits.reshape(-1, logits.shape[-1]), labels.long().reshape(-1).to(device))
    # loss on tpu: 2.16250
    print('loss',loss, logits.shape)

    return model



if __name__ == '__main__':
    main()

