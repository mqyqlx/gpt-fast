
from ctypes import c_void_p, c_long
import torch
import math
import random
import os
import tempfile
from math import inf, nan
from torch._inductor.hooks import run_intermediate_hooks
from torch._inductor.utils import maybe_profile

from torch import empty_strided, device
from torch._inductor.codecache import AsyncCompile
from torch._inductor.select_algorithm import extern_kernels
from torch.profiler import profile, record_function, ProfilerActivity


aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_mengqy/fx/cfxbaxpd6x6dp4pr4yj2mqmpkc4swyh5sec5fhsl74se7veahcol.py
# Source Nodes: [l__mod___tok_embeddings], Original ATen: [aten.embedding]
# l__mod___tok_embeddings => embedding
triton_poi_fused_embedding_0 = async_compile.triton('triton_poi_fused_embedding_0', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_0', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]})
@triton.jit
def triton_poi_fused_embedding_0(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 1024) % 2048
    x2 = (xindex // 2097152)
    x0 = xindex % 1024
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x1 + (2049*x2)), None, eviction_policy='evict_last')
    tmp1 = tl.where(tmp0 < 0, tmp0 + 50257, tmp0)
    tl.device_assert((0 <= tmp1) & (tmp1 < 50257), "index out of bounds: 0 <= tmp1 < 50257")
    tmp2 = tl.load(in_ptr1 + (x0 + (1024*tmp1)), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp2, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_mengqy/3z/c3zrn5vg5bd7nw3nprwlzie6kgnfcqnxthvntjwxj2glxcoxvhec.py
# Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
# stack => cat
# stack_1 => cat_1
triton_poi_fused_stack_1 = async_compile.triton('triton_poi_fused_stack_1', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_1', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]})
@triton.jit
def triton_poi_fused_stack_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 3145728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 32
    x4 = (xindex // 32)
    x2 = (xindex // 512) % 2048
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x4)), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + ((2*x0) + (64*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (32 + x0 + (64*x4)), None).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (1 + (2*x0) + (64*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr2 + (x0 + (64*x4)), None).to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (32 + x0 + (64*x4)), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp1 * tmp3
    tmp6 = tmp5.to(tl.float32)
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp6 * tmp8
    tmp10 = tmp4 - tmp9
    tmp11 = tmp6 * tmp3
    tmp12 = tmp1 * tmp8
    tmp13 = tmp11 + tmp12
    tmp15 = tmp14.to(tl.float32)
    tmp16 = tmp15 * tmp3
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp8
    tmp20 = tmp16 - tmp19
    tmp21 = tmp18 * tmp3
    tmp22 = tmp15 * tmp8
    tmp23 = tmp21 + tmp22
    tl.store(out_ptr0 + (2*x5), tmp10, None)
    tl.store(out_ptr1 + (2*x5), tmp13, None)
    tl.store(out_ptr2 + (2*x5), tmp20, None)
    tl.store(out_ptr3 + (2*x5), tmp23, None)
''')


# kernel path: /tmp/torchinductor_mengqy/3w/c3w45fwangida7ybcckdclp5sbjqkbqowfjhkhpskne5aq4kkabn.py
# Source Nodes: [getitem, getitem_18], Original ATen: [aten.index, aten.slice]
# getitem => index
# getitem_18 => slice_1, slice_2, slice_3
triton_poi_fused_index_slice_2 = async_compile.triton('triton_poi_fused_index_slice_2', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*i1', 1: '*i1', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_slice_2', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_index_slice_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4194304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/bf/cbfmcrq2pug25kfmtsuyqocukxa4zwufzdckytddnwstv6kv767x.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone
triton_poi_fused_clone_3 = async_compile.triton('triton_poi_fused_clone_3', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_clone_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 2048
    x2 = (xindex // 131072) % 16
    x3 = (xindex // 2097152)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1) + (2097152*x3)), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp1, None)
''')


# kernel path: /tmp/torchinductor_mengqy/lx/clxafc5klxfgh5ozbxxbydg26ycpu4oywo72xlkbtpggo6sfr3ff.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_1
triton_poi_fused_clone_4 = async_compile.triton('triton_poi_fused_clone_4', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[4096, 2048], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]})
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 3072
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 1024
    y1 = (yindex // 1024)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x2) + (2097152*y1)), None, eviction_policy='evict_last')
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x2 + (2048*y3)), tmp1, None)
''')


# kernel path: /tmp/torchinductor_mengqy/yp/cypzyephpx4vyoo4y2vy7rjk725qyprlabgldpynphvoehlmyrak.py
# Source Nodes: [where], Original ATen: [aten.scalar_tensor]
# where => scalar_tensor
triton_poi_fused_scalar_tensor_5 = async_compile.triton('triton_poi_fused_scalar_tensor_5', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[1], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_scalar_tensor_5', 'configs': [instance_descriptor(divisible_by_16=(0,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]})
@triton.jit
def triton_poi_fused_scalar_tensor_5(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    tmp0 = -65504.0
    tl.store(out_ptr0 + (tl.full([XBLOCK], 0, tl.int32)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/nm/cnmdqouabgsysnznlsxslsgv2itbevnvbqglcomrt6uiymv5jdey.py
# Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
# mul_11 => mul_11
# softmax => amax, convert_element_type_6, convert_element_type_7, div, exp, sub_2, sum_1
# where => where
triton_red_fused__softmax_mul_where_6 = async_compile.triton('triton_red_fused__softmax_mul_where_6', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 2048],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_mul_where_6', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_red_fused__softmax_mul_where_6(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2048
    x3 = xindex
    tmp4 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    _tmp9 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (r2 + (2048*x0)), rmask, eviction_policy='evict_last')
        tmp1 = tl.load(in_ptr1 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = 0.125
        tmp3 = tmp1 * tmp2
        tmp6 = tl.where(tmp0, tmp3, tmp5)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = triton_helpers.maximum(_tmp9, tmp8)
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
    tmp9 = triton_helpers.max2(_tmp9, 1)[:, None]
    tmp15 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
    _tmp22 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp11 = tl.load(in_ptr0 + (r2 + (2048*x0)), rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr1 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp13 = 0.125
        tmp14 = tmp12 * tmp13
        tmp17 = tl.where(tmp11, tmp14, tmp16)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 - tmp9
        tmp20 = tl.exp(tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp28 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp24 = tl.load(in_ptr0 + (r2 + (2048*x0)), rmask, eviction_policy='evict_last')
        tmp25 = tl.load(in_ptr1 + (r2 + (2048*x3)), rmask, other=0).to(tl.float32)
        tmp26 = 0.125
        tmp27 = tmp25 * tmp26
        tmp30 = tl.where(tmp24, tmp27, tmp29)
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp31 - tmp9
        tmp33 = tl.exp(tmp32)
        tmp34 = tmp33 / tmp22
        tmp35 = tmp34.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (2048*x3)), tmp35, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/rz/crzn7fbdc2upf7k2p7nwlximxov6dg5vuvmq5myn65ar7drjkdz2.py
# Source Nodes: [matmul_1], Original ATen: [aten.clone]
# matmul_1 => clone_2
triton_poi_fused_clone_7 = async_compile.triton('triton_poi_fused_clone_7', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_clone_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 2048
    x2 = (xindex // 131072) % 16
    x3 = (xindex // 2097152)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1) + (2097152*x3)), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/sm/csmyzfnoqghv3byhphoraqw5dktvw5cxvstns6ls23myvaa4ydpo.py
# Source Nodes: [contiguous], Original ATen: [aten.clone]
# contiguous => clone_3
triton_poi_fused_clone_8 = async_compile.triton('triton_poi_fused_clone_8', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_clone_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 16
    x2 = (xindex // 1024) % 2048
    x3 = (xindex // 2097152)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (131072*x1) + (2097152*x3)), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/sb/csb7d7g2c6sc7flqoisdcnmqle7hs7b77skuxw4iki2x5u3ysjst.py
# Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_3 => add_3
# add_4 => add_4
# float_4 => convert_element_type_8
# mean_1 => mean_1
# mul_12 => mul_12
# mul_13 => mul_13
# mul_14 => mul_14
# rsqrt_1 => rsqrt_1
# type_as_3 => convert_element_type_9
triton_per_fused__to_copy_add_mean_mul_rsqrt_9 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_rsqrt_9', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_rsqrt_9', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_rsqrt_9(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel):
    xnumel = 6144
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp16 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = 1024.0
    tmp10 = tmp8 / tmp9
    tmp11 = 1e-05
    tmp12 = tmp10 + tmp11
    tmp13 = tl.math.rsqrt(tmp12)
    tmp14 = tmp3 * tmp13
    tmp15 = tmp14.to(tl.float32)
    tmp17 = tmp15 * tmp16
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp17, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/du/cdu46vphnvm5ppmbm7xf7bsha6vmhi5maoe7svuxmtszbvodm3ds.py
# Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
# mul_15 => mul_16
# silu => convert_element_type_10, convert_element_type_11, mul_15, sigmoid
triton_poi_fused_mul_silu_10 = async_compile.triton('triton_poi_fused_mul_silu_10', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_10', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_mul_silu_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17301504
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, None)
''')


# kernel path: /tmp/torchinductor_mengqy/ts/ctsb6vpi3lat6svut5zxew6trpl7smi42b2bjn4z2t5jxz2twu62.py
# Source Nodes: [add, add_3, add_5, l__mod___dynamic_dense_0_w1, mean, mul, pow_1, rsqrt], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
# add => add_6
# add_3 => add_3
# add_5 => add_5
# l__mod___dynamic_dense_0_w1 => view_30
# mean => mean_2
# mul => mul_17
# pow_1 => pow_1
# rsqrt => rsqrt_2
triton_per_fused_add_mean_mul_pow_rsqrt_view_11 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_view_11', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_view_11', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_view_11(in_ptr0, in_ptr1, in_ptr2, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 6144
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp3 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 + tmp3
    tmp5 = tmp4 * tmp4
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp11 = 1024.0
    tmp12 = tmp10 / tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp4 * tmp16
    tl.store(out_ptr1 + (x0), tmp16, None)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp17, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/nm/cnm6kcc5qxxubfdtbi3ed6psarqczlcuuxwflzqkf64lgvqcbzii.py
# Source Nodes: [l__mod___dynamic_dense_0_act, l__mod___dynamic_dense_0_w2], Original ATen: [aten.gelu, aten.view]
# l__mod___dynamic_dense_0_act => add_7, convert_element_type_12, convert_element_type_13, erf, mul_18, mul_19, mul_20
# l__mod___dynamic_dense_0_w2 => view_32
triton_poi_fused_gelu_view_12 = async_compile.triton('triton_poi_fused_gelu_view_12', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_12', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_gelu_view_12(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp1 * tmp4
    tmp6 = tl.math.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp10, None)
''')


# kernel path: /tmp/torchinductor_mengqy/jn/cjnnl2s5gvcpqvf23pf2bzeyxuksgf3j7kqwgjy2b6noh3rm5bfj.py
# Source Nodes: [add, add_2, add_3, add_4, add_6, float_1, float_2, float_3, mean, mean_1, mean_2, mul, mul_1, mul_2, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, rsqrt, rsqrt_1, rsqrt_2, type_as, type_as_1, type_as_2], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add => add_13
# add_2 => add_9
# add_3 => add_10
# add_4 => add_11, add_14
# add_6 => add_15
# float_1 => convert_element_type_14
# float_2 => convert_element_type_16
# float_3 => convert_element_type_18
# mean => mean_3
# mean_1 => mean_4
# mean_2 => mean_5
# mul => mul_25
# mul_1 => mul_21, mul_26
# mul_2 => mul_22, mul_27
# mul_3 => mul_23, mul_28
# mul_4 => mul_29
# mul_5 => mul_30
# mul_6 => mul_31
# mul_7 => mul_32
# mul_8 => mul_33
# rsqrt => rsqrt_3
# rsqrt_1 => rsqrt_4
# rsqrt_2 => rsqrt_5
# type_as => convert_element_type_15
# type_as_1 => convert_element_type_17
# type_as_2 => convert_element_type_19
triton_per_fused__to_copy_add_mean_mul_rsqrt_13 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_rsqrt_13', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_rsqrt_13', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_rsqrt_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr3, out_ptr4, out_ptr5, xnumel, rnumel):
    xnumel = 6144
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (8*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (0)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp14 = tl.load(in_ptr0 + (2 + (8*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp15 = tl.load(in_ptr1 + (2)).to(tl.float32)
    tmp16 = tl.broadcast_to(tmp15, [RBLOCK])
    tmp26 = tl.load(in_ptr0 + (4 + (8*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp27 = tl.load(in_ptr1 + (4)).to(tl.float32)
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp45 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp52 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp59 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = 0.0
    tmp7 = tmp5 + tmp6
    tmp8 = tmp7.to(tl.float32)
    tmp9 = tmp8 * tmp8
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.where(rmask, tmp10, 0)
    tmp13 = triton_helpers.promote_to_tensor(tl.sum(tmp12, 0))
    tmp17 = tmp14 + tmp16
    tmp18 = tmp17 * tmp4
    tmp19 = tmp18 + tmp6
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp29 = tmp26 + tmp28
    tmp30 = tmp29 * tmp4
    tmp31 = tmp30 + tmp6
    tmp32 = tmp31.to(tl.float32)
    tmp33 = tmp32 * tmp32
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp36 = tl.where(rmask, tmp34, 0)
    tmp37 = triton_helpers.promote_to_tensor(tl.sum(tmp36, 0))
    tmp38 = 1024.0
    tmp39 = tmp13 / tmp38
    tmp40 = 1e-05
    tmp41 = tmp39 + tmp40
    tmp42 = tl.math.rsqrt(tmp41)
    tmp43 = tmp8 * tmp42
    tmp44 = tmp43.to(tl.float32)
    tmp46 = tmp44 * tmp45
    tmp47 = tmp25 / tmp38
    tmp48 = tmp47 + tmp40
    tmp49 = tl.math.rsqrt(tmp48)
    tmp50 = tmp20 * tmp49
    tmp51 = tmp50.to(tl.float32)
    tmp53 = tmp51 * tmp52
    tmp54 = tmp37 / tmp38
    tmp55 = tmp54 + tmp40
    tmp56 = tl.math.rsqrt(tmp55)
    tmp57 = tmp32 * tmp56
    tmp58 = tmp57.to(tl.float32)
    tmp60 = tmp58 * tmp59
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp46, rmask)
    tl.store(out_ptr4 + (r1 + (1024*x0)), tmp53, rmask)
    tl.store(out_ptr5 + (r1 + (1024*x0)), tmp60, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/iz/cizoanwzjfubw7qhtrqduocudobrinbpjwiy2j35yn4qyy2rwt2z.py
# Source Nodes: [add_10, add_11, add_5, float_6, mean_3, mul_18, mul_19, mul_20, mul_4, rsqrt_3, type_as_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_10 => add_18
# add_11 => add_19
# add_5 => add_12
# float_6 => convert_element_type_26
# mean_3 => mean_6
# mul_18 => mul_43
# mul_19 => mul_44
# mul_20 => mul_45
# mul_4 => mul_24
# rsqrt_3 => rsqrt_6
# type_as_5 => convert_element_type_27
triton_per_fused__to_copy_add_mean_mul_rsqrt_14 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_rsqrt_14', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_rsqrt_14', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_rsqrt_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
    xnumel = 6144
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (6 + (8*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (6)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp8 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp23 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = 0.0
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp9.to(tl.float32)
    tmp11 = tmp10 * tmp10
    tmp12 = tl.broadcast_to(tmp11, [RBLOCK])
    tmp14 = tl.where(rmask, tmp12, 0)
    tmp15 = triton_helpers.promote_to_tensor(tl.sum(tmp14, 0))
    tmp16 = 1024.0
    tmp17 = tmp15 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp10 * tmp20
    tmp22 = tmp21.to(tl.float32)
    tmp24 = tmp22 * tmp23
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp24, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/6b/c6b5z6zq35wn7gw5caospwl4lu24ydotek2hxca5guc23ozlyf2p.py
# Source Nodes: [add_10, add_12, add_5, add_6, l__mod___dynamic_dense_1_w1, mean_1, mul_4, mul_5, pow_2, rsqrt_1], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
# add_10 => add_18
# add_12 => add_20
# add_5 => add_12
# add_6 => add_21
# l__mod___dynamic_dense_1_w1 => view_65
# mean_1 => mean_7
# mul_4 => mul_24
# mul_5 => mul_48
# pow_2 => pow_2
# rsqrt_1 => rsqrt_7
triton_per_fused_add_mean_mul_pow_rsqrt_view_15 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_view_15', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_view_15', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_view_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 6144
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (6 + (8*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (6)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp8 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp10 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = 0.0
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp11 = tmp9 + tmp10
    tmp12 = tmp11 * tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tl.broadcast_to(tmp13, [RBLOCK])
    tmp16 = tl.where(rmask, tmp14, 0)
    tmp17 = triton_helpers.promote_to_tensor(tl.sum(tmp16, 0))
    tmp18 = 1024.0
    tmp19 = tmp17 / tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = tl.math.rsqrt(tmp22)
    tmp24 = tmp11 * tmp23
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp11, rmask)
    tl.store(out_ptr1 + (x0), tmp23, None)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp24, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/qo/cqoa6nskzh7327r5tztxsgkhcxymnh4yzhbtocy5h5kv3eraz3po.py
# Source Nodes: [l__mod___dynamic_dense_1_act, l__mod___dynamic_dense_1_w2], Original ATen: [aten.gelu, aten.view]
# l__mod___dynamic_dense_1_act => add_22, convert_element_type_30, convert_element_type_31, erf_1, mul_49, mul_50, mul_51
# l__mod___dynamic_dense_1_w2 => view_67
triton_poi_fused_gelu_view_16 = async_compile.triton('triton_poi_fused_gelu_view_16', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_16', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_gelu_view_16(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp1 * tmp4
    tmp6 = tl.math.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp10, None)
''')


# kernel path: /tmp/torchinductor_mengqy/zi/cziawgdon4mpuhx3yjtafsudfyl33lourxze46nuii6mm7yuqa3v.py
# Source Nodes: [add, add_10, add_11, add_12, add_13, add_14, add_3, add_5, add_8, add_9, float_1, float_2, float_3, mean, mean_1, mean_2, mul, mul_1, mul_10, mul_11, mul_2, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, rsqrt, rsqrt_1, rsqrt_2, type_as, type_as_1, type_as_2], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add => add_32
# add_10 => add_26
# add_11 => add_27
# add_12 => add_28, add_33
# add_13 => add_29
# add_14 => add_34
# add_3 => add_3
# add_5 => add_5
# add_8 => add_24
# add_9 => add_25
# float_1 => convert_element_type_32
# float_2 => convert_element_type_34
# float_3 => convert_element_type_36
# mean => mean_8
# mean_1 => mean_9
# mean_2 => mean_10
# mul => mul_60
# mul_1 => mul_61
# mul_10 => mul_56
# mul_11 => mul_57
# mul_2 => mul_62
# mul_3 => mul_63
# mul_4 => mul_64
# mul_5 => mul_65
# mul_6 => mul_52, mul_66
# mul_7 => mul_53, mul_67
# mul_8 => mul_54, mul_68
# mul_9 => mul_55
# rsqrt => rsqrt_8
# rsqrt_1 => rsqrt_9
# rsqrt_2 => rsqrt_10
# type_as => convert_element_type_33
# type_as_1 => convert_element_type_35
# type_as_2 => convert_element_type_37
triton_per_fused__to_copy_add_mean_mul_rsqrt_17 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_rsqrt_17', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_rsqrt_17', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_rsqrt_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr4, out_ptr6, out_ptr8, xnumel, rnumel):
    xnumel = 6144
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (12*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (0)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (1 + (12*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr1 + (1)).to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp19 = tl.load(in_ptr0 + (3 + (12*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr1 + (3)).to(tl.float32)
    tmp21 = tl.broadcast_to(tmp20, [RBLOCK])
    tmp25 = tl.load(in_ptr0 + (4 + (12*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (4)).to(tl.float32)
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp32 = tl.load(in_ptr0 + (6 + (12*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp33 = tl.load(in_ptr1 + (6)).to(tl.float32)
    tmp34 = tl.broadcast_to(tmp33, [RBLOCK])
    tmp38 = tl.load(in_ptr0 + (7 + (12*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp39 = tl.load(in_ptr1 + (7)).to(tl.float32)
    tmp40 = tl.broadcast_to(tmp39, [RBLOCK])
    tmp57 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp69 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp81 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = 0.0
    tmp7 = tmp5 + tmp6
    tmp11 = tmp8 + tmp10
    tmp13 = tmp4 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp11 * tmp15
    tmp17 = tmp7 + tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp22 = tmp19 + tmp21
    tmp23 = tmp22 * tmp4
    tmp24 = tmp23 + tmp6
    tmp28 = tmp25 + tmp27
    tmp29 = tmp28 * tmp15
    tmp30 = tmp24 + tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp35 = tmp32 + tmp34
    tmp36 = tmp35 * tmp4
    tmp37 = tmp36 + tmp6
    tmp41 = tmp38 + tmp40
    tmp42 = tmp41 * tmp15
    tmp43 = tmp37 + tmp42
    tmp44 = tmp43.to(tl.float32)
    tmp45 = tmp18 * tmp18
    tmp46 = tl.broadcast_to(tmp45, [RBLOCK])
    tmp48 = tl.where(rmask, tmp46, 0)
    tmp49 = triton_helpers.promote_to_tensor(tl.sum(tmp48, 0))
    tmp50 = 1024.0
    tmp51 = tmp49 / tmp50
    tmp52 = 1e-05
    tmp53 = tmp51 + tmp52
    tmp54 = tl.math.rsqrt(tmp53)
    tmp55 = tmp18 * tmp54
    tmp56 = tmp55.to(tl.float32)
    tmp58 = tmp56 * tmp57
    tmp59 = tmp31 * tmp31
    tmp60 = tl.broadcast_to(tmp59, [RBLOCK])
    tmp62 = tl.where(rmask, tmp60, 0)
    tmp63 = triton_helpers.promote_to_tensor(tl.sum(tmp62, 0))
    tmp64 = tmp63 / tmp50
    tmp65 = tmp64 + tmp52
    tmp66 = tl.math.rsqrt(tmp65)
    tmp67 = tmp31 * tmp66
    tmp68 = tmp67.to(tl.float32)
    tmp70 = tmp68 * tmp69
    tmp71 = tmp44 * tmp44
    tmp72 = tl.broadcast_to(tmp71, [RBLOCK])
    tmp74 = tl.where(rmask, tmp72, 0)
    tmp75 = triton_helpers.promote_to_tensor(tl.sum(tmp74, 0))
    tmp76 = tmp75 / tmp50
    tmp77 = tmp76 + tmp52
    tmp78 = tl.math.rsqrt(tmp77)
    tmp79 = tmp44 * tmp78
    tmp80 = tmp79.to(tl.float32)
    tmp82 = tmp80 * tmp81
    tl.store(out_ptr4 + (r1 + (1024*x0)), tmp58, rmask)
    tl.store(out_ptr6 + (r1 + (1024*x0)), tmp70, rmask)
    tl.store(out_ptr8 + (r1 + (1024*x0)), tmp82, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/77/c77i5xrm4cr4ailo2ruvpxglmkiilc5dcomvvwxkosunacec57oa.py
# Source Nodes: [add_14, add_15, add_18, add_19, add_3, add_5, float_6, mean_3, mul_12, mul_13, mul_18, mul_19, mul_20, rsqrt_3, type_as_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_14 => add_30
# add_15 => add_31
# add_18 => add_37
# add_19 => add_38
# add_3 => add_3
# add_5 => add_5
# float_6 => convert_element_type_44
# mean_3 => mean_11
# mul_12 => mul_58
# mul_13 => mul_59
# mul_18 => mul_78
# mul_19 => mul_79
# mul_20 => mul_80
# rsqrt_3 => rsqrt_11
# type_as_5 => convert_element_type_45
triton_per_fused__to_copy_add_mean_mul_rsqrt_18 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_rsqrt_18', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_rsqrt_18', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_rsqrt_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, xnumel, rnumel):
    xnumel = 6144
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (9 + (12*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (9)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (10 + (12*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr1 + (10)).to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp18 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp33 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = 0.0
    tmp7 = tmp5 + tmp6
    tmp11 = tmp8 + tmp10
    tmp13 = tmp4 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp11 * tmp15
    tmp17 = tmp7 + tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tmp19.to(tl.float32)
    tmp21 = tmp20 * tmp20
    tmp22 = tl.broadcast_to(tmp21, [RBLOCK])
    tmp24 = tl.where(rmask, tmp22, 0)
    tmp25 = triton_helpers.promote_to_tensor(tl.sum(tmp24, 0))
    tmp26 = 1024.0
    tmp27 = tmp25 / tmp26
    tmp28 = 1e-05
    tmp29 = tmp27 + tmp28
    tmp30 = tl.math.rsqrt(tmp29)
    tmp31 = tmp20 * tmp30
    tmp32 = tmp31.to(tl.float32)
    tmp34 = tmp32 * tmp33
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp19, rmask)
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp34, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/ux/cuxyj54xsbxo5fucz4z3nqs3o5vfhjzqlciepryvdm2qrdgmon2p.py
# Source Nodes: [add_16, add_20, l__mod___dynamic_dense_2_w1, mean_2, mul_14, pow_3, rsqrt_2], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
# add_16 => add_40
# add_20 => add_39
# l__mod___dynamic_dense_2_w1 => view_100
# mean_2 => mean_12
# mul_14 => mul_83
# pow_3 => pow_3
# rsqrt_2 => rsqrt_12
triton_per_fused_add_mean_mul_pow_rsqrt_view_19 = async_compile.triton('triton_per_fused_add_mean_mul_pow_rsqrt_view_19', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_mean_mul_pow_rsqrt_view_19', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_per_fused_add_mean_mul_pow_rsqrt_view_19(in_ptr0, in_ptr1, out_ptr1, out_ptr2, xnumel, rnumel):
    xnumel = 6144
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp9 = 1024.0
    tmp10 = tmp8 / tmp9
    tmp11 = tmp10.to(tl.float32)
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tmp15 = tmp2 * tmp14
    tl.store(out_ptr1 + (x0), tmp14, None)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp15, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/mb/cmb6wxhcglo6ncbz6qu46ijugw2dz2s34xaqiuzo7tcr2p4j7br4.py
# Source Nodes: [l__mod___dynamic_dense_2_act, l__mod___dynamic_dense_2_w2], Original ATen: [aten.gelu, aten.view]
# l__mod___dynamic_dense_2_act => add_41, convert_element_type_48, convert_element_type_49, erf_2, mul_84, mul_85, mul_86
# l__mod___dynamic_dense_2_w2 => view_102
triton_poi_fused_gelu_view_20 = async_compile.triton('triton_poi_fused_gelu_view_20', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_20', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_gelu_view_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp1 * tmp4
    tmp6 = tl.math.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp10, None)
''')


# kernel path: /tmp/torchinductor_mengqy/y2/cy2ai3a6rl33vx4smpszl7lzh5y3n7ndmjpvlo3zjmuexv3uhndh.py
# Source Nodes: [add, add_18, add_19, add_20, add_21, add_22, add_23, add_24, add_25, add_26, add_27, add_28, add_29, add_3, add_5, float_1, float_2, float_3, mean, mean_1, mean_2, mul, mul_1, mul_15, mul_16, mul_17, mul_18, mul_19, mul_2, mul_20, mul_21, mul_22, mul_23, mul_24, mul_25, mul_26, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, rsqrt, rsqrt_1, rsqrt_2, type_as, type_as_1, type_as_2], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add => add_55
# add_18 => add_43
# add_19 => add_44
# add_20 => add_45
# add_21 => add_46
# add_22 => add_47
# add_23 => add_48
# add_24 => add_49, add_56
# add_25 => add_50
# add_26 => add_51
# add_27 => add_52, add_57
# add_28 => add_53
# add_29 => add_54
# add_3 => add_3
# add_5 => add_5
# float_1 => convert_element_type_50
# float_2 => convert_element_type_52
# float_3 => convert_element_type_54
# mean => mean_13
# mean_1 => mean_14
# mean_2 => mean_15
# mul => mul_99
# mul_1 => mul_100
# mul_15 => mul_87
# mul_16 => mul_88
# mul_17 => mul_89
# mul_18 => mul_90
# mul_19 => mul_91
# mul_2 => mul_101
# mul_20 => mul_92
# mul_21 => mul_93
# mul_22 => mul_94
# mul_23 => mul_95
# mul_24 => mul_96
# mul_25 => mul_97
# mul_26 => mul_98
# mul_3 => mul_102
# mul_4 => mul_103
# mul_5 => mul_104
# mul_6 => mul_105
# mul_7 => mul_106
# mul_8 => mul_107
# rsqrt => rsqrt_13
# rsqrt_1 => rsqrt_14
# rsqrt_2 => rsqrt_15
# type_as => convert_element_type_51
# type_as_1 => convert_element_type_53
# type_as_2 => convert_element_type_55
triton_per_fused__to_copy_add_mean_mul_rsqrt_21 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_rsqrt_21', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_rsqrt_21', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_rsqrt_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, out_ptr3, out_ptr5, out_ptr7, out_ptr9, xnumel, rnumel):
    xnumel = 6144
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (16*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (0)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (1 + (16*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr1 + (1)).to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp18 = tl.load(in_ptr0 + (2 + (16*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp19 = tl.load(in_ptr1 + (2)).to(tl.float32)
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.load(in_ptr5 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp25 = tl.load(in_ptr0 + (4 + (16*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (4)).to(tl.float32)
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp31 = tl.load(in_ptr0 + (5 + (16*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp32 = tl.load(in_ptr1 + (5)).to(tl.float32)
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp37 = tl.load(in_ptr0 + (6 + (16*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp38 = tl.load(in_ptr1 + (6)).to(tl.float32)
    tmp39 = tl.broadcast_to(tmp38, [RBLOCK])
    tmp43 = tl.load(in_ptr0 + (8 + (16*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp44 = tl.load(in_ptr1 + (8)).to(tl.float32)
    tmp45 = tl.broadcast_to(tmp44, [RBLOCK])
    tmp49 = tl.load(in_ptr0 + (9 + (16*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp50 = tl.load(in_ptr1 + (9)).to(tl.float32)
    tmp51 = tl.broadcast_to(tmp50, [RBLOCK])
    tmp55 = tl.load(in_ptr0 + (10 + (16*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp56 = tl.load(in_ptr1 + (10)).to(tl.float32)
    tmp57 = tl.broadcast_to(tmp56, [RBLOCK])
    tmp61 = tl.load(in_ptr0 + (12 + (16*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp62 = tl.load(in_ptr1 + (12)).to(tl.float32)
    tmp63 = tl.broadcast_to(tmp62, [RBLOCK])
    tmp67 = tl.load(in_ptr0 + (13 + (16*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp68 = tl.load(in_ptr1 + (13)).to(tl.float32)
    tmp69 = tl.broadcast_to(tmp68, [RBLOCK])
    tmp73 = tl.load(in_ptr0 + (14 + (16*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp74 = tl.load(in_ptr1 + (14)).to(tl.float32)
    tmp75 = tl.broadcast_to(tmp74, [RBLOCK])
    tmp92 = tl.load(in_ptr6 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp105 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp118 = tl.load(in_ptr8 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = 0.0
    tmp7 = tmp5 + tmp6
    tmp11 = tmp8 + tmp10
    tmp13 = tmp4 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp11 * tmp15
    tmp17 = tmp7 + tmp16
    tmp21 = tmp18 + tmp20
    tmp23 = tmp21 * tmp22
    tmp24 = tmp17 + tmp23
    tmp28 = tmp25 + tmp27
    tmp29 = tmp28 * tmp4
    tmp30 = tmp29 + tmp6
    tmp34 = tmp31 + tmp33
    tmp35 = tmp34 * tmp15
    tmp36 = tmp30 + tmp35
    tmp40 = tmp37 + tmp39
    tmp41 = tmp40 * tmp22
    tmp42 = tmp36 + tmp41
    tmp46 = tmp43 + tmp45
    tmp47 = tmp46 * tmp4
    tmp48 = tmp47 + tmp6
    tmp52 = tmp49 + tmp51
    tmp53 = tmp52 * tmp15
    tmp54 = tmp48 + tmp53
    tmp58 = tmp55 + tmp57
    tmp59 = tmp58 * tmp22
    tmp60 = tmp54 + tmp59
    tmp64 = tmp61 + tmp63
    tmp65 = tmp64 * tmp4
    tmp66 = tmp65 + tmp6
    tmp70 = tmp67 + tmp69
    tmp71 = tmp70 * tmp15
    tmp72 = tmp66 + tmp71
    tmp76 = tmp73 + tmp75
    tmp77 = tmp76 * tmp22
    tmp78 = tmp72 + tmp77
    tmp79 = tmp24.to(tl.float32)
    tmp80 = tmp79 * tmp79
    tmp81 = tl.broadcast_to(tmp80, [RBLOCK])
    tmp83 = tl.where(rmask, tmp81, 0)
    tmp84 = triton_helpers.promote_to_tensor(tl.sum(tmp83, 0))
    tmp85 = 1024.0
    tmp86 = tmp84 / tmp85
    tmp87 = 1e-05
    tmp88 = tmp86 + tmp87
    tmp89 = tl.math.rsqrt(tmp88)
    tmp90 = tmp79 * tmp89
    tmp91 = tmp90.to(tl.float32)
    tmp93 = tmp91 * tmp92
    tmp94 = tmp42.to(tl.float32)
    tmp95 = tmp94 * tmp94
    tmp96 = tl.broadcast_to(tmp95, [RBLOCK])
    tmp98 = tl.where(rmask, tmp96, 0)
    tmp99 = triton_helpers.promote_to_tensor(tl.sum(tmp98, 0))
    tmp100 = tmp99 / tmp85
    tmp101 = tmp100 + tmp87
    tmp102 = tl.math.rsqrt(tmp101)
    tmp103 = tmp94 * tmp102
    tmp104 = tmp103.to(tl.float32)
    tmp106 = tmp104 * tmp105
    tmp107 = tmp60.to(tl.float32)
    tmp108 = tmp107 * tmp107
    tmp109 = tl.broadcast_to(tmp108, [RBLOCK])
    tmp111 = tl.where(rmask, tmp109, 0)
    tmp112 = triton_helpers.promote_to_tensor(tl.sum(tmp111, 0))
    tmp113 = tmp112 / tmp85
    tmp114 = tmp113 + tmp87
    tmp115 = tl.math.rsqrt(tmp114)
    tmp116 = tmp107 * tmp115
    tmp117 = tmp116.to(tl.float32)
    tmp119 = tmp117 * tmp118
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp78, rmask)
    tl.store(out_ptr5 + (r1 + (1024*x0)), tmp93, rmask)
    tl.store(out_ptr7 + (r1 + (1024*x0)), tmp106, rmask)
    tl.store(out_ptr9 + (r1 + (1024*x0)), tmp119, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/4e/c4endhao2qbvyx3ngnslt4xk5ley4ixy6kh2m3bpua7id4jwubhg.py
# Source Nodes: [l__mod___dynamic_dense_3_act, l__mod___dynamic_dense_3_w2], Original ATen: [aten.gelu, aten.view]
# l__mod___dynamic_dense_3_act => add_64, convert_element_type_66, convert_element_type_67, erf_3, mul_123, mul_124, mul_125
# l__mod___dynamic_dense_3_w2 => view_137
triton_poi_fused_gelu_view_22 = async_compile.triton('triton_poi_fused_gelu_view_22', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_view_22', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_gelu_view_22(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 122880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = 0.5
    tmp3 = tmp1 * tmp2
    tmp4 = 0.7071067811865476
    tmp5 = tmp1 * tmp4
    tmp6 = tl.math.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp3 * tmp8
    tmp10 = tmp9.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp10, None)
''')


# kernel path: /tmp/torchinductor_mengqy/tn/ctn4uujx7xvcaj66ujaoacpxbvahe65fpuafuqon7p5ssmmt2fyz.py
# Source Nodes: [add_20, add_3, add_36, add_37, add_38, add_39, add_48, add_5, float_1, l__mod___output, mean_4, mul_32, mul_33, mul_34, mul_35, mul_44, mul_45, mul_46, rsqrt_4, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.view]
# add_20 => add_39
# add_3 => add_3
# add_36 => add_70
# add_37 => add_71
# add_38 => add_72
# add_39 => add_73
# add_48 => add_82
# add_5 => add_5
# float_1 => convert_element_type_68
# l__mod___output => view_140
# mean_4 => mean_18
# mul_32 => mul_130
# mul_33 => mul_131
# mul_34 => mul_132
# mul_35 => mul_133
# mul_44 => mul_142
# mul_45 => mul_143
# mul_46 => mul_144
# rsqrt_4 => rsqrt_18
# type_as => convert_element_type_69
triton_per_fused__to_copy_add_mean_mul_rsqrt_view_23 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_rsqrt_view_23', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[8192, 1024],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_rsqrt_view_23', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_rsqrt_view_23(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, out_ptr1, xnumel, rnumel):
    xnumel = 6144
    XBLOCK: tl.constexpr = 1
    rnumel = 1024
    RBLOCK: tl.constexpr = 1024
    xoffset = tl.program_id(0) * XBLOCK
    xindex = tl.full([1], xoffset, tl.int32)
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[:]
    rmask = rindex < rnumel
    x0 = xindex
    r1 = rindex
    tmp0 = tl.load(in_ptr0 + (5 + (20*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (5)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [RBLOCK])
    tmp4 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp8 = tl.load(in_ptr0 + (6 + (20*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp9 = tl.load(in_ptr1 + (6)).to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [RBLOCK])
    tmp12 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp14 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp18 = tl.load(in_ptr0 + (7 + (20*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp19 = tl.load(in_ptr1 + (7)).to(tl.float32)
    tmp20 = tl.broadcast_to(tmp19, [RBLOCK])
    tmp22 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp25 = tl.load(in_ptr0 + (8 + (20*x0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp26 = tl.load(in_ptr1 + (8)).to(tl.float32)
    tmp27 = tl.broadcast_to(tmp26, [RBLOCK])
    tmp29 = tl.load(in_ptr5 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp30 = tl.load(in_ptr6 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp47 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp3 = tmp0 + tmp2
    tmp5 = tmp3 * tmp4
    tmp6 = 0.0
    tmp7 = tmp5 + tmp6
    tmp11 = tmp8 + tmp10
    tmp13 = tmp4 + tmp12
    tmp15 = tmp13 + tmp14
    tmp16 = tmp11 * tmp15
    tmp17 = tmp7 + tmp16
    tmp21 = tmp18 + tmp20
    tmp23 = tmp21 * tmp22
    tmp24 = tmp17 + tmp23
    tmp28 = tmp25 + tmp27
    tmp31 = tmp29 + tmp30
    tmp32 = tmp28 * tmp31
    tmp33 = tmp24 + tmp32
    tmp34 = tmp33.to(tl.float32)
    tmp35 = tmp34 * tmp34
    tmp36 = tl.broadcast_to(tmp35, [RBLOCK])
    tmp38 = tl.where(rmask, tmp36, 0)
    tmp39 = triton_helpers.promote_to_tensor(tl.sum(tmp38, 0))
    tmp40 = 1024.0
    tmp41 = tmp39 / tmp40
    tmp42 = 1e-05
    tmp43 = tmp41 + tmp42
    tmp44 = tl.math.rsqrt(tmp43)
    tmp45 = tmp34 * tmp44
    tmp46 = tmp45.to(tl.float32)
    tmp48 = tmp46 * tmp47
    tl.debug_barrier()
    tl.store(in_out_ptr1 + (x0), tmp44, None)
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp48, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/af/cafoxopao6bukdpmxilm2nhbciegsujfmizt644qomwkre34kiju.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_24 = async_compile.triton('triton_poi_fused_24', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[1024, 65536], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_24', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_24(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 1024
    xnumel = 50257
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (1024*x1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x1 + (50264*y0)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/cc/cccsh3poxqjecqjfsqdqrzpjtjvvkp7qxed5sqmgwqm4ecl6t37r.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_25 = async_compile.triton('triton_poi_fused_25', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_25', 'configs': [instance_descriptor(divisible_by_16=(1,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]})
@triton.jit
def triton_poi_fused_25(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (50264*x1)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/64/c64whmotvms35lzhfhte6irauy32iagcj6rkuz3kswjiaxhbrzso.py
# Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax]
# cross_entropy => amax_4, convert_element_type_70, convert_element_type_71, exp_4, log, sub_12, sub_13, sum_5
triton_red_fused__log_softmax_26 = async_compile.triton('triton_red_fused__log_softmax_26', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 65536],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_26', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]}
)
@triton.jit
def triton_red_fused__log_softmax_26(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 50257
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp3 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (50264*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
        tmp4 = triton_helpers.maximum(_tmp3, tmp2)
        _tmp3 = tl.where(rmask, tmp4, _tmp3)
    tmp3 = triton_helpers.max2(_tmp3, 1)[:, None]
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp5 = tl.load(in_ptr0 + (r1 + (50264*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp6 - tmp3
        tmp8 = tl.exp(tmp7)
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp12 = tl.load(in_ptr0 + (r1 + (50264*x0)), rmask, other=0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = tmp13 - tmp3
        tmp15 = tl.log(tmp10)
        tmp16 = tmp14 - tmp15
        tmp17 = tmp16.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (50257*x0)), tmp17, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/jh/cjhbsqwbx6vyncxiuurwsm6wiyjwcwbkak4mgm5lqtps7oiinaoc.py
# Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_forward]
# cross_entropy => convert_element_type_72, div_4, ne, neg, scalar_tensor_5, sum_6, sum_7, where_5
triton_red_fused_nll_loss_forward_27 = async_compile.triton('triton_red_fused_nll_loss_forward_27', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_27', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]}
)
@triton.jit
def triton_red_fused_nll_loss_forward_27(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp5 = tl.full([XBLOCK, RBLOCK], 0, tl.int64)
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + ((2049*(r0 // 2048)) + (r0 % 2048)), rmask, other=0)
        tmp1 = tl.full([1, 1], -100, tl.int64)
        tmp2 = tmp0 != tmp1
        tmp3 = tmp2.to(tl.int64)
        tmp4 = tl.broadcast_to(tmp3, [XBLOCK, RBLOCK])
        tmp6 = _tmp5 + tmp4
        _tmp5 = tl.where(rmask, tmp6, _tmp5)
        tmp7 = tl.full([1, 1], 0, tl.int64)
        tmp8 = tl.where(tmp2, tmp0, tmp7)
        tmp9 = tl.where(tmp8 < 0, tmp8 + 50257, tmp8)
        tl.device_assert((0 <= tmp9) & (tmp9 < 50257), "index out of bounds: 0 <= tmp9 < 50257")
        tmp10 = tl.load(in_ptr1 + (tmp9 + (50257*r0)), rmask, other=0).to(tl.float32)
        tmp11 = -tmp10
        tmp12 = 0.0
        tmp13 = tl.where(tmp2, tmp11, tmp12)
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp5 = tl.sum(_tmp5, 1)[:, None]
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tmp17 = tmp5.to(tl.float32)
    tmp18 = tmp15 / tmp17
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp17, None)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp18, None)
''')


# kernel path: /tmp/torchinductor_mengqy/zn/cznjbbaxqvjvfajkiujjziv3q3ofvbpc3fslwssyyiegngu2voir.py
# Source Nodes: [getitem_1], Original ATen: [aten.index]
# getitem_1 => index_1
triton_poi_fused_index_28 = async_compile.triton('triton_poi_fused_index_28', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_28', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_index_28(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 131072
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/am/camiqyffw2krofl4c4pnresf67pq7ejupz2uepdk5jx7plgkrmnw.py
# Source Nodes: [add_1], Original ATen: [aten.add]
# add_1 => add_8
triton_poi_fused_add_29 = async_compile.triton('triton_poi_fused_add_29', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_29', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]})
@triton.jit
def triton_poi_fused_add_29(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2
    x1 = (xindex // 2) % 6144
    x2 = (xindex // 12288)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2*x2) + (8*x1)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + (2*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_mengqy/2n/c2ns3juzlohmsdfawj3akznguycknymxnrr22d22vjpo6etnsq3n.py
# Source Nodes: [add_7], Original ATen: [aten.add]
# add_7 => add_23
triton_poi_fused_add_30 = async_compile.triton('triton_poi_fused_add_30', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_30', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]})
@triton.jit
def triton_poi_fused_add_30(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x1 = (xindex // 3) % 6144
    x2 = (xindex // 18432)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3*x2) + (12*x1)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + (3*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_mengqy/qd/cqd3zal2ufejtw7vi5aixnzqcpa2lwskfiniqccre66lo5qczaob.py
# Source Nodes: [add_17], Original ATen: [aten.add]
# add_17 => add_42
triton_poi_fused_add_31 = async_compile.triton('triton_poi_fused_add_31', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_31', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]})
@triton.jit
def triton_poi_fused_add_31(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = (xindex // 4) % 6144
    x2 = (xindex // 24576)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (4*x2) + (16*x1)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + (4*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


# kernel path: /tmp/torchinductor_mengqy/nh/cnhj4v4gphpjnvucesv3sdwtpy7cgosy4mjh3hiarw7qkgg5p3b4.py
# Source Nodes: [add_31], Original ATen: [aten.add]
# add_31 => add_65
triton_poi_fused_add_32 = async_compile.triton('triton_poi_fused_add_32', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_32', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]})
@triton.jit
def triton_poi_fused_add_32(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 122880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 5
    x1 = (xindex // 5) % 6144
    x2 = (xindex // 30720)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (5*x2) + (20*x1)), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0 + (5*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + (x3), tmp2, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61 = args
    args.clear()
    assert_size_stride(primals_1, (1024, ), (1, ))
    assert_size_stride(primals_2, (1024, 1024), (1024, 1))
    assert_size_stride(primals_3, (1024, 1024), (1024, 1))
    assert_size_stride(primals_4, (1024, 1024), (1024, 1))
    assert_size_stride(primals_5, (1024, 1024), (1024, 1))
    assert_size_stride(primals_6, (1024, ), (1, ))
    assert_size_stride(primals_7, (2816, 1024), (1024, 1))
    assert_size_stride(primals_8, (2816, 1024), (1024, 1))
    assert_size_stride(primals_9, (1024, 2816), (2816, 1))
    assert_size_stride(primals_10, (4, 2), (2, 1))
    assert_size_stride(primals_11, (1024, ), (1, ))
    assert_size_stride(primals_12, (1024, ), (1, ))
    assert_size_stride(primals_13, (1024, ), (1, ))
    assert_size_stride(primals_14, (1024, 1024), (1024, 1))
    assert_size_stride(primals_15, (1024, 1024), (1024, 1))
    assert_size_stride(primals_16, (1024, 1024), (1024, 1))
    assert_size_stride(primals_17, (1024, 1024), (1024, 1))
    assert_size_stride(primals_18, (1024, ), (1, ))
    assert_size_stride(primals_19, (2816, 1024), (1024, 1))
    assert_size_stride(primals_20, (2816, 1024), (1024, 1))
    assert_size_stride(primals_21, (1024, 2816), (2816, 1))
    assert_size_stride(primals_22, (4, 3), (3, 1))
    assert_size_stride(primals_23, (1024, ), (1, ))
    assert_size_stride(primals_24, (1024, ), (1, ))
    assert_size_stride(primals_25, (1024, ), (1, ))
    assert_size_stride(primals_26, (1024, 1024), (1024, 1))
    assert_size_stride(primals_27, (1024, 1024), (1024, 1))
    assert_size_stride(primals_28, (1024, 1024), (1024, 1))
    assert_size_stride(primals_29, (1024, 1024), (1024, 1))
    assert_size_stride(primals_30, (1024, ), (1, ))
    assert_size_stride(primals_31, (2816, 1024), (1024, 1))
    assert_size_stride(primals_32, (2816, 1024), (1024, 1))
    assert_size_stride(primals_33, (1024, 2816), (2816, 1))
    assert_size_stride(primals_34, (4, 4), (4, 1))
    assert_size_stride(primals_35, (1024, ), (1, ))
    assert_size_stride(primals_36, (1024, ), (1, ))
    assert_size_stride(primals_37, (1024, ), (1, ))
    assert_size_stride(primals_38, (1024, 1024), (1024, 1))
    assert_size_stride(primals_39, (1024, 1024), (1024, 1))
    assert_size_stride(primals_40, (1024, 1024), (1024, 1))
    assert_size_stride(primals_41, (1024, 1024), (1024, 1))
    assert_size_stride(primals_42, (1024, ), (1, ))
    assert_size_stride(primals_43, (2816, 1024), (1024, 1))
    assert_size_stride(primals_44, (2816, 1024), (1024, 1))
    assert_size_stride(primals_45, (1024, 2816), (2816, 1))
    assert_size_stride(primals_46, (4, 5), (5, 1))
    assert_size_stride(primals_47, (1024, ), (1, ))
    assert_size_stride(primals_48, (50257, 1024), (1024, 1))
    assert_size_stride(primals_49, (8, 1024), (1024, 1))
    assert_size_stride(primals_50, (8, 8), (8, 1))
    assert_size_stride(primals_51, (12, 1024), (1024, 1))
    assert_size_stride(primals_52, (12, 12), (12, 1))
    assert_size_stride(primals_53, (16, 1024), (1024, 1))
    assert_size_stride(primals_54, (16, 16), (16, 1))
    assert_size_stride(primals_55, (20, 1024), (1024, 1))
    assert_size_stride(primals_56, (20, 20), (20, 1))
    assert_size_stride(primals_57, (50257, 1024), (1024, 1))
    assert_size_stride(primals_58, (2048, 32, 2), (64, 2, 1))
    assert_size_stride(primals_59, (2048, 2048), (2048, 1))
    assert_size_stride(primals_60, (3, 2048), (2049, 1))
    assert_size_stride(primals_61, (3, 2048), (2049, 1))
    with torch.cuda._DeviceGuard(0):
        print_mem(s='start')
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16) # BTD
        # Source Nodes: [l__mod___tok_embeddings], Original ATen: [aten.embedding]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_embedding_0.run(primals_60, primals_48, buf0, 6291456, grid=grid(6291456), stream=stream0)
        del primals_48
        buf1 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16) # q: (BT)D
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_2, (1024, 1024), (1, 1024), 0), out=buf1)
        buf2 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16) # k: (BT)D
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_3, (1024, 1024), (1, 1024), 0), out=buf2)
        buf3 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16) # v: (BT)D
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf0, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_4, (1024, 1024), (1, 1024), 0), out=buf3)
        buf6 = empty_strided((3, 2048, 16, 32, 2), (2097152, 1024, 64, 2, 1), device='cuda', dtype=torch.float32)
        buf4 = reinterpret_tensor(buf6, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 0)  # alias
        buf5 = reinterpret_tensor(buf6, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 1)  # alias
        buf9 = empty_strided((3, 2048, 16, 32, 2), (2097152, 1024, 64, 2, 1), device='cuda', dtype=torch.float32)
        buf7 = reinterpret_tensor(buf9, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 0)  # alias
        buf8 = reinterpret_tensor(buf9, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf1, primals_58, buf2, buf4, buf5, buf7, buf8, 3145728, grid=grid(3145728), stream=stream0) # rope
        buf10 = empty_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda', dtype=torch.bool) # mask
        # Source Nodes: [getitem, getitem_18], Original ATen: [aten.index, aten.slice]
        triton_poi_fused_index_slice_2.run(primals_59, buf10, 4194304, grid=grid(4194304), stream=stream0) 
        del buf4
        del buf5
        del buf7
        del buf8
        del primals_59
        buf11 = reinterpret_tensor(buf2, (3, 16, 2048, 64), (2097152, 131072, 64, 1)); del buf2  # reuse BNTD
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf6, buf11, 6291456, grid=grid(6291456), stream=stream0)
        buf12 = reinterpret_tensor(buf1, (3, 16, 64, 2048), (2097152, 131072, 2048, 1)); del buf1  # reuse BNDT
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf9, buf12, 3072, 2048, grid=grid(3072, 2048), stream=stream0)
        buf13 = empty_strided((48, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16) # (BN)TS
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf11, (48, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf12, (48, 64, 2048), (131072, 2048, 1), 0), out=buf13) # q @ k
        buf14 = empty_strided((), (), device='cuda', dtype=torch.float16)
        # Source Nodes: [where], Original ATen: [aten.scalar_tensor]
        triton_poi_fused_scalar_tensor_5.run(buf14, 1, grid=grid(1), stream=stream0)
        buf17 = empty_strided((3, 16, 2048, 2048), (67108864, 4194304, 2048, 1), device='cuda', dtype=torch.float16) # BNTS
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf13, buf14, buf17, 98304, 2048, grid=grid(98304), stream=stream0) # softmax, mask
        buf18 = reinterpret_tensor(buf12, (3, 16, 2048, 64), (2097152, 131072, 64, 1)); del buf12  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf3, buf18, 6291456, grid=grid(6291456), stream=stream0)
        buf19 = reinterpret_tensor(buf3, (48, 2048, 64), (131072, 64, 1)); del buf3  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf17, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf18, (48, 2048, 64), (131072, 64, 1), 0), out=buf19) # BNTS, BNSd -> BNTd
        buf20 = reinterpret_tensor(buf18, (3, 2048, 16, 64), (2097152, 1024, 64, 1)); del buf18  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf19, buf20, 6291456, grid=grid(6291456), stream=stream0)
        buf21 = reinterpret_tensor(buf19, (6144, 1024), (1024, 1)); del buf19  # reuse mixed v BTD
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_5, (1024, 1024), (1, 1024), 0), out=buf21) # apply wo
        buf23 = reinterpret_tensor(buf20, (3, 2048, 1024), (2097152, 1024, 1)); del buf20  # reuse mlp_input buf
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_per_fused__to_copy_add_mean_mul_rsqrt_9.run(buf0, buf21, primals_6, buf23, 6144, 1024, grid=grid(6144), stream=stream0) # mlp_input: add residual, rmsnorm
        buf24 = empty_strided((6144, 2816), (2816, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_7, (1024, 2816), (1, 1024), 0), out=buf24) # apply w1
        buf25 = empty_strided((6144, 2816), (2816, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_8, (1024, 2816), (1, 1024), 0), out=buf25) # apply wg
        buf26 = reinterpret_tensor(buf24, (3, 2048, 2816), (5767168, 2816, 1)); del buf24  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf26, buf25, 17301504, grid=grid(17301504), stream=stream0) # w1(x) * wg(x)
        buf27 = reinterpret_tensor(buf23, (6144, 1024), (1024, 1)); del buf23  # reuse BTD, mlp_output
        # Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (6144, 2816), (2816, 1), 0), reinterpret_tensor(primals_9, (2816, 1024), (1, 2816), 0), out=buf27)
        print_mem(s='layer1mlp')
        buf29 = empty_strided((3, 2048, 1), (2048, 1, 1), device='cuda', dtype=torch.float16)
        buf30 = reinterpret_tensor(buf11, (6144, 1024), (1024, 1)); del buf11  # reuse BTD
        # Source Nodes: [add, add_3, add_5, l__mod___dynamic_dense_0_w1, mean, mul, pow_1, rsqrt], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_11.run(buf0, buf21, buf27, buf29, buf30, 6144, 1024, grid=grid(6144), stream=stream0)
        buf31 = empty_strided((6144, 8), (8, 1), device='cuda', dtype=torch.float16) # dw1_act
        # Source Nodes: [l__mod___dynamic_dense_0_w1], Original ATen: [aten.mm]
        extern_kernels.mm(buf30, reinterpret_tensor(primals_49, (1024, 8), (1, 1024), 0), out=buf31) # apply dw1: BTD, DK->BTK
        buf32 = empty_strided((6144, 8), (8, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__mod___dynamic_dense_0_act, l__mod___dynamic_dense_0_w2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_12.run(buf31, buf32, 49152, grid=grid(49152), stream=stream0) # gelu
        buf33 = empty_strided((6144, 8), (8, 1), device='cuda', dtype=torch.float16) # dw
        # Source Nodes: [l__mod___dynamic_dense_0_w2], Original ATen: [aten.mm]
        extern_kernels.mm(buf32, reinterpret_tensor(primals_50, (8, 8), (1, 8), 0), out=buf33) # apply dw2
        buf37 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16) # xq: BTD 
        buf39 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16) # xk: BTD
        buf41 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16) # xv: BTD
        # Source Nodes: [add, add_2, add_3, add_4, add_6, float_1, float_2, float_3, mean, mean_1, mean_2, mul, mul_1, mul_2, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, rsqrt, rsqrt_1, rsqrt_2, type_as, type_as_1, type_as_2], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_per_fused__to_copy_add_mean_mul_rsqrt_13.run(buf33, primals_10, buf0, primals_11, primals_12, primals_13, buf37, buf39, buf41, 6144, 1024, grid=grid(6144), stream=stream0) # apply sw+ dw and rmsnorm for xq,xk,xv
        buf38 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf37, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_14, (1024, 1024), (1, 1024), 0), out=buf38)
        buf40 = reinterpret_tensor(buf37, (6144, 1024), (1024, 1)); del buf37  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_15, (1024, 1024), (1, 1024), 0), out=buf40)
        buf42 = reinterpret_tensor(buf39, (6144, 1024), (1024, 1)); del buf39  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf41, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_16, (1024, 1024), (1, 1024), 0), out=buf42)
        buf45 = buf9; del buf9  # reuse
        buf43 = reinterpret_tensor(buf45, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 0)  # alias
        buf44 = reinterpret_tensor(buf45, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 1)  # alias
        buf48 = buf6; del buf6  # reuse
        buf46 = reinterpret_tensor(buf48, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 0)  # alias
        buf47 = reinterpret_tensor(buf48, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf38, primals_58, buf40, buf43, buf44, buf46, buf47, 3145728, grid=grid(3145728), stream=stream0)
        buf49 = reinterpret_tensor(buf40, (3, 16, 2048, 64), (2097152, 131072, 64, 1)); del buf40  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf45, buf49, 6291456, grid=grid(6291456), stream=stream0)
        del buf43
        del buf44
        del buf46
        del buf47
        buf50 = reinterpret_tensor(buf38, (3, 16, 64, 2048), (2097152, 131072, 2048, 1)); del buf38  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf48, buf50, 3072, 2048, grid=grid(3072, 2048), stream=stream0)
        buf51 = reinterpret_tensor(buf17, (48, 2048, 2048), (4194304, 2048, 1)); del buf17  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf49, (48, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf50, (48, 64, 2048), (131072, 2048, 1), 0), out=buf51)
        buf54 = reinterpret_tensor(buf13, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf13  # reuse
        # Source Nodes: [mul_17, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf51, buf14, buf54, 98304, 2048, grid=grid(98304), stream=stream0)
        buf55 = reinterpret_tensor(buf50, (3, 16, 2048, 64), (2097152, 131072, 64, 1)); del buf50  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf42, buf55, 6291456, grid=grid(6291456), stream=stream0)
        buf56 = reinterpret_tensor(buf42, (48, 2048, 64), (131072, 64, 1)); del buf42  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf54, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf55, (48, 2048, 64), (131072, 64, 1), 0), out=buf56)
        buf57 = reinterpret_tensor(buf55, (3, 2048, 16, 64), (2097152, 1024, 64, 1)); del buf55  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf56, buf57, 6291456, grid=grid(6291456), stream=stream0)
        buf58 = reinterpret_tensor(buf56, (6144, 1024), (1024, 1)); del buf56  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf57, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_17, (1024, 1024), (1, 1024), 0), out=buf58)
        buf60 = reinterpret_tensor(buf57, (3, 2048, 1024), (2097152, 1024, 1)); del buf57  # reuse
        # Source Nodes: [add_10, add_11, add_5, float_6, mean_3, mul_18, mul_19, mul_20, mul_4, rsqrt_3, type_as_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_per_fused__to_copy_add_mean_mul_rsqrt_14.run(buf33, primals_10, buf0, buf58, primals_18, buf60, 6144, 1024, grid=grid(6144), stream=stream0)
        buf61 = reinterpret_tensor(buf26, (6144, 2816), (2816, 1)); del buf26  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf60, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_19, (1024, 2816), (1, 1024), 0), out=buf61)
        buf62 = buf25; del buf25  # reuse
        # Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf60, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_20, (1024, 2816), (1, 1024), 0), out=buf62)
        buf63 = reinterpret_tensor(buf61, (3, 2048, 2816), (5767168, 2816, 1)); del buf61  # reuse
        # Source Nodes: [mul_21, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf63, buf62, 17301504, grid=grid(17301504), stream=stream0)
        buf64 = reinterpret_tensor(buf60, (6144, 1024), (1024, 1)); del buf60  # reuse
        # Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (6144, 2816), (2816, 1), 0), reinterpret_tensor(primals_21, (2816, 1024), (1, 2816), 0), out=buf64)
        print_mem(s='layer2mlp')
        buf65 = reinterpret_tensor(buf58, (3, 2048, 1024), (2097152, 1024, 1)); del buf58  # reuse
        buf67 = empty_strided((3, 2048, 1), (2048, 1, 1), device='cuda', dtype=torch.float16)
        buf68 = reinterpret_tensor(buf49, (6144, 1024), (1024, 1)); del buf49  # reuse
        # Source Nodes: [add_10, add_12, add_5, add_6, l__mod___dynamic_dense_1_w1, mean_1, mul_4, mul_5, pow_2, rsqrt_1], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_15.run(buf65, buf33, primals_10, buf0, buf64, buf67, buf68, 6144, 1024, grid=grid(6144), stream=stream0)
        buf69 = empty_strided((6144, 12), (12, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__mod___dynamic_dense_1_w1], Original ATen: [aten.mm]
        extern_kernels.mm(buf68, reinterpret_tensor(primals_51, (1024, 12), (1, 1024), 0), out=buf69)
        buf70 = empty_strided((6144, 12), (12, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__mod___dynamic_dense_1_act, l__mod___dynamic_dense_1_w2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_16.run(buf69, buf70, 73728, grid=grid(73728), stream=stream0)
        buf71 = empty_strided((6144, 12), (12, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__mod___dynamic_dense_1_w2], Original ATen: [aten.mm]
        extern_kernels.mm(buf70, reinterpret_tensor(primals_52, (12, 12), (1, 12), 0), out=buf71)
        buf78 = reinterpret_tensor(buf64, (3, 2048, 1024), (2097152, 1024, 1)); del buf64  # reuse
        buf80 = buf41; del buf41  # reuse
        buf82 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_10, add_11, add_12, add_13, add_14, add_3, add_5, add_8, add_9, float_1, float_2, float_3, mean, mean_1, mean_2, mul, mul_1, mul_10, mul_11, mul_2, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, rsqrt, rsqrt_1, rsqrt_2, type_as, type_as_1, type_as_2], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_per_fused__to_copy_add_mean_mul_rsqrt_17.run(buf71, primals_22, buf0, buf21, buf27, primals_23, primals_24, primals_25, buf78, buf80, buf82, 6144, 1024, grid=grid(6144), stream=stream0)
        buf79 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf78, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_26, (1024, 1024), (1, 1024), 0), out=buf79)
        buf81 = reinterpret_tensor(buf78, (6144, 1024), (1024, 1)); del buf78  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_27, (1024, 1024), (1, 1024), 0), out=buf81)
        buf83 = reinterpret_tensor(buf80, (6144, 1024), (1024, 1)); del buf80  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_28, (1024, 1024), (1, 1024), 0), out=buf83)
        buf86 = buf48; del buf48  # reuse
        buf84 = reinterpret_tensor(buf86, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 0)  # alias
        buf85 = reinterpret_tensor(buf86, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 1)  # alias
        buf89 = buf45; del buf45  # reuse
        buf87 = reinterpret_tensor(buf89, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 0)  # alias
        buf88 = reinterpret_tensor(buf89, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf79, primals_58, buf81, buf84, buf85, buf87, buf88, 3145728, grid=grid(3145728), stream=stream0)
        buf90 = reinterpret_tensor(buf81, (3, 16, 2048, 64), (2097152, 131072, 64, 1)); del buf81  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf86, buf90, 6291456, grid=grid(6291456), stream=stream0)
        del buf84
        del buf85
        del buf87
        del buf88
        buf91 = reinterpret_tensor(buf79, (3, 16, 64, 2048), (2097152, 131072, 2048, 1)); del buf79  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf89, buf91, 3072, 2048, grid=grid(3072, 2048), stream=stream0)
        buf92 = reinterpret_tensor(buf54, (48, 2048, 2048), (4194304, 2048, 1)); del buf54  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf90, (48, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf91, (48, 64, 2048), (131072, 2048, 1), 0), out=buf92)
        buf95 = reinterpret_tensor(buf51, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf51  # reuse
        # Source Nodes: [mul_17, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf92, buf14, buf95, 98304, 2048, grid=grid(98304), stream=stream0)
        buf96 = reinterpret_tensor(buf91, (3, 16, 2048, 64), (2097152, 131072, 64, 1)); del buf91  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf83, buf96, 6291456, grid=grid(6291456), stream=stream0)
        buf97 = reinterpret_tensor(buf83, (48, 2048, 64), (131072, 64, 1)); del buf83  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf95, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf96, (48, 2048, 64), (131072, 64, 1), 0), out=buf97)
        buf98 = reinterpret_tensor(buf96, (3, 2048, 16, 64), (2097152, 1024, 64, 1)); del buf96  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf97, buf98, 6291456, grid=grid(6291456), stream=stream0)
        buf99 = reinterpret_tensor(buf97, (6144, 1024), (1024, 1)); del buf97  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf98, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_29, (1024, 1024), (1, 1024), 0), out=buf99)
        buf100 = reinterpret_tensor(buf99, (3, 2048, 1024), (2097152, 1024, 1)); del buf99  # reuse
        buf102 = reinterpret_tensor(buf98, (3, 2048, 1024), (2097152, 1024, 1)); del buf98  # reuse
        # Source Nodes: [add_14, add_15, add_18, add_19, add_3, add_5, float_6, mean_3, mul_12, mul_13, mul_18, mul_19, mul_20, rsqrt_3, type_as_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_per_fused__to_copy_add_mean_mul_rsqrt_18.run(buf100, buf71, primals_22, buf0, buf21, buf27, primals_30, buf102, 6144, 1024, grid=grid(6144), stream=stream0)
        buf103 = reinterpret_tensor(buf63, (6144, 2816), (2816, 1)); del buf63  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_31, (1024, 2816), (1, 1024), 0), out=buf103)
        buf104 = buf62; del buf62  # reuse
        # Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf102, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_32, (1024, 2816), (1, 1024), 0), out=buf104)
        buf105 = reinterpret_tensor(buf103, (3, 2048, 2816), (5767168, 2816, 1)); del buf103  # reuse
        # Source Nodes: [mul_21, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf105, buf104, 17301504, grid=grid(17301504), stream=stream0)
        buf106 = reinterpret_tensor(buf102, (6144, 1024), (1024, 1)); del buf102  # reuse
        # Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (6144, 2816), (2816, 1), 0), reinterpret_tensor(primals_33, (2816, 1024), (1, 2816), 0), out=buf106)
        print_mem(s='layer3mlp')
        buf108 = empty_strided((3, 2048, 1), (2048, 1, 1), device='cuda', dtype=torch.float16)
        buf109 = reinterpret_tensor(buf90, (6144, 1024), (1024, 1)); del buf90  # reuse
        # Source Nodes: [add_16, add_20, l__mod___dynamic_dense_2_w1, mean_2, mul_14, pow_3, rsqrt_2], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_19.run(buf100, buf106, buf108, buf109, 6144, 1024, grid=grid(6144), stream=stream0)
        buf110 = empty_strided((6144, 16), (16, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__mod___dynamic_dense_2_w1], Original ATen: [aten.mm]
        extern_kernels.mm(buf109, reinterpret_tensor(primals_53, (1024, 16), (1, 1024), 0), out=buf110)
        buf111 = empty_strided((6144, 16), (16, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__mod___dynamic_dense_2_act, l__mod___dynamic_dense_2_w2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_20.run(buf110, buf111, 98304, grid=grid(98304), stream=stream0)
        buf112 = empty_strided((6144, 16), (16, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__mod___dynamic_dense_2_w2], Original ATen: [aten.mm]
        extern_kernels.mm(buf111, reinterpret_tensor(primals_54, (16, 16), (1, 16), 0), out=buf112)
        buf116 = buf82; del buf82  # reuse
        buf120 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        buf122 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        buf124 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_18, add_19, add_20, add_21, add_22, add_23, add_24, add_25, add_26, add_27, add_28, add_29, add_3, add_5, float_1, float_2, float_3, mean, mean_1, mean_2, mul, mul_1, mul_15, mul_16, mul_17, mul_18, mul_19, mul_2, mul_20, mul_21, mul_22, mul_23, mul_24, mul_25, mul_26, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, rsqrt, rsqrt_1, rsqrt_2, type_as, type_as_1, type_as_2], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_per_fused__to_copy_add_mean_mul_rsqrt_21.run(buf112, primals_34, buf0, buf21, buf27, buf65, primals_35, primals_36, primals_37, buf116, buf120, buf122, buf124, 6144, 1024, grid=grid(6144), stream=stream0)
        buf121 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf120, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_38, (1024, 1024), (1, 1024), 0), out=buf121)
        buf123 = reinterpret_tensor(buf120, (6144, 1024), (1024, 1)); del buf120  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_39, (1024, 1024), (1, 1024), 0), out=buf123)
        buf125 = reinterpret_tensor(buf122, (6144, 1024), (1024, 1)); del buf122  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_40, (1024, 1024), (1, 1024), 0), out=buf125)
        del buf124
        buf128 = buf89; del buf89  # reuse
        buf126 = reinterpret_tensor(buf128, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 0)  # alias
        buf127 = reinterpret_tensor(buf128, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 1)  # alias
        buf131 = buf86; del buf86  # reuse
        buf129 = reinterpret_tensor(buf131, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 0)  # alias
        buf130 = reinterpret_tensor(buf131, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf121, primals_58, buf123, buf126, buf127, buf129, buf130, 3145728, grid=grid(3145728), stream=stream0)
        buf132 = reinterpret_tensor(buf123, (3, 16, 2048, 64), (2097152, 131072, 64, 1)); del buf123  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf128, buf132, 6291456, grid=grid(6291456), stream=stream0)
        del buf126
        del buf127
        del buf128
        del buf129
        del buf130
        buf133 = reinterpret_tensor(buf121, (3, 16, 64, 2048), (2097152, 131072, 2048, 1)); del buf121  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf131, buf133, 3072, 2048, grid=grid(3072, 2048), stream=stream0)
        del buf131
        buf134 = reinterpret_tensor(buf95, (48, 2048, 2048), (4194304, 2048, 1)); del buf95  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf132, (48, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf133, (48, 64, 2048), (131072, 2048, 1), 0), out=buf134)
        buf137 = reinterpret_tensor(buf92, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf92  # reuse
        # Source Nodes: [mul_17, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf134, buf14, buf137, 98304, 2048, grid=grid(98304), stream=stream0)
        del buf134
        buf138 = reinterpret_tensor(buf133, (3, 16, 2048, 64), (2097152, 131072, 64, 1)); del buf133  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf125, buf138, 6291456, grid=grid(6291456), stream=stream0)
        buf139 = reinterpret_tensor(buf125, (48, 2048, 64), (131072, 64, 1)); del buf125  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf137, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf138, (48, 2048, 64), (131072, 64, 1), 0), out=buf139)
        del buf137
        buf140 = reinterpret_tensor(buf138, (3, 2048, 16, 64), (2097152, 1024, 64, 1)); del buf138  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf139, buf140, 6291456, grid=grid(6291456), stream=stream0)
        buf141 = reinterpret_tensor(buf139, (6144, 1024), (1024, 1)); del buf139  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf140, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_41, (1024, 1024), (1, 1024), 0), out=buf141)
        buf143 = reinterpret_tensor(buf140, (3, 2048, 1024), (2097152, 1024, 1)); del buf140  # reuse
        # Source Nodes: [add_31, add_32, float_6, mean_3, mul_18, mul_19, mul_20, rsqrt_3, type_as_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_per_fused__to_copy_add_mean_mul_rsqrt_9.run(buf116, buf141, primals_42, buf143, 6144, 1024, grid=grid(6144), stream=stream0)
        buf144 = reinterpret_tensor(buf105, (6144, 2816), (2816, 1)); del buf105  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_43, (1024, 2816), (1, 1024), 0), out=buf144)
        buf145 = buf104; del buf104  # reuse
        # Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (6144, 1024), (1024, 1), 0), reinterpret_tensor(primals_44, (1024, 2816), (1, 1024), 0), out=buf145)
        buf146 = reinterpret_tensor(buf144, (3, 2048, 2816), (5767168, 2816, 1)); del buf144  # reuse
        # Source Nodes: [mul_21, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf146, buf145, 17301504, grid=grid(17301504), stream=stream0)
        del buf145
        buf147 = reinterpret_tensor(buf143, (6144, 1024), (1024, 1)); del buf143  # reuse
        # Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf146, (6144, 2816), (2816, 1), 0), reinterpret_tensor(primals_45, (2816, 1024), (1, 2816), 0), out=buf147)
        print_mem(s='layer4mlp')
        del buf146
        buf149 = empty_strided((3, 2048, 1), (2048, 1, 1), device='cuda', dtype=torch.float16)
        buf150 = reinterpret_tensor(buf132, (6144, 1024), (1024, 1)); del buf132  # reuse
        # Source Nodes: [add_30, add_31, add_33, l__mod___dynamic_dense_3_w1, mean_3, mul_27, pow_4, rsqrt_3], Original ATen: [aten.add, aten.mean, aten.mul, aten.pow, aten.rsqrt, aten.view]
        triton_per_fused_add_mean_mul_pow_rsqrt_view_11.run(buf116, buf141, buf147, buf149, buf150, 6144, 1024, grid=grid(6144), stream=stream0)
        del buf116
        del buf141
        buf151 = empty_strided((6144, 20), (20, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__mod___dynamic_dense_3_w1], Original ATen: [aten.mm]
        extern_kernels.mm(buf150, reinterpret_tensor(primals_55, (1024, 20), (1, 1024), 0), out=buf151)
        buf152 = empty_strided((6144, 20), (20, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__mod___dynamic_dense_3_act, l__mod___dynamic_dense_3_w2], Original ATen: [aten.gelu, aten.view]
        triton_poi_fused_gelu_view_22.run(buf151, buf152, 122880, grid=grid(122880), stream=stream0)
        buf153 = empty_strided((6144, 20), (20, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__mod___dynamic_dense_3_w2], Original ATen: [aten.mm]
        extern_kernels.mm(buf152, reinterpret_tensor(primals_56, (20, 20), (1, 20), 0), out=buf153)
        buf154 = reinterpret_tensor(buf21, (3, 2048, 1024), (2097152, 1024, 1)); del buf21  # reuse
        buf156 = empty_strided((3, 2048, 1), (2048, 1, 6144), device='cuda', dtype=torch.float32)
        buf157 = reinterpret_tensor(buf156, (3, 2048, 1), (2048, 1, 1)); del buf156  # reuse
        buf158 = buf147; del buf147  # reuse
        # Source Nodes: [add_20, add_3, add_36, add_37, add_38, add_39, add_48, add_5, float_1, l__mod___output, mean_4, mul_32, mul_33, mul_34, mul_35, mul_44, mul_45, mul_46, rsqrt_4, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.view]
        triton_per_fused__to_copy_add_mean_mul_rsqrt_view_23.run(buf154, buf157, buf153, primals_46, buf0, buf27, buf65, buf100, buf106, primals_47, buf158, 6144, 1024, grid=grid(6144), stream=stream0)
        print_mem(s='after_last_comp')
        del buf100
        del buf106
        del buf154
        del buf27
        del buf65
        buf161 = empty_strided((1024, 50264), (50264, 1), device='cuda', dtype=torch.float16)
        buf159 = reinterpret_tensor(buf161, (1024, 50257), (50264, 1), 0)  # alias
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_24.run(primals_57, buf159, 1024, 50257, grid=grid(1024, 50257), stream=stream0)
        buf160 = reinterpret_tensor(buf161, (1024, 7), (50264, 1), 50257)  # alias
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_25.run(buf160, 7168, grid=grid(7168), stream=stream0)
        del buf159
        del buf160
        buf162 = empty_strided((6144, 50264), (50264, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf158, buf161, out=buf162)
        del buf161
        buf165 = empty_strided((6144, 50257), (50257, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_26.run(buf162, buf165, 6144, 50257, grid=grid(6144), stream=stream0)
        print_mem(s='after_softmax')
        del buf162
        buf168 = empty_strided((), (), device='cuda', dtype=torch.float16)
        buf167 = empty_strided((), (), device='cuda', dtype=torch.float16)
        buf169 = buf168; del buf168  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_27.run(buf169, primals_61, buf165, buf167, 1, 6144, grid=grid(1), stream=stream0)
        buf170 = empty_strided((2048, 32, 2), (64, 2, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [getitem_1], Original ATen: [aten.index]
        triton_poi_fused_index_28.run(primals_58, buf170, 131072, grid=grid(131072), stream=stream0)
        del primals_58
        buf171 = empty_strided((4, 3, 2048, 2), (12288, 4096, 2, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_1], Original ATen: [aten.add]
        triton_poi_fused_add_29.run(buf33, primals_10, buf171, 49152, grid=grid(49152), stream=stream0)
        del buf33
        del primals_10
        buf172 = empty_strided((4, 3, 2048, 3), (18432, 6144, 3, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_7], Original ATen: [aten.add]
        triton_poi_fused_add_30.run(buf71, primals_22, buf172, 73728, grid=grid(73728), stream=stream0)
        del buf71
        del primals_22
        buf173 = empty_strided((4, 3, 2048, 4), (24576, 8192, 4, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_17], Original ATen: [aten.add]
        triton_poi_fused_add_31.run(buf112, primals_34, buf173, 98304, grid=grid(98304), stream=stream0)
        del buf112
        del primals_34
        buf174 = empty_strided((4, 3, 2048, 5), (30720, 10240, 5, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_31], Original ATen: [aten.add]
        triton_poi_fused_add_32.run(buf153, primals_46, buf174, 122880, grid=grid(122880), stream=stream0)
        del buf153
        del primals_46
        print_mem(s='end')
        print('------------------------')
        return (buf169, primals_6, primals_11, primals_12, primals_13, primals_18, primals_23, primals_24, primals_25, primals_30, primals_35, primals_36, primals_37, primals_42, primals_47, primals_60, primals_61, buf0, reinterpret_tensor(primals_2, (1024, 1024), (1, 1024), 0), reinterpret_tensor(primals_3, (1024, 1024), (1, 1024), 0), reinterpret_tensor(primals_4, (1024, 1024), (1, 1024), 0), reinterpret_tensor(buf170, (1, 2048, 1, 32), (0, 64, 0, 2), 0), reinterpret_tensor(buf170, (1, 2048, 1, 32), (0, 64, 0, 2), 1), buf10, buf14, reinterpret_tensor(primals_5, (1024, 1024), (1, 1024), 0), reinterpret_tensor(primals_7, (1024, 2816), (1, 1024), 0), reinterpret_tensor(primals_8, (1024, 2816), (1, 1024), 0), reinterpret_tensor(primals_9, (2816, 1024), (1, 2816), 0), buf29, buf30, buf31, buf32, reinterpret_tensor(buf171, (3, 2048, 1), (4096, 2, 0), 0), reinterpret_tensor(buf171, (3, 2048, 1), (4096, 2, 0), 12288), reinterpret_tensor(buf171, (3, 2048, 1), (4096, 2, 0), 24576), reinterpret_tensor(buf171, (3, 2048, 1), (4096, 2, 0), 36864), reinterpret_tensor(primals_14, (1024, 1024), (1, 1024), 0), reinterpret_tensor(primals_15, (1024, 1024), (1, 1024), 0), reinterpret_tensor(primals_16, (1024, 1024), (1, 1024), 0), reinterpret_tensor(primals_17, (1024, 1024), (1, 1024), 0), reinterpret_tensor(primals_19, (1024, 2816), (1, 1024), 0), reinterpret_tensor(primals_20, (1024, 2816), (1, 1024), 0), reinterpret_tensor(primals_21, (2816, 1024), (1, 2816), 0), buf67, buf68, buf69, buf70, reinterpret_tensor(buf172, (3, 2048, 1), (6144, 3, 0), 0), reinterpret_tensor(buf172, (3, 2048, 1), (6144, 3, 0), 1), reinterpret_tensor(buf172, (3, 2048, 1), (6144, 3, 0), 18432), reinterpret_tensor(buf172, (3, 2048, 1), (6144, 3, 0), 18433), reinterpret_tensor(buf172, (3, 2048, 1), (6144, 3, 0), 36864), reinterpret_tensor(buf172, (3, 2048, 1), (6144, 3, 0), 36865), reinterpret_tensor(buf172, (3, 2048, 1), (6144, 3, 0), 55296), reinterpret_tensor(buf172, (3, 2048, 1), (6144, 3, 0), 55297), reinterpret_tensor(primals_26, (1024, 1024), (1, 1024), 0), reinterpret_tensor(primals_27, (1024, 1024), (1, 1024), 0), reinterpret_tensor(primals_28, (1024, 1024), (1, 1024), 0), reinterpret_tensor(primals_29, (1024, 1024), (1, 1024), 0), reinterpret_tensor(primals_31, (1024, 2816), (1, 1024), 0), reinterpret_tensor(primals_32, (1024, 2816), (1, 1024), 0), reinterpret_tensor(primals_33, (2816, 1024), (1, 2816), 0), buf108, buf109, buf110, buf111, reinterpret_tensor(buf173, (3, 2048, 1), (8192, 4, 0), 0), reinterpret_tensor(buf173, (3, 2048, 1), (8192, 4, 0), 1), reinterpret_tensor(buf173, (3, 2048, 1), (8192, 4, 0), 2), reinterpret_tensor(buf173, (3, 2048, 1), (8192, 4, 0), 24576), reinterpret_tensor(buf173, (3, 2048, 1), (8192, 4, 0), 24577), reinterpret_tensor(buf173, (3, 2048, 1), (8192, 4, 0), 24578), reinterpret_tensor(buf173, (3, 2048, 1), (8192, 4, 0), 49152), reinterpret_tensor(buf173, (3, 2048, 1), (8192, 4, 0), 49153), reinterpret_tensor(buf173, (3, 2048, 1), (8192, 4, 0), 49154), reinterpret_tensor(buf173, (3, 2048, 1), (8192, 4, 0), 73728), reinterpret_tensor(buf173, (3, 2048, 1), (8192, 4, 0), 73729), reinterpret_tensor(buf173, (3, 2048, 1), (8192, 4, 0), 73730), reinterpret_tensor(primals_38, (1024, 1024), (1, 1024), 0), reinterpret_tensor(primals_39, (1024, 1024), (1, 1024), 0), reinterpret_tensor(primals_40, (1024, 1024), (1, 1024), 0), reinterpret_tensor(primals_41, (1024, 1024), (1, 1024), 0), reinterpret_tensor(primals_43, (1024, 2816), (1, 1024), 0), reinterpret_tensor(primals_44, (1024, 2816), (1, 1024), 0), reinterpret_tensor(primals_45, (2816, 1024), (1, 2816), 0), buf149, buf150, buf151, buf152, reinterpret_tensor(buf174, (3, 2048, 1), (10240, 5, 0), 30720), reinterpret_tensor(buf174, (3, 2048, 1), (10240, 5, 0), 30721), reinterpret_tensor(buf174, (3, 2048, 1), (10240, 5, 0), 30722), reinterpret_tensor(buf174, (3, 2048, 1), (10240, 5, 0), 30723), buf157, buf158, buf165, buf167, reinterpret_tensor(primals_57, (50257, 1024), (1024, 1), 0), reinterpret_tensor(primals_56, (20, 20), (20, 1), 0), reinterpret_tensor(primals_55, (20, 1024), (1024, 1), 0), reinterpret_tensor(primals_54, (16, 16), (16, 1), 0), reinterpret_tensor(primals_53, (16, 1024), (1024, 1), 0), reinterpret_tensor(primals_52, (12, 12), (12, 1), 0), reinterpret_tensor(primals_51, (12, 1024), (1024, 1), 0), reinterpret_tensor(primals_50, (8, 8), (8, 1), 0), reinterpret_tensor(primals_49, (8, 1024), (1024, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    times=10
    repeat=1
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_2 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_3 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_4 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_5 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_6 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_7 = rand_strided((2816, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_8 = rand_strided((2816, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_9 = rand_strided((1024, 2816), (2816, 1), device='cuda:0', dtype=torch.float16)
    primals_10 = rand_strided((4, 2), (2, 1), device='cuda:0', dtype=torch.float16)
    primals_11 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_12 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_13 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_14 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_15 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_16 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_17 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_18 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_19 = rand_strided((2816, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_20 = rand_strided((2816, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_21 = rand_strided((1024, 2816), (2816, 1), device='cuda:0', dtype=torch.float16)
    primals_22 = rand_strided((4, 3), (3, 1), device='cuda:0', dtype=torch.float16)
    primals_23 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_24 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_25 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_26 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_27 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_28 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_29 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_30 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_31 = rand_strided((2816, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_32 = rand_strided((2816, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_33 = rand_strided((1024, 2816), (2816, 1), device='cuda:0', dtype=torch.float16)
    primals_34 = rand_strided((4, 4), (4, 1), device='cuda:0', dtype=torch.float16)
    primals_35 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_36 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_37 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_38 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_39 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_40 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_41 = rand_strided((1024, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_42 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_43 = rand_strided((2816, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_44 = rand_strided((2816, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_45 = rand_strided((1024, 2816), (2816, 1), device='cuda:0', dtype=torch.float16)
    primals_46 = rand_strided((4, 5), (5, 1), device='cuda:0', dtype=torch.float16)
    primals_47 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_48 = rand_strided((50257, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_49 = rand_strided((8, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_50 = rand_strided((8, 8), (8, 1), device='cuda:0', dtype=torch.float16)
    primals_51 = rand_strided((12, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_52 = rand_strided((12, 12), (12, 1), device='cuda:0', dtype=torch.float16)
    primals_53 = rand_strided((16, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_54 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float16)
    primals_55 = rand_strided((20, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_56 = rand_strided((20, 20), (20, 1), device='cuda:0', dtype=torch.float16)
    primals_57 = rand_strided((50257, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    primals_58 = rand_strided((2048, 32, 2), (64, 2, 1), device='cuda:0', dtype=torch.float16) # rope freqs
    primals_59 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bool) # causal mask
    primals_60 = rand_strided((3, 2048), (2049, 1), device='cuda:0', dtype=torch.int64)
    primals_61 = rand_strided((3, 2048), (2049, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61]), times=times, repeat=repeat)

def print_mem(s=''):
    print(f"Memory used at {s}: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB, {torch.cuda.memory_allocated() / 1e9:.02f}, {torch.cuda.memory_allocated() / 1e9:.02f}")


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
