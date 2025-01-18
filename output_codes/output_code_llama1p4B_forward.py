
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

aten = torch.ops.aten
assert_size_stride = torch._C._dynamo.guards.assert_size_stride
reinterpret_tensor = torch.ops.inductor._reinterpret_tensor
async_compile = AsyncCompile()


# kernel path: /tmp/torchinductor_mengqy/6j/c6jdsrvopiferlte2jvjoto3nx7f2z4hy6exqg6c62mksq5fbp5q.py
# Source Nodes: [add, float_1, l__mod___tok_embeddings, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.rsqrt]
# add => add
# float_1 => convert_element_type
# l__mod___tok_embeddings => embedding
# mean => mean
# mul => mul
# mul_1 => mul_1
# mul_2 => mul_2
# rsqrt => rsqrt
# type_as => convert_element_type_1
triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_0 = async_compile.triton('triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_0', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i64', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_0', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    tmp0 = tl.load(in_ptr0 + (x0 + (2049*x1)), None, eviction_policy='evict_last')
    x3 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp1 = tl.where(tmp0 < 0, tmp0 + 50257, tmp0)
        tl.device_assert((0 <= tmp1) & (tmp1 < 50257), "index out of bounds: 0 <= tmp1 < 50257")
        tmp2 = tl.load(in_ptr1 + (r2 + (2048*tmp1)), rmask, other=0).to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
        tl.store(out_ptr0 + (r2 + (2048*x3)), tmp2, rmask)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp8 = tl.load(out_ptr0 + (r2 + (2048*x3)), rmask, other=0).to(tl.float32)
        tmp17 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = 2048.0
        tmp11 = tmp6 / tmp10
        tmp12 = 1e-05
        tmp13 = tmp11 + tmp12
        tmp14 = tl.math.rsqrt(tmp13)
        tmp15 = tmp9 * tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp18 = tmp16 * tmp17
        tl.store(out_ptr2 + (r2 + (2048*x3)), tmp18, rmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_mengqy/6u/c6u6xfynhtff5pferekved546f5ayc5epqvks237nzprxe4qjv2m.py
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

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_1', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]})
@triton.jit
def triton_poi_fused_stack_1(in_ptr0, in_ptr1, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 16
    x4 = (xindex // 1024)
    x2 = (xindex // 1024) % 2048
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x1) + (6144*x4)), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + ((2*x0) + (128*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (6144*x4)), None).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (1 + (2*x0) + (128*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr0 + (2048 + x0 + (128*x1) + (6144*x4)), None).to(tl.float32)
    tmp17 = tl.load(in_ptr0 + (2112 + x0 + (128*x1) + (6144*x4)), None).to(tl.float32)
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
# Source Nodes: [getitem, getitem_21], Original ATen: [aten.index, aten.slice]
# getitem => index
# getitem_21 => slice_1, slice_2, slice_3
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


# kernel path: /tmp/torchinductor_mengqy/jx/cjxmnbvtkhnnj5m2zsrdgjdqkfduuoi6puqrmikcaophraijyv4a.py
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

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_3', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_clone_3(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 2048
    x2 = (xindex // 262144) % 16
    x3 = (xindex // 4194304)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x2) + (2048*x1) + (4194304*x3)), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp1, None)
''')


# kernel path: /tmp/torchinductor_mengqy/x2/cx2jcqwqtxdccs6p34d5p2tkmz3y7bzxvcoyiowwydyex55ph24o.py
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

@pointwise(size_hints=[8192, 2048], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_4', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]})
@triton.jit
def triton_poi_fused_clone_4(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex
    y0 = yindex % 2048
    y1 = (yindex // 2048)
    y3 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (2048*x2) + (4194304*y1)), None, eviction_policy='evict_last')
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


# kernel path: /tmp/torchinductor_mengqy/t5/ct5z4dahsxo7e5klf23kp37l3f6g6bsm77ur2k5a6taay37u5qsk.py
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
        tmp2 = 0.08838834764831843
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
        tmp13 = 0.08838834764831843
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
        tmp26 = 0.08838834764831843
        tmp27 = tmp25 * tmp26
        tmp30 = tl.where(tmp24, tmp27, tmp29)
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tmp31 - tmp9
        tmp33 = tl.exp(tmp32)
        tmp34 = tmp33 / tmp22
        tmp35 = tmp34.to(tl.float32)
        tl.store(out_ptr2 + (r2 + (2048*x3)), tmp35, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/gn/cgn4gllxugume7i547o5jygt3isft63gaxa4ld24vxtn3lsvi5tu.py
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

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_clone_7(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 2048
    x2 = (xindex // 262144) % 16
    x3 = (xindex // 4194304)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (4096 + x0 + (128*x2) + (6144*x1) + (12582912*x3)), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/gf/cgfu5d2zzhdpctx5scw4bwn5gyc4xlfq7l2jig4xcyt6xwfhhofs.py
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

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_clone_8(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 16
    x2 = (xindex // 2048) % 2048
    x3 = (xindex // 4194304)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x2) + (262144*x1) + (4194304*x3)), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/ey/ceyoiaqzvzfnh4ycwsfqvxwrfliuckoadibmj54obytfk42gaifw.py
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
triton_red_fused__to_copy_add_mean_mul_rsqrt_9 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_9', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_9', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_9(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp3 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = 2048.0
        tmp13 = tmp6 / tmp12
        tmp14 = 1e-05
        tmp15 = tmp13 + tmp14
        tmp16 = tl.math.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 * tmp19
        tl.store(out_ptr1 + (r1 + (2048*x0)), tmp20, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/ig/cigb6teo34bxdyuftbxv77qurgj5kcsmjsd4qfovfylz47ghxkln.py
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

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_10', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_mul_silu_10(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 34603008
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


# kernel path: /tmp/torchinductor_mengqy/7j/c7j2s6fm74upbi5tw36alsqvksx5ruazn3y4tsjwjnx6glgg2hos.py
# Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add => add_6
# add_3 => add_3
# add_5 => add_5
# float_1 => convert_element_type_12
# mean => mean_2
# mul => mul_17
# mul_1 => mul_18
# mul_2 => mul_19
# rsqrt => rsqrt_2
# type_as => convert_element_type_13
triton_red_fused__to_copy_add_mean_mul_rsqrt_11 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_11', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_11', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_11(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp1 = tl.load(in_out_ptr0 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
        tl.store(in_out_ptr0 + (r1 + (2048*x0)), tmp4, rmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_out_ptr0 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = 2048.0
        tmp13 = tmp8 / tmp12
        tmp14 = 1e-05
        tmp15 = tmp13 + tmp14
        tmp16 = tl.math.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 * tmp19
        tl.store(out_ptr1 + (r1 + (2048*x0)), tmp20, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/xd/cxdtbkh7nwtk2w5fjpqznpe3ta2m3gqbebqjc32bkc2rm76gn3ch.py
# Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add => add_24
# add_3 => add_21
# add_5 => add_23
# float_1 => convert_element_type_48
# mean => mean_8
# mul => mul_68
# mul_1 => mul_69
# mul_2 => mul_70
# rsqrt => rsqrt_8
# type_as => convert_element_type_49
triton_red_fused__to_copy_add_mean_mul_rsqrt_12 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_12', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_12', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_out_ptr0 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
        tl.store(in_out_ptr0 + (r1 + (2048*x0)), tmp4, rmask)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_out_ptr0 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp11 = tmp10.to(tl.float32)
        tmp12 = 2048.0
        tmp13 = tmp8 / tmp12
        tmp14 = 1e-05
        tmp15 = tmp13 + tmp14
        tmp16 = tl.math.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 * tmp19
        tl.store(out_ptr1 + (r1 + (2048*x0)), tmp20, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/cg/ccgu4s6gx56c3kg74iepc66mgl7r5gaji4i7uh23y4nijufacyqi.py
# Source Nodes: [add, add_3, add_5, float_1, l__mod___output, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.view]
# add => add_144
# add_3 => add_141
# add_5 => add_143
# float_1 => convert_element_type_288
# l__mod___output => view_624
# mean => mean_48
# mul => mul_408
# mul_1 => mul_409
# mul_2 => mul_410
# rsqrt => rsqrt_48
# type_as => convert_element_type_289
triton_red_fused__to_copy_add_mean_mul_rsqrt_view_13 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_view_13', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8192, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_view_13', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_view_13(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp10 = 2048.0
    tmp11 = tmp8 / tmp10
    tmp12 = 1e-05
    tmp13 = tmp11 + tmp12
    tmp14 = tl.math.rsqrt(tmp13)
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x0), tmp14, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp15 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp16 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp23 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp17 = tmp15 + tmp16
        tmp19 = tmp17 + tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp21 = tmp20 * tmp14
        tmp22 = tmp21.to(tl.float32)
        tmp24 = tmp22 * tmp23
        tl.store(out_ptr0 + (r1 + (2048*x0)), tmp24, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/w5/cw5ca2k7v46daoje5n22rczrawzo3ez5jxkir6itvbgfjgjlmzuq.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_14 = async_compile.triton('triton_poi_fused_14', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[2048, 65536], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_14', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_14(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 2048
    xnumel = 50257
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (2048*x1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x1 + (50264*y0)), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/ti/ctierqqlakwr3zd7bptzlt5scqykh7ampdghfqyobyfkhi36xq55.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_15 = async_compile.triton('triton_poi_fused_15', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_15', 'configs': [instance_descriptor(divisible_by_16=(1,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]})
@triton.jit
def triton_poi_fused_15(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (50264*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/z5/cz5oenb343cvgj7txrderzmuvsrrqzgcyq7hmcz4jck7ltccjh7e.py
# Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax]
# cross_entropy => amax_24, convert_element_type_290, convert_element_type_291, exp_24, log, sub_72, sub_73, sum_25
triton_red_fused__log_softmax_16 = async_compile.triton('triton_red_fused__log_softmax_16', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_16', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]}
)
@triton.jit
def triton_red_fused__log_softmax_16(in_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_mengqy/bf/cbflwjvcv7d6bd2gyqvh7fy3xfrc3e4dmhdqsgjm2wgbd2ken4ku.py
# Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_forward]
# cross_entropy => convert_element_type_292, div_24, ne, neg, scalar_tensor_25, sum_26, sum_27, where_25
triton_red_fused_nll_loss_forward_17 = async_compile.triton('triton_red_fused_nll_loss_forward_17', '''
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
    meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_nll_loss_forward_17', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]}
)
@triton.jit
def triton_red_fused_nll_loss_forward_17(in_out_ptr0, in_ptr0, in_ptr1, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_mengqy/vl/cvlwlyc6jg55zndadss5a2sryzrqsw4kattxlms65g6q6rcqt6nu.py
# Source Nodes: [getitem_1], Original ATen: [aten.index]
# getitem_1 => index_1
triton_poi_fused_index_18 = async_compile.triton('triton_poi_fused_index_18', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[262144], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_18', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_index_18(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 262144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175 = args
    args.clear()
    assert_size_stride(primals_1, (2048, ), (1, ))
    assert_size_stride(primals_2, (6144, 2048), (2048, 1))
    assert_size_stride(primals_3, (2048, 2048), (2048, 1))
    assert_size_stride(primals_4, (2048, ), (1, ))
    assert_size_stride(primals_5, (5632, 2048), (2048, 1))
    assert_size_stride(primals_6, (5632, 2048), (2048, 1))
    assert_size_stride(primals_7, (2048, 5632), (5632, 1))
    assert_size_stride(primals_8, (2048, ), (1, ))
    assert_size_stride(primals_9, (6144, 2048), (2048, 1))
    assert_size_stride(primals_10, (2048, 2048), (2048, 1))
    assert_size_stride(primals_11, (2048, ), (1, ))
    assert_size_stride(primals_12, (5632, 2048), (2048, 1))
    assert_size_stride(primals_13, (5632, 2048), (2048, 1))
    assert_size_stride(primals_14, (2048, 5632), (5632, 1))
    assert_size_stride(primals_15, (2048, ), (1, ))
    assert_size_stride(primals_16, (6144, 2048), (2048, 1))
    assert_size_stride(primals_17, (2048, 2048), (2048, 1))
    assert_size_stride(primals_18, (2048, ), (1, ))
    assert_size_stride(primals_19, (5632, 2048), (2048, 1))
    assert_size_stride(primals_20, (5632, 2048), (2048, 1))
    assert_size_stride(primals_21, (2048, 5632), (5632, 1))
    assert_size_stride(primals_22, (2048, ), (1, ))
    assert_size_stride(primals_23, (6144, 2048), (2048, 1))
    assert_size_stride(primals_24, (2048, 2048), (2048, 1))
    assert_size_stride(primals_25, (2048, ), (1, ))
    assert_size_stride(primals_26, (5632, 2048), (2048, 1))
    assert_size_stride(primals_27, (5632, 2048), (2048, 1))
    assert_size_stride(primals_28, (2048, 5632), (5632, 1))
    assert_size_stride(primals_29, (2048, ), (1, ))
    assert_size_stride(primals_30, (6144, 2048), (2048, 1))
    assert_size_stride(primals_31, (2048, 2048), (2048, 1))
    assert_size_stride(primals_32, (2048, ), (1, ))
    assert_size_stride(primals_33, (5632, 2048), (2048, 1))
    assert_size_stride(primals_34, (5632, 2048), (2048, 1))
    assert_size_stride(primals_35, (2048, 5632), (5632, 1))
    assert_size_stride(primals_36, (2048, ), (1, ))
    assert_size_stride(primals_37, (6144, 2048), (2048, 1))
    assert_size_stride(primals_38, (2048, 2048), (2048, 1))
    assert_size_stride(primals_39, (2048, ), (1, ))
    assert_size_stride(primals_40, (5632, 2048), (2048, 1))
    assert_size_stride(primals_41, (5632, 2048), (2048, 1))
    assert_size_stride(primals_42, (2048, 5632), (5632, 1))
    assert_size_stride(primals_43, (2048, ), (1, ))
    assert_size_stride(primals_44, (6144, 2048), (2048, 1))
    assert_size_stride(primals_45, (2048, 2048), (2048, 1))
    assert_size_stride(primals_46, (2048, ), (1, ))
    assert_size_stride(primals_47, (5632, 2048), (2048, 1))
    assert_size_stride(primals_48, (5632, 2048), (2048, 1))
    assert_size_stride(primals_49, (2048, 5632), (5632, 1))
    assert_size_stride(primals_50, (2048, ), (1, ))
    assert_size_stride(primals_51, (6144, 2048), (2048, 1))
    assert_size_stride(primals_52, (2048, 2048), (2048, 1))
    assert_size_stride(primals_53, (2048, ), (1, ))
    assert_size_stride(primals_54, (5632, 2048), (2048, 1))
    assert_size_stride(primals_55, (5632, 2048), (2048, 1))
    assert_size_stride(primals_56, (2048, 5632), (5632, 1))
    assert_size_stride(primals_57, (2048, ), (1, ))
    assert_size_stride(primals_58, (6144, 2048), (2048, 1))
    assert_size_stride(primals_59, (2048, 2048), (2048, 1))
    assert_size_stride(primals_60, (2048, ), (1, ))
    assert_size_stride(primals_61, (5632, 2048), (2048, 1))
    assert_size_stride(primals_62, (5632, 2048), (2048, 1))
    assert_size_stride(primals_63, (2048, 5632), (5632, 1))
    assert_size_stride(primals_64, (2048, ), (1, ))
    assert_size_stride(primals_65, (6144, 2048), (2048, 1))
    assert_size_stride(primals_66, (2048, 2048), (2048, 1))
    assert_size_stride(primals_67, (2048, ), (1, ))
    assert_size_stride(primals_68, (5632, 2048), (2048, 1))
    assert_size_stride(primals_69, (5632, 2048), (2048, 1))
    assert_size_stride(primals_70, (2048, 5632), (5632, 1))
    assert_size_stride(primals_71, (2048, ), (1, ))
    assert_size_stride(primals_72, (6144, 2048), (2048, 1))
    assert_size_stride(primals_73, (2048, 2048), (2048, 1))
    assert_size_stride(primals_74, (2048, ), (1, ))
    assert_size_stride(primals_75, (5632, 2048), (2048, 1))
    assert_size_stride(primals_76, (5632, 2048), (2048, 1))
    assert_size_stride(primals_77, (2048, 5632), (5632, 1))
    assert_size_stride(primals_78, (2048, ), (1, ))
    assert_size_stride(primals_79, (6144, 2048), (2048, 1))
    assert_size_stride(primals_80, (2048, 2048), (2048, 1))
    assert_size_stride(primals_81, (2048, ), (1, ))
    assert_size_stride(primals_82, (5632, 2048), (2048, 1))
    assert_size_stride(primals_83, (5632, 2048), (2048, 1))
    assert_size_stride(primals_84, (2048, 5632), (5632, 1))
    assert_size_stride(primals_85, (2048, ), (1, ))
    assert_size_stride(primals_86, (6144, 2048), (2048, 1))
    assert_size_stride(primals_87, (2048, 2048), (2048, 1))
    assert_size_stride(primals_88, (2048, ), (1, ))
    assert_size_stride(primals_89, (5632, 2048), (2048, 1))
    assert_size_stride(primals_90, (5632, 2048), (2048, 1))
    assert_size_stride(primals_91, (2048, 5632), (5632, 1))
    assert_size_stride(primals_92, (2048, ), (1, ))
    assert_size_stride(primals_93, (6144, 2048), (2048, 1))
    assert_size_stride(primals_94, (2048, 2048), (2048, 1))
    assert_size_stride(primals_95, (2048, ), (1, ))
    assert_size_stride(primals_96, (5632, 2048), (2048, 1))
    assert_size_stride(primals_97, (5632, 2048), (2048, 1))
    assert_size_stride(primals_98, (2048, 5632), (5632, 1))
    assert_size_stride(primals_99, (2048, ), (1, ))
    assert_size_stride(primals_100, (6144, 2048), (2048, 1))
    assert_size_stride(primals_101, (2048, 2048), (2048, 1))
    assert_size_stride(primals_102, (2048, ), (1, ))
    assert_size_stride(primals_103, (5632, 2048), (2048, 1))
    assert_size_stride(primals_104, (5632, 2048), (2048, 1))
    assert_size_stride(primals_105, (2048, 5632), (5632, 1))
    assert_size_stride(primals_106, (2048, ), (1, ))
    assert_size_stride(primals_107, (6144, 2048), (2048, 1))
    assert_size_stride(primals_108, (2048, 2048), (2048, 1))
    assert_size_stride(primals_109, (2048, ), (1, ))
    assert_size_stride(primals_110, (5632, 2048), (2048, 1))
    assert_size_stride(primals_111, (5632, 2048), (2048, 1))
    assert_size_stride(primals_112, (2048, 5632), (5632, 1))
    assert_size_stride(primals_113, (2048, ), (1, ))
    assert_size_stride(primals_114, (6144, 2048), (2048, 1))
    assert_size_stride(primals_115, (2048, 2048), (2048, 1))
    assert_size_stride(primals_116, (2048, ), (1, ))
    assert_size_stride(primals_117, (5632, 2048), (2048, 1))
    assert_size_stride(primals_118, (5632, 2048), (2048, 1))
    assert_size_stride(primals_119, (2048, 5632), (5632, 1))
    assert_size_stride(primals_120, (2048, ), (1, ))
    assert_size_stride(primals_121, (6144, 2048), (2048, 1))
    assert_size_stride(primals_122, (2048, 2048), (2048, 1))
    assert_size_stride(primals_123, (2048, ), (1, ))
    assert_size_stride(primals_124, (5632, 2048), (2048, 1))
    assert_size_stride(primals_125, (5632, 2048), (2048, 1))
    assert_size_stride(primals_126, (2048, 5632), (5632, 1))
    assert_size_stride(primals_127, (2048, ), (1, ))
    assert_size_stride(primals_128, (6144, 2048), (2048, 1))
    assert_size_stride(primals_129, (2048, 2048), (2048, 1))
    assert_size_stride(primals_130, (2048, ), (1, ))
    assert_size_stride(primals_131, (5632, 2048), (2048, 1))
    assert_size_stride(primals_132, (5632, 2048), (2048, 1))
    assert_size_stride(primals_133, (2048, 5632), (5632, 1))
    assert_size_stride(primals_134, (2048, ), (1, ))
    assert_size_stride(primals_135, (6144, 2048), (2048, 1))
    assert_size_stride(primals_136, (2048, 2048), (2048, 1))
    assert_size_stride(primals_137, (2048, ), (1, ))
    assert_size_stride(primals_138, (5632, 2048), (2048, 1))
    assert_size_stride(primals_139, (5632, 2048), (2048, 1))
    assert_size_stride(primals_140, (2048, 5632), (5632, 1))
    assert_size_stride(primals_141, (2048, ), (1, ))
    assert_size_stride(primals_142, (6144, 2048), (2048, 1))
    assert_size_stride(primals_143, (2048, 2048), (2048, 1))
    assert_size_stride(primals_144, (2048, ), (1, ))
    assert_size_stride(primals_145, (5632, 2048), (2048, 1))
    assert_size_stride(primals_146, (5632, 2048), (2048, 1))
    assert_size_stride(primals_147, (2048, 5632), (5632, 1))
    assert_size_stride(primals_148, (2048, ), (1, ))
    assert_size_stride(primals_149, (6144, 2048), (2048, 1))
    assert_size_stride(primals_150, (2048, 2048), (2048, 1))
    assert_size_stride(primals_151, (2048, ), (1, ))
    assert_size_stride(primals_152, (5632, 2048), (2048, 1))
    assert_size_stride(primals_153, (5632, 2048), (2048, 1))
    assert_size_stride(primals_154, (2048, 5632), (5632, 1))
    assert_size_stride(primals_155, (2048, ), (1, ))
    assert_size_stride(primals_156, (6144, 2048), (2048, 1))
    assert_size_stride(primals_157, (2048, 2048), (2048, 1))
    assert_size_stride(primals_158, (2048, ), (1, ))
    assert_size_stride(primals_159, (5632, 2048), (2048, 1))
    assert_size_stride(primals_160, (5632, 2048), (2048, 1))
    assert_size_stride(primals_161, (2048, 5632), (5632, 1))
    assert_size_stride(primals_162, (2048, ), (1, ))
    assert_size_stride(primals_163, (6144, 2048), (2048, 1))
    assert_size_stride(primals_164, (2048, 2048), (2048, 1))
    assert_size_stride(primals_165, (2048, ), (1, ))
    assert_size_stride(primals_166, (5632, 2048), (2048, 1))
    assert_size_stride(primals_167, (5632, 2048), (2048, 1))
    assert_size_stride(primals_168, (2048, 5632), (5632, 1))
    assert_size_stride(primals_169, (2048, ), (1, ))
    assert_size_stride(primals_170, (50257, 2048), (2048, 1))
    assert_size_stride(primals_171, (50257, 2048), (2048, 1))
    assert_size_stride(primals_172, (2048, 64, 2), (128, 2, 1))
    assert_size_stride(primals_173, (2048, 2048), (2048, 1))
    assert_size_stride(primals_174, (3, 2048), (2049, 1))
    assert_size_stride(primals_175, (3, 2048), (2049, 1))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        buf2 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, l__mod___tok_embeddings, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.rsqrt]
        stream0 = get_cuda_stream(0)
        triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_0.run(primals_174, primals_170, primals_1, buf0, buf2, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_170
        buf3 = empty_strided((6144, 6144), (6144, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf2, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_2, (2048, 6144), (1, 2048), 0), out=buf3)
        buf6 = empty_strided((3, 2048, 16, 64, 2), (4194304, 2048, 128, 2, 1), device='cuda', dtype=torch.float32)
        buf4 = reinterpret_tensor(buf6, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf5 = reinterpret_tensor(buf6, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf9 = empty_strided((3, 2048, 16, 64, 2), (4194304, 2048, 128, 2, 1), device='cuda', dtype=torch.float32)
        buf7 = reinterpret_tensor(buf9, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf8 = reinterpret_tensor(buf9, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf3, primals_172, buf4, buf5, buf7, buf8, 6291456, grid=grid(6291456), stream=stream0)
        buf10 = empty_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda', dtype=torch.bool)
        # Source Nodes: [getitem, getitem_21], Original ATen: [aten.index, aten.slice]
        triton_poi_fused_index_slice_2.run(primals_173, buf10, 4194304, grid=grid(4194304), stream=stream0)
        del buf4
        del buf5
        del buf7
        del buf8
        del primals_173
        buf11 = reinterpret_tensor(buf2, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf2  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf6, buf11, 12582912, grid=grid(12582912), stream=stream0)
        buf12 = empty_strided((3, 16, 128, 2048), (4194304, 262144, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf9, buf12, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf13 = empty_strided((48, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf11, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf12, (48, 128, 2048), (262144, 2048, 1), 0), out=buf13)
        buf14 = empty_strided((), (), device='cuda', dtype=torch.float16)
        # Source Nodes: [where], Original ATen: [aten.scalar_tensor]
        triton_poi_fused_scalar_tensor_5.run(buf14, 1, grid=grid(1), stream=stream0)
        buf17 = empty_strided((3, 16, 2048, 2048), (67108864, 4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf13, buf14, buf17, 98304, 2048, grid=grid(98304), stream=stream0)
        buf18 = reinterpret_tensor(buf12, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf12  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf3, buf18, 12582912, grid=grid(12582912), stream=stream0)
        buf19 = reinterpret_tensor(buf11, (48, 2048, 128), (262144, 128, 1)); del buf11  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf17, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf18, (48, 2048, 128), (262144, 128, 1), 0), out=buf19)
        buf20 = reinterpret_tensor(buf18, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf18  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf19, buf20, 12582912, grid=grid(12582912), stream=stream0)
        buf21 = reinterpret_tensor(buf19, (6144, 2048), (2048, 1)); del buf19  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_3, (2048, 2048), (1, 2048), 0), out=buf21)
        buf23 = reinterpret_tensor(buf20, (3, 2048, 2048), (4194304, 2048, 1)); del buf20  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf0, buf21, primals_4, buf23, 6144, 2048, grid=grid(6144), stream=stream0)
        buf24 = empty_strided((6144, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_5, (2048, 5632), (1, 2048), 0), out=buf24)
        buf25 = empty_strided((6144, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_6, (2048, 5632), (1, 2048), 0), out=buf25)
        buf26 = reinterpret_tensor(buf24, (3, 2048, 5632), (11534336, 5632, 1)); del buf24  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf26, buf25, 34603008, grid=grid(34603008), stream=stream0)
        buf27 = reinterpret_tensor(buf23, (6144, 2048), (2048, 1)); del buf23  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_7, (5632, 2048), (1, 5632), 0), out=buf27)
        buf28 = reinterpret_tensor(buf21, (3, 2048, 2048), (4194304, 2048, 1)); del buf21  # reuse
        buf30 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf28, buf0, buf27, primals_8, buf30, 6144, 2048, grid=grid(6144), stream=stream0)
        buf31 = buf3; del buf3  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf30, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_9, (2048, 6144), (1, 2048), 0), out=buf31)
        buf34 = buf9; del buf9  # reuse
        buf32 = reinterpret_tensor(buf34, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf33 = reinterpret_tensor(buf34, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf37 = buf6; del buf6  # reuse
        buf35 = reinterpret_tensor(buf37, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf36 = reinterpret_tensor(buf37, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf31, primals_172, buf32, buf33, buf35, buf36, 6291456, grid=grid(6291456), stream=stream0)
        buf38 = reinterpret_tensor(buf30, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf30  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf34, buf38, 12582912, grid=grid(12582912), stream=stream0)
        del buf32
        del buf33
        del buf35
        del buf36
        buf39 = reinterpret_tensor(buf27, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf27  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf37, buf39, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf40 = reinterpret_tensor(buf17, (48, 2048, 2048), (4194304, 2048, 1)); del buf17  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf38, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf39, (48, 128, 2048), (262144, 2048, 1), 0), out=buf40)
        buf43 = reinterpret_tensor(buf13, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf13  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf40, buf14, buf43, 98304, 2048, grid=grid(98304), stream=stream0)
        buf44 = reinterpret_tensor(buf39, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf39  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf31, buf44, 12582912, grid=grid(12582912), stream=stream0)
        buf45 = reinterpret_tensor(buf38, (48, 2048, 128), (262144, 128, 1)); del buf38  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf43, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf44, (48, 2048, 128), (262144, 128, 1), 0), out=buf45)
        buf46 = reinterpret_tensor(buf44, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf44  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf45, buf46, 12582912, grid=grid(12582912), stream=stream0)
        buf47 = reinterpret_tensor(buf45, (6144, 2048), (2048, 1)); del buf45  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf46, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_10, (2048, 2048), (1, 2048), 0), out=buf47)
        buf49 = reinterpret_tensor(buf46, (3, 2048, 2048), (4194304, 2048, 1)); del buf46  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf28, buf47, primals_11, buf49, 6144, 2048, grid=grid(6144), stream=stream0)
        buf50 = reinterpret_tensor(buf26, (6144, 5632), (5632, 1)); del buf26  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_12, (2048, 5632), (1, 2048), 0), out=buf50)
        buf51 = buf25; del buf25  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf49, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_13, (2048, 5632), (1, 2048), 0), out=buf51)
        buf52 = reinterpret_tensor(buf50, (3, 2048, 5632), (11534336, 5632, 1)); del buf50  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf52, buf51, 34603008, grid=grid(34603008), stream=stream0)
        buf53 = reinterpret_tensor(buf49, (6144, 2048), (2048, 1)); del buf49  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_14, (5632, 2048), (1, 5632), 0), out=buf53)
        buf54 = reinterpret_tensor(buf47, (3, 2048, 2048), (4194304, 2048, 1)); del buf47  # reuse
        buf56 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf54, buf28, buf53, primals_15, buf56, 6144, 2048, grid=grid(6144), stream=stream0)
        buf57 = buf31; del buf31  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf56, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_16, (2048, 6144), (1, 2048), 0), out=buf57)
        buf60 = buf37; del buf37  # reuse
        buf58 = reinterpret_tensor(buf60, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf59 = reinterpret_tensor(buf60, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf63 = buf34; del buf34  # reuse
        buf61 = reinterpret_tensor(buf63, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf62 = reinterpret_tensor(buf63, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf57, primals_172, buf58, buf59, buf61, buf62, 6291456, grid=grid(6291456), stream=stream0)
        buf64 = reinterpret_tensor(buf56, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf56  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf60, buf64, 12582912, grid=grid(12582912), stream=stream0)
        del buf58
        del buf59
        del buf61
        del buf62
        buf65 = reinterpret_tensor(buf53, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf53  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf63, buf65, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf66 = reinterpret_tensor(buf43, (48, 2048, 2048), (4194304, 2048, 1)); del buf43  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf64, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf65, (48, 128, 2048), (262144, 2048, 1), 0), out=buf66)
        buf69 = reinterpret_tensor(buf40, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf40  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf66, buf14, buf69, 98304, 2048, grid=grid(98304), stream=stream0)
        buf70 = reinterpret_tensor(buf65, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf65  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf57, buf70, 12582912, grid=grid(12582912), stream=stream0)
        buf71 = reinterpret_tensor(buf64, (48, 2048, 128), (262144, 128, 1)); del buf64  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf69, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf70, (48, 2048, 128), (262144, 128, 1), 0), out=buf71)
        buf72 = reinterpret_tensor(buf70, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf70  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf71, buf72, 12582912, grid=grid(12582912), stream=stream0)
        buf73 = reinterpret_tensor(buf71, (6144, 2048), (2048, 1)); del buf71  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_17, (2048, 2048), (1, 2048), 0), out=buf73)
        buf75 = reinterpret_tensor(buf72, (3, 2048, 2048), (4194304, 2048, 1)); del buf72  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf54, buf73, primals_18, buf75, 6144, 2048, grid=grid(6144), stream=stream0)
        buf76 = reinterpret_tensor(buf52, (6144, 5632), (5632, 1)); del buf52  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf75, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_19, (2048, 5632), (1, 2048), 0), out=buf76)
        buf77 = buf51; del buf51  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf75, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_20, (2048, 5632), (1, 2048), 0), out=buf77)
        buf78 = reinterpret_tensor(buf76, (3, 2048, 5632), (11534336, 5632, 1)); del buf76  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf78, buf77, 34603008, grid=grid(34603008), stream=stream0)
        buf79 = reinterpret_tensor(buf75, (6144, 2048), (2048, 1)); del buf75  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf78, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_21, (5632, 2048), (1, 5632), 0), out=buf79)
        buf80 = reinterpret_tensor(buf73, (3, 2048, 2048), (4194304, 2048, 1)); del buf73  # reuse
        buf82 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf80, buf54, buf79, primals_22, buf82, 6144, 2048, grid=grid(6144), stream=stream0)
        buf83 = buf57; del buf57  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf82, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_23, (2048, 6144), (1, 2048), 0), out=buf83)
        buf86 = buf63; del buf63  # reuse
        buf84 = reinterpret_tensor(buf86, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf85 = reinterpret_tensor(buf86, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf89 = buf60; del buf60  # reuse
        buf87 = reinterpret_tensor(buf89, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf88 = reinterpret_tensor(buf89, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf83, primals_172, buf84, buf85, buf87, buf88, 6291456, grid=grid(6291456), stream=stream0)
        buf90 = reinterpret_tensor(buf82, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf82  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf86, buf90, 12582912, grid=grid(12582912), stream=stream0)
        del buf84
        del buf85
        del buf87
        del buf88
        buf91 = reinterpret_tensor(buf79, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf79  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf89, buf91, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf92 = reinterpret_tensor(buf69, (48, 2048, 2048), (4194304, 2048, 1)); del buf69  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf90, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf91, (48, 128, 2048), (262144, 2048, 1), 0), out=buf92)
        buf95 = reinterpret_tensor(buf66, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf66  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf92, buf14, buf95, 98304, 2048, grid=grid(98304), stream=stream0)
        buf96 = reinterpret_tensor(buf91, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf91  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf83, buf96, 12582912, grid=grid(12582912), stream=stream0)
        buf97 = reinterpret_tensor(buf90, (48, 2048, 128), (262144, 128, 1)); del buf90  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf95, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf96, (48, 2048, 128), (262144, 128, 1), 0), out=buf97)
        buf98 = reinterpret_tensor(buf96, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf96  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf97, buf98, 12582912, grid=grid(12582912), stream=stream0)
        buf99 = reinterpret_tensor(buf97, (6144, 2048), (2048, 1)); del buf97  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf98, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_24, (2048, 2048), (1, 2048), 0), out=buf99)
        buf101 = reinterpret_tensor(buf98, (3, 2048, 2048), (4194304, 2048, 1)); del buf98  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf80, buf99, primals_25, buf101, 6144, 2048, grid=grid(6144), stream=stream0)
        buf102 = reinterpret_tensor(buf78, (6144, 5632), (5632, 1)); del buf78  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_26, (2048, 5632), (1, 2048), 0), out=buf102)
        buf103 = buf77; del buf77  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_27, (2048, 5632), (1, 2048), 0), out=buf103)
        buf104 = reinterpret_tensor(buf102, (3, 2048, 5632), (11534336, 5632, 1)); del buf102  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf104, buf103, 34603008, grid=grid(34603008), stream=stream0)
        buf105 = reinterpret_tensor(buf101, (6144, 2048), (2048, 1)); del buf101  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf104, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_28, (5632, 2048), (1, 5632), 0), out=buf105)
        buf106 = reinterpret_tensor(buf105, (3, 2048, 2048), (4194304, 2048, 1)); del buf105  # reuse
        buf108 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_12.run(buf106, buf80, buf99, primals_29, buf108, 6144, 2048, grid=grid(6144), stream=stream0)
        buf109 = buf83; del buf83  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_30, (2048, 6144), (1, 2048), 0), out=buf109)
        buf112 = buf89; del buf89  # reuse
        buf110 = reinterpret_tensor(buf112, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf111 = reinterpret_tensor(buf112, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf115 = buf86; del buf86  # reuse
        buf113 = reinterpret_tensor(buf115, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf114 = reinterpret_tensor(buf115, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf109, primals_172, buf110, buf111, buf113, buf114, 6291456, grid=grid(6291456), stream=stream0)
        buf116 = reinterpret_tensor(buf108, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf108  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf112, buf116, 12582912, grid=grid(12582912), stream=stream0)
        del buf110
        del buf111
        del buf113
        del buf114
        buf117 = reinterpret_tensor(buf99, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf99  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf115, buf117, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf118 = reinterpret_tensor(buf95, (48, 2048, 2048), (4194304, 2048, 1)); del buf95  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf116, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf117, (48, 128, 2048), (262144, 2048, 1), 0), out=buf118)
        buf121 = reinterpret_tensor(buf92, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf92  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf118, buf14, buf121, 98304, 2048, grid=grid(98304), stream=stream0)
        buf122 = reinterpret_tensor(buf117, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf117  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf109, buf122, 12582912, grid=grid(12582912), stream=stream0)
        buf123 = reinterpret_tensor(buf116, (48, 2048, 128), (262144, 128, 1)); del buf116  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf121, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf122, (48, 2048, 128), (262144, 128, 1), 0), out=buf123)
        buf124 = reinterpret_tensor(buf122, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf122  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf123, buf124, 12582912, grid=grid(12582912), stream=stream0)
        buf125 = reinterpret_tensor(buf123, (6144, 2048), (2048, 1)); del buf123  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf124, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_31, (2048, 2048), (1, 2048), 0), out=buf125)
        buf127 = reinterpret_tensor(buf124, (3, 2048, 2048), (4194304, 2048, 1)); del buf124  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf106, buf125, primals_32, buf127, 6144, 2048, grid=grid(6144), stream=stream0)
        buf128 = reinterpret_tensor(buf104, (6144, 5632), (5632, 1)); del buf104  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_33, (2048, 5632), (1, 2048), 0), out=buf128)
        buf129 = buf103; del buf103  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf127, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_34, (2048, 5632), (1, 2048), 0), out=buf129)
        buf130 = reinterpret_tensor(buf128, (3, 2048, 5632), (11534336, 5632, 1)); del buf128  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf130, buf129, 34603008, grid=grid(34603008), stream=stream0)
        buf131 = reinterpret_tensor(buf127, (6144, 2048), (2048, 1)); del buf127  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf130, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_35, (5632, 2048), (1, 5632), 0), out=buf131)
        buf132 = reinterpret_tensor(buf125, (3, 2048, 2048), (4194304, 2048, 1)); del buf125  # reuse
        buf134 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf132, buf106, buf131, primals_36, buf134, 6144, 2048, grid=grid(6144), stream=stream0)
        buf135 = buf109; del buf109  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf134, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_37, (2048, 6144), (1, 2048), 0), out=buf135)
        buf138 = buf115; del buf115  # reuse
        buf136 = reinterpret_tensor(buf138, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf137 = reinterpret_tensor(buf138, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf141 = buf112; del buf112  # reuse
        buf139 = reinterpret_tensor(buf141, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf140 = reinterpret_tensor(buf141, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf135, primals_172, buf136, buf137, buf139, buf140, 6291456, grid=grid(6291456), stream=stream0)
        buf142 = reinterpret_tensor(buf134, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf134  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf138, buf142, 12582912, grid=grid(12582912), stream=stream0)
        del buf136
        del buf137
        del buf139
        del buf140
        buf143 = reinterpret_tensor(buf131, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf131  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf141, buf143, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf144 = reinterpret_tensor(buf121, (48, 2048, 2048), (4194304, 2048, 1)); del buf121  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf142, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf143, (48, 128, 2048), (262144, 2048, 1), 0), out=buf144)
        buf147 = reinterpret_tensor(buf118, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf118  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf144, buf14, buf147, 98304, 2048, grid=grid(98304), stream=stream0)
        buf148 = reinterpret_tensor(buf143, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf143  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf135, buf148, 12582912, grid=grid(12582912), stream=stream0)
        buf149 = reinterpret_tensor(buf142, (48, 2048, 128), (262144, 128, 1)); del buf142  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf147, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf148, (48, 2048, 128), (262144, 128, 1), 0), out=buf149)
        buf150 = reinterpret_tensor(buf148, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf148  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf149, buf150, 12582912, grid=grid(12582912), stream=stream0)
        buf151 = reinterpret_tensor(buf149, (6144, 2048), (2048, 1)); del buf149  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf150, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_38, (2048, 2048), (1, 2048), 0), out=buf151)
        buf153 = reinterpret_tensor(buf150, (3, 2048, 2048), (4194304, 2048, 1)); del buf150  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf132, buf151, primals_39, buf153, 6144, 2048, grid=grid(6144), stream=stream0)
        buf154 = reinterpret_tensor(buf130, (6144, 5632), (5632, 1)); del buf130  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf153, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_40, (2048, 5632), (1, 2048), 0), out=buf154)
        buf155 = buf129; del buf129  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf153, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_41, (2048, 5632), (1, 2048), 0), out=buf155)
        buf156 = reinterpret_tensor(buf154, (3, 2048, 5632), (11534336, 5632, 1)); del buf154  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf156, buf155, 34603008, grid=grid(34603008), stream=stream0)
        buf157 = reinterpret_tensor(buf153, (6144, 2048), (2048, 1)); del buf153  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf156, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_42, (5632, 2048), (1, 5632), 0), out=buf157)
        buf158 = reinterpret_tensor(buf151, (3, 2048, 2048), (4194304, 2048, 1)); del buf151  # reuse
        buf160 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf158, buf132, buf157, primals_43, buf160, 6144, 2048, grid=grid(6144), stream=stream0)
        buf161 = buf135; del buf135  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf160, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_44, (2048, 6144), (1, 2048), 0), out=buf161)
        buf164 = buf141; del buf141  # reuse
        buf162 = reinterpret_tensor(buf164, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf163 = reinterpret_tensor(buf164, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf167 = buf138; del buf138  # reuse
        buf165 = reinterpret_tensor(buf167, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf166 = reinterpret_tensor(buf167, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf161, primals_172, buf162, buf163, buf165, buf166, 6291456, grid=grid(6291456), stream=stream0)
        buf168 = reinterpret_tensor(buf160, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf160  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf164, buf168, 12582912, grid=grid(12582912), stream=stream0)
        del buf162
        del buf163
        del buf165
        del buf166
        buf169 = reinterpret_tensor(buf157, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf157  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf167, buf169, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf170 = reinterpret_tensor(buf147, (48, 2048, 2048), (4194304, 2048, 1)); del buf147  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf168, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf169, (48, 128, 2048), (262144, 2048, 1), 0), out=buf170)
        buf173 = reinterpret_tensor(buf144, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf144  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf170, buf14, buf173, 98304, 2048, grid=grid(98304), stream=stream0)
        buf174 = reinterpret_tensor(buf169, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf169  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf161, buf174, 12582912, grid=grid(12582912), stream=stream0)
        buf175 = reinterpret_tensor(buf168, (48, 2048, 128), (262144, 128, 1)); del buf168  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf173, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf174, (48, 2048, 128), (262144, 128, 1), 0), out=buf175)
        buf176 = reinterpret_tensor(buf174, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf174  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf175, buf176, 12582912, grid=grid(12582912), stream=stream0)
        buf177 = reinterpret_tensor(buf175, (6144, 2048), (2048, 1)); del buf175  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf176, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_45, (2048, 2048), (1, 2048), 0), out=buf177)
        buf179 = reinterpret_tensor(buf176, (3, 2048, 2048), (4194304, 2048, 1)); del buf176  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf158, buf177, primals_46, buf179, 6144, 2048, grid=grid(6144), stream=stream0)
        buf180 = reinterpret_tensor(buf156, (6144, 5632), (5632, 1)); del buf156  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_47, (2048, 5632), (1, 2048), 0), out=buf180)
        buf181 = buf155; del buf155  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_48, (2048, 5632), (1, 2048), 0), out=buf181)
        buf182 = reinterpret_tensor(buf180, (3, 2048, 5632), (11534336, 5632, 1)); del buf180  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf182, buf181, 34603008, grid=grid(34603008), stream=stream0)
        buf183 = reinterpret_tensor(buf179, (6144, 2048), (2048, 1)); del buf179  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf182, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_49, (5632, 2048), (1, 5632), 0), out=buf183)
        buf184 = reinterpret_tensor(buf177, (3, 2048, 2048), (4194304, 2048, 1)); del buf177  # reuse
        buf186 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf184, buf158, buf183, primals_50, buf186, 6144, 2048, grid=grid(6144), stream=stream0)
        buf187 = buf161; del buf161  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf186, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_51, (2048, 6144), (1, 2048), 0), out=buf187)
        buf190 = buf167; del buf167  # reuse
        buf188 = reinterpret_tensor(buf190, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf189 = reinterpret_tensor(buf190, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf193 = buf164; del buf164  # reuse
        buf191 = reinterpret_tensor(buf193, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf192 = reinterpret_tensor(buf193, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf187, primals_172, buf188, buf189, buf191, buf192, 6291456, grid=grid(6291456), stream=stream0)
        buf194 = reinterpret_tensor(buf186, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf186  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf190, buf194, 12582912, grid=grid(12582912), stream=stream0)
        del buf188
        del buf189
        del buf191
        del buf192
        buf195 = reinterpret_tensor(buf183, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf183  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf193, buf195, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf196 = reinterpret_tensor(buf173, (48, 2048, 2048), (4194304, 2048, 1)); del buf173  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf194, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf195, (48, 128, 2048), (262144, 2048, 1), 0), out=buf196)
        buf199 = reinterpret_tensor(buf170, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf170  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf196, buf14, buf199, 98304, 2048, grid=grid(98304), stream=stream0)
        buf200 = reinterpret_tensor(buf195, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf195  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf187, buf200, 12582912, grid=grid(12582912), stream=stream0)
        buf201 = reinterpret_tensor(buf194, (48, 2048, 128), (262144, 128, 1)); del buf194  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf199, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf200, (48, 2048, 128), (262144, 128, 1), 0), out=buf201)
        buf202 = reinterpret_tensor(buf200, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf200  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf201, buf202, 12582912, grid=grid(12582912), stream=stream0)
        buf203 = reinterpret_tensor(buf201, (6144, 2048), (2048, 1)); del buf201  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf202, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_52, (2048, 2048), (1, 2048), 0), out=buf203)
        buf205 = reinterpret_tensor(buf202, (3, 2048, 2048), (4194304, 2048, 1)); del buf202  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf184, buf203, primals_53, buf205, 6144, 2048, grid=grid(6144), stream=stream0)
        buf206 = reinterpret_tensor(buf182, (6144, 5632), (5632, 1)); del buf182  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_54, (2048, 5632), (1, 2048), 0), out=buf206)
        buf207 = buf181; del buf181  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf205, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_55, (2048, 5632), (1, 2048), 0), out=buf207)
        buf208 = reinterpret_tensor(buf206, (3, 2048, 5632), (11534336, 5632, 1)); del buf206  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf208, buf207, 34603008, grid=grid(34603008), stream=stream0)
        buf209 = reinterpret_tensor(buf205, (6144, 2048), (2048, 1)); del buf205  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf208, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_56, (5632, 2048), (1, 5632), 0), out=buf209)
        buf210 = reinterpret_tensor(buf203, (3, 2048, 2048), (4194304, 2048, 1)); del buf203  # reuse
        buf212 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf210, buf184, buf209, primals_57, buf212, 6144, 2048, grid=grid(6144), stream=stream0)
        buf213 = buf187; del buf187  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf212, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_58, (2048, 6144), (1, 2048), 0), out=buf213)
        buf216 = buf193; del buf193  # reuse
        buf214 = reinterpret_tensor(buf216, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf215 = reinterpret_tensor(buf216, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf219 = buf190; del buf190  # reuse
        buf217 = reinterpret_tensor(buf219, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf218 = reinterpret_tensor(buf219, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf213, primals_172, buf214, buf215, buf217, buf218, 6291456, grid=grid(6291456), stream=stream0)
        buf220 = reinterpret_tensor(buf212, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf212  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf216, buf220, 12582912, grid=grid(12582912), stream=stream0)
        del buf214
        del buf215
        del buf217
        del buf218
        buf221 = reinterpret_tensor(buf209, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf209  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf219, buf221, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf222 = reinterpret_tensor(buf199, (48, 2048, 2048), (4194304, 2048, 1)); del buf199  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf220, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf221, (48, 128, 2048), (262144, 2048, 1), 0), out=buf222)
        buf225 = reinterpret_tensor(buf196, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf196  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf222, buf14, buf225, 98304, 2048, grid=grid(98304), stream=stream0)
        buf226 = reinterpret_tensor(buf221, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf221  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf213, buf226, 12582912, grid=grid(12582912), stream=stream0)
        buf227 = reinterpret_tensor(buf220, (48, 2048, 128), (262144, 128, 1)); del buf220  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf225, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf226, (48, 2048, 128), (262144, 128, 1), 0), out=buf227)
        buf228 = reinterpret_tensor(buf226, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf226  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf227, buf228, 12582912, grid=grid(12582912), stream=stream0)
        buf229 = reinterpret_tensor(buf227, (6144, 2048), (2048, 1)); del buf227  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf228, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_59, (2048, 2048), (1, 2048), 0), out=buf229)
        buf231 = reinterpret_tensor(buf228, (3, 2048, 2048), (4194304, 2048, 1)); del buf228  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf210, buf229, primals_60, buf231, 6144, 2048, grid=grid(6144), stream=stream0)
        buf232 = reinterpret_tensor(buf208, (6144, 5632), (5632, 1)); del buf208  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_61, (2048, 5632), (1, 2048), 0), out=buf232)
        buf233 = buf207; del buf207  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_62, (2048, 5632), (1, 2048), 0), out=buf233)
        buf234 = reinterpret_tensor(buf232, (3, 2048, 5632), (11534336, 5632, 1)); del buf232  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf234, buf233, 34603008, grid=grid(34603008), stream=stream0)
        buf235 = reinterpret_tensor(buf231, (6144, 2048), (2048, 1)); del buf231  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf234, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_63, (5632, 2048), (1, 5632), 0), out=buf235)
        buf236 = reinterpret_tensor(buf229, (3, 2048, 2048), (4194304, 2048, 1)); del buf229  # reuse
        buf238 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf236, buf210, buf235, primals_64, buf238, 6144, 2048, grid=grid(6144), stream=stream0)
        buf239 = buf213; del buf213  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf238, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_65, (2048, 6144), (1, 2048), 0), out=buf239)
        buf242 = buf219; del buf219  # reuse
        buf240 = reinterpret_tensor(buf242, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf241 = reinterpret_tensor(buf242, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf245 = buf216; del buf216  # reuse
        buf243 = reinterpret_tensor(buf245, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf244 = reinterpret_tensor(buf245, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf239, primals_172, buf240, buf241, buf243, buf244, 6291456, grid=grid(6291456), stream=stream0)
        buf246 = reinterpret_tensor(buf238, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf238  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf242, buf246, 12582912, grid=grid(12582912), stream=stream0)
        del buf240
        del buf241
        del buf243
        del buf244
        buf247 = reinterpret_tensor(buf235, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf235  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf245, buf247, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf248 = reinterpret_tensor(buf225, (48, 2048, 2048), (4194304, 2048, 1)); del buf225  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf246, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf247, (48, 128, 2048), (262144, 2048, 1), 0), out=buf248)
        buf251 = reinterpret_tensor(buf222, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf222  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf248, buf14, buf251, 98304, 2048, grid=grid(98304), stream=stream0)
        buf252 = reinterpret_tensor(buf247, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf247  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf239, buf252, 12582912, grid=grid(12582912), stream=stream0)
        buf253 = reinterpret_tensor(buf246, (48, 2048, 128), (262144, 128, 1)); del buf246  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf251, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf252, (48, 2048, 128), (262144, 128, 1), 0), out=buf253)
        buf254 = reinterpret_tensor(buf252, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf252  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf253, buf254, 12582912, grid=grid(12582912), stream=stream0)
        buf255 = reinterpret_tensor(buf253, (6144, 2048), (2048, 1)); del buf253  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf254, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_66, (2048, 2048), (1, 2048), 0), out=buf255)
        buf257 = reinterpret_tensor(buf254, (3, 2048, 2048), (4194304, 2048, 1)); del buf254  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf236, buf255, primals_67, buf257, 6144, 2048, grid=grid(6144), stream=stream0)
        buf258 = reinterpret_tensor(buf234, (6144, 5632), (5632, 1)); del buf234  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf257, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_68, (2048, 5632), (1, 2048), 0), out=buf258)
        buf259 = buf233; del buf233  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf257, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_69, (2048, 5632), (1, 2048), 0), out=buf259)
        buf260 = reinterpret_tensor(buf258, (3, 2048, 5632), (11534336, 5632, 1)); del buf258  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf260, buf259, 34603008, grid=grid(34603008), stream=stream0)
        buf261 = reinterpret_tensor(buf257, (6144, 2048), (2048, 1)); del buf257  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf260, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_70, (5632, 2048), (1, 5632), 0), out=buf261)
        buf262 = reinterpret_tensor(buf255, (3, 2048, 2048), (4194304, 2048, 1)); del buf255  # reuse
        buf264 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf262, buf236, buf261, primals_71, buf264, 6144, 2048, grid=grid(6144), stream=stream0)
        buf265 = buf239; del buf239  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf264, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_72, (2048, 6144), (1, 2048), 0), out=buf265)
        buf268 = buf245; del buf245  # reuse
        buf266 = reinterpret_tensor(buf268, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf267 = reinterpret_tensor(buf268, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf271 = buf242; del buf242  # reuse
        buf269 = reinterpret_tensor(buf271, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf270 = reinterpret_tensor(buf271, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf265, primals_172, buf266, buf267, buf269, buf270, 6291456, grid=grid(6291456), stream=stream0)
        buf272 = reinterpret_tensor(buf264, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf264  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf268, buf272, 12582912, grid=grid(12582912), stream=stream0)
        del buf266
        del buf267
        del buf269
        del buf270
        buf273 = reinterpret_tensor(buf261, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf261  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf271, buf273, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf274 = reinterpret_tensor(buf251, (48, 2048, 2048), (4194304, 2048, 1)); del buf251  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf272, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf273, (48, 128, 2048), (262144, 2048, 1), 0), out=buf274)
        buf277 = reinterpret_tensor(buf248, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf248  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf274, buf14, buf277, 98304, 2048, grid=grid(98304), stream=stream0)
        buf278 = reinterpret_tensor(buf273, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf273  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf265, buf278, 12582912, grid=grid(12582912), stream=stream0)
        buf279 = reinterpret_tensor(buf272, (48, 2048, 128), (262144, 128, 1)); del buf272  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf277, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf278, (48, 2048, 128), (262144, 128, 1), 0), out=buf279)
        buf280 = reinterpret_tensor(buf278, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf278  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf279, buf280, 12582912, grid=grid(12582912), stream=stream0)
        buf281 = reinterpret_tensor(buf279, (6144, 2048), (2048, 1)); del buf279  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_73, (2048, 2048), (1, 2048), 0), out=buf281)
        buf283 = reinterpret_tensor(buf280, (3, 2048, 2048), (4194304, 2048, 1)); del buf280  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf262, buf281, primals_74, buf283, 6144, 2048, grid=grid(6144), stream=stream0)
        buf284 = reinterpret_tensor(buf260, (6144, 5632), (5632, 1)); del buf260  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf283, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_75, (2048, 5632), (1, 2048), 0), out=buf284)
        buf285 = buf259; del buf259  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf283, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_76, (2048, 5632), (1, 2048), 0), out=buf285)
        buf286 = reinterpret_tensor(buf284, (3, 2048, 5632), (11534336, 5632, 1)); del buf284  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf286, buf285, 34603008, grid=grid(34603008), stream=stream0)
        buf287 = reinterpret_tensor(buf283, (6144, 2048), (2048, 1)); del buf283  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf286, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_77, (5632, 2048), (1, 5632), 0), out=buf287)
        buf288 = reinterpret_tensor(buf281, (3, 2048, 2048), (4194304, 2048, 1)); del buf281  # reuse
        buf290 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf288, buf262, buf287, primals_78, buf290, 6144, 2048, grid=grid(6144), stream=stream0)
        buf291 = buf265; del buf265  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf290, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_79, (2048, 6144), (1, 2048), 0), out=buf291)
        buf294 = buf271; del buf271  # reuse
        buf292 = reinterpret_tensor(buf294, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf293 = reinterpret_tensor(buf294, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf297 = buf268; del buf268  # reuse
        buf295 = reinterpret_tensor(buf297, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf296 = reinterpret_tensor(buf297, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf291, primals_172, buf292, buf293, buf295, buf296, 6291456, grid=grid(6291456), stream=stream0)
        buf298 = reinterpret_tensor(buf290, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf290  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf294, buf298, 12582912, grid=grid(12582912), stream=stream0)
        del buf292
        del buf293
        del buf295
        del buf296
        buf299 = reinterpret_tensor(buf287, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf287  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf297, buf299, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf300 = reinterpret_tensor(buf277, (48, 2048, 2048), (4194304, 2048, 1)); del buf277  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf298, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf299, (48, 128, 2048), (262144, 2048, 1), 0), out=buf300)
        buf303 = reinterpret_tensor(buf274, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf274  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf300, buf14, buf303, 98304, 2048, grid=grid(98304), stream=stream0)
        buf304 = reinterpret_tensor(buf299, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf299  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf291, buf304, 12582912, grid=grid(12582912), stream=stream0)
        buf305 = reinterpret_tensor(buf298, (48, 2048, 128), (262144, 128, 1)); del buf298  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf303, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf304, (48, 2048, 128), (262144, 128, 1), 0), out=buf305)
        buf306 = reinterpret_tensor(buf304, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf304  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf305, buf306, 12582912, grid=grid(12582912), stream=stream0)
        buf307 = reinterpret_tensor(buf305, (6144, 2048), (2048, 1)); del buf305  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf306, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_80, (2048, 2048), (1, 2048), 0), out=buf307)
        buf309 = reinterpret_tensor(buf306, (3, 2048, 2048), (4194304, 2048, 1)); del buf306  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf288, buf307, primals_81, buf309, 6144, 2048, grid=grid(6144), stream=stream0)
        buf310 = reinterpret_tensor(buf286, (6144, 5632), (5632, 1)); del buf286  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_82, (2048, 5632), (1, 2048), 0), out=buf310)
        buf311 = buf285; del buf285  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf309, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_83, (2048, 5632), (1, 2048), 0), out=buf311)
        buf312 = reinterpret_tensor(buf310, (3, 2048, 5632), (11534336, 5632, 1)); del buf310  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf312, buf311, 34603008, grid=grid(34603008), stream=stream0)
        buf313 = reinterpret_tensor(buf309, (6144, 2048), (2048, 1)); del buf309  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf312, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_84, (5632, 2048), (1, 5632), 0), out=buf313)
        buf314 = reinterpret_tensor(buf307, (3, 2048, 2048), (4194304, 2048, 1)); del buf307  # reuse
        buf316 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf314, buf288, buf313, primals_85, buf316, 6144, 2048, grid=grid(6144), stream=stream0)
        buf317 = buf291; del buf291  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_86, (2048, 6144), (1, 2048), 0), out=buf317)
        buf320 = buf297; del buf297  # reuse
        buf318 = reinterpret_tensor(buf320, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf319 = reinterpret_tensor(buf320, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf323 = buf294; del buf294  # reuse
        buf321 = reinterpret_tensor(buf323, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf322 = reinterpret_tensor(buf323, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf317, primals_172, buf318, buf319, buf321, buf322, 6291456, grid=grid(6291456), stream=stream0)
        buf324 = reinterpret_tensor(buf316, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf316  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf320, buf324, 12582912, grid=grid(12582912), stream=stream0)
        del buf318
        del buf319
        del buf321
        del buf322
        buf325 = reinterpret_tensor(buf313, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf313  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf323, buf325, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf326 = reinterpret_tensor(buf303, (48, 2048, 2048), (4194304, 2048, 1)); del buf303  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf324, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf325, (48, 128, 2048), (262144, 2048, 1), 0), out=buf326)
        buf329 = reinterpret_tensor(buf300, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf300  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf326, buf14, buf329, 98304, 2048, grid=grid(98304), stream=stream0)
        buf330 = reinterpret_tensor(buf325, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf325  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf317, buf330, 12582912, grid=grid(12582912), stream=stream0)
        buf331 = reinterpret_tensor(buf324, (48, 2048, 128), (262144, 128, 1)); del buf324  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf329, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf330, (48, 2048, 128), (262144, 128, 1), 0), out=buf331)
        buf332 = reinterpret_tensor(buf330, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf330  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf331, buf332, 12582912, grid=grid(12582912), stream=stream0)
        buf333 = reinterpret_tensor(buf331, (6144, 2048), (2048, 1)); del buf331  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf332, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_87, (2048, 2048), (1, 2048), 0), out=buf333)
        buf335 = reinterpret_tensor(buf332, (3, 2048, 2048), (4194304, 2048, 1)); del buf332  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf314, buf333, primals_88, buf335, 6144, 2048, grid=grid(6144), stream=stream0)
        buf336 = reinterpret_tensor(buf312, (6144, 5632), (5632, 1)); del buf312  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf335, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_89, (2048, 5632), (1, 2048), 0), out=buf336)
        buf337 = buf311; del buf311  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf335, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_90, (2048, 5632), (1, 2048), 0), out=buf337)
        buf338 = reinterpret_tensor(buf336, (3, 2048, 5632), (11534336, 5632, 1)); del buf336  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf338, buf337, 34603008, grid=grid(34603008), stream=stream0)
        buf339 = reinterpret_tensor(buf335, (6144, 2048), (2048, 1)); del buf335  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf338, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_91, (5632, 2048), (1, 5632), 0), out=buf339)
        buf340 = reinterpret_tensor(buf333, (3, 2048, 2048), (4194304, 2048, 1)); del buf333  # reuse
        buf342 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf340, buf314, buf339, primals_92, buf342, 6144, 2048, grid=grid(6144), stream=stream0)
        buf343 = buf317; del buf317  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf342, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_93, (2048, 6144), (1, 2048), 0), out=buf343)
        buf346 = buf323; del buf323  # reuse
        buf344 = reinterpret_tensor(buf346, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf345 = reinterpret_tensor(buf346, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf349 = buf320; del buf320  # reuse
        buf347 = reinterpret_tensor(buf349, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf348 = reinterpret_tensor(buf349, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf343, primals_172, buf344, buf345, buf347, buf348, 6291456, grid=grid(6291456), stream=stream0)
        buf350 = reinterpret_tensor(buf342, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf342  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf346, buf350, 12582912, grid=grid(12582912), stream=stream0)
        del buf344
        del buf345
        del buf347
        del buf348
        buf351 = reinterpret_tensor(buf339, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf339  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf349, buf351, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf352 = reinterpret_tensor(buf329, (48, 2048, 2048), (4194304, 2048, 1)); del buf329  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf350, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf351, (48, 128, 2048), (262144, 2048, 1), 0), out=buf352)
        buf355 = reinterpret_tensor(buf326, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf326  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf352, buf14, buf355, 98304, 2048, grid=grid(98304), stream=stream0)
        buf356 = reinterpret_tensor(buf351, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf351  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf343, buf356, 12582912, grid=grid(12582912), stream=stream0)
        buf357 = reinterpret_tensor(buf350, (48, 2048, 128), (262144, 128, 1)); del buf350  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf355, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf356, (48, 2048, 128), (262144, 128, 1), 0), out=buf357)
        buf358 = reinterpret_tensor(buf356, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf356  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf357, buf358, 12582912, grid=grid(12582912), stream=stream0)
        buf359 = reinterpret_tensor(buf357, (6144, 2048), (2048, 1)); del buf357  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf358, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_94, (2048, 2048), (1, 2048), 0), out=buf359)
        buf361 = reinterpret_tensor(buf358, (3, 2048, 2048), (4194304, 2048, 1)); del buf358  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf340, buf359, primals_95, buf361, 6144, 2048, grid=grid(6144), stream=stream0)
        buf362 = reinterpret_tensor(buf338, (6144, 5632), (5632, 1)); del buf338  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf361, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_96, (2048, 5632), (1, 2048), 0), out=buf362)
        buf363 = buf337; del buf337  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf361, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_97, (2048, 5632), (1, 2048), 0), out=buf363)
        buf364 = reinterpret_tensor(buf362, (3, 2048, 5632), (11534336, 5632, 1)); del buf362  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf364, buf363, 34603008, grid=grid(34603008), stream=stream0)
        buf365 = reinterpret_tensor(buf361, (6144, 2048), (2048, 1)); del buf361  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf364, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_98, (5632, 2048), (1, 5632), 0), out=buf365)
        buf366 = reinterpret_tensor(buf359, (3, 2048, 2048), (4194304, 2048, 1)); del buf359  # reuse
        buf368 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf366, buf340, buf365, primals_99, buf368, 6144, 2048, grid=grid(6144), stream=stream0)
        buf369 = buf343; del buf343  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf368, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_100, (2048, 6144), (1, 2048), 0), out=buf369)
        buf372 = buf349; del buf349  # reuse
        buf370 = reinterpret_tensor(buf372, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf371 = reinterpret_tensor(buf372, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf375 = buf346; del buf346  # reuse
        buf373 = reinterpret_tensor(buf375, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf374 = reinterpret_tensor(buf375, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf369, primals_172, buf370, buf371, buf373, buf374, 6291456, grid=grid(6291456), stream=stream0)
        buf376 = reinterpret_tensor(buf368, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf368  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf372, buf376, 12582912, grid=grid(12582912), stream=stream0)
        del buf370
        del buf371
        del buf373
        del buf374
        buf377 = reinterpret_tensor(buf365, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf365  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf375, buf377, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf378 = reinterpret_tensor(buf355, (48, 2048, 2048), (4194304, 2048, 1)); del buf355  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf376, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf377, (48, 128, 2048), (262144, 2048, 1), 0), out=buf378)
        buf381 = reinterpret_tensor(buf352, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf352  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf378, buf14, buf381, 98304, 2048, grid=grid(98304), stream=stream0)
        buf382 = reinterpret_tensor(buf377, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf377  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf369, buf382, 12582912, grid=grid(12582912), stream=stream0)
        buf383 = reinterpret_tensor(buf376, (48, 2048, 128), (262144, 128, 1)); del buf376  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf381, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf382, (48, 2048, 128), (262144, 128, 1), 0), out=buf383)
        buf384 = reinterpret_tensor(buf382, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf382  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf383, buf384, 12582912, grid=grid(12582912), stream=stream0)
        buf385 = reinterpret_tensor(buf383, (6144, 2048), (2048, 1)); del buf383  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf384, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_101, (2048, 2048), (1, 2048), 0), out=buf385)
        buf387 = reinterpret_tensor(buf384, (3, 2048, 2048), (4194304, 2048, 1)); del buf384  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf366, buf385, primals_102, buf387, 6144, 2048, grid=grid(6144), stream=stream0)
        buf388 = reinterpret_tensor(buf364, (6144, 5632), (5632, 1)); del buf364  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_103, (2048, 5632), (1, 2048), 0), out=buf388)
        buf389 = buf363; del buf363  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf387, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_104, (2048, 5632), (1, 2048), 0), out=buf389)
        buf390 = reinterpret_tensor(buf388, (3, 2048, 5632), (11534336, 5632, 1)); del buf388  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf390, buf389, 34603008, grid=grid(34603008), stream=stream0)
        buf391 = reinterpret_tensor(buf387, (6144, 2048), (2048, 1)); del buf387  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf390, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_105, (5632, 2048), (1, 5632), 0), out=buf391)
        buf392 = reinterpret_tensor(buf385, (3, 2048, 2048), (4194304, 2048, 1)); del buf385  # reuse
        buf394 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf392, buf366, buf391, primals_106, buf394, 6144, 2048, grid=grid(6144), stream=stream0)
        buf395 = buf369; del buf369  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf394, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_107, (2048, 6144), (1, 2048), 0), out=buf395)
        buf398 = buf375; del buf375  # reuse
        buf396 = reinterpret_tensor(buf398, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf397 = reinterpret_tensor(buf398, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf401 = buf372; del buf372  # reuse
        buf399 = reinterpret_tensor(buf401, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf400 = reinterpret_tensor(buf401, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf395, primals_172, buf396, buf397, buf399, buf400, 6291456, grid=grid(6291456), stream=stream0)
        buf402 = reinterpret_tensor(buf394, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf394  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf398, buf402, 12582912, grid=grid(12582912), stream=stream0)
        del buf396
        del buf397
        del buf399
        del buf400
        buf403 = reinterpret_tensor(buf391, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf391  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf401, buf403, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf404 = reinterpret_tensor(buf381, (48, 2048, 2048), (4194304, 2048, 1)); del buf381  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf402, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf403, (48, 128, 2048), (262144, 2048, 1), 0), out=buf404)
        buf407 = reinterpret_tensor(buf378, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf378  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf404, buf14, buf407, 98304, 2048, grid=grid(98304), stream=stream0)
        buf408 = reinterpret_tensor(buf403, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf403  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf395, buf408, 12582912, grid=grid(12582912), stream=stream0)
        buf409 = reinterpret_tensor(buf402, (48, 2048, 128), (262144, 128, 1)); del buf402  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf407, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf408, (48, 2048, 128), (262144, 128, 1), 0), out=buf409)
        buf410 = reinterpret_tensor(buf408, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf408  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf409, buf410, 12582912, grid=grid(12582912), stream=stream0)
        buf411 = reinterpret_tensor(buf409, (6144, 2048), (2048, 1)); del buf409  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf410, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_108, (2048, 2048), (1, 2048), 0), out=buf411)
        buf413 = reinterpret_tensor(buf410, (3, 2048, 2048), (4194304, 2048, 1)); del buf410  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf392, buf411, primals_109, buf413, 6144, 2048, grid=grid(6144), stream=stream0)
        buf414 = reinterpret_tensor(buf390, (6144, 5632), (5632, 1)); del buf390  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf413, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_110, (2048, 5632), (1, 2048), 0), out=buf414)
        buf415 = buf389; del buf389  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf413, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_111, (2048, 5632), (1, 2048), 0), out=buf415)
        buf416 = reinterpret_tensor(buf414, (3, 2048, 5632), (11534336, 5632, 1)); del buf414  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf416, buf415, 34603008, grid=grid(34603008), stream=stream0)
        buf417 = reinterpret_tensor(buf413, (6144, 2048), (2048, 1)); del buf413  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf416, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_112, (5632, 2048), (1, 5632), 0), out=buf417)
        buf418 = reinterpret_tensor(buf411, (3, 2048, 2048), (4194304, 2048, 1)); del buf411  # reuse
        buf420 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf418, buf392, buf417, primals_113, buf420, 6144, 2048, grid=grid(6144), stream=stream0)
        buf421 = buf395; del buf395  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf420, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_114, (2048, 6144), (1, 2048), 0), out=buf421)
        buf424 = buf401; del buf401  # reuse
        buf422 = reinterpret_tensor(buf424, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf423 = reinterpret_tensor(buf424, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf427 = buf398; del buf398  # reuse
        buf425 = reinterpret_tensor(buf427, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf426 = reinterpret_tensor(buf427, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf421, primals_172, buf422, buf423, buf425, buf426, 6291456, grid=grid(6291456), stream=stream0)
        buf428 = reinterpret_tensor(buf420, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf420  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf424, buf428, 12582912, grid=grid(12582912), stream=stream0)
        del buf422
        del buf423
        del buf425
        del buf426
        buf429 = reinterpret_tensor(buf417, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf417  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf427, buf429, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf430 = reinterpret_tensor(buf407, (48, 2048, 2048), (4194304, 2048, 1)); del buf407  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf428, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf429, (48, 128, 2048), (262144, 2048, 1), 0), out=buf430)
        buf433 = reinterpret_tensor(buf404, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf404  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf430, buf14, buf433, 98304, 2048, grid=grid(98304), stream=stream0)
        buf434 = reinterpret_tensor(buf429, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf429  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf421, buf434, 12582912, grid=grid(12582912), stream=stream0)
        buf435 = reinterpret_tensor(buf428, (48, 2048, 128), (262144, 128, 1)); del buf428  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf433, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf434, (48, 2048, 128), (262144, 128, 1), 0), out=buf435)
        buf436 = reinterpret_tensor(buf434, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf434  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf435, buf436, 12582912, grid=grid(12582912), stream=stream0)
        buf437 = reinterpret_tensor(buf435, (6144, 2048), (2048, 1)); del buf435  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf436, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_115, (2048, 2048), (1, 2048), 0), out=buf437)
        buf439 = reinterpret_tensor(buf436, (3, 2048, 2048), (4194304, 2048, 1)); del buf436  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf418, buf437, primals_116, buf439, 6144, 2048, grid=grid(6144), stream=stream0)
        buf440 = reinterpret_tensor(buf416, (6144, 5632), (5632, 1)); del buf416  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf439, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_117, (2048, 5632), (1, 2048), 0), out=buf440)
        buf441 = buf415; del buf415  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf439, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_118, (2048, 5632), (1, 2048), 0), out=buf441)
        buf442 = reinterpret_tensor(buf440, (3, 2048, 5632), (11534336, 5632, 1)); del buf440  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf442, buf441, 34603008, grid=grid(34603008), stream=stream0)
        buf443 = reinterpret_tensor(buf439, (6144, 2048), (2048, 1)); del buf439  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf442, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_119, (5632, 2048), (1, 5632), 0), out=buf443)
        buf444 = reinterpret_tensor(buf437, (3, 2048, 2048), (4194304, 2048, 1)); del buf437  # reuse
        buf446 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf444, buf418, buf443, primals_120, buf446, 6144, 2048, grid=grid(6144), stream=stream0)
        buf447 = buf421; del buf421  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf446, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_121, (2048, 6144), (1, 2048), 0), out=buf447)
        buf450 = buf427; del buf427  # reuse
        buf448 = reinterpret_tensor(buf450, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf449 = reinterpret_tensor(buf450, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf453 = buf424; del buf424  # reuse
        buf451 = reinterpret_tensor(buf453, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf452 = reinterpret_tensor(buf453, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf447, primals_172, buf448, buf449, buf451, buf452, 6291456, grid=grid(6291456), stream=stream0)
        buf454 = reinterpret_tensor(buf446, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf446  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf450, buf454, 12582912, grid=grid(12582912), stream=stream0)
        del buf448
        del buf449
        del buf451
        del buf452
        buf455 = reinterpret_tensor(buf443, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf443  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf453, buf455, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf456 = reinterpret_tensor(buf433, (48, 2048, 2048), (4194304, 2048, 1)); del buf433  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf454, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf455, (48, 128, 2048), (262144, 2048, 1), 0), out=buf456)
        buf459 = reinterpret_tensor(buf430, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf430  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf456, buf14, buf459, 98304, 2048, grid=grid(98304), stream=stream0)
        buf460 = reinterpret_tensor(buf455, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf455  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf447, buf460, 12582912, grid=grid(12582912), stream=stream0)
        buf461 = reinterpret_tensor(buf454, (48, 2048, 128), (262144, 128, 1)); del buf454  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf459, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf460, (48, 2048, 128), (262144, 128, 1), 0), out=buf461)
        buf462 = reinterpret_tensor(buf460, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf460  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf461, buf462, 12582912, grid=grid(12582912), stream=stream0)
        buf463 = reinterpret_tensor(buf461, (6144, 2048), (2048, 1)); del buf461  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf462, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_122, (2048, 2048), (1, 2048), 0), out=buf463)
        buf465 = reinterpret_tensor(buf462, (3, 2048, 2048), (4194304, 2048, 1)); del buf462  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf444, buf463, primals_123, buf465, 6144, 2048, grid=grid(6144), stream=stream0)
        buf466 = reinterpret_tensor(buf442, (6144, 5632), (5632, 1)); del buf442  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf465, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_124, (2048, 5632), (1, 2048), 0), out=buf466)
        buf467 = buf441; del buf441  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf465, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_125, (2048, 5632), (1, 2048), 0), out=buf467)
        buf468 = reinterpret_tensor(buf466, (3, 2048, 5632), (11534336, 5632, 1)); del buf466  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf468, buf467, 34603008, grid=grid(34603008), stream=stream0)
        buf469 = reinterpret_tensor(buf465, (6144, 2048), (2048, 1)); del buf465  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf468, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_126, (5632, 2048), (1, 5632), 0), out=buf469)
        buf470 = reinterpret_tensor(buf463, (3, 2048, 2048), (4194304, 2048, 1)); del buf463  # reuse
        buf472 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf470, buf444, buf469, primals_127, buf472, 6144, 2048, grid=grid(6144), stream=stream0)
        buf473 = buf447; del buf447  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf472, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_128, (2048, 6144), (1, 2048), 0), out=buf473)
        buf476 = buf453; del buf453  # reuse
        buf474 = reinterpret_tensor(buf476, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf475 = reinterpret_tensor(buf476, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf479 = buf450; del buf450  # reuse
        buf477 = reinterpret_tensor(buf479, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf478 = reinterpret_tensor(buf479, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf473, primals_172, buf474, buf475, buf477, buf478, 6291456, grid=grid(6291456), stream=stream0)
        buf480 = reinterpret_tensor(buf472, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf472  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf476, buf480, 12582912, grid=grid(12582912), stream=stream0)
        del buf474
        del buf475
        del buf477
        del buf478
        buf481 = reinterpret_tensor(buf469, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf469  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf479, buf481, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf482 = reinterpret_tensor(buf459, (48, 2048, 2048), (4194304, 2048, 1)); del buf459  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf480, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf481, (48, 128, 2048), (262144, 2048, 1), 0), out=buf482)
        buf485 = reinterpret_tensor(buf456, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf456  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf482, buf14, buf485, 98304, 2048, grid=grid(98304), stream=stream0)
        buf486 = reinterpret_tensor(buf481, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf481  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf473, buf486, 12582912, grid=grid(12582912), stream=stream0)
        buf487 = reinterpret_tensor(buf480, (48, 2048, 128), (262144, 128, 1)); del buf480  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf485, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf486, (48, 2048, 128), (262144, 128, 1), 0), out=buf487)
        buf488 = reinterpret_tensor(buf486, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf486  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf487, buf488, 12582912, grid=grid(12582912), stream=stream0)
        buf489 = reinterpret_tensor(buf487, (6144, 2048), (2048, 1)); del buf487  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf488, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_129, (2048, 2048), (1, 2048), 0), out=buf489)
        buf491 = reinterpret_tensor(buf488, (3, 2048, 2048), (4194304, 2048, 1)); del buf488  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf470, buf489, primals_130, buf491, 6144, 2048, grid=grid(6144), stream=stream0)
        buf492 = reinterpret_tensor(buf468, (6144, 5632), (5632, 1)); del buf468  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf491, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_131, (2048, 5632), (1, 2048), 0), out=buf492)
        buf493 = buf467; del buf467  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf491, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_132, (2048, 5632), (1, 2048), 0), out=buf493)
        buf494 = reinterpret_tensor(buf492, (3, 2048, 5632), (11534336, 5632, 1)); del buf492  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf494, buf493, 34603008, grid=grid(34603008), stream=stream0)
        buf495 = reinterpret_tensor(buf491, (6144, 2048), (2048, 1)); del buf491  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf494, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_133, (5632, 2048), (1, 5632), 0), out=buf495)
        buf496 = reinterpret_tensor(buf489, (3, 2048, 2048), (4194304, 2048, 1)); del buf489  # reuse
        buf498 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf496, buf470, buf495, primals_134, buf498, 6144, 2048, grid=grid(6144), stream=stream0)
        buf499 = buf473; del buf473  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf498, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_135, (2048, 6144), (1, 2048), 0), out=buf499)
        buf502 = buf479; del buf479  # reuse
        buf500 = reinterpret_tensor(buf502, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf501 = reinterpret_tensor(buf502, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf505 = buf476; del buf476  # reuse
        buf503 = reinterpret_tensor(buf505, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf504 = reinterpret_tensor(buf505, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf499, primals_172, buf500, buf501, buf503, buf504, 6291456, grid=grid(6291456), stream=stream0)
        buf506 = reinterpret_tensor(buf498, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf498  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf502, buf506, 12582912, grid=grid(12582912), stream=stream0)
        del buf500
        del buf501
        del buf503
        del buf504
        buf507 = reinterpret_tensor(buf495, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf495  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf505, buf507, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf508 = reinterpret_tensor(buf485, (48, 2048, 2048), (4194304, 2048, 1)); del buf485  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf506, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf507, (48, 128, 2048), (262144, 2048, 1), 0), out=buf508)
        buf511 = reinterpret_tensor(buf482, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf482  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf508, buf14, buf511, 98304, 2048, grid=grid(98304), stream=stream0)
        buf512 = reinterpret_tensor(buf507, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf507  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf499, buf512, 12582912, grid=grid(12582912), stream=stream0)
        buf513 = reinterpret_tensor(buf506, (48, 2048, 128), (262144, 128, 1)); del buf506  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf511, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf512, (48, 2048, 128), (262144, 128, 1), 0), out=buf513)
        buf514 = reinterpret_tensor(buf512, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf512  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf513, buf514, 12582912, grid=grid(12582912), stream=stream0)
        buf515 = reinterpret_tensor(buf513, (6144, 2048), (2048, 1)); del buf513  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf514, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_136, (2048, 2048), (1, 2048), 0), out=buf515)
        buf517 = reinterpret_tensor(buf514, (3, 2048, 2048), (4194304, 2048, 1)); del buf514  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf496, buf515, primals_137, buf517, 6144, 2048, grid=grid(6144), stream=stream0)
        buf518 = reinterpret_tensor(buf494, (6144, 5632), (5632, 1)); del buf494  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf517, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_138, (2048, 5632), (1, 2048), 0), out=buf518)
        buf519 = buf493; del buf493  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf517, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_139, (2048, 5632), (1, 2048), 0), out=buf519)
        buf520 = reinterpret_tensor(buf518, (3, 2048, 5632), (11534336, 5632, 1)); del buf518  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf520, buf519, 34603008, grid=grid(34603008), stream=stream0)
        buf521 = reinterpret_tensor(buf517, (6144, 2048), (2048, 1)); del buf517  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf520, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_140, (5632, 2048), (1, 5632), 0), out=buf521)
        buf522 = reinterpret_tensor(buf515, (3, 2048, 2048), (4194304, 2048, 1)); del buf515  # reuse
        buf524 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf522, buf496, buf521, primals_141, buf524, 6144, 2048, grid=grid(6144), stream=stream0)
        buf525 = buf499; del buf499  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf524, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_142, (2048, 6144), (1, 2048), 0), out=buf525)
        buf528 = buf505; del buf505  # reuse
        buf526 = reinterpret_tensor(buf528, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf527 = reinterpret_tensor(buf528, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf531 = buf502; del buf502  # reuse
        buf529 = reinterpret_tensor(buf531, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf530 = reinterpret_tensor(buf531, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf525, primals_172, buf526, buf527, buf529, buf530, 6291456, grid=grid(6291456), stream=stream0)
        buf532 = reinterpret_tensor(buf524, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf524  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf528, buf532, 12582912, grid=grid(12582912), stream=stream0)
        del buf526
        del buf527
        del buf529
        del buf530
        buf533 = reinterpret_tensor(buf521, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf521  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf531, buf533, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf534 = reinterpret_tensor(buf511, (48, 2048, 2048), (4194304, 2048, 1)); del buf511  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf532, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf533, (48, 128, 2048), (262144, 2048, 1), 0), out=buf534)
        buf537 = reinterpret_tensor(buf508, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf508  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf534, buf14, buf537, 98304, 2048, grid=grid(98304), stream=stream0)
        buf538 = reinterpret_tensor(buf533, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf533  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf525, buf538, 12582912, grid=grid(12582912), stream=stream0)
        buf539 = reinterpret_tensor(buf532, (48, 2048, 128), (262144, 128, 1)); del buf532  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf537, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf538, (48, 2048, 128), (262144, 128, 1), 0), out=buf539)
        buf540 = reinterpret_tensor(buf538, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf538  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf539, buf540, 12582912, grid=grid(12582912), stream=stream0)
        buf541 = reinterpret_tensor(buf539, (6144, 2048), (2048, 1)); del buf539  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf540, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_143, (2048, 2048), (1, 2048), 0), out=buf541)
        buf543 = reinterpret_tensor(buf540, (3, 2048, 2048), (4194304, 2048, 1)); del buf540  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf522, buf541, primals_144, buf543, 6144, 2048, grid=grid(6144), stream=stream0)
        buf544 = reinterpret_tensor(buf520, (6144, 5632), (5632, 1)); del buf520  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf543, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_145, (2048, 5632), (1, 2048), 0), out=buf544)
        buf545 = buf519; del buf519  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf543, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_146, (2048, 5632), (1, 2048), 0), out=buf545)
        buf546 = reinterpret_tensor(buf544, (3, 2048, 5632), (11534336, 5632, 1)); del buf544  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf546, buf545, 34603008, grid=grid(34603008), stream=stream0)
        buf547 = reinterpret_tensor(buf543, (6144, 2048), (2048, 1)); del buf543  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf546, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_147, (5632, 2048), (1, 5632), 0), out=buf547)
        buf548 = reinterpret_tensor(buf541, (3, 2048, 2048), (4194304, 2048, 1)); del buf541  # reuse
        buf550 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf548, buf522, buf547, primals_148, buf550, 6144, 2048, grid=grid(6144), stream=stream0)
        buf551 = buf525; del buf525  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf550, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_149, (2048, 6144), (1, 2048), 0), out=buf551)
        buf554 = buf531; del buf531  # reuse
        buf552 = reinterpret_tensor(buf554, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf553 = reinterpret_tensor(buf554, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf557 = buf528; del buf528  # reuse
        buf555 = reinterpret_tensor(buf557, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf556 = reinterpret_tensor(buf557, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf551, primals_172, buf552, buf553, buf555, buf556, 6291456, grid=grid(6291456), stream=stream0)
        buf558 = reinterpret_tensor(buf550, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf550  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf554, buf558, 12582912, grid=grid(12582912), stream=stream0)
        del buf552
        del buf553
        del buf555
        del buf556
        buf559 = reinterpret_tensor(buf547, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf547  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf557, buf559, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf560 = reinterpret_tensor(buf537, (48, 2048, 2048), (4194304, 2048, 1)); del buf537  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf558, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf559, (48, 128, 2048), (262144, 2048, 1), 0), out=buf560)
        buf563 = reinterpret_tensor(buf534, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf534  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf560, buf14, buf563, 98304, 2048, grid=grid(98304), stream=stream0)
        buf564 = reinterpret_tensor(buf559, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf559  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf551, buf564, 12582912, grid=grid(12582912), stream=stream0)
        buf565 = reinterpret_tensor(buf558, (48, 2048, 128), (262144, 128, 1)); del buf558  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf563, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf564, (48, 2048, 128), (262144, 128, 1), 0), out=buf565)
        buf566 = reinterpret_tensor(buf564, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf564  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf565, buf566, 12582912, grid=grid(12582912), stream=stream0)
        buf567 = reinterpret_tensor(buf565, (6144, 2048), (2048, 1)); del buf565  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf566, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_150, (2048, 2048), (1, 2048), 0), out=buf567)
        buf569 = reinterpret_tensor(buf566, (3, 2048, 2048), (4194304, 2048, 1)); del buf566  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf548, buf567, primals_151, buf569, 6144, 2048, grid=grid(6144), stream=stream0)
        buf570 = reinterpret_tensor(buf546, (6144, 5632), (5632, 1)); del buf546  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf569, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_152, (2048, 5632), (1, 2048), 0), out=buf570)
        buf571 = buf545; del buf545  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf569, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_153, (2048, 5632), (1, 2048), 0), out=buf571)
        buf572 = reinterpret_tensor(buf570, (3, 2048, 5632), (11534336, 5632, 1)); del buf570  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf572, buf571, 34603008, grid=grid(34603008), stream=stream0)
        buf573 = reinterpret_tensor(buf569, (6144, 2048), (2048, 1)); del buf569  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf572, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_154, (5632, 2048), (1, 5632), 0), out=buf573)
        buf574 = reinterpret_tensor(buf567, (3, 2048, 2048), (4194304, 2048, 1)); del buf567  # reuse
        buf576 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf574, buf548, buf573, primals_155, buf576, 6144, 2048, grid=grid(6144), stream=stream0)
        buf577 = buf551; del buf551  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf576, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_156, (2048, 6144), (1, 2048), 0), out=buf577)
        buf580 = buf557; del buf557  # reuse
        buf578 = reinterpret_tensor(buf580, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf579 = reinterpret_tensor(buf580, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf583 = buf554; del buf554  # reuse
        buf581 = reinterpret_tensor(buf583, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf582 = reinterpret_tensor(buf583, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf577, primals_172, buf578, buf579, buf581, buf582, 6291456, grid=grid(6291456), stream=stream0)
        buf584 = reinterpret_tensor(buf576, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf576  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf580, buf584, 12582912, grid=grid(12582912), stream=stream0)
        del buf578
        del buf579
        del buf581
        del buf582
        buf585 = reinterpret_tensor(buf573, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf573  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf583, buf585, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf586 = reinterpret_tensor(buf563, (48, 2048, 2048), (4194304, 2048, 1)); del buf563  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf584, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf585, (48, 128, 2048), (262144, 2048, 1), 0), out=buf586)
        buf589 = reinterpret_tensor(buf560, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf560  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf586, buf14, buf589, 98304, 2048, grid=grid(98304), stream=stream0)
        buf590 = reinterpret_tensor(buf585, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf585  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf577, buf590, 12582912, grid=grid(12582912), stream=stream0)
        buf591 = reinterpret_tensor(buf584, (48, 2048, 128), (262144, 128, 1)); del buf584  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf589, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf590, (48, 2048, 128), (262144, 128, 1), 0), out=buf591)
        buf592 = reinterpret_tensor(buf590, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf590  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf591, buf592, 12582912, grid=grid(12582912), stream=stream0)
        buf593 = reinterpret_tensor(buf591, (6144, 2048), (2048, 1)); del buf591  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf592, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_157, (2048, 2048), (1, 2048), 0), out=buf593)
        buf595 = reinterpret_tensor(buf592, (3, 2048, 2048), (4194304, 2048, 1)); del buf592  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf574, buf593, primals_158, buf595, 6144, 2048, grid=grid(6144), stream=stream0)
        buf596 = reinterpret_tensor(buf572, (6144, 5632), (5632, 1)); del buf572  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf595, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_159, (2048, 5632), (1, 2048), 0), out=buf596)
        buf597 = buf571; del buf571  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf595, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_160, (2048, 5632), (1, 2048), 0), out=buf597)
        buf598 = reinterpret_tensor(buf596, (3, 2048, 5632), (11534336, 5632, 1)); del buf596  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf598, buf597, 34603008, grid=grid(34603008), stream=stream0)
        buf599 = reinterpret_tensor(buf595, (6144, 2048), (2048, 1)); del buf595  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf598, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_161, (5632, 2048), (1, 5632), 0), out=buf599)
        buf600 = reinterpret_tensor(buf593, (3, 2048, 2048), (4194304, 2048, 1)); del buf593  # reuse
        buf602 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf600, buf574, buf599, primals_162, buf602, 6144, 2048, grid=grid(6144), stream=stream0)
        buf603 = buf577; del buf577  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf602, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_163, (2048, 6144), (1, 2048), 0), out=buf603)
        buf606 = buf583; del buf583  # reuse
        buf604 = reinterpret_tensor(buf606, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf605 = reinterpret_tensor(buf606, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf609 = buf580; del buf580  # reuse
        buf607 = reinterpret_tensor(buf609, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf608 = reinterpret_tensor(buf609, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf603, primals_172, buf604, buf605, buf607, buf608, 6291456, grid=grid(6291456), stream=stream0)
        buf610 = reinterpret_tensor(buf602, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf602  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_3.run(buf606, buf610, 12582912, grid=grid(12582912), stream=stream0)
        del buf604
        del buf605
        del buf606
        del buf607
        del buf608
        buf611 = reinterpret_tensor(buf599, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf599  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_4.run(buf609, buf611, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        del buf609
        buf612 = reinterpret_tensor(buf589, (48, 2048, 2048), (4194304, 2048, 1)); del buf589  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf610, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf611, (48, 128, 2048), (262144, 2048, 1), 0), out=buf612)
        buf615 = reinterpret_tensor(buf586, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf586  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_6.run(buf10, buf612, buf14, buf615, 98304, 2048, grid=grid(98304), stream=stream0)
        del buf612
        buf616 = reinterpret_tensor(buf611, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf611  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf603, buf616, 12582912, grid=grid(12582912), stream=stream0)
        del buf603
        buf617 = reinterpret_tensor(buf610, (48, 2048, 128), (262144, 128, 1)); del buf610  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf615, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf616, (48, 2048, 128), (262144, 128, 1), 0), out=buf617)
        del buf615
        buf618 = reinterpret_tensor(buf616, (3, 2048, 16, 128), (4194304, 2048, 128, 1)); del buf616  # reuse
        # Source Nodes: [contiguous], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf617, buf618, 12582912, grid=grid(12582912), stream=stream0)
        buf619 = reinterpret_tensor(buf617, (6144, 2048), (2048, 1)); del buf617  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf618, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_164, (2048, 2048), (1, 2048), 0), out=buf619)
        buf621 = reinterpret_tensor(buf618, (3, 2048, 2048), (4194304, 2048, 1)); del buf618  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_9.run(buf600, buf619, primals_165, buf621, 6144, 2048, grid=grid(6144), stream=stream0)
        buf622 = reinterpret_tensor(buf598, (6144, 5632), (5632, 1)); del buf598  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf621, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_166, (2048, 5632), (1, 2048), 0), out=buf622)
        buf623 = buf597; del buf597  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf621, (6144, 2048), (2048, 1), 0), reinterpret_tensor(primals_167, (2048, 5632), (1, 2048), 0), out=buf623)
        buf624 = reinterpret_tensor(buf622, (3, 2048, 5632), (11534336, 5632, 1)); del buf622  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_10.run(buf624, buf623, 34603008, grid=grid(34603008), stream=stream0)
        del buf623
        buf625 = reinterpret_tensor(buf621, (6144, 2048), (2048, 1)); del buf621  # reuse
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf624, (6144, 5632), (5632, 1), 0), reinterpret_tensor(primals_168, (5632, 2048), (1, 5632), 0), out=buf625)
        del buf624
        buf626 = empty_strided((3, 2048, 1), (2048, 1, 6144), device='cuda', dtype=torch.float32)
        buf627 = reinterpret_tensor(buf626, (3, 2048, 1), (2048, 1, 1)); del buf626  # reuse
        buf628 = empty_strided((6144, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_3, add_5, float_1, l__mod___output, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.view]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_view_13.run(buf627, buf600, buf619, buf625, primals_169, buf628, 6144, 2048, grid=grid(6144), stream=stream0)
        del buf619
        del buf625
        buf631 = empty_strided((2048, 50264), (50264, 1), device='cuda', dtype=torch.float16)
        buf629 = reinterpret_tensor(buf631, (2048, 50257), (50264, 1), 0)  # alias
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_14.run(primals_171, buf629, 2048, 50257, grid=grid(2048, 50257), stream=stream0)
        buf630 = reinterpret_tensor(buf631, (2048, 7), (50264, 1), 50257)  # alias
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_15.run(buf630, 14336, grid=grid(14336), stream=stream0)
        del buf629
        del buf630
        buf632 = empty_strided((6144, 50264), (50264, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf628, buf631, out=buf632)
        del buf631
        buf635 = empty_strided((6144, 50257), (50257, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax]
        triton_red_fused__log_softmax_16.run(buf632, buf635, 6144, 50257, grid=grid(6144), stream=stream0)
        del buf632
        buf638 = empty_strided((), (), device='cuda', dtype=torch.float16)
        buf637 = empty_strided((), (), device='cuda', dtype=torch.float16)
        buf639 = buf638; del buf638  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_forward]
        triton_red_fused_nll_loss_forward_17.run(buf639, primals_175, buf635, buf637, 1, 6144, grid=grid(1), stream=stream0)
        buf640 = empty_strided((2048, 64, 2), (128, 2, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [getitem_1], Original ATen: [aten.index]
        triton_poi_fused_index_18.run(primals_172, buf640, 262144, grid=grid(262144), stream=stream0)
        del primals_172
        return (buf639, primals_1, primals_4, primals_8, primals_11, primals_15, primals_18, primals_22, primals_25, primals_29, primals_32, primals_36, primals_39, primals_43, primals_46, primals_50, primals_53, primals_57, primals_60, primals_64, primals_67, primals_71, primals_74, primals_78, primals_81, primals_85, primals_88, primals_92, primals_95, primals_99, primals_102, primals_106, primals_109, primals_113, primals_116, primals_120, primals_123, primals_127, primals_130, primals_134, primals_137, primals_141, primals_144, primals_148, primals_151, primals_155, primals_158, primals_162, primals_165, primals_169, primals_174, primals_175, buf0, reinterpret_tensor(primals_2, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf640, (1, 2048, 1, 64), (0, 128, 0, 2), 0), reinterpret_tensor(buf640, (1, 2048, 1, 64), (0, 128, 0, 2), 1), buf10, buf14, reinterpret_tensor(primals_3, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_5, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_6, (2048, 5632), (1, 2048), 0), buf28, reinterpret_tensor(primals_9, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_10, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_12, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_13, (2048, 5632), (1, 2048), 0), buf54, reinterpret_tensor(primals_16, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_17, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_19, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_20, (2048, 5632), (1, 2048), 0), buf80, reinterpret_tensor(primals_23, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_24, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_26, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_27, (2048, 5632), (1, 2048), 0), buf106, reinterpret_tensor(primals_30, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_31, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_33, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_34, (2048, 5632), (1, 2048), 0), buf132, reinterpret_tensor(primals_37, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_38, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_40, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_41, (2048, 5632), (1, 2048), 0), buf158, reinterpret_tensor(primals_44, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_45, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_47, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_48, (2048, 5632), (1, 2048), 0), buf184, reinterpret_tensor(primals_51, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_52, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_54, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_55, (2048, 5632), (1, 2048), 0), buf210, reinterpret_tensor(primals_58, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_59, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_61, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_62, (2048, 5632), (1, 2048), 0), buf236, reinterpret_tensor(primals_65, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_66, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_68, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_69, (2048, 5632), (1, 2048), 0), buf262, reinterpret_tensor(primals_72, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_73, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_75, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_76, (2048, 5632), (1, 2048), 0), buf288, reinterpret_tensor(primals_79, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_80, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_82, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_83, (2048, 5632), (1, 2048), 0), buf314, reinterpret_tensor(primals_86, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_87, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_89, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_90, (2048, 5632), (1, 2048), 0), buf340, reinterpret_tensor(primals_93, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_94, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_96, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_97, (2048, 5632), (1, 2048), 0), buf366, reinterpret_tensor(primals_100, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_101, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_103, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_104, (2048, 5632), (1, 2048), 0), buf392, reinterpret_tensor(primals_107, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_108, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_110, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_111, (2048, 5632), (1, 2048), 0), buf418, reinterpret_tensor(primals_114, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_115, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_117, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_118, (2048, 5632), (1, 2048), 0), buf444, reinterpret_tensor(primals_121, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_122, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_124, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_125, (2048, 5632), (1, 2048), 0), buf470, reinterpret_tensor(primals_128, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_129, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_131, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_132, (2048, 5632), (1, 2048), 0), buf496, reinterpret_tensor(primals_135, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_136, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_138, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_139, (2048, 5632), (1, 2048), 0), buf522, reinterpret_tensor(primals_142, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_143, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_145, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_146, (2048, 5632), (1, 2048), 0), buf548, reinterpret_tensor(primals_149, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_150, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_152, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_153, (2048, 5632), (1, 2048), 0), buf574, reinterpret_tensor(primals_156, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_157, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_159, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_160, (2048, 5632), (1, 2048), 0), buf600, reinterpret_tensor(primals_163, (2048, 6144), (1, 2048), 0), reinterpret_tensor(primals_164, (2048, 2048), (1, 2048), 0), reinterpret_tensor(primals_166, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_167, (2048, 5632), (1, 2048), 0), reinterpret_tensor(primals_168, (5632, 2048), (1, 5632), 0), buf627, buf628, buf635, buf637, reinterpret_tensor(primals_171, (50257, 2048), (2048, 1), 0), reinterpret_tensor(primals_161, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_154, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_147, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_140, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_133, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_126, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_119, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_112, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_105, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_98, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_91, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_84, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_77, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_70, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_63, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_56, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_49, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_42, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_35, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_28, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_21, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_14, (2048, 5632), (5632, 1), 0), reinterpret_tensor(primals_7, (2048, 5632), (5632, 1), 0), )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_2 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_3 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_4 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_5 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_6 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_7 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_8 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_9 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_10 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_11 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_12 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_13 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_14 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_15 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_16 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_17 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_18 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_19 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_20 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_21 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_22 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_23 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_24 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_25 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_26 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_27 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_28 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_29 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_30 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_31 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_32 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_33 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_34 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_35 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_36 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_37 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_38 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_39 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_40 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_41 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_42 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_43 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_44 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_45 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_46 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_47 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_48 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_49 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_50 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_51 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_52 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_53 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_54 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_55 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_56 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_57 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_58 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_59 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_60 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_61 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_62 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_63 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_64 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_65 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_66 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_67 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_68 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_69 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_70 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_71 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_72 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_73 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_74 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_75 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_76 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_77 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_78 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_79 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_80 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_81 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_82 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_83 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_84 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_85 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_86 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_87 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_88 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_89 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_90 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_91 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_92 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_93 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_94 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_95 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_96 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_97 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_98 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_99 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_100 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_101 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_102 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_103 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_104 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_105 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_106 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_107 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_108 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_109 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_110 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_111 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_112 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_113 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_114 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_115 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_116 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_117 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_118 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_119 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_120 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_121 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_122 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_123 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_124 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_125 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_126 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_127 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_128 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_129 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_130 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_131 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_132 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_133 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_134 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_135 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_136 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_137 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_138 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_139 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_140 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_141 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_142 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_143 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_144 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_145 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_146 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_147 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_148 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_149 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_150 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_151 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_152 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_153 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_154 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_155 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_156 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_157 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_158 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_159 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_160 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_161 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_162 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_163 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_164 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_165 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_166 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_167 = rand_strided((5632, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_168 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    primals_169 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_170 = rand_strided((50257, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_171 = rand_strided((50257, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    primals_172 = rand_strided((2048, 64, 2), (128, 2, 1), device='cuda:0', dtype=torch.float16)
    primals_173 = rand_strided((2048, 2048), (2048, 1), device='cuda:0', dtype=torch.bool)
    primals_174 = rand_strided((3, 2048), (2049, 1), device='cuda:0', dtype=torch.int64)
    primals_175 = rand_strided((3, 2048), (2049, 1), device='cuda:0', dtype=torch.int64)
    return print_performance(lambda: call([primals_1, primals_2, primals_3, primals_4, primals_5, primals_6, primals_7, primals_8, primals_9, primals_10, primals_11, primals_12, primals_13, primals_14, primals_15, primals_16, primals_17, primals_18, primals_19, primals_20, primals_21, primals_22, primals_23, primals_24, primals_25, primals_26, primals_27, primals_28, primals_29, primals_30, primals_31, primals_32, primals_33, primals_34, primals_35, primals_36, primals_37, primals_38, primals_39, primals_40, primals_41, primals_42, primals_43, primals_44, primals_45, primals_46, primals_47, primals_48, primals_49, primals_50, primals_51, primals_52, primals_53, primals_54, primals_55, primals_56, primals_57, primals_58, primals_59, primals_60, primals_61, primals_62, primals_63, primals_64, primals_65, primals_66, primals_67, primals_68, primals_69, primals_70, primals_71, primals_72, primals_73, primals_74, primals_75, primals_76, primals_77, primals_78, primals_79, primals_80, primals_81, primals_82, primals_83, primals_84, primals_85, primals_86, primals_87, primals_88, primals_89, primals_90, primals_91, primals_92, primals_93, primals_94, primals_95, primals_96, primals_97, primals_98, primals_99, primals_100, primals_101, primals_102, primals_103, primals_104, primals_105, primals_106, primals_107, primals_108, primals_109, primals_110, primals_111, primals_112, primals_113, primals_114, primals_115, primals_116, primals_117, primals_118, primals_119, primals_120, primals_121, primals_122, primals_123, primals_124, primals_125, primals_126, primals_127, primals_128, primals_129, primals_130, primals_131, primals_132, primals_133, primals_134, primals_135, primals_136, primals_137, primals_138, primals_139, primals_140, primals_141, primals_142, primals_143, primals_144, primals_145, primals_146, primals_147, primals_148, primals_149, primals_150, primals_151, primals_152, primals_153, primals_154, primals_155, primals_156, primals_157, primals_158, primals_159, primals_160, primals_161, primals_162, primals_163, primals_164, primals_165, primals_166, primals_167, primals_168, primals_169, primals_170, primals_171, primals_172, primals_173, primals_174, primals_175]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

