
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


# kernel path: /tmp/torchinductor_mengqy/4o/c4oyqxwmt73g7duz6z6jiqycgi4j7w3pyv7c6p3vm2taebjyvyor.py
# Source Nodes: [add, float_1, l__model___tok_embeddings, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.rsqrt]
# add => add
# float_1 => convert_element_type
# l__model___tok_embeddings => embedding
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
    size_hints=[1, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_0', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp2 = tl.where(tmp1 < 0, tmp1 + 32000, tmp1)
        tl.device_assert((0 <= tmp2) & (tmp2 < 32000), "index out of bounds: 0 <= tmp2 < 32000")
        tmp3 = tl.load(in_ptr1 + (r0 + (4096*tmp2)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tmp3.to(tl.float32)
        tmp5 = tmp4 * tmp4
        tmp6 = tl.broadcast_to(tmp5, [XBLOCK, RBLOCK])
        tmp8 = _tmp7 + tmp6
        _tmp7 = tl.where(rmask, tmp8, _tmp7)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp9 = tl.load(in_ptr0 + (0))
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp21 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp11 = tl.where(tmp10 < 0, tmp10 + 32000, tmp10)
        tl.device_assert((0 <= tmp11) & (tmp11 < 32000), "index out of bounds: 0 <= tmp11 < 32000")
        tmp12 = tl.load(in_ptr1 + (r0 + (4096*tmp11)), rmask, other=0).to(tl.float32)
        tmp13 = tmp12.to(tl.float32)
        tmp14 = 4096.0
        tmp15 = tmp7 / tmp14
        tmp16 = 1e-05
        tmp17 = tmp15 + tmp16
        tmp18 = tl.math.rsqrt(tmp17)
        tmp19 = tmp13 * tmp18
        tmp20 = tmp19.to(tl.float32)
        tmp22 = tmp20 * tmp21
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp22, rmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_mengqy/uq/cuqaq5wbj7wyo43aqcnhpsibujk3sucwfn234vtu5ybgjnejqbxm.py
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

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i32', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_1', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]})
@triton.jit
def triton_poi_fused_stack_1(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (2*x2), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp8 = tl.load(in_ptr0 + (1 + (2*x2)), None).to(tl.float32)
    tmp17 = tl.load(in_ptr0 + (4096 + (2*x2)), None).to(tl.float32)
    tmp20 = tl.load(in_ptr0 + (4097 + (2*x2)), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp4 = tl.where(tmp3 < 0, tmp3 + 2048, tmp3)
    tl.device_assert((0 <= tmp4) & (tmp4 < 2048), "index out of bounds: 0 <= tmp4 < 2048")
    tmp5 = tl.load(in_ptr2 + ((2*x0) + (128*tmp4)), None).to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp1 * tmp6
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tl.load(in_ptr2 + (1 + (2*x0) + (128*tmp4)), None).to(tl.float32)
    tmp11 = tmp10.to(tl.float32)
    tmp12 = tmp9 * tmp11
    tmp13 = tmp7 - tmp12
    tmp14 = tmp9 * tmp6
    tmp15 = tmp1 * tmp11
    tmp16 = tmp14 + tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = tmp18 * tmp6
    tmp21 = tmp20.to(tl.float32)
    tmp22 = tmp21 * tmp11
    tmp23 = tmp19 - tmp22
    tmp24 = tmp21 * tmp6
    tmp25 = tmp18 * tmp11
    tmp26 = tmp24 + tmp25
    tl.store(out_ptr0 + (2*x2), tmp13, None)
    tl.store(out_ptr1 + (2*x2), tmp16, None)
    tl.store(out_ptr2 + (2*x2), tmp23, None)
    tl.store(out_ptr3 + (2*x2), tmp26, None)
''')


# kernel path: /tmp/torchinductor_mengqy/5b/c5b7rfgcpslka7h7hzybwyuoh6ixzxjoahteewf6ohykkzx6kluz.py
# Source Nodes: [setitem], Original ATen: [aten.index_put]
# setitem => index_put
triton_poi_fused_index_put_2 = async_compile.triton('triton_poi_fused_index_put_2', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_2', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_index_put_2(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4718592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/ji/cjieuak532hz45nife677oionfrtpnbirdkgrnlanrsvwutklp3t.py
# Source Nodes: [setitem, setitem_1], Original ATen: [aten.index_put]
# setitem => index_put
# setitem_1 => index_put_1
triton_poi_fused_index_put_3 = async_compile.triton('triton_poi_fused_index_put_3', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*i32', 1: '*fp32', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['out_ptr0', 'out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_3', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]})
@triton.jit
def triton_poi_fused_index_put_3(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 128
    x1 = (xindex // 128)
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK])
    tmp3 = tl.load(in_ptr1 + (x2), None)
    tmp5 = tl.load(in_ptr2 + (8192 + x2), None).to(tl.float32)
    tmp2 = tl.where(tmp1 < 0, tmp1 + 1152, tmp1)
    tl.device_assert((0 <= tmp2) & (tmp2 < 1152), "index out of bounds: 0 <= tmp2 < 1152")
    tmp4 = tmp3.to(tl.float32)
    tl.store(out_ptr0 + (x0 + (128*tmp2) + (147456*x1)), tmp4, None)
    tl.store(out_ptr1 + (x0 + (128*tmp2) + (147456*x1)), tmp5, None)
''')


# kernel path: /tmp/torchinductor_mengqy/7z/c7z6l46gzmmf5g35bf5zgcthy5is3dxkj5ji35acie7zgwmn4sup.py
# Source Nodes: [type_as_1], Original ATen: [aten._to_copy]
# type_as_1 => convert_element_type_3
triton_poi_fused__to_copy_4 = async_compile.triton('triton_poi_fused__to_copy_4', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_4', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused__to_copy_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''')


# kernel path: /tmp/torchinductor_mengqy/ta/ctagidxkwhgfobqcu27xenocmymkeyzb5ilnqcauwyj5mqd7ffl2.py
# Source Nodes: [setitem], Original ATen: [aten.slice_scatter]
# setitem => slice_scatter, slice_scatter_1
triton_poi_fused_slice_scatter_5 = async_compile.triton('triton_poi_fused_slice_scatter_5', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_scatter_5', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]})
@triton.jit
def triton_poi_fused_slice_scatter_5(in_ptr0, out_ptr0, out_ptr1, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4718592
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
    tl.store(out_ptr1 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/hy/chyk3tr7olbhngy7lqa2tecbzrbzz7mwrq3qkjup3lzuvmdf4row.py
# Source Nodes: [getitem, mul_11, softmax, where], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
# getitem => index
# mul_11 => mul_11
# softmax => amax, convert_element_type_6, convert_element_type_7, div, exp, sub_2, sum_1
# where => full_default, where
triton_red_fused__softmax_index_mul_scalar_tensor_where_6 = async_compile.triton('triton_red_fused__softmax_index_mul_scalar_tensor_where_6', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i32', 1: '*i1', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_index_mul_scalar_tensor_where_6', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_red_fused__softmax_index_mul_scalar_tensor_where_6(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 32
    rnumel = 1152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    x0 = xindex
    _tmp11 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp4 = tl.load(in_ptr2 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tl.where(tmp1 < 0, tmp1 + 1152, tmp1)
        tl.device_assert((0 <= tmp2) & (tmp2 < 1152), "index out of bounds: 0 <= tmp2 < 1152")
        tmp3 = tl.load(in_ptr1 + (r1 + (1152*tmp2)), rmask, eviction_policy='evict_last')
        tmp5 = 0.08838834764831843
        tmp6 = tmp4 * tmp5
        tmp7 = -65504.0
        tmp8 = tl.where(tmp3, tmp6, tmp7)
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = triton_helpers.maximum(_tmp11, tmp10)
        _tmp11 = tl.where(rmask & xmask, tmp12, _tmp11)
    tmp11 = triton_helpers.max2(_tmp11, 1)[:, None]
    tmp13 = tl.load(in_ptr0 + (0))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    _tmp26 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp17 = tl.load(in_ptr2 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp15 = tl.where(tmp14 < 0, tmp14 + 1152, tmp14)
        tl.device_assert((0 <= tmp15) & (tmp15 < 1152), "index out of bounds: 0 <= tmp15 < 1152")
        tmp16 = tl.load(in_ptr1 + (r1 + (1152*tmp15)), rmask, eviction_policy='evict_last')
        tmp18 = 0.08838834764831843
        tmp19 = tmp17 * tmp18
        tmp20 = -65504.0
        tmp21 = tl.where(tmp16, tmp19, tmp20)
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp22 - tmp11
        tmp24 = tl.exp(tmp23)
        tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
        tmp27 = _tmp26 + tmp25
        _tmp26 = tl.where(rmask & xmask, tmp27, _tmp26)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp28 = tl.load(in_ptr0 + (0))
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp32 = tl.load(in_ptr2 + (r1 + (1152*x0)), rmask & xmask, other=0).to(tl.float32)
        tmp30 = tl.where(tmp29 < 0, tmp29 + 1152, tmp29)
        tl.device_assert((0 <= tmp30) & (tmp30 < 1152), "index out of bounds: 0 <= tmp30 < 1152")
        tmp31 = tl.load(in_ptr1 + (r1 + (1152*tmp30)), rmask)
        tmp33 = 0.08838834764831843
        tmp34 = tmp32 * tmp33
        tmp35 = -65504.0
        tmp36 = tl.where(tmp31, tmp34, tmp35)
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp37 - tmp11
        tmp39 = tl.exp(tmp38)
        tmp40 = tmp39 / tmp26
        tmp41 = tmp40.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (1152*x0)), tmp41, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/sz/cszjva3orvxghquc24t3lk7lra76ixzkdnlbslgukltj5uls7eej.py
# Source Nodes: [add_3, add_4, float_4, l__model___tok_embeddings, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.rsqrt]
# add_3 => add_3
# add_4 => add_4
# float_4 => convert_element_type_8
# l__model___tok_embeddings => embedding
# mean_1 => mean_1
# mul_12 => mul_12
# mul_13 => mul_13
# mul_14 => mul_14
# rsqrt_1 => rsqrt_1
# type_as_3 => convert_element_type_9
triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_7 = async_compile.triton('triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_7', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_7', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp4 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tl.where(tmp1 < 0, tmp1 + 32000, tmp1)
        tl.device_assert((0 <= tmp2) & (tmp2 < 32000), "index out of bounds: 0 <= tmp2 < 32000")
        tmp3 = tl.load(in_ptr1 + (r0 + (4096*tmp2)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp6 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp11 = tl.load(in_ptr0 + (0))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp15 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp25 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp13 = tl.where(tmp12 < 0, tmp12 + 32000, tmp12)
        tl.device_assert((0 <= tmp13) & (tmp13 < 32000), "index out of bounds: 0 <= tmp13 < 32000")
        tmp14 = tl.load(in_ptr1 + (r0 + (4096*tmp13)), rmask, other=0).to(tl.float32)
        tmp16 = tmp14 + tmp15
        tmp17 = tmp16.to(tl.float32)
        tmp18 = 4096.0
        tmp19 = tmp9 / tmp18
        tmp20 = 1e-05
        tmp21 = tmp19 + tmp20
        tmp22 = tl.math.rsqrt(tmp21)
        tmp23 = tmp17 * tmp22
        tmp24 = tmp23.to(tl.float32)
        tmp26 = tmp24 * tmp25
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp26, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/54/c544i4wrtradxsbcut7bpg3wt5x6gy4gehbzas2a2woddimanzio.py
# Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
# mul_15 => mul_16
# silu => convert_element_type_10, convert_element_type_11, mul_15, sigmoid
triton_poi_fused_mul_silu_8 = async_compile.triton('triton_poi_fused_mul_silu_8', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_8', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_mul_silu_8(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 11008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(in_out_ptr0 + (x0), tmp6, xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/2s/c2su33oixoxdalexgko4vyx4wns7fej6aheyehqsttbv22sjtmgt.py
# Source Nodes: [add_3, add_5, add_6, float_5, l__model___tok_embeddings, mean_2, mul_16, mul_17, mul_18, rsqrt_2, type_as_4], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.rsqrt]
# add_3 => add_3
# add_5 => add_5
# add_6 => add_6
# float_5 => convert_element_type_12
# l__model___tok_embeddings => embedding
# mean_2 => mean_2
# mul_16 => mul_17
# mul_17 => mul_18
# mul_18 => mul_19
# rsqrt_2 => rsqrt_2
# type_as_4 => convert_element_type_13
triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_9 = async_compile.triton('triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_9', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*i32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_9', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    _tmp11 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp4 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tl.where(tmp1 < 0, tmp1 + 32000, tmp1)
        tl.device_assert((0 <= tmp2) & (tmp2 < 32000), "index out of bounds: 0 <= tmp2 < 32000")
        tmp3 = tl.load(in_ptr1 + (r0 + (4096*tmp2)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp7.to(tl.float32)
        tmp9 = tmp8 * tmp8
        tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
        tmp12 = _tmp11 + tmp10
        _tmp11 = tl.where(rmask, tmp12, _tmp11)
    tmp11 = tl.sum(_tmp11, 1)[:, None]
    tmp13 = tl.load(in_ptr0 + (0))
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp17 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp29 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp15 = tl.where(tmp14 < 0, tmp14 + 32000, tmp14)
        tl.device_assert((0 <= tmp15) & (tmp15 < 32000), "index out of bounds: 0 <= tmp15 < 32000")
        tmp16 = tl.load(in_ptr1 + (r0 + (4096*tmp15)), rmask, other=0).to(tl.float32)
        tmp18 = tmp16 + tmp17
        tmp20 = tmp18 + tmp19
        tmp21 = tmp20.to(tl.float32)
        tmp22 = 4096.0
        tmp23 = tmp11 / tmp22
        tmp24 = 1e-05
        tmp25 = tmp23 + tmp24
        tmp26 = tl.math.rsqrt(tmp25)
        tmp27 = tmp21 * tmp26
        tmp28 = tmp27.to(tl.float32)
        tmp30 = tmp28 * tmp29
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp30, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/xg/cxgnpuamaezztb3o64jvudabjrjixpzmiaxmtpvrubdodxdmc2yq.py
# Source Nodes: [add_10, add_3, add_5, add_9, float_8, l__model___tok_embeddings, mean_3, mul_28, mul_29, mul_30, rsqrt_3, type_as_7], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.rsqrt]
# add_10 => add_10
# add_3 => add_3
# add_5 => add_5
# add_9 => add_9
# float_8 => convert_element_type_20
# l__model___tok_embeddings => embedding
# mean_3 => mean_3
# mul_28 => mul_29
# mul_29 => mul_30
# mul_30 => mul_31
# rsqrt_3 => rsqrt_3
# type_as_7 => convert_element_type_21
triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_10 = async_compile.triton('triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_10', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*i32', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_10', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp4 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp8 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp2 = tl.where(tmp1 < 0, tmp1 + 32000, tmp1)
        tl.device_assert((0 <= tmp2) & (tmp2 < 32000), "index out of bounds: 0 <= tmp2 < 32000")
        tmp3 = tl.load(in_ptr1 + (r0 + (4096*tmp2)), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp9, rmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp15 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp24 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 4096.0
        tmp18 = tmp13 / tmp17
        tmp19 = 1e-05
        tmp20 = tmp18 + tmp19
        tmp21 = tl.math.rsqrt(tmp20)
        tmp22 = tmp16 * tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp25 = tmp23 * tmp24
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp25, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/ez/cez6vylkge5l5klopwsy26ag4pwt742u47jpyotdipbqpsy7upzw.py
# Source Nodes: [add_11, add_12, float_9, mean_4, mul_32, mul_33, mul_34, rsqrt_4, type_as_8], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_11 => add_11
# add_12 => add_12
# float_9 => convert_element_type_24
# mean_4 => mean_4
# mul_32 => mul_34
# mul_33 => mul_35
# mul_34 => mul_36
# rsqrt_4 => rsqrt_4
# type_as_8 => convert_element_type_25
triton_red_fused__to_copy_add_mean_mul_rsqrt_11 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_11', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_11', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_11(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp6 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
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
        r0 = rindex
        tmp8 = tl.load(in_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp9 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp10 = tmp8 + tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = 4096.0
        tmp13 = tmp6 / tmp12
        tmp14 = 1e-05
        tmp15 = tmp13 + tmp14
        tmp16 = tl.math.rsqrt(tmp15)
        tmp17 = tmp11 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp20 = tmp18 * tmp19
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp20, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/z2/cz2w6yex4dki74bs6s32dae2sh4syeffuajcxtaxqf6vm6v3ored.py
# Source Nodes: [add_11, add_15, add_16, float_12, mean_5, mul_44, mul_45, mul_46, rsqrt_5, type_as_11], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_11 => add_11
# add_15 => add_15
# add_16 => add_16
# float_12 => convert_element_type_32
# mean_5 => mean_5
# mul_44 => mul_46
# mul_45 => mul_47
# mul_46 => mul_48
# rsqrt_5 => rsqrt_5
# type_as_11 => convert_element_type_33
triton_red_fused__to_copy_add_mean_mul_rsqrt_12 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_12', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_12', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp8 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp5 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp10 = tl.load(in_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp23 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp12 = tmp10 + tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = tmp14.to(tl.float32)
        tmp16 = 4096.0
        tmp17 = tmp8 / tmp16
        tmp18 = 1e-05
        tmp19 = tmp17 + tmp18
        tmp20 = tl.math.rsqrt(tmp19)
        tmp21 = tmp15 * tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp24 = tmp22 * tmp23
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp24, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/ei/ceipkuwphumiydbejs3dmutmbqrc4natbn7csxv4vq5hj5m5jtez.py
# Source Nodes: [add_11, add_15, add_17, add_18, float_13, mean_6, mul_48, mul_49, mul_50, rsqrt_6, type_as_12], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_11 => add_11
# add_15 => add_15
# add_17 => add_17
# add_18 => add_18
# float_13 => convert_element_type_36
# mean_6 => mean_6
# mul_48 => mul_51
# mul_49 => mul_52
# mul_50 => mul_53
# rsqrt_6 => rsqrt_6
# type_as_12 => convert_element_type_37
triton_red_fused__to_copy_add_mean_mul_rsqrt_13 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_13', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_13', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_13(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp10 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r0), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp7 = tmp6.to(tl.float32)
        tmp8 = tmp7 * tmp7
        tmp9 = tl.broadcast_to(tmp8, [XBLOCK, RBLOCK])
        tmp11 = _tmp10 + tmp9
        _tmp10 = tl.where(rmask, tmp11, _tmp10)
    tmp10 = tl.sum(_tmp10, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp12 = tl.load(in_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp17 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp27 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp18 = tmp16 + tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp20 = 4096.0
        tmp21 = tmp10 / tmp20
        tmp22 = 1e-05
        tmp23 = tmp21 + tmp22
        tmp24 = tl.math.rsqrt(tmp23)
        tmp25 = tmp19 * tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp28 = tmp26 * tmp27
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp28, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/fu/cfuhamfsdj4ok2r3cua3b7dgnxkzrwp3r5gg45ogmwrk4cp5kjzn.py
# Source Nodes: [add_11, add_15, add_17, add_21, add_22, float_16, mean_7, mul_60, mul_61, mul_62, rsqrt_7, type_as_15], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_11 => add_11
# add_15 => add_15
# add_17 => add_17
# add_21 => add_21
# add_22 => add_22
# float_16 => convert_element_type_44
# mean_7 => mean_7
# mul_60 => mul_63
# mul_61 => mul_64
# mul_62 => mul_65
# rsqrt_7 => rsqrt_7
# type_as_15 => convert_element_type_45
triton_red_fused__to_copy_add_mean_mul_rsqrt_14 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_14', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_14', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_14(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp7 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp8, rmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp14 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp23 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = 4096.0
        tmp17 = tmp12 / tmp16
        tmp18 = 1e-05
        tmp19 = tmp17 + tmp18
        tmp20 = tl.math.rsqrt(tmp19)
        tmp21 = tmp15 * tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp24 = tmp22 * tmp23
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp24, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/ko/ckoa67di6lgt6ahjlxrlhjfwtj5ng6hha36osqn2viyjtz4fjvdn.py
# Source Nodes: [add_23, add_27, add_29, add_33, add_34, float_24, mean_11, mul_92, mul_93, mul_94, rsqrt_11, type_as_23], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_23 => add_23
# add_27 => add_27
# add_29 => add_29
# add_33 => add_33
# add_34 => add_34
# float_24 => convert_element_type_68
# mean_11 => mean_11
# mul_92 => mul_97
# mul_93 => mul_98
# mul_94 => mul_99
# rsqrt_11 => rsqrt_11
# type_as_23 => convert_element_type_69
triton_red_fused__to_copy_add_mean_mul_rsqrt_15 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_15', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1, 4096],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_15', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp7 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 + tmp3
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp9 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp8, rmask)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp14 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp23 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp15 = tmp14.to(tl.float32)
        tmp16 = 4096.0
        tmp17 = tmp12 / tmp16
        tmp18 = 1e-05
        tmp19 = tmp17 + tmp18
        tmp20 = tl.math.rsqrt(tmp19)
        tmp21 = tmp15 * tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp24 = tmp22 * tmp23
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp24, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/wy/cwy6bzljvilbazsxtskkn7cnfvikifzowqsnyu5y7uid3eyaglcd.py
# Source Nodes: [truediv], Original ATen: [aten.div]
# truediv => div_32
triton_poi_fused_div_16 = async_compile.triton('triton_poi_fused_div_16', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_16', 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]})
@triton.jit
def triton_poi_fused_div_16(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 32000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), xmask).to(tl.float32)
    tmp1 = 0.5
    tmp2 = tmp0 / tmp1
    tl.store(in_out_ptr0 + (x0), tmp2, xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/4p/c4pjbojzbrhmkdvhpwyr6lwdm67lbrdj3jofahqcj27gwtkfr5mk.py
# Source Nodes: [lt, softmax_32, where_32], Original ATen: [aten._softmax, aten.lt, aten.scalar_tensor, aten.where]
# lt => lt
# softmax_32 => amax_32, convert_element_type_386
# where_32 => full_default_32, where_32
triton_red_fused__softmax_lt_scalar_tensor_where_17 = async_compile.triton('triton_red_fused__softmax_lt_scalar_tensor_where_17', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_lt_scalar_tensor_where_17', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]}
)
@triton.jit
def triton_red_fused__softmax_lt_scalar_tensor_where_17(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 8000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (199)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    _tmp8 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (8000*x0)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp3 = tmp0 < tmp2
        tmp4 = float("-inf")
        tmp5 = tl.where(tmp3, tmp4, tmp0)
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = triton_helpers.maximum(_tmp8, tmp7)
        _tmp8 = tl.where(rmask & xmask, tmp9, _tmp8)
    tmp8 = triton_helpers.max2(_tmp8, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp8, xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/3x/c3xtlyuh2u6zk4qzwfjzeygu4h5zy3m4fgffmf7fsnccfcyoh6du.py
# Source Nodes: [lt, softmax_32, where_32], Original ATen: [aten._softmax, aten.lt, aten.scalar_tensor, aten.where]
# lt => lt
# softmax_32 => amax_32, convert_element_type_386
# where_32 => full_default_32, where_32
triton_per_fused__softmax_lt_scalar_tensor_where_18 = async_compile.triton('triton_per_fused__softmax_lt_scalar_tensor_where_18', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_lt_scalar_tensor_where_18', 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]}
)
@triton.jit
def triton_per_fused__softmax_lt_scalar_tensor_where_18(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, float("-inf"))
    tmp4 = triton_helpers.max2(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
''')


# kernel path: /tmp/torchinductor_mengqy/vb/cvbo42xero3m5n74xtgtlcpu6cwnacsn3pjvd5nsjxeg4skdh4rt.py
# Source Nodes: [lt, softmax_32, where_32], Original ATen: [aten._softmax, aten.lt, aten.scalar_tensor, aten.where]
# lt => lt
# softmax_32 => convert_element_type_386, exp_32, sub_96, sum_33
# where_32 => full_default_32, where_32
triton_red_fused__softmax_lt_scalar_tensor_where_19 = async_compile.triton('triton_red_fused__softmax_lt_scalar_tensor_where_19', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[4, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_lt_scalar_tensor_where_19', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]}
)
@triton.jit
def triton_red_fused__softmax_lt_scalar_tensor_where_19(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 4
    rnumel = 8000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + (199)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp7 = tl.load(in_ptr2 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (8000*x0)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp3 = tmp0 < tmp2
        tmp4 = float("-inf")
        tmp5 = tl.where(tmp3, tmp4, tmp0)
        tmp6 = tmp5.to(tl.float32)
        tmp9 = tmp6 - tmp8
        tmp10 = tl.exp(tmp9)
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask & xmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp12, xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/kr/ckr7za65slc6kojpqqrotfk2lqukuzdmce4spowirirpqk3dsloy.py
# Source Nodes: [lt, softmax_32, where_32], Original ATen: [aten._softmax, aten.lt, aten.scalar_tensor, aten.where]
# lt => lt
# softmax_32 => convert_element_type_386, exp_32, sub_96, sum_33
# where_32 => full_default_32, where_32
triton_per_fused__softmax_lt_scalar_tensor_where_20 = async_compile.triton('triton_per_fused__softmax_lt_scalar_tensor_where_20', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1, 4],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_lt_scalar_tensor_where_20', 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]}
)
@triton.jit
def triton_per_fused__softmax_lt_scalar_tensor_where_20(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4
    RBLOCK: tl.constexpr = 4
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r0 = rindex
    tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp4, None)
''')


# kernel path: /tmp/torchinductor_mengqy/3x/c3x6rncmke2tjouqgljgevy3vxoxk4uuzfg4cz6bwhiyqipf6b3h.py
# Source Nodes: [argmax, exponential_, lt, softmax_32, to, truediv_1, where_32], Original ATen: [aten._softmax, aten._to_copy, aten.argmax, aten.div, aten.exponential, aten.lt, aten.scalar_tensor, aten.where]
# argmax => argmax
# exponential_ => convert_element_type_389, log1p, mul_547, neg
# lt => lt
# softmax_32 => convert_element_type_386, convert_element_type_387, div_33, exp_32, sub_96
# to => convert_element_type_390
# truediv_1 => div_34
# where_32 => full_default_32, where_32
triton_red_fused__softmax__to_copy_argmax_div_exponential_lt_scalar_tensor_where_21 = async_compile.triton('triton_red_fused__softmax__to_copy_argmax_div_exponential_lt_scalar_tensor_where_21', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1, 32768],
    reduction_hint=ReductionHint.DEFAULT,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*i32', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_argmax_div_exponential_lt_scalar_tensor_where_21', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_argmax_div_exponential_lt_scalar_tensor_where_21(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 32000
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (199)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp7 = tl.load(in_ptr1 + (0))
    tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
    tmp11 = tl.load(in_ptr2 + (0))
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    _tmp25 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    _tmp25_index = tl.full([XBLOCK, RBLOCK], 9223372036854775807, tl.int64)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tmp0 < tmp2
        tmp4 = float("-inf")
        tmp5 = tl.where(tmp3, tmp4, tmp0)
        tmp6 = tmp5.to(tl.float32)
        tmp9 = tmp6 - tmp8
        tmp10 = tl.exp(tmp9)
        tmp13 = tmp10 / tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tl.load(in_ptr3 + load_seed_offset)
        tmp16 = r0
        tmp17 = tl.rand(tmp15, (tmp16).to(tl.uint32))
        tmp18 = -tmp17
        tmp19 = tl.math.log1p(tmp18)
        tmp20 = -1.0
        tmp21 = tmp19 * tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp23 = tmp14 / tmp22
        tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
        _tmp25_next, _tmp25_index_next = triton_helpers.maximum_with_index(
            _tmp25, _tmp25_index, tmp24, rindex
        )
        _tmp25 = tl.where(rmask, _tmp25_next, _tmp25)
        _tmp25_index = tl.where(rmask, _tmp25_index_next, _tmp25_index)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp14, rmask)
    _, tmp25_tmp = triton_helpers.max_with_index(_tmp25, _tmp25_index, 1)
    tmp25 = tmp25_tmp[:, None]
    tmp26 = tmp25.to(tl.int32)
    tl.store(out_ptr2 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp26, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4096, ), (1, ))
    assert_size_stride(arg1_1, (4096, ), (1, ))
    assert_size_stride(arg2_1, (4096, ), (1, ))
    assert_size_stride(arg3_1, (4096, ), (1, ))
    assert_size_stride(arg4_1, (4096, ), (1, ))
    assert_size_stride(arg5_1, (4096, ), (1, ))
    assert_size_stride(arg6_1, (4096, ), (1, ))
    assert_size_stride(arg7_1, (4096, ), (1, ))
    assert_size_stride(arg8_1, (4096, ), (1, ))
    assert_size_stride(arg9_1, (4096, ), (1, ))
    assert_size_stride(arg10_1, (4096, ), (1, ))
    assert_size_stride(arg11_1, (4096, ), (1, ))
    assert_size_stride(arg12_1, (4096, ), (1, ))
    assert_size_stride(arg13_1, (4096, ), (1, ))
    assert_size_stride(arg14_1, (4096, ), (1, ))
    assert_size_stride(arg15_1, (4096, ), (1, ))
    assert_size_stride(arg16_1, (4096, ), (1, ))
    assert_size_stride(arg17_1, (4096, ), (1, ))
    assert_size_stride(arg18_1, (4096, ), (1, ))
    assert_size_stride(arg19_1, (4096, ), (1, ))
    assert_size_stride(arg20_1, (4096, ), (1, ))
    assert_size_stride(arg21_1, (4096, ), (1, ))
    assert_size_stride(arg22_1, (4096, ), (1, ))
    assert_size_stride(arg23_1, (4096, ), (1, ))
    assert_size_stride(arg24_1, (4096, ), (1, ))
    assert_size_stride(arg25_1, (4096, ), (1, ))
    assert_size_stride(arg26_1, (4096, ), (1, ))
    assert_size_stride(arg27_1, (4096, ), (1, ))
    assert_size_stride(arg28_1, (4096, ), (1, ))
    assert_size_stride(arg29_1, (4096, ), (1, ))
    assert_size_stride(arg30_1, (4096, ), (1, ))
    assert_size_stride(arg31_1, (4096, ), (1, ))
    assert_size_stride(arg32_1, (4096, ), (1, ))
    assert_size_stride(arg33_1, (4096, ), (1, ))
    assert_size_stride(arg34_1, (4096, ), (1, ))
    assert_size_stride(arg35_1, (4096, ), (1, ))
    assert_size_stride(arg36_1, (4096, ), (1, ))
    assert_size_stride(arg37_1, (4096, ), (1, ))
    assert_size_stride(arg38_1, (4096, ), (1, ))
    assert_size_stride(arg39_1, (4096, ), (1, ))
    assert_size_stride(arg40_1, (4096, ), (1, ))
    assert_size_stride(arg41_1, (4096, ), (1, ))
    assert_size_stride(arg42_1, (4096, ), (1, ))
    assert_size_stride(arg43_1, (4096, ), (1, ))
    assert_size_stride(arg44_1, (4096, ), (1, ))
    assert_size_stride(arg45_1, (4096, ), (1, ))
    assert_size_stride(arg46_1, (4096, ), (1, ))
    assert_size_stride(arg47_1, (4096, ), (1, ))
    assert_size_stride(arg48_1, (4096, ), (1, ))
    assert_size_stride(arg49_1, (4096, ), (1, ))
    assert_size_stride(arg50_1, (4096, ), (1, ))
    assert_size_stride(arg51_1, (4096, ), (1, ))
    assert_size_stride(arg52_1, (4096, ), (1, ))
    assert_size_stride(arg53_1, (4096, ), (1, ))
    assert_size_stride(arg54_1, (4096, ), (1, ))
    assert_size_stride(arg55_1, (4096, ), (1, ))
    assert_size_stride(arg56_1, (4096, ), (1, ))
    assert_size_stride(arg57_1, (4096, ), (1, ))
    assert_size_stride(arg58_1, (4096, ), (1, ))
    assert_size_stride(arg59_1, (4096, ), (1, ))
    assert_size_stride(arg60_1, (4096, ), (1, ))
    assert_size_stride(arg61_1, (4096, ), (1, ))
    assert_size_stride(arg62_1, (4096, ), (1, ))
    assert_size_stride(arg63_1, (4096, ), (1, ))
    assert_size_stride(arg64_1, (4096, ), (1, ))
    assert_size_stride(arg65_1, (32000, 4096), (4096, 1))
    assert_size_stride(arg66_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg67_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg68_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg69_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg70_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg71_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg72_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg73_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg74_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg75_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg76_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg77_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg78_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg79_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg80_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg81_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg82_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg83_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg84_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg85_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg86_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg87_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg88_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg89_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg90_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg91_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg92_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg93_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg94_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg95_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg96_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg97_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg98_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg99_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg100_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg101_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg102_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg103_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg104_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg105_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg106_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg107_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg108_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg109_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg110_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg111_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg112_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg113_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg114_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg115_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg116_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg117_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg118_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg119_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg120_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg121_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg122_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg123_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg124_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg125_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg126_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg127_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg128_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg129_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg130_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg131_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg132_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg133_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg134_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg135_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg136_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg137_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg138_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg139_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg140_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg141_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg142_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg143_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg144_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg145_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg146_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg147_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg148_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg149_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg150_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg151_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg152_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg153_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg154_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg155_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg156_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg157_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg158_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg159_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg160_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg161_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg162_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg163_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg164_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg165_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg166_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg167_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg168_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg169_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg170_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg171_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg172_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg173_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg174_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg175_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg176_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg177_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg178_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg179_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg180_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg181_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg182_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg183_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg184_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg185_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg186_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg187_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg188_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg189_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg190_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg191_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg192_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg193_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg194_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg195_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg196_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg197_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg198_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg199_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg200_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg201_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg202_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg203_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg204_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg205_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg206_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg207_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg208_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg209_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg210_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg211_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg212_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg213_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg214_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg215_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg216_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg217_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg218_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg219_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg220_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg221_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg222_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg223_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg224_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg225_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg226_1, (32000, 4096), (4096, 1))
    assert_size_stride(arg227_1, (2048, 64, 2), (128, 2, 1))
    assert_size_stride(arg228_1, (1152, 1152), (1152, 1))
    assert_size_stride(arg229_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg230_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg231_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg232_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg233_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg234_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg235_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg236_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg237_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg238_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg239_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg240_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg241_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg242_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg243_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg244_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg245_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg246_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg247_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg248_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg249_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg250_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg251_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg252_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg253_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg254_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg255_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg256_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg257_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg258_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg259_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg260_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg261_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg262_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg263_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg264_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg265_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg266_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg267_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg268_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg269_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg270_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg271_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg272_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg273_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg274_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg275_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg276_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg277_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg278_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg279_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg280_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg281_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg282_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg283_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg284_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg285_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg286_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg287_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg288_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg289_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg290_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg291_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg292_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg293_1, (1, 1), (1, 1))
    assert_size_stride(arg294_1, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf1 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, l__model___tok_embeddings, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.rsqrt]
        stream0 = get_cuda_stream(0)
        triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_0.run(arg293_1, arg65_1, arg0_1, buf1, 1, 4096, grid=grid(1), stream=stream0)
        del arg0_1
        buf2 = empty_strided((1, 12288), (12288, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_0_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (1, 4096), (0, 1), 0), reinterpret_tensor(arg66_1, (4096, 12288), (1, 4096), 0), out=buf2)
        del arg66_1
        buf5 = empty_strided((1, 1, 32, 64, 2), (4096, 4096, 128, 2, 1), device='cuda', dtype=torch.float32)
        buf3 = reinterpret_tensor(buf5, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf4 = reinterpret_tensor(buf5, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf8 = empty_strided((1, 1, 32, 64, 2), (4096, 4096, 128, 2, 1), device='cuda', dtype=torch.float32)
        buf6 = reinterpret_tensor(buf8, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf7 = reinterpret_tensor(buf8, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf2, arg294_1, arg227_1, buf3, buf4, buf6, buf7, 2048, grid=grid(2048), stream=stream0)
        buf9 = empty_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [setitem], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg229_1, buf9, 4718592, grid=grid(4718592), stream=stream0)
        del buf3
        del buf4
        del buf6
        del buf7
        buf16 = empty_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [setitem_1], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg230_1, buf16, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem, setitem_1], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf8, buf2, buf9, buf16, 4096, grid=grid(4096), stream=stream0)
        buf11 = reinterpret_tensor(buf1, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf1  # reuse
        # Source Nodes: [type_as_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf5, buf11, 4096, grid=grid(4096), stream=stream0)
        buf12 = empty_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [setitem], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf9, buf12, arg229_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg229_1
        buf13 = empty_strided((32, 1, 1152), (1152, 1152, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf11, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf12, (32, 128, 1152), (147456, 1, 128), 0), out=buf13)
        buf18 = empty_strided((1, 32, 1, 1152), (36864, 1152, 1152, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [getitem, mul_11, softmax, where], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf13, buf18, 32, 1152, grid=grid(32), stream=stream0)
        buf19 = buf12; del buf12  # reuse
        # Source Nodes: [setitem_1], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf16, buf19, arg230_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg230_1
        buf20 = reinterpret_tensor(buf11, (32, 1, 128), (128, 128, 1)); del buf11  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf18, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf19, (32, 1152, 128), (147456, 128, 1), 0), out=buf20)
        buf21 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_0_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf20, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg67_1, (4096, 4096), (1, 4096), 0), out=buf21)
        del arg67_1
        buf23 = reinterpret_tensor(buf20, (1, 1, 4096), (4096, 4096, 1)); del buf20  # reuse
        # Source Nodes: [add_3, add_4, float_4, l__model___tok_embeddings, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_7.run(arg293_1, arg65_1, buf21, arg1_1, buf23, 1, 4096, grid=grid(1), stream=stream0)
        del arg1_1
        buf24 = empty_strided((1, 11008), (11008, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_0_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg68_1, (4096, 11008), (1, 4096), 0), out=buf24)
        del arg68_1
        buf25 = empty_strided((1, 11008), (11008, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_0_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg69_1, (4096, 11008), (1, 4096), 0), out=buf25)
        del arg69_1
        buf26 = reinterpret_tensor(buf24, (1, 1, 11008), (11008, 11008, 1)); del buf24  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf26, buf25, 11008, grid=grid(11008), stream=stream0)
        buf27 = reinterpret_tensor(buf23, (1, 4096), (4096, 1)); del buf23  # reuse
        # Source Nodes: [l__model___layers_0_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (1, 11008), (0, 1), 0), reinterpret_tensor(arg70_1, (11008, 4096), (1, 11008), 0), out=buf27)
        del arg70_1
        buf29 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_5, add_6, float_5, l__model___tok_embeddings, mean_2, mul_16, mul_17, mul_18, rsqrt_2, type_as_4], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_9.run(arg293_1, arg65_1, buf21, buf27, arg2_1, buf29, 1, 4096, grid=grid(1), stream=stream0)
        del arg2_1
        buf30 = buf2; del buf2  # reuse
        # Source Nodes: [l__model___layers_1_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (1, 4096), (0, 1), 0), reinterpret_tensor(arg71_1, (4096, 12288), (1, 4096), 0), out=buf30)
        del arg71_1
        buf33 = buf5; del buf5  # reuse
        buf31 = reinterpret_tensor(buf33, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf32 = reinterpret_tensor(buf33, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf36 = buf8; del buf8  # reuse
        buf34 = reinterpret_tensor(buf36, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf35 = reinterpret_tensor(buf36, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_2, stack_3], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf30, arg294_1, arg227_1, buf31, buf32, buf34, buf35, 2048, grid=grid(2048), stream=stream0)
        buf37 = buf19; del buf19  # reuse
        # Source Nodes: [setitem_2], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg231_1, buf37, 4718592, grid=grid(4718592), stream=stream0)
        del buf31
        del buf32
        del buf34
        del buf35
        buf44 = buf16; del buf16  # reuse
        # Source Nodes: [setitem_3], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg232_1, buf44, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_2, setitem_3], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf36, buf30, buf37, buf44, 4096, grid=grid(4096), stream=stream0)
        buf39 = reinterpret_tensor(buf29, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf29  # reuse
        # Source Nodes: [type_as_5], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf33, buf39, 4096, grid=grid(4096), stream=stream0)
        buf40 = buf9; del buf9  # reuse
        # Source Nodes: [setitem_2], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf37, buf40, arg231_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg231_1
        buf41 = reinterpret_tensor(buf18, (32, 1, 1152), (1152, 1152, 1)); del buf18  # reuse
        # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf39, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf40, (32, 128, 1152), (147456, 1, 128), 0), out=buf41)
        buf46 = reinterpret_tensor(buf13, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf13  # reuse
        # Source Nodes: [getitem, mul_27, softmax_1, where_1], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf41, buf46, 32, 1152, grid=grid(32), stream=stream0)
        buf47 = buf40; del buf40  # reuse
        # Source Nodes: [setitem_3], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf44, buf47, arg232_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg232_1
        buf48 = reinterpret_tensor(buf39, (32, 1, 128), (128, 128, 1)); del buf39  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf46, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf47, (32, 1152, 128), (147456, 128, 1), 0), out=buf48)
        buf49 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_1_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf48, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg72_1, (4096, 4096), (1, 4096), 0), out=buf49)
        del arg72_1
        buf50 = reinterpret_tensor(buf21, (1, 1, 4096), (4096, 4096, 1)); del buf21  # reuse
        buf52 = reinterpret_tensor(buf48, (1, 1, 4096), (4096, 4096, 1)); del buf48  # reuse
        # Source Nodes: [add_10, add_3, add_5, add_9, float_8, l__model___tok_embeddings, mean_3, mul_28, mul_29, mul_30, rsqrt_3, type_as_7], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_10.run(buf50, arg293_1, arg65_1, buf27, buf49, arg3_1, buf52, 1, 4096, grid=grid(1), stream=stream0)
        del arg293_1
        del arg3_1
        del arg65_1
        buf53 = reinterpret_tensor(buf26, (1, 11008), (11008, 1)); del buf26  # reuse
        # Source Nodes: [l__model___layers_1_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (1, 4096), (0, 1), 0), reinterpret_tensor(arg73_1, (4096, 11008), (1, 4096), 0), out=buf53)
        del arg73_1
        buf54 = buf25; del buf25  # reuse
        # Source Nodes: [l__model___layers_1_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg74_1, (4096, 11008), (1, 4096), 0), out=buf54)
        del arg74_1
        buf55 = reinterpret_tensor(buf53, (1, 1, 11008), (11008, 11008, 1)); del buf53  # reuse
        # Source Nodes: [mul_31, silu_1], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf55, buf54, 11008, grid=grid(11008), stream=stream0)
        buf56 = reinterpret_tensor(buf52, (1, 4096), (4096, 1)); del buf52  # reuse
        # Source Nodes: [l__model___layers_1_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf55, (1, 11008), (0, 1), 0), reinterpret_tensor(arg75_1, (11008, 4096), (1, 11008), 0), out=buf56)
        del arg75_1
        buf58 = reinterpret_tensor(buf49, (1, 1, 4096), (4096, 4096, 1)); del buf49  # reuse
        # Source Nodes: [add_11, add_12, float_9, mean_4, mul_32, mul_33, mul_34, rsqrt_4, type_as_8], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf50, buf56, arg4_1, buf58, 1, 4096, grid=grid(1), stream=stream0)
        del arg4_1
        buf59 = buf30; del buf30  # reuse
        # Source Nodes: [l__model___layers_2_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf58, (1, 4096), (0, 1), 0), reinterpret_tensor(arg76_1, (4096, 12288), (1, 4096), 0), out=buf59)
        del arg76_1
        buf62 = buf33; del buf33  # reuse
        buf60 = reinterpret_tensor(buf62, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf61 = reinterpret_tensor(buf62, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf65 = buf36; del buf36  # reuse
        buf63 = reinterpret_tensor(buf65, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf64 = reinterpret_tensor(buf65, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_4, stack_5], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf59, arg294_1, arg227_1, buf60, buf61, buf63, buf64, 2048, grid=grid(2048), stream=stream0)
        buf66 = buf47; del buf47  # reuse
        # Source Nodes: [setitem_4], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg233_1, buf66, 4718592, grid=grid(4718592), stream=stream0)
        del buf60
        del buf61
        del buf63
        del buf64
        buf73 = buf44; del buf44  # reuse
        # Source Nodes: [setitem_5], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg234_1, buf73, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_4, setitem_5], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf65, buf59, buf66, buf73, 4096, grid=grid(4096), stream=stream0)
        buf68 = reinterpret_tensor(buf58, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf58  # reuse
        # Source Nodes: [type_as_9], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf62, buf68, 4096, grid=grid(4096), stream=stream0)
        buf69 = buf37; del buf37  # reuse
        # Source Nodes: [setitem_4], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf66, buf69, arg233_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg233_1
        buf70 = reinterpret_tensor(buf46, (32, 1, 1152), (1152, 1152, 1)); del buf46  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf68, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf69, (32, 128, 1152), (147456, 1, 128), 0), out=buf70)
        buf75 = reinterpret_tensor(buf41, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf41  # reuse
        # Source Nodes: [getitem, mul_43, softmax_2, where_2], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf70, buf75, 32, 1152, grid=grid(32), stream=stream0)
        buf76 = buf69; del buf69  # reuse
        # Source Nodes: [setitem_5], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf73, buf76, arg234_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg234_1
        buf77 = reinterpret_tensor(buf68, (32, 1, 128), (128, 128, 1)); del buf68  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf75, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf76, (32, 1152, 128), (147456, 128, 1), 0), out=buf77)
        buf78 = buf27; del buf27  # reuse
        # Source Nodes: [l__model___layers_2_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf77, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg77_1, (4096, 4096), (1, 4096), 0), out=buf78)
        del arg77_1
        buf80 = reinterpret_tensor(buf77, (1, 1, 4096), (4096, 4096, 1)); del buf77  # reuse
        # Source Nodes: [add_11, add_15, add_16, float_12, mean_5, mul_44, mul_45, mul_46, rsqrt_5, type_as_11], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_12.run(buf50, buf56, buf78, arg5_1, buf80, 1, 4096, grid=grid(1), stream=stream0)
        del arg5_1
        buf81 = reinterpret_tensor(buf55, (1, 11008), (11008, 1)); del buf55  # reuse
        # Source Nodes: [l__model___layers_2_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg78_1, (4096, 11008), (1, 4096), 0), out=buf81)
        del arg78_1
        buf82 = buf54; del buf54  # reuse
        # Source Nodes: [l__model___layers_2_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg79_1, (4096, 11008), (1, 4096), 0), out=buf82)
        del arg79_1
        buf83 = reinterpret_tensor(buf81, (1, 1, 11008), (11008, 11008, 1)); del buf81  # reuse
        # Source Nodes: [mul_47, silu_2], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf83, buf82, 11008, grid=grid(11008), stream=stream0)
        buf84 = reinterpret_tensor(buf80, (1, 4096), (4096, 1)); del buf80  # reuse
        # Source Nodes: [l__model___layers_2_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (1, 11008), (0, 1), 0), reinterpret_tensor(arg80_1, (11008, 4096), (1, 11008), 0), out=buf84)
        del arg80_1
        buf86 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_11, add_15, add_17, add_18, float_13, mean_6, mul_48, mul_49, mul_50, rsqrt_6, type_as_12], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_13.run(buf50, buf56, buf78, buf84, arg6_1, buf86, 1, 4096, grid=grid(1), stream=stream0)
        del arg6_1
        buf87 = buf59; del buf59  # reuse
        # Source Nodes: [l__model___layers_3_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (1, 4096), (0, 1), 0), reinterpret_tensor(arg81_1, (4096, 12288), (1, 4096), 0), out=buf87)
        del arg81_1
        buf90 = buf62; del buf62  # reuse
        buf88 = reinterpret_tensor(buf90, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf89 = reinterpret_tensor(buf90, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf93 = buf65; del buf65  # reuse
        buf91 = reinterpret_tensor(buf93, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf92 = reinterpret_tensor(buf93, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_6, stack_7], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf87, arg294_1, arg227_1, buf88, buf89, buf91, buf92, 2048, grid=grid(2048), stream=stream0)
        buf94 = buf76; del buf76  # reuse
        # Source Nodes: [setitem_6], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg235_1, buf94, 4718592, grid=grid(4718592), stream=stream0)
        del buf88
        del buf89
        del buf91
        del buf92
        buf101 = buf73; del buf73  # reuse
        # Source Nodes: [setitem_7], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg236_1, buf101, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_6, setitem_7], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf93, buf87, buf94, buf101, 4096, grid=grid(4096), stream=stream0)
        buf96 = reinterpret_tensor(buf86, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf86  # reuse
        # Source Nodes: [type_as_13], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf90, buf96, 4096, grid=grid(4096), stream=stream0)
        buf97 = buf66; del buf66  # reuse
        # Source Nodes: [setitem_6], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf94, buf97, arg235_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg235_1
        buf98 = reinterpret_tensor(buf75, (32, 1, 1152), (1152, 1152, 1)); del buf75  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf96, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf97, (32, 128, 1152), (147456, 1, 128), 0), out=buf98)
        buf103 = reinterpret_tensor(buf70, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf70  # reuse
        # Source Nodes: [getitem, mul_59, softmax_3, where_3], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf98, buf103, 32, 1152, grid=grid(32), stream=stream0)
        buf104 = buf97; del buf97  # reuse
        # Source Nodes: [setitem_7], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf101, buf104, arg236_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg236_1
        buf105 = reinterpret_tensor(buf96, (32, 1, 128), (128, 128, 1)); del buf96  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf103, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf104, (32, 1152, 128), (147456, 128, 1), 0), out=buf105)
        buf106 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_3_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg82_1, (4096, 4096), (1, 4096), 0), out=buf106)
        del arg82_1
        buf107 = reinterpret_tensor(buf106, (1, 1, 4096), (4096, 4096, 1)); del buf106  # reuse
        buf109 = reinterpret_tensor(buf105, (1, 1, 4096), (4096, 4096, 1)); del buf105  # reuse
        # Source Nodes: [add_11, add_15, add_17, add_21, add_22, float_16, mean_7, mul_60, mul_61, mul_62, rsqrt_7, type_as_15], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_14.run(buf107, buf50, buf56, buf78, buf84, arg7_1, buf109, 1, 4096, grid=grid(1), stream=stream0)
        del arg7_1
        buf110 = reinterpret_tensor(buf83, (1, 11008), (11008, 1)); del buf83  # reuse
        # Source Nodes: [l__model___layers_3_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (1, 4096), (0, 1), 0), reinterpret_tensor(arg83_1, (4096, 11008), (1, 4096), 0), out=buf110)
        del arg83_1
        buf111 = buf82; del buf82  # reuse
        # Source Nodes: [l__model___layers_3_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf109, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg84_1, (4096, 11008), (1, 4096), 0), out=buf111)
        del arg84_1
        buf112 = reinterpret_tensor(buf110, (1, 1, 11008), (11008, 11008, 1)); del buf110  # reuse
        # Source Nodes: [mul_63, silu_3], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf112, buf111, 11008, grid=grid(11008), stream=stream0)
        buf113 = reinterpret_tensor(buf109, (1, 4096), (4096, 1)); del buf109  # reuse
        # Source Nodes: [l__model___layers_3_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf112, (1, 11008), (0, 1), 0), reinterpret_tensor(arg85_1, (11008, 4096), (1, 11008), 0), out=buf113)
        del arg85_1
        buf115 = reinterpret_tensor(buf84, (1, 1, 4096), (4096, 4096, 1)); del buf84  # reuse
        # Source Nodes: [add_23, add_24, float_17, mean_8, mul_64, mul_65, mul_66, rsqrt_8, type_as_16], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf107, buf113, arg8_1, buf115, 1, 4096, grid=grid(1), stream=stream0)
        del arg8_1
        buf116 = buf87; del buf87  # reuse
        # Source Nodes: [l__model___layers_4_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (1, 4096), (0, 1), 0), reinterpret_tensor(arg86_1, (4096, 12288), (1, 4096), 0), out=buf116)
        del arg86_1
        buf119 = buf90; del buf90  # reuse
        buf117 = reinterpret_tensor(buf119, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf118 = reinterpret_tensor(buf119, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf122 = buf93; del buf93  # reuse
        buf120 = reinterpret_tensor(buf122, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf121 = reinterpret_tensor(buf122, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_8, stack_9], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf116, arg294_1, arg227_1, buf117, buf118, buf120, buf121, 2048, grid=grid(2048), stream=stream0)
        buf123 = buf104; del buf104  # reuse
        # Source Nodes: [setitem_8], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg237_1, buf123, 4718592, grid=grid(4718592), stream=stream0)
        del buf117
        del buf118
        del buf120
        del buf121
        buf130 = buf101; del buf101  # reuse
        # Source Nodes: [setitem_9], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg238_1, buf130, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_8, setitem_9], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf122, buf116, buf123, buf130, 4096, grid=grid(4096), stream=stream0)
        buf125 = reinterpret_tensor(buf115, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf115  # reuse
        # Source Nodes: [type_as_17], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf119, buf125, 4096, grid=grid(4096), stream=stream0)
        buf126 = buf94; del buf94  # reuse
        # Source Nodes: [setitem_8], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf123, buf126, arg237_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg237_1
        buf127 = reinterpret_tensor(buf103, (32, 1, 1152), (1152, 1152, 1)); del buf103  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf125, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf126, (32, 128, 1152), (147456, 1, 128), 0), out=buf127)
        buf132 = reinterpret_tensor(buf98, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf98  # reuse
        # Source Nodes: [getitem, mul_75, softmax_4, where_4], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf127, buf132, 32, 1152, grid=grid(32), stream=stream0)
        buf133 = buf126; del buf126  # reuse
        # Source Nodes: [setitem_9], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf130, buf133, arg238_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg238_1
        buf134 = reinterpret_tensor(buf125, (32, 1, 128), (128, 128, 1)); del buf125  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf132, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf133, (32, 1152, 128), (147456, 128, 1), 0), out=buf134)
        buf135 = buf78; del buf78  # reuse
        # Source Nodes: [l__model___layers_4_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf134, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg87_1, (4096, 4096), (1, 4096), 0), out=buf135)
        del arg87_1
        buf137 = reinterpret_tensor(buf134, (1, 1, 4096), (4096, 4096, 1)); del buf134  # reuse
        # Source Nodes: [add_23, add_27, add_28, float_20, mean_9, mul_76, mul_77, mul_78, rsqrt_9, type_as_19], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_12.run(buf107, buf113, buf135, arg9_1, buf137, 1, 4096, grid=grid(1), stream=stream0)
        del arg9_1
        buf138 = reinterpret_tensor(buf112, (1, 11008), (11008, 1)); del buf112  # reuse
        # Source Nodes: [l__model___layers_4_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf137, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg88_1, (4096, 11008), (1, 4096), 0), out=buf138)
        del arg88_1
        buf139 = buf111; del buf111  # reuse
        # Source Nodes: [l__model___layers_4_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf137, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg89_1, (4096, 11008), (1, 4096), 0), out=buf139)
        del arg89_1
        buf140 = reinterpret_tensor(buf138, (1, 1, 11008), (11008, 11008, 1)); del buf138  # reuse
        # Source Nodes: [mul_79, silu_4], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf140, buf139, 11008, grid=grid(11008), stream=stream0)
        buf141 = reinterpret_tensor(buf137, (1, 4096), (4096, 1)); del buf137  # reuse
        # Source Nodes: [l__model___layers_4_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf140, (1, 11008), (0, 1), 0), reinterpret_tensor(arg90_1, (11008, 4096), (1, 11008), 0), out=buf141)
        del arg90_1
        buf143 = reinterpret_tensor(buf56, (1, 1, 4096), (4096, 4096, 1)); del buf56  # reuse
        # Source Nodes: [add_23, add_27, add_29, add_30, float_21, mean_10, mul_80, mul_81, mul_82, rsqrt_10, type_as_20], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_13.run(buf107, buf113, buf135, buf141, arg10_1, buf143, 1, 4096, grid=grid(1), stream=stream0)
        del arg10_1
        buf144 = buf116; del buf116  # reuse
        # Source Nodes: [l__model___layers_5_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf143, (1, 4096), (0, 1), 0), reinterpret_tensor(arg91_1, (4096, 12288), (1, 4096), 0), out=buf144)
        del arg91_1
        buf147 = buf119; del buf119  # reuse
        buf145 = reinterpret_tensor(buf147, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf146 = reinterpret_tensor(buf147, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf150 = buf122; del buf122  # reuse
        buf148 = reinterpret_tensor(buf150, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf149 = reinterpret_tensor(buf150, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_10, stack_11], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf144, arg294_1, arg227_1, buf145, buf146, buf148, buf149, 2048, grid=grid(2048), stream=stream0)
        buf151 = buf133; del buf133  # reuse
        # Source Nodes: [setitem_10], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg239_1, buf151, 4718592, grid=grid(4718592), stream=stream0)
        del buf145
        del buf146
        del buf148
        del buf149
        buf158 = buf130; del buf130  # reuse
        # Source Nodes: [setitem_11], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg240_1, buf158, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_10, setitem_11], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf150, buf144, buf151, buf158, 4096, grid=grid(4096), stream=stream0)
        buf153 = reinterpret_tensor(buf143, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf143  # reuse
        # Source Nodes: [type_as_21], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf147, buf153, 4096, grid=grid(4096), stream=stream0)
        buf154 = buf123; del buf123  # reuse
        # Source Nodes: [setitem_10], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf151, buf154, arg239_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg239_1
        buf155 = reinterpret_tensor(buf132, (32, 1, 1152), (1152, 1152, 1)); del buf132  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf153, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf154, (32, 128, 1152), (147456, 1, 128), 0), out=buf155)
        buf160 = reinterpret_tensor(buf127, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf127  # reuse
        # Source Nodes: [getitem, mul_91, softmax_5, where_5], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf155, buf160, 32, 1152, grid=grid(32), stream=stream0)
        buf161 = buf154; del buf154  # reuse
        # Source Nodes: [setitem_11], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf158, buf161, arg240_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg240_1
        buf162 = reinterpret_tensor(buf153, (32, 1, 128), (128, 128, 1)); del buf153  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf160, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf161, (32, 1152, 128), (147456, 128, 1), 0), out=buf162)
        buf163 = reinterpret_tensor(buf50, (1, 4096), (4096, 1)); del buf50  # reuse
        # Source Nodes: [l__model___layers_5_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf162, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg92_1, (4096, 4096), (1, 4096), 0), out=buf163)
        del arg92_1
        buf164 = buf107; del buf107  # reuse
        buf166 = reinterpret_tensor(buf162, (1, 1, 4096), (4096, 4096, 1)); del buf162  # reuse
        # Source Nodes: [add_23, add_27, add_29, add_33, add_34, float_24, mean_11, mul_92, mul_93, mul_94, rsqrt_11, type_as_23], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(buf164, buf113, buf135, buf141, buf163, arg11_1, buf166, 1, 4096, grid=grid(1), stream=stream0)
        del arg11_1
        buf167 = reinterpret_tensor(buf140, (1, 11008), (11008, 1)); del buf140  # reuse
        # Source Nodes: [l__model___layers_5_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (1, 4096), (0, 1), 0), reinterpret_tensor(arg93_1, (4096, 11008), (1, 4096), 0), out=buf167)
        del arg93_1
        buf168 = buf139; del buf139  # reuse
        # Source Nodes: [l__model___layers_5_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg94_1, (4096, 11008), (1, 4096), 0), out=buf168)
        del arg94_1
        buf169 = reinterpret_tensor(buf167, (1, 1, 11008), (11008, 11008, 1)); del buf167  # reuse
        # Source Nodes: [mul_95, silu_5], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf169, buf168, 11008, grid=grid(11008), stream=stream0)
        buf170 = reinterpret_tensor(buf166, (1, 4096), (4096, 1)); del buf166  # reuse
        # Source Nodes: [l__model___layers_5_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (1, 11008), (0, 1), 0), reinterpret_tensor(arg95_1, (11008, 4096), (1, 11008), 0), out=buf170)
        del arg95_1
        buf172 = reinterpret_tensor(buf163, (1, 1, 4096), (4096, 4096, 1)); del buf163  # reuse
        # Source Nodes: [add_35, add_36, float_25, mean_12, mul_96, mul_97, mul_98, rsqrt_12, type_as_24], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf164, buf170, arg12_1, buf172, 1, 4096, grid=grid(1), stream=stream0)
        del arg12_1
        buf173 = buf144; del buf144  # reuse
        # Source Nodes: [l__model___layers_6_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf172, (1, 4096), (0, 1), 0), reinterpret_tensor(arg96_1, (4096, 12288), (1, 4096), 0), out=buf173)
        del arg96_1
        buf176 = buf147; del buf147  # reuse
        buf174 = reinterpret_tensor(buf176, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf175 = reinterpret_tensor(buf176, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf179 = buf150; del buf150  # reuse
        buf177 = reinterpret_tensor(buf179, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf178 = reinterpret_tensor(buf179, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_12, stack_13], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf173, arg294_1, arg227_1, buf174, buf175, buf177, buf178, 2048, grid=grid(2048), stream=stream0)
        buf180 = buf161; del buf161  # reuse
        # Source Nodes: [setitem_12], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg241_1, buf180, 4718592, grid=grid(4718592), stream=stream0)
        del buf174
        del buf175
        del buf177
        del buf178
        buf187 = buf158; del buf158  # reuse
        # Source Nodes: [setitem_13], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg242_1, buf187, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_12, setitem_13], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf179, buf173, buf180, buf187, 4096, grid=grid(4096), stream=stream0)
        buf182 = reinterpret_tensor(buf172, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf172  # reuse
        # Source Nodes: [type_as_25], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf176, buf182, 4096, grid=grid(4096), stream=stream0)
        buf183 = buf151; del buf151  # reuse
        # Source Nodes: [setitem_12], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf180, buf183, arg241_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg241_1
        buf184 = reinterpret_tensor(buf160, (32, 1, 1152), (1152, 1152, 1)); del buf160  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf182, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf183, (32, 128, 1152), (147456, 1, 128), 0), out=buf184)
        buf189 = reinterpret_tensor(buf155, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf155  # reuse
        # Source Nodes: [getitem, mul_107, softmax_6, where_6], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf184, buf189, 32, 1152, grid=grid(32), stream=stream0)
        buf190 = buf183; del buf183  # reuse
        # Source Nodes: [setitem_13], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf187, buf190, arg242_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg242_1
        buf191 = reinterpret_tensor(buf182, (32, 1, 128), (128, 128, 1)); del buf182  # reuse
        # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf189, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf190, (32, 1152, 128), (147456, 128, 1), 0), out=buf191)
        buf192 = buf141; del buf141  # reuse
        # Source Nodes: [l__model___layers_6_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf191, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg97_1, (4096, 4096), (1, 4096), 0), out=buf192)
        del arg97_1
        buf194 = reinterpret_tensor(buf191, (1, 1, 4096), (4096, 4096, 1)); del buf191  # reuse
        # Source Nodes: [add_35, add_39, add_40, float_28, mean_13, mul_108, mul_109, mul_110, rsqrt_13, type_as_27], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_12.run(buf164, buf170, buf192, arg13_1, buf194, 1, 4096, grid=grid(1), stream=stream0)
        del arg13_1
        buf195 = reinterpret_tensor(buf169, (1, 11008), (11008, 1)); del buf169  # reuse
        # Source Nodes: [l__model___layers_6_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf194, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg98_1, (4096, 11008), (1, 4096), 0), out=buf195)
        del arg98_1
        buf196 = buf168; del buf168  # reuse
        # Source Nodes: [l__model___layers_6_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf194, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg99_1, (4096, 11008), (1, 4096), 0), out=buf196)
        del arg99_1
        buf197 = reinterpret_tensor(buf195, (1, 1, 11008), (11008, 11008, 1)); del buf195  # reuse
        # Source Nodes: [mul_111, silu_6], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf197, buf196, 11008, grid=grid(11008), stream=stream0)
        buf198 = reinterpret_tensor(buf194, (1, 4096), (4096, 1)); del buf194  # reuse
        # Source Nodes: [l__model___layers_6_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf197, (1, 11008), (0, 1), 0), reinterpret_tensor(arg100_1, (11008, 4096), (1, 11008), 0), out=buf198)
        del arg100_1
        buf200 = reinterpret_tensor(buf135, (1, 1, 4096), (4096, 4096, 1)); del buf135  # reuse
        # Source Nodes: [add_35, add_39, add_41, add_42, float_29, mean_14, mul_112, mul_113, mul_114, rsqrt_14, type_as_28], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_13.run(buf164, buf170, buf192, buf198, arg14_1, buf200, 1, 4096, grid=grid(1), stream=stream0)
        del arg14_1
        buf201 = buf173; del buf173  # reuse
        # Source Nodes: [l__model___layers_7_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf200, (1, 4096), (0, 1), 0), reinterpret_tensor(arg101_1, (4096, 12288), (1, 4096), 0), out=buf201)
        del arg101_1
        buf204 = buf176; del buf176  # reuse
        buf202 = reinterpret_tensor(buf204, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf203 = reinterpret_tensor(buf204, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf207 = buf179; del buf179  # reuse
        buf205 = reinterpret_tensor(buf207, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf206 = reinterpret_tensor(buf207, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_14, stack_15], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf201, arg294_1, arg227_1, buf202, buf203, buf205, buf206, 2048, grid=grid(2048), stream=stream0)
        buf208 = buf190; del buf190  # reuse
        # Source Nodes: [setitem_14], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg243_1, buf208, 4718592, grid=grid(4718592), stream=stream0)
        del buf202
        del buf203
        del buf205
        del buf206
        buf215 = buf187; del buf187  # reuse
        # Source Nodes: [setitem_15], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg244_1, buf215, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_14, setitem_15], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf207, buf201, buf208, buf215, 4096, grid=grid(4096), stream=stream0)
        buf210 = reinterpret_tensor(buf200, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf200  # reuse
        # Source Nodes: [type_as_29], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf204, buf210, 4096, grid=grid(4096), stream=stream0)
        buf211 = buf180; del buf180  # reuse
        # Source Nodes: [setitem_14], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf208, buf211, arg243_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg243_1
        buf212 = reinterpret_tensor(buf189, (32, 1, 1152), (1152, 1152, 1)); del buf189  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf210, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf211, (32, 128, 1152), (147456, 1, 128), 0), out=buf212)
        buf217 = reinterpret_tensor(buf184, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf184  # reuse
        # Source Nodes: [getitem, mul_123, softmax_7, where_7], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf212, buf217, 32, 1152, grid=grid(32), stream=stream0)
        buf218 = buf211; del buf211  # reuse
        # Source Nodes: [setitem_15], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf215, buf218, arg244_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg244_1
        buf219 = reinterpret_tensor(buf210, (32, 1, 128), (128, 128, 1)); del buf210  # reuse
        # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf217, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf218, (32, 1152, 128), (147456, 128, 1), 0), out=buf219)
        buf220 = buf113; del buf113  # reuse
        # Source Nodes: [l__model___layers_7_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf219, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg102_1, (4096, 4096), (1, 4096), 0), out=buf220)
        del arg102_1
        buf221 = buf164; del buf164  # reuse
        buf223 = reinterpret_tensor(buf219, (1, 1, 4096), (4096, 4096, 1)); del buf219  # reuse
        # Source Nodes: [add_35, add_39, add_41, add_45, add_46, float_32, mean_15, mul_124, mul_125, mul_126, rsqrt_15, type_as_31], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(buf221, buf170, buf192, buf198, buf220, arg15_1, buf223, 1, 4096, grid=grid(1), stream=stream0)
        del arg15_1
        buf224 = reinterpret_tensor(buf197, (1, 11008), (11008, 1)); del buf197  # reuse
        # Source Nodes: [l__model___layers_7_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (1, 4096), (0, 1), 0), reinterpret_tensor(arg103_1, (4096, 11008), (1, 4096), 0), out=buf224)
        del arg103_1
        buf225 = buf196; del buf196  # reuse
        # Source Nodes: [l__model___layers_7_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg104_1, (4096, 11008), (1, 4096), 0), out=buf225)
        del arg104_1
        buf226 = reinterpret_tensor(buf224, (1, 1, 11008), (11008, 11008, 1)); del buf224  # reuse
        # Source Nodes: [mul_127, silu_7], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf226, buf225, 11008, grid=grid(11008), stream=stream0)
        buf227 = reinterpret_tensor(buf223, (1, 4096), (4096, 1)); del buf223  # reuse
        # Source Nodes: [l__model___layers_7_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf226, (1, 11008), (0, 1), 0), reinterpret_tensor(arg105_1, (11008, 4096), (1, 11008), 0), out=buf227)
        del arg105_1
        buf229 = reinterpret_tensor(buf220, (1, 1, 4096), (4096, 4096, 1)); del buf220  # reuse
        # Source Nodes: [add_47, add_48, float_33, mean_16, mul_128, mul_129, mul_130, rsqrt_16, type_as_32], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf221, buf227, arg16_1, buf229, 1, 4096, grid=grid(1), stream=stream0)
        del arg16_1
        buf230 = buf201; del buf201  # reuse
        # Source Nodes: [l__model___layers_8_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf229, (1, 4096), (0, 1), 0), reinterpret_tensor(arg106_1, (4096, 12288), (1, 4096), 0), out=buf230)
        del arg106_1
        buf233 = buf204; del buf204  # reuse
        buf231 = reinterpret_tensor(buf233, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf232 = reinterpret_tensor(buf233, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf236 = buf207; del buf207  # reuse
        buf234 = reinterpret_tensor(buf236, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf235 = reinterpret_tensor(buf236, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_16, stack_17], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf230, arg294_1, arg227_1, buf231, buf232, buf234, buf235, 2048, grid=grid(2048), stream=stream0)
        buf237 = buf218; del buf218  # reuse
        # Source Nodes: [setitem_16], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg245_1, buf237, 4718592, grid=grid(4718592), stream=stream0)
        del buf231
        del buf232
        del buf234
        del buf235
        buf244 = buf215; del buf215  # reuse
        # Source Nodes: [setitem_17], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg246_1, buf244, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_16, setitem_17], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf236, buf230, buf237, buf244, 4096, grid=grid(4096), stream=stream0)
        buf239 = reinterpret_tensor(buf229, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf229  # reuse
        # Source Nodes: [type_as_33], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf233, buf239, 4096, grid=grid(4096), stream=stream0)
        buf240 = buf208; del buf208  # reuse
        # Source Nodes: [setitem_16], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf237, buf240, arg245_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg245_1
        buf241 = reinterpret_tensor(buf217, (32, 1, 1152), (1152, 1152, 1)); del buf217  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf239, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf240, (32, 128, 1152), (147456, 1, 128), 0), out=buf241)
        buf246 = reinterpret_tensor(buf212, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf212  # reuse
        # Source Nodes: [getitem, mul_139, softmax_8, where_8], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf241, buf246, 32, 1152, grid=grid(32), stream=stream0)
        buf247 = buf240; del buf240  # reuse
        # Source Nodes: [setitem_17], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf244, buf247, arg246_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg246_1
        buf248 = reinterpret_tensor(buf239, (32, 1, 128), (128, 128, 1)); del buf239  # reuse
        # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf246, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf247, (32, 1152, 128), (147456, 128, 1), 0), out=buf248)
        buf249 = buf198; del buf198  # reuse
        # Source Nodes: [l__model___layers_8_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf248, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg107_1, (4096, 4096), (1, 4096), 0), out=buf249)
        del arg107_1
        buf251 = reinterpret_tensor(buf248, (1, 1, 4096), (4096, 4096, 1)); del buf248  # reuse
        # Source Nodes: [add_47, add_51, add_52, float_36, mean_17, mul_140, mul_141, mul_142, rsqrt_17, type_as_35], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_12.run(buf221, buf227, buf249, arg17_1, buf251, 1, 4096, grid=grid(1), stream=stream0)
        del arg17_1
        buf252 = reinterpret_tensor(buf226, (1, 11008), (11008, 1)); del buf226  # reuse
        # Source Nodes: [l__model___layers_8_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf251, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg108_1, (4096, 11008), (1, 4096), 0), out=buf252)
        del arg108_1
        buf253 = buf225; del buf225  # reuse
        # Source Nodes: [l__model___layers_8_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf251, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg109_1, (4096, 11008), (1, 4096), 0), out=buf253)
        del arg109_1
        buf254 = reinterpret_tensor(buf252, (1, 1, 11008), (11008, 11008, 1)); del buf252  # reuse
        # Source Nodes: [mul_143, silu_8], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf254, buf253, 11008, grid=grid(11008), stream=stream0)
        buf255 = reinterpret_tensor(buf251, (1, 4096), (4096, 1)); del buf251  # reuse
        # Source Nodes: [l__model___layers_8_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf254, (1, 11008), (0, 1), 0), reinterpret_tensor(arg110_1, (11008, 4096), (1, 11008), 0), out=buf255)
        del arg110_1
        buf257 = reinterpret_tensor(buf192, (1, 1, 4096), (4096, 4096, 1)); del buf192  # reuse
        # Source Nodes: [add_47, add_51, add_53, add_54, float_37, mean_18, mul_144, mul_145, mul_146, rsqrt_18, type_as_36], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_13.run(buf221, buf227, buf249, buf255, arg18_1, buf257, 1, 4096, grid=grid(1), stream=stream0)
        del arg18_1
        buf258 = buf230; del buf230  # reuse
        # Source Nodes: [l__model___layers_9_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf257, (1, 4096), (0, 1), 0), reinterpret_tensor(arg111_1, (4096, 12288), (1, 4096), 0), out=buf258)
        del arg111_1
        buf261 = buf233; del buf233  # reuse
        buf259 = reinterpret_tensor(buf261, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf260 = reinterpret_tensor(buf261, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf264 = buf236; del buf236  # reuse
        buf262 = reinterpret_tensor(buf264, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf263 = reinterpret_tensor(buf264, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_18, stack_19], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf258, arg294_1, arg227_1, buf259, buf260, buf262, buf263, 2048, grid=grid(2048), stream=stream0)
        buf265 = buf247; del buf247  # reuse
        # Source Nodes: [setitem_18], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg247_1, buf265, 4718592, grid=grid(4718592), stream=stream0)
        del buf259
        del buf260
        del buf262
        del buf263
        buf272 = buf244; del buf244  # reuse
        # Source Nodes: [setitem_19], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg248_1, buf272, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_18, setitem_19], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf264, buf258, buf265, buf272, 4096, grid=grid(4096), stream=stream0)
        buf267 = reinterpret_tensor(buf257, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf257  # reuse
        # Source Nodes: [type_as_37], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf261, buf267, 4096, grid=grid(4096), stream=stream0)
        buf268 = buf237; del buf237  # reuse
        # Source Nodes: [setitem_18], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf265, buf268, arg247_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg247_1
        buf269 = reinterpret_tensor(buf246, (32, 1, 1152), (1152, 1152, 1)); del buf246  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf267, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf268, (32, 128, 1152), (147456, 1, 128), 0), out=buf269)
        buf274 = reinterpret_tensor(buf241, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf241  # reuse
        # Source Nodes: [getitem, mul_155, softmax_9, where_9], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf269, buf274, 32, 1152, grid=grid(32), stream=stream0)
        buf275 = buf268; del buf268  # reuse
        # Source Nodes: [setitem_19], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf272, buf275, arg248_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg248_1
        buf276 = reinterpret_tensor(buf267, (32, 1, 128), (128, 128, 1)); del buf267  # reuse
        # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf274, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf275, (32, 1152, 128), (147456, 128, 1), 0), out=buf276)
        buf277 = buf170; del buf170  # reuse
        # Source Nodes: [l__model___layers_9_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf276, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg112_1, (4096, 4096), (1, 4096), 0), out=buf277)
        del arg112_1
        buf278 = buf221; del buf221  # reuse
        buf280 = reinterpret_tensor(buf276, (1, 1, 4096), (4096, 4096, 1)); del buf276  # reuse
        # Source Nodes: [add_47, add_51, add_53, add_57, add_58, float_40, mean_19, mul_156, mul_157, mul_158, rsqrt_19, type_as_39], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(buf278, buf227, buf249, buf255, buf277, arg19_1, buf280, 1, 4096, grid=grid(1), stream=stream0)
        del arg19_1
        buf281 = reinterpret_tensor(buf254, (1, 11008), (11008, 1)); del buf254  # reuse
        # Source Nodes: [l__model___layers_9_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (1, 4096), (0, 1), 0), reinterpret_tensor(arg113_1, (4096, 11008), (1, 4096), 0), out=buf281)
        del arg113_1
        buf282 = buf253; del buf253  # reuse
        # Source Nodes: [l__model___layers_9_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf280, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg114_1, (4096, 11008), (1, 4096), 0), out=buf282)
        del arg114_1
        buf283 = reinterpret_tensor(buf281, (1, 1, 11008), (11008, 11008, 1)); del buf281  # reuse
        # Source Nodes: [mul_159, silu_9], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf283, buf282, 11008, grid=grid(11008), stream=stream0)
        buf284 = reinterpret_tensor(buf280, (1, 4096), (4096, 1)); del buf280  # reuse
        # Source Nodes: [l__model___layers_9_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf283, (1, 11008), (0, 1), 0), reinterpret_tensor(arg115_1, (11008, 4096), (1, 11008), 0), out=buf284)
        del arg115_1
        buf286 = reinterpret_tensor(buf277, (1, 1, 4096), (4096, 4096, 1)); del buf277  # reuse
        # Source Nodes: [add_59, add_60, float_41, mean_20, mul_160, mul_161, mul_162, rsqrt_20, type_as_40], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf278, buf284, arg20_1, buf286, 1, 4096, grid=grid(1), stream=stream0)
        del arg20_1
        buf287 = buf258; del buf258  # reuse
        # Source Nodes: [l__model___layers_10_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf286, (1, 4096), (0, 1), 0), reinterpret_tensor(arg116_1, (4096, 12288), (1, 4096), 0), out=buf287)
        del arg116_1
        buf290 = buf261; del buf261  # reuse
        buf288 = reinterpret_tensor(buf290, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf289 = reinterpret_tensor(buf290, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf293 = buf264; del buf264  # reuse
        buf291 = reinterpret_tensor(buf293, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf292 = reinterpret_tensor(buf293, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_20, stack_21], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf287, arg294_1, arg227_1, buf288, buf289, buf291, buf292, 2048, grid=grid(2048), stream=stream0)
        buf294 = buf275; del buf275  # reuse
        # Source Nodes: [setitem_20], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg249_1, buf294, 4718592, grid=grid(4718592), stream=stream0)
        del buf288
        del buf289
        del buf291
        del buf292
        buf301 = buf272; del buf272  # reuse
        # Source Nodes: [setitem_21], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg250_1, buf301, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_20, setitem_21], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf293, buf287, buf294, buf301, 4096, grid=grid(4096), stream=stream0)
        buf296 = reinterpret_tensor(buf286, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf286  # reuse
        # Source Nodes: [type_as_41], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf290, buf296, 4096, grid=grid(4096), stream=stream0)
        buf297 = buf265; del buf265  # reuse
        # Source Nodes: [setitem_20], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf294, buf297, arg249_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg249_1
        buf298 = reinterpret_tensor(buf274, (32, 1, 1152), (1152, 1152, 1)); del buf274  # reuse
        # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf296, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf297, (32, 128, 1152), (147456, 1, 128), 0), out=buf298)
        buf303 = reinterpret_tensor(buf269, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf269  # reuse
        # Source Nodes: [getitem, mul_171, softmax_10, where_10], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf298, buf303, 32, 1152, grid=grid(32), stream=stream0)
        buf304 = buf297; del buf297  # reuse
        # Source Nodes: [setitem_21], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf301, buf304, arg250_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg250_1
        buf305 = reinterpret_tensor(buf296, (32, 1, 128), (128, 128, 1)); del buf296  # reuse
        # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf303, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf304, (32, 1152, 128), (147456, 128, 1), 0), out=buf305)
        buf306 = buf255; del buf255  # reuse
        # Source Nodes: [l__model___layers_10_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf305, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg117_1, (4096, 4096), (1, 4096), 0), out=buf306)
        del arg117_1
        buf308 = reinterpret_tensor(buf305, (1, 1, 4096), (4096, 4096, 1)); del buf305  # reuse
        # Source Nodes: [add_59, add_63, add_64, float_44, mean_21, mul_172, mul_173, mul_174, rsqrt_21, type_as_43], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_12.run(buf278, buf284, buf306, arg21_1, buf308, 1, 4096, grid=grid(1), stream=stream0)
        del arg21_1
        buf309 = reinterpret_tensor(buf283, (1, 11008), (11008, 1)); del buf283  # reuse
        # Source Nodes: [l__model___layers_10_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf308, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg118_1, (4096, 11008), (1, 4096), 0), out=buf309)
        del arg118_1
        buf310 = buf282; del buf282  # reuse
        # Source Nodes: [l__model___layers_10_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf308, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg119_1, (4096, 11008), (1, 4096), 0), out=buf310)
        del arg119_1
        buf311 = reinterpret_tensor(buf309, (1, 1, 11008), (11008, 11008, 1)); del buf309  # reuse
        # Source Nodes: [mul_175, silu_10], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf311, buf310, 11008, grid=grid(11008), stream=stream0)
        buf312 = reinterpret_tensor(buf308, (1, 4096), (4096, 1)); del buf308  # reuse
        # Source Nodes: [l__model___layers_10_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf311, (1, 11008), (0, 1), 0), reinterpret_tensor(arg120_1, (11008, 4096), (1, 11008), 0), out=buf312)
        del arg120_1
        buf314 = reinterpret_tensor(buf249, (1, 1, 4096), (4096, 4096, 1)); del buf249  # reuse
        # Source Nodes: [add_59, add_63, add_65, add_66, float_45, mean_22, mul_176, mul_177, mul_178, rsqrt_22, type_as_44], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_13.run(buf278, buf284, buf306, buf312, arg22_1, buf314, 1, 4096, grid=grid(1), stream=stream0)
        del arg22_1
        buf315 = buf287; del buf287  # reuse
        # Source Nodes: [l__model___layers_11_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf314, (1, 4096), (0, 1), 0), reinterpret_tensor(arg121_1, (4096, 12288), (1, 4096), 0), out=buf315)
        del arg121_1
        buf318 = buf290; del buf290  # reuse
        buf316 = reinterpret_tensor(buf318, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf317 = reinterpret_tensor(buf318, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf321 = buf293; del buf293  # reuse
        buf319 = reinterpret_tensor(buf321, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf320 = reinterpret_tensor(buf321, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_22, stack_23], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf315, arg294_1, arg227_1, buf316, buf317, buf319, buf320, 2048, grid=grid(2048), stream=stream0)
        buf322 = buf304; del buf304  # reuse
        # Source Nodes: [setitem_22], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg251_1, buf322, 4718592, grid=grid(4718592), stream=stream0)
        del buf316
        del buf317
        del buf319
        del buf320
        buf329 = buf301; del buf301  # reuse
        # Source Nodes: [setitem_23], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg252_1, buf329, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_22, setitem_23], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf321, buf315, buf322, buf329, 4096, grid=grid(4096), stream=stream0)
        buf324 = reinterpret_tensor(buf314, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf314  # reuse
        # Source Nodes: [type_as_45], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf318, buf324, 4096, grid=grid(4096), stream=stream0)
        buf325 = buf294; del buf294  # reuse
        # Source Nodes: [setitem_22], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf322, buf325, arg251_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg251_1
        buf326 = reinterpret_tensor(buf303, (32, 1, 1152), (1152, 1152, 1)); del buf303  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf324, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf325, (32, 128, 1152), (147456, 1, 128), 0), out=buf326)
        buf331 = reinterpret_tensor(buf298, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf298  # reuse
        # Source Nodes: [getitem, mul_187, softmax_11, where_11], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf326, buf331, 32, 1152, grid=grid(32), stream=stream0)
        buf332 = buf325; del buf325  # reuse
        # Source Nodes: [setitem_23], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf329, buf332, arg252_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg252_1
        buf333 = reinterpret_tensor(buf324, (32, 1, 128), (128, 128, 1)); del buf324  # reuse
        # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf331, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf332, (32, 1152, 128), (147456, 128, 1), 0), out=buf333)
        buf334 = buf227; del buf227  # reuse
        # Source Nodes: [l__model___layers_11_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf333, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg122_1, (4096, 4096), (1, 4096), 0), out=buf334)
        del arg122_1
        buf335 = buf278; del buf278  # reuse
        buf337 = reinterpret_tensor(buf333, (1, 1, 4096), (4096, 4096, 1)); del buf333  # reuse
        # Source Nodes: [add_59, add_63, add_65, add_69, add_70, float_48, mean_23, mul_188, mul_189, mul_190, rsqrt_23, type_as_47], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(buf335, buf284, buf306, buf312, buf334, arg23_1, buf337, 1, 4096, grid=grid(1), stream=stream0)
        del arg23_1
        buf338 = reinterpret_tensor(buf311, (1, 11008), (11008, 1)); del buf311  # reuse
        # Source Nodes: [l__model___layers_11_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (1, 4096), (0, 1), 0), reinterpret_tensor(arg123_1, (4096, 11008), (1, 4096), 0), out=buf338)
        del arg123_1
        buf339 = buf310; del buf310  # reuse
        # Source Nodes: [l__model___layers_11_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg124_1, (4096, 11008), (1, 4096), 0), out=buf339)
        del arg124_1
        buf340 = reinterpret_tensor(buf338, (1, 1, 11008), (11008, 11008, 1)); del buf338  # reuse
        # Source Nodes: [mul_191, silu_11], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf340, buf339, 11008, grid=grid(11008), stream=stream0)
        buf341 = reinterpret_tensor(buf337, (1, 4096), (4096, 1)); del buf337  # reuse
        # Source Nodes: [l__model___layers_11_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf340, (1, 11008), (0, 1), 0), reinterpret_tensor(arg125_1, (11008, 4096), (1, 11008), 0), out=buf341)
        del arg125_1
        buf343 = reinterpret_tensor(buf334, (1, 1, 4096), (4096, 4096, 1)); del buf334  # reuse
        # Source Nodes: [add_71, add_72, float_49, mean_24, mul_192, mul_193, mul_194, rsqrt_24, type_as_48], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf335, buf341, arg24_1, buf343, 1, 4096, grid=grid(1), stream=stream0)
        del arg24_1
        buf344 = buf315; del buf315  # reuse
        # Source Nodes: [l__model___layers_12_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf343, (1, 4096), (0, 1), 0), reinterpret_tensor(arg126_1, (4096, 12288), (1, 4096), 0), out=buf344)
        del arg126_1
        buf347 = buf318; del buf318  # reuse
        buf345 = reinterpret_tensor(buf347, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf346 = reinterpret_tensor(buf347, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf350 = buf321; del buf321  # reuse
        buf348 = reinterpret_tensor(buf350, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf349 = reinterpret_tensor(buf350, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_24, stack_25], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf344, arg294_1, arg227_1, buf345, buf346, buf348, buf349, 2048, grid=grid(2048), stream=stream0)
        buf351 = buf332; del buf332  # reuse
        # Source Nodes: [setitem_24], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg253_1, buf351, 4718592, grid=grid(4718592), stream=stream0)
        del buf345
        del buf346
        del buf348
        del buf349
        buf358 = buf329; del buf329  # reuse
        # Source Nodes: [setitem_25], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg254_1, buf358, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_24, setitem_25], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf350, buf344, buf351, buf358, 4096, grid=grid(4096), stream=stream0)
        buf353 = reinterpret_tensor(buf343, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf343  # reuse
        # Source Nodes: [type_as_49], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf347, buf353, 4096, grid=grid(4096), stream=stream0)
        buf354 = buf322; del buf322  # reuse
        # Source Nodes: [setitem_24], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf351, buf354, arg253_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg253_1
        buf355 = reinterpret_tensor(buf331, (32, 1, 1152), (1152, 1152, 1)); del buf331  # reuse
        # Source Nodes: [matmul_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf353, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf354, (32, 128, 1152), (147456, 1, 128), 0), out=buf355)
        buf360 = reinterpret_tensor(buf326, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf326  # reuse
        # Source Nodes: [getitem, mul_203, softmax_12, where_12], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf355, buf360, 32, 1152, grid=grid(32), stream=stream0)
        buf361 = buf354; del buf354  # reuse
        # Source Nodes: [setitem_25], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf358, buf361, arg254_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg254_1
        buf362 = reinterpret_tensor(buf353, (32, 1, 128), (128, 128, 1)); del buf353  # reuse
        # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf360, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf361, (32, 1152, 128), (147456, 128, 1), 0), out=buf362)
        buf363 = buf312; del buf312  # reuse
        # Source Nodes: [l__model___layers_12_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf362, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg127_1, (4096, 4096), (1, 4096), 0), out=buf363)
        del arg127_1
        buf365 = reinterpret_tensor(buf362, (1, 1, 4096), (4096, 4096, 1)); del buf362  # reuse
        # Source Nodes: [add_71, add_75, add_76, float_52, mean_25, mul_204, mul_205, mul_206, rsqrt_25, type_as_51], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_12.run(buf335, buf341, buf363, arg25_1, buf365, 1, 4096, grid=grid(1), stream=stream0)
        del arg25_1
        buf366 = reinterpret_tensor(buf340, (1, 11008), (11008, 1)); del buf340  # reuse
        # Source Nodes: [l__model___layers_12_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf365, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg128_1, (4096, 11008), (1, 4096), 0), out=buf366)
        del arg128_1
        buf367 = buf339; del buf339  # reuse
        # Source Nodes: [l__model___layers_12_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf365, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg129_1, (4096, 11008), (1, 4096), 0), out=buf367)
        del arg129_1
        buf368 = reinterpret_tensor(buf366, (1, 1, 11008), (11008, 11008, 1)); del buf366  # reuse
        # Source Nodes: [mul_207, silu_12], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf368, buf367, 11008, grid=grid(11008), stream=stream0)
        buf369 = reinterpret_tensor(buf365, (1, 4096), (4096, 1)); del buf365  # reuse
        # Source Nodes: [l__model___layers_12_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf368, (1, 11008), (0, 1), 0), reinterpret_tensor(arg130_1, (11008, 4096), (1, 11008), 0), out=buf369)
        del arg130_1
        buf371 = reinterpret_tensor(buf306, (1, 1, 4096), (4096, 4096, 1)); del buf306  # reuse
        # Source Nodes: [add_71, add_75, add_77, add_78, float_53, mean_26, mul_208, mul_209, mul_210, rsqrt_26, type_as_52], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_13.run(buf335, buf341, buf363, buf369, arg26_1, buf371, 1, 4096, grid=grid(1), stream=stream0)
        del arg26_1
        buf372 = buf344; del buf344  # reuse
        # Source Nodes: [l__model___layers_13_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf371, (1, 4096), (0, 1), 0), reinterpret_tensor(arg131_1, (4096, 12288), (1, 4096), 0), out=buf372)
        del arg131_1
        buf375 = buf347; del buf347  # reuse
        buf373 = reinterpret_tensor(buf375, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf374 = reinterpret_tensor(buf375, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf378 = buf350; del buf350  # reuse
        buf376 = reinterpret_tensor(buf378, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf377 = reinterpret_tensor(buf378, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_26, stack_27], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf372, arg294_1, arg227_1, buf373, buf374, buf376, buf377, 2048, grid=grid(2048), stream=stream0)
        buf379 = buf361; del buf361  # reuse
        # Source Nodes: [setitem_26], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg255_1, buf379, 4718592, grid=grid(4718592), stream=stream0)
        del buf373
        del buf374
        del buf376
        del buf377
        buf386 = buf358; del buf358  # reuse
        # Source Nodes: [setitem_27], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg256_1, buf386, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_26, setitem_27], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf378, buf372, buf379, buf386, 4096, grid=grid(4096), stream=stream0)
        buf381 = reinterpret_tensor(buf371, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf371  # reuse
        # Source Nodes: [type_as_53], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf375, buf381, 4096, grid=grid(4096), stream=stream0)
        buf382 = buf351; del buf351  # reuse
        # Source Nodes: [setitem_26], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf379, buf382, arg255_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg255_1
        buf383 = reinterpret_tensor(buf360, (32, 1, 1152), (1152, 1152, 1)); del buf360  # reuse
        # Source Nodes: [matmul_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf381, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf382, (32, 128, 1152), (147456, 1, 128), 0), out=buf383)
        buf388 = reinterpret_tensor(buf355, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf355  # reuse
        # Source Nodes: [getitem, mul_219, softmax_13, where_13], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf383, buf388, 32, 1152, grid=grid(32), stream=stream0)
        buf389 = buf382; del buf382  # reuse
        # Source Nodes: [setitem_27], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf386, buf389, arg256_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg256_1
        buf390 = reinterpret_tensor(buf381, (32, 1, 128), (128, 128, 1)); del buf381  # reuse
        # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf388, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf389, (32, 1152, 128), (147456, 128, 1), 0), out=buf390)
        buf391 = buf284; del buf284  # reuse
        # Source Nodes: [l__model___layers_13_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf390, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg132_1, (4096, 4096), (1, 4096), 0), out=buf391)
        del arg132_1
        buf392 = buf335; del buf335  # reuse
        buf394 = reinterpret_tensor(buf390, (1, 1, 4096), (4096, 4096, 1)); del buf390  # reuse
        # Source Nodes: [add_71, add_75, add_77, add_81, add_82, float_56, mean_27, mul_220, mul_221, mul_222, rsqrt_27, type_as_55], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(buf392, buf341, buf363, buf369, buf391, arg27_1, buf394, 1, 4096, grid=grid(1), stream=stream0)
        del arg27_1
        buf395 = reinterpret_tensor(buf368, (1, 11008), (11008, 1)); del buf368  # reuse
        # Source Nodes: [l__model___layers_13_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf394, (1, 4096), (0, 1), 0), reinterpret_tensor(arg133_1, (4096, 11008), (1, 4096), 0), out=buf395)
        del arg133_1
        buf396 = buf367; del buf367  # reuse
        # Source Nodes: [l__model___layers_13_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf394, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg134_1, (4096, 11008), (1, 4096), 0), out=buf396)
        del arg134_1
        buf397 = reinterpret_tensor(buf395, (1, 1, 11008), (11008, 11008, 1)); del buf395  # reuse
        # Source Nodes: [mul_223, silu_13], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf397, buf396, 11008, grid=grid(11008), stream=stream0)
        buf398 = reinterpret_tensor(buf394, (1, 4096), (4096, 1)); del buf394  # reuse
        # Source Nodes: [l__model___layers_13_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf397, (1, 11008), (0, 1), 0), reinterpret_tensor(arg135_1, (11008, 4096), (1, 11008), 0), out=buf398)
        del arg135_1
        buf400 = reinterpret_tensor(buf391, (1, 1, 4096), (4096, 4096, 1)); del buf391  # reuse
        # Source Nodes: [add_83, add_84, float_57, mean_28, mul_224, mul_225, mul_226, rsqrt_28, type_as_56], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf392, buf398, arg28_1, buf400, 1, 4096, grid=grid(1), stream=stream0)
        del arg28_1
        buf401 = buf372; del buf372  # reuse
        # Source Nodes: [l__model___layers_14_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf400, (1, 4096), (0, 1), 0), reinterpret_tensor(arg136_1, (4096, 12288), (1, 4096), 0), out=buf401)
        del arg136_1
        buf404 = buf375; del buf375  # reuse
        buf402 = reinterpret_tensor(buf404, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf403 = reinterpret_tensor(buf404, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf407 = buf378; del buf378  # reuse
        buf405 = reinterpret_tensor(buf407, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf406 = reinterpret_tensor(buf407, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_28, stack_29], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf401, arg294_1, arg227_1, buf402, buf403, buf405, buf406, 2048, grid=grid(2048), stream=stream0)
        buf408 = buf389; del buf389  # reuse
        # Source Nodes: [setitem_28], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg257_1, buf408, 4718592, grid=grid(4718592), stream=stream0)
        del buf402
        del buf403
        del buf405
        del buf406
        buf415 = buf386; del buf386  # reuse
        # Source Nodes: [setitem_29], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg258_1, buf415, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_28, setitem_29], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf407, buf401, buf408, buf415, 4096, grid=grid(4096), stream=stream0)
        buf410 = reinterpret_tensor(buf400, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf400  # reuse
        # Source Nodes: [type_as_57], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf404, buf410, 4096, grid=grid(4096), stream=stream0)
        buf411 = buf379; del buf379  # reuse
        # Source Nodes: [setitem_28], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf408, buf411, arg257_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg257_1
        buf412 = reinterpret_tensor(buf388, (32, 1, 1152), (1152, 1152, 1)); del buf388  # reuse
        # Source Nodes: [matmul_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf410, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf411, (32, 128, 1152), (147456, 1, 128), 0), out=buf412)
        buf417 = reinterpret_tensor(buf383, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf383  # reuse
        # Source Nodes: [getitem, mul_235, softmax_14, where_14], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf412, buf417, 32, 1152, grid=grid(32), stream=stream0)
        buf418 = buf411; del buf411  # reuse
        # Source Nodes: [setitem_29], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf415, buf418, arg258_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg258_1
        buf419 = reinterpret_tensor(buf410, (32, 1, 128), (128, 128, 1)); del buf410  # reuse
        # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf417, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf418, (32, 1152, 128), (147456, 128, 1), 0), out=buf419)
        buf420 = buf369; del buf369  # reuse
        # Source Nodes: [l__model___layers_14_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf419, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg137_1, (4096, 4096), (1, 4096), 0), out=buf420)
        del arg137_1
        buf422 = reinterpret_tensor(buf419, (1, 1, 4096), (4096, 4096, 1)); del buf419  # reuse
        # Source Nodes: [add_83, add_87, add_88, float_60, mean_29, mul_236, mul_237, mul_238, rsqrt_29, type_as_59], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_12.run(buf392, buf398, buf420, arg29_1, buf422, 1, 4096, grid=grid(1), stream=stream0)
        del arg29_1
        buf423 = reinterpret_tensor(buf397, (1, 11008), (11008, 1)); del buf397  # reuse
        # Source Nodes: [l__model___layers_14_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf422, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg138_1, (4096, 11008), (1, 4096), 0), out=buf423)
        del arg138_1
        buf424 = buf396; del buf396  # reuse
        # Source Nodes: [l__model___layers_14_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf422, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg139_1, (4096, 11008), (1, 4096), 0), out=buf424)
        del arg139_1
        buf425 = reinterpret_tensor(buf423, (1, 1, 11008), (11008, 11008, 1)); del buf423  # reuse
        # Source Nodes: [mul_239, silu_14], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf425, buf424, 11008, grid=grid(11008), stream=stream0)
        buf426 = reinterpret_tensor(buf422, (1, 4096), (4096, 1)); del buf422  # reuse
        # Source Nodes: [l__model___layers_14_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf425, (1, 11008), (0, 1), 0), reinterpret_tensor(arg140_1, (11008, 4096), (1, 11008), 0), out=buf426)
        del arg140_1
        buf428 = reinterpret_tensor(buf363, (1, 1, 4096), (4096, 4096, 1)); del buf363  # reuse
        # Source Nodes: [add_83, add_87, add_89, add_90, float_61, mean_30, mul_240, mul_241, mul_242, rsqrt_30, type_as_60], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_13.run(buf392, buf398, buf420, buf426, arg30_1, buf428, 1, 4096, grid=grid(1), stream=stream0)
        del arg30_1
        buf429 = buf401; del buf401  # reuse
        # Source Nodes: [l__model___layers_15_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (1, 4096), (0, 1), 0), reinterpret_tensor(arg141_1, (4096, 12288), (1, 4096), 0), out=buf429)
        del arg141_1
        buf432 = buf404; del buf404  # reuse
        buf430 = reinterpret_tensor(buf432, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf431 = reinterpret_tensor(buf432, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf435 = buf407; del buf407  # reuse
        buf433 = reinterpret_tensor(buf435, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf434 = reinterpret_tensor(buf435, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_30, stack_31], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf429, arg294_1, arg227_1, buf430, buf431, buf433, buf434, 2048, grid=grid(2048), stream=stream0)
        buf436 = buf418; del buf418  # reuse
        # Source Nodes: [setitem_30], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg259_1, buf436, 4718592, grid=grid(4718592), stream=stream0)
        del buf430
        del buf431
        del buf433
        del buf434
        buf443 = buf415; del buf415  # reuse
        # Source Nodes: [setitem_31], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg260_1, buf443, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_30, setitem_31], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf435, buf429, buf436, buf443, 4096, grid=grid(4096), stream=stream0)
        buf438 = reinterpret_tensor(buf428, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf428  # reuse
        # Source Nodes: [type_as_61], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf432, buf438, 4096, grid=grid(4096), stream=stream0)
        buf439 = buf408; del buf408  # reuse
        # Source Nodes: [setitem_30], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf436, buf439, arg259_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg259_1
        buf440 = reinterpret_tensor(buf417, (32, 1, 1152), (1152, 1152, 1)); del buf417  # reuse
        # Source Nodes: [matmul_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf438, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf439, (32, 128, 1152), (147456, 1, 128), 0), out=buf440)
        buf445 = reinterpret_tensor(buf412, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf412  # reuse
        # Source Nodes: [getitem, mul_251, softmax_15, where_15], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf440, buf445, 32, 1152, grid=grid(32), stream=stream0)
        buf446 = buf439; del buf439  # reuse
        # Source Nodes: [setitem_31], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf443, buf446, arg260_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg260_1
        buf447 = reinterpret_tensor(buf438, (32, 1, 128), (128, 128, 1)); del buf438  # reuse
        # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf445, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf446, (32, 1152, 128), (147456, 128, 1), 0), out=buf447)
        buf448 = buf341; del buf341  # reuse
        # Source Nodes: [l__model___layers_15_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg142_1, (4096, 4096), (1, 4096), 0), out=buf448)
        del arg142_1
        buf449 = buf392; del buf392  # reuse
        buf451 = reinterpret_tensor(buf447, (1, 1, 4096), (4096, 4096, 1)); del buf447  # reuse
        # Source Nodes: [add_83, add_87, add_89, add_93, add_94, float_64, mean_31, mul_252, mul_253, mul_254, rsqrt_31, type_as_63], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(buf449, buf398, buf420, buf426, buf448, arg31_1, buf451, 1, 4096, grid=grid(1), stream=stream0)
        del arg31_1
        buf452 = reinterpret_tensor(buf425, (1, 11008), (11008, 1)); del buf425  # reuse
        # Source Nodes: [l__model___layers_15_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf451, (1, 4096), (0, 1), 0), reinterpret_tensor(arg143_1, (4096, 11008), (1, 4096), 0), out=buf452)
        del arg143_1
        buf453 = buf424; del buf424  # reuse
        # Source Nodes: [l__model___layers_15_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf451, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg144_1, (4096, 11008), (1, 4096), 0), out=buf453)
        del arg144_1
        buf454 = reinterpret_tensor(buf452, (1, 1, 11008), (11008, 11008, 1)); del buf452  # reuse
        # Source Nodes: [mul_255, silu_15], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf454, buf453, 11008, grid=grid(11008), stream=stream0)
        buf455 = reinterpret_tensor(buf451, (1, 4096), (4096, 1)); del buf451  # reuse
        # Source Nodes: [l__model___layers_15_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf454, (1, 11008), (0, 1), 0), reinterpret_tensor(arg145_1, (11008, 4096), (1, 11008), 0), out=buf455)
        del arg145_1
        buf457 = reinterpret_tensor(buf448, (1, 1, 4096), (4096, 4096, 1)); del buf448  # reuse
        # Source Nodes: [add_95, add_96, float_65, mean_32, mul_256, mul_257, mul_258, rsqrt_32, type_as_64], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf449, buf455, arg32_1, buf457, 1, 4096, grid=grid(1), stream=stream0)
        del arg32_1
        buf458 = buf429; del buf429  # reuse
        # Source Nodes: [l__model___layers_16_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf457, (1, 4096), (0, 1), 0), reinterpret_tensor(arg146_1, (4096, 12288), (1, 4096), 0), out=buf458)
        del arg146_1
        buf461 = buf432; del buf432  # reuse
        buf459 = reinterpret_tensor(buf461, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf460 = reinterpret_tensor(buf461, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf464 = buf435; del buf435  # reuse
        buf462 = reinterpret_tensor(buf464, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf463 = reinterpret_tensor(buf464, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_32, stack_33], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf458, arg294_1, arg227_1, buf459, buf460, buf462, buf463, 2048, grid=grid(2048), stream=stream0)
        buf465 = buf446; del buf446  # reuse
        # Source Nodes: [setitem_32], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg261_1, buf465, 4718592, grid=grid(4718592), stream=stream0)
        del buf459
        del buf460
        del buf462
        del buf463
        buf472 = buf443; del buf443  # reuse
        # Source Nodes: [setitem_33], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg262_1, buf472, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_32, setitem_33], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf464, buf458, buf465, buf472, 4096, grid=grid(4096), stream=stream0)
        buf467 = reinterpret_tensor(buf457, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf457  # reuse
        # Source Nodes: [type_as_65], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf461, buf467, 4096, grid=grid(4096), stream=stream0)
        buf468 = buf436; del buf436  # reuse
        # Source Nodes: [setitem_32], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf465, buf468, arg261_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg261_1
        buf469 = reinterpret_tensor(buf445, (32, 1, 1152), (1152, 1152, 1)); del buf445  # reuse
        # Source Nodes: [matmul_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf467, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf468, (32, 128, 1152), (147456, 1, 128), 0), out=buf469)
        buf474 = reinterpret_tensor(buf440, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf440  # reuse
        # Source Nodes: [getitem, mul_267, softmax_16, where_16], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf469, buf474, 32, 1152, grid=grid(32), stream=stream0)
        buf475 = buf468; del buf468  # reuse
        # Source Nodes: [setitem_33], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf472, buf475, arg262_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg262_1
        buf476 = reinterpret_tensor(buf467, (32, 1, 128), (128, 128, 1)); del buf467  # reuse
        # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf474, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf475, (32, 1152, 128), (147456, 128, 1), 0), out=buf476)
        buf477 = buf426; del buf426  # reuse
        # Source Nodes: [l__model___layers_16_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf476, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg147_1, (4096, 4096), (1, 4096), 0), out=buf477)
        del arg147_1
        buf479 = reinterpret_tensor(buf476, (1, 1, 4096), (4096, 4096, 1)); del buf476  # reuse
        # Source Nodes: [add_100, add_95, add_99, float_68, mean_33, mul_268, mul_269, mul_270, rsqrt_33, type_as_67], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_12.run(buf449, buf455, buf477, arg33_1, buf479, 1, 4096, grid=grid(1), stream=stream0)
        del arg33_1
        buf480 = reinterpret_tensor(buf454, (1, 11008), (11008, 1)); del buf454  # reuse
        # Source Nodes: [l__model___layers_16_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf479, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg148_1, (4096, 11008), (1, 4096), 0), out=buf480)
        del arg148_1
        buf481 = buf453; del buf453  # reuse
        # Source Nodes: [l__model___layers_16_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf479, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg149_1, (4096, 11008), (1, 4096), 0), out=buf481)
        del arg149_1
        buf482 = reinterpret_tensor(buf480, (1, 1, 11008), (11008, 11008, 1)); del buf480  # reuse
        # Source Nodes: [mul_271, silu_16], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf482, buf481, 11008, grid=grid(11008), stream=stream0)
        buf483 = reinterpret_tensor(buf479, (1, 4096), (4096, 1)); del buf479  # reuse
        # Source Nodes: [l__model___layers_16_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf482, (1, 11008), (0, 1), 0), reinterpret_tensor(arg150_1, (11008, 4096), (1, 11008), 0), out=buf483)
        del arg150_1
        buf485 = reinterpret_tensor(buf420, (1, 1, 4096), (4096, 4096, 1)); del buf420  # reuse
        # Source Nodes: [add_101, add_102, add_95, add_99, float_69, mean_34, mul_272, mul_273, mul_274, rsqrt_34, type_as_68], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_13.run(buf449, buf455, buf477, buf483, arg34_1, buf485, 1, 4096, grid=grid(1), stream=stream0)
        del arg34_1
        buf486 = buf458; del buf458  # reuse
        # Source Nodes: [l__model___layers_17_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf485, (1, 4096), (0, 1), 0), reinterpret_tensor(arg151_1, (4096, 12288), (1, 4096), 0), out=buf486)
        del arg151_1
        buf489 = buf461; del buf461  # reuse
        buf487 = reinterpret_tensor(buf489, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf488 = reinterpret_tensor(buf489, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf492 = buf464; del buf464  # reuse
        buf490 = reinterpret_tensor(buf492, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf491 = reinterpret_tensor(buf492, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_34, stack_35], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf486, arg294_1, arg227_1, buf487, buf488, buf490, buf491, 2048, grid=grid(2048), stream=stream0)
        buf493 = buf475; del buf475  # reuse
        # Source Nodes: [setitem_34], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg263_1, buf493, 4718592, grid=grid(4718592), stream=stream0)
        del buf487
        del buf488
        del buf490
        del buf491
        buf500 = buf472; del buf472  # reuse
        # Source Nodes: [setitem_35], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg264_1, buf500, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_34, setitem_35], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf492, buf486, buf493, buf500, 4096, grid=grid(4096), stream=stream0)
        buf495 = reinterpret_tensor(buf485, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf485  # reuse
        # Source Nodes: [type_as_69], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf489, buf495, 4096, grid=grid(4096), stream=stream0)
        buf496 = buf465; del buf465  # reuse
        # Source Nodes: [setitem_34], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf493, buf496, arg263_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg263_1
        buf497 = reinterpret_tensor(buf474, (32, 1, 1152), (1152, 1152, 1)); del buf474  # reuse
        # Source Nodes: [matmul_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf495, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf496, (32, 128, 1152), (147456, 1, 128), 0), out=buf497)
        buf502 = reinterpret_tensor(buf469, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf469  # reuse
        # Source Nodes: [getitem, mul_283, softmax_17, where_17], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf497, buf502, 32, 1152, grid=grid(32), stream=stream0)
        buf503 = buf496; del buf496  # reuse
        # Source Nodes: [setitem_35], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf500, buf503, arg264_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg264_1
        buf504 = reinterpret_tensor(buf495, (32, 1, 128), (128, 128, 1)); del buf495  # reuse
        # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf502, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf503, (32, 1152, 128), (147456, 128, 1), 0), out=buf504)
        buf505 = buf398; del buf398  # reuse
        # Source Nodes: [l__model___layers_17_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf504, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg152_1, (4096, 4096), (1, 4096), 0), out=buf505)
        del arg152_1
        buf506 = buf449; del buf449  # reuse
        buf508 = reinterpret_tensor(buf504, (1, 1, 4096), (4096, 4096, 1)); del buf504  # reuse
        # Source Nodes: [add_101, add_105, add_106, add_95, add_99, float_72, mean_35, mul_284, mul_285, mul_286, rsqrt_35, type_as_71], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(buf506, buf455, buf477, buf483, buf505, arg35_1, buf508, 1, 4096, grid=grid(1), stream=stream0)
        del arg35_1
        buf509 = reinterpret_tensor(buf482, (1, 11008), (11008, 1)); del buf482  # reuse
        # Source Nodes: [l__model___layers_17_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf508, (1, 4096), (0, 1), 0), reinterpret_tensor(arg153_1, (4096, 11008), (1, 4096), 0), out=buf509)
        del arg153_1
        buf510 = buf481; del buf481  # reuse
        # Source Nodes: [l__model___layers_17_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf508, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg154_1, (4096, 11008), (1, 4096), 0), out=buf510)
        del arg154_1
        buf511 = reinterpret_tensor(buf509, (1, 1, 11008), (11008, 11008, 1)); del buf509  # reuse
        # Source Nodes: [mul_287, silu_17], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf511, buf510, 11008, grid=grid(11008), stream=stream0)
        buf512 = reinterpret_tensor(buf508, (1, 4096), (4096, 1)); del buf508  # reuse
        # Source Nodes: [l__model___layers_17_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf511, (1, 11008), (0, 1), 0), reinterpret_tensor(arg155_1, (11008, 4096), (1, 11008), 0), out=buf512)
        del arg155_1
        buf514 = reinterpret_tensor(buf505, (1, 1, 4096), (4096, 4096, 1)); del buf505  # reuse
        # Source Nodes: [add_107, add_108, float_73, mean_36, mul_288, mul_289, mul_290, rsqrt_36, type_as_72], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf506, buf512, arg36_1, buf514, 1, 4096, grid=grid(1), stream=stream0)
        del arg36_1
        buf515 = buf486; del buf486  # reuse
        # Source Nodes: [l__model___layers_18_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf514, (1, 4096), (0, 1), 0), reinterpret_tensor(arg156_1, (4096, 12288), (1, 4096), 0), out=buf515)
        del arg156_1
        buf518 = buf489; del buf489  # reuse
        buf516 = reinterpret_tensor(buf518, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf517 = reinterpret_tensor(buf518, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf521 = buf492; del buf492  # reuse
        buf519 = reinterpret_tensor(buf521, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf520 = reinterpret_tensor(buf521, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_36, stack_37], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf515, arg294_1, arg227_1, buf516, buf517, buf519, buf520, 2048, grid=grid(2048), stream=stream0)
        buf522 = buf503; del buf503  # reuse
        # Source Nodes: [setitem_36], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg265_1, buf522, 4718592, grid=grid(4718592), stream=stream0)
        del buf516
        del buf517
        del buf519
        del buf520
        buf529 = buf500; del buf500  # reuse
        # Source Nodes: [setitem_37], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg266_1, buf529, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_36, setitem_37], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf521, buf515, buf522, buf529, 4096, grid=grid(4096), stream=stream0)
        buf524 = reinterpret_tensor(buf514, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf514  # reuse
        # Source Nodes: [type_as_73], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf518, buf524, 4096, grid=grid(4096), stream=stream0)
        buf525 = buf493; del buf493  # reuse
        # Source Nodes: [setitem_36], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf522, buf525, arg265_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg265_1
        buf526 = reinterpret_tensor(buf502, (32, 1, 1152), (1152, 1152, 1)); del buf502  # reuse
        # Source Nodes: [matmul_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf524, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf525, (32, 128, 1152), (147456, 1, 128), 0), out=buf526)
        buf531 = reinterpret_tensor(buf497, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf497  # reuse
        # Source Nodes: [getitem, mul_299, softmax_18, where_18], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf526, buf531, 32, 1152, grid=grid(32), stream=stream0)
        buf532 = buf525; del buf525  # reuse
        # Source Nodes: [setitem_37], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf529, buf532, arg266_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg266_1
        buf533 = reinterpret_tensor(buf524, (32, 1, 128), (128, 128, 1)); del buf524  # reuse
        # Source Nodes: [matmul_37], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf531, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf532, (32, 1152, 128), (147456, 128, 1), 0), out=buf533)
        buf534 = buf483; del buf483  # reuse
        # Source Nodes: [l__model___layers_18_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf533, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg157_1, (4096, 4096), (1, 4096), 0), out=buf534)
        del arg157_1
        buf536 = reinterpret_tensor(buf533, (1, 1, 4096), (4096, 4096, 1)); del buf533  # reuse
        # Source Nodes: [add_107, add_111, add_112, float_76, mean_37, mul_300, mul_301, mul_302, rsqrt_37, type_as_75], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_12.run(buf506, buf512, buf534, arg37_1, buf536, 1, 4096, grid=grid(1), stream=stream0)
        del arg37_1
        buf537 = reinterpret_tensor(buf511, (1, 11008), (11008, 1)); del buf511  # reuse
        # Source Nodes: [l__model___layers_18_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf536, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg158_1, (4096, 11008), (1, 4096), 0), out=buf537)
        del arg158_1
        buf538 = buf510; del buf510  # reuse
        # Source Nodes: [l__model___layers_18_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf536, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg159_1, (4096, 11008), (1, 4096), 0), out=buf538)
        del arg159_1
        buf539 = reinterpret_tensor(buf537, (1, 1, 11008), (11008, 11008, 1)); del buf537  # reuse
        # Source Nodes: [mul_303, silu_18], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf539, buf538, 11008, grid=grid(11008), stream=stream0)
        buf540 = reinterpret_tensor(buf536, (1, 4096), (4096, 1)); del buf536  # reuse
        # Source Nodes: [l__model___layers_18_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf539, (1, 11008), (0, 1), 0), reinterpret_tensor(arg160_1, (11008, 4096), (1, 11008), 0), out=buf540)
        del arg160_1
        buf542 = reinterpret_tensor(buf477, (1, 1, 4096), (4096, 4096, 1)); del buf477  # reuse
        # Source Nodes: [add_107, add_111, add_113, add_114, float_77, mean_38, mul_304, mul_305, mul_306, rsqrt_38, type_as_76], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_13.run(buf506, buf512, buf534, buf540, arg38_1, buf542, 1, 4096, grid=grid(1), stream=stream0)
        del arg38_1
        buf543 = buf515; del buf515  # reuse
        # Source Nodes: [l__model___layers_19_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf542, (1, 4096), (0, 1), 0), reinterpret_tensor(arg161_1, (4096, 12288), (1, 4096), 0), out=buf543)
        del arg161_1
        buf546 = buf518; del buf518  # reuse
        buf544 = reinterpret_tensor(buf546, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf545 = reinterpret_tensor(buf546, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf549 = buf521; del buf521  # reuse
        buf547 = reinterpret_tensor(buf549, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf548 = reinterpret_tensor(buf549, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_38, stack_39], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf543, arg294_1, arg227_1, buf544, buf545, buf547, buf548, 2048, grid=grid(2048), stream=stream0)
        buf550 = buf532; del buf532  # reuse
        # Source Nodes: [setitem_38], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg267_1, buf550, 4718592, grid=grid(4718592), stream=stream0)
        del buf544
        del buf545
        del buf547
        del buf548
        buf557 = buf529; del buf529  # reuse
        # Source Nodes: [setitem_39], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg268_1, buf557, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_38, setitem_39], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf549, buf543, buf550, buf557, 4096, grid=grid(4096), stream=stream0)
        buf552 = reinterpret_tensor(buf542, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf542  # reuse
        # Source Nodes: [type_as_77], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf546, buf552, 4096, grid=grid(4096), stream=stream0)
        buf553 = buf522; del buf522  # reuse
        # Source Nodes: [setitem_38], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf550, buf553, arg267_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg267_1
        buf554 = reinterpret_tensor(buf531, (32, 1, 1152), (1152, 1152, 1)); del buf531  # reuse
        # Source Nodes: [matmul_38], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf552, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf553, (32, 128, 1152), (147456, 1, 128), 0), out=buf554)
        buf559 = reinterpret_tensor(buf526, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf526  # reuse
        # Source Nodes: [getitem, mul_315, softmax_19, where_19], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf554, buf559, 32, 1152, grid=grid(32), stream=stream0)
        buf560 = buf553; del buf553  # reuse
        # Source Nodes: [setitem_39], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf557, buf560, arg268_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg268_1
        buf561 = reinterpret_tensor(buf552, (32, 1, 128), (128, 128, 1)); del buf552  # reuse
        # Source Nodes: [matmul_39], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf559, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf560, (32, 1152, 128), (147456, 128, 1), 0), out=buf561)
        buf562 = buf455; del buf455  # reuse
        # Source Nodes: [l__model___layers_19_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf561, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg162_1, (4096, 4096), (1, 4096), 0), out=buf562)
        del arg162_1
        buf563 = buf506; del buf506  # reuse
        buf565 = reinterpret_tensor(buf561, (1, 1, 4096), (4096, 4096, 1)); del buf561  # reuse
        # Source Nodes: [add_107, add_111, add_113, add_117, add_118, float_80, mean_39, mul_316, mul_317, mul_318, rsqrt_39, type_as_79], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(buf563, buf512, buf534, buf540, buf562, arg39_1, buf565, 1, 4096, grid=grid(1), stream=stream0)
        del arg39_1
        buf566 = reinterpret_tensor(buf539, (1, 11008), (11008, 1)); del buf539  # reuse
        # Source Nodes: [l__model___layers_19_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf565, (1, 4096), (0, 1), 0), reinterpret_tensor(arg163_1, (4096, 11008), (1, 4096), 0), out=buf566)
        del arg163_1
        buf567 = buf538; del buf538  # reuse
        # Source Nodes: [l__model___layers_19_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf565, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg164_1, (4096, 11008), (1, 4096), 0), out=buf567)
        del arg164_1
        buf568 = reinterpret_tensor(buf566, (1, 1, 11008), (11008, 11008, 1)); del buf566  # reuse
        # Source Nodes: [mul_319, silu_19], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf568, buf567, 11008, grid=grid(11008), stream=stream0)
        buf569 = reinterpret_tensor(buf565, (1, 4096), (4096, 1)); del buf565  # reuse
        # Source Nodes: [l__model___layers_19_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf568, (1, 11008), (0, 1), 0), reinterpret_tensor(arg165_1, (11008, 4096), (1, 11008), 0), out=buf569)
        del arg165_1
        buf571 = reinterpret_tensor(buf562, (1, 1, 4096), (4096, 4096, 1)); del buf562  # reuse
        # Source Nodes: [add_119, add_120, float_81, mean_40, mul_320, mul_321, mul_322, rsqrt_40, type_as_80], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf563, buf569, arg40_1, buf571, 1, 4096, grid=grid(1), stream=stream0)
        del arg40_1
        buf572 = buf543; del buf543  # reuse
        # Source Nodes: [l__model___layers_20_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf571, (1, 4096), (0, 1), 0), reinterpret_tensor(arg166_1, (4096, 12288), (1, 4096), 0), out=buf572)
        del arg166_1
        buf575 = buf546; del buf546  # reuse
        buf573 = reinterpret_tensor(buf575, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf574 = reinterpret_tensor(buf575, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf578 = buf549; del buf549  # reuse
        buf576 = reinterpret_tensor(buf578, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf577 = reinterpret_tensor(buf578, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_40, stack_41], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf572, arg294_1, arg227_1, buf573, buf574, buf576, buf577, 2048, grid=grid(2048), stream=stream0)
        buf579 = buf560; del buf560  # reuse
        # Source Nodes: [setitem_40], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg269_1, buf579, 4718592, grid=grid(4718592), stream=stream0)
        del buf573
        del buf574
        del buf576
        del buf577
        buf586 = buf557; del buf557  # reuse
        # Source Nodes: [setitem_41], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg270_1, buf586, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_40, setitem_41], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf578, buf572, buf579, buf586, 4096, grid=grid(4096), stream=stream0)
        buf581 = reinterpret_tensor(buf571, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf571  # reuse
        # Source Nodes: [type_as_81], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf575, buf581, 4096, grid=grid(4096), stream=stream0)
        buf582 = buf550; del buf550  # reuse
        # Source Nodes: [setitem_40], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf579, buf582, arg269_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg269_1
        buf583 = reinterpret_tensor(buf559, (32, 1, 1152), (1152, 1152, 1)); del buf559  # reuse
        # Source Nodes: [matmul_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf581, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf582, (32, 128, 1152), (147456, 1, 128), 0), out=buf583)
        buf588 = reinterpret_tensor(buf554, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf554  # reuse
        # Source Nodes: [getitem, mul_331, softmax_20, where_20], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf583, buf588, 32, 1152, grid=grid(32), stream=stream0)
        buf589 = buf582; del buf582  # reuse
        # Source Nodes: [setitem_41], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf586, buf589, arg270_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg270_1
        buf590 = reinterpret_tensor(buf581, (32, 1, 128), (128, 128, 1)); del buf581  # reuse
        # Source Nodes: [matmul_41], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf588, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf589, (32, 1152, 128), (147456, 128, 1), 0), out=buf590)
        buf591 = buf540; del buf540  # reuse
        # Source Nodes: [l__model___layers_20_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf590, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg167_1, (4096, 4096), (1, 4096), 0), out=buf591)
        del arg167_1
        buf593 = reinterpret_tensor(buf590, (1, 1, 4096), (4096, 4096, 1)); del buf590  # reuse
        # Source Nodes: [add_119, add_123, add_124, float_84, mean_41, mul_332, mul_333, mul_334, rsqrt_41, type_as_83], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_12.run(buf563, buf569, buf591, arg41_1, buf593, 1, 4096, grid=grid(1), stream=stream0)
        del arg41_1
        buf594 = reinterpret_tensor(buf568, (1, 11008), (11008, 1)); del buf568  # reuse
        # Source Nodes: [l__model___layers_20_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf593, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg168_1, (4096, 11008), (1, 4096), 0), out=buf594)
        del arg168_1
        buf595 = buf567; del buf567  # reuse
        # Source Nodes: [l__model___layers_20_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf593, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg169_1, (4096, 11008), (1, 4096), 0), out=buf595)
        del arg169_1
        buf596 = reinterpret_tensor(buf594, (1, 1, 11008), (11008, 11008, 1)); del buf594  # reuse
        # Source Nodes: [mul_335, silu_20], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf596, buf595, 11008, grid=grid(11008), stream=stream0)
        buf597 = reinterpret_tensor(buf593, (1, 4096), (4096, 1)); del buf593  # reuse
        # Source Nodes: [l__model___layers_20_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf596, (1, 11008), (0, 1), 0), reinterpret_tensor(arg170_1, (11008, 4096), (1, 11008), 0), out=buf597)
        del arg170_1
        buf599 = reinterpret_tensor(buf534, (1, 1, 4096), (4096, 4096, 1)); del buf534  # reuse
        # Source Nodes: [add_119, add_123, add_125, add_126, float_85, mean_42, mul_336, mul_337, mul_338, rsqrt_42, type_as_84], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_13.run(buf563, buf569, buf591, buf597, arg42_1, buf599, 1, 4096, grid=grid(1), stream=stream0)
        del arg42_1
        buf600 = buf572; del buf572  # reuse
        # Source Nodes: [l__model___layers_21_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf599, (1, 4096), (0, 1), 0), reinterpret_tensor(arg171_1, (4096, 12288), (1, 4096), 0), out=buf600)
        del arg171_1
        buf603 = buf575; del buf575  # reuse
        buf601 = reinterpret_tensor(buf603, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf602 = reinterpret_tensor(buf603, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf606 = buf578; del buf578  # reuse
        buf604 = reinterpret_tensor(buf606, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf605 = reinterpret_tensor(buf606, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_42, stack_43], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf600, arg294_1, arg227_1, buf601, buf602, buf604, buf605, 2048, grid=grid(2048), stream=stream0)
        buf607 = buf589; del buf589  # reuse
        # Source Nodes: [setitem_42], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg271_1, buf607, 4718592, grid=grid(4718592), stream=stream0)
        del buf601
        del buf602
        del buf604
        del buf605
        buf614 = buf586; del buf586  # reuse
        # Source Nodes: [setitem_43], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg272_1, buf614, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_42, setitem_43], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf606, buf600, buf607, buf614, 4096, grid=grid(4096), stream=stream0)
        buf609 = reinterpret_tensor(buf599, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf599  # reuse
        # Source Nodes: [type_as_85], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf603, buf609, 4096, grid=grid(4096), stream=stream0)
        buf610 = buf579; del buf579  # reuse
        # Source Nodes: [setitem_42], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf607, buf610, arg271_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg271_1
        buf611 = reinterpret_tensor(buf588, (32, 1, 1152), (1152, 1152, 1)); del buf588  # reuse
        # Source Nodes: [matmul_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf609, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf610, (32, 128, 1152), (147456, 1, 128), 0), out=buf611)
        buf616 = reinterpret_tensor(buf583, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf583  # reuse
        # Source Nodes: [getitem, mul_347, softmax_21, where_21], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf611, buf616, 32, 1152, grid=grid(32), stream=stream0)
        buf617 = buf610; del buf610  # reuse
        # Source Nodes: [setitem_43], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf614, buf617, arg272_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg272_1
        buf618 = reinterpret_tensor(buf609, (32, 1, 128), (128, 128, 1)); del buf609  # reuse
        # Source Nodes: [matmul_43], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf616, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf617, (32, 1152, 128), (147456, 128, 1), 0), out=buf618)
        buf619 = buf512; del buf512  # reuse
        # Source Nodes: [l__model___layers_21_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf618, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg172_1, (4096, 4096), (1, 4096), 0), out=buf619)
        del arg172_1
        buf620 = buf563; del buf563  # reuse
        buf622 = reinterpret_tensor(buf618, (1, 1, 4096), (4096, 4096, 1)); del buf618  # reuse
        # Source Nodes: [add_119, add_123, add_125, add_129, add_130, float_88, mean_43, mul_348, mul_349, mul_350, rsqrt_43, type_as_87], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(buf620, buf569, buf591, buf597, buf619, arg43_1, buf622, 1, 4096, grid=grid(1), stream=stream0)
        del arg43_1
        buf623 = reinterpret_tensor(buf596, (1, 11008), (11008, 1)); del buf596  # reuse
        # Source Nodes: [l__model___layers_21_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf622, (1, 4096), (0, 1), 0), reinterpret_tensor(arg173_1, (4096, 11008), (1, 4096), 0), out=buf623)
        del arg173_1
        buf624 = buf595; del buf595  # reuse
        # Source Nodes: [l__model___layers_21_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf622, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg174_1, (4096, 11008), (1, 4096), 0), out=buf624)
        del arg174_1
        buf625 = reinterpret_tensor(buf623, (1, 1, 11008), (11008, 11008, 1)); del buf623  # reuse
        # Source Nodes: [mul_351, silu_21], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf625, buf624, 11008, grid=grid(11008), stream=stream0)
        buf626 = reinterpret_tensor(buf622, (1, 4096), (4096, 1)); del buf622  # reuse
        # Source Nodes: [l__model___layers_21_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf625, (1, 11008), (0, 1), 0), reinterpret_tensor(arg175_1, (11008, 4096), (1, 11008), 0), out=buf626)
        del arg175_1
        buf628 = reinterpret_tensor(buf619, (1, 1, 4096), (4096, 4096, 1)); del buf619  # reuse
        # Source Nodes: [add_131, add_132, float_89, mean_44, mul_352, mul_353, mul_354, rsqrt_44, type_as_88], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf620, buf626, arg44_1, buf628, 1, 4096, grid=grid(1), stream=stream0)
        del arg44_1
        buf629 = buf600; del buf600  # reuse
        # Source Nodes: [l__model___layers_22_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf628, (1, 4096), (0, 1), 0), reinterpret_tensor(arg176_1, (4096, 12288), (1, 4096), 0), out=buf629)
        del arg176_1
        buf632 = buf603; del buf603  # reuse
        buf630 = reinterpret_tensor(buf632, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf631 = reinterpret_tensor(buf632, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf635 = buf606; del buf606  # reuse
        buf633 = reinterpret_tensor(buf635, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf634 = reinterpret_tensor(buf635, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_44, stack_45], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf629, arg294_1, arg227_1, buf630, buf631, buf633, buf634, 2048, grid=grid(2048), stream=stream0)
        buf636 = buf617; del buf617  # reuse
        # Source Nodes: [setitem_44], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg273_1, buf636, 4718592, grid=grid(4718592), stream=stream0)
        del buf630
        del buf631
        del buf633
        del buf634
        buf643 = buf614; del buf614  # reuse
        # Source Nodes: [setitem_45], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg274_1, buf643, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_44, setitem_45], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf635, buf629, buf636, buf643, 4096, grid=grid(4096), stream=stream0)
        buf638 = reinterpret_tensor(buf628, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf628  # reuse
        # Source Nodes: [type_as_89], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf632, buf638, 4096, grid=grid(4096), stream=stream0)
        buf639 = buf607; del buf607  # reuse
        # Source Nodes: [setitem_44], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf636, buf639, arg273_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg273_1
        buf640 = reinterpret_tensor(buf616, (32, 1, 1152), (1152, 1152, 1)); del buf616  # reuse
        # Source Nodes: [matmul_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf638, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf639, (32, 128, 1152), (147456, 1, 128), 0), out=buf640)
        buf645 = reinterpret_tensor(buf611, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf611  # reuse
        # Source Nodes: [getitem, mul_363, softmax_22, where_22], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf640, buf645, 32, 1152, grid=grid(32), stream=stream0)
        buf646 = buf639; del buf639  # reuse
        # Source Nodes: [setitem_45], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf643, buf646, arg274_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg274_1
        buf647 = reinterpret_tensor(buf638, (32, 1, 128), (128, 128, 1)); del buf638  # reuse
        # Source Nodes: [matmul_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf645, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf646, (32, 1152, 128), (147456, 128, 1), 0), out=buf647)
        buf648 = buf597; del buf597  # reuse
        # Source Nodes: [l__model___layers_22_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf647, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg177_1, (4096, 4096), (1, 4096), 0), out=buf648)
        del arg177_1
        buf650 = reinterpret_tensor(buf647, (1, 1, 4096), (4096, 4096, 1)); del buf647  # reuse
        # Source Nodes: [add_131, add_135, add_136, float_92, mean_45, mul_364, mul_365, mul_366, rsqrt_45, type_as_91], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_12.run(buf620, buf626, buf648, arg45_1, buf650, 1, 4096, grid=grid(1), stream=stream0)
        del arg45_1
        buf651 = reinterpret_tensor(buf625, (1, 11008), (11008, 1)); del buf625  # reuse
        # Source Nodes: [l__model___layers_22_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf650, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg178_1, (4096, 11008), (1, 4096), 0), out=buf651)
        del arg178_1
        buf652 = buf624; del buf624  # reuse
        # Source Nodes: [l__model___layers_22_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf650, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg179_1, (4096, 11008), (1, 4096), 0), out=buf652)
        del arg179_1
        buf653 = reinterpret_tensor(buf651, (1, 1, 11008), (11008, 11008, 1)); del buf651  # reuse
        # Source Nodes: [mul_367, silu_22], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf653, buf652, 11008, grid=grid(11008), stream=stream0)
        buf654 = reinterpret_tensor(buf650, (1, 4096), (4096, 1)); del buf650  # reuse
        # Source Nodes: [l__model___layers_22_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf653, (1, 11008), (0, 1), 0), reinterpret_tensor(arg180_1, (11008, 4096), (1, 11008), 0), out=buf654)
        del arg180_1
        buf656 = reinterpret_tensor(buf591, (1, 1, 4096), (4096, 4096, 1)); del buf591  # reuse
        # Source Nodes: [add_131, add_135, add_137, add_138, float_93, mean_46, mul_368, mul_369, mul_370, rsqrt_46, type_as_92], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_13.run(buf620, buf626, buf648, buf654, arg46_1, buf656, 1, 4096, grid=grid(1), stream=stream0)
        del arg46_1
        buf657 = buf629; del buf629  # reuse
        # Source Nodes: [l__model___layers_23_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf656, (1, 4096), (0, 1), 0), reinterpret_tensor(arg181_1, (4096, 12288), (1, 4096), 0), out=buf657)
        del arg181_1
        buf660 = buf632; del buf632  # reuse
        buf658 = reinterpret_tensor(buf660, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf659 = reinterpret_tensor(buf660, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf663 = buf635; del buf635  # reuse
        buf661 = reinterpret_tensor(buf663, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf662 = reinterpret_tensor(buf663, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_46, stack_47], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf657, arg294_1, arg227_1, buf658, buf659, buf661, buf662, 2048, grid=grid(2048), stream=stream0)
        buf664 = buf646; del buf646  # reuse
        # Source Nodes: [setitem_46], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg275_1, buf664, 4718592, grid=grid(4718592), stream=stream0)
        del buf658
        del buf659
        del buf661
        del buf662
        buf671 = buf643; del buf643  # reuse
        # Source Nodes: [setitem_47], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg276_1, buf671, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_46, setitem_47], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf663, buf657, buf664, buf671, 4096, grid=grid(4096), stream=stream0)
        buf666 = reinterpret_tensor(buf656, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf656  # reuse
        # Source Nodes: [type_as_93], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf660, buf666, 4096, grid=grid(4096), stream=stream0)
        buf667 = buf636; del buf636  # reuse
        # Source Nodes: [setitem_46], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf664, buf667, arg275_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg275_1
        buf668 = reinterpret_tensor(buf645, (32, 1, 1152), (1152, 1152, 1)); del buf645  # reuse
        # Source Nodes: [matmul_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf666, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf667, (32, 128, 1152), (147456, 1, 128), 0), out=buf668)
        buf673 = reinterpret_tensor(buf640, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf640  # reuse
        # Source Nodes: [getitem, mul_379, softmax_23, where_23], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf668, buf673, 32, 1152, grid=grid(32), stream=stream0)
        buf674 = buf667; del buf667  # reuse
        # Source Nodes: [setitem_47], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf671, buf674, arg276_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg276_1
        buf675 = reinterpret_tensor(buf666, (32, 1, 128), (128, 128, 1)); del buf666  # reuse
        # Source Nodes: [matmul_47], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf673, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf674, (32, 1152, 128), (147456, 128, 1), 0), out=buf675)
        buf676 = buf569; del buf569  # reuse
        # Source Nodes: [l__model___layers_23_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf675, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg182_1, (4096, 4096), (1, 4096), 0), out=buf676)
        del arg182_1
        buf677 = buf620; del buf620  # reuse
        buf679 = reinterpret_tensor(buf675, (1, 1, 4096), (4096, 4096, 1)); del buf675  # reuse
        # Source Nodes: [add_131, add_135, add_137, add_141, add_142, float_96, mean_47, mul_380, mul_381, mul_382, rsqrt_47, type_as_95], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(buf677, buf626, buf648, buf654, buf676, arg47_1, buf679, 1, 4096, grid=grid(1), stream=stream0)
        del arg47_1
        buf680 = reinterpret_tensor(buf653, (1, 11008), (11008, 1)); del buf653  # reuse
        # Source Nodes: [l__model___layers_23_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf679, (1, 4096), (0, 1), 0), reinterpret_tensor(arg183_1, (4096, 11008), (1, 4096), 0), out=buf680)
        del arg183_1
        buf681 = buf652; del buf652  # reuse
        # Source Nodes: [l__model___layers_23_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf679, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg184_1, (4096, 11008), (1, 4096), 0), out=buf681)
        del arg184_1
        buf682 = reinterpret_tensor(buf680, (1, 1, 11008), (11008, 11008, 1)); del buf680  # reuse
        # Source Nodes: [mul_383, silu_23], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf682, buf681, 11008, grid=grid(11008), stream=stream0)
        buf683 = reinterpret_tensor(buf679, (1, 4096), (4096, 1)); del buf679  # reuse
        # Source Nodes: [l__model___layers_23_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf682, (1, 11008), (0, 1), 0), reinterpret_tensor(arg185_1, (11008, 4096), (1, 11008), 0), out=buf683)
        del arg185_1
        buf685 = reinterpret_tensor(buf676, (1, 1, 4096), (4096, 4096, 1)); del buf676  # reuse
        # Source Nodes: [add_143, add_144, float_97, mean_48, mul_384, mul_385, mul_386, rsqrt_48, type_as_96], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf677, buf683, arg48_1, buf685, 1, 4096, grid=grid(1), stream=stream0)
        del arg48_1
        buf686 = buf657; del buf657  # reuse
        # Source Nodes: [l__model___layers_24_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf685, (1, 4096), (0, 1), 0), reinterpret_tensor(arg186_1, (4096, 12288), (1, 4096), 0), out=buf686)
        del arg186_1
        buf689 = buf660; del buf660  # reuse
        buf687 = reinterpret_tensor(buf689, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf688 = reinterpret_tensor(buf689, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf692 = buf663; del buf663  # reuse
        buf690 = reinterpret_tensor(buf692, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf691 = reinterpret_tensor(buf692, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_48, stack_49], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf686, arg294_1, arg227_1, buf687, buf688, buf690, buf691, 2048, grid=grid(2048), stream=stream0)
        buf693 = buf674; del buf674  # reuse
        # Source Nodes: [setitem_48], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg277_1, buf693, 4718592, grid=grid(4718592), stream=stream0)
        del buf687
        del buf688
        del buf690
        del buf691
        buf700 = buf671; del buf671  # reuse
        # Source Nodes: [setitem_49], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg278_1, buf700, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_48, setitem_49], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf692, buf686, buf693, buf700, 4096, grid=grid(4096), stream=stream0)
        buf695 = reinterpret_tensor(buf685, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf685  # reuse
        # Source Nodes: [type_as_97], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf689, buf695, 4096, grid=grid(4096), stream=stream0)
        buf696 = buf664; del buf664  # reuse
        # Source Nodes: [setitem_48], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf693, buf696, arg277_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg277_1
        buf697 = reinterpret_tensor(buf673, (32, 1, 1152), (1152, 1152, 1)); del buf673  # reuse
        # Source Nodes: [matmul_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf695, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf696, (32, 128, 1152), (147456, 1, 128), 0), out=buf697)
        buf702 = reinterpret_tensor(buf668, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf668  # reuse
        # Source Nodes: [getitem, mul_395, softmax_24, where_24], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf697, buf702, 32, 1152, grid=grid(32), stream=stream0)
        buf703 = buf696; del buf696  # reuse
        # Source Nodes: [setitem_49], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf700, buf703, arg278_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg278_1
        buf704 = reinterpret_tensor(buf695, (32, 1, 128), (128, 128, 1)); del buf695  # reuse
        # Source Nodes: [matmul_49], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf702, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf703, (32, 1152, 128), (147456, 128, 1), 0), out=buf704)
        buf705 = buf654; del buf654  # reuse
        # Source Nodes: [l__model___layers_24_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf704, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg187_1, (4096, 4096), (1, 4096), 0), out=buf705)
        del arg187_1
        buf707 = reinterpret_tensor(buf704, (1, 1, 4096), (4096, 4096, 1)); del buf704  # reuse
        # Source Nodes: [add_143, add_147, add_148, float_100, mean_49, mul_396, mul_397, mul_398, rsqrt_49, type_as_99], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_12.run(buf677, buf683, buf705, arg49_1, buf707, 1, 4096, grid=grid(1), stream=stream0)
        del arg49_1
        buf708 = reinterpret_tensor(buf682, (1, 11008), (11008, 1)); del buf682  # reuse
        # Source Nodes: [l__model___layers_24_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf707, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg188_1, (4096, 11008), (1, 4096), 0), out=buf708)
        del arg188_1
        buf709 = buf681; del buf681  # reuse
        # Source Nodes: [l__model___layers_24_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf707, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg189_1, (4096, 11008), (1, 4096), 0), out=buf709)
        del arg189_1
        buf710 = reinterpret_tensor(buf708, (1, 1, 11008), (11008, 11008, 1)); del buf708  # reuse
        # Source Nodes: [mul_399, silu_24], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf710, buf709, 11008, grid=grid(11008), stream=stream0)
        buf711 = reinterpret_tensor(buf707, (1, 4096), (4096, 1)); del buf707  # reuse
        # Source Nodes: [l__model___layers_24_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf710, (1, 11008), (0, 1), 0), reinterpret_tensor(arg190_1, (11008, 4096), (1, 11008), 0), out=buf711)
        del arg190_1
        buf713 = reinterpret_tensor(buf648, (1, 1, 4096), (4096, 4096, 1)); del buf648  # reuse
        # Source Nodes: [add_143, add_147, add_149, add_150, float_101, mean_50, mul_400, mul_401, mul_402, rsqrt_50, type_as_100], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_13.run(buf677, buf683, buf705, buf711, arg50_1, buf713, 1, 4096, grid=grid(1), stream=stream0)
        del arg50_1
        buf714 = buf686; del buf686  # reuse
        # Source Nodes: [l__model___layers_25_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf713, (1, 4096), (0, 1), 0), reinterpret_tensor(arg191_1, (4096, 12288), (1, 4096), 0), out=buf714)
        del arg191_1
        buf717 = buf689; del buf689  # reuse
        buf715 = reinterpret_tensor(buf717, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf716 = reinterpret_tensor(buf717, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf720 = buf692; del buf692  # reuse
        buf718 = reinterpret_tensor(buf720, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf719 = reinterpret_tensor(buf720, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_50, stack_51], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf714, arg294_1, arg227_1, buf715, buf716, buf718, buf719, 2048, grid=grid(2048), stream=stream0)
        buf721 = buf703; del buf703  # reuse
        # Source Nodes: [setitem_50], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg279_1, buf721, 4718592, grid=grid(4718592), stream=stream0)
        del buf715
        del buf716
        del buf718
        del buf719
        buf728 = buf700; del buf700  # reuse
        # Source Nodes: [setitem_51], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg280_1, buf728, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_50, setitem_51], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf720, buf714, buf721, buf728, 4096, grid=grid(4096), stream=stream0)
        buf723 = reinterpret_tensor(buf713, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf713  # reuse
        # Source Nodes: [type_as_101], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf717, buf723, 4096, grid=grid(4096), stream=stream0)
        buf724 = buf693; del buf693  # reuse
        # Source Nodes: [setitem_50], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf721, buf724, arg279_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg279_1
        buf725 = reinterpret_tensor(buf702, (32, 1, 1152), (1152, 1152, 1)); del buf702  # reuse
        # Source Nodes: [matmul_50], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf723, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf724, (32, 128, 1152), (147456, 1, 128), 0), out=buf725)
        buf730 = reinterpret_tensor(buf697, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf697  # reuse
        # Source Nodes: [getitem, mul_411, softmax_25, where_25], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf725, buf730, 32, 1152, grid=grid(32), stream=stream0)
        buf731 = buf724; del buf724  # reuse
        # Source Nodes: [setitem_51], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf728, buf731, arg280_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg280_1
        buf732 = reinterpret_tensor(buf723, (32, 1, 128), (128, 128, 1)); del buf723  # reuse
        # Source Nodes: [matmul_51], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf730, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf731, (32, 1152, 128), (147456, 128, 1), 0), out=buf732)
        buf733 = buf626; del buf626  # reuse
        # Source Nodes: [l__model___layers_25_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf732, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg192_1, (4096, 4096), (1, 4096), 0), out=buf733)
        del arg192_1
        buf734 = buf677; del buf677  # reuse
        buf736 = reinterpret_tensor(buf732, (1, 1, 4096), (4096, 4096, 1)); del buf732  # reuse
        # Source Nodes: [add_143, add_147, add_149, add_153, add_154, float_104, mean_51, mul_412, mul_413, mul_414, rsqrt_51, type_as_103], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(buf734, buf683, buf705, buf711, buf733, arg51_1, buf736, 1, 4096, grid=grid(1), stream=stream0)
        del arg51_1
        buf737 = reinterpret_tensor(buf710, (1, 11008), (11008, 1)); del buf710  # reuse
        # Source Nodes: [l__model___layers_25_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf736, (1, 4096), (0, 1), 0), reinterpret_tensor(arg193_1, (4096, 11008), (1, 4096), 0), out=buf737)
        del arg193_1
        buf738 = buf709; del buf709  # reuse
        # Source Nodes: [l__model___layers_25_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf736, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg194_1, (4096, 11008), (1, 4096), 0), out=buf738)
        del arg194_1
        buf739 = reinterpret_tensor(buf737, (1, 1, 11008), (11008, 11008, 1)); del buf737  # reuse
        # Source Nodes: [mul_415, silu_25], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf739, buf738, 11008, grid=grid(11008), stream=stream0)
        buf740 = reinterpret_tensor(buf736, (1, 4096), (4096, 1)); del buf736  # reuse
        # Source Nodes: [l__model___layers_25_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf739, (1, 11008), (0, 1), 0), reinterpret_tensor(arg195_1, (11008, 4096), (1, 11008), 0), out=buf740)
        del arg195_1
        buf742 = reinterpret_tensor(buf733, (1, 1, 4096), (4096, 4096, 1)); del buf733  # reuse
        # Source Nodes: [add_155, add_156, float_105, mean_52, mul_416, mul_417, mul_418, rsqrt_52, type_as_104], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf734, buf740, arg52_1, buf742, 1, 4096, grid=grid(1), stream=stream0)
        del arg52_1
        buf743 = buf714; del buf714  # reuse
        # Source Nodes: [l__model___layers_26_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf742, (1, 4096), (0, 1), 0), reinterpret_tensor(arg196_1, (4096, 12288), (1, 4096), 0), out=buf743)
        del arg196_1
        buf746 = buf717; del buf717  # reuse
        buf744 = reinterpret_tensor(buf746, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf745 = reinterpret_tensor(buf746, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf749 = buf720; del buf720  # reuse
        buf747 = reinterpret_tensor(buf749, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf748 = reinterpret_tensor(buf749, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_52, stack_53], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf743, arg294_1, arg227_1, buf744, buf745, buf747, buf748, 2048, grid=grid(2048), stream=stream0)
        buf750 = buf731; del buf731  # reuse
        # Source Nodes: [setitem_52], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg281_1, buf750, 4718592, grid=grid(4718592), stream=stream0)
        del buf744
        del buf745
        del buf747
        del buf748
        buf757 = buf728; del buf728  # reuse
        # Source Nodes: [setitem_53], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg282_1, buf757, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_52, setitem_53], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf749, buf743, buf750, buf757, 4096, grid=grid(4096), stream=stream0)
        buf752 = reinterpret_tensor(buf742, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf742  # reuse
        # Source Nodes: [type_as_105], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf746, buf752, 4096, grid=grid(4096), stream=stream0)
        buf753 = buf721; del buf721  # reuse
        # Source Nodes: [setitem_52], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf750, buf753, arg281_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg281_1
        buf754 = reinterpret_tensor(buf730, (32, 1, 1152), (1152, 1152, 1)); del buf730  # reuse
        # Source Nodes: [matmul_52], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf752, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf753, (32, 128, 1152), (147456, 1, 128), 0), out=buf754)
        buf759 = reinterpret_tensor(buf725, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf725  # reuse
        # Source Nodes: [getitem, mul_427, softmax_26, where_26], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf754, buf759, 32, 1152, grid=grid(32), stream=stream0)
        buf760 = buf753; del buf753  # reuse
        # Source Nodes: [setitem_53], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf757, buf760, arg282_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg282_1
        buf761 = reinterpret_tensor(buf752, (32, 1, 128), (128, 128, 1)); del buf752  # reuse
        # Source Nodes: [matmul_53], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf759, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf760, (32, 1152, 128), (147456, 128, 1), 0), out=buf761)
        buf762 = buf711; del buf711  # reuse
        # Source Nodes: [l__model___layers_26_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf761, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg197_1, (4096, 4096), (1, 4096), 0), out=buf762)
        del arg197_1
        buf764 = reinterpret_tensor(buf761, (1, 1, 4096), (4096, 4096, 1)); del buf761  # reuse
        # Source Nodes: [add_155, add_159, add_160, float_108, mean_53, mul_428, mul_429, mul_430, rsqrt_53, type_as_107], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_12.run(buf734, buf740, buf762, arg53_1, buf764, 1, 4096, grid=grid(1), stream=stream0)
        del arg53_1
        buf765 = reinterpret_tensor(buf739, (1, 11008), (11008, 1)); del buf739  # reuse
        # Source Nodes: [l__model___layers_26_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf764, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg198_1, (4096, 11008), (1, 4096), 0), out=buf765)
        del arg198_1
        buf766 = buf738; del buf738  # reuse
        # Source Nodes: [l__model___layers_26_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf764, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg199_1, (4096, 11008), (1, 4096), 0), out=buf766)
        del arg199_1
        buf767 = reinterpret_tensor(buf765, (1, 1, 11008), (11008, 11008, 1)); del buf765  # reuse
        # Source Nodes: [mul_431, silu_26], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf767, buf766, 11008, grid=grid(11008), stream=stream0)
        buf768 = reinterpret_tensor(buf764, (1, 4096), (4096, 1)); del buf764  # reuse
        # Source Nodes: [l__model___layers_26_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf767, (1, 11008), (0, 1), 0), reinterpret_tensor(arg200_1, (11008, 4096), (1, 11008), 0), out=buf768)
        del arg200_1
        buf770 = reinterpret_tensor(buf705, (1, 1, 4096), (4096, 4096, 1)); del buf705  # reuse
        # Source Nodes: [add_155, add_159, add_161, add_162, float_109, mean_54, mul_432, mul_433, mul_434, rsqrt_54, type_as_108], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_13.run(buf734, buf740, buf762, buf768, arg54_1, buf770, 1, 4096, grid=grid(1), stream=stream0)
        del arg54_1
        buf771 = buf743; del buf743  # reuse
        # Source Nodes: [l__model___layers_27_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf770, (1, 4096), (0, 1), 0), reinterpret_tensor(arg201_1, (4096, 12288), (1, 4096), 0), out=buf771)
        del arg201_1
        buf774 = buf746; del buf746  # reuse
        buf772 = reinterpret_tensor(buf774, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf773 = reinterpret_tensor(buf774, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf777 = buf749; del buf749  # reuse
        buf775 = reinterpret_tensor(buf777, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf776 = reinterpret_tensor(buf777, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_54, stack_55], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf771, arg294_1, arg227_1, buf772, buf773, buf775, buf776, 2048, grid=grid(2048), stream=stream0)
        buf778 = buf760; del buf760  # reuse
        # Source Nodes: [setitem_54], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg283_1, buf778, 4718592, grid=grid(4718592), stream=stream0)
        del buf772
        del buf773
        del buf775
        del buf776
        buf785 = buf757; del buf757  # reuse
        # Source Nodes: [setitem_55], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg284_1, buf785, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_54, setitem_55], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf777, buf771, buf778, buf785, 4096, grid=grid(4096), stream=stream0)
        buf780 = reinterpret_tensor(buf770, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf770  # reuse
        # Source Nodes: [type_as_109], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf774, buf780, 4096, grid=grid(4096), stream=stream0)
        buf781 = buf750; del buf750  # reuse
        # Source Nodes: [setitem_54], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf778, buf781, arg283_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg283_1
        buf782 = reinterpret_tensor(buf759, (32, 1, 1152), (1152, 1152, 1)); del buf759  # reuse
        # Source Nodes: [matmul_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf780, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf781, (32, 128, 1152), (147456, 1, 128), 0), out=buf782)
        buf787 = reinterpret_tensor(buf754, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf754  # reuse
        # Source Nodes: [getitem, mul_443, softmax_27, where_27], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf782, buf787, 32, 1152, grid=grid(32), stream=stream0)
        buf788 = buf781; del buf781  # reuse
        # Source Nodes: [setitem_55], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf785, buf788, arg284_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg284_1
        buf789 = reinterpret_tensor(buf780, (32, 1, 128), (128, 128, 1)); del buf780  # reuse
        # Source Nodes: [matmul_55], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf787, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf788, (32, 1152, 128), (147456, 128, 1), 0), out=buf789)
        buf790 = buf683; del buf683  # reuse
        # Source Nodes: [l__model___layers_27_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf789, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg202_1, (4096, 4096), (1, 4096), 0), out=buf790)
        del arg202_1
        buf791 = buf734; del buf734  # reuse
        buf793 = reinterpret_tensor(buf789, (1, 1, 4096), (4096, 4096, 1)); del buf789  # reuse
        # Source Nodes: [add_155, add_159, add_161, add_165, add_166, float_112, mean_55, mul_444, mul_445, mul_446, rsqrt_55, type_as_111], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(buf791, buf740, buf762, buf768, buf790, arg55_1, buf793, 1, 4096, grid=grid(1), stream=stream0)
        del arg55_1
        buf794 = reinterpret_tensor(buf767, (1, 11008), (11008, 1)); del buf767  # reuse
        # Source Nodes: [l__model___layers_27_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf793, (1, 4096), (0, 1), 0), reinterpret_tensor(arg203_1, (4096, 11008), (1, 4096), 0), out=buf794)
        del arg203_1
        buf795 = buf766; del buf766  # reuse
        # Source Nodes: [l__model___layers_27_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf793, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg204_1, (4096, 11008), (1, 4096), 0), out=buf795)
        del arg204_1
        buf796 = reinterpret_tensor(buf794, (1, 1, 11008), (11008, 11008, 1)); del buf794  # reuse
        # Source Nodes: [mul_447, silu_27], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf796, buf795, 11008, grid=grid(11008), stream=stream0)
        buf797 = reinterpret_tensor(buf793, (1, 4096), (4096, 1)); del buf793  # reuse
        # Source Nodes: [l__model___layers_27_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf796, (1, 11008), (0, 1), 0), reinterpret_tensor(arg205_1, (11008, 4096), (1, 11008), 0), out=buf797)
        del arg205_1
        buf799 = reinterpret_tensor(buf790, (1, 1, 4096), (4096, 4096, 1)); del buf790  # reuse
        # Source Nodes: [add_167, add_168, float_113, mean_56, mul_448, mul_449, mul_450, rsqrt_56, type_as_112], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf791, buf797, arg56_1, buf799, 1, 4096, grid=grid(1), stream=stream0)
        del arg56_1
        buf800 = buf771; del buf771  # reuse
        # Source Nodes: [l__model___layers_28_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf799, (1, 4096), (0, 1), 0), reinterpret_tensor(arg206_1, (4096, 12288), (1, 4096), 0), out=buf800)
        del arg206_1
        buf803 = buf774; del buf774  # reuse
        buf801 = reinterpret_tensor(buf803, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf802 = reinterpret_tensor(buf803, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf806 = buf777; del buf777  # reuse
        buf804 = reinterpret_tensor(buf806, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf805 = reinterpret_tensor(buf806, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_56, stack_57], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf800, arg294_1, arg227_1, buf801, buf802, buf804, buf805, 2048, grid=grid(2048), stream=stream0)
        buf807 = buf788; del buf788  # reuse
        # Source Nodes: [setitem_56], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg285_1, buf807, 4718592, grid=grid(4718592), stream=stream0)
        del buf801
        del buf802
        del buf804
        del buf805
        buf814 = buf785; del buf785  # reuse
        # Source Nodes: [setitem_57], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg286_1, buf814, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_56, setitem_57], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf806, buf800, buf807, buf814, 4096, grid=grid(4096), stream=stream0)
        buf809 = reinterpret_tensor(buf799, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf799  # reuse
        # Source Nodes: [type_as_113], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf803, buf809, 4096, grid=grid(4096), stream=stream0)
        buf810 = buf778; del buf778  # reuse
        # Source Nodes: [setitem_56], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf807, buf810, arg285_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg285_1
        buf811 = reinterpret_tensor(buf787, (32, 1, 1152), (1152, 1152, 1)); del buf787  # reuse
        # Source Nodes: [matmul_56], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf809, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf810, (32, 128, 1152), (147456, 1, 128), 0), out=buf811)
        buf816 = reinterpret_tensor(buf782, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf782  # reuse
        # Source Nodes: [getitem, mul_459, softmax_28, where_28], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf811, buf816, 32, 1152, grid=grid(32), stream=stream0)
        buf817 = buf810; del buf810  # reuse
        # Source Nodes: [setitem_57], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf814, buf817, arg286_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg286_1
        buf818 = reinterpret_tensor(buf809, (32, 1, 128), (128, 128, 1)); del buf809  # reuse
        # Source Nodes: [matmul_57], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf816, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf817, (32, 1152, 128), (147456, 128, 1), 0), out=buf818)
        buf819 = buf768; del buf768  # reuse
        # Source Nodes: [l__model___layers_28_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf818, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg207_1, (4096, 4096), (1, 4096), 0), out=buf819)
        del arg207_1
        buf821 = reinterpret_tensor(buf818, (1, 1, 4096), (4096, 4096, 1)); del buf818  # reuse
        # Source Nodes: [add_167, add_171, add_172, float_116, mean_57, mul_460, mul_461, mul_462, rsqrt_57, type_as_115], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_12.run(buf791, buf797, buf819, arg57_1, buf821, 1, 4096, grid=grid(1), stream=stream0)
        del arg57_1
        buf822 = reinterpret_tensor(buf796, (1, 11008), (11008, 1)); del buf796  # reuse
        # Source Nodes: [l__model___layers_28_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf821, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg208_1, (4096, 11008), (1, 4096), 0), out=buf822)
        del arg208_1
        buf823 = buf795; del buf795  # reuse
        # Source Nodes: [l__model___layers_28_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf821, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg209_1, (4096, 11008), (1, 4096), 0), out=buf823)
        del arg209_1
        buf824 = reinterpret_tensor(buf822, (1, 1, 11008), (11008, 11008, 1)); del buf822  # reuse
        # Source Nodes: [mul_463, silu_28], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf824, buf823, 11008, grid=grid(11008), stream=stream0)
        buf825 = reinterpret_tensor(buf821, (1, 4096), (4096, 1)); del buf821  # reuse
        # Source Nodes: [l__model___layers_28_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf824, (1, 11008), (0, 1), 0), reinterpret_tensor(arg210_1, (11008, 4096), (1, 11008), 0), out=buf825)
        del arg210_1
        buf827 = reinterpret_tensor(buf762, (1, 1, 4096), (4096, 4096, 1)); del buf762  # reuse
        # Source Nodes: [add_167, add_171, add_173, add_174, float_117, mean_58, mul_464, mul_465, mul_466, rsqrt_58, type_as_116], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_13.run(buf791, buf797, buf819, buf825, arg58_1, buf827, 1, 4096, grid=grid(1), stream=stream0)
        del arg58_1
        buf828 = buf800; del buf800  # reuse
        # Source Nodes: [l__model___layers_29_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf827, (1, 4096), (0, 1), 0), reinterpret_tensor(arg211_1, (4096, 12288), (1, 4096), 0), out=buf828)
        del arg211_1
        buf831 = buf803; del buf803  # reuse
        buf829 = reinterpret_tensor(buf831, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf830 = reinterpret_tensor(buf831, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf834 = buf806; del buf806  # reuse
        buf832 = reinterpret_tensor(buf834, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf833 = reinterpret_tensor(buf834, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_58, stack_59], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf828, arg294_1, arg227_1, buf829, buf830, buf832, buf833, 2048, grid=grid(2048), stream=stream0)
        buf835 = buf817; del buf817  # reuse
        # Source Nodes: [setitem_58], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg287_1, buf835, 4718592, grid=grid(4718592), stream=stream0)
        del buf829
        del buf830
        del buf832
        del buf833
        buf842 = buf814; del buf814  # reuse
        # Source Nodes: [setitem_59], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg288_1, buf842, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_58, setitem_59], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf834, buf828, buf835, buf842, 4096, grid=grid(4096), stream=stream0)
        buf837 = reinterpret_tensor(buf827, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf827  # reuse
        # Source Nodes: [type_as_117], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf831, buf837, 4096, grid=grid(4096), stream=stream0)
        buf838 = buf807; del buf807  # reuse
        # Source Nodes: [setitem_58], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf835, buf838, arg287_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg287_1
        buf839 = reinterpret_tensor(buf816, (32, 1, 1152), (1152, 1152, 1)); del buf816  # reuse
        # Source Nodes: [matmul_58], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf837, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf838, (32, 128, 1152), (147456, 1, 128), 0), out=buf839)
        buf844 = reinterpret_tensor(buf811, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf811  # reuse
        # Source Nodes: [getitem, mul_475, softmax_29, where_29], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf839, buf844, 32, 1152, grid=grid(32), stream=stream0)
        buf845 = buf838; del buf838  # reuse
        # Source Nodes: [setitem_59], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf842, buf845, arg288_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg288_1
        buf846 = reinterpret_tensor(buf837, (32, 1, 128), (128, 128, 1)); del buf837  # reuse
        # Source Nodes: [matmul_59], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf844, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf845, (32, 1152, 128), (147456, 128, 1), 0), out=buf846)
        buf847 = buf740; del buf740  # reuse
        # Source Nodes: [l__model___layers_29_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf846, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg212_1, (4096, 4096), (1, 4096), 0), out=buf847)
        del arg212_1
        buf848 = buf791; del buf791  # reuse
        buf850 = reinterpret_tensor(buf846, (1, 1, 4096), (4096, 4096, 1)); del buf846  # reuse
        # Source Nodes: [add_167, add_171, add_173, add_177, add_178, float_120, mean_59, mul_476, mul_477, mul_478, rsqrt_59, type_as_119], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(buf848, buf797, buf819, buf825, buf847, arg59_1, buf850, 1, 4096, grid=grid(1), stream=stream0)
        del arg59_1
        buf851 = reinterpret_tensor(buf824, (1, 11008), (11008, 1)); del buf824  # reuse
        # Source Nodes: [l__model___layers_29_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf850, (1, 4096), (0, 1), 0), reinterpret_tensor(arg213_1, (4096, 11008), (1, 4096), 0), out=buf851)
        del arg213_1
        buf852 = buf823; del buf823  # reuse
        # Source Nodes: [l__model___layers_29_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf850, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg214_1, (4096, 11008), (1, 4096), 0), out=buf852)
        del arg214_1
        buf853 = reinterpret_tensor(buf851, (1, 1, 11008), (11008, 11008, 1)); del buf851  # reuse
        # Source Nodes: [mul_479, silu_29], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf853, buf852, 11008, grid=grid(11008), stream=stream0)
        buf854 = reinterpret_tensor(buf850, (1, 4096), (4096, 1)); del buf850  # reuse
        # Source Nodes: [l__model___layers_29_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf853, (1, 11008), (0, 1), 0), reinterpret_tensor(arg215_1, (11008, 4096), (1, 11008), 0), out=buf854)
        del arg215_1
        buf856 = reinterpret_tensor(buf847, (1, 1, 4096), (4096, 4096, 1)); del buf847  # reuse
        # Source Nodes: [add_179, add_180, float_121, mean_60, mul_480, mul_481, mul_482, rsqrt_60, type_as_120], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf848, buf854, arg60_1, buf856, 1, 4096, grid=grid(1), stream=stream0)
        del arg60_1
        buf857 = buf828; del buf828  # reuse
        # Source Nodes: [l__model___layers_30_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf856, (1, 4096), (0, 1), 0), reinterpret_tensor(arg216_1, (4096, 12288), (1, 4096), 0), out=buf857)
        del arg216_1
        buf860 = buf831; del buf831  # reuse
        buf858 = reinterpret_tensor(buf860, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf859 = reinterpret_tensor(buf860, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf863 = buf834; del buf834  # reuse
        buf861 = reinterpret_tensor(buf863, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf862 = reinterpret_tensor(buf863, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_60, stack_61], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf857, arg294_1, arg227_1, buf858, buf859, buf861, buf862, 2048, grid=grid(2048), stream=stream0)
        buf864 = buf845; del buf845  # reuse
        # Source Nodes: [setitem_60], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg289_1, buf864, 4718592, grid=grid(4718592), stream=stream0)
        del buf858
        del buf859
        del buf861
        del buf862
        buf871 = buf842; del buf842  # reuse
        # Source Nodes: [setitem_61], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg290_1, buf871, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_60, setitem_61], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf863, buf857, buf864, buf871, 4096, grid=grid(4096), stream=stream0)
        buf866 = reinterpret_tensor(buf856, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf856  # reuse
        # Source Nodes: [type_as_121], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf860, buf866, 4096, grid=grid(4096), stream=stream0)
        buf867 = buf835; del buf835  # reuse
        # Source Nodes: [setitem_60], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf864, buf867, arg289_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg289_1
        buf868 = reinterpret_tensor(buf844, (32, 1, 1152), (1152, 1152, 1)); del buf844  # reuse
        # Source Nodes: [matmul_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf866, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf867, (32, 128, 1152), (147456, 1, 128), 0), out=buf868)
        buf873 = reinterpret_tensor(buf839, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf839  # reuse
        # Source Nodes: [getitem, mul_491, softmax_30, where_30], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf868, buf873, 32, 1152, grid=grid(32), stream=stream0)
        buf874 = buf867; del buf867  # reuse
        # Source Nodes: [setitem_61], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf871, buf874, arg290_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg290_1
        buf875 = reinterpret_tensor(buf866, (32, 1, 128), (128, 128, 1)); del buf866  # reuse
        # Source Nodes: [matmul_61], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf873, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf874, (32, 1152, 128), (147456, 128, 1), 0), out=buf875)
        buf876 = buf825; del buf825  # reuse
        # Source Nodes: [l__model___layers_30_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf875, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg217_1, (4096, 4096), (1, 4096), 0), out=buf876)
        del arg217_1
        buf878 = reinterpret_tensor(buf875, (1, 1, 4096), (4096, 4096, 1)); del buf875  # reuse
        # Source Nodes: [add_179, add_183, add_184, float_124, mean_61, mul_492, mul_493, mul_494, rsqrt_61, type_as_123], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_12.run(buf848, buf854, buf876, arg61_1, buf878, 1, 4096, grid=grid(1), stream=stream0)
        del arg61_1
        buf879 = reinterpret_tensor(buf853, (1, 11008), (11008, 1)); del buf853  # reuse
        # Source Nodes: [l__model___layers_30_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf878, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg218_1, (4096, 11008), (1, 4096), 0), out=buf879)
        del arg218_1
        buf880 = buf852; del buf852  # reuse
        # Source Nodes: [l__model___layers_30_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf878, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg219_1, (4096, 11008), (1, 4096), 0), out=buf880)
        del arg219_1
        buf881 = reinterpret_tensor(buf879, (1, 1, 11008), (11008, 11008, 1)); del buf879  # reuse
        # Source Nodes: [mul_495, silu_30], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf881, buf880, 11008, grid=grid(11008), stream=stream0)
        buf882 = reinterpret_tensor(buf878, (1, 4096), (4096, 1)); del buf878  # reuse
        # Source Nodes: [l__model___layers_30_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf881, (1, 11008), (0, 1), 0), reinterpret_tensor(arg220_1, (11008, 4096), (1, 11008), 0), out=buf882)
        del arg220_1
        buf884 = reinterpret_tensor(buf819, (1, 1, 4096), (4096, 4096, 1)); del buf819  # reuse
        # Source Nodes: [add_179, add_183, add_185, add_186, float_125, mean_62, mul_496, mul_497, mul_498, rsqrt_62, type_as_124], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_13.run(buf848, buf854, buf876, buf882, arg62_1, buf884, 1, 4096, grid=grid(1), stream=stream0)
        del arg62_1
        buf885 = buf857; del buf857  # reuse
        # Source Nodes: [l__model___layers_31_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf884, (1, 4096), (0, 1), 0), reinterpret_tensor(arg221_1, (4096, 12288), (1, 4096), 0), out=buf885)
        del arg221_1
        buf888 = buf860; del buf860  # reuse
        buf886 = reinterpret_tensor(buf888, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf887 = reinterpret_tensor(buf888, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf891 = buf863; del buf863  # reuse
        buf889 = reinterpret_tensor(buf891, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf890 = reinterpret_tensor(buf891, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_62, stack_63], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf885, arg294_1, arg227_1, buf886, buf887, buf889, buf890, 2048, grid=grid(2048), stream=stream0)
        del arg227_1
        buf892 = buf874; del buf874  # reuse
        # Source Nodes: [setitem_62], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg291_1, buf892, 4718592, grid=grid(4718592), stream=stream0)
        del buf886
        del buf887
        del buf889
        del buf890
        buf899 = buf871; del buf871  # reuse
        # Source Nodes: [setitem_63], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg292_1, buf899, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_62, setitem_63], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg294_1, buf891, buf885, buf892, buf899, 4096, grid=grid(4096), stream=stream0)
        del buf885
        del buf891
        buf894 = reinterpret_tensor(buf884, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf884  # reuse
        # Source Nodes: [type_as_125], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf888, buf894, 4096, grid=grid(4096), stream=stream0)
        del buf888
        buf895 = buf864; del buf864  # reuse
        # Source Nodes: [setitem_62], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf892, buf895, arg291_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg291_1
        del buf892
        buf896 = reinterpret_tensor(buf873, (32, 1, 1152), (1152, 1152, 1)); del buf873  # reuse
        # Source Nodes: [matmul_62], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf894, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf895, (32, 128, 1152), (147456, 1, 128), 0), out=buf896)
        buf901 = reinterpret_tensor(buf868, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf868  # reuse
        # Source Nodes: [getitem, mul_507, softmax_31, where_31], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg294_1, arg228_1, buf896, buf901, 32, 1152, grid=grid(32), stream=stream0)
        del arg228_1
        del arg294_1
        del buf896
        buf902 = buf895; del buf895  # reuse
        # Source Nodes: [setitem_63], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf899, buf902, arg292_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg292_1
        del buf899
        buf903 = reinterpret_tensor(buf894, (32, 1, 128), (128, 128, 1)); del buf894  # reuse
        # Source Nodes: [matmul_63], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf901, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf902, (32, 1152, 128), (147456, 128, 1), 0), out=buf903)
        del buf901
        del buf902
        buf904 = buf797; del buf797  # reuse
        # Source Nodes: [l__model___layers_31_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf903, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg222_1, (4096, 4096), (1, 4096), 0), out=buf904)
        del arg222_1
        buf905 = buf848; del buf848  # reuse
        buf907 = reinterpret_tensor(buf903, (1, 1, 4096), (4096, 4096, 1)); del buf903  # reuse
        # Source Nodes: [add_179, add_183, add_185, add_189, add_190, float_128, mean_63, mul_508, mul_509, mul_510, rsqrt_63, type_as_127], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(buf905, buf854, buf876, buf882, buf904, arg63_1, buf907, 1, 4096, grid=grid(1), stream=stream0)
        del arg63_1
        del buf854
        del buf876
        del buf882
        buf908 = reinterpret_tensor(buf881, (1, 11008), (11008, 1)); del buf881  # reuse
        # Source Nodes: [l__model___layers_31_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf907, (1, 4096), (0, 1), 0), reinterpret_tensor(arg223_1, (4096, 11008), (1, 4096), 0), out=buf908)
        del arg223_1
        buf909 = buf880; del buf880  # reuse
        # Source Nodes: [l__model___layers_31_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf907, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg224_1, (4096, 11008), (1, 4096), 0), out=buf909)
        del arg224_1
        buf910 = reinterpret_tensor(buf908, (1, 1, 11008), (11008, 11008, 1)); del buf908  # reuse
        # Source Nodes: [mul_511, silu_31], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf910, buf909, 11008, grid=grid(11008), stream=stream0)
        del buf909
        buf911 = reinterpret_tensor(buf907, (1, 4096), (4096, 1)); del buf907  # reuse
        # Source Nodes: [l__model___layers_31_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf910, (1, 11008), (0, 1), 0), reinterpret_tensor(arg225_1, (11008, 4096), (1, 11008), 0), out=buf911)
        del arg225_1
        del buf910
        buf913 = reinterpret_tensor(buf904, (1, 1, 4096), (4096, 4096, 1)); del buf904  # reuse
        # Source Nodes: [add_191, add_192, float_129, mean_64, mul_512, mul_513, mul_514, rsqrt_64, type_as_128], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf905, buf911, arg64_1, buf913, 1, 4096, grid=grid(1), stream=stream0)
        del arg64_1
        del buf905
        del buf911
        buf914 = empty_strided((1, 32000), (32000, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___output], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf913, (1, 4096), (0, 1), 0), reinterpret_tensor(arg226_1, (4096, 32000), (1, 4096), 0), out=buf914)
        del arg226_1
        del buf913
        buf915 = buf914; del buf914  # reuse
        # Source Nodes: [truediv], Original ATen: [aten.div]
        triton_poi_fused_div_16.run(buf915, 32000, grid=grid(32000), stream=stream0)
        # Source Nodes: [topk, truediv], Original ATen: [aten.div, aten.topk]
        buf916 = aten.topk(buf915, 200)
        buf917 = buf916[0]
        assert_size_stride(buf917, (1, 200), (200, 1))
        del buf916
        buf919 = empty_strided((1, 1, 4), (4, 4, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [lt, softmax_32, where_32], Original ATen: [aten._softmax, aten.lt, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_lt_scalar_tensor_where_17.run(buf915, buf917, buf919, 4, 8000, grid=grid(4), stream=stream0)
        buf920 = empty_strided((1, 1), (1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [lt, softmax_32, where_32], Original ATen: [aten._softmax, aten.lt, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_lt_scalar_tensor_where_18.run(buf919, buf920, 1, 4, grid=grid(1), stream=stream0)
        buf921 = buf919; del buf919  # reuse
        # Source Nodes: [lt, softmax_32, where_32], Original ATen: [aten._softmax, aten.lt, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_lt_scalar_tensor_where_19.run(buf915, buf917, buf920, buf921, 4, 8000, grid=grid(4), stream=stream0)
        buf922 = empty_strided((1, 1), (1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [lt, softmax_32, where_32], Original ATen: [aten._softmax, aten.lt, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_lt_scalar_tensor_where_20.run(buf921, buf922, 1, 4, grid=grid(1), stream=stream0)
        del buf921
        buf924 = empty_strided((1, ), (1, ), device='cuda', dtype=torch.int64)
        # Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf924)
        buf923 = buf915; del buf915  # reuse
        buf927 = empty_strided((1, 1), (1, 1), device='cuda', dtype=torch.int32)
        # Source Nodes: [argmax, exponential_, lt, softmax_32, to, truediv_1, where_32], Original ATen: [aten._softmax, aten._to_copy, aten.argmax, aten.div, aten.exponential, aten.lt, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax__to_copy_argmax_div_exponential_lt_scalar_tensor_where_21.run(buf923, buf917, buf920, buf922, buf924, buf927, 0, 1, 32000, grid=grid(1), stream=stream0)
        return (buf927, buf923, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg7_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg8_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg9_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg10_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg11_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg12_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg13_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg14_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg15_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg16_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg17_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg18_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg19_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg20_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg21_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg22_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg23_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg24_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg25_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg26_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg27_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg28_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg29_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg30_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg31_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg32_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg33_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg34_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg35_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg36_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg37_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg38_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg39_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg40_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg41_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg42_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg43_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg44_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg45_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg46_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg47_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg48_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg49_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg50_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg51_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg52_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg53_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg54_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg55_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg56_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg57_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg58_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg59_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg60_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg61_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg62_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg63_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg64_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg65_1 = rand_strided((32000, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg66_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg67_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg68_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg69_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg70_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg71_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg72_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg73_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg74_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg75_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg76_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg77_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg78_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg79_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg80_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg81_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg82_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg83_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg84_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg85_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg86_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg87_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg88_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg89_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg90_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg91_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg92_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg93_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg94_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg95_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg96_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg97_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg98_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg99_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg100_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg101_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg102_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg103_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg104_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg105_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg106_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg107_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg108_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg109_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg110_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg111_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg112_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg113_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg114_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg115_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg116_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg117_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg118_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg119_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg120_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg121_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg122_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg123_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg124_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg125_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg126_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg127_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg128_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg129_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg130_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg131_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg132_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg133_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg134_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg135_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg136_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg137_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg138_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg139_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg140_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg141_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg142_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg143_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg144_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg145_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg146_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg147_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg148_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg149_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg150_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg151_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg152_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg153_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg154_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg155_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg156_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg157_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg158_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg159_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg160_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg161_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg162_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg163_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg164_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg165_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg166_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg167_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg168_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg169_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg170_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg171_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg172_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg173_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg174_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg175_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg176_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg177_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg178_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg179_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg180_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg181_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg182_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg183_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg184_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg185_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg186_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg187_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg188_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg189_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg190_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg191_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg192_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg193_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg194_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg195_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg196_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg197_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg198_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg199_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg200_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg201_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg202_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg203_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg204_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg205_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg206_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg207_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg208_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg209_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg210_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg211_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg212_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg213_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg214_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg215_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg216_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg217_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg218_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg219_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg220_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg221_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg222_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg223_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg224_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg225_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg226_1 = rand_strided((32000, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg227_1 = rand_strided((2048, 64, 2), (128, 2, 1), device='cuda:0', dtype=torch.float16)
    arg228_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bool)
    arg229_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg230_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg231_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg232_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg233_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg234_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg235_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg236_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg237_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg238_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg239_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg240_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg241_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg242_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg243_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg244_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg245_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg246_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg247_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg248_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg249_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg250_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg251_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg252_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg253_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg254_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg255_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg256_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg257_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg258_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg259_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg260_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg261_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg262_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg263_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg264_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg265_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg266_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg267_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg268_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg269_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg270_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg271_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg272_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg273_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg274_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg275_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg276_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg277_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg278_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg279_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg280_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg281_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg282_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg283_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg284_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg285_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg286_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg287_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg288_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg289_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg290_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg291_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg292_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg293_1 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int32)
    arg294_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
