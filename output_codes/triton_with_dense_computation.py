
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


# kernel path: /tmp/torchinductor_mengqy/lw/clwlwbnl4ftkuh5llybxm67z3qldimcvxbuwpextfsvcw7s3w3zv.py
# Source Nodes: [add, add_6, add_7, float_1, float_5, l__model___tok_embeddings, mean, mean_2, mul, mul_1, mul_16, mul_17, mul_18, mul_19, mul_2, rsqrt, rsqrt_2, type_as, type_as_4], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.rsqrt]
# add => add
# add_6 => add_6
# add_7 => add_7
# float_1 => convert_element_type
# float_5 => convert_element_type_12
# l__model___tok_embeddings => embedding
# mean => mean
# mean_2 => mean_2
# mul => mul
# mul_1 => mul_1
# mul_16 => mul_17
# mul_17 => mul_18
# mul_18 => mul_19
# mul_19 => mul_20
# mul_2 => mul_2
# rsqrt => rsqrt
# rsqrt_2 => rsqrt_2
# type_as => convert_element_type_1
# type_as_4 => convert_element_type_13
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
    meta={'signature': {0: '*i32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_0', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_0(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr2, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    _tmp7 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp9 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
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
        tmp11 = tmp10 * tmp3
        tmp12 = 0.0
        tmp13 = tmp11 + tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp14 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask, tmp18, _tmp17)
    tmp7 = tl.sum(_tmp7, 1)[:, None]
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tmp19 = tl.load(in_ptr0 + (0))
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp33 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp31 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp44 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp21 = tl.where(tmp20 < 0, tmp20 + 32000, tmp20)
        tl.device_assert((0 <= tmp21) & (tmp21 < 32000), "index out of bounds: 0 <= tmp21 < 32000")
        tmp22 = tl.load(in_ptr1 + (r0 + (4096*tmp21)), rmask, other=0).to(tl.float32)
        tmp23 = tmp22.to(tl.float32)
        tmp24 = 4096.0
        tmp25 = tmp7 / tmp24
        tmp26 = 1e-05
        tmp27 = tmp25 + tmp26
        tmp28 = tl.math.rsqrt(tmp27)
        tmp29 = tmp23 * tmp28
        tmp30 = tmp29.to(tl.float32)
        tmp32 = tmp30 * tmp31
        tmp35 = tmp34 * tmp22
        tmp36 = 0.0
        tmp37 = tmp35 + tmp36
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp17 / tmp24
        tmp40 = tmp39 + tmp26
        tmp41 = tl.math.rsqrt(tmp40)
        tmp42 = tmp38 * tmp41
        tmp43 = tmp42.to(tl.float32)
        tmp45 = tmp43 * tmp44
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp32, rmask)
        tl.store(out_ptr3 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp45, rmask)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_mengqy/mr/cmrs5lgx6ezeofjhdl5g5z4htt6ylbjlibtheukqcpcyj2pllr56.py
# Source Nodes: [stack, stack_1, stack_2, stack_3], Original ATen: [aten.stack]
# stack => cat
# stack_1 => cat_1
# stack_2 => cat_2
# stack_3 => cat_3
triton_poi_fused_stack_1 = async_compile.triton('triton_poi_fused_stack_1', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i32', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_1', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 8, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]})
@triton.jit
def triton_poi_fused_stack_1(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, XBLOCK : tl.constexpr):
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
    tmp27 = tl.load(in_ptr3 + (2*x2), None).to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (1 + (2*x2)), None).to(tl.float32)
    tmp37 = tl.load(in_ptr3 + (4096 + (2*x2)), None).to(tl.float32)
    tmp40 = tl.load(in_ptr3 + (4097 + (2*x2)), None).to(tl.float32)
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
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28 * tmp6
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp31 * tmp11
    tmp33 = tmp29 - tmp32
    tmp34 = tmp31 * tmp6
    tmp35 = tmp28 * tmp11
    tmp36 = tmp34 + tmp35
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp38 * tmp6
    tmp41 = tmp40.to(tl.float32)
    tmp42 = tmp41 * tmp11
    tmp43 = tmp39 - tmp42
    tmp44 = tmp41 * tmp6
    tmp45 = tmp38 * tmp11
    tmp46 = tmp44 + tmp45
    tl.store(out_ptr0 + (2*x2), tmp13, None)
    tl.store(out_ptr1 + (2*x2), tmp16, None)
    tl.store(out_ptr2 + (2*x2), tmp23, None)
    tl.store(out_ptr3 + (2*x2), tmp26, None)
    tl.store(out_ptr4 + (2*x2), tmp33, None)
    tl.store(out_ptr5 + (2*x2), tmp36, None)
    tl.store(out_ptr6 + (2*x2), tmp43, None)
    tl.store(out_ptr7 + (2*x2), tmp46, None)
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


# kernel path: /tmp/torchinductor_mengqy/6o/c6oqxdj7vjtdit5rq3dz6ripwl7otcjsmm7oykk7bakprum5jel2.py
# Source Nodes: [setitem, setitem_1, setitem_2, setitem_3], Original ATen: [aten.index_put]
# setitem => index_put
# setitem_1 => index_put_1
# setitem_2 => index_put_2
# setitem_3 => index_put_3
triton_poi_fused_index_put_3 = async_compile.triton('triton_poi_fused_index_put_3', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[4096], filename=__file__, meta={'signature': {0: '*i32', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['out_ptr0', 'out_ptr1', 'out_ptr2', 'out_ptr3'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_index_put_3', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]})
@triton.jit
def triton_poi_fused_index_put_3(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp6 = tl.load(in_ptr3 + (x2), None)
    tmp8 = tl.load(in_ptr4 + (8192 + x2), None).to(tl.float32)
    tmp2 = tl.where(tmp1 < 0, tmp1 + 1152, tmp1)
    tl.device_assert((0 <= tmp2) & (tmp2 < 1152), "index out of bounds: 0 <= tmp2 < 1152")
    tmp4 = tmp3.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tl.store(out_ptr0 + (x0 + (128*tmp2) + (147456*x1)), tmp4, None)
    tl.store(out_ptr1 + (x0 + (128*tmp2) + (147456*x1)), tmp5, None)
    tl.store(out_ptr2 + (x0 + (128*tmp2) + (147456*x1)), tmp7, None)
    tl.store(out_ptr3 + (x0 + (128*tmp2) + (147456*x1)), tmp8, None)
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


# kernel path: /tmp/torchinductor_mengqy/fd/cfdv4ea2onmaacnb244qclquhxyxc4hw4plry5wg546ub5liuvpd.py
# Source Nodes: [getitem, mul_11, mul_28, softmax, softmax_1, where, where_1], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
# getitem => index
# mul_11 => mul_11
# mul_28 => mul_29
# softmax => amax, convert_element_type_6, convert_element_type_7, div, exp, sub_2, sum_1
# softmax_1 => amax_1, convert_element_type_18, convert_element_type_19, div_1, exp_1, sub_5, sum_2
# where => full_default, where
# where_1 => full_default_1, where_1
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
    meta={'signature': {0: '*i32', 1: '*i1', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_index_mul_scalar_tensor_where_6', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]}
)
@triton.jit
def triton_red_fused__softmax_index_mul_scalar_tensor_where_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp33 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp17 = tl.load(in_ptr2 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp28 = tl.load(in_ptr3 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
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
        tmp29 = tmp28 * tmp18
        tmp30 = tl.where(tmp16, tmp29, tmp20)
        tmp31 = tmp30.to(tl.float32)
        tmp32 = tl.broadcast_to(tmp31, [XBLOCK, RBLOCK])
        tmp34 = triton_helpers.maximum(_tmp33, tmp32)
        _tmp33 = tl.where(rmask & xmask, tmp34, _tmp33)
    tmp26 = tl.sum(_tmp26, 1)[:, None]
    tmp33 = triton_helpers.max2(_tmp33, 1)[:, None]
    tmp35 = tl.load(in_ptr0 + (0))
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
    _tmp48 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp39 = tl.load(in_ptr3 + (r1 + (1152*x0)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp50 = tl.load(in_ptr2 + (r1 + (1152*x0)), rmask & xmask, other=0).to(tl.float32)
        tmp37 = tl.where(tmp36 < 0, tmp36 + 1152, tmp36)
        tl.device_assert((0 <= tmp37) & (tmp37 < 1152), "index out of bounds: 0 <= tmp37 < 1152")
        tmp38 = tl.load(in_ptr1 + (r1 + (1152*tmp37)), rmask, eviction_policy='evict_last')
        tmp40 = 0.08838834764831843
        tmp41 = tmp39 * tmp40
        tmp42 = -65504.0
        tmp43 = tl.where(tmp38, tmp41, tmp42)
        tmp44 = tmp43.to(tl.float32)
        tmp45 = tmp44 - tmp33
        tmp46 = tl.exp(tmp45)
        tmp47 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
        tmp49 = _tmp48 + tmp47
        _tmp48 = tl.where(rmask & xmask, tmp49, _tmp48)
        tmp51 = tmp50 * tmp40
        tmp52 = tl.where(tmp38, tmp51, tmp42)
        tmp53 = tmp52.to(tl.float32)
        tmp54 = tmp53 - tmp11
        tmp55 = tl.exp(tmp54)
        tmp56 = tmp55 / tmp26
        tmp57 = tmp56.to(tl.float32)
        tl.store(out_ptr4 + (r1 + (1152*x0)), tmp57, rmask & xmask)
    tmp48 = tl.sum(_tmp48, 1)[:, None]
    tmp58 = tl.load(in_ptr0 + (0))
    tmp59 = tl.broadcast_to(tmp58, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp62 = tl.load(in_ptr3 + (r1 + (1152*x0)), rmask & xmask, other=0).to(tl.float32)
        tmp60 = tl.where(tmp59 < 0, tmp59 + 1152, tmp59)
        tl.device_assert((0 <= tmp60) & (tmp60 < 1152), "index out of bounds: 0 <= tmp60 < 1152")
        tmp61 = tl.load(in_ptr1 + (r1 + (1152*tmp60)), rmask)
        tmp63 = 0.08838834764831843
        tmp64 = tmp62 * tmp63
        tmp65 = -65504.0
        tmp66 = tl.where(tmp61, tmp64, tmp65)
        tmp67 = tmp66.to(tl.float32)
        tmp68 = tmp67 - tmp33
        tmp69 = tl.exp(tmp68)
        tmp70 = tmp69 / tmp48
        tmp71 = tmp70.to(tl.float32)
        tl.store(out_ptr5 + (r1 + (1152*x0)), tmp71, rmask & xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/7a/c7aqfft7vfclnsxbe5floqduocjckahxzvgfq5yqiu3pcpnibxos.py
# Source Nodes: [add_10, add_3, add_4, add_6, float_4, float_8, l__model___tok_embeddings, mean_1, mean_3, mul_12, mul_13, mul_14, mul_16, mul_29, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.rsqrt]
# add_10 => add_10
# add_3 => add_3
# add_4 => add_4
# add_6 => add_6
# float_4 => convert_element_type_8
# float_8 => convert_element_type_20
# l__model___tok_embeddings => embedding
# mean_1 => mean_1
# mean_3 => mean_3
# mul_12 => mul_12
# mul_13 => mul_13
# mul_14 => mul_14
# mul_16 => mul_17
# mul_29 => mul_30
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
    meta={'signature': {0: '*i32', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp32', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_7', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_7(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0 = tl.load(in_ptr0 + (0))
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    _tmp9 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    tmp11 = tl.load(in_ptr3 + (0)).to(tl.float32)
    tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp4 = tl.load(in_ptr2 + (r0), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp16 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp2 = tl.where(tmp1 < 0, tmp1 + 32000, tmp1)
        tl.device_assert((0 <= tmp2) & (tmp2 < 32000), "index out of bounds: 0 <= tmp2 < 32000")
        tmp3 = tl.load(in_ptr1 + (r0 + (4096*tmp2)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp7 = tmp6 * tmp6
        tmp8 = tl.broadcast_to(tmp7, [XBLOCK, RBLOCK])
        tmp10 = _tmp9 + tmp8
        _tmp9 = tl.where(rmask, tmp10, _tmp9)
        tmp13 = tmp12 * tmp3
        tmp14 = 0.0
        tmp15 = tmp13 + tmp14
        tmp17 = tmp15 + tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask, tmp22, _tmp21)
    tmp9 = tl.sum(_tmp9, 1)[:, None]
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr1 + (tl.full([XBLOCK, 1], 0, tl.int32)), tmp21, None)
    tmp23 = tl.load(in_ptr0 + (0))
    tmp24 = tl.broadcast_to(tmp23, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp27 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp37 = tl.load(in_ptr5 + (r0), rmask, other=0).to(tl.float32)
        tmp25 = tl.where(tmp24 < 0, tmp24 + 32000, tmp24)
        tl.device_assert((0 <= tmp25) & (tmp25 < 32000), "index out of bounds: 0 <= tmp25 < 32000")
        tmp26 = tl.load(in_ptr1 + (r0 + (4096*tmp25)), rmask, other=0).to(tl.float32)
        tmp28 = tmp26 + tmp27
        tmp29 = tmp28.to(tl.float32)
        tmp30 = 4096.0
        tmp31 = tmp9 / tmp30
        tmp32 = 1e-05
        tmp33 = tmp31 + tmp32
        tmp34 = tl.math.rsqrt(tmp33)
        tmp35 = tmp29 * tmp34
        tmp36 = tmp35.to(tl.float32)
        tmp38 = tmp36 * tmp37
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp38, rmask)
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


# kernel path: /tmp/torchinductor_mengqy/s6/cs66tcgi3mqvdbqe5kbzft7vzlhfnpm4p6wbonufjekbljc7osz7.py
# Source Nodes: [add_10, add_11, add_13, add_14, add_15, add_3, add_5, add_6, float_8, float_9, l__model___tok_embeddings, mean_3, mean_4, mul_16, mul_29, mul_30, mul_31, mul_33, mul_34, mul_35, mul_36, mul_37, rsqrt_3, rsqrt_4, type_as_7, type_as_8], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.rsqrt]
# add_10 => add_10
# add_11 => add_11
# add_13 => add_13
# add_14 => add_14
# add_15 => add_15
# add_3 => add_3
# add_5 => add_5
# add_6 => add_6
# float_8 => convert_element_type_20
# float_9 => convert_element_type_24
# l__model___tok_embeddings => embedding
# mean_3 => mean_3
# mean_4 => mean_4
# mul_16 => mul_17
# mul_29 => mul_30
# mul_30 => mul_31
# mul_31 => mul_32
# mul_33 => mul_35
# mul_34 => mul_36
# mul_35 => mul_37
# mul_36 => mul_38
# mul_37 => mul_39
# rsqrt_3 => rsqrt_3
# rsqrt_4 => rsqrt_4
# type_as_7 => convert_element_type_21
# type_as_8 => convert_element_type_25
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
    meta={'signature': {0: '*fp16', 1: '*i32', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_9', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(14,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_9(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0 = tl.load(in_ptr0 + (0)).to(tl.float32)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp9 = tl.load(in_ptr0 + (1)).to(tl.float32)
    tmp10 = tl.broadcast_to(tmp9, [XBLOCK, RBLOCK])
    tmp17 = tl.load(in_ptr5 + (0)).to(tl.float32)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp24 = tl.load(in_ptr7 + (0))
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    _tmp38 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp11 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp21 = tl.load(in_ptr6 + (r0), rmask, other=0).to(tl.float32)
        tmp33 = tl.load(in_ptr8 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.where(tmp3 < 0, tmp3 + 32000, tmp3)
        tl.device_assert((0 <= tmp4) & (tmp4 < 32000), "index out of bounds: 0 <= tmp4 < 32000")
        tmp5 = tl.load(in_ptr2 + (r0 + (4096*tmp4)), rmask, other=0).to(tl.float32)
        tmp6 = tmp1 * tmp5
        tmp7 = 0.0
        tmp8 = tmp6 + tmp7
        tmp12 = tmp5 + tmp11
        tmp14 = tmp12 + tmp13
        tmp15 = tmp10 * tmp14
        tmp16 = tmp8 + tmp15
        tmp19 = tmp18 * tmp5
        tmp20 = tmp19 + tmp7
        tmp22 = tmp20 + tmp21
        tmp23 = tmp22.to(tl.float32)
        tmp26 = 4096.0
        tmp27 = tmp25 / tmp26
        tmp28 = 1e-05
        tmp29 = tmp27 + tmp28
        tmp30 = tl.math.rsqrt(tmp29)
        tmp31 = tmp23 * tmp30
        tmp32 = tmp31.to(tl.float32)
        tmp34 = tmp32 * tmp33
        tmp35 = tmp16.to(tl.float32)
        tmp36 = tmp35 * tmp35
        tmp37 = tl.broadcast_to(tmp36, [XBLOCK, RBLOCK])
        tmp39 = _tmp38 + tmp37
        _tmp38 = tl.where(rmask, tmp39, _tmp38)
        tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp16, rmask)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp34, rmask)
    tmp38 = tl.sum(_tmp38, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp40 = tl.load(out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp49 = tl.load(in_ptr9 + (r0), rmask, other=0).to(tl.float32)
        tmp41 = tmp40.to(tl.float32)
        tmp42 = 4096.0
        tmp43 = tmp38 / tmp42
        tmp44 = 1e-05
        tmp45 = tmp43 + tmp44
        tmp46 = tl.math.rsqrt(tmp45)
        tmp47 = tmp41 * tmp46
        tmp48 = tmp47.to(tl.float32)
        tmp50 = tmp48 * tmp49
        tl.store(out_ptr3 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp50, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/lg/clgpduunzueyaffeu3jqq2penizpbsw5uszfjtgoavs5rddabu2t.py
# Source Nodes: [add_10, add_105, add_106, add_107, add_12, add_121, add_122, add_123, add_138, add_139, add_140, add_156, add_157, add_158, add_175, add_176, add_177, add_195, add_196, add_197, add_21, add_216, add_217, add_218, add_22, add_23, add_238, add_239, add_24, add_240, add_261, add_262, add_263, add_285, add_286, add_287, add_3, add_30, add_31, add_310, add_311, add_312, add_32, add_336, add_337, add_338, add_363, add_364, add_365, add_391, add_392, add_393, add_40, add_41, add_42, add_420, add_421, add_422, add_450, add_451, add_452, add_481, add_482, add_483, add_5, add_51, add_513, add_514, add_515, add_52, add_53, add_546, add_547, add_548, add_580, add_581, add_582, add_6, add_615, add_616, add_617, add_63, add_64, add_65, add_651, add_652, add_653, add_688, add_689, add_690, add_76, add_77, add_78, add_90, add_91, add_92, float_13, l__model___tok_embeddings, mean_6, mul_1008, mul_1009, mul_1010, mul_111, mul_112, mul_113, mul_133, mul_134, mul_135, mul_156, mul_157, mul_158, mul_16, mul_180, mul_181, mul_182, mul_205, mul_206, mul_207, mul_231, mul_232, mul_233, mul_258, mul_259, mul_260, mul_286, mul_287, mul_288, mul_315, mul_316, mul_317, mul_345, mul_346, mul_347, mul_376, mul_377, mul_378, mul_408, mul_409, mul_410, mul_441, mul_442, mul_443, mul_475, mul_476, mul_477, mul_51, mul_510, mul_511, mul_512, mul_52, mul_53, mul_54, mul_546, mul_547, mul_548, mul_55, mul_56, mul_583, mul_584, mul_585, mul_621, mul_622, mul_623, mul_660, mul_661, mul_662, mul_70, mul_700, mul_701, mul_702, mul_71, mul_72, mul_741, mul_742, mul_743, mul_783, mul_784, mul_785, mul_826, mul_827, mul_828, mul_870, mul_871, mul_872, mul_90, mul_91, mul_915, mul_916, mul_917, mul_92, mul_961, mul_962, mul_963, rsqrt_6, type_as_12], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.rsqrt]
# add_10 => add_10
# add_105 => add_105
# add_106 => add_106
# add_107 => add_107
# add_12 => add_12
# add_121 => add_121
# add_122 => add_122
# add_123 => add_123
# add_138 => add_138
# add_139 => add_139
# add_140 => add_140
# add_156 => add_156
# add_157 => add_157
# add_158 => add_158
# add_175 => add_175
# add_176 => add_176
# add_177 => add_177
# add_195 => add_195
# add_196 => add_196
# add_197 => add_197
# add_21 => add_21
# add_216 => add_216
# add_217 => add_217
# add_218 => add_218
# add_22 => add_22
# add_23 => add_23
# add_238 => add_238
# add_239 => add_239
# add_24 => add_24
# add_240 => add_240
# add_261 => add_261
# add_262 => add_262
# add_263 => add_263
# add_285 => add_285
# add_286 => add_286
# add_287 => add_287
# add_3 => add_3
# add_30 => add_30
# add_31 => add_31
# add_310 => add_310
# add_311 => add_311
# add_312 => add_312
# add_32 => add_32
# add_336 => add_336
# add_337 => add_337
# add_338 => add_338
# add_363 => add_363
# add_364 => add_364
# add_365 => add_365
# add_391 => add_391
# add_392 => add_392
# add_393 => add_393
# add_40 => add_40
# add_41 => add_41
# add_42 => add_42
# add_420 => add_420
# add_421 => add_421
# add_422 => add_422
# add_450 => add_450
# add_451 => add_451
# add_452 => add_452
# add_481 => add_481
# add_482 => add_482
# add_483 => add_483
# add_5 => add_5
# add_51 => add_51
# add_513 => add_513
# add_514 => add_514
# add_515 => add_515
# add_52 => add_52
# add_53 => add_53
# add_546 => add_546
# add_547 => add_547
# add_548 => add_548
# add_580 => add_580
# add_581 => add_581
# add_582 => add_582
# add_6 => add_6
# add_615 => add_615
# add_616 => add_616
# add_617 => add_617
# add_63 => add_63
# add_64 => add_64
# add_65 => add_65
# add_651 => add_651
# add_652 => add_652
# add_653 => add_653
# add_688 => add_688
# add_689 => add_689
# add_690 => add_690
# add_76 => add_76
# add_77 => add_77
# add_78 => add_78
# add_90 => add_90
# add_91 => add_91
# add_92 => add_92
# float_13 => convert_element_type_36
# l__model___tok_embeddings => embedding
# mean_6 => mean_6
# mul_1008 => mul_1040
# mul_1009 => mul_1041
# mul_1010 => mul_1042
# mul_111 => mul_117
# mul_112 => mul_118
# mul_113 => mul_119
# mul_133 => mul_140
# mul_134 => mul_141
# mul_135 => mul_142
# mul_156 => mul_164
# mul_157 => mul_165
# mul_158 => mul_166
# mul_16 => mul_17
# mul_180 => mul_189
# mul_181 => mul_190
# mul_182 => mul_191
# mul_205 => mul_215
# mul_206 => mul_216
# mul_207 => mul_217
# mul_231 => mul_242
# mul_232 => mul_243
# mul_233 => mul_244
# mul_258 => mul_270
# mul_259 => mul_271
# mul_260 => mul_272
# mul_286 => mul_299
# mul_287 => mul_300
# mul_288 => mul_301
# mul_315 => mul_329
# mul_316 => mul_330
# mul_317 => mul_331
# mul_345 => mul_360
# mul_346 => mul_361
# mul_347 => mul_362
# mul_376 => mul_392
# mul_377 => mul_393
# mul_378 => mul_394
# mul_408 => mul_425
# mul_409 => mul_426
# mul_410 => mul_427
# mul_441 => mul_459
# mul_442 => mul_460
# mul_443 => mul_461
# mul_475 => mul_494
# mul_476 => mul_495
# mul_477 => mul_496
# mul_51 => mul_54
# mul_510 => mul_530
# mul_511 => mul_531
# mul_512 => mul_532
# mul_52 => mul_55
# mul_53 => mul_56
# mul_54 => mul_57
# mul_546 => mul_567
# mul_547 => mul_568
# mul_548 => mul_569
# mul_55 => mul_58
# mul_56 => mul_59
# mul_583 => mul_605
# mul_584 => mul_606
# mul_585 => mul_607
# mul_621 => mul_644
# mul_622 => mul_645
# mul_623 => mul_646
# mul_660 => mul_684
# mul_661 => mul_685
# mul_662 => mul_686
# mul_70 => mul_74
# mul_700 => mul_725
# mul_701 => mul_726
# mul_702 => mul_727
# mul_71 => mul_75
# mul_72 => mul_76
# mul_741 => mul_767
# mul_742 => mul_768
# mul_743 => mul_769
# mul_783 => mul_810
# mul_784 => mul_811
# mul_785 => mul_812
# mul_826 => mul_854
# mul_827 => mul_855
# mul_828 => mul_856
# mul_870 => mul_899
# mul_871 => mul_900
# mul_872 => mul_901
# mul_90 => mul_95
# mul_91 => mul_96
# mul_915 => mul_945
# mul_916 => mul_946
# mul_917 => mul_947
# mul_92 => mul_97
# mul_961 => mul_992
# mul_962 => mul_993
# mul_963 => mul_994
# rsqrt_6 => rsqrt_6
# type_as_12 => convert_element_type_37
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*i32', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: '*fp16', 21: '*fp16', 22: '*fp16', 23: '*fp16', 24: '*fp16', 25: '*fp16', 26: '*fp16', 27: '*fp16', 28: '*fp16', 29: '*fp16', 30: '*fp16', 31: '*fp16', 32: '*fp16', 33: '*fp16', 34: '*fp16', 35: '*fp16', 36: '*fp16', 37: '*fp16', 38: '*fp16', 39: '*fp16', 40: '*fp16', 41: '*fp16', 42: '*fp16', 43: '*fp16', 44: '*fp16', 45: '*fp16', 46: '*fp16', 47: '*fp16', 48: '*fp16', 49: '*fp16', 50: '*fp16', 51: '*fp16', 52: '*fp16', 53: '*fp16', 54: '*fp16', 55: '*fp16', 56: '*fp16', 57: '*fp16', 58: '*fp16', 59: '*fp16', 60: '*fp16', 61: '*fp16', 62: '*fp16', 63: '*fp16', 64: '*fp16', 65: '*fp16', 66: '*fp16', 67: '*fp16', 68: '*fp16', 69: 'i32', 70: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_10', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 70), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(70,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_10(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, out_ptr18, out_ptr19, out_ptr20, out_ptr21, out_ptr22, out_ptr23, out_ptr24, out_ptr25, out_ptr26, out_ptr27, out_ptr28, out_ptr29, out_ptr31, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp0 = tl.load(in_ptr0 + (0)).to(tl.float32)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
    tmp13 = tl.load(in_ptr4 + (0)).to(tl.float32)
    tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
    tmp17 = tl.load(in_ptr4 + (1)).to(tl.float32)
    tmp18 = tl.broadcast_to(tmp17, [XBLOCK, RBLOCK])
    tmp25 = tl.load(in_ptr4 + (2)).to(tl.float32)
    tmp26 = tl.broadcast_to(tmp25, [XBLOCK, RBLOCK])
    tmp29 = tl.load(in_ptr7 + (0)).to(tl.float32)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    tmp33 = tl.load(in_ptr7 + (1)).to(tl.float32)
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
    tmp37 = tl.load(in_ptr7 + (2)).to(tl.float32)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
    tmp41 = tl.load(in_ptr8 + (0)).to(tl.float32)
    tmp42 = tl.broadcast_to(tmp41, [XBLOCK, RBLOCK])
    tmp45 = tl.load(in_ptr8 + (1)).to(tl.float32)
    tmp46 = tl.broadcast_to(tmp45, [XBLOCK, RBLOCK])
    tmp49 = tl.load(in_ptr8 + (2)).to(tl.float32)
    tmp50 = tl.broadcast_to(tmp49, [XBLOCK, RBLOCK])
    tmp53 = tl.load(in_ptr9 + (0)).to(tl.float32)
    tmp54 = tl.broadcast_to(tmp53, [XBLOCK, RBLOCK])
    tmp57 = tl.load(in_ptr9 + (1)).to(tl.float32)
    tmp58 = tl.broadcast_to(tmp57, [XBLOCK, RBLOCK])
    tmp61 = tl.load(in_ptr9 + (2)).to(tl.float32)
    tmp62 = tl.broadcast_to(tmp61, [XBLOCK, RBLOCK])
    tmp65 = tl.load(in_ptr10 + (0)).to(tl.float32)
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK, RBLOCK])
    tmp69 = tl.load(in_ptr10 + (1)).to(tl.float32)
    tmp70 = tl.broadcast_to(tmp69, [XBLOCK, RBLOCK])
    tmp73 = tl.load(in_ptr10 + (2)).to(tl.float32)
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK, RBLOCK])
    tmp77 = tl.load(in_ptr11 + (0)).to(tl.float32)
    tmp78 = tl.broadcast_to(tmp77, [XBLOCK, RBLOCK])
    tmp81 = tl.load(in_ptr11 + (1)).to(tl.float32)
    tmp82 = tl.broadcast_to(tmp81, [XBLOCK, RBLOCK])
    tmp85 = tl.load(in_ptr11 + (2)).to(tl.float32)
    tmp86 = tl.broadcast_to(tmp85, [XBLOCK, RBLOCK])
    tmp89 = tl.load(in_ptr12 + (0)).to(tl.float32)
    tmp90 = tl.broadcast_to(tmp89, [XBLOCK, RBLOCK])
    tmp93 = tl.load(in_ptr12 + (1)).to(tl.float32)
    tmp94 = tl.broadcast_to(tmp93, [XBLOCK, RBLOCK])
    tmp97 = tl.load(in_ptr12 + (2)).to(tl.float32)
    tmp98 = tl.broadcast_to(tmp97, [XBLOCK, RBLOCK])
    tmp101 = tl.load(in_ptr13 + (0)).to(tl.float32)
    tmp102 = tl.broadcast_to(tmp101, [XBLOCK, RBLOCK])
    tmp105 = tl.load(in_ptr13 + (1)).to(tl.float32)
    tmp106 = tl.broadcast_to(tmp105, [XBLOCK, RBLOCK])
    tmp109 = tl.load(in_ptr13 + (2)).to(tl.float32)
    tmp110 = tl.broadcast_to(tmp109, [XBLOCK, RBLOCK])
    tmp113 = tl.load(in_ptr14 + (0)).to(tl.float32)
    tmp114 = tl.broadcast_to(tmp113, [XBLOCK, RBLOCK])
    tmp117 = tl.load(in_ptr14 + (1)).to(tl.float32)
    tmp118 = tl.broadcast_to(tmp117, [XBLOCK, RBLOCK])
    tmp121 = tl.load(in_ptr14 + (2)).to(tl.float32)
    tmp122 = tl.broadcast_to(tmp121, [XBLOCK, RBLOCK])
    tmp125 = tl.load(in_ptr15 + (0)).to(tl.float32)
    tmp126 = tl.broadcast_to(tmp125, [XBLOCK, RBLOCK])
    tmp129 = tl.load(in_ptr15 + (1)).to(tl.float32)
    tmp130 = tl.broadcast_to(tmp129, [XBLOCK, RBLOCK])
    tmp133 = tl.load(in_ptr15 + (2)).to(tl.float32)
    tmp134 = tl.broadcast_to(tmp133, [XBLOCK, RBLOCK])
    tmp137 = tl.load(in_ptr16 + (0)).to(tl.float32)
    tmp138 = tl.broadcast_to(tmp137, [XBLOCK, RBLOCK])
    tmp141 = tl.load(in_ptr16 + (1)).to(tl.float32)
    tmp142 = tl.broadcast_to(tmp141, [XBLOCK, RBLOCK])
    tmp145 = tl.load(in_ptr16 + (2)).to(tl.float32)
    tmp146 = tl.broadcast_to(tmp145, [XBLOCK, RBLOCK])
    tmp149 = tl.load(in_ptr17 + (0)).to(tl.float32)
    tmp150 = tl.broadcast_to(tmp149, [XBLOCK, RBLOCK])
    tmp153 = tl.load(in_ptr17 + (1)).to(tl.float32)
    tmp154 = tl.broadcast_to(tmp153, [XBLOCK, RBLOCK])
    tmp157 = tl.load(in_ptr17 + (2)).to(tl.float32)
    tmp158 = tl.broadcast_to(tmp157, [XBLOCK, RBLOCK])
    tmp161 = tl.load(in_ptr18 + (0)).to(tl.float32)
    tmp162 = tl.broadcast_to(tmp161, [XBLOCK, RBLOCK])
    tmp165 = tl.load(in_ptr18 + (1)).to(tl.float32)
    tmp166 = tl.broadcast_to(tmp165, [XBLOCK, RBLOCK])
    tmp169 = tl.load(in_ptr18 + (2)).to(tl.float32)
    tmp170 = tl.broadcast_to(tmp169, [XBLOCK, RBLOCK])
    tmp173 = tl.load(in_ptr19 + (0)).to(tl.float32)
    tmp174 = tl.broadcast_to(tmp173, [XBLOCK, RBLOCK])
    tmp177 = tl.load(in_ptr19 + (1)).to(tl.float32)
    tmp178 = tl.broadcast_to(tmp177, [XBLOCK, RBLOCK])
    tmp181 = tl.load(in_ptr19 + (2)).to(tl.float32)
    tmp182 = tl.broadcast_to(tmp181, [XBLOCK, RBLOCK])
    tmp185 = tl.load(in_ptr20 + (0)).to(tl.float32)
    tmp186 = tl.broadcast_to(tmp185, [XBLOCK, RBLOCK])
    tmp189 = tl.load(in_ptr20 + (1)).to(tl.float32)
    tmp190 = tl.broadcast_to(tmp189, [XBLOCK, RBLOCK])
    tmp193 = tl.load(in_ptr20 + (2)).to(tl.float32)
    tmp194 = tl.broadcast_to(tmp193, [XBLOCK, RBLOCK])
    tmp197 = tl.load(in_ptr21 + (0)).to(tl.float32)
    tmp198 = tl.broadcast_to(tmp197, [XBLOCK, RBLOCK])
    tmp201 = tl.load(in_ptr21 + (1)).to(tl.float32)
    tmp202 = tl.broadcast_to(tmp201, [XBLOCK, RBLOCK])
    tmp205 = tl.load(in_ptr21 + (2)).to(tl.float32)
    tmp206 = tl.broadcast_to(tmp205, [XBLOCK, RBLOCK])
    tmp209 = tl.load(in_ptr22 + (0)).to(tl.float32)
    tmp210 = tl.broadcast_to(tmp209, [XBLOCK, RBLOCK])
    tmp213 = tl.load(in_ptr22 + (1)).to(tl.float32)
    tmp214 = tl.broadcast_to(tmp213, [XBLOCK, RBLOCK])
    tmp217 = tl.load(in_ptr22 + (2)).to(tl.float32)
    tmp218 = tl.broadcast_to(tmp217, [XBLOCK, RBLOCK])
    tmp221 = tl.load(in_ptr23 + (0)).to(tl.float32)
    tmp222 = tl.broadcast_to(tmp221, [XBLOCK, RBLOCK])
    tmp225 = tl.load(in_ptr23 + (1)).to(tl.float32)
    tmp226 = tl.broadcast_to(tmp225, [XBLOCK, RBLOCK])
    tmp229 = tl.load(in_ptr23 + (2)).to(tl.float32)
    tmp230 = tl.broadcast_to(tmp229, [XBLOCK, RBLOCK])
    tmp233 = tl.load(in_ptr24 + (0)).to(tl.float32)
    tmp234 = tl.broadcast_to(tmp233, [XBLOCK, RBLOCK])
    tmp237 = tl.load(in_ptr24 + (1)).to(tl.float32)
    tmp238 = tl.broadcast_to(tmp237, [XBLOCK, RBLOCK])
    tmp241 = tl.load(in_ptr24 + (2)).to(tl.float32)
    tmp242 = tl.broadcast_to(tmp241, [XBLOCK, RBLOCK])
    tmp245 = tl.load(in_ptr25 + (0)).to(tl.float32)
    tmp246 = tl.broadcast_to(tmp245, [XBLOCK, RBLOCK])
    tmp249 = tl.load(in_ptr25 + (1)).to(tl.float32)
    tmp250 = tl.broadcast_to(tmp249, [XBLOCK, RBLOCK])
    tmp253 = tl.load(in_ptr25 + (2)).to(tl.float32)
    tmp254 = tl.broadcast_to(tmp253, [XBLOCK, RBLOCK])
    tmp257 = tl.load(in_ptr26 + (0)).to(tl.float32)
    tmp258 = tl.broadcast_to(tmp257, [XBLOCK, RBLOCK])
    tmp261 = tl.load(in_ptr26 + (1)).to(tl.float32)
    tmp262 = tl.broadcast_to(tmp261, [XBLOCK, RBLOCK])
    tmp265 = tl.load(in_ptr26 + (2)).to(tl.float32)
    tmp266 = tl.broadcast_to(tmp265, [XBLOCK, RBLOCK])
    tmp269 = tl.load(in_ptr27 + (0)).to(tl.float32)
    tmp270 = tl.broadcast_to(tmp269, [XBLOCK, RBLOCK])
    tmp273 = tl.load(in_ptr27 + (1)).to(tl.float32)
    tmp274 = tl.broadcast_to(tmp273, [XBLOCK, RBLOCK])
    tmp277 = tl.load(in_ptr27 + (2)).to(tl.float32)
    tmp278 = tl.broadcast_to(tmp277, [XBLOCK, RBLOCK])
    tmp281 = tl.load(in_ptr28 + (0)).to(tl.float32)
    tmp282 = tl.broadcast_to(tmp281, [XBLOCK, RBLOCK])
    tmp285 = tl.load(in_ptr28 + (1)).to(tl.float32)
    tmp286 = tl.broadcast_to(tmp285, [XBLOCK, RBLOCK])
    tmp289 = tl.load(in_ptr28 + (2)).to(tl.float32)
    tmp290 = tl.broadcast_to(tmp289, [XBLOCK, RBLOCK])
    tmp293 = tl.load(in_ptr29 + (0)).to(tl.float32)
    tmp294 = tl.broadcast_to(tmp293, [XBLOCK, RBLOCK])
    tmp297 = tl.load(in_ptr29 + (1)).to(tl.float32)
    tmp298 = tl.broadcast_to(tmp297, [XBLOCK, RBLOCK])
    tmp301 = tl.load(in_ptr29 + (2)).to(tl.float32)
    tmp302 = tl.broadcast_to(tmp301, [XBLOCK, RBLOCK])
    tmp305 = tl.load(in_ptr30 + (0)).to(tl.float32)
    tmp306 = tl.broadcast_to(tmp305, [XBLOCK, RBLOCK])
    tmp309 = tl.load(in_ptr30 + (1)).to(tl.float32)
    tmp310 = tl.broadcast_to(tmp309, [XBLOCK, RBLOCK])
    tmp313 = tl.load(in_ptr30 + (2)).to(tl.float32)
    tmp314 = tl.broadcast_to(tmp313, [XBLOCK, RBLOCK])
    tmp317 = tl.load(in_ptr31 + (0)).to(tl.float32)
    tmp318 = tl.broadcast_to(tmp317, [XBLOCK, RBLOCK])
    tmp321 = tl.load(in_ptr31 + (1)).to(tl.float32)
    tmp322 = tl.broadcast_to(tmp321, [XBLOCK, RBLOCK])
    tmp325 = tl.load(in_ptr31 + (2)).to(tl.float32)
    tmp326 = tl.broadcast_to(tmp325, [XBLOCK, RBLOCK])
    tmp329 = tl.load(in_ptr32 + (0)).to(tl.float32)
    tmp330 = tl.broadcast_to(tmp329, [XBLOCK, RBLOCK])
    tmp333 = tl.load(in_ptr32 + (1)).to(tl.float32)
    tmp334 = tl.broadcast_to(tmp333, [XBLOCK, RBLOCK])
    tmp337 = tl.load(in_ptr32 + (2)).to(tl.float32)
    tmp338 = tl.broadcast_to(tmp337, [XBLOCK, RBLOCK])
    tmp341 = tl.load(in_ptr33 + (0)).to(tl.float32)
    tmp342 = tl.broadcast_to(tmp341, [XBLOCK, RBLOCK])
    tmp345 = tl.load(in_ptr33 + (1)).to(tl.float32)
    tmp346 = tl.broadcast_to(tmp345, [XBLOCK, RBLOCK])
    tmp349 = tl.load(in_ptr33 + (2)).to(tl.float32)
    tmp350 = tl.broadcast_to(tmp349, [XBLOCK, RBLOCK])
    tmp353 = tl.load(in_ptr34 + (0)).to(tl.float32)
    tmp354 = tl.broadcast_to(tmp353, [XBLOCK, RBLOCK])
    tmp357 = tl.load(in_ptr34 + (1)).to(tl.float32)
    tmp358 = tl.broadcast_to(tmp357, [XBLOCK, RBLOCK])
    tmp361 = tl.load(in_ptr34 + (2)).to(tl.float32)
    tmp362 = tl.broadcast_to(tmp361, [XBLOCK, RBLOCK])
    tmp365 = tl.load(in_ptr35 + (0)).to(tl.float32)
    tmp366 = tl.broadcast_to(tmp365, [XBLOCK, RBLOCK])
    tmp369 = tl.load(in_ptr35 + (1)).to(tl.float32)
    tmp370 = tl.broadcast_to(tmp369, [XBLOCK, RBLOCK])
    tmp373 = tl.load(in_ptr35 + (2)).to(tl.float32)
    tmp374 = tl.broadcast_to(tmp373, [XBLOCK, RBLOCK])
    _tmp380 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp9 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp11 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_ptr5 + (r0), rmask, other=0).to(tl.float32)
        tmp21 = tl.load(in_ptr6 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.where(tmp3 < 0, tmp3 + 32000, tmp3)
        tl.device_assert((0 <= tmp4) & (tmp4 < 32000), "index out of bounds: 0 <= tmp4 < 32000")
        tmp5 = tl.load(in_ptr2 + (r0 + (4096*tmp4)), rmask, other=0).to(tl.float32)
        tmp6 = tmp1 * tmp5
        tmp7 = 0.0
        tmp8 = tmp6 + tmp7
        tmp10 = tmp8 + tmp9
        tmp12 = tmp10 + tmp11
        tmp15 = tmp14 * tmp5
        tmp16 = tmp15 + tmp7
        tmp20 = tmp5 + tmp19
        tmp22 = tmp20 + tmp21
        tmp23 = tmp18 * tmp22
        tmp24 = tmp16 + tmp23
        tmp27 = tmp26 * tmp12
        tmp28 = tmp24 + tmp27
        tmp31 = tmp30 * tmp5
        tmp32 = tmp31 + tmp7
        tmp35 = tmp34 * tmp22
        tmp36 = tmp32 + tmp35
        tmp39 = tmp38 * tmp12
        tmp40 = tmp36 + tmp39
        tmp43 = tmp42 * tmp5
        tmp44 = tmp43 + tmp7
        tmp47 = tmp46 * tmp22
        tmp48 = tmp44 + tmp47
        tmp51 = tmp50 * tmp12
        tmp52 = tmp48 + tmp51
        tmp55 = tmp54 * tmp5
        tmp56 = tmp55 + tmp7
        tmp59 = tmp58 * tmp22
        tmp60 = tmp56 + tmp59
        tmp63 = tmp62 * tmp12
        tmp64 = tmp60 + tmp63
        tmp67 = tmp66 * tmp5
        tmp68 = tmp67 + tmp7
        tmp71 = tmp70 * tmp22
        tmp72 = tmp68 + tmp71
        tmp75 = tmp74 * tmp12
        tmp76 = tmp72 + tmp75
        tmp79 = tmp78 * tmp5
        tmp80 = tmp79 + tmp7
        tmp83 = tmp82 * tmp22
        tmp84 = tmp80 + tmp83
        tmp87 = tmp86 * tmp12
        tmp88 = tmp84 + tmp87
        tmp91 = tmp90 * tmp5
        tmp92 = tmp91 + tmp7
        tmp95 = tmp94 * tmp22
        tmp96 = tmp92 + tmp95
        tmp99 = tmp98 * tmp12
        tmp100 = tmp96 + tmp99
        tmp103 = tmp102 * tmp5
        tmp104 = tmp103 + tmp7
        tmp107 = tmp106 * tmp22
        tmp108 = tmp104 + tmp107
        tmp111 = tmp110 * tmp12
        tmp112 = tmp108 + tmp111
        tmp115 = tmp114 * tmp5
        tmp116 = tmp115 + tmp7
        tmp119 = tmp118 * tmp22
        tmp120 = tmp116 + tmp119
        tmp123 = tmp122 * tmp12
        tmp124 = tmp120 + tmp123
        tmp127 = tmp126 * tmp5
        tmp128 = tmp127 + tmp7
        tmp131 = tmp130 * tmp22
        tmp132 = tmp128 + tmp131
        tmp135 = tmp134 * tmp12
        tmp136 = tmp132 + tmp135
        tmp139 = tmp138 * tmp5
        tmp140 = tmp139 + tmp7
        tmp143 = tmp142 * tmp22
        tmp144 = tmp140 + tmp143
        tmp147 = tmp146 * tmp12
        tmp148 = tmp144 + tmp147
        tmp151 = tmp150 * tmp5
        tmp152 = tmp151 + tmp7
        tmp155 = tmp154 * tmp22
        tmp156 = tmp152 + tmp155
        tmp159 = tmp158 * tmp12
        tmp160 = tmp156 + tmp159
        tmp163 = tmp162 * tmp5
        tmp164 = tmp163 + tmp7
        tmp167 = tmp166 * tmp22
        tmp168 = tmp164 + tmp167
        tmp171 = tmp170 * tmp12
        tmp172 = tmp168 + tmp171
        tmp175 = tmp174 * tmp5
        tmp176 = tmp175 + tmp7
        tmp179 = tmp178 * tmp22
        tmp180 = tmp176 + tmp179
        tmp183 = tmp182 * tmp12
        tmp184 = tmp180 + tmp183
        tmp187 = tmp186 * tmp5
        tmp188 = tmp187 + tmp7
        tmp191 = tmp190 * tmp22
        tmp192 = tmp188 + tmp191
        tmp195 = tmp194 * tmp12
        tmp196 = tmp192 + tmp195
        tmp199 = tmp198 * tmp5
        tmp200 = tmp199 + tmp7
        tmp203 = tmp202 * tmp22
        tmp204 = tmp200 + tmp203
        tmp207 = tmp206 * tmp12
        tmp208 = tmp204 + tmp207
        tmp211 = tmp210 * tmp5
        tmp212 = tmp211 + tmp7
        tmp215 = tmp214 * tmp22
        tmp216 = tmp212 + tmp215
        tmp219 = tmp218 * tmp12
        tmp220 = tmp216 + tmp219
        tmp223 = tmp222 * tmp5
        tmp224 = tmp223 + tmp7
        tmp227 = tmp226 * tmp22
        tmp228 = tmp224 + tmp227
        tmp231 = tmp230 * tmp12
        tmp232 = tmp228 + tmp231
        tmp235 = tmp234 * tmp5
        tmp236 = tmp235 + tmp7
        tmp239 = tmp238 * tmp22
        tmp240 = tmp236 + tmp239
        tmp243 = tmp242 * tmp12
        tmp244 = tmp240 + tmp243
        tmp247 = tmp246 * tmp5
        tmp248 = tmp247 + tmp7
        tmp251 = tmp250 * tmp22
        tmp252 = tmp248 + tmp251
        tmp255 = tmp254 * tmp12
        tmp256 = tmp252 + tmp255
        tmp259 = tmp258 * tmp5
        tmp260 = tmp259 + tmp7
        tmp263 = tmp262 * tmp22
        tmp264 = tmp260 + tmp263
        tmp267 = tmp266 * tmp12
        tmp268 = tmp264 + tmp267
        tmp271 = tmp270 * tmp5
        tmp272 = tmp271 + tmp7
        tmp275 = tmp274 * tmp22
        tmp276 = tmp272 + tmp275
        tmp279 = tmp278 * tmp12
        tmp280 = tmp276 + tmp279
        tmp283 = tmp282 * tmp5
        tmp284 = tmp283 + tmp7
        tmp287 = tmp286 * tmp22
        tmp288 = tmp284 + tmp287
        tmp291 = tmp290 * tmp12
        tmp292 = tmp288 + tmp291
        tmp295 = tmp294 * tmp5
        tmp296 = tmp295 + tmp7
        tmp299 = tmp298 * tmp22
        tmp300 = tmp296 + tmp299
        tmp303 = tmp302 * tmp12
        tmp304 = tmp300 + tmp303
        tmp307 = tmp306 * tmp5
        tmp308 = tmp307 + tmp7
        tmp311 = tmp310 * tmp22
        tmp312 = tmp308 + tmp311
        tmp315 = tmp314 * tmp12
        tmp316 = tmp312 + tmp315
        tmp319 = tmp318 * tmp5
        tmp320 = tmp319 + tmp7
        tmp323 = tmp322 * tmp22
        tmp324 = tmp320 + tmp323
        tmp327 = tmp326 * tmp12
        tmp328 = tmp324 + tmp327
        tmp331 = tmp330 * tmp5
        tmp332 = tmp331 + tmp7
        tmp335 = tmp334 * tmp22
        tmp336 = tmp332 + tmp335
        tmp339 = tmp338 * tmp12
        tmp340 = tmp336 + tmp339
        tmp343 = tmp342 * tmp5
        tmp344 = tmp343 + tmp7
        tmp347 = tmp346 * tmp22
        tmp348 = tmp344 + tmp347
        tmp351 = tmp350 * tmp12
        tmp352 = tmp348 + tmp351
        tmp355 = tmp354 * tmp5
        tmp356 = tmp355 + tmp7
        tmp359 = tmp358 * tmp22
        tmp360 = tmp356 + tmp359
        tmp363 = tmp362 * tmp12
        tmp364 = tmp360 + tmp363
        tmp367 = tmp366 * tmp5
        tmp368 = tmp367 + tmp7
        tmp371 = tmp370 * tmp22
        tmp372 = tmp368 + tmp371
        tmp375 = tmp374 * tmp12
        tmp376 = tmp372 + tmp375
        tmp377 = tmp28.to(tl.float32)
        tmp378 = tmp377 * tmp377
        tmp379 = tl.broadcast_to(tmp378, [XBLOCK, RBLOCK])
        tmp381 = _tmp380 + tmp379
        _tmp380 = tl.where(rmask, tmp381, _tmp380)
        tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp28, rmask)
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp40, rmask)
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp52, rmask)
        tl.store(out_ptr3 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp64, rmask)
        tl.store(out_ptr4 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp76, rmask)
        tl.store(out_ptr5 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp88, rmask)
        tl.store(out_ptr6 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp100, rmask)
        tl.store(out_ptr7 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp112, rmask)
        tl.store(out_ptr8 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp124, rmask)
        tl.store(out_ptr9 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp136, rmask)
        tl.store(out_ptr10 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp148, rmask)
        tl.store(out_ptr11 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp160, rmask)
        tl.store(out_ptr12 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp172, rmask)
        tl.store(out_ptr13 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp184, rmask)
        tl.store(out_ptr14 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp196, rmask)
        tl.store(out_ptr15 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp208, rmask)
        tl.store(out_ptr16 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp220, rmask)
        tl.store(out_ptr17 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp232, rmask)
        tl.store(out_ptr18 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp244, rmask)
        tl.store(out_ptr19 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp256, rmask)
        tl.store(out_ptr20 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp268, rmask)
        tl.store(out_ptr21 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp280, rmask)
        tl.store(out_ptr22 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp292, rmask)
        tl.store(out_ptr23 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp304, rmask)
        tl.store(out_ptr24 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp316, rmask)
        tl.store(out_ptr25 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp328, rmask)
        tl.store(out_ptr26 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp340, rmask)
        tl.store(out_ptr27 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp352, rmask)
        tl.store(out_ptr28 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp364, rmask)
        tl.store(out_ptr29 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp376, rmask)
    tmp380 = tl.sum(_tmp380, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp382 = tl.load(out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp391 = tl.load(in_ptr36 + (r0), rmask, other=0).to(tl.float32)
        tmp383 = tmp382.to(tl.float32)
        tmp384 = 4096.0
        tmp385 = tmp380 / tmp384
        tmp386 = 1e-05
        tmp387 = tmp385 + tmp386
        tmp388 = tl.math.rsqrt(tmp387)
        tmp389 = tmp383 * tmp388
        tmp390 = tmp389.to(tl.float32)
        tmp392 = tmp390 * tmp391
        tl.store(out_ptr31 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp392, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/ez/cez6vylkge5l5klopwsy26ag4pwt742u47jpyotdipbqpsy7upzw.py
# Source Nodes: [add_18, add_19, float_12, mean_5, mul_47, mul_48, mul_49, rsqrt_5, type_as_11], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_18 => add_18
# add_19 => add_19
# float_12 => convert_element_type_32
# mean_5 => mean_5
# mul_47 => mul_49
# mul_48 => mul_50
# mul_49 => mul_51
# rsqrt_5 => rsqrt_5
# type_as_11 => convert_element_type_33
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


# kernel path: /tmp/torchinductor_mengqy/xd/cxdonia2eoe3tegeex5weminoyp56l4dthc3kidhyi25k2drhfnd.py
# Source Nodes: [add_18, add_20, add_33, add_34, float_17, mean_8, mul_73, mul_74, mul_75, mul_76, rsqrt_8, type_as_16], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_18 => add_18
# add_20 => add_20
# add_33 => add_33
# add_34 => add_34
# float_17 => convert_element_type_48
# mean_8 => mean_8
# mul_73 => mul_77
# mul_74 => mul_78
# mul_75 => mul_79
# mul_76 => mul_80
# rsqrt_8 => rsqrt_8
# type_as_16 => convert_element_type_49
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_12', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_12(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (3)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
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


# kernel path: /tmp/torchinductor_mengqy/ce/ccega744mqi62zbxl2owgxp4qwvwz2t2rvqxd2pja6k3orl3ljz5.py
# Source Nodes: [add_18, add_20, add_27, add_29, add_43, add_44, add_45, add_54, add_55, add_66, add_67, float_21, mean_10, mul_114, mul_115, mul_136, mul_137, mul_93, mul_94, mul_95, mul_96, mul_97, rsqrt_10, type_as_20], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_18 => add_18
# add_20 => add_20
# add_27 => add_27
# add_29 => add_29
# add_43 => add_43
# add_44 => add_44
# add_45 => add_45
# add_54 => add_54
# add_55 => add_55
# add_66 => add_66
# add_67 => add_67
# float_21 => convert_element_type_60
# mean_10 => mean_10
# mul_114 => mul_120
# mul_115 => mul_121
# mul_136 => mul_143
# mul_137 => mul_144
# mul_93 => mul_98
# mul_94 => mul_99
# mul_95 => mul_100
# mul_96 => mul_101
# mul_97 => mul_102
# rsqrt_10 => rsqrt_10
# type_as_20 => convert_element_type_61
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_13', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_13(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (3)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp10 = tl.load(in_ptr0 + (4)).to(tl.float32)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp20 = tl.load(in_ptr7 + (3)).to(tl.float32)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp24 = tl.load(in_ptr7 + (4)).to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp29 = tl.load(in_ptr8 + (3)).to(tl.float32)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    tmp33 = tl.load(in_ptr8 + (4)).to(tl.float32)
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
    _tmp40 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr5 + (r0), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr6 + (r0), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_out_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp28 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp11 * tmp16
        tmp18 = tmp9 + tmp17
        tmp22 = tmp21 * tmp7
        tmp23 = tmp19 + tmp22
        tmp26 = tmp25 * tmp16
        tmp27 = tmp23 + tmp26
        tmp31 = tmp30 * tmp7
        tmp32 = tmp28 + tmp31
        tmp35 = tmp34 * tmp16
        tmp36 = tmp32 + tmp35
        tmp37 = tmp18.to(tl.float32)
        tmp38 = tmp37 * tmp37
        tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
        tmp41 = _tmp40 + tmp39
        _tmp40 = tl.where(rmask, tmp41, _tmp40)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp18, rmask)
        tl.store(in_out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp27, rmask)
        tl.store(in_out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp36, rmask)
    tmp40 = tl.sum(_tmp40, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp42 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp51 = tl.load(in_ptr9 + (r0), rmask, other=0).to(tl.float32)
        tmp43 = tmp42.to(tl.float32)
        tmp44 = 4096.0
        tmp45 = tmp40 / tmp44
        tmp46 = 1e-05
        tmp47 = tmp45 + tmp46
        tmp48 = tl.math.rsqrt(tmp47)
        tmp49 = tmp43 * tmp48
        tmp50 = tmp49.to(tl.float32)
        tmp52 = tmp50 * tmp51
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp52, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/rn/crn2f4sb6egfy3jrnrecgt5dvie7wsmxlazh2lba6f7ax7a5k7i6.py
# Source Nodes: [add_37, add_39, add_56, add_57, float_25, mean_12, mul_116, mul_117, mul_118, mul_119, rsqrt_12, type_as_24], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_37 => add_37
# add_39 => add_39
# add_56 => add_56
# add_57 => add_57
# float_25 => convert_element_type_72
# mean_12 => mean_12
# mul_116 => mul_122
# mul_117 => mul_123
# mul_118 => mul_124
# mul_119 => mul_125
# rsqrt_12 => rsqrt_12
# type_as_24 => convert_element_type_73
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
    tmp1 = tl.load(in_ptr0 + (5)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
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


# kernel path: /tmp/torchinductor_mengqy/uk/cukn7drch7z5md4hyqxegslhznlby7xnhy5y7uaqfcl35urqf6nx.py
# Source Nodes: [add_108, add_109, add_110, add_111, add_124, add_125, add_126, add_127, add_141, add_142, add_143, add_144, add_159, add_160, add_161, add_162, add_178, add_179, add_18, add_180, add_181, add_198, add_199, add_20, add_200, add_201, add_219, add_220, add_221, add_222, add_241, add_242, add_243, add_244, add_264, add_265, add_266, add_267, add_27, add_288, add_289, add_29, add_290, add_291, add_313, add_314, add_315, add_316, add_339, add_340, add_341, add_342, add_366, add_367, add_368, add_369, add_37, add_39, add_394, add_395, add_396, add_397, add_423, add_424, add_425, add_426, add_453, add_454, add_455, add_456, add_48, add_484, add_485, add_486, add_487, add_50, add_516, add_517, add_518, add_519, add_549, add_550, add_551, add_552, add_583, add_584, add_585, add_586, add_618, add_619, add_620, add_621, add_654, add_655, add_656, add_657, add_68, add_69, add_691, add_692, add_693, add_694, add_70, add_79, add_80, add_81, add_82, add_93, add_94, add_95, add_96, float_29, mean_14, mul_1011, mul_1012, mul_1013, mul_1014, mul_138, mul_139, mul_140, mul_141, mul_142, mul_159, mul_160, mul_161, mul_162, mul_183, mul_184, mul_185, mul_186, mul_208, mul_209, mul_210, mul_211, mul_234, mul_235, mul_236, mul_237, mul_261, mul_262, mul_263, mul_264, mul_289, mul_290, mul_291, mul_292, mul_318, mul_319, mul_320, mul_321, mul_348, mul_349, mul_350, mul_351, mul_379, mul_380, mul_381, mul_382, mul_411, mul_412, mul_413, mul_414, mul_444, mul_445, mul_446, mul_447, mul_478, mul_479, mul_480, mul_481, mul_513, mul_514, mul_515, mul_516, mul_549, mul_550, mul_551, mul_552, mul_586, mul_587, mul_588, mul_589, mul_624, mul_625, mul_626, mul_627, mul_663, mul_664, mul_665, mul_666, mul_703, mul_704, mul_705, mul_706, mul_744, mul_745, mul_746, mul_747, mul_786, mul_787, mul_788, mul_789, mul_829, mul_830, mul_831, mul_832, mul_873, mul_874, mul_875, mul_876, mul_918, mul_919, mul_920, mul_921, mul_964, mul_965, mul_966, mul_967, rsqrt_14, type_as_28], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_108 => add_108
# add_109 => add_109
# add_110 => add_110
# add_111 => add_111
# add_124 => add_124
# add_125 => add_125
# add_126 => add_126
# add_127 => add_127
# add_141 => add_141
# add_142 => add_142
# add_143 => add_143
# add_144 => add_144
# add_159 => add_159
# add_160 => add_160
# add_161 => add_161
# add_162 => add_162
# add_178 => add_178
# add_179 => add_179
# add_18 => add_18
# add_180 => add_180
# add_181 => add_181
# add_198 => add_198
# add_199 => add_199
# add_20 => add_20
# add_200 => add_200
# add_201 => add_201
# add_219 => add_219
# add_220 => add_220
# add_221 => add_221
# add_222 => add_222
# add_241 => add_241
# add_242 => add_242
# add_243 => add_243
# add_244 => add_244
# add_264 => add_264
# add_265 => add_265
# add_266 => add_266
# add_267 => add_267
# add_27 => add_27
# add_288 => add_288
# add_289 => add_289
# add_29 => add_29
# add_290 => add_290
# add_291 => add_291
# add_313 => add_313
# add_314 => add_314
# add_315 => add_315
# add_316 => add_316
# add_339 => add_339
# add_340 => add_340
# add_341 => add_341
# add_342 => add_342
# add_366 => add_366
# add_367 => add_367
# add_368 => add_368
# add_369 => add_369
# add_37 => add_37
# add_39 => add_39
# add_394 => add_394
# add_395 => add_395
# add_396 => add_396
# add_397 => add_397
# add_423 => add_423
# add_424 => add_424
# add_425 => add_425
# add_426 => add_426
# add_453 => add_453
# add_454 => add_454
# add_455 => add_455
# add_456 => add_456
# add_48 => add_48
# add_484 => add_484
# add_485 => add_485
# add_486 => add_486
# add_487 => add_487
# add_50 => add_50
# add_516 => add_516
# add_517 => add_517
# add_518 => add_518
# add_519 => add_519
# add_549 => add_549
# add_550 => add_550
# add_551 => add_551
# add_552 => add_552
# add_583 => add_583
# add_584 => add_584
# add_585 => add_585
# add_586 => add_586
# add_618 => add_618
# add_619 => add_619
# add_620 => add_620
# add_621 => add_621
# add_654 => add_654
# add_655 => add_655
# add_656 => add_656
# add_657 => add_657
# add_68 => add_68
# add_69 => add_69
# add_691 => add_691
# add_692 => add_692
# add_693 => add_693
# add_694 => add_694
# add_70 => add_70
# add_79 => add_79
# add_80 => add_80
# add_81 => add_81
# add_82 => add_82
# add_93 => add_93
# add_94 => add_94
# add_95 => add_95
# add_96 => add_96
# float_29 => convert_element_type_84
# mean_14 => mean_14
# mul_1011 => mul_1043
# mul_1012 => mul_1044
# mul_1013 => mul_1045
# mul_1014 => mul_1046
# mul_138 => mul_145
# mul_139 => mul_146
# mul_140 => mul_147
# mul_141 => mul_148
# mul_142 => mul_149
# mul_159 => mul_167
# mul_160 => mul_168
# mul_161 => mul_169
# mul_162 => mul_170
# mul_183 => mul_192
# mul_184 => mul_193
# mul_185 => mul_194
# mul_186 => mul_195
# mul_208 => mul_218
# mul_209 => mul_219
# mul_210 => mul_220
# mul_211 => mul_221
# mul_234 => mul_245
# mul_235 => mul_246
# mul_236 => mul_247
# mul_237 => mul_248
# mul_261 => mul_273
# mul_262 => mul_274
# mul_263 => mul_275
# mul_264 => mul_276
# mul_289 => mul_302
# mul_290 => mul_303
# mul_291 => mul_304
# mul_292 => mul_305
# mul_318 => mul_332
# mul_319 => mul_333
# mul_320 => mul_334
# mul_321 => mul_335
# mul_348 => mul_363
# mul_349 => mul_364
# mul_350 => mul_365
# mul_351 => mul_366
# mul_379 => mul_395
# mul_380 => mul_396
# mul_381 => mul_397
# mul_382 => mul_398
# mul_411 => mul_428
# mul_412 => mul_429
# mul_413 => mul_430
# mul_414 => mul_431
# mul_444 => mul_462
# mul_445 => mul_463
# mul_446 => mul_464
# mul_447 => mul_465
# mul_478 => mul_497
# mul_479 => mul_498
# mul_480 => mul_499
# mul_481 => mul_500
# mul_513 => mul_533
# mul_514 => mul_534
# mul_515 => mul_535
# mul_516 => mul_536
# mul_549 => mul_570
# mul_550 => mul_571
# mul_551 => mul_572
# mul_552 => mul_573
# mul_586 => mul_608
# mul_587 => mul_609
# mul_588 => mul_610
# mul_589 => mul_611
# mul_624 => mul_647
# mul_625 => mul_648
# mul_626 => mul_649
# mul_627 => mul_650
# mul_663 => mul_687
# mul_664 => mul_688
# mul_665 => mul_689
# mul_666 => mul_690
# mul_703 => mul_728
# mul_704 => mul_729
# mul_705 => mul_730
# mul_706 => mul_731
# mul_744 => mul_770
# mul_745 => mul_771
# mul_746 => mul_772
# mul_747 => mul_773
# mul_786 => mul_813
# mul_787 => mul_814
# mul_788 => mul_815
# mul_789 => mul_816
# mul_829 => mul_857
# mul_830 => mul_858
# mul_831 => mul_859
# mul_832 => mul_860
# mul_873 => mul_902
# mul_874 => mul_903
# mul_875 => mul_904
# mul_876 => mul_905
# mul_918 => mul_948
# mul_919 => mul_949
# mul_920 => mul_950
# mul_921 => mul_951
# mul_964 => mul_995
# mul_965 => mul_996
# mul_966 => mul_997
# mul_967 => mul_998
# rsqrt_14 => rsqrt_14
# type_as_28 => convert_element_type_85
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: '*fp16', 21: '*fp16', 22: '*fp16', 23: '*fp16', 24: '*fp16', 25: '*fp16', 26: '*fp16', 27: '*fp16', 28: '*fp16', 29: '*fp16', 30: '*fp16', 31: '*fp16', 32: '*fp16', 33: '*fp16', 34: '*fp16', 35: '*fp16', 36: '*fp16', 37: '*fp16', 38: '*fp16', 39: '*fp16', 40: '*fp16', 41: '*fp16', 42: '*fp16', 43: '*fp16', 44: '*fp16', 45: '*fp16', 46: '*fp16', 47: '*fp16', 48: '*fp16', 49: '*fp16', 50: '*fp16', 51: '*fp16', 52: '*fp16', 53: '*fp16', 54: '*fp16', 55: '*fp16', 56: '*fp16', 57: '*fp16', 58: '*fp16', 59: '*fp16', 60: '*fp16', 61: '*fp16', 62: '*fp16', 63: '*fp16', 64: '*fp16', 65: '*fp16', 66: 'i32', 67: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr10', 'in_out_ptr11', 'in_out_ptr12', 'in_out_ptr13', 'in_out_ptr14', 'in_out_ptr15', 'in_out_ptr16', 'in_out_ptr17', 'in_out_ptr18', 'in_out_ptr19', 'in_out_ptr2', 'in_out_ptr20', 'in_out_ptr21', 'in_out_ptr22', 'in_out_ptr23', 'in_out_ptr24', 'in_out_ptr25', 'in_out_ptr3', 'in_out_ptr4', 'in_out_ptr5', 'in_out_ptr6', 'in_out_ptr7', 'in_out_ptr8', 'in_out_ptr9'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_15', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(67,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_15(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_out_ptr3, in_out_ptr4, in_out_ptr5, in_out_ptr6, in_out_ptr7, in_out_ptr8, in_out_ptr9, in_out_ptr10, in_out_ptr11, in_out_ptr12, in_out_ptr13, in_out_ptr14, in_out_ptr15, in_out_ptr16, in_out_ptr17, in_out_ptr18, in_out_ptr19, in_out_ptr20, in_out_ptr21, in_out_ptr22, in_out_ptr23, in_out_ptr24, in_out_ptr25, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (3)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp10 = tl.load(in_ptr0 + (4)).to(tl.float32)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp20 = tl.load(in_ptr7 + (3)).to(tl.float32)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp24 = tl.load(in_ptr7 + (4)).to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp29 = tl.load(in_ptr8 + (3)).to(tl.float32)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    tmp33 = tl.load(in_ptr8 + (4)).to(tl.float32)
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
    tmp38 = tl.load(in_ptr9 + (3)).to(tl.float32)
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
    tmp42 = tl.load(in_ptr9 + (4)).to(tl.float32)
    tmp43 = tl.broadcast_to(tmp42, [XBLOCK, RBLOCK])
    tmp47 = tl.load(in_ptr10 + (3)).to(tl.float32)
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK, RBLOCK])
    tmp51 = tl.load(in_ptr10 + (4)).to(tl.float32)
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK, RBLOCK])
    tmp56 = tl.load(in_ptr11 + (3)).to(tl.float32)
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK, RBLOCK])
    tmp60 = tl.load(in_ptr11 + (4)).to(tl.float32)
    tmp61 = tl.broadcast_to(tmp60, [XBLOCK, RBLOCK])
    tmp65 = tl.load(in_ptr12 + (3)).to(tl.float32)
    tmp66 = tl.broadcast_to(tmp65, [XBLOCK, RBLOCK])
    tmp69 = tl.load(in_ptr12 + (4)).to(tl.float32)
    tmp70 = tl.broadcast_to(tmp69, [XBLOCK, RBLOCK])
    tmp74 = tl.load(in_ptr13 + (3)).to(tl.float32)
    tmp75 = tl.broadcast_to(tmp74, [XBLOCK, RBLOCK])
    tmp78 = tl.load(in_ptr13 + (4)).to(tl.float32)
    tmp79 = tl.broadcast_to(tmp78, [XBLOCK, RBLOCK])
    tmp83 = tl.load(in_ptr14 + (3)).to(tl.float32)
    tmp84 = tl.broadcast_to(tmp83, [XBLOCK, RBLOCK])
    tmp87 = tl.load(in_ptr14 + (4)).to(tl.float32)
    tmp88 = tl.broadcast_to(tmp87, [XBLOCK, RBLOCK])
    tmp92 = tl.load(in_ptr15 + (3)).to(tl.float32)
    tmp93 = tl.broadcast_to(tmp92, [XBLOCK, RBLOCK])
    tmp96 = tl.load(in_ptr15 + (4)).to(tl.float32)
    tmp97 = tl.broadcast_to(tmp96, [XBLOCK, RBLOCK])
    tmp101 = tl.load(in_ptr16 + (3)).to(tl.float32)
    tmp102 = tl.broadcast_to(tmp101, [XBLOCK, RBLOCK])
    tmp105 = tl.load(in_ptr16 + (4)).to(tl.float32)
    tmp106 = tl.broadcast_to(tmp105, [XBLOCK, RBLOCK])
    tmp110 = tl.load(in_ptr17 + (3)).to(tl.float32)
    tmp111 = tl.broadcast_to(tmp110, [XBLOCK, RBLOCK])
    tmp114 = tl.load(in_ptr17 + (4)).to(tl.float32)
    tmp115 = tl.broadcast_to(tmp114, [XBLOCK, RBLOCK])
    tmp119 = tl.load(in_ptr18 + (3)).to(tl.float32)
    tmp120 = tl.broadcast_to(tmp119, [XBLOCK, RBLOCK])
    tmp123 = tl.load(in_ptr18 + (4)).to(tl.float32)
    tmp124 = tl.broadcast_to(tmp123, [XBLOCK, RBLOCK])
    tmp128 = tl.load(in_ptr19 + (3)).to(tl.float32)
    tmp129 = tl.broadcast_to(tmp128, [XBLOCK, RBLOCK])
    tmp132 = tl.load(in_ptr19 + (4)).to(tl.float32)
    tmp133 = tl.broadcast_to(tmp132, [XBLOCK, RBLOCK])
    tmp137 = tl.load(in_ptr20 + (3)).to(tl.float32)
    tmp138 = tl.broadcast_to(tmp137, [XBLOCK, RBLOCK])
    tmp141 = tl.load(in_ptr20 + (4)).to(tl.float32)
    tmp142 = tl.broadcast_to(tmp141, [XBLOCK, RBLOCK])
    tmp146 = tl.load(in_ptr21 + (3)).to(tl.float32)
    tmp147 = tl.broadcast_to(tmp146, [XBLOCK, RBLOCK])
    tmp150 = tl.load(in_ptr21 + (4)).to(tl.float32)
    tmp151 = tl.broadcast_to(tmp150, [XBLOCK, RBLOCK])
    tmp155 = tl.load(in_ptr22 + (3)).to(tl.float32)
    tmp156 = tl.broadcast_to(tmp155, [XBLOCK, RBLOCK])
    tmp159 = tl.load(in_ptr22 + (4)).to(tl.float32)
    tmp160 = tl.broadcast_to(tmp159, [XBLOCK, RBLOCK])
    tmp164 = tl.load(in_ptr23 + (3)).to(tl.float32)
    tmp165 = tl.broadcast_to(tmp164, [XBLOCK, RBLOCK])
    tmp168 = tl.load(in_ptr23 + (4)).to(tl.float32)
    tmp169 = tl.broadcast_to(tmp168, [XBLOCK, RBLOCK])
    tmp173 = tl.load(in_ptr24 + (3)).to(tl.float32)
    tmp174 = tl.broadcast_to(tmp173, [XBLOCK, RBLOCK])
    tmp177 = tl.load(in_ptr24 + (4)).to(tl.float32)
    tmp178 = tl.broadcast_to(tmp177, [XBLOCK, RBLOCK])
    tmp182 = tl.load(in_ptr25 + (3)).to(tl.float32)
    tmp183 = tl.broadcast_to(tmp182, [XBLOCK, RBLOCK])
    tmp186 = tl.load(in_ptr25 + (4)).to(tl.float32)
    tmp187 = tl.broadcast_to(tmp186, [XBLOCK, RBLOCK])
    tmp191 = tl.load(in_ptr26 + (3)).to(tl.float32)
    tmp192 = tl.broadcast_to(tmp191, [XBLOCK, RBLOCK])
    tmp195 = tl.load(in_ptr26 + (4)).to(tl.float32)
    tmp196 = tl.broadcast_to(tmp195, [XBLOCK, RBLOCK])
    tmp200 = tl.load(in_ptr27 + (3)).to(tl.float32)
    tmp201 = tl.broadcast_to(tmp200, [XBLOCK, RBLOCK])
    tmp204 = tl.load(in_ptr27 + (4)).to(tl.float32)
    tmp205 = tl.broadcast_to(tmp204, [XBLOCK, RBLOCK])
    tmp209 = tl.load(in_ptr28 + (3)).to(tl.float32)
    tmp210 = tl.broadcast_to(tmp209, [XBLOCK, RBLOCK])
    tmp213 = tl.load(in_ptr28 + (4)).to(tl.float32)
    tmp214 = tl.broadcast_to(tmp213, [XBLOCK, RBLOCK])
    tmp218 = tl.load(in_ptr29 + (3)).to(tl.float32)
    tmp219 = tl.broadcast_to(tmp218, [XBLOCK, RBLOCK])
    tmp222 = tl.load(in_ptr29 + (4)).to(tl.float32)
    tmp223 = tl.broadcast_to(tmp222, [XBLOCK, RBLOCK])
    tmp227 = tl.load(in_ptr30 + (3)).to(tl.float32)
    tmp228 = tl.broadcast_to(tmp227, [XBLOCK, RBLOCK])
    tmp231 = tl.load(in_ptr30 + (4)).to(tl.float32)
    tmp232 = tl.broadcast_to(tmp231, [XBLOCK, RBLOCK])
    tmp236 = tl.load(in_ptr31 + (5)).to(tl.float32)
    tmp237 = tl.broadcast_to(tmp236, [XBLOCK, RBLOCK])
    tmp245 = tl.load(in_ptr31 + (6)).to(tl.float32)
    tmp246 = tl.broadcast_to(tmp245, [XBLOCK, RBLOCK])
    tmp254 = tl.load(in_ptr28 + (5)).to(tl.float32)
    tmp255 = tl.broadcast_to(tmp254, [XBLOCK, RBLOCK])
    tmp258 = tl.load(in_ptr28 + (6)).to(tl.float32)
    tmp259 = tl.broadcast_to(tmp258, [XBLOCK, RBLOCK])
    tmp262 = tl.load(in_ptr29 + (5)).to(tl.float32)
    tmp263 = tl.broadcast_to(tmp262, [XBLOCK, RBLOCK])
    tmp266 = tl.load(in_ptr29 + (6)).to(tl.float32)
    tmp267 = tl.broadcast_to(tmp266, [XBLOCK, RBLOCK])
    tmp270 = tl.load(in_ptr30 + (5)).to(tl.float32)
    tmp271 = tl.broadcast_to(tmp270, [XBLOCK, RBLOCK])
    tmp274 = tl.load(in_ptr30 + (6)).to(tl.float32)
    tmp275 = tl.broadcast_to(tmp274, [XBLOCK, RBLOCK])
    tmp278 = tl.load(in_ptr26 + (5)).to(tl.float32)
    tmp279 = tl.broadcast_to(tmp278, [XBLOCK, RBLOCK])
    tmp282 = tl.load(in_ptr26 + (6)).to(tl.float32)
    tmp283 = tl.broadcast_to(tmp282, [XBLOCK, RBLOCK])
    tmp286 = tl.load(in_ptr27 + (5)).to(tl.float32)
    tmp287 = tl.broadcast_to(tmp286, [XBLOCK, RBLOCK])
    tmp290 = tl.load(in_ptr27 + (6)).to(tl.float32)
    tmp291 = tl.broadcast_to(tmp290, [XBLOCK, RBLOCK])
    tmp294 = tl.load(in_ptr24 + (5)).to(tl.float32)
    tmp295 = tl.broadcast_to(tmp294, [XBLOCK, RBLOCK])
    tmp298 = tl.load(in_ptr24 + (6)).to(tl.float32)
    tmp299 = tl.broadcast_to(tmp298, [XBLOCK, RBLOCK])
    tmp302 = tl.load(in_ptr25 + (5)).to(tl.float32)
    tmp303 = tl.broadcast_to(tmp302, [XBLOCK, RBLOCK])
    tmp306 = tl.load(in_ptr25 + (6)).to(tl.float32)
    tmp307 = tl.broadcast_to(tmp306, [XBLOCK, RBLOCK])
    tmp310 = tl.load(in_ptr22 + (5)).to(tl.float32)
    tmp311 = tl.broadcast_to(tmp310, [XBLOCK, RBLOCK])
    tmp314 = tl.load(in_ptr22 + (6)).to(tl.float32)
    tmp315 = tl.broadcast_to(tmp314, [XBLOCK, RBLOCK])
    tmp318 = tl.load(in_ptr23 + (5)).to(tl.float32)
    tmp319 = tl.broadcast_to(tmp318, [XBLOCK, RBLOCK])
    tmp322 = tl.load(in_ptr23 + (6)).to(tl.float32)
    tmp323 = tl.broadcast_to(tmp322, [XBLOCK, RBLOCK])
    tmp326 = tl.load(in_ptr20 + (5)).to(tl.float32)
    tmp327 = tl.broadcast_to(tmp326, [XBLOCK, RBLOCK])
    tmp330 = tl.load(in_ptr20 + (6)).to(tl.float32)
    tmp331 = tl.broadcast_to(tmp330, [XBLOCK, RBLOCK])
    tmp334 = tl.load(in_ptr21 + (5)).to(tl.float32)
    tmp335 = tl.broadcast_to(tmp334, [XBLOCK, RBLOCK])
    tmp338 = tl.load(in_ptr21 + (6)).to(tl.float32)
    tmp339 = tl.broadcast_to(tmp338, [XBLOCK, RBLOCK])
    tmp342 = tl.load(in_ptr18 + (5)).to(tl.float32)
    tmp343 = tl.broadcast_to(tmp342, [XBLOCK, RBLOCK])
    tmp346 = tl.load(in_ptr18 + (6)).to(tl.float32)
    tmp347 = tl.broadcast_to(tmp346, [XBLOCK, RBLOCK])
    tmp350 = tl.load(in_ptr19 + (5)).to(tl.float32)
    tmp351 = tl.broadcast_to(tmp350, [XBLOCK, RBLOCK])
    tmp354 = tl.load(in_ptr19 + (6)).to(tl.float32)
    tmp355 = tl.broadcast_to(tmp354, [XBLOCK, RBLOCK])
    tmp358 = tl.load(in_ptr16 + (5)).to(tl.float32)
    tmp359 = tl.broadcast_to(tmp358, [XBLOCK, RBLOCK])
    tmp362 = tl.load(in_ptr16 + (6)).to(tl.float32)
    tmp363 = tl.broadcast_to(tmp362, [XBLOCK, RBLOCK])
    tmp366 = tl.load(in_ptr17 + (5)).to(tl.float32)
    tmp367 = tl.broadcast_to(tmp366, [XBLOCK, RBLOCK])
    tmp370 = tl.load(in_ptr17 + (6)).to(tl.float32)
    tmp371 = tl.broadcast_to(tmp370, [XBLOCK, RBLOCK])
    tmp374 = tl.load(in_ptr14 + (5)).to(tl.float32)
    tmp375 = tl.broadcast_to(tmp374, [XBLOCK, RBLOCK])
    tmp378 = tl.load(in_ptr14 + (6)).to(tl.float32)
    tmp379 = tl.broadcast_to(tmp378, [XBLOCK, RBLOCK])
    tmp382 = tl.load(in_ptr15 + (5)).to(tl.float32)
    tmp383 = tl.broadcast_to(tmp382, [XBLOCK, RBLOCK])
    tmp386 = tl.load(in_ptr15 + (6)).to(tl.float32)
    tmp387 = tl.broadcast_to(tmp386, [XBLOCK, RBLOCK])
    tmp390 = tl.load(in_ptr12 + (5)).to(tl.float32)
    tmp391 = tl.broadcast_to(tmp390, [XBLOCK, RBLOCK])
    tmp394 = tl.load(in_ptr12 + (6)).to(tl.float32)
    tmp395 = tl.broadcast_to(tmp394, [XBLOCK, RBLOCK])
    tmp398 = tl.load(in_ptr13 + (5)).to(tl.float32)
    tmp399 = tl.broadcast_to(tmp398, [XBLOCK, RBLOCK])
    tmp402 = tl.load(in_ptr13 + (6)).to(tl.float32)
    tmp403 = tl.broadcast_to(tmp402, [XBLOCK, RBLOCK])
    tmp406 = tl.load(in_ptr10 + (5)).to(tl.float32)
    tmp407 = tl.broadcast_to(tmp406, [XBLOCK, RBLOCK])
    tmp410 = tl.load(in_ptr10 + (6)).to(tl.float32)
    tmp411 = tl.broadcast_to(tmp410, [XBLOCK, RBLOCK])
    tmp414 = tl.load(in_ptr11 + (5)).to(tl.float32)
    tmp415 = tl.broadcast_to(tmp414, [XBLOCK, RBLOCK])
    tmp418 = tl.load(in_ptr11 + (6)).to(tl.float32)
    tmp419 = tl.broadcast_to(tmp418, [XBLOCK, RBLOCK])
    tmp422 = tl.load(in_ptr8 + (5)).to(tl.float32)
    tmp423 = tl.broadcast_to(tmp422, [XBLOCK, RBLOCK])
    tmp426 = tl.load(in_ptr8 + (6)).to(tl.float32)
    tmp427 = tl.broadcast_to(tmp426, [XBLOCK, RBLOCK])
    tmp430 = tl.load(in_ptr9 + (5)).to(tl.float32)
    tmp431 = tl.broadcast_to(tmp430, [XBLOCK, RBLOCK])
    tmp434 = tl.load(in_ptr9 + (6)).to(tl.float32)
    tmp435 = tl.broadcast_to(tmp434, [XBLOCK, RBLOCK])
    tmp438 = tl.load(in_ptr0 + (5)).to(tl.float32)
    tmp439 = tl.broadcast_to(tmp438, [XBLOCK, RBLOCK])
    tmp442 = tl.load(in_ptr0 + (6)).to(tl.float32)
    tmp443 = tl.broadcast_to(tmp442, [XBLOCK, RBLOCK])
    tmp446 = tl.load(in_ptr7 + (5)).to(tl.float32)
    tmp447 = tl.broadcast_to(tmp446, [XBLOCK, RBLOCK])
    tmp450 = tl.load(in_ptr7 + (6)).to(tl.float32)
    tmp451 = tl.broadcast_to(tmp450, [XBLOCK, RBLOCK])
    _tmp457 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr5 + (r0), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr6 + (r0), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_out_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp28 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp37 = tl.load(in_out_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp46 = tl.load(in_out_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp55 = tl.load(in_out_ptr5 + (r0), rmask, other=0).to(tl.float32)
        tmp64 = tl.load(in_out_ptr6 + (r0), rmask, other=0).to(tl.float32)
        tmp73 = tl.load(in_out_ptr7 + (r0), rmask, other=0).to(tl.float32)
        tmp82 = tl.load(in_out_ptr8 + (r0), rmask, other=0).to(tl.float32)
        tmp91 = tl.load(in_out_ptr9 + (r0), rmask, other=0).to(tl.float32)
        tmp100 = tl.load(in_out_ptr10 + (r0), rmask, other=0).to(tl.float32)
        tmp109 = tl.load(in_out_ptr11 + (r0), rmask, other=0).to(tl.float32)
        tmp118 = tl.load(in_out_ptr12 + (r0), rmask, other=0).to(tl.float32)
        tmp127 = tl.load(in_out_ptr13 + (r0), rmask, other=0).to(tl.float32)
        tmp136 = tl.load(in_out_ptr14 + (r0), rmask, other=0).to(tl.float32)
        tmp145 = tl.load(in_out_ptr15 + (r0), rmask, other=0).to(tl.float32)
        tmp154 = tl.load(in_out_ptr16 + (r0), rmask, other=0).to(tl.float32)
        tmp163 = tl.load(in_out_ptr17 + (r0), rmask, other=0).to(tl.float32)
        tmp172 = tl.load(in_out_ptr18 + (r0), rmask, other=0).to(tl.float32)
        tmp181 = tl.load(in_out_ptr19 + (r0), rmask, other=0).to(tl.float32)
        tmp190 = tl.load(in_out_ptr20 + (r0), rmask, other=0).to(tl.float32)
        tmp199 = tl.load(in_out_ptr21 + (r0), rmask, other=0).to(tl.float32)
        tmp208 = tl.load(in_out_ptr22 + (r0), rmask, other=0).to(tl.float32)
        tmp217 = tl.load(in_out_ptr23 + (r0), rmask, other=0).to(tl.float32)
        tmp226 = tl.load(in_out_ptr24 + (r0), rmask, other=0).to(tl.float32)
        tmp235 = tl.load(in_out_ptr25 + (r0), rmask, other=0).to(tl.float32)
        tmp238 = tl.load(in_ptr32 + (r0), rmask, other=0).to(tl.float32)
        tmp239 = tl.load(in_ptr33 + (r0), rmask, other=0).to(tl.float32)
        tmp241 = tl.load(in_ptr34 + (r0), rmask, other=0).to(tl.float32)
        tmp247 = tl.load(in_ptr35 + (r0), rmask, other=0).to(tl.float32)
        tmp248 = tl.load(in_ptr36 + (r0), rmask, other=0).to(tl.float32)
        tmp250 = tl.load(in_ptr37 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp11 * tmp16
        tmp18 = tmp9 + tmp17
        tmp22 = tmp21 * tmp7
        tmp23 = tmp19 + tmp22
        tmp26 = tmp25 * tmp16
        tmp27 = tmp23 + tmp26
        tmp31 = tmp30 * tmp7
        tmp32 = tmp28 + tmp31
        tmp35 = tmp34 * tmp16
        tmp36 = tmp32 + tmp35
        tmp40 = tmp39 * tmp7
        tmp41 = tmp37 + tmp40
        tmp44 = tmp43 * tmp16
        tmp45 = tmp41 + tmp44
        tmp49 = tmp48 * tmp7
        tmp50 = tmp46 + tmp49
        tmp53 = tmp52 * tmp16
        tmp54 = tmp50 + tmp53
        tmp58 = tmp57 * tmp7
        tmp59 = tmp55 + tmp58
        tmp62 = tmp61 * tmp16
        tmp63 = tmp59 + tmp62
        tmp67 = tmp66 * tmp7
        tmp68 = tmp64 + tmp67
        tmp71 = tmp70 * tmp16
        tmp72 = tmp68 + tmp71
        tmp76 = tmp75 * tmp7
        tmp77 = tmp73 + tmp76
        tmp80 = tmp79 * tmp16
        tmp81 = tmp77 + tmp80
        tmp85 = tmp84 * tmp7
        tmp86 = tmp82 + tmp85
        tmp89 = tmp88 * tmp16
        tmp90 = tmp86 + tmp89
        tmp94 = tmp93 * tmp7
        tmp95 = tmp91 + tmp94
        tmp98 = tmp97 * tmp16
        tmp99 = tmp95 + tmp98
        tmp103 = tmp102 * tmp7
        tmp104 = tmp100 + tmp103
        tmp107 = tmp106 * tmp16
        tmp108 = tmp104 + tmp107
        tmp112 = tmp111 * tmp7
        tmp113 = tmp109 + tmp112
        tmp116 = tmp115 * tmp16
        tmp117 = tmp113 + tmp116
        tmp121 = tmp120 * tmp7
        tmp122 = tmp118 + tmp121
        tmp125 = tmp124 * tmp16
        tmp126 = tmp122 + tmp125
        tmp130 = tmp129 * tmp7
        tmp131 = tmp127 + tmp130
        tmp134 = tmp133 * tmp16
        tmp135 = tmp131 + tmp134
        tmp139 = tmp138 * tmp7
        tmp140 = tmp136 + tmp139
        tmp143 = tmp142 * tmp16
        tmp144 = tmp140 + tmp143
        tmp148 = tmp147 * tmp7
        tmp149 = tmp145 + tmp148
        tmp152 = tmp151 * tmp16
        tmp153 = tmp149 + tmp152
        tmp157 = tmp156 * tmp7
        tmp158 = tmp154 + tmp157
        tmp161 = tmp160 * tmp16
        tmp162 = tmp158 + tmp161
        tmp166 = tmp165 * tmp7
        tmp167 = tmp163 + tmp166
        tmp170 = tmp169 * tmp16
        tmp171 = tmp167 + tmp170
        tmp175 = tmp174 * tmp7
        tmp176 = tmp172 + tmp175
        tmp179 = tmp178 * tmp16
        tmp180 = tmp176 + tmp179
        tmp184 = tmp183 * tmp7
        tmp185 = tmp181 + tmp184
        tmp188 = tmp187 * tmp16
        tmp189 = tmp185 + tmp188
        tmp193 = tmp192 * tmp7
        tmp194 = tmp190 + tmp193
        tmp197 = tmp196 * tmp16
        tmp198 = tmp194 + tmp197
        tmp202 = tmp201 * tmp7
        tmp203 = tmp199 + tmp202
        tmp206 = tmp205 * tmp16
        tmp207 = tmp203 + tmp206
        tmp211 = tmp210 * tmp7
        tmp212 = tmp208 + tmp211
        tmp215 = tmp214 * tmp16
        tmp216 = tmp212 + tmp215
        tmp220 = tmp219 * tmp7
        tmp221 = tmp217 + tmp220
        tmp224 = tmp223 * tmp16
        tmp225 = tmp221 + tmp224
        tmp229 = tmp228 * tmp7
        tmp230 = tmp226 + tmp229
        tmp233 = tmp232 * tmp16
        tmp234 = tmp230 + tmp233
        tmp240 = tmp238 + tmp239
        tmp242 = tmp240 + tmp241
        tmp243 = tmp237 * tmp242
        tmp244 = tmp235 + tmp243
        tmp249 = tmp247 + tmp248
        tmp251 = tmp249 + tmp250
        tmp252 = tmp246 * tmp251
        tmp253 = tmp244 + tmp252
        tmp256 = tmp255 * tmp242
        tmp257 = tmp216 + tmp256
        tmp260 = tmp259 * tmp251
        tmp261 = tmp257 + tmp260
        tmp264 = tmp263 * tmp242
        tmp265 = tmp225 + tmp264
        tmp268 = tmp267 * tmp251
        tmp269 = tmp265 + tmp268
        tmp272 = tmp271 * tmp242
        tmp273 = tmp234 + tmp272
        tmp276 = tmp275 * tmp251
        tmp277 = tmp273 + tmp276
        tmp280 = tmp279 * tmp242
        tmp281 = tmp198 + tmp280
        tmp284 = tmp283 * tmp251
        tmp285 = tmp281 + tmp284
        tmp288 = tmp287 * tmp242
        tmp289 = tmp207 + tmp288
        tmp292 = tmp291 * tmp251
        tmp293 = tmp289 + tmp292
        tmp296 = tmp295 * tmp242
        tmp297 = tmp180 + tmp296
        tmp300 = tmp299 * tmp251
        tmp301 = tmp297 + tmp300
        tmp304 = tmp303 * tmp242
        tmp305 = tmp189 + tmp304
        tmp308 = tmp307 * tmp251
        tmp309 = tmp305 + tmp308
        tmp312 = tmp311 * tmp242
        tmp313 = tmp162 + tmp312
        tmp316 = tmp315 * tmp251
        tmp317 = tmp313 + tmp316
        tmp320 = tmp319 * tmp242
        tmp321 = tmp171 + tmp320
        tmp324 = tmp323 * tmp251
        tmp325 = tmp321 + tmp324
        tmp328 = tmp327 * tmp242
        tmp329 = tmp144 + tmp328
        tmp332 = tmp331 * tmp251
        tmp333 = tmp329 + tmp332
        tmp336 = tmp335 * tmp242
        tmp337 = tmp153 + tmp336
        tmp340 = tmp339 * tmp251
        tmp341 = tmp337 + tmp340
        tmp344 = tmp343 * tmp242
        tmp345 = tmp126 + tmp344
        tmp348 = tmp347 * tmp251
        tmp349 = tmp345 + tmp348
        tmp352 = tmp351 * tmp242
        tmp353 = tmp135 + tmp352
        tmp356 = tmp355 * tmp251
        tmp357 = tmp353 + tmp356
        tmp360 = tmp359 * tmp242
        tmp361 = tmp108 + tmp360
        tmp364 = tmp363 * tmp251
        tmp365 = tmp361 + tmp364
        tmp368 = tmp367 * tmp242
        tmp369 = tmp117 + tmp368
        tmp372 = tmp371 * tmp251
        tmp373 = tmp369 + tmp372
        tmp376 = tmp375 * tmp242
        tmp377 = tmp90 + tmp376
        tmp380 = tmp379 * tmp251
        tmp381 = tmp377 + tmp380
        tmp384 = tmp383 * tmp242
        tmp385 = tmp99 + tmp384
        tmp388 = tmp387 * tmp251
        tmp389 = tmp385 + tmp388
        tmp392 = tmp391 * tmp242
        tmp393 = tmp72 + tmp392
        tmp396 = tmp395 * tmp251
        tmp397 = tmp393 + tmp396
        tmp400 = tmp399 * tmp242
        tmp401 = tmp81 + tmp400
        tmp404 = tmp403 * tmp251
        tmp405 = tmp401 + tmp404
        tmp408 = tmp407 * tmp242
        tmp409 = tmp54 + tmp408
        tmp412 = tmp411 * tmp251
        tmp413 = tmp409 + tmp412
        tmp416 = tmp415 * tmp242
        tmp417 = tmp63 + tmp416
        tmp420 = tmp419 * tmp251
        tmp421 = tmp417 + tmp420
        tmp424 = tmp423 * tmp242
        tmp425 = tmp36 + tmp424
        tmp428 = tmp427 * tmp251
        tmp429 = tmp425 + tmp428
        tmp432 = tmp431 * tmp242
        tmp433 = tmp45 + tmp432
        tmp436 = tmp435 * tmp251
        tmp437 = tmp433 + tmp436
        tmp440 = tmp439 * tmp242
        tmp441 = tmp18 + tmp440
        tmp444 = tmp443 * tmp251
        tmp445 = tmp441 + tmp444
        tmp448 = tmp447 * tmp242
        tmp449 = tmp27 + tmp448
        tmp452 = tmp451 * tmp251
        tmp453 = tmp449 + tmp452
        tmp454 = tmp253.to(tl.float32)
        tmp455 = tmp454 * tmp454
        tmp456 = tl.broadcast_to(tmp455, [XBLOCK, RBLOCK])
        tmp458 = _tmp457 + tmp456
        _tmp457 = tl.where(rmask, tmp458, _tmp457)
        tl.store(in_out_ptr25 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp253, rmask)
        tl.store(in_out_ptr22 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp261, rmask)
        tl.store(in_out_ptr23 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp269, rmask)
        tl.store(in_out_ptr24 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp277, rmask)
        tl.store(in_out_ptr20 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp285, rmask)
        tl.store(in_out_ptr21 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp293, rmask)
        tl.store(in_out_ptr18 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp301, rmask)
        tl.store(in_out_ptr19 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp309, rmask)
        tl.store(in_out_ptr16 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp317, rmask)
        tl.store(in_out_ptr17 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp325, rmask)
        tl.store(in_out_ptr14 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp333, rmask)
        tl.store(in_out_ptr15 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp341, rmask)
        tl.store(in_out_ptr12 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp349, rmask)
        tl.store(in_out_ptr13 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp357, rmask)
        tl.store(in_out_ptr10 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp365, rmask)
        tl.store(in_out_ptr11 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp373, rmask)
        tl.store(in_out_ptr8 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp381, rmask)
        tl.store(in_out_ptr9 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp389, rmask)
        tl.store(in_out_ptr6 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp397, rmask)
        tl.store(in_out_ptr7 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp405, rmask)
        tl.store(in_out_ptr4 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp413, rmask)
        tl.store(in_out_ptr5 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp421, rmask)
        tl.store(in_out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp429, rmask)
        tl.store(in_out_ptr3 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp437, rmask)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp445, rmask)
        tl.store(in_out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp453, rmask)
    tmp457 = tl.sum(_tmp457, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp459 = tl.load(in_out_ptr25 + (r0), rmask, other=0).to(tl.float32)
        tmp468 = tl.load(in_ptr38 + (r0), rmask, other=0).to(tl.float32)
        tmp460 = tmp459.to(tl.float32)
        tmp461 = 4096.0
        tmp462 = tmp457 / tmp461
        tmp463 = 1e-05
        tmp464 = tmp462 + tmp463
        tmp465 = tl.math.rsqrt(tmp464)
        tmp466 = tmp460 * tmp465
        tmp467 = tmp466.to(tl.float32)
        tmp469 = tmp467 * tmp468
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp469, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/hr/chrxawqctbtngyzpmt6wpjenumxwq3ysenneln2lgqlrpocp5gva.py
# Source Nodes: [add_60, add_62, add_83, add_84, float_33, mean_16, mul_163, mul_164, mul_165, mul_166, rsqrt_16, type_as_32], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_60 => add_60
# add_62 => add_62
# add_83 => add_83
# add_84 => add_84
# float_33 => convert_element_type_96
# mean_16 => mean_16
# mul_163 => mul_171
# mul_164 => mul_172
# mul_165 => mul_173
# mul_166 => mul_174
# rsqrt_16 => rsqrt_16
# type_as_32 => convert_element_type_97
triton_red_fused__to_copy_add_mean_mul_rsqrt_16 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_16', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_16', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_16(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (7)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
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


# kernel path: /tmp/torchinductor_mengqy/sj/csjnd47csjzj2piloht47qo23qyy7xkmjw6dmu5hlrdt4a4d2tnu.py
# Source Nodes: [add_112, add_113, add_128, add_129, add_60, add_62, add_73, add_75, add_97, add_98, add_99, float_37, mean_18, mul_187, mul_188, mul_189, mul_190, mul_191, mul_212, mul_213, mul_238, mul_239, rsqrt_18, type_as_36], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_112 => add_112
# add_113 => add_113
# add_128 => add_128
# add_129 => add_129
# add_60 => add_60
# add_62 => add_62
# add_73 => add_73
# add_75 => add_75
# add_97 => add_97
# add_98 => add_98
# add_99 => add_99
# float_37 => convert_element_type_108
# mean_18 => mean_18
# mul_187 => mul_196
# mul_188 => mul_197
# mul_189 => mul_198
# mul_190 => mul_199
# mul_191 => mul_200
# mul_212 => mul_222
# mul_213 => mul_223
# mul_238 => mul_249
# mul_239 => mul_250
# rsqrt_18 => rsqrt_18
# type_as_36 => convert_element_type_109
triton_red_fused__to_copy_add_mean_mul_rsqrt_17 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_17', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: 'i32', 15: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_17', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 15), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(15,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_17(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (7)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp10 = tl.load(in_ptr0 + (8)).to(tl.float32)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp20 = tl.load(in_ptr7 + (7)).to(tl.float32)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp24 = tl.load(in_ptr7 + (8)).to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp29 = tl.load(in_ptr8 + (7)).to(tl.float32)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    tmp33 = tl.load(in_ptr8 + (8)).to(tl.float32)
    tmp34 = tl.broadcast_to(tmp33, [XBLOCK, RBLOCK])
    _tmp40 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr5 + (r0), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr6 + (r0), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_out_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp28 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp11 * tmp16
        tmp18 = tmp9 + tmp17
        tmp22 = tmp21 * tmp7
        tmp23 = tmp19 + tmp22
        tmp26 = tmp25 * tmp16
        tmp27 = tmp23 + tmp26
        tmp31 = tmp30 * tmp7
        tmp32 = tmp28 + tmp31
        tmp35 = tmp34 * tmp16
        tmp36 = tmp32 + tmp35
        tmp37 = tmp18.to(tl.float32)
        tmp38 = tmp37 * tmp37
        tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
        tmp41 = _tmp40 + tmp39
        _tmp40 = tl.where(rmask, tmp41, _tmp40)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp18, rmask)
        tl.store(in_out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp27, rmask)
        tl.store(in_out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp36, rmask)
    tmp40 = tl.sum(_tmp40, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp42 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp51 = tl.load(in_ptr9 + (r0), rmask, other=0).to(tl.float32)
        tmp43 = tmp42.to(tl.float32)
        tmp44 = 4096.0
        tmp45 = tmp40 / tmp44
        tmp46 = 1e-05
        tmp47 = tmp45 + tmp46
        tmp48 = tl.math.rsqrt(tmp47)
        tmp49 = tmp43 * tmp48
        tmp50 = tmp49.to(tl.float32)
        tmp52 = tmp50 * tmp51
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp52, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/o2/co2tqsxyvygdzx6ge36emest6jw3tkw353vo4oef2zn7qbqwxukg.py
# Source Nodes: [add_114, add_115, add_87, add_89, float_41, mean_20, mul_214, mul_215, mul_216, mul_217, rsqrt_20, type_as_40], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_114 => add_114
# add_115 => add_115
# add_87 => add_87
# add_89 => add_89
# float_41 => convert_element_type_120
# mean_20 => mean_20
# mul_214 => mul_224
# mul_215 => mul_225
# mul_216 => mul_226
# mul_217 => mul_227
# rsqrt_20 => rsqrt_20
# type_as_40 => convert_element_type_121
triton_red_fused__to_copy_add_mean_mul_rsqrt_18 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_18', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_18', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_18(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (9)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
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


# kernel path: /tmp/torchinductor_mengqy/ng/cngwbeotgg4svrltrc6nbww2vr4f42dlxflzkiisny6t2jbu5r5r.py
# Source Nodes: [add_102, add_104, add_130, add_131, add_132, add_145, add_146, add_147, add_148, add_163, add_164, add_165, add_166, add_60, add_62, add_73, add_75, add_87, add_89, float_45, mean_22, mul_240, mul_241, mul_242, mul_243, mul_244, mul_265, mul_266, mul_267, mul_268, mul_293, mul_294, mul_295, mul_296, rsqrt_22, type_as_44], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_102 => add_102
# add_104 => add_104
# add_130 => add_130
# add_131 => add_131
# add_132 => add_132
# add_145 => add_145
# add_146 => add_146
# add_147 => add_147
# add_148 => add_148
# add_163 => add_163
# add_164 => add_164
# add_165 => add_165
# add_166 => add_166
# add_60 => add_60
# add_62 => add_62
# add_73 => add_73
# add_75 => add_75
# add_87 => add_87
# add_89 => add_89
# float_45 => convert_element_type_132
# mean_22 => mean_22
# mul_240 => mul_251
# mul_241 => mul_252
# mul_242 => mul_253
# mul_243 => mul_254
# mul_244 => mul_255
# mul_265 => mul_277
# mul_266 => mul_278
# mul_267 => mul_279
# mul_268 => mul_280
# mul_293 => mul_306
# mul_294 => mul_307
# mul_295 => mul_308
# mul_296 => mul_309
# rsqrt_22 => rsqrt_22
# type_as_44 => convert_element_type_133
triton_red_fused__to_copy_add_mean_mul_rsqrt_19 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_19', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: 'i32', 21: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_19', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 21), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(21,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_19(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (7)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp10 = tl.load(in_ptr0 + (8)).to(tl.float32)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp20 = tl.load(in_ptr7 + (7)).to(tl.float32)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp24 = tl.load(in_ptr7 + (8)).to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp29 = tl.load(in_ptr8 + (9)).to(tl.float32)
    tmp30 = tl.broadcast_to(tmp29, [XBLOCK, RBLOCK])
    tmp38 = tl.load(in_ptr8 + (10)).to(tl.float32)
    tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
    tmp47 = tl.load(in_ptr0 + (9)).to(tl.float32)
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK, RBLOCK])
    tmp51 = tl.load(in_ptr0 + (10)).to(tl.float32)
    tmp52 = tl.broadcast_to(tmp51, [XBLOCK, RBLOCK])
    tmp55 = tl.load(in_ptr7 + (9)).to(tl.float32)
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK, RBLOCK])
    tmp59 = tl.load(in_ptr7 + (10)).to(tl.float32)
    tmp60 = tl.broadcast_to(tmp59, [XBLOCK, RBLOCK])
    _tmp66 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr5 + (r0), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr6 + (r0), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_out_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp28 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp31 = tl.load(in_ptr9 + (r0), rmask, other=0).to(tl.float32)
        tmp32 = tl.load(in_ptr10 + (r0), rmask, other=0).to(tl.float32)
        tmp34 = tl.load(in_ptr11 + (r0), rmask, other=0).to(tl.float32)
        tmp40 = tl.load(in_ptr12 + (r0), rmask, other=0).to(tl.float32)
        tmp41 = tl.load(in_ptr13 + (r0), rmask, other=0).to(tl.float32)
        tmp43 = tl.load(in_ptr14 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp11 * tmp16
        tmp18 = tmp9 + tmp17
        tmp22 = tmp21 * tmp7
        tmp23 = tmp19 + tmp22
        tmp26 = tmp25 * tmp16
        tmp27 = tmp23 + tmp26
        tmp33 = tmp31 + tmp32
        tmp35 = tmp33 + tmp34
        tmp36 = tmp30 * tmp35
        tmp37 = tmp28 + tmp36
        tmp42 = tmp40 + tmp41
        tmp44 = tmp42 + tmp43
        tmp45 = tmp39 * tmp44
        tmp46 = tmp37 + tmp45
        tmp49 = tmp48 * tmp35
        tmp50 = tmp18 + tmp49
        tmp53 = tmp52 * tmp44
        tmp54 = tmp50 + tmp53
        tmp57 = tmp56 * tmp35
        tmp58 = tmp27 + tmp57
        tmp61 = tmp60 * tmp44
        tmp62 = tmp58 + tmp61
        tmp63 = tmp46.to(tl.float32)
        tmp64 = tmp63 * tmp63
        tmp65 = tl.broadcast_to(tmp64, [XBLOCK, RBLOCK])
        tmp67 = _tmp66 + tmp65
        _tmp66 = tl.where(rmask, tmp67, _tmp66)
        tl.store(in_out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp46, rmask)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp54, rmask)
        tl.store(in_out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp62, rmask)
    tmp66 = tl.sum(_tmp66, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp68 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp77 = tl.load(in_ptr15 + (r0), rmask, other=0).to(tl.float32)
        tmp69 = tmp68.to(tl.float32)
        tmp70 = 4096.0
        tmp71 = tmp66 / tmp70
        tmp72 = 1e-05
        tmp73 = tmp71 + tmp72
        tmp74 = tl.math.rsqrt(tmp73)
        tmp75 = tmp69 * tmp74
        tmp76 = tmp75.to(tl.float32)
        tmp78 = tmp76 * tmp77
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp78, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/sw/csw3tir63ku6labvlaloazvl6vriil6r6luajj3yjnt64sqq63wg.py
# Source Nodes: [add_118, add_120, add_149, add_150, float_49, mean_24, mul_269, mul_270, mul_271, mul_272, rsqrt_24, type_as_48], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_118 => add_118
# add_120 => add_120
# add_149 => add_149
# add_150 => add_150
# float_49 => convert_element_type_144
# mean_24 => mean_24
# mul_269 => mul_281
# mul_270 => mul_282
# mul_271 => mul_283
# mul_272 => mul_284
# rsqrt_24 => rsqrt_24
# type_as_48 => convert_element_type_145
triton_red_fused__to_copy_add_mean_mul_rsqrt_20 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_20', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_20', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (11)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
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


# kernel path: /tmp/torchinductor_mengqy/we/cwer7wsvyfvi4ckdef74pjas4a7auifxlcmg2vwgjs3totwvaptp.py
# Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_167, add_168, add_169, add_182, add_183, add_184, add_185, add_186, add_187, add_202, add_203, add_204, add_205, add_206, add_207, add_60, add_62, add_73, add_75, add_87, add_89, float_53, mean_26, mul_297, mul_298, mul_299, mul_300, mul_301, mul_322, mul_323, mul_324, mul_325, mul_326, mul_327, mul_352, mul_353, mul_354, mul_355, mul_356, mul_357, rsqrt_26, type_as_52], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_102 => add_102
# add_104 => add_104
# add_118 => add_118
# add_120 => add_120
# add_135 => add_135
# add_137 => add_137
# add_167 => add_167
# add_168 => add_168
# add_169 => add_169
# add_182 => add_182
# add_183 => add_183
# add_184 => add_184
# add_185 => add_185
# add_186 => add_186
# add_187 => add_187
# add_202 => add_202
# add_203 => add_203
# add_204 => add_204
# add_205 => add_205
# add_206 => add_206
# add_207 => add_207
# add_60 => add_60
# add_62 => add_62
# add_73 => add_73
# add_75 => add_75
# add_87 => add_87
# add_89 => add_89
# float_53 => convert_element_type_156
# mean_26 => mean_26
# mul_297 => mul_310
# mul_298 => mul_311
# mul_299 => mul_312
# mul_300 => mul_313
# mul_301 => mul_314
# mul_322 => mul_336
# mul_323 => mul_337
# mul_324 => mul_338
# mul_325 => mul_339
# mul_326 => mul_340
# mul_327 => mul_341
# mul_352 => mul_367
# mul_353 => mul_368
# mul_354 => mul_369
# mul_355 => mul_370
# mul_356 => mul_371
# mul_357 => mul_372
# rsqrt_26 => rsqrt_26
# type_as_52 => convert_element_type_157
triton_red_fused__to_copy_add_mean_mul_rsqrt_21 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_21', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: '*fp16', 21: '*fp16', 22: '*fp16', 23: '*fp16', 24: '*fp16', 25: '*fp16', 26: 'i32', 27: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_21', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(27,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_21(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (7)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp10 = tl.load(in_ptr0 + (8)).to(tl.float32)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp20 = tl.load(in_ptr7 + (7)).to(tl.float32)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp24 = tl.load(in_ptr7 + (8)).to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp28 = tl.load(in_ptr0 + (9)).to(tl.float32)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp37 = tl.load(in_ptr0 + (10)).to(tl.float32)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
    tmp46 = tl.load(in_ptr7 + (9)).to(tl.float32)
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
    tmp50 = tl.load(in_ptr7 + (10)).to(tl.float32)
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
    tmp55 = tl.load(in_ptr14 + (11)).to(tl.float32)
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK, RBLOCK])
    tmp64 = tl.load(in_ptr14 + (12)).to(tl.float32)
    tmp65 = tl.broadcast_to(tmp64, [XBLOCK, RBLOCK])
    tmp73 = tl.load(in_ptr0 + (11)).to(tl.float32)
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK, RBLOCK])
    tmp77 = tl.load(in_ptr0 + (12)).to(tl.float32)
    tmp78 = tl.broadcast_to(tmp77, [XBLOCK, RBLOCK])
    tmp81 = tl.load(in_ptr7 + (11)).to(tl.float32)
    tmp82 = tl.broadcast_to(tmp81, [XBLOCK, RBLOCK])
    tmp85 = tl.load(in_ptr7 + (12)).to(tl.float32)
    tmp86 = tl.broadcast_to(tmp85, [XBLOCK, RBLOCK])
    _tmp92 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr5 + (r0), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr6 + (r0), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_out_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp30 = tl.load(in_ptr8 + (r0), rmask, other=0).to(tl.float32)
        tmp31 = tl.load(in_ptr9 + (r0), rmask, other=0).to(tl.float32)
        tmp33 = tl.load(in_ptr10 + (r0), rmask, other=0).to(tl.float32)
        tmp39 = tl.load(in_ptr11 + (r0), rmask, other=0).to(tl.float32)
        tmp40 = tl.load(in_ptr12 + (r0), rmask, other=0).to(tl.float32)
        tmp42 = tl.load(in_ptr13 + (r0), rmask, other=0).to(tl.float32)
        tmp54 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp57 = tl.load(in_ptr15 + (r0), rmask, other=0).to(tl.float32)
        tmp58 = tl.load(in_ptr16 + (r0), rmask, other=0).to(tl.float32)
        tmp60 = tl.load(in_ptr17 + (r0), rmask, other=0).to(tl.float32)
        tmp66 = tl.load(in_ptr18 + (r0), rmask, other=0).to(tl.float32)
        tmp67 = tl.load(in_ptr19 + (r0), rmask, other=0).to(tl.float32)
        tmp69 = tl.load(in_ptr20 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp11 * tmp16
        tmp18 = tmp9 + tmp17
        tmp22 = tmp21 * tmp7
        tmp23 = tmp19 + tmp22
        tmp26 = tmp25 * tmp16
        tmp27 = tmp23 + tmp26
        tmp32 = tmp30 + tmp31
        tmp34 = tmp32 + tmp33
        tmp35 = tmp29 * tmp34
        tmp36 = tmp18 + tmp35
        tmp41 = tmp39 + tmp40
        tmp43 = tmp41 + tmp42
        tmp44 = tmp38 * tmp43
        tmp45 = tmp36 + tmp44
        tmp48 = tmp47 * tmp34
        tmp49 = tmp27 + tmp48
        tmp52 = tmp51 * tmp43
        tmp53 = tmp49 + tmp52
        tmp59 = tmp57 + tmp58
        tmp61 = tmp59 + tmp60
        tmp62 = tmp56 * tmp61
        tmp63 = tmp54 + tmp62
        tmp68 = tmp66 + tmp67
        tmp70 = tmp68 + tmp69
        tmp71 = tmp65 * tmp70
        tmp72 = tmp63 + tmp71
        tmp75 = tmp74 * tmp61
        tmp76 = tmp45 + tmp75
        tmp79 = tmp78 * tmp70
        tmp80 = tmp76 + tmp79
        tmp83 = tmp82 * tmp61
        tmp84 = tmp53 + tmp83
        tmp87 = tmp86 * tmp70
        tmp88 = tmp84 + tmp87
        tmp89 = tmp72.to(tl.float32)
        tmp90 = tmp89 * tmp89
        tmp91 = tl.broadcast_to(tmp90, [XBLOCK, RBLOCK])
        tmp93 = _tmp92 + tmp91
        _tmp92 = tl.where(rmask, tmp93, _tmp92)
        tl.store(in_out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp72, rmask)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp80, rmask)
        tl.store(in_out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp88, rmask)
    tmp92 = tl.sum(_tmp92, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp94 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp103 = tl.load(in_ptr21 + (r0), rmask, other=0).to(tl.float32)
        tmp95 = tmp94.to(tl.float32)
        tmp96 = 4096.0
        tmp97 = tmp92 / tmp96
        tmp98 = 1e-05
        tmp99 = tmp97 + tmp98
        tmp100 = tl.math.rsqrt(tmp99)
        tmp101 = tmp95 * tmp100
        tmp102 = tmp101.to(tl.float32)
        tmp104 = tmp102 * tmp103
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp104, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/ql/cqlvfxtjpwjhydsd6p6pibiheq3zgej4jayy3uu6acbgcgiazvon.py
# Source Nodes: [add_153, add_155, add_188, add_189, float_57, mean_28, mul_328, mul_329, mul_330, mul_331, rsqrt_28, type_as_56], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_153 => add_153
# add_155 => add_155
# add_188 => add_188
# add_189 => add_189
# float_57 => convert_element_type_168
# mean_28 => mean_28
# mul_328 => mul_342
# mul_329 => mul_343
# mul_330 => mul_344
# mul_331 => mul_345
# rsqrt_28 => rsqrt_28
# type_as_56 => convert_element_type_169
triton_red_fused__to_copy_add_mean_mul_rsqrt_22 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_22', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_22', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_22(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (13)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
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


# kernel path: /tmp/torchinductor_mengqy/5i/c5iaojngcoqixwknikra6zqtcqrwechtcuccrg35zggx332ap5p6.py
# Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_153, add_155, add_172, add_174, add_208, add_209, add_210, add_223, add_224, add_225, add_226, add_227, add_228, add_229, add_230, add_245, add_246, add_247, add_248, add_249, add_250, add_251, add_252, add_60, add_62, add_73, add_75, add_87, add_89, float_61, mean_30, mul_358, mul_359, mul_360, mul_361, mul_362, mul_383, mul_384, mul_385, mul_386, mul_387, mul_388, mul_389, mul_390, mul_415, mul_416, mul_417, mul_418, mul_419, mul_420, mul_421, mul_422, rsqrt_30, type_as_60], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_102 => add_102
# add_104 => add_104
# add_118 => add_118
# add_120 => add_120
# add_135 => add_135
# add_137 => add_137
# add_153 => add_153
# add_155 => add_155
# add_172 => add_172
# add_174 => add_174
# add_208 => add_208
# add_209 => add_209
# add_210 => add_210
# add_223 => add_223
# add_224 => add_224
# add_225 => add_225
# add_226 => add_226
# add_227 => add_227
# add_228 => add_228
# add_229 => add_229
# add_230 => add_230
# add_245 => add_245
# add_246 => add_246
# add_247 => add_247
# add_248 => add_248
# add_249 => add_249
# add_250 => add_250
# add_251 => add_251
# add_252 => add_252
# add_60 => add_60
# add_62 => add_62
# add_73 => add_73
# add_75 => add_75
# add_87 => add_87
# add_89 => add_89
# float_61 => convert_element_type_180
# mean_30 => mean_30
# mul_358 => mul_373
# mul_359 => mul_374
# mul_360 => mul_375
# mul_361 => mul_376
# mul_362 => mul_377
# mul_383 => mul_399
# mul_384 => mul_400
# mul_385 => mul_401
# mul_386 => mul_402
# mul_387 => mul_403
# mul_388 => mul_404
# mul_389 => mul_405
# mul_390 => mul_406
# mul_415 => mul_432
# mul_416 => mul_433
# mul_417 => mul_434
# mul_418 => mul_435
# mul_419 => mul_436
# mul_420 => mul_437
# mul_421 => mul_438
# mul_422 => mul_439
# rsqrt_30 => rsqrt_30
# type_as_60 => convert_element_type_181
triton_red_fused__to_copy_add_mean_mul_rsqrt_23 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_23', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: '*fp16', 21: '*fp16', 22: '*fp16', 23: '*fp16', 24: '*fp16', 25: '*fp16', 26: '*fp16', 27: '*fp16', 28: '*fp16', 29: '*fp16', 30: '*fp16', 31: '*fp16', 32: 'i32', 33: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_23', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 33), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(33,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_23(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (7)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp10 = tl.load(in_ptr0 + (8)).to(tl.float32)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp20 = tl.load(in_ptr7 + (7)).to(tl.float32)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp24 = tl.load(in_ptr7 + (8)).to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp28 = tl.load(in_ptr0 + (9)).to(tl.float32)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp37 = tl.load(in_ptr0 + (10)).to(tl.float32)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
    tmp46 = tl.load(in_ptr7 + (9)).to(tl.float32)
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
    tmp50 = tl.load(in_ptr7 + (10)).to(tl.float32)
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
    tmp54 = tl.load(in_ptr0 + (11)).to(tl.float32)
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK, RBLOCK])
    tmp63 = tl.load(in_ptr0 + (12)).to(tl.float32)
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, RBLOCK])
    tmp72 = tl.load(in_ptr7 + (11)).to(tl.float32)
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp76 = tl.load(in_ptr7 + (12)).to(tl.float32)
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK, RBLOCK])
    tmp81 = tl.load(in_ptr20 + (13)).to(tl.float32)
    tmp82 = tl.broadcast_to(tmp81, [XBLOCK, RBLOCK])
    tmp90 = tl.load(in_ptr20 + (14)).to(tl.float32)
    tmp91 = tl.broadcast_to(tmp90, [XBLOCK, RBLOCK])
    tmp99 = tl.load(in_ptr0 + (13)).to(tl.float32)
    tmp100 = tl.broadcast_to(tmp99, [XBLOCK, RBLOCK])
    tmp103 = tl.load(in_ptr0 + (14)).to(tl.float32)
    tmp104 = tl.broadcast_to(tmp103, [XBLOCK, RBLOCK])
    tmp107 = tl.load(in_ptr7 + (13)).to(tl.float32)
    tmp108 = tl.broadcast_to(tmp107, [XBLOCK, RBLOCK])
    tmp111 = tl.load(in_ptr7 + (14)).to(tl.float32)
    tmp112 = tl.broadcast_to(tmp111, [XBLOCK, RBLOCK])
    _tmp118 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr5 + (r0), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr6 + (r0), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_out_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp30 = tl.load(in_ptr8 + (r0), rmask, other=0).to(tl.float32)
        tmp31 = tl.load(in_ptr9 + (r0), rmask, other=0).to(tl.float32)
        tmp33 = tl.load(in_ptr10 + (r0), rmask, other=0).to(tl.float32)
        tmp39 = tl.load(in_ptr11 + (r0), rmask, other=0).to(tl.float32)
        tmp40 = tl.load(in_ptr12 + (r0), rmask, other=0).to(tl.float32)
        tmp42 = tl.load(in_ptr13 + (r0), rmask, other=0).to(tl.float32)
        tmp56 = tl.load(in_ptr14 + (r0), rmask, other=0).to(tl.float32)
        tmp57 = tl.load(in_ptr15 + (r0), rmask, other=0).to(tl.float32)
        tmp59 = tl.load(in_ptr16 + (r0), rmask, other=0).to(tl.float32)
        tmp65 = tl.load(in_ptr17 + (r0), rmask, other=0).to(tl.float32)
        tmp66 = tl.load(in_ptr18 + (r0), rmask, other=0).to(tl.float32)
        tmp68 = tl.load(in_ptr19 + (r0), rmask, other=0).to(tl.float32)
        tmp80 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp83 = tl.load(in_ptr21 + (r0), rmask, other=0).to(tl.float32)
        tmp84 = tl.load(in_ptr22 + (r0), rmask, other=0).to(tl.float32)
        tmp86 = tl.load(in_ptr23 + (r0), rmask, other=0).to(tl.float32)
        tmp92 = tl.load(in_ptr24 + (r0), rmask, other=0).to(tl.float32)
        tmp93 = tl.load(in_ptr25 + (r0), rmask, other=0).to(tl.float32)
        tmp95 = tl.load(in_ptr26 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp11 * tmp16
        tmp18 = tmp9 + tmp17
        tmp22 = tmp21 * tmp7
        tmp23 = tmp19 + tmp22
        tmp26 = tmp25 * tmp16
        tmp27 = tmp23 + tmp26
        tmp32 = tmp30 + tmp31
        tmp34 = tmp32 + tmp33
        tmp35 = tmp29 * tmp34
        tmp36 = tmp18 + tmp35
        tmp41 = tmp39 + tmp40
        tmp43 = tmp41 + tmp42
        tmp44 = tmp38 * tmp43
        tmp45 = tmp36 + tmp44
        tmp48 = tmp47 * tmp34
        tmp49 = tmp27 + tmp48
        tmp52 = tmp51 * tmp43
        tmp53 = tmp49 + tmp52
        tmp58 = tmp56 + tmp57
        tmp60 = tmp58 + tmp59
        tmp61 = tmp55 * tmp60
        tmp62 = tmp45 + tmp61
        tmp67 = tmp65 + tmp66
        tmp69 = tmp67 + tmp68
        tmp70 = tmp64 * tmp69
        tmp71 = tmp62 + tmp70
        tmp74 = tmp73 * tmp60
        tmp75 = tmp53 + tmp74
        tmp78 = tmp77 * tmp69
        tmp79 = tmp75 + tmp78
        tmp85 = tmp83 + tmp84
        tmp87 = tmp85 + tmp86
        tmp88 = tmp82 * tmp87
        tmp89 = tmp80 + tmp88
        tmp94 = tmp92 + tmp93
        tmp96 = tmp94 + tmp95
        tmp97 = tmp91 * tmp96
        tmp98 = tmp89 + tmp97
        tmp101 = tmp100 * tmp87
        tmp102 = tmp71 + tmp101
        tmp105 = tmp104 * tmp96
        tmp106 = tmp102 + tmp105
        tmp109 = tmp108 * tmp87
        tmp110 = tmp79 + tmp109
        tmp113 = tmp112 * tmp96
        tmp114 = tmp110 + tmp113
        tmp115 = tmp98.to(tl.float32)
        tmp116 = tmp115 * tmp115
        tmp117 = tl.broadcast_to(tmp116, [XBLOCK, RBLOCK])
        tmp119 = _tmp118 + tmp117
        _tmp118 = tl.where(rmask, tmp119, _tmp118)
        tl.store(in_out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp98, rmask)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp106, rmask)
        tl.store(in_out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp114, rmask)
    tmp118 = tl.sum(_tmp118, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp120 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp129 = tl.load(in_ptr27 + (r0), rmask, other=0).to(tl.float32)
        tmp121 = tmp120.to(tl.float32)
        tmp122 = 4096.0
        tmp123 = tmp118 / tmp122
        tmp124 = 1e-05
        tmp125 = tmp123 + tmp124
        tmp126 = tl.math.rsqrt(tmp125)
        tmp127 = tmp121 * tmp126
        tmp128 = tmp127.to(tl.float32)
        tmp130 = tmp128 * tmp129
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp130, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/hl/chlfdc4czolhio2cvxxu6ihw6vos6mzmg5vnaxtaaubq4ewy7zy4.py
# Source Nodes: [add_192, add_194, add_231, add_232, float_65, mean_32, mul_391, mul_392, mul_393, mul_394, rsqrt_32, type_as_64], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_192 => add_192
# add_194 => add_194
# add_231 => add_231
# add_232 => add_232
# float_65 => convert_element_type_192
# mean_32 => mean_32
# mul_391 => mul_407
# mul_392 => mul_408
# mul_393 => mul_409
# mul_394 => mul_410
# rsqrt_32 => rsqrt_32
# type_as_64 => convert_element_type_193
triton_red_fused__to_copy_add_mean_mul_rsqrt_24 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_24', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_24', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_24(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (15)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
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


# kernel path: /tmp/torchinductor_mengqy/o6/co6av5s7lbvjy7bae4r5pwmbmwqdvcnucalfk2tp32h6oidiskmc.py
# Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_153, add_155, add_172, add_174, add_192, add_194, add_213, add_215, add_253, add_254, add_255, add_268, add_269, add_270, add_271, add_272, add_273, add_274, add_275, add_276, add_277, add_292, add_293, add_294, add_295, add_296, add_297, add_298, add_299, add_300, add_301, add_60, add_62, add_73, add_75, add_87, add_89, float_69, mean_34, mul_423, mul_424, mul_425, mul_426, mul_427, mul_448, mul_449, mul_450, mul_451, mul_452, mul_453, mul_454, mul_455, mul_456, mul_457, mul_482, mul_483, mul_484, mul_485, mul_486, mul_487, mul_488, mul_489, mul_490, mul_491, rsqrt_34, type_as_68], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_102 => add_102
# add_104 => add_104
# add_118 => add_118
# add_120 => add_120
# add_135 => add_135
# add_137 => add_137
# add_153 => add_153
# add_155 => add_155
# add_172 => add_172
# add_174 => add_174
# add_192 => add_192
# add_194 => add_194
# add_213 => add_213
# add_215 => add_215
# add_253 => add_253
# add_254 => add_254
# add_255 => add_255
# add_268 => add_268
# add_269 => add_269
# add_270 => add_270
# add_271 => add_271
# add_272 => add_272
# add_273 => add_273
# add_274 => add_274
# add_275 => add_275
# add_276 => add_276
# add_277 => add_277
# add_292 => add_292
# add_293 => add_293
# add_294 => add_294
# add_295 => add_295
# add_296 => add_296
# add_297 => add_297
# add_298 => add_298
# add_299 => add_299
# add_300 => add_300
# add_301 => add_301
# add_60 => add_60
# add_62 => add_62
# add_73 => add_73
# add_75 => add_75
# add_87 => add_87
# add_89 => add_89
# float_69 => convert_element_type_204
# mean_34 => mean_34
# mul_423 => mul_440
# mul_424 => mul_441
# mul_425 => mul_442
# mul_426 => mul_443
# mul_427 => mul_444
# mul_448 => mul_466
# mul_449 => mul_467
# mul_450 => mul_468
# mul_451 => mul_469
# mul_452 => mul_470
# mul_453 => mul_471
# mul_454 => mul_472
# mul_455 => mul_473
# mul_456 => mul_474
# mul_457 => mul_475
# mul_482 => mul_501
# mul_483 => mul_502
# mul_484 => mul_503
# mul_485 => mul_504
# mul_486 => mul_505
# mul_487 => mul_506
# mul_488 => mul_507
# mul_489 => mul_508
# mul_490 => mul_509
# mul_491 => mul_510
# rsqrt_34 => rsqrt_34
# type_as_68 => convert_element_type_205
triton_red_fused__to_copy_add_mean_mul_rsqrt_25 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_25', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: '*fp16', 21: '*fp16', 22: '*fp16', 23: '*fp16', 24: '*fp16', 25: '*fp16', 26: '*fp16', 27: '*fp16', 28: '*fp16', 29: '*fp16', 30: '*fp16', 31: '*fp16', 32: '*fp16', 33: '*fp16', 34: '*fp16', 35: '*fp16', 36: '*fp16', 37: '*fp16', 38: 'i32', 39: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_25', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 39), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(39,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_25(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (7)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp10 = tl.load(in_ptr0 + (8)).to(tl.float32)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp20 = tl.load(in_ptr7 + (7)).to(tl.float32)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp24 = tl.load(in_ptr7 + (8)).to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp28 = tl.load(in_ptr0 + (9)).to(tl.float32)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp37 = tl.load(in_ptr0 + (10)).to(tl.float32)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
    tmp46 = tl.load(in_ptr7 + (9)).to(tl.float32)
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
    tmp50 = tl.load(in_ptr7 + (10)).to(tl.float32)
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
    tmp54 = tl.load(in_ptr0 + (11)).to(tl.float32)
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK, RBLOCK])
    tmp63 = tl.load(in_ptr0 + (12)).to(tl.float32)
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, RBLOCK])
    tmp72 = tl.load(in_ptr7 + (11)).to(tl.float32)
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp76 = tl.load(in_ptr7 + (12)).to(tl.float32)
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK, RBLOCK])
    tmp80 = tl.load(in_ptr0 + (13)).to(tl.float32)
    tmp81 = tl.broadcast_to(tmp80, [XBLOCK, RBLOCK])
    tmp89 = tl.load(in_ptr0 + (14)).to(tl.float32)
    tmp90 = tl.broadcast_to(tmp89, [XBLOCK, RBLOCK])
    tmp98 = tl.load(in_ptr7 + (13)).to(tl.float32)
    tmp99 = tl.broadcast_to(tmp98, [XBLOCK, RBLOCK])
    tmp102 = tl.load(in_ptr7 + (14)).to(tl.float32)
    tmp103 = tl.broadcast_to(tmp102, [XBLOCK, RBLOCK])
    tmp107 = tl.load(in_ptr26 + (15)).to(tl.float32)
    tmp108 = tl.broadcast_to(tmp107, [XBLOCK, RBLOCK])
    tmp116 = tl.load(in_ptr26 + (16)).to(tl.float32)
    tmp117 = tl.broadcast_to(tmp116, [XBLOCK, RBLOCK])
    tmp125 = tl.load(in_ptr0 + (15)).to(tl.float32)
    tmp126 = tl.broadcast_to(tmp125, [XBLOCK, RBLOCK])
    tmp129 = tl.load(in_ptr0 + (16)).to(tl.float32)
    tmp130 = tl.broadcast_to(tmp129, [XBLOCK, RBLOCK])
    tmp133 = tl.load(in_ptr7 + (15)).to(tl.float32)
    tmp134 = tl.broadcast_to(tmp133, [XBLOCK, RBLOCK])
    tmp137 = tl.load(in_ptr7 + (16)).to(tl.float32)
    tmp138 = tl.broadcast_to(tmp137, [XBLOCK, RBLOCK])
    _tmp144 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr5 + (r0), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr6 + (r0), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_out_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp30 = tl.load(in_ptr8 + (r0), rmask, other=0).to(tl.float32)
        tmp31 = tl.load(in_ptr9 + (r0), rmask, other=0).to(tl.float32)
        tmp33 = tl.load(in_ptr10 + (r0), rmask, other=0).to(tl.float32)
        tmp39 = tl.load(in_ptr11 + (r0), rmask, other=0).to(tl.float32)
        tmp40 = tl.load(in_ptr12 + (r0), rmask, other=0).to(tl.float32)
        tmp42 = tl.load(in_ptr13 + (r0), rmask, other=0).to(tl.float32)
        tmp56 = tl.load(in_ptr14 + (r0), rmask, other=0).to(tl.float32)
        tmp57 = tl.load(in_ptr15 + (r0), rmask, other=0).to(tl.float32)
        tmp59 = tl.load(in_ptr16 + (r0), rmask, other=0).to(tl.float32)
        tmp65 = tl.load(in_ptr17 + (r0), rmask, other=0).to(tl.float32)
        tmp66 = tl.load(in_ptr18 + (r0), rmask, other=0).to(tl.float32)
        tmp68 = tl.load(in_ptr19 + (r0), rmask, other=0).to(tl.float32)
        tmp82 = tl.load(in_ptr20 + (r0), rmask, other=0).to(tl.float32)
        tmp83 = tl.load(in_ptr21 + (r0), rmask, other=0).to(tl.float32)
        tmp85 = tl.load(in_ptr22 + (r0), rmask, other=0).to(tl.float32)
        tmp91 = tl.load(in_ptr23 + (r0), rmask, other=0).to(tl.float32)
        tmp92 = tl.load(in_ptr24 + (r0), rmask, other=0).to(tl.float32)
        tmp94 = tl.load(in_ptr25 + (r0), rmask, other=0).to(tl.float32)
        tmp106 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp109 = tl.load(in_ptr27 + (r0), rmask, other=0).to(tl.float32)
        tmp110 = tl.load(in_ptr28 + (r0), rmask, other=0).to(tl.float32)
        tmp112 = tl.load(in_ptr29 + (r0), rmask, other=0).to(tl.float32)
        tmp118 = tl.load(in_ptr30 + (r0), rmask, other=0).to(tl.float32)
        tmp119 = tl.load(in_ptr31 + (r0), rmask, other=0).to(tl.float32)
        tmp121 = tl.load(in_ptr32 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp11 * tmp16
        tmp18 = tmp9 + tmp17
        tmp22 = tmp21 * tmp7
        tmp23 = tmp19 + tmp22
        tmp26 = tmp25 * tmp16
        tmp27 = tmp23 + tmp26
        tmp32 = tmp30 + tmp31
        tmp34 = tmp32 + tmp33
        tmp35 = tmp29 * tmp34
        tmp36 = tmp18 + tmp35
        tmp41 = tmp39 + tmp40
        tmp43 = tmp41 + tmp42
        tmp44 = tmp38 * tmp43
        tmp45 = tmp36 + tmp44
        tmp48 = tmp47 * tmp34
        tmp49 = tmp27 + tmp48
        tmp52 = tmp51 * tmp43
        tmp53 = tmp49 + tmp52
        tmp58 = tmp56 + tmp57
        tmp60 = tmp58 + tmp59
        tmp61 = tmp55 * tmp60
        tmp62 = tmp45 + tmp61
        tmp67 = tmp65 + tmp66
        tmp69 = tmp67 + tmp68
        tmp70 = tmp64 * tmp69
        tmp71 = tmp62 + tmp70
        tmp74 = tmp73 * tmp60
        tmp75 = tmp53 + tmp74
        tmp78 = tmp77 * tmp69
        tmp79 = tmp75 + tmp78
        tmp84 = tmp82 + tmp83
        tmp86 = tmp84 + tmp85
        tmp87 = tmp81 * tmp86
        tmp88 = tmp71 + tmp87
        tmp93 = tmp91 + tmp92
        tmp95 = tmp93 + tmp94
        tmp96 = tmp90 * tmp95
        tmp97 = tmp88 + tmp96
        tmp100 = tmp99 * tmp86
        tmp101 = tmp79 + tmp100
        tmp104 = tmp103 * tmp95
        tmp105 = tmp101 + tmp104
        tmp111 = tmp109 + tmp110
        tmp113 = tmp111 + tmp112
        tmp114 = tmp108 * tmp113
        tmp115 = tmp106 + tmp114
        tmp120 = tmp118 + tmp119
        tmp122 = tmp120 + tmp121
        tmp123 = tmp117 * tmp122
        tmp124 = tmp115 + tmp123
        tmp127 = tmp126 * tmp113
        tmp128 = tmp97 + tmp127
        tmp131 = tmp130 * tmp122
        tmp132 = tmp128 + tmp131
        tmp135 = tmp134 * tmp113
        tmp136 = tmp105 + tmp135
        tmp139 = tmp138 * tmp122
        tmp140 = tmp136 + tmp139
        tmp141 = tmp124.to(tl.float32)
        tmp142 = tmp141 * tmp141
        tmp143 = tl.broadcast_to(tmp142, [XBLOCK, RBLOCK])
        tmp145 = _tmp144 + tmp143
        _tmp144 = tl.where(rmask, tmp145, _tmp144)
        tl.store(in_out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp124, rmask)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp132, rmask)
        tl.store(in_out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp140, rmask)
    tmp144 = tl.sum(_tmp144, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp146 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp155 = tl.load(in_ptr33 + (r0), rmask, other=0).to(tl.float32)
        tmp147 = tmp146.to(tl.float32)
        tmp148 = 4096.0
        tmp149 = tmp144 / tmp148
        tmp150 = 1e-05
        tmp151 = tmp149 + tmp150
        tmp152 = tl.math.rsqrt(tmp151)
        tmp153 = tmp147 * tmp152
        tmp154 = tmp153.to(tl.float32)
        tmp156 = tmp154 * tmp155
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp156, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/rf/crfejxnel3gctmhaf747hqe2aa6suk55vnau4yp2ckii67ujbqwh.py
# Source Nodes: [add_235, add_237, add_278, add_279, float_73, mean_36, mul_458, mul_459, mul_460, mul_461, rsqrt_36, type_as_72], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_235 => add_235
# add_237 => add_237
# add_278 => add_278
# add_279 => add_279
# float_73 => convert_element_type_216
# mean_36 => mean_36
# mul_458 => mul_476
# mul_459 => mul_477
# mul_460 => mul_478
# mul_461 => mul_479
# rsqrt_36 => rsqrt_36
# type_as_72 => convert_element_type_217
triton_red_fused__to_copy_add_mean_mul_rsqrt_26 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_26', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_26', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (17)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
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


# kernel path: /tmp/torchinductor_mengqy/3k/c3kei2menadawjkh6qvoykyy3v5mtxv6m34lvrcit2zzilzzbhhe.py
# Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_153, add_155, add_172, add_174, add_192, add_194, add_213, add_215, add_235, add_237, add_258, add_260, add_302, add_303, add_304, add_317, add_318, add_319, add_320, add_321, add_322, add_323, add_324, add_325, add_326, add_327, add_328, add_343, add_344, add_345, add_346, add_347, add_348, add_349, add_350, add_351, add_352, add_353, add_354, add_60, add_62, add_73, add_75, add_87, add_89, float_77, mean_38, mul_492, mul_493, mul_494, mul_495, mul_496, mul_517, mul_518, mul_519, mul_520, mul_521, mul_522, mul_523, mul_524, mul_525, mul_526, mul_527, mul_528, mul_553, mul_554, mul_555, mul_556, mul_557, mul_558, mul_559, mul_560, mul_561, mul_562, mul_563, mul_564, rsqrt_38, type_as_76], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_102 => add_102
# add_104 => add_104
# add_118 => add_118
# add_120 => add_120
# add_135 => add_135
# add_137 => add_137
# add_153 => add_153
# add_155 => add_155
# add_172 => add_172
# add_174 => add_174
# add_192 => add_192
# add_194 => add_194
# add_213 => add_213
# add_215 => add_215
# add_235 => add_235
# add_237 => add_237
# add_258 => add_258
# add_260 => add_260
# add_302 => add_302
# add_303 => add_303
# add_304 => add_304
# add_317 => add_317
# add_318 => add_318
# add_319 => add_319
# add_320 => add_320
# add_321 => add_321
# add_322 => add_322
# add_323 => add_323
# add_324 => add_324
# add_325 => add_325
# add_326 => add_326
# add_327 => add_327
# add_328 => add_328
# add_343 => add_343
# add_344 => add_344
# add_345 => add_345
# add_346 => add_346
# add_347 => add_347
# add_348 => add_348
# add_349 => add_349
# add_350 => add_350
# add_351 => add_351
# add_352 => add_352
# add_353 => add_353
# add_354 => add_354
# add_60 => add_60
# add_62 => add_62
# add_73 => add_73
# add_75 => add_75
# add_87 => add_87
# add_89 => add_89
# float_77 => convert_element_type_228
# mean_38 => mean_38
# mul_492 => mul_511
# mul_493 => mul_512
# mul_494 => mul_513
# mul_495 => mul_514
# mul_496 => mul_515
# mul_517 => mul_537
# mul_518 => mul_538
# mul_519 => mul_539
# mul_520 => mul_540
# mul_521 => mul_541
# mul_522 => mul_542
# mul_523 => mul_543
# mul_524 => mul_544
# mul_525 => mul_545
# mul_526 => mul_546
# mul_527 => mul_547
# mul_528 => mul_548
# mul_553 => mul_574
# mul_554 => mul_575
# mul_555 => mul_576
# mul_556 => mul_577
# mul_557 => mul_578
# mul_558 => mul_579
# mul_559 => mul_580
# mul_560 => mul_581
# mul_561 => mul_582
# mul_562 => mul_583
# mul_563 => mul_584
# mul_564 => mul_585
# rsqrt_38 => rsqrt_38
# type_as_76 => convert_element_type_229
triton_red_fused__to_copy_add_mean_mul_rsqrt_27 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_27', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: '*fp16', 21: '*fp16', 22: '*fp16', 23: '*fp16', 24: '*fp16', 25: '*fp16', 26: '*fp16', 27: '*fp16', 28: '*fp16', 29: '*fp16', 30: '*fp16', 31: '*fp16', 32: '*fp16', 33: '*fp16', 34: '*fp16', 35: '*fp16', 36: '*fp16', 37: '*fp16', 38: '*fp16', 39: '*fp16', 40: '*fp16', 41: '*fp16', 42: '*fp16', 43: '*fp16', 44: 'i32', 45: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_27', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 45), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(45,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_27(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (7)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp10 = tl.load(in_ptr0 + (8)).to(tl.float32)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp20 = tl.load(in_ptr7 + (7)).to(tl.float32)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp24 = tl.load(in_ptr7 + (8)).to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp28 = tl.load(in_ptr0 + (9)).to(tl.float32)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp37 = tl.load(in_ptr0 + (10)).to(tl.float32)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
    tmp46 = tl.load(in_ptr7 + (9)).to(tl.float32)
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
    tmp50 = tl.load(in_ptr7 + (10)).to(tl.float32)
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
    tmp54 = tl.load(in_ptr0 + (11)).to(tl.float32)
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK, RBLOCK])
    tmp63 = tl.load(in_ptr0 + (12)).to(tl.float32)
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, RBLOCK])
    tmp72 = tl.load(in_ptr7 + (11)).to(tl.float32)
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp76 = tl.load(in_ptr7 + (12)).to(tl.float32)
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK, RBLOCK])
    tmp80 = tl.load(in_ptr0 + (13)).to(tl.float32)
    tmp81 = tl.broadcast_to(tmp80, [XBLOCK, RBLOCK])
    tmp89 = tl.load(in_ptr0 + (14)).to(tl.float32)
    tmp90 = tl.broadcast_to(tmp89, [XBLOCK, RBLOCK])
    tmp98 = tl.load(in_ptr7 + (13)).to(tl.float32)
    tmp99 = tl.broadcast_to(tmp98, [XBLOCK, RBLOCK])
    tmp102 = tl.load(in_ptr7 + (14)).to(tl.float32)
    tmp103 = tl.broadcast_to(tmp102, [XBLOCK, RBLOCK])
    tmp106 = tl.load(in_ptr0 + (15)).to(tl.float32)
    tmp107 = tl.broadcast_to(tmp106, [XBLOCK, RBLOCK])
    tmp115 = tl.load(in_ptr0 + (16)).to(tl.float32)
    tmp116 = tl.broadcast_to(tmp115, [XBLOCK, RBLOCK])
    tmp124 = tl.load(in_ptr7 + (15)).to(tl.float32)
    tmp125 = tl.broadcast_to(tmp124, [XBLOCK, RBLOCK])
    tmp128 = tl.load(in_ptr7 + (16)).to(tl.float32)
    tmp129 = tl.broadcast_to(tmp128, [XBLOCK, RBLOCK])
    tmp133 = tl.load(in_ptr32 + (17)).to(tl.float32)
    tmp134 = tl.broadcast_to(tmp133, [XBLOCK, RBLOCK])
    tmp142 = tl.load(in_ptr32 + (18)).to(tl.float32)
    tmp143 = tl.broadcast_to(tmp142, [XBLOCK, RBLOCK])
    tmp151 = tl.load(in_ptr0 + (17)).to(tl.float32)
    tmp152 = tl.broadcast_to(tmp151, [XBLOCK, RBLOCK])
    tmp155 = tl.load(in_ptr0 + (18)).to(tl.float32)
    tmp156 = tl.broadcast_to(tmp155, [XBLOCK, RBLOCK])
    tmp159 = tl.load(in_ptr7 + (17)).to(tl.float32)
    tmp160 = tl.broadcast_to(tmp159, [XBLOCK, RBLOCK])
    tmp163 = tl.load(in_ptr7 + (18)).to(tl.float32)
    tmp164 = tl.broadcast_to(tmp163, [XBLOCK, RBLOCK])
    _tmp170 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr5 + (r0), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr6 + (r0), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_out_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp30 = tl.load(in_ptr8 + (r0), rmask, other=0).to(tl.float32)
        tmp31 = tl.load(in_ptr9 + (r0), rmask, other=0).to(tl.float32)
        tmp33 = tl.load(in_ptr10 + (r0), rmask, other=0).to(tl.float32)
        tmp39 = tl.load(in_ptr11 + (r0), rmask, other=0).to(tl.float32)
        tmp40 = tl.load(in_ptr12 + (r0), rmask, other=0).to(tl.float32)
        tmp42 = tl.load(in_ptr13 + (r0), rmask, other=0).to(tl.float32)
        tmp56 = tl.load(in_ptr14 + (r0), rmask, other=0).to(tl.float32)
        tmp57 = tl.load(in_ptr15 + (r0), rmask, other=0).to(tl.float32)
        tmp59 = tl.load(in_ptr16 + (r0), rmask, other=0).to(tl.float32)
        tmp65 = tl.load(in_ptr17 + (r0), rmask, other=0).to(tl.float32)
        tmp66 = tl.load(in_ptr18 + (r0), rmask, other=0).to(tl.float32)
        tmp68 = tl.load(in_ptr19 + (r0), rmask, other=0).to(tl.float32)
        tmp82 = tl.load(in_ptr20 + (r0), rmask, other=0).to(tl.float32)
        tmp83 = tl.load(in_ptr21 + (r0), rmask, other=0).to(tl.float32)
        tmp85 = tl.load(in_ptr22 + (r0), rmask, other=0).to(tl.float32)
        tmp91 = tl.load(in_ptr23 + (r0), rmask, other=0).to(tl.float32)
        tmp92 = tl.load(in_ptr24 + (r0), rmask, other=0).to(tl.float32)
        tmp94 = tl.load(in_ptr25 + (r0), rmask, other=0).to(tl.float32)
        tmp108 = tl.load(in_ptr26 + (r0), rmask, other=0).to(tl.float32)
        tmp109 = tl.load(in_ptr27 + (r0), rmask, other=0).to(tl.float32)
        tmp111 = tl.load(in_ptr28 + (r0), rmask, other=0).to(tl.float32)
        tmp117 = tl.load(in_ptr29 + (r0), rmask, other=0).to(tl.float32)
        tmp118 = tl.load(in_ptr30 + (r0), rmask, other=0).to(tl.float32)
        tmp120 = tl.load(in_ptr31 + (r0), rmask, other=0).to(tl.float32)
        tmp132 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp135 = tl.load(in_ptr33 + (r0), rmask, other=0).to(tl.float32)
        tmp136 = tl.load(in_ptr34 + (r0), rmask, other=0).to(tl.float32)
        tmp138 = tl.load(in_ptr35 + (r0), rmask, other=0).to(tl.float32)
        tmp144 = tl.load(in_ptr36 + (r0), rmask, other=0).to(tl.float32)
        tmp145 = tl.load(in_ptr37 + (r0), rmask, other=0).to(tl.float32)
        tmp147 = tl.load(in_ptr38 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp11 * tmp16
        tmp18 = tmp9 + tmp17
        tmp22 = tmp21 * tmp7
        tmp23 = tmp19 + tmp22
        tmp26 = tmp25 * tmp16
        tmp27 = tmp23 + tmp26
        tmp32 = tmp30 + tmp31
        tmp34 = tmp32 + tmp33
        tmp35 = tmp29 * tmp34
        tmp36 = tmp18 + tmp35
        tmp41 = tmp39 + tmp40
        tmp43 = tmp41 + tmp42
        tmp44 = tmp38 * tmp43
        tmp45 = tmp36 + tmp44
        tmp48 = tmp47 * tmp34
        tmp49 = tmp27 + tmp48
        tmp52 = tmp51 * tmp43
        tmp53 = tmp49 + tmp52
        tmp58 = tmp56 + tmp57
        tmp60 = tmp58 + tmp59
        tmp61 = tmp55 * tmp60
        tmp62 = tmp45 + tmp61
        tmp67 = tmp65 + tmp66
        tmp69 = tmp67 + tmp68
        tmp70 = tmp64 * tmp69
        tmp71 = tmp62 + tmp70
        tmp74 = tmp73 * tmp60
        tmp75 = tmp53 + tmp74
        tmp78 = tmp77 * tmp69
        tmp79 = tmp75 + tmp78
        tmp84 = tmp82 + tmp83
        tmp86 = tmp84 + tmp85
        tmp87 = tmp81 * tmp86
        tmp88 = tmp71 + tmp87
        tmp93 = tmp91 + tmp92
        tmp95 = tmp93 + tmp94
        tmp96 = tmp90 * tmp95
        tmp97 = tmp88 + tmp96
        tmp100 = tmp99 * tmp86
        tmp101 = tmp79 + tmp100
        tmp104 = tmp103 * tmp95
        tmp105 = tmp101 + tmp104
        tmp110 = tmp108 + tmp109
        tmp112 = tmp110 + tmp111
        tmp113 = tmp107 * tmp112
        tmp114 = tmp97 + tmp113
        tmp119 = tmp117 + tmp118
        tmp121 = tmp119 + tmp120
        tmp122 = tmp116 * tmp121
        tmp123 = tmp114 + tmp122
        tmp126 = tmp125 * tmp112
        tmp127 = tmp105 + tmp126
        tmp130 = tmp129 * tmp121
        tmp131 = tmp127 + tmp130
        tmp137 = tmp135 + tmp136
        tmp139 = tmp137 + tmp138
        tmp140 = tmp134 * tmp139
        tmp141 = tmp132 + tmp140
        tmp146 = tmp144 + tmp145
        tmp148 = tmp146 + tmp147
        tmp149 = tmp143 * tmp148
        tmp150 = tmp141 + tmp149
        tmp153 = tmp152 * tmp139
        tmp154 = tmp123 + tmp153
        tmp157 = tmp156 * tmp148
        tmp158 = tmp154 + tmp157
        tmp161 = tmp160 * tmp139
        tmp162 = tmp131 + tmp161
        tmp165 = tmp164 * tmp148
        tmp166 = tmp162 + tmp165
        tmp167 = tmp150.to(tl.float32)
        tmp168 = tmp167 * tmp167
        tmp169 = tl.broadcast_to(tmp168, [XBLOCK, RBLOCK])
        tmp171 = _tmp170 + tmp169
        _tmp170 = tl.where(rmask, tmp171, _tmp170)
        tl.store(in_out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp150, rmask)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp158, rmask)
        tl.store(in_out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp166, rmask)
    tmp170 = tl.sum(_tmp170, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp172 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp181 = tl.load(in_ptr39 + (r0), rmask, other=0).to(tl.float32)
        tmp173 = tmp172.to(tl.float32)
        tmp174 = 4096.0
        tmp175 = tmp170 / tmp174
        tmp176 = 1e-05
        tmp177 = tmp175 + tmp176
        tmp178 = tl.math.rsqrt(tmp177)
        tmp179 = tmp173 * tmp178
        tmp180 = tmp179.to(tl.float32)
        tmp182 = tmp180 * tmp181
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp182, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/ub/cubz5nehowphrzk6amqn76swobvjxpifj3ydxl7zdc7itd3fa7er.py
# Source Nodes: [add_282, add_284, add_329, add_330, float_81, mean_40, mul_529, mul_530, mul_531, mul_532, rsqrt_40, type_as_80], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_282 => add_282
# add_284 => add_284
# add_329 => add_329
# add_330 => add_330
# float_81 => convert_element_type_240
# mean_40 => mean_40
# mul_529 => mul_549
# mul_530 => mul_550
# mul_531 => mul_551
# mul_532 => mul_552
# rsqrt_40 => rsqrt_40
# type_as_80 => convert_element_type_241
triton_red_fused__to_copy_add_mean_mul_rsqrt_28 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_28', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_28', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_28(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (19)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
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


# kernel path: /tmp/torchinductor_mengqy/ha/chaxza7be6hc6iosav6uumkajvmns3lan4vmzx22wmdzzcqq6r6u.py
# Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_153, add_155, add_172, add_174, add_192, add_194, add_213, add_215, add_235, add_237, add_258, add_260, add_282, add_284, add_307, add_309, add_355, add_356, add_357, add_370, add_371, add_372, add_373, add_374, add_375, add_376, add_377, add_378, add_379, add_380, add_381, add_382, add_383, add_398, add_399, add_400, add_401, add_402, add_403, add_404, add_405, add_406, add_407, add_408, add_409, add_410, add_411, add_60, add_62, add_73, add_75, add_87, add_89, float_85, mean_42, mul_565, mul_566, mul_567, mul_568, mul_569, mul_590, mul_591, mul_592, mul_593, mul_594, mul_595, mul_596, mul_597, mul_598, mul_599, mul_600, mul_601, mul_602, mul_603, mul_628, mul_629, mul_630, mul_631, mul_632, mul_633, mul_634, mul_635, mul_636, mul_637, mul_638, mul_639, mul_640, mul_641, rsqrt_42, type_as_84], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_102 => add_102
# add_104 => add_104
# add_118 => add_118
# add_120 => add_120
# add_135 => add_135
# add_137 => add_137
# add_153 => add_153
# add_155 => add_155
# add_172 => add_172
# add_174 => add_174
# add_192 => add_192
# add_194 => add_194
# add_213 => add_213
# add_215 => add_215
# add_235 => add_235
# add_237 => add_237
# add_258 => add_258
# add_260 => add_260
# add_282 => add_282
# add_284 => add_284
# add_307 => add_307
# add_309 => add_309
# add_355 => add_355
# add_356 => add_356
# add_357 => add_357
# add_370 => add_370
# add_371 => add_371
# add_372 => add_372
# add_373 => add_373
# add_374 => add_374
# add_375 => add_375
# add_376 => add_376
# add_377 => add_377
# add_378 => add_378
# add_379 => add_379
# add_380 => add_380
# add_381 => add_381
# add_382 => add_382
# add_383 => add_383
# add_398 => add_398
# add_399 => add_399
# add_400 => add_400
# add_401 => add_401
# add_402 => add_402
# add_403 => add_403
# add_404 => add_404
# add_405 => add_405
# add_406 => add_406
# add_407 => add_407
# add_408 => add_408
# add_409 => add_409
# add_410 => add_410
# add_411 => add_411
# add_60 => add_60
# add_62 => add_62
# add_73 => add_73
# add_75 => add_75
# add_87 => add_87
# add_89 => add_89
# float_85 => convert_element_type_252
# mean_42 => mean_42
# mul_565 => mul_586
# mul_566 => mul_587
# mul_567 => mul_588
# mul_568 => mul_589
# mul_569 => mul_590
# mul_590 => mul_612
# mul_591 => mul_613
# mul_592 => mul_614
# mul_593 => mul_615
# mul_594 => mul_616
# mul_595 => mul_617
# mul_596 => mul_618
# mul_597 => mul_619
# mul_598 => mul_620
# mul_599 => mul_621
# mul_600 => mul_622
# mul_601 => mul_623
# mul_602 => mul_624
# mul_603 => mul_625
# mul_628 => mul_651
# mul_629 => mul_652
# mul_630 => mul_653
# mul_631 => mul_654
# mul_632 => mul_655
# mul_633 => mul_656
# mul_634 => mul_657
# mul_635 => mul_658
# mul_636 => mul_659
# mul_637 => mul_660
# mul_638 => mul_661
# mul_639 => mul_662
# mul_640 => mul_663
# mul_641 => mul_664
# rsqrt_42 => rsqrt_42
# type_as_84 => convert_element_type_253
triton_red_fused__to_copy_add_mean_mul_rsqrt_29 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_29', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: '*fp16', 21: '*fp16', 22: '*fp16', 23: '*fp16', 24: '*fp16', 25: '*fp16', 26: '*fp16', 27: '*fp16', 28: '*fp16', 29: '*fp16', 30: '*fp16', 31: '*fp16', 32: '*fp16', 33: '*fp16', 34: '*fp16', 35: '*fp16', 36: '*fp16', 37: '*fp16', 38: '*fp16', 39: '*fp16', 40: '*fp16', 41: '*fp16', 42: '*fp16', 43: '*fp16', 44: '*fp16', 45: '*fp16', 46: '*fp16', 47: '*fp16', 48: '*fp16', 49: '*fp16', 50: 'i32', 51: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_29', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 51), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(51,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_29(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (7)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp10 = tl.load(in_ptr0 + (8)).to(tl.float32)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp20 = tl.load(in_ptr7 + (7)).to(tl.float32)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp24 = tl.load(in_ptr7 + (8)).to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp28 = tl.load(in_ptr0 + (9)).to(tl.float32)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp37 = tl.load(in_ptr0 + (10)).to(tl.float32)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
    tmp46 = tl.load(in_ptr7 + (9)).to(tl.float32)
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
    tmp50 = tl.load(in_ptr7 + (10)).to(tl.float32)
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
    tmp54 = tl.load(in_ptr0 + (11)).to(tl.float32)
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK, RBLOCK])
    tmp63 = tl.load(in_ptr0 + (12)).to(tl.float32)
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, RBLOCK])
    tmp72 = tl.load(in_ptr7 + (11)).to(tl.float32)
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp76 = tl.load(in_ptr7 + (12)).to(tl.float32)
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK, RBLOCK])
    tmp80 = tl.load(in_ptr0 + (13)).to(tl.float32)
    tmp81 = tl.broadcast_to(tmp80, [XBLOCK, RBLOCK])
    tmp89 = tl.load(in_ptr0 + (14)).to(tl.float32)
    tmp90 = tl.broadcast_to(tmp89, [XBLOCK, RBLOCK])
    tmp98 = tl.load(in_ptr7 + (13)).to(tl.float32)
    tmp99 = tl.broadcast_to(tmp98, [XBLOCK, RBLOCK])
    tmp102 = tl.load(in_ptr7 + (14)).to(tl.float32)
    tmp103 = tl.broadcast_to(tmp102, [XBLOCK, RBLOCK])
    tmp106 = tl.load(in_ptr0 + (15)).to(tl.float32)
    tmp107 = tl.broadcast_to(tmp106, [XBLOCK, RBLOCK])
    tmp115 = tl.load(in_ptr0 + (16)).to(tl.float32)
    tmp116 = tl.broadcast_to(tmp115, [XBLOCK, RBLOCK])
    tmp124 = tl.load(in_ptr7 + (15)).to(tl.float32)
    tmp125 = tl.broadcast_to(tmp124, [XBLOCK, RBLOCK])
    tmp128 = tl.load(in_ptr7 + (16)).to(tl.float32)
    tmp129 = tl.broadcast_to(tmp128, [XBLOCK, RBLOCK])
    tmp132 = tl.load(in_ptr0 + (17)).to(tl.float32)
    tmp133 = tl.broadcast_to(tmp132, [XBLOCK, RBLOCK])
    tmp141 = tl.load(in_ptr0 + (18)).to(tl.float32)
    tmp142 = tl.broadcast_to(tmp141, [XBLOCK, RBLOCK])
    tmp150 = tl.load(in_ptr7 + (17)).to(tl.float32)
    tmp151 = tl.broadcast_to(tmp150, [XBLOCK, RBLOCK])
    tmp154 = tl.load(in_ptr7 + (18)).to(tl.float32)
    tmp155 = tl.broadcast_to(tmp154, [XBLOCK, RBLOCK])
    tmp159 = tl.load(in_ptr38 + (19)).to(tl.float32)
    tmp160 = tl.broadcast_to(tmp159, [XBLOCK, RBLOCK])
    tmp168 = tl.load(in_ptr38 + (20)).to(tl.float32)
    tmp169 = tl.broadcast_to(tmp168, [XBLOCK, RBLOCK])
    tmp177 = tl.load(in_ptr0 + (19)).to(tl.float32)
    tmp178 = tl.broadcast_to(tmp177, [XBLOCK, RBLOCK])
    tmp181 = tl.load(in_ptr0 + (20)).to(tl.float32)
    tmp182 = tl.broadcast_to(tmp181, [XBLOCK, RBLOCK])
    tmp185 = tl.load(in_ptr7 + (19)).to(tl.float32)
    tmp186 = tl.broadcast_to(tmp185, [XBLOCK, RBLOCK])
    tmp189 = tl.load(in_ptr7 + (20)).to(tl.float32)
    tmp190 = tl.broadcast_to(tmp189, [XBLOCK, RBLOCK])
    _tmp196 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr5 + (r0), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr6 + (r0), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_out_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp30 = tl.load(in_ptr8 + (r0), rmask, other=0).to(tl.float32)
        tmp31 = tl.load(in_ptr9 + (r0), rmask, other=0).to(tl.float32)
        tmp33 = tl.load(in_ptr10 + (r0), rmask, other=0).to(tl.float32)
        tmp39 = tl.load(in_ptr11 + (r0), rmask, other=0).to(tl.float32)
        tmp40 = tl.load(in_ptr12 + (r0), rmask, other=0).to(tl.float32)
        tmp42 = tl.load(in_ptr13 + (r0), rmask, other=0).to(tl.float32)
        tmp56 = tl.load(in_ptr14 + (r0), rmask, other=0).to(tl.float32)
        tmp57 = tl.load(in_ptr15 + (r0), rmask, other=0).to(tl.float32)
        tmp59 = tl.load(in_ptr16 + (r0), rmask, other=0).to(tl.float32)
        tmp65 = tl.load(in_ptr17 + (r0), rmask, other=0).to(tl.float32)
        tmp66 = tl.load(in_ptr18 + (r0), rmask, other=0).to(tl.float32)
        tmp68 = tl.load(in_ptr19 + (r0), rmask, other=0).to(tl.float32)
        tmp82 = tl.load(in_ptr20 + (r0), rmask, other=0).to(tl.float32)
        tmp83 = tl.load(in_ptr21 + (r0), rmask, other=0).to(tl.float32)
        tmp85 = tl.load(in_ptr22 + (r0), rmask, other=0).to(tl.float32)
        tmp91 = tl.load(in_ptr23 + (r0), rmask, other=0).to(tl.float32)
        tmp92 = tl.load(in_ptr24 + (r0), rmask, other=0).to(tl.float32)
        tmp94 = tl.load(in_ptr25 + (r0), rmask, other=0).to(tl.float32)
        tmp108 = tl.load(in_ptr26 + (r0), rmask, other=0).to(tl.float32)
        tmp109 = tl.load(in_ptr27 + (r0), rmask, other=0).to(tl.float32)
        tmp111 = tl.load(in_ptr28 + (r0), rmask, other=0).to(tl.float32)
        tmp117 = tl.load(in_ptr29 + (r0), rmask, other=0).to(tl.float32)
        tmp118 = tl.load(in_ptr30 + (r0), rmask, other=0).to(tl.float32)
        tmp120 = tl.load(in_ptr31 + (r0), rmask, other=0).to(tl.float32)
        tmp134 = tl.load(in_ptr32 + (r0), rmask, other=0).to(tl.float32)
        tmp135 = tl.load(in_ptr33 + (r0), rmask, other=0).to(tl.float32)
        tmp137 = tl.load(in_ptr34 + (r0), rmask, other=0).to(tl.float32)
        tmp143 = tl.load(in_ptr35 + (r0), rmask, other=0).to(tl.float32)
        tmp144 = tl.load(in_ptr36 + (r0), rmask, other=0).to(tl.float32)
        tmp146 = tl.load(in_ptr37 + (r0), rmask, other=0).to(tl.float32)
        tmp158 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp161 = tl.load(in_ptr39 + (r0), rmask, other=0).to(tl.float32)
        tmp162 = tl.load(in_ptr40 + (r0), rmask, other=0).to(tl.float32)
        tmp164 = tl.load(in_ptr41 + (r0), rmask, other=0).to(tl.float32)
        tmp170 = tl.load(in_ptr42 + (r0), rmask, other=0).to(tl.float32)
        tmp171 = tl.load(in_ptr43 + (r0), rmask, other=0).to(tl.float32)
        tmp173 = tl.load(in_ptr44 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp11 * tmp16
        tmp18 = tmp9 + tmp17
        tmp22 = tmp21 * tmp7
        tmp23 = tmp19 + tmp22
        tmp26 = tmp25 * tmp16
        tmp27 = tmp23 + tmp26
        tmp32 = tmp30 + tmp31
        tmp34 = tmp32 + tmp33
        tmp35 = tmp29 * tmp34
        tmp36 = tmp18 + tmp35
        tmp41 = tmp39 + tmp40
        tmp43 = tmp41 + tmp42
        tmp44 = tmp38 * tmp43
        tmp45 = tmp36 + tmp44
        tmp48 = tmp47 * tmp34
        tmp49 = tmp27 + tmp48
        tmp52 = tmp51 * tmp43
        tmp53 = tmp49 + tmp52
        tmp58 = tmp56 + tmp57
        tmp60 = tmp58 + tmp59
        tmp61 = tmp55 * tmp60
        tmp62 = tmp45 + tmp61
        tmp67 = tmp65 + tmp66
        tmp69 = tmp67 + tmp68
        tmp70 = tmp64 * tmp69
        tmp71 = tmp62 + tmp70
        tmp74 = tmp73 * tmp60
        tmp75 = tmp53 + tmp74
        tmp78 = tmp77 * tmp69
        tmp79 = tmp75 + tmp78
        tmp84 = tmp82 + tmp83
        tmp86 = tmp84 + tmp85
        tmp87 = tmp81 * tmp86
        tmp88 = tmp71 + tmp87
        tmp93 = tmp91 + tmp92
        tmp95 = tmp93 + tmp94
        tmp96 = tmp90 * tmp95
        tmp97 = tmp88 + tmp96
        tmp100 = tmp99 * tmp86
        tmp101 = tmp79 + tmp100
        tmp104 = tmp103 * tmp95
        tmp105 = tmp101 + tmp104
        tmp110 = tmp108 + tmp109
        tmp112 = tmp110 + tmp111
        tmp113 = tmp107 * tmp112
        tmp114 = tmp97 + tmp113
        tmp119 = tmp117 + tmp118
        tmp121 = tmp119 + tmp120
        tmp122 = tmp116 * tmp121
        tmp123 = tmp114 + tmp122
        tmp126 = tmp125 * tmp112
        tmp127 = tmp105 + tmp126
        tmp130 = tmp129 * tmp121
        tmp131 = tmp127 + tmp130
        tmp136 = tmp134 + tmp135
        tmp138 = tmp136 + tmp137
        tmp139 = tmp133 * tmp138
        tmp140 = tmp123 + tmp139
        tmp145 = tmp143 + tmp144
        tmp147 = tmp145 + tmp146
        tmp148 = tmp142 * tmp147
        tmp149 = tmp140 + tmp148
        tmp152 = tmp151 * tmp138
        tmp153 = tmp131 + tmp152
        tmp156 = tmp155 * tmp147
        tmp157 = tmp153 + tmp156
        tmp163 = tmp161 + tmp162
        tmp165 = tmp163 + tmp164
        tmp166 = tmp160 * tmp165
        tmp167 = tmp158 + tmp166
        tmp172 = tmp170 + tmp171
        tmp174 = tmp172 + tmp173
        tmp175 = tmp169 * tmp174
        tmp176 = tmp167 + tmp175
        tmp179 = tmp178 * tmp165
        tmp180 = tmp149 + tmp179
        tmp183 = tmp182 * tmp174
        tmp184 = tmp180 + tmp183
        tmp187 = tmp186 * tmp165
        tmp188 = tmp157 + tmp187
        tmp191 = tmp190 * tmp174
        tmp192 = tmp188 + tmp191
        tmp193 = tmp176.to(tl.float32)
        tmp194 = tmp193 * tmp193
        tmp195 = tl.broadcast_to(tmp194, [XBLOCK, RBLOCK])
        tmp197 = _tmp196 + tmp195
        _tmp196 = tl.where(rmask, tmp197, _tmp196)
        tl.store(in_out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp176, rmask)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp184, rmask)
        tl.store(in_out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp192, rmask)
    tmp196 = tl.sum(_tmp196, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp198 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp207 = tl.load(in_ptr45 + (r0), rmask, other=0).to(tl.float32)
        tmp199 = tmp198.to(tl.float32)
        tmp200 = 4096.0
        tmp201 = tmp196 / tmp200
        tmp202 = 1e-05
        tmp203 = tmp201 + tmp202
        tmp204 = tl.math.rsqrt(tmp203)
        tmp205 = tmp199 * tmp204
        tmp206 = tmp205.to(tl.float32)
        tmp208 = tmp206 * tmp207
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp208, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/cq/ccqrlkxw5h6zkpzy2ntojy5r75dld422ppceyu2yyohf5emvby2c.py
# Source Nodes: [add_333, add_335, add_384, add_385, float_89, mean_44, mul_604, mul_605, mul_606, mul_607, rsqrt_44, type_as_88], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_333 => add_333
# add_335 => add_335
# add_384 => add_384
# add_385 => add_385
# float_89 => convert_element_type_264
# mean_44 => mean_44
# mul_604 => mul_626
# mul_605 => mul_627
# mul_606 => mul_628
# mul_607 => mul_629
# rsqrt_44 => rsqrt_44
# type_as_88 => convert_element_type_265
triton_red_fused__to_copy_add_mean_mul_rsqrt_30 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_30', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_30', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (21)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
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


# kernel path: /tmp/torchinductor_mengqy/e2/ce2v3pzhqxrvppkvltw4geicpomebltxh4tdsrka32bh3pwqi2lb.py
# Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_153, add_155, add_172, add_174, add_192, add_194, add_213, add_215, add_235, add_237, add_258, add_260, add_282, add_284, add_307, add_309, add_333, add_335, add_360, add_362, add_412, add_413, add_414, add_427, add_428, add_429, add_430, add_431, add_432, add_433, add_434, add_435, add_436, add_437, add_438, add_439, add_440, add_441, add_442, add_457, add_458, add_459, add_460, add_461, add_462, add_463, add_464, add_465, add_466, add_467, add_468, add_469, add_470, add_471, add_472, add_60, add_62, add_73, add_75, add_87, add_89, float_93, mean_46, mul_642, mul_643, mul_644, mul_645, mul_646, mul_667, mul_668, mul_669, mul_670, mul_671, mul_672, mul_673, mul_674, mul_675, mul_676, mul_677, mul_678, mul_679, mul_680, mul_681, mul_682, mul_707, mul_708, mul_709, mul_710, mul_711, mul_712, mul_713, mul_714, mul_715, mul_716, mul_717, mul_718, mul_719, mul_720, mul_721, mul_722, rsqrt_46, type_as_92], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_102 => add_102
# add_104 => add_104
# add_118 => add_118
# add_120 => add_120
# add_135 => add_135
# add_137 => add_137
# add_153 => add_153
# add_155 => add_155
# add_172 => add_172
# add_174 => add_174
# add_192 => add_192
# add_194 => add_194
# add_213 => add_213
# add_215 => add_215
# add_235 => add_235
# add_237 => add_237
# add_258 => add_258
# add_260 => add_260
# add_282 => add_282
# add_284 => add_284
# add_307 => add_307
# add_309 => add_309
# add_333 => add_333
# add_335 => add_335
# add_360 => add_360
# add_362 => add_362
# add_412 => add_412
# add_413 => add_413
# add_414 => add_414
# add_427 => add_427
# add_428 => add_428
# add_429 => add_429
# add_430 => add_430
# add_431 => add_431
# add_432 => add_432
# add_433 => add_433
# add_434 => add_434
# add_435 => add_435
# add_436 => add_436
# add_437 => add_437
# add_438 => add_438
# add_439 => add_439
# add_440 => add_440
# add_441 => add_441
# add_442 => add_442
# add_457 => add_457
# add_458 => add_458
# add_459 => add_459
# add_460 => add_460
# add_461 => add_461
# add_462 => add_462
# add_463 => add_463
# add_464 => add_464
# add_465 => add_465
# add_466 => add_466
# add_467 => add_467
# add_468 => add_468
# add_469 => add_469
# add_470 => add_470
# add_471 => add_471
# add_472 => add_472
# add_60 => add_60
# add_62 => add_62
# add_73 => add_73
# add_75 => add_75
# add_87 => add_87
# add_89 => add_89
# float_93 => convert_element_type_276
# mean_46 => mean_46
# mul_642 => mul_665
# mul_643 => mul_666
# mul_644 => mul_667
# mul_645 => mul_668
# mul_646 => mul_669
# mul_667 => mul_691
# mul_668 => mul_692
# mul_669 => mul_693
# mul_670 => mul_694
# mul_671 => mul_695
# mul_672 => mul_696
# mul_673 => mul_697
# mul_674 => mul_698
# mul_675 => mul_699
# mul_676 => mul_700
# mul_677 => mul_701
# mul_678 => mul_702
# mul_679 => mul_703
# mul_680 => mul_704
# mul_681 => mul_705
# mul_682 => mul_706
# mul_707 => mul_732
# mul_708 => mul_733
# mul_709 => mul_734
# mul_710 => mul_735
# mul_711 => mul_736
# mul_712 => mul_737
# mul_713 => mul_738
# mul_714 => mul_739
# mul_715 => mul_740
# mul_716 => mul_741
# mul_717 => mul_742
# mul_718 => mul_743
# mul_719 => mul_744
# mul_720 => mul_745
# mul_721 => mul_746
# mul_722 => mul_747
# rsqrt_46 => rsqrt_46
# type_as_92 => convert_element_type_277
triton_red_fused__to_copy_add_mean_mul_rsqrt_31 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_31', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: '*fp16', 21: '*fp16', 22: '*fp16', 23: '*fp16', 24: '*fp16', 25: '*fp16', 26: '*fp16', 27: '*fp16', 28: '*fp16', 29: '*fp16', 30: '*fp16', 31: '*fp16', 32: '*fp16', 33: '*fp16', 34: '*fp16', 35: '*fp16', 36: '*fp16', 37: '*fp16', 38: '*fp16', 39: '*fp16', 40: '*fp16', 41: '*fp16', 42: '*fp16', 43: '*fp16', 44: '*fp16', 45: '*fp16', 46: '*fp16', 47: '*fp16', 48: '*fp16', 49: '*fp16', 50: '*fp16', 51: '*fp16', 52: '*fp16', 53: '*fp16', 54: '*fp16', 55: '*fp16', 56: 'i32', 57: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_31', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 57), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(57,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_31(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (7)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp10 = tl.load(in_ptr0 + (8)).to(tl.float32)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp20 = tl.load(in_ptr7 + (7)).to(tl.float32)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp24 = tl.load(in_ptr7 + (8)).to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp28 = tl.load(in_ptr0 + (9)).to(tl.float32)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp37 = tl.load(in_ptr0 + (10)).to(tl.float32)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
    tmp46 = tl.load(in_ptr7 + (9)).to(tl.float32)
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
    tmp50 = tl.load(in_ptr7 + (10)).to(tl.float32)
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
    tmp54 = tl.load(in_ptr0 + (11)).to(tl.float32)
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK, RBLOCK])
    tmp63 = tl.load(in_ptr0 + (12)).to(tl.float32)
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, RBLOCK])
    tmp72 = tl.load(in_ptr7 + (11)).to(tl.float32)
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp76 = tl.load(in_ptr7 + (12)).to(tl.float32)
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK, RBLOCK])
    tmp80 = tl.load(in_ptr0 + (13)).to(tl.float32)
    tmp81 = tl.broadcast_to(tmp80, [XBLOCK, RBLOCK])
    tmp89 = tl.load(in_ptr0 + (14)).to(tl.float32)
    tmp90 = tl.broadcast_to(tmp89, [XBLOCK, RBLOCK])
    tmp98 = tl.load(in_ptr7 + (13)).to(tl.float32)
    tmp99 = tl.broadcast_to(tmp98, [XBLOCK, RBLOCK])
    tmp102 = tl.load(in_ptr7 + (14)).to(tl.float32)
    tmp103 = tl.broadcast_to(tmp102, [XBLOCK, RBLOCK])
    tmp106 = tl.load(in_ptr0 + (15)).to(tl.float32)
    tmp107 = tl.broadcast_to(tmp106, [XBLOCK, RBLOCK])
    tmp115 = tl.load(in_ptr0 + (16)).to(tl.float32)
    tmp116 = tl.broadcast_to(tmp115, [XBLOCK, RBLOCK])
    tmp124 = tl.load(in_ptr7 + (15)).to(tl.float32)
    tmp125 = tl.broadcast_to(tmp124, [XBLOCK, RBLOCK])
    tmp128 = tl.load(in_ptr7 + (16)).to(tl.float32)
    tmp129 = tl.broadcast_to(tmp128, [XBLOCK, RBLOCK])
    tmp132 = tl.load(in_ptr0 + (17)).to(tl.float32)
    tmp133 = tl.broadcast_to(tmp132, [XBLOCK, RBLOCK])
    tmp141 = tl.load(in_ptr0 + (18)).to(tl.float32)
    tmp142 = tl.broadcast_to(tmp141, [XBLOCK, RBLOCK])
    tmp150 = tl.load(in_ptr7 + (17)).to(tl.float32)
    tmp151 = tl.broadcast_to(tmp150, [XBLOCK, RBLOCK])
    tmp154 = tl.load(in_ptr7 + (18)).to(tl.float32)
    tmp155 = tl.broadcast_to(tmp154, [XBLOCK, RBLOCK])
    tmp158 = tl.load(in_ptr0 + (19)).to(tl.float32)
    tmp159 = tl.broadcast_to(tmp158, [XBLOCK, RBLOCK])
    tmp167 = tl.load(in_ptr0 + (20)).to(tl.float32)
    tmp168 = tl.broadcast_to(tmp167, [XBLOCK, RBLOCK])
    tmp176 = tl.load(in_ptr7 + (19)).to(tl.float32)
    tmp177 = tl.broadcast_to(tmp176, [XBLOCK, RBLOCK])
    tmp180 = tl.load(in_ptr7 + (20)).to(tl.float32)
    tmp181 = tl.broadcast_to(tmp180, [XBLOCK, RBLOCK])
    tmp185 = tl.load(in_ptr44 + (21)).to(tl.float32)
    tmp186 = tl.broadcast_to(tmp185, [XBLOCK, RBLOCK])
    tmp194 = tl.load(in_ptr44 + (22)).to(tl.float32)
    tmp195 = tl.broadcast_to(tmp194, [XBLOCK, RBLOCK])
    tmp203 = tl.load(in_ptr0 + (21)).to(tl.float32)
    tmp204 = tl.broadcast_to(tmp203, [XBLOCK, RBLOCK])
    tmp207 = tl.load(in_ptr0 + (22)).to(tl.float32)
    tmp208 = tl.broadcast_to(tmp207, [XBLOCK, RBLOCK])
    tmp211 = tl.load(in_ptr7 + (21)).to(tl.float32)
    tmp212 = tl.broadcast_to(tmp211, [XBLOCK, RBLOCK])
    tmp215 = tl.load(in_ptr7 + (22)).to(tl.float32)
    tmp216 = tl.broadcast_to(tmp215, [XBLOCK, RBLOCK])
    _tmp222 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr5 + (r0), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr6 + (r0), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_out_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp30 = tl.load(in_ptr8 + (r0), rmask, other=0).to(tl.float32)
        tmp31 = tl.load(in_ptr9 + (r0), rmask, other=0).to(tl.float32)
        tmp33 = tl.load(in_ptr10 + (r0), rmask, other=0).to(tl.float32)
        tmp39 = tl.load(in_ptr11 + (r0), rmask, other=0).to(tl.float32)
        tmp40 = tl.load(in_ptr12 + (r0), rmask, other=0).to(tl.float32)
        tmp42 = tl.load(in_ptr13 + (r0), rmask, other=0).to(tl.float32)
        tmp56 = tl.load(in_ptr14 + (r0), rmask, other=0).to(tl.float32)
        tmp57 = tl.load(in_ptr15 + (r0), rmask, other=0).to(tl.float32)
        tmp59 = tl.load(in_ptr16 + (r0), rmask, other=0).to(tl.float32)
        tmp65 = tl.load(in_ptr17 + (r0), rmask, other=0).to(tl.float32)
        tmp66 = tl.load(in_ptr18 + (r0), rmask, other=0).to(tl.float32)
        tmp68 = tl.load(in_ptr19 + (r0), rmask, other=0).to(tl.float32)
        tmp82 = tl.load(in_ptr20 + (r0), rmask, other=0).to(tl.float32)
        tmp83 = tl.load(in_ptr21 + (r0), rmask, other=0).to(tl.float32)
        tmp85 = tl.load(in_ptr22 + (r0), rmask, other=0).to(tl.float32)
        tmp91 = tl.load(in_ptr23 + (r0), rmask, other=0).to(tl.float32)
        tmp92 = tl.load(in_ptr24 + (r0), rmask, other=0).to(tl.float32)
        tmp94 = tl.load(in_ptr25 + (r0), rmask, other=0).to(tl.float32)
        tmp108 = tl.load(in_ptr26 + (r0), rmask, other=0).to(tl.float32)
        tmp109 = tl.load(in_ptr27 + (r0), rmask, other=0).to(tl.float32)
        tmp111 = tl.load(in_ptr28 + (r0), rmask, other=0).to(tl.float32)
        tmp117 = tl.load(in_ptr29 + (r0), rmask, other=0).to(tl.float32)
        tmp118 = tl.load(in_ptr30 + (r0), rmask, other=0).to(tl.float32)
        tmp120 = tl.load(in_ptr31 + (r0), rmask, other=0).to(tl.float32)
        tmp134 = tl.load(in_ptr32 + (r0), rmask, other=0).to(tl.float32)
        tmp135 = tl.load(in_ptr33 + (r0), rmask, other=0).to(tl.float32)
        tmp137 = tl.load(in_ptr34 + (r0), rmask, other=0).to(tl.float32)
        tmp143 = tl.load(in_ptr35 + (r0), rmask, other=0).to(tl.float32)
        tmp144 = tl.load(in_ptr36 + (r0), rmask, other=0).to(tl.float32)
        tmp146 = tl.load(in_ptr37 + (r0), rmask, other=0).to(tl.float32)
        tmp160 = tl.load(in_ptr38 + (r0), rmask, other=0).to(tl.float32)
        tmp161 = tl.load(in_ptr39 + (r0), rmask, other=0).to(tl.float32)
        tmp163 = tl.load(in_ptr40 + (r0), rmask, other=0).to(tl.float32)
        tmp169 = tl.load(in_ptr41 + (r0), rmask, other=0).to(tl.float32)
        tmp170 = tl.load(in_ptr42 + (r0), rmask, other=0).to(tl.float32)
        tmp172 = tl.load(in_ptr43 + (r0), rmask, other=0).to(tl.float32)
        tmp184 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp187 = tl.load(in_ptr45 + (r0), rmask, other=0).to(tl.float32)
        tmp188 = tl.load(in_ptr46 + (r0), rmask, other=0).to(tl.float32)
        tmp190 = tl.load(in_ptr47 + (r0), rmask, other=0).to(tl.float32)
        tmp196 = tl.load(in_ptr48 + (r0), rmask, other=0).to(tl.float32)
        tmp197 = tl.load(in_ptr49 + (r0), rmask, other=0).to(tl.float32)
        tmp199 = tl.load(in_ptr50 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp11 * tmp16
        tmp18 = tmp9 + tmp17
        tmp22 = tmp21 * tmp7
        tmp23 = tmp19 + tmp22
        tmp26 = tmp25 * tmp16
        tmp27 = tmp23 + tmp26
        tmp32 = tmp30 + tmp31
        tmp34 = tmp32 + tmp33
        tmp35 = tmp29 * tmp34
        tmp36 = tmp18 + tmp35
        tmp41 = tmp39 + tmp40
        tmp43 = tmp41 + tmp42
        tmp44 = tmp38 * tmp43
        tmp45 = tmp36 + tmp44
        tmp48 = tmp47 * tmp34
        tmp49 = tmp27 + tmp48
        tmp52 = tmp51 * tmp43
        tmp53 = tmp49 + tmp52
        tmp58 = tmp56 + tmp57
        tmp60 = tmp58 + tmp59
        tmp61 = tmp55 * tmp60
        tmp62 = tmp45 + tmp61
        tmp67 = tmp65 + tmp66
        tmp69 = tmp67 + tmp68
        tmp70 = tmp64 * tmp69
        tmp71 = tmp62 + tmp70
        tmp74 = tmp73 * tmp60
        tmp75 = tmp53 + tmp74
        tmp78 = tmp77 * tmp69
        tmp79 = tmp75 + tmp78
        tmp84 = tmp82 + tmp83
        tmp86 = tmp84 + tmp85
        tmp87 = tmp81 * tmp86
        tmp88 = tmp71 + tmp87
        tmp93 = tmp91 + tmp92
        tmp95 = tmp93 + tmp94
        tmp96 = tmp90 * tmp95
        tmp97 = tmp88 + tmp96
        tmp100 = tmp99 * tmp86
        tmp101 = tmp79 + tmp100
        tmp104 = tmp103 * tmp95
        tmp105 = tmp101 + tmp104
        tmp110 = tmp108 + tmp109
        tmp112 = tmp110 + tmp111
        tmp113 = tmp107 * tmp112
        tmp114 = tmp97 + tmp113
        tmp119 = tmp117 + tmp118
        tmp121 = tmp119 + tmp120
        tmp122 = tmp116 * tmp121
        tmp123 = tmp114 + tmp122
        tmp126 = tmp125 * tmp112
        tmp127 = tmp105 + tmp126
        tmp130 = tmp129 * tmp121
        tmp131 = tmp127 + tmp130
        tmp136 = tmp134 + tmp135
        tmp138 = tmp136 + tmp137
        tmp139 = tmp133 * tmp138
        tmp140 = tmp123 + tmp139
        tmp145 = tmp143 + tmp144
        tmp147 = tmp145 + tmp146
        tmp148 = tmp142 * tmp147
        tmp149 = tmp140 + tmp148
        tmp152 = tmp151 * tmp138
        tmp153 = tmp131 + tmp152
        tmp156 = tmp155 * tmp147
        tmp157 = tmp153 + tmp156
        tmp162 = tmp160 + tmp161
        tmp164 = tmp162 + tmp163
        tmp165 = tmp159 * tmp164
        tmp166 = tmp149 + tmp165
        tmp171 = tmp169 + tmp170
        tmp173 = tmp171 + tmp172
        tmp174 = tmp168 * tmp173
        tmp175 = tmp166 + tmp174
        tmp178 = tmp177 * tmp164
        tmp179 = tmp157 + tmp178
        tmp182 = tmp181 * tmp173
        tmp183 = tmp179 + tmp182
        tmp189 = tmp187 + tmp188
        tmp191 = tmp189 + tmp190
        tmp192 = tmp186 * tmp191
        tmp193 = tmp184 + tmp192
        tmp198 = tmp196 + tmp197
        tmp200 = tmp198 + tmp199
        tmp201 = tmp195 * tmp200
        tmp202 = tmp193 + tmp201
        tmp205 = tmp204 * tmp191
        tmp206 = tmp175 + tmp205
        tmp209 = tmp208 * tmp200
        tmp210 = tmp206 + tmp209
        tmp213 = tmp212 * tmp191
        tmp214 = tmp183 + tmp213
        tmp217 = tmp216 * tmp200
        tmp218 = tmp214 + tmp217
        tmp219 = tmp202.to(tl.float32)
        tmp220 = tmp219 * tmp219
        tmp221 = tl.broadcast_to(tmp220, [XBLOCK, RBLOCK])
        tmp223 = _tmp222 + tmp221
        _tmp222 = tl.where(rmask, tmp223, _tmp222)
        tl.store(in_out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp202, rmask)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp210, rmask)
        tl.store(in_out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp218, rmask)
    tmp222 = tl.sum(_tmp222, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp224 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp233 = tl.load(in_ptr51 + (r0), rmask, other=0).to(tl.float32)
        tmp225 = tmp224.to(tl.float32)
        tmp226 = 4096.0
        tmp227 = tmp222 / tmp226
        tmp228 = 1e-05
        tmp229 = tmp227 + tmp228
        tmp230 = tl.math.rsqrt(tmp229)
        tmp231 = tmp225 * tmp230
        tmp232 = tmp231.to(tl.float32)
        tmp234 = tmp232 * tmp233
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp234, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/nl/cnlnk7yrypr4iyls5yuagn2pns2h4khhyuuj7y43uf4rwlrgkyrt.py
# Source Nodes: [add_388, add_390, add_443, add_444, float_97, mean_48, mul_683, mul_684, mul_685, mul_686, rsqrt_48, type_as_96], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_388 => add_388
# add_390 => add_390
# add_443 => add_443
# add_444 => add_444
# float_97 => convert_element_type_288
# mean_48 => mean_48
# mul_683 => mul_707
# mul_684 => mul_708
# mul_685 => mul_709
# mul_686 => mul_710
# rsqrt_48 => rsqrt_48
# type_as_96 => convert_element_type_289
triton_red_fused__to_copy_add_mean_mul_rsqrt_32 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_32', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_32', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_32(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (23)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
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


# kernel path: /tmp/torchinductor_mengqy/dk/cdkx6kci274ixs7zf26srfrktame3ksizotx53ed4u2ut54omrdt.py
# Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_153, add_155, add_172, add_174, add_192, add_194, add_213, add_215, add_235, add_237, add_258, add_260, add_282, add_284, add_307, add_309, add_333, add_335, add_360, add_362, add_388, add_390, add_417, add_419, add_473, add_474, add_475, add_488, add_489, add_490, add_491, add_492, add_493, add_494, add_495, add_496, add_497, add_498, add_499, add_500, add_501, add_502, add_503, add_504, add_505, add_520, add_521, add_522, add_523, add_524, add_525, add_526, add_527, add_528, add_529, add_530, add_531, add_532, add_533, add_534, add_535, add_536, add_537, add_60, add_62, add_73, add_75, add_87, add_89, float_101, mean_50, mul_723, mul_724, mul_725, mul_726, mul_727, mul_748, mul_749, mul_750, mul_751, mul_752, mul_753, mul_754, mul_755, mul_756, mul_757, mul_758, mul_759, mul_760, mul_761, mul_762, mul_763, mul_764, mul_765, mul_790, mul_791, mul_792, mul_793, mul_794, mul_795, mul_796, mul_797, mul_798, mul_799, mul_800, mul_801, mul_802, mul_803, mul_804, mul_805, mul_806, mul_807, rsqrt_50, type_as_100], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_102 => add_102
# add_104 => add_104
# add_118 => add_118
# add_120 => add_120
# add_135 => add_135
# add_137 => add_137
# add_153 => add_153
# add_155 => add_155
# add_172 => add_172
# add_174 => add_174
# add_192 => add_192
# add_194 => add_194
# add_213 => add_213
# add_215 => add_215
# add_235 => add_235
# add_237 => add_237
# add_258 => add_258
# add_260 => add_260
# add_282 => add_282
# add_284 => add_284
# add_307 => add_307
# add_309 => add_309
# add_333 => add_333
# add_335 => add_335
# add_360 => add_360
# add_362 => add_362
# add_388 => add_388
# add_390 => add_390
# add_417 => add_417
# add_419 => add_419
# add_473 => add_473
# add_474 => add_474
# add_475 => add_475
# add_488 => add_488
# add_489 => add_489
# add_490 => add_490
# add_491 => add_491
# add_492 => add_492
# add_493 => add_493
# add_494 => add_494
# add_495 => add_495
# add_496 => add_496
# add_497 => add_497
# add_498 => add_498
# add_499 => add_499
# add_500 => add_500
# add_501 => add_501
# add_502 => add_502
# add_503 => add_503
# add_504 => add_504
# add_505 => add_505
# add_520 => add_520
# add_521 => add_521
# add_522 => add_522
# add_523 => add_523
# add_524 => add_524
# add_525 => add_525
# add_526 => add_526
# add_527 => add_527
# add_528 => add_528
# add_529 => add_529
# add_530 => add_530
# add_531 => add_531
# add_532 => add_532
# add_533 => add_533
# add_534 => add_534
# add_535 => add_535
# add_536 => add_536
# add_537 => add_537
# add_60 => add_60
# add_62 => add_62
# add_73 => add_73
# add_75 => add_75
# add_87 => add_87
# add_89 => add_89
# float_101 => convert_element_type_300
# mean_50 => mean_50
# mul_723 => mul_748
# mul_724 => mul_749
# mul_725 => mul_750
# mul_726 => mul_751
# mul_727 => mul_752
# mul_748 => mul_774
# mul_749 => mul_775
# mul_750 => mul_776
# mul_751 => mul_777
# mul_752 => mul_778
# mul_753 => mul_779
# mul_754 => mul_780
# mul_755 => mul_781
# mul_756 => mul_782
# mul_757 => mul_783
# mul_758 => mul_784
# mul_759 => mul_785
# mul_760 => mul_786
# mul_761 => mul_787
# mul_762 => mul_788
# mul_763 => mul_789
# mul_764 => mul_790
# mul_765 => mul_791
# mul_790 => mul_817
# mul_791 => mul_818
# mul_792 => mul_819
# mul_793 => mul_820
# mul_794 => mul_821
# mul_795 => mul_822
# mul_796 => mul_823
# mul_797 => mul_824
# mul_798 => mul_825
# mul_799 => mul_826
# mul_800 => mul_827
# mul_801 => mul_828
# mul_802 => mul_829
# mul_803 => mul_830
# mul_804 => mul_831
# mul_805 => mul_832
# mul_806 => mul_833
# mul_807 => mul_834
# rsqrt_50 => rsqrt_50
# type_as_100 => convert_element_type_301
triton_red_fused__to_copy_add_mean_mul_rsqrt_33 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_33', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: '*fp16', 21: '*fp16', 22: '*fp16', 23: '*fp16', 24: '*fp16', 25: '*fp16', 26: '*fp16', 27: '*fp16', 28: '*fp16', 29: '*fp16', 30: '*fp16', 31: '*fp16', 32: '*fp16', 33: '*fp16', 34: '*fp16', 35: '*fp16', 36: '*fp16', 37: '*fp16', 38: '*fp16', 39: '*fp16', 40: '*fp16', 41: '*fp16', 42: '*fp16', 43: '*fp16', 44: '*fp16', 45: '*fp16', 46: '*fp16', 47: '*fp16', 48: '*fp16', 49: '*fp16', 50: '*fp16', 51: '*fp16', 52: '*fp16', 53: '*fp16', 54: '*fp16', 55: '*fp16', 56: '*fp16', 57: '*fp16', 58: '*fp16', 59: '*fp16', 60: '*fp16', 61: '*fp16', 62: 'i32', 63: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_33', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 63), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(63,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_33(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (7)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp10 = tl.load(in_ptr0 + (8)).to(tl.float32)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp20 = tl.load(in_ptr7 + (7)).to(tl.float32)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp24 = tl.load(in_ptr7 + (8)).to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp28 = tl.load(in_ptr0 + (9)).to(tl.float32)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp37 = tl.load(in_ptr0 + (10)).to(tl.float32)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
    tmp46 = tl.load(in_ptr7 + (9)).to(tl.float32)
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
    tmp50 = tl.load(in_ptr7 + (10)).to(tl.float32)
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
    tmp54 = tl.load(in_ptr0 + (11)).to(tl.float32)
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK, RBLOCK])
    tmp63 = tl.load(in_ptr0 + (12)).to(tl.float32)
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, RBLOCK])
    tmp72 = tl.load(in_ptr7 + (11)).to(tl.float32)
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp76 = tl.load(in_ptr7 + (12)).to(tl.float32)
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK, RBLOCK])
    tmp80 = tl.load(in_ptr0 + (13)).to(tl.float32)
    tmp81 = tl.broadcast_to(tmp80, [XBLOCK, RBLOCK])
    tmp89 = tl.load(in_ptr0 + (14)).to(tl.float32)
    tmp90 = tl.broadcast_to(tmp89, [XBLOCK, RBLOCK])
    tmp98 = tl.load(in_ptr7 + (13)).to(tl.float32)
    tmp99 = tl.broadcast_to(tmp98, [XBLOCK, RBLOCK])
    tmp102 = tl.load(in_ptr7 + (14)).to(tl.float32)
    tmp103 = tl.broadcast_to(tmp102, [XBLOCK, RBLOCK])
    tmp106 = tl.load(in_ptr0 + (15)).to(tl.float32)
    tmp107 = tl.broadcast_to(tmp106, [XBLOCK, RBLOCK])
    tmp115 = tl.load(in_ptr0 + (16)).to(tl.float32)
    tmp116 = tl.broadcast_to(tmp115, [XBLOCK, RBLOCK])
    tmp124 = tl.load(in_ptr7 + (15)).to(tl.float32)
    tmp125 = tl.broadcast_to(tmp124, [XBLOCK, RBLOCK])
    tmp128 = tl.load(in_ptr7 + (16)).to(tl.float32)
    tmp129 = tl.broadcast_to(tmp128, [XBLOCK, RBLOCK])
    tmp132 = tl.load(in_ptr0 + (17)).to(tl.float32)
    tmp133 = tl.broadcast_to(tmp132, [XBLOCK, RBLOCK])
    tmp141 = tl.load(in_ptr0 + (18)).to(tl.float32)
    tmp142 = tl.broadcast_to(tmp141, [XBLOCK, RBLOCK])
    tmp150 = tl.load(in_ptr7 + (17)).to(tl.float32)
    tmp151 = tl.broadcast_to(tmp150, [XBLOCK, RBLOCK])
    tmp154 = tl.load(in_ptr7 + (18)).to(tl.float32)
    tmp155 = tl.broadcast_to(tmp154, [XBLOCK, RBLOCK])
    tmp158 = tl.load(in_ptr0 + (19)).to(tl.float32)
    tmp159 = tl.broadcast_to(tmp158, [XBLOCK, RBLOCK])
    tmp167 = tl.load(in_ptr0 + (20)).to(tl.float32)
    tmp168 = tl.broadcast_to(tmp167, [XBLOCK, RBLOCK])
    tmp176 = tl.load(in_ptr7 + (19)).to(tl.float32)
    tmp177 = tl.broadcast_to(tmp176, [XBLOCK, RBLOCK])
    tmp180 = tl.load(in_ptr7 + (20)).to(tl.float32)
    tmp181 = tl.broadcast_to(tmp180, [XBLOCK, RBLOCK])
    tmp184 = tl.load(in_ptr0 + (21)).to(tl.float32)
    tmp185 = tl.broadcast_to(tmp184, [XBLOCK, RBLOCK])
    tmp193 = tl.load(in_ptr0 + (22)).to(tl.float32)
    tmp194 = tl.broadcast_to(tmp193, [XBLOCK, RBLOCK])
    tmp202 = tl.load(in_ptr7 + (21)).to(tl.float32)
    tmp203 = tl.broadcast_to(tmp202, [XBLOCK, RBLOCK])
    tmp206 = tl.load(in_ptr7 + (22)).to(tl.float32)
    tmp207 = tl.broadcast_to(tmp206, [XBLOCK, RBLOCK])
    tmp211 = tl.load(in_ptr50 + (23)).to(tl.float32)
    tmp212 = tl.broadcast_to(tmp211, [XBLOCK, RBLOCK])
    tmp220 = tl.load(in_ptr50 + (24)).to(tl.float32)
    tmp221 = tl.broadcast_to(tmp220, [XBLOCK, RBLOCK])
    tmp229 = tl.load(in_ptr0 + (23)).to(tl.float32)
    tmp230 = tl.broadcast_to(tmp229, [XBLOCK, RBLOCK])
    tmp233 = tl.load(in_ptr0 + (24)).to(tl.float32)
    tmp234 = tl.broadcast_to(tmp233, [XBLOCK, RBLOCK])
    tmp237 = tl.load(in_ptr7 + (23)).to(tl.float32)
    tmp238 = tl.broadcast_to(tmp237, [XBLOCK, RBLOCK])
    tmp241 = tl.load(in_ptr7 + (24)).to(tl.float32)
    tmp242 = tl.broadcast_to(tmp241, [XBLOCK, RBLOCK])
    _tmp248 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr5 + (r0), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr6 + (r0), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_out_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp30 = tl.load(in_ptr8 + (r0), rmask, other=0).to(tl.float32)
        tmp31 = tl.load(in_ptr9 + (r0), rmask, other=0).to(tl.float32)
        tmp33 = tl.load(in_ptr10 + (r0), rmask, other=0).to(tl.float32)
        tmp39 = tl.load(in_ptr11 + (r0), rmask, other=0).to(tl.float32)
        tmp40 = tl.load(in_ptr12 + (r0), rmask, other=0).to(tl.float32)
        tmp42 = tl.load(in_ptr13 + (r0), rmask, other=0).to(tl.float32)
        tmp56 = tl.load(in_ptr14 + (r0), rmask, other=0).to(tl.float32)
        tmp57 = tl.load(in_ptr15 + (r0), rmask, other=0).to(tl.float32)
        tmp59 = tl.load(in_ptr16 + (r0), rmask, other=0).to(tl.float32)
        tmp65 = tl.load(in_ptr17 + (r0), rmask, other=0).to(tl.float32)
        tmp66 = tl.load(in_ptr18 + (r0), rmask, other=0).to(tl.float32)
        tmp68 = tl.load(in_ptr19 + (r0), rmask, other=0).to(tl.float32)
        tmp82 = tl.load(in_ptr20 + (r0), rmask, other=0).to(tl.float32)
        tmp83 = tl.load(in_ptr21 + (r0), rmask, other=0).to(tl.float32)
        tmp85 = tl.load(in_ptr22 + (r0), rmask, other=0).to(tl.float32)
        tmp91 = tl.load(in_ptr23 + (r0), rmask, other=0).to(tl.float32)
        tmp92 = tl.load(in_ptr24 + (r0), rmask, other=0).to(tl.float32)
        tmp94 = tl.load(in_ptr25 + (r0), rmask, other=0).to(tl.float32)
        tmp108 = tl.load(in_ptr26 + (r0), rmask, other=0).to(tl.float32)
        tmp109 = tl.load(in_ptr27 + (r0), rmask, other=0).to(tl.float32)
        tmp111 = tl.load(in_ptr28 + (r0), rmask, other=0).to(tl.float32)
        tmp117 = tl.load(in_ptr29 + (r0), rmask, other=0).to(tl.float32)
        tmp118 = tl.load(in_ptr30 + (r0), rmask, other=0).to(tl.float32)
        tmp120 = tl.load(in_ptr31 + (r0), rmask, other=0).to(tl.float32)
        tmp134 = tl.load(in_ptr32 + (r0), rmask, other=0).to(tl.float32)
        tmp135 = tl.load(in_ptr33 + (r0), rmask, other=0).to(tl.float32)
        tmp137 = tl.load(in_ptr34 + (r0), rmask, other=0).to(tl.float32)
        tmp143 = tl.load(in_ptr35 + (r0), rmask, other=0).to(tl.float32)
        tmp144 = tl.load(in_ptr36 + (r0), rmask, other=0).to(tl.float32)
        tmp146 = tl.load(in_ptr37 + (r0), rmask, other=0).to(tl.float32)
        tmp160 = tl.load(in_ptr38 + (r0), rmask, other=0).to(tl.float32)
        tmp161 = tl.load(in_ptr39 + (r0), rmask, other=0).to(tl.float32)
        tmp163 = tl.load(in_ptr40 + (r0), rmask, other=0).to(tl.float32)
        tmp169 = tl.load(in_ptr41 + (r0), rmask, other=0).to(tl.float32)
        tmp170 = tl.load(in_ptr42 + (r0), rmask, other=0).to(tl.float32)
        tmp172 = tl.load(in_ptr43 + (r0), rmask, other=0).to(tl.float32)
        tmp186 = tl.load(in_ptr44 + (r0), rmask, other=0).to(tl.float32)
        tmp187 = tl.load(in_ptr45 + (r0), rmask, other=0).to(tl.float32)
        tmp189 = tl.load(in_ptr46 + (r0), rmask, other=0).to(tl.float32)
        tmp195 = tl.load(in_ptr47 + (r0), rmask, other=0).to(tl.float32)
        tmp196 = tl.load(in_ptr48 + (r0), rmask, other=0).to(tl.float32)
        tmp198 = tl.load(in_ptr49 + (r0), rmask, other=0).to(tl.float32)
        tmp210 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp213 = tl.load(in_ptr51 + (r0), rmask, other=0).to(tl.float32)
        tmp214 = tl.load(in_ptr52 + (r0), rmask, other=0).to(tl.float32)
        tmp216 = tl.load(in_ptr53 + (r0), rmask, other=0).to(tl.float32)
        tmp222 = tl.load(in_ptr54 + (r0), rmask, other=0).to(tl.float32)
        tmp223 = tl.load(in_ptr55 + (r0), rmask, other=0).to(tl.float32)
        tmp225 = tl.load(in_ptr56 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp11 * tmp16
        tmp18 = tmp9 + tmp17
        tmp22 = tmp21 * tmp7
        tmp23 = tmp19 + tmp22
        tmp26 = tmp25 * tmp16
        tmp27 = tmp23 + tmp26
        tmp32 = tmp30 + tmp31
        tmp34 = tmp32 + tmp33
        tmp35 = tmp29 * tmp34
        tmp36 = tmp18 + tmp35
        tmp41 = tmp39 + tmp40
        tmp43 = tmp41 + tmp42
        tmp44 = tmp38 * tmp43
        tmp45 = tmp36 + tmp44
        tmp48 = tmp47 * tmp34
        tmp49 = tmp27 + tmp48
        tmp52 = tmp51 * tmp43
        tmp53 = tmp49 + tmp52
        tmp58 = tmp56 + tmp57
        tmp60 = tmp58 + tmp59
        tmp61 = tmp55 * tmp60
        tmp62 = tmp45 + tmp61
        tmp67 = tmp65 + tmp66
        tmp69 = tmp67 + tmp68
        tmp70 = tmp64 * tmp69
        tmp71 = tmp62 + tmp70
        tmp74 = tmp73 * tmp60
        tmp75 = tmp53 + tmp74
        tmp78 = tmp77 * tmp69
        tmp79 = tmp75 + tmp78
        tmp84 = tmp82 + tmp83
        tmp86 = tmp84 + tmp85
        tmp87 = tmp81 * tmp86
        tmp88 = tmp71 + tmp87
        tmp93 = tmp91 + tmp92
        tmp95 = tmp93 + tmp94
        tmp96 = tmp90 * tmp95
        tmp97 = tmp88 + tmp96
        tmp100 = tmp99 * tmp86
        tmp101 = tmp79 + tmp100
        tmp104 = tmp103 * tmp95
        tmp105 = tmp101 + tmp104
        tmp110 = tmp108 + tmp109
        tmp112 = tmp110 + tmp111
        tmp113 = tmp107 * tmp112
        tmp114 = tmp97 + tmp113
        tmp119 = tmp117 + tmp118
        tmp121 = tmp119 + tmp120
        tmp122 = tmp116 * tmp121
        tmp123 = tmp114 + tmp122
        tmp126 = tmp125 * tmp112
        tmp127 = tmp105 + tmp126
        tmp130 = tmp129 * tmp121
        tmp131 = tmp127 + tmp130
        tmp136 = tmp134 + tmp135
        tmp138 = tmp136 + tmp137
        tmp139 = tmp133 * tmp138
        tmp140 = tmp123 + tmp139
        tmp145 = tmp143 + tmp144
        tmp147 = tmp145 + tmp146
        tmp148 = tmp142 * tmp147
        tmp149 = tmp140 + tmp148
        tmp152 = tmp151 * tmp138
        tmp153 = tmp131 + tmp152
        tmp156 = tmp155 * tmp147
        tmp157 = tmp153 + tmp156
        tmp162 = tmp160 + tmp161
        tmp164 = tmp162 + tmp163
        tmp165 = tmp159 * tmp164
        tmp166 = tmp149 + tmp165
        tmp171 = tmp169 + tmp170
        tmp173 = tmp171 + tmp172
        tmp174 = tmp168 * tmp173
        tmp175 = tmp166 + tmp174
        tmp178 = tmp177 * tmp164
        tmp179 = tmp157 + tmp178
        tmp182 = tmp181 * tmp173
        tmp183 = tmp179 + tmp182
        tmp188 = tmp186 + tmp187
        tmp190 = tmp188 + tmp189
        tmp191 = tmp185 * tmp190
        tmp192 = tmp175 + tmp191
        tmp197 = tmp195 + tmp196
        tmp199 = tmp197 + tmp198
        tmp200 = tmp194 * tmp199
        tmp201 = tmp192 + tmp200
        tmp204 = tmp203 * tmp190
        tmp205 = tmp183 + tmp204
        tmp208 = tmp207 * tmp199
        tmp209 = tmp205 + tmp208
        tmp215 = tmp213 + tmp214
        tmp217 = tmp215 + tmp216
        tmp218 = tmp212 * tmp217
        tmp219 = tmp210 + tmp218
        tmp224 = tmp222 + tmp223
        tmp226 = tmp224 + tmp225
        tmp227 = tmp221 * tmp226
        tmp228 = tmp219 + tmp227
        tmp231 = tmp230 * tmp217
        tmp232 = tmp201 + tmp231
        tmp235 = tmp234 * tmp226
        tmp236 = tmp232 + tmp235
        tmp239 = tmp238 * tmp217
        tmp240 = tmp209 + tmp239
        tmp243 = tmp242 * tmp226
        tmp244 = tmp240 + tmp243
        tmp245 = tmp228.to(tl.float32)
        tmp246 = tmp245 * tmp245
        tmp247 = tl.broadcast_to(tmp246, [XBLOCK, RBLOCK])
        tmp249 = _tmp248 + tmp247
        _tmp248 = tl.where(rmask, tmp249, _tmp248)
        tl.store(in_out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp228, rmask)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp236, rmask)
        tl.store(in_out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp244, rmask)
    tmp248 = tl.sum(_tmp248, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp250 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp259 = tl.load(in_ptr57 + (r0), rmask, other=0).to(tl.float32)
        tmp251 = tmp250.to(tl.float32)
        tmp252 = 4096.0
        tmp253 = tmp248 / tmp252
        tmp254 = 1e-05
        tmp255 = tmp253 + tmp254
        tmp256 = tl.math.rsqrt(tmp255)
        tmp257 = tmp251 * tmp256
        tmp258 = tmp257.to(tl.float32)
        tmp260 = tmp258 * tmp259
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp260, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/vu/cvuflg2qmtaziupe7llplquxhp3zyxebkzk7amyvgrlspckb6x6l.py
# Source Nodes: [add_447, add_449, add_506, add_507, float_105, mean_52, mul_766, mul_767, mul_768, mul_769, rsqrt_52, type_as_104], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_447 => add_447
# add_449 => add_449
# add_506 => add_506
# add_507 => add_507
# float_105 => convert_element_type_312
# mean_52 => mean_52
# mul_766 => mul_792
# mul_767 => mul_793
# mul_768 => mul_794
# mul_769 => mul_795
# rsqrt_52 => rsqrt_52
# type_as_104 => convert_element_type_313
triton_red_fused__to_copy_add_mean_mul_rsqrt_34 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_34', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_34', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_34(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (25)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
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


# kernel path: /tmp/torchinductor_mengqy/pv/cpvkfuc4oixsnoymlwi6qoqp5lrckrupa4nkjtwkltgyv43uaat5.py
# Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_153, add_155, add_172, add_174, add_192, add_194, add_213, add_215, add_235, add_237, add_258, add_260, add_282, add_284, add_307, add_309, add_333, add_335, add_360, add_362, add_388, add_390, add_417, add_419, add_447, add_449, add_478, add_480, add_538, add_539, add_540, add_553, add_554, add_555, add_556, add_557, add_558, add_559, add_560, add_561, add_562, add_563, add_564, add_565, add_566, add_567, add_568, add_569, add_570, add_571, add_572, add_587, add_588, add_589, add_590, add_591, add_592, add_593, add_594, add_595, add_596, add_597, add_598, add_599, add_60, add_600, add_601, add_602, add_603, add_604, add_605, add_606, add_62, add_73, add_75, add_87, add_89, float_109, mean_54, mul_808, mul_809, mul_810, mul_811, mul_812, mul_833, mul_834, mul_835, mul_836, mul_837, mul_838, mul_839, mul_840, mul_841, mul_842, mul_843, mul_844, mul_845, mul_846, mul_847, mul_848, mul_849, mul_850, mul_851, mul_852, mul_877, mul_878, mul_879, mul_880, mul_881, mul_882, mul_883, mul_884, mul_885, mul_886, mul_887, mul_888, mul_889, mul_890, mul_891, mul_892, mul_893, mul_894, mul_895, mul_896, rsqrt_54, type_as_108], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_102 => add_102
# add_104 => add_104
# add_118 => add_118
# add_120 => add_120
# add_135 => add_135
# add_137 => add_137
# add_153 => add_153
# add_155 => add_155
# add_172 => add_172
# add_174 => add_174
# add_192 => add_192
# add_194 => add_194
# add_213 => add_213
# add_215 => add_215
# add_235 => add_235
# add_237 => add_237
# add_258 => add_258
# add_260 => add_260
# add_282 => add_282
# add_284 => add_284
# add_307 => add_307
# add_309 => add_309
# add_333 => add_333
# add_335 => add_335
# add_360 => add_360
# add_362 => add_362
# add_388 => add_388
# add_390 => add_390
# add_417 => add_417
# add_419 => add_419
# add_447 => add_447
# add_449 => add_449
# add_478 => add_478
# add_480 => add_480
# add_538 => add_538
# add_539 => add_539
# add_540 => add_540
# add_553 => add_553
# add_554 => add_554
# add_555 => add_555
# add_556 => add_556
# add_557 => add_557
# add_558 => add_558
# add_559 => add_559
# add_560 => add_560
# add_561 => add_561
# add_562 => add_562
# add_563 => add_563
# add_564 => add_564
# add_565 => add_565
# add_566 => add_566
# add_567 => add_567
# add_568 => add_568
# add_569 => add_569
# add_570 => add_570
# add_571 => add_571
# add_572 => add_572
# add_587 => add_587
# add_588 => add_588
# add_589 => add_589
# add_590 => add_590
# add_591 => add_591
# add_592 => add_592
# add_593 => add_593
# add_594 => add_594
# add_595 => add_595
# add_596 => add_596
# add_597 => add_597
# add_598 => add_598
# add_599 => add_599
# add_60 => add_60
# add_600 => add_600
# add_601 => add_601
# add_602 => add_602
# add_603 => add_603
# add_604 => add_604
# add_605 => add_605
# add_606 => add_606
# add_62 => add_62
# add_73 => add_73
# add_75 => add_75
# add_87 => add_87
# add_89 => add_89
# float_109 => convert_element_type_324
# mean_54 => mean_54
# mul_808 => mul_835
# mul_809 => mul_836
# mul_810 => mul_837
# mul_811 => mul_838
# mul_812 => mul_839
# mul_833 => mul_861
# mul_834 => mul_862
# mul_835 => mul_863
# mul_836 => mul_864
# mul_837 => mul_865
# mul_838 => mul_866
# mul_839 => mul_867
# mul_840 => mul_868
# mul_841 => mul_869
# mul_842 => mul_870
# mul_843 => mul_871
# mul_844 => mul_872
# mul_845 => mul_873
# mul_846 => mul_874
# mul_847 => mul_875
# mul_848 => mul_876
# mul_849 => mul_877
# mul_850 => mul_878
# mul_851 => mul_879
# mul_852 => mul_880
# mul_877 => mul_906
# mul_878 => mul_907
# mul_879 => mul_908
# mul_880 => mul_909
# mul_881 => mul_910
# mul_882 => mul_911
# mul_883 => mul_912
# mul_884 => mul_913
# mul_885 => mul_914
# mul_886 => mul_915
# mul_887 => mul_916
# mul_888 => mul_917
# mul_889 => mul_918
# mul_890 => mul_919
# mul_891 => mul_920
# mul_892 => mul_921
# mul_893 => mul_922
# mul_894 => mul_923
# mul_895 => mul_924
# mul_896 => mul_925
# rsqrt_54 => rsqrt_54
# type_as_108 => convert_element_type_325
triton_red_fused__to_copy_add_mean_mul_rsqrt_35 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_35', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: '*fp16', 21: '*fp16', 22: '*fp16', 23: '*fp16', 24: '*fp16', 25: '*fp16', 26: '*fp16', 27: '*fp16', 28: '*fp16', 29: '*fp16', 30: '*fp16', 31: '*fp16', 32: '*fp16', 33: '*fp16', 34: '*fp16', 35: '*fp16', 36: '*fp16', 37: '*fp16', 38: '*fp16', 39: '*fp16', 40: '*fp16', 41: '*fp16', 42: '*fp16', 43: '*fp16', 44: '*fp16', 45: '*fp16', 46: '*fp16', 47: '*fp16', 48: '*fp16', 49: '*fp16', 50: '*fp16', 51: '*fp16', 52: '*fp16', 53: '*fp16', 54: '*fp16', 55: '*fp16', 56: '*fp16', 57: '*fp16', 58: '*fp16', 59: '*fp16', 60: '*fp16', 61: '*fp16', 62: '*fp16', 63: '*fp16', 64: '*fp16', 65: '*fp16', 66: '*fp16', 67: '*fp16', 68: 'i32', 69: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_35', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 69), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(69,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_35(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (7)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp10 = tl.load(in_ptr0 + (8)).to(tl.float32)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp20 = tl.load(in_ptr7 + (7)).to(tl.float32)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp24 = tl.load(in_ptr7 + (8)).to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp28 = tl.load(in_ptr0 + (9)).to(tl.float32)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp37 = tl.load(in_ptr0 + (10)).to(tl.float32)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
    tmp46 = tl.load(in_ptr7 + (9)).to(tl.float32)
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
    tmp50 = tl.load(in_ptr7 + (10)).to(tl.float32)
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
    tmp54 = tl.load(in_ptr0 + (11)).to(tl.float32)
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK, RBLOCK])
    tmp63 = tl.load(in_ptr0 + (12)).to(tl.float32)
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, RBLOCK])
    tmp72 = tl.load(in_ptr7 + (11)).to(tl.float32)
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp76 = tl.load(in_ptr7 + (12)).to(tl.float32)
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK, RBLOCK])
    tmp80 = tl.load(in_ptr0 + (13)).to(tl.float32)
    tmp81 = tl.broadcast_to(tmp80, [XBLOCK, RBLOCK])
    tmp89 = tl.load(in_ptr0 + (14)).to(tl.float32)
    tmp90 = tl.broadcast_to(tmp89, [XBLOCK, RBLOCK])
    tmp98 = tl.load(in_ptr7 + (13)).to(tl.float32)
    tmp99 = tl.broadcast_to(tmp98, [XBLOCK, RBLOCK])
    tmp102 = tl.load(in_ptr7 + (14)).to(tl.float32)
    tmp103 = tl.broadcast_to(tmp102, [XBLOCK, RBLOCK])
    tmp106 = tl.load(in_ptr0 + (15)).to(tl.float32)
    tmp107 = tl.broadcast_to(tmp106, [XBLOCK, RBLOCK])
    tmp115 = tl.load(in_ptr0 + (16)).to(tl.float32)
    tmp116 = tl.broadcast_to(tmp115, [XBLOCK, RBLOCK])
    tmp124 = tl.load(in_ptr7 + (15)).to(tl.float32)
    tmp125 = tl.broadcast_to(tmp124, [XBLOCK, RBLOCK])
    tmp128 = tl.load(in_ptr7 + (16)).to(tl.float32)
    tmp129 = tl.broadcast_to(tmp128, [XBLOCK, RBLOCK])
    tmp132 = tl.load(in_ptr0 + (17)).to(tl.float32)
    tmp133 = tl.broadcast_to(tmp132, [XBLOCK, RBLOCK])
    tmp141 = tl.load(in_ptr0 + (18)).to(tl.float32)
    tmp142 = tl.broadcast_to(tmp141, [XBLOCK, RBLOCK])
    tmp150 = tl.load(in_ptr7 + (17)).to(tl.float32)
    tmp151 = tl.broadcast_to(tmp150, [XBLOCK, RBLOCK])
    tmp154 = tl.load(in_ptr7 + (18)).to(tl.float32)
    tmp155 = tl.broadcast_to(tmp154, [XBLOCK, RBLOCK])
    tmp158 = tl.load(in_ptr0 + (19)).to(tl.float32)
    tmp159 = tl.broadcast_to(tmp158, [XBLOCK, RBLOCK])
    tmp167 = tl.load(in_ptr0 + (20)).to(tl.float32)
    tmp168 = tl.broadcast_to(tmp167, [XBLOCK, RBLOCK])
    tmp176 = tl.load(in_ptr7 + (19)).to(tl.float32)
    tmp177 = tl.broadcast_to(tmp176, [XBLOCK, RBLOCK])
    tmp180 = tl.load(in_ptr7 + (20)).to(tl.float32)
    tmp181 = tl.broadcast_to(tmp180, [XBLOCK, RBLOCK])
    tmp184 = tl.load(in_ptr0 + (21)).to(tl.float32)
    tmp185 = tl.broadcast_to(tmp184, [XBLOCK, RBLOCK])
    tmp193 = tl.load(in_ptr0 + (22)).to(tl.float32)
    tmp194 = tl.broadcast_to(tmp193, [XBLOCK, RBLOCK])
    tmp202 = tl.load(in_ptr7 + (21)).to(tl.float32)
    tmp203 = tl.broadcast_to(tmp202, [XBLOCK, RBLOCK])
    tmp206 = tl.load(in_ptr7 + (22)).to(tl.float32)
    tmp207 = tl.broadcast_to(tmp206, [XBLOCK, RBLOCK])
    tmp210 = tl.load(in_ptr0 + (23)).to(tl.float32)
    tmp211 = tl.broadcast_to(tmp210, [XBLOCK, RBLOCK])
    tmp219 = tl.load(in_ptr0 + (24)).to(tl.float32)
    tmp220 = tl.broadcast_to(tmp219, [XBLOCK, RBLOCK])
    tmp228 = tl.load(in_ptr7 + (23)).to(tl.float32)
    tmp229 = tl.broadcast_to(tmp228, [XBLOCK, RBLOCK])
    tmp232 = tl.load(in_ptr7 + (24)).to(tl.float32)
    tmp233 = tl.broadcast_to(tmp232, [XBLOCK, RBLOCK])
    tmp237 = tl.load(in_ptr56 + (25)).to(tl.float32)
    tmp238 = tl.broadcast_to(tmp237, [XBLOCK, RBLOCK])
    tmp246 = tl.load(in_ptr56 + (26)).to(tl.float32)
    tmp247 = tl.broadcast_to(tmp246, [XBLOCK, RBLOCK])
    tmp255 = tl.load(in_ptr0 + (25)).to(tl.float32)
    tmp256 = tl.broadcast_to(tmp255, [XBLOCK, RBLOCK])
    tmp259 = tl.load(in_ptr0 + (26)).to(tl.float32)
    tmp260 = tl.broadcast_to(tmp259, [XBLOCK, RBLOCK])
    tmp263 = tl.load(in_ptr7 + (25)).to(tl.float32)
    tmp264 = tl.broadcast_to(tmp263, [XBLOCK, RBLOCK])
    tmp267 = tl.load(in_ptr7 + (26)).to(tl.float32)
    tmp268 = tl.broadcast_to(tmp267, [XBLOCK, RBLOCK])
    _tmp274 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr5 + (r0), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr6 + (r0), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_out_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp30 = tl.load(in_ptr8 + (r0), rmask, other=0).to(tl.float32)
        tmp31 = tl.load(in_ptr9 + (r0), rmask, other=0).to(tl.float32)
        tmp33 = tl.load(in_ptr10 + (r0), rmask, other=0).to(tl.float32)
        tmp39 = tl.load(in_ptr11 + (r0), rmask, other=0).to(tl.float32)
        tmp40 = tl.load(in_ptr12 + (r0), rmask, other=0).to(tl.float32)
        tmp42 = tl.load(in_ptr13 + (r0), rmask, other=0).to(tl.float32)
        tmp56 = tl.load(in_ptr14 + (r0), rmask, other=0).to(tl.float32)
        tmp57 = tl.load(in_ptr15 + (r0), rmask, other=0).to(tl.float32)
        tmp59 = tl.load(in_ptr16 + (r0), rmask, other=0).to(tl.float32)
        tmp65 = tl.load(in_ptr17 + (r0), rmask, other=0).to(tl.float32)
        tmp66 = tl.load(in_ptr18 + (r0), rmask, other=0).to(tl.float32)
        tmp68 = tl.load(in_ptr19 + (r0), rmask, other=0).to(tl.float32)
        tmp82 = tl.load(in_ptr20 + (r0), rmask, other=0).to(tl.float32)
        tmp83 = tl.load(in_ptr21 + (r0), rmask, other=0).to(tl.float32)
        tmp85 = tl.load(in_ptr22 + (r0), rmask, other=0).to(tl.float32)
        tmp91 = tl.load(in_ptr23 + (r0), rmask, other=0).to(tl.float32)
        tmp92 = tl.load(in_ptr24 + (r0), rmask, other=0).to(tl.float32)
        tmp94 = tl.load(in_ptr25 + (r0), rmask, other=0).to(tl.float32)
        tmp108 = tl.load(in_ptr26 + (r0), rmask, other=0).to(tl.float32)
        tmp109 = tl.load(in_ptr27 + (r0), rmask, other=0).to(tl.float32)
        tmp111 = tl.load(in_ptr28 + (r0), rmask, other=0).to(tl.float32)
        tmp117 = tl.load(in_ptr29 + (r0), rmask, other=0).to(tl.float32)
        tmp118 = tl.load(in_ptr30 + (r0), rmask, other=0).to(tl.float32)
        tmp120 = tl.load(in_ptr31 + (r0), rmask, other=0).to(tl.float32)
        tmp134 = tl.load(in_ptr32 + (r0), rmask, other=0).to(tl.float32)
        tmp135 = tl.load(in_ptr33 + (r0), rmask, other=0).to(tl.float32)
        tmp137 = tl.load(in_ptr34 + (r0), rmask, other=0).to(tl.float32)
        tmp143 = tl.load(in_ptr35 + (r0), rmask, other=0).to(tl.float32)
        tmp144 = tl.load(in_ptr36 + (r0), rmask, other=0).to(tl.float32)
        tmp146 = tl.load(in_ptr37 + (r0), rmask, other=0).to(tl.float32)
        tmp160 = tl.load(in_ptr38 + (r0), rmask, other=0).to(tl.float32)
        tmp161 = tl.load(in_ptr39 + (r0), rmask, other=0).to(tl.float32)
        tmp163 = tl.load(in_ptr40 + (r0), rmask, other=0).to(tl.float32)
        tmp169 = tl.load(in_ptr41 + (r0), rmask, other=0).to(tl.float32)
        tmp170 = tl.load(in_ptr42 + (r0), rmask, other=0).to(tl.float32)
        tmp172 = tl.load(in_ptr43 + (r0), rmask, other=0).to(tl.float32)
        tmp186 = tl.load(in_ptr44 + (r0), rmask, other=0).to(tl.float32)
        tmp187 = tl.load(in_ptr45 + (r0), rmask, other=0).to(tl.float32)
        tmp189 = tl.load(in_ptr46 + (r0), rmask, other=0).to(tl.float32)
        tmp195 = tl.load(in_ptr47 + (r0), rmask, other=0).to(tl.float32)
        tmp196 = tl.load(in_ptr48 + (r0), rmask, other=0).to(tl.float32)
        tmp198 = tl.load(in_ptr49 + (r0), rmask, other=0).to(tl.float32)
        tmp212 = tl.load(in_ptr50 + (r0), rmask, other=0).to(tl.float32)
        tmp213 = tl.load(in_ptr51 + (r0), rmask, other=0).to(tl.float32)
        tmp215 = tl.load(in_ptr52 + (r0), rmask, other=0).to(tl.float32)
        tmp221 = tl.load(in_ptr53 + (r0), rmask, other=0).to(tl.float32)
        tmp222 = tl.load(in_ptr54 + (r0), rmask, other=0).to(tl.float32)
        tmp224 = tl.load(in_ptr55 + (r0), rmask, other=0).to(tl.float32)
        tmp236 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp239 = tl.load(in_ptr57 + (r0), rmask, other=0).to(tl.float32)
        tmp240 = tl.load(in_ptr58 + (r0), rmask, other=0).to(tl.float32)
        tmp242 = tl.load(in_ptr59 + (r0), rmask, other=0).to(tl.float32)
        tmp248 = tl.load(in_ptr60 + (r0), rmask, other=0).to(tl.float32)
        tmp249 = tl.load(in_ptr61 + (r0), rmask, other=0).to(tl.float32)
        tmp251 = tl.load(in_ptr62 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp11 * tmp16
        tmp18 = tmp9 + tmp17
        tmp22 = tmp21 * tmp7
        tmp23 = tmp19 + tmp22
        tmp26 = tmp25 * tmp16
        tmp27 = tmp23 + tmp26
        tmp32 = tmp30 + tmp31
        tmp34 = tmp32 + tmp33
        tmp35 = tmp29 * tmp34
        tmp36 = tmp18 + tmp35
        tmp41 = tmp39 + tmp40
        tmp43 = tmp41 + tmp42
        tmp44 = tmp38 * tmp43
        tmp45 = tmp36 + tmp44
        tmp48 = tmp47 * tmp34
        tmp49 = tmp27 + tmp48
        tmp52 = tmp51 * tmp43
        tmp53 = tmp49 + tmp52
        tmp58 = tmp56 + tmp57
        tmp60 = tmp58 + tmp59
        tmp61 = tmp55 * tmp60
        tmp62 = tmp45 + tmp61
        tmp67 = tmp65 + tmp66
        tmp69 = tmp67 + tmp68
        tmp70 = tmp64 * tmp69
        tmp71 = tmp62 + tmp70
        tmp74 = tmp73 * tmp60
        tmp75 = tmp53 + tmp74
        tmp78 = tmp77 * tmp69
        tmp79 = tmp75 + tmp78
        tmp84 = tmp82 + tmp83
        tmp86 = tmp84 + tmp85
        tmp87 = tmp81 * tmp86
        tmp88 = tmp71 + tmp87
        tmp93 = tmp91 + tmp92
        tmp95 = tmp93 + tmp94
        tmp96 = tmp90 * tmp95
        tmp97 = tmp88 + tmp96
        tmp100 = tmp99 * tmp86
        tmp101 = tmp79 + tmp100
        tmp104 = tmp103 * tmp95
        tmp105 = tmp101 + tmp104
        tmp110 = tmp108 + tmp109
        tmp112 = tmp110 + tmp111
        tmp113 = tmp107 * tmp112
        tmp114 = tmp97 + tmp113
        tmp119 = tmp117 + tmp118
        tmp121 = tmp119 + tmp120
        tmp122 = tmp116 * tmp121
        tmp123 = tmp114 + tmp122
        tmp126 = tmp125 * tmp112
        tmp127 = tmp105 + tmp126
        tmp130 = tmp129 * tmp121
        tmp131 = tmp127 + tmp130
        tmp136 = tmp134 + tmp135
        tmp138 = tmp136 + tmp137
        tmp139 = tmp133 * tmp138
        tmp140 = tmp123 + tmp139
        tmp145 = tmp143 + tmp144
        tmp147 = tmp145 + tmp146
        tmp148 = tmp142 * tmp147
        tmp149 = tmp140 + tmp148
        tmp152 = tmp151 * tmp138
        tmp153 = tmp131 + tmp152
        tmp156 = tmp155 * tmp147
        tmp157 = tmp153 + tmp156
        tmp162 = tmp160 + tmp161
        tmp164 = tmp162 + tmp163
        tmp165 = tmp159 * tmp164
        tmp166 = tmp149 + tmp165
        tmp171 = tmp169 + tmp170
        tmp173 = tmp171 + tmp172
        tmp174 = tmp168 * tmp173
        tmp175 = tmp166 + tmp174
        tmp178 = tmp177 * tmp164
        tmp179 = tmp157 + tmp178
        tmp182 = tmp181 * tmp173
        tmp183 = tmp179 + tmp182
        tmp188 = tmp186 + tmp187
        tmp190 = tmp188 + tmp189
        tmp191 = tmp185 * tmp190
        tmp192 = tmp175 + tmp191
        tmp197 = tmp195 + tmp196
        tmp199 = tmp197 + tmp198
        tmp200 = tmp194 * tmp199
        tmp201 = tmp192 + tmp200
        tmp204 = tmp203 * tmp190
        tmp205 = tmp183 + tmp204
        tmp208 = tmp207 * tmp199
        tmp209 = tmp205 + tmp208
        tmp214 = tmp212 + tmp213
        tmp216 = tmp214 + tmp215
        tmp217 = tmp211 * tmp216
        tmp218 = tmp201 + tmp217
        tmp223 = tmp221 + tmp222
        tmp225 = tmp223 + tmp224
        tmp226 = tmp220 * tmp225
        tmp227 = tmp218 + tmp226
        tmp230 = tmp229 * tmp216
        tmp231 = tmp209 + tmp230
        tmp234 = tmp233 * tmp225
        tmp235 = tmp231 + tmp234
        tmp241 = tmp239 + tmp240
        tmp243 = tmp241 + tmp242
        tmp244 = tmp238 * tmp243
        tmp245 = tmp236 + tmp244
        tmp250 = tmp248 + tmp249
        tmp252 = tmp250 + tmp251
        tmp253 = tmp247 * tmp252
        tmp254 = tmp245 + tmp253
        tmp257 = tmp256 * tmp243
        tmp258 = tmp227 + tmp257
        tmp261 = tmp260 * tmp252
        tmp262 = tmp258 + tmp261
        tmp265 = tmp264 * tmp243
        tmp266 = tmp235 + tmp265
        tmp269 = tmp268 * tmp252
        tmp270 = tmp266 + tmp269
        tmp271 = tmp254.to(tl.float32)
        tmp272 = tmp271 * tmp271
        tmp273 = tl.broadcast_to(tmp272, [XBLOCK, RBLOCK])
        tmp275 = _tmp274 + tmp273
        _tmp274 = tl.where(rmask, tmp275, _tmp274)
        tl.store(in_out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp254, rmask)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp262, rmask)
        tl.store(in_out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp270, rmask)
    tmp274 = tl.sum(_tmp274, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp276 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp285 = tl.load(in_ptr63 + (r0), rmask, other=0).to(tl.float32)
        tmp277 = tmp276.to(tl.float32)
        tmp278 = 4096.0
        tmp279 = tmp274 / tmp278
        tmp280 = 1e-05
        tmp281 = tmp279 + tmp280
        tmp282 = tl.math.rsqrt(tmp281)
        tmp283 = tmp277 * tmp282
        tmp284 = tmp283.to(tl.float32)
        tmp286 = tmp284 * tmp285
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp286, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/td/ctdpsv63isvxryurv3elzwmmijvftjfvys3ik4co66mjy23tkqwq.py
# Source Nodes: [add_510, add_512, add_573, add_574, float_113, mean_56, mul_853, mul_854, mul_855, mul_856, rsqrt_56, type_as_112], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_510 => add_510
# add_512 => add_512
# add_573 => add_573
# add_574 => add_574
# float_113 => convert_element_type_336
# mean_56 => mean_56
# mul_853 => mul_881
# mul_854 => mul_882
# mul_855 => mul_883
# mul_856 => mul_884
# rsqrt_56 => rsqrt_56
# type_as_112 => convert_element_type_337
triton_red_fused__to_copy_add_mean_mul_rsqrt_36 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_36', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_36', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_36(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (27)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
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


# kernel path: /tmp/torchinductor_mengqy/7d/c7d7s5u2opl4jib6xu4zfgdzugho4jj4nppzptfystoboifnayqg.py
# Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_153, add_155, add_172, add_174, add_192, add_194, add_213, add_215, add_235, add_237, add_258, add_260, add_282, add_284, add_307, add_309, add_333, add_335, add_360, add_362, add_388, add_390, add_417, add_419, add_447, add_449, add_478, add_480, add_510, add_512, add_543, add_545, add_60, add_607, add_608, add_609, add_62, add_622, add_623, add_624, add_625, add_626, add_627, add_628, add_629, add_630, add_631, add_632, add_633, add_634, add_635, add_636, add_637, add_638, add_639, add_640, add_641, add_642, add_643, add_658, add_659, add_660, add_661, add_662, add_663, add_664, add_665, add_666, add_667, add_668, add_669, add_670, add_671, add_672, add_673, add_674, add_675, add_676, add_677, add_678, add_679, add_73, add_75, add_87, add_89, float_117, mean_58, mul_897, mul_898, mul_899, mul_900, mul_901, mul_922, mul_923, mul_924, mul_925, mul_926, mul_927, mul_928, mul_929, mul_930, mul_931, mul_932, mul_933, mul_934, mul_935, mul_936, mul_937, mul_938, mul_939, mul_940, mul_941, mul_942, mul_943, mul_968, mul_969, mul_970, mul_971, mul_972, mul_973, mul_974, mul_975, mul_976, mul_977, mul_978, mul_979, mul_980, mul_981, mul_982, mul_983, mul_984, mul_985, mul_986, mul_987, mul_988, mul_989, rsqrt_58, type_as_116], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_102 => add_102
# add_104 => add_104
# add_118 => add_118
# add_120 => add_120
# add_135 => add_135
# add_137 => add_137
# add_153 => add_153
# add_155 => add_155
# add_172 => add_172
# add_174 => add_174
# add_192 => add_192
# add_194 => add_194
# add_213 => add_213
# add_215 => add_215
# add_235 => add_235
# add_237 => add_237
# add_258 => add_258
# add_260 => add_260
# add_282 => add_282
# add_284 => add_284
# add_307 => add_307
# add_309 => add_309
# add_333 => add_333
# add_335 => add_335
# add_360 => add_360
# add_362 => add_362
# add_388 => add_388
# add_390 => add_390
# add_417 => add_417
# add_419 => add_419
# add_447 => add_447
# add_449 => add_449
# add_478 => add_478
# add_480 => add_480
# add_510 => add_510
# add_512 => add_512
# add_543 => add_543
# add_545 => add_545
# add_60 => add_60
# add_607 => add_607
# add_608 => add_608
# add_609 => add_609
# add_62 => add_62
# add_622 => add_622
# add_623 => add_623
# add_624 => add_624
# add_625 => add_625
# add_626 => add_626
# add_627 => add_627
# add_628 => add_628
# add_629 => add_629
# add_630 => add_630
# add_631 => add_631
# add_632 => add_632
# add_633 => add_633
# add_634 => add_634
# add_635 => add_635
# add_636 => add_636
# add_637 => add_637
# add_638 => add_638
# add_639 => add_639
# add_640 => add_640
# add_641 => add_641
# add_642 => add_642
# add_643 => add_643
# add_658 => add_658
# add_659 => add_659
# add_660 => add_660
# add_661 => add_661
# add_662 => add_662
# add_663 => add_663
# add_664 => add_664
# add_665 => add_665
# add_666 => add_666
# add_667 => add_667
# add_668 => add_668
# add_669 => add_669
# add_670 => add_670
# add_671 => add_671
# add_672 => add_672
# add_673 => add_673
# add_674 => add_674
# add_675 => add_675
# add_676 => add_676
# add_677 => add_677
# add_678 => add_678
# add_679 => add_679
# add_73 => add_73
# add_75 => add_75
# add_87 => add_87
# add_89 => add_89
# float_117 => convert_element_type_348
# mean_58 => mean_58
# mul_897 => mul_926
# mul_898 => mul_927
# mul_899 => mul_928
# mul_900 => mul_929
# mul_901 => mul_930
# mul_922 => mul_952
# mul_923 => mul_953
# mul_924 => mul_954
# mul_925 => mul_955
# mul_926 => mul_956
# mul_927 => mul_957
# mul_928 => mul_958
# mul_929 => mul_959
# mul_930 => mul_960
# mul_931 => mul_961
# mul_932 => mul_962
# mul_933 => mul_963
# mul_934 => mul_964
# mul_935 => mul_965
# mul_936 => mul_966
# mul_937 => mul_967
# mul_938 => mul_968
# mul_939 => mul_969
# mul_940 => mul_970
# mul_941 => mul_971
# mul_942 => mul_972
# mul_943 => mul_973
# mul_968 => mul_999
# mul_969 => mul_1000
# mul_970 => mul_1001
# mul_971 => mul_1002
# mul_972 => mul_1003
# mul_973 => mul_1004
# mul_974 => mul_1005
# mul_975 => mul_1006
# mul_976 => mul_1007
# mul_977 => mul_1008
# mul_978 => mul_1009
# mul_979 => mul_1010
# mul_980 => mul_1011
# mul_981 => mul_1012
# mul_982 => mul_1013
# mul_983 => mul_1014
# mul_984 => mul_1015
# mul_985 => mul_1016
# mul_986 => mul_1017
# mul_987 => mul_1018
# mul_988 => mul_1019
# mul_989 => mul_1020
# rsqrt_58 => rsqrt_58
# type_as_116 => convert_element_type_349
triton_red_fused__to_copy_add_mean_mul_rsqrt_37 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_37', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: '*fp16', 21: '*fp16', 22: '*fp16', 23: '*fp16', 24: '*fp16', 25: '*fp16', 26: '*fp16', 27: '*fp16', 28: '*fp16', 29: '*fp16', 30: '*fp16', 31: '*fp16', 32: '*fp16', 33: '*fp16', 34: '*fp16', 35: '*fp16', 36: '*fp16', 37: '*fp16', 38: '*fp16', 39: '*fp16', 40: '*fp16', 41: '*fp16', 42: '*fp16', 43: '*fp16', 44: '*fp16', 45: '*fp16', 46: '*fp16', 47: '*fp16', 48: '*fp16', 49: '*fp16', 50: '*fp16', 51: '*fp16', 52: '*fp16', 53: '*fp16', 54: '*fp16', 55: '*fp16', 56: '*fp16', 57: '*fp16', 58: '*fp16', 59: '*fp16', 60: '*fp16', 61: '*fp16', 62: '*fp16', 63: '*fp16', 64: '*fp16', 65: '*fp16', 66: '*fp16', 67: '*fp16', 68: '*fp16', 69: '*fp16', 70: '*fp16', 71: '*fp16', 72: '*fp16', 73: '*fp16', 74: 'i32', 75: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_37', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 75), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(75,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_37(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (7)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp10 = tl.load(in_ptr0 + (8)).to(tl.float32)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp20 = tl.load(in_ptr7 + (7)).to(tl.float32)
    tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
    tmp24 = tl.load(in_ptr7 + (8)).to(tl.float32)
    tmp25 = tl.broadcast_to(tmp24, [XBLOCK, RBLOCK])
    tmp28 = tl.load(in_ptr0 + (9)).to(tl.float32)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp37 = tl.load(in_ptr0 + (10)).to(tl.float32)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
    tmp46 = tl.load(in_ptr7 + (9)).to(tl.float32)
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
    tmp50 = tl.load(in_ptr7 + (10)).to(tl.float32)
    tmp51 = tl.broadcast_to(tmp50, [XBLOCK, RBLOCK])
    tmp54 = tl.load(in_ptr0 + (11)).to(tl.float32)
    tmp55 = tl.broadcast_to(tmp54, [XBLOCK, RBLOCK])
    tmp63 = tl.load(in_ptr0 + (12)).to(tl.float32)
    tmp64 = tl.broadcast_to(tmp63, [XBLOCK, RBLOCK])
    tmp72 = tl.load(in_ptr7 + (11)).to(tl.float32)
    tmp73 = tl.broadcast_to(tmp72, [XBLOCK, RBLOCK])
    tmp76 = tl.load(in_ptr7 + (12)).to(tl.float32)
    tmp77 = tl.broadcast_to(tmp76, [XBLOCK, RBLOCK])
    tmp80 = tl.load(in_ptr0 + (13)).to(tl.float32)
    tmp81 = tl.broadcast_to(tmp80, [XBLOCK, RBLOCK])
    tmp89 = tl.load(in_ptr0 + (14)).to(tl.float32)
    tmp90 = tl.broadcast_to(tmp89, [XBLOCK, RBLOCK])
    tmp98 = tl.load(in_ptr7 + (13)).to(tl.float32)
    tmp99 = tl.broadcast_to(tmp98, [XBLOCK, RBLOCK])
    tmp102 = tl.load(in_ptr7 + (14)).to(tl.float32)
    tmp103 = tl.broadcast_to(tmp102, [XBLOCK, RBLOCK])
    tmp106 = tl.load(in_ptr0 + (15)).to(tl.float32)
    tmp107 = tl.broadcast_to(tmp106, [XBLOCK, RBLOCK])
    tmp115 = tl.load(in_ptr0 + (16)).to(tl.float32)
    tmp116 = tl.broadcast_to(tmp115, [XBLOCK, RBLOCK])
    tmp124 = tl.load(in_ptr7 + (15)).to(tl.float32)
    tmp125 = tl.broadcast_to(tmp124, [XBLOCK, RBLOCK])
    tmp128 = tl.load(in_ptr7 + (16)).to(tl.float32)
    tmp129 = tl.broadcast_to(tmp128, [XBLOCK, RBLOCK])
    tmp132 = tl.load(in_ptr0 + (17)).to(tl.float32)
    tmp133 = tl.broadcast_to(tmp132, [XBLOCK, RBLOCK])
    tmp141 = tl.load(in_ptr0 + (18)).to(tl.float32)
    tmp142 = tl.broadcast_to(tmp141, [XBLOCK, RBLOCK])
    tmp150 = tl.load(in_ptr7 + (17)).to(tl.float32)
    tmp151 = tl.broadcast_to(tmp150, [XBLOCK, RBLOCK])
    tmp154 = tl.load(in_ptr7 + (18)).to(tl.float32)
    tmp155 = tl.broadcast_to(tmp154, [XBLOCK, RBLOCK])
    tmp158 = tl.load(in_ptr0 + (19)).to(tl.float32)
    tmp159 = tl.broadcast_to(tmp158, [XBLOCK, RBLOCK])
    tmp167 = tl.load(in_ptr0 + (20)).to(tl.float32)
    tmp168 = tl.broadcast_to(tmp167, [XBLOCK, RBLOCK])
    tmp176 = tl.load(in_ptr7 + (19)).to(tl.float32)
    tmp177 = tl.broadcast_to(tmp176, [XBLOCK, RBLOCK])
    tmp180 = tl.load(in_ptr7 + (20)).to(tl.float32)
    tmp181 = tl.broadcast_to(tmp180, [XBLOCK, RBLOCK])
    tmp184 = tl.load(in_ptr0 + (21)).to(tl.float32)
    tmp185 = tl.broadcast_to(tmp184, [XBLOCK, RBLOCK])
    tmp193 = tl.load(in_ptr0 + (22)).to(tl.float32)
    tmp194 = tl.broadcast_to(tmp193, [XBLOCK, RBLOCK])
    tmp202 = tl.load(in_ptr7 + (21)).to(tl.float32)
    tmp203 = tl.broadcast_to(tmp202, [XBLOCK, RBLOCK])
    tmp206 = tl.load(in_ptr7 + (22)).to(tl.float32)
    tmp207 = tl.broadcast_to(tmp206, [XBLOCK, RBLOCK])
    tmp210 = tl.load(in_ptr0 + (23)).to(tl.float32)
    tmp211 = tl.broadcast_to(tmp210, [XBLOCK, RBLOCK])
    tmp219 = tl.load(in_ptr0 + (24)).to(tl.float32)
    tmp220 = tl.broadcast_to(tmp219, [XBLOCK, RBLOCK])
    tmp228 = tl.load(in_ptr7 + (23)).to(tl.float32)
    tmp229 = tl.broadcast_to(tmp228, [XBLOCK, RBLOCK])
    tmp232 = tl.load(in_ptr7 + (24)).to(tl.float32)
    tmp233 = tl.broadcast_to(tmp232, [XBLOCK, RBLOCK])
    tmp236 = tl.load(in_ptr0 + (25)).to(tl.float32)
    tmp237 = tl.broadcast_to(tmp236, [XBLOCK, RBLOCK])
    tmp245 = tl.load(in_ptr0 + (26)).to(tl.float32)
    tmp246 = tl.broadcast_to(tmp245, [XBLOCK, RBLOCK])
    tmp254 = tl.load(in_ptr7 + (25)).to(tl.float32)
    tmp255 = tl.broadcast_to(tmp254, [XBLOCK, RBLOCK])
    tmp258 = tl.load(in_ptr7 + (26)).to(tl.float32)
    tmp259 = tl.broadcast_to(tmp258, [XBLOCK, RBLOCK])
    tmp263 = tl.load(in_ptr62 + (27)).to(tl.float32)
    tmp264 = tl.broadcast_to(tmp263, [XBLOCK, RBLOCK])
    tmp272 = tl.load(in_ptr62 + (28)).to(tl.float32)
    tmp273 = tl.broadcast_to(tmp272, [XBLOCK, RBLOCK])
    tmp281 = tl.load(in_ptr0 + (27)).to(tl.float32)
    tmp282 = tl.broadcast_to(tmp281, [XBLOCK, RBLOCK])
    tmp285 = tl.load(in_ptr0 + (28)).to(tl.float32)
    tmp286 = tl.broadcast_to(tmp285, [XBLOCK, RBLOCK])
    tmp289 = tl.load(in_ptr7 + (27)).to(tl.float32)
    tmp290 = tl.broadcast_to(tmp289, [XBLOCK, RBLOCK])
    tmp293 = tl.load(in_ptr7 + (28)).to(tl.float32)
    tmp294 = tl.broadcast_to(tmp293, [XBLOCK, RBLOCK])
    _tmp300 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr5 + (r0), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr6 + (r0), rmask, other=0).to(tl.float32)
        tmp19 = tl.load(in_out_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp30 = tl.load(in_ptr8 + (r0), rmask, other=0).to(tl.float32)
        tmp31 = tl.load(in_ptr9 + (r0), rmask, other=0).to(tl.float32)
        tmp33 = tl.load(in_ptr10 + (r0), rmask, other=0).to(tl.float32)
        tmp39 = tl.load(in_ptr11 + (r0), rmask, other=0).to(tl.float32)
        tmp40 = tl.load(in_ptr12 + (r0), rmask, other=0).to(tl.float32)
        tmp42 = tl.load(in_ptr13 + (r0), rmask, other=0).to(tl.float32)
        tmp56 = tl.load(in_ptr14 + (r0), rmask, other=0).to(tl.float32)
        tmp57 = tl.load(in_ptr15 + (r0), rmask, other=0).to(tl.float32)
        tmp59 = tl.load(in_ptr16 + (r0), rmask, other=0).to(tl.float32)
        tmp65 = tl.load(in_ptr17 + (r0), rmask, other=0).to(tl.float32)
        tmp66 = tl.load(in_ptr18 + (r0), rmask, other=0).to(tl.float32)
        tmp68 = tl.load(in_ptr19 + (r0), rmask, other=0).to(tl.float32)
        tmp82 = tl.load(in_ptr20 + (r0), rmask, other=0).to(tl.float32)
        tmp83 = tl.load(in_ptr21 + (r0), rmask, other=0).to(tl.float32)
        tmp85 = tl.load(in_ptr22 + (r0), rmask, other=0).to(tl.float32)
        tmp91 = tl.load(in_ptr23 + (r0), rmask, other=0).to(tl.float32)
        tmp92 = tl.load(in_ptr24 + (r0), rmask, other=0).to(tl.float32)
        tmp94 = tl.load(in_ptr25 + (r0), rmask, other=0).to(tl.float32)
        tmp108 = tl.load(in_ptr26 + (r0), rmask, other=0).to(tl.float32)
        tmp109 = tl.load(in_ptr27 + (r0), rmask, other=0).to(tl.float32)
        tmp111 = tl.load(in_ptr28 + (r0), rmask, other=0).to(tl.float32)
        tmp117 = tl.load(in_ptr29 + (r0), rmask, other=0).to(tl.float32)
        tmp118 = tl.load(in_ptr30 + (r0), rmask, other=0).to(tl.float32)
        tmp120 = tl.load(in_ptr31 + (r0), rmask, other=0).to(tl.float32)
        tmp134 = tl.load(in_ptr32 + (r0), rmask, other=0).to(tl.float32)
        tmp135 = tl.load(in_ptr33 + (r0), rmask, other=0).to(tl.float32)
        tmp137 = tl.load(in_ptr34 + (r0), rmask, other=0).to(tl.float32)
        tmp143 = tl.load(in_ptr35 + (r0), rmask, other=0).to(tl.float32)
        tmp144 = tl.load(in_ptr36 + (r0), rmask, other=0).to(tl.float32)
        tmp146 = tl.load(in_ptr37 + (r0), rmask, other=0).to(tl.float32)
        tmp160 = tl.load(in_ptr38 + (r0), rmask, other=0).to(tl.float32)
        tmp161 = tl.load(in_ptr39 + (r0), rmask, other=0).to(tl.float32)
        tmp163 = tl.load(in_ptr40 + (r0), rmask, other=0).to(tl.float32)
        tmp169 = tl.load(in_ptr41 + (r0), rmask, other=0).to(tl.float32)
        tmp170 = tl.load(in_ptr42 + (r0), rmask, other=0).to(tl.float32)
        tmp172 = tl.load(in_ptr43 + (r0), rmask, other=0).to(tl.float32)
        tmp186 = tl.load(in_ptr44 + (r0), rmask, other=0).to(tl.float32)
        tmp187 = tl.load(in_ptr45 + (r0), rmask, other=0).to(tl.float32)
        tmp189 = tl.load(in_ptr46 + (r0), rmask, other=0).to(tl.float32)
        tmp195 = tl.load(in_ptr47 + (r0), rmask, other=0).to(tl.float32)
        tmp196 = tl.load(in_ptr48 + (r0), rmask, other=0).to(tl.float32)
        tmp198 = tl.load(in_ptr49 + (r0), rmask, other=0).to(tl.float32)
        tmp212 = tl.load(in_ptr50 + (r0), rmask, other=0).to(tl.float32)
        tmp213 = tl.load(in_ptr51 + (r0), rmask, other=0).to(tl.float32)
        tmp215 = tl.load(in_ptr52 + (r0), rmask, other=0).to(tl.float32)
        tmp221 = tl.load(in_ptr53 + (r0), rmask, other=0).to(tl.float32)
        tmp222 = tl.load(in_ptr54 + (r0), rmask, other=0).to(tl.float32)
        tmp224 = tl.load(in_ptr55 + (r0), rmask, other=0).to(tl.float32)
        tmp238 = tl.load(in_ptr56 + (r0), rmask, other=0).to(tl.float32)
        tmp239 = tl.load(in_ptr57 + (r0), rmask, other=0).to(tl.float32)
        tmp241 = tl.load(in_ptr58 + (r0), rmask, other=0).to(tl.float32)
        tmp247 = tl.load(in_ptr59 + (r0), rmask, other=0).to(tl.float32)
        tmp248 = tl.load(in_ptr60 + (r0), rmask, other=0).to(tl.float32)
        tmp250 = tl.load(in_ptr61 + (r0), rmask, other=0).to(tl.float32)
        tmp262 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp265 = tl.load(in_ptr63 + (r0), rmask, other=0).to(tl.float32)
        tmp266 = tl.load(in_ptr64 + (r0), rmask, other=0).to(tl.float32)
        tmp268 = tl.load(in_ptr65 + (r0), rmask, other=0).to(tl.float32)
        tmp274 = tl.load(in_ptr66 + (r0), rmask, other=0).to(tl.float32)
        tmp275 = tl.load(in_ptr67 + (r0), rmask, other=0).to(tl.float32)
        tmp277 = tl.load(in_ptr68 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp11 * tmp16
        tmp18 = tmp9 + tmp17
        tmp22 = tmp21 * tmp7
        tmp23 = tmp19 + tmp22
        tmp26 = tmp25 * tmp16
        tmp27 = tmp23 + tmp26
        tmp32 = tmp30 + tmp31
        tmp34 = tmp32 + tmp33
        tmp35 = tmp29 * tmp34
        tmp36 = tmp18 + tmp35
        tmp41 = tmp39 + tmp40
        tmp43 = tmp41 + tmp42
        tmp44 = tmp38 * tmp43
        tmp45 = tmp36 + tmp44
        tmp48 = tmp47 * tmp34
        tmp49 = tmp27 + tmp48
        tmp52 = tmp51 * tmp43
        tmp53 = tmp49 + tmp52
        tmp58 = tmp56 + tmp57
        tmp60 = tmp58 + tmp59
        tmp61 = tmp55 * tmp60
        tmp62 = tmp45 + tmp61
        tmp67 = tmp65 + tmp66
        tmp69 = tmp67 + tmp68
        tmp70 = tmp64 * tmp69
        tmp71 = tmp62 + tmp70
        tmp74 = tmp73 * tmp60
        tmp75 = tmp53 + tmp74
        tmp78 = tmp77 * tmp69
        tmp79 = tmp75 + tmp78
        tmp84 = tmp82 + tmp83
        tmp86 = tmp84 + tmp85
        tmp87 = tmp81 * tmp86
        tmp88 = tmp71 + tmp87
        tmp93 = tmp91 + tmp92
        tmp95 = tmp93 + tmp94
        tmp96 = tmp90 * tmp95
        tmp97 = tmp88 + tmp96
        tmp100 = tmp99 * tmp86
        tmp101 = tmp79 + tmp100
        tmp104 = tmp103 * tmp95
        tmp105 = tmp101 + tmp104
        tmp110 = tmp108 + tmp109
        tmp112 = tmp110 + tmp111
        tmp113 = tmp107 * tmp112
        tmp114 = tmp97 + tmp113
        tmp119 = tmp117 + tmp118
        tmp121 = tmp119 + tmp120
        tmp122 = tmp116 * tmp121
        tmp123 = tmp114 + tmp122
        tmp126 = tmp125 * tmp112
        tmp127 = tmp105 + tmp126
        tmp130 = tmp129 * tmp121
        tmp131 = tmp127 + tmp130
        tmp136 = tmp134 + tmp135
        tmp138 = tmp136 + tmp137
        tmp139 = tmp133 * tmp138
        tmp140 = tmp123 + tmp139
        tmp145 = tmp143 + tmp144
        tmp147 = tmp145 + tmp146
        tmp148 = tmp142 * tmp147
        tmp149 = tmp140 + tmp148
        tmp152 = tmp151 * tmp138
        tmp153 = tmp131 + tmp152
        tmp156 = tmp155 * tmp147
        tmp157 = tmp153 + tmp156
        tmp162 = tmp160 + tmp161
        tmp164 = tmp162 + tmp163
        tmp165 = tmp159 * tmp164
        tmp166 = tmp149 + tmp165
        tmp171 = tmp169 + tmp170
        tmp173 = tmp171 + tmp172
        tmp174 = tmp168 * tmp173
        tmp175 = tmp166 + tmp174
        tmp178 = tmp177 * tmp164
        tmp179 = tmp157 + tmp178
        tmp182 = tmp181 * tmp173
        tmp183 = tmp179 + tmp182
        tmp188 = tmp186 + tmp187
        tmp190 = tmp188 + tmp189
        tmp191 = tmp185 * tmp190
        tmp192 = tmp175 + tmp191
        tmp197 = tmp195 + tmp196
        tmp199 = tmp197 + tmp198
        tmp200 = tmp194 * tmp199
        tmp201 = tmp192 + tmp200
        tmp204 = tmp203 * tmp190
        tmp205 = tmp183 + tmp204
        tmp208 = tmp207 * tmp199
        tmp209 = tmp205 + tmp208
        tmp214 = tmp212 + tmp213
        tmp216 = tmp214 + tmp215
        tmp217 = tmp211 * tmp216
        tmp218 = tmp201 + tmp217
        tmp223 = tmp221 + tmp222
        tmp225 = tmp223 + tmp224
        tmp226 = tmp220 * tmp225
        tmp227 = tmp218 + tmp226
        tmp230 = tmp229 * tmp216
        tmp231 = tmp209 + tmp230
        tmp234 = tmp233 * tmp225
        tmp235 = tmp231 + tmp234
        tmp240 = tmp238 + tmp239
        tmp242 = tmp240 + tmp241
        tmp243 = tmp237 * tmp242
        tmp244 = tmp227 + tmp243
        tmp249 = tmp247 + tmp248
        tmp251 = tmp249 + tmp250
        tmp252 = tmp246 * tmp251
        tmp253 = tmp244 + tmp252
        tmp256 = tmp255 * tmp242
        tmp257 = tmp235 + tmp256
        tmp260 = tmp259 * tmp251
        tmp261 = tmp257 + tmp260
        tmp267 = tmp265 + tmp266
        tmp269 = tmp267 + tmp268
        tmp270 = tmp264 * tmp269
        tmp271 = tmp262 + tmp270
        tmp276 = tmp274 + tmp275
        tmp278 = tmp276 + tmp277
        tmp279 = tmp273 * tmp278
        tmp280 = tmp271 + tmp279
        tmp283 = tmp282 * tmp269
        tmp284 = tmp253 + tmp283
        tmp287 = tmp286 * tmp278
        tmp288 = tmp284 + tmp287
        tmp291 = tmp290 * tmp269
        tmp292 = tmp261 + tmp291
        tmp295 = tmp294 * tmp278
        tmp296 = tmp292 + tmp295
        tmp297 = tmp280.to(tl.float32)
        tmp298 = tmp297 * tmp297
        tmp299 = tl.broadcast_to(tmp298, [XBLOCK, RBLOCK])
        tmp301 = _tmp300 + tmp299
        _tmp300 = tl.where(rmask, tmp301, _tmp300)
        tl.store(in_out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp280, rmask)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp288, rmask)
        tl.store(in_out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp296, rmask)
    tmp300 = tl.sum(_tmp300, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp302 = tl.load(in_out_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp311 = tl.load(in_ptr69 + (r0), rmask, other=0).to(tl.float32)
        tmp303 = tmp302.to(tl.float32)
        tmp304 = 4096.0
        tmp305 = tmp300 / tmp304
        tmp306 = 1e-05
        tmp307 = tmp305 + tmp306
        tmp308 = tl.math.rsqrt(tmp307)
        tmp309 = tmp303 * tmp308
        tmp310 = tmp309.to(tl.float32)
        tmp312 = tmp310 * tmp311
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp312, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/jp/cjpmsezktacyg7vubbndac5m4rltiysgec6sjtbyqsedw6cfsumz.py
# Source Nodes: [add_577, add_579, add_644, add_645, float_121, mean_60, mul_944, mul_945, mul_946, mul_947, rsqrt_60, type_as_120], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_577 => add_577
# add_579 => add_579
# add_644 => add_644
# add_645 => add_645
# float_121 => convert_element_type_360
# mean_60 => mean_60
# mul_944 => mul_974
# mul_945 => mul_975
# mul_946 => mul_976
# mul_947 => mul_977
# rsqrt_60 => rsqrt_60
# type_as_120 => convert_element_type_361
triton_red_fused__to_copy_add_mean_mul_rsqrt_38 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_38', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_38', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_38(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (29)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
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


# kernel path: /tmp/torchinductor_mengqy/hw/chwxg5pasfzaf4jocwbnz2utbnbvmnxmuf6wbxtepa5i23wghw7o.py
# Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_153, add_155, add_172, add_174, add_192, add_194, add_213, add_215, add_235, add_237, add_258, add_260, add_282, add_284, add_307, add_309, add_333, add_335, add_360, add_362, add_388, add_390, add_417, add_419, add_447, add_449, add_478, add_480, add_510, add_512, add_543, add_545, add_577, add_579, add_60, add_612, add_614, add_62, add_680, add_681, add_682, add_695, add_696, add_697, add_698, add_699, add_700, add_701, add_702, add_703, add_704, add_705, add_706, add_707, add_708, add_709, add_710, add_711, add_712, add_713, add_714, add_715, add_716, add_717, add_718, add_73, add_75, add_87, add_89, float_125, mean_62, mul_1015, mul_1016, mul_1017, mul_1018, mul_1019, mul_1020, mul_1021, mul_1022, mul_1023, mul_1024, mul_1025, mul_1026, mul_1027, mul_1028, mul_1029, mul_1030, mul_1031, mul_1032, mul_1033, mul_1034, mul_1035, mul_1036, mul_1037, mul_1038, mul_990, mul_991, mul_992, mul_993, mul_994, rsqrt_62, type_as_124], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_102 => add_102
# add_104 => add_104
# add_118 => add_118
# add_120 => add_120
# add_135 => add_135
# add_137 => add_137
# add_153 => add_153
# add_155 => add_155
# add_172 => add_172
# add_174 => add_174
# add_192 => add_192
# add_194 => add_194
# add_213 => add_213
# add_215 => add_215
# add_235 => add_235
# add_237 => add_237
# add_258 => add_258
# add_260 => add_260
# add_282 => add_282
# add_284 => add_284
# add_307 => add_307
# add_309 => add_309
# add_333 => add_333
# add_335 => add_335
# add_360 => add_360
# add_362 => add_362
# add_388 => add_388
# add_390 => add_390
# add_417 => add_417
# add_419 => add_419
# add_447 => add_447
# add_449 => add_449
# add_478 => add_478
# add_480 => add_480
# add_510 => add_510
# add_512 => add_512
# add_543 => add_543
# add_545 => add_545
# add_577 => add_577
# add_579 => add_579
# add_60 => add_60
# add_612 => add_612
# add_614 => add_614
# add_62 => add_62
# add_680 => add_680
# add_681 => add_681
# add_682 => add_682
# add_695 => add_695
# add_696 => add_696
# add_697 => add_697
# add_698 => add_698
# add_699 => add_699
# add_700 => add_700
# add_701 => add_701
# add_702 => add_702
# add_703 => add_703
# add_704 => add_704
# add_705 => add_705
# add_706 => add_706
# add_707 => add_707
# add_708 => add_708
# add_709 => add_709
# add_710 => add_710
# add_711 => add_711
# add_712 => add_712
# add_713 => add_713
# add_714 => add_714
# add_715 => add_715
# add_716 => add_716
# add_717 => add_717
# add_718 => add_718
# add_73 => add_73
# add_75 => add_75
# add_87 => add_87
# add_89 => add_89
# float_125 => convert_element_type_372
# mean_62 => mean_62
# mul_1015 => mul_1047
# mul_1016 => mul_1048
# mul_1017 => mul_1049
# mul_1018 => mul_1050
# mul_1019 => mul_1051
# mul_1020 => mul_1052
# mul_1021 => mul_1053
# mul_1022 => mul_1054
# mul_1023 => mul_1055
# mul_1024 => mul_1056
# mul_1025 => mul_1057
# mul_1026 => mul_1058
# mul_1027 => mul_1059
# mul_1028 => mul_1060
# mul_1029 => mul_1061
# mul_1030 => mul_1062
# mul_1031 => mul_1063
# mul_1032 => mul_1064
# mul_1033 => mul_1065
# mul_1034 => mul_1066
# mul_1035 => mul_1067
# mul_1036 => mul_1068
# mul_1037 => mul_1069
# mul_1038 => mul_1070
# mul_990 => mul_1021
# mul_991 => mul_1022
# mul_992 => mul_1023
# mul_993 => mul_1024
# mul_994 => mul_1025
# rsqrt_62 => rsqrt_62
# type_as_124 => convert_element_type_373
triton_red_fused__to_copy_add_mean_mul_rsqrt_39 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_39', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: '*fp16', 21: '*fp16', 22: '*fp16', 23: '*fp16', 24: '*fp16', 25: '*fp16', 26: '*fp16', 27: '*fp16', 28: '*fp16', 29: '*fp16', 30: '*fp16', 31: '*fp16', 32: '*fp16', 33: '*fp16', 34: '*fp16', 35: '*fp16', 36: '*fp16', 37: '*fp16', 38: '*fp16', 39: '*fp16', 40: '*fp16', 41: '*fp16', 42: '*fp16', 43: '*fp16', 44: '*fp16', 45: '*fp16', 46: '*fp16', 47: '*fp16', 48: '*fp16', 49: '*fp16', 50: '*fp16', 51: '*fp16', 52: '*fp16', 53: '*fp16', 54: '*fp16', 55: '*fp16', 56: '*fp16', 57: '*fp16', 58: '*fp16', 59: '*fp16', 60: '*fp16', 61: '*fp16', 62: '*fp16', 63: '*fp16', 64: '*fp16', 65: '*fp16', 66: '*fp16', 67: '*fp16', 68: '*fp16', 69: '*fp16', 70: '*fp16', 71: '*fp16', 72: '*fp16', 73: '*fp16', 74: '*fp16', 75: '*fp16', 76: '*fp16', 77: '*fp16', 78: 'i32', 79: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_39', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 79), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(79,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_39(in_out_ptr0, in_out_ptr1, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, in_ptr30, in_ptr31, in_ptr32, in_ptr33, in_ptr34, in_ptr35, in_ptr36, in_ptr37, in_ptr38, in_ptr39, in_ptr40, in_ptr41, in_ptr42, in_ptr43, in_ptr44, in_ptr45, in_ptr46, in_ptr47, in_ptr48, in_ptr49, in_ptr50, in_ptr51, in_ptr52, in_ptr53, in_ptr54, in_ptr55, in_ptr56, in_ptr57, in_ptr58, in_ptr59, in_ptr60, in_ptr61, in_ptr62, in_ptr63, in_ptr64, in_ptr65, in_ptr66, in_ptr67, in_ptr68, in_ptr69, in_ptr70, in_ptr71, in_ptr72, in_ptr73, in_ptr74, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr0 + (7)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    tmp10 = tl.load(in_ptr0 + (8)).to(tl.float32)
    tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
    tmp19 = tl.load(in_ptr0 + (9)).to(tl.float32)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp28 = tl.load(in_ptr0 + (10)).to(tl.float32)
    tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
    tmp37 = tl.load(in_ptr0 + (11)).to(tl.float32)
    tmp38 = tl.broadcast_to(tmp37, [XBLOCK, RBLOCK])
    tmp46 = tl.load(in_ptr0 + (12)).to(tl.float32)
    tmp47 = tl.broadcast_to(tmp46, [XBLOCK, RBLOCK])
    tmp55 = tl.load(in_ptr0 + (13)).to(tl.float32)
    tmp56 = tl.broadcast_to(tmp55, [XBLOCK, RBLOCK])
    tmp64 = tl.load(in_ptr0 + (14)).to(tl.float32)
    tmp65 = tl.broadcast_to(tmp64, [XBLOCK, RBLOCK])
    tmp73 = tl.load(in_ptr0 + (15)).to(tl.float32)
    tmp74 = tl.broadcast_to(tmp73, [XBLOCK, RBLOCK])
    tmp82 = tl.load(in_ptr0 + (16)).to(tl.float32)
    tmp83 = tl.broadcast_to(tmp82, [XBLOCK, RBLOCK])
    tmp91 = tl.load(in_ptr0 + (17)).to(tl.float32)
    tmp92 = tl.broadcast_to(tmp91, [XBLOCK, RBLOCK])
    tmp100 = tl.load(in_ptr0 + (18)).to(tl.float32)
    tmp101 = tl.broadcast_to(tmp100, [XBLOCK, RBLOCK])
    tmp109 = tl.load(in_ptr0 + (19)).to(tl.float32)
    tmp110 = tl.broadcast_to(tmp109, [XBLOCK, RBLOCK])
    tmp118 = tl.load(in_ptr0 + (20)).to(tl.float32)
    tmp119 = tl.broadcast_to(tmp118, [XBLOCK, RBLOCK])
    tmp127 = tl.load(in_ptr0 + (21)).to(tl.float32)
    tmp128 = tl.broadcast_to(tmp127, [XBLOCK, RBLOCK])
    tmp136 = tl.load(in_ptr0 + (22)).to(tl.float32)
    tmp137 = tl.broadcast_to(tmp136, [XBLOCK, RBLOCK])
    tmp145 = tl.load(in_ptr0 + (23)).to(tl.float32)
    tmp146 = tl.broadcast_to(tmp145, [XBLOCK, RBLOCK])
    tmp154 = tl.load(in_ptr0 + (24)).to(tl.float32)
    tmp155 = tl.broadcast_to(tmp154, [XBLOCK, RBLOCK])
    tmp163 = tl.load(in_ptr0 + (25)).to(tl.float32)
    tmp164 = tl.broadcast_to(tmp163, [XBLOCK, RBLOCK])
    tmp172 = tl.load(in_ptr0 + (26)).to(tl.float32)
    tmp173 = tl.broadcast_to(tmp172, [XBLOCK, RBLOCK])
    tmp181 = tl.load(in_ptr0 + (27)).to(tl.float32)
    tmp182 = tl.broadcast_to(tmp181, [XBLOCK, RBLOCK])
    tmp190 = tl.load(in_ptr0 + (28)).to(tl.float32)
    tmp191 = tl.broadcast_to(tmp190, [XBLOCK, RBLOCK])
    tmp200 = tl.load(in_ptr67 + (29)).to(tl.float32)
    tmp201 = tl.broadcast_to(tmp200, [XBLOCK, RBLOCK])
    tmp209 = tl.load(in_ptr67 + (30)).to(tl.float32)
    tmp210 = tl.broadcast_to(tmp209, [XBLOCK, RBLOCK])
    tmp218 = tl.load(in_ptr0 + (29)).to(tl.float32)
    tmp219 = tl.broadcast_to(tmp218, [XBLOCK, RBLOCK])
    tmp222 = tl.load(in_ptr0 + (30)).to(tl.float32)
    tmp223 = tl.broadcast_to(tmp222, [XBLOCK, RBLOCK])
    _tmp229 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_out_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp12 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr5 + (r0), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr6 + (r0), rmask, other=0).to(tl.float32)
        tmp21 = tl.load(in_ptr7 + (r0), rmask, other=0).to(tl.float32)
        tmp22 = tl.load(in_ptr8 + (r0), rmask, other=0).to(tl.float32)
        tmp24 = tl.load(in_ptr9 + (r0), rmask, other=0).to(tl.float32)
        tmp30 = tl.load(in_ptr10 + (r0), rmask, other=0).to(tl.float32)
        tmp31 = tl.load(in_ptr11 + (r0), rmask, other=0).to(tl.float32)
        tmp33 = tl.load(in_ptr12 + (r0), rmask, other=0).to(tl.float32)
        tmp39 = tl.load(in_ptr13 + (r0), rmask, other=0).to(tl.float32)
        tmp40 = tl.load(in_ptr14 + (r0), rmask, other=0).to(tl.float32)
        tmp42 = tl.load(in_ptr15 + (r0), rmask, other=0).to(tl.float32)
        tmp48 = tl.load(in_ptr16 + (r0), rmask, other=0).to(tl.float32)
        tmp49 = tl.load(in_ptr17 + (r0), rmask, other=0).to(tl.float32)
        tmp51 = tl.load(in_ptr18 + (r0), rmask, other=0).to(tl.float32)
        tmp57 = tl.load(in_ptr19 + (r0), rmask, other=0).to(tl.float32)
        tmp58 = tl.load(in_ptr20 + (r0), rmask, other=0).to(tl.float32)
        tmp60 = tl.load(in_ptr21 + (r0), rmask, other=0).to(tl.float32)
        tmp66 = tl.load(in_ptr22 + (r0), rmask, other=0).to(tl.float32)
        tmp67 = tl.load(in_ptr23 + (r0), rmask, other=0).to(tl.float32)
        tmp69 = tl.load(in_ptr24 + (r0), rmask, other=0).to(tl.float32)
        tmp75 = tl.load(in_ptr25 + (r0), rmask, other=0).to(tl.float32)
        tmp76 = tl.load(in_ptr26 + (r0), rmask, other=0).to(tl.float32)
        tmp78 = tl.load(in_ptr27 + (r0), rmask, other=0).to(tl.float32)
        tmp84 = tl.load(in_ptr28 + (r0), rmask, other=0).to(tl.float32)
        tmp85 = tl.load(in_ptr29 + (r0), rmask, other=0).to(tl.float32)
        tmp87 = tl.load(in_ptr30 + (r0), rmask, other=0).to(tl.float32)
        tmp93 = tl.load(in_ptr31 + (r0), rmask, other=0).to(tl.float32)
        tmp94 = tl.load(in_ptr32 + (r0), rmask, other=0).to(tl.float32)
        tmp96 = tl.load(in_ptr33 + (r0), rmask, other=0).to(tl.float32)
        tmp102 = tl.load(in_ptr34 + (r0), rmask, other=0).to(tl.float32)
        tmp103 = tl.load(in_ptr35 + (r0), rmask, other=0).to(tl.float32)
        tmp105 = tl.load(in_ptr36 + (r0), rmask, other=0).to(tl.float32)
        tmp111 = tl.load(in_ptr37 + (r0), rmask, other=0).to(tl.float32)
        tmp112 = tl.load(in_ptr38 + (r0), rmask, other=0).to(tl.float32)
        tmp114 = tl.load(in_ptr39 + (r0), rmask, other=0).to(tl.float32)
        tmp120 = tl.load(in_ptr40 + (r0), rmask, other=0).to(tl.float32)
        tmp121 = tl.load(in_ptr41 + (r0), rmask, other=0).to(tl.float32)
        tmp123 = tl.load(in_ptr42 + (r0), rmask, other=0).to(tl.float32)
        tmp129 = tl.load(in_ptr43 + (r0), rmask, other=0).to(tl.float32)
        tmp130 = tl.load(in_ptr44 + (r0), rmask, other=0).to(tl.float32)
        tmp132 = tl.load(in_ptr45 + (r0), rmask, other=0).to(tl.float32)
        tmp138 = tl.load(in_ptr46 + (r0), rmask, other=0).to(tl.float32)
        tmp139 = tl.load(in_ptr47 + (r0), rmask, other=0).to(tl.float32)
        tmp141 = tl.load(in_ptr48 + (r0), rmask, other=0).to(tl.float32)
        tmp147 = tl.load(in_ptr49 + (r0), rmask, other=0).to(tl.float32)
        tmp148 = tl.load(in_ptr50 + (r0), rmask, other=0).to(tl.float32)
        tmp150 = tl.load(in_ptr51 + (r0), rmask, other=0).to(tl.float32)
        tmp156 = tl.load(in_ptr52 + (r0), rmask, other=0).to(tl.float32)
        tmp157 = tl.load(in_ptr53 + (r0), rmask, other=0).to(tl.float32)
        tmp159 = tl.load(in_ptr54 + (r0), rmask, other=0).to(tl.float32)
        tmp165 = tl.load(in_ptr55 + (r0), rmask, other=0).to(tl.float32)
        tmp166 = tl.load(in_ptr56 + (r0), rmask, other=0).to(tl.float32)
        tmp168 = tl.load(in_ptr57 + (r0), rmask, other=0).to(tl.float32)
        tmp174 = tl.load(in_ptr58 + (r0), rmask, other=0).to(tl.float32)
        tmp175 = tl.load(in_ptr59 + (r0), rmask, other=0).to(tl.float32)
        tmp177 = tl.load(in_ptr60 + (r0), rmask, other=0).to(tl.float32)
        tmp183 = tl.load(in_ptr61 + (r0), rmask, other=0).to(tl.float32)
        tmp184 = tl.load(in_ptr62 + (r0), rmask, other=0).to(tl.float32)
        tmp186 = tl.load(in_ptr63 + (r0), rmask, other=0).to(tl.float32)
        tmp192 = tl.load(in_ptr64 + (r0), rmask, other=0).to(tl.float32)
        tmp193 = tl.load(in_ptr65 + (r0), rmask, other=0).to(tl.float32)
        tmp195 = tl.load(in_ptr66 + (r0), rmask, other=0).to(tl.float32)
        tmp199 = tl.load(in_out_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp202 = tl.load(in_ptr68 + (r0), rmask, other=0).to(tl.float32)
        tmp203 = tl.load(in_ptr69 + (r0), rmask, other=0).to(tl.float32)
        tmp205 = tl.load(in_ptr70 + (r0), rmask, other=0).to(tl.float32)
        tmp211 = tl.load(in_ptr71 + (r0), rmask, other=0).to(tl.float32)
        tmp212 = tl.load(in_ptr72 + (r0), rmask, other=0).to(tl.float32)
        tmp214 = tl.load(in_ptr73 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
        tmp14 = tmp12 + tmp13
        tmp16 = tmp14 + tmp15
        tmp17 = tmp11 * tmp16
        tmp18 = tmp9 + tmp17
        tmp23 = tmp21 + tmp22
        tmp25 = tmp23 + tmp24
        tmp26 = tmp20 * tmp25
        tmp27 = tmp18 + tmp26
        tmp32 = tmp30 + tmp31
        tmp34 = tmp32 + tmp33
        tmp35 = tmp29 * tmp34
        tmp36 = tmp27 + tmp35
        tmp41 = tmp39 + tmp40
        tmp43 = tmp41 + tmp42
        tmp44 = tmp38 * tmp43
        tmp45 = tmp36 + tmp44
        tmp50 = tmp48 + tmp49
        tmp52 = tmp50 + tmp51
        tmp53 = tmp47 * tmp52
        tmp54 = tmp45 + tmp53
        tmp59 = tmp57 + tmp58
        tmp61 = tmp59 + tmp60
        tmp62 = tmp56 * tmp61
        tmp63 = tmp54 + tmp62
        tmp68 = tmp66 + tmp67
        tmp70 = tmp68 + tmp69
        tmp71 = tmp65 * tmp70
        tmp72 = tmp63 + tmp71
        tmp77 = tmp75 + tmp76
        tmp79 = tmp77 + tmp78
        tmp80 = tmp74 * tmp79
        tmp81 = tmp72 + tmp80
        tmp86 = tmp84 + tmp85
        tmp88 = tmp86 + tmp87
        tmp89 = tmp83 * tmp88
        tmp90 = tmp81 + tmp89
        tmp95 = tmp93 + tmp94
        tmp97 = tmp95 + tmp96
        tmp98 = tmp92 * tmp97
        tmp99 = tmp90 + tmp98
        tmp104 = tmp102 + tmp103
        tmp106 = tmp104 + tmp105
        tmp107 = tmp101 * tmp106
        tmp108 = tmp99 + tmp107
        tmp113 = tmp111 + tmp112
        tmp115 = tmp113 + tmp114
        tmp116 = tmp110 * tmp115
        tmp117 = tmp108 + tmp116
        tmp122 = tmp120 + tmp121
        tmp124 = tmp122 + tmp123
        tmp125 = tmp119 * tmp124
        tmp126 = tmp117 + tmp125
        tmp131 = tmp129 + tmp130
        tmp133 = tmp131 + tmp132
        tmp134 = tmp128 * tmp133
        tmp135 = tmp126 + tmp134
        tmp140 = tmp138 + tmp139
        tmp142 = tmp140 + tmp141
        tmp143 = tmp137 * tmp142
        tmp144 = tmp135 + tmp143
        tmp149 = tmp147 + tmp148
        tmp151 = tmp149 + tmp150
        tmp152 = tmp146 * tmp151
        tmp153 = tmp144 + tmp152
        tmp158 = tmp156 + tmp157
        tmp160 = tmp158 + tmp159
        tmp161 = tmp155 * tmp160
        tmp162 = tmp153 + tmp161
        tmp167 = tmp165 + tmp166
        tmp169 = tmp167 + tmp168
        tmp170 = tmp164 * tmp169
        tmp171 = tmp162 + tmp170
        tmp176 = tmp174 + tmp175
        tmp178 = tmp176 + tmp177
        tmp179 = tmp173 * tmp178
        tmp180 = tmp171 + tmp179
        tmp185 = tmp183 + tmp184
        tmp187 = tmp185 + tmp186
        tmp188 = tmp182 * tmp187
        tmp189 = tmp180 + tmp188
        tmp194 = tmp192 + tmp193
        tmp196 = tmp194 + tmp195
        tmp197 = tmp191 * tmp196
        tmp198 = tmp189 + tmp197
        tmp204 = tmp202 + tmp203
        tmp206 = tmp204 + tmp205
        tmp207 = tmp201 * tmp206
        tmp208 = tmp199 + tmp207
        tmp213 = tmp211 + tmp212
        tmp215 = tmp213 + tmp214
        tmp216 = tmp210 * tmp215
        tmp217 = tmp208 + tmp216
        tmp220 = tmp219 * tmp206
        tmp221 = tmp198 + tmp220
        tmp224 = tmp223 * tmp215
        tmp225 = tmp221 + tmp224
        tmp226 = tmp217.to(tl.float32)
        tmp227 = tmp226 * tmp226
        tmp228 = tl.broadcast_to(tmp227, [XBLOCK, RBLOCK])
        tmp230 = _tmp229 + tmp228
        _tmp229 = tl.where(rmask, tmp230, _tmp229)
        tl.store(in_out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp217, rmask)
        tl.store(in_out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp225, rmask)
    tmp229 = tl.sum(_tmp229, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp231 = tl.load(in_out_ptr1 + (r0), rmask, other=0).to(tl.float32)
        tmp240 = tl.load(in_ptr74 + (r0), rmask, other=0).to(tl.float32)
        tmp232 = tmp231.to(tl.float32)
        tmp233 = 4096.0
        tmp234 = tmp229 / tmp233
        tmp235 = 1e-05
        tmp236 = tmp234 + tmp235
        tmp237 = tl.math.rsqrt(tmp236)
        tmp238 = tmp232 * tmp237
        tmp239 = tmp238.to(tl.float32)
        tmp241 = tmp239 * tmp240
        tl.store(out_ptr1 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp241, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/64/c64pgjpzn3a35sqgh6qjbypukilrch6mlpawnosesbxd5nr42d2g.py
# Source Nodes: [stack_60, stack_61, stack_63], Original ATen: [aten.stack]
# stack_60 => cat_60
# stack_61 => cat_61
# stack_63 => cat_63
triton_poi_fused_stack_40 = async_compile.triton('triton_poi_fused_stack_40', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[2048], filename=__file__, meta={'signature': {0: '*fp16', 1: '*i32', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_40', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 8, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10,))]})
@triton.jit
def triton_poi_fused_stack_40(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, xnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x0 = xindex % 64
    tmp0 = tl.load(in_ptr0 + (4096 + (2*x2)), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + (0))
    tmp3 = tl.broadcast_to(tmp2, [XBLOCK])
    tmp8 = tl.load(in_ptr0 + (4097 + (2*x2)), None).to(tl.float32)
    tmp17 = tl.load(in_ptr3 + (2*x2), None).to(tl.float32)
    tmp20 = tl.load(in_ptr3 + (1 + (2*x2)), None).to(tl.float32)
    tmp27 = tl.load(in_ptr3 + (4096 + (2*x2)), None).to(tl.float32)
    tmp30 = tl.load(in_ptr3 + (4097 + (2*x2)), None).to(tl.float32)
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
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28 * tmp6
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp31 * tmp11
    tmp33 = tmp29 - tmp32
    tmp34 = tmp31 * tmp6
    tmp35 = tmp28 * tmp11
    tmp36 = tmp34 + tmp35
    tl.store(out_ptr0 + (2*x2), tmp13, None)
    tl.store(out_ptr1 + (2*x2), tmp16, None)
    tl.store(out_ptr2 + (2*x2), tmp23, None)
    tl.store(out_ptr3 + (2*x2), tmp26, None)
    tl.store(out_ptr4 + (2*x2), tmp33, None)
    tl.store(out_ptr5 + (2*x2), tmp36, None)
''')


# kernel path: /tmp/torchinductor_mengqy/2c/c2ciho2qudbkiddp4ee7imdtnw7q5fvexgdhvm55o4cwttbde2rj.py
# Source Nodes: [getitem, mul_956, softmax_30, where_30], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
# getitem => index
# mul_956 => mul_986
# softmax_30 => amax_30, convert_element_type_366, convert_element_type_367, div_30, exp_30, sub_92, sum_31
# where_30 => full_default_30, where_30
triton_red_fused__softmax_index_mul_scalar_tensor_where_41 = async_compile.triton('triton_red_fused__softmax_index_mul_scalar_tensor_where_41', '''
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
    meta={'signature': {0: '*i32', 1: '*i1', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_index_mul_scalar_tensor_where_41', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_red_fused__softmax_index_mul_scalar_tensor_where_41(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_mengqy/5d/c5dnx4j2dt2x354xjb23jpod44s5th2qzdsopo2r2q4l2en3pvgr.py
# Source Nodes: [add_648, add_650, add_719, add_720, float_129, mean_64, mul_1039, mul_1040, mul_1041, mul_1042, rsqrt_64, type_as_128], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_648 => add_648
# add_650 => add_650
# add_719 => add_719
# add_720 => add_720
# float_129 => convert_element_type_384
# mean_64 => mean_64
# mul_1039 => mul_1071
# mul_1040 => mul_1072
# mul_1041 => mul_1073
# mul_1042 => mul_1074
# rsqrt_64 => rsqrt_64
# type_as_128 => convert_element_type_385
triton_red_fused__to_copy_add_mean_mul_rsqrt_42 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_42', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp32', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_42', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9,))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 1
    rnumel = 4096
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    tmp1 = tl.load(in_ptr1 + (31)).to(tl.float32)
    tmp2 = tl.broadcast_to(tmp1, [XBLOCK, RBLOCK])
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp0 = tl.load(in_ptr0 + (r0), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r0), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (r0), rmask, other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr4 + (r0), rmask, other=0).to(tl.float32)
        tmp5 = tmp3 + tmp4
        tmp7 = tmp5 + tmp6
        tmp8 = tmp2 * tmp7
        tmp9 = tmp0 + tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp10 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
        tl.store(out_ptr0 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp10, rmask)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r0 = rindex
        tmp15 = tl.load(out_ptr0 + (r0), rmask, other=0)
        tmp23 = tl.load(in_ptr5 + (r0), rmask, other=0).to(tl.float32)
        tmp16 = 4096.0
        tmp17 = tmp13 / tmp16
        tmp18 = 1e-05
        tmp19 = tmp17 + tmp18
        tmp20 = tl.math.rsqrt(tmp19)
        tmp21 = tmp15 * tmp20
        tmp22 = tmp21.to(tl.float32)
        tmp24 = tmp22 * tmp23
        tl.store(out_ptr2 + (tl.broadcast_to(r0, [XBLOCK, RBLOCK])), tmp24, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/mt/cmtcodlqc2vbpzf4honpbas3zhnrtqbqsiceppcqfjw2qe37kb6a.py
# Source Nodes: [truediv], Original ATen: [aten.div]
# truediv => div_32
triton_poi_fused_div_43 = async_compile.triton('triton_poi_fused_div_43', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[32768], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_div_43', 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]})
@triton.jit
def triton_poi_fused_div_43(in_out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_mengqy/zn/czn4cp2ebmutzrjkzsvturqloin5cdpfdura4sck3u46j5kigf56.py
# Source Nodes: [lt, softmax_32, where_32], Original ATen: [aten._softmax, aten.lt, aten.scalar_tensor, aten.where]
# lt => lt
# softmax_32 => amax_32, convert_element_type_386
# where_32 => full_default_31, where_32
triton_red_fused__softmax_lt_scalar_tensor_where_44 = async_compile.triton('triton_red_fused__softmax_lt_scalar_tensor_where_44', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: 'i32', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_lt_scalar_tensor_where_44', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]}
)
@triton.jit
def triton_red_fused__softmax_lt_scalar_tensor_where_44(in_ptr0, in_ptr1, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_mengqy/le/clexn72i2qvy4kq2mag4ls3vp2bpd2r3xmzbtxpxyxz2qkf3kslq.py
# Source Nodes: [lt, softmax_32, where_32], Original ATen: [aten._softmax, aten.lt, aten.scalar_tensor, aten.where]
# lt => lt
# softmax_32 => amax_32, convert_element_type_386
# where_32 => full_default_31, where_32
triton_per_fused__softmax_lt_scalar_tensor_where_45 = async_compile.triton('triton_per_fused__softmax_lt_scalar_tensor_where_45', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_lt_scalar_tensor_where_45', 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]}
)
@triton.jit
def triton_per_fused__softmax_lt_scalar_tensor_where_45(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_mengqy/i6/ci6u5vbkcyn7cst5imfnek7pphjkoetpv7kancu2zou5fn56hbt3.py
# Source Nodes: [lt, softmax_32, where_32], Original ATen: [aten._softmax, aten.lt, aten.scalar_tensor, aten.where]
# lt => lt
# softmax_32 => convert_element_type_386, exp_32, sub_96, sum_33
# where_32 => full_default_31, where_32
triton_red_fused__softmax_lt_scalar_tensor_where_46 = async_compile.triton('triton_red_fused__softmax_lt_scalar_tensor_where_46', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_lt_scalar_tensor_where_46', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]}
)
@triton.jit
def triton_red_fused__softmax_lt_scalar_tensor_where_46(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_mengqy/fl/cflewazx7yxug6gyalr564ukc3fsx3y5wvowyxs3wmmlyhi6gdac.py
# Source Nodes: [lt, softmax_32, where_32], Original ATen: [aten._softmax, aten.lt, aten.scalar_tensor, aten.where]
# lt => lt
# softmax_32 => convert_element_type_386, exp_32, sub_96, sum_33
# where_32 => full_default_31, where_32
triton_per_fused__softmax_lt_scalar_tensor_where_47 = async_compile.triton('triton_per_fused__softmax_lt_scalar_tensor_where_47', '''
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
    meta={'signature': {0: '*fp32', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__softmax_lt_scalar_tensor_where_47', 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=())]}
)
@triton.jit
def triton_per_fused__softmax_lt_scalar_tensor_where_47(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_mengqy/me/cmejydngql234dtbfj6voq37g5ae7rkvcohuo3xqzlz672pkargg.py
# Source Nodes: [argmax, exponential_, lt, softmax_32, to, truediv_1, where_32], Original ATen: [aten._softmax, aten._to_copy, aten.argmax, aten.div, aten.exponential, aten.lt, aten.scalar_tensor, aten.where]
# argmax => argmax
# exponential_ => convert_element_type_389, log1p, mul_1075, neg
# lt => lt
# softmax_32 => convert_element_type_386, convert_element_type_387, div_33, exp_32, sub_96
# to => convert_element_type_390
# truediv_1 => div_34
# where_32 => full_default_31, where_32
triton_red_fused__softmax__to_copy_argmax_div_exponential_lt_scalar_tensor_where_48 = async_compile.triton('triton_red_fused__softmax__to_copy_argmax_div_exponential_lt_scalar_tensor_where_48', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*i64', 5: '*i32', 6: 'i32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__to_copy_argmax_div_exponential_lt_scalar_tensor_where_48', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]}
)
@triton.jit
def triton_red_fused__softmax__to_copy_argmax_div_exponential_lt_scalar_tensor_where_48(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr2, load_seed_offset, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_mengqy/v4/cv43porpej3k2so7zegfvj4qtdl6rd6bmwuwt7vl4zamy3w233xp.py
# Source Nodes: [setitem_62], Original ATen: [aten.slice_scatter]
# setitem_62 => slice_scatter_124, slice_scatter_125
triton_poi_fused_slice_scatter_49 = async_compile.triton('triton_poi_fused_slice_scatter_49', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_slice_scatter_49', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_slice_scatter_49(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 4718592
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
    arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1 = args
    args.clear()
    assert_size_stride(arg0_1, (4096, ), (1, ))
    assert_size_stride(arg1_1, (4096, ), (1, ))
    assert_size_stride(arg2_1, (2, ), (1, ))
    assert_size_stride(arg3_1, (4096, ), (1, ))
    assert_size_stride(arg4_1, (4096, ), (1, ))
    assert_size_stride(arg5_1, (3, ), (1, ))
    assert_size_stride(arg6_1, (4096, ), (1, ))
    assert_size_stride(arg7_1, (4096, ), (1, ))
    assert_size_stride(arg8_1, (4, ), (1, ))
    assert_size_stride(arg9_1, (4096, ), (1, ))
    assert_size_stride(arg10_1, (4096, ), (1, ))
    assert_size_stride(arg11_1, (5, ), (1, ))
    assert_size_stride(arg12_1, (4096, ), (1, ))
    assert_size_stride(arg13_1, (4096, ), (1, ))
    assert_size_stride(arg14_1, (6, ), (1, ))
    assert_size_stride(arg15_1, (4096, ), (1, ))
    assert_size_stride(arg16_1, (4096, ), (1, ))
    assert_size_stride(arg17_1, (7, ), (1, ))
    assert_size_stride(arg18_1, (4096, ), (1, ))
    assert_size_stride(arg19_1, (4096, ), (1, ))
    assert_size_stride(arg20_1, (8, ), (1, ))
    assert_size_stride(arg21_1, (4096, ), (1, ))
    assert_size_stride(arg22_1, (4096, ), (1, ))
    assert_size_stride(arg23_1, (9, ), (1, ))
    assert_size_stride(arg24_1, (4096, ), (1, ))
    assert_size_stride(arg25_1, (4096, ), (1, ))
    assert_size_stride(arg26_1, (10, ), (1, ))
    assert_size_stride(arg27_1, (4096, ), (1, ))
    assert_size_stride(arg28_1, (4096, ), (1, ))
    assert_size_stride(arg29_1, (11, ), (1, ))
    assert_size_stride(arg30_1, (4096, ), (1, ))
    assert_size_stride(arg31_1, (4096, ), (1, ))
    assert_size_stride(arg32_1, (12, ), (1, ))
    assert_size_stride(arg33_1, (4096, ), (1, ))
    assert_size_stride(arg34_1, (4096, ), (1, ))
    assert_size_stride(arg35_1, (13, ), (1, ))
    assert_size_stride(arg36_1, (4096, ), (1, ))
    assert_size_stride(arg37_1, (4096, ), (1, ))
    assert_size_stride(arg38_1, (14, ), (1, ))
    assert_size_stride(arg39_1, (4096, ), (1, ))
    assert_size_stride(arg40_1, (4096, ), (1, ))
    assert_size_stride(arg41_1, (15, ), (1, ))
    assert_size_stride(arg42_1, (4096, ), (1, ))
    assert_size_stride(arg43_1, (4096, ), (1, ))
    assert_size_stride(arg44_1, (16, ), (1, ))
    assert_size_stride(arg45_1, (4096, ), (1, ))
    assert_size_stride(arg46_1, (4096, ), (1, ))
    assert_size_stride(arg47_1, (17, ), (1, ))
    assert_size_stride(arg48_1, (4096, ), (1, ))
    assert_size_stride(arg49_1, (4096, ), (1, ))
    assert_size_stride(arg50_1, (18, ), (1, ))
    assert_size_stride(arg51_1, (4096, ), (1, ))
    assert_size_stride(arg52_1, (4096, ), (1, ))
    assert_size_stride(arg53_1, (19, ), (1, ))
    assert_size_stride(arg54_1, (4096, ), (1, ))
    assert_size_stride(arg55_1, (4096, ), (1, ))
    assert_size_stride(arg56_1, (20, ), (1, ))
    assert_size_stride(arg57_1, (4096, ), (1, ))
    assert_size_stride(arg58_1, (4096, ), (1, ))
    assert_size_stride(arg59_1, (21, ), (1, ))
    assert_size_stride(arg60_1, (4096, ), (1, ))
    assert_size_stride(arg61_1, (4096, ), (1, ))
    assert_size_stride(arg62_1, (22, ), (1, ))
    assert_size_stride(arg63_1, (4096, ), (1, ))
    assert_size_stride(arg64_1, (4096, ), (1, ))
    assert_size_stride(arg65_1, (23, ), (1, ))
    assert_size_stride(arg66_1, (4096, ), (1, ))
    assert_size_stride(arg67_1, (4096, ), (1, ))
    assert_size_stride(arg68_1, (24, ), (1, ))
    assert_size_stride(arg69_1, (4096, ), (1, ))
    assert_size_stride(arg70_1, (4096, ), (1, ))
    assert_size_stride(arg71_1, (25, ), (1, ))
    assert_size_stride(arg72_1, (4096, ), (1, ))
    assert_size_stride(arg73_1, (4096, ), (1, ))
    assert_size_stride(arg74_1, (26, ), (1, ))
    assert_size_stride(arg75_1, (4096, ), (1, ))
    assert_size_stride(arg76_1, (4096, ), (1, ))
    assert_size_stride(arg77_1, (27, ), (1, ))
    assert_size_stride(arg78_1, (4096, ), (1, ))
    assert_size_stride(arg79_1, (4096, ), (1, ))
    assert_size_stride(arg80_1, (28, ), (1, ))
    assert_size_stride(arg81_1, (4096, ), (1, ))
    assert_size_stride(arg82_1, (4096, ), (1, ))
    assert_size_stride(arg83_1, (29, ), (1, ))
    assert_size_stride(arg84_1, (4096, ), (1, ))
    assert_size_stride(arg85_1, (4096, ), (1, ))
    assert_size_stride(arg86_1, (30, ), (1, ))
    assert_size_stride(arg87_1, (4096, ), (1, ))
    assert_size_stride(arg88_1, (4096, ), (1, ))
    assert_size_stride(arg89_1, (31, ), (1, ))
    assert_size_stride(arg90_1, (4096, ), (1, ))
    assert_size_stride(arg91_1, (4096, ), (1, ))
    assert_size_stride(arg92_1, (32, ), (1, ))
    assert_size_stride(arg93_1, (4096, ), (1, ))
    assert_size_stride(arg94_1, (4096, ), (1, ))
    assert_size_stride(arg95_1, (33, ), (1, ))
    assert_size_stride(arg96_1, (4096, ), (1, ))
    assert_size_stride(arg97_1, (32000, 4096), (4096, 1))
    assert_size_stride(arg98_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg99_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg100_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg101_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg102_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg103_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg104_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg105_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg106_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg107_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg108_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg109_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg110_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg111_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg112_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg113_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg114_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg115_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg116_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg117_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg118_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg119_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg120_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg121_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg122_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg123_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg124_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg125_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg126_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg127_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg128_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg129_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg130_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg131_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg132_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg133_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg134_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg135_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg136_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg137_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg138_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg139_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg140_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg141_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg142_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg143_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg144_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg145_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg146_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg147_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg148_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg149_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg150_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg151_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg152_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg153_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg154_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg155_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg156_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg157_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg158_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg159_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg160_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg161_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg162_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg163_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg164_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg165_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg166_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg167_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg168_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg169_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg170_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg171_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg172_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg173_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg174_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg175_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg176_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg177_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg178_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg179_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg180_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg181_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg182_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg183_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg184_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg185_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg186_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg187_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg188_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg189_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg190_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg191_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg192_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg193_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg194_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg195_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg196_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg197_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg198_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg199_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg200_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg201_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg202_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg203_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg204_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg205_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg206_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg207_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg208_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg209_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg210_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg211_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg212_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg213_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg214_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg215_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg216_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg217_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg218_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg219_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg220_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg221_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg222_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg223_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg224_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg225_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg226_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg227_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg228_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg229_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg230_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg231_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg232_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg233_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg234_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg235_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg236_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg237_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg238_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg239_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg240_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg241_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg242_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg243_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg244_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg245_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg246_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg247_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg248_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg249_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg250_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg251_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg252_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg253_1, (12288, 4096), (4096, 1))
    assert_size_stride(arg254_1, (4096, 4096), (4096, 1))
    assert_size_stride(arg255_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg256_1, (11008, 4096), (4096, 1))
    assert_size_stride(arg257_1, (4096, 11008), (11008, 1))
    assert_size_stride(arg258_1, (32000, 4096), (4096, 1))
    assert_size_stride(arg259_1, (2048, 64, 2), (128, 2, 1))
    assert_size_stride(arg260_1, (1152, 1152), (1152, 1))
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
    assert_size_stride(arg293_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg294_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg295_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg296_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg297_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg298_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg299_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg300_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg301_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg302_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg303_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg304_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg305_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg306_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg307_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg308_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg309_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg310_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg311_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg312_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg313_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg314_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg315_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg316_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg317_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg318_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg319_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg320_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg321_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg322_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg323_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg324_1, (1, 32, 1152, 128), (4718592, 147456, 128, 1))
    assert_size_stride(arg325_1, (1, 1), (1, 1))
    assert_size_stride(arg326_1, (1, ), (1, ))
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf1 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf4 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_6, add_7, float_1, float_5, l__model___tok_embeddings, mean, mean_2, mul, mul_1, mul_16, mul_17, mul_18, mul_19, mul_2, rsqrt, rsqrt_2, type_as, type_as_4], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.rsqrt]
        stream0 = get_cuda_stream(0)
        triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_0.run(arg325_1, arg97_1, arg2_1, arg0_1, arg3_1, buf1, buf4, 1, 4096, grid=grid(1), stream=stream0)
        del arg0_1
        del arg3_1
        buf2 = empty_strided((1, 12288), (12288, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_0_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1, (1, 4096), (0, 1), 0), reinterpret_tensor(arg98_1, (4096, 12288), (1, 4096), 0), out=buf2)
        del arg98_1
        buf5 = empty_strided((1, 12288), (12288, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_1_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf4, (1, 4096), (0, 1), 0), reinterpret_tensor(arg103_1, (4096, 12288), (1, 4096), 0), out=buf5)
        del arg103_1
        buf8 = empty_strided((1, 1, 32, 64, 2), (4096, 4096, 128, 2, 1), device='cuda', dtype=torch.float32)
        buf6 = reinterpret_tensor(buf8, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf7 = reinterpret_tensor(buf8, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf11 = empty_strided((1, 1, 32, 64, 2), (4096, 4096, 128, 2, 1), device='cuda', dtype=torch.float32)
        buf9 = reinterpret_tensor(buf11, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf10 = reinterpret_tensor(buf11, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf37 = empty_strided((1, 1, 32, 64, 2), (4096, 4096, 128, 2, 1), device='cuda', dtype=torch.float32)
        buf35 = reinterpret_tensor(buf37, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf36 = reinterpret_tensor(buf37, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf40 = empty_strided((1, 1, 32, 64, 2), (4096, 4096, 128, 2, 1), device='cuda', dtype=torch.float32)
        buf38 = reinterpret_tensor(buf40, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf39 = reinterpret_tensor(buf40, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1, stack_2, stack_3], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf2, arg326_1, arg259_1, buf5, buf6, buf7, buf9, buf10, buf35, buf36, buf38, buf39, 2048, grid=grid(2048), stream=stream0)
        buf12 = empty_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [setitem], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg261_1, buf12, 4718592, grid=grid(4718592), stream=stream0)
        del buf10
        del buf6
        del buf7
        del buf9
        buf19 = empty_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [setitem_1], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg262_1, buf19, 4718592, grid=grid(4718592), stream=stream0)
        buf41 = empty_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [setitem_2], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg263_1, buf41, 4718592, grid=grid(4718592), stream=stream0)
        del buf38
        del buf39
        buf48 = empty_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [setitem_3], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg264_1, buf48, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem, setitem_1, setitem_2, setitem_3], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg326_1, buf11, buf2, buf40, buf5, buf12, buf19, buf41, buf48, 4096, grid=grid(4096), stream=stream0)
        buf14 = reinterpret_tensor(buf4, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf4  # reuse
        # Source Nodes: [type_as_1], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf8, buf14, 4096, grid=grid(4096), stream=stream0)
        buf15 = empty_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [setitem], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf12, buf15, arg261_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg261_1
        buf16 = empty_strided((32, 1, 1152), (1152, 1152, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf14, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf15, (32, 128, 1152), (147456, 1, 128), 0), out=buf16)
        buf43 = buf14; del buf14  # reuse
        # Source Nodes: [type_as_5], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf37, buf43, 4096, grid=grid(4096), stream=stream0)
        del buf35
        del buf36
        buf44 = buf15; del buf15  # reuse
        # Source Nodes: [setitem_2], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf41, buf44, arg263_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg263_1
        buf45 = empty_strided((32, 1, 1152), (1152, 1152, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [matmul_2], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf43, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf44, (32, 128, 1152), (147456, 1, 128), 0), out=buf45)
        buf21 = empty_strided((1, 32, 1, 1152), (36864, 1152, 1152, 1), device='cuda', dtype=torch.float16)
        buf50 = empty_strided((1, 32, 1, 1152), (36864, 1152, 1152, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [getitem, mul_11, mul_28, softmax, softmax_1, where, where_1], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg326_1, arg260_1, buf16, buf45, buf21, buf50, 32, 1152, grid=grid(32), stream=stream0)
        buf22 = buf44; del buf44  # reuse
        # Source Nodes: [setitem_1], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf19, buf22, arg262_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg262_1
        buf23 = reinterpret_tensor(buf43, (32, 1, 128), (128, 128, 1)); del buf43  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf21, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf22, (32, 1152, 128), (147456, 128, 1), 0), out=buf23)
        buf24 = reinterpret_tensor(buf1, (1, 4096), (4096, 1)); del buf1  # reuse
        # Source Nodes: [l__model___layers_0_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf23, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg99_1, (4096, 4096), (1, 4096), 0), out=buf24)
        del arg99_1
        buf51 = buf22; del buf22  # reuse
        # Source Nodes: [setitem_3], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf48, buf51, arg264_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg264_1
        buf52 = buf23; del buf23  # reuse
        # Source Nodes: [matmul_3], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf50, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf51, (32, 1152, 128), (147456, 128, 1), 0), out=buf52)
        buf53 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_1_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf52, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg104_1, (4096, 4096), (1, 4096), 0), out=buf53)
        del arg104_1
        buf54 = empty_strided((1, 1, 1), (1, 1, 1), device='cuda', dtype=torch.float32)
        buf26 = reinterpret_tensor(buf52, (1, 1, 4096), (4096, 4096, 1)); del buf52  # reuse
        # Source Nodes: [add_10, add_3, add_4, add_6, float_4, float_8, l__model___tok_embeddings, mean_1, mean_3, mul_12, mul_13, mul_14, mul_16, mul_29, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_7.run(arg325_1, arg97_1, buf24, arg2_1, buf53, arg1_1, buf54, buf26, 1, 4096, grid=grid(1), stream=stream0)
        del arg1_1
        buf27 = empty_strided((1, 11008), (11008, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_0_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg100_1, (4096, 11008), (1, 4096), 0), out=buf27)
        del arg100_1
        buf28 = empty_strided((1, 11008), (11008, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_0_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf26, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg101_1, (4096, 11008), (1, 4096), 0), out=buf28)
        del arg101_1
        buf29 = reinterpret_tensor(buf27, (1, 1, 11008), (11008, 11008, 1)); del buf27  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf29, buf28, 11008, grid=grid(11008), stream=stream0)
        buf30 = reinterpret_tensor(buf26, (1, 4096), (4096, 1)); del buf26  # reuse
        # Source Nodes: [l__model___layers_0_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf29, (1, 11008), (0, 1), 0), reinterpret_tensor(arg102_1, (11008, 4096), (1, 11008), 0), out=buf30)
        del arg102_1
        buf31 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf55 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf33 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_10, add_11, add_13, add_14, add_15, add_3, add_5, add_6, float_8, float_9, l__model___tok_embeddings, mean_3, mean_4, mul_16, mul_29, mul_30, mul_31, mul_33, mul_34, mul_35, mul_36, mul_37, rsqrt_3, rsqrt_4, type_as_7, type_as_8], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_9.run(arg5_1, arg325_1, arg97_1, buf24, buf30, arg2_1, buf53, buf54, arg4_1, arg6_1, buf31, buf55, buf33, 1, 4096, grid=grid(1), stream=stream0)
        del arg4_1
        del arg5_1
        del arg6_1
        buf34 = buf5; del buf5  # reuse
        # Source Nodes: [l__model___layers_2_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf33, (1, 4096), (0, 1), 0), reinterpret_tensor(arg108_1, (4096, 12288), (1, 4096), 0), out=buf34)
        del arg108_1
        buf56 = reinterpret_tensor(buf29, (1, 11008), (11008, 1)); del buf29  # reuse
        # Source Nodes: [l__model___layers_1_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf55, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg105_1, (4096, 11008), (1, 4096), 0), out=buf56)
        del arg105_1
        buf57 = buf28; del buf28  # reuse
        # Source Nodes: [l__model___layers_1_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf55, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg106_1, (4096, 11008), (1, 4096), 0), out=buf57)
        del arg106_1
        buf58 = reinterpret_tensor(buf56, (1, 1, 11008), (11008, 11008, 1)); del buf56  # reuse
        # Source Nodes: [mul_32, silu_1], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf58, buf57, 11008, grid=grid(11008), stream=stream0)
        buf59 = reinterpret_tensor(buf55, (1, 4096), (4096, 1)); del buf55  # reuse
        # Source Nodes: [l__model___layers_1_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf58, (1, 11008), (0, 1), 0), reinterpret_tensor(arg107_1, (11008, 4096), (1, 11008), 0), out=buf59)
        del arg107_1
        buf60 = reinterpret_tensor(buf53, (1, 1, 4096), (4096, 4096, 1)); del buf53  # reuse
        buf61 = buf33; del buf33  # reuse
        buf65 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf95 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf125 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf156 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf187 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf219 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf251 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf284 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf317 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf351 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf385 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf420 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf455 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf491 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf527 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf564 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf601 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf639 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf677 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf716 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf755 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf795 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf835 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf876 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf917 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf959 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf1001 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf1044 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf1094 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        buf63 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_10, add_105, add_106, add_107, add_12, add_121, add_122, add_123, add_138, add_139, add_140, add_156, add_157, add_158, add_175, add_176, add_177, add_195, add_196, add_197, add_21, add_216, add_217, add_218, add_22, add_23, add_238, add_239, add_24, add_240, add_261, add_262, add_263, add_285, add_286, add_287, add_3, add_30, add_31, add_310, add_311, add_312, add_32, add_336, add_337, add_338, add_363, add_364, add_365, add_391, add_392, add_393, add_40, add_41, add_42, add_420, add_421, add_422, add_450, add_451, add_452, add_481, add_482, add_483, add_5, add_51, add_513, add_514, add_515, add_52, add_53, add_546, add_547, add_548, add_580, add_581, add_582, add_6, add_615, add_616, add_617, add_63, add_64, add_65, add_651, add_652, add_653, add_688, add_689, add_690, add_76, add_77, add_78, add_90, add_91, add_92, float_13, l__model___tok_embeddings, mean_6, mul_1008, mul_1009, mul_1010, mul_111, mul_112, mul_113, mul_133, mul_134, mul_135, mul_156, mul_157, mul_158, mul_16, mul_180, mul_181, mul_182, mul_205, mul_206, mul_207, mul_231, mul_232, mul_233, mul_258, mul_259, mul_260, mul_286, mul_287, mul_288, mul_315, mul_316, mul_317, mul_345, mul_346, mul_347, mul_376, mul_377, mul_378, mul_408, mul_409, mul_410, mul_441, mul_442, mul_443, mul_475, mul_476, mul_477, mul_51, mul_510, mul_511, mul_512, mul_52, mul_53, mul_54, mul_546, mul_547, mul_548, mul_55, mul_56, mul_583, mul_584, mul_585, mul_621, mul_622, mul_623, mul_660, mul_661, mul_662, mul_70, mul_700, mul_701, mul_702, mul_71, mul_72, mul_741, mul_742, mul_743, mul_783, mul_784, mul_785, mul_826, mul_827, mul_828, mul_870, mul_871, mul_872, mul_90, mul_91, mul_915, mul_916, mul_917, mul_92, mul_961, mul_962, mul_963, rsqrt_6, type_as_12], Original ATen: [aten._to_copy, aten.add, aten.embedding, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_embedding_mean_mul_rsqrt_10.run(buf60, arg2_1, arg325_1, arg97_1, buf59, arg8_1, buf24, buf30, arg11_1, arg14_1, arg17_1, arg20_1, arg23_1, arg26_1, arg29_1, arg32_1, arg35_1, arg38_1, arg41_1, arg44_1, arg47_1, arg50_1, arg53_1, arg56_1, arg59_1, arg62_1, arg65_1, arg68_1, arg71_1, arg74_1, arg77_1, arg80_1, arg83_1, arg86_1, arg89_1, arg92_1, arg95_1, arg9_1, buf61, buf65, buf95, buf125, buf156, buf187, buf219, buf251, buf284, buf317, buf351, buf385, buf420, buf455, buf491, buf527, buf564, buf601, buf639, buf677, buf716, buf755, buf795, buf835, buf876, buf917, buf959, buf1001, buf1044, buf1094, buf63, 1, 4096, grid=grid(1), stream=stream0)
        del arg2_1
        del arg325_1
        del arg8_1
        del arg97_1
        del arg9_1
        buf64 = buf2; del buf2  # reuse
        # Source Nodes: [l__model___layers_3_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf63, (1, 4096), (0, 1), 0), reinterpret_tensor(arg113_1, (4096, 12288), (1, 4096), 0), out=buf64)
        del arg113_1
        buf68 = buf37; del buf37  # reuse
        buf66 = reinterpret_tensor(buf68, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf67 = reinterpret_tensor(buf68, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf71 = buf8; del buf8  # reuse
        buf69 = reinterpret_tensor(buf71, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf70 = reinterpret_tensor(buf71, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf98 = buf40; del buf40  # reuse
        buf96 = reinterpret_tensor(buf98, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf97 = reinterpret_tensor(buf98, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf101 = buf11; del buf11  # reuse
        buf99 = reinterpret_tensor(buf101, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf100 = reinterpret_tensor(buf101, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_4, stack_5, stack_6, stack_7], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf34, arg326_1, arg259_1, buf64, buf66, buf67, buf69, buf70, buf96, buf97, buf99, buf100, 2048, grid=grid(2048), stream=stream0)
        buf72 = buf51; del buf51  # reuse
        # Source Nodes: [setitem_4], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg265_1, buf72, 4718592, grid=grid(4718592), stream=stream0)
        del buf66
        del buf67
        del buf69
        del buf70
        buf102 = buf48; del buf48  # reuse
        # Source Nodes: [setitem_6], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg267_1, buf102, 4718592, grid=grid(4718592), stream=stream0)
        del buf100
        del buf99
        buf109 = buf19; del buf19  # reuse
        # Source Nodes: [setitem_7], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg268_1, buf109, 4718592, grid=grid(4718592), stream=stream0)
        buf79 = buf41; del buf41  # reuse
        # Source Nodes: [setitem_5], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg266_1, buf79, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_4, setitem_5, setitem_6, setitem_7], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg326_1, buf71, buf34, buf101, buf64, buf72, buf79, buf102, buf109, 4096, grid=grid(4096), stream=stream0)
        buf74 = reinterpret_tensor(buf63, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf63  # reuse
        # Source Nodes: [type_as_9], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf68, buf74, 4096, grid=grid(4096), stream=stream0)
        buf75 = buf12; del buf12  # reuse
        # Source Nodes: [setitem_4], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf72, buf75, arg265_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg265_1
        buf76 = reinterpret_tensor(buf50, (32, 1, 1152), (1152, 1152, 1)); del buf50  # reuse
        # Source Nodes: [matmul_4], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf74, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf75, (32, 128, 1152), (147456, 1, 128), 0), out=buf76)
        buf104 = buf74; del buf74  # reuse
        # Source Nodes: [type_as_13], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf98, buf104, 4096, grid=grid(4096), stream=stream0)
        del buf96
        del buf97
        buf105 = buf75; del buf75  # reuse
        # Source Nodes: [setitem_6], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf102, buf105, arg267_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg267_1
        buf106 = reinterpret_tensor(buf21, (32, 1, 1152), (1152, 1152, 1)); del buf21  # reuse
        # Source Nodes: [matmul_6], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf104, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf105, (32, 128, 1152), (147456, 1, 128), 0), out=buf106)
        buf81 = reinterpret_tensor(buf45, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf45  # reuse
        buf111 = reinterpret_tensor(buf16, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf16  # reuse
        # Source Nodes: [getitem, mul_46, mul_65, softmax_2, softmax_3, where_2, where_3], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg326_1, arg260_1, buf76, buf106, buf81, buf111, 32, 1152, grid=grid(32), stream=stream0)
        buf82 = buf105; del buf105  # reuse
        # Source Nodes: [setitem_5], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf79, buf82, arg266_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg266_1
        buf83 = reinterpret_tensor(buf104, (32, 1, 128), (128, 128, 1)); del buf104  # reuse
        # Source Nodes: [matmul_5], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf81, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf82, (32, 1152, 128), (147456, 128, 1), 0), out=buf83)
        buf84 = reinterpret_tensor(buf60, (1, 4096), (4096, 1)); del buf60  # reuse
        # Source Nodes: [l__model___layers_2_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf83, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg109_1, (4096, 4096), (1, 4096), 0), out=buf84)
        del arg109_1
        buf86 = reinterpret_tensor(buf83, (1, 1, 4096), (4096, 4096, 1)); del buf83  # reuse
        # Source Nodes: [add_18, add_19, float_12, mean_5, mul_47, mul_48, mul_49, rsqrt_5, type_as_11], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf31, buf84, arg7_1, buf86, 1, 4096, grid=grid(1), stream=stream0)
        del arg7_1
        buf87 = reinterpret_tensor(buf58, (1, 11008), (11008, 1)); del buf58  # reuse
        # Source Nodes: [l__model___layers_2_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (1, 4096), (0, 1), 0), reinterpret_tensor(arg110_1, (4096, 11008), (1, 4096), 0), out=buf87)
        del arg110_1
        buf88 = buf57; del buf57  # reuse
        # Source Nodes: [l__model___layers_2_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf86, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg111_1, (4096, 11008), (1, 4096), 0), out=buf88)
        del arg111_1
        buf89 = reinterpret_tensor(buf87, (1, 1, 11008), (11008, 11008, 1)); del buf87  # reuse
        # Source Nodes: [mul_50, silu_2], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf89, buf88, 11008, grid=grid(11008), stream=stream0)
        buf90 = reinterpret_tensor(buf86, (1, 4096), (4096, 1)); del buf86  # reuse
        # Source Nodes: [l__model___layers_2_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf89, (1, 11008), (0, 1), 0), reinterpret_tensor(arg112_1, (11008, 4096), (1, 11008), 0), out=buf90)
        del arg112_1
        buf91 = buf65; del buf65  # reuse
        buf93 = reinterpret_tensor(buf59, (1, 1, 4096), (4096, 4096, 1)); del buf59  # reuse
        # Source Nodes: [add_18, add_20, add_33, add_34, float_17, mean_8, mul_73, mul_74, mul_75, mul_76, rsqrt_8, type_as_16], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_12.run(buf91, arg11_1, buf31, buf84, buf90, arg12_1, buf93, 1, 4096, grid=grid(1), stream=stream0)
        del arg11_1
        del arg12_1
        buf94 = buf64; del buf64  # reuse
        # Source Nodes: [l__model___layers_4_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf93, (1, 4096), (0, 1), 0), reinterpret_tensor(arg118_1, (4096, 12288), (1, 4096), 0), out=buf94)
        del arg118_1
        buf112 = buf82; del buf82  # reuse
        # Source Nodes: [setitem_7], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf109, buf112, arg268_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg268_1
        buf113 = reinterpret_tensor(buf93, (32, 1, 128), (128, 128, 1)); del buf93  # reuse
        # Source Nodes: [matmul_7], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf111, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf112, (32, 1152, 128), (147456, 128, 1), 0), out=buf113)
        buf114 = buf30; del buf30  # reuse
        # Source Nodes: [l__model___layers_3_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf113, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg114_1, (4096, 4096), (1, 4096), 0), out=buf114)
        del arg114_1
        buf116 = reinterpret_tensor(buf113, (1, 1, 4096), (4096, 4096, 1)); del buf113  # reuse
        # Source Nodes: [add_27, add_28, float_16, mean_7, mul_66, mul_67, mul_68, rsqrt_7, type_as_15], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf61, buf114, arg10_1, buf116, 1, 4096, grid=grid(1), stream=stream0)
        del arg10_1
        buf117 = reinterpret_tensor(buf89, (1, 11008), (11008, 1)); del buf89  # reuse
        # Source Nodes: [l__model___layers_3_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (1, 4096), (0, 1), 0), reinterpret_tensor(arg115_1, (4096, 11008), (1, 4096), 0), out=buf117)
        del arg115_1
        buf118 = buf88; del buf88  # reuse
        # Source Nodes: [l__model___layers_3_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf116, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg116_1, (4096, 11008), (1, 4096), 0), out=buf118)
        del arg116_1
        buf119 = reinterpret_tensor(buf117, (1, 1, 11008), (11008, 11008, 1)); del buf117  # reuse
        # Source Nodes: [mul_69, silu_3], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf119, buf118, 11008, grid=grid(11008), stream=stream0)
        buf120 = reinterpret_tensor(buf116, (1, 4096), (4096, 1)); del buf116  # reuse
        # Source Nodes: [l__model___layers_3_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf119, (1, 11008), (0, 1), 0), reinterpret_tensor(arg117_1, (11008, 4096), (1, 11008), 0), out=buf120)
        del arg117_1
        buf121 = buf95; del buf95  # reuse
        buf126 = buf125; del buf125  # reuse
        buf157 = buf156; del buf156  # reuse
        buf123 = reinterpret_tensor(buf24, (1, 1, 4096), (4096, 4096, 1)); del buf24  # reuse
        # Source Nodes: [add_18, add_20, add_27, add_29, add_43, add_44, add_45, add_54, add_55, add_66, add_67, float_21, mean_10, mul_114, mul_115, mul_136, mul_137, mul_93, mul_94, mul_95, mul_96, mul_97, rsqrt_10, type_as_20], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_13.run(buf121, buf126, buf157, arg14_1, buf31, buf84, buf90, buf61, buf114, buf120, arg17_1, arg20_1, arg15_1, buf123, 1, 4096, grid=grid(1), stream=stream0)
        del arg14_1
        del arg15_1
        buf124 = buf34; del buf34  # reuse
        # Source Nodes: [l__model___layers_5_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf123, (1, 4096), (0, 1), 0), reinterpret_tensor(arg123_1, (4096, 12288), (1, 4096), 0), out=buf124)
        del arg123_1
        buf129 = buf98; del buf98  # reuse
        buf127 = reinterpret_tensor(buf129, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf128 = reinterpret_tensor(buf129, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf132 = buf68; del buf68  # reuse
        buf130 = reinterpret_tensor(buf132, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf131 = reinterpret_tensor(buf132, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf160 = buf71; del buf71  # reuse
        buf158 = reinterpret_tensor(buf160, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf159 = reinterpret_tensor(buf160, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf163 = buf101; del buf101  # reuse
        buf161 = reinterpret_tensor(buf163, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf162 = reinterpret_tensor(buf163, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_10, stack_11, stack_8, stack_9], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf94, arg326_1, arg259_1, buf124, buf127, buf128, buf130, buf131, buf158, buf159, buf161, buf162, 2048, grid=grid(2048), stream=stream0)
        buf133 = buf112; del buf112  # reuse
        # Source Nodes: [setitem_8], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg269_1, buf133, 4718592, grid=grid(4718592), stream=stream0)
        del buf127
        del buf128
        del buf130
        del buf131
        buf140 = buf109; del buf109  # reuse
        # Source Nodes: [setitem_9], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg270_1, buf140, 4718592, grid=grid(4718592), stream=stream0)
        buf164 = buf79; del buf79  # reuse
        # Source Nodes: [setitem_10], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg271_1, buf164, 4718592, grid=grid(4718592), stream=stream0)
        del buf161
        del buf162
        buf171 = buf102; del buf102  # reuse
        # Source Nodes: [setitem_11], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg272_1, buf171, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_10, setitem_11, setitem_8, setitem_9], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg326_1, buf132, buf94, buf163, buf124, buf133, buf140, buf164, buf171, 4096, grid=grid(4096), stream=stream0)
        buf135 = reinterpret_tensor(buf123, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf123  # reuse
        # Source Nodes: [type_as_17], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf129, buf135, 4096, grid=grid(4096), stream=stream0)
        buf136 = buf72; del buf72  # reuse
        # Source Nodes: [setitem_8], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf133, buf136, arg269_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg269_1
        buf137 = reinterpret_tensor(buf111, (32, 1, 1152), (1152, 1152, 1)); del buf111  # reuse
        # Source Nodes: [matmul_8], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf135, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf136, (32, 128, 1152), (147456, 1, 128), 0), out=buf137)
        buf166 = buf135; del buf135  # reuse
        # Source Nodes: [type_as_21], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf160, buf166, 4096, grid=grid(4096), stream=stream0)
        del buf158
        del buf159
        buf167 = buf136; del buf136  # reuse
        # Source Nodes: [setitem_10], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf164, buf167, arg271_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg271_1
        buf168 = reinterpret_tensor(buf81, (32, 1, 1152), (1152, 1152, 1)); del buf81  # reuse
        # Source Nodes: [matmul_10], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf166, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf167, (32, 128, 1152), (147456, 1, 128), 0), out=buf168)
        buf142 = reinterpret_tensor(buf76, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf76  # reuse
        buf173 = reinterpret_tensor(buf106, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf106  # reuse
        # Source Nodes: [getitem, mul_106, mul_85, softmax_4, softmax_5, where_4, where_5], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg326_1, arg260_1, buf137, buf168, buf142, buf173, 32, 1152, grid=grid(32), stream=stream0)
        buf143 = buf167; del buf167  # reuse
        # Source Nodes: [setitem_9], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf140, buf143, arg270_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg270_1
        buf144 = reinterpret_tensor(buf166, (32, 1, 128), (128, 128, 1)); del buf166  # reuse
        # Source Nodes: [matmul_9], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf142, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf143, (32, 1152, 128), (147456, 128, 1), 0), out=buf144)
        buf145 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_4_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf144, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg119_1, (4096, 4096), (1, 4096), 0), out=buf145)
        del arg119_1
        buf147 = reinterpret_tensor(buf144, (1, 1, 4096), (4096, 4096, 1)); del buf144  # reuse
        # Source Nodes: [add_37, add_38, float_20, mean_9, mul_86, mul_87, mul_88, rsqrt_9, type_as_19], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf91, buf145, arg13_1, buf147, 1, 4096, grid=grid(1), stream=stream0)
        del arg13_1
        buf148 = reinterpret_tensor(buf119, (1, 11008), (11008, 1)); del buf119  # reuse
        # Source Nodes: [l__model___layers_4_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf147, (1, 4096), (0, 1), 0), reinterpret_tensor(arg120_1, (4096, 11008), (1, 4096), 0), out=buf148)
        del arg120_1
        buf149 = buf118; del buf118  # reuse
        # Source Nodes: [l__model___layers_4_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf147, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg121_1, (4096, 11008), (1, 4096), 0), out=buf149)
        del arg121_1
        buf150 = reinterpret_tensor(buf148, (1, 1, 11008), (11008, 11008, 1)); del buf148  # reuse
        # Source Nodes: [mul_89, silu_4], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf150, buf149, 11008, grid=grid(11008), stream=stream0)
        buf151 = reinterpret_tensor(buf147, (1, 4096), (4096, 1)); del buf147  # reuse
        # Source Nodes: [l__model___layers_4_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf150, (1, 11008), (0, 1), 0), reinterpret_tensor(arg122_1, (11008, 4096), (1, 11008), 0), out=buf151)
        del arg122_1
        buf152 = buf126; del buf126  # reuse
        buf154 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_37, add_39, add_56, add_57, float_25, mean_12, mul_116, mul_117, mul_118, mul_119, rsqrt_12, type_as_24], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_14.run(buf152, arg17_1, buf91, buf145, buf151, arg18_1, buf154, 1, 4096, grid=grid(1), stream=stream0)
        del arg17_1
        del arg18_1
        buf155 = buf94; del buf94  # reuse
        # Source Nodes: [l__model___layers_6_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf154, (1, 4096), (0, 1), 0), reinterpret_tensor(arg128_1, (4096, 12288), (1, 4096), 0), out=buf155)
        del arg128_1
        buf174 = buf143; del buf143  # reuse
        # Source Nodes: [setitem_11], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf171, buf174, arg272_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg272_1
        buf175 = reinterpret_tensor(buf154, (32, 1, 128), (128, 128, 1)); del buf154  # reuse
        # Source Nodes: [matmul_11], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf173, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf174, (32, 1152, 128), (147456, 128, 1), 0), out=buf175)
        buf176 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_5_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf175, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg124_1, (4096, 4096), (1, 4096), 0), out=buf176)
        del arg124_1
        buf178 = reinterpret_tensor(buf175, (1, 1, 4096), (4096, 4096, 1)); del buf175  # reuse
        # Source Nodes: [add_48, add_49, float_24, mean_11, mul_107, mul_108, mul_109, rsqrt_11, type_as_23], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf121, buf176, arg16_1, buf178, 1, 4096, grid=grid(1), stream=stream0)
        del arg16_1
        buf179 = reinterpret_tensor(buf150, (1, 11008), (11008, 1)); del buf150  # reuse
        # Source Nodes: [l__model___layers_5_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf178, (1, 4096), (0, 1), 0), reinterpret_tensor(arg125_1, (4096, 11008), (1, 4096), 0), out=buf179)
        del arg125_1
        buf180 = buf149; del buf149  # reuse
        # Source Nodes: [l__model___layers_5_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf178, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg126_1, (4096, 11008), (1, 4096), 0), out=buf180)
        del arg126_1
        buf181 = reinterpret_tensor(buf179, (1, 1, 11008), (11008, 11008, 1)); del buf179  # reuse
        # Source Nodes: [mul_110, silu_5], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf181, buf180, 11008, grid=grid(11008), stream=stream0)
        buf182 = reinterpret_tensor(buf178, (1, 4096), (4096, 1)); del buf178  # reuse
        # Source Nodes: [l__model___layers_5_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf181, (1, 11008), (0, 1), 0), reinterpret_tensor(arg127_1, (11008, 4096), (1, 11008), 0), out=buf182)
        del arg127_1
        buf1045 = buf1044; del buf1044  # reuse
        buf1095 = buf1094; del buf1094  # reuse
        buf960 = buf959; del buf959  # reuse
        buf1002 = buf1001; del buf1001  # reuse
        buf877 = buf876; del buf876  # reuse
        buf918 = buf917; del buf917  # reuse
        buf796 = buf795; del buf795  # reuse
        buf836 = buf835; del buf835  # reuse
        buf717 = buf716; del buf716  # reuse
        buf756 = buf755; del buf755  # reuse
        buf640 = buf639; del buf639  # reuse
        buf678 = buf677; del buf677  # reuse
        buf565 = buf564; del buf564  # reuse
        buf602 = buf601; del buf601  # reuse
        buf492 = buf491; del buf491  # reuse
        buf528 = buf527; del buf527  # reuse
        buf421 = buf420; del buf420  # reuse
        buf456 = buf455; del buf455  # reuse
        buf352 = buf351; del buf351  # reuse
        buf386 = buf385; del buf385  # reuse
        buf285 = buf284; del buf284  # reuse
        buf318 = buf317; del buf317  # reuse
        buf188 = buf187; del buf187  # reuse
        buf220 = buf219; del buf219  # reuse
        buf252 = buf251; del buf251  # reuse
        buf183 = buf157; del buf157  # reuse
        buf189 = buf188; del buf188  # reuse
        buf221 = buf220; del buf220  # reuse
        buf253 = buf252; del buf252  # reuse
        buf286 = buf285; del buf285  # reuse
        buf319 = buf318; del buf318  # reuse
        buf353 = buf352; del buf352  # reuse
        buf387 = buf386; del buf386  # reuse
        buf422 = buf421; del buf421  # reuse
        buf457 = buf456; del buf456  # reuse
        buf493 = buf492; del buf492  # reuse
        buf529 = buf528; del buf528  # reuse
        buf566 = buf565; del buf565  # reuse
        buf603 = buf602; del buf602  # reuse
        buf641 = buf640; del buf640  # reuse
        buf679 = buf678; del buf678  # reuse
        buf718 = buf717; del buf717  # reuse
        buf757 = buf756; del buf756  # reuse
        buf797 = buf796; del buf796  # reuse
        buf837 = buf836; del buf836  # reuse
        buf878 = buf877; del buf877  # reuse
        buf919 = buf918; del buf918  # reuse
        buf961 = buf960; del buf960  # reuse
        buf1003 = buf1002; del buf1002  # reuse
        buf1046 = buf1045; del buf1045  # reuse
        buf1096 = buf1095; del buf1095  # reuse
        buf185 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_108, add_109, add_110, add_111, add_124, add_125, add_126, add_127, add_141, add_142, add_143, add_144, add_159, add_160, add_161, add_162, add_178, add_179, add_18, add_180, add_181, add_198, add_199, add_20, add_200, add_201, add_219, add_220, add_221, add_222, add_241, add_242, add_243, add_244, add_264, add_265, add_266, add_267, add_27, add_288, add_289, add_29, add_290, add_291, add_313, add_314, add_315, add_316, add_339, add_340, add_341, add_342, add_366, add_367, add_368, add_369, add_37, add_39, add_394, add_395, add_396, add_397, add_423, add_424, add_425, add_426, add_453, add_454, add_455, add_456, add_48, add_484, add_485, add_486, add_487, add_50, add_516, add_517, add_518, add_519, add_549, add_550, add_551, add_552, add_583, add_584, add_585, add_586, add_618, add_619, add_620, add_621, add_654, add_655, add_656, add_657, add_68, add_69, add_691, add_692, add_693, add_694, add_70, add_79, add_80, add_81, add_82, add_93, add_94, add_95, add_96, float_29, mean_14, mul_1011, mul_1012, mul_1013, mul_1014, mul_138, mul_139, mul_140, mul_141, mul_142, mul_159, mul_160, mul_161, mul_162, mul_183, mul_184, mul_185, mul_186, mul_208, mul_209, mul_210, mul_211, mul_234, mul_235, mul_236, mul_237, mul_261, mul_262, mul_263, mul_264, mul_289, mul_290, mul_291, mul_292, mul_318, mul_319, mul_320, mul_321, mul_348, mul_349, mul_350, mul_351, mul_379, mul_380, mul_381, mul_382, mul_411, mul_412, mul_413, mul_414, mul_444, mul_445, mul_446, mul_447, mul_478, mul_479, mul_480, mul_481, mul_513, mul_514, mul_515, mul_516, mul_549, mul_550, mul_551, mul_552, mul_586, mul_587, mul_588, mul_589, mul_624, mul_625, mul_626, mul_627, mul_663, mul_664, mul_665, mul_666, mul_703, mul_704, mul_705, mul_706, mul_744, mul_745, mul_746, mul_747, mul_786, mul_787, mul_788, mul_789, mul_829, mul_830, mul_831, mul_832, mul_873, mul_874, mul_875, mul_876, mul_918, mul_919, mul_920, mul_921, mul_964, mul_965, mul_966, mul_967, rsqrt_14, type_as_28], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(buf1046, buf1096, buf961, buf1003, buf878, buf919, buf797, buf837, buf718, buf757, buf641, buf679, buf566, buf603, buf493, buf529, buf422, buf457, buf353, buf387, buf286, buf319, buf189, buf221, buf253, buf183, arg92_1, buf31, buf84, buf90, buf61, buf114, buf120, arg95_1, arg86_1, arg89_1, arg80_1, arg83_1, arg74_1, arg77_1, arg68_1, arg71_1, arg62_1, arg65_1, arg56_1, arg59_1, arg50_1, arg53_1, arg44_1, arg47_1, arg38_1, arg41_1, arg32_1, arg35_1, arg23_1, arg26_1, arg29_1, arg20_1, buf91, buf145, buf151, buf121, buf176, buf182, arg21_1, buf185, 1, 4096, grid=grid(1), stream=stream0)
        del arg20_1
        del arg21_1
        buf186 = buf124; del buf124  # reuse
        # Source Nodes: [l__model___layers_7_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf185, (1, 4096), (0, 1), 0), reinterpret_tensor(arg133_1, (4096, 12288), (1, 4096), 0), out=buf186)
        del arg133_1
        buf192 = buf160; del buf160  # reuse
        buf190 = reinterpret_tensor(buf192, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf191 = reinterpret_tensor(buf192, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf195 = buf129; del buf129  # reuse
        buf193 = reinterpret_tensor(buf195, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf194 = reinterpret_tensor(buf195, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf224 = buf163; del buf163  # reuse
        buf222 = reinterpret_tensor(buf224, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf223 = reinterpret_tensor(buf224, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf227 = buf132; del buf132  # reuse
        buf225 = reinterpret_tensor(buf227, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf226 = reinterpret_tensor(buf227, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_12, stack_13, stack_14, stack_15], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf155, arg326_1, arg259_1, buf186, buf190, buf191, buf193, buf194, buf222, buf223, buf225, buf226, 2048, grid=grid(2048), stream=stream0)
        buf196 = buf174; del buf174  # reuse
        # Source Nodes: [setitem_12], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg273_1, buf196, 4718592, grid=grid(4718592), stream=stream0)
        del buf190
        del buf191
        del buf193
        del buf194
        buf203 = buf171; del buf171  # reuse
        # Source Nodes: [setitem_13], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg274_1, buf203, 4718592, grid=grid(4718592), stream=stream0)
        buf228 = buf140; del buf140  # reuse
        # Source Nodes: [setitem_14], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg275_1, buf228, 4718592, grid=grid(4718592), stream=stream0)
        del buf225
        del buf226
        buf235 = buf164; del buf164  # reuse
        # Source Nodes: [setitem_15], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg276_1, buf235, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_12, setitem_13, setitem_14, setitem_15], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg326_1, buf195, buf155, buf227, buf186, buf196, buf203, buf228, buf235, 4096, grid=grid(4096), stream=stream0)
        buf198 = reinterpret_tensor(buf185, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf185  # reuse
        # Source Nodes: [type_as_25], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf192, buf198, 4096, grid=grid(4096), stream=stream0)
        buf199 = buf133; del buf133  # reuse
        # Source Nodes: [setitem_12], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf196, buf199, arg273_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg273_1
        buf200 = reinterpret_tensor(buf173, (32, 1, 1152), (1152, 1152, 1)); del buf173  # reuse
        # Source Nodes: [matmul_12], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf198, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf199, (32, 128, 1152), (147456, 1, 128), 0), out=buf200)
        buf230 = buf198; del buf198  # reuse
        # Source Nodes: [type_as_29], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf224, buf230, 4096, grid=grid(4096), stream=stream0)
        del buf222
        del buf223
        buf231 = buf199; del buf199  # reuse
        # Source Nodes: [setitem_14], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf228, buf231, arg275_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg275_1
        buf232 = reinterpret_tensor(buf142, (32, 1, 1152), (1152, 1152, 1)); del buf142  # reuse
        # Source Nodes: [matmul_14], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf230, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf231, (32, 128, 1152), (147456, 1, 128), 0), out=buf232)
        buf205 = reinterpret_tensor(buf168, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf168  # reuse
        buf237 = reinterpret_tensor(buf137, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf137  # reuse
        # Source Nodes: [getitem, mul_128, mul_151, softmax_6, softmax_7, where_6, where_7], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg326_1, arg260_1, buf200, buf232, buf205, buf237, 32, 1152, grid=grid(32), stream=stream0)
        buf206 = buf231; del buf231  # reuse
        # Source Nodes: [setitem_13], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf203, buf206, arg274_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg274_1
        buf207 = reinterpret_tensor(buf230, (32, 1, 128), (128, 128, 1)); del buf230  # reuse
        # Source Nodes: [matmul_13], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf205, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf206, (32, 1152, 128), (147456, 128, 1), 0), out=buf207)
        buf208 = reinterpret_tensor(buf91, (1, 4096), (4096, 1)); del buf91  # reuse
        # Source Nodes: [l__model___layers_6_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf207, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg129_1, (4096, 4096), (1, 4096), 0), out=buf208)
        del arg129_1
        buf210 = reinterpret_tensor(buf207, (1, 1, 4096), (4096, 4096, 1)); del buf207  # reuse
        # Source Nodes: [add_60, add_61, float_28, mean_13, mul_129, mul_130, mul_131, rsqrt_13, type_as_27], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf152, buf208, arg19_1, buf210, 1, 4096, grid=grid(1), stream=stream0)
        del arg19_1
        buf211 = reinterpret_tensor(buf181, (1, 11008), (11008, 1)); del buf181  # reuse
        # Source Nodes: [l__model___layers_6_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf210, (1, 4096), (0, 1), 0), reinterpret_tensor(arg130_1, (4096, 11008), (1, 4096), 0), out=buf211)
        del arg130_1
        buf212 = buf180; del buf180  # reuse
        # Source Nodes: [l__model___layers_6_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf210, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg131_1, (4096, 11008), (1, 4096), 0), out=buf212)
        del arg131_1
        buf213 = reinterpret_tensor(buf211, (1, 1, 11008), (11008, 11008, 1)); del buf211  # reuse
        # Source Nodes: [mul_132, silu_6], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf213, buf212, 11008, grid=grid(11008), stream=stream0)
        buf214 = reinterpret_tensor(buf210, (1, 4096), (4096, 1)); del buf210  # reuse
        # Source Nodes: [l__model___layers_6_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf213, (1, 11008), (0, 1), 0), reinterpret_tensor(arg132_1, (11008, 4096), (1, 11008), 0), out=buf214)
        del arg132_1
        buf215 = buf189; del buf189  # reuse
        buf217 = reinterpret_tensor(buf90, (1, 1, 4096), (4096, 4096, 1)); del buf90  # reuse
        # Source Nodes: [add_60, add_62, add_83, add_84, float_33, mean_16, mul_163, mul_164, mul_165, mul_166, rsqrt_16, type_as_32], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_16.run(buf215, arg23_1, buf152, buf208, buf214, arg24_1, buf217, 1, 4096, grid=grid(1), stream=stream0)
        del arg23_1
        del arg24_1
        buf218 = buf186; del buf186  # reuse
        # Source Nodes: [l__model___layers_8_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf217, (1, 4096), (0, 1), 0), reinterpret_tensor(arg138_1, (4096, 12288), (1, 4096), 0), out=buf218)
        del arg138_1
        buf238 = buf206; del buf206  # reuse
        # Source Nodes: [setitem_15], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf235, buf238, arg276_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg276_1
        buf239 = reinterpret_tensor(buf217, (32, 1, 128), (128, 128, 1)); del buf217  # reuse
        # Source Nodes: [matmul_15], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf237, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf238, (32, 1152, 128), (147456, 128, 1), 0), out=buf239)
        buf240 = buf84; del buf84  # reuse
        # Source Nodes: [l__model___layers_7_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf239, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg134_1, (4096, 4096), (1, 4096), 0), out=buf240)
        del arg134_1
        buf242 = reinterpret_tensor(buf239, (1, 1, 4096), (4096, 4096, 1)); del buf239  # reuse
        # Source Nodes: [add_73, add_74, float_32, mean_15, mul_152, mul_153, mul_154, rsqrt_15, type_as_31], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf183, buf240, arg22_1, buf242, 1, 4096, grid=grid(1), stream=stream0)
        del arg22_1
        buf243 = reinterpret_tensor(buf213, (1, 11008), (11008, 1)); del buf213  # reuse
        # Source Nodes: [l__model___layers_7_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf242, (1, 4096), (0, 1), 0), reinterpret_tensor(arg135_1, (4096, 11008), (1, 4096), 0), out=buf243)
        del arg135_1
        buf244 = buf212; del buf212  # reuse
        # Source Nodes: [l__model___layers_7_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf242, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg136_1, (4096, 11008), (1, 4096), 0), out=buf244)
        del arg136_1
        buf245 = reinterpret_tensor(buf243, (1, 1, 11008), (11008, 11008, 1)); del buf243  # reuse
        # Source Nodes: [mul_155, silu_7], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf245, buf244, 11008, grid=grid(11008), stream=stream0)
        buf246 = reinterpret_tensor(buf242, (1, 4096), (4096, 1)); del buf242  # reuse
        # Source Nodes: [l__model___layers_7_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf245, (1, 11008), (0, 1), 0), reinterpret_tensor(arg137_1, (11008, 4096), (1, 11008), 0), out=buf246)
        del arg137_1
        buf247 = buf221; del buf221  # reuse
        buf254 = buf253; del buf253  # reuse
        buf287 = buf286; del buf286  # reuse
        buf249 = buf61; del buf61  # reuse
        # Source Nodes: [add_112, add_113, add_128, add_129, add_60, add_62, add_73, add_75, add_97, add_98, add_99, float_37, mean_18, mul_187, mul_188, mul_189, mul_190, mul_191, mul_212, mul_213, mul_238, mul_239, rsqrt_18, type_as_36], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_17.run(buf247, buf254, buf287, arg26_1, buf152, buf208, buf214, buf183, buf240, buf246, arg29_1, arg32_1, arg27_1, buf249, 1, 4096, grid=grid(1), stream=stream0)
        del arg26_1
        del arg27_1
        buf250 = buf155; del buf155  # reuse
        # Source Nodes: [l__model___layers_9_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (1, 4096), (0, 1), 0), reinterpret_tensor(arg143_1, (4096, 12288), (1, 4096), 0), out=buf250)
        del arg143_1
        buf257 = buf224; del buf224  # reuse
        buf255 = reinterpret_tensor(buf257, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf256 = reinterpret_tensor(buf257, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf260 = buf192; del buf192  # reuse
        buf258 = reinterpret_tensor(buf260, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf259 = reinterpret_tensor(buf260, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf290 = buf227; del buf227  # reuse
        buf288 = reinterpret_tensor(buf290, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf289 = reinterpret_tensor(buf290, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf293 = buf195; del buf195  # reuse
        buf291 = reinterpret_tensor(buf293, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf292 = reinterpret_tensor(buf293, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_16, stack_17, stack_18, stack_19], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf218, arg326_1, arg259_1, buf250, buf255, buf256, buf258, buf259, buf288, buf289, buf291, buf292, 2048, grid=grid(2048), stream=stream0)
        buf261 = buf238; del buf238  # reuse
        # Source Nodes: [setitem_16], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg277_1, buf261, 4718592, grid=grid(4718592), stream=stream0)
        del buf255
        del buf256
        del buf258
        del buf259
        buf268 = buf235; del buf235  # reuse
        # Source Nodes: [setitem_17], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg278_1, buf268, 4718592, grid=grid(4718592), stream=stream0)
        buf294 = buf203; del buf203  # reuse
        # Source Nodes: [setitem_18], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg279_1, buf294, 4718592, grid=grid(4718592), stream=stream0)
        del buf291
        del buf292
        buf301 = buf228; del buf228  # reuse
        # Source Nodes: [setitem_19], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg280_1, buf301, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_16, setitem_17, setitem_18, setitem_19], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg326_1, buf260, buf218, buf293, buf250, buf261, buf268, buf294, buf301, 4096, grid=grid(4096), stream=stream0)
        buf263 = reinterpret_tensor(buf249, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf249  # reuse
        # Source Nodes: [type_as_33], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf257, buf263, 4096, grid=grid(4096), stream=stream0)
        buf264 = buf196; del buf196  # reuse
        # Source Nodes: [setitem_16], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf261, buf264, arg277_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg277_1
        buf265 = reinterpret_tensor(buf237, (32, 1, 1152), (1152, 1152, 1)); del buf237  # reuse
        # Source Nodes: [matmul_16], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf263, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf264, (32, 128, 1152), (147456, 1, 128), 0), out=buf265)
        buf296 = buf263; del buf263  # reuse
        # Source Nodes: [type_as_37], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf290, buf296, 4096, grid=grid(4096), stream=stream0)
        del buf288
        del buf289
        buf297 = buf264; del buf264  # reuse
        # Source Nodes: [setitem_18], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf294, buf297, arg279_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg279_1
        buf298 = reinterpret_tensor(buf205, (32, 1, 1152), (1152, 1152, 1)); del buf205  # reuse
        # Source Nodes: [matmul_18], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf296, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf297, (32, 128, 1152), (147456, 1, 128), 0), out=buf298)
        buf270 = reinterpret_tensor(buf232, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf232  # reuse
        buf303 = reinterpret_tensor(buf200, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf200  # reuse
        # Source Nodes: [getitem, mul_175, mul_200, softmax_8, softmax_9, where_8, where_9], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg326_1, arg260_1, buf265, buf298, buf270, buf303, 32, 1152, grid=grid(32), stream=stream0)
        buf271 = buf297; del buf297  # reuse
        # Source Nodes: [setitem_17], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf268, buf271, arg278_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg278_1
        buf272 = reinterpret_tensor(buf296, (32, 1, 128), (128, 128, 1)); del buf296  # reuse
        # Source Nodes: [matmul_17], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf270, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf271, (32, 1152, 128), (147456, 128, 1), 0), out=buf272)
        buf273 = reinterpret_tensor(buf31, (1, 4096), (4096, 1)); del buf31  # reuse
        # Source Nodes: [l__model___layers_8_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf272, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg139_1, (4096, 4096), (1, 4096), 0), out=buf273)
        del arg139_1
        buf275 = reinterpret_tensor(buf272, (1, 1, 4096), (4096, 4096, 1)); del buf272  # reuse
        # Source Nodes: [add_87, add_88, float_36, mean_17, mul_176, mul_177, mul_178, rsqrt_17, type_as_35], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf215, buf273, arg25_1, buf275, 1, 4096, grid=grid(1), stream=stream0)
        del arg25_1
        buf276 = reinterpret_tensor(buf245, (1, 11008), (11008, 1)); del buf245  # reuse
        # Source Nodes: [l__model___layers_8_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf275, (1, 4096), (0, 1), 0), reinterpret_tensor(arg140_1, (4096, 11008), (1, 4096), 0), out=buf276)
        del arg140_1
        buf277 = buf244; del buf244  # reuse
        # Source Nodes: [l__model___layers_8_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf275, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg141_1, (4096, 11008), (1, 4096), 0), out=buf277)
        del arg141_1
        buf278 = reinterpret_tensor(buf276, (1, 1, 11008), (11008, 11008, 1)); del buf276  # reuse
        # Source Nodes: [mul_179, silu_8], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf278, buf277, 11008, grid=grid(11008), stream=stream0)
        buf279 = reinterpret_tensor(buf275, (1, 4096), (4096, 1)); del buf275  # reuse
        # Source Nodes: [l__model___layers_8_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf278, (1, 11008), (0, 1), 0), reinterpret_tensor(arg142_1, (11008, 4096), (1, 11008), 0), out=buf279)
        del arg142_1
        buf280 = buf254; del buf254  # reuse
        buf282 = reinterpret_tensor(buf182, (1, 1, 4096), (4096, 4096, 1)); del buf182  # reuse
        # Source Nodes: [add_114, add_115, add_87, add_89, float_41, mean_20, mul_214, mul_215, mul_216, mul_217, rsqrt_20, type_as_40], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_18.run(buf280, arg29_1, buf215, buf273, buf279, arg30_1, buf282, 1, 4096, grid=grid(1), stream=stream0)
        del arg29_1
        del arg30_1
        buf283 = buf250; del buf250  # reuse
        # Source Nodes: [l__model___layers_10_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf282, (1, 4096), (0, 1), 0), reinterpret_tensor(arg148_1, (4096, 12288), (1, 4096), 0), out=buf283)
        del arg148_1
        buf304 = buf271; del buf271  # reuse
        # Source Nodes: [setitem_19], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf301, buf304, arg280_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg280_1
        buf305 = reinterpret_tensor(buf282, (32, 1, 128), (128, 128, 1)); del buf282  # reuse
        # Source Nodes: [matmul_19], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf303, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf304, (32, 1152, 128), (147456, 128, 1), 0), out=buf305)
        buf306 = buf176; del buf176  # reuse
        # Source Nodes: [l__model___layers_9_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf305, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg144_1, (4096, 4096), (1, 4096), 0), out=buf306)
        del arg144_1
        buf308 = reinterpret_tensor(buf305, (1, 1, 4096), (4096, 4096, 1)); del buf305  # reuse
        # Source Nodes: [add_102, add_103, float_40, mean_19, mul_201, mul_202, mul_203, rsqrt_19, type_as_39], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf247, buf306, arg28_1, buf308, 1, 4096, grid=grid(1), stream=stream0)
        del arg28_1
        buf309 = reinterpret_tensor(buf278, (1, 11008), (11008, 1)); del buf278  # reuse
        # Source Nodes: [l__model___layers_9_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf308, (1, 4096), (0, 1), 0), reinterpret_tensor(arg145_1, (4096, 11008), (1, 4096), 0), out=buf309)
        del arg145_1
        buf310 = buf277; del buf277  # reuse
        # Source Nodes: [l__model___layers_9_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf308, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg146_1, (4096, 11008), (1, 4096), 0), out=buf310)
        del arg146_1
        buf311 = reinterpret_tensor(buf309, (1, 1, 11008), (11008, 11008, 1)); del buf309  # reuse
        # Source Nodes: [mul_204, silu_9], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf311, buf310, 11008, grid=grid(11008), stream=stream0)
        buf312 = reinterpret_tensor(buf308, (1, 4096), (4096, 1)); del buf308  # reuse
        # Source Nodes: [l__model___layers_9_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf311, (1, 11008), (0, 1), 0), reinterpret_tensor(arg147_1, (11008, 4096), (1, 11008), 0), out=buf312)
        del arg147_1
        buf320 = buf319; del buf319  # reuse
        buf354 = buf353; del buf353  # reuse
        buf313 = buf287; del buf287  # reuse
        buf321 = buf320; del buf320  # reuse
        buf355 = buf354; del buf354  # reuse
        buf315 = reinterpret_tensor(buf151, (1, 1, 4096), (4096, 4096, 1)); del buf151  # reuse
        # Source Nodes: [add_102, add_104, add_130, add_131, add_132, add_145, add_146, add_147, add_148, add_163, add_164, add_165, add_166, add_60, add_62, add_73, add_75, add_87, add_89, float_45, mean_22, mul_240, mul_241, mul_242, mul_243, mul_244, mul_265, mul_266, mul_267, mul_268, mul_293, mul_294, mul_295, mul_296, rsqrt_22, type_as_44], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_19.run(buf321, buf355, buf313, arg35_1, buf152, buf208, buf214, buf183, buf240, buf246, arg38_1, arg32_1, buf215, buf273, buf279, buf247, buf306, buf312, arg33_1, buf315, 1, 4096, grid=grid(1), stream=stream0)
        del arg32_1
        del arg33_1
        buf316 = buf218; del buf218  # reuse
        # Source Nodes: [l__model___layers_11_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (1, 4096), (0, 1), 0), reinterpret_tensor(arg153_1, (4096, 12288), (1, 4096), 0), out=buf316)
        del arg153_1
        buf324 = buf290; del buf290  # reuse
        buf322 = reinterpret_tensor(buf324, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf323 = reinterpret_tensor(buf324, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf327 = buf257; del buf257  # reuse
        buf325 = reinterpret_tensor(buf327, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf326 = reinterpret_tensor(buf327, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf358 = buf293; del buf293  # reuse
        buf356 = reinterpret_tensor(buf358, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf357 = reinterpret_tensor(buf358, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf361 = buf260; del buf260  # reuse
        buf359 = reinterpret_tensor(buf361, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf360 = reinterpret_tensor(buf361, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_20, stack_21, stack_22, stack_23], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf283, arg326_1, arg259_1, buf316, buf322, buf323, buf325, buf326, buf356, buf357, buf359, buf360, 2048, grid=grid(2048), stream=stream0)
        buf328 = buf304; del buf304  # reuse
        # Source Nodes: [setitem_20], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg281_1, buf328, 4718592, grid=grid(4718592), stream=stream0)
        del buf322
        del buf323
        del buf325
        del buf326
        buf335 = buf301; del buf301  # reuse
        # Source Nodes: [setitem_21], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg282_1, buf335, 4718592, grid=grid(4718592), stream=stream0)
        buf362 = buf268; del buf268  # reuse
        # Source Nodes: [setitem_22], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg283_1, buf362, 4718592, grid=grid(4718592), stream=stream0)
        del buf359
        del buf360
        buf369 = buf294; del buf294  # reuse
        # Source Nodes: [setitem_23], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg284_1, buf369, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_20, setitem_21, setitem_22, setitem_23], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg326_1, buf327, buf283, buf361, buf316, buf328, buf335, buf362, buf369, 4096, grid=grid(4096), stream=stream0)
        buf330 = reinterpret_tensor(buf315, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf315  # reuse
        # Source Nodes: [type_as_41], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf324, buf330, 4096, grid=grid(4096), stream=stream0)
        buf331 = buf261; del buf261  # reuse
        # Source Nodes: [setitem_20], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf328, buf331, arg281_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg281_1
        buf332 = reinterpret_tensor(buf303, (32, 1, 1152), (1152, 1152, 1)); del buf303  # reuse
        # Source Nodes: [matmul_20], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf330, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf331, (32, 128, 1152), (147456, 1, 128), 0), out=buf332)
        buf364 = buf330; del buf330  # reuse
        # Source Nodes: [type_as_45], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf358, buf364, 4096, grid=grid(4096), stream=stream0)
        del buf356
        del buf357
        buf365 = buf331; del buf331  # reuse
        # Source Nodes: [setitem_22], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf362, buf365, arg283_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg283_1
        buf366 = reinterpret_tensor(buf270, (32, 1, 1152), (1152, 1152, 1)); del buf270  # reuse
        # Source Nodes: [matmul_22], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf364, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf365, (32, 128, 1152), (147456, 1, 128), 0), out=buf366)
        buf337 = reinterpret_tensor(buf298, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf298  # reuse
        buf371 = reinterpret_tensor(buf265, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf265  # reuse
        # Source Nodes: [getitem, mul_226, mul_253, softmax_10, softmax_11, where_10, where_11], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg326_1, arg260_1, buf332, buf366, buf337, buf371, 32, 1152, grid=grid(32), stream=stream0)
        buf338 = buf365; del buf365  # reuse
        # Source Nodes: [setitem_21], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf335, buf338, arg282_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg282_1
        buf339 = reinterpret_tensor(buf364, (32, 1, 128), (128, 128, 1)); del buf364  # reuse
        # Source Nodes: [matmul_21], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf337, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf338, (32, 1152, 128), (147456, 128, 1), 0), out=buf339)
        buf340 = buf145; del buf145  # reuse
        # Source Nodes: [l__model___layers_10_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf339, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg149_1, (4096, 4096), (1, 4096), 0), out=buf340)
        del arg149_1
        buf342 = reinterpret_tensor(buf339, (1, 1, 4096), (4096, 4096, 1)); del buf339  # reuse
        # Source Nodes: [add_118, add_119, float_44, mean_21, mul_227, mul_228, mul_229, rsqrt_21, type_as_43], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf280, buf340, arg31_1, buf342, 1, 4096, grid=grid(1), stream=stream0)
        del arg31_1
        buf343 = reinterpret_tensor(buf311, (1, 11008), (11008, 1)); del buf311  # reuse
        # Source Nodes: [l__model___layers_10_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf342, (1, 4096), (0, 1), 0), reinterpret_tensor(arg150_1, (4096, 11008), (1, 4096), 0), out=buf343)
        del arg150_1
        buf344 = buf310; del buf310  # reuse
        # Source Nodes: [l__model___layers_10_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf342, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg151_1, (4096, 11008), (1, 4096), 0), out=buf344)
        del arg151_1
        buf345 = reinterpret_tensor(buf343, (1, 1, 11008), (11008, 11008, 1)); del buf343  # reuse
        # Source Nodes: [mul_230, silu_10], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf345, buf344, 11008, grid=grid(11008), stream=stream0)
        buf346 = reinterpret_tensor(buf342, (1, 4096), (4096, 1)); del buf342  # reuse
        # Source Nodes: [l__model___layers_10_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf345, (1, 11008), (0, 1), 0), reinterpret_tensor(arg152_1, (11008, 4096), (1, 11008), 0), out=buf346)
        del arg152_1
        buf347 = buf321; del buf321  # reuse
        buf349 = buf121; del buf121  # reuse
        # Source Nodes: [add_118, add_120, add_149, add_150, float_49, mean_24, mul_269, mul_270, mul_271, mul_272, rsqrt_24, type_as_48], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_20.run(buf347, arg35_1, buf280, buf340, buf346, arg36_1, buf349, 1, 4096, grid=grid(1), stream=stream0)
        del arg35_1
        del arg36_1
        buf350 = buf316; del buf316  # reuse
        # Source Nodes: [l__model___layers_12_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf349, (1, 4096), (0, 1), 0), reinterpret_tensor(arg158_1, (4096, 12288), (1, 4096), 0), out=buf350)
        del arg158_1
        buf372 = buf338; del buf338  # reuse
        # Source Nodes: [setitem_23], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf369, buf372, arg284_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg284_1
        buf373 = reinterpret_tensor(buf349, (32, 1, 128), (128, 128, 1)); del buf349  # reuse
        # Source Nodes: [matmul_23], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf371, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf372, (32, 1152, 128), (147456, 128, 1), 0), out=buf373)
        buf374 = buf120; del buf120  # reuse
        # Source Nodes: [l__model___layers_11_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf373, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg154_1, (4096, 4096), (1, 4096), 0), out=buf374)
        del arg154_1
        buf376 = reinterpret_tensor(buf373, (1, 1, 4096), (4096, 4096, 1)); del buf373  # reuse
        # Source Nodes: [add_135, add_136, float_48, mean_23, mul_254, mul_255, mul_256, rsqrt_23, type_as_47], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf313, buf374, arg34_1, buf376, 1, 4096, grid=grid(1), stream=stream0)
        del arg34_1
        buf377 = reinterpret_tensor(buf345, (1, 11008), (11008, 1)); del buf345  # reuse
        # Source Nodes: [l__model___layers_11_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf376, (1, 4096), (0, 1), 0), reinterpret_tensor(arg155_1, (4096, 11008), (1, 4096), 0), out=buf377)
        del arg155_1
        buf378 = buf344; del buf344  # reuse
        # Source Nodes: [l__model___layers_11_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf376, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg156_1, (4096, 11008), (1, 4096), 0), out=buf378)
        del arg156_1
        buf379 = reinterpret_tensor(buf377, (1, 1, 11008), (11008, 11008, 1)); del buf377  # reuse
        # Source Nodes: [mul_257, silu_11], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf379, buf378, 11008, grid=grid(11008), stream=stream0)
        buf380 = reinterpret_tensor(buf376, (1, 4096), (4096, 1)); del buf376  # reuse
        # Source Nodes: [l__model___layers_11_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf379, (1, 11008), (0, 1), 0), reinterpret_tensor(arg157_1, (11008, 4096), (1, 11008), 0), out=buf380)
        del arg157_1
        buf388 = buf387; del buf387  # reuse
        buf423 = buf422; del buf422  # reuse
        buf389 = buf388; del buf388  # reuse
        buf424 = buf423; del buf423  # reuse
        buf381 = buf355; del buf355  # reuse
        buf390 = buf389; del buf389  # reuse
        buf425 = buf424; del buf424  # reuse
        buf383 = reinterpret_tensor(buf114, (1, 1, 4096), (4096, 4096, 1)); del buf114  # reuse
        # Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_167, add_168, add_169, add_182, add_183, add_184, add_185, add_186, add_187, add_202, add_203, add_204, add_205, add_206, add_207, add_60, add_62, add_73, add_75, add_87, add_89, float_53, mean_26, mul_297, mul_298, mul_299, mul_300, mul_301, mul_322, mul_323, mul_324, mul_325, mul_326, mul_327, mul_352, mul_353, mul_354, mul_355, mul_356, mul_357, rsqrt_26, type_as_52], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_21.run(buf390, buf425, buf381, arg41_1, buf152, buf208, buf214, buf183, buf240, buf246, arg44_1, buf215, buf273, buf279, buf247, buf306, buf312, arg38_1, buf280, buf340, buf346, buf313, buf374, buf380, arg39_1, buf383, 1, 4096, grid=grid(1), stream=stream0)
        del arg38_1
        del arg39_1
        buf384 = buf283; del buf283  # reuse
        # Source Nodes: [l__model___layers_13_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf383, (1, 4096), (0, 1), 0), reinterpret_tensor(arg163_1, (4096, 12288), (1, 4096), 0), out=buf384)
        del arg163_1
        buf393 = buf358; del buf358  # reuse
        buf391 = reinterpret_tensor(buf393, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf392 = reinterpret_tensor(buf393, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf396 = buf324; del buf324  # reuse
        buf394 = reinterpret_tensor(buf396, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf395 = reinterpret_tensor(buf396, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf428 = buf361; del buf361  # reuse
        buf426 = reinterpret_tensor(buf428, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf427 = reinterpret_tensor(buf428, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf431 = buf327; del buf327  # reuse
        buf429 = reinterpret_tensor(buf431, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf430 = reinterpret_tensor(buf431, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_24, stack_25, stack_26, stack_27], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf350, arg326_1, arg259_1, buf384, buf391, buf392, buf394, buf395, buf426, buf427, buf429, buf430, 2048, grid=grid(2048), stream=stream0)
        buf397 = buf372; del buf372  # reuse
        # Source Nodes: [setitem_24], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg285_1, buf397, 4718592, grid=grid(4718592), stream=stream0)
        del buf391
        del buf392
        del buf394
        del buf395
        buf404 = buf369; del buf369  # reuse
        # Source Nodes: [setitem_25], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg286_1, buf404, 4718592, grid=grid(4718592), stream=stream0)
        buf432 = buf335; del buf335  # reuse
        # Source Nodes: [setitem_26], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg287_1, buf432, 4718592, grid=grid(4718592), stream=stream0)
        del buf429
        del buf430
        buf439 = buf362; del buf362  # reuse
        # Source Nodes: [setitem_27], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg288_1, buf439, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_24, setitem_25, setitem_26, setitem_27], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg326_1, buf396, buf350, buf431, buf384, buf397, buf404, buf432, buf439, 4096, grid=grid(4096), stream=stream0)
        buf399 = reinterpret_tensor(buf383, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf383  # reuse
        # Source Nodes: [type_as_49], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf393, buf399, 4096, grid=grid(4096), stream=stream0)
        buf400 = buf328; del buf328  # reuse
        # Source Nodes: [setitem_24], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf397, buf400, arg285_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg285_1
        buf401 = reinterpret_tensor(buf371, (32, 1, 1152), (1152, 1152, 1)); del buf371  # reuse
        # Source Nodes: [matmul_24], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf399, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf400, (32, 128, 1152), (147456, 1, 128), 0), out=buf401)
        buf434 = buf399; del buf399  # reuse
        # Source Nodes: [type_as_53], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf428, buf434, 4096, grid=grid(4096), stream=stream0)
        del buf426
        del buf427
        buf435 = buf400; del buf400  # reuse
        # Source Nodes: [setitem_26], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf432, buf435, arg287_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg287_1
        buf436 = reinterpret_tensor(buf337, (32, 1, 1152), (1152, 1152, 1)); del buf337  # reuse
        # Source Nodes: [matmul_26], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf434, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf435, (32, 128, 1152), (147456, 1, 128), 0), out=buf436)
        buf406 = reinterpret_tensor(buf366, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf366  # reuse
        buf441 = reinterpret_tensor(buf332, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf332  # reuse
        # Source Nodes: [getitem, mul_281, mul_310, softmax_12, softmax_13, where_12, where_13], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg326_1, arg260_1, buf401, buf436, buf406, buf441, 32, 1152, grid=grid(32), stream=stream0)
        buf407 = buf435; del buf435  # reuse
        # Source Nodes: [setitem_25], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf404, buf407, arg286_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg286_1
        buf408 = reinterpret_tensor(buf434, (32, 1, 128), (128, 128, 1)); del buf434  # reuse
        # Source Nodes: [matmul_25], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf406, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf407, (32, 1152, 128), (147456, 128, 1), 0), out=buf408)
        buf409 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_12_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf408, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg159_1, (4096, 4096), (1, 4096), 0), out=buf409)
        del arg159_1
        buf411 = reinterpret_tensor(buf408, (1, 1, 4096), (4096, 4096, 1)); del buf408  # reuse
        # Source Nodes: [add_153, add_154, float_52, mean_25, mul_282, mul_283, mul_284, rsqrt_25, type_as_51], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf347, buf409, arg37_1, buf411, 1, 4096, grid=grid(1), stream=stream0)
        del arg37_1
        buf412 = reinterpret_tensor(buf379, (1, 11008), (11008, 1)); del buf379  # reuse
        # Source Nodes: [l__model___layers_12_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf411, (1, 4096), (0, 1), 0), reinterpret_tensor(arg160_1, (4096, 11008), (1, 4096), 0), out=buf412)
        del arg160_1
        buf413 = buf378; del buf378  # reuse
        # Source Nodes: [l__model___layers_12_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf411, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg161_1, (4096, 11008), (1, 4096), 0), out=buf413)
        del arg161_1
        buf414 = reinterpret_tensor(buf412, (1, 1, 11008), (11008, 11008, 1)); del buf412  # reuse
        # Source Nodes: [mul_285, silu_12], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf414, buf413, 11008, grid=grid(11008), stream=stream0)
        buf415 = reinterpret_tensor(buf411, (1, 4096), (4096, 1)); del buf411  # reuse
        # Source Nodes: [l__model___layers_12_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf414, (1, 11008), (0, 1), 0), reinterpret_tensor(arg162_1, (11008, 4096), (1, 11008), 0), out=buf415)
        del arg162_1
        buf416 = buf390; del buf390  # reuse
        buf418 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_153, add_155, add_188, add_189, float_57, mean_28, mul_328, mul_329, mul_330, mul_331, rsqrt_28, type_as_56], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_22.run(buf416, arg41_1, buf347, buf409, buf415, arg42_1, buf418, 1, 4096, grid=grid(1), stream=stream0)
        del arg41_1
        del arg42_1
        buf419 = buf384; del buf384  # reuse
        # Source Nodes: [l__model___layers_14_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf418, (1, 4096), (0, 1), 0), reinterpret_tensor(arg168_1, (4096, 12288), (1, 4096), 0), out=buf419)
        del arg168_1
        buf442 = buf407; del buf407  # reuse
        # Source Nodes: [setitem_27], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf439, buf442, arg288_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg288_1
        buf443 = reinterpret_tensor(buf418, (32, 1, 128), (128, 128, 1)); del buf418  # reuse
        # Source Nodes: [matmul_27], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf441, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf442, (32, 1152, 128), (147456, 128, 1), 0), out=buf443)
        buf444 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_13_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf443, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg164_1, (4096, 4096), (1, 4096), 0), out=buf444)
        del arg164_1
        buf446 = reinterpret_tensor(buf443, (1, 1, 4096), (4096, 4096, 1)); del buf443  # reuse
        # Source Nodes: [add_172, add_173, float_56, mean_27, mul_311, mul_312, mul_313, rsqrt_27, type_as_55], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf381, buf444, arg40_1, buf446, 1, 4096, grid=grid(1), stream=stream0)
        del arg40_1
        buf447 = reinterpret_tensor(buf414, (1, 11008), (11008, 1)); del buf414  # reuse
        # Source Nodes: [l__model___layers_13_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf446, (1, 4096), (0, 1), 0), reinterpret_tensor(arg165_1, (4096, 11008), (1, 4096), 0), out=buf447)
        del arg165_1
        buf448 = buf413; del buf413  # reuse
        # Source Nodes: [l__model___layers_13_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf446, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg166_1, (4096, 11008), (1, 4096), 0), out=buf448)
        del arg166_1
        buf449 = reinterpret_tensor(buf447, (1, 1, 11008), (11008, 11008, 1)); del buf447  # reuse
        # Source Nodes: [mul_314, silu_13], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf449, buf448, 11008, grid=grid(11008), stream=stream0)
        buf450 = reinterpret_tensor(buf446, (1, 4096), (4096, 1)); del buf446  # reuse
        # Source Nodes: [l__model___layers_13_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf449, (1, 11008), (0, 1), 0), reinterpret_tensor(arg167_1, (11008, 4096), (1, 11008), 0), out=buf450)
        del arg167_1
        buf458 = buf457; del buf457  # reuse
        buf494 = buf493; del buf493  # reuse
        buf459 = buf458; del buf458  # reuse
        buf495 = buf494; del buf494  # reuse
        buf460 = buf459; del buf459  # reuse
        buf496 = buf495; del buf495  # reuse
        buf451 = buf425; del buf425  # reuse
        buf461 = buf460; del buf460  # reuse
        buf497 = buf496; del buf496  # reuse
        buf453 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_153, add_155, add_172, add_174, add_208, add_209, add_210, add_223, add_224, add_225, add_226, add_227, add_228, add_229, add_230, add_245, add_246, add_247, add_248, add_249, add_250, add_251, add_252, add_60, add_62, add_73, add_75, add_87, add_89, float_61, mean_30, mul_358, mul_359, mul_360, mul_361, mul_362, mul_383, mul_384, mul_385, mul_386, mul_387, mul_388, mul_389, mul_390, mul_415, mul_416, mul_417, mul_418, mul_419, mul_420, mul_421, mul_422, rsqrt_30, type_as_60], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_23.run(buf461, buf497, buf451, arg47_1, buf152, buf208, buf214, buf183, buf240, buf246, arg50_1, buf215, buf273, buf279, buf247, buf306, buf312, buf280, buf340, buf346, buf313, buf374, buf380, arg44_1, buf347, buf409, buf415, buf381, buf444, buf450, arg45_1, buf453, 1, 4096, grid=grid(1), stream=stream0)
        del arg44_1
        del arg45_1
        buf454 = buf350; del buf350  # reuse
        # Source Nodes: [l__model___layers_15_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf453, (1, 4096), (0, 1), 0), reinterpret_tensor(arg173_1, (4096, 12288), (1, 4096), 0), out=buf454)
        del arg173_1
        buf464 = buf428; del buf428  # reuse
        buf462 = reinterpret_tensor(buf464, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf463 = reinterpret_tensor(buf464, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf467 = buf393; del buf393  # reuse
        buf465 = reinterpret_tensor(buf467, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf466 = reinterpret_tensor(buf467, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf500 = buf431; del buf431  # reuse
        buf498 = reinterpret_tensor(buf500, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf499 = reinterpret_tensor(buf500, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf503 = buf396; del buf396  # reuse
        buf501 = reinterpret_tensor(buf503, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf502 = reinterpret_tensor(buf503, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_28, stack_29, stack_30, stack_31], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf419, arg326_1, arg259_1, buf454, buf462, buf463, buf465, buf466, buf498, buf499, buf501, buf502, 2048, grid=grid(2048), stream=stream0)
        buf468 = buf442; del buf442  # reuse
        # Source Nodes: [setitem_28], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg289_1, buf468, 4718592, grid=grid(4718592), stream=stream0)
        del buf462
        del buf463
        del buf465
        del buf466
        buf475 = buf439; del buf439  # reuse
        # Source Nodes: [setitem_29], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg290_1, buf475, 4718592, grid=grid(4718592), stream=stream0)
        buf504 = buf404; del buf404  # reuse
        # Source Nodes: [setitem_30], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg291_1, buf504, 4718592, grid=grid(4718592), stream=stream0)
        del buf501
        del buf502
        buf511 = buf432; del buf432  # reuse
        # Source Nodes: [setitem_31], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg292_1, buf511, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_28, setitem_29, setitem_30, setitem_31], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg326_1, buf467, buf419, buf503, buf454, buf468, buf475, buf504, buf511, 4096, grid=grid(4096), stream=stream0)
        buf470 = reinterpret_tensor(buf453, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf453  # reuse
        # Source Nodes: [type_as_57], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf464, buf470, 4096, grid=grid(4096), stream=stream0)
        buf471 = buf397; del buf397  # reuse
        # Source Nodes: [setitem_28], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf468, buf471, arg289_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg289_1
        buf472 = reinterpret_tensor(buf441, (32, 1, 1152), (1152, 1152, 1)); del buf441  # reuse
        # Source Nodes: [matmul_28], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf470, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf471, (32, 128, 1152), (147456, 1, 128), 0), out=buf472)
        buf506 = buf470; del buf470  # reuse
        # Source Nodes: [type_as_61], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf500, buf506, 4096, grid=grid(4096), stream=stream0)
        del buf498
        del buf499
        buf507 = buf471; del buf471  # reuse
        # Source Nodes: [setitem_30], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf504, buf507, arg291_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg291_1
        buf508 = reinterpret_tensor(buf406, (32, 1, 1152), (1152, 1152, 1)); del buf406  # reuse
        # Source Nodes: [matmul_30], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf506, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf507, (32, 128, 1152), (147456, 1, 128), 0), out=buf508)
        buf477 = reinterpret_tensor(buf436, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf436  # reuse
        buf513 = reinterpret_tensor(buf401, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf401  # reuse
        # Source Nodes: [getitem, mul_340, mul_371, softmax_14, softmax_15, where_14, where_15], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg326_1, arg260_1, buf472, buf508, buf477, buf513, 32, 1152, grid=grid(32), stream=stream0)
        buf478 = buf507; del buf507  # reuse
        # Source Nodes: [setitem_29], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf475, buf478, arg290_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg290_1
        buf479 = reinterpret_tensor(buf506, (32, 1, 128), (128, 128, 1)); del buf506  # reuse
        # Source Nodes: [matmul_29], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf477, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf478, (32, 1152, 128), (147456, 128, 1), 0), out=buf479)
        buf480 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_14_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf479, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg169_1, (4096, 4096), (1, 4096), 0), out=buf480)
        del arg169_1
        buf482 = reinterpret_tensor(buf479, (1, 1, 4096), (4096, 4096, 1)); del buf479  # reuse
        # Source Nodes: [add_192, add_193, float_60, mean_29, mul_341, mul_342, mul_343, rsqrt_29, type_as_59], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf416, buf480, arg43_1, buf482, 1, 4096, grid=grid(1), stream=stream0)
        del arg43_1
        buf483 = reinterpret_tensor(buf449, (1, 11008), (11008, 1)); del buf449  # reuse
        # Source Nodes: [l__model___layers_14_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf482, (1, 4096), (0, 1), 0), reinterpret_tensor(arg170_1, (4096, 11008), (1, 4096), 0), out=buf483)
        del arg170_1
        buf484 = buf448; del buf448  # reuse
        # Source Nodes: [l__model___layers_14_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf482, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg171_1, (4096, 11008), (1, 4096), 0), out=buf484)
        del arg171_1
        buf485 = reinterpret_tensor(buf483, (1, 1, 11008), (11008, 11008, 1)); del buf483  # reuse
        # Source Nodes: [mul_344, silu_14], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf485, buf484, 11008, grid=grid(11008), stream=stream0)
        buf486 = reinterpret_tensor(buf482, (1, 4096), (4096, 1)); del buf482  # reuse
        # Source Nodes: [l__model___layers_14_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf485, (1, 11008), (0, 1), 0), reinterpret_tensor(arg172_1, (11008, 4096), (1, 11008), 0), out=buf486)
        del arg172_1
        buf487 = buf461; del buf461  # reuse
        buf489 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_192, add_194, add_231, add_232, float_65, mean_32, mul_391, mul_392, mul_393, mul_394, rsqrt_32, type_as_64], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_24.run(buf487, arg47_1, buf416, buf480, buf486, arg48_1, buf489, 1, 4096, grid=grid(1), stream=stream0)
        del arg47_1
        del arg48_1
        buf490 = buf454; del buf454  # reuse
        # Source Nodes: [l__model___layers_16_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf489, (1, 4096), (0, 1), 0), reinterpret_tensor(arg178_1, (4096, 12288), (1, 4096), 0), out=buf490)
        del arg178_1
        buf514 = buf478; del buf478  # reuse
        # Source Nodes: [setitem_31], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf511, buf514, arg292_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg292_1
        buf515 = reinterpret_tensor(buf489, (32, 1, 128), (128, 128, 1)); del buf489  # reuse
        # Source Nodes: [matmul_31], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf513, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf514, (32, 1152, 128), (147456, 128, 1), 0), out=buf515)
        buf516 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_15_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf515, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg174_1, (4096, 4096), (1, 4096), 0), out=buf516)
        del arg174_1
        buf518 = reinterpret_tensor(buf515, (1, 1, 4096), (4096, 4096, 1)); del buf515  # reuse
        # Source Nodes: [add_213, add_214, float_64, mean_31, mul_372, mul_373, mul_374, rsqrt_31, type_as_63], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf451, buf516, arg46_1, buf518, 1, 4096, grid=grid(1), stream=stream0)
        del arg46_1
        buf519 = reinterpret_tensor(buf485, (1, 11008), (11008, 1)); del buf485  # reuse
        # Source Nodes: [l__model___layers_15_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf518, (1, 4096), (0, 1), 0), reinterpret_tensor(arg175_1, (4096, 11008), (1, 4096), 0), out=buf519)
        del arg175_1
        buf520 = buf484; del buf484  # reuse
        # Source Nodes: [l__model___layers_15_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf518, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg176_1, (4096, 11008), (1, 4096), 0), out=buf520)
        del arg176_1
        buf521 = reinterpret_tensor(buf519, (1, 1, 11008), (11008, 11008, 1)); del buf519  # reuse
        # Source Nodes: [mul_375, silu_15], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf521, buf520, 11008, grid=grid(11008), stream=stream0)
        buf522 = reinterpret_tensor(buf518, (1, 4096), (4096, 1)); del buf518  # reuse
        # Source Nodes: [l__model___layers_15_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf521, (1, 11008), (0, 1), 0), reinterpret_tensor(arg177_1, (11008, 4096), (1, 11008), 0), out=buf522)
        del arg177_1
        buf530 = buf529; del buf529  # reuse
        buf567 = buf566; del buf566  # reuse
        buf531 = buf530; del buf530  # reuse
        buf568 = buf567; del buf567  # reuse
        buf532 = buf531; del buf531  # reuse
        buf569 = buf568; del buf568  # reuse
        buf533 = buf532; del buf532  # reuse
        buf570 = buf569; del buf569  # reuse
        buf523 = buf497; del buf497  # reuse
        buf534 = buf533; del buf533  # reuse
        buf571 = buf570; del buf570  # reuse
        buf525 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_153, add_155, add_172, add_174, add_192, add_194, add_213, add_215, add_253, add_254, add_255, add_268, add_269, add_270, add_271, add_272, add_273, add_274, add_275, add_276, add_277, add_292, add_293, add_294, add_295, add_296, add_297, add_298, add_299, add_300, add_301, add_60, add_62, add_73, add_75, add_87, add_89, float_69, mean_34, mul_423, mul_424, mul_425, mul_426, mul_427, mul_448, mul_449, mul_450, mul_451, mul_452, mul_453, mul_454, mul_455, mul_456, mul_457, mul_482, mul_483, mul_484, mul_485, mul_486, mul_487, mul_488, mul_489, mul_490, mul_491, rsqrt_34, type_as_68], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_25.run(buf534, buf571, buf523, arg53_1, buf152, buf208, buf214, buf183, buf240, buf246, arg56_1, buf215, buf273, buf279, buf247, buf306, buf312, buf280, buf340, buf346, buf313, buf374, buf380, buf347, buf409, buf415, buf381, buf444, buf450, arg50_1, buf416, buf480, buf486, buf451, buf516, buf522, arg51_1, buf525, 1, 4096, grid=grid(1), stream=stream0)
        del arg50_1
        del arg51_1
        buf526 = buf419; del buf419  # reuse
        # Source Nodes: [l__model___layers_17_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf525, (1, 4096), (0, 1), 0), reinterpret_tensor(arg183_1, (4096, 12288), (1, 4096), 0), out=buf526)
        del arg183_1
        buf537 = buf500; del buf500  # reuse
        buf535 = reinterpret_tensor(buf537, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf536 = reinterpret_tensor(buf537, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf540 = buf464; del buf464  # reuse
        buf538 = reinterpret_tensor(buf540, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf539 = reinterpret_tensor(buf540, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf574 = buf503; del buf503  # reuse
        buf572 = reinterpret_tensor(buf574, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf573 = reinterpret_tensor(buf574, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf577 = buf467; del buf467  # reuse
        buf575 = reinterpret_tensor(buf577, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf576 = reinterpret_tensor(buf577, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_32, stack_33, stack_34, stack_35], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf490, arg326_1, arg259_1, buf526, buf535, buf536, buf538, buf539, buf572, buf573, buf575, buf576, 2048, grid=grid(2048), stream=stream0)
        buf541 = buf514; del buf514  # reuse
        # Source Nodes: [setitem_32], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg293_1, buf541, 4718592, grid=grid(4718592), stream=stream0)
        del buf535
        del buf536
        del buf538
        del buf539
        buf548 = buf511; del buf511  # reuse
        # Source Nodes: [setitem_33], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg294_1, buf548, 4718592, grid=grid(4718592), stream=stream0)
        buf578 = buf475; del buf475  # reuse
        # Source Nodes: [setitem_34], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg295_1, buf578, 4718592, grid=grid(4718592), stream=stream0)
        del buf575
        del buf576
        buf585 = buf504; del buf504  # reuse
        # Source Nodes: [setitem_35], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg296_1, buf585, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_32, setitem_33, setitem_34, setitem_35], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg326_1, buf540, buf490, buf577, buf526, buf541, buf548, buf578, buf585, 4096, grid=grid(4096), stream=stream0)
        buf543 = reinterpret_tensor(buf525, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf525  # reuse
        # Source Nodes: [type_as_65], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf537, buf543, 4096, grid=grid(4096), stream=stream0)
        buf544 = buf468; del buf468  # reuse
        # Source Nodes: [setitem_32], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf541, buf544, arg293_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg293_1
        buf545 = reinterpret_tensor(buf513, (32, 1, 1152), (1152, 1152, 1)); del buf513  # reuse
        # Source Nodes: [matmul_32], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf543, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf544, (32, 128, 1152), (147456, 1, 128), 0), out=buf545)
        buf580 = buf543; del buf543  # reuse
        # Source Nodes: [type_as_69], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf574, buf580, 4096, grid=grid(4096), stream=stream0)
        del buf572
        del buf573
        buf581 = buf544; del buf544  # reuse
        # Source Nodes: [setitem_34], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf578, buf581, arg295_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg295_1
        buf582 = reinterpret_tensor(buf477, (32, 1, 1152), (1152, 1152, 1)); del buf477  # reuse
        # Source Nodes: [matmul_34], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf580, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf581, (32, 128, 1152), (147456, 1, 128), 0), out=buf582)
        buf550 = reinterpret_tensor(buf508, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf508  # reuse
        buf587 = reinterpret_tensor(buf472, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf472  # reuse
        # Source Nodes: [getitem, mul_403, mul_436, softmax_16, softmax_17, where_16, where_17], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg326_1, arg260_1, buf545, buf582, buf550, buf587, 32, 1152, grid=grid(32), stream=stream0)
        buf551 = buf581; del buf581  # reuse
        # Source Nodes: [setitem_33], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf548, buf551, arg294_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg294_1
        buf552 = reinterpret_tensor(buf580, (32, 1, 128), (128, 128, 1)); del buf580  # reuse
        # Source Nodes: [matmul_33], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf550, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf551, (32, 1152, 128), (147456, 128, 1), 0), out=buf552)
        buf553 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_16_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf552, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg179_1, (4096, 4096), (1, 4096), 0), out=buf553)
        del arg179_1
        buf555 = reinterpret_tensor(buf552, (1, 1, 4096), (4096, 4096, 1)); del buf552  # reuse
        # Source Nodes: [add_235, add_236, float_68, mean_33, mul_404, mul_405, mul_406, rsqrt_33, type_as_67], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf487, buf553, arg49_1, buf555, 1, 4096, grid=grid(1), stream=stream0)
        del arg49_1
        buf556 = reinterpret_tensor(buf521, (1, 11008), (11008, 1)); del buf521  # reuse
        # Source Nodes: [l__model___layers_16_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf555, (1, 4096), (0, 1), 0), reinterpret_tensor(arg180_1, (4096, 11008), (1, 4096), 0), out=buf556)
        del arg180_1
        buf557 = buf520; del buf520  # reuse
        # Source Nodes: [l__model___layers_16_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf555, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg181_1, (4096, 11008), (1, 4096), 0), out=buf557)
        del arg181_1
        buf558 = reinterpret_tensor(buf556, (1, 1, 11008), (11008, 11008, 1)); del buf556  # reuse
        # Source Nodes: [mul_407, silu_16], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf558, buf557, 11008, grid=grid(11008), stream=stream0)
        buf559 = reinterpret_tensor(buf555, (1, 4096), (4096, 1)); del buf555  # reuse
        # Source Nodes: [l__model___layers_16_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf558, (1, 11008), (0, 1), 0), reinterpret_tensor(arg182_1, (11008, 4096), (1, 11008), 0), out=buf559)
        del arg182_1
        buf560 = buf534; del buf534  # reuse
        buf562 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_235, add_237, add_278, add_279, float_73, mean_36, mul_458, mul_459, mul_460, mul_461, rsqrt_36, type_as_72], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_26.run(buf560, arg53_1, buf487, buf553, buf559, arg54_1, buf562, 1, 4096, grid=grid(1), stream=stream0)
        del arg53_1
        del arg54_1
        buf563 = buf526; del buf526  # reuse
        # Source Nodes: [l__model___layers_18_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf562, (1, 4096), (0, 1), 0), reinterpret_tensor(arg188_1, (4096, 12288), (1, 4096), 0), out=buf563)
        del arg188_1
        buf588 = buf551; del buf551  # reuse
        # Source Nodes: [setitem_35], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf585, buf588, arg296_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg296_1
        buf589 = reinterpret_tensor(buf562, (32, 1, 128), (128, 128, 1)); del buf562  # reuse
        # Source Nodes: [matmul_35], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf587, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf588, (32, 1152, 128), (147456, 128, 1), 0), out=buf589)
        buf590 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_17_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf589, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg184_1, (4096, 4096), (1, 4096), 0), out=buf590)
        del arg184_1
        buf592 = reinterpret_tensor(buf589, (1, 1, 4096), (4096, 4096, 1)); del buf589  # reuse
        # Source Nodes: [add_258, add_259, float_72, mean_35, mul_437, mul_438, mul_439, rsqrt_35, type_as_71], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf523, buf590, arg52_1, buf592, 1, 4096, grid=grid(1), stream=stream0)
        del arg52_1
        buf593 = reinterpret_tensor(buf558, (1, 11008), (11008, 1)); del buf558  # reuse
        # Source Nodes: [l__model___layers_17_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf592, (1, 4096), (0, 1), 0), reinterpret_tensor(arg185_1, (4096, 11008), (1, 4096), 0), out=buf593)
        del arg185_1
        buf594 = buf557; del buf557  # reuse
        # Source Nodes: [l__model___layers_17_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf592, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg186_1, (4096, 11008), (1, 4096), 0), out=buf594)
        del arg186_1
        buf595 = reinterpret_tensor(buf593, (1, 1, 11008), (11008, 11008, 1)); del buf593  # reuse
        # Source Nodes: [mul_440, silu_17], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf595, buf594, 11008, grid=grid(11008), stream=stream0)
        buf596 = reinterpret_tensor(buf592, (1, 4096), (4096, 1)); del buf592  # reuse
        # Source Nodes: [l__model___layers_17_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf595, (1, 11008), (0, 1), 0), reinterpret_tensor(arg187_1, (11008, 4096), (1, 11008), 0), out=buf596)
        del arg187_1
        buf604 = buf603; del buf603  # reuse
        buf642 = buf641; del buf641  # reuse
        buf605 = buf604; del buf604  # reuse
        buf643 = buf642; del buf642  # reuse
        buf606 = buf605; del buf605  # reuse
        buf644 = buf643; del buf643  # reuse
        buf607 = buf606; del buf606  # reuse
        buf645 = buf644; del buf644  # reuse
        buf608 = buf607; del buf607  # reuse
        buf646 = buf645; del buf645  # reuse
        buf597 = buf571; del buf571  # reuse
        buf609 = buf608; del buf608  # reuse
        buf647 = buf646; del buf646  # reuse
        buf599 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_153, add_155, add_172, add_174, add_192, add_194, add_213, add_215, add_235, add_237, add_258, add_260, add_302, add_303, add_304, add_317, add_318, add_319, add_320, add_321, add_322, add_323, add_324, add_325, add_326, add_327, add_328, add_343, add_344, add_345, add_346, add_347, add_348, add_349, add_350, add_351, add_352, add_353, add_354, add_60, add_62, add_73, add_75, add_87, add_89, float_77, mean_38, mul_492, mul_493, mul_494, mul_495, mul_496, mul_517, mul_518, mul_519, mul_520, mul_521, mul_522, mul_523, mul_524, mul_525, mul_526, mul_527, mul_528, mul_553, mul_554, mul_555, mul_556, mul_557, mul_558, mul_559, mul_560, mul_561, mul_562, mul_563, mul_564, rsqrt_38, type_as_76], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_27.run(buf609, buf647, buf597, arg59_1, buf152, buf208, buf214, buf183, buf240, buf246, arg62_1, buf215, buf273, buf279, buf247, buf306, buf312, buf280, buf340, buf346, buf313, buf374, buf380, buf347, buf409, buf415, buf381, buf444, buf450, buf416, buf480, buf486, buf451, buf516, buf522, arg56_1, buf487, buf553, buf559, buf523, buf590, buf596, arg57_1, buf599, 1, 4096, grid=grid(1), stream=stream0)
        del arg56_1
        del arg57_1
        buf600 = buf490; del buf490  # reuse
        # Source Nodes: [l__model___layers_19_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf599, (1, 4096), (0, 1), 0), reinterpret_tensor(arg193_1, (4096, 12288), (1, 4096), 0), out=buf600)
        del arg193_1
        buf612 = buf574; del buf574  # reuse
        buf610 = reinterpret_tensor(buf612, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf611 = reinterpret_tensor(buf612, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf615 = buf537; del buf537  # reuse
        buf613 = reinterpret_tensor(buf615, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf614 = reinterpret_tensor(buf615, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf650 = buf577; del buf577  # reuse
        buf648 = reinterpret_tensor(buf650, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf649 = reinterpret_tensor(buf650, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf653 = buf540; del buf540  # reuse
        buf651 = reinterpret_tensor(buf653, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf652 = reinterpret_tensor(buf653, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_36, stack_37, stack_38, stack_39], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf563, arg326_1, arg259_1, buf600, buf610, buf611, buf613, buf614, buf648, buf649, buf651, buf652, 2048, grid=grid(2048), stream=stream0)
        buf616 = buf588; del buf588  # reuse
        # Source Nodes: [setitem_36], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg297_1, buf616, 4718592, grid=grid(4718592), stream=stream0)
        del buf610
        del buf611
        del buf613
        del buf614
        buf623 = buf585; del buf585  # reuse
        # Source Nodes: [setitem_37], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg298_1, buf623, 4718592, grid=grid(4718592), stream=stream0)
        buf654 = buf548; del buf548  # reuse
        # Source Nodes: [setitem_38], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg299_1, buf654, 4718592, grid=grid(4718592), stream=stream0)
        del buf651
        del buf652
        buf661 = buf578; del buf578  # reuse
        # Source Nodes: [setitem_39], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg300_1, buf661, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_36, setitem_37, setitem_38, setitem_39], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg326_1, buf615, buf563, buf653, buf600, buf616, buf623, buf654, buf661, 4096, grid=grid(4096), stream=stream0)
        buf618 = reinterpret_tensor(buf599, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf599  # reuse
        # Source Nodes: [type_as_73], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf612, buf618, 4096, grid=grid(4096), stream=stream0)
        buf619 = buf541; del buf541  # reuse
        # Source Nodes: [setitem_36], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf616, buf619, arg297_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg297_1
        buf620 = reinterpret_tensor(buf587, (32, 1, 1152), (1152, 1152, 1)); del buf587  # reuse
        # Source Nodes: [matmul_36], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf618, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf619, (32, 128, 1152), (147456, 1, 128), 0), out=buf620)
        buf656 = buf618; del buf618  # reuse
        # Source Nodes: [type_as_77], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf650, buf656, 4096, grid=grid(4096), stream=stream0)
        del buf648
        del buf649
        buf657 = buf619; del buf619  # reuse
        # Source Nodes: [setitem_38], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf654, buf657, arg299_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg299_1
        buf658 = reinterpret_tensor(buf550, (32, 1, 1152), (1152, 1152, 1)); del buf550  # reuse
        # Source Nodes: [matmul_38], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf656, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf657, (32, 128, 1152), (147456, 1, 128), 0), out=buf658)
        buf625 = reinterpret_tensor(buf582, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf582  # reuse
        buf663 = reinterpret_tensor(buf545, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf545  # reuse
        # Source Nodes: [getitem, mul_470, mul_505, softmax_18, softmax_19, where_18, where_19], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg326_1, arg260_1, buf620, buf658, buf625, buf663, 32, 1152, grid=grid(32), stream=stream0)
        buf626 = buf657; del buf657  # reuse
        # Source Nodes: [setitem_37], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf623, buf626, arg298_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg298_1
        buf627 = reinterpret_tensor(buf656, (32, 1, 128), (128, 128, 1)); del buf656  # reuse
        # Source Nodes: [matmul_37], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf625, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf626, (32, 1152, 128), (147456, 128, 1), 0), out=buf627)
        buf628 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_18_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf627, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg189_1, (4096, 4096), (1, 4096), 0), out=buf628)
        del arg189_1
        buf630 = reinterpret_tensor(buf627, (1, 1, 4096), (4096, 4096, 1)); del buf627  # reuse
        # Source Nodes: [add_282, add_283, float_76, mean_37, mul_471, mul_472, mul_473, rsqrt_37, type_as_75], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf560, buf628, arg55_1, buf630, 1, 4096, grid=grid(1), stream=stream0)
        del arg55_1
        buf631 = reinterpret_tensor(buf595, (1, 11008), (11008, 1)); del buf595  # reuse
        # Source Nodes: [l__model___layers_18_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf630, (1, 4096), (0, 1), 0), reinterpret_tensor(arg190_1, (4096, 11008), (1, 4096), 0), out=buf631)
        del arg190_1
        buf632 = buf594; del buf594  # reuse
        # Source Nodes: [l__model___layers_18_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf630, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg191_1, (4096, 11008), (1, 4096), 0), out=buf632)
        del arg191_1
        buf633 = reinterpret_tensor(buf631, (1, 1, 11008), (11008, 11008, 1)); del buf631  # reuse
        # Source Nodes: [mul_474, silu_18], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf633, buf632, 11008, grid=grid(11008), stream=stream0)
        buf634 = reinterpret_tensor(buf630, (1, 4096), (4096, 1)); del buf630  # reuse
        # Source Nodes: [l__model___layers_18_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf633, (1, 11008), (0, 1), 0), reinterpret_tensor(arg192_1, (11008, 4096), (1, 11008), 0), out=buf634)
        del arg192_1
        buf635 = buf609; del buf609  # reuse
        buf637 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_282, add_284, add_329, add_330, float_81, mean_40, mul_529, mul_530, mul_531, mul_532, rsqrt_40, type_as_80], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_28.run(buf635, arg59_1, buf560, buf628, buf634, arg60_1, buf637, 1, 4096, grid=grid(1), stream=stream0)
        del arg59_1
        del arg60_1
        buf638 = buf600; del buf600  # reuse
        # Source Nodes: [l__model___layers_20_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf637, (1, 4096), (0, 1), 0), reinterpret_tensor(arg198_1, (4096, 12288), (1, 4096), 0), out=buf638)
        del arg198_1
        buf664 = buf626; del buf626  # reuse
        # Source Nodes: [setitem_39], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf661, buf664, arg300_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg300_1
        buf665 = reinterpret_tensor(buf637, (32, 1, 128), (128, 128, 1)); del buf637  # reuse
        # Source Nodes: [matmul_39], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf663, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf664, (32, 1152, 128), (147456, 128, 1), 0), out=buf665)
        buf666 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_19_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf665, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg194_1, (4096, 4096), (1, 4096), 0), out=buf666)
        del arg194_1
        buf668 = reinterpret_tensor(buf665, (1, 1, 4096), (4096, 4096, 1)); del buf665  # reuse
        # Source Nodes: [add_307, add_308, float_80, mean_39, mul_506, mul_507, mul_508, rsqrt_39, type_as_79], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf597, buf666, arg58_1, buf668, 1, 4096, grid=grid(1), stream=stream0)
        del arg58_1
        buf669 = reinterpret_tensor(buf633, (1, 11008), (11008, 1)); del buf633  # reuse
        # Source Nodes: [l__model___layers_19_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf668, (1, 4096), (0, 1), 0), reinterpret_tensor(arg195_1, (4096, 11008), (1, 4096), 0), out=buf669)
        del arg195_1
        buf670 = buf632; del buf632  # reuse
        # Source Nodes: [l__model___layers_19_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf668, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg196_1, (4096, 11008), (1, 4096), 0), out=buf670)
        del arg196_1
        buf671 = reinterpret_tensor(buf669, (1, 1, 11008), (11008, 11008, 1)); del buf669  # reuse
        # Source Nodes: [mul_509, silu_19], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf671, buf670, 11008, grid=grid(11008), stream=stream0)
        buf672 = reinterpret_tensor(buf668, (1, 4096), (4096, 1)); del buf668  # reuse
        # Source Nodes: [l__model___layers_19_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf671, (1, 11008), (0, 1), 0), reinterpret_tensor(arg197_1, (11008, 4096), (1, 11008), 0), out=buf672)
        del arg197_1
        buf680 = buf679; del buf679  # reuse
        buf719 = buf718; del buf718  # reuse
        buf681 = buf680; del buf680  # reuse
        buf720 = buf719; del buf719  # reuse
        buf682 = buf681; del buf681  # reuse
        buf721 = buf720; del buf720  # reuse
        buf683 = buf682; del buf682  # reuse
        buf722 = buf721; del buf721  # reuse
        buf684 = buf683; del buf683  # reuse
        buf723 = buf722; del buf722  # reuse
        buf685 = buf684; del buf684  # reuse
        buf724 = buf723; del buf723  # reuse
        buf673 = buf647; del buf647  # reuse
        buf686 = buf685; del buf685  # reuse
        buf725 = buf724; del buf724  # reuse
        buf675 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_153, add_155, add_172, add_174, add_192, add_194, add_213, add_215, add_235, add_237, add_258, add_260, add_282, add_284, add_307, add_309, add_355, add_356, add_357, add_370, add_371, add_372, add_373, add_374, add_375, add_376, add_377, add_378, add_379, add_380, add_381, add_382, add_383, add_398, add_399, add_400, add_401, add_402, add_403, add_404, add_405, add_406, add_407, add_408, add_409, add_410, add_411, add_60, add_62, add_73, add_75, add_87, add_89, float_85, mean_42, mul_565, mul_566, mul_567, mul_568, mul_569, mul_590, mul_591, mul_592, mul_593, mul_594, mul_595, mul_596, mul_597, mul_598, mul_599, mul_600, mul_601, mul_602, mul_603, mul_628, mul_629, mul_630, mul_631, mul_632, mul_633, mul_634, mul_635, mul_636, mul_637, mul_638, mul_639, mul_640, mul_641, rsqrt_42, type_as_84], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_29.run(buf686, buf725, buf673, arg65_1, buf152, buf208, buf214, buf183, buf240, buf246, arg68_1, buf215, buf273, buf279, buf247, buf306, buf312, buf280, buf340, buf346, buf313, buf374, buf380, buf347, buf409, buf415, buf381, buf444, buf450, buf416, buf480, buf486, buf451, buf516, buf522, buf487, buf553, buf559, buf523, buf590, buf596, arg62_1, buf560, buf628, buf634, buf597, buf666, buf672, arg63_1, buf675, 1, 4096, grid=grid(1), stream=stream0)
        del arg62_1
        del arg63_1
        buf676 = buf563; del buf563  # reuse
        # Source Nodes: [l__model___layers_21_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf675, (1, 4096), (0, 1), 0), reinterpret_tensor(arg203_1, (4096, 12288), (1, 4096), 0), out=buf676)
        del arg203_1
        buf689 = buf650; del buf650  # reuse
        buf687 = reinterpret_tensor(buf689, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf688 = reinterpret_tensor(buf689, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf692 = buf612; del buf612  # reuse
        buf690 = reinterpret_tensor(buf692, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf691 = reinterpret_tensor(buf692, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf728 = buf653; del buf653  # reuse
        buf726 = reinterpret_tensor(buf728, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf727 = reinterpret_tensor(buf728, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf731 = buf615; del buf615  # reuse
        buf729 = reinterpret_tensor(buf731, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf730 = reinterpret_tensor(buf731, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_40, stack_41, stack_42, stack_43], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf638, arg326_1, arg259_1, buf676, buf687, buf688, buf690, buf691, buf726, buf727, buf729, buf730, 2048, grid=grid(2048), stream=stream0)
        buf693 = buf664; del buf664  # reuse
        # Source Nodes: [setitem_40], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg301_1, buf693, 4718592, grid=grid(4718592), stream=stream0)
        del buf687
        del buf688
        del buf690
        del buf691
        buf700 = buf661; del buf661  # reuse
        # Source Nodes: [setitem_41], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg302_1, buf700, 4718592, grid=grid(4718592), stream=stream0)
        buf732 = buf623; del buf623  # reuse
        # Source Nodes: [setitem_42], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg303_1, buf732, 4718592, grid=grid(4718592), stream=stream0)
        del buf729
        del buf730
        buf739 = buf654; del buf654  # reuse
        # Source Nodes: [setitem_43], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg304_1, buf739, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_40, setitem_41, setitem_42, setitem_43], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg326_1, buf692, buf638, buf731, buf676, buf693, buf700, buf732, buf739, 4096, grid=grid(4096), stream=stream0)
        buf695 = reinterpret_tensor(buf675, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf675  # reuse
        # Source Nodes: [type_as_81], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf689, buf695, 4096, grid=grid(4096), stream=stream0)
        buf696 = buf616; del buf616  # reuse
        # Source Nodes: [setitem_40], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf693, buf696, arg301_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg301_1
        buf697 = reinterpret_tensor(buf663, (32, 1, 1152), (1152, 1152, 1)); del buf663  # reuse
        # Source Nodes: [matmul_40], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf695, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf696, (32, 128, 1152), (147456, 1, 128), 0), out=buf697)
        buf734 = buf695; del buf695  # reuse
        # Source Nodes: [type_as_85], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf728, buf734, 4096, grid=grid(4096), stream=stream0)
        del buf726
        del buf727
        buf735 = buf696; del buf696  # reuse
        # Source Nodes: [setitem_42], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf732, buf735, arg303_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg303_1
        buf736 = reinterpret_tensor(buf625, (32, 1, 1152), (1152, 1152, 1)); del buf625  # reuse
        # Source Nodes: [matmul_42], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf734, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf735, (32, 128, 1152), (147456, 1, 128), 0), out=buf736)
        buf702 = reinterpret_tensor(buf658, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf658  # reuse
        buf741 = reinterpret_tensor(buf620, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf620  # reuse
        # Source Nodes: [getitem, mul_541, mul_578, softmax_20, softmax_21, where_20, where_21], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg326_1, arg260_1, buf697, buf736, buf702, buf741, 32, 1152, grid=grid(32), stream=stream0)
        buf703 = buf735; del buf735  # reuse
        # Source Nodes: [setitem_41], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf700, buf703, arg302_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg302_1
        buf704 = reinterpret_tensor(buf734, (32, 1, 128), (128, 128, 1)); del buf734  # reuse
        # Source Nodes: [matmul_41], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf702, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf703, (32, 1152, 128), (147456, 128, 1), 0), out=buf704)
        buf705 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_20_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf704, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg199_1, (4096, 4096), (1, 4096), 0), out=buf705)
        del arg199_1
        buf707 = reinterpret_tensor(buf704, (1, 1, 4096), (4096, 4096, 1)); del buf704  # reuse
        # Source Nodes: [add_333, add_334, float_84, mean_41, mul_542, mul_543, mul_544, rsqrt_41, type_as_83], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf635, buf705, arg61_1, buf707, 1, 4096, grid=grid(1), stream=stream0)
        del arg61_1
        buf708 = reinterpret_tensor(buf671, (1, 11008), (11008, 1)); del buf671  # reuse
        # Source Nodes: [l__model___layers_20_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf707, (1, 4096), (0, 1), 0), reinterpret_tensor(arg200_1, (4096, 11008), (1, 4096), 0), out=buf708)
        del arg200_1
        buf709 = buf670; del buf670  # reuse
        # Source Nodes: [l__model___layers_20_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf707, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg201_1, (4096, 11008), (1, 4096), 0), out=buf709)
        del arg201_1
        buf710 = reinterpret_tensor(buf708, (1, 1, 11008), (11008, 11008, 1)); del buf708  # reuse
        # Source Nodes: [mul_545, silu_20], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf710, buf709, 11008, grid=grid(11008), stream=stream0)
        buf711 = reinterpret_tensor(buf707, (1, 4096), (4096, 1)); del buf707  # reuse
        # Source Nodes: [l__model___layers_20_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf710, (1, 11008), (0, 1), 0), reinterpret_tensor(arg202_1, (11008, 4096), (1, 11008), 0), out=buf711)
        del arg202_1
        buf712 = buf686; del buf686  # reuse
        buf714 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_333, add_335, add_384, add_385, float_89, mean_44, mul_604, mul_605, mul_606, mul_607, rsqrt_44, type_as_88], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_30.run(buf712, arg65_1, buf635, buf705, buf711, arg66_1, buf714, 1, 4096, grid=grid(1), stream=stream0)
        del arg65_1
        del arg66_1
        buf715 = buf676; del buf676  # reuse
        # Source Nodes: [l__model___layers_22_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf714, (1, 4096), (0, 1), 0), reinterpret_tensor(arg208_1, (4096, 12288), (1, 4096), 0), out=buf715)
        del arg208_1
        buf742 = buf703; del buf703  # reuse
        # Source Nodes: [setitem_43], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf739, buf742, arg304_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg304_1
        buf743 = reinterpret_tensor(buf714, (32, 1, 128), (128, 128, 1)); del buf714  # reuse
        # Source Nodes: [matmul_43], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf741, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf742, (32, 1152, 128), (147456, 128, 1), 0), out=buf743)
        buf744 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_21_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf743, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg204_1, (4096, 4096), (1, 4096), 0), out=buf744)
        del arg204_1
        buf746 = reinterpret_tensor(buf743, (1, 1, 4096), (4096, 4096, 1)); del buf743  # reuse
        # Source Nodes: [add_360, add_361, float_88, mean_43, mul_579, mul_580, mul_581, rsqrt_43, type_as_87], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf673, buf744, arg64_1, buf746, 1, 4096, grid=grid(1), stream=stream0)
        del arg64_1
        buf747 = reinterpret_tensor(buf710, (1, 11008), (11008, 1)); del buf710  # reuse
        # Source Nodes: [l__model___layers_21_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf746, (1, 4096), (0, 1), 0), reinterpret_tensor(arg205_1, (4096, 11008), (1, 4096), 0), out=buf747)
        del arg205_1
        buf748 = buf709; del buf709  # reuse
        # Source Nodes: [l__model___layers_21_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf746, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg206_1, (4096, 11008), (1, 4096), 0), out=buf748)
        del arg206_1
        buf749 = reinterpret_tensor(buf747, (1, 1, 11008), (11008, 11008, 1)); del buf747  # reuse
        # Source Nodes: [mul_582, silu_21], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf749, buf748, 11008, grid=grid(11008), stream=stream0)
        buf750 = reinterpret_tensor(buf746, (1, 4096), (4096, 1)); del buf746  # reuse
        # Source Nodes: [l__model___layers_21_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf749, (1, 11008), (0, 1), 0), reinterpret_tensor(arg207_1, (11008, 4096), (1, 11008), 0), out=buf750)
        del arg207_1
        buf758 = buf757; del buf757  # reuse
        buf798 = buf797; del buf797  # reuse
        buf759 = buf758; del buf758  # reuse
        buf799 = buf798; del buf798  # reuse
        buf760 = buf759; del buf759  # reuse
        buf800 = buf799; del buf799  # reuse
        buf761 = buf760; del buf760  # reuse
        buf801 = buf800; del buf800  # reuse
        buf762 = buf761; del buf761  # reuse
        buf802 = buf801; del buf801  # reuse
        buf763 = buf762; del buf762  # reuse
        buf803 = buf802; del buf802  # reuse
        buf764 = buf763; del buf763  # reuse
        buf804 = buf803; del buf803  # reuse
        buf751 = buf725; del buf725  # reuse
        buf765 = buf764; del buf764  # reuse
        buf805 = buf804; del buf804  # reuse
        buf753 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_153, add_155, add_172, add_174, add_192, add_194, add_213, add_215, add_235, add_237, add_258, add_260, add_282, add_284, add_307, add_309, add_333, add_335, add_360, add_362, add_412, add_413, add_414, add_427, add_428, add_429, add_430, add_431, add_432, add_433, add_434, add_435, add_436, add_437, add_438, add_439, add_440, add_441, add_442, add_457, add_458, add_459, add_460, add_461, add_462, add_463, add_464, add_465, add_466, add_467, add_468, add_469, add_470, add_471, add_472, add_60, add_62, add_73, add_75, add_87, add_89, float_93, mean_46, mul_642, mul_643, mul_644, mul_645, mul_646, mul_667, mul_668, mul_669, mul_670, mul_671, mul_672, mul_673, mul_674, mul_675, mul_676, mul_677, mul_678, mul_679, mul_680, mul_681, mul_682, mul_707, mul_708, mul_709, mul_710, mul_711, mul_712, mul_713, mul_714, mul_715, mul_716, mul_717, mul_718, mul_719, mul_720, mul_721, mul_722, rsqrt_46, type_as_92], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_31.run(buf765, buf805, buf751, arg71_1, buf152, buf208, buf214, buf183, buf240, buf246, arg74_1, buf215, buf273, buf279, buf247, buf306, buf312, buf280, buf340, buf346, buf313, buf374, buf380, buf347, buf409, buf415, buf381, buf444, buf450, buf416, buf480, buf486, buf451, buf516, buf522, buf487, buf553, buf559, buf523, buf590, buf596, buf560, buf628, buf634, buf597, buf666, buf672, arg68_1, buf635, buf705, buf711, buf673, buf744, buf750, arg69_1, buf753, 1, 4096, grid=grid(1), stream=stream0)
        del arg68_1
        del arg69_1
        buf754 = buf638; del buf638  # reuse
        # Source Nodes: [l__model___layers_23_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf753, (1, 4096), (0, 1), 0), reinterpret_tensor(arg213_1, (4096, 12288), (1, 4096), 0), out=buf754)
        del arg213_1
        buf768 = buf728; del buf728  # reuse
        buf766 = reinterpret_tensor(buf768, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf767 = reinterpret_tensor(buf768, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf771 = buf689; del buf689  # reuse
        buf769 = reinterpret_tensor(buf771, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf770 = reinterpret_tensor(buf771, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf808 = buf731; del buf731  # reuse
        buf806 = reinterpret_tensor(buf808, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf807 = reinterpret_tensor(buf808, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf811 = buf692; del buf692  # reuse
        buf809 = reinterpret_tensor(buf811, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf810 = reinterpret_tensor(buf811, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_44, stack_45, stack_46, stack_47], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf715, arg326_1, arg259_1, buf754, buf766, buf767, buf769, buf770, buf806, buf807, buf809, buf810, 2048, grid=grid(2048), stream=stream0)
        buf772 = buf742; del buf742  # reuse
        # Source Nodes: [setitem_44], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg305_1, buf772, 4718592, grid=grid(4718592), stream=stream0)
        del buf766
        del buf767
        del buf769
        del buf770
        buf779 = buf739; del buf739  # reuse
        # Source Nodes: [setitem_45], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg306_1, buf779, 4718592, grid=grid(4718592), stream=stream0)
        buf812 = buf700; del buf700  # reuse
        # Source Nodes: [setitem_46], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg307_1, buf812, 4718592, grid=grid(4718592), stream=stream0)
        del buf809
        del buf810
        buf819 = buf732; del buf732  # reuse
        # Source Nodes: [setitem_47], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg308_1, buf819, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_44, setitem_45, setitem_46, setitem_47], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg326_1, buf771, buf715, buf811, buf754, buf772, buf779, buf812, buf819, 4096, grid=grid(4096), stream=stream0)
        buf774 = reinterpret_tensor(buf753, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf753  # reuse
        # Source Nodes: [type_as_89], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf768, buf774, 4096, grid=grid(4096), stream=stream0)
        buf775 = buf693; del buf693  # reuse
        # Source Nodes: [setitem_44], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf772, buf775, arg305_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg305_1
        buf776 = reinterpret_tensor(buf741, (32, 1, 1152), (1152, 1152, 1)); del buf741  # reuse
        # Source Nodes: [matmul_44], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf774, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf775, (32, 128, 1152), (147456, 1, 128), 0), out=buf776)
        buf814 = buf774; del buf774  # reuse
        # Source Nodes: [type_as_93], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf808, buf814, 4096, grid=grid(4096), stream=stream0)
        del buf806
        del buf807
        buf815 = buf775; del buf775  # reuse
        # Source Nodes: [setitem_46], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf812, buf815, arg307_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg307_1
        buf816 = reinterpret_tensor(buf702, (32, 1, 1152), (1152, 1152, 1)); del buf702  # reuse
        # Source Nodes: [matmul_46], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf814, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf815, (32, 128, 1152), (147456, 1, 128), 0), out=buf816)
        buf781 = reinterpret_tensor(buf736, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf736  # reuse
        buf821 = reinterpret_tensor(buf697, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf697  # reuse
        # Source Nodes: [getitem, mul_616, mul_655, softmax_22, softmax_23, where_22, where_23], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg326_1, arg260_1, buf776, buf816, buf781, buf821, 32, 1152, grid=grid(32), stream=stream0)
        buf782 = buf815; del buf815  # reuse
        # Source Nodes: [setitem_45], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf779, buf782, arg306_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg306_1
        buf783 = reinterpret_tensor(buf814, (32, 1, 128), (128, 128, 1)); del buf814  # reuse
        # Source Nodes: [matmul_45], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf781, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf782, (32, 1152, 128), (147456, 128, 1), 0), out=buf783)
        buf784 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_22_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf783, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg209_1, (4096, 4096), (1, 4096), 0), out=buf784)
        del arg209_1
        buf786 = reinterpret_tensor(buf783, (1, 1, 4096), (4096, 4096, 1)); del buf783  # reuse
        # Source Nodes: [add_388, add_389, float_92, mean_45, mul_617, mul_618, mul_619, rsqrt_45, type_as_91], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf712, buf784, arg67_1, buf786, 1, 4096, grid=grid(1), stream=stream0)
        del arg67_1
        buf787 = reinterpret_tensor(buf749, (1, 11008), (11008, 1)); del buf749  # reuse
        # Source Nodes: [l__model___layers_22_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf786, (1, 4096), (0, 1), 0), reinterpret_tensor(arg210_1, (4096, 11008), (1, 4096), 0), out=buf787)
        del arg210_1
        buf788 = buf748; del buf748  # reuse
        # Source Nodes: [l__model___layers_22_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf786, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg211_1, (4096, 11008), (1, 4096), 0), out=buf788)
        del arg211_1
        buf789 = reinterpret_tensor(buf787, (1, 1, 11008), (11008, 11008, 1)); del buf787  # reuse
        # Source Nodes: [mul_620, silu_22], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf789, buf788, 11008, grid=grid(11008), stream=stream0)
        buf790 = reinterpret_tensor(buf786, (1, 4096), (4096, 1)); del buf786  # reuse
        # Source Nodes: [l__model___layers_22_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf789, (1, 11008), (0, 1), 0), reinterpret_tensor(arg212_1, (11008, 4096), (1, 11008), 0), out=buf790)
        del arg212_1
        buf791 = buf765; del buf765  # reuse
        buf793 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_388, add_390, add_443, add_444, float_97, mean_48, mul_683, mul_684, mul_685, mul_686, rsqrt_48, type_as_96], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_32.run(buf791, arg71_1, buf712, buf784, buf790, arg72_1, buf793, 1, 4096, grid=grid(1), stream=stream0)
        del arg71_1
        del arg72_1
        buf794 = buf754; del buf754  # reuse
        # Source Nodes: [l__model___layers_24_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf793, (1, 4096), (0, 1), 0), reinterpret_tensor(arg218_1, (4096, 12288), (1, 4096), 0), out=buf794)
        del arg218_1
        buf822 = buf782; del buf782  # reuse
        # Source Nodes: [setitem_47], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf819, buf822, arg308_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg308_1
        buf823 = reinterpret_tensor(buf793, (32, 1, 128), (128, 128, 1)); del buf793  # reuse
        # Source Nodes: [matmul_47], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf821, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf822, (32, 1152, 128), (147456, 128, 1), 0), out=buf823)
        buf824 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_23_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf823, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg214_1, (4096, 4096), (1, 4096), 0), out=buf824)
        del arg214_1
        buf826 = reinterpret_tensor(buf823, (1, 1, 4096), (4096, 4096, 1)); del buf823  # reuse
        # Source Nodes: [add_417, add_418, float_96, mean_47, mul_656, mul_657, mul_658, rsqrt_47, type_as_95], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf751, buf824, arg70_1, buf826, 1, 4096, grid=grid(1), stream=stream0)
        del arg70_1
        buf827 = reinterpret_tensor(buf789, (1, 11008), (11008, 1)); del buf789  # reuse
        # Source Nodes: [l__model___layers_23_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf826, (1, 4096), (0, 1), 0), reinterpret_tensor(arg215_1, (4096, 11008), (1, 4096), 0), out=buf827)
        del arg215_1
        buf828 = buf788; del buf788  # reuse
        # Source Nodes: [l__model___layers_23_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf826, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg216_1, (4096, 11008), (1, 4096), 0), out=buf828)
        del arg216_1
        buf829 = reinterpret_tensor(buf827, (1, 1, 11008), (11008, 11008, 1)); del buf827  # reuse
        # Source Nodes: [mul_659, silu_23], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf829, buf828, 11008, grid=grid(11008), stream=stream0)
        buf830 = reinterpret_tensor(buf826, (1, 4096), (4096, 1)); del buf826  # reuse
        # Source Nodes: [l__model___layers_23_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf829, (1, 11008), (0, 1), 0), reinterpret_tensor(arg217_1, (11008, 4096), (1, 11008), 0), out=buf830)
        del arg217_1
        buf838 = buf837; del buf837  # reuse
        buf879 = buf878; del buf878  # reuse
        buf839 = buf838; del buf838  # reuse
        buf880 = buf879; del buf879  # reuse
        buf840 = buf839; del buf839  # reuse
        buf881 = buf880; del buf880  # reuse
        buf841 = buf840; del buf840  # reuse
        buf882 = buf881; del buf881  # reuse
        buf842 = buf841; del buf841  # reuse
        buf883 = buf882; del buf882  # reuse
        buf843 = buf842; del buf842  # reuse
        buf884 = buf883; del buf883  # reuse
        buf844 = buf843; del buf843  # reuse
        buf885 = buf884; del buf884  # reuse
        buf845 = buf844; del buf844  # reuse
        buf886 = buf885; del buf885  # reuse
        buf831 = buf805; del buf805  # reuse
        buf846 = buf845; del buf845  # reuse
        buf887 = buf886; del buf886  # reuse
        buf833 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_153, add_155, add_172, add_174, add_192, add_194, add_213, add_215, add_235, add_237, add_258, add_260, add_282, add_284, add_307, add_309, add_333, add_335, add_360, add_362, add_388, add_390, add_417, add_419, add_473, add_474, add_475, add_488, add_489, add_490, add_491, add_492, add_493, add_494, add_495, add_496, add_497, add_498, add_499, add_500, add_501, add_502, add_503, add_504, add_505, add_520, add_521, add_522, add_523, add_524, add_525, add_526, add_527, add_528, add_529, add_530, add_531, add_532, add_533, add_534, add_535, add_536, add_537, add_60, add_62, add_73, add_75, add_87, add_89, float_101, mean_50, mul_723, mul_724, mul_725, mul_726, mul_727, mul_748, mul_749, mul_750, mul_751, mul_752, mul_753, mul_754, mul_755, mul_756, mul_757, mul_758, mul_759, mul_760, mul_761, mul_762, mul_763, mul_764, mul_765, mul_790, mul_791, mul_792, mul_793, mul_794, mul_795, mul_796, mul_797, mul_798, mul_799, mul_800, mul_801, mul_802, mul_803, mul_804, mul_805, mul_806, mul_807, rsqrt_50, type_as_100], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_33.run(buf846, buf887, buf831, arg77_1, buf152, buf208, buf214, buf183, buf240, buf246, arg80_1, buf215, buf273, buf279, buf247, buf306, buf312, buf280, buf340, buf346, buf313, buf374, buf380, buf347, buf409, buf415, buf381, buf444, buf450, buf416, buf480, buf486, buf451, buf516, buf522, buf487, buf553, buf559, buf523, buf590, buf596, buf560, buf628, buf634, buf597, buf666, buf672, buf635, buf705, buf711, buf673, buf744, buf750, arg74_1, buf712, buf784, buf790, buf751, buf824, buf830, arg75_1, buf833, 1, 4096, grid=grid(1), stream=stream0)
        del arg74_1
        del arg75_1
        buf834 = buf715; del buf715  # reuse
        # Source Nodes: [l__model___layers_25_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf833, (1, 4096), (0, 1), 0), reinterpret_tensor(arg223_1, (4096, 12288), (1, 4096), 0), out=buf834)
        del arg223_1
        buf849 = buf808; del buf808  # reuse
        buf847 = reinterpret_tensor(buf849, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf848 = reinterpret_tensor(buf849, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf852 = buf768; del buf768  # reuse
        buf850 = reinterpret_tensor(buf852, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf851 = reinterpret_tensor(buf852, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf890 = buf811; del buf811  # reuse
        buf888 = reinterpret_tensor(buf890, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf889 = reinterpret_tensor(buf890, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf893 = buf771; del buf771  # reuse
        buf891 = reinterpret_tensor(buf893, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf892 = reinterpret_tensor(buf893, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_48, stack_49, stack_50, stack_51], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf794, arg326_1, arg259_1, buf834, buf847, buf848, buf850, buf851, buf888, buf889, buf891, buf892, 2048, grid=grid(2048), stream=stream0)
        buf853 = buf822; del buf822  # reuse
        # Source Nodes: [setitem_48], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg309_1, buf853, 4718592, grid=grid(4718592), stream=stream0)
        del buf847
        del buf848
        del buf850
        del buf851
        buf860 = buf819; del buf819  # reuse
        # Source Nodes: [setitem_49], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg310_1, buf860, 4718592, grid=grid(4718592), stream=stream0)
        buf894 = buf779; del buf779  # reuse
        # Source Nodes: [setitem_50], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg311_1, buf894, 4718592, grid=grid(4718592), stream=stream0)
        del buf891
        del buf892
        buf901 = buf812; del buf812  # reuse
        # Source Nodes: [setitem_51], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg312_1, buf901, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_48, setitem_49, setitem_50, setitem_51], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg326_1, buf852, buf794, buf893, buf834, buf853, buf860, buf894, buf901, 4096, grid=grid(4096), stream=stream0)
        buf855 = reinterpret_tensor(buf833, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf833  # reuse
        # Source Nodes: [type_as_97], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf849, buf855, 4096, grid=grid(4096), stream=stream0)
        buf856 = buf772; del buf772  # reuse
        # Source Nodes: [setitem_48], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf853, buf856, arg309_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg309_1
        buf857 = reinterpret_tensor(buf821, (32, 1, 1152), (1152, 1152, 1)); del buf821  # reuse
        # Source Nodes: [matmul_48], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf855, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf856, (32, 128, 1152), (147456, 1, 128), 0), out=buf857)
        buf896 = buf855; del buf855  # reuse
        # Source Nodes: [type_as_101], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf890, buf896, 4096, grid=grid(4096), stream=stream0)
        del buf888
        del buf889
        buf897 = buf856; del buf856  # reuse
        # Source Nodes: [setitem_50], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf894, buf897, arg311_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg311_1
        buf898 = reinterpret_tensor(buf781, (32, 1, 1152), (1152, 1152, 1)); del buf781  # reuse
        # Source Nodes: [matmul_50], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf896, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf897, (32, 128, 1152), (147456, 1, 128), 0), out=buf898)
        buf862 = reinterpret_tensor(buf816, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf816  # reuse
        buf903 = reinterpret_tensor(buf776, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf776  # reuse
        # Source Nodes: [getitem, mul_695, mul_736, softmax_24, softmax_25, where_24, where_25], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg326_1, arg260_1, buf857, buf898, buf862, buf903, 32, 1152, grid=grid(32), stream=stream0)
        buf863 = buf897; del buf897  # reuse
        # Source Nodes: [setitem_49], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf860, buf863, arg310_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg310_1
        buf864 = reinterpret_tensor(buf896, (32, 1, 128), (128, 128, 1)); del buf896  # reuse
        # Source Nodes: [matmul_49], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf862, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf863, (32, 1152, 128), (147456, 128, 1), 0), out=buf864)
        buf865 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_24_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf864, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg219_1, (4096, 4096), (1, 4096), 0), out=buf865)
        del arg219_1
        buf867 = reinterpret_tensor(buf864, (1, 1, 4096), (4096, 4096, 1)); del buf864  # reuse
        # Source Nodes: [add_447, add_448, float_100, mean_49, mul_696, mul_697, mul_698, rsqrt_49, type_as_99], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf791, buf865, arg73_1, buf867, 1, 4096, grid=grid(1), stream=stream0)
        del arg73_1
        buf868 = reinterpret_tensor(buf829, (1, 11008), (11008, 1)); del buf829  # reuse
        # Source Nodes: [l__model___layers_24_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf867, (1, 4096), (0, 1), 0), reinterpret_tensor(arg220_1, (4096, 11008), (1, 4096), 0), out=buf868)
        del arg220_1
        buf869 = buf828; del buf828  # reuse
        # Source Nodes: [l__model___layers_24_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf867, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg221_1, (4096, 11008), (1, 4096), 0), out=buf869)
        del arg221_1
        buf870 = reinterpret_tensor(buf868, (1, 1, 11008), (11008, 11008, 1)); del buf868  # reuse
        # Source Nodes: [mul_699, silu_24], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf870, buf869, 11008, grid=grid(11008), stream=stream0)
        buf871 = reinterpret_tensor(buf867, (1, 4096), (4096, 1)); del buf867  # reuse
        # Source Nodes: [l__model___layers_24_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf870, (1, 11008), (0, 1), 0), reinterpret_tensor(arg222_1, (11008, 4096), (1, 11008), 0), out=buf871)
        del arg222_1
        buf872 = buf846; del buf846  # reuse
        buf874 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_447, add_449, add_506, add_507, float_105, mean_52, mul_766, mul_767, mul_768, mul_769, rsqrt_52, type_as_104], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_34.run(buf872, arg77_1, buf791, buf865, buf871, arg78_1, buf874, 1, 4096, grid=grid(1), stream=stream0)
        del arg77_1
        del arg78_1
        buf875 = buf834; del buf834  # reuse
        # Source Nodes: [l__model___layers_26_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf874, (1, 4096), (0, 1), 0), reinterpret_tensor(arg228_1, (4096, 12288), (1, 4096), 0), out=buf875)
        del arg228_1
        buf904 = buf863; del buf863  # reuse
        # Source Nodes: [setitem_51], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf901, buf904, arg312_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg312_1
        buf905 = reinterpret_tensor(buf874, (32, 1, 128), (128, 128, 1)); del buf874  # reuse
        # Source Nodes: [matmul_51], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf903, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf904, (32, 1152, 128), (147456, 128, 1), 0), out=buf905)
        buf906 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_25_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf905, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg224_1, (4096, 4096), (1, 4096), 0), out=buf906)
        del arg224_1
        buf908 = reinterpret_tensor(buf905, (1, 1, 4096), (4096, 4096, 1)); del buf905  # reuse
        # Source Nodes: [add_478, add_479, float_104, mean_51, mul_737, mul_738, mul_739, rsqrt_51, type_as_103], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf831, buf906, arg76_1, buf908, 1, 4096, grid=grid(1), stream=stream0)
        del arg76_1
        buf909 = reinterpret_tensor(buf870, (1, 11008), (11008, 1)); del buf870  # reuse
        # Source Nodes: [l__model___layers_25_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf908, (1, 4096), (0, 1), 0), reinterpret_tensor(arg225_1, (4096, 11008), (1, 4096), 0), out=buf909)
        del arg225_1
        buf910 = buf869; del buf869  # reuse
        # Source Nodes: [l__model___layers_25_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf908, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg226_1, (4096, 11008), (1, 4096), 0), out=buf910)
        del arg226_1
        buf911 = reinterpret_tensor(buf909, (1, 1, 11008), (11008, 11008, 1)); del buf909  # reuse
        # Source Nodes: [mul_740, silu_25], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf911, buf910, 11008, grid=grid(11008), stream=stream0)
        buf912 = reinterpret_tensor(buf908, (1, 4096), (4096, 1)); del buf908  # reuse
        # Source Nodes: [l__model___layers_25_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf911, (1, 11008), (0, 1), 0), reinterpret_tensor(arg227_1, (11008, 4096), (1, 11008), 0), out=buf912)
        del arg227_1
        buf920 = buf919; del buf919  # reuse
        buf962 = buf961; del buf961  # reuse
        buf921 = buf920; del buf920  # reuse
        buf963 = buf962; del buf962  # reuse
        buf922 = buf921; del buf921  # reuse
        buf964 = buf963; del buf963  # reuse
        buf923 = buf922; del buf922  # reuse
        buf965 = buf964; del buf964  # reuse
        buf924 = buf923; del buf923  # reuse
        buf966 = buf965; del buf965  # reuse
        buf925 = buf924; del buf924  # reuse
        buf967 = buf966; del buf966  # reuse
        buf926 = buf925; del buf925  # reuse
        buf968 = buf967; del buf967  # reuse
        buf927 = buf926; del buf926  # reuse
        buf969 = buf968; del buf968  # reuse
        buf928 = buf927; del buf927  # reuse
        buf970 = buf969; del buf969  # reuse
        buf913 = buf887; del buf887  # reuse
        buf929 = buf928; del buf928  # reuse
        buf971 = buf970; del buf970  # reuse
        buf915 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_153, add_155, add_172, add_174, add_192, add_194, add_213, add_215, add_235, add_237, add_258, add_260, add_282, add_284, add_307, add_309, add_333, add_335, add_360, add_362, add_388, add_390, add_417, add_419, add_447, add_449, add_478, add_480, add_538, add_539, add_540, add_553, add_554, add_555, add_556, add_557, add_558, add_559, add_560, add_561, add_562, add_563, add_564, add_565, add_566, add_567, add_568, add_569, add_570, add_571, add_572, add_587, add_588, add_589, add_590, add_591, add_592, add_593, add_594, add_595, add_596, add_597, add_598, add_599, add_60, add_600, add_601, add_602, add_603, add_604, add_605, add_606, add_62, add_73, add_75, add_87, add_89, float_109, mean_54, mul_808, mul_809, mul_810, mul_811, mul_812, mul_833, mul_834, mul_835, mul_836, mul_837, mul_838, mul_839, mul_840, mul_841, mul_842, mul_843, mul_844, mul_845, mul_846, mul_847, mul_848, mul_849, mul_850, mul_851, mul_852, mul_877, mul_878, mul_879, mul_880, mul_881, mul_882, mul_883, mul_884, mul_885, mul_886, mul_887, mul_888, mul_889, mul_890, mul_891, mul_892, mul_893, mul_894, mul_895, mul_896, rsqrt_54, type_as_108], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_35.run(buf929, buf971, buf913, arg83_1, buf152, buf208, buf214, buf183, buf240, buf246, arg86_1, buf215, buf273, buf279, buf247, buf306, buf312, buf280, buf340, buf346, buf313, buf374, buf380, buf347, buf409, buf415, buf381, buf444, buf450, buf416, buf480, buf486, buf451, buf516, buf522, buf487, buf553, buf559, buf523, buf590, buf596, buf560, buf628, buf634, buf597, buf666, buf672, buf635, buf705, buf711, buf673, buf744, buf750, buf712, buf784, buf790, buf751, buf824, buf830, arg80_1, buf791, buf865, buf871, buf831, buf906, buf912, arg81_1, buf915, 1, 4096, grid=grid(1), stream=stream0)
        del arg80_1
        del arg81_1
        buf916 = buf794; del buf794  # reuse
        # Source Nodes: [l__model___layers_27_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf915, (1, 4096), (0, 1), 0), reinterpret_tensor(arg233_1, (4096, 12288), (1, 4096), 0), out=buf916)
        del arg233_1
        buf932 = buf890; del buf890  # reuse
        buf930 = reinterpret_tensor(buf932, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf931 = reinterpret_tensor(buf932, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf935 = buf849; del buf849  # reuse
        buf933 = reinterpret_tensor(buf935, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf934 = reinterpret_tensor(buf935, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf974 = buf893; del buf893  # reuse
        buf972 = reinterpret_tensor(buf974, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf973 = reinterpret_tensor(buf974, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf977 = buf852; del buf852  # reuse
        buf975 = reinterpret_tensor(buf977, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf976 = reinterpret_tensor(buf977, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_52, stack_53, stack_54, stack_55], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf875, arg326_1, arg259_1, buf916, buf930, buf931, buf933, buf934, buf972, buf973, buf975, buf976, 2048, grid=grid(2048), stream=stream0)
        buf936 = buf904; del buf904  # reuse
        # Source Nodes: [setitem_52], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg313_1, buf936, 4718592, grid=grid(4718592), stream=stream0)
        del buf930
        del buf931
        del buf933
        del buf934
        buf943 = buf901; del buf901  # reuse
        # Source Nodes: [setitem_53], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg314_1, buf943, 4718592, grid=grid(4718592), stream=stream0)
        buf978 = buf860; del buf860  # reuse
        # Source Nodes: [setitem_54], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg315_1, buf978, 4718592, grid=grid(4718592), stream=stream0)
        del buf975
        del buf976
        buf985 = buf894; del buf894  # reuse
        # Source Nodes: [setitem_55], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg316_1, buf985, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_52, setitem_53, setitem_54, setitem_55], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg326_1, buf935, buf875, buf977, buf916, buf936, buf943, buf978, buf985, 4096, grid=grid(4096), stream=stream0)
        buf938 = reinterpret_tensor(buf915, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf915  # reuse
        # Source Nodes: [type_as_105], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf932, buf938, 4096, grid=grid(4096), stream=stream0)
        buf939 = buf853; del buf853  # reuse
        # Source Nodes: [setitem_52], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf936, buf939, arg313_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg313_1
        buf940 = reinterpret_tensor(buf903, (32, 1, 1152), (1152, 1152, 1)); del buf903  # reuse
        # Source Nodes: [matmul_52], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf938, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf939, (32, 128, 1152), (147456, 1, 128), 0), out=buf940)
        buf980 = buf938; del buf938  # reuse
        # Source Nodes: [type_as_109], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf974, buf980, 4096, grid=grid(4096), stream=stream0)
        del buf972
        del buf973
        buf981 = buf939; del buf939  # reuse
        # Source Nodes: [setitem_54], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf978, buf981, arg315_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg315_1
        buf982 = reinterpret_tensor(buf862, (32, 1, 1152), (1152, 1152, 1)); del buf862  # reuse
        # Source Nodes: [matmul_54], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf980, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf981, (32, 128, 1152), (147456, 1, 128), 0), out=buf982)
        buf945 = reinterpret_tensor(buf898, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf898  # reuse
        buf987 = reinterpret_tensor(buf857, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf857  # reuse
        # Source Nodes: [getitem, mul_778, mul_821, softmax_26, softmax_27, where_26, where_27], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg326_1, arg260_1, buf940, buf982, buf945, buf987, 32, 1152, grid=grid(32), stream=stream0)
        buf946 = buf981; del buf981  # reuse
        # Source Nodes: [setitem_53], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf943, buf946, arg314_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg314_1
        buf947 = reinterpret_tensor(buf980, (32, 1, 128), (128, 128, 1)); del buf980  # reuse
        # Source Nodes: [matmul_53], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf945, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf946, (32, 1152, 128), (147456, 128, 1), 0), out=buf947)
        buf948 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_26_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf947, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg229_1, (4096, 4096), (1, 4096), 0), out=buf948)
        del arg229_1
        buf950 = reinterpret_tensor(buf947, (1, 1, 4096), (4096, 4096, 1)); del buf947  # reuse
        # Source Nodes: [add_510, add_511, float_108, mean_53, mul_779, mul_780, mul_781, rsqrt_53, type_as_107], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf872, buf948, arg79_1, buf950, 1, 4096, grid=grid(1), stream=stream0)
        del arg79_1
        buf951 = reinterpret_tensor(buf911, (1, 11008), (11008, 1)); del buf911  # reuse
        # Source Nodes: [l__model___layers_26_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf950, (1, 4096), (0, 1), 0), reinterpret_tensor(arg230_1, (4096, 11008), (1, 4096), 0), out=buf951)
        del arg230_1
        buf952 = buf910; del buf910  # reuse
        # Source Nodes: [l__model___layers_26_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf950, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg231_1, (4096, 11008), (1, 4096), 0), out=buf952)
        del arg231_1
        buf953 = reinterpret_tensor(buf951, (1, 1, 11008), (11008, 11008, 1)); del buf951  # reuse
        # Source Nodes: [mul_782, silu_26], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf953, buf952, 11008, grid=grid(11008), stream=stream0)
        buf954 = reinterpret_tensor(buf950, (1, 4096), (4096, 1)); del buf950  # reuse
        # Source Nodes: [l__model___layers_26_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf953, (1, 11008), (0, 1), 0), reinterpret_tensor(arg232_1, (11008, 4096), (1, 11008), 0), out=buf954)
        del arg232_1
        buf955 = buf929; del buf929  # reuse
        buf957 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_510, add_512, add_573, add_574, float_113, mean_56, mul_853, mul_854, mul_855, mul_856, rsqrt_56, type_as_112], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_36.run(buf955, arg83_1, buf872, buf948, buf954, arg84_1, buf957, 1, 4096, grid=grid(1), stream=stream0)
        del arg83_1
        del arg84_1
        buf958 = buf916; del buf916  # reuse
        # Source Nodes: [l__model___layers_28_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf957, (1, 4096), (0, 1), 0), reinterpret_tensor(arg238_1, (4096, 12288), (1, 4096), 0), out=buf958)
        del arg238_1
        buf988 = buf946; del buf946  # reuse
        # Source Nodes: [setitem_55], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf985, buf988, arg316_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg316_1
        buf989 = reinterpret_tensor(buf957, (32, 1, 128), (128, 128, 1)); del buf957  # reuse
        # Source Nodes: [matmul_55], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf987, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf988, (32, 1152, 128), (147456, 128, 1), 0), out=buf989)
        buf990 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_27_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf989, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg234_1, (4096, 4096), (1, 4096), 0), out=buf990)
        del arg234_1
        buf992 = reinterpret_tensor(buf989, (1, 1, 4096), (4096, 4096, 1)); del buf989  # reuse
        # Source Nodes: [add_543, add_544, float_112, mean_55, mul_822, mul_823, mul_824, rsqrt_55, type_as_111], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf913, buf990, arg82_1, buf992, 1, 4096, grid=grid(1), stream=stream0)
        del arg82_1
        buf993 = reinterpret_tensor(buf953, (1, 11008), (11008, 1)); del buf953  # reuse
        # Source Nodes: [l__model___layers_27_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf992, (1, 4096), (0, 1), 0), reinterpret_tensor(arg235_1, (4096, 11008), (1, 4096), 0), out=buf993)
        del arg235_1
        buf994 = buf952; del buf952  # reuse
        # Source Nodes: [l__model___layers_27_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf992, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg236_1, (4096, 11008), (1, 4096), 0), out=buf994)
        del arg236_1
        buf995 = reinterpret_tensor(buf993, (1, 1, 11008), (11008, 11008, 1)); del buf993  # reuse
        # Source Nodes: [mul_825, silu_27], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf995, buf994, 11008, grid=grid(11008), stream=stream0)
        buf996 = reinterpret_tensor(buf992, (1, 4096), (4096, 1)); del buf992  # reuse
        # Source Nodes: [l__model___layers_27_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf995, (1, 11008), (0, 1), 0), reinterpret_tensor(arg237_1, (11008, 4096), (1, 11008), 0), out=buf996)
        del arg237_1
        buf1004 = buf1003; del buf1003  # reuse
        buf1047 = buf1046; del buf1046  # reuse
        buf1005 = buf1004; del buf1004  # reuse
        buf1048 = buf1047; del buf1047  # reuse
        buf1006 = buf1005; del buf1005  # reuse
        buf1049 = buf1048; del buf1048  # reuse
        buf1007 = buf1006; del buf1006  # reuse
        buf1050 = buf1049; del buf1049  # reuse
        buf1008 = buf1007; del buf1007  # reuse
        buf1051 = buf1050; del buf1050  # reuse
        buf1009 = buf1008; del buf1008  # reuse
        buf1052 = buf1051; del buf1051  # reuse
        buf1010 = buf1009; del buf1009  # reuse
        buf1053 = buf1052; del buf1052  # reuse
        buf1011 = buf1010; del buf1010  # reuse
        buf1054 = buf1053; del buf1053  # reuse
        buf1012 = buf1011; del buf1011  # reuse
        buf1055 = buf1054; del buf1054  # reuse
        buf1013 = buf1012; del buf1012  # reuse
        buf1056 = buf1055; del buf1055  # reuse
        buf997 = buf971; del buf971  # reuse
        buf1014 = buf1013; del buf1013  # reuse
        buf1057 = buf1056; del buf1056  # reuse
        buf999 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_153, add_155, add_172, add_174, add_192, add_194, add_213, add_215, add_235, add_237, add_258, add_260, add_282, add_284, add_307, add_309, add_333, add_335, add_360, add_362, add_388, add_390, add_417, add_419, add_447, add_449, add_478, add_480, add_510, add_512, add_543, add_545, add_60, add_607, add_608, add_609, add_62, add_622, add_623, add_624, add_625, add_626, add_627, add_628, add_629, add_630, add_631, add_632, add_633, add_634, add_635, add_636, add_637, add_638, add_639, add_640, add_641, add_642, add_643, add_658, add_659, add_660, add_661, add_662, add_663, add_664, add_665, add_666, add_667, add_668, add_669, add_670, add_671, add_672, add_673, add_674, add_675, add_676, add_677, add_678, add_679, add_73, add_75, add_87, add_89, float_117, mean_58, mul_897, mul_898, mul_899, mul_900, mul_901, mul_922, mul_923, mul_924, mul_925, mul_926, mul_927, mul_928, mul_929, mul_930, mul_931, mul_932, mul_933, mul_934, mul_935, mul_936, mul_937, mul_938, mul_939, mul_940, mul_941, mul_942, mul_943, mul_968, mul_969, mul_970, mul_971, mul_972, mul_973, mul_974, mul_975, mul_976, mul_977, mul_978, mul_979, mul_980, mul_981, mul_982, mul_983, mul_984, mul_985, mul_986, mul_987, mul_988, mul_989, rsqrt_58, type_as_116], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_37.run(buf1014, buf1057, buf997, arg89_1, buf152, buf208, buf214, buf183, buf240, buf246, arg92_1, buf215, buf273, buf279, buf247, buf306, buf312, buf280, buf340, buf346, buf313, buf374, buf380, buf347, buf409, buf415, buf381, buf444, buf450, buf416, buf480, buf486, buf451, buf516, buf522, buf487, buf553, buf559, buf523, buf590, buf596, buf560, buf628, buf634, buf597, buf666, buf672, buf635, buf705, buf711, buf673, buf744, buf750, buf712, buf784, buf790, buf751, buf824, buf830, buf791, buf865, buf871, buf831, buf906, buf912, arg86_1, buf872, buf948, buf954, buf913, buf990, buf996, arg87_1, buf999, 1, 4096, grid=grid(1), stream=stream0)
        del arg86_1
        del arg87_1
        buf1000 = buf875; del buf875  # reuse
        # Source Nodes: [l__model___layers_29_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf999, (1, 4096), (0, 1), 0), reinterpret_tensor(arg243_1, (4096, 12288), (1, 4096), 0), out=buf1000)
        del arg243_1
        buf1017 = buf974; del buf974  # reuse
        buf1015 = reinterpret_tensor(buf1017, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf1016 = reinterpret_tensor(buf1017, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf1020 = buf932; del buf932  # reuse
        buf1018 = reinterpret_tensor(buf1020, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf1019 = reinterpret_tensor(buf1020, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf1060 = buf977; del buf977  # reuse
        buf1058 = reinterpret_tensor(buf1060, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf1059 = reinterpret_tensor(buf1060, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf1063 = buf935; del buf935  # reuse
        buf1061 = reinterpret_tensor(buf1063, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf1062 = reinterpret_tensor(buf1063, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_56, stack_57, stack_58, stack_59], Original ATen: [aten.stack]
        triton_poi_fused_stack_1.run(buf958, arg326_1, arg259_1, buf1000, buf1015, buf1016, buf1018, buf1019, buf1058, buf1059, buf1061, buf1062, 2048, grid=grid(2048), stream=stream0)
        buf1021 = buf988; del buf988  # reuse
        # Source Nodes: [setitem_56], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg317_1, buf1021, 4718592, grid=grid(4718592), stream=stream0)
        del buf1015
        del buf1016
        del buf1018
        del buf1019
        buf1028 = buf985; del buf985  # reuse
        # Source Nodes: [setitem_57], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg318_1, buf1028, 4718592, grid=grid(4718592), stream=stream0)
        buf1064 = buf943; del buf943  # reuse
        # Source Nodes: [setitem_58], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg319_1, buf1064, 4718592, grid=grid(4718592), stream=stream0)
        del buf1061
        del buf1062
        buf1071 = buf978; del buf978  # reuse
        # Source Nodes: [setitem_59], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg320_1, buf1071, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_56, setitem_57, setitem_58, setitem_59], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg326_1, buf1020, buf958, buf1063, buf1000, buf1021, buf1028, buf1064, buf1071, 4096, grid=grid(4096), stream=stream0)
        del buf1020
        buf1023 = reinterpret_tensor(buf999, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf999  # reuse
        # Source Nodes: [type_as_113], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf1017, buf1023, 4096, grid=grid(4096), stream=stream0)
        buf1024 = buf936; del buf936  # reuse
        # Source Nodes: [setitem_56], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf1021, buf1024, arg317_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg317_1
        buf1025 = reinterpret_tensor(buf987, (32, 1, 1152), (1152, 1152, 1)); del buf987  # reuse
        # Source Nodes: [matmul_56], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1023, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf1024, (32, 128, 1152), (147456, 1, 128), 0), out=buf1025)
        buf1066 = buf1023; del buf1023  # reuse
        # Source Nodes: [type_as_117], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf1060, buf1066, 4096, grid=grid(4096), stream=stream0)
        del buf1058
        del buf1059
        buf1067 = buf1024; del buf1024  # reuse
        # Source Nodes: [setitem_58], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf1064, buf1067, arg319_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg319_1
        buf1068 = reinterpret_tensor(buf945, (32, 1, 1152), (1152, 1152, 1)); del buf945  # reuse
        # Source Nodes: [matmul_58], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1066, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf1067, (32, 128, 1152), (147456, 1, 128), 0), out=buf1068)
        buf1030 = reinterpret_tensor(buf982, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf982  # reuse
        buf1073 = reinterpret_tensor(buf940, (1, 32, 1, 1152), (36864, 1152, 1152, 1)); del buf940  # reuse
        # Source Nodes: [getitem, mul_865, mul_910, softmax_28, softmax_29, where_28, where_29], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_6.run(arg326_1, arg260_1, buf1025, buf1068, buf1030, buf1073, 32, 1152, grid=grid(32), stream=stream0)
        del buf1025
        del buf1068
        buf1031 = buf1067; del buf1067  # reuse
        # Source Nodes: [setitem_57], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf1028, buf1031, arg318_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg318_1
        buf1032 = reinterpret_tensor(buf1066, (32, 1, 128), (128, 128, 1)); del buf1066  # reuse
        # Source Nodes: [matmul_57], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1030, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf1031, (32, 1152, 128), (147456, 128, 1), 0), out=buf1032)
        buf1033 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_28_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1032, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg239_1, (4096, 4096), (1, 4096), 0), out=buf1033)
        del arg239_1
        buf1035 = reinterpret_tensor(buf1032, (1, 1, 4096), (4096, 4096, 1)); del buf1032  # reuse
        # Source Nodes: [add_577, add_578, float_116, mean_57, mul_866, mul_867, mul_868, rsqrt_57, type_as_115], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf955, buf1033, arg85_1, buf1035, 1, 4096, grid=grid(1), stream=stream0)
        del arg85_1
        buf1036 = reinterpret_tensor(buf995, (1, 11008), (11008, 1)); del buf995  # reuse
        # Source Nodes: [l__model___layers_28_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1035, (1, 4096), (0, 1), 0), reinterpret_tensor(arg240_1, (4096, 11008), (1, 4096), 0), out=buf1036)
        del arg240_1
        buf1037 = buf994; del buf994  # reuse
        # Source Nodes: [l__model___layers_28_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1035, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg241_1, (4096, 11008), (1, 4096), 0), out=buf1037)
        del arg241_1
        buf1038 = reinterpret_tensor(buf1036, (1, 1, 11008), (11008, 11008, 1)); del buf1036  # reuse
        # Source Nodes: [mul_869, silu_28], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf1038, buf1037, 11008, grid=grid(11008), stream=stream0)
        buf1039 = reinterpret_tensor(buf1035, (1, 4096), (4096, 1)); del buf1035  # reuse
        # Source Nodes: [l__model___layers_28_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1038, (1, 11008), (0, 1), 0), reinterpret_tensor(arg242_1, (11008, 4096), (1, 11008), 0), out=buf1039)
        del arg242_1
        buf1040 = buf1014; del buf1014  # reuse
        buf1042 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_577, add_579, add_644, add_645, float_121, mean_60, mul_944, mul_945, mul_946, mul_947, rsqrt_60, type_as_120], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_38.run(buf1040, arg89_1, buf955, buf1033, buf1039, arg90_1, buf1042, 1, 4096, grid=grid(1), stream=stream0)
        del arg89_1
        del arg90_1
        buf1043 = buf958; del buf958  # reuse
        # Source Nodes: [l__model___layers_30_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1042, (1, 4096), (0, 1), 0), reinterpret_tensor(arg248_1, (4096, 12288), (1, 4096), 0), out=buf1043)
        del arg248_1
        buf1074 = buf1031; del buf1031  # reuse
        # Source Nodes: [setitem_59], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf1071, buf1074, arg320_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg320_1
        buf1075 = reinterpret_tensor(buf1042, (32, 1, 128), (128, 128, 1)); del buf1042  # reuse
        # Source Nodes: [matmul_59], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1073, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf1074, (32, 1152, 128), (147456, 128, 1), 0), out=buf1075)
        buf1076 = empty_strided((1, 4096), (4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___layers_29_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1075, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg244_1, (4096, 4096), (1, 4096), 0), out=buf1076)
        del arg244_1
        buf1078 = reinterpret_tensor(buf1075, (1, 1, 4096), (4096, 4096, 1)); del buf1075  # reuse
        # Source Nodes: [add_612, add_613, float_120, mean_59, mul_911, mul_912, mul_913, rsqrt_59, type_as_119], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf997, buf1076, arg88_1, buf1078, 1, 4096, grid=grid(1), stream=stream0)
        del arg88_1
        buf1079 = reinterpret_tensor(buf1038, (1, 11008), (11008, 1)); del buf1038  # reuse
        # Source Nodes: [l__model___layers_29_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1078, (1, 4096), (0, 1), 0), reinterpret_tensor(arg245_1, (4096, 11008), (1, 4096), 0), out=buf1079)
        del arg245_1
        buf1080 = buf1037; del buf1037  # reuse
        # Source Nodes: [l__model___layers_29_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1078, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg246_1, (4096, 11008), (1, 4096), 0), out=buf1080)
        del arg246_1
        buf1081 = reinterpret_tensor(buf1079, (1, 1, 11008), (11008, 11008, 1)); del buf1079  # reuse
        # Source Nodes: [mul_914, silu_29], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf1081, buf1080, 11008, grid=grid(11008), stream=stream0)
        buf1082 = reinterpret_tensor(buf1078, (1, 4096), (4096, 1)); del buf1078  # reuse
        # Source Nodes: [l__model___layers_29_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1081, (1, 11008), (0, 1), 0), reinterpret_tensor(arg247_1, (11008, 4096), (1, 11008), 0), out=buf1082)
        del arg247_1
        buf1097 = buf1096; del buf1096  # reuse
        buf1098 = buf1097; del buf1097  # reuse
        buf1099 = buf1098; del buf1098  # reuse
        buf1100 = buf1099; del buf1099  # reuse
        buf1101 = buf1100; del buf1100  # reuse
        buf1102 = buf1101; del buf1101  # reuse
        buf1103 = buf1102; del buf1102  # reuse
        buf1104 = buf1103; del buf1103  # reuse
        buf1105 = buf1104; del buf1104  # reuse
        buf1106 = buf1105; del buf1105  # reuse
        buf1107 = buf1106; del buf1106  # reuse
        buf1083 = buf1057; del buf1057  # reuse
        buf1108 = buf1107; del buf1107  # reuse
        buf1085 = empty_strided((1, 1, 4096), (4096, 4096, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_102, add_104, add_118, add_120, add_135, add_137, add_153, add_155, add_172, add_174, add_192, add_194, add_213, add_215, add_235, add_237, add_258, add_260, add_282, add_284, add_307, add_309, add_333, add_335, add_360, add_362, add_388, add_390, add_417, add_419, add_447, add_449, add_478, add_480, add_510, add_512, add_543, add_545, add_577, add_579, add_60, add_612, add_614, add_62, add_680, add_681, add_682, add_695, add_696, add_697, add_698, add_699, add_700, add_701, add_702, add_703, add_704, add_705, add_706, add_707, add_708, add_709, add_710, add_711, add_712, add_713, add_714, add_715, add_716, add_717, add_718, add_73, add_75, add_87, add_89, float_125, mean_62, mul_1015, mul_1016, mul_1017, mul_1018, mul_1019, mul_1020, mul_1021, mul_1022, mul_1023, mul_1024, mul_1025, mul_1026, mul_1027, mul_1028, mul_1029, mul_1030, mul_1031, mul_1032, mul_1033, mul_1034, mul_1035, mul_1036, mul_1037, mul_1038, mul_990, mul_991, mul_992, mul_993, mul_994, rsqrt_62, type_as_124], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_39.run(buf1108, buf1083, arg95_1, buf152, buf208, buf214, buf183, buf240, buf246, buf215, buf273, buf279, buf247, buf306, buf312, buf280, buf340, buf346, buf313, buf374, buf380, buf347, buf409, buf415, buf381, buf444, buf450, buf416, buf480, buf486, buf451, buf516, buf522, buf487, buf553, buf559, buf523, buf590, buf596, buf560, buf628, buf634, buf597, buf666, buf672, buf635, buf705, buf711, buf673, buf744, buf750, buf712, buf784, buf790, buf751, buf824, buf830, buf791, buf865, buf871, buf831, buf906, buf912, buf872, buf948, buf954, buf913, buf990, buf996, arg92_1, buf955, buf1033, buf1039, buf997, buf1076, buf1082, arg93_1, buf1085, 1, 4096, grid=grid(1), stream=stream0)
        del arg92_1
        del arg93_1
        del buf1033
        del buf1039
        del buf1076
        del buf1082
        del buf1083
        del buf152
        del buf183
        del buf208
        del buf214
        del buf215
        del buf240
        del buf246
        del buf247
        del buf273
        del buf279
        del buf280
        del buf306
        del buf312
        del buf313
        del buf340
        del buf346
        del buf347
        del buf374
        del buf380
        del buf381
        del buf409
        del buf415
        del buf416
        del buf444
        del buf450
        del buf451
        del buf480
        del buf486
        del buf487
        del buf516
        del buf522
        del buf523
        del buf553
        del buf559
        del buf560
        del buf590
        del buf596
        del buf597
        del buf628
        del buf634
        del buf635
        del buf666
        del buf672
        del buf673
        del buf705
        del buf711
        del buf712
        del buf744
        del buf750
        del buf751
        del buf784
        del buf790
        del buf791
        del buf824
        del buf830
        del buf831
        del buf865
        del buf871
        del buf872
        del buf906
        del buf912
        del buf913
        del buf948
        del buf954
        del buf955
        del buf990
        buf1086 = buf1000; del buf1000  # reuse
        # Source Nodes: [l__model___layers_31_attention_wqkv], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1085, (1, 4096), (0, 1), 0), reinterpret_tensor(arg253_1, (4096, 12288), (1, 4096), 0), out=buf1086)
        del arg253_1
        buf1089 = buf1060; del buf1060  # reuse
        buf1087 = reinterpret_tensor(buf1089, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf1088 = reinterpret_tensor(buf1089, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf1111 = buf1017; del buf1017  # reuse
        buf1109 = reinterpret_tensor(buf1111, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf1110 = reinterpret_tensor(buf1111, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        buf1114 = buf1063; del buf1063  # reuse
        buf1112 = reinterpret_tensor(buf1114, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 0)  # alias
        buf1113 = reinterpret_tensor(buf1114, (1, 1, 32, 64, 1), (4096, 4096, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack_60, stack_61, stack_63], Original ATen: [aten.stack]
        triton_poi_fused_stack_40.run(buf1086, arg326_1, arg259_1, buf1043, buf1087, buf1088, buf1109, buf1110, buf1112, buf1113, 2048, grid=grid(2048), stream=stream0)
        del arg259_1
        buf1090 = buf1074; del buf1074  # reuse
        # Source Nodes: [setitem_62], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg323_1, buf1090, 4718592, grid=grid(4718592), stream=stream0)
        del buf1087
        del buf1088
        buf1092 = buf1071; del buf1071  # reuse
        # Source Nodes: [setitem_63], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg324_1, buf1092, 4718592, grid=grid(4718592), stream=stream0)
        buf1115 = buf1028; del buf1028  # reuse
        # Source Nodes: [setitem_60], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg321_1, buf1115, 4718592, grid=grid(4718592), stream=stream0)
        del buf1112
        del buf1113
        buf1122 = buf1064; del buf1064  # reuse
        # Source Nodes: [setitem_61], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_2.run(arg322_1, buf1122, 4718592, grid=grid(4718592), stream=stream0)
        # Source Nodes: [setitem_60, setitem_61, setitem_62, setitem_63], Original ATen: [aten.index_put]
        triton_poi_fused_index_put_3.run(arg326_1, buf1089, buf1086, buf1114, buf1043, buf1090, buf1092, buf1115, buf1122, 4096, grid=grid(4096), stream=stream0)
        del buf1043
        del buf1086
        del buf1089
        del buf1114
        buf1117 = reinterpret_tensor(buf1085, (1, 1, 32, 128), (4096, 4096, 128, 1)); del buf1085  # reuse
        # Source Nodes: [type_as_121], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_4.run(buf1111, buf1117, 4096, grid=grid(4096), stream=stream0)
        del buf1109
        del buf1110
        buf1118 = buf1021; del buf1021  # reuse
        # Source Nodes: [setitem_60], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf1115, buf1118, arg321_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg321_1
        del buf1115
        buf1119 = reinterpret_tensor(buf1073, (32, 1, 1152), (1152, 1152, 1)); del buf1073  # reuse
        # Source Nodes: [matmul_60], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1117, (32, 1, 128), (128, 0, 1), 0), reinterpret_tensor(buf1118, (32, 128, 1152), (147456, 1, 128), 0), out=buf1119)
        buf1124 = buf1030; del buf1030  # reuse
        # Source Nodes: [getitem, mul_956, softmax_30, where_30], Original ATen: [aten._softmax, aten.index, aten.mul, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_index_mul_scalar_tensor_where_41.run(arg326_1, arg260_1, buf1119, buf1124, 32, 1152, grid=grid(32), stream=stream0)
        del arg260_1
        del arg326_1
        del buf1119
        buf1125 = buf1118; del buf1118  # reuse
        # Source Nodes: [setitem_61], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_5.run(buf1122, buf1125, arg322_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg322_1
        del buf1122
        buf1126 = reinterpret_tensor(buf1117, (32, 1, 128), (128, 128, 1)); del buf1117  # reuse
        # Source Nodes: [matmul_61], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1124, (32, 1, 1152), (1152, 0, 1), 0), reinterpret_tensor(buf1125, (32, 1152, 128), (147456, 128, 1), 0), out=buf1126)
        del buf1124
        del buf1125
        buf1127 = reinterpret_tensor(buf997, (1, 4096), (4096, 1)); del buf997  # reuse
        # Source Nodes: [l__model___layers_30_attention_wo], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1126, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg249_1, (4096, 4096), (1, 4096), 0), out=buf1127)
        del arg249_1
        buf1129 = reinterpret_tensor(buf1126, (1, 1, 4096), (4096, 4096, 1)); del buf1126  # reuse
        # Source Nodes: [add_648, add_649, float_124, mean_61, mul_957, mul_958, mul_959, rsqrt_61, type_as_123], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_11.run(buf1040, buf1127, arg91_1, buf1129, 1, 4096, grid=grid(1), stream=stream0)
        del arg91_1
        buf1130 = reinterpret_tensor(buf1081, (1, 11008), (11008, 1)); del buf1081  # reuse
        # Source Nodes: [l__model___layers_30_feed_forward_w1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1129, (1, 4096), (0, 1), 0), reinterpret_tensor(arg250_1, (4096, 11008), (1, 4096), 0), out=buf1130)
        del arg250_1
        buf1131 = buf1080; del buf1080  # reuse
        # Source Nodes: [l__model___layers_30_feed_forward_w3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1129, (1, 4096), (4096, 1), 0), reinterpret_tensor(arg251_1, (4096, 11008), (1, 4096), 0), out=buf1131)
        del arg251_1
        buf1132 = reinterpret_tensor(buf1130, (1, 1, 11008), (11008, 11008, 1)); del buf1130  # reuse
        # Source Nodes: [mul_960, silu_30], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_8.run(buf1132, buf1131, 11008, grid=grid(11008), stream=stream0)
        del buf1131
        buf1133 = reinterpret_tensor(buf1129, (1, 4096), (4096, 1)); del buf1129  # reuse
        # Source Nodes: [l__model___layers_30_feed_forward_w2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1132, (1, 11008), (0, 1), 0), reinterpret_tensor(arg252_1, (11008, 4096), (1, 11008), 0), out=buf1133)
        del arg252_1
        del buf1132
        buf1134 = reinterpret_tensor(buf1111, (1, 1, 4096), (4096, 4096, 1)); del buf1111  # reuse
        buf1136 = reinterpret_tensor(buf996, (1, 1, 4096), (4096, 4096, 1)); del buf996  # reuse
        # Source Nodes: [add_648, add_650, add_719, add_720, float_129, mean_64, mul_1039, mul_1040, mul_1041, mul_1042, rsqrt_64, type_as_128], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_42.run(buf1108, arg95_1, buf1040, buf1127, buf1133, arg96_1, buf1134, buf1136, 1, 4096, grid=grid(1), stream=stream0)
        del arg95_1
        del arg96_1
        del buf1040
        del buf1108
        del buf1127
        del buf1133
        del buf1134
        buf1137 = empty_strided((1, 32000), (32000, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [l__model___output], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1136, (1, 4096), (0, 1), 0), reinterpret_tensor(arg258_1, (4096, 32000), (1, 4096), 0), out=buf1137)
        del arg258_1
        del buf1136
        buf1138 = buf1137; del buf1137  # reuse
        # Source Nodes: [truediv], Original ATen: [aten.div]
        triton_poi_fused_div_43.run(buf1138, 32000, grid=grid(32000), stream=stream0)
        # Source Nodes: [topk, truediv], Original ATen: [aten.div, aten.topk]
        buf1139 = aten.topk(buf1138, 200)
        buf1140 = buf1139[0]
        assert_size_stride(buf1140, (1, 200), (200, 1))
        del buf1139
        buf1142 = empty_strided((1, 1, 4), (4, 4, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [lt, softmax_32, where_32], Original ATen: [aten._softmax, aten.lt, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_lt_scalar_tensor_where_44.run(buf1138, buf1140, buf1142, 4, 8000, grid=grid(4), stream=stream0)
        buf1143 = reinterpret_tensor(buf54, (1, 1), (1, 1)); del buf54  # reuse
        # Source Nodes: [lt, softmax_32, where_32], Original ATen: [aten._softmax, aten.lt, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_lt_scalar_tensor_where_45.run(buf1142, buf1143, 1, 4, grid=grid(1), stream=stream0)
        buf1144 = buf1142; del buf1142  # reuse
        # Source Nodes: [lt, softmax_32, where_32], Original ATen: [aten._softmax, aten.lt, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax_lt_scalar_tensor_where_46.run(buf1138, buf1140, buf1143, buf1144, 4, 8000, grid=grid(4), stream=stream0)
        buf1145 = empty_strided((1, 1), (1, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [lt, softmax_32, where_32], Original ATen: [aten._softmax, aten.lt, aten.scalar_tensor, aten.where]
        triton_per_fused__softmax_lt_scalar_tensor_where_47.run(buf1144, buf1145, 1, 4, grid=grid(1), stream=stream0)
        del buf1144
        buf1147 = empty_strided((1, ), (1, ), device='cuda', dtype=torch.int64)
        # Source Nodes: [], Original ATen: []
        aten.randint.low_out(-9223372036854775808, 9223372036854775807, [1], out=buf1147)
        buf1146 = buf1138; del buf1138  # reuse
        buf1150 = empty_strided((1, 1), (1, 1), device='cuda', dtype=torch.int32)
        # Source Nodes: [argmax, exponential_, lt, softmax_32, to, truediv_1, where_32], Original ATen: [aten._softmax, aten._to_copy, aten.argmax, aten.div, aten.exponential, aten.lt, aten.scalar_tensor, aten.where]
        triton_red_fused__softmax__to_copy_argmax_div_exponential_lt_scalar_tensor_where_48.run(buf1146, buf1140, buf1143, buf1145, buf1147, buf1150, 0, 1, 32000, grid=grid(1), stream=stream0)
        del buf1140
        del buf1143
        del buf1145
        del buf1147
        # Source Nodes: [setitem_62], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_49.run(buf1090, arg323_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg323_1
        del buf1090
        # Source Nodes: [setitem_63], Original ATen: [aten.slice_scatter]
        triton_poi_fused_slice_scatter_49.run(buf1092, arg324_1, 4718592, grid=grid(4718592), stream=stream0)
        del arg324_1
        return (buf1150, buf1146, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    arg0_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg1_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg2_1 = rand_strided((2, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg3_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg4_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg5_1 = rand_strided((3, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg6_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg7_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg8_1 = rand_strided((4, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg9_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg10_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg11_1 = rand_strided((5, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg12_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg13_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg14_1 = rand_strided((6, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg15_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg16_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg17_1 = rand_strided((7, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg18_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg19_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg20_1 = rand_strided((8, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg21_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg22_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg23_1 = rand_strided((9, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg24_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg25_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg26_1 = rand_strided((10, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg27_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg28_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg29_1 = rand_strided((11, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg30_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg31_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg32_1 = rand_strided((12, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg33_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg34_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg35_1 = rand_strided((13, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg36_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg37_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg38_1 = rand_strided((14, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg39_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg40_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg41_1 = rand_strided((15, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg42_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg43_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg44_1 = rand_strided((16, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg45_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg46_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg47_1 = rand_strided((17, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg48_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg49_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg50_1 = rand_strided((18, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg51_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg52_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg53_1 = rand_strided((19, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg54_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg55_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg56_1 = rand_strided((20, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg57_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg58_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg59_1 = rand_strided((21, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg60_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg61_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg62_1 = rand_strided((22, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg63_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg64_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg65_1 = rand_strided((23, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg66_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg67_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg68_1 = rand_strided((24, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg69_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg70_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg71_1 = rand_strided((25, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg72_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg73_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg74_1 = rand_strided((26, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg75_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg76_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg77_1 = rand_strided((27, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg78_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg79_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg80_1 = rand_strided((28, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg81_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg82_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg83_1 = rand_strided((29, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg84_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg85_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg86_1 = rand_strided((30, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg87_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg88_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg89_1 = rand_strided((31, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg90_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg91_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg92_1 = rand_strided((32, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg93_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg94_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg95_1 = rand_strided((33, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg96_1 = rand_strided((4096, ), (1, ), device='cuda:0', dtype=torch.float16)
    arg97_1 = rand_strided((32000, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg98_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg99_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg100_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg101_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg102_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg103_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg104_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg105_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg106_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg107_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg108_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg109_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg110_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg111_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg112_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg113_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg114_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg115_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg116_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg117_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg118_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg119_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg120_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg121_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg122_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg123_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg124_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg125_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg126_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg127_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg128_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg129_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg130_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg131_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg132_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg133_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg134_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg135_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg136_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg137_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg138_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg139_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg140_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg141_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg142_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg143_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg144_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg145_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg146_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg147_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg148_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg149_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg150_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg151_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg152_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg153_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg154_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg155_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg156_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg157_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg158_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg159_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg160_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg161_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg162_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg163_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg164_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg165_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg166_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg167_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg168_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg169_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg170_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg171_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg172_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg173_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg174_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg175_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg176_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg177_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg178_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg179_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg180_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg181_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg182_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg183_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg184_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg185_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg186_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg187_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg188_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg189_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg190_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg191_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg192_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg193_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg194_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg195_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg196_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg197_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg198_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg199_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg200_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg201_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg202_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg203_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg204_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg205_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg206_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg207_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg208_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg209_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg210_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg211_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg212_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg213_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg214_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg215_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg216_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg217_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg218_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg219_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg220_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg221_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg222_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg223_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg224_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg225_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg226_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg227_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg228_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg229_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg230_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg231_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg232_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg233_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg234_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg235_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg236_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg237_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg238_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg239_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg240_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg241_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg242_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg243_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg244_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg245_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg246_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg247_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg248_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg249_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg250_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg251_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg252_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg253_1 = rand_strided((12288, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg254_1 = rand_strided((4096, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg255_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg256_1 = rand_strided((11008, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg257_1 = rand_strided((4096, 11008), (11008, 1), device='cuda:0', dtype=torch.float16)
    arg258_1 = rand_strided((32000, 4096), (4096, 1), device='cuda:0', dtype=torch.float16)
    arg259_1 = rand_strided((2048, 64, 2), (128, 2, 1), device='cuda:0', dtype=torch.float16)
    arg260_1 = rand_strided((1152, 1152), (1152, 1), device='cuda:0', dtype=torch.bool)
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
    arg293_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg294_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg295_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg296_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg297_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg298_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg299_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg300_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg301_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg302_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg303_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg304_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg305_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg306_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg307_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg308_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg309_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg310_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg311_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg312_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg313_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg314_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg315_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg316_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg317_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg318_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg319_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg320_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg321_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg322_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg323_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg324_1 = rand_strided((1, 32, 1152, 128), (4718592, 147456, 128, 1), device='cuda:0', dtype=torch.float16)
    arg325_1 = rand_strided((1, 1), (1, 1), device='cuda:0', dtype=torch.int32)
    arg326_1 = rand_strided((1, ), (1, ), device='cuda:0', dtype=torch.int32)
    return print_performance(lambda: call([arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1, arg15_1, arg16_1, arg17_1, arg18_1, arg19_1, arg20_1, arg21_1, arg22_1, arg23_1, arg24_1, arg25_1, arg26_1, arg27_1, arg28_1, arg29_1, arg30_1, arg31_1, arg32_1, arg33_1, arg34_1, arg35_1, arg36_1, arg37_1, arg38_1, arg39_1, arg40_1, arg41_1, arg42_1, arg43_1, arg44_1, arg45_1, arg46_1, arg47_1, arg48_1, arg49_1, arg50_1, arg51_1, arg52_1, arg53_1, arg54_1, arg55_1, arg56_1, arg57_1, arg58_1, arg59_1, arg60_1, arg61_1, arg62_1, arg63_1, arg64_1, arg65_1, arg66_1, arg67_1, arg68_1, arg69_1, arg70_1, arg71_1, arg72_1, arg73_1, arg74_1, arg75_1, arg76_1, arg77_1, arg78_1, arg79_1, arg80_1, arg81_1, arg82_1, arg83_1, arg84_1, arg85_1, arg86_1, arg87_1, arg88_1, arg89_1, arg90_1, arg91_1, arg92_1, arg93_1, arg94_1, arg95_1, arg96_1, arg97_1, arg98_1, arg99_1, arg100_1, arg101_1, arg102_1, arg103_1, arg104_1, arg105_1, arg106_1, arg107_1, arg108_1, arg109_1, arg110_1, arg111_1, arg112_1, arg113_1, arg114_1, arg115_1, arg116_1, arg117_1, arg118_1, arg119_1, arg120_1, arg121_1, arg122_1, arg123_1, arg124_1, arg125_1, arg126_1, arg127_1, arg128_1, arg129_1, arg130_1, arg131_1, arg132_1, arg133_1, arg134_1, arg135_1, arg136_1, arg137_1, arg138_1, arg139_1, arg140_1, arg141_1, arg142_1, arg143_1, arg144_1, arg145_1, arg146_1, arg147_1, arg148_1, arg149_1, arg150_1, arg151_1, arg152_1, arg153_1, arg154_1, arg155_1, arg156_1, arg157_1, arg158_1, arg159_1, arg160_1, arg161_1, arg162_1, arg163_1, arg164_1, arg165_1, arg166_1, arg167_1, arg168_1, arg169_1, arg170_1, arg171_1, arg172_1, arg173_1, arg174_1, arg175_1, arg176_1, arg177_1, arg178_1, arg179_1, arg180_1, arg181_1, arg182_1, arg183_1, arg184_1, arg185_1, arg186_1, arg187_1, arg188_1, arg189_1, arg190_1, arg191_1, arg192_1, arg193_1, arg194_1, arg195_1, arg196_1, arg197_1, arg198_1, arg199_1, arg200_1, arg201_1, arg202_1, arg203_1, arg204_1, arg205_1, arg206_1, arg207_1, arg208_1, arg209_1, arg210_1, arg211_1, arg212_1, arg213_1, arg214_1, arg215_1, arg216_1, arg217_1, arg218_1, arg219_1, arg220_1, arg221_1, arg222_1, arg223_1, arg224_1, arg225_1, arg226_1, arg227_1, arg228_1, arg229_1, arg230_1, arg231_1, arg232_1, arg233_1, arg234_1, arg235_1, arg236_1, arg237_1, arg238_1, arg239_1, arg240_1, arg241_1, arg242_1, arg243_1, arg244_1, arg245_1, arg246_1, arg247_1, arg248_1, arg249_1, arg250_1, arg251_1, arg252_1, arg253_1, arg254_1, arg255_1, arg256_1, arg257_1, arg258_1, arg259_1, arg260_1, arg261_1, arg262_1, arg263_1, arg264_1, arg265_1, arg266_1, arg267_1, arg268_1, arg269_1, arg270_1, arg271_1, arg272_1, arg273_1, arg274_1, arg275_1, arg276_1, arg277_1, arg278_1, arg279_1, arg280_1, arg281_1, arg282_1, arg283_1, arg284_1, arg285_1, arg286_1, arg287_1, arg288_1, arg289_1, arg290_1, arg291_1, arg292_1, arg293_1, arg294_1, arg295_1, arg296_1, arg297_1, arg298_1, arg299_1, arg300_1, arg301_1, arg302_1, arg303_1, arg304_1, arg305_1, arg306_1, arg307_1, arg308_1, arg309_1, arg310_1, arg311_1, arg312_1, arg313_1, arg314_1, arg315_1, arg316_1, arg317_1, arg318_1, arg319_1, arg320_1, arg321_1, arg322_1, arg323_1, arg324_1, arg325_1, arg326_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
