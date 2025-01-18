
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


# kernel path: /tmp/torchinductor_mengqy/hb/chb3hasgpfrh5wt3ctsnnpxsd6dga3audi2vhpk53mhpfp6qzfsx.py
# Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# cross_entropy => scalar_tensor_24
triton_poi_fused_nll_loss_backward_nll_loss_forward_0 = async_compile.triton('triton_poi_fused_nll_loss_backward_nll_loss_forward_0', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[536870912], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_0', 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]})
@triton.jit
def triton_poi_fused_nll_loss_backward_nll_loss_forward_0(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 308779008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')

import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid, start_graph, end_graph
from torch._C import _cuda_getCurrentRawStream as get_cuda_stream


# kernel path: /tmp/torchinductor_mengqy/cd/ccdd2gbsqx3bp3fsxsmj76sy6h7rexeu3o5plvlg7ywvo23iddpw.py
# Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
# cross_entropy => scalar_tensor_24
triton_poi_fused_nll_loss_backward_nll_loss_forward_1 = async_compile.triton('triton_poi_fused_nll_loss_backward_nll_loss_forward_1', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*i64', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_nll_loss_backward_nll_loss_forward_1', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_nll_loss_backward_nll_loss_forward_1(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + ((2049*(x0 // 2048)) + (x0 % 2048)), None)
    tmp1 = tl.full([1], -100, tl.int64)
    tmp2 = tmp0 != tmp1
    tmp3 = tl.full([1], 0, tl.int64)
    tmp4 = tl.where(tmp2, tmp0, tmp3)
    tmp5 = tl.where(tmp4 < 0, tmp4 + 50257, tmp4)
    tl.device_assert((0 <= tmp5) & (tmp5 < 50257), "index out of bounds: 0 <= tmp5 < 50257")
    tmp6 = -1.0
    tl.store(out_ptr0 + (tmp5 + (50257*x0)), tmp6, None)
''')


# kernel path: /tmp/torchinductor_mengqy/dt/cdtnadz33kk7hy4xl5jjlpbmq3lh7xvkr5vh7dtl6ydg73lmycjt.py
# Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
# cross_entropy => scalar_tensor_25
triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2 = async_compile.triton('triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2', '''
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
    meta={'signature': {0: '*fp16', 1: '*i64', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]}
)
@triton.jit
def triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 50257
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    tmp1 = tl.load(in_ptr1 + ((2049*(x0 // 2048)) + (x0 % 2048)), None, eviction_policy='evict_last')
    tmp4 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
    tmp6 = tl.load(in_ptr3 + (0)).to(tl.float32)
    tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (50257*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tl.full([1, 1], -100, tl.int64)
        tmp3 = tmp1 != tmp2
        tmp8 = tmp5 / tmp7
        tmp9 = 0.0
        tmp10 = tl.where(tmp3, tmp8, tmp9)
        tmp11 = tmp0 * tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tmp19 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
    tmp21 = tl.load(in_ptr3 + (0)).to(tl.float32)
    tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp16 = tl.load(in_ptr0 + (r1 + (50257*x0)), rmask, other=0).to(tl.float32)
        tmp28 = tl.load(in_ptr4 + (r1 + (50257*x0)), rmask, other=0).to(tl.float32)
        tmp17 = tl.full([1, 1], -100, tl.int64)
        tmp18 = tmp1 != tmp17
        tmp23 = tmp20 / tmp22
        tmp24 = 0.0
        tmp25 = tl.where(tmp18, tmp23, tmp24)
        tmp26 = tmp16 * tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tl.exp(tmp29)
        tmp31 = tmp30 * tmp14
        tmp32 = tmp27 - tmp31
        tmp33 = tmp32.to(tl.float32)
        tl.store(out_ptr1 + (r1 + (50257*x0)), tmp33, rmask)
        tl.store(out_ptr2 + (r1 + (50264*x0)), tmp33, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/iu/ciuqwdekvqjevtlkd57temlpm4nkr4exxbwzqerzkp4btewlaiww.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[65536, 8192], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]})
@triton.jit
def triton_poi_fused_3(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 50257
    xnumel = 6144
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x1 = xindex
    y0 = yindex
    tmp0 = tl.load(in_ptr0 + (y0 + (50257*x1)), ymask, eviction_policy='evict_last').to(tl.float32)
    tl.store(out_ptr0 + (x1 + (6144*y0)), tmp0, ymask)
''')


# kernel path: /tmp/torchinductor_mengqy/y6/cy6feskldwd7pxjlry4dsamx7j5pytnfrshq4ppregnyxnsrclhv.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_4 = async_compile.triton('triton_poi_fused_4', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]})
@triton.jit
def triton_poi_fused_4(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/hc/chcatdp7bczvsqlzmso4bsbafg5risvkd5he57ujh3xt2wwadp3q.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_5 = async_compile.triton('triton_poi_fused_5', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'configs': [instance_descriptor(divisible_by_16=(1,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]})
@triton.jit
def triton_poi_fused_5(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (50264*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/2l/c2lebak4vvn4x77tq3wwlf7pjcgvwuftc6wnb4gyunybzojmsvcz.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_6 = async_compile.triton('triton_poi_fused_6', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_6', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_6(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102926336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/oy/coy7ukaihla62qymebe7777clwfijqh57kg4gpuqufdpnb4rb7qi.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_7 = async_compile.triton('triton_poi_fused_7', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[16384], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_7', 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]})
@triton.jit
def triton_poi_fused_7(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 14336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/6m/c6mh2hvaxaj6z2n6rmzmdijoetonwervjyjwfvbwojo5uyip7fkr.py
# Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add => add_138
# float_1 => convert_element_type_276
# mean => mean_46
# mul => mul_391
# mul_1 => mul_392
# mul_2 => mul_393
# rsqrt => rsqrt_46
# type_as => convert_element_type_277
triton_red_fused__to_copy_add_mean_mul_rsqrt_8 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_8', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_8', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_8(in_ptr0, in_ptr1, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp4 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp2 = tmp1 * tmp1
        tmp3 = tl.broadcast_to(tmp2, [XBLOCK, RBLOCK])
        tmp5 = _tmp4 + tmp3
        _tmp4 = tl.where(rmask, tmp5, _tmp4)
    tmp4 = tl.sum(_tmp4, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp6 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp7 = tmp6.to(tl.float32)
        tmp8 = 2048.0
        tmp9 = tmp4 / tmp8
        tmp10 = 1e-05
        tmp11 = tmp9 + tmp10
        tmp12 = tl.math.rsqrt(tmp11)
        tmp13 = tmp7 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp16 = tmp14 * tmp15
        tl.store(out_ptr1 + (r1 + (2048*x0)), tmp16, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/2m/c2mebwsiokrtbfrhtacidadb2dpa4zwwz4urpubfq36elchw3gcq.py
# Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
# stack => cat_46
# stack_1 => cat_47
triton_poi_fused_stack_9 = async_compile.triton('triton_poi_fused_stack_9', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_9', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 5, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7,))]})
@triton.jit
def triton_poi_fused_stack_9(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 16
    x4 = (xindex // 1024)
    x2 = (xindex // 1024) % 2048
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (2048 + x0 + (128*x1) + (6144*x4)), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + ((2*x0) + (128*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (2112 + x0 + (128*x1) + (6144*x4)), None).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + ((2*x0) + (128*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr0 + (x0 + (128*x1) + (6144*x4)), None).to(tl.float32)
    tmp17 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (6144*x4)), None).to(tl.float32)
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


# kernel path: /tmp/torchinductor_mengqy/4h/c4hrwr3mgoyjmxpzj4ifnwcd6juzoilh5ekqmkhhoooelsmabey2.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_92
triton_poi_fused_clone_10 = async_compile.triton('triton_poi_fused_clone_10', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_clone_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_mengqy/be/cbe3xromgbkztqozljmu7od3rtw77rvyygfcqwisprc2knjrqfku.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_93
triton_poi_fused_clone_11 = async_compile.triton('triton_poi_fused_clone_11', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[8192, 2048], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_11', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]})
@triton.jit
def triton_poi_fused_clone_11(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_mengqy/xe/cxej46543h5bht4anmv4wvthm52kswawxkejf5dpl2jv5tr3a4w4.py
# Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
# mul_11 => mul_402
# softmax => amax_23, convert_element_type_282, convert_element_type_283, div_23, exp_23, sub_71, sum_24
# where => where_23
triton_red_fused__softmax_mul_where_12 = async_compile.triton('triton_red_fused__softmax_mul_where_12', '''
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
    meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_mul_where_12', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_red_fused__softmax_mul_where_12(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_mengqy/5f/c5fjhsgehu3e2lqcwsnwa22gigm6v6yat4uruhdvcffnjyq7xlgd.py
# Source Nodes: [matmul_1], Original ATen: [aten.clone]
# matmul_1 => clone_94
triton_poi_fused_clone_13 = async_compile.triton('triton_poi_fused_clone_13', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_13', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_clone_13(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_mengqy/cg/ccgmisrzilbaeoch3wf6klnuosnmvv42ulunl7llg7fofj7yrtmy.py
# Source Nodes: [linear_1], Original ATen: [aten.view]
# linear_1 => view_616
triton_poi_fused_view_14 = async_compile.triton('triton_poi_fused_view_14', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_14', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_view_14(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((128*(x1 % 2048)) + (262144*(x0 // 128)) + (4194304*(x1 // 2048)) + (x0 % 128)), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/3w/c3wshtjfvn43wcudar45pd4uko4zgu5qksprhcjmkecdhihkppj2.py
# Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_3 => add_141
# add_4 => add_142
# float_4 => convert_element_type_284
# mean_1 => mean_47
# mul_12 => mul_403
# mul_13 => mul_404
# mul_14 => mul_405
# rsqrt_1 => rsqrt_47
# type_as_3 => convert_element_type_285
triton_red_fused__to_copy_add_mean_mul_rsqrt_15 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_15', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_15', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_15(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    tl.store(out_ptr0 + (x0), tmp6, None)
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


# kernel path: /tmp/torchinductor_mengqy/6z/c6z2kejtpneazw436utid2nieent64yqurosnps765uavatoenuv.py
# Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
# mul_15 => mul_407
# silu => convert_element_type_286, convert_element_type_287, mul_406, sigmoid_23
triton_poi_fused_mul_silu_16 = async_compile.triton('triton_poi_fused_mul_silu_16', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_16', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]})
@triton.jit
def triton_poi_fused_mul_silu_16(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 34603008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tl.store(out_ptr0 + (x0), tmp6, None)
''')


# kernel path: /tmp/torchinductor_mengqy/x5/cx5umqzw27y5c4q47shxycyr3xap3e2zmrqqziuvtkijq6znuiua.py
# Source Nodes: [add_3, add_5, float_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.sum]
# add_3 => add_141
# add_5 => add_143
# float_1 => convert_element_type_288
triton_red_fused__to_copy_add_div_mul_sum_17 = async_compile.triton('triton_red_fused__to_copy_add_div_mul_sum_17', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mul_sum_17', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mul_sum_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp7 = tl.load(in_ptr4 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tmp0 * tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp6 = tmp4 + tmp5
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp3 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp16 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp22 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp23 = tl.load(in_ptr3 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp25 = tl.load(in_ptr4 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp29 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp30 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp14 = -0.5
        tmp15 = tmp12 * tmp14
        tmp17 = tmp16 * tmp16
        tmp18 = tmp17 * tmp16
        tmp19 = tmp15 * tmp18
        tmp20 = 2048.0
        tmp21 = tmp19 / tmp20
        tmp24 = tmp22 + tmp23
        tmp26 = tmp24 + tmp25
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp21 * tmp27
        tmp31 = tmp29 * tmp30
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp32 * tmp16
        tmp34 = tmp33 + tmp28
        tmp35 = tmp34 + tmp28
        tmp36 = tmp35.to(tl.float32)
        tl.store(out_ptr2 + (r1 + (2048*x0)), tmp36, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/ob/cobfqv4cz5ajfu3nwk6cvojrd47zr23rocyhopigja33noc5j7tr.py
# Source Nodes: [silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# silu => convert_element_type_286, convert_element_type_287, mul_406, sigmoid_23
triton_poi_fused_add_fill_mul_sigmoid_silu_sub_18 = async_compile.triton('triton_poi_fused_add_fill_mul_sigmoid_silu_sub_18', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_silu_sub_18', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]})
@triton.jit
def triton_poi_fused_add_fill_mul_sigmoid_silu_sub_18(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 34603008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp7 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp3 = tl.sigmoid(tmp2)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp0 * tmp5
    tmp8 = tmp0 * tmp7
    tmp9 = tl.sigmoid(tmp1)
    tmp10 = 1.0
    tmp11 = tmp10 - tmp9
    tmp12 = tmp1 * tmp11
    tmp13 = tmp12 + tmp10
    tmp14 = tmp9 * tmp13
    tmp15 = tmp8 * tmp14
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(in_out_ptr0 + (x0), tmp15, None)
''')


# kernel path: /tmp/torchinductor_mengqy/w7/cw7g7asdm3t2udspaxywjj66oary2seivxi5bnm6gbwkna3j2qw6.py
# Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add_3 => add_141
# add_4 => add_142
# float_4 => convert_element_type_284
# mean_1 => mean_47
# mul_12 => mul_403
# rsqrt_1 => rsqrt_47
triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_19 = async_compile.triton('triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_19', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp16', 7: '*fp16', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_19', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_19(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp7 = tl.load(in_ptr4 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp5 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp20 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp17 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp33 = tl.load(in_ptr3 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp34 = tl.load(in_ptr4 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp40 = tl.load(in_ptr6 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp16 = tmp14 + tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp21 = 2048.0
        tmp22 = tmp20 / tmp21
        tmp23 = 1e-05
        tmp24 = tmp22 + tmp23
        tmp25 = tl.math.rsqrt(tmp24)
        tmp26 = tmp19 * tmp25
        tmp27 = -0.5
        tmp28 = tmp12 * tmp27
        tmp29 = tmp25 * tmp25
        tmp30 = tmp29 * tmp25
        tmp31 = tmp28 * tmp30
        tmp32 = tmp31 / tmp21
        tmp35 = tmp33 + tmp34
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tmp32 * tmp36
        tmp38 = tmp26 + tmp37
        tmp39 = tmp38 + tmp37
        tmp41 = tmp39.to(tl.float32)
        tmp42 = tmp40 + tmp41
        tl.store(out_ptr2 + (r1 + (2048*x0)), tmp42, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/pp/cppq47xwxapclcf4lusrrdwn4h4anphdljzsp3lp4aa3nytzud5r.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_20 = async_compile.triton('triton_poi_fused_clone_20', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_20', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_clone_20(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 128
    x1 = (xindex // 128) % 2048
    x2 = (xindex // 262144) % 16
    x3 = (xindex // 4194304)
    x4 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (128*x2) + (2048*x1) + (4194304*x3)), None).to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/jg/cjgf76cazhoa5t5bzchrcopwehj6mqrlqj33kfhekjda6yyxhrrl.py
# Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
# cross_entropy => scalar_tensor_25
triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21 = async_compile.triton('triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 2048],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*i1', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
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
        tmp2 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tmp0.to(tl.float32)
        tmp3 = tmp2.to(tl.float32)
        tmp4 = tmp1 * tmp3
        tmp5 = tl.broadcast_to(tmp4, [XBLOCK, RBLOCK])
        tmp7 = _tmp6 + tmp5
        _tmp6 = tl.where(rmask, tmp7, _tmp6)
    tmp6 = tl.sum(_tmp6, 1)[:, None]
    x2 = xindex % 2048
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp8 = tl.load(in_ptr2 + (r1 + (2048*x2)), rmask, eviction_policy='evict_last')
        tmp9 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp11 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp10 = tmp9.to(tl.float32)
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp10 * tmp12
        tmp14 = tmp12 * tmp6
        tmp15 = tmp13 - tmp14
        tmp16 = tmp15.to(tl.float32)
        tmp17 = 0.0
        tmp18 = tl.where(tmp8, tmp16, tmp17)
        tmp19 = 0.08838834764831843
        tmp20 = tmp18 * tmp19
        tl.store(out_ptr1 + (r1 + (2048*x0)), tmp20, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/tj/ctj6hmeinlgxzvf7i2g6o4uo2eehqtnfqsdnq5wnv33h7wglaek7.py
# Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]

triton_poi_fused__to_copy_cat_22 = async_compile.triton('triton_poi_fused__to_copy_cat_22', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[8192, 2048], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_cat_22', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]})
@triton.jit
def triton_poi_fused__to_copy_cat_22(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr1, out_ptr3, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 2048
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 128
    x3 = (xindex // 128)
    y0 = yindex % 2048
    y1 = (yindex // 2048)
    x5 = xindex
    y4 = yindex
    tmp3 = tl.load(in_ptr0 + (2048 + y0 + (4096*(x2 % 64)) + (262144*x3) + (4194304*y1)), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr1 + ((2*(x2 % 64)) + (128*y0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr2 + ((2*(x2 % 64)) + (128*y0)), None, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr0 + (y0 + (4096*(x2 % 64)) + (262144*x3) + (4194304*y1)), None, eviction_policy='evict_last').to(tl.float32)
    tmp27 = tl.load(in_ptr3 + (1 + (2*(x2 % 64)) + (128*y0) + (262144*x3) + (4194304*y1)), None, eviction_policy='evict_last').to(tl.float32)
    tmp34 = tl.load(in_ptr3 + ((2*(x2 % 64)) + (128*y0) + (262144*x3) + (4194304*y1)), None, eviction_policy='evict_last').to(tl.float32)
    tmp0 = (x2 // 64)
    tmp1 = tl.full([1, 1], 0, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp5.to(tl.float32)
    tmp7 = tmp4 * tmp6
    tmp8 = 0.0
    tmp9 = tl.where(tmp2, tmp7, tmp8)
    tmp10 = tl.full([1, 1], 1, tl.int32)
    tmp11 = tmp0 == tmp10
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp4 * tmp13
    tmp15 = tl.where(tmp11, tmp14, tmp8)
    tmp16 = tmp9 + tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp19 = -tmp18
    tmp20 = tmp19 * tmp6
    tmp21 = tl.where(tmp11, tmp20, tmp8)
    tmp22 = tmp16 + tmp21
    tmp23 = tmp18 * tmp13
    tmp24 = tl.where(tmp2, tmp23, tmp8)
    tmp25 = tmp22 + tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28 * tmp6
    tmp30 = tl.where(tmp2, tmp29, tmp8)
    tmp31 = tmp28 * tmp13
    tmp32 = tl.where(tmp11, tmp31, tmp8)
    tmp33 = tmp30 + tmp32
    tmp35 = tmp34.to(tl.float32)
    tmp36 = -tmp35
    tmp37 = tmp36 * tmp6
    tmp38 = tl.where(tmp11, tmp37, tmp8)
    tmp39 = tmp33 + tmp38
    tmp40 = tmp35 * tmp13
    tmp41 = tl.where(tmp2, tmp40, tmp8)
    tmp42 = tmp39 + tmp41
    tmp43 = tmp42.to(tl.float32)
    tl.store(out_ptr1 + (x5 + (6144*y4)), tmp26, None)
    tl.store(out_ptr3 + (x5 + (6144*y4)), tmp43, None)
''')


# kernel path: /tmp/torchinductor_mengqy/z2/cz2llnfcvf4ufpt3eooy4a6tl2wci37yqpa2vmoiyphj5gpq4e6n.py
# Source Nodes: [], Original ATen: [aten.cat]

triton_poi_fused_cat_23 = async_compile.triton('triton_poi_fused_cat_23', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[16777216], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_cat_23', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_cat_23(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 12582912
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 2048
    x1 = (xindex // 2048) % 2048
    x2 = (xindex // 4194304)
    x3 = (xindex // 2048)
    tmp0 = tl.load(in_ptr0 + ((128*x1) + (262144*(x0 // 128)) + (4194304*x2) + (x0 % 128)), None).to(tl.float32)
    tl.store(out_ptr0 + (x0 + (6144*x3)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/vo/cvoqcjj3augw7h37lu5duqs4rel6wwifdd2p3jmoy7pilekxhbyd.py
# Source Nodes: [add, add_3, add_4, add_5, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add => add_138
# add_3 => add_141
# add_4 => add_142
# add_5 => add_143
# float_1 => convert_element_type_276, convert_element_type_288
# float_4 => convert_element_type_284
# mean => mean_46
# mean_1 => mean_47
# mul => mul_391
# mul_1 => mul_392, mul_409
# mul_12 => mul_403
# mul_13 => mul_404
# rsqrt => rsqrt_46
# rsqrt_1 => rsqrt_47
# type_as => convert_element_type_277, convert_element_type_289
# type_as_3 => convert_element_type_285
triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_24 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_24', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp16', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: '*fp32', 13: 'i32', 14: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_24', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(13, 14))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, out_ptr1, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp40 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2) + (262144*x1)), rmask, other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + (2048*r2) + (262144*x1)), rmask, other=0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + (x0 + (2048*r2) + (262144*x1)), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (x0 + (2048*r2) + (262144*x1)), rmask, other=0).to(tl.float32)
        tmp7 = tl.load(in_ptr4 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp14 = tl.load(in_ptr5 + (x0 + (2048*r2) + (262144*x1)), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr6 + (x0 + (2048*r2) + (262144*x1)), rmask, other=0).to(tl.float32)
        tmp18 = tl.load(in_ptr7 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp30 = tl.load(in_ptr8 + (x0 + (2048*r2) + (262144*x1)), rmask, other=0).to(tl.float32)
        tmp32 = tl.load(in_ptr9 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp3 = tmp1 + tmp2
        tmp5 = tmp3 + tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp0 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
        tmp16 = tmp14 + tmp15
        tmp17 = tmp3.to(tl.float32)
        tmp19 = 2048.0
        tmp20 = tmp18 / tmp19
        tmp21 = 1e-05
        tmp22 = tmp20 + tmp21
        tmp23 = tl.math.rsqrt(tmp22)
        tmp24 = tmp17 * tmp23
        tmp25 = tmp24.to(tl.float32)
        tmp26 = tmp16 * tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask, tmp29, _tmp28)
        tmp31 = tmp1.to(tl.float32)
        tmp33 = tmp32 / tmp19
        tmp34 = tmp33 + tmp21
        tmp35 = tl.math.rsqrt(tmp34)
        tmp36 = tmp31 * tmp35
        tmp37 = tmp36.to(tl.float32)
        tmp38 = tmp30 * tmp37
        tmp39 = tl.broadcast_to(tmp38, [XBLOCK, RBLOCK])
        tmp41 = _tmp40 + tmp39
        _tmp40 = tl.where(rmask, tmp41, _tmp40)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp28, None)
    tmp40 = tl.sum(_tmp40, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp40, None)
''')


# kernel path: /tmp/torchinductor_mengqy/5a/c5a7mfs5san3xksnhtsfacpwyppugu2yv6gqkro4flakvdqgojp5.py
# Source Nodes: [add_3, add_5, float_1, mul_1, type_as], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
# add_3 => add_141
# add_5 => add_143
# float_1 => convert_element_type_288
# mul_1 => mul_409
# type_as => convert_element_type_289
triton_per_fused__to_copy_add_mul_sum_25 = async_compile.triton('triton_per_fused__to_copy_add_mul_sum_25', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[2048, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mul_sum_25', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]}
)
@triton.jit
def triton_per_fused__to_copy_add_mul_sum_25(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 2048
    rnumel = 48
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (2048*r1)), rmask, other=0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, None)
''')


# kernel path: /tmp/torchinductor_mengqy/gz/cgzyzfv2majn2telhgojpft7yu33tyq63i6pyxmhhdoopslrqxko.py
# Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add => add_138
# float_1 => convert_element_type_276
# mean => mean_46
# mul => mul_391
# rsqrt => rsqrt_46
triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26 = async_compile.triton('triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tmp0 * tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    tmp15 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp10 = tl.load(in_out_ptr0 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp11 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp28 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp13 = tmp11 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp16 = 2048.0
        tmp17 = tmp15 / tmp16
        tmp18 = 1e-05
        tmp19 = tmp17 + tmp18
        tmp20 = tl.math.rsqrt(tmp19)
        tmp21 = tmp14 * tmp20
        tmp22 = -0.5
        tmp23 = tmp8 * tmp22
        tmp24 = tmp20 * tmp20
        tmp25 = tmp24 * tmp20
        tmp26 = tmp23 * tmp25
        tmp27 = tmp26 / tmp16
        tmp29 = tmp28.to(tl.float32)
        tmp30 = tmp27 * tmp29
        tmp31 = tmp21 + tmp30
        tmp32 = tmp31 + tmp30
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp10 + tmp33
        tl.store(in_out_ptr0 + (r1 + (2048*x0)), tmp34, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/4w/c4wt4fp46qljwq2354qeikaixffp5doqsbo47jlwaro7s4jnvsja.py
# Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
# stack => cat_42, cat_44
# stack_1 => cat_43, cat_45
triton_poi_fused_stack_27 = async_compile.triton('triton_poi_fused_stack_27', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp32', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_27', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 8, 10, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]})
@triton.jit
def triton_poi_fused_stack_27(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 64
    x1 = (xindex // 64) % 16
    x4 = (xindex // 1024)
    x2 = (xindex // 1024) % 2048
    x5 = xindex
    tmp0 = tl.load(in_ptr0 + (2048 + x0 + (128*x1) + (6144*x4)), None).to(tl.float32)
    tmp2 = tl.load(in_ptr1 + ((2*x0) + (128*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr0 + (2112 + x0 + (128*x1) + (6144*x4)), None).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + ((2*x0) + (128*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr0 + (x0 + (128*x1) + (6144*x4)), None).to(tl.float32)
    tmp17 = tl.load(in_ptr0 + (64 + x0 + (128*x1) + (6144*x4)), None).to(tl.float32)
    tmp24 = tl.load(in_ptr3 + (2048 + x0 + (128*x1) + (6144*x4)), None).to(tl.float32)
    tmp27 = tl.load(in_ptr3 + (2112 + x0 + (128*x1) + (6144*x4)), None).to(tl.float32)
    tmp34 = tl.load(in_ptr3 + (x0 + (128*x1) + (6144*x4)), None).to(tl.float32)
    tmp37 = tl.load(in_ptr3 + (64 + x0 + (128*x1) + (6144*x4)), None).to(tl.float32)
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
    tmp25 = tmp24.to(tl.float32)
    tmp26 = tmp25 * tmp3
    tmp28 = tmp27.to(tl.float32)
    tmp29 = tmp28 * tmp8
    tmp30 = tmp26 - tmp29
    tmp31 = tmp28 * tmp3
    tmp32 = tmp25 * tmp8
    tmp33 = tmp31 + tmp32
    tmp35 = tmp34.to(tl.float32)
    tmp36 = tmp35 * tmp3
    tmp38 = tmp37.to(tl.float32)
    tmp39 = tmp38 * tmp8
    tmp40 = tmp36 - tmp39
    tmp41 = tmp38 * tmp3
    tmp42 = tmp35 * tmp8
    tmp43 = tmp41 + tmp42
    tl.store(out_ptr0 + (2*x5), tmp10, None)
    tl.store(out_ptr1 + (2*x5), tmp13, None)
    tl.store(out_ptr2 + (2*x5), tmp20, None)
    tl.store(out_ptr3 + (2*x5), tmp23, None)
    tl.store(out_ptr4 + (2*x5), tmp30, None)
    tl.store(out_ptr5 + (2*x5), tmp33, None)
    tl.store(out_ptr6 + (2*x5), tmp40, None)
    tl.store(out_ptr7 + (2*x5), tmp43, None)
''')


# kernel path: /tmp/torchinductor_mengqy/2i/c2i73qhfkj7sbkqkrdcvh2mmtryajzzy4cu5imcftv4vr5kkdao6.py
# Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
# mul_11 => mul_368, mul_385
# softmax => amax_21, amax_22, convert_element_type_258, convert_element_type_259, convert_element_type_270, convert_element_type_271, div_21, div_22, exp_21, exp_22, sub_65, sub_68, sum_22, sum_23
# where => where_21, where_22
triton_red_fused__softmax_mul_where_28 = async_compile.triton('triton_red_fused__softmax_mul_where_28', '''
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
    meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_mul_where_28', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]}
)
@triton.jit
def triton_red_fused__softmax_mul_where_28(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr4, out_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp29 = tl.full([XBLOCK, RBLOCK], float("-inf"), tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp11 = tl.load(in_ptr0 + (r2 + (2048*x0)), rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr1 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp24 = tl.load(in_ptr3 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp13 = 0.08838834764831843
        tmp14 = tmp12 * tmp13
        tmp17 = tl.where(tmp11, tmp14, tmp16)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 - tmp9
        tmp20 = tl.exp(tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
        tmp25 = tmp24 * tmp13
        tmp26 = tl.where(tmp11, tmp25, tmp16)
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = triton_helpers.maximum(_tmp29, tmp28)
        _tmp29 = tl.where(rmask, tmp30, _tmp29)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp29 = triton_helpers.max2(_tmp29, 1)[:, None]
    tmp35 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp36 = tl.broadcast_to(tmp35, [XBLOCK, RBLOCK])
    _tmp42 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp31 = tl.load(in_ptr0 + (r2 + (2048*x0)), rmask, eviction_policy='evict_last')
        tmp32 = tl.load(in_ptr3 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp44 = tl.load(in_ptr1 + (r2 + (2048*x3)), rmask, other=0).to(tl.float32)
        tmp33 = 0.08838834764831843
        tmp34 = tmp32 * tmp33
        tmp37 = tl.where(tmp31, tmp34, tmp36)
        tmp38 = tmp37.to(tl.float32)
        tmp39 = tmp38 - tmp29
        tmp40 = tl.exp(tmp39)
        tmp41 = tl.broadcast_to(tmp40, [XBLOCK, RBLOCK])
        tmp43 = _tmp42 + tmp41
        _tmp42 = tl.where(rmask, tmp43, _tmp42)
        tmp45 = tmp44 * tmp33
        tmp46 = tl.where(tmp31, tmp45, tmp36)
        tmp47 = tmp46.to(tl.float32)
        tmp48 = tmp47 - tmp9
        tmp49 = tl.exp(tmp48)
        tmp50 = tmp49 / tmp22
        tmp51 = tmp50.to(tl.float32)
        tl.store(out_ptr4 + (r2 + (2048*x3)), tmp51, rmask)
    tmp42 = tl.sum(_tmp42, 1)[:, None]
    tmp56 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp57 = tl.broadcast_to(tmp56, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp52 = tl.load(in_ptr0 + (r2 + (2048*x0)), rmask, eviction_policy='evict_last')
        tmp53 = tl.load(in_ptr3 + (r2 + (2048*x3)), rmask, other=0).to(tl.float32)
        tmp54 = 0.08838834764831843
        tmp55 = tmp53 * tmp54
        tmp58 = tl.where(tmp52, tmp55, tmp57)
        tmp59 = tmp58.to(tl.float32)
        tmp60 = tmp59 - tmp29
        tmp61 = tl.exp(tmp60)
        tmp62 = tmp61 / tmp42
        tmp63 = tmp62.to(tl.float32)
        tl.store(out_ptr5 + (r2 + (2048*x3)), tmp63, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/4k/c4kbhktubk6fs72iqo2dyb36xuxajzbuzqfigzgsv7qjqefrnhep.py
# Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# mul_15 => mul_390
# silu => convert_element_type_274, convert_element_type_275, mul_389, sigmoid_22
triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29 = async_compile.triton('triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]})
@triton.jit
def triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, out_ptr2, xnumel, XBLOCK : tl.constexpr):
    xnumel = 34603008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp5 = tl.load(in_ptr1 + (x0), None).to(tl.float32)
    tmp7 = tl.load(in_ptr2 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp2 = tl.sigmoid(tmp1)
    tmp3 = tmp1 * tmp2
    tmp4 = tmp3.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tmp8 = tmp7 * tmp4
    tmp9 = tmp7 * tmp5
    tmp10 = tl.sigmoid(tmp0)
    tmp11 = 1.0
    tmp12 = tmp11 - tmp10
    tmp13 = tmp0 * tmp12
    tmp14 = tmp13 + tmp11
    tmp15 = tmp10 * tmp14
    tmp16 = tmp9 * tmp15
    tl.store(out_ptr0 + (x0), tmp6, None)
    tl.store(out_ptr1 + (x0), tmp8, None)
    tl.store(out_ptr2 + (x0), tmp16, None)
''')


# kernel path: /tmp/torchinductor_mengqy/5j/c5jxhbcelvonogj2vbwzvoqvjba3jw7pd5ghuc744fk5urdtgniv.py
# Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add_3 => add_135
# add_4 => add_136
# float_4 => convert_element_type_272
# mean_1 => mean_45
# mul_12 => mul_386
# rsqrt_1 => rsqrt_45
triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30 = async_compile.triton('triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]}
)
@triton.jit
def triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 6144
    rnumel = 2048
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp0 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp6 = tl.load(in_ptr3 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp7 = tl.load(in_ptr4 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp2 * tmp3
        tmp5 = tmp4.to(tl.float32)
        tmp8 = tmp6 + tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp5 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tmp20 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp14 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp17 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp33 = tl.load(in_ptr3 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp34 = tl.load(in_ptr4 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp40 = tl.load(in_out_ptr0 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp16 = tmp14 + tmp15
        tmp18 = tmp16 * tmp17
        tmp19 = tmp18.to(tl.float32)
        tmp21 = 2048.0
        tmp22 = tmp20 / tmp21
        tmp23 = 1e-05
        tmp24 = tmp22 + tmp23
        tmp25 = tl.math.rsqrt(tmp24)
        tmp26 = tmp19 * tmp25
        tmp27 = -0.5
        tmp28 = tmp12 * tmp27
        tmp29 = tmp25 * tmp25
        tmp30 = tmp29 * tmp25
        tmp31 = tmp28 * tmp30
        tmp32 = tmp31 / tmp21
        tmp35 = tmp33 + tmp34
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tmp32 * tmp36
        tmp38 = tmp26 + tmp37
        tmp39 = tmp38 + tmp37
        tmp41 = tmp39.to(tl.float32)
        tmp42 = tmp40 + tmp41
        tl.store(in_out_ptr0 + (r1 + (2048*x0)), tmp42, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/pg/cpgv3cgsokqozh5dkiadluealbq454ntugtbcplxvbqezocrbwo3.py
# Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add => add_132
# add_3 => add_135
# add_4 => add_136
# float_1 => convert_element_type_264
# float_4 => convert_element_type_272
# mean => mean_44
# mean_1 => mean_45
# mul => mul_374
# mul_1 => mul_375
# mul_12 => mul_386
# mul_13 => mul_387
# rsqrt => rsqrt_44
# rsqrt_1 => rsqrt_45
# type_as => convert_element_type_265
# type_as_3 => convert_element_type_273
triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[131072, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp16', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr0, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 98304
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 2048
    x1 = (xindex // 2048)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp29 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (2048*r2) + (262144*x1)), rmask, other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + (2048*r2) + (262144*x1)), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + (2048*r2) + (262144*x1)), rmask, other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (x0 + (2048*r2) + (262144*x1)), rmask, other=0).to(tl.float32)
        tmp7 = tl.load(in_ptr4 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp19 = tl.load(in_ptr5 + (x0 + (2048*r2) + (262144*x1)), rmask, other=0).to(tl.float32)
        tmp21 = tl.load(in_ptr6 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp8 = 2048.0
        tmp9 = tmp7 / tmp8
        tmp10 = 1e-05
        tmp11 = tmp9 + tmp10
        tmp12 = tl.math.rsqrt(tmp11)
        tmp13 = tmp6 * tmp12
        tmp14 = tmp13.to(tl.float32)
        tmp15 = tmp2 * tmp14
        tmp16 = tl.broadcast_to(tmp15, [XBLOCK, RBLOCK])
        tmp18 = _tmp17 + tmp16
        _tmp17 = tl.where(rmask, tmp18, _tmp17)
        tmp20 = tmp3.to(tl.float32)
        tmp22 = tmp21 / tmp8
        tmp23 = tmp22 + tmp10
        tmp24 = tl.math.rsqrt(tmp23)
        tmp25 = tmp20 * tmp24
        tmp26 = tmp25.to(tl.float32)
        tmp27 = tmp19 * tmp26
        tmp28 = tl.broadcast_to(tmp27, [XBLOCK, RBLOCK])
        tmp30 = _tmp29 + tmp28
        _tmp29 = tl.where(rmask, tmp30, _tmp29)
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, None)
    tmp29 = tl.sum(_tmp29, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp29, None)
''')


# kernel path: /tmp/torchinductor_mengqy/on/con7h7elt6js3b5oh5hj4mypt7qqwn56hux5iv26mil7kc5fjp2d.py
# Source Nodes: [cross_entropy, mul_11, softmax, where], Original ATen: [aten._softmax, aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
# cross_entropy => scalar_tensor_25
# mul_11 => mul_11
# softmax => amax, convert_element_type_6, convert_element_type_7, div, exp, sub_2, sum_1
# where => where
triton_red_fused__softmax__softmax_backward_data_mul_nll_loss_forward_where_32 = async_compile.triton('triton_red_fused__softmax__softmax_backward_data_mul_nll_loss_forward_where_32', '''
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
    meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax__softmax_backward_data_mul_nll_loss_forward_where_32', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]}
)
@triton.jit
def triton_red_fused__softmax__softmax_backward_data_mul_nll_loss_forward_where_32(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
    _tmp30 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp11 = tl.load(in_ptr0 + (r2 + (2048*x0)), rmask, eviction_policy='evict_last')
        tmp12 = tl.load(in_ptr1 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp24 = tl.load(in_ptr3 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp26 = tl.load(in_ptr4 + (r2 + (2048*x3)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp13 = 0.08838834764831843
        tmp14 = tmp12 * tmp13
        tmp17 = tl.where(tmp11, tmp14, tmp16)
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp18 - tmp9
        tmp20 = tl.exp(tmp19)
        tmp21 = tl.broadcast_to(tmp20, [XBLOCK, RBLOCK])
        tmp23 = _tmp22 + tmp21
        _tmp22 = tl.where(rmask, tmp23, _tmp22)
        tmp25 = tmp24.to(tl.float32)
        tmp27 = tmp26.to(tl.float32)
        tmp28 = tmp25 * tmp27
        tmp29 = tl.broadcast_to(tmp28, [XBLOCK, RBLOCK])
        tmp31 = _tmp30 + tmp29
        _tmp30 = tl.where(rmask, tmp31, _tmp30)
    tmp22 = tl.sum(_tmp22, 1)[:, None]
    tmp30 = tl.sum(_tmp30, 1)[:, None]
    tmp47 = tl.load(in_ptr2 + (0)).to(tl.float32)
    tmp48 = tl.broadcast_to(tmp47, [XBLOCK, RBLOCK])
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp32 = tl.load(in_ptr0 + (r2 + (2048*x0)), rmask, eviction_policy='evict_last')
        tmp33 = tl.load(in_ptr3 + (r2 + (2048*x3)), rmask, other=0).to(tl.float32)
        tmp35 = tl.load(in_ptr4 + (r2 + (2048*x3)), rmask, other=0).to(tl.float32)
        tmp45 = tl.load(in_ptr1 + (r2 + (2048*x3)), rmask, other=0).to(tl.float32)
        tmp34 = tmp33.to(tl.float32)
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tmp34 * tmp36
        tmp38 = tmp36 * tmp30
        tmp39 = tmp37 - tmp38
        tmp40 = tmp39.to(tl.float32)
        tmp41 = 0.0
        tmp42 = tl.where(tmp32, tmp40, tmp41)
        tmp43 = 0.08838834764831843
        tmp44 = tmp42 * tmp43
        tmp46 = tmp45 * tmp43
        tmp49 = tl.where(tmp32, tmp46, tmp48)
        tmp50 = tmp49.to(tl.float32)
        tmp51 = tmp50 - tmp9
        tmp52 = tl.exp(tmp51)
        tmp53 = tmp52 / tmp22
        tmp54 = tmp53.to(tl.float32)
        tl.store(out_ptr3 + (r2 + (2048*x3)), tmp44, rmask)
        tl.store(out_ptr4 + (r2 + (2048*x3)), tmp54, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/5p/c5plcyrvwxzkcvxwrpjmymodnht2y7kliczga7ca42awxgfcbs2m.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_33 = async_compile.triton('triton_poi_fused_embedding_dense_backward_33', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_33', 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]})
@triton.jit
def triton_poi_fused_embedding_dense_backward_33(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102926336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/pu/cpuxgjiilpgnwbz77tlg25gba52umtopcrlgrld5ucswlt2glll3.py
# Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.embedding_dense_backward, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add => add
# float_1 => convert_element_type
# mean => mean
# mul => mul
# rsqrt => rsqrt
triton_red_fused__to_copy_add_div_embedding_dense_backward_mean_mul_rsqrt_sum_34 = async_compile.triton('triton_red_fused__to_copy_add_div_embedding_dense_backward_mean_mul_rsqrt_sum_34', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*i64', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_div_embedding_dense_backward_mean_mul_rsqrt_sum_34', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]}
)
@triton.jit
def triton_red_fused__to_copy_add_div_embedding_dense_backward_mean_mul_rsqrt_sum_34(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp2 = tmp0 * tmp1
        tmp3 = tmp2.to(tl.float32)
        tmp5 = tmp4.to(tl.float32)
        tmp6 = tmp3 * tmp5
        tmp7 = tl.broadcast_to(tmp6, [XBLOCK, RBLOCK])
        tmp9 = _tmp8 + tmp7
        _tmp8 = tl.where(rmask, tmp9, _tmp8)
    tmp8 = tl.sum(_tmp8, 1)[:, None]
    x2 = xindex % 2048
    x3 = (xindex // 2048)
    tmp10 = tl.load(in_ptr3 + (x2 + (2049*x3)), None, eviction_policy='evict_last')
    tmp18 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r1 = rindex
        tmp13 = tl.load(in_ptr4 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp14 = tl.load(in_ptr0 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp15 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp31 = tl.load(in_ptr2 + (r1 + (2048*x0)), rmask, other=0).to(tl.float32)
        tmp11 = tl.full([1, 1], -1, tl.int64)
        tmp12 = tmp10 == tmp11
        tmp16 = tmp14 * tmp15
        tmp17 = tmp16.to(tl.float32)
        tmp19 = 2048.0
        tmp20 = tmp18 / tmp19
        tmp21 = 1e-05
        tmp22 = tmp20 + tmp21
        tmp23 = tl.math.rsqrt(tmp22)
        tmp24 = tmp17 * tmp23
        tmp25 = -0.5
        tmp26 = tmp8 * tmp25
        tmp27 = tmp23 * tmp23
        tmp28 = tmp27 * tmp23
        tmp29 = tmp26 * tmp28
        tmp30 = tmp29 / tmp19
        tmp32 = tmp31.to(tl.float32)
        tmp33 = tmp30 * tmp32
        tmp34 = tmp24 + tmp33
        tmp35 = tmp34 + tmp33
        tmp36 = tmp35.to(tl.float32)
        tmp37 = tmp13 + tmp36
        tmp38 = tmp37.to(tl.float32)
        tmp39 = 0.0
        tmp40 = tl.where(tmp12, tmp39, tmp38)
        tmp41 = tl.where(tmp10 < 0, tmp10 + 50257, tmp10)
        tl.atomic_add(out_ptr2 + (tl.broadcast_to(r1 + (2048*tmp41), [XBLOCK, RBLOCK])), tmp40, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/v3/cv3rmqdcevkzvln76xcflv7ole6r4rnrfpf5jgplnbvlkdto2p62.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_35 = async_compile.triton('triton_poi_fused_embedding_dense_backward_35', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[134217728], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_35', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_embedding_dense_backward_35(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 102926336
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, None)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_1, primals_4, primals_8, primals_11, primals_15, primals_18, primals_22, primals_25, primals_29, primals_32, primals_36, primals_39, primals_43, primals_46, primals_50, primals_53, primals_57, primals_60, primals_64, primals_67, primals_71, primals_74, primals_78, primals_81, primals_85, primals_88, primals_92, primals_95, primals_99, primals_102, primals_106, primals_109, primals_113, primals_116, primals_120, primals_123, primals_127, primals_130, primals_134, primals_137, primals_141, primals_144, primals_148, primals_151, primals_155, primals_158, primals_162, primals_165, primals_169, primals_174, primals_175, embedding, permute, select_1, select_3, slice_3, scalar_tensor, permute_8, permute_9, permute_10, add_5, permute_12, permute_20, permute_21, permute_22, add_11, permute_24, permute_32, permute_33, permute_34, add_17, permute_36, permute_44, permute_45, permute_46, add_23, permute_48, permute_56, permute_57, permute_58, add_29, permute_60, permute_68, permute_69, permute_70, add_35, permute_72, permute_80, permute_81, permute_82, add_41, permute_84, permute_92, permute_93, permute_94, add_47, permute_96, permute_104, permute_105, permute_106, add_53, permute_108, permute_116, permute_117, permute_118, add_59, permute_120, permute_128, permute_129, permute_130, add_65, permute_132, permute_140, permute_141, permute_142, add_71, permute_144, permute_152, permute_153, permute_154, add_77, permute_156, permute_164, permute_165, permute_166, add_83, permute_168, permute_176, permute_177, permute_178, add_89, permute_180, permute_188, permute_189, permute_190, add_95, permute_192, permute_200, permute_201, permute_202, add_101, permute_204, permute_212, permute_213, permute_214, add_107, permute_216, permute_224, permute_225, permute_226, add_113, permute_228, permute_236, permute_237, permute_238, add_119, permute_240, permute_248, permute_249, permute_250, add_125, permute_252, permute_260, permute_261, permute_262, add_131, permute_264, permute_272, permute_273, permute_274, add_137, permute_276, permute_284, permute_285, permute_286, permute_287, rsqrt_48, view_624, convert_element_type_291, convert_element_type_292, permute_291, permute_327, permute_359, permute_391, permute_423, permute_455, permute_487, permute_519, permute_551, permute_583, permute_615, permute_647, permute_679, permute_711, permute_743, permute_775, permute_807, permute_839, permute_871, permute_903, permute_935, permute_967, permute_999, permute_1031, tangents_1 = args
    args.clear()
    assert_size_stride(primals_1, (2048, ), (1, ))
    assert_size_stride(primals_4, (2048, ), (1, ))
    assert_size_stride(primals_8, (2048, ), (1, ))
    assert_size_stride(primals_11, (2048, ), (1, ))
    assert_size_stride(primals_15, (2048, ), (1, ))
    assert_size_stride(primals_18, (2048, ), (1, ))
    assert_size_stride(primals_22, (2048, ), (1, ))
    assert_size_stride(primals_25, (2048, ), (1, ))
    assert_size_stride(primals_29, (2048, ), (1, ))
    assert_size_stride(primals_32, (2048, ), (1, ))
    assert_size_stride(primals_36, (2048, ), (1, ))
    assert_size_stride(primals_39, (2048, ), (1, ))
    assert_size_stride(primals_43, (2048, ), (1, ))
    assert_size_stride(primals_46, (2048, ), (1, ))
    assert_size_stride(primals_50, (2048, ), (1, ))
    assert_size_stride(primals_53, (2048, ), (1, ))
    assert_size_stride(primals_57, (2048, ), (1, ))
    assert_size_stride(primals_60, (2048, ), (1, ))
    assert_size_stride(primals_64, (2048, ), (1, ))
    assert_size_stride(primals_67, (2048, ), (1, ))
    assert_size_stride(primals_71, (2048, ), (1, ))
    assert_size_stride(primals_74, (2048, ), (1, ))
    assert_size_stride(primals_78, (2048, ), (1, ))
    assert_size_stride(primals_81, (2048, ), (1, ))
    assert_size_stride(primals_85, (2048, ), (1, ))
    assert_size_stride(primals_88, (2048, ), (1, ))
    assert_size_stride(primals_92, (2048, ), (1, ))
    assert_size_stride(primals_95, (2048, ), (1, ))
    assert_size_stride(primals_99, (2048, ), (1, ))
    assert_size_stride(primals_102, (2048, ), (1, ))
    assert_size_stride(primals_106, (2048, ), (1, ))
    assert_size_stride(primals_109, (2048, ), (1, ))
    assert_size_stride(primals_113, (2048, ), (1, ))
    assert_size_stride(primals_116, (2048, ), (1, ))
    assert_size_stride(primals_120, (2048, ), (1, ))
    assert_size_stride(primals_123, (2048, ), (1, ))
    assert_size_stride(primals_127, (2048, ), (1, ))
    assert_size_stride(primals_130, (2048, ), (1, ))
    assert_size_stride(primals_134, (2048, ), (1, ))
    assert_size_stride(primals_137, (2048, ), (1, ))
    assert_size_stride(primals_141, (2048, ), (1, ))
    assert_size_stride(primals_144, (2048, ), (1, ))
    assert_size_stride(primals_148, (2048, ), (1, ))
    assert_size_stride(primals_151, (2048, ), (1, ))
    assert_size_stride(primals_155, (2048, ), (1, ))
    assert_size_stride(primals_158, (2048, ), (1, ))
    assert_size_stride(primals_162, (2048, ), (1, ))
    assert_size_stride(primals_165, (2048, ), (1, ))
    assert_size_stride(primals_169, (2048, ), (1, ))
    assert_size_stride(primals_174, (3, 2048), (2049, 1))
    assert_size_stride(primals_175, (3, 2048), (2049, 1))
    assert_size_stride(embedding, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute, (2048, 6144), (1, 2048))
    assert_size_stride(select_1, (1, 2048, 1, 64), (0, 128, 0, 2))
    assert_size_stride(select_3, (1, 2048, 1, 64), (0, 128, 0, 2))
    assert_size_stride(slice_3, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(scalar_tensor, (), ())
    assert_size_stride(permute_8, (2048, 2048), (1, 2048))
    assert_size_stride(permute_9, (2048, 5632), (1, 2048))
    assert_size_stride(permute_10, (2048, 5632), (1, 2048))
    assert_size_stride(add_5, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_12, (2048, 6144), (1, 2048))
    assert_size_stride(permute_20, (2048, 2048), (1, 2048))
    assert_size_stride(permute_21, (2048, 5632), (1, 2048))
    assert_size_stride(permute_22, (2048, 5632), (1, 2048))
    assert_size_stride(add_11, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_24, (2048, 6144), (1, 2048))
    assert_size_stride(permute_32, (2048, 2048), (1, 2048))
    assert_size_stride(permute_33, (2048, 5632), (1, 2048))
    assert_size_stride(permute_34, (2048, 5632), (1, 2048))
    assert_size_stride(add_17, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_36, (2048, 6144), (1, 2048))
    assert_size_stride(permute_44, (2048, 2048), (1, 2048))
    assert_size_stride(permute_45, (2048, 5632), (1, 2048))
    assert_size_stride(permute_46, (2048, 5632), (1, 2048))
    assert_size_stride(add_23, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_48, (2048, 6144), (1, 2048))
    assert_size_stride(permute_56, (2048, 2048), (1, 2048))
    assert_size_stride(permute_57, (2048, 5632), (1, 2048))
    assert_size_stride(permute_58, (2048, 5632), (1, 2048))
    assert_size_stride(add_29, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_60, (2048, 6144), (1, 2048))
    assert_size_stride(permute_68, (2048, 2048), (1, 2048))
    assert_size_stride(permute_69, (2048, 5632), (1, 2048))
    assert_size_stride(permute_70, (2048, 5632), (1, 2048))
    assert_size_stride(add_35, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_72, (2048, 6144), (1, 2048))
    assert_size_stride(permute_80, (2048, 2048), (1, 2048))
    assert_size_stride(permute_81, (2048, 5632), (1, 2048))
    assert_size_stride(permute_82, (2048, 5632), (1, 2048))
    assert_size_stride(add_41, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_84, (2048, 6144), (1, 2048))
    assert_size_stride(permute_92, (2048, 2048), (1, 2048))
    assert_size_stride(permute_93, (2048, 5632), (1, 2048))
    assert_size_stride(permute_94, (2048, 5632), (1, 2048))
    assert_size_stride(add_47, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_96, (2048, 6144), (1, 2048))
    assert_size_stride(permute_104, (2048, 2048), (1, 2048))
    assert_size_stride(permute_105, (2048, 5632), (1, 2048))
    assert_size_stride(permute_106, (2048, 5632), (1, 2048))
    assert_size_stride(add_53, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_108, (2048, 6144), (1, 2048))
    assert_size_stride(permute_116, (2048, 2048), (1, 2048))
    assert_size_stride(permute_117, (2048, 5632), (1, 2048))
    assert_size_stride(permute_118, (2048, 5632), (1, 2048))
    assert_size_stride(add_59, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_120, (2048, 6144), (1, 2048))
    assert_size_stride(permute_128, (2048, 2048), (1, 2048))
    assert_size_stride(permute_129, (2048, 5632), (1, 2048))
    assert_size_stride(permute_130, (2048, 5632), (1, 2048))
    assert_size_stride(add_65, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_132, (2048, 6144), (1, 2048))
    assert_size_stride(permute_140, (2048, 2048), (1, 2048))
    assert_size_stride(permute_141, (2048, 5632), (1, 2048))
    assert_size_stride(permute_142, (2048, 5632), (1, 2048))
    assert_size_stride(add_71, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_144, (2048, 6144), (1, 2048))
    assert_size_stride(permute_152, (2048, 2048), (1, 2048))
    assert_size_stride(permute_153, (2048, 5632), (1, 2048))
    assert_size_stride(permute_154, (2048, 5632), (1, 2048))
    assert_size_stride(add_77, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_156, (2048, 6144), (1, 2048))
    assert_size_stride(permute_164, (2048, 2048), (1, 2048))
    assert_size_stride(permute_165, (2048, 5632), (1, 2048))
    assert_size_stride(permute_166, (2048, 5632), (1, 2048))
    assert_size_stride(add_83, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_168, (2048, 6144), (1, 2048))
    assert_size_stride(permute_176, (2048, 2048), (1, 2048))
    assert_size_stride(permute_177, (2048, 5632), (1, 2048))
    assert_size_stride(permute_178, (2048, 5632), (1, 2048))
    assert_size_stride(add_89, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_180, (2048, 6144), (1, 2048))
    assert_size_stride(permute_188, (2048, 2048), (1, 2048))
    assert_size_stride(permute_189, (2048, 5632), (1, 2048))
    assert_size_stride(permute_190, (2048, 5632), (1, 2048))
    assert_size_stride(add_95, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_192, (2048, 6144), (1, 2048))
    assert_size_stride(permute_200, (2048, 2048), (1, 2048))
    assert_size_stride(permute_201, (2048, 5632), (1, 2048))
    assert_size_stride(permute_202, (2048, 5632), (1, 2048))
    assert_size_stride(add_101, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_204, (2048, 6144), (1, 2048))
    assert_size_stride(permute_212, (2048, 2048), (1, 2048))
    assert_size_stride(permute_213, (2048, 5632), (1, 2048))
    assert_size_stride(permute_214, (2048, 5632), (1, 2048))
    assert_size_stride(add_107, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_216, (2048, 6144), (1, 2048))
    assert_size_stride(permute_224, (2048, 2048), (1, 2048))
    assert_size_stride(permute_225, (2048, 5632), (1, 2048))
    assert_size_stride(permute_226, (2048, 5632), (1, 2048))
    assert_size_stride(add_113, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_228, (2048, 6144), (1, 2048))
    assert_size_stride(permute_236, (2048, 2048), (1, 2048))
    assert_size_stride(permute_237, (2048, 5632), (1, 2048))
    assert_size_stride(permute_238, (2048, 5632), (1, 2048))
    assert_size_stride(add_119, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_240, (2048, 6144), (1, 2048))
    assert_size_stride(permute_248, (2048, 2048), (1, 2048))
    assert_size_stride(permute_249, (2048, 5632), (1, 2048))
    assert_size_stride(permute_250, (2048, 5632), (1, 2048))
    assert_size_stride(add_125, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_252, (2048, 6144), (1, 2048))
    assert_size_stride(permute_260, (2048, 2048), (1, 2048))
    assert_size_stride(permute_261, (2048, 5632), (1, 2048))
    assert_size_stride(permute_262, (2048, 5632), (1, 2048))
    assert_size_stride(add_131, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_264, (2048, 6144), (1, 2048))
    assert_size_stride(permute_272, (2048, 2048), (1, 2048))
    assert_size_stride(permute_273, (2048, 5632), (1, 2048))
    assert_size_stride(permute_274, (2048, 5632), (1, 2048))
    assert_size_stride(add_137, (3, 2048, 2048), (4194304, 2048, 1))
    assert_size_stride(permute_276, (2048, 6144), (1, 2048))
    assert_size_stride(permute_284, (2048, 2048), (1, 2048))
    assert_size_stride(permute_285, (2048, 5632), (1, 2048))
    assert_size_stride(permute_286, (2048, 5632), (1, 2048))
    assert_size_stride(permute_287, (5632, 2048), (1, 5632))
    assert_size_stride(rsqrt_48, (3, 2048, 1), (2048, 1, 1))
    assert_size_stride(view_624, (6144, 2048), (2048, 1))
    assert_size_stride(convert_element_type_291, (6144, 50257), (50257, 1))
    assert_size_stride(convert_element_type_292, (), ())
    assert_size_stride(permute_291, (50257, 2048), (2048, 1))
    assert_size_stride(permute_327, (2048, 5632), (5632, 1))
    assert_size_stride(permute_359, (2048, 5632), (5632, 1))
    assert_size_stride(permute_391, (2048, 5632), (5632, 1))
    assert_size_stride(permute_423, (2048, 5632), (5632, 1))
    assert_size_stride(permute_455, (2048, 5632), (5632, 1))
    assert_size_stride(permute_487, (2048, 5632), (5632, 1))
    assert_size_stride(permute_519, (2048, 5632), (5632, 1))
    assert_size_stride(permute_551, (2048, 5632), (5632, 1))
    assert_size_stride(permute_583, (2048, 5632), (5632, 1))
    assert_size_stride(permute_615, (2048, 5632), (5632, 1))
    assert_size_stride(permute_647, (2048, 5632), (5632, 1))
    assert_size_stride(permute_679, (2048, 5632), (5632, 1))
    assert_size_stride(permute_711, (2048, 5632), (5632, 1))
    assert_size_stride(permute_743, (2048, 5632), (5632, 1))
    assert_size_stride(permute_775, (2048, 5632), (5632, 1))
    assert_size_stride(permute_807, (2048, 5632), (5632, 1))
    assert_size_stride(permute_839, (2048, 5632), (5632, 1))
    assert_size_stride(permute_871, (2048, 5632), (5632, 1))
    assert_size_stride(permute_903, (2048, 5632), (5632, 1))
    assert_size_stride(permute_935, (2048, 5632), (5632, 1))
    assert_size_stride(permute_967, (2048, 5632), (5632, 1))
    assert_size_stride(permute_999, (2048, 5632), (5632, 1))
    assert_size_stride(permute_1031, (2048, 5632), (5632, 1))
    assert_size_stride(tangents_1, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((6144, 50257), (50257, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_0.run(buf0, 308779008, grid=grid(308779008), stream=stream0)
        # Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_1.run(primals_175, buf0, 6144, grid=grid(6144), stream=stream0)
        buf3 = empty_strided((6144, 50257), (50257, 1), device='cuda', dtype=torch.float16)
        buf10 = empty_strided((6144, 50264), (50264, 1), device='cuda', dtype=torch.float16)
        buf8 = reinterpret_tensor(buf10, (6144, 50257), (50264, 1), 0)  # alias
        # Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2.run(buf0, primals_175, tangents_1, convert_element_type_292, convert_element_type_291, buf3, buf8, 6144, 50257, grid=grid(6144), stream=stream0)
        del buf0
        del convert_element_type_291
        del convert_element_type_292
        del primals_175
        del tangents_1
        buf6 = empty_strided((50264, 6144), (6144, 1), device='cuda', dtype=torch.float16)
        buf4 = reinterpret_tensor(buf6, (50257, 6144), (6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf3, buf4, 50257, 6144, grid=grid(50257, 6144), stream=stream0)
        del buf3
        buf5 = reinterpret_tensor(buf6, (7, 6144), (6144, 1), 308779008)  # alias
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(buf5, 43008, grid=grid(43008), stream=stream0)
        del buf4
        del buf5
        buf7 = empty_strided((50264, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf6, view_624, out=buf7)
        del buf6
        del view_624
        buf9 = reinterpret_tensor(buf10, (6144, 7), (50264, 1), 50257)  # alias
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(buf9, 43008, grid=grid(43008), stream=stream0)
        buf13 = empty_strided((50264, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        buf11 = reinterpret_tensor(buf13, (50257, 2048), (2048, 1), 0)  # alias
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_6.run(permute_291, buf11, 102926336, grid=grid(102926336), stream=stream0)
        del buf8
        del buf9
        del permute_291
        buf12 = reinterpret_tensor(buf13, (7, 2048), (2048, 1), 102926336)  # alias
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_7.run(buf12, 14336, grid=grid(14336), stream=stream0)
        del buf11
        del buf12
        buf14 = empty_strided((6144, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf10, buf13, out=buf14)
        del buf10
        del buf13
        buf15 = empty_strided((3, 2048, 1), (2048, 1, 6144), device='cuda', dtype=torch.float32)
        buf16 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_137, primals_162, buf15, buf16, 6144, 2048, grid=grid(6144), stream=stream0)
        buf17 = empty_strided((6144, 6144), (6144, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf16, (6144, 2048), (2048, 1), 0), permute_276, out=buf17)
        buf20 = empty_strided((3, 2048, 16, 64, 2), (4194304, 2048, 128, 2, 1), device='cuda', dtype=torch.float32)
        buf18 = reinterpret_tensor(buf20, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf19 = reinterpret_tensor(buf20, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf23 = empty_strided((3, 2048, 16, 64, 2), (4194304, 2048, 128, 2, 1), device='cuda', dtype=torch.float32)
        buf21 = reinterpret_tensor(buf23, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf22 = reinterpret_tensor(buf23, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_9.run(buf17, select_1, select_3, buf18, buf19, buf21, buf22, 6291456, grid=grid(6291456), stream=stream0)
        buf24 = empty_strided((3, 16, 2048, 128), (4194304, 262144, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf23, buf24, 12582912, grid=grid(12582912), stream=stream0)
        del buf18
        del buf19
        del buf21
        del buf22
        buf25 = empty_strided((3, 16, 128, 2048), (4194304, 262144, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf20, buf25, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf26 = empty_strided((48, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf24, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf25, (48, 128, 2048), (262144, 2048, 1), 0), out=buf26)
        buf29 = empty_strided((3, 16, 2048, 2048), (67108864, 4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_12.run(slice_3, buf26, scalar_tensor, buf29, 98304, 2048, grid=grid(98304), stream=stream0)
        buf30 = empty_strided((3, 16, 2048, 128), (4194304, 262144, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf17, buf30, 12582912, grid=grid(12582912), stream=stream0)
        buf31 = empty_strided((48, 2048, 128), (262144, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf29, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf30, (48, 2048, 128), (262144, 128, 1), 0), out=buf31)
        buf32 = empty_strided((6144, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf31, buf32, 12582912, grid=grid(12582912), stream=stream0)
        buf33 = reinterpret_tensor(buf31, (6144, 2048), (2048, 1)); del buf31  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf32, permute_284, out=buf33)
        buf34 = empty_strided((3, 2048, 1), (2048, 1, 6144), device='cuda', dtype=torch.float32)
        buf35 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_137, buf33, primals_165, buf34, buf35, 6144, 2048, grid=grid(6144), stream=stream0)
        buf36 = empty_strided((6144, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (6144, 2048), (2048, 1), 0), permute_285, out=buf36)
        buf37 = empty_strided((6144, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (6144, 2048), (2048, 1), 0), permute_286, out=buf37)
        buf38 = empty_strided((3, 2048, 5632), (11534336, 5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_16.run(buf36, buf37, buf38, 34603008, grid=grid(34603008), stream=stream0)
        buf39 = empty_strided((6144, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf38, (6144, 5632), (5632, 1), 0), permute_287, out=buf39)
        buf44 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_5, float_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.sum]
        triton_red_fused__to_copy_add_div_mul_sum_17.run(buf14, primals_169, add_137, buf33, buf39, rsqrt_48, buf44, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_169
        buf46 = empty_strided((6144, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_287, (2048, 5632), (5632, 1), 0), out=buf46)
        del permute_287
        buf47 = empty_strided((3, 2048, 5632), (11534336, 5632, 1), device='cuda', dtype=torch.float16)
        buf50 = reinterpret_tensor(buf37, (3, 2048, 5632), (11534336, 5632, 1)); del buf37  # reuse
        # Source Nodes: [silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_18.run(buf50, buf46, buf36, buf47, 34603008, grid=grid(34603008), stream=stream0)
        buf49 = empty_strided((6144, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_286, (5632, 2048), (2048, 1), 0), out=buf49)
        del permute_286
        buf52 = empty_strided((6144, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_285, (5632, 2048), (2048, 1), 0), out=buf52)
        del permute_285
        buf57 = empty_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_19.run(buf49, buf52, primals_165, add_137, buf33, buf34, buf44, buf57, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_165
        buf59 = empty_strided((6144, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf57, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_284, (2048, 2048), (2048, 1), 0), out=buf59)
        del permute_284
        buf60 = empty_strided((3, 16, 2048, 128), (4194304, 262144, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf59, buf60, 12582912, grid=grid(12582912), stream=stream0)
        buf62 = buf26; del buf26  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf60, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf30, (48, 128, 2048), (262144, 1, 128), 0), out=buf62)
        buf64 = empty_strided((3, 16, 2048, 2048), (67108864, 4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf62, buf29, slice_3, buf64, 98304, 2048, grid=grid(98304), stream=stream0)
        buf65 = reinterpret_tensor(buf30, (48, 128, 2048), (262144, 2048, 1)); del buf30  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf24, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf64, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf65)
        buf66 = reinterpret_tensor(buf24, (48, 2048, 128), (262144, 128, 1)); del buf24  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf64, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf25, (48, 2048, 128), (262144, 1, 2048), 0), out=buf66)
        buf72 = reinterpret_tensor(buf17, (3, 2048, 6144), (12582912, 6144, 1)); del buf17  # reuse
        buf70 = reinterpret_tensor(buf72, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf69 = reinterpret_tensor(buf72, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf65, select_3, select_1, buf66, buf70, buf69, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf61 = buf66; del buf66  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf29, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf60, (48, 2048, 128), (262144, 128, 1), 0), out=buf61)
        buf71 = reinterpret_tensor(buf72, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf61, buf71, 12582912, grid=grid(12582912), stream=stream0)
        del buf69
        del buf70
        del buf71
        buf74 = reinterpret_tensor(buf61, (6144, 2048), (2048, 1)); del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_276, (6144, 2048), (2048, 1), 0), out=buf74)
        del permute_276
        buf40 = empty_strided((1, 1, 2048, 48), (98304, 98304, 1, 2048), device='cuda', dtype=torch.float32)
        buf53 = empty_strided((1, 1, 2048, 48), (98304, 98304, 1, 2048), device='cuda', dtype=torch.float32)
        buf75 = empty_strided((1, 1, 2048, 48), (98304, 98304, 1, 2048), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, add_3, add_4, add_5, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_24.run(buf14, add_137, buf33, buf39, rsqrt_48, buf49, buf52, buf34, buf74, buf15, buf40, buf53, buf75, 98304, 128, grid=grid(98304), stream=stream0)
        del rsqrt_48
        buf41 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_5, float_1, mul_1, type_as], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf40, buf41, 2048, 48, grid=grid(2048), stream=stream0)
        del buf40
        buf45 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf44, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf38, (6144, 5632), (5632, 1), 0), out=buf45)
        buf48 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf47, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf35, (6144, 2048), (2048, 1), 0), out=buf48)
        buf51 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf50, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf35, (6144, 2048), (2048, 1), 0), out=buf51)
        buf54 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf53, buf54, 2048, 48, grid=grid(2048), stream=stream0)
        buf58 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf57, (2048, 6144), (1, 2048), 0), buf32, out=buf58)
        buf73 = buf32; del buf32  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf72, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf16, (6144, 2048), (2048, 1), 0), out=buf73)
        buf76 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf75, buf76, 2048, 48, grid=grid(2048), stream=stream0)
        buf78 = buf57; del buf57  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf78, buf74, primals_162, add_137, buf15, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_137
        del primals_162
        buf79 = buf15; del buf15  # reuse
        buf80 = reinterpret_tensor(buf74, (3, 2048, 2048), (4194304, 2048, 1)); del buf74  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_131, primals_155, buf79, buf80, 6144, 2048, grid=grid(6144), stream=stream0)
        buf81 = reinterpret_tensor(buf72, (6144, 6144), (6144, 1)); del buf72  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf80, (6144, 2048), (2048, 1), 0), permute_264, out=buf81)
        buf137 = buf34; del buf34  # reuse
        buf138 = buf16; del buf16  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_125, primals_148, buf137, buf138, 6144, 2048, grid=grid(6144), stream=stream0)
        buf139 = empty_strided((6144, 6144), (6144, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf138, (6144, 2048), (2048, 1), 0), permute_252, out=buf139)
        buf84 = buf20; del buf20  # reuse
        buf82 = reinterpret_tensor(buf84, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf83 = reinterpret_tensor(buf84, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf87 = buf23; del buf23  # reuse
        buf85 = reinterpret_tensor(buf87, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf86 = reinterpret_tensor(buf87, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf142 = empty_strided((3, 2048, 16, 64, 2), (4194304, 2048, 128, 2, 1), device='cuda', dtype=torch.float32)
        buf140 = reinterpret_tensor(buf142, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf141 = reinterpret_tensor(buf142, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf145 = empty_strided((3, 2048, 16, 64, 2), (4194304, 2048, 128, 2, 1), device='cuda', dtype=torch.float32)
        buf143 = reinterpret_tensor(buf145, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf144 = reinterpret_tensor(buf145, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_27.run(buf81, select_1, select_3, buf139, buf82, buf83, buf85, buf86, buf140, buf141, buf143, buf144, 6291456, grid=grid(6291456), stream=stream0)
        buf88 = reinterpret_tensor(buf35, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf35  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf87, buf88, 12582912, grid=grid(12582912), stream=stream0)
        del buf82
        del buf83
        del buf85
        del buf86
        buf89 = reinterpret_tensor(buf44, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf44  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf84, buf89, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf90 = reinterpret_tensor(buf29, (48, 2048, 2048), (4194304, 2048, 1)); del buf29  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf88, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf89, (48, 128, 2048), (262144, 2048, 1), 0), out=buf90)
        buf146 = reinterpret_tensor(buf52, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf52  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf145, buf146, 12582912, grid=grid(12582912), stream=stream0)
        del buf143
        del buf144
        buf147 = reinterpret_tensor(buf49, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf49  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf142, buf147, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        del buf140
        del buf141
        buf148 = reinterpret_tensor(buf64, (48, 2048, 2048), (4194304, 2048, 1)); del buf64  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf146, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf147, (48, 128, 2048), (262144, 2048, 1), 0), out=buf148)
        buf93 = reinterpret_tensor(buf62, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf62  # reuse
        buf151 = empty_strided((3, 16, 2048, 2048), (67108864, 4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_28.run(slice_3, buf90, scalar_tensor, buf148, buf93, buf151, 98304, 2048, grid=grid(98304), stream=stream0)
        buf94 = reinterpret_tensor(buf39, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf39  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf81, buf94, 12582912, grid=grid(12582912), stream=stream0)
        buf95 = reinterpret_tensor(buf33, (48, 2048, 128), (262144, 128, 1)); del buf33  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf93, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf94, (48, 2048, 128), (262144, 128, 1), 0), out=buf95)
        buf96 = buf14; del buf14  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf95, buf96, 12582912, grid=grid(12582912), stream=stream0)
        buf97 = reinterpret_tensor(buf95, (6144, 2048), (2048, 1)); del buf95  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf96, permute_272, out=buf97)
        buf98 = empty_strided((3, 2048, 1), (2048, 1, 6144), device='cuda', dtype=torch.float32)
        buf99 = reinterpret_tensor(buf60, (3, 2048, 2048), (4194304, 2048, 1)); del buf60  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_131, buf97, primals_158, buf98, buf99, 6144, 2048, grid=grid(6144), stream=stream0)
        buf100 = reinterpret_tensor(buf50, (6144, 5632), (5632, 1)); del buf50  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (6144, 2048), (2048, 1), 0), permute_273, out=buf100)
        buf101 = reinterpret_tensor(buf47, (6144, 5632), (5632, 1)); del buf47  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf99, (6144, 2048), (2048, 1), 0), permute_274, out=buf101)
        buf104 = reinterpret_tensor(buf38, (6144, 5632), (5632, 1)); del buf38  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf78, (6144, 2048), (2048, 1), 0), permute_327, out=buf104)
        del permute_327
        buf102 = reinterpret_tensor(buf46, (3, 2048, 5632), (11534336, 5632, 1)); del buf46  # reuse
        buf105 = reinterpret_tensor(buf36, (3, 2048, 5632), (11534336, 5632, 1)); del buf36  # reuse
        buf108 = empty_strided((3, 2048, 5632), (11534336, 5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf100, buf101, buf104, buf102, buf105, buf108, 34603008, grid=grid(34603008), stream=stream0)
        buf103 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf78, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf102, (6144, 5632), (5632, 1), 0), out=buf103)
        buf106 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf99, (6144, 2048), (2048, 1), 0), out=buf106)
        buf107 = reinterpret_tensor(buf65, (6144, 2048), (2048, 1)); del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf105, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_274, (5632, 2048), (2048, 1), 0), out=buf107)
        del permute_274
        buf109 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf99, (6144, 2048), (2048, 1), 0), out=buf109)
        buf110 = reinterpret_tensor(buf99, (6144, 2048), (2048, 1)); del buf99  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf108, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_273, (5632, 2048), (2048, 1), 0), out=buf110)
        del permute_273
        buf115 = buf78; del buf78  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf115, buf107, buf110, primals_158, add_131, buf97, buf98, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_158
        buf117 = reinterpret_tensor(buf25, (6144, 2048), (2048, 1)); del buf25  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_272, (2048, 2048), (2048, 1), 0), out=buf117)
        del permute_272
        buf118 = reinterpret_tensor(buf59, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf117, buf118, 12582912, grid=grid(12582912), stream=stream0)
        buf120 = buf90; del buf90  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf118, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf94, (48, 128, 2048), (262144, 1, 128), 0), out=buf120)
        buf122 = reinterpret_tensor(buf148, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf148  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf120, buf93, slice_3, buf122, 98304, 2048, grid=grid(98304), stream=stream0)
        buf123 = reinterpret_tensor(buf94, (48, 128, 2048), (262144, 2048, 1)); del buf94  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf88, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf122, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf123)
        buf124 = reinterpret_tensor(buf88, (48, 2048, 128), (262144, 128, 1)); del buf88  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf122, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf89, (48, 2048, 128), (262144, 1, 2048), 0), out=buf124)
        buf130 = reinterpret_tensor(buf81, (3, 2048, 6144), (12582912, 6144, 1)); del buf81  # reuse
        buf128 = reinterpret_tensor(buf130, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf127 = reinterpret_tensor(buf130, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf123, select_3, select_1, buf124, buf128, buf127, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf119 = buf124; del buf124  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf93, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf118, (48, 2048, 128), (262144, 128, 1), 0), out=buf119)
        buf129 = reinterpret_tensor(buf130, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf119, buf129, 12582912, grid=grid(12582912), stream=stream0)
        del buf127
        del buf128
        del buf129
        buf132 = reinterpret_tensor(buf119, (6144, 2048), (2048, 1)); del buf119  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf130, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_264, (6144, 2048), (2048, 1), 0), out=buf132)
        del permute_264
        buf111 = buf75; del buf75  # reuse
        buf133 = buf53; del buf53  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf107, buf110, add_131, buf97, buf98, buf132, buf79, buf111, buf133, 98304, 128, grid=grid(98304), stream=stream0)
        buf112 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf111, buf112, 2048, 48, grid=grid(2048), stream=stream0)
        buf116 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf115, (2048, 6144), (1, 2048), 0), buf96, out=buf116)
        buf131 = buf96; del buf96  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf130, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf80, (6144, 2048), (2048, 1), 0), out=buf131)
        buf134 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf133, buf134, 2048, 48, grid=grid(2048), stream=stream0)
        buf136 = buf115; del buf115  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf136, buf132, primals_155, add_131, buf79, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_131
        del primals_155
        buf152 = reinterpret_tensor(buf132, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf132  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf139, buf152, 12582912, grid=grid(12582912), stream=stream0)
        buf153 = reinterpret_tensor(buf80, (48, 2048, 128), (262144, 128, 1)); del buf80  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf151, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf152, (48, 2048, 128), (262144, 128, 1), 0), out=buf153)
        buf154 = buf97; del buf97  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf153, buf154, 12582912, grid=grid(12582912), stream=stream0)
        buf155 = reinterpret_tensor(buf153, (6144, 2048), (2048, 1)); del buf153  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf154, permute_260, out=buf155)
        buf156 = buf79; del buf79  # reuse
        buf157 = reinterpret_tensor(buf110, (3, 2048, 2048), (4194304, 2048, 1)); del buf110  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_125, buf155, primals_151, buf156, buf157, 6144, 2048, grid=grid(6144), stream=stream0)
        buf158 = reinterpret_tensor(buf108, (6144, 5632), (5632, 1)); del buf108  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (6144, 2048), (2048, 1), 0), permute_261, out=buf158)
        buf159 = reinterpret_tensor(buf105, (6144, 5632), (5632, 1)); del buf105  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf157, (6144, 2048), (2048, 1), 0), permute_262, out=buf159)
        buf162 = reinterpret_tensor(buf102, (6144, 5632), (5632, 1)); del buf102  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf136, (6144, 2048), (2048, 1), 0), permute_359, out=buf162)
        del permute_359
        buf160 = reinterpret_tensor(buf104, (3, 2048, 5632), (11534336, 5632, 1)); del buf104  # reuse
        buf163 = reinterpret_tensor(buf101, (3, 2048, 5632), (11534336, 5632, 1)); del buf101  # reuse
        buf166 = reinterpret_tensor(buf100, (3, 2048, 5632), (11534336, 5632, 1)); del buf100  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf158, buf159, buf162, buf160, buf163, buf166, 34603008, grid=grid(34603008), stream=stream0)
        buf161 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf136, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf160, (6144, 5632), (5632, 1), 0), out=buf161)
        buf164 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf157, (6144, 2048), (2048, 1), 0), out=buf164)
        buf165 = buf107; del buf107  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf163, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_262, (5632, 2048), (2048, 1), 0), out=buf165)
        del permute_262
        buf167 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf157, (6144, 2048), (2048, 1), 0), out=buf167)
        buf168 = reinterpret_tensor(buf157, (6144, 2048), (2048, 1)); del buf157  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf166, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_261, (5632, 2048), (2048, 1), 0), out=buf168)
        del permute_261
        buf173 = buf136; del buf136  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf173, buf165, buf168, primals_151, add_125, buf155, buf156, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_151
        buf175 = reinterpret_tensor(buf118, (6144, 2048), (2048, 1)); del buf118  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf173, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_260, (2048, 2048), (2048, 1), 0), out=buf175)
        del permute_260
        buf176 = reinterpret_tensor(buf123, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf123  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf175, buf176, 12582912, grid=grid(12582912), stream=stream0)
        buf178 = reinterpret_tensor(buf93, (48, 2048, 2048), (4194304, 2048, 1)); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf176, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf152, (48, 128, 2048), (262144, 1, 128), 0), out=buf178)
        buf180 = buf122; del buf122  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf178, buf151, slice_3, buf180, 98304, 2048, grid=grid(98304), stream=stream0)
        buf181 = reinterpret_tensor(buf152, (48, 128, 2048), (262144, 2048, 1)); del buf152  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf146, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf180, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf181)
        buf182 = reinterpret_tensor(buf146, (48, 2048, 128), (262144, 128, 1)); del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf180, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf147, (48, 2048, 128), (262144, 1, 2048), 0), out=buf182)
        buf188 = reinterpret_tensor(buf139, (3, 2048, 6144), (12582912, 6144, 1)); del buf139  # reuse
        buf186 = reinterpret_tensor(buf188, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf185 = reinterpret_tensor(buf188, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf181, select_3, select_1, buf182, buf186, buf185, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf177 = buf182; del buf182  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf151, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf176, (48, 2048, 128), (262144, 128, 1), 0), out=buf177)
        buf187 = reinterpret_tensor(buf188, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf177, buf187, 12582912, grid=grid(12582912), stream=stream0)
        del buf185
        del buf186
        del buf187
        buf190 = reinterpret_tensor(buf177, (6144, 2048), (2048, 1)); del buf177  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_252, (6144, 2048), (2048, 1), 0), out=buf190)
        del permute_252
        buf169 = buf133; del buf133  # reuse
        buf191 = buf111; del buf111  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf165, buf168, add_125, buf155, buf156, buf190, buf137, buf169, buf191, 98304, 128, grid=grid(98304), stream=stream0)
        buf170 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf169, buf170, 2048, 48, grid=grid(2048), stream=stream0)
        buf174 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf173, (2048, 6144), (1, 2048), 0), buf154, out=buf174)
        buf189 = buf154; del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf188, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf138, (6144, 2048), (2048, 1), 0), out=buf189)
        buf192 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf191, buf192, 2048, 48, grid=grid(2048), stream=stream0)
        buf194 = buf173; del buf173  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf194, buf190, primals_148, add_125, buf137, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_125
        del primals_148
        buf195 = buf137; del buf137  # reuse
        buf196 = reinterpret_tensor(buf190, (3, 2048, 2048), (4194304, 2048, 1)); del buf190  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_119, primals_141, buf195, buf196, 6144, 2048, grid=grid(6144), stream=stream0)
        buf197 = reinterpret_tensor(buf188, (6144, 6144), (6144, 1)); del buf188  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf196, (6144, 2048), (2048, 1), 0), permute_240, out=buf197)
        buf253 = buf156; del buf156  # reuse
        buf254 = buf138; del buf138  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_113, primals_134, buf253, buf254, 6144, 2048, grid=grid(6144), stream=stream0)
        buf255 = reinterpret_tensor(buf130, (6144, 6144), (6144, 1)); del buf130  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf254, (6144, 2048), (2048, 1), 0), permute_228, out=buf255)
        buf200 = buf142; del buf142  # reuse
        buf198 = reinterpret_tensor(buf200, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf199 = reinterpret_tensor(buf200, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf203 = buf145; del buf145  # reuse
        buf201 = reinterpret_tensor(buf203, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf202 = reinterpret_tensor(buf203, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf258 = buf84; del buf84  # reuse
        buf256 = reinterpret_tensor(buf258, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf257 = reinterpret_tensor(buf258, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf261 = buf87; del buf87  # reuse
        buf259 = reinterpret_tensor(buf261, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf260 = reinterpret_tensor(buf261, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_27.run(buf197, select_1, select_3, buf255, buf198, buf199, buf201, buf202, buf256, buf257, buf259, buf260, 6291456, grid=grid(6291456), stream=stream0)
        buf204 = reinterpret_tensor(buf168, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf168  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf203, buf204, 12582912, grid=grid(12582912), stream=stream0)
        del buf198
        del buf199
        del buf201
        del buf202
        buf205 = reinterpret_tensor(buf165, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf165  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf200, buf205, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf206 = reinterpret_tensor(buf151, (48, 2048, 2048), (4194304, 2048, 1)); del buf151  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf204, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf205, (48, 128, 2048), (262144, 2048, 1), 0), out=buf206)
        buf262 = reinterpret_tensor(buf155, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf155  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf261, buf262, 12582912, grid=grid(12582912), stream=stream0)
        del buf259
        del buf260
        buf263 = reinterpret_tensor(buf176, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf176  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf258, buf263, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        del buf256
        del buf257
        buf264 = reinterpret_tensor(buf180, (48, 2048, 2048), (4194304, 2048, 1)); del buf180  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf262, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf263, (48, 128, 2048), (262144, 2048, 1), 0), out=buf264)
        buf209 = reinterpret_tensor(buf178, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf178  # reuse
        buf267 = reinterpret_tensor(buf120, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf120  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_28.run(slice_3, buf206, scalar_tensor, buf264, buf209, buf267, 98304, 2048, grid=grid(98304), stream=stream0)
        buf210 = reinterpret_tensor(buf181, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf181  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf197, buf210, 12582912, grid=grid(12582912), stream=stream0)
        buf211 = reinterpret_tensor(buf147, (48, 2048, 128), (262144, 128, 1)); del buf147  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf209, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf210, (48, 2048, 128), (262144, 128, 1), 0), out=buf211)
        buf212 = buf175; del buf175  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf211, buf212, 12582912, grid=grid(12582912), stream=stream0)
        buf213 = reinterpret_tensor(buf211, (6144, 2048), (2048, 1)); del buf211  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf212, permute_248, out=buf213)
        buf214 = buf98; del buf98  # reuse
        buf215 = reinterpret_tensor(buf89, (3, 2048, 2048), (4194304, 2048, 1)); del buf89  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_119, buf213, primals_144, buf214, buf215, 6144, 2048, grid=grid(6144), stream=stream0)
        buf216 = reinterpret_tensor(buf166, (6144, 5632), (5632, 1)); del buf166  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (6144, 2048), (2048, 1), 0), permute_249, out=buf216)
        buf217 = reinterpret_tensor(buf163, (6144, 5632), (5632, 1)); del buf163  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf215, (6144, 2048), (2048, 1), 0), permute_250, out=buf217)
        buf220 = reinterpret_tensor(buf160, (6144, 5632), (5632, 1)); del buf160  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf194, (6144, 2048), (2048, 1), 0), permute_391, out=buf220)
        del permute_391
        buf218 = reinterpret_tensor(buf162, (3, 2048, 5632), (11534336, 5632, 1)); del buf162  # reuse
        buf221 = reinterpret_tensor(buf159, (3, 2048, 5632), (11534336, 5632, 1)); del buf159  # reuse
        buf224 = reinterpret_tensor(buf158, (3, 2048, 5632), (11534336, 5632, 1)); del buf158  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf216, buf217, buf220, buf218, buf221, buf224, 34603008, grid=grid(34603008), stream=stream0)
        buf219 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf194, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf218, (6144, 5632), (5632, 1), 0), out=buf219)
        buf222 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf215, (6144, 2048), (2048, 1), 0), out=buf222)
        buf223 = buf117; del buf117  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf221, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_250, (5632, 2048), (2048, 1), 0), out=buf223)
        del permute_250
        buf225 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf224, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf215, (6144, 2048), (2048, 1), 0), out=buf225)
        buf226 = reinterpret_tensor(buf215, (6144, 2048), (2048, 1)); del buf215  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf224, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_249, (5632, 2048), (2048, 1), 0), out=buf226)
        del permute_249
        buf231 = buf194; del buf194  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf231, buf223, buf226, primals_144, add_119, buf213, buf214, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_144
        buf233 = empty_strided((6144, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_248, (2048, 2048), (2048, 1), 0), out=buf233)
        del permute_248
        buf234 = empty_strided((3, 16, 2048, 128), (4194304, 262144, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf233, buf234, 12582912, grid=grid(12582912), stream=stream0)
        buf236 = buf264; del buf264  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf234, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf210, (48, 128, 2048), (262144, 1, 128), 0), out=buf236)
        buf238 = reinterpret_tensor(buf206, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf206  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf236, buf209, slice_3, buf238, 98304, 2048, grid=grid(98304), stream=stream0)
        buf239 = reinterpret_tensor(buf210, (48, 128, 2048), (262144, 2048, 1)); del buf210  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf204, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf238, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf239)
        buf240 = reinterpret_tensor(buf204, (48, 2048, 128), (262144, 128, 1)); del buf204  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf238, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf205, (48, 2048, 128), (262144, 1, 2048), 0), out=buf240)
        buf246 = reinterpret_tensor(buf197, (3, 2048, 6144), (12582912, 6144, 1)); del buf197  # reuse
        buf244 = reinterpret_tensor(buf246, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf243 = reinterpret_tensor(buf246, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf239, select_3, select_1, buf240, buf244, buf243, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf235 = buf240; del buf240  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf209, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf234, (48, 2048, 128), (262144, 128, 1), 0), out=buf235)
        buf245 = reinterpret_tensor(buf246, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf235, buf245, 12582912, grid=grid(12582912), stream=stream0)
        del buf243
        del buf244
        del buf245
        buf248 = reinterpret_tensor(buf235, (6144, 2048), (2048, 1)); del buf235  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf246, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_240, (6144, 2048), (2048, 1), 0), out=buf248)
        del permute_240
        buf227 = buf191; del buf191  # reuse
        buf249 = buf169; del buf169  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf223, buf226, add_119, buf213, buf214, buf248, buf195, buf227, buf249, 98304, 128, grid=grid(98304), stream=stream0)
        buf228 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf227, buf228, 2048, 48, grid=grid(2048), stream=stream0)
        buf232 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf231, (2048, 6144), (1, 2048), 0), buf212, out=buf232)
        buf247 = buf212; del buf212  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf246, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf196, (6144, 2048), (2048, 1), 0), out=buf247)
        buf250 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf249, buf250, 2048, 48, grid=grid(2048), stream=stream0)
        buf252 = buf231; del buf231  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf252, buf248, primals_141, add_119, buf195, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_119
        del primals_141
        buf268 = reinterpret_tensor(buf248, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf248  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf255, buf268, 12582912, grid=grid(12582912), stream=stream0)
        buf269 = reinterpret_tensor(buf196, (48, 2048, 128), (262144, 128, 1)); del buf196  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf267, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf268, (48, 2048, 128), (262144, 128, 1), 0), out=buf269)
        buf270 = buf226; del buf226  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf269, buf270, 12582912, grid=grid(12582912), stream=stream0)
        buf271 = reinterpret_tensor(buf269, (6144, 2048), (2048, 1)); del buf269  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf270, permute_236, out=buf271)
        buf272 = buf195; del buf195  # reuse
        buf273 = reinterpret_tensor(buf223, (3, 2048, 2048), (4194304, 2048, 1)); del buf223  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_113, buf271, primals_137, buf272, buf273, 6144, 2048, grid=grid(6144), stream=stream0)
        buf274 = reinterpret_tensor(buf224, (6144, 5632), (5632, 1)); del buf224  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (6144, 2048), (2048, 1), 0), permute_237, out=buf274)
        buf275 = reinterpret_tensor(buf221, (6144, 5632), (5632, 1)); del buf221  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf273, (6144, 2048), (2048, 1), 0), permute_238, out=buf275)
        buf278 = reinterpret_tensor(buf218, (6144, 5632), (5632, 1)); del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (6144, 2048), (2048, 1), 0), permute_423, out=buf278)
        del permute_423
        buf276 = reinterpret_tensor(buf220, (3, 2048, 5632), (11534336, 5632, 1)); del buf220  # reuse
        buf279 = reinterpret_tensor(buf217, (3, 2048, 5632), (11534336, 5632, 1)); del buf217  # reuse
        buf282 = reinterpret_tensor(buf216, (3, 2048, 5632), (11534336, 5632, 1)); del buf216  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf274, buf275, buf278, buf276, buf279, buf282, 34603008, grid=grid(34603008), stream=stream0)
        buf277 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf252, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf276, (6144, 5632), (5632, 1), 0), out=buf277)
        buf280 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf273, (6144, 2048), (2048, 1), 0), out=buf280)
        buf281 = buf213; del buf213  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf279, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_238, (5632, 2048), (2048, 1), 0), out=buf281)
        del permute_238
        buf283 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf282, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf273, (6144, 2048), (2048, 1), 0), out=buf283)
        buf284 = reinterpret_tensor(buf273, (6144, 2048), (2048, 1)); del buf273  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf282, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_237, (5632, 2048), (2048, 1), 0), out=buf284)
        del permute_237
        buf289 = buf252; del buf252  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf289, buf281, buf284, primals_137, add_113, buf271, buf272, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_137
        buf291 = reinterpret_tensor(buf234, (6144, 2048), (2048, 1)); del buf234  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_236, (2048, 2048), (2048, 1), 0), out=buf291)
        del permute_236
        buf292 = reinterpret_tensor(buf239, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf291, buf292, 12582912, grid=grid(12582912), stream=stream0)
        buf294 = reinterpret_tensor(buf209, (48, 2048, 2048), (4194304, 2048, 1)); del buf209  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf292, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf268, (48, 128, 2048), (262144, 1, 128), 0), out=buf294)
        buf296 = buf238; del buf238  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf294, buf267, slice_3, buf296, 98304, 2048, grid=grid(98304), stream=stream0)
        buf297 = reinterpret_tensor(buf268, (48, 128, 2048), (262144, 2048, 1)); del buf268  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf262, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf296, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf297)
        buf298 = reinterpret_tensor(buf262, (48, 2048, 128), (262144, 128, 1)); del buf262  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf296, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf263, (48, 2048, 128), (262144, 1, 2048), 0), out=buf298)
        buf304 = reinterpret_tensor(buf255, (3, 2048, 6144), (12582912, 6144, 1)); del buf255  # reuse
        buf302 = reinterpret_tensor(buf304, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf301 = reinterpret_tensor(buf304, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf297, select_3, select_1, buf298, buf302, buf301, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf293 = buf298; del buf298  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf267, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf292, (48, 2048, 128), (262144, 128, 1), 0), out=buf293)
        buf303 = reinterpret_tensor(buf304, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf293, buf303, 12582912, grid=grid(12582912), stream=stream0)
        del buf301
        del buf302
        del buf303
        buf306 = reinterpret_tensor(buf293, (6144, 2048), (2048, 1)); del buf293  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_228, (6144, 2048), (2048, 1), 0), out=buf306)
        del permute_228
        buf285 = buf249; del buf249  # reuse
        buf307 = buf227; del buf227  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf281, buf284, add_113, buf271, buf272, buf306, buf253, buf285, buf307, 98304, 128, grid=grid(98304), stream=stream0)
        buf286 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf285, buf286, 2048, 48, grid=grid(2048), stream=stream0)
        buf290 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf289, (2048, 6144), (1, 2048), 0), buf270, out=buf290)
        buf305 = buf270; del buf270  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf254, (6144, 2048), (2048, 1), 0), out=buf305)
        buf308 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf307, buf308, 2048, 48, grid=grid(2048), stream=stream0)
        buf310 = buf289; del buf289  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf310, buf306, primals_134, add_113, buf253, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_113
        del primals_134
        buf311 = buf253; del buf253  # reuse
        buf312 = reinterpret_tensor(buf306, (3, 2048, 2048), (4194304, 2048, 1)); del buf306  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_107, primals_127, buf311, buf312, 6144, 2048, grid=grid(6144), stream=stream0)
        buf313 = reinterpret_tensor(buf304, (6144, 6144), (6144, 1)); del buf304  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf312, (6144, 2048), (2048, 1), 0), permute_216, out=buf313)
        buf369 = buf272; del buf272  # reuse
        buf370 = buf254; del buf254  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_101, primals_120, buf369, buf370, 6144, 2048, grid=grid(6144), stream=stream0)
        buf371 = reinterpret_tensor(buf246, (6144, 6144), (6144, 1)); del buf246  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf370, (6144, 2048), (2048, 1), 0), permute_204, out=buf371)
        buf316 = buf258; del buf258  # reuse
        buf314 = reinterpret_tensor(buf316, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf315 = reinterpret_tensor(buf316, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf319 = buf261; del buf261  # reuse
        buf317 = reinterpret_tensor(buf319, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf318 = reinterpret_tensor(buf319, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf374 = buf200; del buf200  # reuse
        buf372 = reinterpret_tensor(buf374, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf373 = reinterpret_tensor(buf374, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf377 = buf203; del buf203  # reuse
        buf375 = reinterpret_tensor(buf377, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf376 = reinterpret_tensor(buf377, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_27.run(buf313, select_1, select_3, buf371, buf314, buf315, buf317, buf318, buf372, buf373, buf375, buf376, 6291456, grid=grid(6291456), stream=stream0)
        buf320 = reinterpret_tensor(buf284, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf284  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf319, buf320, 12582912, grid=grid(12582912), stream=stream0)
        del buf314
        del buf315
        del buf317
        del buf318
        buf321 = reinterpret_tensor(buf281, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf281  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf316, buf321, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf322 = reinterpret_tensor(buf267, (48, 2048, 2048), (4194304, 2048, 1)); del buf267  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf320, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf321, (48, 128, 2048), (262144, 2048, 1), 0), out=buf322)
        buf378 = reinterpret_tensor(buf271, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf271  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf377, buf378, 12582912, grid=grid(12582912), stream=stream0)
        del buf375
        del buf376
        buf379 = reinterpret_tensor(buf292, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf292  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf374, buf379, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        del buf372
        del buf373
        buf380 = reinterpret_tensor(buf296, (48, 2048, 2048), (4194304, 2048, 1)); del buf296  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf378, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf379, (48, 128, 2048), (262144, 2048, 1), 0), out=buf380)
        buf325 = reinterpret_tensor(buf294, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf294  # reuse
        buf383 = reinterpret_tensor(buf236, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf236  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_28.run(slice_3, buf322, scalar_tensor, buf380, buf325, buf383, 98304, 2048, grid=grid(98304), stream=stream0)
        buf326 = reinterpret_tensor(buf297, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf297  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf313, buf326, 12582912, grid=grid(12582912), stream=stream0)
        buf327 = reinterpret_tensor(buf263, (48, 2048, 128), (262144, 128, 1)); del buf263  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf325, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf326, (48, 2048, 128), (262144, 128, 1), 0), out=buf327)
        buf328 = buf291; del buf291  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf327, buf328, 12582912, grid=grid(12582912), stream=stream0)
        buf329 = reinterpret_tensor(buf327, (6144, 2048), (2048, 1)); del buf327  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf328, permute_224, out=buf329)
        buf330 = buf214; del buf214  # reuse
        buf331 = reinterpret_tensor(buf205, (3, 2048, 2048), (4194304, 2048, 1)); del buf205  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_107, buf329, primals_130, buf330, buf331, 6144, 2048, grid=grid(6144), stream=stream0)
        buf332 = reinterpret_tensor(buf282, (6144, 5632), (5632, 1)); del buf282  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf331, (6144, 2048), (2048, 1), 0), permute_225, out=buf332)
        buf333 = reinterpret_tensor(buf279, (6144, 5632), (5632, 1)); del buf279  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf331, (6144, 2048), (2048, 1), 0), permute_226, out=buf333)
        buf336 = reinterpret_tensor(buf276, (6144, 5632), (5632, 1)); del buf276  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (6144, 2048), (2048, 1), 0), permute_455, out=buf336)
        del permute_455
        buf334 = reinterpret_tensor(buf278, (3, 2048, 5632), (11534336, 5632, 1)); del buf278  # reuse
        buf337 = reinterpret_tensor(buf275, (3, 2048, 5632), (11534336, 5632, 1)); del buf275  # reuse
        buf340 = reinterpret_tensor(buf274, (3, 2048, 5632), (11534336, 5632, 1)); del buf274  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf332, buf333, buf336, buf334, buf337, buf340, 34603008, grid=grid(34603008), stream=stream0)
        buf335 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf310, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf334, (6144, 5632), (5632, 1), 0), out=buf335)
        buf338 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf331, (6144, 2048), (2048, 1), 0), out=buf338)
        buf339 = buf233; del buf233  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf337, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_226, (5632, 2048), (2048, 1), 0), out=buf339)
        del permute_226
        buf341 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf340, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf331, (6144, 2048), (2048, 1), 0), out=buf341)
        buf342 = reinterpret_tensor(buf331, (6144, 2048), (2048, 1)); del buf331  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf340, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_225, (5632, 2048), (2048, 1), 0), out=buf342)
        del permute_225
        buf347 = buf310; del buf310  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf347, buf339, buf342, primals_130, add_107, buf329, buf330, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_130
        buf349 = empty_strided((6144, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf347, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_224, (2048, 2048), (2048, 1), 0), out=buf349)
        del permute_224
        buf350 = empty_strided((3, 16, 2048, 128), (4194304, 262144, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf349, buf350, 12582912, grid=grid(12582912), stream=stream0)
        buf352 = buf380; del buf380  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf350, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf326, (48, 128, 2048), (262144, 1, 128), 0), out=buf352)
        buf354 = reinterpret_tensor(buf322, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf322  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf352, buf325, slice_3, buf354, 98304, 2048, grid=grid(98304), stream=stream0)
        buf355 = reinterpret_tensor(buf326, (48, 128, 2048), (262144, 2048, 1)); del buf326  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf320, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf354, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf355)
        buf356 = reinterpret_tensor(buf320, (48, 2048, 128), (262144, 128, 1)); del buf320  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf354, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf321, (48, 2048, 128), (262144, 1, 2048), 0), out=buf356)
        buf362 = reinterpret_tensor(buf313, (3, 2048, 6144), (12582912, 6144, 1)); del buf313  # reuse
        buf360 = reinterpret_tensor(buf362, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf359 = reinterpret_tensor(buf362, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf355, select_3, select_1, buf356, buf360, buf359, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf351 = buf356; del buf356  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf325, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf350, (48, 2048, 128), (262144, 128, 1), 0), out=buf351)
        buf361 = reinterpret_tensor(buf362, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf351, buf361, 12582912, grid=grid(12582912), stream=stream0)
        del buf359
        del buf360
        del buf361
        buf364 = reinterpret_tensor(buf351, (6144, 2048), (2048, 1)); del buf351  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf362, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_216, (6144, 2048), (2048, 1), 0), out=buf364)
        del permute_216
        buf343 = buf307; del buf307  # reuse
        buf365 = buf285; del buf285  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf339, buf342, add_107, buf329, buf330, buf364, buf311, buf343, buf365, 98304, 128, grid=grid(98304), stream=stream0)
        buf344 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf343, buf344, 2048, 48, grid=grid(2048), stream=stream0)
        buf348 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf347, (2048, 6144), (1, 2048), 0), buf328, out=buf348)
        buf363 = buf328; del buf328  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf362, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf312, (6144, 2048), (2048, 1), 0), out=buf363)
        buf366 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf365, buf366, 2048, 48, grid=grid(2048), stream=stream0)
        buf368 = buf347; del buf347  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf368, buf364, primals_127, add_107, buf311, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_107
        del primals_127
        buf384 = reinterpret_tensor(buf364, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf364  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf371, buf384, 12582912, grid=grid(12582912), stream=stream0)
        buf385 = reinterpret_tensor(buf312, (48, 2048, 128), (262144, 128, 1)); del buf312  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf383, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf384, (48, 2048, 128), (262144, 128, 1), 0), out=buf385)
        buf386 = buf342; del buf342  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf385, buf386, 12582912, grid=grid(12582912), stream=stream0)
        buf387 = reinterpret_tensor(buf385, (6144, 2048), (2048, 1)); del buf385  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf386, permute_212, out=buf387)
        buf388 = buf311; del buf311  # reuse
        buf389 = reinterpret_tensor(buf339, (3, 2048, 2048), (4194304, 2048, 1)); del buf339  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_101, buf387, primals_123, buf388, buf389, 6144, 2048, grid=grid(6144), stream=stream0)
        buf390 = reinterpret_tensor(buf340, (6144, 5632), (5632, 1)); del buf340  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf389, (6144, 2048), (2048, 1), 0), permute_213, out=buf390)
        buf391 = reinterpret_tensor(buf337, (6144, 5632), (5632, 1)); del buf337  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf389, (6144, 2048), (2048, 1), 0), permute_214, out=buf391)
        buf394 = reinterpret_tensor(buf334, (6144, 5632), (5632, 1)); del buf334  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf368, (6144, 2048), (2048, 1), 0), permute_487, out=buf394)
        del permute_487
        buf392 = reinterpret_tensor(buf336, (3, 2048, 5632), (11534336, 5632, 1)); del buf336  # reuse
        buf395 = reinterpret_tensor(buf333, (3, 2048, 5632), (11534336, 5632, 1)); del buf333  # reuse
        buf398 = reinterpret_tensor(buf332, (3, 2048, 5632), (11534336, 5632, 1)); del buf332  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf390, buf391, buf394, buf392, buf395, buf398, 34603008, grid=grid(34603008), stream=stream0)
        buf393 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf368, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf392, (6144, 5632), (5632, 1), 0), out=buf393)
        buf396 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf395, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf389, (6144, 2048), (2048, 1), 0), out=buf396)
        buf397 = buf329; del buf329  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf395, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_214, (5632, 2048), (2048, 1), 0), out=buf397)
        del permute_214
        buf399 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf398, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf389, (6144, 2048), (2048, 1), 0), out=buf399)
        buf400 = reinterpret_tensor(buf389, (6144, 2048), (2048, 1)); del buf389  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf398, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_213, (5632, 2048), (2048, 1), 0), out=buf400)
        del permute_213
        buf405 = buf368; del buf368  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf405, buf397, buf400, primals_123, add_101, buf387, buf388, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_123
        buf407 = reinterpret_tensor(buf350, (6144, 2048), (2048, 1)); del buf350  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_212, (2048, 2048), (2048, 1), 0), out=buf407)
        del permute_212
        buf408 = reinterpret_tensor(buf355, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf355  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf407, buf408, 12582912, grid=grid(12582912), stream=stream0)
        buf410 = reinterpret_tensor(buf325, (48, 2048, 2048), (4194304, 2048, 1)); del buf325  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf408, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf384, (48, 128, 2048), (262144, 1, 128), 0), out=buf410)
        buf412 = buf354; del buf354  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf410, buf383, slice_3, buf412, 98304, 2048, grid=grid(98304), stream=stream0)
        buf413 = reinterpret_tensor(buf384, (48, 128, 2048), (262144, 2048, 1)); del buf384  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf378, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf412, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf413)
        buf414 = reinterpret_tensor(buf378, (48, 2048, 128), (262144, 128, 1)); del buf378  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf412, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf379, (48, 2048, 128), (262144, 1, 2048), 0), out=buf414)
        buf420 = reinterpret_tensor(buf371, (3, 2048, 6144), (12582912, 6144, 1)); del buf371  # reuse
        buf418 = reinterpret_tensor(buf420, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf417 = reinterpret_tensor(buf420, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf413, select_3, select_1, buf414, buf418, buf417, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf409 = buf414; del buf414  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf383, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf408, (48, 2048, 128), (262144, 128, 1), 0), out=buf409)
        buf419 = reinterpret_tensor(buf420, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf409, buf419, 12582912, grid=grid(12582912), stream=stream0)
        del buf417
        del buf418
        del buf419
        buf422 = reinterpret_tensor(buf409, (6144, 2048), (2048, 1)); del buf409  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf420, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_204, (6144, 2048), (2048, 1), 0), out=buf422)
        del permute_204
        buf401 = buf365; del buf365  # reuse
        buf423 = buf343; del buf343  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf397, buf400, add_101, buf387, buf388, buf422, buf369, buf401, buf423, 98304, 128, grid=grid(98304), stream=stream0)
        buf402 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf401, buf402, 2048, 48, grid=grid(2048), stream=stream0)
        buf406 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf405, (2048, 6144), (1, 2048), 0), buf386, out=buf406)
        buf421 = buf386; del buf386  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf420, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf370, (6144, 2048), (2048, 1), 0), out=buf421)
        buf424 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf423, buf424, 2048, 48, grid=grid(2048), stream=stream0)
        buf426 = buf405; del buf405  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf426, buf422, primals_120, add_101, buf369, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_101
        del primals_120
        buf427 = buf369; del buf369  # reuse
        buf428 = reinterpret_tensor(buf422, (3, 2048, 2048), (4194304, 2048, 1)); del buf422  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_95, primals_113, buf427, buf428, 6144, 2048, grid=grid(6144), stream=stream0)
        buf429 = reinterpret_tensor(buf420, (6144, 6144), (6144, 1)); del buf420  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf428, (6144, 2048), (2048, 1), 0), permute_192, out=buf429)
        buf485 = buf388; del buf388  # reuse
        buf486 = buf370; del buf370  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_89, primals_106, buf485, buf486, 6144, 2048, grid=grid(6144), stream=stream0)
        buf487 = reinterpret_tensor(buf362, (6144, 6144), (6144, 1)); del buf362  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf486, (6144, 2048), (2048, 1), 0), permute_180, out=buf487)
        buf432 = buf374; del buf374  # reuse
        buf430 = reinterpret_tensor(buf432, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf431 = reinterpret_tensor(buf432, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf435 = buf377; del buf377  # reuse
        buf433 = reinterpret_tensor(buf435, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf434 = reinterpret_tensor(buf435, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf490 = buf316; del buf316  # reuse
        buf488 = reinterpret_tensor(buf490, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf489 = reinterpret_tensor(buf490, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf493 = buf319; del buf319  # reuse
        buf491 = reinterpret_tensor(buf493, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf492 = reinterpret_tensor(buf493, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_27.run(buf429, select_1, select_3, buf487, buf430, buf431, buf433, buf434, buf488, buf489, buf491, buf492, 6291456, grid=grid(6291456), stream=stream0)
        buf436 = reinterpret_tensor(buf400, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf400  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf435, buf436, 12582912, grid=grid(12582912), stream=stream0)
        del buf430
        del buf431
        del buf433
        del buf434
        buf437 = reinterpret_tensor(buf397, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf397  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf432, buf437, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf438 = reinterpret_tensor(buf383, (48, 2048, 2048), (4194304, 2048, 1)); del buf383  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf436, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf437, (48, 128, 2048), (262144, 2048, 1), 0), out=buf438)
        buf494 = reinterpret_tensor(buf387, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf387  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf493, buf494, 12582912, grid=grid(12582912), stream=stream0)
        del buf491
        del buf492
        buf495 = reinterpret_tensor(buf408, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf408  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf490, buf495, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        del buf488
        del buf489
        buf496 = reinterpret_tensor(buf412, (48, 2048, 2048), (4194304, 2048, 1)); del buf412  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf494, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf495, (48, 128, 2048), (262144, 2048, 1), 0), out=buf496)
        buf441 = reinterpret_tensor(buf410, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf410  # reuse
        buf499 = reinterpret_tensor(buf352, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf352  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_28.run(slice_3, buf438, scalar_tensor, buf496, buf441, buf499, 98304, 2048, grid=grid(98304), stream=stream0)
        buf442 = reinterpret_tensor(buf413, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf413  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf429, buf442, 12582912, grid=grid(12582912), stream=stream0)
        buf443 = reinterpret_tensor(buf379, (48, 2048, 128), (262144, 128, 1)); del buf379  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf441, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf442, (48, 2048, 128), (262144, 128, 1), 0), out=buf443)
        buf444 = buf407; del buf407  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf443, buf444, 12582912, grid=grid(12582912), stream=stream0)
        buf445 = reinterpret_tensor(buf443, (6144, 2048), (2048, 1)); del buf443  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf444, permute_200, out=buf445)
        buf446 = buf330; del buf330  # reuse
        buf447 = reinterpret_tensor(buf321, (3, 2048, 2048), (4194304, 2048, 1)); del buf321  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_95, buf445, primals_116, buf446, buf447, 6144, 2048, grid=grid(6144), stream=stream0)
        buf448 = reinterpret_tensor(buf398, (6144, 5632), (5632, 1)); del buf398  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (6144, 2048), (2048, 1), 0), permute_201, out=buf448)
        buf449 = reinterpret_tensor(buf395, (6144, 5632), (5632, 1)); del buf395  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf447, (6144, 2048), (2048, 1), 0), permute_202, out=buf449)
        buf452 = reinterpret_tensor(buf392, (6144, 5632), (5632, 1)); del buf392  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf426, (6144, 2048), (2048, 1), 0), permute_519, out=buf452)
        del permute_519
        buf450 = reinterpret_tensor(buf394, (3, 2048, 5632), (11534336, 5632, 1)); del buf394  # reuse
        buf453 = reinterpret_tensor(buf391, (3, 2048, 5632), (11534336, 5632, 1)); del buf391  # reuse
        buf456 = reinterpret_tensor(buf390, (3, 2048, 5632), (11534336, 5632, 1)); del buf390  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf448, buf449, buf452, buf450, buf453, buf456, 34603008, grid=grid(34603008), stream=stream0)
        buf451 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf426, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf450, (6144, 5632), (5632, 1), 0), out=buf451)
        buf454 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf453, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf447, (6144, 2048), (2048, 1), 0), out=buf454)
        buf455 = buf349; del buf349  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf453, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_202, (5632, 2048), (2048, 1), 0), out=buf455)
        del permute_202
        buf457 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf456, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf447, (6144, 2048), (2048, 1), 0), out=buf457)
        buf458 = reinterpret_tensor(buf447, (6144, 2048), (2048, 1)); del buf447  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf456, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_201, (5632, 2048), (2048, 1), 0), out=buf458)
        del permute_201
        buf463 = buf426; del buf426  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf463, buf455, buf458, primals_116, add_95, buf445, buf446, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_116
        buf465 = empty_strided((6144, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf463, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_200, (2048, 2048), (2048, 1), 0), out=buf465)
        del permute_200
        buf466 = empty_strided((3, 16, 2048, 128), (4194304, 262144, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf465, buf466, 12582912, grid=grid(12582912), stream=stream0)
        buf468 = buf496; del buf496  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf466, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf442, (48, 128, 2048), (262144, 1, 128), 0), out=buf468)
        buf470 = reinterpret_tensor(buf438, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf438  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf468, buf441, slice_3, buf470, 98304, 2048, grid=grid(98304), stream=stream0)
        buf471 = reinterpret_tensor(buf442, (48, 128, 2048), (262144, 2048, 1)); del buf442  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf436, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf470, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf471)
        buf472 = reinterpret_tensor(buf436, (48, 2048, 128), (262144, 128, 1)); del buf436  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf470, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf437, (48, 2048, 128), (262144, 1, 2048), 0), out=buf472)
        buf478 = reinterpret_tensor(buf429, (3, 2048, 6144), (12582912, 6144, 1)); del buf429  # reuse
        buf476 = reinterpret_tensor(buf478, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf475 = reinterpret_tensor(buf478, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf471, select_3, select_1, buf472, buf476, buf475, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf467 = buf472; del buf472  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf441, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf466, (48, 2048, 128), (262144, 128, 1), 0), out=buf467)
        buf477 = reinterpret_tensor(buf478, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf467, buf477, 12582912, grid=grid(12582912), stream=stream0)
        del buf475
        del buf476
        del buf477
        buf480 = reinterpret_tensor(buf467, (6144, 2048), (2048, 1)); del buf467  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf478, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_192, (6144, 2048), (2048, 1), 0), out=buf480)
        del permute_192
        buf459 = buf423; del buf423  # reuse
        buf481 = buf401; del buf401  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf455, buf458, add_95, buf445, buf446, buf480, buf427, buf459, buf481, 98304, 128, grid=grid(98304), stream=stream0)
        buf460 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf459, buf460, 2048, 48, grid=grid(2048), stream=stream0)
        buf464 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf463, (2048, 6144), (1, 2048), 0), buf444, out=buf464)
        buf479 = buf444; del buf444  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf478, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf428, (6144, 2048), (2048, 1), 0), out=buf479)
        buf482 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf481, buf482, 2048, 48, grid=grid(2048), stream=stream0)
        buf484 = buf463; del buf463  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf484, buf480, primals_113, add_95, buf427, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_95
        del primals_113
        buf500 = reinterpret_tensor(buf480, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf480  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf487, buf500, 12582912, grid=grid(12582912), stream=stream0)
        buf501 = reinterpret_tensor(buf428, (48, 2048, 128), (262144, 128, 1)); del buf428  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf499, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf500, (48, 2048, 128), (262144, 128, 1), 0), out=buf501)
        buf502 = buf458; del buf458  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf501, buf502, 12582912, grid=grid(12582912), stream=stream0)
        buf503 = reinterpret_tensor(buf501, (6144, 2048), (2048, 1)); del buf501  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf502, permute_188, out=buf503)
        buf504 = buf427; del buf427  # reuse
        buf505 = reinterpret_tensor(buf455, (3, 2048, 2048), (4194304, 2048, 1)); del buf455  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_89, buf503, primals_109, buf504, buf505, 6144, 2048, grid=grid(6144), stream=stream0)
        buf506 = reinterpret_tensor(buf456, (6144, 5632), (5632, 1)); del buf456  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf505, (6144, 2048), (2048, 1), 0), permute_189, out=buf506)
        buf507 = reinterpret_tensor(buf453, (6144, 5632), (5632, 1)); del buf453  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf505, (6144, 2048), (2048, 1), 0), permute_190, out=buf507)
        buf510 = reinterpret_tensor(buf450, (6144, 5632), (5632, 1)); del buf450  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (6144, 2048), (2048, 1), 0), permute_551, out=buf510)
        del permute_551
        buf508 = reinterpret_tensor(buf452, (3, 2048, 5632), (11534336, 5632, 1)); del buf452  # reuse
        buf511 = reinterpret_tensor(buf449, (3, 2048, 5632), (11534336, 5632, 1)); del buf449  # reuse
        buf514 = reinterpret_tensor(buf448, (3, 2048, 5632), (11534336, 5632, 1)); del buf448  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf506, buf507, buf510, buf508, buf511, buf514, 34603008, grid=grid(34603008), stream=stream0)
        buf509 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf484, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf508, (6144, 5632), (5632, 1), 0), out=buf509)
        buf512 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf511, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf505, (6144, 2048), (2048, 1), 0), out=buf512)
        buf513 = buf445; del buf445  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf511, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_190, (5632, 2048), (2048, 1), 0), out=buf513)
        del permute_190
        buf515 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf514, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf505, (6144, 2048), (2048, 1), 0), out=buf515)
        buf516 = reinterpret_tensor(buf505, (6144, 2048), (2048, 1)); del buf505  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf514, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_189, (5632, 2048), (2048, 1), 0), out=buf516)
        del permute_189
        buf521 = buf484; del buf484  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf521, buf513, buf516, primals_109, add_89, buf503, buf504, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_109
        buf523 = reinterpret_tensor(buf466, (6144, 2048), (2048, 1)); del buf466  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf521, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_188, (2048, 2048), (2048, 1), 0), out=buf523)
        del permute_188
        buf524 = reinterpret_tensor(buf471, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf471  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf523, buf524, 12582912, grid=grid(12582912), stream=stream0)
        buf526 = reinterpret_tensor(buf441, (48, 2048, 2048), (4194304, 2048, 1)); del buf441  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf524, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf500, (48, 128, 2048), (262144, 1, 128), 0), out=buf526)
        buf528 = buf470; del buf470  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf526, buf499, slice_3, buf528, 98304, 2048, grid=grid(98304), stream=stream0)
        buf529 = reinterpret_tensor(buf500, (48, 128, 2048), (262144, 2048, 1)); del buf500  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf494, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf528, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf529)
        buf530 = reinterpret_tensor(buf494, (48, 2048, 128), (262144, 128, 1)); del buf494  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf528, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf495, (48, 2048, 128), (262144, 1, 2048), 0), out=buf530)
        buf536 = reinterpret_tensor(buf487, (3, 2048, 6144), (12582912, 6144, 1)); del buf487  # reuse
        buf534 = reinterpret_tensor(buf536, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf533 = reinterpret_tensor(buf536, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf529, select_3, select_1, buf530, buf534, buf533, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf525 = buf530; del buf530  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf499, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf524, (48, 2048, 128), (262144, 128, 1), 0), out=buf525)
        buf535 = reinterpret_tensor(buf536, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf525, buf535, 12582912, grid=grid(12582912), stream=stream0)
        del buf533
        del buf534
        del buf535
        buf538 = reinterpret_tensor(buf525, (6144, 2048), (2048, 1)); del buf525  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf536, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_180, (6144, 2048), (2048, 1), 0), out=buf538)
        del permute_180
        buf517 = buf481; del buf481  # reuse
        buf539 = buf459; del buf459  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf513, buf516, add_89, buf503, buf504, buf538, buf485, buf517, buf539, 98304, 128, grid=grid(98304), stream=stream0)
        buf518 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf517, buf518, 2048, 48, grid=grid(2048), stream=stream0)
        buf522 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf521, (2048, 6144), (1, 2048), 0), buf502, out=buf522)
        buf537 = buf502; del buf502  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf536, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf486, (6144, 2048), (2048, 1), 0), out=buf537)
        buf540 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf539, buf540, 2048, 48, grid=grid(2048), stream=stream0)
        buf542 = buf521; del buf521  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf542, buf538, primals_106, add_89, buf485, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_89
        del primals_106
        buf543 = buf485; del buf485  # reuse
        buf544 = reinterpret_tensor(buf538, (3, 2048, 2048), (4194304, 2048, 1)); del buf538  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_83, primals_99, buf543, buf544, 6144, 2048, grid=grid(6144), stream=stream0)
        buf545 = reinterpret_tensor(buf536, (6144, 6144), (6144, 1)); del buf536  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf544, (6144, 2048), (2048, 1), 0), permute_168, out=buf545)
        buf601 = buf504; del buf504  # reuse
        buf602 = buf486; del buf486  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_77, primals_92, buf601, buf602, 6144, 2048, grid=grid(6144), stream=stream0)
        buf603 = reinterpret_tensor(buf478, (6144, 6144), (6144, 1)); del buf478  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf602, (6144, 2048), (2048, 1), 0), permute_156, out=buf603)
        buf548 = buf490; del buf490  # reuse
        buf546 = reinterpret_tensor(buf548, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf547 = reinterpret_tensor(buf548, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf551 = buf493; del buf493  # reuse
        buf549 = reinterpret_tensor(buf551, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf550 = reinterpret_tensor(buf551, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf606 = buf432; del buf432  # reuse
        buf604 = reinterpret_tensor(buf606, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf605 = reinterpret_tensor(buf606, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf609 = buf435; del buf435  # reuse
        buf607 = reinterpret_tensor(buf609, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf608 = reinterpret_tensor(buf609, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_27.run(buf545, select_1, select_3, buf603, buf546, buf547, buf549, buf550, buf604, buf605, buf607, buf608, 6291456, grid=grid(6291456), stream=stream0)
        buf552 = reinterpret_tensor(buf516, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf516  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf551, buf552, 12582912, grid=grid(12582912), stream=stream0)
        del buf546
        del buf547
        del buf549
        del buf550
        buf553 = reinterpret_tensor(buf513, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf513  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf548, buf553, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf554 = reinterpret_tensor(buf499, (48, 2048, 2048), (4194304, 2048, 1)); del buf499  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf552, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf553, (48, 128, 2048), (262144, 2048, 1), 0), out=buf554)
        buf610 = reinterpret_tensor(buf503, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf503  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf609, buf610, 12582912, grid=grid(12582912), stream=stream0)
        del buf607
        del buf608
        buf611 = reinterpret_tensor(buf524, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf524  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf606, buf611, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        del buf604
        del buf605
        buf612 = reinterpret_tensor(buf528, (48, 2048, 2048), (4194304, 2048, 1)); del buf528  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf610, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf611, (48, 128, 2048), (262144, 2048, 1), 0), out=buf612)
        buf557 = reinterpret_tensor(buf526, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf526  # reuse
        buf615 = reinterpret_tensor(buf468, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf468  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_28.run(slice_3, buf554, scalar_tensor, buf612, buf557, buf615, 98304, 2048, grid=grid(98304), stream=stream0)
        buf558 = reinterpret_tensor(buf529, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf529  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf545, buf558, 12582912, grid=grid(12582912), stream=stream0)
        buf559 = reinterpret_tensor(buf495, (48, 2048, 128), (262144, 128, 1)); del buf495  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf557, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf558, (48, 2048, 128), (262144, 128, 1), 0), out=buf559)
        buf560 = buf523; del buf523  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf559, buf560, 12582912, grid=grid(12582912), stream=stream0)
        buf561 = reinterpret_tensor(buf559, (6144, 2048), (2048, 1)); del buf559  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf560, permute_176, out=buf561)
        buf562 = buf446; del buf446  # reuse
        buf563 = reinterpret_tensor(buf437, (3, 2048, 2048), (4194304, 2048, 1)); del buf437  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_83, buf561, primals_102, buf562, buf563, 6144, 2048, grid=grid(6144), stream=stream0)
        buf564 = reinterpret_tensor(buf514, (6144, 5632), (5632, 1)); del buf514  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf563, (6144, 2048), (2048, 1), 0), permute_177, out=buf564)
        buf565 = reinterpret_tensor(buf511, (6144, 5632), (5632, 1)); del buf511  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf563, (6144, 2048), (2048, 1), 0), permute_178, out=buf565)
        buf568 = reinterpret_tensor(buf508, (6144, 5632), (5632, 1)); del buf508  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf542, (6144, 2048), (2048, 1), 0), permute_583, out=buf568)
        del permute_583
        buf566 = reinterpret_tensor(buf510, (3, 2048, 5632), (11534336, 5632, 1)); del buf510  # reuse
        buf569 = reinterpret_tensor(buf507, (3, 2048, 5632), (11534336, 5632, 1)); del buf507  # reuse
        buf572 = reinterpret_tensor(buf506, (3, 2048, 5632), (11534336, 5632, 1)); del buf506  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf564, buf565, buf568, buf566, buf569, buf572, 34603008, grid=grid(34603008), stream=stream0)
        buf567 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf542, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf566, (6144, 5632), (5632, 1), 0), out=buf567)
        buf570 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf569, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf563, (6144, 2048), (2048, 1), 0), out=buf570)
        buf571 = buf465; del buf465  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf569, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_178, (5632, 2048), (2048, 1), 0), out=buf571)
        del permute_178
        buf573 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf572, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf563, (6144, 2048), (2048, 1), 0), out=buf573)
        buf574 = reinterpret_tensor(buf563, (6144, 2048), (2048, 1)); del buf563  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf572, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_177, (5632, 2048), (2048, 1), 0), out=buf574)
        del permute_177
        buf579 = buf542; del buf542  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf579, buf571, buf574, primals_102, add_83, buf561, buf562, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_102
        buf581 = empty_strided((6144, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf579, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_176, (2048, 2048), (2048, 1), 0), out=buf581)
        del permute_176
        buf582 = empty_strided((3, 16, 2048, 128), (4194304, 262144, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf581, buf582, 12582912, grid=grid(12582912), stream=stream0)
        buf584 = buf612; del buf612  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf582, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf558, (48, 128, 2048), (262144, 1, 128), 0), out=buf584)
        buf586 = reinterpret_tensor(buf554, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf554  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf584, buf557, slice_3, buf586, 98304, 2048, grid=grid(98304), stream=stream0)
        buf587 = reinterpret_tensor(buf558, (48, 128, 2048), (262144, 2048, 1)); del buf558  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf552, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf586, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf587)
        buf588 = reinterpret_tensor(buf552, (48, 2048, 128), (262144, 128, 1)); del buf552  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf586, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf553, (48, 2048, 128), (262144, 1, 2048), 0), out=buf588)
        buf594 = reinterpret_tensor(buf545, (3, 2048, 6144), (12582912, 6144, 1)); del buf545  # reuse
        buf592 = reinterpret_tensor(buf594, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf591 = reinterpret_tensor(buf594, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf587, select_3, select_1, buf588, buf592, buf591, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf583 = buf588; del buf588  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf557, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf582, (48, 2048, 128), (262144, 128, 1), 0), out=buf583)
        buf593 = reinterpret_tensor(buf594, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf583, buf593, 12582912, grid=grid(12582912), stream=stream0)
        del buf591
        del buf592
        del buf593
        buf596 = reinterpret_tensor(buf583, (6144, 2048), (2048, 1)); del buf583  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf594, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_168, (6144, 2048), (2048, 1), 0), out=buf596)
        del permute_168
        buf575 = buf539; del buf539  # reuse
        buf597 = buf517; del buf517  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf571, buf574, add_83, buf561, buf562, buf596, buf543, buf575, buf597, 98304, 128, grid=grid(98304), stream=stream0)
        buf576 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf575, buf576, 2048, 48, grid=grid(2048), stream=stream0)
        buf580 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf579, (2048, 6144), (1, 2048), 0), buf560, out=buf580)
        buf595 = buf560; del buf560  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf594, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf544, (6144, 2048), (2048, 1), 0), out=buf595)
        buf598 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf597, buf598, 2048, 48, grid=grid(2048), stream=stream0)
        buf600 = buf579; del buf579  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf600, buf596, primals_99, add_83, buf543, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_83
        del primals_99
        buf616 = reinterpret_tensor(buf596, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf596  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf603, buf616, 12582912, grid=grid(12582912), stream=stream0)
        buf617 = reinterpret_tensor(buf544, (48, 2048, 128), (262144, 128, 1)); del buf544  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf615, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf616, (48, 2048, 128), (262144, 128, 1), 0), out=buf617)
        buf618 = buf574; del buf574  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf617, buf618, 12582912, grid=grid(12582912), stream=stream0)
        buf619 = reinterpret_tensor(buf617, (6144, 2048), (2048, 1)); del buf617  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf618, permute_164, out=buf619)
        buf620 = buf543; del buf543  # reuse
        buf621 = reinterpret_tensor(buf571, (3, 2048, 2048), (4194304, 2048, 1)); del buf571  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_77, buf619, primals_95, buf620, buf621, 6144, 2048, grid=grid(6144), stream=stream0)
        buf622 = reinterpret_tensor(buf572, (6144, 5632), (5632, 1)); del buf572  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf621, (6144, 2048), (2048, 1), 0), permute_165, out=buf622)
        buf623 = reinterpret_tensor(buf569, (6144, 5632), (5632, 1)); del buf569  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf621, (6144, 2048), (2048, 1), 0), permute_166, out=buf623)
        buf626 = reinterpret_tensor(buf566, (6144, 5632), (5632, 1)); del buf566  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf600, (6144, 2048), (2048, 1), 0), permute_615, out=buf626)
        del permute_615
        buf624 = reinterpret_tensor(buf568, (3, 2048, 5632), (11534336, 5632, 1)); del buf568  # reuse
        buf627 = reinterpret_tensor(buf565, (3, 2048, 5632), (11534336, 5632, 1)); del buf565  # reuse
        buf630 = reinterpret_tensor(buf564, (3, 2048, 5632), (11534336, 5632, 1)); del buf564  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf622, buf623, buf626, buf624, buf627, buf630, 34603008, grid=grid(34603008), stream=stream0)
        buf625 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf600, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf624, (6144, 5632), (5632, 1), 0), out=buf625)
        buf628 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf627, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf621, (6144, 2048), (2048, 1), 0), out=buf628)
        buf629 = buf561; del buf561  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf627, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_166, (5632, 2048), (2048, 1), 0), out=buf629)
        del permute_166
        buf631 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf630, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf621, (6144, 2048), (2048, 1), 0), out=buf631)
        buf632 = reinterpret_tensor(buf621, (6144, 2048), (2048, 1)); del buf621  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf630, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_165, (5632, 2048), (2048, 1), 0), out=buf632)
        del permute_165
        buf637 = buf600; del buf600  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf637, buf629, buf632, primals_95, add_77, buf619, buf620, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_95
        buf639 = reinterpret_tensor(buf582, (6144, 2048), (2048, 1)); del buf582  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf637, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_164, (2048, 2048), (2048, 1), 0), out=buf639)
        del permute_164
        buf640 = reinterpret_tensor(buf587, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf587  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf639, buf640, 12582912, grid=grid(12582912), stream=stream0)
        buf642 = reinterpret_tensor(buf557, (48, 2048, 2048), (4194304, 2048, 1)); del buf557  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf640, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf616, (48, 128, 2048), (262144, 1, 128), 0), out=buf642)
        buf644 = buf586; del buf586  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf642, buf615, slice_3, buf644, 98304, 2048, grid=grid(98304), stream=stream0)
        buf645 = reinterpret_tensor(buf616, (48, 128, 2048), (262144, 2048, 1)); del buf616  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf610, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf644, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf645)
        buf646 = reinterpret_tensor(buf610, (48, 2048, 128), (262144, 128, 1)); del buf610  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf644, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf611, (48, 2048, 128), (262144, 1, 2048), 0), out=buf646)
        buf652 = reinterpret_tensor(buf603, (3, 2048, 6144), (12582912, 6144, 1)); del buf603  # reuse
        buf650 = reinterpret_tensor(buf652, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf649 = reinterpret_tensor(buf652, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf645, select_3, select_1, buf646, buf650, buf649, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf641 = buf646; del buf646  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf615, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf640, (48, 2048, 128), (262144, 128, 1), 0), out=buf641)
        buf651 = reinterpret_tensor(buf652, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf641, buf651, 12582912, grid=grid(12582912), stream=stream0)
        del buf649
        del buf650
        del buf651
        buf654 = reinterpret_tensor(buf641, (6144, 2048), (2048, 1)); del buf641  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf652, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_156, (6144, 2048), (2048, 1), 0), out=buf654)
        del permute_156
        buf633 = buf597; del buf597  # reuse
        buf655 = buf575; del buf575  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf629, buf632, add_77, buf619, buf620, buf654, buf601, buf633, buf655, 98304, 128, grid=grid(98304), stream=stream0)
        buf634 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf633, buf634, 2048, 48, grid=grid(2048), stream=stream0)
        buf638 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf637, (2048, 6144), (1, 2048), 0), buf618, out=buf638)
        buf653 = buf618; del buf618  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf652, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf602, (6144, 2048), (2048, 1), 0), out=buf653)
        buf656 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf655, buf656, 2048, 48, grid=grid(2048), stream=stream0)
        buf658 = buf637; del buf637  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf658, buf654, primals_92, add_77, buf601, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_77
        del primals_92
        buf659 = buf601; del buf601  # reuse
        buf660 = reinterpret_tensor(buf654, (3, 2048, 2048), (4194304, 2048, 1)); del buf654  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_71, primals_85, buf659, buf660, 6144, 2048, grid=grid(6144), stream=stream0)
        buf661 = reinterpret_tensor(buf652, (6144, 6144), (6144, 1)); del buf652  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf660, (6144, 2048), (2048, 1), 0), permute_144, out=buf661)
        buf717 = buf620; del buf620  # reuse
        buf718 = buf602; del buf602  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_65, primals_78, buf717, buf718, 6144, 2048, grid=grid(6144), stream=stream0)
        buf719 = reinterpret_tensor(buf594, (6144, 6144), (6144, 1)); del buf594  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf718, (6144, 2048), (2048, 1), 0), permute_132, out=buf719)
        buf664 = buf606; del buf606  # reuse
        buf662 = reinterpret_tensor(buf664, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf663 = reinterpret_tensor(buf664, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf667 = buf609; del buf609  # reuse
        buf665 = reinterpret_tensor(buf667, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf666 = reinterpret_tensor(buf667, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf722 = buf548; del buf548  # reuse
        buf720 = reinterpret_tensor(buf722, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf721 = reinterpret_tensor(buf722, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf725 = buf551; del buf551  # reuse
        buf723 = reinterpret_tensor(buf725, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf724 = reinterpret_tensor(buf725, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_27.run(buf661, select_1, select_3, buf719, buf662, buf663, buf665, buf666, buf720, buf721, buf723, buf724, 6291456, grid=grid(6291456), stream=stream0)
        buf668 = reinterpret_tensor(buf632, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf632  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf667, buf668, 12582912, grid=grid(12582912), stream=stream0)
        del buf662
        del buf663
        del buf665
        del buf666
        buf669 = reinterpret_tensor(buf629, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf629  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf664, buf669, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf670 = reinterpret_tensor(buf615, (48, 2048, 2048), (4194304, 2048, 1)); del buf615  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf668, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf669, (48, 128, 2048), (262144, 2048, 1), 0), out=buf670)
        buf726 = reinterpret_tensor(buf619, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf619  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf725, buf726, 12582912, grid=grid(12582912), stream=stream0)
        del buf723
        del buf724
        buf727 = reinterpret_tensor(buf640, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf640  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf722, buf727, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        del buf720
        del buf721
        buf728 = reinterpret_tensor(buf644, (48, 2048, 2048), (4194304, 2048, 1)); del buf644  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf726, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf727, (48, 128, 2048), (262144, 2048, 1), 0), out=buf728)
        buf673 = reinterpret_tensor(buf642, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf642  # reuse
        buf731 = reinterpret_tensor(buf584, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf584  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_28.run(slice_3, buf670, scalar_tensor, buf728, buf673, buf731, 98304, 2048, grid=grid(98304), stream=stream0)
        buf674 = reinterpret_tensor(buf645, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf645  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf661, buf674, 12582912, grid=grid(12582912), stream=stream0)
        buf675 = reinterpret_tensor(buf611, (48, 2048, 128), (262144, 128, 1)); del buf611  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf673, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf674, (48, 2048, 128), (262144, 128, 1), 0), out=buf675)
        buf676 = buf639; del buf639  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf675, buf676, 12582912, grid=grid(12582912), stream=stream0)
        buf677 = reinterpret_tensor(buf675, (6144, 2048), (2048, 1)); del buf675  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf676, permute_152, out=buf677)
        buf678 = buf562; del buf562  # reuse
        buf679 = reinterpret_tensor(buf553, (3, 2048, 2048), (4194304, 2048, 1)); del buf553  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_71, buf677, primals_88, buf678, buf679, 6144, 2048, grid=grid(6144), stream=stream0)
        buf680 = reinterpret_tensor(buf630, (6144, 5632), (5632, 1)); del buf630  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf679, (6144, 2048), (2048, 1), 0), permute_153, out=buf680)
        buf681 = reinterpret_tensor(buf627, (6144, 5632), (5632, 1)); del buf627  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf679, (6144, 2048), (2048, 1), 0), permute_154, out=buf681)
        buf684 = reinterpret_tensor(buf624, (6144, 5632), (5632, 1)); del buf624  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf658, (6144, 2048), (2048, 1), 0), permute_647, out=buf684)
        del permute_647
        buf682 = reinterpret_tensor(buf626, (3, 2048, 5632), (11534336, 5632, 1)); del buf626  # reuse
        buf685 = reinterpret_tensor(buf623, (3, 2048, 5632), (11534336, 5632, 1)); del buf623  # reuse
        buf688 = reinterpret_tensor(buf622, (3, 2048, 5632), (11534336, 5632, 1)); del buf622  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf680, buf681, buf684, buf682, buf685, buf688, 34603008, grid=grid(34603008), stream=stream0)
        buf683 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf658, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf682, (6144, 5632), (5632, 1), 0), out=buf683)
        buf686 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf685, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf679, (6144, 2048), (2048, 1), 0), out=buf686)
        buf687 = buf581; del buf581  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf685, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_154, (5632, 2048), (2048, 1), 0), out=buf687)
        del permute_154
        buf689 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf688, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf679, (6144, 2048), (2048, 1), 0), out=buf689)
        buf690 = reinterpret_tensor(buf679, (6144, 2048), (2048, 1)); del buf679  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf688, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_153, (5632, 2048), (2048, 1), 0), out=buf690)
        del permute_153
        buf695 = buf658; del buf658  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf695, buf687, buf690, primals_88, add_71, buf677, buf678, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_88
        buf697 = empty_strided((6144, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf695, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_152, (2048, 2048), (2048, 1), 0), out=buf697)
        del permute_152
        buf698 = empty_strided((3, 16, 2048, 128), (4194304, 262144, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf697, buf698, 12582912, grid=grid(12582912), stream=stream0)
        buf700 = buf728; del buf728  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf698, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf674, (48, 128, 2048), (262144, 1, 128), 0), out=buf700)
        buf702 = reinterpret_tensor(buf670, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf670  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf700, buf673, slice_3, buf702, 98304, 2048, grid=grid(98304), stream=stream0)
        buf703 = reinterpret_tensor(buf674, (48, 128, 2048), (262144, 2048, 1)); del buf674  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf668, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf702, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf703)
        buf704 = reinterpret_tensor(buf668, (48, 2048, 128), (262144, 128, 1)); del buf668  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf702, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf669, (48, 2048, 128), (262144, 1, 2048), 0), out=buf704)
        buf710 = reinterpret_tensor(buf661, (3, 2048, 6144), (12582912, 6144, 1)); del buf661  # reuse
        buf708 = reinterpret_tensor(buf710, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf707 = reinterpret_tensor(buf710, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf703, select_3, select_1, buf704, buf708, buf707, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf699 = buf704; del buf704  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf673, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf698, (48, 2048, 128), (262144, 128, 1), 0), out=buf699)
        buf709 = reinterpret_tensor(buf710, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf699, buf709, 12582912, grid=grid(12582912), stream=stream0)
        del buf707
        del buf708
        del buf709
        buf712 = reinterpret_tensor(buf699, (6144, 2048), (2048, 1)); del buf699  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf710, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_144, (6144, 2048), (2048, 1), 0), out=buf712)
        del permute_144
        buf691 = buf655; del buf655  # reuse
        buf713 = buf633; del buf633  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf687, buf690, add_71, buf677, buf678, buf712, buf659, buf691, buf713, 98304, 128, grid=grid(98304), stream=stream0)
        buf692 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf691, buf692, 2048, 48, grid=grid(2048), stream=stream0)
        buf696 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf695, (2048, 6144), (1, 2048), 0), buf676, out=buf696)
        buf711 = buf676; del buf676  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf710, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf660, (6144, 2048), (2048, 1), 0), out=buf711)
        buf714 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf713, buf714, 2048, 48, grid=grid(2048), stream=stream0)
        buf716 = buf695; del buf695  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf716, buf712, primals_85, add_71, buf659, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_71
        del primals_85
        buf732 = reinterpret_tensor(buf712, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf712  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf719, buf732, 12582912, grid=grid(12582912), stream=stream0)
        buf733 = reinterpret_tensor(buf660, (48, 2048, 128), (262144, 128, 1)); del buf660  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf731, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf732, (48, 2048, 128), (262144, 128, 1), 0), out=buf733)
        buf734 = buf690; del buf690  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf733, buf734, 12582912, grid=grid(12582912), stream=stream0)
        buf735 = reinterpret_tensor(buf733, (6144, 2048), (2048, 1)); del buf733  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf734, permute_140, out=buf735)
        buf736 = buf659; del buf659  # reuse
        buf737 = reinterpret_tensor(buf687, (3, 2048, 2048), (4194304, 2048, 1)); del buf687  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_65, buf735, primals_81, buf736, buf737, 6144, 2048, grid=grid(6144), stream=stream0)
        buf738 = reinterpret_tensor(buf688, (6144, 5632), (5632, 1)); del buf688  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf737, (6144, 2048), (2048, 1), 0), permute_141, out=buf738)
        buf739 = reinterpret_tensor(buf685, (6144, 5632), (5632, 1)); del buf685  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf737, (6144, 2048), (2048, 1), 0), permute_142, out=buf739)
        buf742 = reinterpret_tensor(buf682, (6144, 5632), (5632, 1)); del buf682  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf716, (6144, 2048), (2048, 1), 0), permute_679, out=buf742)
        del permute_679
        buf740 = reinterpret_tensor(buf684, (3, 2048, 5632), (11534336, 5632, 1)); del buf684  # reuse
        buf743 = reinterpret_tensor(buf681, (3, 2048, 5632), (11534336, 5632, 1)); del buf681  # reuse
        buf746 = reinterpret_tensor(buf680, (3, 2048, 5632), (11534336, 5632, 1)); del buf680  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf738, buf739, buf742, buf740, buf743, buf746, 34603008, grid=grid(34603008), stream=stream0)
        buf741 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf716, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf740, (6144, 5632), (5632, 1), 0), out=buf741)
        buf744 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf743, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf737, (6144, 2048), (2048, 1), 0), out=buf744)
        buf745 = buf677; del buf677  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf743, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_142, (5632, 2048), (2048, 1), 0), out=buf745)
        del permute_142
        buf747 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf746, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf737, (6144, 2048), (2048, 1), 0), out=buf747)
        buf748 = reinterpret_tensor(buf737, (6144, 2048), (2048, 1)); del buf737  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf746, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_141, (5632, 2048), (2048, 1), 0), out=buf748)
        del permute_141
        buf753 = buf716; del buf716  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf753, buf745, buf748, primals_81, add_65, buf735, buf736, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_81
        buf755 = reinterpret_tensor(buf698, (6144, 2048), (2048, 1)); del buf698  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf753, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_140, (2048, 2048), (2048, 1), 0), out=buf755)
        del permute_140
        buf756 = reinterpret_tensor(buf703, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf703  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf755, buf756, 12582912, grid=grid(12582912), stream=stream0)
        buf758 = reinterpret_tensor(buf673, (48, 2048, 2048), (4194304, 2048, 1)); del buf673  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf756, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf732, (48, 128, 2048), (262144, 1, 128), 0), out=buf758)
        buf760 = buf702; del buf702  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf758, buf731, slice_3, buf760, 98304, 2048, grid=grid(98304), stream=stream0)
        buf761 = reinterpret_tensor(buf732, (48, 128, 2048), (262144, 2048, 1)); del buf732  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf726, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf760, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf761)
        buf762 = reinterpret_tensor(buf726, (48, 2048, 128), (262144, 128, 1)); del buf726  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf760, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf727, (48, 2048, 128), (262144, 1, 2048), 0), out=buf762)
        buf768 = reinterpret_tensor(buf719, (3, 2048, 6144), (12582912, 6144, 1)); del buf719  # reuse
        buf766 = reinterpret_tensor(buf768, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf765 = reinterpret_tensor(buf768, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf761, select_3, select_1, buf762, buf766, buf765, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf757 = buf762; del buf762  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf731, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf756, (48, 2048, 128), (262144, 128, 1), 0), out=buf757)
        buf767 = reinterpret_tensor(buf768, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf757, buf767, 12582912, grid=grid(12582912), stream=stream0)
        del buf765
        del buf766
        del buf767
        buf770 = reinterpret_tensor(buf757, (6144, 2048), (2048, 1)); del buf757  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf768, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_132, (6144, 2048), (2048, 1), 0), out=buf770)
        del permute_132
        buf749 = buf713; del buf713  # reuse
        buf771 = buf691; del buf691  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf745, buf748, add_65, buf735, buf736, buf770, buf717, buf749, buf771, 98304, 128, grid=grid(98304), stream=stream0)
        buf750 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf749, buf750, 2048, 48, grid=grid(2048), stream=stream0)
        buf754 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf753, (2048, 6144), (1, 2048), 0), buf734, out=buf754)
        buf769 = buf734; del buf734  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf768, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf718, (6144, 2048), (2048, 1), 0), out=buf769)
        buf772 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf771, buf772, 2048, 48, grid=grid(2048), stream=stream0)
        buf774 = buf753; del buf753  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf774, buf770, primals_78, add_65, buf717, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_65
        del primals_78
        buf775 = buf717; del buf717  # reuse
        buf776 = reinterpret_tensor(buf770, (3, 2048, 2048), (4194304, 2048, 1)); del buf770  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_59, primals_71, buf775, buf776, 6144, 2048, grid=grid(6144), stream=stream0)
        buf777 = reinterpret_tensor(buf768, (6144, 6144), (6144, 1)); del buf768  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf776, (6144, 2048), (2048, 1), 0), permute_120, out=buf777)
        buf833 = buf736; del buf736  # reuse
        buf834 = buf718; del buf718  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_53, primals_64, buf833, buf834, 6144, 2048, grid=grid(6144), stream=stream0)
        buf835 = reinterpret_tensor(buf710, (6144, 6144), (6144, 1)); del buf710  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf834, (6144, 2048), (2048, 1), 0), permute_108, out=buf835)
        buf780 = buf722; del buf722  # reuse
        buf778 = reinterpret_tensor(buf780, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf779 = reinterpret_tensor(buf780, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf783 = buf725; del buf725  # reuse
        buf781 = reinterpret_tensor(buf783, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf782 = reinterpret_tensor(buf783, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf838 = buf664; del buf664  # reuse
        buf836 = reinterpret_tensor(buf838, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf837 = reinterpret_tensor(buf838, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf841 = buf667; del buf667  # reuse
        buf839 = reinterpret_tensor(buf841, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf840 = reinterpret_tensor(buf841, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_27.run(buf777, select_1, select_3, buf835, buf778, buf779, buf781, buf782, buf836, buf837, buf839, buf840, 6291456, grid=grid(6291456), stream=stream0)
        buf784 = reinterpret_tensor(buf748, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf748  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf783, buf784, 12582912, grid=grid(12582912), stream=stream0)
        del buf778
        del buf779
        del buf781
        del buf782
        buf785 = reinterpret_tensor(buf745, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf745  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf780, buf785, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf786 = reinterpret_tensor(buf731, (48, 2048, 2048), (4194304, 2048, 1)); del buf731  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf784, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf785, (48, 128, 2048), (262144, 2048, 1), 0), out=buf786)
        buf842 = reinterpret_tensor(buf735, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf735  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf841, buf842, 12582912, grid=grid(12582912), stream=stream0)
        del buf839
        del buf840
        buf843 = reinterpret_tensor(buf756, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf756  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf838, buf843, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        del buf836
        del buf837
        buf844 = reinterpret_tensor(buf760, (48, 2048, 2048), (4194304, 2048, 1)); del buf760  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf842, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf843, (48, 128, 2048), (262144, 2048, 1), 0), out=buf844)
        buf789 = reinterpret_tensor(buf758, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf758  # reuse
        buf847 = reinterpret_tensor(buf700, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf700  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_28.run(slice_3, buf786, scalar_tensor, buf844, buf789, buf847, 98304, 2048, grid=grid(98304), stream=stream0)
        buf790 = reinterpret_tensor(buf761, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf761  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf777, buf790, 12582912, grid=grid(12582912), stream=stream0)
        buf791 = reinterpret_tensor(buf727, (48, 2048, 128), (262144, 128, 1)); del buf727  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf789, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf790, (48, 2048, 128), (262144, 128, 1), 0), out=buf791)
        buf792 = buf755; del buf755  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf791, buf792, 12582912, grid=grid(12582912), stream=stream0)
        buf793 = reinterpret_tensor(buf791, (6144, 2048), (2048, 1)); del buf791  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf792, permute_128, out=buf793)
        buf794 = buf678; del buf678  # reuse
        buf795 = reinterpret_tensor(buf669, (3, 2048, 2048), (4194304, 2048, 1)); del buf669  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_59, buf793, primals_74, buf794, buf795, 6144, 2048, grid=grid(6144), stream=stream0)
        buf796 = reinterpret_tensor(buf746, (6144, 5632), (5632, 1)); del buf746  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf795, (6144, 2048), (2048, 1), 0), permute_129, out=buf796)
        buf797 = reinterpret_tensor(buf743, (6144, 5632), (5632, 1)); del buf743  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf795, (6144, 2048), (2048, 1), 0), permute_130, out=buf797)
        buf800 = reinterpret_tensor(buf740, (6144, 5632), (5632, 1)); del buf740  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf774, (6144, 2048), (2048, 1), 0), permute_711, out=buf800)
        del permute_711
        buf798 = reinterpret_tensor(buf742, (3, 2048, 5632), (11534336, 5632, 1)); del buf742  # reuse
        buf801 = reinterpret_tensor(buf739, (3, 2048, 5632), (11534336, 5632, 1)); del buf739  # reuse
        buf804 = reinterpret_tensor(buf738, (3, 2048, 5632), (11534336, 5632, 1)); del buf738  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf796, buf797, buf800, buf798, buf801, buf804, 34603008, grid=grid(34603008), stream=stream0)
        buf799 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf774, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf798, (6144, 5632), (5632, 1), 0), out=buf799)
        buf802 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf801, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf795, (6144, 2048), (2048, 1), 0), out=buf802)
        buf803 = buf697; del buf697  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf801, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_130, (5632, 2048), (2048, 1), 0), out=buf803)
        del permute_130
        buf805 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf804, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf795, (6144, 2048), (2048, 1), 0), out=buf805)
        buf806 = reinterpret_tensor(buf795, (6144, 2048), (2048, 1)); del buf795  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf804, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_129, (5632, 2048), (2048, 1), 0), out=buf806)
        del permute_129
        buf811 = buf774; del buf774  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf811, buf803, buf806, primals_74, add_59, buf793, buf794, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_74
        buf813 = empty_strided((6144, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf811, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_128, (2048, 2048), (2048, 1), 0), out=buf813)
        del permute_128
        buf814 = empty_strided((3, 16, 2048, 128), (4194304, 262144, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf813, buf814, 12582912, grid=grid(12582912), stream=stream0)
        buf816 = buf844; del buf844  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf814, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf790, (48, 128, 2048), (262144, 1, 128), 0), out=buf816)
        buf818 = reinterpret_tensor(buf786, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf786  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf816, buf789, slice_3, buf818, 98304, 2048, grid=grid(98304), stream=stream0)
        buf819 = reinterpret_tensor(buf790, (48, 128, 2048), (262144, 2048, 1)); del buf790  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf784, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf818, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf819)
        buf820 = reinterpret_tensor(buf784, (48, 2048, 128), (262144, 128, 1)); del buf784  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf818, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf785, (48, 2048, 128), (262144, 1, 2048), 0), out=buf820)
        buf826 = reinterpret_tensor(buf777, (3, 2048, 6144), (12582912, 6144, 1)); del buf777  # reuse
        buf824 = reinterpret_tensor(buf826, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf823 = reinterpret_tensor(buf826, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf819, select_3, select_1, buf820, buf824, buf823, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf815 = buf820; del buf820  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf789, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf814, (48, 2048, 128), (262144, 128, 1), 0), out=buf815)
        buf825 = reinterpret_tensor(buf826, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf815, buf825, 12582912, grid=grid(12582912), stream=stream0)
        del buf823
        del buf824
        del buf825
        buf828 = reinterpret_tensor(buf815, (6144, 2048), (2048, 1)); del buf815  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf826, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_120, (6144, 2048), (2048, 1), 0), out=buf828)
        del permute_120
        buf807 = buf771; del buf771  # reuse
        buf829 = buf749; del buf749  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf803, buf806, add_59, buf793, buf794, buf828, buf775, buf807, buf829, 98304, 128, grid=grid(98304), stream=stream0)
        buf808 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf807, buf808, 2048, 48, grid=grid(2048), stream=stream0)
        buf812 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf811, (2048, 6144), (1, 2048), 0), buf792, out=buf812)
        buf827 = buf792; del buf792  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf826, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf776, (6144, 2048), (2048, 1), 0), out=buf827)
        buf830 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf829, buf830, 2048, 48, grid=grid(2048), stream=stream0)
        buf832 = buf811; del buf811  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf832, buf828, primals_71, add_59, buf775, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_59
        del primals_71
        buf848 = reinterpret_tensor(buf828, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf828  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf835, buf848, 12582912, grid=grid(12582912), stream=stream0)
        buf849 = reinterpret_tensor(buf776, (48, 2048, 128), (262144, 128, 1)); del buf776  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf847, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf848, (48, 2048, 128), (262144, 128, 1), 0), out=buf849)
        buf850 = buf806; del buf806  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf849, buf850, 12582912, grid=grid(12582912), stream=stream0)
        buf851 = reinterpret_tensor(buf849, (6144, 2048), (2048, 1)); del buf849  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf850, permute_116, out=buf851)
        buf852 = buf775; del buf775  # reuse
        buf853 = reinterpret_tensor(buf803, (3, 2048, 2048), (4194304, 2048, 1)); del buf803  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_53, buf851, primals_67, buf852, buf853, 6144, 2048, grid=grid(6144), stream=stream0)
        buf854 = reinterpret_tensor(buf804, (6144, 5632), (5632, 1)); del buf804  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf853, (6144, 2048), (2048, 1), 0), permute_117, out=buf854)
        buf855 = reinterpret_tensor(buf801, (6144, 5632), (5632, 1)); del buf801  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf853, (6144, 2048), (2048, 1), 0), permute_118, out=buf855)
        buf858 = reinterpret_tensor(buf798, (6144, 5632), (5632, 1)); del buf798  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf832, (6144, 2048), (2048, 1), 0), permute_743, out=buf858)
        del permute_743
        buf856 = reinterpret_tensor(buf800, (3, 2048, 5632), (11534336, 5632, 1)); del buf800  # reuse
        buf859 = reinterpret_tensor(buf797, (3, 2048, 5632), (11534336, 5632, 1)); del buf797  # reuse
        buf862 = reinterpret_tensor(buf796, (3, 2048, 5632), (11534336, 5632, 1)); del buf796  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf854, buf855, buf858, buf856, buf859, buf862, 34603008, grid=grid(34603008), stream=stream0)
        buf857 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf832, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf856, (6144, 5632), (5632, 1), 0), out=buf857)
        buf860 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf859, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf853, (6144, 2048), (2048, 1), 0), out=buf860)
        buf861 = buf793; del buf793  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf859, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_118, (5632, 2048), (2048, 1), 0), out=buf861)
        del permute_118
        buf863 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf862, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf853, (6144, 2048), (2048, 1), 0), out=buf863)
        buf864 = reinterpret_tensor(buf853, (6144, 2048), (2048, 1)); del buf853  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf862, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_117, (5632, 2048), (2048, 1), 0), out=buf864)
        del permute_117
        buf869 = buf832; del buf832  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf869, buf861, buf864, primals_67, add_53, buf851, buf852, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_67
        buf871 = reinterpret_tensor(buf814, (6144, 2048), (2048, 1)); del buf814  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf869, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_116, (2048, 2048), (2048, 1), 0), out=buf871)
        del permute_116
        buf872 = reinterpret_tensor(buf819, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf819  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf871, buf872, 12582912, grid=grid(12582912), stream=stream0)
        buf874 = reinterpret_tensor(buf789, (48, 2048, 2048), (4194304, 2048, 1)); del buf789  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf872, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf848, (48, 128, 2048), (262144, 1, 128), 0), out=buf874)
        buf876 = buf818; del buf818  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf874, buf847, slice_3, buf876, 98304, 2048, grid=grid(98304), stream=stream0)
        buf877 = reinterpret_tensor(buf848, (48, 128, 2048), (262144, 2048, 1)); del buf848  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf842, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf876, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf877)
        buf878 = reinterpret_tensor(buf842, (48, 2048, 128), (262144, 128, 1)); del buf842  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf876, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf843, (48, 2048, 128), (262144, 1, 2048), 0), out=buf878)
        buf884 = reinterpret_tensor(buf835, (3, 2048, 6144), (12582912, 6144, 1)); del buf835  # reuse
        buf882 = reinterpret_tensor(buf884, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf881 = reinterpret_tensor(buf884, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf877, select_3, select_1, buf878, buf882, buf881, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf873 = buf878; del buf878  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf847, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf872, (48, 2048, 128), (262144, 128, 1), 0), out=buf873)
        buf883 = reinterpret_tensor(buf884, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf873, buf883, 12582912, grid=grid(12582912), stream=stream0)
        del buf881
        del buf882
        del buf883
        buf886 = reinterpret_tensor(buf873, (6144, 2048), (2048, 1)); del buf873  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf884, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_108, (6144, 2048), (2048, 1), 0), out=buf886)
        del permute_108
        buf865 = buf829; del buf829  # reuse
        buf887 = buf807; del buf807  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf861, buf864, add_53, buf851, buf852, buf886, buf833, buf865, buf887, 98304, 128, grid=grid(98304), stream=stream0)
        buf866 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf865, buf866, 2048, 48, grid=grid(2048), stream=stream0)
        buf870 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf869, (2048, 6144), (1, 2048), 0), buf850, out=buf870)
        buf885 = buf850; del buf850  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf884, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf834, (6144, 2048), (2048, 1), 0), out=buf885)
        buf888 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf887, buf888, 2048, 48, grid=grid(2048), stream=stream0)
        buf890 = buf869; del buf869  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf890, buf886, primals_64, add_53, buf833, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_53
        del primals_64
        buf891 = buf833; del buf833  # reuse
        buf892 = reinterpret_tensor(buf886, (3, 2048, 2048), (4194304, 2048, 1)); del buf886  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_47, primals_57, buf891, buf892, 6144, 2048, grid=grid(6144), stream=stream0)
        buf893 = reinterpret_tensor(buf884, (6144, 6144), (6144, 1)); del buf884  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf892, (6144, 2048), (2048, 1), 0), permute_96, out=buf893)
        buf949 = buf852; del buf852  # reuse
        buf950 = buf834; del buf834  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_41, primals_50, buf949, buf950, 6144, 2048, grid=grid(6144), stream=stream0)
        buf951 = reinterpret_tensor(buf826, (6144, 6144), (6144, 1)); del buf826  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf950, (6144, 2048), (2048, 1), 0), permute_84, out=buf951)
        buf896 = buf838; del buf838  # reuse
        buf894 = reinterpret_tensor(buf896, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf895 = reinterpret_tensor(buf896, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf899 = buf841; del buf841  # reuse
        buf897 = reinterpret_tensor(buf899, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf898 = reinterpret_tensor(buf899, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf954 = buf780; del buf780  # reuse
        buf952 = reinterpret_tensor(buf954, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf953 = reinterpret_tensor(buf954, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf957 = buf783; del buf783  # reuse
        buf955 = reinterpret_tensor(buf957, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf956 = reinterpret_tensor(buf957, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_27.run(buf893, select_1, select_3, buf951, buf894, buf895, buf897, buf898, buf952, buf953, buf955, buf956, 6291456, grid=grid(6291456), stream=stream0)
        buf900 = reinterpret_tensor(buf864, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf864  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf899, buf900, 12582912, grid=grid(12582912), stream=stream0)
        del buf894
        del buf895
        del buf897
        del buf898
        buf901 = reinterpret_tensor(buf861, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf861  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf896, buf901, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf902 = reinterpret_tensor(buf847, (48, 2048, 2048), (4194304, 2048, 1)); del buf847  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf900, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf901, (48, 128, 2048), (262144, 2048, 1), 0), out=buf902)
        buf958 = reinterpret_tensor(buf851, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf851  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf957, buf958, 12582912, grid=grid(12582912), stream=stream0)
        del buf955
        del buf956
        buf959 = reinterpret_tensor(buf872, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf872  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf954, buf959, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        del buf952
        del buf953
        buf960 = reinterpret_tensor(buf876, (48, 2048, 2048), (4194304, 2048, 1)); del buf876  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf958, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf959, (48, 128, 2048), (262144, 2048, 1), 0), out=buf960)
        buf905 = reinterpret_tensor(buf874, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf874  # reuse
        buf963 = reinterpret_tensor(buf816, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf816  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_28.run(slice_3, buf902, scalar_tensor, buf960, buf905, buf963, 98304, 2048, grid=grid(98304), stream=stream0)
        buf906 = reinterpret_tensor(buf877, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf877  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf893, buf906, 12582912, grid=grid(12582912), stream=stream0)
        buf907 = reinterpret_tensor(buf843, (48, 2048, 128), (262144, 128, 1)); del buf843  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf905, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf906, (48, 2048, 128), (262144, 128, 1), 0), out=buf907)
        buf908 = buf871; del buf871  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf907, buf908, 12582912, grid=grid(12582912), stream=stream0)
        buf909 = reinterpret_tensor(buf907, (6144, 2048), (2048, 1)); del buf907  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf908, permute_104, out=buf909)
        buf910 = buf794; del buf794  # reuse
        buf911 = reinterpret_tensor(buf785, (3, 2048, 2048), (4194304, 2048, 1)); del buf785  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_47, buf909, primals_60, buf910, buf911, 6144, 2048, grid=grid(6144), stream=stream0)
        buf912 = reinterpret_tensor(buf862, (6144, 5632), (5632, 1)); del buf862  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf911, (6144, 2048), (2048, 1), 0), permute_105, out=buf912)
        buf913 = reinterpret_tensor(buf859, (6144, 5632), (5632, 1)); del buf859  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf911, (6144, 2048), (2048, 1), 0), permute_106, out=buf913)
        buf916 = reinterpret_tensor(buf856, (6144, 5632), (5632, 1)); del buf856  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf890, (6144, 2048), (2048, 1), 0), permute_775, out=buf916)
        del permute_775
        buf914 = reinterpret_tensor(buf858, (3, 2048, 5632), (11534336, 5632, 1)); del buf858  # reuse
        buf917 = reinterpret_tensor(buf855, (3, 2048, 5632), (11534336, 5632, 1)); del buf855  # reuse
        buf920 = reinterpret_tensor(buf854, (3, 2048, 5632), (11534336, 5632, 1)); del buf854  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf912, buf913, buf916, buf914, buf917, buf920, 34603008, grid=grid(34603008), stream=stream0)
        buf915 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf890, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf914, (6144, 5632), (5632, 1), 0), out=buf915)
        buf918 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf917, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf911, (6144, 2048), (2048, 1), 0), out=buf918)
        buf919 = buf813; del buf813  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf917, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_106, (5632, 2048), (2048, 1), 0), out=buf919)
        del permute_106
        buf921 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf920, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf911, (6144, 2048), (2048, 1), 0), out=buf921)
        buf922 = reinterpret_tensor(buf911, (6144, 2048), (2048, 1)); del buf911  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf920, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_105, (5632, 2048), (2048, 1), 0), out=buf922)
        del permute_105
        buf927 = buf890; del buf890  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf927, buf919, buf922, primals_60, add_47, buf909, buf910, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_60
        buf929 = empty_strided((6144, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf927, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_104, (2048, 2048), (2048, 1), 0), out=buf929)
        del permute_104
        buf930 = empty_strided((3, 16, 2048, 128), (4194304, 262144, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf929, buf930, 12582912, grid=grid(12582912), stream=stream0)
        buf932 = buf960; del buf960  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf930, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf906, (48, 128, 2048), (262144, 1, 128), 0), out=buf932)
        buf934 = reinterpret_tensor(buf902, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf902  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf932, buf905, slice_3, buf934, 98304, 2048, grid=grid(98304), stream=stream0)
        buf935 = reinterpret_tensor(buf906, (48, 128, 2048), (262144, 2048, 1)); del buf906  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf900, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf934, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf935)
        buf936 = reinterpret_tensor(buf900, (48, 2048, 128), (262144, 128, 1)); del buf900  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf934, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf901, (48, 2048, 128), (262144, 1, 2048), 0), out=buf936)
        buf942 = reinterpret_tensor(buf893, (3, 2048, 6144), (12582912, 6144, 1)); del buf893  # reuse
        buf940 = reinterpret_tensor(buf942, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf939 = reinterpret_tensor(buf942, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf935, select_3, select_1, buf936, buf940, buf939, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf931 = buf936; del buf936  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf905, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf930, (48, 2048, 128), (262144, 128, 1), 0), out=buf931)
        buf941 = reinterpret_tensor(buf942, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf931, buf941, 12582912, grid=grid(12582912), stream=stream0)
        del buf939
        del buf940
        del buf941
        buf944 = reinterpret_tensor(buf931, (6144, 2048), (2048, 1)); del buf931  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf942, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_96, (6144, 2048), (2048, 1), 0), out=buf944)
        del permute_96
        buf923 = buf887; del buf887  # reuse
        buf945 = buf865; del buf865  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf919, buf922, add_47, buf909, buf910, buf944, buf891, buf923, buf945, 98304, 128, grid=grid(98304), stream=stream0)
        buf924 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf923, buf924, 2048, 48, grid=grid(2048), stream=stream0)
        buf928 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf927, (2048, 6144), (1, 2048), 0), buf908, out=buf928)
        buf943 = buf908; del buf908  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf942, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf892, (6144, 2048), (2048, 1), 0), out=buf943)
        buf946 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf945, buf946, 2048, 48, grid=grid(2048), stream=stream0)
        buf948 = buf927; del buf927  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf948, buf944, primals_57, add_47, buf891, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_47
        del primals_57
        buf964 = reinterpret_tensor(buf944, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf944  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf951, buf964, 12582912, grid=grid(12582912), stream=stream0)
        buf965 = reinterpret_tensor(buf892, (48, 2048, 128), (262144, 128, 1)); del buf892  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf963, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf964, (48, 2048, 128), (262144, 128, 1), 0), out=buf965)
        buf966 = buf922; del buf922  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf965, buf966, 12582912, grid=grid(12582912), stream=stream0)
        buf967 = reinterpret_tensor(buf965, (6144, 2048), (2048, 1)); del buf965  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf966, permute_92, out=buf967)
        buf968 = buf891; del buf891  # reuse
        buf969 = reinterpret_tensor(buf919, (3, 2048, 2048), (4194304, 2048, 1)); del buf919  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_41, buf967, primals_53, buf968, buf969, 6144, 2048, grid=grid(6144), stream=stream0)
        buf970 = reinterpret_tensor(buf920, (6144, 5632), (5632, 1)); del buf920  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf969, (6144, 2048), (2048, 1), 0), permute_93, out=buf970)
        buf971 = reinterpret_tensor(buf917, (6144, 5632), (5632, 1)); del buf917  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf969, (6144, 2048), (2048, 1), 0), permute_94, out=buf971)
        buf974 = reinterpret_tensor(buf914, (6144, 5632), (5632, 1)); del buf914  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf948, (6144, 2048), (2048, 1), 0), permute_807, out=buf974)
        del permute_807
        buf972 = reinterpret_tensor(buf916, (3, 2048, 5632), (11534336, 5632, 1)); del buf916  # reuse
        buf975 = reinterpret_tensor(buf913, (3, 2048, 5632), (11534336, 5632, 1)); del buf913  # reuse
        buf978 = reinterpret_tensor(buf912, (3, 2048, 5632), (11534336, 5632, 1)); del buf912  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf970, buf971, buf974, buf972, buf975, buf978, 34603008, grid=grid(34603008), stream=stream0)
        buf973 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf948, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf972, (6144, 5632), (5632, 1), 0), out=buf973)
        buf976 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf975, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf969, (6144, 2048), (2048, 1), 0), out=buf976)
        buf977 = buf909; del buf909  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf975, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_94, (5632, 2048), (2048, 1), 0), out=buf977)
        del permute_94
        buf979 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf978, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf969, (6144, 2048), (2048, 1), 0), out=buf979)
        buf980 = reinterpret_tensor(buf969, (6144, 2048), (2048, 1)); del buf969  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf978, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_93, (5632, 2048), (2048, 1), 0), out=buf980)
        del permute_93
        buf985 = buf948; del buf948  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf985, buf977, buf980, primals_53, add_41, buf967, buf968, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_53
        buf987 = reinterpret_tensor(buf930, (6144, 2048), (2048, 1)); del buf930  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf985, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_92, (2048, 2048), (2048, 1), 0), out=buf987)
        del permute_92
        buf988 = reinterpret_tensor(buf935, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf935  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf987, buf988, 12582912, grid=grid(12582912), stream=stream0)
        buf990 = reinterpret_tensor(buf905, (48, 2048, 2048), (4194304, 2048, 1)); del buf905  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf988, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf964, (48, 128, 2048), (262144, 1, 128), 0), out=buf990)
        buf992 = buf934; del buf934  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf990, buf963, slice_3, buf992, 98304, 2048, grid=grid(98304), stream=stream0)
        buf993 = reinterpret_tensor(buf964, (48, 128, 2048), (262144, 2048, 1)); del buf964  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf958, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf992, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf993)
        buf994 = reinterpret_tensor(buf958, (48, 2048, 128), (262144, 128, 1)); del buf958  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf992, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf959, (48, 2048, 128), (262144, 1, 2048), 0), out=buf994)
        buf1000 = reinterpret_tensor(buf951, (3, 2048, 6144), (12582912, 6144, 1)); del buf951  # reuse
        buf998 = reinterpret_tensor(buf1000, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf997 = reinterpret_tensor(buf1000, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf993, select_3, select_1, buf994, buf998, buf997, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf989 = buf994; del buf994  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf963, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf988, (48, 2048, 128), (262144, 128, 1), 0), out=buf989)
        buf999 = reinterpret_tensor(buf1000, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf989, buf999, 12582912, grid=grid(12582912), stream=stream0)
        del buf997
        del buf998
        del buf999
        buf1002 = reinterpret_tensor(buf989, (6144, 2048), (2048, 1)); del buf989  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1000, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_84, (6144, 2048), (2048, 1), 0), out=buf1002)
        del permute_84
        buf981 = buf945; del buf945  # reuse
        buf1003 = buf923; del buf923  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf977, buf980, add_41, buf967, buf968, buf1002, buf949, buf981, buf1003, 98304, 128, grid=grid(98304), stream=stream0)
        buf982 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf981, buf982, 2048, 48, grid=grid(2048), stream=stream0)
        buf986 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf985, (2048, 6144), (1, 2048), 0), buf966, out=buf986)
        buf1001 = buf966; del buf966  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1000, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf950, (6144, 2048), (2048, 1), 0), out=buf1001)
        buf1004 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf1003, buf1004, 2048, 48, grid=grid(2048), stream=stream0)
        buf1006 = buf985; del buf985  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf1006, buf1002, primals_50, add_41, buf949, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_41
        del primals_50
        buf1007 = buf949; del buf949  # reuse
        buf1008 = reinterpret_tensor(buf1002, (3, 2048, 2048), (4194304, 2048, 1)); del buf1002  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_35, primals_43, buf1007, buf1008, 6144, 2048, grid=grid(6144), stream=stream0)
        buf1009 = reinterpret_tensor(buf1000, (6144, 6144), (6144, 1)); del buf1000  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1008, (6144, 2048), (2048, 1), 0), permute_72, out=buf1009)
        buf1065 = buf968; del buf968  # reuse
        buf1066 = buf950; del buf950  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_29, primals_36, buf1065, buf1066, 6144, 2048, grid=grid(6144), stream=stream0)
        buf1067 = reinterpret_tensor(buf942, (6144, 6144), (6144, 1)); del buf942  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1066, (6144, 2048), (2048, 1), 0), permute_60, out=buf1067)
        buf1012 = buf954; del buf954  # reuse
        buf1010 = reinterpret_tensor(buf1012, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf1011 = reinterpret_tensor(buf1012, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf1015 = buf957; del buf957  # reuse
        buf1013 = reinterpret_tensor(buf1015, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf1014 = reinterpret_tensor(buf1015, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf1070 = buf896; del buf896  # reuse
        buf1068 = reinterpret_tensor(buf1070, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf1069 = reinterpret_tensor(buf1070, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf1073 = buf899; del buf899  # reuse
        buf1071 = reinterpret_tensor(buf1073, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf1072 = reinterpret_tensor(buf1073, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_27.run(buf1009, select_1, select_3, buf1067, buf1010, buf1011, buf1013, buf1014, buf1068, buf1069, buf1071, buf1072, 6291456, grid=grid(6291456), stream=stream0)
        buf1016 = reinterpret_tensor(buf980, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf980  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf1015, buf1016, 12582912, grid=grid(12582912), stream=stream0)
        del buf1010
        del buf1011
        del buf1013
        del buf1014
        buf1017 = reinterpret_tensor(buf977, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf977  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf1012, buf1017, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf1018 = reinterpret_tensor(buf963, (48, 2048, 2048), (4194304, 2048, 1)); del buf963  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1016, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf1017, (48, 128, 2048), (262144, 2048, 1), 0), out=buf1018)
        buf1074 = reinterpret_tensor(buf967, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf967  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf1073, buf1074, 12582912, grid=grid(12582912), stream=stream0)
        del buf1071
        del buf1072
        buf1075 = reinterpret_tensor(buf988, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf988  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf1070, buf1075, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        del buf1068
        del buf1069
        buf1076 = reinterpret_tensor(buf992, (48, 2048, 2048), (4194304, 2048, 1)); del buf992  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1074, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf1075, (48, 128, 2048), (262144, 2048, 1), 0), out=buf1076)
        buf1021 = reinterpret_tensor(buf990, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf990  # reuse
        buf1079 = reinterpret_tensor(buf932, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf932  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_28.run(slice_3, buf1018, scalar_tensor, buf1076, buf1021, buf1079, 98304, 2048, grid=grid(98304), stream=stream0)
        buf1022 = reinterpret_tensor(buf993, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf993  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf1009, buf1022, 12582912, grid=grid(12582912), stream=stream0)
        buf1023 = reinterpret_tensor(buf959, (48, 2048, 128), (262144, 128, 1)); del buf959  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1021, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf1022, (48, 2048, 128), (262144, 128, 1), 0), out=buf1023)
        buf1024 = buf987; del buf987  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf1023, buf1024, 12582912, grid=grid(12582912), stream=stream0)
        buf1025 = reinterpret_tensor(buf1023, (6144, 2048), (2048, 1)); del buf1023  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf1024, permute_80, out=buf1025)
        buf1026 = buf910; del buf910  # reuse
        buf1027 = reinterpret_tensor(buf901, (3, 2048, 2048), (4194304, 2048, 1)); del buf901  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_35, buf1025, primals_46, buf1026, buf1027, 6144, 2048, grid=grid(6144), stream=stream0)
        buf1028 = reinterpret_tensor(buf978, (6144, 5632), (5632, 1)); del buf978  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1027, (6144, 2048), (2048, 1), 0), permute_81, out=buf1028)
        buf1029 = reinterpret_tensor(buf975, (6144, 5632), (5632, 1)); del buf975  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1027, (6144, 2048), (2048, 1), 0), permute_82, out=buf1029)
        buf1032 = reinterpret_tensor(buf972, (6144, 5632), (5632, 1)); del buf972  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1006, (6144, 2048), (2048, 1), 0), permute_839, out=buf1032)
        del permute_839
        buf1030 = reinterpret_tensor(buf974, (3, 2048, 5632), (11534336, 5632, 1)); del buf974  # reuse
        buf1033 = reinterpret_tensor(buf971, (3, 2048, 5632), (11534336, 5632, 1)); del buf971  # reuse
        buf1036 = reinterpret_tensor(buf970, (3, 2048, 5632), (11534336, 5632, 1)); del buf970  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf1028, buf1029, buf1032, buf1030, buf1033, buf1036, 34603008, grid=grid(34603008), stream=stream0)
        buf1031 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1006, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf1030, (6144, 5632), (5632, 1), 0), out=buf1031)
        buf1034 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1033, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf1027, (6144, 2048), (2048, 1), 0), out=buf1034)
        buf1035 = buf929; del buf929  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1033, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_82, (5632, 2048), (2048, 1), 0), out=buf1035)
        del permute_82
        buf1037 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1036, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf1027, (6144, 2048), (2048, 1), 0), out=buf1037)
        buf1038 = reinterpret_tensor(buf1027, (6144, 2048), (2048, 1)); del buf1027  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1036, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_81, (5632, 2048), (2048, 1), 0), out=buf1038)
        del permute_81
        buf1043 = buf1006; del buf1006  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf1043, buf1035, buf1038, primals_46, add_35, buf1025, buf1026, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_46
        buf1045 = empty_strided((6144, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1043, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_80, (2048, 2048), (2048, 1), 0), out=buf1045)
        del permute_80
        buf1046 = empty_strided((3, 16, 2048, 128), (4194304, 262144, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf1045, buf1046, 12582912, grid=grid(12582912), stream=stream0)
        buf1048 = buf1076; del buf1076  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1046, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf1022, (48, 128, 2048), (262144, 1, 128), 0), out=buf1048)
        buf1050 = reinterpret_tensor(buf1018, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf1018  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf1048, buf1021, slice_3, buf1050, 98304, 2048, grid=grid(98304), stream=stream0)
        buf1051 = reinterpret_tensor(buf1022, (48, 128, 2048), (262144, 2048, 1)); del buf1022  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1016, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf1050, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf1051)
        buf1052 = reinterpret_tensor(buf1016, (48, 2048, 128), (262144, 128, 1)); del buf1016  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1050, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf1017, (48, 2048, 128), (262144, 1, 2048), 0), out=buf1052)
        buf1058 = reinterpret_tensor(buf1009, (3, 2048, 6144), (12582912, 6144, 1)); del buf1009  # reuse
        buf1056 = reinterpret_tensor(buf1058, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf1055 = reinterpret_tensor(buf1058, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf1051, select_3, select_1, buf1052, buf1056, buf1055, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf1047 = buf1052; del buf1052  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1021, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf1046, (48, 2048, 128), (262144, 128, 1), 0), out=buf1047)
        buf1057 = reinterpret_tensor(buf1058, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf1047, buf1057, 12582912, grid=grid(12582912), stream=stream0)
        del buf1055
        del buf1056
        del buf1057
        buf1060 = reinterpret_tensor(buf1047, (6144, 2048), (2048, 1)); del buf1047  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1058, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_72, (6144, 2048), (2048, 1), 0), out=buf1060)
        del permute_72
        buf1039 = buf1003; del buf1003  # reuse
        buf1061 = buf981; del buf981  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf1035, buf1038, add_35, buf1025, buf1026, buf1060, buf1007, buf1039, buf1061, 98304, 128, grid=grid(98304), stream=stream0)
        buf1040 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf1039, buf1040, 2048, 48, grid=grid(2048), stream=stream0)
        buf1044 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1043, (2048, 6144), (1, 2048), 0), buf1024, out=buf1044)
        buf1059 = buf1024; del buf1024  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1058, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf1008, (6144, 2048), (2048, 1), 0), out=buf1059)
        buf1062 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf1061, buf1062, 2048, 48, grid=grid(2048), stream=stream0)
        buf1064 = buf1043; del buf1043  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf1064, buf1060, primals_43, add_35, buf1007, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_35
        del primals_43
        buf1080 = reinterpret_tensor(buf1060, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf1060  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf1067, buf1080, 12582912, grid=grid(12582912), stream=stream0)
        buf1081 = reinterpret_tensor(buf1008, (48, 2048, 128), (262144, 128, 1)); del buf1008  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1079, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf1080, (48, 2048, 128), (262144, 128, 1), 0), out=buf1081)
        buf1082 = buf1038; del buf1038  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf1081, buf1082, 12582912, grid=grid(12582912), stream=stream0)
        buf1083 = reinterpret_tensor(buf1081, (6144, 2048), (2048, 1)); del buf1081  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf1082, permute_68, out=buf1083)
        buf1084 = buf1007; del buf1007  # reuse
        buf1085 = reinterpret_tensor(buf1035, (3, 2048, 2048), (4194304, 2048, 1)); del buf1035  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_29, buf1083, primals_39, buf1084, buf1085, 6144, 2048, grid=grid(6144), stream=stream0)
        buf1086 = reinterpret_tensor(buf1036, (6144, 5632), (5632, 1)); del buf1036  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1085, (6144, 2048), (2048, 1), 0), permute_69, out=buf1086)
        buf1087 = reinterpret_tensor(buf1033, (6144, 5632), (5632, 1)); del buf1033  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1085, (6144, 2048), (2048, 1), 0), permute_70, out=buf1087)
        buf1090 = reinterpret_tensor(buf1030, (6144, 5632), (5632, 1)); del buf1030  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1064, (6144, 2048), (2048, 1), 0), permute_871, out=buf1090)
        del permute_871
        buf1088 = reinterpret_tensor(buf1032, (3, 2048, 5632), (11534336, 5632, 1)); del buf1032  # reuse
        buf1091 = reinterpret_tensor(buf1029, (3, 2048, 5632), (11534336, 5632, 1)); del buf1029  # reuse
        buf1094 = reinterpret_tensor(buf1028, (3, 2048, 5632), (11534336, 5632, 1)); del buf1028  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf1086, buf1087, buf1090, buf1088, buf1091, buf1094, 34603008, grid=grid(34603008), stream=stream0)
        buf1089 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1064, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf1088, (6144, 5632), (5632, 1), 0), out=buf1089)
        buf1092 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1091, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf1085, (6144, 2048), (2048, 1), 0), out=buf1092)
        buf1093 = buf1025; del buf1025  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1091, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_70, (5632, 2048), (2048, 1), 0), out=buf1093)
        del permute_70
        buf1095 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1094, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf1085, (6144, 2048), (2048, 1), 0), out=buf1095)
        buf1096 = reinterpret_tensor(buf1085, (6144, 2048), (2048, 1)); del buf1085  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1094, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_69, (5632, 2048), (2048, 1), 0), out=buf1096)
        del permute_69
        buf1101 = buf1064; del buf1064  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf1101, buf1093, buf1096, primals_39, add_29, buf1083, buf1084, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_39
        buf1103 = reinterpret_tensor(buf1046, (6144, 2048), (2048, 1)); del buf1046  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1101, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_68, (2048, 2048), (2048, 1), 0), out=buf1103)
        del permute_68
        buf1104 = reinterpret_tensor(buf1051, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf1051  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf1103, buf1104, 12582912, grid=grid(12582912), stream=stream0)
        buf1106 = reinterpret_tensor(buf1021, (48, 2048, 2048), (4194304, 2048, 1)); del buf1021  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1104, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf1080, (48, 128, 2048), (262144, 1, 128), 0), out=buf1106)
        buf1108 = buf1050; del buf1050  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf1106, buf1079, slice_3, buf1108, 98304, 2048, grid=grid(98304), stream=stream0)
        buf1109 = reinterpret_tensor(buf1080, (48, 128, 2048), (262144, 2048, 1)); del buf1080  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1074, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf1108, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf1109)
        buf1110 = reinterpret_tensor(buf1074, (48, 2048, 128), (262144, 128, 1)); del buf1074  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1108, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf1075, (48, 2048, 128), (262144, 1, 2048), 0), out=buf1110)
        buf1116 = reinterpret_tensor(buf1067, (3, 2048, 6144), (12582912, 6144, 1)); del buf1067  # reuse
        buf1114 = reinterpret_tensor(buf1116, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf1113 = reinterpret_tensor(buf1116, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf1109, select_3, select_1, buf1110, buf1114, buf1113, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf1105 = buf1110; del buf1110  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1079, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf1104, (48, 2048, 128), (262144, 128, 1), 0), out=buf1105)
        buf1115 = reinterpret_tensor(buf1116, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf1105, buf1115, 12582912, grid=grid(12582912), stream=stream0)
        del buf1113
        del buf1114
        del buf1115
        buf1118 = reinterpret_tensor(buf1105, (6144, 2048), (2048, 1)); del buf1105  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1116, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_60, (6144, 2048), (2048, 1), 0), out=buf1118)
        del permute_60
        buf1097 = buf1061; del buf1061  # reuse
        buf1119 = buf1039; del buf1039  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf1093, buf1096, add_29, buf1083, buf1084, buf1118, buf1065, buf1097, buf1119, 98304, 128, grid=grid(98304), stream=stream0)
        buf1098 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf1097, buf1098, 2048, 48, grid=grid(2048), stream=stream0)
        buf1102 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1101, (2048, 6144), (1, 2048), 0), buf1082, out=buf1102)
        buf1117 = buf1082; del buf1082  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1116, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf1066, (6144, 2048), (2048, 1), 0), out=buf1117)
        buf1120 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf1119, buf1120, 2048, 48, grid=grid(2048), stream=stream0)
        buf1122 = buf1101; del buf1101  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf1122, buf1118, primals_36, add_29, buf1065, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_29
        del primals_36
        buf1123 = buf1065; del buf1065  # reuse
        buf1124 = reinterpret_tensor(buf1118, (3, 2048, 2048), (4194304, 2048, 1)); del buf1118  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_23, primals_29, buf1123, buf1124, 6144, 2048, grid=grid(6144), stream=stream0)
        buf1125 = reinterpret_tensor(buf1116, (6144, 6144), (6144, 1)); del buf1116  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1124, (6144, 2048), (2048, 1), 0), permute_48, out=buf1125)
        buf1181 = buf1084; del buf1084  # reuse
        buf1182 = buf1066; del buf1066  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_17, primals_22, buf1181, buf1182, 6144, 2048, grid=grid(6144), stream=stream0)
        buf1183 = reinterpret_tensor(buf1058, (6144, 6144), (6144, 1)); del buf1058  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1182, (6144, 2048), (2048, 1), 0), permute_36, out=buf1183)
        buf1128 = buf1070; del buf1070  # reuse
        buf1126 = reinterpret_tensor(buf1128, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf1127 = reinterpret_tensor(buf1128, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf1131 = buf1073; del buf1073  # reuse
        buf1129 = reinterpret_tensor(buf1131, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf1130 = reinterpret_tensor(buf1131, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf1186 = buf1012; del buf1012  # reuse
        buf1184 = reinterpret_tensor(buf1186, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf1185 = reinterpret_tensor(buf1186, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf1189 = buf1015; del buf1015  # reuse
        buf1187 = reinterpret_tensor(buf1189, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf1188 = reinterpret_tensor(buf1189, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_27.run(buf1125, select_1, select_3, buf1183, buf1126, buf1127, buf1129, buf1130, buf1184, buf1185, buf1187, buf1188, 6291456, grid=grid(6291456), stream=stream0)
        buf1132 = reinterpret_tensor(buf1096, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf1096  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf1131, buf1132, 12582912, grid=grid(12582912), stream=stream0)
        del buf1126
        del buf1127
        del buf1129
        del buf1130
        buf1133 = reinterpret_tensor(buf1093, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf1093  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf1128, buf1133, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf1134 = reinterpret_tensor(buf1079, (48, 2048, 2048), (4194304, 2048, 1)); del buf1079  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1132, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf1133, (48, 128, 2048), (262144, 2048, 1), 0), out=buf1134)
        buf1190 = reinterpret_tensor(buf1083, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf1083  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf1189, buf1190, 12582912, grid=grid(12582912), stream=stream0)
        del buf1187
        del buf1188
        buf1191 = reinterpret_tensor(buf1104, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf1104  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf1186, buf1191, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        del buf1184
        del buf1185
        buf1192 = reinterpret_tensor(buf1108, (48, 2048, 2048), (4194304, 2048, 1)); del buf1108  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1190, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf1191, (48, 128, 2048), (262144, 2048, 1), 0), out=buf1192)
        buf1137 = reinterpret_tensor(buf1106, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf1106  # reuse
        buf1195 = reinterpret_tensor(buf1048, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf1048  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_28.run(slice_3, buf1134, scalar_tensor, buf1192, buf1137, buf1195, 98304, 2048, grid=grid(98304), stream=stream0)
        buf1138 = reinterpret_tensor(buf1109, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf1109  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf1125, buf1138, 12582912, grid=grid(12582912), stream=stream0)
        buf1139 = reinterpret_tensor(buf1075, (48, 2048, 128), (262144, 128, 1)); del buf1075  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1137, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf1138, (48, 2048, 128), (262144, 128, 1), 0), out=buf1139)
        buf1140 = buf1103; del buf1103  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf1139, buf1140, 12582912, grid=grid(12582912), stream=stream0)
        buf1141 = reinterpret_tensor(buf1139, (6144, 2048), (2048, 1)); del buf1139  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf1140, permute_56, out=buf1141)
        buf1142 = buf1026; del buf1026  # reuse
        buf1143 = reinterpret_tensor(buf1017, (3, 2048, 2048), (4194304, 2048, 1)); del buf1017  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_23, buf1141, primals_32, buf1142, buf1143, 6144, 2048, grid=grid(6144), stream=stream0)
        buf1144 = reinterpret_tensor(buf1094, (6144, 5632), (5632, 1)); del buf1094  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1143, (6144, 2048), (2048, 1), 0), permute_57, out=buf1144)
        buf1145 = reinterpret_tensor(buf1091, (6144, 5632), (5632, 1)); del buf1091  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1143, (6144, 2048), (2048, 1), 0), permute_58, out=buf1145)
        buf1148 = reinterpret_tensor(buf1088, (6144, 5632), (5632, 1)); del buf1088  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1122, (6144, 2048), (2048, 1), 0), permute_903, out=buf1148)
        del permute_903
        buf1146 = reinterpret_tensor(buf1090, (3, 2048, 5632), (11534336, 5632, 1)); del buf1090  # reuse
        buf1149 = reinterpret_tensor(buf1087, (3, 2048, 5632), (11534336, 5632, 1)); del buf1087  # reuse
        buf1152 = reinterpret_tensor(buf1086, (3, 2048, 5632), (11534336, 5632, 1)); del buf1086  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf1144, buf1145, buf1148, buf1146, buf1149, buf1152, 34603008, grid=grid(34603008), stream=stream0)
        buf1147 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1122, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf1146, (6144, 5632), (5632, 1), 0), out=buf1147)
        buf1150 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1149, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf1143, (6144, 2048), (2048, 1), 0), out=buf1150)
        buf1151 = buf1045; del buf1045  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1149, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_58, (5632, 2048), (2048, 1), 0), out=buf1151)
        del permute_58
        buf1153 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1152, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf1143, (6144, 2048), (2048, 1), 0), out=buf1153)
        buf1154 = reinterpret_tensor(buf1143, (6144, 2048), (2048, 1)); del buf1143  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1152, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_57, (5632, 2048), (2048, 1), 0), out=buf1154)
        del permute_57
        buf1159 = buf1122; del buf1122  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf1159, buf1151, buf1154, primals_32, add_23, buf1141, buf1142, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_32
        buf1161 = empty_strided((6144, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1159, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_56, (2048, 2048), (2048, 1), 0), out=buf1161)
        del permute_56
        buf1162 = empty_strided((3, 16, 2048, 128), (4194304, 262144, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf1161, buf1162, 12582912, grid=grid(12582912), stream=stream0)
        buf1164 = buf1192; del buf1192  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1162, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf1138, (48, 128, 2048), (262144, 1, 128), 0), out=buf1164)
        buf1166 = reinterpret_tensor(buf1134, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf1134  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf1164, buf1137, slice_3, buf1166, 98304, 2048, grid=grid(98304), stream=stream0)
        buf1167 = reinterpret_tensor(buf1138, (48, 128, 2048), (262144, 2048, 1)); del buf1138  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1132, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf1166, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf1167)
        buf1168 = reinterpret_tensor(buf1132, (48, 2048, 128), (262144, 128, 1)); del buf1132  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1166, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf1133, (48, 2048, 128), (262144, 1, 2048), 0), out=buf1168)
        buf1174 = reinterpret_tensor(buf1125, (3, 2048, 6144), (12582912, 6144, 1)); del buf1125  # reuse
        buf1172 = reinterpret_tensor(buf1174, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf1171 = reinterpret_tensor(buf1174, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf1167, select_3, select_1, buf1168, buf1172, buf1171, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf1163 = buf1168; del buf1168  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1137, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf1162, (48, 2048, 128), (262144, 128, 1), 0), out=buf1163)
        buf1173 = reinterpret_tensor(buf1174, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf1163, buf1173, 12582912, grid=grid(12582912), stream=stream0)
        del buf1171
        del buf1172
        del buf1173
        buf1176 = reinterpret_tensor(buf1163, (6144, 2048), (2048, 1)); del buf1163  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1174, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_48, (6144, 2048), (2048, 1), 0), out=buf1176)
        del permute_48
        buf1155 = buf1119; del buf1119  # reuse
        buf1177 = buf1097; del buf1097  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf1151, buf1154, add_23, buf1141, buf1142, buf1176, buf1123, buf1155, buf1177, 98304, 128, grid=grid(98304), stream=stream0)
        buf1156 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf1155, buf1156, 2048, 48, grid=grid(2048), stream=stream0)
        buf1160 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1159, (2048, 6144), (1, 2048), 0), buf1140, out=buf1160)
        buf1175 = buf1140; del buf1140  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1174, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf1124, (6144, 2048), (2048, 1), 0), out=buf1175)
        buf1178 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf1177, buf1178, 2048, 48, grid=grid(2048), stream=stream0)
        buf1180 = buf1159; del buf1159  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf1180, buf1176, primals_29, add_23, buf1123, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_23
        del primals_29
        buf1196 = reinterpret_tensor(buf1176, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf1176  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf1183, buf1196, 12582912, grid=grid(12582912), stream=stream0)
        buf1197 = reinterpret_tensor(buf1124, (48, 2048, 128), (262144, 128, 1)); del buf1124  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1195, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf1196, (48, 2048, 128), (262144, 128, 1), 0), out=buf1197)
        buf1198 = buf1154; del buf1154  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf1197, buf1198, 12582912, grid=grid(12582912), stream=stream0)
        buf1199 = reinterpret_tensor(buf1197, (6144, 2048), (2048, 1)); del buf1197  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf1198, permute_44, out=buf1199)
        buf1200 = buf1123; del buf1123  # reuse
        buf1201 = reinterpret_tensor(buf1151, (3, 2048, 2048), (4194304, 2048, 1)); del buf1151  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_17, buf1199, primals_25, buf1200, buf1201, 6144, 2048, grid=grid(6144), stream=stream0)
        buf1202 = reinterpret_tensor(buf1152, (6144, 5632), (5632, 1)); del buf1152  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1201, (6144, 2048), (2048, 1), 0), permute_45, out=buf1202)
        buf1203 = reinterpret_tensor(buf1149, (6144, 5632), (5632, 1)); del buf1149  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1201, (6144, 2048), (2048, 1), 0), permute_46, out=buf1203)
        buf1206 = reinterpret_tensor(buf1146, (6144, 5632), (5632, 1)); del buf1146  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1180, (6144, 2048), (2048, 1), 0), permute_935, out=buf1206)
        del permute_935
        buf1204 = reinterpret_tensor(buf1148, (3, 2048, 5632), (11534336, 5632, 1)); del buf1148  # reuse
        buf1207 = reinterpret_tensor(buf1145, (3, 2048, 5632), (11534336, 5632, 1)); del buf1145  # reuse
        buf1210 = reinterpret_tensor(buf1144, (3, 2048, 5632), (11534336, 5632, 1)); del buf1144  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf1202, buf1203, buf1206, buf1204, buf1207, buf1210, 34603008, grid=grid(34603008), stream=stream0)
        buf1205 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1180, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf1204, (6144, 5632), (5632, 1), 0), out=buf1205)
        buf1208 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1207, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf1201, (6144, 2048), (2048, 1), 0), out=buf1208)
        buf1209 = buf1141; del buf1141  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1207, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_46, (5632, 2048), (2048, 1), 0), out=buf1209)
        del permute_46
        buf1211 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1210, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf1201, (6144, 2048), (2048, 1), 0), out=buf1211)
        buf1212 = reinterpret_tensor(buf1201, (6144, 2048), (2048, 1)); del buf1201  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1210, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_45, (5632, 2048), (2048, 1), 0), out=buf1212)
        del permute_45
        buf1217 = buf1180; del buf1180  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf1217, buf1209, buf1212, primals_25, add_17, buf1199, buf1200, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_25
        buf1219 = reinterpret_tensor(buf1162, (6144, 2048), (2048, 1)); del buf1162  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1217, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_44, (2048, 2048), (2048, 1), 0), out=buf1219)
        del permute_44
        buf1220 = reinterpret_tensor(buf1167, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf1167  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf1219, buf1220, 12582912, grid=grid(12582912), stream=stream0)
        buf1222 = reinterpret_tensor(buf1137, (48, 2048, 2048), (4194304, 2048, 1)); del buf1137  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1220, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf1196, (48, 128, 2048), (262144, 1, 128), 0), out=buf1222)
        buf1224 = buf1166; del buf1166  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf1222, buf1195, slice_3, buf1224, 98304, 2048, grid=grid(98304), stream=stream0)
        buf1225 = reinterpret_tensor(buf1196, (48, 128, 2048), (262144, 2048, 1)); del buf1196  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1190, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf1224, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf1225)
        buf1226 = reinterpret_tensor(buf1190, (48, 2048, 128), (262144, 128, 1)); del buf1190  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1224, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf1191, (48, 2048, 128), (262144, 1, 2048), 0), out=buf1226)
        buf1232 = reinterpret_tensor(buf1183, (3, 2048, 6144), (12582912, 6144, 1)); del buf1183  # reuse
        buf1230 = reinterpret_tensor(buf1232, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf1229 = reinterpret_tensor(buf1232, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf1225, select_3, select_1, buf1226, buf1230, buf1229, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf1221 = buf1226; del buf1226  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1195, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf1220, (48, 2048, 128), (262144, 128, 1), 0), out=buf1221)
        buf1231 = reinterpret_tensor(buf1232, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf1221, buf1231, 12582912, grid=grid(12582912), stream=stream0)
        del buf1229
        del buf1230
        del buf1231
        buf1234 = reinterpret_tensor(buf1221, (6144, 2048), (2048, 1)); del buf1221  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1232, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_36, (6144, 2048), (2048, 1), 0), out=buf1234)
        del permute_36
        buf1213 = buf1177; del buf1177  # reuse
        buf1235 = buf1155; del buf1155  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf1209, buf1212, add_17, buf1199, buf1200, buf1234, buf1181, buf1213, buf1235, 98304, 128, grid=grid(98304), stream=stream0)
        buf1214 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf1213, buf1214, 2048, 48, grid=grid(2048), stream=stream0)
        buf1218 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1217, (2048, 6144), (1, 2048), 0), buf1198, out=buf1218)
        buf1233 = buf1198; del buf1198  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1232, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf1182, (6144, 2048), (2048, 1), 0), out=buf1233)
        buf1236 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf1235, buf1236, 2048, 48, grid=grid(2048), stream=stream0)
        buf1238 = buf1217; del buf1217  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf1238, buf1234, primals_22, add_17, buf1181, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_17
        del primals_22
        buf1239 = buf1181; del buf1181  # reuse
        buf1240 = reinterpret_tensor(buf1234, (3, 2048, 2048), (4194304, 2048, 1)); del buf1234  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_11, primals_15, buf1239, buf1240, 6144, 2048, grid=grid(6144), stream=stream0)
        buf1241 = reinterpret_tensor(buf1232, (6144, 6144), (6144, 1)); del buf1232  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1240, (6144, 2048), (2048, 1), 0), permute_24, out=buf1241)
        buf1297 = buf1200; del buf1200  # reuse
        buf1298 = buf1182; del buf1182  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(add_5, primals_8, buf1297, buf1298, 6144, 2048, grid=grid(6144), stream=stream0)
        buf1299 = reinterpret_tensor(buf1174, (6144, 6144), (6144, 1)); del buf1174  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1298, (6144, 2048), (2048, 1), 0), permute_12, out=buf1299)
        buf1244 = buf1186; del buf1186  # reuse
        buf1242 = reinterpret_tensor(buf1244, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf1243 = reinterpret_tensor(buf1244, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf1247 = buf1189; del buf1189  # reuse
        buf1245 = reinterpret_tensor(buf1247, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf1246 = reinterpret_tensor(buf1247, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf1302 = buf1128; del buf1128  # reuse
        buf1300 = reinterpret_tensor(buf1302, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf1301 = reinterpret_tensor(buf1302, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf1305 = buf1131; del buf1131  # reuse
        buf1303 = reinterpret_tensor(buf1305, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf1304 = reinterpret_tensor(buf1305, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_27.run(buf1241, select_1, select_3, buf1299, buf1242, buf1243, buf1245, buf1246, buf1300, buf1301, buf1303, buf1304, 6291456, grid=grid(6291456), stream=stream0)
        buf1248 = reinterpret_tensor(buf1212, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf1212  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf1247, buf1248, 12582912, grid=grid(12582912), stream=stream0)
        del buf1242
        del buf1243
        del buf1245
        del buf1246
        del buf1247
        buf1249 = reinterpret_tensor(buf1209, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf1209  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf1244, buf1249, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        del buf1244
        buf1250 = reinterpret_tensor(buf1195, (48, 2048, 2048), (4194304, 2048, 1)); del buf1195  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1248, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf1249, (48, 128, 2048), (262144, 2048, 1), 0), out=buf1250)
        buf1306 = reinterpret_tensor(buf1199, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf1199  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf1305, buf1306, 12582912, grid=grid(12582912), stream=stream0)
        del buf1303
        del buf1304
        buf1307 = reinterpret_tensor(buf1220, (3, 16, 128, 2048), (4194304, 262144, 2048, 1)); del buf1220  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf1302, buf1307, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        del buf1300
        del buf1301
        buf1308 = reinterpret_tensor(buf1224, (48, 2048, 2048), (4194304, 2048, 1)); del buf1224  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1306, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf1307, (48, 128, 2048), (262144, 2048, 1), 0), out=buf1308)
        buf1253 = reinterpret_tensor(buf1222, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf1222  # reuse
        buf1311 = reinterpret_tensor(buf1164, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf1164  # reuse
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_28.run(slice_3, buf1250, scalar_tensor, buf1308, buf1253, buf1311, 98304, 2048, grid=grid(98304), stream=stream0)
        buf1254 = reinterpret_tensor(buf1225, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf1225  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf1241, buf1254, 12582912, grid=grid(12582912), stream=stream0)
        buf1255 = reinterpret_tensor(buf1191, (48, 2048, 128), (262144, 128, 1)); del buf1191  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1253, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf1254, (48, 2048, 128), (262144, 128, 1), 0), out=buf1255)
        buf1256 = buf1219; del buf1219  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf1255, buf1256, 12582912, grid=grid(12582912), stream=stream0)
        buf1257 = reinterpret_tensor(buf1255, (6144, 2048), (2048, 1)); del buf1255  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf1256, permute_32, out=buf1257)
        buf1258 = buf1142; del buf1142  # reuse
        buf1259 = reinterpret_tensor(buf1133, (3, 2048, 2048), (4194304, 2048, 1)); del buf1133  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_11, buf1257, primals_18, buf1258, buf1259, 6144, 2048, grid=grid(6144), stream=stream0)
        buf1260 = reinterpret_tensor(buf1210, (6144, 5632), (5632, 1)); del buf1210  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1259, (6144, 2048), (2048, 1), 0), permute_33, out=buf1260)
        buf1261 = reinterpret_tensor(buf1207, (6144, 5632), (5632, 1)); del buf1207  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1259, (6144, 2048), (2048, 1), 0), permute_34, out=buf1261)
        buf1264 = reinterpret_tensor(buf1204, (6144, 5632), (5632, 1)); del buf1204  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1238, (6144, 2048), (2048, 1), 0), permute_967, out=buf1264)
        del permute_967
        buf1262 = reinterpret_tensor(buf1206, (3, 2048, 5632), (11534336, 5632, 1)); del buf1206  # reuse
        buf1265 = reinterpret_tensor(buf1203, (3, 2048, 5632), (11534336, 5632, 1)); del buf1203  # reuse
        buf1268 = reinterpret_tensor(buf1202, (3, 2048, 5632), (11534336, 5632, 1)); del buf1202  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf1260, buf1261, buf1264, buf1262, buf1265, buf1268, 34603008, grid=grid(34603008), stream=stream0)
        buf1263 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1238, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf1262, (6144, 5632), (5632, 1), 0), out=buf1263)
        buf1266 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1265, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf1259, (6144, 2048), (2048, 1), 0), out=buf1266)
        buf1267 = buf1161; del buf1161  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1265, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_34, (5632, 2048), (2048, 1), 0), out=buf1267)
        del permute_34
        buf1269 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1268, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf1259, (6144, 2048), (2048, 1), 0), out=buf1269)
        buf1270 = reinterpret_tensor(buf1259, (6144, 2048), (2048, 1)); del buf1259  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1268, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_33, (5632, 2048), (2048, 1), 0), out=buf1270)
        del permute_33
        buf1275 = buf1238; del buf1238  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf1275, buf1267, buf1270, primals_18, add_11, buf1257, buf1258, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_18
        buf1277 = empty_strided((6144, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1275, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_32, (2048, 2048), (2048, 1), 0), out=buf1277)
        del permute_32
        buf1278 = empty_strided((3, 16, 2048, 128), (4194304, 262144, 128, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf1277, buf1278, 12582912, grid=grid(12582912), stream=stream0)
        buf1280 = buf1308; del buf1308  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1278, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf1254, (48, 128, 2048), (262144, 1, 128), 0), out=buf1280)
        buf1282 = reinterpret_tensor(buf1250, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf1250  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf1280, buf1253, slice_3, buf1282, 98304, 2048, grid=grid(98304), stream=stream0)
        buf1283 = reinterpret_tensor(buf1254, (48, 128, 2048), (262144, 2048, 1)); del buf1254  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1248, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf1282, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf1283)
        buf1284 = reinterpret_tensor(buf1248, (48, 2048, 128), (262144, 128, 1)); del buf1248  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1282, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf1249, (48, 2048, 128), (262144, 1, 2048), 0), out=buf1284)
        buf1290 = reinterpret_tensor(buf1241, (3, 2048, 6144), (12582912, 6144, 1)); del buf1241  # reuse
        buf1288 = reinterpret_tensor(buf1290, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf1287 = reinterpret_tensor(buf1290, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf1283, select_3, select_1, buf1284, buf1288, buf1287, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf1279 = buf1284; del buf1284  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1253, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf1278, (48, 2048, 128), (262144, 128, 1), 0), out=buf1279)
        buf1289 = reinterpret_tensor(buf1290, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf1279, buf1289, 12582912, grid=grid(12582912), stream=stream0)
        del buf1287
        del buf1288
        del buf1289
        buf1292 = reinterpret_tensor(buf1279, (6144, 2048), (2048, 1)); del buf1279  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1290, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_24, (6144, 2048), (2048, 1), 0), out=buf1292)
        del permute_24
        buf1271 = buf1235; del buf1235  # reuse
        buf1293 = buf1213; del buf1213  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf1267, buf1270, add_11, buf1257, buf1258, buf1292, buf1239, buf1271, buf1293, 98304, 128, grid=grid(98304), stream=stream0)
        buf1272 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf1271, buf1272, 2048, 48, grid=grid(2048), stream=stream0)
        buf1276 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1275, (2048, 6144), (1, 2048), 0), buf1256, out=buf1276)
        buf1291 = buf1256; del buf1256  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1290, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf1240, (6144, 2048), (2048, 1), 0), out=buf1291)
        buf1294 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf1293, buf1294, 2048, 48, grid=grid(2048), stream=stream0)
        buf1296 = buf1275; del buf1275  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf1296, buf1292, primals_15, add_11, buf1239, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_11
        del primals_15
        buf1312 = reinterpret_tensor(buf1292, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf1292  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf1299, buf1312, 12582912, grid=grid(12582912), stream=stream0)
        buf1313 = reinterpret_tensor(buf1240, (48, 2048, 128), (262144, 128, 1)); del buf1240  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1311, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf1312, (48, 2048, 128), (262144, 128, 1), 0), out=buf1313)
        buf1314 = buf1270; del buf1270  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf1313, buf1314, 12582912, grid=grid(12582912), stream=stream0)
        buf1315 = reinterpret_tensor(buf1313, (6144, 2048), (2048, 1)); del buf1313  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf1314, permute_20, out=buf1315)
        buf1316 = buf1239; del buf1239  # reuse
        buf1317 = reinterpret_tensor(buf1267, (3, 2048, 2048), (4194304, 2048, 1)); del buf1267  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(add_5, buf1315, primals_11, buf1316, buf1317, 6144, 2048, grid=grid(6144), stream=stream0)
        buf1318 = reinterpret_tensor(buf1268, (6144, 5632), (5632, 1)); del buf1268  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1317, (6144, 2048), (2048, 1), 0), permute_21, out=buf1318)
        buf1319 = reinterpret_tensor(buf1265, (6144, 5632), (5632, 1)); del buf1265  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1317, (6144, 2048), (2048, 1), 0), permute_22, out=buf1319)
        buf1322 = reinterpret_tensor(buf1262, (6144, 5632), (5632, 1)); del buf1262  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1296, (6144, 2048), (2048, 1), 0), permute_999, out=buf1322)
        del permute_999
        buf1320 = reinterpret_tensor(buf1264, (3, 2048, 5632), (11534336, 5632, 1)); del buf1264  # reuse
        buf1323 = reinterpret_tensor(buf1261, (3, 2048, 5632), (11534336, 5632, 1)); del buf1261  # reuse
        buf1326 = reinterpret_tensor(buf1260, (3, 2048, 5632), (11534336, 5632, 1)); del buf1260  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf1318, buf1319, buf1322, buf1320, buf1323, buf1326, 34603008, grid=grid(34603008), stream=stream0)
        buf1321 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1296, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf1320, (6144, 5632), (5632, 1), 0), out=buf1321)
        buf1324 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1323, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf1317, (6144, 2048), (2048, 1), 0), out=buf1324)
        buf1325 = buf1257; del buf1257  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1323, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_22, (5632, 2048), (2048, 1), 0), out=buf1325)
        del permute_22
        buf1327 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1326, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf1317, (6144, 2048), (2048, 1), 0), out=buf1327)
        buf1328 = reinterpret_tensor(buf1317, (6144, 2048), (2048, 1)); del buf1317  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1326, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_21, (5632, 2048), (2048, 1), 0), out=buf1328)
        del permute_21
        buf1333 = buf1296; del buf1296  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf1333, buf1325, buf1328, primals_11, add_5, buf1315, buf1316, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_11
        buf1335 = reinterpret_tensor(buf1278, (6144, 2048), (2048, 1)); del buf1278  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1333, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_20, (2048, 2048), (2048, 1), 0), out=buf1335)
        del permute_20
        buf1336 = reinterpret_tensor(buf1283, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf1283  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf1335, buf1336, 12582912, grid=grid(12582912), stream=stream0)
        buf1338 = reinterpret_tensor(buf1253, (48, 2048, 2048), (4194304, 2048, 1)); del buf1253  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1336, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf1312, (48, 128, 2048), (262144, 1, 128), 0), out=buf1338)
        buf1355 = buf1258; del buf1258  # reuse
        buf1356 = reinterpret_tensor(buf1312, (3, 2048, 2048), (4194304, 2048, 1)); del buf1312  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, mul_2, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_8.run(embedding, primals_1, buf1355, buf1356, 6144, 2048, grid=grid(6144), stream=stream0)
        buf1357 = buf1299; del buf1299  # reuse
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1356, (6144, 2048), (2048, 1), 0), permute, out=buf1357)
        buf1360 = buf1302; del buf1302  # reuse
        buf1358 = reinterpret_tensor(buf1360, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf1359 = reinterpret_tensor(buf1360, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        buf1363 = buf1305; del buf1305  # reuse
        buf1361 = reinterpret_tensor(buf1363, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 0)  # alias
        buf1362 = reinterpret_tensor(buf1363, (3, 2048, 16, 64, 1), (4194304, 2048, 128, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_9.run(buf1357, select_1, select_3, buf1358, buf1359, buf1361, buf1362, 6291456, grid=grid(6291456), stream=stream0)
        buf1364 = reinterpret_tensor(buf1335, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf1335  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf1363, buf1364, 12582912, grid=grid(12582912), stream=stream0)
        del buf1361
        del buf1362
        del buf1363
        buf1365 = buf1249; del buf1249  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_11.run(buf1360, buf1365, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        del buf1358
        del buf1359
        del buf1360
        buf1366 = reinterpret_tensor(buf1282, (48, 2048, 2048), (4194304, 2048, 1)); del buf1282  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1364, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf1365, (48, 128, 2048), (262144, 2048, 1), 0), out=buf1366)
        buf1340 = reinterpret_tensor(buf1280, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf1280  # reuse
        buf1369 = empty_strided((3, 16, 2048, 2048), (67108864, 4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [cross_entropy, mul_11, softmax, where], Original ATen: [aten._softmax, aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax__softmax_backward_data_mul_nll_loss_forward_where_32.run(slice_3, buf1366, scalar_tensor, buf1338, buf1311, buf1340, buf1369, 98304, 2048, grid=grid(98304), stream=stream0)
        del buf1338
        del buf1366
        del scalar_tensor
        buf1341 = reinterpret_tensor(buf1277, (48, 128, 2048), (262144, 2048, 1)); del buf1277  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1306, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf1340, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf1341)
        buf1342 = reinterpret_tensor(buf1306, (48, 2048, 128), (262144, 128, 1)); del buf1306  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1340, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf1307, (48, 2048, 128), (262144, 1, 2048), 0), out=buf1342)
        del buf1307
        buf1348 = buf1290; del buf1290  # reuse
        buf1346 = reinterpret_tensor(buf1348, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf1345 = reinterpret_tensor(buf1348, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf1341, select_3, select_1, buf1342, buf1346, buf1345, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        buf1337 = buf1342; del buf1342  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1311, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf1336, (48, 2048, 128), (262144, 128, 1), 0), out=buf1337)
        buf1347 = reinterpret_tensor(buf1348, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf1337, buf1347, 12582912, grid=grid(12582912), stream=stream0)
        del buf1345
        del buf1346
        del buf1347
        buf1350 = reinterpret_tensor(buf1337, (6144, 2048), (2048, 1)); del buf1337  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1348, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute_12, (6144, 2048), (2048, 1), 0), out=buf1350)
        del permute_12
        buf1329 = buf1293; del buf1293  # reuse
        buf1351 = buf1271; del buf1271  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf1325, buf1328, add_5, buf1315, buf1316, buf1350, buf1297, buf1329, buf1351, 98304, 128, grid=grid(98304), stream=stream0)
        del buf1316
        buf1330 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf1329, buf1330, 2048, 48, grid=grid(2048), stream=stream0)
        buf1334 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1333, (2048, 6144), (1, 2048), 0), buf1314, out=buf1334)
        buf1349 = buf1314; del buf1314  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1348, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf1298, (6144, 2048), (2048, 1), 0), out=buf1349)
        del buf1348
        buf1352 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf1351, buf1352, 2048, 48, grid=grid(2048), stream=stream0)
        buf1354 = buf1333; del buf1333  # reuse
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_26.run(buf1354, buf1350, primals_8, add_5, buf1297, 6144, 2048, grid=grid(6144), stream=stream0)
        del add_5
        del primals_8
        buf1370 = reinterpret_tensor(buf1350, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf1350  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_13.run(buf1357, buf1370, 12582912, grid=grid(12582912), stream=stream0)
        buf1371 = reinterpret_tensor(buf1298, (48, 2048, 128), (262144, 128, 1)); del buf1298  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1369, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf1370, (48, 2048, 128), (262144, 128, 1), 0), out=buf1371)
        buf1372 = buf1328; del buf1328  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.view]
        triton_poi_fused_view_14.run(buf1371, buf1372, 12582912, grid=grid(12582912), stream=stream0)
        buf1373 = reinterpret_tensor(buf1371, (6144, 2048), (2048, 1)); del buf1371  # reuse
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(buf1372, permute_8, out=buf1373)
        buf1374 = buf1297; del buf1297  # reuse
        buf1375 = reinterpret_tensor(buf1325, (3, 2048, 2048), (4194304, 2048, 1)); del buf1325  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, mul_14, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_15.run(embedding, buf1373, primals_4, buf1374, buf1375, 6144, 2048, grid=grid(6144), stream=stream0)
        buf1376 = reinterpret_tensor(buf1326, (6144, 5632), (5632, 1)); del buf1326  # reuse
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1375, (6144, 2048), (2048, 1), 0), permute_9, out=buf1376)
        buf1377 = reinterpret_tensor(buf1323, (6144, 5632), (5632, 1)); del buf1323  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1375, (6144, 2048), (2048, 1), 0), permute_10, out=buf1377)
        buf1380 = reinterpret_tensor(buf1320, (6144, 5632), (5632, 1)); del buf1320  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1354, (6144, 2048), (2048, 1), 0), permute_1031, out=buf1380)
        del permute_1031
        buf1378 = reinterpret_tensor(buf1322, (3, 2048, 5632), (11534336, 5632, 1)); del buf1322  # reuse
        buf1381 = reinterpret_tensor(buf1319, (3, 2048, 5632), (11534336, 5632, 1)); del buf1319  # reuse
        buf1384 = reinterpret_tensor(buf1318, (3, 2048, 5632), (11534336, 5632, 1)); del buf1318  # reuse
        # Source Nodes: [mul_15, silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_29.run(buf1376, buf1377, buf1380, buf1378, buf1381, buf1384, 34603008, grid=grid(34603008), stream=stream0)
        del buf1376
        del buf1377
        del buf1380
        buf1379 = empty_strided((2048, 5632), (5632, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1354, (2048, 6144), (1, 2048), 0), reinterpret_tensor(buf1378, (6144, 5632), (5632, 1), 0), out=buf1379)
        del buf1378
        buf1382 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1381, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf1375, (6144, 2048), (2048, 1), 0), out=buf1382)
        buf1383 = buf1315; del buf1315  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1381, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_10, (5632, 2048), (2048, 1), 0), out=buf1383)
        del buf1381
        del permute_10
        buf1385 = empty_strided((5632, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1384, (5632, 6144), (1, 5632), 0), reinterpret_tensor(buf1375, (6144, 2048), (2048, 1), 0), out=buf1385)
        buf1386 = reinterpret_tensor(buf1375, (6144, 2048), (2048, 1)); del buf1375  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1384, (6144, 5632), (5632, 1), 0), reinterpret_tensor(permute_9, (5632, 2048), (2048, 1), 0), out=buf1386)
        del buf1384
        del permute_9
        buf1391 = buf1354; del buf1354  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_mean_mul_rsqrt_sum_30.run(buf1391, buf1383, buf1386, primals_4, embedding, buf1373, buf1374, 6144, 2048, grid=grid(6144), stream=stream0)
        del primals_4
        buf1393 = reinterpret_tensor(buf1336, (6144, 2048), (2048, 1)); del buf1336  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1391, (6144, 2048), (2048, 1), 0), reinterpret_tensor(permute_8, (2048, 2048), (2048, 1), 0), out=buf1393)
        del permute_8
        buf1394 = reinterpret_tensor(buf1341, (3, 16, 2048, 128), (4194304, 262144, 128, 1)); del buf1341  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_20.run(buf1393, buf1394, 12582912, grid=grid(12582912), stream=stream0)
        del buf1393
        buf1396 = reinterpret_tensor(buf1311, (48, 2048, 2048), (4194304, 2048, 1)); del buf1311  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1394, (48, 2048, 128), (262144, 128, 1), 0), reinterpret_tensor(buf1370, (48, 128, 2048), (262144, 1, 128), 0), out=buf1396)
        buf1398 = buf1340; del buf1340  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_21.run(buf1396, buf1369, slice_3, buf1398, 98304, 2048, grid=grid(98304), stream=stream0)
        del buf1396
        del slice_3
        buf1399 = reinterpret_tensor(buf1370, (48, 128, 2048), (262144, 2048, 1)); del buf1370  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1364, (48, 128, 2048), (262144, 1, 128), 0), reinterpret_tensor(buf1398, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf1399)
        buf1400 = reinterpret_tensor(buf1364, (48, 2048, 128), (262144, 128, 1)); del buf1364  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1398, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf1365, (48, 2048, 128), (262144, 1, 2048), 0), out=buf1400)
        del buf1365
        del buf1398
        buf1406 = reinterpret_tensor(buf1357, (3, 2048, 6144), (12582912, 6144, 1)); del buf1357  # reuse
        buf1404 = reinterpret_tensor(buf1406, (3, 2048, 2048), (12582912, 6144, 1), 2048)  # alias
        buf1403 = reinterpret_tensor(buf1406, (3, 2048, 2048), (12582912, 6144, 1), 0)  # alias
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.cat]
        triton_poi_fused__to_copy_cat_22.run(buf1399, select_3, select_1, buf1400, buf1404, buf1403, 6144, 2048, grid=grid(6144, 2048), stream=stream0)
        del buf1399
        del select_1
        del select_3
        buf1395 = buf1400; del buf1400  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf1369, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf1394, (48, 2048, 128), (262144, 128, 1), 0), out=buf1395)
        del buf1369
        del buf1394
        buf1405 = reinterpret_tensor(buf1406, (3, 2048, 2048), (12582912, 6144, 1), 4096)  # alias
        # Source Nodes: [], Original ATen: [aten.cat]
        triton_poi_fused_cat_23.run(buf1395, buf1405, 12582912, grid=grid(12582912), stream=stream0)
        del buf1403
        del buf1404
        del buf1405
        buf1408 = reinterpret_tensor(buf1395, (6144, 2048), (2048, 1)); del buf1395  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1406, (6144, 6144), (6144, 1), 0), reinterpret_tensor(permute, (6144, 2048), (2048, 1), 0), out=buf1408)
        del permute
        buf1387 = buf1351; del buf1351  # reuse
        buf1409 = buf1329; del buf1329  # reuse
        # Source Nodes: [add, add_3, add_4, float_1, float_4, mean, mean_1, mul, mul_1, mul_12, mul_13, rsqrt, rsqrt_1, type_as, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_31.run(buf1383, buf1386, embedding, buf1373, buf1374, buf1408, buf1355, buf1387, buf1409, 98304, 128, grid=grid(98304), stream=stream0)
        del buf1373
        del buf1374
        del buf1383
        del buf1386
        buf1388 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf1387, buf1388, 2048, 48, grid=grid(2048), stream=stream0)
        del buf1387
        buf1392 = empty_strided((2048, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1391, (2048, 6144), (1, 2048), 0), buf1372, out=buf1392)
        buf1407 = buf1372; del buf1372  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf1406, (6144, 6144), (1, 6144), 0), reinterpret_tensor(buf1356, (6144, 2048), (2048, 1), 0), out=buf1407)
        del buf1356
        del buf1406
        buf1410 = empty_strided((1, 1, 2048), (2048, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_25.run(buf1409, buf1410, 2048, 48, grid=grid(2048), stream=stream0)
        del buf1409
        buf1413 = empty_strided((50257, 2048), (2048, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_33.run(buf1413, 102926336, grid=grid(102926336), stream=stream0)
        # Source Nodes: [add, float_1, mean, mul, rsqrt], Original ATen: [aten._to_copy, aten.add, aten.div, aten.embedding_dense_backward, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_div_embedding_dense_backward_mean_mul_rsqrt_sum_34.run(buf1408, primals_1, embedding, primals_174, buf1391, buf1355, buf1413, 6144, 2048, grid=grid(6144), stream=stream0)
        del buf1355
        del buf1391
        del buf1408
        del embedding
        del primals_1
        del primals_174
        buf1415 = empty_strided((50257, 2048), (2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_35.run(buf1413, buf1415, 102926336, grid=grid(102926336), stream=stream0)
        return (reinterpret_tensor(buf1410, (2048, ), (1, ), 0), reinterpret_tensor(buf1407, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf1392, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf1388, (2048, ), (1, ), 0), reinterpret_tensor(buf1385, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf1382, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf1379, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf1352, (2048, ), (1, ), 0), reinterpret_tensor(buf1349, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf1334, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf1330, (2048, ), (1, ), 0), reinterpret_tensor(buf1327, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf1324, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf1321, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf1294, (2048, ), (1, ), 0), reinterpret_tensor(buf1291, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf1276, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf1272, (2048, ), (1, ), 0), reinterpret_tensor(buf1269, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf1266, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf1263, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf1236, (2048, ), (1, ), 0), reinterpret_tensor(buf1233, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf1218, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf1214, (2048, ), (1, ), 0), reinterpret_tensor(buf1211, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf1208, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf1205, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf1178, (2048, ), (1, ), 0), reinterpret_tensor(buf1175, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf1160, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf1156, (2048, ), (1, ), 0), reinterpret_tensor(buf1153, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf1150, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf1147, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf1120, (2048, ), (1, ), 0), reinterpret_tensor(buf1117, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf1102, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf1098, (2048, ), (1, ), 0), reinterpret_tensor(buf1095, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf1092, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf1089, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf1062, (2048, ), (1, ), 0), reinterpret_tensor(buf1059, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf1044, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf1040, (2048, ), (1, ), 0), reinterpret_tensor(buf1037, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf1034, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf1031, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf1004, (2048, ), (1, ), 0), reinterpret_tensor(buf1001, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf986, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf982, (2048, ), (1, ), 0), reinterpret_tensor(buf979, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf976, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf973, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf946, (2048, ), (1, ), 0), reinterpret_tensor(buf943, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf928, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf924, (2048, ), (1, ), 0), reinterpret_tensor(buf921, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf918, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf915, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf888, (2048, ), (1, ), 0), reinterpret_tensor(buf885, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf870, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf866, (2048, ), (1, ), 0), reinterpret_tensor(buf863, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf860, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf857, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf830, (2048, ), (1, ), 0), reinterpret_tensor(buf827, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf812, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf808, (2048, ), (1, ), 0), reinterpret_tensor(buf805, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf802, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf799, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf772, (2048, ), (1, ), 0), reinterpret_tensor(buf769, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf754, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf750, (2048, ), (1, ), 0), reinterpret_tensor(buf747, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf744, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf741, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf714, (2048, ), (1, ), 0), reinterpret_tensor(buf711, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf696, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf692, (2048, ), (1, ), 0), reinterpret_tensor(buf689, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf686, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf683, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf656, (2048, ), (1, ), 0), reinterpret_tensor(buf653, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf638, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf634, (2048, ), (1, ), 0), reinterpret_tensor(buf631, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf628, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf625, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf598, (2048, ), (1, ), 0), reinterpret_tensor(buf595, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf580, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf576, (2048, ), (1, ), 0), reinterpret_tensor(buf573, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf570, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf567, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf540, (2048, ), (1, ), 0), reinterpret_tensor(buf537, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf522, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf518, (2048, ), (1, ), 0), reinterpret_tensor(buf515, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf512, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf509, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf482, (2048, ), (1, ), 0), reinterpret_tensor(buf479, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf464, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf460, (2048, ), (1, ), 0), reinterpret_tensor(buf457, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf454, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf451, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf424, (2048, ), (1, ), 0), reinterpret_tensor(buf421, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf406, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf402, (2048, ), (1, ), 0), reinterpret_tensor(buf399, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf396, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf393, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf366, (2048, ), (1, ), 0), reinterpret_tensor(buf363, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf348, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf344, (2048, ), (1, ), 0), reinterpret_tensor(buf341, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf338, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf335, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf308, (2048, ), (1, ), 0), reinterpret_tensor(buf305, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf290, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf286, (2048, ), (1, ), 0), reinterpret_tensor(buf283, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf280, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf277, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf250, (2048, ), (1, ), 0), reinterpret_tensor(buf247, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf232, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf228, (2048, ), (1, ), 0), reinterpret_tensor(buf225, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf222, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf219, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf192, (2048, ), (1, ), 0), reinterpret_tensor(buf189, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf174, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf170, (2048, ), (1, ), 0), reinterpret_tensor(buf167, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf164, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf161, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf134, (2048, ), (1, ), 0), reinterpret_tensor(buf131, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf116, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf112, (2048, ), (1, ), 0), reinterpret_tensor(buf109, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf106, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf103, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf76, (2048, ), (1, ), 0), reinterpret_tensor(buf73, (6144, 2048), (2048, 1), 0), reinterpret_tensor(buf58, (2048, 2048), (2048, 1), 0), reinterpret_tensor(buf54, (2048, ), (1, ), 0), reinterpret_tensor(buf51, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf48, (5632, 2048), (2048, 1), 0), reinterpret_tensor(buf45, (2048, 5632), (5632, 1), 0), reinterpret_tensor(buf41, (2048, ), (1, ), 0), buf1415, reinterpret_tensor(buf7, (50257, 2048), (2048, 1), 0), None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_1 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_4 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_8 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_11 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_15 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_18 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_22 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_25 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_29 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_32 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_36 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_39 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_43 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_46 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_50 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_53 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_57 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_60 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_64 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_67 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_71 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_74 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_78 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_81 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_85 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_88 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_92 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_95 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_99 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_102 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_106 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_109 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_113 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_116 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_120 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_123 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_127 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_130 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_134 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_137 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_141 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_144 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_148 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_151 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_155 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_158 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_162 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_165 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_169 = rand_strided((2048, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_174 = rand_strided((3, 2048), (2049, 1), device='cuda:0', dtype=torch.int64)
    primals_175 = rand_strided((3, 2048), (2049, 1), device='cuda:0', dtype=torch.int64)
    embedding = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    select_1 = rand_strided((1, 2048, 1, 64), (0, 128, 0, 2), device='cuda:0', dtype=torch.float16)
    select_3 = rand_strided((1, 2048, 1, 64), (0, 128, 0, 2), device='cuda:0', dtype=torch.float16)
    slice_3 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    scalar_tensor = rand_strided((), (), device='cuda:0', dtype=torch.float16)
    permute_8 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_9 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_10 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_5 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_12 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_20 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_21 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_22 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_11 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_24 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_32 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_33 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_34 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_17 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_36 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_44 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_45 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_46 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_23 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_48 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_56 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_57 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_58 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_29 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_60 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_68 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_69 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_70 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_35 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_72 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_80 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_81 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_82 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_41 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_84 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_92 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_93 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_94 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_47 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_96 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_104 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_105 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_106 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_53 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_108 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_116 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_117 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_118 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_59 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_120 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_128 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_129 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_130 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_65 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_132 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_140 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_141 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_142 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_71 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_144 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_152 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_153 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_154 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_77 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_156 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_164 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_165 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_166 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_83 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_168 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_176 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_177 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_178 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_89 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_180 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_188 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_189 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_190 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_95 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_192 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_200 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_201 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_202 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_101 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_204 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_212 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_213 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_214 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_107 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_216 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_224 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_225 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_226 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_113 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_228 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_236 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_237 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_238 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_119 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_240 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_248 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_249 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_250 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_125 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_252 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_260 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_261 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_262 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_131 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_264 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_272 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_273 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_274 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    add_137 = rand_strided((3, 2048, 2048), (4194304, 2048, 1), device='cuda:0', dtype=torch.float16)
    permute_276 = rand_strided((2048, 6144), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_284 = rand_strided((2048, 2048), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_285 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_286 = rand_strided((2048, 5632), (1, 2048), device='cuda:0', dtype=torch.float16)
    permute_287 = rand_strided((5632, 2048), (1, 5632), device='cuda:0', dtype=torch.float16)
    rsqrt_48 = rand_strided((3, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_624 = rand_strided((6144, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_291 = rand_strided((6144, 50257), (50257, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_292 = rand_strided((), (), device='cuda:0', dtype=torch.float16)
    permute_291 = rand_strided((50257, 2048), (2048, 1), device='cuda:0', dtype=torch.float16)
    permute_327 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_359 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_391 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_423 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_455 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_487 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_519 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_551 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_583 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_615 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_647 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_679 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_711 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_743 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_775 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_807 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_839 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_871 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_903 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_935 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_967 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_999 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    permute_1031 = rand_strided((2048, 5632), (5632, 1), device='cuda:0', dtype=torch.float16)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float16)
    return print_performance(lambda: call([primals_1, primals_4, primals_8, primals_11, primals_15, primals_18, primals_22, primals_25, primals_29, primals_32, primals_36, primals_39, primals_43, primals_46, primals_50, primals_53, primals_57, primals_60, primals_64, primals_67, primals_71, primals_74, primals_78, primals_81, primals_85, primals_88, primals_92, primals_95, primals_99, primals_102, primals_106, primals_109, primals_113, primals_116, primals_120, primals_123, primals_127, primals_130, primals_134, primals_137, primals_141, primals_144, primals_148, primals_151, primals_155, primals_158, primals_162, primals_165, primals_169, primals_174, primals_175, embedding, permute, select_1, select_3, slice_3, scalar_tensor, permute_8, permute_9, permute_10, add_5, permute_12, permute_20, permute_21, permute_22, add_11, permute_24, permute_32, permute_33, permute_34, add_17, permute_36, permute_44, permute_45, permute_46, add_23, permute_48, permute_56, permute_57, permute_58, add_29, permute_60, permute_68, permute_69, permute_70, add_35, permute_72, permute_80, permute_81, permute_82, add_41, permute_84, permute_92, permute_93, permute_94, add_47, permute_96, permute_104, permute_105, permute_106, add_53, permute_108, permute_116, permute_117, permute_118, add_59, permute_120, permute_128, permute_129, permute_130, add_65, permute_132, permute_140, permute_141, permute_142, add_71, permute_144, permute_152, permute_153, permute_154, add_77, permute_156, permute_164, permute_165, permute_166, add_83, permute_168, permute_176, permute_177, permute_178, add_89, permute_180, permute_188, permute_189, permute_190, add_95, permute_192, permute_200, permute_201, permute_202, add_101, permute_204, permute_212, permute_213, permute_214, add_107, permute_216, permute_224, permute_225, permute_226, add_113, permute_228, permute_236, permute_237, permute_238, add_119, permute_240, permute_248, permute_249, permute_250, add_125, permute_252, permute_260, permute_261, permute_262, add_131, permute_264, permute_272, permute_273, permute_274, add_137, permute_276, permute_284, permute_285, permute_286, permute_287, rsqrt_48, view_624, convert_element_type_291, convert_element_type_292, permute_291, permute_327, permute_359, permute_391, permute_423, permute_455, permute_487, permute_519, permute_551, permute_583, permute_615, permute_647, permute_679, permute_711, permute_743, permute_775, permute_807, permute_839, permute_871, permute_903, permute_935, permute_967, permute_999, permute_1031, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")

