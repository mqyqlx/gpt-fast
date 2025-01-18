
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
# cross_entropy => scalar_tensor_4
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
# cross_entropy => scalar_tensor_4
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
# cross_entropy => scalar_tensor_5
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


# kernel path: /tmp/torchinductor_mengqy/uw/cuwsj7cbrsfnfya4arnuwdga7ywgzroyb3jeim6rx47dztryluju.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_3 = async_compile.triton('triton_poi_fused_3', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_3', 'configs': [instance_descriptor(divisible_by_16=(1,), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]})
@triton.jit
def triton_poi_fused_3(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 43008
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 7
    x1 = (xindex // 7)
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0 + (50264*x1)), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/4p/c4ptz47q72md2bks5rlfa3epzmvyajcmr24izhr3zvjuoekeslpd.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_4 = async_compile.triton('triton_poi_fused_4', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_4', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_4(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51463168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask).to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/gz/cgzquiepdyde5xb5ii7qz5tio6ocen73ttemvfrkhutt6hmnsoix.py
# Source Nodes: [], Original ATen: []

triton_poi_fused_5 = async_compile.triton('triton_poi_fused_5', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[8192], filename=__file__, meta={'signature': {0: '*fp16', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_5', 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]})
@triton.jit
def triton_poi_fused_5(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 7168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/tv/ctvv4xd7cl6b27e7jqbsyjf573ngqo46yknmc6pf2csts7c6gklg.py
# Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
# stack => cat
# stack_1 => cat_1
triton_poi_fused_stack_6 = async_compile.triton('triton_poi_fused_stack_6', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[4194304], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: '*fp32', 7: '*fp32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_stack_6', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8,))]})
@triton.jit
def triton_poi_fused_stack_6(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, out_ptr2, out_ptr3, xnumel, XBLOCK : tl.constexpr):
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
    tmp7 = tl.load(in_ptr2 + ((2*x0) + (64*x2)), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr3 + (x0 + (64*x4)), None).to(tl.float32)
    tmp17 = tl.load(in_ptr3 + (32 + x0 + (64*x4)), None).to(tl.float32)
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


# kernel path: /tmp/torchinductor_mengqy/3b/c3bbxakqyheex4npoam6miy2kdwfwzubpnxmcbmpw6h3n6fti6ls.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone
triton_poi_fused_clone_7 = async_compile.triton('triton_poi_fused_clone_7', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_7', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
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
    tmp0 = tl.load(in_ptr0 + (x0 + (64*x2) + (1024*x1) + (2097152*x3)), None)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x4), tmp1, None)
''')


# kernel path: /tmp/torchinductor_mengqy/xd/cxddrcigxlqhp52mobbzig2d25glycv74uqvrb4kraknflm3xkgo.py
# Source Nodes: [matmul], Original ATen: [aten.clone]
# matmul => clone_1
triton_poi_fused_clone_8 = async_compile.triton('triton_poi_fused_clone_8', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[4096, 2048], tile_hint=TileHint.SQUARE,filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_8', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]})
@triton.jit
def triton_poi_fused_clone_8(in_ptr0, out_ptr0, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_mengqy/3u/c3udmp3cly6zd4iwhy5yy7fjeoh35xfwj47atkvkl7psoqathuso.py
# Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
# mul_11 => mul_11
# softmax => amax, convert_element_type_6, convert_element_type_7, div, exp, sub_2, sum_1
# where => where
triton_red_fused__softmax_mul_where_9 = async_compile.triton('triton_red_fused__softmax_mul_where_9', '''
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
    meta={'signature': {0: '*i1', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_mul_where_9', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_red_fused__softmax_mul_where_9(in_ptr0, in_ptr1, in_ptr2, out_ptr2, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_mengqy/lb/clbz7r7v53id3yh3fg4kkmkdu4vof54nstohorbwdacobqhraiqo.py
# Source Nodes: [matmul_1], Original ATen: [aten.clone]
# matmul_1 => clone_2
triton_poi_fused_clone_10 = async_compile.triton('triton_poi_fused_clone_10', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_10', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_clone_10(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
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


# kernel path: /tmp/torchinductor_mengqy/73/c73chbktynjs7wfmdfr4lpfa3qdqi5avdz4lssj76zx6y52so4zk.py
# Source Nodes: [linear_3], Original ATen: [aten.view]
# linear_3 => view_22
triton_poi_fused_view_11 = async_compile.triton('triton_poi_fused_view_11', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_view_11', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_view_11(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    x2 = xindex
    tmp0 = tl.load(in_ptr0 + ((64*(x1 % 2048)) + (131072*(x0 // 64)) + (2097152*(x1 // 2048)) + (x0 % 64)), None).to(tl.float32)
    tl.store(out_ptr0 + (x2), tmp0, None)
''')


# kernel path: /tmp/torchinductor_mengqy/45/c45hk43vf5gjl2flu2ucr2dy6hbt7gznd2kpiv2z23tdecnsfcws.py
# Source Nodes: [add_2, add_3, add_4, float_1, float_2, float_3, float_4, mean, mean_1, mean_2, mul, mul_1, mul_12, mul_13, mul_14, mul_2, mul_3, mul_6, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_2 => add_9
# add_3 => add_10, add_3
# add_4 => add_11, add_4
# float_1 => convert_element_type_14
# float_2 => convert_element_type_16
# float_3 => convert_element_type_18
# float_4 => convert_element_type_8
# mean => mean_3
# mean_1 => mean_1, mean_4
# mean_2 => mean_5
# mul => mul_25
# mul_1 => mul_21
# mul_12 => mul_12
# mul_13 => mul_13
# mul_14 => mul_14
# mul_2 => mul_22
# mul_3 => mul_23, mul_28
# mul_6 => mul_31
# rsqrt_1 => rsqrt_1
# type_as_3 => convert_element_type_9
triton_per_fused__to_copy_add_mean_mul_rsqrt_12 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_rsqrt_12', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp32', 7: '*fp32', 8: '*fp32', 9: '*fp32', 10: '*fp16', 11: 'i32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_rsqrt_12', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11, 12))]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_rsqrt_12(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp9 = tl.load(in_ptr2 + (2*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp19 = tl.load(in_ptr3 + (2*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp28 = tl.load(in_ptr4 + (2*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp44 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp4 = tmp3 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp10 = tmp9 * tmp0
    tmp11 = 0.0
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp13 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp20 = tmp19 * tmp0
    tmp21 = tmp20 + tmp11
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp22 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp29 = tmp28 * tmp0
    tmp30 = tmp29 + tmp11
    tmp31 = tmp30.to(tl.float32)
    tmp32 = tmp31 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tmp37 = 1024.0
    tmp38 = tmp8 / tmp37
    tmp39 = 1e-05
    tmp40 = tmp38 + tmp39
    tmp41 = tl.math.rsqrt(tmp40)
    tmp42 = tmp3 * tmp41
    tmp43 = tmp42.to(tl.float32)
    tmp45 = tmp43 * tmp44
    tl.store(out_ptr4 + (r1 + (1024*x0)), tmp45, rmask)
    tl.store(out_ptr0 + (x0), tmp8, None)
    tl.store(out_ptr1 + (x0), tmp18, None)
    tl.store(out_ptr2 + (x0), tmp27, None)
    tl.store(out_ptr3 + (x0), tmp36, None)
''')


# kernel path: /tmp/torchinductor_mengqy/hf/chfop25tov5jylbx45fprehxuyqtav32xlo3a4ummjomhmzb2i2i.py
# Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
# mul_15 => mul_16
# silu => convert_element_type_10, convert_element_type_11, mul_15, sigmoid
triton_poi_fused_mul_silu_13 = async_compile.triton('triton_poi_fused_mul_silu_13', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_mul_silu_13', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]})
@triton.jit
def triton_poi_fused_mul_silu_13(in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17301504
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


# kernel path: /tmp/torchinductor_mengqy/7b/c7byq7ugvjikk35vsdlr77bo32jzzbfrnyfnt2g53g2krbjpfndq.py
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp32', 13: '*fp32', 14: '*fp32', 15: '*fp32', 16: '*fp16', 17: '*fp32', 18: '*fp16', 19: '*fp32', 20: '*fp16', 21: 'i32', 22: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_rsqrt_14', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(21, 22))]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_rsqrt_14(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (3*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (3*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp13 = tl.load(in_ptr5 + (3*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr6 + (3*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr7 + (3*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp23 = tl.load(in_ptr8 + (3*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp39 = tl.load(in_ptr9 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp51 = tl.load(in_ptr10 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp63 = tl.load(in_ptr11 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = 0.0
    tmp4 = tmp2 + tmp3
    tmp7 = tmp1 + tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp5 * tmp9
    tmp11 = tmp4 + tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp13 * tmp1
    tmp15 = tmp14 + tmp3
    tmp17 = tmp16 * tmp9
    tmp18 = tmp15 + tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp21 = tmp20 * tmp1
    tmp22 = tmp21 + tmp3
    tmp24 = tmp23 * tmp9
    tmp25 = tmp22 + tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = tmp12 * tmp12
    tmp28 = tl.broadcast_to(tmp27, [RBLOCK])
    tmp30 = tl.where(rmask, tmp28, 0)
    tmp31 = triton_helpers.promote_to_tensor(tl.sum(tmp30, 0))
    tmp32 = 1024.0
    tmp33 = tmp31 / tmp32
    tmp34 = 1e-05
    tmp35 = tmp33 + tmp34
    tmp36 = tl.math.rsqrt(tmp35)
    tmp37 = tmp12 * tmp36
    tmp38 = tmp37.to(tl.float32)
    tmp40 = tmp38 * tmp39
    tmp41 = tmp19 * tmp19
    tmp42 = tl.broadcast_to(tmp41, [RBLOCK])
    tmp44 = tl.where(rmask, tmp42, 0)
    tmp45 = triton_helpers.promote_to_tensor(tl.sum(tmp44, 0))
    tmp46 = tmp45 / tmp32
    tmp47 = tmp46 + tmp34
    tmp48 = tl.math.rsqrt(tmp47)
    tmp49 = tmp19 * tmp48
    tmp50 = tmp49.to(tl.float32)
    tmp52 = tmp50 * tmp51
    tmp53 = tmp26 * tmp26
    tmp54 = tl.broadcast_to(tmp53, [RBLOCK])
    tmp56 = tl.where(rmask, tmp54, 0)
    tmp57 = triton_helpers.promote_to_tensor(tl.sum(tmp56, 0))
    tmp58 = tmp57 / tmp32
    tmp59 = tmp58 + tmp34
    tmp60 = tl.math.rsqrt(tmp59)
    tmp61 = tmp26 * tmp60
    tmp62 = tmp61.to(tl.float32)
    tmp64 = tmp62 * tmp63
    tl.store(out_ptr0 + (r1 + (1024*x0)), tmp12, rmask)
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp19, rmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp26, rmask)
    tl.store(out_ptr4 + (r1 + (1024*x0)), tmp40, rmask)
    tl.store(out_ptr6 + (r1 + (1024*x0)), tmp52, rmask)
    tl.store(out_ptr8 + (r1 + (1024*x0)), tmp64, rmask)
    tl.store(out_ptr3 + (x0), tmp31, None)
    tl.store(out_ptr5 + (x0), tmp45, None)
    tl.store(out_ptr7 + (x0), tmp57, None)
''')


# kernel path: /tmp/torchinductor_mengqy/xt/cxtcdcqqufye2gutf6vs3tyu4ntlrgj2mzkxqrd7wig3pfe56uro.py
# Source Nodes: [add, add_14, add_15, add_18, add_19, add_2, add_3, add_4, add_5, add_6, float_1, float_2, float_3, float_6, mean, mean_1, mean_2, mean_3, mul, mul_1, mul_12, mul_13, mul_18, mul_19, mul_2, mul_20, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, rsqrt, rsqrt_1, rsqrt_2, rsqrt_3, type_as, type_as_1, type_as_2, type_as_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add => add_13
# add_14 => add_30
# add_15 => add_31
# add_18 => add_37
# add_19 => add_38
# add_2 => add_9
# add_3 => add_10, add_3
# add_4 => add_11, add_14
# add_5 => add_5
# add_6 => add_15
# float_1 => convert_element_type_14
# float_2 => convert_element_type_16
# float_3 => convert_element_type_18
# float_6 => convert_element_type_44
# mean => mean_3
# mean_1 => mean_4
# mean_2 => mean_5
# mean_3 => mean_11
# mul => mul_25
# mul_1 => mul_21, mul_26
# mul_12 => mul_58
# mul_13 => mul_59
# mul_18 => mul_78
# mul_19 => mul_79
# mul_2 => mul_22, mul_27
# mul_20 => mul_80
# mul_3 => mul_23, mul_28
# mul_4 => mul_29
# mul_5 => mul_30
# mul_6 => mul_31
# mul_7 => mul_32
# mul_8 => mul_33
# rsqrt => rsqrt_3
# rsqrt_1 => rsqrt_4
# rsqrt_2 => rsqrt_5
# rsqrt_3 => rsqrt_11
# type_as => convert_element_type_15
# type_as_1 => convert_element_type_17
# type_as_2 => convert_element_type_19
# type_as_5 => convert_element_type_45
triton_per_fused__to_copy_add_mean_mul_rsqrt_15 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_rsqrt_15', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp16', 9: '*fp16', 10: '*fp32', 11: '*fp16', 12: '*fp16', 13: '*fp32', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp32', 20: '*fp16', 21: 'i32', 22: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_rsqrt_15', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(21, 22))]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_rsqrt_15(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (3*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (3*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp12 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp14 = tl.load(in_ptr5 + (2*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp26 = tl.load(in_ptr7 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp28 = tl.load(in_ptr8 + (2*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp32 = tl.load(in_ptr9 + (x0), None, eviction_policy='evict_last')
    tmp38 = tl.load(in_ptr10 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp40 = tl.load(in_ptr11 + (2*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp44 = tl.load(in_ptr12 + (x0), None, eviction_policy='evict_last')
    tmp50 = tl.load(in_ptr13 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp63 = tl.load(in_ptr14 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = 0.0
    tmp4 = tmp2 + tmp3
    tmp7 = tmp1 + tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp5 * tmp9
    tmp11 = tmp4 + tmp10
    tmp13 = tmp11 + tmp12
    tmp15 = tmp14 * tmp1
    tmp16 = tmp15 + tmp3
    tmp17 = tmp16.to(tl.float32)
    tmp19 = 1024.0
    tmp20 = tmp18 / tmp19
    tmp21 = 1e-05
    tmp22 = tmp20 + tmp21
    tmp23 = tl.math.rsqrt(tmp22)
    tmp24 = tmp17 * tmp23
    tmp25 = tmp24.to(tl.float32)
    tmp27 = tmp25 * tmp26
    tmp29 = tmp28 * tmp1
    tmp30 = tmp29 + tmp3
    tmp31 = tmp30.to(tl.float32)
    tmp33 = tmp32 / tmp19
    tmp34 = tmp33 + tmp21
    tmp35 = tl.math.rsqrt(tmp34)
    tmp36 = tmp31 * tmp35
    tmp37 = tmp36.to(tl.float32)
    tmp39 = tmp37 * tmp38
    tmp41 = tmp40 * tmp1
    tmp42 = tmp41 + tmp3
    tmp43 = tmp42.to(tl.float32)
    tmp45 = tmp44 / tmp19
    tmp46 = tmp45 + tmp21
    tmp47 = tl.math.rsqrt(tmp46)
    tmp48 = tmp43 * tmp47
    tmp49 = tmp48.to(tl.float32)
    tmp51 = tmp49 * tmp50
    tmp52 = tmp13.to(tl.float32)
    tmp53 = tmp52 * tmp52
    tmp54 = tl.broadcast_to(tmp53, [RBLOCK])
    tmp56 = tl.where(rmask, tmp54, 0)
    tmp57 = triton_helpers.promote_to_tensor(tl.sum(tmp56, 0))
    tmp58 = tmp57 / tmp19
    tmp59 = tmp58 + tmp21
    tmp60 = tl.math.rsqrt(tmp59)
    tmp61 = tmp52 * tmp60
    tmp62 = tmp61.to(tl.float32)
    tmp64 = tmp62 * tmp63
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp13, rmask)
    tl.store(out_ptr0 + (r1 + (1024*x0)), tmp27, rmask)
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp39, rmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp51, rmask)
    tl.store(out_ptr4 + (r1 + (1024*x0)), tmp64, rmask)
    tl.store(out_ptr3 + (x0), tmp57, None)
''')


# kernel path: /tmp/torchinductor_mengqy/sa/csa7ysp3kl6qws56uraqtaplgwsu3sahgnxp36xli5y64e3fwwga.py
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
triton_per_fused__to_copy_add_mean_mul_rsqrt_16 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_rsqrt_16', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_rsqrt_16', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_rsqrt_16(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (2*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp20 = tl.load(in_ptr3 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = 0.0
    tmp4 = tmp2 + tmp3
    tmp6 = tmp4 + tmp5
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp7 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp13 = 1024.0
    tmp14 = tmp12 / tmp13
    tmp15 = 1e-05
    tmp16 = tmp14 + tmp15
    tmp17 = tl.math.rsqrt(tmp16)
    tmp18 = tmp7 * tmp17
    tmp19 = tmp18.to(tl.float32)
    tmp21 = tmp19 * tmp20
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp21, rmask)
    tl.store(out_ptr0 + (x0), tmp12, None)
''')


# kernel path: /tmp/torchinductor_mengqy/bk/cbk7b6qdwemicfbvje26n7qrvopl5sw3uhnaqkru6r3m4whpwjel.py
# Source Nodes: [add, add_10, add_12, add_18, add_19, add_20, add_21, add_22, add_23, add_24, add_25, add_26, add_27, add_28, add_29, add_3, add_36, add_37, add_38, add_39, add_5, float_1, float_2, float_3, mean, mean_1, mean_2, mul, mul_1, mul_15, mul_16, mul_17, mul_18, mul_19, mul_2, mul_20, mul_21, mul_22, mul_23, mul_24, mul_25, mul_26, mul_3, mul_32, mul_33, mul_34, mul_35, mul_4, mul_5, mul_6, mul_7, mul_8, rsqrt, rsqrt_1, rsqrt_2, type_as, type_as_1, type_as_2], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add => add_55
# add_10 => add_18
# add_12 => add_20
# add_18 => add_43
# add_19 => add_44
# add_20 => add_39, add_45
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
# add_36 => add_70
# add_37 => add_71
# add_38 => add_72
# add_39 => add_73
# add_5 => add_12, add_5
# float_1 => convert_element_type_50, convert_element_type_68
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
# mul_32 => mul_130
# mul_33 => mul_131
# mul_34 => mul_132
# mul_35 => mul_133
# mul_4 => mul_103, mul_24
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
triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_17 = async_compile.triton('triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_17', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: '*fp16', 21: '*fp16', 22: '*fp16', 23: '*fp16', 24: '*fp16', 25: '*fp16', 26: '*fp16', 27: '*fp16', 28: '*fp16', 29: '*fp32', 30: '*fp16', 31: '*fp16', 32: '*fp16', 33: '*fp16', 34: '*fp16', 35: '*fp32', 36: '*fp16', 37: '*fp32', 38: '*fp16', 39: '*fp32', 40: '*fp16', 41: '*fp32', 42: '*fp16', 43: '*fp16', 44: '*fp16', 45: '*fp16', 46: 'i32', 47: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_17', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(46, 47))]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_17(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, in_ptr20, in_ptr21, in_ptr22, in_ptr23, in_ptr24, in_ptr25, in_ptr26, in_ptr27, in_ptr28, in_ptr29, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, xnumel, rnumel):
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
    tmp0 = tl.load(in_ptr0 + (5*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = tl.load(in_ptr1 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp5 = tl.load(in_ptr2 + (5*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp12 = tl.load(in_ptr5 + (5*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr6 + (2*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr7 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp18 = tl.load(in_ptr8 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp22 = tl.load(in_ptr9 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp25 = tl.load(in_ptr10 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp28 = tl.load(in_ptr11 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp31 = tl.load(in_ptr12 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp34 = tl.load(in_ptr13 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp37 = tl.load(in_ptr14 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp40 = tl.load(in_ptr15 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp43 = tl.load(in_ptr16 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp46 = tl.load(in_ptr17 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp49 = tl.load(in_ptr18 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp52 = tl.load(in_ptr19 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp55 = tl.load(in_ptr20 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp71 = tl.load(in_ptr21 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp84 = tl.load(in_ptr22 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp97 = tl.load(in_ptr23 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp99 = tl.load(in_ptr24 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp100 = tl.load(in_ptr25 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp103 = tl.load(in_ptr26 + (5*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp104 = tl.load(in_ptr27 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp105 = tl.load(in_ptr28 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp117 = tl.load(in_ptr29 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = 0.0
    tmp4 = tmp2 + tmp3
    tmp7 = tmp1 + tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp5 * tmp9
    tmp11 = tmp4 + tmp10
    tmp14 = tmp13 * tmp1
    tmp15 = tmp14 + tmp3
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp20 = tmp12 * tmp19
    tmp21 = tmp11 + tmp20
    tmp23 = tmp22 * tmp1
    tmp24 = tmp23 + tmp3
    tmp26 = tmp25 * tmp9
    tmp27 = tmp24 + tmp26
    tmp29 = tmp28 * tmp19
    tmp30 = tmp27 + tmp29
    tmp32 = tmp31 * tmp1
    tmp33 = tmp32 + tmp3
    tmp35 = tmp34 * tmp9
    tmp36 = tmp33 + tmp35
    tmp38 = tmp37 * tmp19
    tmp39 = tmp36 + tmp38
    tmp41 = tmp40 * tmp1
    tmp42 = tmp41 + tmp3
    tmp44 = tmp43 * tmp9
    tmp45 = tmp42 + tmp44
    tmp47 = tmp46 * tmp19
    tmp48 = tmp45 + tmp47
    tmp50 = tmp49 * tmp1
    tmp51 = tmp50 + tmp3
    tmp53 = tmp52 * tmp9
    tmp54 = tmp51 + tmp53
    tmp56 = tmp55 * tmp19
    tmp57 = tmp54 + tmp56
    tmp58 = tmp30.to(tl.float32)
    tmp59 = tmp58 * tmp58
    tmp60 = tl.broadcast_to(tmp59, [RBLOCK])
    tmp62 = tl.where(rmask, tmp60, 0)
    tmp63 = triton_helpers.promote_to_tensor(tl.sum(tmp62, 0))
    tmp64 = 1024.0
    tmp65 = tmp63 / tmp64
    tmp66 = 1e-05
    tmp67 = tmp65 + tmp66
    tmp68 = tl.math.rsqrt(tmp67)
    tmp69 = tmp58 * tmp68
    tmp70 = tmp69.to(tl.float32)
    tmp72 = tmp70 * tmp71
    tmp73 = tmp39.to(tl.float32)
    tmp74 = tmp73 * tmp73
    tmp75 = tl.broadcast_to(tmp74, [RBLOCK])
    tmp77 = tl.where(rmask, tmp75, 0)
    tmp78 = triton_helpers.promote_to_tensor(tl.sum(tmp77, 0))
    tmp79 = tmp78 / tmp64
    tmp80 = tmp79 + tmp66
    tmp81 = tl.math.rsqrt(tmp80)
    tmp82 = tmp73 * tmp81
    tmp83 = tmp82.to(tl.float32)
    tmp85 = tmp83 * tmp84
    tmp86 = tmp48.to(tl.float32)
    tmp87 = tmp86 * tmp86
    tmp88 = tl.broadcast_to(tmp87, [RBLOCK])
    tmp90 = tl.where(rmask, tmp88, 0)
    tmp91 = triton_helpers.promote_to_tensor(tl.sum(tmp90, 0))
    tmp92 = tmp91 / tmp64
    tmp93 = tmp92 + tmp66
    tmp94 = tl.math.rsqrt(tmp93)
    tmp95 = tmp86 * tmp94
    tmp96 = tmp95.to(tl.float32)
    tmp98 = tmp96 * tmp97
    tmp101 = tmp99 * tmp100
    tmp102 = tmp101.to(tl.float32)
    tmp106 = tmp104 + tmp105
    tmp107 = tmp103 * tmp106
    tmp108 = tmp21 + tmp107
    tmp109 = tmp108.to(tl.float32)
    tmp110 = tmp102 * tmp109
    tmp111 = tl.broadcast_to(tmp110, [RBLOCK])
    tmp113 = tl.where(rmask, tmp111, 0)
    tmp114 = triton_helpers.promote_to_tensor(tl.sum(tmp113, 0))
    tmp115 = -0.5
    tmp116 = tmp114 * tmp115
    tmp118 = tmp117 * tmp117
    tmp119 = tmp118 * tmp117
    tmp120 = tmp116 * tmp119
    tmp121 = tmp120 / tmp64
    tmp122 = tmp121 * tmp109
    tmp123 = tmp102 * tmp117
    tmp124 = tmp123 + tmp122
    tmp125 = tmp124 + tmp122
    tmp126 = tmp125.to(tl.float32)
    tmp127 = tmp126 * tmp106
    tmp128 = tl.broadcast_to(tmp127, [RBLOCK])
    tmp130 = tl.where(rmask, tmp128, 0)
    tmp131 = triton_helpers.promote_to_tensor(tl.sum(tmp130, 0))
    tmp132 = tmp126 * tmp19
    tmp133 = tl.broadcast_to(tmp132, [RBLOCK])
    tmp135 = tl.where(rmask, tmp133, 0)
    tmp136 = triton_helpers.promote_to_tensor(tl.sum(tmp135, 0))
    tmp137 = tmp126 * tmp9
    tmp138 = tl.broadcast_to(tmp137, [RBLOCK])
    tmp140 = tl.where(rmask, tmp138, 0)
    tmp141 = triton_helpers.promote_to_tensor(tl.sum(tmp140, 0))
    tmp142 = tmp126 * tmp1
    tmp143 = tl.broadcast_to(tmp142, [RBLOCK])
    tmp145 = tl.where(rmask, tmp143, 0)
    tmp146 = triton_helpers.promote_to_tensor(tl.sum(tmp145, 0))
    tl.store(out_ptr0 + (r1 + (1024*x0)), tmp21, rmask)
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp30, rmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp39, rmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp48, rmask)
    tl.store(out_ptr4 + (r1 + (1024*x0)), tmp57, rmask)
    tl.store(out_ptr6 + (r1 + (1024*x0)), tmp72, rmask)
    tl.store(out_ptr8 + (r1 + (1024*x0)), tmp85, rmask)
    tl.store(out_ptr10 + (r1 + (1024*x0)), tmp98, rmask)
    tl.store(out_ptr12 + (r1 + (1024*x0)), tmp122, rmask)
    tl.store(out_ptr5 + (x0), tmp63, None)
    tl.store(out_ptr7 + (x0), tmp78, None)
    tl.store(out_ptr9 + (x0), tmp91, None)
    tl.store(out_ptr13 + (x0), tmp131, None)
    tl.store(out_ptr14 + (x0), tmp136, None)
    tl.store(out_ptr15 + (x0), tmp141, None)
    tl.store(out_ptr16 + (x0), tmp146, None)
''')


# kernel path: /tmp/torchinductor_mengqy/22/c2246u3ov72drg4e3le27raiyz6lc25gipwl6isbqdimywn6br2o.py
# Source Nodes: [add_20, add_39, float_1, mul_35, mul_45, type_as], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
# add_20 => add_39
# add_39 => add_73
# float_1 => convert_element_type_68
# mul_35 => mul_133
# mul_45 => mul_143
# type_as => convert_element_type_69
triton_red_fused__to_copy_add_mul_sum_18 = async_compile.triton('triton_red_fused__to_copy_add_mul_sum_18', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mul_sum_18', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mul_sum_18(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp14 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*r2) + (131072*x1)), rmask, other=0).to(tl.float32)
        tmp2 = tl.load(in_ptr2 + ((5*r2) + (640*x1)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr3 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr4 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp9 = tl.load(in_ptr5 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp5 = tmp3 + tmp4
        tmp6 = tmp2 * tmp5
        tmp7 = tmp1 + tmp6
        tmp8 = tmp7.to(tl.float32)
        tmp10 = tmp8 * tmp9
        tmp11 = tmp10.to(tl.float32)
        tmp12 = tmp0 * tmp11
        tmp13 = tl.broadcast_to(tmp12, [XBLOCK, RBLOCK])
        tmp15 = _tmp14 + tmp13
        _tmp14 = tl.where(rmask, tmp15, _tmp14)
    tmp14 = tl.sum(_tmp14, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp14, None)
''')


# kernel path: /tmp/torchinductor_mengqy/zk/czkeskpjb5lsu3jngssrjsa33bgv5mtibnl4w2fidf3gpekqg66r.py
# Source Nodes: [add_20, add_39, float_1, mul_35, mul_45, type_as], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
# add_20 => add_39
# add_39 => add_73
# float_1 => convert_element_type_68
# mul_35 => mul_133
# mul_45 => mul_143
# type_as => convert_element_type_69
triton_per_fused__to_copy_add_mul_sum_19 = async_compile.triton('triton_per_fused__to_copy_add_mul_sum_19', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[1024, 64],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mul_sum_19', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]}
)
@triton.jit
def triton_per_fused__to_copy_add_mul_sum_19(in_ptr0, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 1024
    rnumel = 48
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r1 = rindex
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (1024*r1)), rmask & xmask, other=0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.store(out_ptr0 + (x0), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/mk/cmkwxuqvoc4t4apzfblncqw2wmucylzeofqm6ikdyv3ddfluvlpr.py
# Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward, aten.sum]

triton_red_fused_add_select_backward_slice_backward_sum_20 = async_compile.triton('triton_red_fused_add_select_backward_slice_backward_sum_20', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[32, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_select_backward_slice_backward_sum_20', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]}
)
@triton.jit
def triton_red_fused_add_select_backward_slice_backward_sum_20(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 20
    rnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 5)
    x0 = xindex % 5
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp6 = tl.load(in_ptr0 + (r2), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp17 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp23 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp0 = x1
        tmp1 = tl.full([1, 1], 1, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp3 = x0
        tmp4 = tl.full([1, 1], 3, tl.int32)
        tmp5 = tmp3 == tmp4
        tmp7 = 0.0
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tl.where(tmp2, tmp8, tmp7)
        tmp10 = tl.full([1, 1], 2, tl.int32)
        tmp11 = tmp3 == tmp10
        tmp13 = tl.where(tmp11, tmp12, tmp7)
        tmp14 = tl.where(tmp2, tmp13, tmp7)
        tmp15 = tmp9 + tmp14
        tmp16 = tmp3 == tmp1
        tmp18 = tl.where(tmp16, tmp17, tmp7)
        tmp19 = tl.where(tmp2, tmp18, tmp7)
        tmp20 = tmp15 + tmp19
        tmp21 = tl.full([1, 1], 0, tl.int32)
        tmp22 = tmp3 == tmp21
        tmp24 = tl.where(tmp22, tmp23, tmp7)
        tmp25 = tl.where(tmp2, tmp24, tmp7)
        tmp26 = tmp20 + tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/dc/cdcdj267ym5j3ex3hfoconbsjrkih3bnowxjoiprw2c34jugwhbt.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_21 = async_compile.triton('triton_poi_fused_clone_21', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_21', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]})
@triton.jit
def triton_poi_fused_clone_21(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 122880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 5) % 4
    x0 = xindex % 5
    x2 = (xindex // 20)
    x3 = xindex
    tmp6 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp23 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x1
    tmp1 = tl.full([1], 1, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 3, tl.int32)
    tmp5 = tmp3 == tmp4
    tmp7 = 0.0
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tl.where(tmp2, tmp8, tmp7)
    tmp10 = tl.full([1], 2, tl.int32)
    tmp11 = tmp3 == tmp10
    tmp13 = tl.where(tmp11, tmp12, tmp7)
    tmp14 = tl.where(tmp2, tmp13, tmp7)
    tmp15 = tmp9 + tmp14
    tmp16 = tmp3 == tmp1
    tmp18 = tl.where(tmp16, tmp17, tmp7)
    tmp19 = tl.where(tmp2, tmp18, tmp7)
    tmp20 = tmp15 + tmp19
    tmp21 = tl.full([1], 0, tl.int32)
    tmp22 = tmp3 == tmp21
    tmp24 = tl.where(tmp22, tmp23, tmp7)
    tmp25 = tl.where(tmp2, tmp24, tmp7)
    tmp26 = tmp20 + tmp25
    tl.store(out_ptr0 + (x3), tmp26, None)
''')


# kernel path: /tmp/torchinductor_mengqy/hy/chytonudhjdpttwbop5hgita3xljmyl365mzzundzioeezix4zcy.py
# Source Nodes: [l__mod___dynamic_dense_3_act], Original ATen: [aten.gelu, aten.gelu_backward]
# l__mod___dynamic_dense_3_act => add_64, convert_element_type_66, erf_3, mul_124
triton_poi_fused_gelu_gelu_backward_22 = async_compile.triton('triton_poi_fused_gelu_gelu_backward_22', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_22', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_gelu_gelu_backward_22(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 122880
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = tl.math.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = 0.3989422804014327
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tmp1 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp20, None)
''')


# kernel path: /tmp/torchinductor_mengqy/ks/cks2hym32uz3hwyqa6vi2mpja6w5jbbgk3fi3znqqsoua3cwrsec.py
# Source Nodes: [add_31, add_32, float_6, mean_3, mul_18, mul_19, mul_20, rsqrt_3, type_as_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
# add_31 => add_60
# add_32 => add_61
# float_6 => convert_element_type_62
# mean_3 => mean_16
# mul_18 => mul_117
# mul_19 => mul_118
# mul_20 => mul_119
# rsqrt_3 => rsqrt_16
# type_as_5 => convert_element_type_63
triton_per_fused__to_copy_add_mean_mul_rsqrt_23 = async_compile.triton('triton_per_fused__to_copy_add_mean_mul_rsqrt_23', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_mean_mul_rsqrt_23', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_per_fused__to_copy_add_mean_mul_rsqrt_23(in_ptr0, in_ptr1, in_ptr2, out_ptr0, out_ptr1, xnumel, rnumel):
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
    tl.store(out_ptr0 + (x0), tmp8, None)
''')


# kernel path: /tmp/torchinductor_mengqy/57/c57o4awf63vhudzr2zcqfolbmm54vmv77prtnmd2qtj7cl6o7ibf.py
# Source Nodes: [add_31, add_33], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
# add_31 => add_60
# add_33 => add_62
triton_per_fused_add_div_mul_pow_sum_24 = async_compile.triton('triton_per_fused_add_div_mul_pow_sum_24', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_pow_sum_24', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]}
)
@triton.jit
def triton_per_fused_add_div_mul_pow_sum_24(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr1, xnumel, rnumel):
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
    tmp2 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp11 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = tmp0 * tmp11
    tmp13 = -0.5
    tmp14 = tmp10 * tmp13
    tmp15 = tmp11 * tmp11
    tmp16 = tmp15 * tmp11
    tmp17 = tmp14 * tmp16
    tmp18 = 1024.0
    tmp19 = tmp17 / tmp18
    tmp20 = 2.0
    tmp21 = tmp5 * tmp20
    tmp22 = tmp19 * tmp21
    tmp23 = tmp12 + tmp22
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp23, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/x5/cx55gi5zn3t4oiglo5uphnul6cetb4lvf7wmho7doul77kheq2ui.py
# Source Nodes: [silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
# silu => convert_element_type_64, convert_element_type_65, mul_120, sigmoid_3
triton_poi_fused_add_fill_mul_sigmoid_silu_sub_25 = async_compile.triton('triton_poi_fused_add_fill_mul_sigmoid_silu_sub_25', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[33554432], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_fill_mul_sigmoid_silu_sub_25', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]})
@triton.jit
def triton_poi_fused_add_fill_mul_sigmoid_silu_sub_25(in_out_ptr0, in_ptr0, in_ptr1, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 17301504
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


# kernel path: /tmp/torchinductor_mengqy/gd/cgdyedpwvzxrbkafxbulentjvj6wcsbfuebrifdfr273wtmc4ssd.py
# Source Nodes: [add_31, add_32, float_6, mean_3, mul_18, mul_19, rsqrt_3, type_as_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add_31 => add_60
# add_32 => add_61
# float_6 => convert_element_type_62
# mean_3 => mean_16
# mul_18 => mul_117
# mul_19 => mul_118
# rsqrt_3 => rsqrt_16
# type_as_5 => convert_element_type_63
triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_26 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_26', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_26', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_26(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp17 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp7 = tl.load(in_ptr4 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 + tmp4
        tmp6 = tmp5.to(tl.float32)
        tmp8 = 1024.0
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
    tmp17 = tl.sum(_tmp17, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp17, None)
''')


# kernel path: /tmp/torchinductor_mengqy/jc/cjcpri5ukrttewmsjw6gwjmvl2noxl754565wwua44ftp2e45u7o.py
# Source Nodes: [add_31, add_32, float_6, mean_3, mul_18, rsqrt_3], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add_31 => add_60
# add_32 => add_61
# float_6 => convert_element_type_62
# mean_3 => mean_16
# mul_18 => mul_117
# rsqrt_3 => rsqrt_16
triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_27 = async_compile.triton('triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_27', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp32', 7: 'i32', 8: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_27', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(7, 8))]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_27(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, xnumel, rnumel):
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
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp5 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 1024.0
    tmp17 = tmp15 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp5 * tmp20
    tmp22 = -0.5
    tmp23 = tmp14 * tmp22
    tmp24 = tmp20 * tmp20
    tmp25 = tmp24 * tmp20
    tmp26 = tmp23 * tmp25
    tmp27 = tmp26 / tmp16
    tmp28 = tmp27 * tmp9
    tmp29 = tmp21 + tmp28
    tmp30 = tmp29 + tmp28
    tmp32 = tmp30.to(tl.float32)
    tmp33 = tmp31 + tmp32
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp33, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/ry/crywavnepaizoudcf7b6ebl44wsushyypjdjs4gvrhz5floscd2y.py
# Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
# cross_entropy => scalar_tensor_5
triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_28 = async_compile.triton('triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_28', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*i1', 3: '*fp16', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_28', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_28(in_ptr0, in_ptr1, in_ptr2, out_ptr1, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
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
        tmp19 = 0.125
        tmp20 = tmp18 * tmp19
        tl.store(out_ptr1 + (r1 + (2048*x0)), tmp20, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/ya/cyak2p6noe2r3woiyz3pzzn7x7gvkus6j2ckq67uglsgdyyjz2xa.py
# Source Nodes: [], Original ATen: [aten._to_copy]

triton_poi_fused__to_copy_29 = async_compile.triton('triton_poi_fused__to_copy_29', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[8192, 1024], tile_hint=TileHint.DEFAULT,filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_29', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]})
@triton.jit
def triton_poi_fused__to_copy_29(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, out_ptr1, ynumel, xnumel, YBLOCK : tl.constexpr, XBLOCK : tl.constexpr):
    ynumel = 6144
    xnumel = 1024
    yoffset = tl.program_id(1) * YBLOCK
    yindex = yoffset + tl.arange(0, YBLOCK)[None, :]
    ymask = yindex < ynumel
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    x2 = xindex % 64
    x3 = (xindex // 64)
    y0 = yindex % 2048
    y1 = (yindex // 2048)
    x5 = xindex
    y4 = yindex
    tmp3 = tl.load(in_ptr0 + (2048 + y0 + (4096*(x2 % 32)) + (131072*x3) + (2097152*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr1 + ((2*(x2 % 32)) + (64*y0)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr2 + ((2*(x2 % 32)) + (64*y0)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp17 = tl.load(in_ptr0 + (y0 + (4096*(x2 % 32)) + (131072*x3) + (2097152*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp27 = tl.load(in_ptr3 + (1 + (2*(x2 % 32)) + (64*y0) + (131072*x3) + (2097152*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp34 = tl.load(in_ptr3 + ((2*(x2 % 32)) + (64*y0) + (131072*x3) + (2097152*y1)), xmask, eviction_policy='evict_last').to(tl.float32)
    tmp0 = (x2 // 32)
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
    tl.store(out_ptr0 + (x5 + (1024*y4)), tmp26, xmask)
    tl.store(out_ptr1 + (x5 + (1024*y4)), tmp43, xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/y7/cy76t6m3gltzyornvqj56wj3cvmhlwr53mxk3frueb44jdkqidjs.py
# Source Nodes: [add_27, float_3, mean_2, mul_6, mul_7, rsqrt_2, type_as_2], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add_27 => add_57
# float_3 => convert_element_type_54
# mean_2 => mean_15
# mul_6 => mul_105
# mul_7 => mul_106
# rsqrt_2 => rsqrt_15
# type_as_2 => convert_element_type_55
triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_30 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_30', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_30', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_30(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = 1024.0
        tmp5 = tmp3 / tmp4
        tmp6 = 1e-05
        tmp7 = tmp5 + tmp6
        tmp8 = tl.math.rsqrt(tmp7)
        tmp9 = tmp2 * tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_mengqy/rq/crqycrea7b5gqkw6fzlk75nfhss26s7mtx67y4k7cyntm7r67htn.py
# Source Nodes: [add, add_10, add_12, add_24, add_27, add_3, add_5, float_1, float_2, float_3, mean, mean_1, mean_2, mul, mul_3, mul_4, mul_6, rsqrt, rsqrt_1, rsqrt_2], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add => add_55
# add_10 => add_18
# add_12 => add_20
# add_24 => add_56
# add_27 => add_57
# add_3 => add_3
# add_5 => add_12, add_5
# float_1 => convert_element_type_50
# float_2 => convert_element_type_52
# float_3 => convert_element_type_54
# mean => mean_13
# mean_1 => mean_14
# mean_2 => mean_15
# mul => mul_99
# mul_3 => mul_102
# mul_4 => mul_24
# mul_6 => mul_105
# rsqrt => rsqrt_13
# rsqrt_1 => rsqrt_14
# rsqrt_2 => rsqrt_15
triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_31 = async_compile.triton('triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_31', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp32', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: '*fp16', 21: '*fp16', 22: '*fp16', 23: '*fp16', 24: '*fp16', 25: '*fp16', 26: '*fp16', 27: '*fp16', 28: '*fp16', 29: '*fp16', 30: '*fp16', 31: '*fp16', 32: '*fp16', 33: '*fp16', 34: 'i32', 35: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_31', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(34, 35))]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_31(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, out_ptr1, out_ptr3, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, out_ptr14, out_ptr15, out_ptr16, out_ptr17, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp11 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp28 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp29 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp32 = tl.load(in_ptr6 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp39 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp53 = tl.load(in_ptr8 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp54 = tl.load(in_ptr9 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp57 = tl.load(in_ptr10 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp64 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp78 = tl.load(in_ptr12 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp79 = tl.load(in_ptr13 + (2*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp80 = tl.load(in_ptr14 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp84 = tl.load(in_ptr15 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp86 = tl.load(in_ptr16 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp108 = tl.load(in_ptr17 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp110 = tl.load(in_ptr18 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp4.to(tl.float32)
    tmp6 = tmp3 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp12 = 1024.0
    tmp13 = tmp11 / tmp12
    tmp14 = 1e-05
    tmp15 = tmp13 + tmp14
    tmp16 = tl.math.rsqrt(tmp15)
    tmp17 = tmp3 * tmp16
    tmp18 = -0.5
    tmp19 = tmp10 * tmp18
    tmp20 = tmp16 * tmp16
    tmp21 = tmp20 * tmp16
    tmp22 = tmp19 * tmp21
    tmp23 = tmp22 / tmp12
    tmp24 = tmp23 * tmp5
    tmp25 = tmp17 + tmp24
    tmp26 = tmp25 + tmp24
    tmp27 = tmp26.to(tl.float32)
    tmp30 = tmp28 * tmp29
    tmp31 = tmp30.to(tl.float32)
    tmp33 = tmp32.to(tl.float32)
    tmp34 = tmp31 * tmp33
    tmp35 = tl.broadcast_to(tmp34, [RBLOCK])
    tmp37 = tl.where(rmask, tmp35, 0)
    tmp38 = triton_helpers.promote_to_tensor(tl.sum(tmp37, 0))
    tmp40 = tmp39 / tmp12
    tmp41 = tmp40 + tmp14
    tmp42 = tl.math.rsqrt(tmp41)
    tmp43 = tmp31 * tmp42
    tmp44 = tmp38 * tmp18
    tmp45 = tmp42 * tmp42
    tmp46 = tmp45 * tmp42
    tmp47 = tmp44 * tmp46
    tmp48 = tmp47 / tmp12
    tmp49 = tmp48 * tmp33
    tmp50 = tmp43 + tmp49
    tmp51 = tmp50 + tmp49
    tmp52 = tmp51.to(tl.float32)
    tmp55 = tmp53 * tmp54
    tmp56 = tmp55.to(tl.float32)
    tmp58 = tmp57.to(tl.float32)
    tmp59 = tmp56 * tmp58
    tmp60 = tl.broadcast_to(tmp59, [RBLOCK])
    tmp62 = tl.where(rmask, tmp60, 0)
    tmp63 = triton_helpers.promote_to_tensor(tl.sum(tmp62, 0))
    tmp65 = tmp64 / tmp12
    tmp66 = tmp65 + tmp14
    tmp67 = tl.math.rsqrt(tmp66)
    tmp68 = tmp56 * tmp67
    tmp69 = tmp63 * tmp18
    tmp70 = tmp67 * tmp67
    tmp71 = tmp70 * tmp67
    tmp72 = tmp69 * tmp71
    tmp73 = tmp72 / tmp12
    tmp74 = tmp73 * tmp58
    tmp75 = tmp68 + tmp74
    tmp76 = tmp75 + tmp74
    tmp77 = tmp76.to(tl.float32)
    tmp81 = tmp79 * tmp80
    tmp82 = 0.0
    tmp83 = tmp81 + tmp82
    tmp85 = tmp83 + tmp84
    tmp87 = tmp85 + tmp86
    tmp88 = tmp78 * tmp87
    tmp89 = tl.broadcast_to(tmp88, [RBLOCK])
    tmp91 = tl.where(rmask, tmp89, 0)
    tmp92 = triton_helpers.promote_to_tensor(tl.sum(tmp91, 0))
    tmp93 = tmp77 * tmp87
    tmp94 = tl.broadcast_to(tmp93, [RBLOCK])
    tmp96 = tl.where(rmask, tmp94, 0)
    tmp97 = triton_helpers.promote_to_tensor(tl.sum(tmp96, 0))
    tmp98 = tmp27 * tmp87
    tmp99 = tl.broadcast_to(tmp98, [RBLOCK])
    tmp101 = tl.where(rmask, tmp99, 0)
    tmp102 = triton_helpers.promote_to_tensor(tl.sum(tmp101, 0))
    tmp103 = tmp52 * tmp87
    tmp104 = tl.broadcast_to(tmp103, [RBLOCK])
    tmp106 = tl.where(rmask, tmp104, 0)
    tmp107 = triton_helpers.promote_to_tensor(tl.sum(tmp106, 0))
    tmp109 = tmp80 + tmp108
    tmp111 = tmp109 + tmp110
    tmp112 = tmp78 * tmp111
    tmp113 = tl.broadcast_to(tmp112, [RBLOCK])
    tmp115 = tl.where(rmask, tmp113, 0)
    tmp116 = triton_helpers.promote_to_tensor(tl.sum(tmp115, 0))
    tmp117 = tmp77 * tmp111
    tmp118 = tl.broadcast_to(tmp117, [RBLOCK])
    tmp120 = tl.where(rmask, tmp118, 0)
    tmp121 = triton_helpers.promote_to_tensor(tl.sum(tmp120, 0))
    tmp122 = tmp27 * tmp111
    tmp123 = tl.broadcast_to(tmp122, [RBLOCK])
    tmp125 = tl.where(rmask, tmp123, 0)
    tmp126 = triton_helpers.promote_to_tensor(tl.sum(tmp125, 0))
    tmp127 = tmp52 * tmp111
    tmp128 = tl.broadcast_to(tmp127, [RBLOCK])
    tmp130 = tl.where(rmask, tmp128, 0)
    tmp131 = triton_helpers.promote_to_tensor(tl.sum(tmp130, 0))
    tmp132 = tmp78 * tmp80
    tmp133 = tl.broadcast_to(tmp132, [RBLOCK])
    tmp135 = tl.where(rmask, tmp133, 0)
    tmp136 = triton_helpers.promote_to_tensor(tl.sum(tmp135, 0))
    tmp137 = tmp27 * tmp80
    tmp138 = tl.broadcast_to(tmp137, [RBLOCK])
    tmp140 = tl.where(rmask, tmp138, 0)
    tmp141 = triton_helpers.promote_to_tensor(tl.sum(tmp140, 0))
    tmp142 = tmp52 * tmp80
    tmp143 = tl.broadcast_to(tmp142, [RBLOCK])
    tmp145 = tl.where(rmask, tmp143, 0)
    tmp146 = triton_helpers.promote_to_tensor(tl.sum(tmp145, 0))
    tmp147 = tmp77 * tmp80
    tmp148 = tl.broadcast_to(tmp147, [RBLOCK])
    tmp150 = tl.where(rmask, tmp148, 0)
    tmp151 = triton_helpers.promote_to_tensor(tl.sum(tmp150, 0))
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp27, rmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp52, rmask)
    tl.store(out_ptr5 + (r1 + (1024*x0)), tmp77, rmask)
    tl.store(out_ptr6 + (x0), tmp92, None)
    tl.store(out_ptr7 + (x0), tmp97, None)
    tl.store(out_ptr8 + (x0), tmp102, None)
    tl.store(out_ptr9 + (x0), tmp107, None)
    tl.store(out_ptr10 + (x0), tmp116, None)
    tl.store(out_ptr11 + (x0), tmp121, None)
    tl.store(out_ptr12 + (x0), tmp126, None)
    tl.store(out_ptr13 + (x0), tmp131, None)
    tl.store(out_ptr14 + (x0), tmp136, None)
    tl.store(out_ptr15 + (x0), tmp141, None)
    tl.store(out_ptr16 + (x0), tmp146, None)
    tl.store(out_ptr17 + (x0), tmp151, None)
''')


# kernel path: /tmp/torchinductor_mengqy/ce/cceu77x34yhgajzxtwg35cwrjswjub4dhylvkdgghcxfmyltvvi7.py
# Source Nodes: [add_24, float_2, mean_1, mul_3, mul_4, rsqrt_1, type_as_1], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add_24 => add_56
# float_2 => convert_element_type_52
# mean_1 => mean_14
# mul_3 => mul_102
# mul_4 => mul_103
# rsqrt_1 => rsqrt_14
# type_as_1 => convert_element_type_53
triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_32 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_32', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_32', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_32(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp13 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*r2) + (131072*x1)), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp2 = tmp1.to(tl.float32)
        tmp4 = 1024.0
        tmp5 = tmp3 / tmp4
        tmp6 = 1e-05
        tmp7 = tmp5 + tmp6
        tmp8 = tl.math.rsqrt(tmp7)
        tmp9 = tmp2 * tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp11 = tmp0 * tmp10
        tmp12 = tl.broadcast_to(tmp11, [XBLOCK, RBLOCK])
        tmp14 = _tmp13 + tmp12
        _tmp13 = tl.where(rmask, tmp14, _tmp13)
    tmp13 = tl.sum(_tmp13, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp13, None)
''')


# kernel path: /tmp/torchinductor_mengqy/pj/cpj2l45kuhfq5642yyjav3w26mmfbehdlcogrthdavbsvl722ara.py
# Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward]

triton_poi_fused_add_select_backward_slice_backward_33 = async_compile.triton('triton_poi_fused_add_select_backward_slice_backward_33', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_select_backward_slice_backward_33', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(11,))]})
@triton.jit
def triton_poi_fused_add_select_backward_slice_backward_33(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 24576)
    x0 = xindex % 4
    x1 = (xindex // 4) % 6144
    x3 = xindex
    tmp6 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp23 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp27 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp31 = tl.load(in_ptr5 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp36 = tl.load(in_ptr6 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp40 = tl.load(in_ptr7 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp44 = tl.load(in_ptr8 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp49 = tl.load(in_ptr9 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x2
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 2, tl.int32)
    tmp5 = tmp3 == tmp4
    tmp7 = 0.0
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tl.where(tmp2, tmp8, tmp7)
    tmp10 = tl.full([1], 1, tl.int32)
    tmp11 = tmp3 == tmp10
    tmp13 = tl.where(tmp11, tmp12, tmp7)
    tmp14 = tl.where(tmp2, tmp13, tmp7)
    tmp15 = tmp9 + tmp14
    tmp16 = tl.full([1], 0, tl.int32)
    tmp17 = tmp3 == tmp16
    tmp19 = tl.where(tmp17, tmp18, tmp7)
    tmp20 = tl.where(tmp2, tmp19, tmp7)
    tmp21 = tmp15 + tmp20
    tmp22 = tmp0 == tmp4
    tmp24 = tl.where(tmp5, tmp23, tmp7)
    tmp25 = tl.where(tmp22, tmp24, tmp7)
    tmp26 = tmp21 + tmp25
    tmp28 = tl.where(tmp11, tmp27, tmp7)
    tmp29 = tl.where(tmp22, tmp28, tmp7)
    tmp30 = tmp26 + tmp29
    tmp32 = tl.where(tmp17, tmp31, tmp7)
    tmp33 = tl.where(tmp22, tmp32, tmp7)
    tmp34 = tmp30 + tmp33
    tmp35 = tmp0 == tmp10
    tmp37 = tl.where(tmp5, tmp36, tmp7)
    tmp38 = tl.where(tmp35, tmp37, tmp7)
    tmp39 = tmp34 + tmp38
    tmp41 = tl.where(tmp11, tmp40, tmp7)
    tmp42 = tl.where(tmp35, tmp41, tmp7)
    tmp43 = tmp39 + tmp42
    tmp45 = tl.where(tmp17, tmp44, tmp7)
    tmp46 = tl.where(tmp35, tmp45, tmp7)
    tmp47 = tmp43 + tmp46
    tmp48 = tmp0 == tmp16
    tmp50 = tl.where(tmp5, tmp49, tmp7)
    tmp51 = tl.where(tmp48, tmp50, tmp7)
    tmp52 = tmp47 + tmp51
    tl.store(in_out_ptr0 + (x3), tmp52, None)
''')


# kernel path: /tmp/torchinductor_mengqy/f5/cf5norajuxczyj2t34sqvojpgooxwdmotpaai6lq7opcflzocod4.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_34 = async_compile.triton('triton_poi_fused_clone_34', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_34', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4,))]})
@triton.jit
def triton_poi_fused_clone_34(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 4
    x1 = (xindex // 4) % 4
    x2 = (xindex // 16)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (4*x2) + (24576*x1)), None).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp13 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = x1
    tmp2 = tl.full([1], 0, tl.int32)
    tmp3 = tmp1 == tmp2
    tmp4 = x0
    tmp5 = tl.full([1], 1, tl.int32)
    tmp6 = tmp4 == tmp5
    tmp8 = 0.0
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tl.where(tmp3, tmp9, tmp8)
    tmp11 = tmp0 + tmp10
    tmp12 = tmp4 == tmp2
    tmp14 = tl.where(tmp12, tmp13, tmp8)
    tmp15 = tl.where(tmp3, tmp14, tmp8)
    tmp16 = tmp11 + tmp15
    tl.store(out_ptr0 + (x3), tmp16, None)
''')


# kernel path: /tmp/torchinductor_mengqy/xg/cxg7hh4h5kwcb255kpjgy4dccjwaulo7hm373of6gwc32o4ltp5p.py
# Source Nodes: [l__mod___dynamic_dense_2_act], Original ATen: [aten.gelu, aten.gelu_backward]
# l__mod___dynamic_dense_2_act => add_41, convert_element_type_48, erf_2, mul_85
triton_poi_fused_gelu_gelu_backward_35 = async_compile.triton('triton_poi_fused_gelu_gelu_backward_35', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_35', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_gelu_gelu_backward_35(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 98304
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = tl.math.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = 0.3989422804014327
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tmp1 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp20, None)
''')


# kernel path: /tmp/torchinductor_mengqy/l2/cl25bysrvnnzdcf4k53fovv46fk5yow6uxtoby3vrgfnjqbnjk2s.py
# Source Nodes: [add_20], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.pow, aten.sum]
# add_20 => add_39
triton_per_fused__to_copy_add_div_mul_pow_sum_36 = async_compile.triton('triton_per_fused__to_copy_add_div_mul_pow_sum_36', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp32', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: '*fp16', 21: '*fp16', 22: '*fp16', 23: '*fp16', 24: 'i32', 25: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mul_pow_sum_36', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(24, 25))]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mul_pow_sum_36(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel):
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
    tmp2 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp9 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp10 = tl.load(in_ptr4 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp13 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp15 = tl.load(in_ptr6 + (r1 + (1024*x0)), rmask, other=0)
    tmp19 = tl.load(in_ptr7 + (5*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp21 = tl.load(in_ptr8 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp22 = tl.load(in_ptr9 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp25 = tl.load(in_ptr10 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp26 = tl.load(in_ptr11 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp29 = tl.load(in_ptr12 + (5*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp31 = tl.load(in_ptr13 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp34 = tl.load(in_ptr14 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp37 = tl.load(in_ptr15 + (5*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp39 = tl.load(in_ptr16 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp42 = tl.load(in_ptr17 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp45 = tl.load(in_ptr18 + (5*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp47 = tl.load(in_ptr19 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp4 = tmp0 * tmp3
    tmp5 = tl.broadcast_to(tmp4, [RBLOCK])
    tmp7 = tl.where(rmask, tmp5, 0)
    tmp8 = triton_helpers.promote_to_tensor(tl.sum(tmp7, 0))
    tmp11 = tmp9 * tmp10
    tmp12 = tmp11.to(tl.float32)
    tmp14 = tmp12 * tmp13
    tmp16 = tmp14 + tmp15
    tmp17 = tmp16 + tmp15
    tmp18 = tmp17.to(tl.float32)
    tmp20 = tmp18 * tmp19
    tmp23 = tmp21 * tmp22
    tmp24 = tmp20 + tmp23
    tmp27 = tmp25 * tmp26
    tmp28 = tmp24 + tmp27
    tmp30 = tmp18 * tmp29
    tmp32 = tmp21 * tmp31
    tmp33 = tmp30 + tmp32
    tmp35 = tmp25 * tmp34
    tmp36 = tmp33 + tmp35
    tmp38 = tmp18 * tmp37
    tmp40 = tmp21 * tmp39
    tmp41 = tmp38 + tmp40
    tmp43 = tmp25 * tmp42
    tmp44 = tmp41 + tmp43
    tmp46 = tmp18 * tmp45
    tmp48 = tmp0 * tmp47
    tmp49 = tmp46 + tmp48
    tmp50 = -0.5
    tmp51 = tmp8 * tmp50
    tmp52 = tmp47 * tmp47
    tmp53 = tmp52 * tmp47
    tmp54 = tmp51 * tmp53
    tmp55 = 1024.0
    tmp56 = tmp54 / tmp55
    tmp57 = 2.0
    tmp58 = tmp3 * tmp57
    tmp59 = tmp56 * tmp58
    tmp60 = tmp49 + tmp59
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp28, rmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp36, rmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp44, rmask)
    tl.store(out_ptr4 + (r1 + (1024*x0)), tmp60, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/hv/chv5aeytaes74nzgcq2wdjglazraed3pchjbbo67v4ozv2oscr2i.py
# Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward, aten.sum]

triton_red_fused_add_select_backward_slice_backward_sum_37 = async_compile.triton('triton_red_fused_add_select_backward_slice_backward_sum_37', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_select_backward_slice_backward_sum_37', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_red_fused_add_select_backward_slice_backward_sum_37(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 768
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 4
    x4 = (xindex // 4)
    x2 = (xindex // 192)
    x1 = (xindex // 4) % 48
    _tmp18 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (4*r3) + (512*x4)), rmask & xmask, other=0).to(tl.float32)
        tmp7 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp13 = tl.load(in_ptr2 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = x2
        tmp2 = tl.full([1, 1], 0, tl.int32)
        tmp3 = tmp1 == tmp2
        tmp4 = x0
        tmp5 = tl.full([1, 1], 1, tl.int32)
        tmp6 = tmp4 == tmp5
        tmp8 = 0.0
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tmp0 + tmp10
        tmp12 = tmp4 == tmp2
        tmp14 = tl.where(tmp12, tmp13, tmp8)
        tmp15 = tl.where(tmp3, tmp14, tmp8)
        tmp16 = tmp11 + tmp15
        tmp17 = tl.broadcast_to(tmp16, [XBLOCK, RBLOCK])
        tmp19 = _tmp18 + tmp17
        _tmp18 = tl.where(rmask & xmask, tmp19, _tmp18)
    tmp18 = tl.sum(_tmp18, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp18, xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/kz/ckzjdmn6eg7i6bbyjty2tkblubkajsgnwo37la3nnbv2cwdxnce5.py
# Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward, aten.sum]

triton_per_fused_add_select_backward_slice_backward_sum_38 = async_compile.triton('triton_per_fused_add_select_backward_slice_backward_sum_38', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_select_backward_slice_backward_sum_38', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2, 3))]}
)
@triton.jit
def triton_per_fused_add_select_backward_slice_backward_sum_38(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 16
    rnumel = 48
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 4
    x1 = (xindex // 4)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (4*r2) + (192*x1)), rmask & xmask, other=0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/4e/c4eo5xwmpdpsfj7huwqte3n4zsfxhgsgajocf6skz7mzy2lyljwk.py
# Source Nodes: [add_19, float_6, mean_3, mul_18, mul_19, rsqrt_3, type_as_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add_19 => add_38
# float_6 => convert_element_type_44
# mean_3 => mean_11
# mul_18 => mul_78
# mul_19 => mul_79
# rsqrt_3 => rsqrt_11
# type_as_5 => convert_element_type_45
triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_39 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_39', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp32', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_39', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_39(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp15 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp5 = tl.load(in_ptr3 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp2 = tmp0 + tmp1
        tmp4 = tmp3.to(tl.float32)
        tmp6 = 1024.0
        tmp7 = tmp5 / tmp6
        tmp8 = 1e-05
        tmp9 = tmp7 + tmp8
        tmp10 = tl.math.rsqrt(tmp9)
        tmp11 = tmp4 * tmp10
        tmp12 = tmp11.to(tl.float32)
        tmp13 = tmp2 * tmp12
        tmp14 = tl.broadcast_to(tmp13, [XBLOCK, RBLOCK])
        tmp16 = _tmp15 + tmp14
        _tmp15 = tl.where(rmask, tmp16, _tmp15)
    tmp15 = tl.sum(_tmp15, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp15, None)
''')


# kernel path: /tmp/torchinductor_mengqy/hi/chi3juvcridn544rjpj3scmafpley5m3yqen7u32piy23s4o42ko.py
# Source Nodes: [add_19, float_6, mean_3, mul_18, rsqrt_3], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add_19 => add_38
# float_6 => convert_element_type_44
# mean_3 => mean_11
# mul_18 => mul_78
# rsqrt_3 => rsqrt_11
triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_40 = async_compile.triton('triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_40', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: 'i32', 7: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_40', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6, 7))]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_40(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, xnumel, rnumel):
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
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp13 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp14 = tl.load(in_ptr4 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp7 = tmp6.to(tl.float32)
    tmp8 = tmp5 * tmp7
    tmp9 = tl.broadcast_to(tmp8, [RBLOCK])
    tmp11 = tl.where(rmask, tmp9, 0)
    tmp12 = triton_helpers.promote_to_tensor(tl.sum(tmp11, 0))
    tmp15 = 1024.0
    tmp16 = tmp14 / tmp15
    tmp17 = 1e-05
    tmp18 = tmp16 + tmp17
    tmp19 = tl.math.rsqrt(tmp18)
    tmp20 = tmp5 * tmp19
    tmp21 = -0.5
    tmp22 = tmp12 * tmp21
    tmp23 = tmp19 * tmp19
    tmp24 = tmp23 * tmp19
    tmp25 = tmp22 * tmp24
    tmp26 = tmp25 / tmp15
    tmp27 = tmp26 * tmp7
    tmp28 = tmp20 + tmp27
    tmp29 = tmp28 + tmp27
    tmp30 = tmp29.to(tl.float32)
    tmp31 = tmp13 + tmp30
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp31, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/zq/czqclywb3ivp4vi66opj2mqtibyv4tdxjc753ninoicy44ltkrui.py
# Source Nodes: [add_14, mean_2, mul_6, mul_7, rsqrt_2, type_as_2], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add_14 => add_34
# mean_2 => mean_10
# mul_6 => mul_66
# mul_7 => mul_67
# rsqrt_2 => rsqrt_10
# type_as_2 => convert_element_type_37
triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_41 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_41', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_41', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_41(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*r2) + (131072*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp2 = tl.load(in_ptr2 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp3 = 1024.0
        tmp4 = tmp2 / tmp3
        tmp5 = 1e-05
        tmp6 = tmp4 + tmp5
        tmp7 = tl.math.rsqrt(tmp6)
        tmp8 = tmp1 * tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp0 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_mengqy/54/c54wsl5h2oiyudmrdw44i2ljr2rjdylfmretfwgxxfm2up4cshth.py
# Source Nodes: [add, add_12, add_14, add_3, add_5, mean, mean_1, mean_2, mul, mul_3, mul_6, rsqrt, rsqrt_1, rsqrt_2], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add => add_32
# add_12 => add_33
# add_14 => add_34
# add_3 => add_3
# add_5 => add_5
# mean => mean_8
# mean_1 => mean_9
# mean_2 => mean_10
# mul => mul_60
# mul_3 => mul_63
# mul_6 => mul_66
# rsqrt => rsqrt_8
# rsqrt_1 => rsqrt_9
# rsqrt_2 => rsqrt_10
triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_42 = async_compile.triton('triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_42', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp32', 3: '*fp32', 4: '*fp16', 5: '*fp16', 6: '*fp32', 7: '*fp32', 8: '*fp16', 9: '*fp16', 10: '*fp32', 11: '*fp32', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: '*fp16', 21: '*fp16', 22: '*fp16', 23: '*fp16', 24: '*fp16', 25: '*fp16', 26: '*fp16', 27: 'i32', 28: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_42', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(27, 28))]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_42(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, out_ptr1, out_ptr3, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, out_ptr10, out_ptr11, out_ptr12, out_ptr13, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0)
    tmp10 = tl.load(in_ptr3 + (x0), None, eviction_policy='evict_last')
    tmp27 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp28 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp31 = tl.load(in_ptr6 + (r1 + (1024*x0)), rmask, other=0)
    tmp37 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp51 = tl.load(in_ptr8 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp52 = tl.load(in_ptr9 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp55 = tl.load(in_ptr10 + (r1 + (1024*x0)), rmask, other=0)
    tmp61 = tl.load(in_ptr11 + (x0), None, eviction_policy='evict_last')
    tmp75 = tl.load(in_ptr12 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp76 = tl.load(in_ptr13 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp77 = tl.load(in_ptr14 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp79 = tl.load(in_ptr15 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp5 = tmp3 * tmp4
    tmp6 = tl.broadcast_to(tmp5, [RBLOCK])
    tmp8 = tl.where(rmask, tmp6, 0)
    tmp9 = triton_helpers.promote_to_tensor(tl.sum(tmp8, 0))
    tmp11 = 1024.0
    tmp12 = tmp10 / tmp11
    tmp13 = 1e-05
    tmp14 = tmp12 + tmp13
    tmp15 = tl.math.rsqrt(tmp14)
    tmp16 = tmp3 * tmp15
    tmp17 = -0.5
    tmp18 = tmp9 * tmp17
    tmp19 = tmp15 * tmp15
    tmp20 = tmp19 * tmp15
    tmp21 = tmp18 * tmp20
    tmp22 = tmp21 / tmp11
    tmp23 = tmp22 * tmp4
    tmp24 = tmp16 + tmp23
    tmp25 = tmp24 + tmp23
    tmp26 = tmp25.to(tl.float32)
    tmp29 = tmp27 * tmp28
    tmp30 = tmp29.to(tl.float32)
    tmp32 = tmp30 * tmp31
    tmp33 = tl.broadcast_to(tmp32, [RBLOCK])
    tmp35 = tl.where(rmask, tmp33, 0)
    tmp36 = triton_helpers.promote_to_tensor(tl.sum(tmp35, 0))
    tmp38 = tmp37 / tmp11
    tmp39 = tmp38 + tmp13
    tmp40 = tl.math.rsqrt(tmp39)
    tmp41 = tmp30 * tmp40
    tmp42 = tmp36 * tmp17
    tmp43 = tmp40 * tmp40
    tmp44 = tmp43 * tmp40
    tmp45 = tmp42 * tmp44
    tmp46 = tmp45 / tmp11
    tmp47 = tmp46 * tmp31
    tmp48 = tmp41 + tmp47
    tmp49 = tmp48 + tmp47
    tmp50 = tmp49.to(tl.float32)
    tmp53 = tmp51 * tmp52
    tmp54 = tmp53.to(tl.float32)
    tmp56 = tmp54 * tmp55
    tmp57 = tl.broadcast_to(tmp56, [RBLOCK])
    tmp59 = tl.where(rmask, tmp57, 0)
    tmp60 = triton_helpers.promote_to_tensor(tl.sum(tmp59, 0))
    tmp62 = tmp61 / tmp11
    tmp63 = tmp62 + tmp13
    tmp64 = tl.math.rsqrt(tmp63)
    tmp65 = tmp54 * tmp64
    tmp66 = tmp60 * tmp17
    tmp67 = tmp64 * tmp64
    tmp68 = tmp67 * tmp64
    tmp69 = tmp66 * tmp68
    tmp70 = tmp69 / tmp11
    tmp71 = tmp70 * tmp55
    tmp72 = tmp65 + tmp71
    tmp73 = tmp72 + tmp71
    tmp74 = tmp73.to(tl.float32)
    tmp78 = tmp76 + tmp77
    tmp80 = tmp78 + tmp79
    tmp81 = tmp75 * tmp80
    tmp82 = tl.broadcast_to(tmp81, [RBLOCK])
    tmp84 = tl.where(rmask, tmp82, 0)
    tmp85 = triton_helpers.promote_to_tensor(tl.sum(tmp84, 0))
    tmp86 = tmp50 * tmp80
    tmp87 = tl.broadcast_to(tmp86, [RBLOCK])
    tmp89 = tl.where(rmask, tmp87, 0)
    tmp90 = triton_helpers.promote_to_tensor(tl.sum(tmp89, 0))
    tmp91 = tmp26 * tmp80
    tmp92 = tl.broadcast_to(tmp91, [RBLOCK])
    tmp94 = tl.where(rmask, tmp92, 0)
    tmp95 = triton_helpers.promote_to_tensor(tl.sum(tmp94, 0))
    tmp96 = tmp74 * tmp80
    tmp97 = tl.broadcast_to(tmp96, [RBLOCK])
    tmp99 = tl.where(rmask, tmp97, 0)
    tmp100 = triton_helpers.promote_to_tensor(tl.sum(tmp99, 0))
    tmp101 = tmp75 * tmp76
    tmp102 = tl.broadcast_to(tmp101, [RBLOCK])
    tmp104 = tl.where(rmask, tmp102, 0)
    tmp105 = triton_helpers.promote_to_tensor(tl.sum(tmp104, 0))
    tmp106 = tmp74 * tmp76
    tmp107 = tl.broadcast_to(tmp106, [RBLOCK])
    tmp109 = tl.where(rmask, tmp107, 0)
    tmp110 = triton_helpers.promote_to_tensor(tl.sum(tmp109, 0))
    tmp111 = tmp50 * tmp76
    tmp112 = tl.broadcast_to(tmp111, [RBLOCK])
    tmp114 = tl.where(rmask, tmp112, 0)
    tmp115 = triton_helpers.promote_to_tensor(tl.sum(tmp114, 0))
    tmp116 = tmp26 * tmp76
    tmp117 = tl.broadcast_to(tmp116, [RBLOCK])
    tmp119 = tl.where(rmask, tmp117, 0)
    tmp120 = triton_helpers.promote_to_tensor(tl.sum(tmp119, 0))
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp26, rmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp50, rmask)
    tl.store(out_ptr5 + (r1 + (1024*x0)), tmp74, rmask)
    tl.store(out_ptr6 + (x0), tmp85, None)
    tl.store(out_ptr7 + (x0), tmp90, None)
    tl.store(out_ptr8 + (x0), tmp95, None)
    tl.store(out_ptr9 + (x0), tmp100, None)
    tl.store(out_ptr10 + (x0), tmp105, None)
    tl.store(out_ptr11 + (x0), tmp110, None)
    tl.store(out_ptr12 + (x0), tmp115, None)
    tl.store(out_ptr13 + (x0), tmp120, None)
''')


# kernel path: /tmp/torchinductor_mengqy/b3/cb3zesln6xau52i4l4q2aovqkk7merl7p6qfvm3leqwmgwow5tpg.py
# Source Nodes: [add_12, mean_1, mul_3, mul_4, rsqrt_1, type_as_1], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add_12 => add_33
# mean_1 => mean_9
# mul_3 => mul_63
# mul_4 => mul_64
# rsqrt_1 => rsqrt_9
# type_as_1 => convert_element_type_35
triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_43 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_43', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp32', 3: '*fp32', 4: 'i32', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_43', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(4, 5))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_43(in_ptr0, in_ptr1, in_ptr2, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp12 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*r2) + (131072*x1)), rmask, other=0)
        tmp2 = tl.load(in_ptr2 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp3 = 1024.0
        tmp4 = tmp2 / tmp3
        tmp5 = 1e-05
        tmp6 = tmp4 + tmp5
        tmp7 = tl.math.rsqrt(tmp6)
        tmp8 = tmp1 * tmp7
        tmp9 = tmp8.to(tl.float32)
        tmp10 = tmp0 * tmp9
        tmp11 = tl.broadcast_to(tmp10, [XBLOCK, RBLOCK])
        tmp13 = _tmp12 + tmp11
        _tmp12 = tl.where(rmask, tmp13, _tmp12)
    tmp12 = tl.sum(_tmp12, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp12, None)
''')


# kernel path: /tmp/torchinductor_mengqy/7s/c7s7cccyt3yshl3tks4ipkmtewvucytq5jvab7eftyaq5mjv26ri.py
# Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward]

triton_poi_fused_add_select_backward_slice_backward_44 = async_compile.triton('triton_poi_fused_add_select_backward_slice_backward_44', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_select_backward_slice_backward_44', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(6,))]})
@triton.jit
def triton_poi_fused_add_select_backward_slice_backward_44(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = (xindex // 18432)
    x0 = xindex % 3
    x1 = (xindex // 3) % 6144
    x3 = xindex
    tmp6 = tl.load(in_ptr0 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr1 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr2 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp22 = tl.load(in_ptr3 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp27 = tl.load(in_ptr4 + (x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x2
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 1, tl.int32)
    tmp5 = tmp3 == tmp4
    tmp7 = 0.0
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tl.where(tmp2, tmp8, tmp7)
    tmp10 = tl.full([1], 0, tl.int32)
    tmp11 = tmp3 == tmp10
    tmp13 = tl.where(tmp11, tmp12, tmp7)
    tmp14 = tl.where(tmp2, tmp13, tmp7)
    tmp15 = tmp9 + tmp14
    tmp16 = tl.full([1], 2, tl.int32)
    tmp17 = tmp0 == tmp16
    tmp19 = tl.where(tmp5, tmp18, tmp7)
    tmp20 = tl.where(tmp17, tmp19, tmp7)
    tmp21 = tmp15 + tmp20
    tmp23 = tl.where(tmp11, tmp22, tmp7)
    tmp24 = tl.where(tmp17, tmp23, tmp7)
    tmp25 = tmp21 + tmp24
    tmp26 = tmp0 == tmp4
    tmp28 = tl.where(tmp5, tmp27, tmp7)
    tmp29 = tl.where(tmp26, tmp28, tmp7)
    tmp30 = tmp25 + tmp29
    tl.store(out_ptr0 + (x3), tmp30, None)
''')


# kernel path: /tmp/torchinductor_mengqy/7n/c7ncfb2knk3wb5ghrz4g52cmua4o53twpv2c3fnxu4zcniha2dol.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_45 = async_compile.triton('triton_poi_fused_clone_45', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_45', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]})
@triton.jit
def triton_poi_fused_clone_45(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex % 3
    x1 = (xindex // 3) % 4
    x2 = (xindex // 12)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3*x2) + (18432*x1)), None).to(tl.float32)
    tmp7 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp14 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp1 = x1
    tmp2 = tl.full([1], 1, tl.int32)
    tmp3 = tmp1 == tmp2
    tmp4 = x0
    tmp5 = tl.full([1], 0, tl.int32)
    tmp6 = tmp4 == tmp5
    tmp8 = 0.0
    tmp9 = tl.where(tmp6, tmp7, tmp8)
    tmp10 = tl.where(tmp3, tmp9, tmp8)
    tmp11 = tmp0 + tmp10
    tmp12 = tmp1 == tmp5
    tmp13 = tmp4 == tmp2
    tmp15 = tl.where(tmp13, tmp14, tmp8)
    tmp16 = tl.where(tmp12, tmp15, tmp8)
    tmp17 = tmp11 + tmp16
    tmp19 = tl.where(tmp6, tmp18, tmp8)
    tmp20 = tl.where(tmp12, tmp19, tmp8)
    tmp21 = tmp17 + tmp20
    tl.store(out_ptr0 + (x3), tmp21, None)
''')


# kernel path: /tmp/torchinductor_mengqy/5b/c5bmo6pxnjm4g3kjb7bqc4iouku2pz7layzt7tp3pzndtqgcrmaf.py
# Source Nodes: [l__mod___dynamic_dense_1_act], Original ATen: [aten.gelu, aten.gelu_backward]
# l__mod___dynamic_dense_1_act => add_22, convert_element_type_30, erf_1, mul_50
triton_poi_fused_gelu_gelu_backward_46 = async_compile.triton('triton_poi_fused_gelu_gelu_backward_46', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[131072], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_46', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_gelu_gelu_backward_46(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 73728
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = tl.math.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = 0.3989422804014327
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tmp1 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp20, None)
''')


# kernel path: /tmp/torchinductor_mengqy/rw/crworyo2plzzgxoxu3jzorcbnshjc2mj224t5h257ob3kta5imls.py
# Source Nodes: [add_10, add_12, add_5, mul_4], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
# add_10 => add_18
# add_12 => add_20
# add_5 => add_12
# mul_4 => mul_24
triton_per_fused_add_div_mul_pow_sum_47 = async_compile.triton('triton_per_fused_add_div_mul_pow_sum_47', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp16', 20: '*fp16', 21: '*fp16', 22: '*fp16', 23: 'i32', 24: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'in_out_ptr1', 'in_out_ptr2'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_pow_sum_47', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(23, 24))]}
)
@triton.jit
def triton_per_fused_add_div_mul_pow_sum_47(in_out_ptr0, in_out_ptr1, in_out_ptr2, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, in_ptr19, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (2*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp2 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp8 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp15 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp17 = tl.load(in_ptr6 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr7 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp21 = tl.load(in_ptr8 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp24 = tl.load(in_ptr9 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp25 = tl.load(in_ptr10 + (3*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp28 = tl.load(in_ptr11 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp29 = tl.load(in_ptr12 + (3*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp32 = tl.load(in_out_ptr1 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp33 = tl.load(in_ptr13 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp36 = tl.load(in_ptr14 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp39 = tl.load(in_ptr15 + (3*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp42 = tl.load(in_ptr16 + (3*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp45 = tl.load(in_out_ptr2 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp46 = tl.load(in_ptr17 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp49 = tl.load(in_ptr18 + (4*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp52 = tl.load(in_ptr19 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 * tmp2
    tmp4 = 0.0
    tmp5 = tmp3 + tmp4
    tmp7 = tmp5 + tmp6
    tmp9 = tmp7 + tmp8
    tmp10 = tmp0 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp22 = tmp20 * tmp21
    tmp23 = tmp19 + tmp22
    tmp26 = tmp24 * tmp25
    tmp27 = tmp23 + tmp26
    tmp30 = tmp28 * tmp29
    tmp31 = tmp27 + tmp30
    tmp34 = tmp16 * tmp33
    tmp35 = tmp32 + tmp34
    tmp37 = tmp20 * tmp36
    tmp38 = tmp35 + tmp37
    tmp40 = tmp24 * tmp39
    tmp41 = tmp38 + tmp40
    tmp43 = tmp28 * tmp42
    tmp44 = tmp41 + tmp43
    tmp47 = tmp16 * tmp46
    tmp48 = tmp45 + tmp47
    tmp50 = tmp20 * tmp49
    tmp51 = tmp48 + tmp50
    tmp53 = tmp0 * tmp52
    tmp54 = tmp51 + tmp53
    tmp55 = -0.5
    tmp56 = tmp14 * tmp55
    tmp57 = tmp52 * tmp52
    tmp58 = tmp57 * tmp52
    tmp59 = tmp56 * tmp58
    tmp60 = 1024.0
    tmp61 = tmp59 / tmp60
    tmp62 = 2.0
    tmp63 = tmp9 * tmp62
    tmp64 = tmp61 * tmp63
    tmp65 = tmp54 + tmp64
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp31, rmask)
    tl.store(in_out_ptr1 + (r1 + (1024*x0)), tmp44, rmask)
    tl.store(in_out_ptr2 + (r1 + (1024*x0)), tmp65, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/jm/cjmqdkkqo54g44a6fslejqmdpur65ds5a6ak3xmvlyvhivb4yppx.py
# Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward, aten.sum]

triton_red_fused_add_select_backward_slice_backward_sum_48 = async_compile.triton('triton_red_fused_add_select_backward_slice_backward_sum_48', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[1024, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp32', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_select_backward_slice_backward_sum_48', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_red_fused_add_select_backward_slice_backward_sum_48(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 576
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 3
    x4 = (xindex // 3)
    x2 = (xindex // 144)
    x1 = (xindex // 3) % 48
    _tmp23 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x5 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r3 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (3*r3) + (384*x4)), rmask & xmask, other=0).to(tl.float32)
        tmp7 = tl.load(in_ptr1 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp14 = tl.load(in_ptr2 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp18 = tl.load(in_ptr3 + (r3 + (128*x1)), rmask & xmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp1 = x2
        tmp2 = tl.full([1, 1], 1, tl.int32)
        tmp3 = tmp1 == tmp2
        tmp4 = x0
        tmp5 = tl.full([1, 1], 0, tl.int32)
        tmp6 = tmp4 == tmp5
        tmp8 = 0.0
        tmp9 = tl.where(tmp6, tmp7, tmp8)
        tmp10 = tl.where(tmp3, tmp9, tmp8)
        tmp11 = tmp0 + tmp10
        tmp12 = tmp1 == tmp5
        tmp13 = tmp4 == tmp2
        tmp15 = tl.where(tmp13, tmp14, tmp8)
        tmp16 = tl.where(tmp12, tmp15, tmp8)
        tmp17 = tmp11 + tmp16
        tmp19 = tl.where(tmp6, tmp18, tmp8)
        tmp20 = tl.where(tmp12, tmp19, tmp8)
        tmp21 = tmp17 + tmp20
        tmp22 = tl.broadcast_to(tmp21, [XBLOCK, RBLOCK])
        tmp24 = _tmp23 + tmp22
        _tmp23 = tl.where(rmask & xmask, tmp24, _tmp23)
    tmp23 = tl.sum(_tmp23, 1)[:, None]
    tl.store(out_ptr0 + (x5), tmp23, xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/ox/coxogny3ilrskou4jtf2ftldbq442cj53u6eb5jel25gheb3aliq.py
# Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward, aten.sum]

triton_per_fused_add_select_backward_slice_backward_sum_49 = async_compile.triton('triton_per_fused_add_select_backward_slice_backward_sum_49', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, persistent_reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@persistent_reduction(
    size_hints=[16, 64],
    reduction_hint=ReductionHint.OUTER_TINY,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp32', 2: 'i32', 3: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_select_backward_slice_backward_sum_49', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 3), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(3,))]}
)
@triton.jit
def triton_per_fused_add_select_backward_slice_backward_sum_49(in_out_ptr0, in_ptr0, xnumel, rnumel, XBLOCK : tl.constexpr):
    xnumel = 12
    rnumel = 48
    RBLOCK: tl.constexpr = 64
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rindex = tl.arange(0, RBLOCK)[None, :]
    rmask = rindex < rnumel
    r2 = rindex
    x0 = xindex % 3
    x1 = (xindex // 3)
    x3 = xindex
    tmp0 = tl.load(in_ptr0 + (x0 + (3*r2) + (144*x1)), rmask & xmask, other=0)
    tmp1 = tl.broadcast_to(tmp0, [XBLOCK, RBLOCK])
    tmp3 = tl.where(rmask & xmask, tmp1, 0)
    tmp4 = tl.sum(tmp3, 1)[:, None]
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp4, xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/ie/ciebr3qshoyjmqkkhuxtvvvm5r554a44mbq746537im6j6fexodf.py
# Source Nodes: [add_10, add_11, add_5, float_6, mean_3, mul_18, mul_4, rsqrt_3], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add_10 => add_18
# add_11 => add_19
# add_5 => add_12
# float_6 => convert_element_type_26
# mean_3 => mean_6
# mul_18 => mul_43
# mul_4 => mul_24
# rsqrt_3 => rsqrt_6
triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_50 = async_compile.triton('triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_50', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp32', 8: 'i32', 9: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_50', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(8, 9))]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_50(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, xnumel, rnumel):
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
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (2*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp11 = tl.load(in_ptr5 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp21 = tl.load(in_ptr6 + (x0), None, eviction_policy='evict_last')
    tmp32 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = tmp6 * tmp7
    tmp9 = 0.0
    tmp10 = tmp8 + tmp9
    tmp12 = tmp10 + tmp11
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp5 * tmp13
    tmp15 = tl.broadcast_to(tmp14, [RBLOCK])
    tmp17 = tl.where(rmask, tmp15, 0)
    tmp18 = triton_helpers.promote_to_tensor(tl.sum(tmp17, 0))
    tmp19 = -0.5
    tmp20 = tmp18 * tmp19
    tmp22 = 1024.0
    tmp23 = tmp21 / tmp22
    tmp24 = 1e-05
    tmp25 = tmp23 + tmp24
    tmp26 = tl.math.rsqrt(tmp25)
    tmp27 = tmp26 * tmp26
    tmp28 = tmp27 * tmp26
    tmp29 = tmp20 * tmp28
    tmp30 = tmp29 / tmp22
    tmp31 = tmp30 * tmp13
    tmp33 = tmp5 * tmp26
    tmp34 = tmp33 + tmp31
    tmp35 = tmp34 + tmp31
    tmp36 = tmp35.to(tl.float32)
    tmp37 = tmp32 + tmp36
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp37, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/ir/cirlphky4r6s3hljewoq7spsd6dqpbeezn73hcx7boy2opah63rd.py
# Source Nodes: [add, add_2, add_3, add_4, add_6, float_1, float_2, float_3, mean, mean_1, mean_2, mul, mul_1, mul_2, mul_3, mul_6, rsqrt, rsqrt_1, rsqrt_2], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
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
# mul_1 => mul_21
# mul_2 => mul_22
# mul_3 => mul_23, mul_28
# mul_6 => mul_31
# rsqrt => rsqrt_3
# rsqrt_1 => rsqrt_4
# rsqrt_2 => rsqrt_5
triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_51 = async_compile.triton('triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_51', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp32', 9: '*fp16', 10: '*fp16', 11: '*fp16', 12: '*fp16', 13: '*fp16', 14: '*fp16', 15: '*fp32', 16: '*fp16', 17: '*fp16', 18: '*fp16', 19: '*fp32', 20: '*fp32', 21: '*fp16', 22: '*fp16', 23: '*fp32', 24: '*fp16', 25: '*fp16', 26: 'i32', 27: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_51', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(26, 27))]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_51(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, out_ptr3, out_ptr5, out_ptr6, out_ptr7, out_ptr8, out_ptr9, xnumel, rnumel):
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
    tmp1 = tl.load(in_ptr1 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp4 = tl.load(in_ptr2 + (2*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp5 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp15 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp16 = tl.load(in_ptr5 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp19 = tl.load(in_ptr6 + (2*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp28 = tl.load(in_ptr7 + (x0), None, eviction_policy='evict_last')
    tmp44 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp45 = tl.load(in_ptr8 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp46 = tl.load(in_ptr9 + (3*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp49 = tl.load(in_ptr10 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp50 = tl.load(in_ptr11 + (3*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp53 = tl.load(in_ptr12 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp54 = tl.load(in_ptr13 + (2*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp60 = tl.load(in_ptr14 + (x0), None, eviction_policy='evict_last')
    tmp73 = tl.load(in_ptr15 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp74 = tl.load(in_ptr16 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp77 = tl.load(in_ptr17 + (2*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp96 = tl.load(in_ptr18 + (x0), None, eviction_policy='evict_last')
    tmp2 = tmp0 * tmp1
    tmp3 = tmp2.to(tl.float32)
    tmp6 = tmp4 * tmp5
    tmp7 = 0.0
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp3 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp17 = tmp15 * tmp16
    tmp18 = tmp17.to(tl.float32)
    tmp20 = tmp19 * tmp5
    tmp21 = tmp20 + tmp7
    tmp22 = tmp21.to(tl.float32)
    tmp23 = tmp18 * tmp22
    tmp24 = tl.broadcast_to(tmp23, [RBLOCK])
    tmp26 = tl.where(rmask, tmp24, 0)
    tmp27 = triton_helpers.promote_to_tensor(tl.sum(tmp26, 0))
    tmp29 = 1024.0
    tmp30 = tmp28 / tmp29
    tmp31 = 1e-05
    tmp32 = tmp30 + tmp31
    tmp33 = tl.math.rsqrt(tmp32)
    tmp34 = tmp3 * tmp33
    tmp35 = -0.5
    tmp36 = tmp14 * tmp35
    tmp37 = tmp33 * tmp33
    tmp38 = tmp37 * tmp33
    tmp39 = tmp36 * tmp38
    tmp40 = tmp39 / tmp29
    tmp41 = tmp40 * tmp9
    tmp42 = tmp34 + tmp41
    tmp43 = tmp42 + tmp41
    tmp47 = tmp45 * tmp46
    tmp48 = tmp44 + tmp47
    tmp51 = tmp49 * tmp50
    tmp52 = tmp48 + tmp51
    tmp55 = tmp53 * tmp54
    tmp56 = tmp52 + tmp55
    tmp57 = tmp43.to(tl.float32)
    tmp58 = tmp57 * tmp4
    tmp59 = tmp56 + tmp58
    tmp61 = tmp60 / tmp29
    tmp62 = tmp61 + tmp31
    tmp63 = tl.math.rsqrt(tmp62)
    tmp64 = tmp18 * tmp63
    tmp65 = tmp27 * tmp35
    tmp66 = tmp63 * tmp63
    tmp67 = tmp66 * tmp63
    tmp68 = tmp65 * tmp67
    tmp69 = tmp68 / tmp29
    tmp70 = tmp69 * tmp22
    tmp71 = tmp64 + tmp70
    tmp72 = tmp71 + tmp70
    tmp75 = tmp73 * tmp74
    tmp76 = tmp75.to(tl.float32)
    tmp78 = tmp77 * tmp5
    tmp79 = tmp78 + tmp7
    tmp80 = tmp79.to(tl.float32)
    tmp81 = tmp76 * tmp80
    tmp82 = tl.broadcast_to(tmp81, [RBLOCK])
    tmp84 = tl.where(rmask, tmp82, 0)
    tmp85 = triton_helpers.promote_to_tensor(tl.sum(tmp84, 0))
    tmp86 = tmp53 * tmp5
    tmp87 = tl.broadcast_to(tmp86, [RBLOCK])
    tmp89 = tl.where(rmask, tmp87, 0)
    tmp90 = triton_helpers.promote_to_tensor(tl.sum(tmp89, 0))
    tmp91 = tmp57 * tmp5
    tmp92 = tl.broadcast_to(tmp91, [RBLOCK])
    tmp94 = tl.where(rmask, tmp92, 0)
    tmp95 = triton_helpers.promote_to_tensor(tl.sum(tmp94, 0))
    tmp97 = tmp96 / tmp29
    tmp98 = tmp97 + tmp31
    tmp99 = tl.math.rsqrt(tmp98)
    tmp100 = tmp76 * tmp99
    tmp101 = tmp85 * tmp35
    tmp102 = tmp99 * tmp99
    tmp103 = tmp102 * tmp99
    tmp104 = tmp101 * tmp103
    tmp105 = tmp104 / tmp29
    tmp106 = tmp105 * tmp80
    tmp107 = tmp100 + tmp106
    tmp108 = tmp107 + tmp106
    tmp109 = tmp72.to(tl.float32)
    tmp110 = tmp109 * tmp5
    tmp111 = tl.broadcast_to(tmp110, [RBLOCK])
    tmp113 = tl.where(rmask, tmp111, 0)
    tmp114 = triton_helpers.promote_to_tensor(tl.sum(tmp113, 0))
    tmp115 = tmp108.to(tl.float32)
    tmp116 = tmp115 * tmp5
    tmp117 = tl.broadcast_to(tmp116, [RBLOCK])
    tmp119 = tl.where(rmask, tmp117, 0)
    tmp120 = triton_helpers.promote_to_tensor(tl.sum(tmp119, 0))
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp59, rmask)
    tl.store(out_ptr3 + (r1 + (1024*x0)), tmp72, rmask)
    tl.store(out_ptr7 + (r1 + (1024*x0)), tmp108, rmask)
    tl.store(out_ptr5 + (x0), tmp90, None)
    tl.store(out_ptr6 + (x0), tmp95, None)
    tl.store(out_ptr8 + (x0), tmp114, None)
    tl.store(out_ptr9 + (x0), tmp120, None)
''')


# kernel path: /tmp/torchinductor_mengqy/q4/cq4pb4o6aedfqeyucfvw5xewygceiolp5gzvjyy6wlpxrnjdstmh.py
# Source Nodes: [], Original ATen: [aten.clone]

triton_poi_fused_clone_52 = async_compile.triton('triton_poi_fused_clone_52', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_clone_52', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5,))]})
@triton.jit
def triton_poi_fused_clone_52(in_ptr0, in_ptr1, in_ptr2, in_ptr3, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x1 = (xindex // 2) % 4
    x0 = xindex % 2
    x2 = (xindex // 8)
    x3 = xindex
    tmp6 = tl.load(in_ptr0 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp12 = tl.load(in_ptr1 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp18 = tl.load(in_ptr2 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp23 = tl.load(in_ptr3 + (x2), None, eviction_policy='evict_last').to(tl.float32)
    tmp0 = x1
    tmp1 = tl.full([1], 3, tl.int32)
    tmp2 = tmp0 == tmp1
    tmp3 = x0
    tmp4 = tl.full([1], 0, tl.int32)
    tmp5 = tmp3 == tmp4
    tmp7 = 0.0
    tmp8 = tl.where(tmp5, tmp6, tmp7)
    tmp9 = tl.where(tmp2, tmp8, tmp7)
    tmp10 = tl.full([1], 2, tl.int32)
    tmp11 = tmp0 == tmp10
    tmp13 = tl.where(tmp5, tmp12, tmp7)
    tmp14 = tl.where(tmp11, tmp13, tmp7)
    tmp15 = tmp9 + tmp14
    tmp16 = tl.full([1], 1, tl.int32)
    tmp17 = tmp0 == tmp16
    tmp19 = tl.where(tmp5, tmp18, tmp7)
    tmp20 = tl.where(tmp17, tmp19, tmp7)
    tmp21 = tmp15 + tmp20
    tmp22 = tmp0 == tmp4
    tmp24 = tl.where(tmp5, tmp23, tmp7)
    tmp25 = tl.where(tmp22, tmp24, tmp7)
    tmp26 = tmp21 + tmp25
    tl.store(out_ptr0 + (x3), tmp26, None)
''')


# kernel path: /tmp/torchinductor_mengqy/l7/cl75c3paoamop3d4e2xhosx64hzh7neu66wuevn3xhlvi4k6ngsz.py
# Source Nodes: [l__mod___dynamic_dense_0_act], Original ATen: [aten.gelu, aten.gelu_backward]
# l__mod___dynamic_dense_0_act => add_7, convert_element_type_12, erf, mul_19
triton_poi_fused_gelu_gelu_backward_53 = async_compile.triton('triton_poi_fused_gelu_gelu_backward_53', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[65536], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_gelu_gelu_backward_53', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_gelu_gelu_backward_53(in_out_ptr0, in_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 49152
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_out_ptr0 + (x0), None).to(tl.float32)
    tmp2 = tl.load(in_ptr0 + (x0), None).to(tl.float32)
    tmp1 = tmp0.to(tl.float32)
    tmp3 = tmp2.to(tl.float32)
    tmp4 = 0.7071067811865476
    tmp5 = tmp3 * tmp4
    tmp6 = tl.math.erf(tmp5)
    tmp7 = 1.0
    tmp8 = tmp6 + tmp7
    tmp9 = 0.5
    tmp10 = tmp8 * tmp9
    tmp11 = tmp3 * tmp3
    tmp12 = -0.5
    tmp13 = tmp11 * tmp12
    tmp14 = tl.exp(tmp13)
    tmp15 = 0.3989422804014327
    tmp16 = tmp14 * tmp15
    tmp17 = tmp3 * tmp16
    tmp18 = tmp10 + tmp17
    tmp19 = tmp1 * tmp18
    tmp20 = tmp19.to(tl.float32)
    tl.store(in_out_ptr0 + (x0), tmp20, None)
''')


# kernel path: /tmp/torchinductor_mengqy/ul/culx7xlm7zqkdd25edtuvxzgqyxbq7e2w76knsdnf6lzltai4dsl.py
# Source Nodes: [add_3, add_5], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
# add_3 => add_3
# add_5 => add_5
triton_per_fused_add_div_mul_pow_sum_54 = async_compile.triton('triton_per_fused_add_div_mul_pow_sum_54', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp16', 6: '*fp16', 7: '*fp16', 8: '*fp16', 9: '*fp16', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused_add_div_mul_pow_sum_54', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(10, 11))]}
)
@triton.jit
def triton_per_fused_add_div_mul_pow_sum_54(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, xnumel, rnumel):
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
    tmp2 = tl.load(in_ptr2 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp4 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp11 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp12 = tl.load(in_out_ptr0 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp13 = tl.load(in_ptr5 + (3*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp16 = tl.load(in_ptr6 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp17 = tl.load(in_ptr7 + (3*x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp20 = tl.load(in_ptr8 + (x0), None, eviction_policy='evict_last').to(tl.float32)
    tmp3 = tmp1 + tmp2
    tmp5 = tmp3 + tmp4
    tmp6 = tmp0 * tmp5
    tmp7 = tl.broadcast_to(tmp6, [RBLOCK])
    tmp9 = tl.where(rmask, tmp7, 0)
    tmp10 = triton_helpers.promote_to_tensor(tl.sum(tmp9, 0))
    tmp14 = tmp12 * tmp13
    tmp15 = tmp11 + tmp14
    tmp18 = tmp16 * tmp17
    tmp19 = tmp15 + tmp18
    tmp21 = tmp0 * tmp20
    tmp22 = tmp19 + tmp21
    tmp23 = -0.5
    tmp24 = tmp10 * tmp23
    tmp25 = tmp20 * tmp20
    tmp26 = tmp25 * tmp20
    tmp27 = tmp24 * tmp26
    tmp28 = 1024.0
    tmp29 = tmp27 / tmp28
    tmp30 = 2.0
    tmp31 = tmp5 * tmp30
    tmp32 = tmp29 * tmp31
    tmp33 = tmp22 + tmp32
    tl.store(in_out_ptr0 + (r1 + (1024*x0)), tmp33, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/rj/crjmufsdad7gcfbimdt5d63usebwvd5lejwr765vjupyf6vy7d5e.py
# Source Nodes: [add, add_10, add_11, add_2, add_3, add_4, add_5, add_6, float_1, float_2, float_3, float_4, float_6, mean, mean_1, mean_2, mean_3, mul, mul_1, mul_12, mul_13, mul_18, mul_19, mul_2, mul_3, mul_4, mul_6, mul_7, rsqrt, rsqrt_1, rsqrt_2, rsqrt_3, type_as, type_as_1, type_as_2, type_as_3, type_as_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add => add_13
# add_10 => add_18
# add_11 => add_19
# add_2 => add_9
# add_3 => add_10, add_3
# add_4 => add_11, add_14, add_4
# add_5 => add_12
# add_6 => add_15
# float_1 => convert_element_type_14
# float_2 => convert_element_type_16
# float_3 => convert_element_type_18
# float_4 => convert_element_type_8
# float_6 => convert_element_type_26
# mean => mean_3
# mean_1 => mean_1, mean_4
# mean_2 => mean_5
# mean_3 => mean_6
# mul => mul_25
# mul_1 => mul_21, mul_26
# mul_12 => mul_12
# mul_13 => mul_13
# mul_18 => mul_43
# mul_19 => mul_44
# mul_2 => mul_22
# mul_3 => mul_23, mul_28
# mul_4 => mul_24, mul_29
# mul_6 => mul_31
# mul_7 => mul_32
# rsqrt => rsqrt_3
# rsqrt_1 => rsqrt_1, rsqrt_4
# rsqrt_2 => rsqrt_5
# rsqrt_3 => rsqrt_6
# type_as => convert_element_type_15
# type_as_1 => convert_element_type_17
# type_as_2 => convert_element_type_19
# type_as_3 => convert_element_type_9
# type_as_5 => convert_element_type_27
triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_55 = async_compile.triton('triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_55', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[65536, 128],
    reduction_hint=ReductionHint.OUTER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp16', 7: '*fp16', 8: '*fp32', 9: '*fp16', 10: '*fp16', 11: '*fp32', 12: '*fp16', 13: '*fp16', 14: '*fp32', 15: '*fp16', 16: '*fp16', 17: '*fp16', 18: '*fp32', 19: '*fp32', 20: '*fp32', 21: '*fp32', 22: '*fp32', 23: '*fp32', 24: 'i32', 25: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_55', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(24, 25))]}
)
@triton.jit
def triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_55(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, in_ptr10, in_ptr11, in_ptr12, in_ptr13, in_ptr14, in_ptr15, in_ptr16, in_ptr17, in_ptr18, out_ptr0, out_ptr1, out_ptr2, out_ptr3, out_ptr4, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 49152
    rnumel = 128
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x0 = xindex % 1024
    x1 = (xindex // 1024)
    _tmp21 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    _tmp36 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp51 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp66 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    _tmp82 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp0 = tl.load(in_ptr0 + (x0 + (1024*r2) + (131072*x1)), rmask, other=0).to(tl.float32)
        tmp1 = tl.load(in_ptr1 + (x0 + (1024*r2) + (131072*x1)), rmask, other=0).to(tl.float32)
        tmp3 = tl.load(in_ptr2 + ((2*r2) + (256*x1)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp4 = tl.load(in_ptr3 + (x0 + (1024*r2) + (131072*x1)), rmask, other=0).to(tl.float32)
        tmp8 = tl.load(in_ptr4 + (x0 + (1024*r2) + (131072*x1)), rmask, other=0).to(tl.float32)
        tmp11 = tl.load(in_ptr5 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp23 = tl.load(in_ptr6 + (x0 + (1024*r2) + (131072*x1)), rmask, other=0).to(tl.float32)
        tmp24 = tl.load(in_ptr7 + ((2*r2) + (256*x1)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp28 = tl.load(in_ptr8 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp38 = tl.load(in_ptr9 + (x0 + (1024*r2) + (131072*x1)), rmask, other=0).to(tl.float32)
        tmp39 = tl.load(in_ptr10 + ((2*r2) + (256*x1)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp43 = tl.load(in_ptr11 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp53 = tl.load(in_ptr12 + (x0 + (1024*r2) + (131072*x1)), rmask, other=0).to(tl.float32)
        tmp54 = tl.load(in_ptr13 + ((2*r2) + (256*x1)), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp58 = tl.load(in_ptr14 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp68 = tl.load(in_ptr15 + (x0 + (1024*r2) + (131072*x1)), rmask, other=0).to(tl.float32)
        tmp69 = tl.load(in_ptr16 + (x0 + (1024*r2) + (131072*x1)), rmask, other=0).to(tl.float32)
        tmp71 = tl.load(in_ptr17 + (x0 + (1024*r2) + (131072*x1)), rmask, other=0).to(tl.float32)
        tmp74 = tl.load(in_ptr18 + (r2 + (128*x1)), rmask, eviction_policy='evict_last', other=0)
        tmp2 = tmp0 + tmp1
        tmp5 = tmp3 * tmp4
        tmp6 = 0.0
        tmp7 = tmp5 + tmp6
        tmp9 = tmp7 + tmp8
        tmp10 = tmp9.to(tl.float32)
        tmp12 = 1024.0
        tmp13 = tmp11 / tmp12
        tmp14 = 1e-05
        tmp15 = tmp13 + tmp14
        tmp16 = tl.math.rsqrt(tmp15)
        tmp17 = tmp10 * tmp16
        tmp18 = tmp17.to(tl.float32)
        tmp19 = tmp2 * tmp18
        tmp20 = tl.broadcast_to(tmp19, [XBLOCK, RBLOCK])
        tmp22 = _tmp21 + tmp20
        _tmp21 = tl.where(rmask, tmp22, _tmp21)
        tmp25 = tmp24 * tmp4
        tmp26 = tmp25 + tmp6
        tmp27 = tmp26.to(tl.float32)
        tmp29 = tmp28 / tmp12
        tmp30 = tmp29 + tmp14
        tmp31 = tl.math.rsqrt(tmp30)
        tmp32 = tmp27 * tmp31
        tmp33 = tmp32.to(tl.float32)
        tmp34 = tmp23 * tmp33
        tmp35 = tl.broadcast_to(tmp34, [XBLOCK, RBLOCK])
        tmp37 = _tmp36 + tmp35
        _tmp36 = tl.where(rmask, tmp37, _tmp36)
        tmp40 = tmp39 * tmp4
        tmp41 = tmp40 + tmp6
        tmp42 = tmp41.to(tl.float32)
        tmp44 = tmp43 / tmp12
        tmp45 = tmp44 + tmp14
        tmp46 = tl.math.rsqrt(tmp45)
        tmp47 = tmp42 * tmp46
        tmp48 = tmp47.to(tl.float32)
        tmp49 = tmp38 * tmp48
        tmp50 = tl.broadcast_to(tmp49, [XBLOCK, RBLOCK])
        tmp52 = _tmp51 + tmp50
        _tmp51 = tl.where(rmask, tmp52, _tmp51)
        tmp55 = tmp54 * tmp4
        tmp56 = tmp55 + tmp6
        tmp57 = tmp56.to(tl.float32)
        tmp59 = tmp58 / tmp12
        tmp60 = tmp59 + tmp14
        tmp61 = tl.math.rsqrt(tmp60)
        tmp62 = tmp57 * tmp61
        tmp63 = tmp62.to(tl.float32)
        tmp64 = tmp53 * tmp63
        tmp65 = tl.broadcast_to(tmp64, [XBLOCK, RBLOCK])
        tmp67 = _tmp66 + tmp65
        _tmp66 = tl.where(rmask, tmp67, _tmp66)
        tmp70 = tmp68 + tmp69
        tmp72 = tmp4 + tmp71
        tmp73 = tmp72.to(tl.float32)
        tmp75 = tmp74 / tmp12
        tmp76 = tmp75 + tmp14
        tmp77 = tl.math.rsqrt(tmp76)
        tmp78 = tmp73 * tmp77
        tmp79 = tmp78.to(tl.float32)
        tmp80 = tmp70 * tmp79
        tmp81 = tl.broadcast_to(tmp80, [XBLOCK, RBLOCK])
        tmp83 = _tmp82 + tmp81
        _tmp82 = tl.where(rmask, tmp83, _tmp82)
    tmp21 = tl.sum(_tmp21, 1)[:, None]
    tl.store(out_ptr0 + (x3), tmp21, None)
    tmp36 = tl.sum(_tmp36, 1)[:, None]
    tl.store(out_ptr1 + (x3), tmp36, None)
    tmp51 = tl.sum(_tmp51, 1)[:, None]
    tl.store(out_ptr2 + (x3), tmp51, None)
    tmp66 = tl.sum(_tmp66, 1)[:, None]
    tl.store(out_ptr3 + (x3), tmp66, None)
    tmp82 = tl.sum(_tmp82, 1)[:, None]
    tl.store(out_ptr4 + (x3), tmp82, None)
''')


# kernel path: /tmp/torchinductor_mengqy/aa/caawwybkfmg2bj3bsnkjj3x4xstecrjf46xmyozstc4akozebv4u.py
# Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward, aten.sum]

triton_red_fused_add_select_backward_slice_backward_sum_56 = async_compile.triton('triton_red_fused_add_select_backward_slice_backward_sum_56', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, reduction
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@reduction(
    size_hints=[8, 8192],
    reduction_hint=ReductionHint.INNER,
    filename=__file__,
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: 'i32', 6: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_red_fused_add_select_backward_slice_backward_sum_56', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 6), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(5, 6))]}
)
@triton.jit
def triton_red_fused_add_select_backward_slice_backward_sum_56(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, xnumel, rnumel, XBLOCK : tl.constexpr, RBLOCK : tl.constexpr):
    xnumel = 8
    rnumel = 6144
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:, None]
    xmask = xindex < xnumel
    rbase = tl.arange(0, RBLOCK)[None, :]
    x1 = (xindex // 2)
    x0 = xindex % 2
    _tmp28 = tl.full([XBLOCK, RBLOCK], 0, tl.float32)
    x3 = xindex
    for roffset in range(0, rnumel, RBLOCK):
        rindex = roffset + rbase
        rmask = rindex < rnumel
        r2 = rindex
        tmp6 = tl.load(in_ptr0 + (r2), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp12 = tl.load(in_ptr1 + (r2), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp18 = tl.load(in_ptr2 + (r2), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp23 = tl.load(in_ptr3 + (r2), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
        tmp0 = x1
        tmp1 = tl.full([1, 1], 3, tl.int32)
        tmp2 = tmp0 == tmp1
        tmp3 = x0
        tmp4 = tl.full([1, 1], 0, tl.int32)
        tmp5 = tmp3 == tmp4
        tmp7 = 0.0
        tmp8 = tl.where(tmp5, tmp6, tmp7)
        tmp9 = tl.where(tmp2, tmp8, tmp7)
        tmp10 = tl.full([1, 1], 2, tl.int32)
        tmp11 = tmp0 == tmp10
        tmp13 = tl.where(tmp5, tmp12, tmp7)
        tmp14 = tl.where(tmp11, tmp13, tmp7)
        tmp15 = tmp9 + tmp14
        tmp16 = tl.full([1, 1], 1, tl.int32)
        tmp17 = tmp0 == tmp16
        tmp19 = tl.where(tmp5, tmp18, tmp7)
        tmp20 = tl.where(tmp17, tmp19, tmp7)
        tmp21 = tmp15 + tmp20
        tmp22 = tmp0 == tmp4
        tmp24 = tl.where(tmp5, tmp23, tmp7)
        tmp25 = tl.where(tmp22, tmp24, tmp7)
        tmp26 = tmp21 + tmp25
        tmp27 = tl.broadcast_to(tmp26, [XBLOCK, RBLOCK])
        tmp29 = _tmp28 + tmp27
        _tmp28 = tl.where(rmask & xmask, tmp29, _tmp28)
    tmp28 = tl.sum(_tmp28, 1)[:, None]
    tl.debug_barrier()
    tl.store(in_out_ptr0 + (x3), tmp28, xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/5g/c5gls7gkgz4b7lvdwbmtghanb7oofqh3g6o4637qnye4ouqfgtls.py
# Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
# add_3 => add_3
# add_4 => add_4
# float_4 => convert_element_type_8
# mean_1 => mean_1
# mul_12 => mul_12
# rsqrt_1 => rsqrt_1
triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_57 = async_compile.triton('triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_57', '''
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
    meta={'signature': {0: '*fp16', 1: '*fp16', 2: '*fp16', 3: '*fp16', 4: '*fp16', 5: '*fp32', 6: '*fp16', 7: '*fp32', 8: '*fp16', 9: 'i32', 10: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_57', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(9, 10))]}
)
@triton.jit
def triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_57(in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, out_ptr1, out_ptr2, xnumel, rnumel):
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
    tmp3 = tl.load(in_ptr2 + (r1), rmask, eviction_policy='evict_last', other=0).to(tl.float32)
    tmp6 = tl.load(in_ptr3 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp7 = tl.load(in_ptr4 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp15 = tl.load(in_ptr5 + (x0), None, eviction_policy='evict_last')
    tmp31 = tl.load(in_ptr6 + (r1 + (1024*x0)), rmask, other=0).to(tl.float32)
    tmp2 = tmp0 + tmp1
    tmp4 = tmp2 * tmp3
    tmp5 = tmp4.to(tl.float32)
    tmp8 = tmp6 + tmp7
    tmp9 = tmp8.to(tl.float32)
    tmp10 = tmp5 * tmp9
    tmp11 = tl.broadcast_to(tmp10, [RBLOCK])
    tmp13 = tl.where(rmask, tmp11, 0)
    tmp14 = triton_helpers.promote_to_tensor(tl.sum(tmp13, 0))
    tmp16 = 1024.0
    tmp17 = tmp15 / tmp16
    tmp18 = 1e-05
    tmp19 = tmp17 + tmp18
    tmp20 = tl.math.rsqrt(tmp19)
    tmp21 = tmp5 * tmp20
    tmp22 = -0.5
    tmp23 = tmp14 * tmp22
    tmp24 = tmp20 * tmp20
    tmp25 = tmp24 * tmp20
    tmp26 = tmp23 * tmp25
    tmp27 = tmp26 / tmp16
    tmp28 = tmp27 * tmp9
    tmp29 = tmp21 + tmp28
    tmp30 = tmp29 + tmp28
    tmp32 = tmp30.to(tl.float32)
    tmp33 = tmp31 + tmp32
    tl.store(out_ptr1 + (r1 + (1024*x0)), tmp30, rmask)
    tl.store(out_ptr2 + (r1 + (1024*x0)), tmp33, rmask)
''')


# kernel path: /tmp/torchinductor_mengqy/cc/cccvwlgum73fmxborvsc743plokrldyi453vxeimmdnabokddnc7.py
# Source Nodes: [], Original ATen: [aten.add, aten.embedding_dense_backward]

triton_poi_fused_add_embedding_dense_backward_58 = async_compile.triton('triton_poi_fused_add_embedding_dense_backward_58', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_add_embedding_dense_backward_58', 'configs': [instance_descriptor(divisible_by_16=(0, 1), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(1,))]})
@triton.jit
def triton_poi_fused_add_embedding_dense_backward_58(out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51463168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = 0.0
    tl.store(out_ptr0 + (x0), tmp0, xmask)
''')


# kernel path: /tmp/torchinductor_mengqy/6c/c6cgteomz4fdpxtf6y76mj457oijxj4isrbolkyr5d3faf2gjuzq.py
# Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.embedding_dense_backward, aten.mul]

triton_poi_fused__to_copy_add_embedding_dense_backward_mul_59 = async_compile.triton('triton_poi_fused__to_copy_add_embedding_dense_backward_mul_59', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[8388608], filename=__file__, meta={'signature': {0: '*fp16', 1: '*fp32', 2: '*fp16', 3: '*fp32', 4: '*fp16', 5: '*fp16', 6: '*fp32', 7: '*fp16', 8: '*fp16', 9: '*i64', 10: '*fp16', 11: '*fp32', 12: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': ['in_out_ptr0', 'out_ptr0'], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused__to_copy_add_embedding_dense_backward_mul_59', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(12,))]})
@triton.jit
def triton_poi_fused__to_copy_add_embedding_dense_backward_mul_59(in_out_ptr0, in_ptr0, in_ptr1, in_ptr2, in_ptr3, in_ptr4, in_ptr5, in_ptr6, in_ptr7, in_ptr8, in_ptr9, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 6291456
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x2 = xindex
    x1 = (xindex // 1024)
    x3 = (xindex // 1024) % 2048
    x4 = (xindex // 2097152)
    x0 = xindex % 1024
    tmp0 = tl.load(in_out_ptr0 + (x2), None).to(tl.float32)
    tmp1 = tl.load(in_ptr0 + (x2), None)
    tmp3 = tl.load(in_ptr1 + (2*x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp6 = tl.load(in_ptr2 + (x2), None)
    tmp8 = tl.load(in_ptr3 + (2*x1), None, eviction_policy='evict_last').to(tl.float32)
    tmp11 = tl.load(in_ptr4 + (x2), None).to(tl.float32)
    tmp12 = tl.load(in_ptr5 + (x2), None)
    tmp16 = tl.load(in_ptr6 + (x2), None).to(tl.float32)
    tmp18 = tl.load(in_ptr7 + (x2), None).to(tl.float32)
    tmp20 = tl.load(in_ptr8 + (x3 + (2049*x4)), None, eviction_policy='evict_last')
    tmp24 = tl.load(in_ptr9 + (x2), None).to(tl.float32)
    tmp2 = tmp1.to(tl.float32)
    tmp4 = tmp2 * tmp3
    tmp5 = tmp0 + tmp4
    tmp7 = tmp6.to(tl.float32)
    tmp9 = tmp7 * tmp8
    tmp10 = tmp5 + tmp9
    tmp13 = tmp12.to(tl.float32)
    tmp14 = tmp11 + tmp13
    tmp15 = tmp10 + tmp14
    tmp17 = tmp15 + tmp16
    tmp19 = tmp17 + tmp18
    tmp21 = tl.where(tmp20 < 0, tmp20 + 50257, tmp20)
    tmp22 = tl.full([1], -1, tl.int64)
    tmp23 = tmp20 == tmp22
    tmp25 = tmp19 + tmp24
    tmp26 = tmp25.to(tl.float32)
    tmp27 = 0.0
    tmp28 = tl.where(tmp23, tmp27, tmp26)
    tl.atomic_add(out_ptr0 + (x0 + (1024*tmp21)), tmp28, None)
''')


# kernel path: /tmp/torchinductor_mengqy/ew/cew5etetls5cessewbccmf3lq522ntjc2jlil2pguampeetm6ehm.py
# Source Nodes: [], Original ATen: [aten.embedding_dense_backward]

triton_poi_fused_embedding_dense_backward_60 = async_compile.triton('triton_poi_fused_embedding_dense_backward_60', '''
import triton
import triton.language as tl
from torch._inductor.ir import ReductionHint
from torch._inductor.ir import TileHint
from torch._inductor.triton_heuristics import AutotuneHint, pointwise
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@pointwise(size_hints=[67108864], filename=__file__, meta={'signature': {0: '*fp32', 1: '*fp16', 2: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'triton_poi_fused_embedding_dense_backward_60', 'configs': [instance_descriptor(divisible_by_16=(0, 1, 2), equal_to_1=(), ids_of_folded_args=(), divisible_by_8=(2,))]})
@triton.jit
def triton_poi_fused_embedding_dense_backward_60(in_ptr0, out_ptr0, xnumel, XBLOCK : tl.constexpr):
    xnumel = 51463168
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)[:]
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + (x0), xmask)
    tmp1 = tmp0.to(tl.float32)
    tl.store(out_ptr0 + (x0), tmp1, xmask)
''')


async_compile.wait(globals())
del async_compile

def call(args):
    primals_6, primals_11, primals_12, primals_13, primals_18, primals_23, primals_24, primals_25, primals_30, primals_35, primals_36, primals_37, primals_42, primals_47, primals_60, primals_61, embedding, permute, permute_1, permute_2, select_1, select_3, slice_3, scalar_tensor, permute_10, permute_11, permute_12, permute_13, rsqrt_2, view_30, mm_7, view_32, unsqueeze_8, unsqueeze_9, unsqueeze_10, unsqueeze_11, permute_17, permute_18, permute_19, permute_27, permute_28, permute_29, permute_30, rsqrt_7, view_65, mm_16, view_67, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23, unsqueeze_24, unsqueeze_25, permute_34, permute_35, permute_36, permute_44, permute_45, permute_46, permute_47, rsqrt_12, view_100, mm_25, view_102, unsqueeze_32, unsqueeze_33, unsqueeze_34, unsqueeze_35, unsqueeze_36, unsqueeze_37, unsqueeze_38, unsqueeze_39, unsqueeze_40, unsqueeze_41, unsqueeze_42, unsqueeze_43, permute_51, permute_52, permute_53, permute_61, permute_62, permute_63, permute_64, rsqrt_17, view_135, mm_34, view_137, unsqueeze_54, unsqueeze_55, unsqueeze_56, unsqueeze_57, rsqrt_18, view_140, convert_element_type_71, convert_element_type_72, permute_71, permute_76, permute_80, permute_125, permute_129, permute_174, permute_178, permute_223, permute_227, tangents_1 = args
    args.clear()
    assert_size_stride(primals_6, (1024, ), (1, ))
    assert_size_stride(primals_11, (1024, ), (1, ))
    assert_size_stride(primals_12, (1024, ), (1, ))
    assert_size_stride(primals_13, (1024, ), (1, ))
    assert_size_stride(primals_18, (1024, ), (1, ))
    assert_size_stride(primals_23, (1024, ), (1, ))
    assert_size_stride(primals_24, (1024, ), (1, ))
    assert_size_stride(primals_25, (1024, ), (1, ))
    assert_size_stride(primals_30, (1024, ), (1, ))
    assert_size_stride(primals_35, (1024, ), (1, ))
    assert_size_stride(primals_36, (1024, ), (1, ))
    assert_size_stride(primals_37, (1024, ), (1, ))
    assert_size_stride(primals_42, (1024, ), (1, ))
    assert_size_stride(primals_47, (1024, ), (1, ))
    assert_size_stride(primals_60, (3, 2048), (2049, 1))
    assert_size_stride(primals_61, (3, 2048), (2049, 1))
    assert_size_stride(embedding, (3, 2048, 1024), (2097152, 1024, 1))
    assert_size_stride(permute, (1024, 1024), (1, 1024))
    assert_size_stride(permute_1, (1024, 1024), (1, 1024))
    assert_size_stride(permute_2, (1024, 1024), (1, 1024))
    assert_size_stride(select_1, (1, 2048, 1, 32), (0, 64, 0, 2))
    assert_size_stride(select_3, (1, 2048, 1, 32), (0, 64, 0, 2))
    assert_size_stride(slice_3, (1, 1, 2048, 2048), (4194304, 4194304, 2048, 1))
    assert_size_stride(scalar_tensor, (), ())
    assert_size_stride(permute_10, (1024, 1024), (1, 1024))
    assert_size_stride(permute_11, (1024, 2816), (1, 1024))
    assert_size_stride(permute_12, (1024, 2816), (1, 1024))
    assert_size_stride(permute_13, (2816, 1024), (1, 2816))
    assert_size_stride(rsqrt_2, (3, 2048, 1), (2048, 1, 1))
    assert_size_stride(view_30, (6144, 1024), (1024, 1))
    assert_size_stride(mm_7, (6144, 8), (8, 1))
    assert_size_stride(view_32, (6144, 8), (8, 1))
    assert_size_stride(unsqueeze_8, (3, 2048, 1), (4096, 2, 0))
    assert_size_stride(unsqueeze_9, (3, 2048, 1), (4096, 2, 0))
    assert_size_stride(unsqueeze_10, (3, 2048, 1), (4096, 2, 0))
    assert_size_stride(unsqueeze_11, (3, 2048, 1), (4096, 2, 0))
    assert_size_stride(permute_17, (1024, 1024), (1, 1024))
    assert_size_stride(permute_18, (1024, 1024), (1, 1024))
    assert_size_stride(permute_19, (1024, 1024), (1, 1024))
    assert_size_stride(permute_27, (1024, 1024), (1, 1024))
    assert_size_stride(permute_28, (1024, 2816), (1, 1024))
    assert_size_stride(permute_29, (1024, 2816), (1, 1024))
    assert_size_stride(permute_30, (2816, 1024), (1, 2816))
    assert_size_stride(rsqrt_7, (3, 2048, 1), (2048, 1, 1))
    assert_size_stride(view_65, (6144, 1024), (1024, 1))
    assert_size_stride(mm_16, (6144, 12), (12, 1))
    assert_size_stride(view_67, (6144, 12), (12, 1))
    assert_size_stride(unsqueeze_18, (3, 2048, 1), (6144, 3, 0))
    assert_size_stride(unsqueeze_19, (3, 2048, 1), (6144, 3, 0))
    assert_size_stride(unsqueeze_20, (3, 2048, 1), (6144, 3, 0))
    assert_size_stride(unsqueeze_21, (3, 2048, 1), (6144, 3, 0))
    assert_size_stride(unsqueeze_22, (3, 2048, 1), (6144, 3, 0))
    assert_size_stride(unsqueeze_23, (3, 2048, 1), (6144, 3, 0))
    assert_size_stride(unsqueeze_24, (3, 2048, 1), (6144, 3, 0))
    assert_size_stride(unsqueeze_25, (3, 2048, 1), (6144, 3, 0))
    assert_size_stride(permute_34, (1024, 1024), (1, 1024))
    assert_size_stride(permute_35, (1024, 1024), (1, 1024))
    assert_size_stride(permute_36, (1024, 1024), (1, 1024))
    assert_size_stride(permute_44, (1024, 1024), (1, 1024))
    assert_size_stride(permute_45, (1024, 2816), (1, 1024))
    assert_size_stride(permute_46, (1024, 2816), (1, 1024))
    assert_size_stride(permute_47, (2816, 1024), (1, 2816))
    assert_size_stride(rsqrt_12, (3, 2048, 1), (2048, 1, 1))
    assert_size_stride(view_100, (6144, 1024), (1024, 1))
    assert_size_stride(mm_25, (6144, 16), (16, 1))
    assert_size_stride(view_102, (6144, 16), (16, 1))
    assert_size_stride(unsqueeze_32, (3, 2048, 1), (8192, 4, 0))
    assert_size_stride(unsqueeze_33, (3, 2048, 1), (8192, 4, 0))
    assert_size_stride(unsqueeze_34, (3, 2048, 1), (8192, 4, 0))
    assert_size_stride(unsqueeze_35, (3, 2048, 1), (8192, 4, 0))
    assert_size_stride(unsqueeze_36, (3, 2048, 1), (8192, 4, 0))
    assert_size_stride(unsqueeze_37, (3, 2048, 1), (8192, 4, 0))
    assert_size_stride(unsqueeze_38, (3, 2048, 1), (8192, 4, 0))
    assert_size_stride(unsqueeze_39, (3, 2048, 1), (8192, 4, 0))
    assert_size_stride(unsqueeze_40, (3, 2048, 1), (8192, 4, 0))
    assert_size_stride(unsqueeze_41, (3, 2048, 1), (8192, 4, 0))
    assert_size_stride(unsqueeze_42, (3, 2048, 1), (8192, 4, 0))
    assert_size_stride(unsqueeze_43, (3, 2048, 1), (8192, 4, 0))
    assert_size_stride(permute_51, (1024, 1024), (1, 1024))
    assert_size_stride(permute_52, (1024, 1024), (1, 1024))
    assert_size_stride(permute_53, (1024, 1024), (1, 1024))
    assert_size_stride(permute_61, (1024, 1024), (1, 1024))
    assert_size_stride(permute_62, (1024, 2816), (1, 1024))
    assert_size_stride(permute_63, (1024, 2816), (1, 1024))
    assert_size_stride(permute_64, (2816, 1024), (1, 2816))
    assert_size_stride(rsqrt_17, (3, 2048, 1), (2048, 1, 1))
    assert_size_stride(view_135, (6144, 1024), (1024, 1))
    assert_size_stride(mm_34, (6144, 20), (20, 1))
    assert_size_stride(view_137, (6144, 20), (20, 1))
    assert_size_stride(unsqueeze_54, (3, 2048, 1), (10240, 5, 0))
    assert_size_stride(unsqueeze_55, (3, 2048, 1), (10240, 5, 0))
    assert_size_stride(unsqueeze_56, (3, 2048, 1), (10240, 5, 0))
    assert_size_stride(unsqueeze_57, (3, 2048, 1), (10240, 5, 0))
    assert_size_stride(rsqrt_18, (3, 2048, 1), (2048, 1, 1))
    assert_size_stride(view_140, (6144, 1024), (1024, 1)) # (BT)D before lm_head and after rmsnorm
    assert_size_stride(convert_element_type_71, (6144, 50257), (50257, 1)) # logits : (BT)E
    assert_size_stride(convert_element_type_72, (), ())
    assert_size_stride(permute_71, (50257, 1024), (1024, 1)) # lm_head
    assert_size_stride(permute_76, (20, 20), (20, 1))
    assert_size_stride(permute_80, (20, 1024), (1024, 1))
    assert_size_stride(permute_125, (16, 16), (16, 1))
    assert_size_stride(permute_129, (16, 1024), (1024, 1))
    assert_size_stride(permute_174, (12, 12), (12, 1))
    assert_size_stride(permute_178, (12, 1024), (1024, 1))
    assert_size_stride(permute_223, (8, 8), (8, 1))
    assert_size_stride(permute_227, (8, 1024), (1024, 1))
    assert_size_stride(tangents_1, (), ())
    with torch.cuda._DeviceGuard(0):
        torch.cuda.set_device(0) # no-op to ensure context
        buf0 = empty_strided((6144, 50257), (50257, 1), device='cuda', dtype=torch.float16) # BTE
        # Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        stream0 = get_cuda_stream(0)
        triton_poi_fused_nll_loss_backward_nll_loss_forward_0.run(buf0, 308779008, grid=grid(308779008), stream=stream0) # set buf0 to 0 
        # Source Nodes: [cross_entropy], Original ATen: [aten.nll_loss_backward, aten.nll_loss_forward]
        triton_poi_fused_nll_loss_backward_nll_loss_forward_1.run(primals_61, buf0, 6144, grid=grid(6144), stream=stream0) # BT, BTE; set buf0(BTE) as -1 where E==labels
        buf3 = empty_strided((6144, 50257), (50257, 1), device='cuda', dtype=torch.float16)
        buf7 = empty_strided((6144, 50264), (50264, 1), device='cuda', dtype=torch.float16)
        buf5 = reinterpret_tensor(buf7, (6144, 50257), (50264, 1), 0)  # alias
        # Source Nodes: [cross_entropy], Original ATen: [aten._log_softmax_backward_data, aten.nll_loss_backward, aten.nll_loss_forward]
        triton_red_fused__log_softmax_backward_data_nll_loss_backward_nll_loss_forward_2.run(buf0, primals_61, tangents_1, convert_element_type_72, convert_element_type_71, buf3, buf5, 6144, 50257, grid=grid(6144), stream=stream0) # buf3 and buf5 contains error : (BT)E
        del buf0
        del convert_element_type_71
        del convert_element_type_72
        del primals_61
        del tangents_1
        buf4 = empty_strided((50257, 1024), (1024, 1), device='cuda', dtype=torch.float16) # lm_head grad: ED 
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf3, (50257, 6144), (1, 50257), 0), view_140, out=buf4) # Err: E(BT), Acts: (BT)D
        del buf3
        del view_140
        buf6 = reinterpret_tensor(buf7, (6144, 7), (50264, 1), 50257)  # alias
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_3.run(buf6, 43008, grid=grid(43008), stream=stream0)
        buf10 = empty_strided((50264, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        buf8 = reinterpret_tensor(buf10, (50257, 1024), (1024, 1), 0)  # alias
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_4.run(permute_71, buf8, 51463168, grid=grid(51463168), stream=stream0)
        del buf5
        del buf6
        del permute_71
        buf9 = reinterpret_tensor(buf10, (7, 1024), (1024, 1), 51463168)  # alias
        # Source Nodes: [], Original ATen: []
        triton_poi_fused_5.run(buf9, 7168, grid=grid(7168), stream=stream0) # set buf9 as 0
        del buf8
        del buf9
        buf11 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16) # (BT)D, err of x after rmsnorm  
        # Source Nodes: [], Original ATen: []
        extern_kernels.mm(buf7, buf10, out=buf11) # Err:(BT)E, lm_head: ED -> (BT)D  
        del buf10
        del buf7
        buf12 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(embedding, (6144, 1024), (1024, 1), 0), permute_1, out=buf12) #q (BT)D, DD->(BT)D
        buf16 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(embedding, (6144, 1024), (1024, 1), 0), permute, out=buf16) #k (BT)D, DD->(BT)D
        buf15 = empty_strided((3, 2048, 16, 32, 2), (2097152, 1024, 64, 2, 1), device='cuda', dtype=torch.float32)
        buf13 = reinterpret_tensor(buf15, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 0)  # alias
        buf14 = reinterpret_tensor(buf15, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 1)  # alias
        buf19 = empty_strided((3, 2048, 16, 32, 2), (2097152, 1024, 64, 2, 1), device='cuda', dtype=torch.float32)
        buf17 = reinterpret_tensor(buf19, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 0)  # alias
        buf18 = reinterpret_tensor(buf19, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_6.run(buf12, select_1, select_3, buf16, buf13, buf14, buf17, buf18, 3145728, grid=grid(3145728), stream=stream0) # rope 
        buf20 = reinterpret_tensor(buf16, (3, 16, 2048, 64), (2097152, 131072, 64, 1)); del buf16  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf19, buf20, 6291456, grid=grid(6291456), stream=stream0)
        del buf13
        del buf14
        del buf17
        del buf18
        buf21 = reinterpret_tensor(buf12, (3, 16, 64, 2048), (2097152, 131072, 2048, 1)); del buf12  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf15, buf21, 3072, 2048, grid=grid(3072, 2048), stream=stream0)
        buf22 = empty_strided((48, 2048, 2048), (4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (48, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf21, (48, 64, 2048), (131072, 2048, 1), 0), out=buf22) # attn_logits q@k: (BN)Td, (BN)dS -> (BN)TS 
        buf25 = empty_strided((3, 16, 2048, 2048), (67108864, 4194304, 2048, 1), device='cuda', dtype=torch.float16) # probs: BNTS
        # Source Nodes: [mul_11, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_9.run(slice_3, buf22, scalar_tensor, buf25, 98304, 2048, grid=grid(98304), stream=stream0) # layer0: attn softmax & mask 
        buf26 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16)  # v: (BT)D
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(embedding, (6144, 1024), (1024, 1), 0), permute_2, out=buf26) # (BT)D, DD->(BT)D
        buf27 = empty_strided((3, 16, 2048, 64), (2097152, 131072, 64, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf26, buf27, 6291456, grid=grid(6291456), stream=stream0) # reshape v
        buf28 = reinterpret_tensor(buf26, (48, 2048, 64), (131072, 64, 1)); del buf26  # reuse mixed_v
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf25, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf27, (48, 2048, 64), (131072, 64, 1), 0), out=buf28) # mixed_v = probs @ v
        buf29 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16) # mixed_v: (BT)D
        # Source Nodes: [linear_3], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf28, buf29, 6291456, grid=grid(6291456), stream=stream0) # reshape mixed_v
        buf30 = reinterpret_tensor(buf28, (6144, 1024), (1024, 1)); del buf28  # reuse attn_out
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf29, permute_10, out=buf30) # attn_out = Wo @ mixed_v
        buf31 = empty_strided((3, 2048, 1), (2048, 1, 6144), device='cuda', dtype=torch.float32)
        buf72 = empty_strided((3, 2048, 1), (2048, 1, 6144), device='cuda', dtype=torch.float32)
        buf78 = empty_strided((3, 2048, 1), (2048, 1, 6144), device='cuda', dtype=torch.float32)
        buf90 = empty_strided((3, 2048, 1), (2048, 1, 6144), device='cuda', dtype=torch.float32)
        buf32 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_2, add_3, add_4, float_1, float_2, float_3, float_4, mean, mean_1, mean_2, mul, mul_1, mul_12, mul_13, mul_14, mul_2, mul_3, mul_6, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_per_fused__to_copy_add_mean_mul_rsqrt_12.run(embedding, buf30, unsqueeze_9, unsqueeze_8, unsqueeze_10, primals_6, buf31, buf72, buf78, buf90, buf32, 6144, 1024, grid=grid(6144), stream=stream0) # out = rmsnorm(emb + attn_out)
        buf33 = empty_strided((6144, 2816), (2816, 1), device='cuda', dtype=torch.float16) # w1_out
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (6144, 1024), (1024, 1), 0), permute_11, out=buf33) # mlp_input: w1 @ x; (BT)DK, DK -> (BT)K
        buf34 = empty_strided((6144, 2816), (2816, 1), device='cuda', dtype=torch.float16) # wg_out
        # Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf32, (6144, 1024), (1024, 1), 0), permute_12, out=buf34) # BTD, DK->BTK
        buf35 = empty_strided((3, 2048, 2816), (5767168, 2816, 1), device='cuda', dtype=torch.float16) # mlp_out
        # Source Nodes: [mul_15, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_13.run(buf33, buf34, buf35, 17301504, grid=grid(17301504), stream=stream0) # silu(w1_out) * wg_out
        buf36 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16) # layer_out BTD
        # Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf35, (6144, 2816), (2816, 1), 0), permute_13, out=buf36)
        buf37 = reinterpret_tensor(buf15, (3, 2048, 1024), (2097152, 1024, 1)); del buf15  # reuse
        buf44 = reinterpret_tensor(buf19, (3, 2048, 1024), (2097152, 1024, 1)); del buf19  # reuse
        buf57 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float32)
        buf38 = empty_strided((3, 2048, 1), (2048, 1, 6144), device='cuda', dtype=torch.float32)
        buf39 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        buf45 = empty_strided((3, 2048, 1), (2048, 1, 6144), device='cuda', dtype=torch.float32)
        buf46 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        buf58 = empty_strided((3, 2048, 1), (2048, 1, 6144), device='cuda', dtype=torch.float32)
        buf59 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_10, add_11, add_12, add_13, add_14, add_3, add_5, add_8, add_9, float_1, float_2, float_3, mean, mean_1, mean_2, mul, mul_1, mul_10, mul_11, mul_2, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, mul_9, rsqrt, rsqrt_1, rsqrt_2, type_as, type_as_1, type_as_2], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_per_fused__to_copy_add_mean_mul_rsqrt_14.run(unsqueeze_20, embedding, unsqueeze_21, buf30, buf36, unsqueeze_18, unsqueeze_19, unsqueeze_22, unsqueeze_23, primals_24, primals_23, primals_25, buf37, buf44, buf57, buf38, buf39, buf45, buf46, buf58, buf59, 6144, 1024, grid=grid(6144), stream=stream0)
        buf40 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16) # layer1:q
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf39, (6144, 1024), (1024, 1), 0), permute_35, out=buf40)
        buf47 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16) # layer1:k
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf46, (6144, 1024), (1024, 1), 0), permute_34, out=buf47)
        buf43 = empty_strided((3, 2048, 16, 32, 2), (2097152, 1024, 64, 2, 1), device='cuda', dtype=torch.float32)
        buf41 = reinterpret_tensor(buf43, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 0)  # alias
        buf42 = reinterpret_tensor(buf43, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 1)  # alias
        buf50 = empty_strided((3, 2048, 16, 32, 2), (2097152, 1024, 64, 2, 1), device='cuda', dtype=torch.float32)
        buf48 = reinterpret_tensor(buf50, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 0)  # alias
        buf49 = reinterpret_tensor(buf50, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_6.run(buf40, select_1, select_3, buf47, buf41, buf42, buf48, buf49, 3145728, grid=grid(3145728), stream=stream0) # layer1:rope
        buf51 = reinterpret_tensor(buf47, (3, 16, 2048, 64), (2097152, 131072, 64, 1)); del buf47  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf50, buf51, 6291456, grid=grid(6291456), stream=stream0)
        del buf41
        del buf42
        del buf48
        del buf49
        buf52 = reinterpret_tensor(buf40, (3, 16, 64, 2048), (2097152, 131072, 2048, 1)); del buf40  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf43, buf52, 3072, 2048, grid=grid(3072, 2048), stream=stream0)
        buf53 = buf22; del buf22  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf51, (48, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf52, (48, 64, 2048), (131072, 2048, 1), 0), out=buf53) # layer1: q @ k
        buf56 = empty_strided((3, 16, 2048, 2048), (67108864, 4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [mul_17, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_9.run(slice_3, buf53, scalar_tensor, buf56, 98304, 2048, grid=grid(98304), stream=stream0) # layer1: softmax
        buf60 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf59, (6144, 1024), (1024, 1), 0), permute_36, out=buf60)
        buf61 = empty_strided((3, 16, 2048, 64), (2097152, 131072, 64, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf60, buf61, 6291456, grid=grid(6291456), stream=stream0) # layer1:v
        buf62 = reinterpret_tensor(buf60, (48, 2048, 64), (131072, 64, 1)); del buf60  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf56, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf61, (48, 2048, 64), (131072, 64, 1), 0), out=buf62) # layer1:mixed_v
        buf63 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_3], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf62, buf63, 6291456, grid=grid(6291456), stream=stream0)
        buf64 = reinterpret_tensor(buf62, (6144, 1024), (1024, 1)); del buf62  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf63, permute_44, out=buf64) # layer1: o
        buf65 = reinterpret_tensor(buf64, (3, 2048, 1024), (2097152, 1024, 1)); del buf64  # reuse
        buf73 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        buf79 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        buf91 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        buf66 = empty_strided((3, 2048, 1), (2048, 1, 6144), device='cuda', dtype=torch.float32)
        buf67 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_14, add_15, add_18, add_19, add_2, add_3, add_4, add_5, add_6, float_1, float_2, float_3, float_6, mean, mean_1, mean_2, mean_3, mul, mul_1, mul_12, mul_13, mul_18, mul_19, mul_2, mul_20, mul_3, mul_4, mul_5, mul_6, mul_7, mul_8, rsqrt, rsqrt_1, rsqrt_2, rsqrt_3, type_as, type_as_1, type_as_2, type_as_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_per_fused__to_copy_add_mean_mul_rsqrt_15.run(buf65, unsqueeze_24, embedding, unsqueeze_25, buf30, buf36, unsqueeze_9, buf72, primals_12, unsqueeze_8, buf78, primals_11, unsqueeze_10, buf90, primals_13, primals_30, buf73, buf79, buf91, buf66, buf67, 6144, 1024, grid=grid(6144), stream=stream0) # dyndense conn after layer1: 
        buf68 = empty_strided((6144, 2816), (2816, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (6144, 1024), (1024, 1), 0), permute_45, out=buf68)
        buf69 = empty_strided((6144, 2816), (2816, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf67, (6144, 1024), (1024, 1), 0), permute_46, out=buf69)
        buf70 = empty_strided((3, 2048, 2816), (5767168, 2816, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [mul_21, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_13.run(buf68, buf69, buf70, 17301504, grid=grid(17301504), stream=stream0)
        buf71 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf70, (6144, 2816), (2816, 1), 0), permute_47, out=buf71)
        buf74 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf73, (6144, 1024), (1024, 1), 0), permute_18, out=buf74)
        buf80 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf79, (6144, 1024), (1024, 1), 0), permute_17, out=buf80)
        buf77 = buf43; del buf43  # reuse
        buf75 = reinterpret_tensor(buf77, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 0)  # alias
        buf76 = reinterpret_tensor(buf77, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 1)  # alias
        buf83 = buf50; del buf50  # reuse
        buf81 = reinterpret_tensor(buf83, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 0)  # alias
        buf82 = reinterpret_tensor(buf83, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_6.run(buf74, select_1, select_3, buf80, buf75, buf76, buf81, buf82, 3145728, grid=grid(3145728), stream=stream0)
        buf84 = reinterpret_tensor(buf80, (3, 16, 2048, 64), (2097152, 131072, 64, 1)); del buf80  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf83, buf84, 6291456, grid=grid(6291456), stream=stream0)
        del buf75
        del buf76
        del buf81
        del buf82
        buf85 = reinterpret_tensor(buf74, (3, 16, 64, 2048), (2097152, 131072, 2048, 1)); del buf74  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf77, buf85, 3072, 2048, grid=grid(3072, 2048), stream=stream0)
        buf86 = buf53; del buf53  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf84, (48, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf85, (48, 64, 2048), (131072, 2048, 1), 0), out=buf86)
        buf89 = empty_strided((3, 16, 2048, 2048), (67108864, 4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [mul_17, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_9.run(slice_3, buf86, scalar_tensor, buf89, 98304, 2048, grid=grid(98304), stream=stream0) #layer2: attn softmax & mask
        buf92 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf91, (6144, 1024), (1024, 1), 0), permute_19, out=buf92)
        buf93 = empty_strided((3, 16, 2048, 64), (2097152, 131072, 64, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf92, buf93, 6291456, grid=grid(6291456), stream=stream0)
        buf94 = reinterpret_tensor(buf92, (48, 2048, 64), (131072, 64, 1)); del buf92  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf89, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf93, (48, 2048, 64), (131072, 64, 1), 0), out=buf94)
        buf95 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_3], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf94, buf95, 6291456, grid=grid(6291456), stream=stream0)
        buf96 = reinterpret_tensor(buf94, (6144, 1024), (1024, 1)); del buf94  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf95, permute_27, out=buf96)
        buf97 = empty_strided((3, 2048, 1), (2048, 1, 6144), device='cuda', dtype=torch.float32)
        buf98 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_10, add_11, add_5, float_6, mean_3, mul_18, mul_19, mul_20, mul_4, rsqrt_3, type_as_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_per_fused__to_copy_add_mean_mul_rsqrt_16.run(unsqueeze_11, embedding, buf96, primals_18, buf97, buf98, 6144, 1024, grid=grid(6144), stream=stream0) # 
        buf99 = empty_strided((6144, 2816), (2816, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf98, (6144, 1024), (1024, 1), 0), permute_28, out=buf99)
        buf100 = empty_strided((6144, 2816), (2816, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf98, (6144, 1024), (1024, 1), 0), permute_29, out=buf100)
        buf101 = empty_strided((3, 2048, 2816), (5767168, 2816, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [mul_21, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_13.run(buf99, buf100, buf101, 17301504, grid=grid(17301504), stream=stream0)
        buf102 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf101, (6144, 2816), (2816, 1), 0), permute_30, out=buf102)
        buf103 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        buf120 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        buf127 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        buf140 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        buf148 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        buf121 = empty_strided((3, 2048, 1), (2048, 1, 6144), device='cuda', dtype=torch.float32)
        buf122 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        buf128 = empty_strided((3, 2048, 1), (2048, 1, 6144), device='cuda', dtype=torch.float32)
        buf129 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        buf141 = empty_strided((3, 2048, 1), (2048, 1, 6144), device='cuda', dtype=torch.float32)
        buf142 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        buf107 = reinterpret_tensor(buf77, (3, 2048, 1024), (2097152, 1024, 1)); del buf77  # reuse
        buf108 = empty_strided((3, 2048, 1), (2048, 1, 1), device='cuda', dtype=torch.float16)
        buf109 = empty_strided((3, 2048, 1), (2048, 1, 1), device='cuda', dtype=torch.float16)
        buf110 = empty_strided((3, 2048, 1), (2048, 1, 1), device='cuda', dtype=torch.float16)
        buf111 = empty_strided((3, 2048, 1), (2048, 1, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_10, add_12, add_18, add_19, add_20, add_21, add_22, add_23, add_24, add_25, add_26, add_27, add_28, add_29, add_3, add_36, add_37, add_38, add_39, add_5, float_1, float_2, float_3, mean, mean_1, mean_2, mul, mul_1, mul_15, mul_16, mul_17, mul_18, mul_19, mul_2, mul_20, mul_21, mul_22, mul_23, mul_24, mul_25, mul_26, mul_3, mul_32, mul_33, mul_34, mul_35, mul_4, mul_5, mul_6, mul_7, mul_8, rsqrt, rsqrt_1, rsqrt_2, type_as, type_as_1, type_as_2], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_17.run(unsqueeze_54, embedding, unsqueeze_55, buf30, buf36, unsqueeze_56, unsqueeze_11, buf96, buf102, unsqueeze_35, unsqueeze_36, unsqueeze_37, unsqueeze_32, unsqueeze_33, unsqueeze_34, unsqueeze_38, unsqueeze_39, unsqueeze_40, unsqueeze_41, unsqueeze_42, unsqueeze_43, primals_36, primals_35, primals_37, buf11, primals_47, unsqueeze_57, buf65, buf71, rsqrt_18, buf103, buf120, buf127, buf140, buf148, buf121, buf122, buf128, buf129, buf141, buf142, buf107, buf108, buf109, buf110, buf111, 6144, 1024, grid=grid(6144), stream=stream0) # dyndense conn after layer2:
        buf104 = empty_strided((1, 1, 1024, 48), (49152, 49152, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add_20, add_39, float_1, mul_35, mul_45, type_as], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_red_fused__to_copy_add_mul_sum_18.run(buf11, buf103, unsqueeze_57, buf65, buf71, rsqrt_18, buf104, 49152, 128, grid=grid(49152), stream=stream0)
        buf105 = empty_strided((1, 1, 1024), (1024, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_20, add_39, float_1, mul_35, mul_45, type_as], Original ATen: [aten._to_copy, aten.add, aten.mul, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_19.run(buf104, buf105, 1024, 48, grid=grid(1024), stream=stream0)
        buf112 = empty_strided((4, 1, 1, 5), (5, 20, 20, 1), device='cuda', dtype=torch.float16)
        buf113 = reinterpret_tensor(buf112, (4, 5), (5, 1)); del buf112  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward, aten.sum]
        triton_red_fused_add_select_backward_slice_backward_sum_20.run(buf113, buf108, buf109, buf110, buf111, 20, 6144, grid=grid(20), stream=stream0)
        buf114 = empty_strided((3, 2048, 4, 5), (40960, 20, 5, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_21.run(buf108, buf109, buf110, buf111, buf114, 122880, grid=grid(122880), stream=stream0)
        buf115 = empty_strided((20, 20), (20, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf114, (20, 6144), (1, 20), 0), view_137, out=buf115)
        del view_137
        buf116 = empty_strided((6144, 20), (20, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf114, (6144, 20), (20, 1), 0), permute_76, out=buf116)
        del buf114
        del permute_76
        buf117 = reinterpret_tensor(buf116, (3, 2048, 20), (40960, 20, 1)); del buf116  # reuse
        # Source Nodes: [l__mod___dynamic_dense_3_act], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_22.run(buf117, mm_34, 122880, grid=grid(122880), stream=stream0)
        del mm_34
        buf118 = empty_strided((20, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (20, 6144), (1, 20), 0), view_135, out=buf118)
        del view_135
        buf119 = reinterpret_tensor(buf103, (6144, 1024), (1024, 1)); del buf103  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf117, (6144, 20), (20, 1), 0), permute_80, out=buf119)
        del buf117
        del permute_80
        buf123 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_1], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf122, (6144, 1024), (1024, 1), 0), permute_52, out=buf123)
        buf130 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf129, (6144, 1024), (1024, 1), 0), permute_51, out=buf130)
        buf126 = buf83; del buf83  # reuse
        buf124 = reinterpret_tensor(buf126, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 0)  # alias
        buf125 = reinterpret_tensor(buf126, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 1)  # alias
        buf133 = empty_strided((3, 2048, 16, 32, 2), (2097152, 1024, 64, 2, 1), device='cuda', dtype=torch.float32)
        buf131 = reinterpret_tensor(buf133, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 0)  # alias
        buf132 = reinterpret_tensor(buf133, (3, 2048, 16, 32, 1), (2097152, 1024, 64, 2, 1), 1)  # alias
        # Source Nodes: [stack, stack_1], Original ATen: [aten.stack]
        triton_poi_fused_stack_6.run(buf123, select_1, select_3, buf130, buf124, buf125, buf131, buf132, 3145728, grid=grid(3145728), stream=stream0)
        buf134 = reinterpret_tensor(buf130, (3, 16, 2048, 64), (2097152, 131072, 64, 1)); del buf130  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_7.run(buf133, buf134, 6291456, grid=grid(6291456), stream=stream0)
        del buf124
        del buf125
        del buf131
        del buf132
        del buf133
        buf135 = reinterpret_tensor(buf123, (3, 16, 64, 2048), (2097152, 131072, 2048, 1)); del buf123  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.clone]
        triton_poi_fused_clone_8.run(buf126, buf135, 3072, 2048, grid=grid(3072, 2048), stream=stream0)
        del buf126
        buf136 = buf86; del buf86  # reuse
        # Source Nodes: [matmul], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf134, (48, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf135, (48, 64, 2048), (131072, 2048, 1), 0), out=buf136)
        buf139 = empty_strided((3, 16, 2048, 2048), (67108864, 4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [mul_17, softmax, where], Original ATen: [aten._softmax, aten.mul, aten.where]
        triton_red_fused__softmax_mul_where_9.run(slice_3, buf136, scalar_tensor, buf139, 98304, 2048, grid=grid(98304), stream=stream0) # layer3: attn softmax & mask
        del scalar_tensor
        buf143 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_2], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf142, (6144, 1024), (1024, 1), 0), permute_53, out=buf143)
        buf144 = empty_strided((3, 16, 2048, 64), (2097152, 131072, 64, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [matmul_1], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf143, buf144, 6291456, grid=grid(6291456), stream=stream0)
        buf145 = reinterpret_tensor(buf143, (48, 2048, 64), (131072, 64, 1)); del buf143  # reuse
        # Source Nodes: [matmul_1], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf139, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf144, (48, 2048, 64), (131072, 64, 1), 0), out=buf145)
        buf146 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_3], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf145, buf146, 6291456, grid=grid(6291456), stream=stream0)
        buf147 = reinterpret_tensor(buf145, (6144, 1024), (1024, 1)); del buf145  # reuse
        # Source Nodes: [linear_3], Original ATen: [aten.mm]
        extern_kernels.mm(buf146, permute_61, out=buf147)
        buf149 = empty_strided((3, 2048, 1), (2048, 1, 6144), device='cuda', dtype=torch.float32)
        buf150 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_31, add_32, float_6, mean_3, mul_18, mul_19, mul_20, rsqrt_3, type_as_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt]
        triton_per_fused__to_copy_add_mean_mul_rsqrt_23.run(buf148, buf147, primals_42, buf149, buf150, 6144, 1024, grid=grid(6144), stream=stream0) 
        buf151 = empty_strided((6144, 2816), (2816, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_4], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf150, (6144, 1024), (1024, 1), 0), permute_62, out=buf151)
        buf152 = empty_strided((6144, 2816), (2816, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_5], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf150, (6144, 1024), (1024, 1), 0), permute_63, out=buf152)
        buf153 = empty_strided((3, 2048, 2816), (5767168, 2816, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [mul_21, silu], Original ATen: [aten.mul, aten.silu]
        triton_poi_fused_mul_silu_13.run(buf151, buf152, buf153, 17301504, grid=grid(17301504), stream=stream0)
        buf154 = empty_strided((6144, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [linear_6], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf153, (6144, 2816), (2816, 1), 0), permute_64, out=buf154)
        buf156 = empty_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_31, add_33], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_24.run(buf119, buf148, buf147, buf154, rsqrt_17, buf156, 6144, 1024, grid=grid(6144), stream=stream0)
        del buf119
        del rsqrt_17
        buf157 = empty_strided((1024, 2816), (2816, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf156, (1024, 6144), (1, 1024), 0), reinterpret_tensor(buf153, (6144, 2816), (2816, 1), 0), out=buf157)
        buf158 = reinterpret_tensor(buf153, (6144, 2816), (2816, 1)); del buf153  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf156, (6144, 1024), (1024, 1), 0), reinterpret_tensor(permute_64, (1024, 2816), (2816, 1), 0), out=buf158)
        del permute_64
        buf159 = empty_strided((3, 2048, 2816), (5767168, 2816, 1), device='cuda', dtype=torch.float16)
        buf162 = reinterpret_tensor(buf152, (3, 2048, 2816), (5767168, 2816, 1)); del buf152  # reuse
        # Source Nodes: [silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_25.run(buf162, buf158, buf151, buf159, 17301504, grid=grid(17301504), stream=stream0)
        del buf151
        del buf158
        buf160 = empty_strided((2816, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (2816, 6144), (1, 2816), 0), reinterpret_tensor(buf150, (6144, 1024), (1024, 1), 0), out=buf160)
        buf161 = buf154; del buf154  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf159, (6144, 2816), (2816, 1), 0), reinterpret_tensor(permute_63, (2816, 1024), (1024, 1), 0), out=buf161)
        del buf159
        del permute_63
        buf163 = empty_strided((2816, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf162, (2816, 6144), (1, 2816), 0), reinterpret_tensor(buf150, (6144, 1024), (1024, 1), 0), out=buf163)
        buf164 = reinterpret_tensor(buf150, (6144, 1024), (1024, 1)); del buf150  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf162, (6144, 2816), (2816, 1), 0), reinterpret_tensor(permute_62, (2816, 1024), (1024, 1), 0), out=buf164)
        del permute_62
        buf165 = buf104; del buf104  # reuse
        # Source Nodes: [add_31, add_32, float_6, mean_3, mul_18, mul_19, rsqrt_3, type_as_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_26.run(buf161, buf164, buf148, buf147, buf149, buf165, 49152, 128, grid=grid(49152), stream=stream0)
        buf166 = empty_strided((1, 1, 1024), (1024, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_31, add_32, float_6, mean_3, mul_18, mul_19, rsqrt_3, type_as_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_19.run(buf165, buf166, 1024, 48, grid=grid(1024), stream=stream0)
        buf169 = buf156; del buf156  # reuse
        # Source Nodes: [add_31, add_32, float_6, mean_3, mul_18, rsqrt_3], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_27.run(buf169, buf161, buf164, primals_42, buf148, buf147, buf149, 6144, 1024, grid=grid(6144), stream=stream0)
        del buf147
        del buf148
        del buf149
        del buf161
        del primals_42
        buf170 = empty_strided((1024, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (1024, 6144), (1, 1024), 0), buf146, out=buf170)
        buf171 = buf146; del buf146  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf169, (6144, 1024), (1024, 1), 0), reinterpret_tensor(permute_61, (1024, 1024), (1024, 1), 0), out=buf171)
        del permute_61
        buf172 = reinterpret_tensor(buf164, (3, 16, 2048, 64), (2097152, 131072, 64, 1)); del buf164  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf171, buf172, 6291456, grid=grid(6291456), stream=stream0)
        buf173 = reinterpret_tensor(buf171, (48, 2048, 64), (131072, 64, 1)); del buf171  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf139, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf172, (48, 2048, 64), (131072, 64, 1), 0), out=buf173)
        buf174 = buf136; del buf136  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf172, (48, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf144, (48, 64, 2048), (131072, 1, 64), 0), out=buf174)
        buf176 = empty_strided((3, 16, 2048, 2048), (67108864, 4194304, 2048, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_28.run(buf174, buf139, slice_3, buf176, 98304, 2048, grid=grid(98304), stream=stream0)
        del buf139
        buf177 = reinterpret_tensor(buf172, (48, 64, 2048), (131072, 2048, 1)); del buf172  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf134, (48, 64, 2048), (131072, 1, 64), 0), reinterpret_tensor(buf176, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf177)
        buf178 = reinterpret_tensor(buf134, (48, 2048, 64), (131072, 64, 1)); del buf134  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf176, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf135, (48, 2048, 64), (131072, 1, 2048), 0), out=buf178)
        buf179 = reinterpret_tensor(buf135, (3, 2048, 16, 64), (2097152, 1024, 64, 1)); del buf135  # reuse
        buf180 = reinterpret_tensor(buf144, (3, 2048, 16, 64), (2097152, 1024, 64, 1)); del buf144  # reuse
        # Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_29.run(buf177, select_3, select_1, buf178, buf179, buf180, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf177
        buf181 = reinterpret_tensor(buf178, (6144, 1024), (1024, 1)); del buf178  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf173, buf181, 6291456, grid=grid(6291456), stream=stream0)
        del buf173
        buf182 = empty_strided((1024, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf181, (1024, 6144), (1, 1024), 0), reinterpret_tensor(buf142, (6144, 1024), (1024, 1), 0), out=buf182)
        buf183 = reinterpret_tensor(buf142, (6144, 1024), (1024, 1)); del buf142  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf181, reinterpret_tensor(permute_53, (1024, 1024), (1024, 1), 0), out=buf183)
        del permute_53
        buf184 = empty_strided((1024, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (1024, 6144), (1, 1024), 0), reinterpret_tensor(buf122, (6144, 1024), (1024, 1), 0), out=buf184)
        buf185 = reinterpret_tensor(buf122, (6144, 1024), (1024, 1)); del buf122  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf179, (6144, 1024), (1024, 1), 0), reinterpret_tensor(permute_52, (1024, 1024), (1024, 1), 0), out=buf185)
        del permute_52
        buf186 = empty_strided((1024, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf180, (1024, 6144), (1, 1024), 0), reinterpret_tensor(buf129, (6144, 1024), (1024, 1), 0), out=buf186)
        buf187 = reinterpret_tensor(buf129, (6144, 1024), (1024, 1)); del buf129  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf180, (6144, 1024), (1024, 1), 0), reinterpret_tensor(permute_51, (1024, 1024), (1024, 1), 0), out=buf187)
        del permute_51
        buf188 = buf165; del buf165  # reuse
        # Source Nodes: [add_27, float_3, mean_2, mul_6, mul_7, rsqrt_2, type_as_2], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_30.run(buf183, buf140, buf141, buf188, 49152, 128, grid=grid(49152), stream=stream0)
        buf189 = empty_strided((1, 1, 1024), (1024, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_27, float_3, mean_2, mul_6, mul_7, rsqrt_2, type_as_2], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_19.run(buf188, buf189, 1024, 48, grid=grid(1024), stream=stream0)
        buf195 = reinterpret_tensor(buf180, (3, 2048, 1024), (2097152, 1024, 1)); del buf180  # reuse
        buf199 = reinterpret_tensor(buf179, (3, 2048, 1024), (2097152, 1024, 1)); del buf179  # reuse
        buf191 = reinterpret_tensor(buf181, (3, 2048, 1024), (2097152, 1024, 1)); del buf181  # reuse
        buf200 = buf111; del buf111  # reuse
        buf203 = buf110; del buf110  # reuse
        buf210 = buf109; del buf109  # reuse
        buf213 = buf108; del buf108  # reuse
        buf201 = empty_strided((3, 2048, 1), (2048, 1, 1), device='cuda', dtype=torch.float16)
        buf205 = empty_strided((3, 2048, 1), (2048, 1, 1), device='cuda', dtype=torch.float16)
        buf211 = empty_strided((3, 2048, 1), (2048, 1, 1), device='cuda', dtype=torch.float16)
        buf215 = empty_strided((3, 2048, 1), (2048, 1, 1), device='cuda', dtype=torch.float16)
        buf202 = empty_strided((3, 2048, 1), (2048, 1, 1), device='cuda', dtype=torch.float16)
        buf212 = empty_strided((3, 2048, 1), (2048, 1, 1), device='cuda', dtype=torch.float16)
        buf216 = empty_strided((3, 2048, 1), (2048, 1, 1), device='cuda', dtype=torch.float16)
        buf208 = empty_strided((3, 2048, 1), (2048, 1, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_10, add_12, add_24, add_27, add_3, add_5, float_1, float_2, float_3, mean, mean_1, mean_2, mul, mul_3, mul_4, mul_6, rsqrt, rsqrt_1, rsqrt_2], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_31.run(buf185, primals_36, buf120, buf121, buf187, primals_35, buf127, buf128, buf183, primals_37, buf140, buf141, buf169, unsqueeze_11, embedding, buf96, buf102, buf30, buf36, buf195, buf199, buf191, buf200, buf203, buf210, buf213, buf201, buf205, buf211, buf215, buf202, buf212, buf216, buf208, 6144, 1024, grid=grid(6144), stream=stream0)
        del buf140
        del buf141
        del primals_35
        del primals_36
        del primals_37
        buf192 = buf188; del buf188  # reuse
        # Source Nodes: [add_24, float_2, mean_1, mul_3, mul_4, rsqrt_1, type_as_1], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_32.run(buf185, buf120, buf121, buf192, 49152, 128, grid=grid(49152), stream=stream0)
        del buf121
        buf193 = empty_strided((1, 1, 1024), (1024, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_24, float_2, mean_1, mul_3, mul_4, rsqrt_1, type_as_1], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_19.run(buf192, buf193, 1024, 48, grid=grid(1024), stream=stream0)
        buf196 = buf192; del buf192  # reuse
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_32.run(buf187, buf127, buf128, buf196, 49152, 128, grid=grid(49152), stream=stream0)
        del buf128
        buf197 = empty_strided((1, 1, 1024), (1024, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_19.run(buf196, buf197, 1024, 48, grid=grid(1024), stream=stream0)
        buf207 = empty_strided((4, 3, 2048, 4), (24576, 8192, 4, 1), device='cuda', dtype=torch.float16)
        buf214 = buf207; del buf207  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_select_backward_slice_backward_33.run(buf214, buf200, buf201, buf202, buf203, buf205, buf208, buf210, buf211, buf212, buf213, 98304, grid=grid(98304), stream=stream0)
        del buf200
        del buf201
        del buf202
        del buf203
        buf220 = empty_strided((3, 2048, 4, 4), (32768, 16, 4, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_34.run(buf214, buf215, buf216, buf220, 98304, grid=grid(98304), stream=stream0)
        buf222 = empty_strided((6144, 16), (16, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf220, (6144, 16), (16, 1), 0), permute_125, out=buf222)
        del permute_125
        buf223 = reinterpret_tensor(buf222, (3, 2048, 16), (32768, 16, 1)); del buf222  # reuse
        # Source Nodes: [l__mod___dynamic_dense_2_act], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_35.run(buf223, mm_25, 98304, grid=grid(98304), stream=stream0)
        del mm_25
        buf225 = buf187; del buf187  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (6144, 16), (16, 1), 0), permute_129, out=buf225)
        del permute_129
        buf204 = buf127; del buf127  # reuse
        buf206 = reinterpret_tensor(buf185, (3, 2048, 1024), (2097152, 1024, 1)); del buf185  # reuse
        buf209 = buf120; del buf120  # reuse
        buf227 = reinterpret_tensor(buf183, (3, 2048, 1024), (2097152, 1024, 1)); del buf183  # reuse
        # Source Nodes: [add_20], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused__to_copy_add_div_mul_pow_sum_36.run(buf225, buf65, buf71, buf11, primals_47, rsqrt_18, buf107, unsqueeze_56, buf169, unsqueeze_43, buf191, unsqueeze_40, unsqueeze_55, unsqueeze_42, unsqueeze_39, unsqueeze_54, unsqueeze_41, unsqueeze_38, unsqueeze_57, rsqrt_12, buf204, buf206, buf209, buf227, 6144, 1024, grid=grid(6144), stream=stream0)
        del buf107
        del buf11
        del buf169
        del buf191
        del buf225
        del primals_47
        del rsqrt_12
        del rsqrt_18
        del unsqueeze_38
        del unsqueeze_39
        del unsqueeze_40
        del unsqueeze_41
        del unsqueeze_42
        del unsqueeze_43
        del unsqueeze_54
        del unsqueeze_55
        del unsqueeze_56
        del unsqueeze_57
        buf217 = empty_strided((4, 1, 1, 4, 48), (192, 768, 768, 1, 4), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward, aten.sum]
        triton_red_fused_add_select_backward_slice_backward_sum_37.run(buf214, buf215, buf216, buf217, 768, 128, grid=grid(768), stream=stream0)
        del buf214
        buf218 = empty_strided((4, 1, 1, 4), (4, 16, 16, 1), device='cuda', dtype=torch.float16)
        buf219 = reinterpret_tensor(buf218, (4, 4), (4, 1)); del buf218  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward, aten.sum]
        triton_per_fused_add_select_backward_slice_backward_sum_38.run(buf219, buf217, 16, 48, grid=grid(16), stream=stream0)
        del buf217
        buf221 = empty_strided((16, 16), (16, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf220, (16, 6144), (1, 16), 0), view_102, out=buf221)
        del buf220
        del view_102
        buf224 = empty_strided((16, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf223, (16, 6144), (1, 16), 0), view_100, out=buf224)
        del buf223
        del view_100
        buf228 = empty_strided((1024, 2816), (2816, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (1024, 6144), (1, 1024), 0), reinterpret_tensor(buf70, (6144, 2816), (2816, 1), 0), out=buf228)
        buf229 = reinterpret_tensor(buf70, (6144, 2816), (2816, 1)); del buf70  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf227, (6144, 1024), (1024, 1), 0), reinterpret_tensor(permute_47, (1024, 2816), (2816, 1), 0), out=buf229)
        del permute_47
        buf230 = buf162; del buf162  # reuse
        buf233 = reinterpret_tensor(buf69, (3, 2048, 2816), (5767168, 2816, 1)); del buf69  # reuse
        # Source Nodes: [silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_25.run(buf233, buf229, buf68, buf230, 17301504, grid=grid(17301504), stream=stream0)
        del buf229
        del buf68
        buf231 = empty_strided((2816, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf230, (2816, 6144), (1, 2816), 0), reinterpret_tensor(buf67, (6144, 1024), (1024, 1), 0), out=buf231)
        buf232 = buf71; del buf71  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf230, (6144, 2816), (2816, 1), 0), reinterpret_tensor(permute_46, (2816, 1024), (1024, 1), 0), out=buf232)
        del buf230
        del permute_46
        buf234 = empty_strided((2816, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (2816, 6144), (1, 2816), 0), reinterpret_tensor(buf67, (6144, 1024), (1024, 1), 0), out=buf234)
        buf235 = reinterpret_tensor(buf67, (6144, 1024), (1024, 1)); del buf67  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf233, (6144, 2816), (2816, 1), 0), reinterpret_tensor(permute_45, (2816, 1024), (1024, 1), 0), out=buf235)
        del permute_45
        buf236 = buf196; del buf196  # reuse
        # Source Nodes: [add_19, float_6, mean_3, mul_18, mul_19, rsqrt_3, type_as_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_39.run(buf232, buf235, buf65, buf66, buf236, 49152, 128, grid=grid(49152), stream=stream0)
        buf237 = empty_strided((1, 1, 1024), (1024, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_19, float_6, mean_3, mul_18, mul_19, rsqrt_3, type_as_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_19.run(buf236, buf237, 1024, 48, grid=grid(1024), stream=stream0)
        buf239 = buf227; del buf227  # reuse
        # Source Nodes: [add_19, float_6, mean_3, mul_18, rsqrt_3], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_40.run(buf239, buf232, buf235, primals_30, buf65, buf66, 6144, 1024, grid=grid(6144), stream=stream0)
        del buf232
        del buf235
        del buf66
        del primals_30
        buf240 = empty_strided((1024, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf239, (1024, 6144), (1, 1024), 0), buf63, out=buf240)
        buf241 = buf63; del buf63  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf239, (6144, 1024), (1024, 1), 0), reinterpret_tensor(permute_44, (1024, 1024), (1024, 1), 0), out=buf241)
        del permute_44
        buf242 = reinterpret_tensor(buf65, (3, 16, 2048, 64), (2097152, 131072, 64, 1)); del buf65  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf241, buf242, 6291456, grid=grid(6291456), stream=stream0)
        buf243 = reinterpret_tensor(buf241, (48, 2048, 64), (131072, 64, 1)); del buf241  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf56, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf242, (48, 2048, 64), (131072, 64, 1), 0), out=buf243)
        buf244 = reinterpret_tensor(buf176, (48, 2048, 2048), (4194304, 2048, 1)); del buf176  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf242, (48, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf61, (48, 64, 2048), (131072, 1, 64), 0), out=buf244)
        buf246 = reinterpret_tensor(buf174, (3, 16, 2048, 2048), (67108864, 4194304, 2048, 1)); del buf174  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_28.run(buf244, buf56, slice_3, buf246, 98304, 2048, grid=grid(98304), stream=stream0)
        del buf244
        buf247 = reinterpret_tensor(buf61, (48, 64, 2048), (131072, 2048, 1)); del buf61  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf51, (48, 64, 2048), (131072, 1, 64), 0), reinterpret_tensor(buf246, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf247)
        buf248 = reinterpret_tensor(buf51, (48, 2048, 64), (131072, 64, 1)); del buf51  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf246, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf52, (48, 2048, 64), (131072, 1, 2048), 0), out=buf248)
        buf249 = reinterpret_tensor(buf52, (3, 2048, 16, 64), (2097152, 1024, 64, 1)); del buf52  # reuse
        buf250 = reinterpret_tensor(buf242, (3, 2048, 16, 64), (2097152, 1024, 64, 1)); del buf242  # reuse
        # Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_29.run(buf247, select_3, select_1, buf248, buf249, buf250, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf247
        buf251 = reinterpret_tensor(buf248, (6144, 1024), (1024, 1)); del buf248  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf243, buf251, 6291456, grid=grid(6291456), stream=stream0)
        del buf243
        buf252 = empty_strided((1024, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf251, (1024, 6144), (1, 1024), 0), reinterpret_tensor(buf59, (6144, 1024), (1024, 1), 0), out=buf252)
        buf253 = reinterpret_tensor(buf59, (6144, 1024), (1024, 1)); del buf59  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf251, reinterpret_tensor(permute_36, (1024, 1024), (1024, 1), 0), out=buf253)
        del permute_36
        buf254 = empty_strided((1024, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (1024, 6144), (1, 1024), 0), reinterpret_tensor(buf39, (6144, 1024), (1024, 1), 0), out=buf254)
        buf255 = reinterpret_tensor(buf39, (6144, 1024), (1024, 1)); del buf39  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf249, (6144, 1024), (1024, 1), 0), reinterpret_tensor(permute_35, (1024, 1024), (1024, 1), 0), out=buf255)
        del permute_35
        buf256 = empty_strided((1024, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf250, (1024, 6144), (1, 1024), 0), reinterpret_tensor(buf46, (6144, 1024), (1024, 1), 0), out=buf256)
        buf257 = reinterpret_tensor(buf46, (6144, 1024), (1024, 1)); del buf46  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf250, (6144, 1024), (1024, 1), 0), reinterpret_tensor(permute_34, (1024, 1024), (1024, 1), 0), out=buf257)
        del permute_34
        buf258 = buf236; del buf236  # reuse
        # Source Nodes: [add_14, mean_2, mul_6, mul_7, rsqrt_2, type_as_2], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_41.run(buf253, buf57, buf58, buf258, 49152, 128, grid=grid(49152), stream=stream0)
        buf259 = empty_strided((1, 1, 1024), (1024, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_14, mean_2, mul_6, mul_7, rsqrt_2, type_as_2], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_19.run(buf258, buf259, 1024, 48, grid=grid(1024), stream=stream0)
        buf265 = reinterpret_tensor(buf250, (3, 2048, 1024), (2097152, 1024, 1)); del buf250  # reuse
        buf261 = reinterpret_tensor(buf249, (3, 2048, 1024), (2097152, 1024, 1)); del buf249  # reuse
        buf269 = reinterpret_tensor(buf251, (3, 2048, 1024), (2097152, 1024, 1)); del buf251  # reuse
        buf270 = buf216; del buf216  # reuse
        buf272 = buf215; del buf215  # reuse
        buf276 = buf213; del buf213  # reuse
        buf279 = buf212; del buf212  # reuse
        buf271 = buf211; del buf211  # reuse
        buf280 = buf210; del buf210  # reuse
        buf274 = buf208; del buf208  # reuse
        buf278 = buf205; del buf205  # reuse
        # Source Nodes: [add, add_12, add_14, add_3, add_5, mean, mean_1, mean_2, mul, mul_3, mul_6, rsqrt, rsqrt_1, rsqrt_2], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_42.run(buf255, primals_24, buf37, buf38, buf253, primals_25, buf57, buf58, buf257, primals_23, buf44, buf45, buf239, embedding, buf30, buf36, buf265, buf261, buf269, buf270, buf272, buf276, buf279, buf271, buf280, buf274, buf278, 6144, 1024, grid=grid(6144), stream=stream0)
        del buf253
        del buf58
        del primals_23
        del primals_24
        del primals_25
        buf262 = buf258; del buf258  # reuse
        # Source Nodes: [add_12, mean_1, mul_3, mul_4, rsqrt_1, type_as_1], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_43.run(buf255, buf37, buf38, buf262, 49152, 128, grid=grid(49152), stream=stream0)
        del buf255
        del buf38
        buf263 = empty_strided((1, 1, 1024), (1024, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_12, mean_1, mul_3, mul_4, rsqrt_1, type_as_1], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_19.run(buf262, buf263, 1024, 48, grid=grid(1024), stream=stream0)
        buf266 = buf262; del buf262  # reuse
        # Source Nodes: [add, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_43.run(buf257, buf44, buf45, buf266, 49152, 128, grid=grid(49152), stream=stream0)
        del buf45
        buf267 = empty_strided((1, 1, 1024), (1024, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_19.run(buf266, buf267, 1024, 48, grid=grid(1024), stream=stream0)
        buf277 = empty_strided((4, 3, 2048, 3), (18432, 6144, 3, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward]
        triton_poi_fused_add_select_backward_slice_backward_44.run(buf270, buf271, buf272, buf274, buf276, buf277, 73728, grid=grid(73728), stream=stream0)
        del buf270
        del buf271
        del buf272
        del buf274
        buf284 = empty_strided((3, 2048, 4, 3), (24576, 12, 3, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_45.run(buf277, buf278, buf279, buf280, buf284, 73728, grid=grid(73728), stream=stream0)
        buf286 = empty_strided((6144, 12), (12, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf284, (6144, 12), (12, 1), 0), permute_174, out=buf286)
        del permute_174
        buf287 = reinterpret_tensor(buf286, (3, 2048, 12), (24576, 12, 1)); del buf286  # reuse
        # Source Nodes: [l__mod___dynamic_dense_1_act], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_46.run(buf287, mm_16, 73728, grid=grid(73728), stream=stream0)
        del mm_16
        buf289 = buf257; del buf257  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf287, (6144, 12), (12, 1), 0), permute_178, out=buf289)
        del permute_178
        buf273 = buf206; del buf206  # reuse
        buf275 = buf209; del buf209  # reuse
        buf291 = buf204; del buf204  # reuse
        # Source Nodes: [add_10, add_12, add_5, mul_4], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_47.run(buf273, buf275, buf291, buf289, unsqueeze_11, embedding, buf96, buf102, buf195, unsqueeze_36, buf199, unsqueeze_33, buf239, unsqueeze_25, buf261, unsqueeze_23, unsqueeze_35, unsqueeze_32, unsqueeze_24, unsqueeze_22, unsqueeze_37, unsqueeze_34, rsqrt_7, 6144, 1024, grid=grid(6144), stream=stream0)
        del buf102
        del rsqrt_7
        del unsqueeze_22
        del unsqueeze_23
        del unsqueeze_24
        del unsqueeze_25
        del unsqueeze_32
        del unsqueeze_33
        del unsqueeze_34
        del unsqueeze_35
        del unsqueeze_36
        del unsqueeze_37
        buf281 = empty_strided((4, 1, 1, 3, 48), (144, 576, 576, 1, 3), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward, aten.sum]
        triton_red_fused_add_select_backward_slice_backward_sum_48.run(buf277, buf278, buf279, buf280, buf281, 576, 128, grid=grid(576), stream=stream0)
        del buf277
        buf282 = empty_strided((4, 1, 1, 3), (3, 12, 12, 1), device='cuda', dtype=torch.float16)
        buf283 = reinterpret_tensor(buf282, (4, 3), (3, 1)); del buf282  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward, aten.sum]
        triton_per_fused_add_select_backward_slice_backward_sum_49.run(buf283, buf281, 12, 48, grid=grid(12), stream=stream0)
        del buf281
        buf285 = empty_strided((12, 12), (12, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf284, (12, 6144), (1, 12), 0), view_67, out=buf285)
        del buf284
        del view_67
        buf288 = empty_strided((12, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf287, (12, 6144), (1, 12), 0), view_65, out=buf288)
        del buf287
        del view_65
        buf292 = empty_strided((1024, 2816), (2816, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf291, (1024, 6144), (1, 1024), 0), reinterpret_tensor(buf101, (6144, 2816), (2816, 1), 0), out=buf292)
        buf293 = reinterpret_tensor(buf101, (6144, 2816), (2816, 1)); del buf101  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf291, (6144, 1024), (1024, 1), 0), reinterpret_tensor(permute_30, (1024, 2816), (2816, 1), 0), out=buf293)
        del permute_30
        buf294 = buf233; del buf233  # reuse
        buf297 = reinterpret_tensor(buf100, (3, 2048, 2816), (5767168, 2816, 1)); del buf100  # reuse
        # Source Nodes: [silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_25.run(buf297, buf293, buf99, buf294, 17301504, grid=grid(17301504), stream=stream0)
        del buf293
        del buf99
        buf295 = empty_strided((2816, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf294, (2816, 6144), (1, 2816), 0), reinterpret_tensor(buf98, (6144, 1024), (1024, 1), 0), out=buf295)
        buf296 = buf289; del buf289  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf294, (6144, 2816), (2816, 1), 0), reinterpret_tensor(permute_29, (2816, 1024), (1024, 1), 0), out=buf296)
        del permute_29
        buf298 = empty_strided((2816, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf297, (2816, 6144), (1, 2816), 0), reinterpret_tensor(buf98, (6144, 1024), (1024, 1), 0), out=buf298)
        buf299 = reinterpret_tensor(buf98, (6144, 1024), (1024, 1)); del buf98  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf297, (6144, 2816), (2816, 1), 0), reinterpret_tensor(permute_28, (2816, 1024), (1024, 1), 0), out=buf299)
        del permute_28
        buf304 = buf291; del buf291  # reuse
        # Source Nodes: [add_10, add_11, add_5, float_6, mean_3, mul_18, mul_4, rsqrt_3], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_50.run(buf304, buf296, buf299, primals_18, unsqueeze_11, embedding, buf96, buf97, 6144, 1024, grid=grid(6144), stream=stream0)
        del primals_18
        buf306 = reinterpret_tensor(buf261, (6144, 1024), (1024, 1)); del buf261  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (6144, 1024), (1024, 1), 0), reinterpret_tensor(permute_27, (1024, 1024), (1024, 1), 0), out=buf306)
        del permute_27
        buf307 = reinterpret_tensor(buf239, (3, 16, 2048, 64), (2097152, 131072, 64, 1)); del buf239  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf306, buf307, 6291456, grid=grid(6291456), stream=stream0)
        buf308 = reinterpret_tensor(buf306, (48, 2048, 64), (131072, 64, 1)); del buf306  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf89, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf307, (48, 2048, 64), (131072, 64, 1), 0), out=buf308)
        buf316 = reinterpret_tensor(buf199, (6144, 1024), (1024, 1)); del buf199  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf308, buf316, 6291456, grid=grid(6291456), stream=stream0)
        buf318 = reinterpret_tensor(buf308, (6144, 1024), (1024, 1)); del buf308  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf316, reinterpret_tensor(permute_19, (1024, 1024), (1024, 1), 0), out=buf318)
        del permute_19
        buf309 = reinterpret_tensor(buf246, (48, 2048, 2048), (4194304, 2048, 1)); del buf246  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf307, (48, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf93, (48, 64, 2048), (131072, 1, 64), 0), out=buf309)
        buf311 = buf56; del buf56  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_28.run(buf309, buf89, slice_3, buf311, 98304, 2048, grid=grid(98304), stream=stream0)
        del buf309
        buf312 = reinterpret_tensor(buf93, (48, 64, 2048), (131072, 2048, 1)); del buf93  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf84, (48, 64, 2048), (131072, 1, 64), 0), reinterpret_tensor(buf311, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf312)
        buf313 = reinterpret_tensor(buf84, (48, 2048, 64), (131072, 64, 1)); del buf84  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf311, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf85, (48, 2048, 64), (131072, 1, 2048), 0), out=buf313)
        buf314 = reinterpret_tensor(buf85, (3, 2048, 16, 64), (2097152, 1024, 64, 1)); del buf85  # reuse
        buf315 = reinterpret_tensor(buf307, (3, 2048, 16, 64), (2097152, 1024, 64, 1)); del buf307  # reuse
        # Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_29.run(buf312, select_3, select_1, buf313, buf314, buf315, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        buf320 = reinterpret_tensor(buf313, (6144, 1024), (1024, 1)); del buf313  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf314, (6144, 1024), (1024, 1), 0), reinterpret_tensor(permute_18, (1024, 1024), (1024, 1), 0), out=buf320)
        del permute_18
        buf322 = reinterpret_tensor(buf312, (6144, 1024), (1024, 1)); del buf312  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (6144, 1024), (1024, 1), 0), reinterpret_tensor(permute_17, (1024, 1024), (1024, 1), 0), out=buf322)
        del permute_17
        buf337 = buf275; del buf275  # reuse
        buf330 = buf44; del buf44  # reuse
        buf335 = buf280; del buf280  # reuse
        buf336 = buf279; del buf279  # reuse
        buf334 = buf37; del buf37  # reuse
        buf338 = buf278; del buf278  # reuse
        buf339 = buf276; del buf276  # reuse
        # Source Nodes: [add, add_2, add_3, add_4, add_6, float_1, float_2, float_3, mean, mean_1, mean_2, mul, mul_1, mul_2, mul_3, mul_6, rsqrt, rsqrt_1, rsqrt_2], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_51.run(buf337, buf318, primals_13, unsqueeze_10, embedding, buf320, primals_12, unsqueeze_9, buf90, buf265, unsqueeze_20, buf269, unsqueeze_18, buf304, unsqueeze_11, buf72, buf322, primals_11, unsqueeze_8, buf78, buf330, buf335, buf336, buf334, buf338, buf339, 6144, 1024, grid=grid(6144), stream=stream0)
        del primals_11
        del primals_12
        del primals_13
        del unsqueeze_18
        del unsqueeze_20
        buf342 = empty_strided((3, 2048, 4, 2), (16384, 8, 2, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_52.run(buf335, buf336, buf338, buf339, buf342, 49152, grid=grid(49152), stream=stream0)
        buf344 = empty_strided((6144, 8), (8, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf342, (6144, 8), (8, 1), 0), permute_223, out=buf344)
        del permute_223
        buf345 = reinterpret_tensor(buf344, (3, 2048, 8), (16384, 8, 1)); del buf344  # reuse
        # Source Nodes: [l__mod___dynamic_dense_0_act], Original ATen: [aten.gelu, aten.gelu_backward]
        triton_poi_fused_gelu_gelu_backward_53.run(buf345, mm_7, 49152, grid=grid(49152), stream=stream0)
        del mm_7
        buf347 = reinterpret_tensor(buf195, (6144, 1024), (1024, 1)); del buf195  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf345, (6144, 8), (8, 1), 0), permute_227, out=buf347)
        del permute_227
        buf349 = buf265; del buf265  # reuse
        # Source Nodes: [add_3, add_5], Original ATen: [aten.add, aten.div, aten.mul, aten.pow, aten.sum]
        triton_per_fused_add_div_mul_pow_sum_54.run(buf349, buf347, embedding, buf30, buf36, buf273, unsqueeze_21, buf269, unsqueeze_19, rsqrt_2, 6144, 1024, grid=grid(6144), stream=stream0)
        del buf269
        del buf273
        del rsqrt_2
        del unsqueeze_19
        del unsqueeze_21
        buf351 = reinterpret_tensor(buf297, (6144, 2816), (2816, 1)); del buf297  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf349, (6144, 1024), (1024, 1), 0), reinterpret_tensor(permute_13, (1024, 2816), (2816, 1), 0), out=buf351)
        del permute_13
        buf352 = buf294; del buf294  # reuse
        buf355 = reinterpret_tensor(buf34, (3, 2048, 2816), (5767168, 2816, 1)); del buf34  # reuse
        # Source Nodes: [silu], Original ATen: [aten.add, aten.fill, aten.mul, aten.sigmoid, aten.silu, aten.sub]
        triton_poi_fused_add_fill_mul_sigmoid_silu_sub_25.run(buf355, buf351, buf33, buf352, 17301504, grid=grid(17301504), stream=stream0)
        del buf33
        del buf351
        buf354 = buf36; del buf36  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf352, (6144, 2816), (2816, 1), 0), reinterpret_tensor(permute_12, (2816, 1024), (1024, 1), 0), out=buf354)
        del permute_12
        buf357 = buf347; del buf347  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf355, (6144, 2816), (2816, 1), 0), reinterpret_tensor(permute_11, (2816, 1024), (1024, 1), 0), out=buf357)
        del permute_11
        buf300 = buf266; del buf266  # reuse
        buf323 = empty_strided((1, 1, 1024, 48), (49152, 49152, 1, 1024), device='cuda', dtype=torch.float32)
        buf327 = empty_strided((1, 1, 1024, 48), (49152, 49152, 1, 1024), device='cuda', dtype=torch.float32)
        buf331 = empty_strided((1, 1, 1024, 48), (49152, 49152, 1, 1024), device='cuda', dtype=torch.float32)
        buf358 = empty_strided((1, 1, 1024, 48), (49152, 49152, 1, 1024), device='cuda', dtype=torch.float32)
        # Source Nodes: [add, add_10, add_11, add_2, add_3, add_4, add_5, add_6, float_1, float_2, float_3, float_4, float_6, mean, mean_1, mean_2, mean_3, mul, mul_1, mul_12, mul_13, mul_18, mul_19, mul_2, mul_3, mul_4, mul_6, mul_7, rsqrt, rsqrt_1, rsqrt_2, rsqrt_3, type_as, type_as_1, type_as_2, type_as_3, type_as_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_red_fused__to_copy_add_mean_mul_rsqrt_sum_55.run(buf296, buf299, unsqueeze_11, embedding, buf96, buf97, buf318, unsqueeze_10, buf90, buf320, unsqueeze_9, buf72, buf322, unsqueeze_8, buf78, buf354, buf357, buf30, buf31, buf300, buf323, buf327, buf331, buf358, 49152, 128, grid=grid(49152), stream=stream0)
        del buf296
        del buf299
        del buf318
        del buf320
        del buf322
        del buf72
        del buf78
        del buf90
        del buf96
        del buf97
        del unsqueeze_10
        del unsqueeze_11
        buf301 = empty_strided((1, 1, 1024), (1024, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_10, add_11, add_5, float_6, mean_3, mul_18, mul_19, mul_4, rsqrt_3, type_as_5], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_19.run(buf300, buf301, 1024, 48, grid=grid(1024), stream=stream0)
        del buf300
        buf305 = empty_strided((1024, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf304, (1024, 6144), (1, 1024), 0), buf95, out=buf305)
        del buf304
        del buf95
        buf317 = empty_strided((1024, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf316, (1024, 6144), (1, 1024), 0), reinterpret_tensor(buf91, (6144, 1024), (1024, 1), 0), out=buf317)
        del buf316
        del buf91
        buf319 = empty_strided((1024, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf314, (1024, 6144), (1, 1024), 0), reinterpret_tensor(buf73, (6144, 1024), (1024, 1), 0), out=buf319)
        del buf314
        del buf73
        buf321 = empty_strided((1024, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf315, (1024, 6144), (1, 1024), 0), reinterpret_tensor(buf79, (6144, 1024), (1024, 1), 0), out=buf321)
        del buf315
        del buf79
        buf324 = empty_strided((1, 1, 1024), (1024, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_4, add_6, float_3, mean_2, mul_3, mul_6, mul_7, rsqrt_2, type_as_2], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_19.run(buf323, buf324, 1024, 48, grid=grid(1024), stream=stream0)
        del buf323
        buf328 = empty_strided((1, 1, 1024), (1024, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_2, mean_1, mul_2, mul_3, mul_4, rsqrt_1, type_as_1], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_19.run(buf327, buf328, 1024, 48, grid=grid(1024), stream=stream0)
        del buf327
        buf332 = empty_strided((1, 1, 1024), (1024, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add, add_2, float_1, mean, mul, mul_1, rsqrt, type_as], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_19.run(buf331, buf332, 1024, 48, grid=grid(1024), stream=stream0)
        del buf331
        buf340 = empty_strided((4, 1, 1, 2), (2, 8, 8, 1), device='cuda', dtype=torch.float16)
        buf341 = reinterpret_tensor(buf340, (4, 2), (2, 1)); del buf340  # reuse
        # Source Nodes: [], Original ATen: [aten.add, aten.select_backward, aten.slice_backward, aten.sum]
        triton_red_fused_add_select_backward_slice_backward_sum_56.run(buf341, buf335, buf336, buf338, buf339, 8, 6144, grid=grid(8), stream=stream0)
        del buf335
        del buf336
        del buf338
        del buf339
        buf343 = empty_strided((8, 8), (8, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf342, (8, 6144), (1, 8), 0), view_32, out=buf343)
        del buf342
        del view_32
        buf346 = empty_strided((8, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf345, (8, 6144), (1, 8), 0), view_30, out=buf346)
        del buf345
        del view_30
        buf350 = empty_strided((1024, 2816), (2816, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf349, (1024, 6144), (1, 1024), 0), reinterpret_tensor(buf35, (6144, 2816), (2816, 1), 0), out=buf350)
        del buf35
        buf353 = empty_strided((2816, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf352, (2816, 6144), (1, 2816), 0), reinterpret_tensor(buf32, (6144, 1024), (1024, 1), 0), out=buf353)
        del buf352
        buf356 = empty_strided((2816, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf355, (2816, 6144), (1, 2816), 0), reinterpret_tensor(buf32, (6144, 1024), (1024, 1), 0), out=buf356)
        del buf355
        buf359 = empty_strided((1, 1, 1024), (1024, 1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, mul_13, rsqrt_1, type_as_3], Original ATen: [aten._to_copy, aten.add, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_mul_sum_19.run(buf358, buf359, 1024, 48, grid=grid(1024), stream=stream0)
        del buf358
        buf361 = buf57; del buf57  # reuse
        buf362 = buf32; del buf32  # reuse
        # Source Nodes: [add_3, add_4, float_4, mean_1, mul_12, rsqrt_1], Original ATen: [aten._to_copy, aten.add, aten.div, aten.mean, aten.mul, aten.rsqrt, aten.sum]
        triton_per_fused__to_copy_add_div_mean_mul_rsqrt_sum_57.run(buf354, buf357, primals_6, embedding, buf30, buf31, buf349, buf361, buf362, 6144, 1024, grid=grid(6144), stream=stream0)
        del buf30
        del buf31
        del buf354
        del buf357
        del primals_6
        buf363 = empty_strided((1024, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf362, (1024, 6144), (1, 1024), 0), buf29, out=buf363)
        buf364 = buf29; del buf29  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf362, (6144, 1024), (1024, 1), 0), reinterpret_tensor(permute_10, (1024, 1024), (1024, 1), 0), out=buf364)
        del permute_10
        buf365 = reinterpret_tensor(buf362, (3, 16, 2048, 64), (2097152, 131072, 64, 1)); del buf362  # reuse
        # Source Nodes: [], Original ATen: [aten.clone]
        triton_poi_fused_clone_10.run(buf364, buf365, 6291456, grid=grid(6291456), stream=stream0)
        buf366 = reinterpret_tensor(buf364, (48, 2048, 64), (131072, 64, 1)); del buf364  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf25, (48, 2048, 2048), (4194304, 1, 2048), 0), reinterpret_tensor(buf365, (48, 2048, 64), (131072, 64, 1), 0), out=buf366)
        buf367 = reinterpret_tensor(buf311, (48, 2048, 2048), (4194304, 2048, 1)); del buf311  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf365, (48, 2048, 64), (131072, 64, 1), 0), reinterpret_tensor(buf27, (48, 64, 2048), (131072, 1, 64), 0), out=buf367)
        buf369 = buf89; del buf89  # reuse
        # Source Nodes: [cross_entropy], Original ATen: [aten._softmax_backward_data, aten.mul, aten.nll_loss_forward, aten.where]
        triton_red_fused__softmax_backward_data_mul_nll_loss_forward_where_28.run(buf367, buf25, slice_3, buf369, 98304, 2048, grid=grid(98304), stream=stream0)
        del buf25
        del buf367
        del slice_3
        buf370 = reinterpret_tensor(buf365, (48, 64, 2048), (131072, 2048, 1)); del buf365  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf20, (48, 64, 2048), (131072, 1, 64), 0), reinterpret_tensor(buf369, (48, 2048, 2048), (4194304, 2048, 1), 0), out=buf370)
        buf371 = reinterpret_tensor(buf20, (48, 2048, 64), (131072, 64, 1)); del buf20  # reuse
        # Source Nodes: [], Original ATen: [aten.bmm]
        extern_kernels.bmm(reinterpret_tensor(buf369, (48, 2048, 2048), (4194304, 2048, 1), 0), reinterpret_tensor(buf21, (48, 2048, 64), (131072, 1, 2048), 0), out=buf371)
        del buf369
        buf372 = reinterpret_tensor(buf21, (3, 2048, 16, 64), (2097152, 1024, 64, 1)); del buf21  # reuse
        buf373 = reinterpret_tensor(buf27, (3, 2048, 16, 64), (2097152, 1024, 64, 1)); del buf27  # reuse
        # Source Nodes: [], Original ATen: [aten._to_copy]
        triton_poi_fused__to_copy_29.run(buf370, select_3, select_1, buf371, buf372, buf373, 6144, 1024, grid=grid(6144, 1024), stream=stream0)
        del buf370
        del select_1
        del select_3
        buf374 = reinterpret_tensor(buf371, (6144, 1024), (1024, 1)); del buf371  # reuse
        # Source Nodes: [], Original ATen: [aten.view]
        triton_poi_fused_view_11.run(buf366, buf374, 6291456, grid=grid(6291456), stream=stream0)
        buf375 = empty_strided((1024, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf374, (1024, 6144), (1, 1024), 0), reinterpret_tensor(embedding, (6144, 1024), (1024, 1), 0), out=buf375)
        buf376 = reinterpret_tensor(buf366, (6144, 1024), (1024, 1)); del buf366  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(buf374, reinterpret_tensor(permute_2, (1024, 1024), (1024, 1), 0), out=buf376)
        del permute_2
        buf377 = empty_strided((1024, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (1024, 6144), (1, 1024), 0), reinterpret_tensor(embedding, (6144, 1024), (1024, 1), 0), out=buf377)
        buf378 = buf374; del buf374  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf372, (6144, 1024), (1024, 1), 0), reinterpret_tensor(permute_1, (1024, 1024), (1024, 1), 0), out=buf378)
        del permute_1
        buf381 = reinterpret_tensor(buf372, (6144, 1024), (1024, 1)); del buf372  # reuse
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf373, (6144, 1024), (1024, 1), 0), reinterpret_tensor(permute, (1024, 1024), (1024, 1), 0), out=buf381)
        del permute
        buf382 = empty_strided((50257, 1024), (1024, 1), device='cuda', dtype=torch.float32)
        # Source Nodes: [], Original ATen: [aten.add, aten.embedding_dense_backward]
        triton_poi_fused_add_embedding_dense_backward_58.run(buf382, 51463168, grid=grid(51463168), stream=stream0)
        buf379 = buf337; del buf337  # reuse
        # Source Nodes: [], Original ATen: [aten._to_copy, aten.add, aten.embedding_dense_backward, aten.mul]
        triton_poi_fused__to_copy_add_embedding_dense_backward_mul_59.run(buf379, buf330, unsqueeze_9, buf334, unsqueeze_8, buf349, buf361, buf376, buf378, primals_60, buf381, buf382, 6291456, grid=grid(6291456), stream=stream0)
        del buf330
        del buf334
        del buf349
        del buf361
        del buf376
        del buf378
        del buf379
        del buf381
        del primals_60
        del unsqueeze_8
        del unsqueeze_9
        buf380 = empty_strided((1024, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.mm]
        extern_kernels.mm(reinterpret_tensor(buf373, (1024, 6144), (1, 1024), 0), reinterpret_tensor(embedding, (6144, 1024), (1024, 1), 0), out=buf380)
        del buf373
        del embedding
        buf384 = empty_strided((50257, 1024), (1024, 1), device='cuda', dtype=torch.float16)
        # Source Nodes: [], Original ATen: [aten.embedding_dense_backward]
        triton_poi_fused_embedding_dense_backward_60.run(buf382, buf384, 51463168, grid=grid(51463168), stream=stream0)
        return (None, reinterpret_tensor(buf380, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf377, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf375, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf363, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf359, (1024, ), (1, ), 0), reinterpret_tensor(buf356, (2816, 1024), (1024, 1), 0), reinterpret_tensor(buf353, (2816, 1024), (1024, 1), 0), reinterpret_tensor(buf350, (1024, 2816), (2816, 1), 0), buf341, reinterpret_tensor(buf332, (1024, ), (1, ), 0), reinterpret_tensor(buf328, (1024, ), (1, ), 0), reinterpret_tensor(buf324, (1024, ), (1, ), 0), reinterpret_tensor(buf321, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf319, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf317, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf305, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf301, (1024, ), (1, ), 0), reinterpret_tensor(buf298, (2816, 1024), (1024, 1), 0), reinterpret_tensor(buf295, (2816, 1024), (1024, 1), 0), reinterpret_tensor(buf292, (1024, 2816), (2816, 1), 0), buf283, reinterpret_tensor(buf267, (1024, ), (1, ), 0), reinterpret_tensor(buf263, (1024, ), (1, ), 0), reinterpret_tensor(buf259, (1024, ), (1, ), 0), reinterpret_tensor(buf256, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf254, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf252, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf240, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf237, (1024, ), (1, ), 0), reinterpret_tensor(buf234, (2816, 1024), (1024, 1), 0), reinterpret_tensor(buf231, (2816, 1024), (1024, 1), 0), reinterpret_tensor(buf228, (1024, 2816), (2816, 1), 0), buf219, reinterpret_tensor(buf197, (1024, ), (1, ), 0), reinterpret_tensor(buf193, (1024, ), (1, ), 0), reinterpret_tensor(buf189, (1024, ), (1, ), 0), reinterpret_tensor(buf186, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf184, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf182, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf170, (1024, 1024), (1024, 1), 0), reinterpret_tensor(buf166, (1024, ), (1, ), 0), reinterpret_tensor(buf163, (2816, 1024), (1024, 1), 0), reinterpret_tensor(buf160, (2816, 1024), (1024, 1), 0), reinterpret_tensor(buf157, (1024, 2816), (2816, 1), 0), buf113, reinterpret_tensor(buf105, (1024, ), (1, ), 0), buf384, reinterpret_tensor(buf346, (8, 1024), (1024, 1), 0), reinterpret_tensor(buf343, (8, 8), (8, 1), 0), reinterpret_tensor(buf288, (12, 1024), (1024, 1), 0), reinterpret_tensor(buf285, (12, 12), (12, 1), 0), reinterpret_tensor(buf224, (16, 1024), (1024, 1), 0), reinterpret_tensor(buf221, (16, 16), (16, 1), 0), reinterpret_tensor(buf118, (20, 1024), (1024, 1), 0), reinterpret_tensor(buf115, (20, 20), (20, 1), 0), reinterpret_tensor(buf4, (50257, 1024), (1024, 1), 0), None, None, None, None, )


def benchmark_compiled_module(times=10, repeat=10):
    from torch._dynamo.testing import rand_strided
    from torch._inductor.utils import print_performance
    primals_6 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_11 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_12 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_13 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_18 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_23 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_24 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_25 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_30 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_35 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_36 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_37 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_42 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_47 = rand_strided((1024, ), (1, ), device='cuda:0', dtype=torch.float16)
    primals_60 = rand_strided((3, 2048), (2049, 1), device='cuda:0', dtype=torch.int64)
    primals_61 = rand_strided((3, 2048), (2049, 1), device='cuda:0', dtype=torch.int64)
    embedding = rand_strided((3, 2048, 1024), (2097152, 1024, 1), device='cuda:0', dtype=torch.float16)
    permute = rand_strided((1024, 1024), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_1 = rand_strided((1024, 1024), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_2 = rand_strided((1024, 1024), (1, 1024), device='cuda:0', dtype=torch.float16)
    select_1 = rand_strided((1, 2048, 1, 32), (0, 64, 0, 2), device='cuda:0', dtype=torch.float16)
    select_3 = rand_strided((1, 2048, 1, 32), (0, 64, 0, 2), device='cuda:0', dtype=torch.float16)
    slice_3 = rand_strided((1, 1, 2048, 2048), (4194304, 4194304, 2048, 1), device='cuda:0', dtype=torch.bool)
    scalar_tensor = rand_strided((), (), device='cuda:0', dtype=torch.float16)
    permute_10 = rand_strided((1024, 1024), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_11 = rand_strided((1024, 2816), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_12 = rand_strided((1024, 2816), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_13 = rand_strided((2816, 1024), (1, 2816), device='cuda:0', dtype=torch.float16)
    rsqrt_2 = rand_strided((3, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float16)
    view_30 = rand_strided((6144, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    mm_7 = rand_strided((6144, 8), (8, 1), device='cuda:0', dtype=torch.float16)
    view_32 = rand_strided((6144, 8), (8, 1), device='cuda:0', dtype=torch.float16)
    unsqueeze_8 = rand_strided((3, 2048, 1), (4096, 2, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_9 = rand_strided((3, 2048, 1), (4096, 2, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_10 = rand_strided((3, 2048, 1), (4096, 2, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_11 = rand_strided((3, 2048, 1), (4096, 2, 0), device='cuda:0', dtype=torch.float16)
    permute_17 = rand_strided((1024, 1024), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_18 = rand_strided((1024, 1024), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_19 = rand_strided((1024, 1024), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_27 = rand_strided((1024, 1024), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_28 = rand_strided((1024, 2816), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_29 = rand_strided((1024, 2816), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_30 = rand_strided((2816, 1024), (1, 2816), device='cuda:0', dtype=torch.float16)
    rsqrt_7 = rand_strided((3, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float16)
    view_65 = rand_strided((6144, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    mm_16 = rand_strided((6144, 12), (12, 1), device='cuda:0', dtype=torch.float16)
    view_67 = rand_strided((6144, 12), (12, 1), device='cuda:0', dtype=torch.float16)
    unsqueeze_18 = rand_strided((3, 2048, 1), (6144, 3, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_19 = rand_strided((3, 2048, 1), (6144, 3, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_20 = rand_strided((3, 2048, 1), (6144, 3, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_21 = rand_strided((3, 2048, 1), (6144, 3, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_22 = rand_strided((3, 2048, 1), (6144, 3, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_23 = rand_strided((3, 2048, 1), (6144, 3, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_24 = rand_strided((3, 2048, 1), (6144, 3, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_25 = rand_strided((3, 2048, 1), (6144, 3, 0), device='cuda:0', dtype=torch.float16)
    permute_34 = rand_strided((1024, 1024), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_35 = rand_strided((1024, 1024), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_36 = rand_strided((1024, 1024), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_44 = rand_strided((1024, 1024), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_45 = rand_strided((1024, 2816), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_46 = rand_strided((1024, 2816), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_47 = rand_strided((2816, 1024), (1, 2816), device='cuda:0', dtype=torch.float16)
    rsqrt_12 = rand_strided((3, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float16)
    view_100 = rand_strided((6144, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    mm_25 = rand_strided((6144, 16), (16, 1), device='cuda:0', dtype=torch.float16)
    view_102 = rand_strided((6144, 16), (16, 1), device='cuda:0', dtype=torch.float16)
    unsqueeze_32 = rand_strided((3, 2048, 1), (8192, 4, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_33 = rand_strided((3, 2048, 1), (8192, 4, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_34 = rand_strided((3, 2048, 1), (8192, 4, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_35 = rand_strided((3, 2048, 1), (8192, 4, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_36 = rand_strided((3, 2048, 1), (8192, 4, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_37 = rand_strided((3, 2048, 1), (8192, 4, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_38 = rand_strided((3, 2048, 1), (8192, 4, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_39 = rand_strided((3, 2048, 1), (8192, 4, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_40 = rand_strided((3, 2048, 1), (8192, 4, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_41 = rand_strided((3, 2048, 1), (8192, 4, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_42 = rand_strided((3, 2048, 1), (8192, 4, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_43 = rand_strided((3, 2048, 1), (8192, 4, 0), device='cuda:0', dtype=torch.float16)
    permute_51 = rand_strided((1024, 1024), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_52 = rand_strided((1024, 1024), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_53 = rand_strided((1024, 1024), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_61 = rand_strided((1024, 1024), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_62 = rand_strided((1024, 2816), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_63 = rand_strided((1024, 2816), (1, 1024), device='cuda:0', dtype=torch.float16)
    permute_64 = rand_strided((2816, 1024), (1, 2816), device='cuda:0', dtype=torch.float16)
    rsqrt_17 = rand_strided((3, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float16)
    view_135 = rand_strided((6144, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    mm_34 = rand_strided((6144, 20), (20, 1), device='cuda:0', dtype=torch.float16)
    view_137 = rand_strided((6144, 20), (20, 1), device='cuda:0', dtype=torch.float16)
    unsqueeze_54 = rand_strided((3, 2048, 1), (10240, 5, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_55 = rand_strided((3, 2048, 1), (10240, 5, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_56 = rand_strided((3, 2048, 1), (10240, 5, 0), device='cuda:0', dtype=torch.float16)
    unsqueeze_57 = rand_strided((3, 2048, 1), (10240, 5, 0), device='cuda:0', dtype=torch.float16)
    rsqrt_18 = rand_strided((3, 2048, 1), (2048, 1, 1), device='cuda:0', dtype=torch.float32)
    view_140 = rand_strided((6144, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_71 = rand_strided((6144, 50257), (50257, 1), device='cuda:0', dtype=torch.float16)
    convert_element_type_72 = rand_strided((), (), device='cuda:0', dtype=torch.float16)
    permute_71 = rand_strided((50257, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    permute_76 = rand_strided((20, 20), (20, 1), device='cuda:0', dtype=torch.float16)
    permute_80 = rand_strided((20, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    permute_125 = rand_strided((16, 16), (16, 1), device='cuda:0', dtype=torch.float16)
    permute_129 = rand_strided((16, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    permute_174 = rand_strided((12, 12), (12, 1), device='cuda:0', dtype=torch.float16)
    permute_178 = rand_strided((12, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    permute_223 = rand_strided((8, 8), (8, 1), device='cuda:0', dtype=torch.float16)
    permute_227 = rand_strided((8, 1024), (1024, 1), device='cuda:0', dtype=torch.float16)
    tangents_1 = rand_strided((), (), device='cuda:0', dtype=torch.float16)
    return print_performance(lambda: call([primals_6, primals_11, primals_12, primals_13, primals_18, primals_23, primals_24, primals_25, primals_30, primals_35, primals_36, primals_37, primals_42, primals_47, primals_60, primals_61, embedding, permute, permute_1, permute_2, select_1, select_3, slice_3, scalar_tensor, permute_10, permute_11, permute_12, permute_13, rsqrt_2, view_30, mm_7, view_32, unsqueeze_8, unsqueeze_9, unsqueeze_10, unsqueeze_11, permute_17, permute_18, permute_19, permute_27, permute_28, permute_29, permute_30, rsqrt_7, view_65, mm_16, view_67, unsqueeze_18, unsqueeze_19, unsqueeze_20, unsqueeze_21, unsqueeze_22, unsqueeze_23, unsqueeze_24, unsqueeze_25, permute_34, permute_35, permute_36, permute_44, permute_45, permute_46, permute_47, rsqrt_12, view_100, mm_25, view_102, unsqueeze_32, unsqueeze_33, unsqueeze_34, unsqueeze_35, unsqueeze_36, unsqueeze_37, unsqueeze_38, unsqueeze_39, unsqueeze_40, unsqueeze_41, unsqueeze_42, unsqueeze_43, permute_51, permute_52, permute_53, permute_61, permute_62, permute_63, permute_64, rsqrt_17, view_135, mm_34, view_137, unsqueeze_54, unsqueeze_55, unsqueeze_56, unsqueeze_57, rsqrt_18, view_140, convert_element_type_71, convert_element_type_72, permute_71, permute_76, permute_80, permute_125, permute_129, permute_174, permute_178, permute_223, permute_227, tangents_1]), times=times, repeat=repeat)


if __name__ == "__main__":
    from torch._inductor.wrapper_benchmark import compiled_module_main
    compiled_module_main('None', benchmark_compiled_module)
    print(f"Memory used: {torch.cuda.max_memory_reserved() / 1e9:.02f} GB")
