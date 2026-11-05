"""Colab env check: versions, GPU, mmdet<->mmcv guard, MSDeformAttn CUDA op
vs. its PyTorch reference (fp32, and under autocast as in fp16 training).

    /content/uwa/bin/python colab/verify_env.py
"""
import os.path as osp
import sys

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..'))
sys.path.insert(0, osp.join(ROOT, 'detection'))

import cv2
import mmcv
import numpy as np
import timm
import torch
import torchvision

print(f'python {sys.version.split()[0]} | torch {torch.__version__} | torchvision {torchvision.__version__} | '
      f'mmcv {mmcv.__version__} | timm {timm.__version__} | numpy {np.__version__} | cv2 {cv2.__version__}')
assert torch.cuda.is_available(), 'no GPU: Runtime > Change runtime type > T4 GPU'
cap = torch.cuda.get_device_capability(0)
props = torch.cuda.get_device_properties(0)
print(f'GPU {props.name}, sm_{cap[0]}{cap[1]}, {props.total_memory / 2**30:.1f} GiB')

from mmcv.ops import RoIAlign, nms  # noqa: E402  (compiled mmcv-full ops)

import mmdet  # noqa: E402  (vendored 2.28.1; asserts mmcv version range)
import mmdet_custom  # noqa: E402,F401
print(f'mmdet {mmdet.__version__} (vendored)')

b = torch.rand(8, 4, device='cuda') * 50
b[:, 2:] += b[:, :2]
nms(b, torch.rand(8, device='cuda'), 0.5)
RoIAlign(7, 1 / 16.)(torch.rand(1, 8, 32, 32, device='cuda'),
                     torch.tensor([[0, 1., 1., 20., 20.]], device='cuda'))

from ops.functions.ms_deform_attn_func import MSDeformAttnFunction, ms_deform_attn_core_pytorch  # noqa: E402

torch.manual_seed(0)
N, M, D, Lq, L, P = 2, 6, 32, 300, 3, 4
shapes = torch.as_tensor([(64, 80), (32, 40), (16, 20)], dtype=torch.long, device='cuda')
start = torch.cat((shapes.new_zeros(1), shapes.prod(1).cumsum(0)[:-1]))
S = int(shapes.prod(1).sum())
value = torch.rand(N, S, M, D, device='cuda') * 0.01
loc = torch.rand(N, Lq, M, L, P, 2, device='cuda')
attn = torch.rand(N, Lq, M, L, P, device='cuda') + 1e-5
attn /= attn.sum((-1, -2), keepdim=True)
ref = ms_deform_attn_core_pytorch(value, shapes, loc, attn)
out = MSDeformAttnFunction.apply(value, shapes, start, loc, attn, 64)
err = (out - ref).abs().max().item()
with torch.cuda.amp.autocast():
    out16 = MSDeformAttnFunction.apply(value.half(), shapes, start, loc.half(), attn.half(), 64)
err16 = (out16.float() - ref).abs().max().item()
print(f'MSDeformAttn max|cuda - ref|: fp32 {err:.2e}, autocast {err16:.2e}')
assert err < 1e-5 and err16 < 1e-3, 'MSDeformAttn op mismatch'
print('ENV OK')
