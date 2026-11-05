"""Build the trained ViT-UWA backbone without mmcv-full / the CUDA op (CPU, MPS or CUDA).

Stubs the compiled MSDeformAttn extension (routes it through the PyTorch reference
implementation in detection/ops) and the mmdet/mmcv imports that the backbone files
only need for registration and checkpoint loading. Weights are loaded directly from
the detector checkpoint ('backbone.*' keys), strictly.
"""
import importlib.util
import os.path as osp
import sys
import types

import torch
import torch.nn as nn

ROOT = osp.abspath(osp.join(osp.dirname(__file__), '..', '..'))
DET = osp.join(ROOT, 'detection')
BB = osp.join(DET, 'mmdet_custom', 'models', 'backbones')

from ckpt_compat import official_convnext_to_timm  # noqa: E402

IMAGENET_NORM = dict(mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375])   # train pipeline
USIS_NORM = dict(mean=[81.236, 113.761, 117.095], std=[60.598, 58.471, 62.821])     # dumped test pipeline


def _stub(name, **attrs):
    m = sys.modules.get(name) or types.ModuleType(name)
    for k, v in attrs.items():
        setattr(m, k, v)
    sys.modules[name] = m
    return m


def _install_stubs():
    class _Registry:
        def register_module(self, *a, **k):
            return (lambda c: c) if not a or not isinstance(a[0], type) else a[0]

    _stub('mmcv'), _stub('mmcv.runner', BaseModule=nn.Module)
    _stub('mmcv_custom', my_load_checkpoint=None)
    _stub('mmdet'), _stub('mmdet.utils', get_root_logger=lambda *a, **k: None)
    _stub('mmdet.models'), _stub('mmdet.models.builder', BACKBONES=_Registry())
    _stub('MultiScaleDeformableAttention')
    if DET not in sys.path:
        sys.path.insert(0, DET)
    from ops.functions import ms_deform_attn_func as F

    class _PyMSDA:
        @staticmethod
        def apply(value, shapes, level_start_index, loc, attn, im2col_step):
            return F.ms_deform_attn_core_pytorch(value, shapes, loc, attn)

    import ops.modules.ms_deform_attn as M
    M.MSDeformAttnFunction = _PyMSDA


def _load(pkg, name, path):
    spec = importlib.util.spec_from_file_location(f'{pkg}.{name}', path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[f'{pkg}.{name}'] = m
    spec.loader.exec_module(m)
    return m


def build_vituwa(cfg_path, ckpt_path, device='cpu'):
    """cfg_path: dumped mmdet config (.py, plain literals). Returns (backbone in eval mode, backbone cfg)."""
    _install_stubs()
    pkg = 'uwabb'
    for p, path in ((pkg, BB), (pkg + '.base', osp.join(BB, 'base'))):
        m = types.ModuleType(p)
        m.__path__ = [path]
        sys.modules[p] = m
    _load(pkg, 'base.vit', osp.join(BB, 'base', 'vit.py'))
    _load(pkg, 'uwa_modules', osp.join(BB, 'uwa_modules.py'))
    vit_uwa = _load(pkg, 'vit_uwa', osp.join(BB, 'vit_uwa.py'))

    ns = {}
    exec(open(cfg_path).read(), ns)
    cfg = dict(ns['model']['backbone'])
    btype = cfg.pop('type')
    cfg.pop('pretrained', None)
    for k in ('hfcp', 'dam', 'dam_lr_mult'):  # ViTUWAX options: HFCP off, ConvNeXt-V2 Pico branch
        cfg.pop(k, None)

    sd = None
    if ckpt_path is not None:
        ck = torch.load(ckpt_path, map_location='cpu')
        sd = ck.get('state_dict', ck)
    is_v2 = btype in ('ViTUWAX', 'ViTUWAv2') or (sd is not None and any(k.startswith('backbone.dam.cnn.') for k in sd))
    if is_v2:
        # ViTUWAv2 (analysis/uwa_ext): ConvNeXt-V2 DAM, no HFCP. Load its files directly, without the
        # uwa_ext package __init__ (which needs mmdet); weights come from the checkpoint.
        ext = osp.join(ROOT, 'analysis', 'uwa_ext')
        pk = types.ModuleType('uwaext')
        pk.__path__ = [ext]
        sys.modules['uwaext'] = pk
        sys.modules['mmdet_custom'] = types.ModuleType('mmdet_custom')
        for n in ('mmdet_custom.models', 'mmdet_custom.models.backbones'):
            sys.modules[n] = types.ModuleType(n)
        sys.modules['mmdet_custom.models.backbones.uwa_modules'] = sys.modules[f'{pkg}.uwa_modules']
        sys.modules['mmdet_custom.models.backbones.vit_uwa'] = vit_uwa
        _load('uwaext', 'dam_v2', osp.join(ext, 'dam_v2.py'))
        v2 = _load('uwaext', 'vit_uwa_v2', osp.join(ext, 'vit_uwa_v2.py'))
        # projection input widths in the checkpoint (64/128/256/512) identify ConvNeXt-V2 Pico
        model = v2.ViTUWAv2(dam_cfg=dict(model_name='convnextv2_pico.fcmae_ft_in1k', pretrained=False), **cfg)
    else:
        model = vit_uwa.ViTUWA(**cfg)
    if str(device).startswith('mps'):  # MPS has no complex FFT: run HFCP's fft on CPU
        fft = model.fft
        model.fft = lambda x, rate: fft(x.cpu(), rate).to(x.device)
    if ckpt_path is None:  # random weights: pipeline smoke tests only
        return model.to(device).eval(), cfg

    bsd = {k[len('backbone.'):]: v for k, v in sd.items() if k.startswith('backbone.')}
    if is_v2:
        bsd = official_convnext_to_timm(bsd)
    missing, unexpected = model.load_state_dict(bsd, strict=False)
    # layer-scale gammas are created with value 1 and are part of the trained checkpoint;
    # anything still missing means the checkpoint and architecture disagree
    if missing or unexpected:
        raise RuntimeError(f'checkpoint mismatch: missing={missing[:8]} unexpected={unexpected[:8]}')
    return model.to(device).eval(), cfg
