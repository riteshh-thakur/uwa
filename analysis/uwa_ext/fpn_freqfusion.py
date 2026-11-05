"""FPNFreqFusion: mmdet FPN with the 3 top-down "upsample + add" steps
replaced by FreqFusion (Chen et al., TPAMI 2024). Lateral convs, output
convs, extra levels and num_outs are mmdet's FPN, unchanged.

Fusion follows the official integration (FreqFusionCARAFEFPN in the
FreqFusion repo): _, hr, lr = ff(hr_feat=lateral[i-1], lr_feat=lateral[i]);
lateral[i-1] = hr + lr.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.runner import auto_fp16
from mmdet.models.builder import NECKS
from mmdet.models.necks import FPN

from .freqfusion import FreqFusion

try:  # the vendored copy used to define this label; the unmodified upstream file does not
    from .freqfusion import CARAFE_IMPL
except ImportError:
    CARAFE_IMPL = 'mmcv'

FF_DEFAULTS = dict(compressed_channels=64, lowpass_kernel=5, highpass_kernel=3,
                   feature_resample=False)  # the checkpoint analysed here was trained with feature_resample=True


@NECKS.register_module()
class FPNFreqFusion(FPN):

    def __init__(self, *args, ff_cfg=None, ff_fp32=True, **kwargs):
        """
        ff_cfg: FreqFusion kwargs (defaults: FF_DEFAULTS, the code defaults).
        ff_fp32: run FreqFusion in fp32 under fp16 training (the official
            code never runs it under autocast, and mmcv's carafe is a custom
            op that autocast does not cast).
        """
        super().__init__(*args, **kwargs)
        cfg = dict(FF_DEFAULTS, **(ff_cfg or {}))
        n = self.backbone_end_level - self.start_level - 1
        self.freq_fusions = nn.ModuleList(
            FreqFusion(hr_channels=self.out_channels, lr_channels=self.out_channels, **cfg)
            for _ in range(n))
        self.ff_fp32 = ff_fp32
        self.carafe_impl = CARAFE_IMPL
        # inference-only analysis switch: indices of fusion steps replaced by
        # nearest upsample + add (0 = P3->P2, 1 = P4->P3, 2 = P5->P4)
        self.plain_steps = set()

    def init_weights(self):
        # FPN's init_cfg applies Xavier to every Conv2d in the neck, including
        # FreqFusion's kernel encoders; restore FreqFusion's own init after.
        super().init_weights()
        for m in self.freq_fusions:
            m.init_weights()

    def _fuse(self, idx, hr, lr):
        if idx in self.plain_steps:
            return hr + F.interpolate(lr, size=hr.shape[2:], **self.upsample_cfg)
        ff = self.freq_fusions[idx]
        if self.ff_fp32 and hr.is_cuda:
            with torch.cuda.amp.autocast(enabled=False):
                _, hr_out, lr_out = ff(hr_feat=hr.float(), lr_feat=lr.float())
        else:
            _, hr_out, lr_out = ff(hr_feat=hr, lr_feat=lr)
        return hr_out + lr_out

    @auto_fp16()
    def forward(self, inputs):
        """Forward function."""
        assert len(inputs) == len(self.in_channels)

        # build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # build top-down path
        used_backbone_levels = len(laterals)
        for i in range(used_backbone_levels - 1, 0, -1):
            laterals[i - 1] = self._fuse(i - 1, laterals[i - 1], laterals[i])

        # build outputs
        # part 1: from original levels
        outs = [
            self.fpn_convs[i](laterals[i]) for i in range(used_backbone_levels)
        ]
        # part 2: add extra levels
        if self.num_outs > len(outs):
            # use max pool to get more levels on top of outputs
            # (e.g., Faster R-CNN, Mask R-CNN)
            if not self.add_extra_convs:
                for i in range(self.num_outs - used_backbone_levels):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            # add conv layers on top of original feature maps (RetinaNet)
            else:
                if self.add_extra_convs == 'on_input':
                    extra_source = inputs[self.backbone_end_level - 1]
                elif self.add_extra_convs == 'on_lateral':
                    extra_source = laterals[-1]
                elif self.add_extra_convs == 'on_output':
                    extra_source = outs[-1]
                else:
                    raise NotImplementedError
                outs.append(self.fpn_convs[used_backbone_levels](extra_source))
                for i in range(used_backbone_levels + 1, self.num_outs):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))
        return tuple(outs)
