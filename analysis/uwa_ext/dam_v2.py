"""DAMv2: pretrained ConvNeXt-V2 spatial branch with the same interface as DAM.

The original DetailAwareModule (detection/mmdet_custom/models/backbones/
uwa_modules.py) returns, for an input of size H x W (padded to a multiple
of 32):
    c1: (B, D, H/4,  W/4)   2-D map, fused into the final 1/4 pyramid level
    c2: (B, H/8  * W/8,  D) tokens  \
    c3: (B, H/16 * W/16, D) tokens   > concatenated as VDCF keys/values
    c4: (B, H/32 * W/32, D) tokens  /
with D = embed_dim. DAMv2 returns exactly this structure. HFDConv (the 3x3
projection to D) is replaced by a 1x1 conv + GroupNorm(32, D) per level.
"""
import logging

import timm
import torch
import torch.nn as nn

_logger = logging.getLogger(__name__)

IMAGENET_MEAN = (123.675, 116.28, 103.53)
IMAGENET_STD = (58.395, 57.12, 57.375)


class LayerNorm2d(nn.LayerNorm):
    """LayerNorm over the channel dimension of (B, C, H, W) maps."""

    def __init__(self, dim, eps=1e-6):
        super().__init__(dim, eps=eps)

    def forward(self, x):
        return super().forward(x.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)


class DAMv2(nn.Module):

    def __init__(self, embed_dim=768, model_name='convnextv2_atto.fcmae_ft_in1k', proj_norm='gn32',
                 pretrained=True, cnn_checkpoint=None,
                 input_mean=IMAGENET_MEAN, input_std=IMAGENET_STD):
        """
        Args:
            embed_dim: ViT width (768 for B, 192 for T).
            model_name: timm model name.
            pretrained: load pretrained ConvNeXt weights.
            cnn_checkpoint: optional local file (timm or official
                facebookresearch format) used instead of the hub download.
            input_mean / input_std: normalization (0-255 scale) of the images
                the backbone receives, i.e. img_norm_cfg of the config. If it
                differs from the ConvNeXt pretraining statistics, the input is
                re-normalized inside DAMv2.
        """
        super().__init__()
        kwargs = dict(features_only=True, out_indices=(0, 1, 2, 3))
        if pretrained and cnn_checkpoint:
            kwargs['pretrained_cfg_overlay'] = dict(file=cnn_checkpoint)
        self.cnn = timm.create_model(model_name, pretrained=pretrained, **kwargs)
        self.weights_source = (cnn_checkpoint or 'timm/hf-hub:' + model_name) if pretrained else 'random-init'
        chs = self.cnn.feature_info.channels()
        reductions = self.cnn.feature_info.reduction()
        assert list(reductions) == [4, 8, 16, 32], reductions
        # proj_norm: 'gn32' GroupNorm(32, D) | 'gn1' GroupNorm(1, D) | 'ln2d' LayerNorm over channels (eps 1e-6).
        # All have the same parameter names and shapes, so a strict checkpoint load cannot tell them apart.
        norms = dict(gn32=lambda: nn.GroupNorm(32, embed_dim), gn1=lambda: nn.GroupNorm(1, embed_dim),
                     ln2d=lambda: LayerNorm2d(embed_dim))
        self.proj = nn.ModuleList(
            nn.Sequential(nn.Conv2d(c, embed_dim, 1), norms[proj_norm]())
            for c in chs)

        # x_cnn = x * a + b maps the backbone's normalization onto the
        # ConvNeXt pretraining normalization (identity when they match).
        cfg = self.cnn.pretrained_cfg
        cnn_mean = torch.tensor(cfg.get('mean', (0.485, 0.456, 0.406))) * 255.
        cnn_std = torch.tensor(cfg.get('std', (0.229, 0.224, 0.225))) * 255.
        in_mean, in_std = torch.tensor(input_mean), torch.tensor(input_std)
        self.renorm = not (torch.allclose(in_mean, cnn_mean, atol=1e-3)
                           and torch.allclose(in_std, cnn_std, atol=1e-3))
        self.register_buffer('a', (in_std / cnn_std).view(1, 3, 1, 1), persistent=False)
        self.register_buffer('b', ((in_mean - cnn_mean) / cnn_std).view(1, 3, 1, 1), persistent=False)
        _logger.info('DAMv2: %s, channels %s, weights %s, renorm %s',
                     model_name, chs, self.weights_source, self.renorm)

        # inference-only analysis switch: subset of {1, 2, 3, 4} to zero
        self.zero_levels = set()

    def forward(self, x):
        if self.renorm:
            x = x * self.a + self.b
        feats = self.cnn(x)
        c1, c2, c3, c4 = [p(f) for p, f in zip(self.proj, feats)]
        if self.zero_levels:
            c1, c2, c3, c4 = [torch.zeros_like(c) if i + 1 in self.zero_levels else c
                              for i, c in enumerate((c1, c2, c3, c4))]
        bs, dim, _, _ = c1.shape
        c2 = c2.view(bs, dim, -1).transpose(1, 2)  # 8s
        c3 = c3.view(bs, dim, -1).transpose(1, 2)  # 16s
        c4 = c4.view(bs, dim, -1).transpose(1, 2)  # 32s
        return c1, c2, c3, c4
