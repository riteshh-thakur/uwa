"""ViTUWAv2: ViT-UWA without HFCP and with DAMv2 (pretrained ConvNeXt-V2).

Differences from ViTUWA (detection/mmdet_custom/models/backbones/vit_uwa.py):
  1. HFCP removed: the image goes through patch_embed once; no FFT / I_hfc.
     ViTUWA adds patch_embed(x) + patch_embed(hfc(x)), i.e. the patch-embed
     bias twice. To keep the same starting point, init_weights() multiplies
     patch_embed.proj.bias by 2 once, right after loading the pretrained ViT
     weights. It is skipped when the loaded file is already a ViTUWAv2
     checkpoint (contains dam.cnn.* keys). Resuming / load_from / testing
     load the full detector state dict after construction, which overwrites
     the bias with the trained value, so the doubling is never applied twice.
  2. self.dam is DAMv2 (same outputs as DetailAwareModule).
VDCF (self.interactions), level_embed, up, norm1-4 and forward() are
otherwise identical.
"""
import torch
import torch.nn.functional as F
from mmcv_custom import my_load_checkpoint as load_checkpoint
from mmdet.models.builder import BACKBONES
from mmdet.utils import get_root_logger
from mmdet_custom.models.backbones.uwa_modules import deform_inputs
from mmdet_custom.models.backbones.vit_uwa import ViTUWA

from .dam_v2 import DAMv2


@BACKBONES.register_module()
class ViTUWAv2(ViTUWA):

    def __init__(self, dam_cfg=None, double_patch_bias=True, **kwargs):
        # read by init_weights(), which TIMMVisionTransformer.__init__ calls
        self.double_patch_bias = double_patch_bias
        super().__init__(**kwargs)
        # ViTUWA.__init__ built the original DAM and applied _init_weights to
        # it. Replace it afterwards so the ConvNeXt weights are never touched
        # by _init_weights; only the new projections get the backbone init.
        del self.dam
        self.dam = DAMv2(embed_dim=self.embed_dim, **(dam_cfg or {}))
        self.dam.proj.apply(self._init_weights)

    def init_weights(self, pretrained=None):
        if not isinstance(pretrained, str):
            return
        logger = get_root_logger()
        ckpt = load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
        sd = ckpt
        if isinstance(ckpt, dict):
            sd = ckpt.get('state_dict', ckpt.get('model', ckpt))
        is_v2 = any(k.startswith(('dam.cnn.', 'backbone.dam.cnn.')) for k in sd)
        if self.double_patch_bias and not is_v2:
            with torch.no_grad():
                self.patch_embed.proj.bias.mul_(2)
            logger.info('ViTUWAv2: patch_embed.proj.bias *= 2 (replaces the HFCP double bias)')
        else:
            logger.info('ViTUWAv2: patch_embed.proj.bias left unchanged (v2 checkpoint=%s)', is_v2)

    def forward(self, x):
        deform_inputs1, deform_inputs2 = deform_inputs(x)

        # DAM
        c1, c2, c3, c4 = self.dam(x)
        c2, c3, c4 = self._add_level_embed(c2, c3, c4)
        c = torch.cat([c2, c3, c4], dim=1)

        # Patch Embedding (no HFCP)
        x, H, W = self.patch_embed(x)
        bs, n, dim = x.shape
        pos_embed = self._get_pos_embed(self.pos_embed[:, 1:], H, W)
        x = self.pos_drop(x + pos_embed)

        # Interaction
        for i, layer in enumerate(self.interactions):
            indexes = self.interaction_indexes[i]
            x, c = layer(x, c, self.blocks[indexes[0]:indexes[-1] + 1],
                         deform_inputs1, deform_inputs2, H, W)

        # Split & Reshape
        c2 = c[:, 0:c2.size(1), :]
        c3 = c[:, c2.size(1):c2.size(1) + c3.size(1), :]
        c4 = c[:, c2.size(1) + c3.size(1):, :]

        c2 = c2.transpose(1, 2).view(bs, dim, H * 2, W * 2).contiguous()
        c3 = c3.transpose(1, 2).view(bs, dim, H, W).contiguous()
        c4 = c4.transpose(1, 2).view(bs, dim, H // 2, W // 2).contiguous()
        c1 = self.up(c2) + c1

        if self.add_vit_feature:
            x3 = x.transpose(1, 2).view(bs, dim, H, W).contiguous()
            x1 = F.interpolate(x3, scale_factor=4, mode='bilinear', align_corners=False)
            x2 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
            x4 = F.interpolate(x3, scale_factor=0.5, mode='bilinear', align_corners=False)
            c1, c2, c3, c4 = c1 + x1, c2 + x2, c3 + x3, c4 + x4

        # Final Norm
        f1 = self.norm1(c1)
        f2 = self.norm2(c2)
        f3 = self.norm3(c3)
        f4 = self.norm4(c4)
        return [f1, f2, f3, f4]
