"""Evaluation-only extensions: registers BACKBONES.ViTUWAv2 and NECKS.FPNFreqFusion.
Requires the original ViT-UWA repo's detection/ on sys.path (mmdet, mmdet_custom, mmcv_custom, ops) and
uwa_ext/freqfusion.py, which colab/setup_colab_t4.sh downloads from the upstream FreqFusion repository
(it is not redistributed here)."""
import mmcv_custom  # noqa: F401
import mmdet_custom  # noqa: F401

from .dam_v2 import DAMv2
from .fpn_freqfusion import FPNFreqFusion
from .vit_uwa_v2 import ViTUWAv2

__all__ = ['DAMv2', 'ViTUWAv2', 'FPNFreqFusion']
