# ViT-UWA MLP ablation (Colab T4)

Evaluates a trained ViT-UWA-style detector (ViT-B backbone, ConvNeXt-V2 Pico branch, FreqFusion FPN,
Mask R-CNN, USIS10K) with block-wise MLP interventions and reports box / mask AP:

| intervention | meaning |
|---|---|
| `zero:i` | block *i*'s MLP update removed (compute skipped, so latency is real) |
| `mean:i` | replaced by its dataset-mean output (removes the token-specific part) |
| `prune:i,j@k` | keep the top-*k* fraction of hidden neurons (fc1 rows / fc2 columns physically sliced) |

This repository contains **only our additions**. The original ViT-UWA code is cloned from
[Linqirui/ViT-UWA](https://github.com/Linqirui/ViT-UWA) (Apache-2.0) and ours is copied over it.
No weights, data or tokens are stored here.

```
analysis/mlp_study/   mlp_ablate_eval.py (GPU mAP ablation), ckpt_compat.py, build_backbone.py,
                      mlp_probe.py / mlp_groups.py (CPU activation probes), data_utils.py
analysis/uwa_ext/     ViTUWAv2, DAMv2, FPNFreqFusion (evaluation only). freqfusion.py is downloaded
                      from the upstream FreqFusion repo by the setup script (it has no license file).
configs/              training config of the analysed checkpoint (paths sanitised)
colab/                T4 environment: Python 3.9, torch 1.9.0+cu111, mmcv-full 1.4.2, CUDA op build
```

## Before you start
1. Runtime, Change runtime type: **T4 GPU**.
2. Colab sidebar, **Secrets** (key icon): add `HF_TOKEN` (a Hugging Face *read* token with access to the
   checkpoint repo) and enable notebook access. Never paste tokens into cells.
3. Dataset: USIS10K as `MyDrive/USIS10K.zip` (a zip of the `USIS10K/` folder with `val/` and
   `multi_class_annotations/`) or already unpacked in `MyDrive/ViT-UWA-NAWB/datasets/USIS10K`.

## Cells
```python
# 1. Drive + code: original repo (pinned) with this repo copied over it
from google.colab import drive, userdata
drive.mount('/content/drive')
!git clone -q https://github.com/Linqirui/ViT-UWA.git /content/ViT-UWA
!git -C /content/ViT-UWA checkout -q 8cffc86bb6cb5fd2b77d3f5785e16286bd7a304b
!git clone -q https://github.com/riteshh-thakur/uwa.git /content/overlay
!cp -r /content/overlay/analysis /content/overlay/colab /content/overlay/configs /content/ViT-UWA/
R = '/content/ViT-UWA'
```
```python
# 2. Dataset (copied to local disk: much faster than reading from Drive)
import os
D = f'{R}/datasets/USIS10K'
if not os.path.isdir(f'{D}/val'):
    if os.path.isdir('/content/drive/MyDrive/ViT-UWA-NAWB/datasets/USIS10K/val'):
        !mkdir -p {R}/datasets && cp -r /content/drive/MyDrive/ViT-UWA-NAWB/datasets/USIS10K {R}/datasets/
    else:
        !mkdir -p {R}/datasets && unzip -q /content/drive/MyDrive/USIS10K.zip -d {R}/datasets
!ls {D}
```
```python
# 3. Environment (~12 min the first session, ~4 min after a restart)
!bash {R}/colab/setup_colab_t4.sh        # must end with: ENV OK
```
```python
# 4. Checkpoint (1.5 GB) from the private Hugging Face repo
from huggingface_hub import hf_hub_download
hf_hub_download('riteshh31/vit-uwa-b6-best-checkpoint', 'best_segm_mAP_epoch_29.pth',
                token=userdata.get('HF_TOKEN'), local_dir='/content/ckpt')
```
```python
# 5. Load test: base model on 50 images (~3 min). Must say "checkpoint loaded strictly".
!cd {R} && /content/uwa/bin/python analysis/mlp_study/mlp_ablate_eval.py \
    --config configs/b6_best_3x_config.py --checkpoint /content/ckpt/best_segm_mAP_epoch_29.pth \
    --data-root datasets/USIS10K --runs base --subset 50 --out /content/test_base.json
```
```python
# 6a. Gate (~15 min): base on ~1,494 val images (all but the 100 calibration images).
#     Observed here: box 45.1 / mask 44.0 (the checkpoint recorded 45.98 on the full val set).
#     Do NOT gate on small subsets: rare classes (robots n=46, aquatic plants n=69, sea-floor n=75) make
#     subset AP swing by several points (50 / 150 / 500 images gave mask AP 38.4 / 44.4 / 40.9).
!cd {R} && /content/uwa/bin/python analysis/mlp_study/mlp_ablate_eval.py \
    --config configs/b6_best_3x_config.py --checkpoint /content/ckpt/best_segm_mAP_epoch_29.pth \
    --data-root datasets/USIS10K --runs base --subset 0 --n-boot 2 \
    --out /content/drive/MyDrive/mlp_ablation_results/base_fullval.json
```
```python
# 6b. Short ablation (~1.5-2 h): late-block question + two early-block reference points, 800 images
!cd {R} && /content/uwa/bin/python analysis/mlp_study/mlp_ablate_eval.py \
    --config configs/b6_best_3x_config.py --checkpoint /content/ckpt/best_segm_mAP_epoch_29.pth \
    --data-root datasets/USIS10K --subset 800 --n-boot 30 \
    --runs base zero:8 zero:9 zero:10 zero:11 prune:8,9,10,11@0.5 prune:8,9,10,11@0.25 zero:9,10,11 zero:0 zero:5 \
    --out /content/drive/MyDrive/mlp_ablation_results/ablate_short_val800.json
```
```python
# 6c. Full ablation (all 31 configurations, ~6 h at 800 images; split it with --runs across sessions)
!cd {R} && /content/uwa/bin/python analysis/mlp_study/mlp_ablate_eval.py \
    --config configs/b6_best_3x_config.py --checkpoint /content/ckpt/best_segm_mAP_epoch_29.pth \
    --data-root datasets/USIS10K --subset 800 --n-boot 30 --runs default \
    --out /content/drive/MyDrive/mlp_ablation_results/ablate_val800.json
```
Results are written to Drive after every configuration, so a disconnect loses at most the run in progress.
After a restart run cells 1-4 again, then repeat the same cell 6x: finished runs are skipped. A markdown
table with 90 % paired-resample intervals is written next to the JSON.

## What the evaluation does
- Fixed random val subset for evaluation; a disjoint 100-image subset supplies the mean MLP outputs and the
  neuron ranking, so no evaluated image is used for fitting.
- Box and mask AP, AP50/AP75, AP small/medium/large, per-image latency (median, GPU-synchronised), and
  50 paired 80 % image resamples for confidence intervals on the change versus base.
- The analysed checkpoint was trained with training-code classes that are not public (`ViTUWAX`,
  `FreqFusionFPN`). `mlp_ablate_eval.py` maps its config onto `analysis/uwa_ext` and loads the weights through
  `ckpt_compat.remap_checkpoint` with a strict key and shape check; on a CPU-side check the mapping reproduces
  all 587 backbone and FreqFusion tensors exactly. The wiring of the forward pass cannot be checked without a GPU,
  which is what the cell 6a gate is for.

## Troubleshooting
- `checkpoint does not match the model`: the message lists the first missing / unexpected keys; share it.
- Setup error compiling the CUDA op: rerun cell 3; the compiled wheel is cached on Drive for later sessions.
- Base mask AP on the ~1,494-image gate far below 43: the reconstruction does not reproduce the trained model. Stop and report.
- `analysis/mlp_study/variant_sweep.py` tests reconstruction details that leave tensor shapes unchanged (fp16, FreqFusion options, projection norm).
