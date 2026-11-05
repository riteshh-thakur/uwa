#!/usr/bin/env bash
# One-shot env setup for ViT-UWA on a Colab T4 runtime. Run after mounting Drive:
#
#   !bash /content/ViT-UWA/colab/setup_colab_t4.sh
#
# Builds a Python 3.9 venv at /content/uwa (Colab's own Python is 3.12, too new
# for mmcv-full 1.x / torch 1.9). The MSDeformAttn op wheel is cached on Drive:
# the first session installs the CUDA 11.8 toolkit (~5 min) to compile it, later
# sessions reuse the wheel and finish in ~3-4 min.
#
# Env: VENV (/content/uwa), CACHE (/content/drive/MyDrive/uwa_cache), REPO (this repo)
set -euo pipefail
REPO=${REPO:-$(cd "$(dirname "$0")/.." && pwd)}
VENV=${VENV:-/content/uwa}
CACHE=${CACHE:-/content/drive/MyDrive/uwa_cache}
CUDA118=/usr/local/cuda-11.8
PY=$VENV/bin/python
mkdir -p "$CACHE"

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader

# ---- Python 3.9 venv (deadsnakes: gcc-built, sane sysconfig for compiling the op) ----
if ! command -v python3.9 >/dev/null; then
  add-apt-repository -y ppa:deadsnakes/ppa >/dev/null
  apt-get -qq update
  apt-get -qq install -y python3.9 python3.9-dev python3.9-venv python3.9-distutils >/dev/null
fi
[[ -x $PY ]] || python3.9 -m venv "$VENV"
command -v uv >/dev/null || pip install -q uv
pip_() { uv pip install --python "$PY" "$@"; }

# CUDA 11.8 nvcc accepts gcc <= 11; newer Colab images may ship a newer gcc.
if (( $(gcc -dumpversion | cut -d. -f1) > 11 )); then
  apt-get -qq install -y gcc-11 g++-11 >/dev/null
  mkdir -p /content/gcc11
  for t in gcc:gcc-11 g++:g++-11 c++:g++-11 cc:gcc-11; do ln -sf "/usr/bin/${t#*:}" "/content/gcc11/${t%%:*}"; done
  export PATH=/content/gcc11:$PATH CC=gcc-11 CXX=g++-11
fi

# ---- torch / torchvision / mmcv-full (same builds as the A40 reproduction) ----
pip_ "setuptools==59.5.0" "wheel==0.41.3"   # newer wheel needs packaging>=24, which conflicts with the pinned packaging==23.2
pip_ "torch==1.9.0+cu111" "torchvision==0.10.0+cu111" -f https://download.pytorch.org/whl/torch_stable.html
pip_ -r "$REPO/colab/requirements_t4.txt"
pip_ "mmcv-full==1.4.2" -f https://download.openmmlab.com/mmcv/dist/cu111/torch1.9.0/index.html

# ---- FreqFusion (neck of the analysed checkpoint) and a timm with ConvNeXt-V2 -------------------
# The upstream FreqFusion repository ships no license file, so its code is downloaded here
# (pinned commit) instead of being redistributed in this repository.
FF=$REPO/analysis/uwa_ext/freqfusion.py
[[ -s $FF ]] || curl -fsSL https://raw.githubusercontent.com/Linwei-Chen/FreqFusion/3fb0c70637a3c194fb74294d3ce4681958b26241/FreqFusion.py -o "$FF"
# ConvNeXt-V2 needs timm >= 0.9; it goes into analysis/pylib, which the scripts put first on sys.path
# (the ViT code only needs timm.models.layers, which 0.9.16 still provides).
"$PY" -m pip install -q --no-deps --target "$REPO/analysis/pylib" timm==0.9.16

# ---- MSDeformAttn CUDA op ---------------------------------------------------
WHEEL=$(ls "$CACHE"/MultiScaleDeformableAttention-*-cp39-*.whl 2>/dev/null | head -1 || true)
if [[ -z $WHEEL ]]; then
  if [[ ! -x $CUDA118/bin/nvcc ]]; then
    echo "installing CUDA 11.8 toolkit (compiler only, for the op build)"
    RUN=/content/cuda_11.8.0_520.61.05_linux.run
    wget -q -O "$RUN" https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
    sh "$RUN" --silent --toolkit --installpath=$CUDA118 --no-opengl-libs --no-man-page --override
    rm -f "$RUN"
  fi
  ( cd "$REPO/detection/ops" && rm -rf build dist *.egg-info &&
    CUDA_HOME=$CUDA118 PATH=$CUDA118/bin:$PATH TORCH_CUDA_ARCH_LIST="7.5" FORCE_CUDA=1 \
      "$PY" setup.py bdist_wheel )
  cp "$REPO"/detection/ops/dist/MultiScaleDeformableAttention-*.whl "$CACHE"/
  WHEEL=$(ls "$CACHE"/MultiScaleDeformableAttention-*-cp39-*.whl | head -1)
fi
pip_ --reinstall "$WHEEL"

"$PY" "$REPO/colab/verify_env.py"
