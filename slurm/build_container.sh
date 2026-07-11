#!/bin/bash
# One-time build of the TiRex-2 Pyxis SquashFS container (mirrors jaxfibers pattern).
#
# Starts from an NVIDIA PyTorch NGC image (CUDA torch + nvcc), pip-installs
# requirements_gpu.txt, PRE-DOWNLOADS the gated NX-AI/TiRex-2 weights, and
# PRE-COMPILES the sLSTM CUDA kernels for all target GPU arches — so every
# subsequent job starts in seconds with deps + weights + kernels in place.
# Saves to $HOME/containers/tirex2.sqsh; all run_*.sbatch auto-detect it.
#
# IMPORTANT: pick a BASE_IMAGE whose torch is 2.8<=torch<2.10 (tirex-2 0.1.1).
#   NGC tag -> torch:  25.06-py3 ~ 2.8 | 25.08/25.09-py3 ~ 2.9  (verify before build).
# The HF weights + compiled kernels are written to the MOUNTED project dir
# (.hf_cache / .torch_ext), not into the image, so they persist and stay writable.
#
# Usage (from project root, with HF_TOKEN exported for the gated weights):
#   export HF_TOKEN=hf_xxx
#   bash slurm/build_container.sh
#
# Overrides: BASE_IMAGE, OUT, PARTITION/QOS/GRES, PROJECT_ROOT, TIME_LIMIT.
set -euo pipefail

# Resolve repo root from this script's location (slurm/ lives in the repo root), so it works
# regardless of the invoking directory.
cd "$(dirname "${BASH_SOURCE[0]}")/.." || { echo "cannot cd to repo root" >&2; exit 1; }
PROJECT_ROOT="${PROJECT_ROOT:-$(pwd)}"
OUT="${OUT:-${HOME}/containers/tirex2.sqsh}"
BASE_IMAGE="${BASE_IMAGE:-nvcr.io#nvidia/pytorch:25.09-py3}"
PARTITION="${PARTITION:-gpu}"
QOS="${QOS:-a100}"
GRES="${GRES:-gpu:a100:1}"
TIME_LIMIT="${TIME_LIMIT:-1:00:00}"

mkdir -p "$(dirname "${OUT}")"
if [[ -f "${OUT}" ]]; then
  echo "Already exists: ${OUT}  (rm it first to rebuild)"; exit 0
fi
if [[ ! -f "${PROJECT_ROOT}/requirements_gpu.txt" ]]; then
  echo "ERROR: requirements_gpu.txt not found in ${PROJECT_ROOT}" >&2; exit 1
fi
if [[ -z "${HF_TOKEN:-}" ]]; then
  echo "ERROR: export HF_TOKEN=hf_xxx first (needed to pre-download gated NX-AI/TiRex-2)." >&2; exit 1
fi

echo "[build] Output      : ${OUT}"
echo "[build] Base image  : ${BASE_IMAGE}"
echo "[build] GPU (arch)  : ${GRES}  (kernels cross-compiled for 8.0;8.6;9.0;10.0)"
echo ""

export SLURM_STEP_LAUNCH_TIMEOUT=600   # cold Pyxis pulls exceed the 32 s default

srun \
  --partition="${PARTITION}" --qos="${QOS}" --gres="${GRES}" \
  --cpus-per-task=8 --mem=128G -t "${TIME_LIMIT}" \
  --container-image="${BASE_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  --container-save="${OUT}" \
  bash -lc "
set -euo pipefail
echo '[build:in-container] '\$(python -V)
echo '[build:in-container] torch: '\$(python -c 'import torch;print(torch.__version__, torch.version.cuda)')
pip install --upgrade pip --quiet
unset PIP_CONSTRAINT
pip install --upgrade -r '${PROJECT_ROOT}/requirements_gpu.txt'
export HF_HOME='${PROJECT_ROOT}/.hf_cache'
export HF_TOKEN='${HF_TOKEN}'
export TORCH_EXTENSIONS_DIR='${PROJECT_ROOT}/.torch_ext'
export TORCH_CUDA_ARCH_LIST='8.0;8.6;9.0;10.0'
mkdir -p \"\$HF_HOME\" \"\$TORCH_EXTENSIONS_DIR\"
echo '[build:in-container] Pre-downloading gated NX-AI/TiRex-2 weights ...'
python -c \"from huggingface_hub import snapshot_download; snapshot_download('NX-AI/TiRex-2')\"
echo '[build:in-container] Pre-compiling sLSTM kernels + smoke forecast ...'
python -c \"import torch, numpy as np; from tirex2 import load_model, TimeseriesType; \
m=load_model('NX-AI/TiRex-2', device='cuda'); \
ts=TimeseriesType(target=torch.randn(1,120), past_covariates=None, future_covariates=None); \
print('smoke forecast ok:', np.asarray(m.forecast([ts])[0]).shape)\" \
  || echo '[build:in-container] WARNING: kernel pre-compile/smoke skipped — kernels will compile on first job.'
echo '[build:in-container] pip list (key):'
pip list 2>/dev/null | grep -iE 'tirex|torch|vitaldb|flashrnn|mlstm|xlstm|numpy|pandas' || true
echo '[build:in-container] Done. Pyxis will snapshot to SquashFS ...'
"

echo ""; echo "[build] Verifying ..."; ls -lh "${OUT}"
echo "[build] Done. run_*.sbatch will auto-detect ${OUT}."
