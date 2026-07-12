#!/bin/bash
#SBATCH -J tirex2-tsfm-setup
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# ONE-TIME: build a SEPARATE, fully-isolated Python venv for each zero-shot foundation model.
# Each venv ships its OWN torch (NO --system-site-packages) so the three mutually-incompatible
# dependency stacks never collide — the earlier single shared venv broke because uni2ts pins
# torch 2.4 and downgraded the container torch, wrecking chronos + torchvision ABI.
#
# The venvs live on the mounted filesystem and are reused by slurm/run_zeroshot.sbatch.
# Runtime code only needs numpy + torch + pyyaml, all pulled by the model libs (+ pyyaml).
#
#   sbatch slurm/setup_tsfm.sbatch                     # builds all three (~30 min, disk ~15 GB)
#   MODELS=chronos sbatch slurm/setup_tsfm.sbatch      # or just one (rebuild / iterate)
set -euo pipefail

# CPU node: skip the enroot NVIDIA hook (nvidia-container-cli absent on the cpu partition).
export NVIDIA_VISIBLE_DEVICES=void
export NVIDIA_DRIVER_CAPABILITIES=void

MODELS="${MODELS:-chronos timesfm moirai}"
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: submit with 'sbatch slurm/setup_tsfm.sbatch' (not 'bash')." >&2; exit 1
fi
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
[ -f "${PROJECT_ROOT}/requirements_gpu.txt" ] || [ ! -f "${PROJECT_ROOT}/../requirements_gpu.txt" ] || PROJECT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DEFAULT_SQSH="${HOME}/containers/tirex2.sqsh"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$( [ -f "$DEFAULT_SQSH" ] && echo "$DEFAULT_SQSH" || echo "nvcr.io#nvidia/pytorch:25.09-py3" )}"
mkdir -p "${PROJECT_ROOT}/logs"
echo "Project root : ${PROJECT_ROOT}"; echo "Container : ${CONTAINER_IMAGE}"; echo "MODELS=${MODELS}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
echo '[tsfm] container python:' \$(python -V)
# pip spec per model: the model lib pulls its own torch/numpy; we add pyyaml for our loader.
pip_spec() {
  case \"\$1\" in
    chronos) echo 'chronos-forecasting pyyaml' ;;
    timesfm) echo 'timesfm[torch] pyyaml' ;;
    moirai)  echo 'uni2ts pyyaml' ;;
    *) echo 'UNKNOWN' ;;
  esac
}
import_check() {
  case \"\$1\" in
    chronos) python -c 'from chronos import BaseChronosPipeline; print(\"  chronos OK\")' ;;
    timesfm) python -c 'import timesfm; print(\"  timesfm OK\")' ;;
    moirai)  python -c 'from uni2ts.model.moirai import MoiraiForecast, MoiraiModule; print(\"  moirai OK\")' ;;
  esac
}
for M in ${MODELS}; do
  SPEC=\$(pip_spec \"\$M\")
  [ \"\$SPEC\" = 'UNKNOWN' ] && { echo \"[tsfm] skip unknown model \$M\"; continue; }
  VENV=\"${PROJECT_ROOT}/.tsfm-venv-\$M\"
  echo \"[tsfm] ===== \$M -> \$VENV =====\"
  rm -rf \"\$VENV\"
  python -m venv \"\$VENV\"                 # FULLY isolated (no --system-site-packages)
  source \"\$VENV/bin/activate\"
  python -m pip install --upgrade pip --quiet
  echo \"[tsfm] pip install \$SPEC\"
  python -m pip install \$SPEC || { echo \"[tsfm] \$M install FAILED\"; deactivate; continue; }
  echo \"[tsfm] \$M torch:\" \$(python -c 'import torch; print(torch.__version__, torch.version.cuda)')
  import_check \"\$M\" || echo \"[tsfm] \$M import FAILED\"
  deactivate
done
echo '[tsfm] done. venvs: '\$(ls -d ${PROJECT_ROOT}/.tsfm-venv-* 2>/dev/null)
"
