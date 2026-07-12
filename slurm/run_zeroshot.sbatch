#!/bin/bash
#SBATCH -J tirex2-zeroshot
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 04:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# GPU job: run a zero-shot foundation model (Chronos / TimesFM / Moirai) on the SAME held-out
# test windows as TiRex-2 and the trained baselines, writing baseline-<model>_<stem>.* so the
# existing compare.py / figures pick it up unchanged. Requires slurm/setup_tsfm.sbatch first.
#
#   MODEL=chronos sbatch slurm/run_zeroshot.sbatch                 # start here (ungated, fastest)
#   MODEL=timesfm sbatch slurm/run_zeroshot.sbatch
#   MODEL=moirai  sbatch slurm/run_zeroshot.sbatch
#   # phenylephrine cohort:
#   MODEL=chronos MATCH=results/ablation_windows_cases115_covpressor.csv COV=pressor \
#       CONFIG=datasets/vitaldb/configs/data_pressor.yaml sbatch slurm/run_zeroshot.sbatch
# compare.py (TiRex vs this model, classification head-to-head) auto-runs for COV=ce.
set -euo pipefail

MODEL="${MODEL:-chronos}"
COV="${COV:-ce}"
CONFIG="${CONFIG:-datasets/vitaldb/configs/data.yaml}"
MATCH="${MATCH:-results/ablation_windows_all2873.csv}"
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: submit with 'sbatch slurm/run_zeroshot.sbatch' (not 'bash')." >&2; exit 1
fi
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
[ -f "${PROJECT_ROOT}/requirements_gpu.txt" ] || [ ! -f "${PROJECT_ROOT}/../requirements_gpu.txt" ] || PROJECT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DEFAULT_SQSH="${HOME}/containers/tirex2.sqsh"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$( [ -f "$DEFAULT_SQSH" ] && echo "$DEFAULT_SQSH" || echo "nvcr.io#nvidia/pytorch:25.09-py3" )}"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results"
export SLURM_STEP_LAUNCH_TIMEOUT=600
echo "Project root : ${PROJECT_ROOT}"; echo "Container : ${CONTAINER_IMAGE}"
echo "MODEL=${MODEL} COV=${COV} CONFIG=${CONFIG} MATCH=${MATCH}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT='${PROJECT_ROOT}'
source '${PROJECT_ROOT}/slurm/setup_env.sh'
# The foundation weights are UNGATED and not yet cached -> allow network downloads
# (setup_env.sh forces HF_HUB_OFFLINE=1 for the gated TiRex repo; we don't load TiRex here).
unset HF_HUB_OFFLINE || true
VENV='${PROJECT_ROOT}/.tsfm-venv-${MODEL}'   # per-model isolated env (own torch, no conflicts)
if [ ! -f \"\$VENV/bin/activate\" ]; then
  echo \"[zs] ERROR: \$VENV missing — run: MODELS=${MODEL} sbatch slurm/setup_tsfm.sbatch\" >&2; exit 1
fi
source \"\$VENV/bin/activate\"
cd '${PROJECT_ROOT}'
echo '[job] zero-shot ${MODEL} cov=${COV} matched to ${MATCH}'
PYTHONPATH=scripts:datasets/vitaldb python scripts/baselines/zeroshot.py \
    --config ${CONFIG} --model ${MODEL} --cov ${COV} \
    --match-tirex ${MATCH} --device cuda --batch-size 256
STEM=\$(basename ${MATCH} | sed 's/ablation_windows_//;s/.csv//')
if [ '${COV}' = 'ce' ]; then
  echo '[job] matched classification comparison (TiRex vs ${MODEL})'
  PYTHONPATH=scripts:datasets/vitaldb python scripts/baselines/compare.py \
      --tirex \${STEM} --baseline baseline-${MODEL}_\${STEM}
fi
echo '[job] Done.'
"
