#!/bin/bash
#SBATCH -J tirex2-revanalyses
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p cpu
#SBATCH -q cpu
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH -t 02:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# CPU job (no GPU, no model): reviewer-response analyses that need the full per-case cache
# and the project's numpy/scikit-learn (only present inside the container, not on login nodes).
#   * scripts/mover_cohort_table.py  -> results/mover_cohort_characteristics.json  (Table 1, MOVER column)
#   * scripts/naive_baselines.py     -> results/naive_metrics_{all2873,mover_art}.json
#                                       + results/ablation_windows_naive-persist_{tag}.csv
# Reuses the caches built by build_cache.sbatch; reads results/ablation_windows_*.csv for the
# ground-truth windows so the naive floor is scored on IDENTICAL windows as every other model.
#
# Submit from project root:  sbatch slurm/reviewer_analyses.sbatch
set -euo pipefail

if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: submit this with 'sbatch slurm/reviewer_analyses.sbatch' (not 'bash')." >&2; exit 1
fi
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
[ -f "${PROJECT_ROOT}/requirements_gpu.txt" ] || [ ! -f "${PROJECT_ROOT}/../requirements_gpu.txt" ] || PROJECT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DEFAULT_SQSH="${HOME}/containers/tirex2.sqsh"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$( [ -f "$DEFAULT_SQSH" ] && echo "$DEFAULT_SQSH" || echo "nvcr.io#nvidia/pytorch:25.09-py3" )}"
mkdir -p "${PROJECT_ROOT}/logs"
export SLURM_STEP_LAUNCH_TIMEOUT=600
# CPU-only job: skip enroot's NVIDIA hook (CPU nodes lack nvidia-container-cli).
export NVIDIA_VISIBLE_DEVICES=void
export NVIDIA_DRIVER_CAPABILITIES=void
echo "Project root : ${PROJECT_ROOT}"; echo "Container : ${CONTAINER_IMAGE}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
cd '${PROJECT_ROOT}'
echo '[rev] python:' \$(python -V)

echo '[rev] 1/3  MOVER cohort characteristics (Table 1) ...'
PYTHONPATH=scripts:datasets/mover python scripts/mover_cohort_table.py

echo '[rev] 2/3  naive baselines on VitalDB (identical windows) ...'
PYTHONPATH=scripts:datasets/vitaldb python scripts/naive_baselines.py all2873 \
    --cache datasets/vitaldb/cache --clinical datasets/vitaldb/data/clinical_data.csv --write-windows

echo '[rev] 3/3  naive baselines on MOVER (identical windows) ...'
PYTHONPATH=scripts:datasets/mover python scripts/naive_baselines.py mover_art \
    --cache datasets/mover/cache --clinical datasets/mover/clinical_data.csv --write-windows

echo '[rev] done. Pull results/naive_metrics_*.json + results/mover_cohort_characteristics.json to the Mac.'
"
