#!/bin/bash
#SBATCH -J tirex2-figures
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p cpu
#SBATCH -q cpu
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -t 01:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# CPU job: turn a finished ablation run into all paper figures + tables. Runs the
# post-hoc scripts (hypo_eval, clinical_eval, subgroup_forest, plot_kapral_mae) inside
# the container against results/ablation_windows_<TAG>.csv. No GPU, no model.
#
# Usage (after run_ablation.sbatch finished for that TAG):
#   TAG=all2873         sbatch slurm/make_figures.sbatch   # primary model — all figures
#   TAG=all2873_covrate sbatch slurm/make_figures.sbatch   # RATE covariate arm
#   TAG=cases115_covpressor sbatch slurm/make_figures.sbatch  # phenylephrine arm
# Optional: H=5 (subgroup-forest horizon, min), NBOOT=1000 (bootstrap reps).
set -euo pipefail

TAG="${TAG:-all2873}"
H="${H:-5}"
NBOOT="${NBOOT:-1000}"
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: submit this with 'sbatch slurm/make_figures.sbatch' (not 'bash')." >&2; exit 1
fi
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
[ -f "${PROJECT_ROOT}/requirements_gpu.txt" ] || [ ! -f "${PROJECT_ROOT}/../requirements_gpu.txt" ] || PROJECT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DEFAULT_SQSH="${HOME}/containers/tirex2.sqsh"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$( [ -f "$DEFAULT_SQSH" ] && echo "$DEFAULT_SQSH" || echo "nvcr.io#nvidia/pytorch:25.09-py3" )}"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/outputs/figs"
export SLURM_STEP_LAUNCH_TIMEOUT=600
# CPU node: skip enroot's NVIDIA hook (nvidia-container-cli absent on cpu partition).
export NVIDIA_VISIBLE_DEVICES=void
export NVIDIA_DRIVER_CAPABILITIES=void
echo "Project root : ${PROJECT_ROOT}"; echo "Container : ${CONTAINER_IMAGE}"; echo "TAG=${TAG} H=${H} NBOOT=${NBOOT}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT='${PROJECT_ROOT}'
source '${PROJECT_ROOT}/slurm/setup_env.sh'
cd '${PROJECT_ROOT}'
if [ ! -f results/ablation_windows_${TAG}.csv ] && [ -z \"\$(ls results/ablation_windows_${TAG}_sh*of*.csv 2>/dev/null)\" ]; then
  echo \"ERROR: no results/ablation_windows_${TAG}.csv — did run_ablation.sbatch finish for TAG=${TAG}?\" >&2; exit 1
fi
export MPLBACKEND=Agg
echo '[fig] hypotension ROC / PR / calibration / operating points + foil table ...'
PYTHONPATH=scripts:datasets/vitaldb python scripts/hypo_eval.py ${TAG} ${NBOOT}
echo '[fig] clinical: lead time / severity gradient / decision curve ...'
PYTHONPATH=scripts:datasets/vitaldb python scripts/clinical_eval.py ${TAG}
echo '[fig] subgroup forest (AUROC by sex/age/ASA/dept/duration) at h=${H} ...'
PYTHONPATH=scripts:datasets/vitaldb python scripts/subgroup_forest.py ${TAG} ${H} ${NBOOT}
echo '[fig] MAE vs Kapral et al. overlay ...'
PYTHONPATH=scripts:datasets/vitaldb python scripts/plot_kapral_mae.py ${TAG}
echo '[fig] Done. Figures in outputs/figs/*_${TAG}.png ; tables/metrics in results/*_${TAG}.*'
"
