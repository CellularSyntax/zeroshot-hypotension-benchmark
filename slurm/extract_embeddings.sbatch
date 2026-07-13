#!/bin/bash
#SBATCH -J tirex2-embed
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -q a100
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -t 03:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#
# GPU job: extract per-window ENCODER embeddings from one zero-shot foundation model, on a
# seeded stratified subsample of the SAME windows as the matched TiRex-2 comparison. Feeds the
# supplementary representation-explainability analyses (UMAP / linear probes / RSA).
# Writes results/embeddings_<model>_<stem>.npz + results/embeddings_meta_<model>_<stem>.json.
#
# Mirrors slurm/run_zeroshot.sbatch exactly:
#   * chronos/timesfm/moirai  -> per-model isolated venv .tsfm-venv-<model> (run setup_tsfm first)
#   * tirex2                  -> the tirex2 container env (gated repo; keep HF_HUB_OFFLINE=1)
#
# Run ALL FOUR for both cohorts (8 jobs). VitalDB:
#   for M in tirex2 chronos timesfm moirai; do MODEL=$M sbatch slurm/extract_embeddings.sbatch; done
# MOVER (external): add MATCH + CONFIG + COV:
#   for M in tirex2 chronos timesfm moirai; do \
#     MODEL=$M MATCH=results/ablation_windows_mover_art.csv COV=mover_rate \
#     CONFIG=datasets/mover/configs/data.yaml sbatch slurm/extract_embeddings.sbatch; done
# Then pull results/embeddings_*.npz + results/embeddings_meta_*.json back to the Mac's results/.
set -euo pipefail

MODEL="${MODEL:-chronos}"
COV="${COV:-ce}"
CONFIG="${CONFIG:-datasets/vitaldb/configs/data.yaml}"
MATCH="${MATCH:-results/ablation_windows_all2873.csv}"
MAXWIN="${MAXWIN:-4000}"
if [ -z "${SLURM_JOB_ID:-}" ]; then
  echo "ERROR: submit with 'sbatch slurm/extract_embeddings.sbatch' (not 'bash')." >&2; exit 1
fi
PROJECT_ROOT="${PROJECT_ROOT:-$SLURM_SUBMIT_DIR}"
[ -f "${PROJECT_ROOT}/requirements_gpu.txt" ] || [ ! -f "${PROJECT_ROOT}/../requirements_gpu.txt" ] || PROJECT_ROOT="$(cd "${PROJECT_ROOT}/.." && pwd)"
DEFAULT_SQSH="${HOME}/containers/tirex2.sqsh"
CONTAINER_IMAGE="${CONTAINER_IMAGE:-$( [ -f "$DEFAULT_SQSH" ] && echo "$DEFAULT_SQSH" || echo "nvcr.io#nvidia/pytorch:25.09-py3" )}"
mkdir -p "${PROJECT_ROOT}/logs" "${PROJECT_ROOT}/results"
export SLURM_STEP_LAUNCH_TIMEOUT=600
echo "Project root : ${PROJECT_ROOT}"; echo "Container : ${CONTAINER_IMAGE}"
echo "MODEL=${MODEL} COV=${COV} CONFIG=${CONFIG} MATCH=${MATCH} MAXWIN=${MAXWIN}"

srun \
  --container-image="${CONTAINER_IMAGE}" \
  --container-mounts="${PROJECT_ROOT}:${PROJECT_ROOT}" \
  --container-workdir="${PROJECT_ROOT}" \
  bash -lc "
set -euo pipefail
export PROJECT_ROOT='${PROJECT_ROOT}'
source '${PROJECT_ROOT}/slurm/setup_env.sh'
cd '${PROJECT_ROOT}'
if [ '${MODEL}' = 'tirex2' ]; then
  # gated TiRex-2 repo: stay in the container env, keep HF_HUB_OFFLINE=1 (weights cached).
  echo '[emb] tirex2 in container env'
  PYTHONPATH=scripts:datasets/vitaldb:datasets/mover python scripts/baselines/extract_embeddings.py \
      --config ${CONFIG} --model tirex2 --cov ${COV} \
      --match-tirex ${MATCH} --max-windows ${MAXWIN} --device cuda --batch-size 64
else
  # ungated foundation weights -> per-model isolated venv + allow downloads
  unset HF_HUB_OFFLINE || true
  VENV='${PROJECT_ROOT}/.tsfm-venv-${MODEL}'
  if [ ! -f \"\$VENV/bin/activate\" ]; then
    echo \"[emb] ERROR: \$VENV missing — run: MODELS=${MODEL} sbatch slurm/setup_tsfm.sbatch\" >&2; exit 1
  fi
  source \"\$VENV/bin/activate\"
  PYTHONPATH=scripts:datasets/vitaldb:datasets/mover python scripts/baselines/extract_embeddings.py \
      --config ${CONFIG} --model ${MODEL} --cov ${COV} \
      --match-tirex ${MATCH} --max-windows ${MAXWIN} --device cuda --batch-size 64
fi
echo '[emb] Done.'
"
