#!/usr/bin/env bash
# Rebuild every DERIVED artifact from the raw per-window CSVs synced off the cluster.
#
# Why this exists: the cluster is authoritative for the RAW files it produces
#   ablation_windows_*.csv  ablation_primary_*.json  baseline_meta_*.json
#   baseline_history_*.json  baseline_ckpt_*.pt  hypo_metrics_*.json  clinical_eval_*.json
# but the DERIVED files are computed LOCALLY:
#   results/matched_comparison_*.json   (compare.py, all-cases OOF)
#   results/tables/*                    (paper_figures / stats_tests / transfer_figure)
#   outputs/figs/paper/*                results_bundle.{tex,pdf}
# An rsync/scp of the whole results/ folder OVERWRITES the derived files with the cluster's
# (often stale) copies. Rather than fight the sync, treat the window CSVs as the single source
# of truth and regenerate everything derived here. Idempotent — safe to run after every sync.
#
# Usage:  bash scripts/rebuild_local.sh            # rebuild + compile PDF
#         SKIP_PDF=1 bash scripts/rebuild_local.sh # skip pdflatex
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="scripts:datasets/vitaldb${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python3}

echo "==> 1/5  matched_comparison_*.json  (compare.py, all-cases OOF, from window CSVs)"
# One matched comparison per baseline window file present. Tag format: baseline-<model>_<cohort>;
# the matching TiRex cohort is <cohort> (its ablation_windows_<cohort>.csv must exist).
# The comparisons are independent -> run them in parallel batches (each is a case-clustered
# bootstrap, CPU-bound). Cap concurrency with MAXJ (default 4) to bound memory (big CSVs).
MAXJ=${MAXJ:-4}
seen=""; pending=""
for f in results/ablation_windows_baseline-*.csv; do
  [ -e "$f" ] || continue
  bt=$(basename "$f" .csv); bt=${bt#ablation_windows_}
  bt=${bt%_sh*of*}                                   # collapse any shards to the base tag
  case " $seen " in *" $bt "*) continue;; esac
  seen="$seen $bt"
  cohort=${bt#baseline-*_}                            # strip 'baseline-<model>_' -> cohort (e.g. all2873, mover_art)
  if [ ! -e "results/ablation_windows_${cohort}.csv" ] && ! ls results/ablation_windows_${cohort}_sh*of*.csv >/dev/null 2>&1; then
    echo "    [skip] $bt : no TiRex windows for cohort '$cohort'"; continue
  fi
  pending="$pending ${cohort}=${bt}"
done
n=0
for pair in $pending; do
  cohort=${pair%%=*}; bt=${pair#*=}
  echo "    compare  tirex=$cohort  baseline=$bt"
  $PY scripts/baselines/compare.py --tirex "$cohort" --baseline "$bt" >/dev/null &
  n=$((n + 1))
  [ $((n % MAXJ)) -eq 0 ] && wait     # barrier every MAXJ launches (bash 3.2-safe, no wait -n)
done
wait

echo "==> 2/5  figures + tables  (paper_figures.py)"
$PY scripts/paper_figures.py >/dev/null

echo "==> 3/5  significance tests  (stats_tests.py)"
$PY scripts/stats_tests.py >/dev/null

echo "==> 4/5  cross-dataset transfer figure + table  (transfer_figure.py)"
$PY scripts/transfer_figure.py >/dev/null

echo "==> 5/5  results bundle  (make_results_bundle.py)"
$PY scripts/make_results_bundle.py >/dev/null
if [ "${SKIP_PDF:-0}" != "1" ] && command -v pdflatex >/dev/null 2>&1; then
  pdflatex -interaction=nonstopmode results_bundle.tex >/dev/null 2>&1
  pdflatex -interaction=nonstopmode results_bundle.tex >/dev/null 2>&1
  echo "    wrote results_bundle.pdf"
else
  echo "    (skipped pdflatex — run: pdflatex results_bundle.tex)"
fi
echo "==> done. All derived artifacts rebuilt from the raw window CSVs."
