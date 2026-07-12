# Reproducing all results (VitalDB)

End-to-end recipe to regenerate every number, table, and figure in the paper from raw data.
Two environments: **cluster** (SLURM + GPU, for the heavy runs) and **local Mac** (CPU, fine for
the post-hoc figures/tables). All commands run from the repo root with:

```bash
export PYP="PYTHONPATH=scripts:datasets/vitaldb"      # shared pipeline finds the dataset loader
```

Tags used throughout: `all2873` = anesthetic cohort, remi+propofol **CE** covariate (primary);
`all2873_covrate` = same cohort, **RATE** covariate; `cases115_covpressor` = phenylephrine subset.
Baselines are tagged `baseline-<model>_<cohort>` (e.g. `baseline-tft_all2873`).

---

## 0. One-time setup (cluster)

```bash
export HF_TOKEN=hf_xxx                       # gated NX-AI/TiRex-2 weights (put in ~/.bashrc)
bash slurm/download_vitalfiles.sh            # .vital + clinical_data.csv -> datasets/vitaldb/data/
bash slurm/build_container.sh                # one-time Pyxis image (deps + weights + kernels baked)
sbatch slurm/build_cache.sbatch              # CPU: loader caches + cohort_manifest.csv (the long pole)
```
Produces: `datasets/vitaldb/cache/` (anesthetic), `datasets/vitaldb/cache_pressor/` (phenylephrine),
`datasets/vitaldb/cohort_manifest.csv`, `results/cohort_flow.json`.

---

## 1. TiRex-2 zero-shot runs (GPU) — the primary results + covariate ablation

```bash
COV=ce      sbatch slurm/run_ablation.sbatch     # -> results/ablation_{windows,primary}_all2873.{csv,json}
COV=rate    sbatch slurm/run_ablation.sbatch     # -> ..._all2873_covrate.*
COV=pressor sbatch slurm/run_ablation.sbatch     # -> ..._cases115_covpressor.*
# or all three with auto-dependency after the cache job:
bash slurm/submit_all.sh
```
Each writes per-window forecasts (`ablation_windows_*.csv`, phase3 schema) + a summary
(`ablation_primary_*.json`) with MAE/CRPS, the covariate effect **X%** (M0→M1) and **Y%** (vs
persistence), stratified all/transition/steady, with case-clustered CIs.

Zero-shot ⇒ no training. The full cohort runs on one A100 in ~1–2 h each.

---

## 2. Matched supervised baselines (GPU) — TFT & PatchTST

Trained on the **same windows / subject-splits / metrics** as TiRex (canonical 60/20/20 split),
so the comparison is apples-to-apples. Each job trains M1 (with drug covariate) and M0 (without).

```bash
# --- classification head-to-head (CE cohort); auto-runs compare.py ---
MODEL=tft       COV=ce sbatch slurm/train_baseline.sbatch   # -> baseline-tft_all2873.*  + matched_comparison_*.json
MODEL=patchtst  COV=ce sbatch slurm/train_baseline.sbatch   # -> baseline-patchtst_all2873.* + matched_comparison_*.json

# --- covariate-representation parity for Fig 2c (RATE + phenylephrine arms) ---
MODEL=tft COV=rate    sbatch slurm/train_baseline.sbatch     # -> baseline-tft_all2873_covrate.*
MODEL=tft COV=pressor MATCH=results/ablation_windows_cases115_covpressor.csv \
    CONFIG=datasets/vitaldb/configs/data_pressor.yaml \
    sbatch slurm/train_baseline.sbatch                       # -> baseline-tft_cases115_covpressor.*
```
Each writes `ablation_windows_<baseline-tag>.csv` (test split, phase3 schema),
`baseline_history_<tag>.json` (train/val loss curves), `baseline_meta_<tag>.json`, and for COV=ce
`matched_comparison_<baseline-tag>.json` (TiRex vs baseline on identical test subjects + foils).

Add more architectures by registering them in `scripts/baselines/models.py` (`MODELS` dict) and
submitting with `MODEL=<name>`.

---

## 2b. Zero-shot foundation-model sweep (GPU) — other TSFMs as foils

Benchmarks other pretrained time-series foundation models **zero-shot** (no training) on the SAME
held-out test windows as TiRex and the trained baselines. Unlike TiRex, these models don't natively
ingest the known future drug trajectory, so they run **covariate-blind** (one forecast per window,
reused for the M1/M0 columns — their covariate effect is 0 by construction, which is the point).
Output is tagged `baseline-<model>_<stem>` → `compare.py` and the figures pick them up unchanged.

**Licenses / HF access** — all three are ungated (only the HF_TOKEN you already have is needed;
no "agree to license" click is required to pull weights):

| Model | HF repo | License | Covariates | pip |
|---|---|---|---|---|
| Chronos-Bolt | `amazon/chronos-bolt-base` | Apache-2.0 | none (univariate) | `chronos-forecasting` |
| TimesFM 2.5 | `google/timesfm-2.5-200m-pytorch` | Apache-2.0 | none (univariate) | `timesfm[torch]` |
| Moirai-1.1-R | `Salesforce/moirai-1.1-R-large` | CC-BY-NC-4.0 (research OK) | any-variate¹ | `uni2ts` |

¹ Moirai *can* take covariates, but the per-window path in `zeroshot.py` currently runs it
univariate too — verify its adapter on the cluster before a full sweep (see note below).

Each model gets its **own fully-isolated venv** (`.tsfm-venv-<model>`, shipping its own torch) —
the three libraries pin mutually-incompatible torch/numpy versions (uni2ts wants torch 2.4, which
clobbers everything if shared), so they must not share an environment.

```bash
# --- one-time: build a separate isolated venv per model (~30 min all three; disk ~15 GB) ---
sbatch slurm/setup_tsfm.sbatch                     # all three
# or one at a time (rebuild / iterate):
MODELS=chronos sbatch slurm/setup_tsfm.sbatch

# --- run the sweep (start with chronos: ungated, fastest, no covariate fuss) ---
MODEL=chronos sbatch slurm/run_zeroshot.sbatch     # -> baseline-chronos_all2873.* + matched_comparison_*
MODEL=timesfm sbatch slurm/run_zeroshot.sbatch     # -> baseline-timesfm_all2873.*
MODEL=moirai  sbatch slurm/run_zeroshot.sbatch     # -> baseline-moirai_all2873.*  (verify first)

# phenylephrine cohort (same as the trained baselines):
MODEL=chronos MATCH=results/ablation_windows_cases115_covpressor.csv COV=pressor \
    CONFIG=datasets/vitaldb/configs/data_pressor.yaml sbatch slurm/run_zeroshot.sbatch
```
Each writes `ablation_windows_baseline-<model>_<stem>.csv` (test split, phase3 schema),
`baseline_meta_<tag>.json`, and for COV=ce `matched_comparison_baseline-<model>_<stem>.json`.
Weights download once into `.hf_cache` on first run. **Run `chronos` first** — it validates the
whole slot-in with the least dependency risk; if the venv build clobbered torch or an import
fails, that's where it'll show, and it's a quick fix before spending GPU on the others.

Register a new foundation model by adding an adapter to the `ADAPTERS` dict in
`scripts/baselines/zeroshot.py` (implement `load()` + `forecast()`); everything downstream is
model-agnostic.

---

## 3. Pull results to the Mac (for figures/tables)

`outputs/` and the big `ablation_windows_*.csv` are git-ignored, so copy them off the cluster:
```bash
scp -r <cluster>:~/tirex-2/tirex-vitaldb/results ./            # JSONs, matched comparisons, windows CSVs, histories
```
(The small manifest, foil tables, and kapral curves are already in the repo.)

---

## 4. Post-hoc analyses (CPU) — feeds Fig 3/4 & Tables 3–5

Run locally with a Python that has numpy/scipy/pandas/matplotlib (or inside the container on the
cluster via `slurm/make_figures.sbatch`). These read the windows CSVs:
```bash
$PYP python scripts/hypo_eval.py       all2873      # hypotension ROC/PR/calibration/operating-points/pAUROC
$PYP python scripts/clinical_eval.py   all2873      # lead time / severity gradient / decision curves
$PYP python scripts/subgroup_forest.py all2873 5    # subgroup AUROC forest @5 min
$PYP python scripts/plot_kapral_mae.py all2873      # MAE vs Kapral overlay (standalone)
# on the cluster instead: TAG=all2873 sbatch slurm/make_figures.sbatch
```

Matched comparison (re-scores TiRex + a baseline on the identical canonical test split):
```bash
$PYP python scripts/baselines/compare.py --tirex all2873 --baseline baseline-tft_all2873
$PYP python scripts/baselines/compare.py --tirex all2873 --baseline baseline-patchtst_all2873
```

---

## 5. Paper figures + tables (CPU)

One command builds everything:
```bash
$PYP python scripts/paper_figures.py all2873
```
Outputs (Nature-style, PDF + 600-dpi PNG in `outputs/figs/paper/`, tables in `results/tables/`):

| Artifact | Content |
|---|---|
| Fig 1 | study design, cohort funnel, example forecasts |
| Fig 2 | (a) forecast accuracy TiRex vs TFT; (b) covariate value by window type; (c) CE/RATE/pressor forest; (d) MAE vs Kapral; (e) covariate exploitation by model class (0% zero-shot / ~1% TiRex / ~9–14% trained) |
| Fig 3 | **zero-shot TSFM benchmark (headline)** — TiRex vs Chronos/TimesFM/Moirai: (a) AUROC, (b) CRPS, (c) calibration @10 min, (d) AUPRC vs horizon |
| Fig 4 | hypotension vs supervised SOTA: ROC, AUROC-vs-horizon, calibration, PR, decision curve, head-to-head bars — TiRex vs TFT vs PatchTST vs foils |
| Fig 5 | clinical translation: lead time, severity gradient, subgroup forest, operating characteristics |
| Fig S | TFT + PatchTST M1/M0 training curves |
| Table 1 | cohort characteristics (n=2,708 windows-contributing) |
| Table 2 | forecast accuracy + covariate value (TiRex) |
| Table 3 | hypotension classification vs foils |
| Table 4 | matched classification AUROC (TiRex vs TFT vs PatchTST vs foils) |
| Table 5 | matched forecasting CRPS/MAE (TiRex vs TFT vs PatchTST) |
| Table 6 | zero-shot TSFM AUROC (TiRex vs Chronos/TimesFM/Moirai) |

`paper_figures.py` degrades gracefully: panels/tables that need a baseline (Fig 2c/e overlays,
Fig 3/4 baseline curves, Tables 4/5/6, Fig S) are drawn only if the corresponding files are present.

Significance tests + one-document bundle:
```bash
$PYP python scripts/stats_tests.py all2873        # -> results/tables/TableS_stats.* (paired case-clustered bootstrap)
python scripts/make_results_bundle.py             # -> results_bundle.tex (all figs+tables+stats)
pdflatex results_bundle.tex                        # -> results_bundle.pdf (run twice, from repo root)
```

---

## Dependencies between steps
```
setup(0) ── cache ──> TiRex runs(1) ──> post-hoc(4) ─┐
                          └────────────> baselines(2) ┴─> figures/tables(5)
```
Regenerating figures after any run is just step 5 (seconds). The heavy steps (1–2) are
deterministic given the seed, so reruns reproduce identical numbers.

## 6. MOVER external validation + cross-dataset generalization

MOVER support lives in `datasets/mover/`. The loader maps SIS columns onto VitalDB **canonical
channel names** (`HRe`→`Solar8000/HR`, `Propofol  drip` rate→`Orchestra/PPF20_RATE`, …), and the
pipeline is loader-agnostic (`phase3_ablation.get_loader()` reads the `loader:` key in the config),
so every script runs unchanged with the MOVER config + a `mover_*` covariate preset.

MOVER target = invasive `MAP_ART` (1-min, 60 s grid); cohort ≈ 1,866 (arterial ≥30 min + infusion),
827 with a pressor. Split is case-level (`PID`; SIS has no patient ID). Covariate = derived
propofol+remifentanil **rate** (`mover_rate`) or phenylephrine (`mover_pressor`).

> **Ordering / the incremental-CSV trap.** `run_ablation` writes `ablation_windows_mover_art.csv`
> *incrementally* (per chunk). Any job that `--match`es it (zero-shot sweep, CV baselines) must wait
> for TiRex to FINISH — watch for `=== SHARD DONE tag=mover_art ... ===` — or chain with
> `sbatch --dependency=afterok:<tirex_jobid> …`. Otherwise it locks onto a partial cohort.

```bash
# ── M0. one-time MOVER cache (CPU) ───────────────────────────────────────────────────
sbatch slurm/build_mover_cache.sbatch
#   inspect one case:  PYTHONPATH=datasets/mover python datasets/mover/mover_loader.py <PID>

# ── M1. TiRex-2 zero-shot on MOVER (GPU) — produces mover_art.csv (the cohort others match) ──
COV=mover_rate    sbatch slurm/run_ablation.sbatch     # -> results/*_mover_art.*
COV=mover_pressor sbatch slurm/run_ablation.sbatch     # phenylephrine -> *_mover_pressor.*

# ── M2. zero-shot TSFM sweep on MOVER (GPU; CHAIN behind M1) ──────────────────────────
for M in chronos timesfm moirai; do
  MODEL=$M MATCH=results/ablation_windows_mover_art.csv COV=mover_rate \
    CONFIG=datasets/mover/configs/data.yaml \
    sbatch --dependency=afterok:<M1_jobid> slurm/run_zeroshot.sbatch
done
```

### 6a. Internal 5-fold subject CV (robust in-distribution eval — out-of-fold over ALL cases)
```bash
# VitalDB (all2873.csv is complete -> run anytime). Overwrites baseline-<model>_all2873 with OOF.
MODEL=tft      FOLDS=5 sbatch slurm/train_baseline.sbatch
MODEL=patchtst FOLDS=5 sbatch slurm/train_baseline.sbatch
# MOVER (CHAIN behind M1):
for M in tft patchtst; do
  MODEL=$M COV=mover_rate CONFIG=datasets/mover/configs/data.yaml FOLDS=5 \
    MATCH=results/ablation_windows_mover_art.csv \
    sbatch --dependency=afterok:<M1_jobid> slurm/train_baseline.sbatch
done
```
`FOLDS=K` writes `ablation_windows_<tag>.csv` covering ALL cases (each in its held-out fold) +
per-fold curves in `baseline_history_<tag>.json`; `FOLDS=1` (default) = single 60/20/20 split.

### 6b. Cross-dataset transfer — M0 covariate-free, both datasets harmonized to 60 s
`ALLTRAIN=1` trains on the full cohort (manifest; independent of `mover_art.csv`) and saves a
checkpoint; `cross_eval` applies it to the other dataset. Aborts on any cadence/channel mismatch.
```bash
# --- transfer-source checkpoints (run anytime; per model) ---
MODEL=tft      COV=rate       CONFIG=datasets/vitaldb/configs/data_h60.yaml ALLTRAIN=1 CKPT=results/baseline_ckpt_vitaldb60_tft.pt      sbatch slurm/train_baseline.sbatch
MODEL=patchtst COV=rate       CONFIG=datasets/vitaldb/configs/data_h60.yaml ALLTRAIN=1 CKPT=results/baseline_ckpt_vitaldb60_patchtst.pt sbatch slurm/train_baseline.sbatch
MODEL=tft      COV=mover_rate CONFIG=datasets/mover/configs/data.yaml       ALLTRAIN=1 CKPT=results/baseline_ckpt_mover_tft.pt         sbatch slurm/train_baseline.sbatch
MODEL=patchtst COV=mover_rate CONFIG=datasets/mover/configs/data.yaml       ALLTRAIN=1 CKPT=results/baseline_ckpt_mover_patchtst.pt    sbatch slurm/train_baseline.sbatch

# --- transfer both directions (per model); writes xfer-<model>_<A>TO<B> ---
CKPT=results/baseline_ckpt_vitaldb60_tft.pt CONFIG=datasets/mover/configs/data.yaml       COV=mover_rate TAG=xfer-tft_vitaldb60TOmover_art sbatch slurm/cross_eval.sbatch
CKPT=results/baseline_ckpt_mover_tft.pt     CONFIG=datasets/vitaldb/configs/data_h60.yaml COV=rate       TAG=xfer-tft_moverTOvitaldb60   sbatch slurm/cross_eval.sbatch
# (+ the two patchtst directions)
```

### 6c. Pull down + figures
```bash
scp -r <cluster>:~/tirex-2/tirex-vitaldb/results ./
# MOVER external-validation figure set (Fig 3/4/5 re-run on the MOVER tag; VitalDB-only foils dropped):
$PYP python scripts/paper_figures.py mover_art        # (once the figure scripts are MOVER-aware — see NOTE)
```
> **NOTE (figure integration — not automatic yet).** `paper_figures.py` is currently hardwired to
> VitalDB (`TAG=all2873`, Kapral/Zhu foils, single-split `canonical_test_subjects`). To include MOVER
> and use the 5-fold CV / transfer results we need: (i) switch the matched comparison to all-cases OOF
> + report fold SD; (ii) drop VitalDB-only foils when `tag=mover_art`; (iii) a NEW cross-dataset figure
> (train×test transfer matrix + zero-shot-on-both). Tracked as the next figure-build task.

## Adding another dataset
Drop it in as `datasets/<name>/` (a loader exposing `load_config`/`_clinical_index`/`load_case` that
returns the canonical-keyed record, + config with `loader: <name>_loader`), then steps 1–5 run
unchanged — the pipeline in `scripts/` is dataset-agnostic.
