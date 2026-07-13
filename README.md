# Zero-shot time-series foundation models for intraoperative hypotension prediction

Code and derived results for the manuscript:

> **Can a zero-shot time-series foundation model rival task-trained models for intraoperative
> hypotension prediction? A two-cohort benchmark and the role of covariate-awareness.**
> Max Haberbusch, Medical University of Vienna.

We benchmark four **zero-shot** time-series foundation models (TiRex-2, Chronos-Bolt, TimesFM-2.5,
Moirai-1.1-R) against two **task-trained** baselines (Temporal Fusion Transformer, PatchTST) for
forecasting mean arterial pressure (MAP) and predicting impending hypotension (MAP < 65 mmHg) over
1–15 min. The foundation models are applied with **no task-specific training and no labels**.
Development is on **VitalDB** (2,708 cases); external validation is on the independent **MOVER**
cohort (1,827 cases). The two cohorts are always reported stratified, never pooled.

---

## Reproduce every figure and table — one notebook

Everything a reviewer needs to verify the paper is in **[`reproduce_paper.ipynb`](reproduce_paper.ipynb)**.
It regenerates **every main and supplementary figure** and prints **every table's numbers** from the
released result files. **No model is retrained and no foundation-model inference is run** — the notebook
reads the per-window forecasts and aggregate metrics we provide and rebuilds the figures/tables from them.

```bash
# 1. create an environment (Python 3.11+)
pip install -r requirements.txt

# 2. get the released result files (see "Where the data lives" below), then:
jupyter notebook reproduce_paper.ipynb        # Kernel -> Restart & Run All
```

The notebook's final cell prints a figure/table → result-file provenance map for auditability.

## Where the data (result files) lives

The `results/` directory required by the notebook (~1.4 GB: per-window forecasts, aggregate metrics,
embeddings, precomputed tables) is archived on **Zenodo** and released with the paper (DOI added on
acceptance). Download it and unpack into the repository root so `results/` sits next to
`reproduce_paper.ipynb`. This GitHub repository is the **living codebase**; the Zenodo record is the
**citable, frozen snapshot** of code + results at publication.

The raw source datasets are **not** redistributed here (data-use terms):

- **VitalDB** — openly available at <https://vitaldb.net> and via PhysioNet.
- **MOVER** — public-access, from the UCI Machine Learning Repository.

Reproducing the figures/tables does **not** require the raw datasets — only the released `results/`.
Re-running the models from raw data (optional) is documented in [`notes/REPRODUCE.md`](notes/REPRODUCE.md).

## Repository layout

```
reproduce_paper.ipynb     one-shot figure/table regeneration (start here)
requirements.txt          runtime dependencies for the notebook
scripts/                  figure/table generators + shared forecasting-pipeline code
  paper_figures.py          Fig 1–5, several tables
  transfer_figure.py        Fig 6 + transfer table
  decision_curves_figure.py FigS decision curves
  external_table.py         external-validation table
  stats_tests.py            paired significance tests table
  compute_footprint.py      per-model compute footprint (GPU; measurement only)
  compute_footprint_table.py aggregates footprint JSON -> table (stdlib, no GPU)
  explainability/           representation-analysis figures (UMAP, RSA/CKA)
  baselines/                TFT/PatchTST models, training, zero-shot adapters
datasets/vitaldb/         VitalDB loader, cohort builder, configs, DATA_NOTES.md
datasets/mover/           MOVER loader + configs
configs/eval.yaml         shared evaluation protocol (horizons, threshold, bootstrap)
slurm/                    cluster job scripts for the heavy runs (training, inference, embeddings)
manuscript/               LaTeX source, figures, compiled PDF
notes/REPRODUCE.md        end-to-end recipe to regenerate results from raw data
notes/CLUSTER.md          how the heavy runs execute on the SLURM/A100 cluster
MOVER_SCHEMA_REPORT.md    MOVER schema and how it maps onto the VitalDB pipeline
```

Run pipeline scripts with `PYTHONPATH=scripts:datasets/vitaldb:datasets/mover` so the shared pipeline
finds the dataset loaders.

## Cluster (optional — re-running the heavy steps)

Model training, zero-shot inference, embedding extraction and the compute-footprint measurement run on
an A100 GPU via SLURM + a Pyxis/enroot container. See [`notes/CLUSTER.md`](notes/CLUSTER.md) and the
scripts in `slurm/`.

## License

Code is released under the license in [`LICENSE.txt`](LICENSE.txt). Released result files are
distributed under CC BY 4.0. The raw VitalDB and MOVER datasets remain under their respective terms.
