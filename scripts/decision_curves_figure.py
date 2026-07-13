"""Supplementary decision-curve analysis figure (FigS_decision_curves).
import os
Net benefit of the isotonic-recalibrated zero-shot TiRex-2 hypotension risk versus the two
default strategies (treat-all, treat-none) at 1/5/15 min, for VitalDB (development, top row)
and MOVER (external, bottom row). The isotonic recalibration map is fit on each cohort's own
disjoint 20% development split (seeded, subject-level where a clinical file is available) and
applied to the held-out test split, so the curves are honest within each cohort. This mirrors
the calibration/transport analysis in the Results (recalibration is fit locally per cohort).

Reads the per-window prediction CSVs (results/ablation_windows_{all2873,mover_art}.csv) and,
for VitalDB, the clinical file for the subject-level split. Writes
outputs/figs/paper/FigS_decision_curves.{pdf,png,svg} via paper_style.save_fig.

Run from the project root:
    PYTHONPATH=scripts:datasets/vitaldb python scripts/decision_curves_figure.py
"""
from __future__ import annotations
import csv, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
import paper_style as S

RES = "results"
# subject-level split source: prefer the shipped 2-column crosswalk, then the full VitalDB
# clinical table; None -> cohort_split_col falls back to case-level (documented in the bundle README)
CL_V = next((p for p in ("results/vitaldb_case_subject_map.csv",
                         "datasets/vitaldb/data/clinical_data.csv") if os.path.exists(p)), None)
HORIZONS = [1, 5, 15]
PTS = np.linspace(0.02, 0.35, 34)          # clinically plausible alarm-threshold range
COLORS = {"model": "#0173B2", "all": "#949494"}   # focal blue (TiRex-2) + grey, match house palette


def norm_caseid(c):
    s = str(c).strip()
    d = s[1:] if s[:1] in "+-" else s
    return str(int(s)) if d.isdigit() else s


def cohort_split_col(windows_csv, clinical=None):
    """Add a boolean _dev column marking a seeded 20% development split. Split is subject-level
    when a clinical file maps caseid->subjectid (VitalDB), else case-level (MOVER)."""
    df = pd.read_csv(windows_csv)
    if clinical:
        c2s = {}
        for r in csv.DictReader(open(clinical, encoding="utf-8-sig")):
            try:
                c2s[norm_caseid(r["caseid"])] = str(r.get("subjectid", r["caseid"]))
            except KeyError:
                pass
        subj = [c2s.get(norm_caseid(c), norm_caseid(c)) for c in df.caseid]
    else:
        subj = [norm_caseid(c) for c in df.caseid]
    subs = sorted(set(subj))
    rng = np.random.default_rng(0)
    rng.shuffle(subs)
    dev = set(subs[:max(1, int(round(len(subs) * 0.2)))])
    df["_dev"] = [s in dev for s in subj]
    return df


def dca_curve(df, h, pts):
    """Net benefit of recalibrated risk (nb_model) and treat-all (nb_all) over a threshold grid.
    Isotonic map fit on the development split, evaluated on the held-out test split."""
    sub = df[df.h_min == h].dropna(subset=["risk_M1"])
    dev, test = sub[sub._dev], sub[~sub._dev]
    iso = IsotonicRegression(out_of_bounds="clip").fit(
        dev.risk_M1.values, dev.hypo_event.values.astype(float))
    st = iso.predict(test.risk_M1.values)
    yt = test.hypo_event.values.astype(float)
    N, prev = len(yt), yt.mean()
    nb_model, nb_all = [], []
    for pt in pts:
        pred = st >= pt
        tp = np.sum(pred & (yt == 1)); fp = np.sum(pred & (yt == 0))
        nb_model.append(tp / N - fp / N * (pt / (1 - pt)))
        nb_all.append(prev - (1 - prev) * (pt / (1 - pt)))
    return np.array(nb_model), np.array(nb_all), prev


def main():
    dfv = cohort_split_col(f"{RES}/ablation_windows_all2873.csv", CL_V)
    dfm = cohort_split_col(f"{RES}/ablation_windows_mover_art.csv", None)
    cohorts = [("VitalDB (development)", dfv), ("MOVER (external)", dfm)]

    fig, axes = plt.subplots(2, 3, figsize=(12, 7), sharex=True)
    for ri, (cname, dd) in enumerate(cohorts):
        for ci, h in enumerate(HORIZONS):
            ax = axes[ri, ci]
            nbm, nba, prev = dca_curve(dd, h, PTS)
            ax.plot(PTS, nbm, color=COLORS["model"], lw=2, label="TiRex-2 (recalibrated)")
            ax.plot(PTS, nba, color=COLORS["all"], lw=1.2, ls="--", label="Treat all")
            ax.axhline(0, color="k", lw=0.8, ls=":", label="Treat none")
            ax.set_title(f"{cname.split()[0]} — {h} min (prev {prev:.2f})", fontsize=9)
            ax.set_ylim(-0.05, max(nbm.max(), prev) * 1.15)
            if ci == 0:
                ax.set_ylabel("Net benefit", fontsize=9)
            if ri == 1:
                ax.set_xlabel("Threshold probability", fontsize=9)
            if ri == 0 and ci == 2:
                ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout(rect=[0, 0, 1, 1.0])
    S.save_fig(fig, "FigS_decision_curves")


if __name__ == "__main__":
    main()
