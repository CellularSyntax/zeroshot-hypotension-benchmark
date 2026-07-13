"""Cross-dataset transfer / external-validation figure and table.

Question: does a model trained on one institution's cohort still predict impending
hypotension on another's, and how does that compare to TiRex-2 which is *never* trained?

We fill a 2x2 train x test matrix (VitalDB, MOVER) for each supervised baseline (TFT,
PatchTST), all in the COVARIATE-FREE arm (M0) so the transfer is apples-to-apples:
  * on-diagonal  (in-domain)  : held-out predictions on the SAME cohort the model trained on
                                (VitalDB = 5-fold OOF `baseline-*_all2873`; MOVER = 5-fold OOF
                                 `baseline-*_mover_art` when available)
  * off-diagonal (transferred): the all-train checkpoint applied to the OTHER cohort
                                (`xfer-*_moverTOvitaldb60`, `xfer-*_vitaldb60TOmover_art`)
TiRex-2 (zero-shot, `all2873` / `mover_art`) is the training-free anchor on both test sets;
being zero-shot it has no train cohort, so it appears once per test panel.

Any missing source CSV (e.g. the MOVER in-domain CV, if not yet run) is skipped gracefully:
its cell is reported as "pending" in the table and omitted from the figure, and the script
prints exactly which run would fill it.

Writes outputs/figs/paper/Fig6_transfer.{pdf,png} and results/tables/Table7_transfer.{md,csv,tex}.

Run: PYTHONPATH=scripts:datasets/vitaldb python scripts/transfer_figure.py
"""
from __future__ import annotations
import csv, glob, os
import numpy as np
import matplotlib.pyplot as plt
import paper_style as S
import hypo_eval as HE

HORIZONS = [1, 3, 5, 7, 10, 15]
N_BOOT = 1000

# ── cells: (test_set, model, condition) -> source window tag ────────────────────────────
#   TiRex is handled separately (zero-shot anchor, one per test set).
VITALDB = "VitalDB"; MOVER = "MOVER"
CELLS = {
    # test on VitalDB
    (VITALDB, "tft",      "in-domain"):   "baseline-tft_all2873",
    (VITALDB, "tft",      "transfer"):    "xfer-tft_moverTOvitaldb60",
    (VITALDB, "patchtst", "in-domain"):   "baseline-patchtst_all2873",
    (VITALDB, "patchtst", "transfer"):    "xfer-patchtst_moverTOvitaldb60",
    # test on MOVER  (in-domain = 5-fold OOF trained on MOVER; the tag carries the cov suffix)
    (MOVER,   "tft",      "in-domain"):   "baseline-tft_mover_art_covmover_rate",
    (MOVER,   "tft",      "transfer"):    "xfer-tft_vitaldb60TOmover_art",
    (MOVER,   "patchtst", "in-domain"):   "baseline-patchtst_mover_art_covmover_rate",
    (MOVER,   "patchtst", "transfer"):    "xfer-patchtst_vitaldb60TOmover_art",
}
TIREX = {VITALDB: "all2873", MOVER: "mover_art"}
TRAIN_SRC = {  # human-readable training source, for the "transferred from" label
    (VITALDB, "transfer"): "MOVER→", (MOVER, "transfer"): "VitalDB→",
    (VITALDB, "in-domain"): "", (MOVER, "in-domain"): "",
}
MODEL_DISP = {"tft": "TFT", "patchtst": "PatchTST"}
MODEL_COL = {"tft": "#DE8F05", "patchtst": "#CC78BC"}   # match paper_figures baseline palette (colorblind)


def load_rows(tag):
    files = sorted(glob.glob(f"results/ablation_windows_{tag}_sh*of*.csv")) \
        or ([f"results/ablation_windows_{tag}.csv"] if os.path.exists(f"results/ablation_windows_{tag}.csv") else [])
    rows = []
    for f in files:
        for r in csv.DictReader(open(f)):
            if r.get("h_min") in ("", None) or None in r.values():
                continue
            rows.append(r)
    return rows


def auroc_ci(rows, h, risk_col, n_boot=N_BOOT):
    hr = [r for r in rows if int(r["h_min"]) == h and r.get(risk_col) not in ("", "nan", None)]
    if not hr:
        return None
    y = np.array([float(r["hypo_event"]) for r in hr])
    s = np.array([float(r[risk_col]) for r in hr])
    cid = np.array([str(r["caseid"]) for r in hr])
    if y.sum() == 0 or y.sum() == len(y):
        return None
    au = HE.auroc(y, s)
    lo, hi = HE.clustered_boot_ci(cid, y, s, HE.auroc, n_boot=n_boot)
    return dict(auroc=float(au), ci=[float(lo), float(hi)], n=len(y), n_events=int(y.sum()))


def series(tag, risk_col):
    """AUROC[CI] across HORIZONS for one window tag; {} if the tag is absent."""
    rows = load_rows(tag)
    if not rows:
        return {}
    return {h: auroc_ci(rows, h, risk_col) for h in HORIZONS}


def main():
    # ── gather every series once ────────────────────────────────────────────────────────
    tirex = {ts: series(tag, "risk_M0") for ts, tag in TIREX.items()}
    cells, pending = {}, []
    for (ts, m, cond), tag in CELLS.items():
        sr = series(tag, "risk_M0")
        if sr:
            cells[(ts, m, cond)] = sr
        else:
            pending.append((ts, m, cond, tag))

    if pending:
        print("PENDING cells (source CSV not found — cell left blank):")
        for ts, m, cond, tag in pending:
            print(f"  test={ts:8s} {MODEL_DISP[m]:9s} {cond:10s}  needs results/ablation_windows_{tag}.csv")
        print()

    # ── FIGURE: one panel per test set, AUROC vs horizon ────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(S.W2 * 1.02, S.W2 * 0.44), sharey=True)
    for ax, ts in zip(axes, (VITALDB, MOVER)):
        hs = np.array(HORIZONS)
        # TiRex-2 zero-shot anchor (bold teal, filled circles)
        tsr = tirex.get(ts, {})
        ty = [tsr[h]["auroc"] if tsr.get(h) else np.nan for h in HORIZONS]
        tlo = [tsr[h]["ci"][0] if tsr.get(h) else np.nan for h in HORIZONS]
        thi = [tsr[h]["ci"][1] if tsr.get(h) else np.nan for h in HORIZONS]
        ax.fill_between(hs, tlo, thi, color=S.C["M1_light"], alpha=0.35, lw=0, zorder=1)
        ax.plot(hs, ty, "-o", color=S.C["M1"], lw=2.3, ms=4.5, zorder=6,
                label="TiRex-2 (zero-shot)")
        # trained baselines: in-domain (solid) vs transferred (dashed, open marker)
        for m in ("tft", "patchtst"):
            col = MODEL_COL[m]
            for cond, ls, mk, mf in (("in-domain", "-", "s", col), ("transfer", "--", "s", "white")):
                sr = cells.get((ts, m, cond))
                if not sr:
                    continue
                yv = [sr[h]["auroc"] if sr.get(h) else np.nan for h in HORIZONS]
                src = TRAIN_SRC[(ts, cond)]
                lab = f"{MODEL_DISP[m]} ({src}transfer)" if cond == "transfer" else f"{MODEL_DISP[m]} (in-domain)"
                ax.plot(hs, yv, ls=ls, marker=mk, color=col, mfc=mf, mec=col, lw=1.4, ms=3.6,
                        zorder=4, label=lab)
        ax.set_xticks(HORIZONS)
        ax.set_xlabel("forecast horizon (min)")
        ax.set_title(ts, loc="center")
        ax.set_ylim(0.62, 1.0)
        ax.legend(loc="lower left", ncol=1, fontsize=5.6, handlelength=2.1)
    axes[0].set_ylabel("impending-hypotension AUROC")
    S.panel_letter(axes[0], "a"); S.panel_letter(axes[1], "b", dx=-0.06)
    fig.tight_layout(w_pad=1.4)
    S.save_fig(fig, "Fig6_transfer")

    # ── TABLE: AUROC point estimates per horizon, grouped by test set ───────────────────
    header = ["Test cohort", "Model (training source)"] + [f"{h} min" for h in HORIZONS]
    body = []

    def cell_txt(sr, h):
        return f"{sr[h]['auroc']:.3f}" if sr and sr.get(h) else "—"

    for ts in (VITALDB, MOVER):
        # TiRex first
        tsr = tirex.get(ts, {})
        body.append([ts, "TiRex-2 (zero-shot)"] + [cell_txt(tsr, h) for h in HORIZONS])
        for m in ("tft", "patchtst"):
            for cond in ("in-domain", "transfer"):
                sr = cells.get((ts, m, cond))
                if cond == "transfer":
                    src = "MOVER" if ts == VITALDB else "VitalDB"
                    who = f"{MODEL_DISP[m]} (trained {src})"
                else:
                    who = f"{MODEL_DISP[m]} (in-domain)"
                if sr is None:
                    body.append(["", who] + ["pending"] * len(HORIZONS))
                else:
                    body.append(["", who] + [cell_txt(sr, h) for h in HORIZONS])
        # collapse repeated test-cohort label to first row of the block
    # blank out repeated test-cohort labels (keep only first of each block)
    seen = set()
    for r in body:
        if r[0] in seen:
            r[0] = ""
        else:
            seen.add(r[0])

    os.makedirs("results/tables", exist_ok=True)
    cap = ("**Table 7. Cross-dataset transfer of impending-hypotension prediction (covariate-free, M0; "
           "AUROC, all cases). Supervised baselines are shown in-domain (held-out 5-fold OOF on the "
           "training cohort) and transferred (all-train checkpoint applied to the other cohort); "
           "TiRex-2 is zero-shot on both. Cells marked 'pending' await the corresponding "
           "in-domain cross-validation run. 95% CIs are in Fig. 6. VitalDB in-domain is at 15 s cadence; "
           "all transferred and MOVER results are at the harmonised 60 s cadence.**")
    with open("results/tables/Table7_transfer.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(body)
    with open("results/tables/Table7_transfer.md", "w") as f:
        f.write(cap + "\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in body:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
    with open("results/tables/Table7_transfer.tex", "w") as f:
        f.write("\\begin{tabular}{" + "l" * len(header) + "}\n\\hline\n")
        f.write(" & ".join(header) + " \\\\\n\\hline\n")
        for r in body:
            f.write(" & ".join(str(x) for x in r) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")

    # ── console summary ─────────────────────────────────────────────────────────────────
    print("Cross-dataset transfer @5 min (AUROC, M0):")
    for ts in (VITALDB, MOVER):
        t5 = tirex.get(ts, {}).get(5)
        print(f"  test={ts}: TiRex-2 {t5['auroc']:.3f} [{t5['ci'][0]:.3f},{t5['ci'][1]:.3f}]" if t5 else f"  test={ts}: TiRex-2 —")
        for m in ("tft", "patchtst"):
            for cond in ("in-domain", "transfer"):
                sr = cells.get((ts, m, cond))
                if sr and sr.get(5):
                    d = sr[5]
                    print(f"      {MODEL_DISP[m]:9s} {cond:10s} {d['auroc']:.3f} [{d['ci'][0]:.3f},{d['ci'][1]:.3f}]")
    print("\nwrote results/tables/Table7_transfer.{md,csv,tex}")


if __name__ == "__main__":
    main()
