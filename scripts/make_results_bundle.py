"""Assemble all paper results (figures + tables + statistics) into one LaTeX document.

Reads the finished artifacts in outputs/figs/paper/ and results/tables/ and emits
results_bundle.tex at the repo root. Figure captions are inline here; table captions are
read from each table's .md first line (kept in sync with paper_figures). Tables are rendered
from their .csv and wrapped in adjustbox so wide ones shrink to the page.

Run: python scripts/make_results_bundle.py   (then: pdflatex results_bundle.tex, from repo root)
"""
from __future__ import annotations
import csv, os, re

FIG_DIR = "outputs/figs/paper"
TAB_DIR = "results/tables"
OUT = "results_bundle.tex"

FIGURES = [
    ("Fig1_design_cohort",
     "Study design and cohort. (a) Forecasting task: a 30-min MAP context and the known future "
     "drug-infusion covariate condition a 15-min quantile forecast. (b) Cohort flow. (c) Representative "
     "forecasts (steady, transition, hypotensive onset): the context window is shaded, the observed "
     "ground truth is muted, and the TiRex-2 median with 10--90\\% interval is overlaid."),
    ("Fig2_accuracy_covariate",
     "Forecast accuracy and the value of the known drug covariate. (a) MAE vs horizon, zero-shot "
     "TiRex-2 vs trained TFT/PatchTST. (b) TiRex-2 covariate benefit by window type. (c) Covariate "
     "benefit by drug representation and model (transition windows, 7 min; markers = mean, whiskers = "
     "case-clustered 95\\% CI). (d) Instantaneous MAE vs Kapral et al. (e) Covariate value by model "
     "(bar view of the CE row of c). Univariate zero-shot foundation models (Chronos-Bolt, TimesFM, "
     "Moirai) cannot ingest the covariate (0 by construction) and are omitted from panels c and e. "
     "\\textit{Evaluation:} the trained baselines (TFT, PatchTST) are scored on all cases by 5-fold "
     "subject-level out-of-fold cross-validation --- each case is predicted only by the fold in which "
     "it was held out --- while TiRex-2 is zero-shot and thus inherently held out; all models are "
     "scored on identical windows with the same metric code."),
    ("Fig3_zeroshot_tsfm",
     "Zero-shot foundation-model benchmark. TiRex-2 vs Chronos-Bolt, TimesFM-2.5 and Moirai-1.1-R on "
     "identical matched test windows, all evaluated zero-shot: (a) impending-hypotension AUROC, "
     "(b) forecasting CRPS, (c) calibration at 10 min (ECE), (d) AUPRC vs horizon. Only TiRex-2 "
     "ingests the known future drug-infusion covariate."),
    ("Fig4_hypotension_vs_sota",
     "Impending-hypotension prediction vs supervised state of the art (with-covariate, M1 arm). "
     "(a) ROC at 5 and 7 min; (b) AUROC vs horizon; (c) calibration; (d) AUPRC; (e) decision curve; "
     "(f) head-to-head AUROC at 5/7 min --- zero-shot TiRex-2 vs trained TFT and PatchTST, with "
     "external foils (Kapral, Zhu). \\textit{Evaluation:} the trained baselines are scored on all "
     "cases by 5-fold subject-level out-of-fold cross-validation --- each case is predicted only by "
     "the fold in which it was held out --- while TiRex-2 is zero-shot and thus inherently held out; "
     "both are scored on identical windows with the same metric code. Shaded bands in (b) and error "
     "intervals are case-clustered bootstrap 95\\% CIs; external foils are literature values."),
    ("Fig5_clinical_robustness",
     "Clinical translation and robustness: early-warning lead time, severity gradient, subgroup "
     "AUROC forest, and operating characteristics."),
    ("Fig6_transfer",
     "Cross-dataset transfer / external validation (covariate-free, M0). Impending-hypotension AUROC "
     "vs horizon on each test cohort: (a) VitalDB, (b) the external MOVER cohort. TiRex-2 (zero-shot, "
     "training-free) is the bold teal anchor on both; supervised TFT/PatchTST are shown in-domain "
     "(solid, held-out 5-fold OOF) versus transferred from the other cohort (dashed, open marker). "
     "Shaded band = TiRex-2 case-clustered 95\\% CI. VitalDB in-domain is at 15\\,s cadence; all "
     "transferred and MOVER series are at the harmonised 60\\,s cadence."),
    ("FigS_training_curves",
     "Supplementary. Training/validation pinball-loss curves for the supervised baselines "
     "(TFT, PatchTST), M1 (with drug covariate) and M0 (without); convergence without overfitting."),
]

# table name -> optional caption override (None => read from the .md first line)
TABLES = [
    ("Table1_cohort", None),
    ("Table2_accuracy", None),
    ("Table3_classification", None),
    ("Table4_matched", None),
    ("Table5_matched_forecast", None),
    ("Table6_zeroshot", None),
    ("Table7_transfer", None),
    ("TableS_stats",
     "Paired, case-clustered bootstrap significance tests for the key claims (2000 resamples; cases "
     "resampled with replacement; each comparison paired on identical windows, differenced within "
     "resample). *** $p<0.001$, ** $p<0.01$, * $p<0.05$, n.s. not significant."),
]

_ESC = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"), ("#", r"\#"), ("_", r"\_"),
        ("±", r"$\pm$"), ("−", r"$-$"), ("≥", r"$\geq$"), ("≤", r"$\leq$"), ("×", r"$\times$"),
        ("→", r"$\rightarrow$"), ("Δ", r"$\Delta$"), ("<", r"$<$"), (">", r"$>$"),
        ("–", "--"), ("—", "---"), ("’", "'"), ("μ", r"$\mu$"), ("°", r"$^\circ$"),
        ("²", r"$^2$"), ("³", r"$^3$")]


def esc(s):
    s = str(s)
    for a, b in _ESC:
        s = s.replace(a, b)
    # safety net: drop any remaining non-ASCII so pdflatex never fatals on an unmapped glyph
    stripped = s.encode("ascii", "ignore").decode()
    if stripped != s:
        print(f"  [warn] dropped non-ASCII glyph(s) in: {s!r}")
    return stripped


def md_caption(name):
    path = f"{TAB_DIR}/{name}.md"
    if not os.path.exists(path):
        return None
    first = open(path).readline().strip()
    if not first.startswith("**"):
        return None
    cap = first.strip("*").strip()
    return re.sub(r"^Table\s+\d+\.\s*", "", cap)   # drop redundant "Table N." (LaTeX auto-numbers)


def table_tex(name, caption):
    csv_path = f"{TAB_DIR}/{name}.csv"
    if not os.path.exists(csv_path):
        return f"% ({name}.csv missing — skipped)\n"
    rows = list(csv.reader(open(csv_path)))
    if not rows:
        return ""
    header, body = rows[0], rows[1:]
    ncol = len(header)
    L = [r"\begin{table}[H]\centering\footnotesize",
         r"\caption{" + (caption or name) + "}",   # caption already LaTeX-ready (see main)
         r"\begin{adjustbox}{max width=\textwidth}",
         r"\begin{tabular}{" + "l" * ncol + "}", r"\toprule",
         " & ".join(r"\textbf{" + esc(h) + "}" for h in header) + r" \\", r"\midrule"]
    for r in body:
        L.append(" & ".join(esc(x) for x in r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}", r"\end{table}", ""]
    return "\n".join(L)


def main():
    doc = [
        r"\documentclass[10pt]{article}",
        r"\usepackage[a4paper,margin=1.6cm]{geometry}",
        r"\usepackage{graphicx}", r"\usepackage{booktabs}", r"\usepackage{adjustbox}",
        r"\usepackage{float}", r"\usepackage{caption}",
        r"\captionsetup{font=small,labelfont=bf}",
        r"\graphicspath{{" + FIG_DIR + "/}}",
        r"\title{Zero-shot intraoperative mean-arterial-pressure forecasting with TiRex-2 on VitalDB\\"
        r"\large Results bundle: figures, tables and statistics}",
        r"\author{}", r"\date{\today}",
        r"\begin{document}", r"\maketitle",
        r"\noindent This document bundles every figure and table of the analysis. Supervised "
        r"baselines are evaluated on all cases via 5-fold subject-level out-of-fold cross-validation; "
        r"TiRex-2 is zero-shot (inherently held-out). Cross-dataset transfer (Fig.~6, Table~7) uses the "
        r"external MOVER cohort. Inference is case-clustered bootstrap throughout.",
        r"\clearpage", r"\section{Figures}",
    ]
    for name, cap in FIGURES:
        if not os.path.exists(f"{FIG_DIR}/{name}.pdf"):
            doc.append(f"% ({name}.pdf missing — skipped)"); continue
        doc += [r"\begin{figure}[H]\centering",
                r"\includegraphics[width=\linewidth]{" + name + ".pdf}",
                r"\caption{" + cap + "}",
                r"\end{figure}"]
    doc += [r"\clearpage", r"\section{Tables}"]
    for name, override in TABLES:
        # override captions are author-written LaTeX (kept raw); .md captions are plain text -> escape
        cap = override if override is not None else esc(md_caption(name) or name)
        doc.append(table_tex(name, cap))
    doc.append(r"\end{document}")
    open(OUT, "w").write("\n".join(doc) + "\n")
    print(f"wrote {OUT}  ({len(FIGURES)} figures, {len(TABLES)} tables)")
    print("compile with:  pdflatex results_bundle.tex   (run from the repo root)")


if __name__ == "__main__":
    main()
