"""Paired, case-clustered bootstrap significance tests for the paper's key claims.

Everything stays in the SAME inferential framework as the rest of the analysis (case-clustered
bootstrap), so the p-values are consistent with the CIs shown in the figures. Windows are nested
within cases, so we resample CASES (with replacement) — never windows — to avoid pseudo-replication.
Comparisons are PAIRED: the two models are evaluated on the identical windows/cases, and each
bootstrap resample is applied to both before differencing.

Produces results/tables/TableS_stats.{md,csv,tex} and prints the same. Reads the per-window CSVs.

Run: PYTHONPATH=scripts:datasets/vitaldb python scripts/stats_tests.py [tag]
"""
from __future__ import annotations
import csv, os, sys
import numpy as np
import hypo_eval as H
from baselines.splits import subject_split

TAG = sys.argv[1] if len(sys.argv) > 1 else "all2873"
RATE_TAG, PRESSOR_TAG = "all2873_covrate", "cases115_covpressor"
N_BOOT = 2000
SEED = 0


def load_rows(tag):
    return H.load_rows(tag)[0]


def canonical_test_subjects(rows, c2s, seed=0):
    # all cases: trained baselines are out-of-fold (held-out) via 5-fold CV; TiRex is zero-shot
    return {c2s.get(str(r["caseid"]), str(r["caseid"])) for r in rows}


def pfmt(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def stars(p):
    return "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."


# ── paired AUROC difference (A − B) on identical windows ───────────────────────────────
def _join_risk(rowsA, rowsB, c2s, test_subj, h, risk="risk_M1", ev="hypo_event"):
    def index(rows):
        d = {}
        for r in rows:
            if int(r["h_min"]) != h:
                continue
            cid = str(r["caseid"])
            if c2s.get(cid, cid) not in test_subj:
                continue
            v = r.get(risk)
            if v in ("", "nan", None):
                continue
            d[(cid, r["t0"])] = (float(r[ev]), float(v), cid)
        return d
    A, B = index(rowsA), index(rowsB)
    keys = sorted(set(A) & set(B))
    y = np.array([A[k][0] for k in keys])
    sA = np.array([A[k][1] for k in keys]); sB = np.array([B[k][1] for k in keys])
    cid = np.array([A[k][2] for k in keys])
    return y, sA, sB, cid


def paired_auroc_test(rowsA, rowsB, c2s, test_subj, h, n_boot=N_BOOT, seed=SEED, **kw):
    y, sA, sB, cid = _join_risk(rowsA, rowsB, c2s, test_subj, h, **kw)
    if len(y) == 0 or y.sum() == 0 or y.sum() == len(y):
        return None
    obs = H.auroc(y, sA) - H.auroc(y, sB)
    cases = np.unique(cid); idx = {c: np.where(cid == c)[0] for c in cases}
    rng = np.random.default_rng(seed); diffs = []
    for _ in range(n_boot):
        samp = rng.choice(cases, len(cases), replace=True)
        ii = np.concatenate([idx[c] for c in samp])
        yb = y[ii]
        if yb.sum() == 0 or yb.sum() == len(yb):
            continue
        diffs.append(H.auroc(yb, sA[ii]) - H.auroc(yb, sB[ii]))
    diffs = np.array(diffs)
    p = min(1.0, 2 * min((diffs > 0).mean(), (diffs < 0).mean()))
    return dict(diff=float(obs), ci=[float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
                p=float(p), n=int(len(y)), n_events=int(y.sum()))


# ── covariate benefit X% (per-case CRPS) and its paired difference ─────────────────────
def percase_crps(rows, c2s, test_subj, stratum, h):
    d = {}
    for r in rows:
        if int(r["h_min"]) != h or (stratum != "all" and r["stratum"] != stratum):
            continue
        cid = str(r["caseid"])
        if test_subj is not None and c2s.get(cid, cid) not in test_subj:
            continue
        v1, v0 = r.get("crps_M1"), r.get("crps_M0")
        if v1 in ("", "nan", None) or v0 in ("", "nan", None):
            continue
        d.setdefault(cid, []).append((float(v1), float(v0)))
    return {c: (np.mean([p[0] for p in v]), np.mean([p[1] for p in v])) for c, v in d.items()}


def _xpct(pc, cases):
    c1 = np.mean([pc[c][0] for c in cases]); c0 = np.mean([pc[c][1] for c in cases])
    return (c0 - c1) / c0 * 100


def xpct_vs0_test(pc, n_boot=N_BOOT, seed=SEED):
    cases = np.array(sorted(pc))
    if len(cases) < 3:
        return None
    obs = _xpct(pc, cases)
    rng = np.random.default_rng(seed); b = []
    for _ in range(n_boot):
        b.append(_xpct(pc, rng.choice(cases, len(cases), replace=True).tolist()))
    b = np.array(b)
    p = min(1.0, 2 * min((b > 0).mean(), (b < 0).mean()))
    return dict(x=float(obs), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                p=float(p), n_cases=int(len(cases)))


def xpct_diff_test(pcA, pcB, n_boot=N_BOOT, seed=SEED):
    cases = np.array(sorted(set(pcA) & set(pcB)))
    if len(cases) < 3:
        return None
    obs = _xpct(pcA, cases) - _xpct(pcB, cases)
    rng = np.random.default_rng(seed); b = []
    for _ in range(n_boot):
        s = rng.choice(cases, len(cases), replace=True).tolist()
        b.append(_xpct(pcA, s) - _xpct(pcB, s))
    b = np.array(b)
    p = min(1.0, 2 * min((b > 0).mean(), (b < 0).mean()))
    return dict(diff=float(obs), ci=[float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                p=float(p), n_cases=int(len(cases)))


def _write_table(name, header, rows):
    os.makedirs("results/tables", exist_ok=True)
    with open(f"results/tables/{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    with open(f"results/tables/{name}.md", "w") as f:
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"] * len(header)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
    with open(f"results/tables/{name}.tex", "w") as f:
        f.write("\\begin{tabular}{" + "l" * len(header) + "}\n\\hline\n")
        f.write(" & ".join(header) + " \\\\\n\\hline\n")
        for r in rows:
            f.write(" & ".join(str(x).replace("<", "$<$").replace("%", "\\%") for x in r) + " \\\\\n")
        f.write("\\hline\n\\end{tabular}\n")


def main():
    c2s = H.caseid_to_subject()
    trows = load_rows(TAG)
    test_subj = canonical_test_subjects(trows, c2s)
    ZS = [("chronos", "Chronos-Bolt"), ("timesfm", "TimesFM-2.5"), ("moirai", "Moirai-1.1-R")]
    TR = [("tft", "TFT"), ("patchtst", "PatchTST")]
    out = []   # (section, comparison, estimate, ci, p, stars)

    def sec_first(section):
        # show a section label only on its first row (blank thereafter -> narrow column)
        return section if not any(r[0] == section for r in out) else ""

    # ── A. Covariate benefit (CE, transition @7 min): within-model > 0, and between-model ──
    pcs = {"TiRex-2": percase_crps(trows, c2s, test_subj, "transition", 7)}
    for key, disp in TR:
        r = load_rows(f"baseline-{key}_{TAG}")
        pcs[disp] = percase_crps(r, c2s, None, "transition", 7)   # baseline CSV is test-only already
    S1 = "Covariate benefit vs 0 (CE, transition 7 min)"
    for name, pc in pcs.items():
        t = xpct_vs0_test(pc)
        if t:
            out.append([sec_first(S1), name, f"{t['x']:+.2f}%",
                        f"[{t['ci'][0]:+.2f}, {t['ci'][1]:+.2f}]", pfmt(t["p"]), stars(t["p"])])
    S2 = "Covariate benefit, between-model (CE, transition 7 min)"
    for A, B in [("TFT", "TiRex-2"), ("PatchTST", "TiRex-2"), ("PatchTST", "TFT")]:
        t = xpct_diff_test(pcs[A], pcs[B])
        if t:
            out.append([sec_first(S2), f"{A} - {B}", f"{t['diff']:+.2f}%",
                        f"[{t['ci'][0]:+.2f}, {t['ci'][1]:+.2f}]", pfmt(t["p"]), stars(t["p"])])

    # ── B. Zero-shot benchmark (Fig 3): TiRex − other zero-shot TSFM, AUROC by horizon ──
    S3 = "Zero-shot AUROC difference (Fig 3)"
    for key, disp in ZS:
        r = load_rows(f"baseline-{key}_{TAG}")
        for h in (5, 10, 15):
            t = paired_auroc_test(trows, r, c2s, test_subj, h)
            if t:
                out.append([sec_first(S3), f"TiRex-2 - {disp} @{h}min", f"{t['diff']:+.3f}",
                            f"[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]", pfmt(t["p"]), stars(t["p"])])

    # ── C. vs supervised SOTA (Fig 4): TiRex − trained baseline, AUROC by horizon ──
    S4 = "Trained-SOTA AUROC difference (Fig 4)"
    for key, disp in TR:
        r = load_rows(f"baseline-{key}_{TAG}")
        for h in (5, 7, 15):
            t = paired_auroc_test(trows, r, c2s, test_subj, h)
            if t:
                out.append([sec_first(S4), f"TiRex-2 - {disp} @{h}min", f"{t['diff']:+.3f}",
                            f"[{t['ci'][0]:+.3f}, {t['ci'][1]:+.3f}]", pfmt(t["p"]), stars(t["p"])])

    header = ["Test", "Comparison", "Estimate", "95% CI", "p", "sig"]
    _write_table("TableS_stats", header, out)
    w = max(len(r[0]) for r in out)
    print(f"{'Test':<48} {'Comparison':<26} {'Est':>9} {'95% CI':>20} {'p':>8}  sig")
    print("-" * 120)
    last = None
    for sec, comp, est, ci, p, st in out:
        sec_disp = sec if sec != last else ""
        print(f"{sec_disp:<48} {comp:<26} {est:>9} {ci:>20} {p:>8}  {st}"); last = sec
    print("\nwrote results/tables/TableS_stats.{md,csv,tex}")


if __name__ == "__main__":
    main()
