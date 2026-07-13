"""Post-hoc impending-hypotension (MAP<65) classification eval for the TiRex-2 study, §3.5 -> [Z].
Standalone: reads the per-window rows already written by phase3_ablation.py (columns hypo_event,
risk_M1, risk_M0 per horizon) — NO forecast rerun needed. Does not import the live plotting pipeline.

Adds, beyond phase3's pooled AUROC/AUPRC:
  - ROC + PR curves per horizon (M1 vs M0), case-clustered bootstrap AUROC/AUPRC CIs
  - partial AUROC over the clinically relevant high-specificity region (spec>=0.9), McClish-standardised
  - probability calibration: reliability diagram, Brier score, expected calibration error (ECE)
  - operating points selected on a DISJOINT patient-level dev set (§3.5), reported on held-out test:
    Youden's J, fixed specificity 0.90, and max-F1 -> sens/spec/PPV/NPV/F1 with bootstrap CIs

Figures come in two variants: a detailed per-horizon GRID (operating point + auc/sens/spec/ppv/npv
box, clinical specificity-axis style) and a COMBINED 2-panel (M0 | M1) overlay of all horizons.

Run:  PYTHONPATH=scripts <venv>/bin/python scripts/hypo_eval.py n300_s1
Writes results/hypo_metrics_<tag>.json + outputs/figs/hypo_{roc,roc_combined,pr,pr_combined,
calibration,operating}_<tag>.png
"""
import csv, glob, json, os, sys
import numpy as np
try:                                        # plotting is optional — the metric functions
    import matplotlib                       # (auroc/calibration/…) reused by compare.py and the
    matplotlib.use("Agg")                   # zero-shot venvs don't need matplotlib installed.
    import matplotlib.pyplot as plt
except ModuleNotFoundError:
    plt = None

SPEC_FLOOR = 0.90     # high-specificity region for pAUROC + a fixed-spec operating point
OP_RULE = "youden"    # operating point marked on the ROC/PR curves (dev-selected)

# Trained supervised foils, hypotension AUROC per horizon (from notes/RELATED_WORK.md; do NOT re-type
# elsewhere). Kapral: TFT, VitalDB as EXTERNAL set. Zhu: Transformer classifier, VitalDB external.
FOILS_TABLE = {5:  {"Kapral int": 0.909, "Kapral ext": 0.903, "Zhu": 0.904},
               7:  {"Kapral int": 0.880, "Kapral ext": 0.867},
               10: {"Zhu": 0.892},
               15: {"Zhu": 0.882}}


# ---------- metrics (hand-rolled; no sklearn) ----------
def _pairs(y, s):
    m = np.isfinite(s); y = np.asarray(y, float)[m]; s = np.asarray(s, float)[m]
    return y, s


def roc_points(y, s):
    """Return (fpr, tpr) sorted by decreasing threshold, incl. (0,0) and (1,1)."""
    y, s = _pairs(y, s)
    P, N = y.sum(), len(y) - y.sum()
    if P == 0 or N == 0:
        return np.array([0, 1.0]), np.array([0, 1.0]), np.array([np.inf, -np.inf])
    order = np.argsort(-s, kind="mergesort"); ys = y[order]; ss = s[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    keep = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
    tpr = np.r_[0, tp[keep] / P]; fpr = np.r_[0, fp[keep] / N]; thr = np.r_[np.inf, ss[keep]]
    return fpr, tpr, thr


_trapz = getattr(np, "trapezoid", getattr(np, "trapz", None))   # np2 renamed trapz->trapezoid

def auc_xy(x, y):
    return float(_trapz(y, x))


def auroc(y, s):
    fpr, tpr, _ = roc_points(y, s); return auc_xy(fpr, tpr)


def pauroc_highspec(y, s, spec_floor=SPEC_FLOOR):
    """McClish-standardised partial AUROC over FPR in [0, 1-spec_floor] -> [0.5,1] (0.5=chance)."""
    fpr, tpr, _ = roc_points(y, s)
    fmax = 1 - spec_floor
    xs = np.linspace(0, fmax, 200); ys = np.interp(xs, fpr, tpr)
    area = _trapz(ys, xs)
    amin, amax = 0.5 * fmax * fmax, fmax
    if amax <= amin:
        return np.nan
    return float(0.5 * (1 + (area - amin) / (amax - amin)))


def pr_points(y, s):
    y, s = _pairs(y, s); P = y.sum()
    if P == 0:
        return np.array([0, 1.0]), np.array([1.0, 0]), np.array([np.inf, -np.inf])
    order = np.argsort(-s, kind="mergesort"); ys = y[order]; ss = s[order]
    tp = np.cumsum(ys); fp = np.cumsum(1 - ys)
    keep = np.r_[np.where(np.diff(ss))[0], len(ss) - 1]
    prec = tp[keep] / (tp[keep] + fp[keep]); rec = tp[keep] / P
    return np.r_[0, rec], np.r_[prec[0], prec], np.r_[np.inf, ss[keep]]


def auprc(y, s):
    rec, prec, _ = pr_points(y, s); return auc_xy(rec, prec)


def brier(y, s):
    y, s = _pairs(y, s); return float(np.mean((s - y) ** 2))


def calibration(y, s, n_bins=10):
    """Equal-count (quantile) bins -> (mean_pred, obs_freq, count) per bin, plus ECE."""
    y, s = _pairs(y, s)
    if len(y) == 0:
        return np.array([]), np.array([]), np.array([]), np.nan
    edges = np.unique(np.quantile(s, np.linspace(0, 1, n_bins + 1)))
    idx = np.clip(np.digitize(s, edges[1:-1]), 0, len(edges) - 2)
    mp, of, cnt = [], [], []
    for b in range(len(edges) - 1):
        sel = idx == b
        if sel.sum() == 0:
            continue
        mp.append(s[sel].mean()); of.append(y[sel].mean()); cnt.append(int(sel.sum()))
    mp, of, cnt = np.array(mp), np.array(of), np.array(cnt)
    ece = float(np.sum(cnt / cnt.sum() * np.abs(mp - of))) if len(cnt) else np.nan
    return mp, of, cnt, ece


def op_metrics(y, s, thr):
    """sens/spec/ppv/npv/f1/acc at a probability threshold."""
    y, s = _pairs(y, s); pred = s >= thr
    tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
    fn = np.sum(~pred & (y == 1)); tn = np.sum(~pred & (y == 0))
    sens = tp / (tp + fn) if tp + fn else np.nan
    spec = tn / (tn + fp) if tn + fp else np.nan
    ppv = tp / (tp + fp) if tp + fp else np.nan
    npv = tn / (tn + fn) if tn + fn else np.nan
    f1 = 2 * ppv * sens / (ppv + sens) if ppv and sens and (ppv + sens) else np.nan
    acc = (tp + tn) / len(y)
    return {"sens": sens, "spec": spec, "ppv": ppv, "npv": npv, "f1": f1, "acc": acc}


def pick_threshold(y, s, rule):
    """Select an operating threshold on the DEV set by a rule."""
    fpr, tpr, thr = roc_points(y, s)
    if rule == "youden":
        j = np.argmax(tpr - fpr); return float(thr[j])
    if rule == "spec90":
        ok = np.where((1 - fpr) >= SPEC_FLOOR)[0]
        return float(thr[ok[np.argmax(tpr[ok])]]) if len(ok) else float(thr[np.argmax(1 - fpr)])
    if rule == "maxf1":
        cand = np.unique(s); best, bt = -1, 0.5
        for t in cand:
            f = op_metrics(y, s, t)["f1"]
            if np.isfinite(f) and f > best:
                best, bt = f, float(t)
        return bt
    raise ValueError(rule)


# ---------- data ----------
def load_rows(tag):
    files = sorted(glob.glob(f"results/ablation_windows_{tag}_sh*of*.csv")) or \
            sorted(glob.glob(f"results/ablation_windows_{tag}.csv"))
    if not files:
        sys.exit(f"no windows CSVs for tag={tag}")
    rows = []
    for f in files:
        rows += list(csv.DictReader(open(f)))
    return rows, files


def caseid_to_subject():
    """Map caseid->subjectid for subject-level splitting. Prefers HE_CLINICAL, then the
    shipped two-column crosswalk (results/vitaldb_case_subject_map.csv), then the full
    VitalDB clinical table. Returns {} if none is present (caller falls back to case-level)."""
    for path in [os.environ.get("HE_CLINICAL"),
                 "results/vitaldb_case_subject_map.csv",
                 "datasets/vitaldb/data/clinical_data.csv"]:
        if path and os.path.exists(path):
            m = {}
            for r in csv.DictReader(open(path, encoding="utf-8-sig")):
                m[str(int(r["caseid"]))] = str(r["subjectid"])
            return m
    return {}   # no mapping available -> split_subjects treats each caseid as its own subject


def split_subjects(caseids, c2s, dev_frac=0.2, seed=0):
    subs = sorted({c2s.get(str(c), str(c)) for c in caseids})
    rng = np.random.default_rng(seed); rng.shuffle(subs)
    ndev = max(1, int(round(len(subs) * dev_frac)))
    dev = set(subs[:ndev]); return dev


def clustered_boot_ci(caseids, y, s, fn, n_boot=1000, seed=0):
    """Case-clustered bootstrap CI for a threshold-free metric fn(y,s)."""
    caseids = np.asarray(caseids); uc = np.unique(caseids)
    by = {c: np.where(caseids == c)[0] for c in uc}
    rng = np.random.default_rng(seed); vals = []
    for _ in range(n_boot):
        pick = rng.choice(uc, len(uc), replace=True)
        idx = np.concatenate([by[c] for c in pick])
        v = fn(y[idx], s[idx])
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return [np.nan, np.nan]
    return [round(float(np.percentile(vals, 2.5)), 3), round(float(np.percentile(vals, 97.5)), 3)]


# ---------- main ----------
def main():
    tag = sys.argv[1] if len(sys.argv) > 1 else "n300_s1"
    n_boot = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    rows, files = load_rows(tag)
    c2s = caseid_to_subject()
    horizons = sorted({int(r["h_min"]) for r in rows})
    allcases = [r["caseid"] for r in rows]
    dev_subjects = split_subjects(allcases, c2s, seed=0)
    print(f"tag={tag}: {len(files)} shards, {len(rows)} rows, horizons={horizons}, "
          f"dev subjects={len(dev_subjects)}", flush=True)

    out = {"tag": tag, "n_shards": len(files), "horizons_min": horizons, "spec_floor": SPEC_FLOOR,
           "op_rule_marked": OP_RULE, "per_horizon": {}}
    for h in horizons:
        hr = [r for r in rows if int(r["h_min"]) == h and r["risk_M1"] not in ("", "nan")]
        y = np.array([float(r["hypo_event"]) for r in hr])
        cid = np.array([r["caseid"] for r in hr])
        is_dev = np.array([c2s.get(str(c), str(c)) in dev_subjects for c in cid])
        rec = {"n": len(y), "n_events": int(y.sum()), "prevalence": round(float(y.mean()), 4),
               "n_dev": int(is_dev.sum()), "n_test": int((~is_dev).sum())}
        for cond in ("M1", "M0"):
            s = np.array([float(r[f"risk_{cond}"]) for r in hr])
            yt, st, ct = y[~is_dev], s[~is_dev], cid[~is_dev]
            yd, sd = y[is_dev], s[is_dev]
            m = {"auroc": round(auroc(yt, st), 4),
                 "auroc_CI95": clustered_boot_ci(ct, yt, st, auroc, n_boot),
                 "auprc": round(auprc(yt, st), 4),
                 "auprc_CI95": clustered_boot_ci(ct, yt, st, auprc, n_boot),
                 "pauroc_spec90": round(pauroc_highspec(yt, st), 4),
                 "brier": round(brier(yt, st), 4)}
            _, _, _, ece = calibration(yt, st); m["ece"] = round(ece, 4) if np.isfinite(ece) else None
            ops = {}
            for rule in ("youden", "spec90", "maxf1"):
                thr = pick_threshold(yd, sd, rule) if yd.sum() > 0 and (yd == 0).any() else 0.5
                mm = op_metrics(yt, st, thr)
                ops[rule] = {"thr": round(thr, 4), **{k: (round(v, 4) if np.isfinite(v) else None)
                                                      for k, v in mm.items()}}
            m["operating_points"] = ops
            rec[cond] = m
        out["per_horizon"][f"{h}min"] = rec
        print(f"  h={h:>2}min  ev={rec['n_events']}/{rec['n']} ({rec['prevalence']:.3f})  "
              f"M1 AUROC={rec['M1']['auroc']:.3f} CI{rec['M1']['auroc_CI95']}  "
              f"pAUROC(sp90)={rec['M1']['pauroc_spec90']}  ECE={rec['M1']['ece']}  "
              f"F1*={rec['M1']['operating_points']['maxf1']['f1']}", flush=True)

    json.dump(out, open(f"results/hypo_metrics_{tag}.json", "w"), indent=1)
    print(f"\nwrote results/hypo_metrics_{tag}.json", flush=True)
    write_comparison_table(out, tag)
    make_figs(rows, out, tag, c2s, dev_subjects)


def write_comparison_table(out, tag):
    """Head-to-head: our zero-shot M1 AUROC vs the trained supervised foils, per horizon."""
    ph = out["per_horizon"]; L = []
    L.append(f"# Hypotension AUROC — ours vs supervised foils ({tag})\n")
    L.append("Ours = **zero-shot** TiRex-2 (M1, drug covariate), held-out test, case-clustered 95% CI. "
             "Foils = **trained** models; Kapral (TFT) & Zhu (Transformer) both use VitalDB as an "
             "*external* set. Foil numbers from `notes/RELATED_WORK.md`. Caveat: event definitions/"
             "cohorts are not identical across studies — this is an indicative reference, not a matched "
             "benchmark (discuss in paper).\n")
    L.append("| Horizon | Ours M1 AUROC [95% CI] | Kapral internal | Kapral external | Zhu (external) |")
    L.append("|--:|:--|:--|:--|:--|")
    for h in out["horizons_min"]:
        m = ph[f"{h}min"]["M1"]; ci = m["auroc_CI95"]
        f = FOILS_TABLE.get(h, {})
        L.append(f"| {h} min | {m['auroc']:.3f} [{ci[0]:.3f}, {ci[1]:.3f}] | "
                 f"{f.get('Kapral int','—') if isinstance(f.get('Kapral int'),float) else '—'} | "
                 f"{f.get('Kapral ext','—') if isinstance(f.get('Kapral ext'),float) else '—'} | "
                 f"{f.get('Zhu','—') if isinstance(f.get('Zhu'),float) else '—'} |")
    L.append("\nKapral also reports an overall (horizon-averaged) hypotension AUROC of 0.933 internal / "
             "0.919 external.\n")
    open(f"results/comparison_foils_hypo_{tag}.md", "w").write("\n".join(L))
    print(f"wrote results/comparison_foils_hypo_{tag}.md", flush=True)


# ---------- figures ----------
def make_figs(rows, out, tag, c2s, dev_subjects):
    if plt is None:
        raise RuntimeError("make_figs needs matplotlib (install it, or run the metrics-only path)")
    horizons = out["horizons_min"]
    colors = {h: plt.cm.viridis(x) for h, x in zip(horizons, np.linspace(0, 0.88, len(horizons)))}

    def testset(h, cond):
        hr = [r for r in rows if int(r["h_min"]) == h and r["risk_M1"] not in ("", "nan")]
        keep = [r for r in hr if c2s.get(str(r["caseid"]), str(r["caseid"])) not in dev_subjects]
        y = np.array([float(r["hypo_event"]) for r in keep])
        s = np.array([float(r[f"risk_{cond}"]) for r in keep])
        return y, s

    def op(h, cond, rule=OP_RULE):
        d = out["per_horizon"][f"{h}min"][cond]
        return d["operating_points"][rule], d["auroc"], d["auprc"]

    def statbox(a, auc, o, kind="roc"):
        def pc(v):
            return f"{v*100:.1f}%" if isinstance(v, (int, float)) and v is not None else "—"
        txt = (f"auc = {auc:.4f}\nsens = {pc(o['sens'])}\nspec = {pc(o['spec'])}\n"
               f"ppv = {pc(o['ppv'])}\nnpv = {pc(o['npv'])}")
        xy = (0.62, 0.05) if kind == "roc" else (0.05, 0.05)
        a.text(xy[0], xy[1], txt, transform=a.transAxes, fontsize=9, va="bottom",
               ha="left", family="monospace")

    def roc_ax(a):
        a.plot([1, 0], [0, 1], "--", color="grey", lw=0.7)
        a.axvspan(SPEC_FLOOR, 1.0, color="k", alpha=0.04)          # high-specificity region
        a.set_xlim(1.0, 0.0); a.set_ylim(0, 1.02)
        a.set_xlabel("specificity"); a.set_ylabel("sensitivity")

    ncol = 3; nrow = int(np.ceil(len(horizons) / ncol))

    # 1) ROC — detailed grid (M1 solid + M0 faint), op point + stat box (screenshot style)
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.0 * nrow), squeeze=False)
    for i, h in enumerate(horizons):
        a = ax[i // ncol][i % ncol]; roc_ax(a)
        y0, s0 = testset(h, "M0"); f0, t0, _ = roc_points(y0, s0)
        a.plot(1 - f0, t0, color="0.6", lw=1.2, ls="--", label=f"M0 (auc {auroc(y0,s0):.3f})")
        y1, s1 = testset(h, "M1"); f1, t1, _ = roc_points(y1, s1)
        a.plot(1 - f1, t1, color="C1", lw=1.8, label=f"M1 (auc {auroc(y1,s1):.3f})")
        o1, auc1, _ = op(h, "M1")
        if o1["sens"] is not None and o1["spec"] is not None:
            a.plot(o1["spec"], o1["sens"], "o", color="C1", ms=9, zorder=6)
            statbox(a, auc1, o1, "roc")
        a.set_title(f"{h} min  (n={out['per_horizon'][f'{h}min']['n_test']}, "
                    f"ev={out['per_horizon'][f'{h}min']['n_events']})", fontsize=9)
        a.legend(fontsize=7, loc="upper left")
    for j in range(len(horizons), nrow * ncol):
        ax[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Hypotension ROC — per horizon (held-out test, op={OP_RULE}) — {tag}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(f"outputs/figs/hypo_roc_{tag}.png", dpi=110); plt.close(fig)

    # 2) ROC — combined (M0 | M1), all horizons overlaid, op dot per curve
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for a, cond in zip(ax, ("M0", "M1")):
        roc_ax(a)
        for h in horizons:
            y, s = testset(h, cond); f, t, _ = roc_points(y, s)
            o, auc, _ = op(h, cond)
            a.plot(1 - f, t, color=colors[h], lw=1.6, label=f"{h} min (auc {auc:.3f})")
            if o["sens"] is not None and o["spec"] is not None:
                a.plot(o["spec"], o["sens"], "o", color=colors[h], ms=7, zorder=6)
        a.set_title(f"{cond}  ({'with' if cond=='M1' else 'no'} drug covariate)", fontweight="bold")
        a.legend(fontsize=7.5, loc="lower right", title=f"op = {OP_RULE} (dev-selected)")
    fig.suptitle(f"Hypotension ROC — combined across horizons — {tag}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(f"outputs/figs/hypo_roc_combined_{tag}.png", dpi=120); plt.close(fig)

    # 3) PR — detailed grid
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.4 * ncol, 4.0 * nrow), squeeze=False)
    for i, h in enumerate(horizons):
        a = ax[i // ncol][i % ncol]
        prev = out["per_horizon"][f"{h}min"]["prevalence"]
        y0, s0 = testset(h, "M0"); r0, p0, _ = pr_points(y0, s0)
        a.plot(r0, p0, color="0.6", lw=1.2, ls="--", label=f"M0 (ap {auprc(y0,s0):.3f})")
        y1, s1 = testset(h, "M1"); r1, p1, _ = pr_points(y1, s1)
        a.plot(r1, p1, color="C1", lw=1.8, label=f"M1 (ap {auprc(y1,s1):.3f})")
        a.axhline(prev, ls=":", color="grey", lw=0.8, label=f"prevalence {prev:.2f}")
        o1, _, ap1 = op(h, "M1")
        if o1["sens"] is not None and o1["ppv"] is not None:   # recall=sens, precision=ppv
            a.plot(o1["sens"], o1["ppv"], "o", color="C1", ms=9, zorder=6)
            statbox(a, ap1, o1, "pr")
        a.set_xlim(0, 1); a.set_ylim(0, 1.02)
        a.set_xlabel("recall (sensitivity)"); a.set_ylabel("precision (PPV)")
        a.set_title(f"{h} min", fontsize=9); a.legend(fontsize=7, loc="upper right")
    for j in range(len(horizons), nrow * ncol):
        ax[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Hypotension precision–recall — per horizon (held-out test) — {tag}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(f"outputs/figs/hypo_pr_{tag}.png", dpi=110); plt.close(fig)

    # 4) PR — combined (M0 | M1)
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.6))
    for a, cond in zip(ax, ("M0", "M1")):
        for h in horizons:
            y, s = testset(h, cond); r, p, _ = pr_points(y, s)
            _, _, ap = op(h, cond); prev = out["per_horizon"][f"{h}min"]["prevalence"]
            a.plot(r, p, color=colors[h], lw=1.6, label=f"{h} min (ap {ap:.3f})")
            o, _, _ = op(h, cond)
            if o["sens"] is not None and o["ppv"] is not None:
                a.plot(o["sens"], o["ppv"], "o", color=colors[h], ms=7, zorder=6)
        a.set_xlim(0, 1); a.set_ylim(0, 1.02)
        a.set_xlabel("recall (sensitivity)"); a.set_ylabel("precision (PPV)")
        a.set_title(f"{cond}  ({'with' if cond=='M1' else 'no'} drug covariate)", fontweight="bold")
        a.legend(fontsize=7.5, loc="upper right", title=f"op = {OP_RULE}")
    fig.suptitle(f"Hypotension precision–recall — combined across horizons — {tag}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(f"outputs/figs/hypo_pr_combined_{tag}.png", dpi=120); plt.close(fig)

    # 5) Calibration grid (M1)
    fig, ax = plt.subplots(nrow, ncol, figsize=(4.2 * ncol, 3.8 * nrow), squeeze=False)
    for i, h in enumerate(horizons):
        a = ax[i // ncol][i % ncol]
        y, s = testset(h, "M1"); mp, of, cnt, ece = calibration(y, s)
        a.plot([0, 1], [0, 1], "--", color="grey", lw=0.7)
        if len(mp):
            a.plot(mp, of, "-o", color="C1", ms=4)
        a.set_title(f"{h} min  ECE={ece:.3f}" if np.isfinite(ece) else f"{h} min")
        a.set_xlabel("predicted risk"); a.set_ylabel("observed frequency")
        a.set_xlim(0, 1); a.set_ylim(0, 1)
    for j in range(len(horizons), nrow * ncol):
        ax[j // ncol][j % ncol].axis("off")
    fig.suptitle(f"Hypotension calibration, M1 (held-out test) — {tag}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.97]); fig.savefig(f"outputs/figs/hypo_calibration_{tag}.png", dpi=110); plt.close(fig)

    # 6) Operating-point + AUROC summary vs horizon
    fig, ax = plt.subplots(1, 2, figsize=(12, 4.4))
    for key, lbl, c in (("sens", "sensitivity", "C0"), ("spec", "specificity", "C2"),
                        ("ppv", "PPV", "C4"), ("f1", "F1", "C1")):
        vals = [out["per_horizon"][f"{h}min"]["M1"]["operating_points"]["spec90"].get(key) for h in horizons]
        ax[0].plot(horizons, vals, "-o", color=c, label=lbl)
    ax[0].set_title("M1 operating point (thr sel. on dev @ spec≥0.90)"); ax[0].set_xlabel("horizon (min)")
    ax[0].set_ylim(0, 1); ax[0].legend(fontsize=8)
    for cond, c in (("M1", "C1"), ("M0", "C3")):
        ax[1].plot(horizons, [out["per_horizon"][f"{h}min"][cond]["auroc"] for h in horizons], "-o",
                   color=c, label=f"{cond} AUROC")
        ax[1].plot(horizons, [out["per_horizon"][f"{h}min"][cond]["pauroc_spec90"] for h in horizons],
                   "--s", color=c, ms=4, label=f"{cond} pAUROC(sp90)")
    ax[1].set_title("AUROC & partial-AUROC vs horizon"); ax[1].set_xlabel("horizon (min)")
    ax[1].set_ylim(0.5, 1); ax[1].legend(fontsize=7)
    fig.suptitle(f"Hypotension operating summary — {tag}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(f"outputs/figs/hypo_operating_{tag}.png", dpi=110); plt.close(fig)
    print("wrote outputs/figs/hypo_{roc,roc_combined,pr,pr_combined,calibration,operating}_"
          + tag + ".png", flush=True)


if __name__ == "__main__":
    main()
