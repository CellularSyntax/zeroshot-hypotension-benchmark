"""Publication figures + tables for the TiRex-2 / VitalDB paper (Nature Medicine style).

Narrative arc (zero-shot foundation model as the lead):
  Fig 1  Study design & cohort
  Fig 2  Forecast accuracy & the value of the known future drug covariate
  Fig 3  Zero-shot foundation-model benchmark: TiRex-2 vs Chronos/TimesFM/Moirai  (headline)
  Fig 4  Impending-hypotension prediction: zero-shot TiRex-2 vs supervised SOTA
  Fig 5  Clinical translation & robustness
  Tables 1-6  cohort | accuracy | classification-vs-foils | matched-trained | matched-forecast | zero-shot

Reads the finished full-cohort outputs already in results/ (tag=all2873 primary; the
covrate/pressor arms for the covariate-representation panel). Styling lives in paper_style.

Run:  PYTHONPATH=scripts:datasets/vitaldb python scripts/paper_figures.py [tag]
"""
from __future__ import annotations
import json, csv, os, sys
import numpy as np
import matplotlib.pyplot as plt
import paper_style as S
import hypo_eval as H   # roc_points, pr_points, calibration, auroc, load_rows, split_subjects, caseid_to_subject

TAG = sys.argv[1] if len(sys.argv) > 1 else "all2873"
RATE_TAG, PRESSOR_TAG = "all2873_covrate", "cases115_covpressor"
DT_S = 15.0
MAIN_H = [1, 3, 5, 7]            # horizons in main figures (<=7 min); supplement adds 10, 15


# ── data loaders ──────────────────────────────────────────────────────────────
def load_primary(tag):
    return json.load(open(f"results/ablation_primary_{tag}.json"))

def load_hypo(tag):
    return json.load(open(f"results/hypo_metrics_{tag}.json"))

def load_clinical(tag):
    return json.load(open(f"results/clinical_eval_{tag}.json"))

def load_subgroup(tag, h=5):
    return json.load(open(f"results/subgroup_forest_{tag}_h{h}.json"))

def strat(primary, h, s):
    return primary["per_horizon"][S.hkey(h)][s]

def _test_scores(rows, c2s, dev, h, risk_col="risk_M1", ev_col="hypo_event"):
    """(y, s) on the held-out test subjects for horizon h — matches hypo_eval's reported AUROC."""
    y, s = [], []
    for r in rows:
        if int(r["h_min"]) != h or r[risk_col] in ("", "nan"):
            continue
        if c2s.get(str(r["caseid"]), str(r["caseid"])) in dev:
            continue  # keep test only
        y.append(float(r[ev_col])); s.append(float(r[risk_col]))
    return np.array(y), np.array(s)


def _tft_covariate_xpct(brows, c2s, test_subjects, strata, hs):
    """TFT covariate benefit X% = (CRPS_M0 − CRPS_M1)/CRPS_M0 × 100, by stratum, on test windows."""
    out = {}
    for st in strata:
        xs = []
        for h in hs:
            m1, m0 = [], []
            for r in brows:
                if int(r["h_min"]) != h:
                    continue
                if st != "all" and r["stratum"] != st:
                    continue
                cid = str(r["caseid"])
                if c2s.get(cid, cid) not in test_subjects:
                    continue
                v1, v0 = r.get("crps_M1"), r.get("crps_M0")
                if v1 in ("", "nan", None) or v0 in ("", "nan", None):
                    continue
                m1.append(float(v1)); m0.append(float(v0))
            if m1:
                c1, c0 = np.mean(m1), np.mean(m0); xs.append((c0 - c1) / c0 * 100)
            else:
                xs.append(np.nan)
        out[st] = xs
    return out


def _tft_xpct_t7(base_tag):
    """TFT covariate X% in transition windows @7 min, or None if that baseline isn't trained yet."""
    import glob
    if not glob.glob(f"results/ablation_windows_{base_tag}.csv") and \
       not glob.glob(f"results/ablation_windows_{base_tag}_sh*of*.csv"):
        return None
    rows, _ = H.load_rows(base_tag); c2s = H.caseid_to_subject()
    tsub = {c2s.get(str(r["caseid"]), str(r["caseid"])) for r in rows}   # baseline CSV is test-only
    return _tft_covariate_xpct(rows, c2s, tsub, ["transition"], [7])["transition"][0]


def _baseline_xpct_ci(base_tag, stratum="transition", h=7, n_boot=1000, seed=0):
    """Trained-baseline covariate benefit X% = (CRPS_M0−CRPS_M1)/CRPS_M0×100 with a case-clustered
    bootstrap 95% CI (same method as the TiRex primary). Returns (x, lo, hi) or None if absent."""
    import glob
    if not glob.glob(f"results/ablation_windows_{base_tag}.csv"):
        return None
    rows, _ = H.load_rows(base_tag)
    by_case = {}
    for r in rows:
        if int(r["h_min"]) != h or (stratum != "all" and r["stratum"] != stratum):
            continue
        v1, v0 = r.get("crps_M1"), r.get("crps_M0")
        if v1 in ("", "nan", None) or v0 in ("", "nan", None):
            continue
        by_case.setdefault(str(r["caseid"]), []).append((float(v1), float(v0)))
    cids = list(by_case)
    if len(cids) < 3:
        return None
    c1 = np.array([np.mean([p[0] for p in by_case[c]]) for c in cids])
    c0 = np.array([np.mean([p[1] for p in by_case[c]]) for c in cids])
    pt = (c0.mean() - c1.mean()) / c0.mean() * 100
    rng = np.random.default_rng(seed); k = len(cids); b = []
    for _ in range(n_boot):
        s = rng.integers(0, k, k)
        b.append((c0[s].mean() - c1[s].mean()) / c0[s].mean() * 100)
    return float(pt), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))


def _mean_metric_by_h(rows, c2s, test_subjects, metric, hs):
    """Mean of a per-window metric (e.g. mae_M1, crps_M1) per horizon, on canonical test windows."""
    per = {h: [] for h in hs}
    for r in rows:
        cid = str(r["caseid"])
        if c2s.get(cid, cid) not in test_subjects:
            continue
        v = r.get(metric)
        if v in ("", "nan", None):
            continue
        h = int(r["h_min"])
        if h in per:
            per[h].append(float(v))
    return [float(np.mean(per[h])) if per[h] else np.nan for h in hs]


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — study design & cohort
# ══════════════════════════════════════════════════════════════════════════════
def figure1(tag):
    prim = load_primary(tag); hyp = load_hypo(tag)
    flow = json.load(open("results/cohort_flow.json"))
    n_cohort = len(windows_caseids(tag))
    # prefer the curated 3-example set (steady/transition/hypotensive) if present
    cur = f"outputs/figs/examples_curated_{tag}.npz"
    if os.path.exists(cur):
        ex = np.load(cur, allow_pickle=True); picks = [0, 1, 2]
    else:
        ex = np.load(f"outputs/figs/examples_{tag}.npz", allow_pickle=True); picks = _pick_examples(ex)

    fig = plt.figure(figsize=(S.W2 * 1.42, S.W2 * 0.80))     # 16:9 landscape
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=1.05, wspace=0.34)
    ax_sch = fig.add_subplot(gs[0, :2])     # a schematic (wide, top-left)
    ax_flow = fig.add_subplot(gs[0, 2])     # b cohort flow (top-right, same row as a)
    ax_ex = [fig.add_subplot(gs[1, i]) for i in range(3)]  # c examples (bottom row, 3 across)

    # a — task schematic (illustrative)
    _schematic(ax_sch)
    S.panel_letter(ax_sch, "a", dx=-0.10, dy=1.14)

    # b — cohort flow (funnel). Extend its axes down into the otherwise-empty gap so it
    # fills the same vertical footprint as the schematic + its covariate inset (top unchanged).
    pa = ax_sch.get_position(); pb = ax_flow.get_position()
    new_bottom = pa.y0 - 0.40*pa.height       # align with the schematic inset's bottom edge
    ax_flow.set_position([pb.x0, new_bottom, pb.width, pb.y1 - new_bottom])
    mover = None                                    # two-cohort funnel once the MOVER curation JSON is pulled down
    if os.path.exists("results/mover_cohort_flow.json"):
        mover = json.load(open("results/mover_cohort_flow.json"))
        mover["_n_cases"] = len(windows_caseids("mover_art"))
        mp = load_primary("mover_art")
        mover["_n_windows"] = mp["n_windows"] if mp else mover.get("n_cached", 0)
    _cohort_flow(ax_flow, flow, prim, hyp, n_cohort, mover)
    S.panel_letter(ax_flow, "b", dx=-0.10, dy=1.06)

    # c — three representative forecasts (steady / transition / hypotensive onset)
    titles = ["Steady", "Transition", "Hypotensive onset"]
    for ax, idx, tt in zip(ax_ex, picks, titles):
        _example_panel(ax, ex, idx, tt)
    S.panel_letter(ax_ex[0], "c", dx=-0.28, dy=1.14)
    ax_ex[0].legend(loc="upper left", fontsize=5.5, ncol=1)
    S.save_fig(fig, "Fig1_design_cohort")


def _schematic(ax):
    t = np.linspace(-30, 15, 300)
    rng = np.random.default_rng(3)
    base = 78 + 6*np.sin(t/6) - 0.25*t
    map_ = base + rng.normal(0, 1.2, t.size)
    ctx = t <= 0
    ax.plot(t[ctx], map_[ctx], color=S.C["ink"], lw=1.3)
    # median forecast + band on horizon
    fut = t > 0
    med = base[fut]
    ax.plot(t[fut], med, color=S.C["M1"], lw=1.6, label="TiRex-2 median")
    ax.fill_between(t[fut], med-6, med+6, color=S.C["M1_light"], alpha=0.5, lw=0, label="10–90% interval")
    ax.axvspan(-30, 0, color="#F2F2F2", zorder=0)
    ax.axvline(0, color="#888", lw=0.8, ls=":")
    ax.axhline(65, color=S.C["event"], lw=0.8, ls="--")
    ax.text(-15, 57, "context (30 min)", fontsize=6, color="#555", ha="center")
    ax.text(7.5, 57, "forecast horizon (→15 min)", fontsize=6, color="#555", ha="center")
    ax.text(13.5, 66.5, "MAP 65", fontsize=5.5, color=S.C["event"], ha="right")
    # covariate strip
    inf = 0.5*(1+np.tanh((t+2)/4))
    ax2 = ax.inset_axes([0, -0.40, 1, 0.24])
    ax2.plot(t, inf, color=S.C["M0"], lw=1.2)
    ax2.fill_between(t[fut], inf[fut], 0, color=S.C["M0_light"], alpha=0.6, lw=0)
    ax2.axvspan(-30, 0, color="#F2F2F2", zorder=0); ax2.axvline(0, color="#888", lw=0.8, ls=":")
    ax2.set_yticks([]); ax2.set_xlabel("time (min)")
    ax2.set_ylabel("drug\ninfusion", fontsize=6)
    ax2.set_ylim(-0.05, 1.6)                       # headroom so the label clears the line
    ax2.text(15, 1.22, "known future covariate (M1)", fontsize=5.5, color=S.C["M0"], ha="right")
    ax2.spines["left"].set_visible(False)
    ax.set_xlim(-30, 15); ax.set_ylim(55, 100); ax.set_xticks([])
    ax.set_ylabel("MAP (mmHg)"); ax.set_title("Forecasting task", loc="left")
    ax.legend(loc="upper right", fontsize=5.5)


def _funnel(ax, steps, x0, x1, header=None, fs=6.4):
    """Draw a vertical funnel of labelled boxes between axes-fraction x0..x1."""
    n = len(steps); gap = 0.055
    y_top = 0.92 if header else 1.0
    box_h = (y_top - (n - 1) * gap) / n
    xc = (x0 + x1) / 2
    if header:
        ax.text(xc, 0.975, header, transform=ax.transAxes, ha="center", va="center",
                fontsize=fs + 0.6, fontweight="bold")
    for i, (txt, col) in enumerate(steps):
        top = y_top - i * (box_h + gap); bot = top - box_h
        ax.add_patch(plt.Rectangle((x0, bot), x1 - x0, box_h, transform=ax.transAxes,
                     facecolor=col, edgecolor="#777", lw=0.7, zorder=2))
        ax.text(xc, (top + bot) / 2, txt, transform=ax.transAxes, ha="center", va="center",
                fontsize=fs, zorder=3)
        if i < n - 1:
            ax.annotate("", xy=(xc, bot - gap + 0.006), xytext=(xc, bot - 0.004),
                        xycoords="axes fraction", arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.0))


def _cohort_flow(ax, flow, prim, hyp, n_cohort, mover=None):
    """Curation funnel. Development cohort (VitalDB) always; if `mover` (from mover_cohort_flow.json
    + windowed counts) is present, render VitalDB and MOVER side by side as a two-cohort panel."""
    ax.axis("off"); ax.set_ylim(0, 1)
    vd = [
        (f"VitalDB cases scanned\nn = {flow['n_local_scanned']:,}", "#E8EEF2"),
        (f"Anesthetic cohort\n(remifentanil + propofol)\nn = {flow['included_N']:,}", S.C["M1_light"]),
        (f"Cases with ≥ 1 window\nn = {n_cohort:,}", "#CFE3E7"),
        (f"Forecast windows\nn = {prim['n_windows']:,}", "#EAD9BD"),
    ]
    if not mover:
        _funnel(ax, vd, 0.04, 0.96)
        return
    _funnel(ax, vd, 0.02, 0.47, "Development: VitalDB", fs=5.6)
    mv = []
    if mover.get("n_candidates"):
        mv.append((f"SIS cases screened\nn = {mover['n_candidates']:,}", "#E8EEF2"))
    mv += [
        (f"Invasive arterial +\ninfusion cohort\nn = {mover['included_N']:,}", S.C["M1_light"]),
        (f"Cases with ≥ 1 window\nn = {mover['_n_cases']:,}", "#CFE3E7"),
        (f"Forecast windows\nn = {mover['_n_windows']:,}", "#EAD9BD"),
    ]
    _funnel(ax, mv, 0.53, 0.98, "External: MOVER", fs=5.6)


def _pick_examples(ex):
    """Choose a steady, a transition and a hypotensive-onset example from the npz.
    Filters artifact traces (context+horizon min < 60 mmHg) so panels look clean."""
    truth = ex["truth"]; ctx = ex["context"]; n = truth.shape[0]
    fin = lambda a: a[np.isfinite(a)]
    st = []
    for i in range(n):
        t = fin(truth[i]); c = fin(ctx[i])
        both = np.concatenate([c, t]) if (c.size and t.size) else (t if t.size else c)
        if t.size < 5 or both.size < 5:
            st.append(dict(i=i, rng=1e9, mn=0.0, below=0.0, cmin=0.0)); continue
        st.append(dict(i=i, rng=float(t.max()-t.min()), mn=float(t.min()),
                       below=float(np.mean(t < 65)), cmin=float(both.min())))
    hypo = max(st, key=lambda s: s["below"])                     # clearest onset below 65
    phys = [s for s in st if s["cmin"] >= 60 and s["mn"] >= 70]  # no artifacts, stays normotensive
    steady = min(phys or st, key=lambda s: s["rng"])             # flattest
    cand = [s for s in st if s["cmin"] >= 60 and s["mn"] >= 66 and s["i"] not in (steady["i"], hypo["i"])]
    cand.sort(key=lambda s: s["rng"])
    trans = cand[len(cand)//2] if cand else st[min(1, n-1)]      # mid-range drift
    return [steady["i"], trans["i"], hypo["i"]]


def _example_panel(ax, ex, i, title):
    dt_min = DT_S/60.0
    ctx = ex["context"][i]; truth = ex["truth"][i]; q = ex["q_ce"][i]
    tc = np.arange(-len(ctx), 0)*dt_min
    th = (np.arange(len(truth))+1)*dt_min
    ax.axvspan(tc[0], 0, color="#F2F2F2", zorder=0)                   # shade context window (matches panel a)
    ax.plot(tc, ctx, color=S.C["ink"], lw=1.1, zorder=4)             # observed history (model input)
    ax.plot(th, truth, color="#AEB4BA", lw=1.2, zorder=2, label="observed (truth)")   # future truth, muted
    ax.fill_between(th, q[S.Q_LO], q[S.Q_HI], color=S.C["M1_light"], alpha=0.5, lw=0, zorder=1, label="10–90%")
    ax.plot(th, q[S.Q_MED], color=S.C["M1"], lw=1.9, zorder=5, label="TiRex-2 median")  # forecast on top
    ax.axvline(0, color="#888", lw=0.7, ls=":", zorder=3)
    ax.axhline(65, color=S.C["event"], lw=0.7, ls="--", zorder=1)
    ax.set_title(title, loc="center", fontsize=7)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("MAP (mmHg)")
    ax.set_xlim(tc[0], th[-1]); ax.set_ylim(40, 120)     # shared y-range across the 3 examples


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — accuracy & value of the drug covariate
# ══════════════════════════════════════════════════════════════════════════════
def figure2(tag):
    prim = load_primary(tag)
    hs = MAIN_H
    c2s = H.caseid_to_subject()
    trows, _ = H.load_rows(tag); tsub = eval_subjects(trows, c2s)
    bl = available_baselines(tag)          # trained comparators on the CE cohort (TFT[, PatchTST])

    # landscape layout: top row = accuracy (a, d); bottom row = value of the drug covariate (b, c, e).
    # (Univariate zero-shot TSFMs can't read the covariate at all — 0 by construction — noted in the
    # caption rather than drawn as an empty bar in e.)
    fig, axs = plt.subplot_mosaic([["a", "a", "a", "d", "d", "d"],
                                   ["b", "b", "c", "c", "e", "e"]],
                                  figsize=(S.W2 * 1.34, S.W2 * 0.64),
                                  gridspec_kw=dict(hspace=0.5, wspace=1.0,
                                                   left=0.06, right=0.985, top=0.92, bottom=0.11))

    # a — forecast accuracy (MAE vs horizon): zero-shot TiRex vs every trained baseline (matched test)
    a = axs["a"]
    a.plot(hs, _mean_metric_by_h(trows, c2s, tsub, "mae_M1", hs), "-o", color=S.C["M1"], lw=2.2, ms=4.5,
           label="TiRex-2 (zero-shot)", zorder=6)
    for bb in bl:
        br, _ = H.load_rows(bb["tag"])
        a.plot(hs, _mean_metric_by_h(br, c2s, tsub, "mae_M1", hs), bb["ls"], marker=bb["mk"],
               color=bb["col"], ms=3.5, label=f"{bb['disp']} (trained)")
    y7 = strat(prim, 7, "all")["Y_pct_vs_persistence"]
    a.text(0.5, 0.05, f"all beat persistence (−{y7:.0f}% CRPS)", transform=a.transAxes,
           fontsize=6, color="#555", ha="center")
    S.finish(a, "forecast horizon (min)", "MAE (mmHg)", "Forecast accuracy: zero-shot vs trained")
    a.set_xticks(hs); a.legend(loc="upper left"); S.panel_letter(a, "a")

    # b — TiRex-2 covariate benefit by window type (characterization of the deployed model;
    # the cross-model comparison of covariate value lives in panels c and e).
    b = axs["b"]
    for s_name, col, mk in [("all", S.C["M1"], "o"), ("transition", S.C["transition"], "^"),
                            ("steady", S.C["steady"], "v")]:
        xs = [strat(prim, h, s_name)["X_pct_withpast"] for h in hs]
        lo = [strat(prim, h, s_name)["X_pct_withpast_CI95"][0] for h in hs]
        hi = [strat(prim, h, s_name)["X_pct_withpast_CI95"][1] for h in hs]
        b.plot(hs, xs, "-", color=col, marker=mk, label=s_name)
        b.fill_between(hs, lo, hi, color=col, alpha=0.15, lw=0)
    b.axhline(0, color="#999", lw=0.7, ls="--")
    S.finish(b, "forecast horizon (min)", "CRPS reduction M0→M1 (%)", "TiRex-2: value by window type")
    b.set_xticks(hs); b.legend(loc="upper left", fontsize=5.6, title="window"); S.panel_letter(b, "b")

    # c — which drug covariate, and which model exploits it: grouped forest, points + case-clustered
    # 95% CIs (all models on one linear axis — the effect can be ≤0 for phenylephrine, which log can't
    # show). Each trained baseline is drawn per cohort where it exists (auto-fills once trained).
    c = axs["c"]
    arms = [("CE (effect-site)", tag, 2), ("RATE (infusion)", RATE_TAG, 1), ("Phenylephrine", PRESSOR_TAG, 0)]
    offs = np.linspace(0.26, -0.26, 1 + len(MATCHED_BASELINES))
    seen = set()
    for lab, t, yb in arms:
        blk = strat(load_primary(t), 7, "transition")
        x = blk["X_pct_withpast"]; ci = blk["X_pct_withpast_CI95"]
        c.errorbar(x, yb + offs[0], xerr=[[x - ci[0]], [ci[1] - x]], fmt="o", color=S.C["M1"],
                   capsize=2, lw=1.1, ms=4, zorder=6)
        for j, m in enumerate(MATCHED_BASELINES, start=1):
            r = _baseline_xpct_ci(f"baseline-{m['key']}_{t}", "transition", 7)
            if r is None:
                continue
            xv, lo, hi = r
            c.errorbar(xv, yb + offs[j], xerr=[[xv - lo], [hi - xv]], fmt=m["mk"], color=m["col"],
                       capsize=2, lw=1.1, ms=4)
            seen.add(m["disp"])
    c.axvline(0, color="#999", lw=0.7, ls="--")
    c.set_yticks([ar[2] for ar in arms]); c.set_yticklabels([ar[0] for ar in arms])
    c.set_ylim(-0.5, 2.5)
    c.set_xlabel("CRPS reduction, transition @7 min (%)")
    c.set_title("Which covariate, which model?", loc="center")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], color=S.C["M1"], marker="o", ls="none", label="TiRex-2 (zero-shot)")]
    handles += [Line2D([], [], color=m["col"], marker=m["mk"], ls="none", label=f"{m['disp']} (trained)")
                for m in MATCHED_BASELINES if m["disp"] in seen]
    c.legend(handles=handles, loc="lower right", fontsize=5.0); S.panel_letter(c, "c")

    # d — instantaneous MAE vs Kapral (external / internal), TiRex vs trained baselines
    d = axs["d"]
    _kapral_panel(d, tag)
    S.panel_letter(d, "d")

    # e — covariate value by model (CE, transition @7 min): the same effect as panel c's CE row,
    # as bars for an at-a-glance read, with case-clustered 95% CIs. Univariate zero-shot TSFMs
    # (Chronos/TimesFM/Moirai) extract 0 by construction and are noted in the caption, not drawn.
    e = axs["e"]
    tb = strat(prim, 7, "transition")
    bars = [("TiRex-2\n(zero-shot)", tb["X_pct_withpast"], tb["X_pct_withpast_CI95"], S.C["M1"])]
    for m in MATCHED_BASELINES:
        r = _baseline_xpct_ci(f"baseline-{m['key']}_{tag}", "transition", 7)
        if r is not None:
            bars.append((f"{m['disp']}\n(trained)", r[0], [r[1], r[2]], m["col"]))
    xpos = np.arange(len(bars))
    for xi, (lab, v, ci, col) in zip(xpos, bars):
        e.bar(xi, v, width=0.62, color=col, edgecolor="white", lw=0.5)
        top = v
        if ci is not None:
            e.errorbar(xi, v, yerr=[[v - ci[0]], [ci[1] - v]], fmt="none", ecolor="#333", capsize=2.5, lw=1.0)
            top = ci[1]
        e.text(xi, top + 0.3, f"{v:+.1f}%", ha="center", va="bottom", fontsize=5.4)
    e.axhline(0, color="#999", lw=0.7)
    e.set_ylim(top=max((ci[1] if ci else v) for _, v, ci, _ in bars) * 1.16 + 0.8)
    e.set_xticks(xpos); e.set_xticklabels([b[0] for b in bars], fontsize=5.2)
    S.finish(e, None, "CRPS reduction from\ndrug covariate (%)", "Covariate value by model")
    S.panel_letter(e, "e")
    S.save_fig(fig, "Fig2_accuracy_covariate")


def _kapral_panel(ax, tag):
    # our instantaneous endpoint MAE on the matched test split (TiRex + every trained baseline)
    c2s = H.caseid_to_subject()
    rows, _ = H.load_rows(tag)
    tsub = eval_subjects(rows, c2s)
    hs = MAIN_H
    our = _mean_metric_by_h(rows, c2s, tsub, "mae_inst_M1", hs)
    # Kapral digitized curves (instantaneous)
    K = {}
    for r in csv.DictReader(open("results/kapral_mae_curves.csv")):
        K.setdefault((r["dataset"], r["curve"]), []).append((float(r["forecast_min"]), float(r["mae_mmHg"])))
    def curve(ds, c):
        pts = sorted(K.get((ds, c), []));
        return np.array([p[0] for p in pts]), np.array([p[1] for p in pts])
    # foils in muted purple (solid vs dashed to tell them apart); ours pops in bold teal
    for ds, col, ls, lab in [("internal", "#9B8AB0", "-", "Kapral internal"),
                             ("external", "#9B8AB0", "--", "Kapral external")]:
        xm, ym = curve(ds, "mean")
        if xm.size:
            ax.plot(xm, ym, ls, color=col, lw=1.3, label=lab)
            xl, yl = curve(ds, "lower"); xu, yu = curve(ds, "upper")
            if xl.size and xu.size:
                yl_i = np.interp(xm, xl, yl); yu_i = np.interp(xm, xu, yu)
                ax.fill_between(xm, yl_i, yu_i, color=col, alpha=0.10, lw=0)
    for bb in available_baselines(tag):
        br, _ = H.load_rows(bb["tag"])
        ax.plot(hs, _mean_metric_by_h(br, c2s, tsub, "mae_inst_M1", hs), bb["ls"], marker=bb["mk"],
                color=bb["col"], lw=1.6, ms=4, zorder=5, label=f"{bb['disp']} (trained, ours)")
    ax.plot(hs, our, "-o", color=S.C["M1"], lw=2.4, ms=6, mec="white", mew=1.0,
            zorder=6, label="TiRex-2 (zero-shot, ours)")
    ax.set_xlim(0, 7.4); ax.set_ylim(0, None)
    S.finish(ax, "forecast distance (min)", "instantaneous MAE (mmHg)", "Accuracy vs Kapral et al.")
    ax.legend(loc="upper left", fontsize=5.6)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — impending-hypotension prediction: zero-shot vs trained vs SOTA
# ══════════════════════════════════════════════════════════════════════════════
TFT_COL = "#566573"     # slate — the trained TFT baseline
PATCH_COL = "#3D5A80"   # steel blue — the trained PatchTST baseline

# Trained supervised baselines, in display order. Each entry is rendered wherever a matched
# comparator appears; models are distinguished by colour + linestyle + marker so panels stay
# legible. Extend this list to add more architectures — figures/tables pick them up.
MATCHED_BASELINES = [
    dict(key="tft",      disp="TFT",      col=TFT_COL,   ls="--", mk="s"),
    dict(key="patchtst", disp="PatchTST", col=PATCH_COL, ls="-.", mk="^"),
]

def available_baselines(tag):
    """Matched baselines whose files are present for this tag (graceful degrade)."""
    out = []
    for b in MATCHED_BASELINES:
        bt = f"baseline-{b['key']}_{tag}"
        if (os.path.exists(f"results/ablation_windows_{bt}.csv")
                and os.path.exists(f"results/matched_comparison_{bt}.json")):
            out.append({**b, "tag": bt})
    return out

# Other zero-shot time-series foundation models — evaluated identically to TiRex (no training),
# univariate (they don't ingest the drug covariate). Their own visual tier, distinct from the
# trained baselines above. Colours are local to the zero-shot figure/table (no clash there).
ZEROSHOT_TSFM = [
    dict(key="chronos", disp="Chronos-Bolt",  col="#D35400", ls="--", mk="s"),
    dict(key="timesfm", disp="TimesFM-2.5",   col="#8E44AD", ls="-.", mk="^"),
    dict(key="moirai",  disp="Moirai-1.1-R",  col="#2C7A3F", ls=":",  mk="D"),
]

def available_zeroshot(tag):
    out = []
    for m in ZEROSHOT_TSFM:
        bt = f"baseline-{m['key']}_{tag}"
        if (os.path.exists(f"results/ablation_windows_{bt}.csv")
                and os.path.exists(f"results/matched_comparison_{bt}.json")):
            out.append({**m, "tag": bt})
    return out

def load_matched(base_tag):
    return json.load(open(f"results/matched_comparison_{base_tag}.json"))

def eval_subjects(rows, c2s, seed=0):
    """All subjects in the cohort. With 5-fold CV the trained baselines carry out-of-fold (held-out)
    predictions on ALL cases, and TiRex/zero-shot models are inherently held-out (no training), so
    every comparison is scored on the full cohort — not a 20% split. Metric functions naturally use
    each model's available rows, so a model covering fewer cases is scored on what it has."""
    return {c2s.get(str(r["caseid"]), str(r["caseid"])) for r in rows}

def _scores_subj(rows, c2s, test_subjects, h, risk_col="risk_M1", ev="hypo_event"):
    y, s = [], []
    for r in rows:
        if int(r["h_min"]) != h or r.get(risk_col) in ("", "nan", None):
            continue
        if c2s.get(str(r["caseid"]), str(r["caseid"])) not in test_subjects:
            continue
        y.append(float(r[ev])); s.append(float(r[risk_col]))
    return np.array(y), np.array(s)

def figure3(tag):
    rows, _ = H.load_rows(tag); c2s = H.caseid_to_subject()
    bl = available_baselines(tag)                               # trained comparators present
    if not bl:                                                  # nothing trained yet -> fall back to TFT tag
        bl = [{**MATCHED_BASELINES[0], "tag": f"baseline-tft_{tag}"}]
    for b in bl:                                               # attach each baseline's rows + matched json
        b["rows"], _ = H.load_rows(b["tag"]); b["M"] = load_matched(b["tag"])
    primary = bl[0]                                            # TFT — the single-comparator panel (a)
    base_rows = primary["rows"]; M = primary["M"]              # TiRex numbers + foils are shared across baselines
    test_subj = eval_subjects(rows, c2s)             # identical split for every panel
    hs = sorted(int(k) for k in M["per_horizon"])

    fig = plt.figure(figsize=(9.6, 5.4))                  # 16:9 landscape
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.42,
                          left=0.08, right=0.97, top=0.93, bottom=0.11)
    ax_roc = fig.add_subplot(gs[0, 0]); ax_auc = fig.add_subplot(gs[0, 1]); ax_cal = fig.add_subplot(gs[0, 2])
    ax_pr = fig.add_subplot(gs[1, 0]); ax_dca = fig.add_subplot(gs[1, 1]); ax_bar = fig.add_subplot(gs[1, 2])

    # a — ROC at 5 and 7 min: TiRex (solid) vs every trained baseline (its own linestyle), matched test
    for h, col in [(5, S.C["M1"]), (7, S.C["transition"])]:
        y, s = _scores_subj(rows, c2s, test_subj, h)
        fpr, tpr, _ = H.roc_points(y, s); au = H.auroc(y, s)
        ax_roc.plot(fpr, tpr, "-", color=col, lw=1.6, label=f"TiRex-2 {h} min ({au:.3f})")
        ax_roc.plot(0.10, float(np.interp(0.10, fpr, tpr)), "o", color=col, ms=5,
                    mec="white", mew=0.6, zorder=6)                       # spec >= 0.90 operating point
        for b in bl:                                                      # TFT (dashed) + PatchTST (dash-dot)
            yb, sb = _scores_subj(b["rows"], c2s, test_subj, h)
            fb, tb, _ = H.roc_points(yb, sb); aub = H.auroc(yb, sb)
            ax_roc.plot(fb, tb, b["ls"], color=col, lw=1.0, label=f"{b['disp']} {h} min ({aub:.3f})")
    ax_roc.plot([0, 1], [0, 1], color="#BBB", lw=0.7, ls=":")
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.005)
    S.finish(ax_roc, "1 − specificity", "sensitivity", "ROC — zero-shot vs trained")
    ax_roc.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02), fontsize=5.2); S.panel_letter(ax_roc, "a")

    # b — AUROC vs horizon: zero-shot TiRex vs trained TFT (matched) + foils (THE panel)
    def arr(key, field):
        return [M["per_horizon"][str(h)][key][field] if M["per_horizon"][str(h)][key] else np.nan for h in hs]
    ta = arr("tirex_M1", "auroc"); tlo = [M["per_horizon"][str(h)]["tirex_M1"]["ci"][0] for h in hs]; thi = [M["per_horizon"][str(h)]["tirex_M1"]["ci"][1] for h in hs]
    # trained comparators first (muted, thinner) so the TiRex-2 line reads as the protagonist on top
    for b in bl:                                               # every trained baseline (matched)
        ph = b["M"]["per_horizon"]
        fa = [ph[str(h)]["tft_M1"]["auroc"] if ph[str(h)]["tft_M1"] else np.nan for h in hs]
        flo = [ph[str(h)]["tft_M1"]["ci"][0] for h in hs]; fhi = [ph[str(h)]["tft_M1"]["ci"][1] for h in hs]
        ax_auc.fill_between(hs, flo, fhi, color=b["col"], alpha=0.08, lw=0, zorder=1)
        ax_auc.plot(hs, fa, b["ls"], marker=b["mk"], color=b["col"], ms=3.1, lw=1.3, alpha=0.8,
                    zorder=4, label=f"{b['disp']} (trained, ours)")
    ax_auc.fill_between(hs, tlo, thi, color=S.C["M1"], alpha=0.18, lw=0, zorder=2)
    ax_auc.plot(hs, ta, "-o", color=S.C["M1"], lw=2.6, ms=4.6, zorder=6, label="TiRex-2 (zero-shot, ours)")
    for h, (ki, ke) in S.KAPRAL_AUROC.items():
        ax_auc.plot(h, ke, "D", color=S.C["kapral"], ms=5, mec="white", mew=0.5, zorder=6)
    ax_auc.plot([], [], "D", color=S.C["kapral"], label="Kapral (TFT, ext.)")
    for h, z in S.ZHU_AUROC.items():
        ax_auc.plot(h, z, "s", color=S.C["zhu"], ms=5, mec="white", mew=0.5, zorder=6)
    ax_auc.plot([], [], "s", color=S.C["zhu"], label="Zhu (Transformer, ext.)")
    S.finish(ax_auc, "forecast horizon (min)", "hypotension AUROC", "Zero-shot vs trained vs SOTA")
    ax_auc.set_xticks(hs); ax_auc.set_ylim(0.80, 1.0)
    _h, _l = ax_auc.get_legend_handles_labels()          # TiRex drawn last (on top) -> pull it first in legend
    _o = sorted(range(len(_l)), key=lambda i: 0 if "TiRex" in _l[i] else 1)
    ax_auc.legend([_h[i] for i in _o], [_l[i] for i in _o], loc="upper right", fontsize=5.4)
    S.panel_letter(ax_auc, "b")

    # c — calibration at 5 min: TiRex vs each trained baseline (matched test)
    y5, s5 = _scores_subj(rows, c2s, test_subj, 5)
    mp, of, _, ece = H.calibration(y5, s5, n_bins=10)
    ax_cal.plot([0, 1], [0, 1], color="#BBB", lw=0.7, ls=":")
    for b in bl:
        yb, sb = _scores_subj(b["rows"], c2s, test_subj, 5)
        mpb, ofb, _, eceb = H.calibration(yb, sb, n_bins=10)
        ax_cal.plot(mpb, ofb, b["ls"], marker=b["mk"], color=b["col"], ms=3, label=f"{b['disp']} (ECE {eceb:.3f})")
    ax_cal.plot(mp, of, "-o", color=S.C["M1"], ms=3, label=f"TiRex-2 (ECE {ece:.3f})")
    ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)
    S.finish(ax_cal, "predicted risk", "observed frequency", "Calibration @5 min")
    ax_cal.legend(loc="lower right", fontsize=5.4); S.panel_letter(ax_cal, "c")

    # d — AUPRC vs horizon: TiRex vs each trained baseline (matched test), rising-prevalence chance line
    ap, prev = [], []
    for h in hs:
        y, s = _scores_subj(rows, c2s, test_subj, h)
        ap.append(H.auprc(y, s)); prev.append(float(y.mean()) if len(y) else np.nan)
    ax_pr.plot(hs, ap, "-o", color=S.C["M1"], label="TiRex-2 (zero-shot)")
    for b in bl:
        apb = [H.auprc(*_scores_subj(b["rows"], c2s, test_subj, h)) for h in hs]
        ax_pr.plot(hs, apb, b["ls"], marker=b["mk"], color=b["col"], ms=3, label=f"{b['disp']} (trained)")
    ax_pr.plot(hs, prev, ":", color=S.C["persist"], label="prevalence (chance)")
    S.finish(ax_pr, "forecast horizon (min)", "AUPRC", "Precision–recall")
    ax_pr.set_xticks(hs); ax_pr.set_ylim(0, 1); ax_pr.legend(loc="upper right", fontsize=5.4); S.panel_letter(ax_pr, "d")

    # e — decision curve @5 min: TiRex vs each trained baseline (matched test), net benefit inline
    def net_benefit(y, s, pts):
        N = len(y); nb = []
        for p in pts:
            fl = s >= p; tp = np.sum(fl & (y == 1)); fp = np.sum(fl & (y == 0)); w = p / (1 - p)
            nb.append(tp / N - fp / N * w)
        return np.array(nb)
    pts = np.linspace(0.01, 0.5, 40); pv = y5.mean()
    nb = net_benefit(y5, s5, pts)
    nball = np.array([pv - (1 - pv) * (p / (1 - p)) for p in pts])
    ax_dca.plot(pts, nb, color=S.C["M1"], lw=1.4, label="TiRex-2")
    for b in bl:
        yb, sb = _scores_subj(b["rows"], c2s, test_subj, 5)
        ax_dca.plot(pts, net_benefit(yb, sb, pts), b["ls"], color=b["col"], lw=1.2, label=b["disp"])
    ax_dca.plot(pts, nball, color=S.C["persist"], lw=1.0, label="treat all")
    ax_dca.axhline(0, color="#999", lw=0.8, label="treat none")
    ax_dca.set_ylim(-0.02, max(0.02, np.nanmax(nb) * 1.15)); ax_dca.set_xlim(pts.min(), pts.max())
    S.finish(ax_dca, "threshold probability", "net benefit", "Decision curve @5 min")
    ax_dca.legend(loc="upper right", fontsize=5.4); S.panel_letter(ax_dca, "e")

    # f — head-to-head AUROC bars at 5 & 7 min: zero-shot vs trained vs foils (matched split)
    from matplotlib.patches import Patch
    groups = [5, 7]
    def au_of(mj, key):
        return (lambda h: (mj["per_horizon"][str(h)][key] or {}).get("auroc")
                if isinstance(mj["per_horizon"][str(h)][key], dict) else mj["per_horizon"][str(h)][key])
    series_f = [("TiRex-2 (zero-shot)", S.C["M1"], au_of(M, "tirex_M1"))]
    series_f += [(f"{b['disp']} (trained)", b["col"], au_of(b["M"], "tft_M1")) for b in bl]
    series_f += [("Kapral (ext.)", S.C["kapral"], au_of(M, "kapral_ext")),
                 ("Zhu (ext.)", S.C["zhu"], au_of(M, "zhu_ext"))]
    nser = len(series_f); w = min(0.20, 0.86 / nser); x = np.arange(len(groups))
    tir_val = {h: series_f[0][2](h) for h in groups}          # TiRex-2 reference level per group
    for j, (lab, col, fn) in enumerate(series_f):
        is_tir = (j == 0)                                     # TiRex is the protagonist -> make it pop
        for xi, h in zip(x + (j - (nser - 1) / 2) * w, groups):
            v = fn(h)
            if v is None:
                continue
            ax_bar.bar(xi, v, width=w, color=col, alpha=(1.0 if is_tir else 0.65),
                       edgecolor=(S.C["ink"] if is_tir else "white"), lw=(1.1 if is_tir else 0.4),
                       zorder=(4 if is_tir else 3))
    # dashed TiRex-2 line across each group: instantly shows which bars clear the zero-shot level
    for xi, h in zip(x, groups):
        half = (nser / 2 + 0.1) * w
        ax_bar.plot([xi - half, xi + half], [tir_val[h], tir_val[h]], ls=(0, (4, 2)),
                    color=S.C["M1"], lw=1.0, zorder=5)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels([f"{g} min" for g in groups])
    ax_bar.set_ylim(0.80, 1.02)
    S.finish(ax_bar, None, "hypotension AUROC", "Head-to-head @5 / 7 min")
    handles = [Patch(facecolor=S.C["M1"], edgecolor=S.C["ink"], lw=1.1, label="TiRex-2 (zero-shot)")]
    handles += [Patch(facecolor=col, alpha=0.65, label=lab) for lab, col, _ in series_f[1:]]
    ax_bar.legend(handles=handles, loc="upper center", ncol=2, fontsize=5.0, framealpha=0.9,
                  handlelength=1.3, columnspacing=1.1, borderpad=0.4)
    S.panel_letter(ax_bar, "f")

    S.save_fig(fig, "Fig4_hypotension_vs_sota")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — zero-shot foundation-model benchmark (TiRex vs other TSFMs)  (headline)
# ══════════════════════════════════════════════════════════════════════════════
def figure_zeroshot(tag, save="Fig3_zeroshot_tsfm", cohort_note=None):
    """Among zero-shot TSFMs (no training), TiRex vs Chronos/TimesFM/Moirai on the identical
    matched test split: (a) hypotension AUROC, (b) forecasting CRPS, (c) calibration @10 min,
    (d) AUPRC vs horizon. TiRex leads on both tasks and uniquely ingests the drug covariate.
    Pass cohort_note (+ a distinct save name) to render the external-cohort (MOVER) variant."""
    zs = available_zeroshot(tag)
    if not zs:
        print("  (no zero-shot TSFM results — skip Fig 3)", flush=True); return
    for z in zs:
        z["M"] = load_matched(z["tag"]); z["rows"], _ = H.load_rows(z["tag"])
    M = zs[0]["M"]; hs = sorted(int(k) for k in M["per_horizon"])
    c2s = H.caseid_to_subject(); trows, _ = H.load_rows(tag)
    tsub = eval_subjects(trows, c2s)

    fig, axs = plt.subplot_mosaic([["a", "b"], ["c", "d"]], figsize=(S.W2, S.W2 * 0.82),
                                  gridspec_kw=dict(hspace=0.5, wspace=0.28))
    # a — hypotension AUROC vs horizon
    axa = axs["a"]
    ta = [M["per_horizon"][str(h)]["tirex_M1"]["auroc"] for h in hs]
    tlo = [M["per_horizon"][str(h)]["tirex_M1"]["ci"][0] for h in hs]
    thi = [M["per_horizon"][str(h)]["tirex_M1"]["ci"][1] for h in hs]
    axa.fill_between(hs, tlo, thi, color=S.C["M1"], alpha=0.13, lw=0)
    axa.plot(hs, ta, "-o", color=S.C["M1"], lw=2.2, ms=4, label="TiRex-2 (ours)", zorder=6)
    for z in zs:
        a = [z["M"]["per_horizon"][str(h)]["tft_M1"]["auroc"] for h in hs]
        axa.plot(hs, a, z["ls"], marker=z["mk"], color=z["col"], ms=3, lw=1.2, label=z["disp"])
    axa.set_xticks(hs); axa.set_ylim(0.80, 1.0)
    S.finish(axa, "forecast horizon (min)", "hypotension AUROC", "Impending hypotension")
    axa.legend(loc="upper right", fontsize=5.6); S.panel_letter(axa, "a")

    # b — forecasting CRPS vs horizon (lower is better)
    axb = axs["b"]
    crps_tx = _mean_metric_by_h(trows, c2s, tsub, "crps_M1", hs)
    axb.plot(hs, crps_tx, "-o", color=S.C["M1"], lw=2.2, ms=4, label="TiRex-2 (ours)", zorder=6)
    for z in zs:
        c = _mean_metric_by_h(z["rows"], c2s, tsub, "crps_M1", hs)
        axb.plot(hs, c, z["ls"], marker=z["mk"], color=z["col"], ms=3, lw=1.2, label=z["disp"])
    axb.set_xticks(hs)
    S.finish(axb, "forecast horizon (min)", "CRPS (mmHg)", "Probabilistic forecast")
    axb.legend(loc="upper left", fontsize=5.6); S.panel_letter(axb, "b")

    # c — calibration (reliability) at 10 min: predicted risk vs observed hypotension frequency
    axc = axs["c"]
    y, s = _scores_subj(trows, c2s, tsub, 10)
    mp, of, _, ece = H.calibration(y, s, n_bins=10)
    axc.plot([0, 1], [0, 1], color="#BBB", lw=0.7, ls=":")
    for z in zs:
        yz, sz = _scores_subj(z["rows"], c2s, tsub, 10)
        mpz, ofz, _, ecez = H.calibration(yz, sz, n_bins=10)
        axc.plot(mpz, ofz, z["ls"], marker=z["mk"], color=z["col"], ms=2.6, lw=1.1,
                 label=f"{z['disp']} ({ecez:.3f})")
    axc.plot(mp, of, "-o", color=S.C["M1"], ms=3, lw=2.0, label=f"TiRex-2 ({ece:.3f})", zorder=6)
    axc.set_xlim(0, 1); axc.set_ylim(0, 1)
    S.finish(axc, "predicted risk", "observed frequency", "Calibration @10 min (ECE)")
    axc.legend(loc="upper left", fontsize=5.4); S.panel_letter(axc, "c")

    # d — AUPRC vs horizon with rising-prevalence chance line
    axd = axs["d"]
    ap = [H.auprc(*_scores_subj(trows, c2s, tsub, h)) for h in hs]
    prev = [float(_scores_subj(trows, c2s, tsub, h)[0].mean()) for h in hs]
    axd.plot(hs, ap, "-o", color=S.C["M1"], lw=2.2, ms=4, label="TiRex-2 (ours)", zorder=6)
    for z in zs:
        apz = [H.auprc(*_scores_subj(z["rows"], c2s, tsub, h)) for h in hs]
        axd.plot(hs, apz, z["ls"], marker=z["mk"], color=z["col"], ms=3, lw=1.2, label=z["disp"])
    axd.plot(hs, prev, ":", color=S.C["persist"], lw=1.0, label="prevalence (chance)")
    axd.set_xticks(hs); axd.set_ylim(0, 1)
    S.finish(axd, "forecast horizon (min)", "AUPRC", "Precision–recall")
    axd.legend(loc="upper right", fontsize=5.4); S.panel_letter(axd, "d")

    if cohort_note:
        fig.suptitle(cohort_note, y=0.99, fontsize=8, fontweight="bold")
    S.save_fig(fig, save)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — clinical translation & robustness
# ══════════════════════════════════════════════════════════════════════════════
def _load_json(path):
    return json.load(open(path)) if os.path.exists(path) else None


def _mini_strip(fig, cell, keys, models, valfn, ylim, letter, ylabel, blocktitle, hs=(1, 3, 5, 7, 10, 15)):
    """1xN strip of horizon line plots inside one gridspec cell, sharing a common y-axis (only the
    leftmost subplot keeps y ticks/label). keys=[(data_key, subplot_title)];
    valfn(model_tag, data_key, horizon)->float|None. Models overlaid per subplot; TiRex-2 highlighted."""
    n = len(keys)
    sub = cell.subgridspec(1, n, wspace=0.08)
    ax0 = None
    for i, (k, ptitle) in enumerate(keys):
        ax = fig.add_subplot(sub[0, i], sharey=ax0) if ax0 is not None else fig.add_subplot(sub[0, i])
        if ax0 is None:
            ax0 = ax
        for disp, mt, col, is_tir, mk in models:
            ys = [valfn(mt, k, h) for h in hs]
            if all(v is None for v in ys):
                continue
            ax.plot(hs, ys, "-", marker=mk, color=col, lw=(1.7 if is_tir else 0.95),
                    ms=(3.6 if is_tir else 2.9), mew=0.3, mec="white",
                    alpha=(1.0 if is_tir else 0.72), zorder=(5 if is_tir else 3), solid_capstyle="round")
        ax.set_title(ptitle, fontsize=5.8, pad=2.0)
        ax.set_xlim(0.5, 15.5); ax.set_xticks([1, 5, 10, 15]); ax.set_ylim(*ylim)
        ax.tick_params(labelsize=5.0, length=1.8, pad=1.2)
        if i > 0:
            ax.tick_params(labelleft=False)
    pos = cell.get_position(fig)
    fig.text(pos.x0 + pos.width / 2, pos.y1 + 0.022, blocktitle, ha="center", va="bottom",
             fontsize=7.5, fontweight="bold")
    fig.text(pos.x0 + pos.width / 2, pos.y0 - 0.048, "forecast horizon (min)", ha="center", va="top", fontsize=6)
    fig.text(pos.x0 - 0.026, pos.y0 + pos.height / 2, ylabel, rotation=90, ha="right", va="center", fontsize=6)
    S.panel_letter(ax0, letter, dx=-0.26, dy=1.20)


def figure4(tag):
    sg = load_subgroup(tag, 5)
    fig = plt.figure(figsize=(9.8, 6.6))                  # landscape
    # left column stacks a (lead) -> d strip -> b strip; the forest spans the full height on the right
    outer = fig.add_gridspec(1, 2, width_ratios=[1.0, 0.33], wspace=0.16,
                             left=0.085, right=0.965, top=0.92, bottom=0.10)
    left = outer[0, 0].subgridspec(3, 1, height_ratios=[0.62, 1.0, 1.0], hspace=0.72)
    cell_a = left[0, 0]                                  # a: early-warning 1x2 (yield | alarm burden)
    cell_op, cell_sev = left[1, 0], left[2, 0]          # b: operating strip (top); c: severity strip (below)
    # d: subgroup forests — VitalDB (top) and MOVER external (bottom), stacked 2x1
    right = outer[0, 1].subgridspec(2, 1, height_ratios=[1.0, 0.72], hspace=0.42)
    ax_forest = fig.add_subplot(right[0, 0])            # d(i): VitalDB subgroup forest
    ax_forest_mover = fig.add_subplot(right[1, 0])      # d(ii): MOVER subgroup forest

    # the four models overlaid in panels b & d (fixed colours + markers, matching the forest panel)
    MODELS = [("TiRex-2 (zero-shot)", tag, S.C["M1"], True, "o"),
              ("TFT (trained)", f"baseline-tft_{tag}", "#566573", False, "s"),
              ("PatchTST (trained)", f"baseline-patchtst_{tag}", "#3D5A80", False, "^"),
              ("Chronos-Bolt (zero-shot)", f"baseline-chronos_{tag}", "#D35400", False, "D")]

    # a — early-warning 1x2: (a1) detection yield vs lead time, (a2) alarm-burden trade-off
    sub_a = cell_a.subgridspec(1, 2, wspace=0.42)
    a1, a2 = fig.add_subplot(sub_a[0, 0]), fig.add_subplot(sub_a[0, 1])
    # a1 — YIELD curve: of all impending events, % flagged >= t min ahead (t=0 intercept == sensitivity)
    ts = list(range(0, 16))
    tir_lead = None
    for disp, mt, col, tir, mk in MODELS:
        cv = (_load_json(f"results/clinical_eval_{mt}.json") or {}).get("A_early_warning", {}).get("lead_curve")
        if not cv:
            continue
        yvals = [cv.get(str(t)) for t in ts]
        a1.plot(ts, yvals, "-", marker=mk, color=col, lw=(1.9 if tir else 1.1),
                ms=(3.0 if tir else 2.4), mew=0.3, mec="white",
                alpha=(1.0 if tir else 0.75), zorder=(5 if tir else 3), solid_capstyle="round")
        if tir:
            tir_lead = cv
    a1.set_xlim(0, 15); a1.set_xticks([0, 2, 5, 10, 15]); a1.set_ylim(0, None)
    a1.set_xlabel("required lead time (min)", fontsize=6)
    a1.set_ylabel("% of impending\nevents flagged", fontsize=6)
    a1.set_title("Detection yield vs lead time", fontsize=6.4)
    # reference: yield of zero-shot TiRex-2 at a clinically actionable 5-min lead time
    if tir_lead is not None and tir_lead.get("5") is not None:
        y5 = tir_lead["5"]
        a1.axvline(5, color="#888", ls=":", lw=0.8, zorder=1)
        a1.annotate(f"{y5:.0f}% flagged\n$\\geq$5 min ahead", xy=(5, y5), xytext=(7.4, y5 + 12),
                    fontsize=5.2, color="#333", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#888",
                                    connectionstyle="arc3,rad=0.1"))
    # a2 — alarm-burden trade-off: sensitivity vs false-alarms/hour as the alarm threshold sweeps
    tir_at = None
    for disp, mt, col, tir, mk in MODELS:
        at = (_load_json(f"results/clinical_eval_{mt}.json") or {}).get("A_early_warning", {}).get("alarm_tradeoff")
        if not at or not at.get("fa_per_hour"):
            continue
        a2.plot(at["fa_per_hour"], at["sensitivity"], "-", marker=mk, markevery=0.16,
                color=col, lw=(1.9 if tir else 1.1), ms=(3.0 if tir else 2.4), mew=0.3, mec="white",
                alpha=(1.0 if tir else 0.78), zorder=(5 if tir else 3), solid_capstyle="round")
        if tir:
            tir_at = at
    a2.set_xlim(0, None); a2.set_ylim(0, None)
    a2.set_xlabel("false alarms / hour", fontsize=6)
    a2.set_ylabel("sensitivity", fontsize=6)
    a2.set_title("Alarm-burden trade-off", fontsize=6.4)
    # reference: sensitivity of zero-shot TiRex-2 at an alarm budget of 1 false alarm/hour
    if tir_at is not None:
        fa = np.asarray(tir_at["fa_per_hour"], float); se = np.asarray(tir_at["sensitivity"], float)
        order = np.argsort(fa)
        sens1 = float(np.interp(1.0, fa[order], se[order]))
        a2.axvline(1.0, color="#888", ls=":", lw=0.8, zorder=1)
        a2.annotate(f"sensitivity {sens1:.2f}\nat 1 alarm/h", xy=(1.0, sens1),
                    xytext=(1.9, max(sens1 - 0.28, 0.12)), fontsize=5.2, color="#333", va="center",
                    arrowprops=dict(arrowstyle="-", lw=0.6, color="#888",
                                    connectionstyle="arc3,rad=0.1"))
    S.panel_letter(a1, "a", dx=-0.30, dy=1.18)           # one letter for the early-warning block

    # b — operating characteristics (TOP strip): 1x4, one subplot per metric, the four models overlaid
    opdata = {mt: (_load_json(f"results/hypo_metrics_{mt}.json") or {}).get("per_horizon", {})
              for _, mt, _, _, _ in MODELS}
    def op_val(mt, metric, h):
        try:
            return opdata[mt][S.hkey(h)]["M1"]["operating_points"]["spec90"][metric]
        except (KeyError, TypeError):
            return None
    OP_KEYS = [("sens", "Sensitivity"), ("ppv", "PPV"), ("npv", "NPV"), ("f1", "F1")]
    _mini_strip(fig, cell_op, OP_KEYS, MODELS, op_val, (0.25, 1.03), "b",
                "value at spec ≥ 0.90", "Operating characteristics")

    # c — severity-stratified detection (BOTTOM strip): 1x4, one subplot per MAP threshold
    sevdata = {mt: (_load_json(f"results/clinical_eval_{mt}.json") or {}).get("B_severity", {})
               for _, mt, _, _, _ in MODELS}
    def sev_val(mt, k, h):
        d = sevdata.get(mt, {}).get(k, {}).get(str(h))
        return d["auroc"] if d else None
    SEV_KEYS = [("MAP<65 (≥1min)", "MAP < 65 mmHg"), ("MAP<55 (≥1min)", "MAP < 55 mmHg"),
                ("MAP<50 (≥1min)", "MAP < 50 mmHg"), ("MAP<65 (≥5min, sustained)", "MAP < 65 mmHg, sustained")]
    _mini_strip(fig, cell_sev, SEV_KEYS, MODELS, sev_val, (0.85, 1.01), "c",
                "hypotension AUROC", "Severity-stratified detection")

    # shared model legend (colours/markers match every panel) — bottom centre
    from matplotlib.lines import Line2D
    mh = [Line2D([], [], color=col, marker=mk, ms=4, lw=(2.0 if tir else 1.1), label=disp)
          for disp, mt, col, tir, mk in MODELS]
    fig.legend(handles=mh, loc="lower center", ncol=4, fontsize=6.2, frameon=False, bbox_to_anchor=(0.42, 0.012))

    # d — subgroup forest (tall panel): TiRex-2 with CI + comparators overlaid per subgroup
    FOREST_COMP = [("TFT", f"baseline-tft_{tag}", "#566573", "s"),
                   ("PatchTST", f"baseline-patchtst_{tag}", "#3D5A80", "^"),
                   ("Chronos-Bolt", f"baseline-chronos_{tag}", "#D35400", "D")]
    fcomp = []
    for disp, t, col, mk in FOREST_COMP:
        p = f"results/subgroup_forest_{t}_h5.json"
        if os.path.exists(p):
            fcomp.append((disp, json.load(open(p)), col, mk))
    _forest(ax_forest, sg, fcomp, title="Subgroup robustness — VitalDB")
    S.panel_letter(ax_forest, "d", dx=0.02, dy=1.04)

    # d(ii) — MOVER external subgroup forest, same helper/style so the two panels match exactly
    MOVER_TAG = "mover_art"
    MOVER_COMP = [("TFT", "baseline-tft_mover_art_covmover_rate", "#566573", "s"),
                  ("PatchTST", "baseline-patchtst_mover_art_covmover_rate", "#3D5A80", "^"),
                  ("Chronos-Bolt", "baseline-chronos_mover_art", "#D35400", "D")]
    sg_mover_p = f"results/subgroup_forest_{MOVER_TAG}_h5.json"
    if os.path.exists(sg_mover_p):
        sg_mover = json.load(open(sg_mover_p))
        mcomp = []
        for disp, t, col, mk in MOVER_COMP:
            p = f"results/subgroup_forest_{t}_h5.json"
            if os.path.exists(p):
                mcomp.append((disp, json.load(open(p)), col, mk))
        _forest(ax_forest_mover, sg_mover, mcomp,
                title="Subgroup robustness — MOVER (external)", show_legend=False)

    S.save_fig(fig, "Fig5_clinical_robustness")


def _forest(ax, sg, comparators=None, title="Subgroup robustness", show_legend=True):
    """Forest with all text on the RIGHT (outer figure margin) so nothing spills into
    the neighbouring panels on the left. TiRex-2 is the CI point; each comparator (trained
    baseline + best zero-shot foil) is overlaid as a bare marker at the same row, so the
    subgroup-by-subgroup robustness of every model is visible at once."""
    from matplotlib.lines import Line2D
    comparators = comparators or []
    cmaps = [(disp, {(s["var"], s["level"]): s for s in c["subgroups"]}, col, mk)
             for disp, c, col, mk in comparators]
    subs = sg["subgroups"]; overall = sg["overall"]
    rows = []
    last_var = None
    for s in subs:
        if s["var"] != last_var:
            rows.append(("header", None, s)); last_var = s["var"]
        rows.append(("row", None, s))
    rows = rows[::-1]
    y = 0; yticks = []; ylabels = []
    for kind, _, s in rows:
        if kind == "header":
            ax.text(1.03, y, s["var"], fontsize=6.2, fontweight="bold", va="center", ha="left",
                    transform=ax.get_yaxis_transform())
            yticks.append(y); ylabels.append("")
        else:
            au, ci = s["auroc"], s["ci"]
            ax.errorbar(au, y, xerr=[[au-ci[0]], [ci[1]-au]], fmt="o", color=S.C["M1"],
                        ms=3.4, capsize=1.8, lw=1.0, zorder=3)         # TiRex-2 CI underneath
            for disp, cmap, col, mk in cmaps:                         # comparators ON TOP so they're visible
                cs = cmap.get((s["var"], s["level"]))
                if cs:
                    ax.plot(cs["auroc"], y, mk, color=col, ms=2.9, alpha=0.9, mec="white",
                            mew=0.3, zorder=6)
            yticks.append(y); ylabels.append(f"{s['level']} (n={s.get('n_cases')})  {au:.3f}")
        y += 1
    ax.axvline(overall["auroc"], color=S.C["persist"], lw=0.9, ls="--")
    ax.text(0.02, 0.97, f"– –  TiRex-2 overall {overall['auroc']:.3f}", transform=ax.transAxes,
            fontsize=5.4, color="#555", ha="left", va="top",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85), zorder=7)
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=5.6)
    ax.yaxis.tick_right()                             # tick labels on the right
    ax.set_ylim(-0.6, y-0.4); ax.set_xlim(0.83, 0.98); ax.set_xticks([0.85, 0.90, 0.95])
    ax.set_xlabel("hypotension AUROC @5 min"); ax.set_title(title, loc="center")
    ax.spines["left"].set_visible(False); ax.spines["right"].set_visible(True)
    ax.tick_params(axis="y", length=0)
    if cmaps and show_legend:
        handles = [Line2D([], [], marker="o", color=S.C["M1"], ls="none", ms=3.4, label="TiRex-2")]
        handles += [Line2D([], [], marker=mk, color=col, ls="none", ms=2.9, label=disp)
                    for disp, _, col, mk in cmaps]
        ax.legend(handles=handles, loc="lower left", fontsize=4.4, handletextpad=0.3,
                  borderpad=0.3, labelspacing=0.25, framealpha=0.9)


# ══════════════════════════════════════════════════════════════════════════════
# TABLES
# ══════════════════════════════════════════════════════════════════════════════
def _write_table(name, header, rows, caption):
    os.makedirs(S.TAB_DIR, exist_ok=True)
    # markdown
    with open(f"{S.TAB_DIR}/{name}.md", "w") as f:
        f.write(f"**{caption}**\n\n")
        f.write("| " + " | ".join(header) + " |\n")
        f.write("|" + "|".join(["---"]*len(header)) + "|\n")
        for r in rows:
            f.write("| " + " | ".join(str(x) for x in r) + " |\n")
    # csv
    with open(f"{S.TAB_DIR}/{name}.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(header); w.writerows(rows)
    # latex (booktabs)
    with open(f"{S.TAB_DIR}/{name}.tex", "w") as f:
        f.write("\\begin{table}[t]\\centering\\footnotesize\n")
        f.write("\\caption{" + caption + "}\n")
        f.write("\\begin{tabular}{" + "l"*len(header) + "}\n\\toprule\n")
        f.write(" & ".join(header) + " \\\\\n\\midrule\n")
        for r in rows:
            f.write(" & ".join(str(x) for x in r).replace("%", "\\%").replace("±", "$\\pm$") + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n\\end{table}\n")
    print(f"  wrote {S.TAB_DIR}/{name}.{{md,csv,tex}}", flush=True)


def windows_caseids(tag):
    """Case ids that contributed >=1 forecast window — the true cohort denominator."""
    cids = set()
    for r in csv.DictReader(open(f"results/ablation_windows_{tag}.csv")):
        cids.add(str(r["caseid"]))
    return cids

def table1_cohort(tag):
    # Denominator = cases contributing >=1 window; demographics from authoritative clinical_data.csv
    # (the manifest is incomplete). Normalise caseids to strip zero-padding across sources.
    keep = {str(int(c)) for c in windows_caseids(tag)}
    cd = {str(int(r["caseid"])): r for r in csv.DictReader(open("datasets/vitaldb/data/clinical_data.csv", encoding="utf-8-sig"))}
    rows = [cd[c] for c in keep if c in cd]
    hyp = load_hypo(tag); n = len(rows)
    def fnum(r, k):
        try: return float(r[k])
        except (ValueError, TypeError, KeyError): return np.nan
    ages = np.array([fnum(r, "age") for r in rows]); ages = ages[~np.isnan(ages)]
    bmi = np.array([fnum(r, "bmi") for r in rows]); bmi = bmi[~np.isnan(bmi)]
    dur = np.array([(fnum(r, "aneend") - fnum(r, "anestart"))/60 for r in rows]); dur = dur[np.isfinite(dur) & (dur > 0)]
    males = sum(1 for r in rows if str(r.get("sex", "")).upper().startswith("M"))
    asa = {}
    for r in rows: asa[str(r.get("asa", "")).strip()] = asa.get(str(r.get("asa", "")).strip(), 0) + 1
    n_asa12 = asa.get("1", 0) + asa.get("2", 0)
    dept = {}
    for r in rows:
        d = str(r.get("department", "")).strip() or "—"; dept[d] = dept.get(d, 0) + 1
    top_dept = sorted(dept.items(), key=lambda kv: -kv[1])[:3]
    def iqr(a): return f"{np.median(a):.0f} ({np.percentile(a,25):.0f}–{np.percentile(a,75):.0f})"
    H_rows = [
        ("Cases, n", f"{n:,}"),
        ("Age, y — median (IQR)", iqr(ages)),
        ("Male sex, n (%)", f"{males:,} ({males/n*100:.0f}%)"),
        ("BMI, kg/m² — median (IQR)", f"{np.median(bmi):.1f} ({np.percentile(bmi,25):.1f}–{np.percentile(bmi,75):.1f})"),
        ("Anesthesia duration, min — median (IQR)", iqr(dur)),
        ("ASA I–II, n (%)", f"{n_asa12:,} ({n_asa12/n*100:.0f}%)"),
        ("Top departments", "; ".join(f"{k} {v}" for k, v in top_dept)),
        ("Forecast windows, n", f"{load_primary(tag)['n_windows']:,}"),
    ]
    for h in [1, 5, 10, 15]:
        p = hyp["per_horizon"][S.hkey(h)]
        H_rows.append((f"Hypotension prevalence @{h} min, %", f"{p['prevalence']*100:.1f}"))
    _write_table("Table1_cohort", ["Characteristic", "Value"], H_rows,
                 f"Table 1. Cohort characteristics — cases contributing ≥1 forecast window (n={n}, tag={tag}).")


def table2_accuracy(tag):
    p = load_primary(tag); hs = S.horizons_sorted(p["per_horizon"])
    header = ["Horizon (min)", "MAE M1", "MAE M0", "CRPS M1", "CRPS M0", "CRPS persist.",
              "X% covariate [95% CI]", "Y% vs persist. [95% CI]"]
    rows = []
    for h in hs:
        a = strat(p, h, "all")
        rows.append([h, f"{a['mae_M1']:.2f}", f"{a['mae_M0']:.2f}", f"{a['crps_M1']:.3f}",
                     f"{a['crps_M0']:.3f}", f"{a['crps_persistence']:.3f}",
                     f"{a['X_pct_withpast']:+.2f} [{a['X_pct_withpast_CI95'][0]:+.2f}, {a['X_pct_withpast_CI95'][1]:+.2f}]",
                     f"{a['Y_pct_vs_persistence']:.1f} [{a['Y_pct_CI95'][0]:.1f}, {a['Y_pct_CI95'][1]:.1f}]"])
    _write_table("Table2_accuracy", header, rows,
                 f"Table 2. Forecast accuracy and covariate value, all windows (tag={tag}).")


def table3_classification(tag):
    hyp = load_hypo(tag); hs = S.horizons_sorted(hyp["per_horizon"])
    header = ["Horizon (min)", "AUROC M1 [95% CI]", "AUROC M0", "AUPRC M1", "pAUROC(sp≥.9)", "ECE",
              "Sens/PPV/NPV/F1 @sp≥.9", "Kapral ext.", "Zhu ext."]
    rows = []
    for h in hs:
        m1 = hyp["per_horizon"][S.hkey(h)]["M1"]; m0 = hyp["per_horizon"][S.hkey(h)]["M0"]
        op = m1["operating_points"]["spec90"]
        kap = f"{S.KAPRAL_AUROC[h][1]:.3f}" if h in S.KAPRAL_AUROC else "—"
        zhu = f"{S.ZHU_AUROC[h]:.3f}" if h in S.ZHU_AUROC else "—"
        rows.append([h, f"{m1['auroc']:.3f} [{m1['auroc_CI95'][0]:.3f}, {m1['auroc_CI95'][1]:.3f}]",
                     f"{m0['auroc']:.3f}", f"{m1['auprc']:.3f}", f"{m1['pauroc_spec90']:.3f}",
                     f"{m1['ece']:.3f}", f"{op['sens']:.2f}/{op['ppv']:.2f}/{op['npv']:.2f}/{op['f1']:.2f}",
                     kap, zhu])
    _write_table("Table3_classification", header, rows,
                 f"Table 3. Impending-hypotension classification vs supervised foils (tag={tag}). "
                 "Ours = zero-shot; foils = task-trained (external VitalDB).")


def table4_matched(tag):
    """Matched head-to-head: zero-shot TiRex vs every trained baseline on identical data."""
    bl = available_baselines(tag)
    if not bl:
        bl = [{**MATCHED_BASELINES[0], "tag": f"baseline-tft_{tag}"}]
    for b in bl:
        b["M"] = load_matched(b["tag"])
    M = bl[0]["M"]
    hs = sorted(int(k) for k in M["per_horizon"])
    header = (["Horizon (min)", "TiRex-2 zero-shot [95% CI]"]
              + [f"{b['disp']} M1 [95% CI]" for b in bl] + [f"{b['disp']} M0" for b in bl]
              + ["Kapral ext.", "Zhu ext."])
    def f(x): return "—" if not x else f"{x['auroc']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]"
    def fb(x):   # trained baseline: append cross-fold ±SD when the 5-fold OOF is present
        if not x:
            return "—"
        s = f"{x['auroc']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]"
        return s + (f" ±{x['fold_sd']:.3f}" if "fold_sd" in x else "")
    def f0(x): return "—" if not x else f"{x['auroc']:.3f}"
    cv = any("fold_sd" in b["M"]["per_horizon"][str(hs[0])].get("tft_M1", {}) for b in bl)
    rows = []
    for h in hs:
        d = M["per_horizon"][str(h)]
        kap = f"{d['kapral_ext']:.3f}" if d.get("kapral_ext") else "—"
        zhu = f"{d['zhu_ext']:.3f}" if d.get("zhu_ext") else "—"
        row = [h, f(d["tirex_M1"])]
        row += [fb(b["M"]["per_horizon"][str(h)]["tft_M1"]) for b in bl]
        row += [f0(b["M"]["per_horizon"][str(h)]["tft_M0"]) for b in bl]
        rows.append(row + [kap, zhu])
    names = " & ".join(b["disp"] for b in bl)
    split_txt = ("all cases, 5-fold subject-level out-of-fold cross-validation"
                 if cv else "canonical 60/20/20 split")
    _write_table("Table4_matched", header, rows,
                 f"Table 4. Matched hypotension AUROC (n={M['n_test_subjects']} subjects; {split_txt}). "
                 f"Zero-shot TiRex-2 (inherently held out) vs {names}, trained on the same windows and "
                 f"scored by out-of-fold prediction — each case is predicted only by the fold in which it "
                 f"was held out; M1 = with drug covariate, M0 = without. Bracketed intervals are "
                 f"case-clustered bootstrap 95% CIs" + (", ±SD is across the 5 folds" if cv else "") +
                 f"; foils are external literature references.")


def table5_matched_forecast(tag):
    """Matched forecasting accuracy (CRPS/MAE) on identical held-out test windows."""
    c2s = H.caseid_to_subject()
    trows, _ = H.load_rows(tag)
    tsub = eval_subjects(trows, c2s)
    bl = available_baselines(tag)
    if not bl:
        bl = [{**MATCHED_BASELINES[0], "tag": f"baseline-tft_{tag}"}]
    for b in bl:
        b["rows"], _ = H.load_rows(b["tag"])
    hs = sorted({int(r["h_min"]) for r in bl[0]["rows"]})
    crps_tx = _mean_metric_by_h(trows, c2s, tsub, "crps_M1", hs)
    mae_tx = _mean_metric_by_h(trows, c2s, tsub, "mae_M1", hs)
    for b in bl:
        b["crps"] = _mean_metric_by_h(b["rows"], c2s, tsub, "crps_M1", hs)
        b["mae"] = _mean_metric_by_h(b["rows"], c2s, tsub, "mae_M1", hs)
    header = (["Horizon (min)", "CRPS TiRex-2"] + [f"CRPS {b['disp']}" for b in bl]
              + ["MAE TiRex-2 (mmHg)"] + [f"MAE {b['disp']} (mmHg)" for b in bl])
    rows = []
    for i, h in enumerate(hs):
        row = [h, f"{crps_tx[i]:.3f}"] + [f"{b['crps'][i]:.3f}" for b in bl]
        row += [f"{mae_tx[i]:.2f}"] + [f"{b['mae'][i]:.2f}" for b in bl]
        rows.append(row)
    names = ", ".join(b["disp"] for b in bl)
    _write_table("Table5_matched_forecast", header, rows,
                 "Table 5. Matched probabilistic-forecasting accuracy on identical windows (all cases; M1, "
                 "with drug covariate). Zero-shot TiRex-2 (inherently held out) vs "
                 f"{names}, trained and scored by 5-fold subject-level out-of-fold cross-validation.")


def table6_zeroshot(tag, save="Table6_zeroshot", label="Table 6", cohort="held-out"):
    """Matched head-to-head among zero-shot TSFMs (TiRex vs Chronos/TimesFM/Moirai)."""
    zs = available_zeroshot(tag)
    if not zs:
        print(f"  (no zero-shot TSFM results — skip {label})", flush=True); return
    for z in zs:
        z["M"] = load_matched(z["tag"])
    M = zs[0]["M"]; hs = sorted(int(k) for k in M["per_horizon"])
    header = ["Horizon (min)", "TiRex-2 [95% CI]"] + [f"{z['disp']} [95% CI]" for z in zs]
    def f(x): return "—" if not x else f"{x['auroc']:.3f} [{x['ci'][0]:.3f}, {x['ci'][1]:.3f}]"
    rows = []
    for h in hs:
        row = [h, f(M["per_horizon"][str(h)]["tirex_M1"])]
        row += [f(z["M"]["per_horizon"][str(h)]["tft_M1"]) for z in zs]
        rows.append(row)
    names = ", ".join(z["disp"] for z in zs)
    _write_table(save, header, rows,
                 f"{label}. Matched hypotension AUROC among zero-shot time-series foundation models "
                 f"(TiRex-2 vs {names}) on identical {cohort} subjects (n={M['n_test_subjects']} "
                 f"subjects). All evaluated zero-shot (no training); only TiRex-2 ingests the known "
                 f"future drug-infusion covariate.")


def figure_s_training(tag):
    """Supplementary: trained-baseline train/val pinball-loss curves (convergence, no overfitting).
    One row per cohort x baseline (VitalDB / MOVER, TFT / PatchTST), columns = M1 / M0 arms."""
    # (cohort label, history tag). MOVER's trained tag carries the cov suffix.
    COHORTS = [("VitalDB", tag), ("MOVER", "mover_art_covmover_rate")]
    present = []
    for cohort, ctag in COHORTS:
        for b in MATCHED_BASELINES:
            path = f"results/baseline_history_baseline-{b['key']}_{ctag}.json"
            if os.path.exists(path):
                present.append((f"{cohort} · {b['disp']}", json.load(open(path))))
    if not present:
        print("  (no baseline_history_*.json — skip training-curve supplement)", flush=True); return
    nrow = len(present)
    fig, axs = plt.subplots(nrow, 2, figsize=(S.W2, S.W2 * 0.42 * nrow), squeeze=False,
                            gridspec_kw=dict(wspace=0.26, hspace=0.62))
    letters = iter("abcdefghijklmnop")
    for ri, (disp, hist) in enumerate(present):
        folds = hist.get("fold_hist") or {}                  # 5-fold CV: one curve per fold
        for ci, (arm, lab) in enumerate([("M1", "with drug covariate"), ("M0", "no drug covariate")]):
            ax = axs[ri][ci]
            if folds:                                        # all folds, so the CV is honestly shown
                curves = [folds[fk][arm]["curve"] for fk in sorted(folds)
                          if folds[fk].get(arm, {}).get("curve")]
            else:                                            # non-CV fallback: single arm curve
                a = hist.get("arms", {}).get(arm)
                curves = [a["curve"]] if a and a.get("curve") else []
            for k, c in enumerate(curves):
                ep = [r["epoch"] for r in c]
                ax.plot(ep, [r["train_pinball"] for r in c], "-", color=S.C["M1"], lw=0.8, alpha=0.5,
                        label=("train" if k == 0 else None))
                ax.plot(ep, [r["val_pinball"] for r in c], "-", color=S.C["M0"], lw=1.0, alpha=0.8,
                        label=("validation" if k == 0 else None))
            if curves:
                ax.legend(loc="upper right", fontsize=6)
            nf = len(curves)
            ttl = f"{disp} {arm} — {lab}" + (f"  ({nf} folds)" if nf > 1 else "")
            S.finish(ax, "epoch", "pinball loss (normalised)", ttl)
            S.panel_letter(ax, next(letters))
    S.save_fig(fig, "FigS_training_curves")


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(S.FIG_DIR, exist_ok=True); os.makedirs(S.TAB_DIR, exist_ok=True)
    print(f"[paper] tag={TAG}  font={S.SANS}", flush=True)
    print("[paper] Figure 1 ..."); figure1(TAG)
    print("[paper] Figure 2 ..."); figure2(TAG)
    print("[paper] Figure 3 (zero-shot TSFM benchmark) ..."); figure_zeroshot(TAG)
    print("[paper] Figure 4 (hypotension vs supervised SOTA) ..."); figure3(TAG)
    print("[paper] Figure 5 (clinical translation) ..."); figure4(TAG)
    print("[paper] Supp: training curves ..."); figure_s_training(TAG)
    print("[paper] Supp: MOVER zero-shot benchmark (external cohort) ...")
    figure_zeroshot("mover_art", save="FigS_zeroshot_mover", cohort_note="External cohort (MOVER)")
    print("[paper] Tables ..."); table1_cohort(TAG); table2_accuracy(TAG); table3_classification(TAG)
    table4_matched(TAG); table5_matched_forecast(TAG); table6_zeroshot(TAG)
    table6_zeroshot("mover_art", save="TableS_zeroshot_mover", label="Table S1", cohort="external MOVER")
    print("[paper] Done. Figures in outputs/figs/paper/ ; tables in results/tables/", flush=True)


if __name__ == "__main__":
    main()
