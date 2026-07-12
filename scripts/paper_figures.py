"""Publication figures + tables for the TiRex-2 / VitalDB paper (Nature Medicine style).

Classification-first framing:
  Fig 1  Study design & cohort
  Fig 2  Forecast accuracy & the value of the known future drug covariate
  Fig 3  Impending-hypotension prediction: zero-shot TiRex-2 vs supervised SOTA  (headline)
  Fig 4  Clinical translation & robustness
  Tables 1-3  cohort characteristics | accuracy | classification-vs-foils

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

    fig = plt.figure(figsize=(S.W2, S.W2 * 0.80))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0], hspace=1.05, wspace=0.42)
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
    _cohort_flow(ax_flow, flow, prim, hyp, n_cohort)
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


def _cohort_flow(ax, flow, prim, hyp, n_cohort):
    ax.axis("off")
    n0 = flow["n_local_scanned"]; nCand = flow["included_N"]
    nw = prim["n_windows"]
    steps = [
        (f"VitalDB cases scanned\nn = {n0:,}", "#E8EEF2"),
        (f"Anesthetic cohort\n(remifentanil + propofol)\nn = {nCand:,}", S.C["M1_light"]),
        (f"Cases with ≥ 1 window\nn = {n_cohort:,}", "#CFE3E7"),
        (f"Forecast windows\nn = {nw:,}", "#EAD9BD"),
    ]
    n = len(steps); gap = 0.06
    box_h = (1.0 - (n-1)*gap) / n          # fill the full cell height
    for i, (txt, col) in enumerate(steps):
        top = 1.0 - i*(box_h+gap); bot = top - box_h
        ax.add_patch(plt.Rectangle((0.04, bot), 0.92, box_h, transform=ax.transAxes,
                     facecolor=col, edgecolor="#777", lw=0.7, zorder=2))
        ax.text(0.5, (top+bot)/2, txt, transform=ax.transAxes, ha="center", va="center",
                fontsize=6.6, zorder=3)
        if i < n-1:
            ax.annotate("", xy=(0.5, bot-gap+0.006), xytext=(0.5, bot-0.004),
                        xycoords="axes fraction",
                        arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.0))
    ax.set_ylim(0, 1)


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
    ax.plot(tc, ctx, color=S.C["ink"], lw=1.0)
    ax.plot(th, truth, color=S.C["ink"], lw=1.3, label="observed")
    ax.plot(th, q[S.Q_MED], color=S.C["M1"], lw=1.3, label="M1 median")
    ax.fill_between(th, q[S.Q_LO], q[S.Q_HI], color=S.C["M1_light"], alpha=0.55, lw=0, label="10–90%")
    ax.axvline(0, color="#888", lw=0.7, ls=":")
    ax.axhline(65, color=S.C["event"], lw=0.7, ls="--")
    ax.set_title(title, loc="center", fontsize=7)
    ax.set_xlabel("time (min)")
    ax.set_ylabel("MAP (mmHg)")
    ax.set_xlim(tc[0], th[-1]); ax.set_ylim(40, 120)     # shared y-range across the 3 examples


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — accuracy & value of the drug covariate
# ══════════════════════════════════════════════════════════════════════════════
def figure2(tag):
    prim = load_primary(tag)
    fig, axs = plt.subplot_mosaic([["a", "b"], ["c", "d"]], figsize=(S.W2, S.W2*0.72),
                                  gridspec_kw=dict(hspace=0.55, wspace=0.35))

    # a — instantaneous MAE (mmHg) vs horizon, M1 vs M0
    hs = MAIN_H
    mae1 = [strat(prim, h, "all")["mae_M1"] for h in hs]
    mae0 = [strat(prim, h, "all")["mae_M0"] for h in hs]
    a = axs["a"]
    a.plot(hs, mae1, "-o", color=S.C["M1"], label="M1 (+ drug covariate)")
    a.plot(hs, mae0, "-s", color=S.C["M0"], label="M0 (target only)")
    y7 = strat(prim, 7, "all")["Y_pct_vs_persistence"]
    a.text(0.34, 0.05, f"vs persistence: −{y7:.0f}% CRPS", transform=a.transAxes, fontsize=6, color="#555")
    S.finish(a, "forecast horizon (min)", "MAE (mmHg)", "Forecast accuracy")
    a.set_xticks(hs); a.legend(loc="upper left"); S.panel_letter(a, "a")

    # b — covariate benefit X% vs horizon, by stratum
    b = axs["b"]
    for s_name, col, mk in [("all", S.C["M1"], "o"), ("transition", S.C["transition"], "^"),
                            ("steady", S.C["steady"], "v")]:
        xs = [strat(prim, h, s_name)["X_pct_withpast"] for h in hs]
        lo = [strat(prim, h, s_name)["X_pct_withpast_CI95"][0] for h in hs]
        hi = [strat(prim, h, s_name)["X_pct_withpast_CI95"][1] for h in hs]
        b.plot(hs, xs, "-", color=col, marker=mk, label=s_name)
        b.fill_between(hs, lo, hi, color=col, alpha=0.15, lw=0)
    b.axhline(0, color="#999", lw=0.7, ls="--")
    S.finish(b, "forecast horizon (min)", "CRPS reduction M0→M1 (%)", "Value of the drug covariate")
    b.set_xticks(hs); b.legend(loc="upper left"); S.panel_letter(b, "b")

    # c — covariate representation: CE vs RATE vs pressor (transition, 7 min) forest
    c = axs["c"]
    arms = [("CE (effect-site)", tag, S.C["M1"]),
            ("RATE (infusion)", RATE_TAG, S.C["rate"]),
            ("Phenylephrine", PRESSOR_TAG, S.C["pressor"])]
    ypos = list(range(len(arms)))[::-1]
    STATX = 8.15                                       # dedicated stats column, clear of the whiskers
    for yp, (lab, t, col) in zip(ypos, arms):
        p = load_primary(t); blk = strat(p, 7, "transition")
        x = blk["X_pct_withpast"]; ci = blk["X_pct_withpast_CI95"]
        c.errorbar(x, yp, xerr=[[x-ci[0]], [ci[1]-x]], fmt="o", color=col, capsize=2.5, lw=1.2)
        c.text(STATX, yp, f"{x:+.2f}%  [{ci[0]:+.2f}, {ci[1]:+.2f}]", ha="right", va="center", fontsize=5.8)
    c.text(STATX, len(arms)-0.35, "mean [95% CI]", ha="right", va="center", fontsize=5.6, color="#555", style="italic")
    c.axvline(0, color="#999", lw=0.7, ls="--")
    c.set_yticks(ypos); c.set_yticklabels([a[0] for a in arms])
    c.set_xlabel("CRPS reduction in transition windows @7 min (%)")
    c.set_title("Which covariate helps?", loc="center")
    c.set_xlim(-4.2, 8.3); c.set_xticks([-4, -2, 0, 2, 4])
    c.set_ylim(-0.5, len(arms)-0.15); S.panel_letter(c, "c")

    # d — instantaneous MAE vs Kapral (external / internal)
    d = axs["d"]
    _kapral_panel(d, tag)
    S.panel_letter(d, "d")
    S.save_fig(fig, "Fig2_accuracy_covariate")


def _kapral_panel(ax, tag):
    # our instantaneous endpoint MAE from windows
    rows, _ = H.load_rows(tag)
    hs = MAIN_H
    our = []
    for h in hs:
        v = [float(r["mae_inst_M1"]) for r in rows if int(r["h_min"]) == h and r.get("mae_inst_M1") not in ("", "nan", None)]
        our.append(np.mean(v) if v else np.nan)
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
    ax.plot(hs, our, "-o", color=S.C["M1"], lw=2.4, ms=6, mec="white", mew=1.0,
            zorder=6, label="TiRex-2 (ours, M1)")
    ax.set_xlim(0, 7.4); ax.set_ylim(0, None)
    S.finish(ax, "forecast distance (min)", "instantaneous MAE (mmHg)", "Accuracy vs Kapral et al. (TFT)")
    ax.legend(loc="upper left")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — impending-hypotension prediction vs supervised SOTA  (headline)
# ══════════════════════════════════════════════════════════════════════════════
def figure3(tag):
    hyp = load_hypo(tag); clin = load_clinical(tag)
    rows, _ = H.load_rows(tag); c2s = H.caseid_to_subject()
    dev = H.split_subjects([r["caseid"] for r in rows], c2s, seed=0)

    fig = plt.figure(figsize=(9.6, 5.4))                  # 16:9 landscape — room to breathe
    gs = fig.add_gridspec(2, 3, hspace=0.62, wspace=0.42,
                          left=0.08, right=0.97, top=0.93, bottom=0.11)
    ax_roc = fig.add_subplot(gs[0, 0])
    ax_auc = fig.add_subplot(gs[0, 1])
    ax_cal = fig.add_subplot(gs[0, 2])
    ax_pr  = fig.add_subplot(gs[1, 0])
    ax_dca = fig.add_subplot(gs[1, 1])
    ax_bar = fig.add_subplot(gs[1, 2])

    # a — ROC at 5 and 7 min with spec>=0.90 operating points
    for h, col, ls in [(5, S.C["M1"], "-"), (7, S.C["transition"], "-")]:
        y, s = _test_scores(rows, c2s, dev, h)
        fpr, tpr, _ = H.roc_points(y, s)
        au = hyp["per_horizon"][S.hkey(h)]["M1"]["auroc"]
        ax_roc.plot(fpr, tpr, ls, color=col, lw=1.4, label=f"{h} min (AUROC {au:.3f})")
        op = hyp["per_horizon"][S.hkey(h)]["M1"]["operating_points"]["spec90"]
        ax_roc.plot(1-op["spec"], op["sens"], "o", color=col, ms=5, mec="white", mew=0.6, zorder=5)
    ax_roc.plot([0, 1], [0, 1], color="#BBB", lw=0.7, ls=":")
    ax_roc.set_xlim(0, 1); ax_roc.set_ylim(0, 1.005)
    S.finish(ax_roc, "1 − specificity", "sensitivity", "ROC (M1)", ygrid=False)
    ax_roc.legend(loc="lower right", bbox_to_anchor=(1.0, 0.02)); S.panel_letter(ax_roc, "a")

    # b — AUROC vs horizon, M1 vs M0, with foils overlaid (THE panel)
    hs = S.horizons_sorted(hyp["per_horizon"])
    def series(model):
        a = [hyp["per_horizon"][S.hkey(h)][model]["auroc"] for h in hs]
        lo = [hyp["per_horizon"][S.hkey(h)][model]["auroc_CI95"][0] for h in hs]
        hi = [hyp["per_horizon"][S.hkey(h)][model]["auroc_CI95"][1] for h in hs]
        return a, lo, hi
    a1, l1, h1 = series("M1"); a0, l0, h0 = series("M0")
    ax_auc.fill_between(hs, l1, h1, color=S.C["M1"], alpha=0.15, lw=0)
    ax_auc.plot(hs, a1, "-o", color=S.C["M1"], label="TiRex-2 M1 (ours)")
    ax_auc.plot(hs, a0, "--s", color=S.C["M0"], ms=3, label="TiRex-2 M0 (ours)")
    for h, (ki, ke) in S.KAPRAL_AUROC.items():
        ax_auc.plot(h, ke, "D", color=S.C["kapral"], ms=5, mec="white", mew=0.5, zorder=6)
    ax_auc.plot([], [], "D", color=S.C["kapral"], label="Kapral (TFT, ext.)")
    for h, z in S.ZHU_AUROC.items():
        ax_auc.plot(h, z, "s", color=S.C["zhu"], ms=5, mec="white", mew=0.5, zorder=6)
    ax_auc.plot([], [], "s", color=S.C["zhu"], label="Zhu (Transformer, ext.)")
    S.finish(ax_auc, "forecast horizon (min)", "hypotension AUROC", "Zero-shot vs supervised SOTA")
    ax_auc.set_xticks(hs); ax_auc.set_ylim(0.80, 1.0)
    ax_auc.legend(loc="upper right", fontsize=5.6); S.panel_letter(ax_auc, "b")

    # c — calibration at 5 min (M1)
    y5, s5 = _test_scores(rows, c2s, dev, 5)
    mean_pred, obs_freq, _, _ = H.calibration(y5, s5, n_bins=10)
    ece = hyp["per_horizon"]["5min"]["M1"]["ece"]
    ax_cal.plot([0, 1], [0, 1], color="#BBB", lw=0.7, ls=":")
    ax_cal.plot(mean_pred, obs_freq, "-o", color=S.C["M1"], ms=3)
    ax_cal.text(0.05, 0.9, f"ECE = {ece:.3f}", transform=ax_cal.transAxes, fontsize=6)
    ax_cal.set_xlim(0, 1); ax_cal.set_ylim(0, 1)
    S.finish(ax_cal, "predicted risk", "observed frequency", "Calibration @5 min", ygrid=False)
    S.panel_letter(ax_cal, "c")

    # d — AUPRC vs horizon with prevalence baseline
    ap = [hyp["per_horizon"][S.hkey(h)]["M1"]["auprc"] for h in hs]
    apl = [hyp["per_horizon"][S.hkey(h)]["M1"]["auprc_CI95"][0] for h in hs]
    aph = [hyp["per_horizon"][S.hkey(h)]["M1"]["auprc_CI95"][1] for h in hs]
    prev = [hyp["per_horizon"][S.hkey(h)]["prevalence"] for h in hs]
    ax_pr.fill_between(hs, apl, aph, color=S.C["M1"], alpha=0.15, lw=0)
    ax_pr.plot(hs, ap, "-o", color=S.C["M1"], label="AUPRC (M1)")
    ax_pr.plot(hs, prev, "--", color=S.C["persist"], label="prevalence (chance)")
    S.finish(ax_pr, "forecast horizon (min)", "AUPRC", "Precision–recall")
    ax_pr.set_xticks(hs); ax_pr.set_ylim(0, 1); ax_pr.legend(loc="upper right"); S.panel_letter(ax_pr, "d")

    # e — decision curve @5 min
    dc = clin["C_decision_curve"]["5"]
    pt = np.array(dc["pt"]); nb = np.array(dc["nb_model"]); nball = np.array(dc["nb_treat_all"])
    ax_dca.plot(pt, nb, color=S.C["M1"], lw=1.4, label="TiRex-2 M1")
    ax_dca.plot(pt, nball, color=S.C["persist"], lw=1.0, label="treat all")
    ax_dca.axhline(0, color="#999", lw=0.8, label="treat none")
    ax_dca.set_ylim(-0.02, max(0.02, np.nanmax(nb)*1.15)); ax_dca.set_xlim(pt.min(), pt.max())
    S.finish(ax_dca, "threshold probability", "net benefit", "Decision curve @5 min")
    ax_dca.legend(loc="upper right"); S.panel_letter(ax_dca, "e")

    # f — head-to-head AUROC bars: ours (zero-shot) vs supervised foils, at 5 & 7 min
    groups = [5, 7]
    series_f = [("TiRex-2 M1 (ours)", S.C["M1"], lambda h: hyp["per_horizon"][S.hkey(h)]["M1"]["auroc"]),
                ("Kapral (ext.)", S.C["kapral"], lambda h: S.KAPRAL_AUROC.get(h, (None, None))[1]),
                ("Zhu (ext.)", S.C["zhu"], lambda h: S.ZHU_AUROC.get(h))]
    from matplotlib.patches import Patch
    nb = len(series_f); w = 0.26
    x = np.arange(len(groups))
    for j, (lab, col, fn) in enumerate(series_f):
        vals = [fn(h) for h in groups]
        xs = x + (j - (nb-1)/2)*w
        for xi, v in zip(xs, vals):
            if v is None:
                continue
            ax_bar.bar(xi, v, width=w, color=col, edgecolor="white", lw=0.4)
            ax_bar.text(xi, v+0.004, f"{v:.3f}", ha="center", va="bottom", fontsize=5.2, rotation=90)
    ax_bar.set_xticks(x); ax_bar.set_xticklabels([f"{g} min" for g in groups])
    ax_bar.set_ylim(0.80, 1.0)                         # headroom so the legend clears the bars
    S.finish(ax_bar, None, "hypotension AUROC", "Head-to-head vs SOTA")
    ax_bar.legend(handles=[Patch(facecolor=col, label=lab) for lab, col, _ in series_f],
                  loc="upper right", fontsize=5.4, framealpha=0.9); S.panel_letter(ax_bar, "f")

    S.save_fig(fig, "Fig3_hypotension_vs_sota")


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — clinical translation & robustness
# ══════════════════════════════════════════════════════════════════════════════
def figure4(tag):
    clin = load_clinical(tag); hyp = load_hypo(tag); sg = load_subgroup(tag, 5)
    fig = plt.figure(figsize=(8.6, 4.9))                  # landscape
    gs = fig.add_gridspec(2, 3, width_ratios=[1.0, 1.0, 0.95], height_ratios=[0.48, 1.0],
                          hspace=0.55, wspace=0.42, left=0.07, right=0.83, top=0.93, bottom=0.10)
    axs = {"a": fig.add_subplot(gs[0, 0:2]),
           "b": fig.add_subplot(gs[1, 0]),
           "d": fig.add_subplot(gs[1, 1]),
           "c": fig.add_subplot(gs[:, 2])}

    # a — lead time (early warning)
    A = clin["A_early_warning"]
    a = axs["a"]
    med = A["lead_time_min_median"]; iqr = A["lead_time_min_IQR"]
    # pct_detected_* are already in percent (0-100); bars close together in a short panel
    a.barh([0], [A["pct_detected_ge2min_ahead"]], color=S.C["M1_light"], height=0.62, label="≥2 min ahead")
    a.barh([1], [A["pct_detected_ge5min_ahead"]], color=S.C["M1"], height=0.62, label="≥5 min ahead")
    a.set_yticks([0, 1]); a.set_yticklabels(["detected\n≥2 min", "detected\n≥5 min"])
    a.set_ylim(-0.6, 1.6)
    a.set_xlim(0, 100); a.set_xlabel("% of events flagged in advance")
    a.set_title(f"Early warning — median lead {med:.1f} min (IQR {iqr[0]:.1f}–{iqr[1]:.1f})", loc="center", fontsize=6.8)
    for yv, key in [(0, "pct_detected_ge2min_ahead"), (1, "pct_detected_ge5min_ahead")]:
        a.text(min(A[key]+1.5, 92), yv, f"{A[key]:.0f}%", va="center", fontsize=6)
    S.panel_letter(a, "a")

    # b — severity gradient (AUROC by threshold/duration vs horizon)
    b = axs["b"]
    sev = clin["B_severity"]
    styles = {"MAP<65 (≥1min)": (S.C["M1"], "-", "o"), "MAP<55 (≥1min)": (S.C["transition"], "-", "^"),
              "MAP<50 (≥1min)": ("#08313A", "-", "s"), "MAP<65 (≥5min, sustained)": (S.C["M0"], "--", "D")}
    for name, (col, ls, mk) in styles.items():
        d = sev.get(name, {}); hs = sorted(int(k) for k in d)
        au = [d[str(h)]["auroc"] for h in hs]
        b.plot(hs, au, ls, color=col, marker=mk, ms=3, label=name.replace(" (", "\n("))
    S.finish(b, "forecast horizon (min)", "AUROC", "Severity gradient")
    b.set_ylim(0.68, 1.0); b.set_box_aspect(1)
    b.legend(loc="lower left", fontsize=4.8, labelspacing=0.25, handlelength=1.4); S.panel_letter(b, "b")

    # d — operating points vs horizon at spec>=0.90
    d = axs["d"]
    hs = S.horizons_sorted(hyp["per_horizon"])
    for metric, col, mk in [("sens", S.C["M1"], "o"), ("ppv", S.C["M0"], "s"),
                            ("npv", S.C["transition"], "^"), ("f1", S.C["zhu"], "D")]:
        vals = [hyp["per_horizon"][S.hkey(h)]["M1"]["operating_points"]["spec90"][metric] for h in hs]
        d.plot(hs, vals, "-", color=col, marker=mk, ms=3, label=metric.upper())
    S.finish(d, "forecast horizon (min)", "value at spec ≥ 0.90", "Operating characteristics")
    d.set_xticks(hs); d.set_ylim(-0.02, 1.02); d.set_box_aspect(1)
    d.legend(loc="lower left", fontsize=5.0, ncol=2, columnspacing=1.0, handlelength=1.4); S.panel_letter(d, "d")

    # c — subgroup forest (tall panel)
    _forest(axs["c"], sg)
    S.panel_letter(axs["c"], "c", dx=0.02, dy=1.04)

    S.save_fig(fig, "Fig4_clinical_robustness")


def _forest(ax, sg):
    """Forest with all text on the RIGHT (outer figure margin) so nothing spills into
    the neighbouring panels on the left."""
    subs = sg["subgroups"]; overall = sg["overall"]
    rows = []
    last_var = None
    for s in subs:
        if s["var"] != last_var:
            rows.append(("header", s["var"], None, None, None)); last_var = s["var"]
        rows.append(("row", s["level"], s["auroc"], s["ci"], s.get("n_cases")))
    rows = rows[::-1]
    y = 0; yticks = []; ylabels = []
    for kind, lab, au, ci, n in rows:
        if kind == "header":
            ax.text(1.03, y, lab, fontsize=6.2, fontweight="bold", va="center", ha="left",
                    transform=ax.get_yaxis_transform())
            yticks.append(y); ylabels.append("")
        else:
            ax.errorbar(au, y, xerr=[[au-ci[0]], [ci[1]-au]], fmt="o", color=S.C["M1"],
                        ms=3.2, capsize=1.8, lw=1.0)
            yticks.append(y); ylabels.append(f"{lab} (n={n})  {au:.3f}")
        y += 1
    ax.axvline(overall["auroc"], color=S.C["persist"], lw=0.9, ls="--")
    # 'overall' key inside the plot (top-left, empty region) — not on the x-axis
    ax.text(0.03, 0.995, f"– –  overall {overall['auroc']:.3f}", transform=ax.transAxes,
            fontsize=5.6, color="#555", ha="left", va="top")
    ax.set_yticks(yticks); ax.set_yticklabels(ylabels, fontsize=5.6)
    ax.yaxis.tick_right()                             # tick labels on the right
    ax.set_ylim(-0.6, y-0.4); ax.set_xlim(0.88, 0.96); ax.set_xticks([0.88, 0.90, 0.92, 0.94, 0.96])
    ax.set_xlabel("hypotension AUROC @5 min"); ax.set_title("Subgroup robustness", loc="center")
    ax.spines["left"].set_visible(False); ax.spines["right"].set_visible(True)
    ax.tick_params(axis="y", length=0)


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


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    os.makedirs(S.FIG_DIR, exist_ok=True); os.makedirs(S.TAB_DIR, exist_ok=True)
    print(f"[paper] tag={TAG}  font={S.SANS}", flush=True)
    print("[paper] Figure 1 ..."); figure1(TAG)
    print("[paper] Figure 2 ..."); figure2(TAG)
    print("[paper] Figure 3 ..."); figure3(TAG)
    print("[paper] Figure 4 ..."); figure4(TAG)
    print("[paper] Tables ..."); table1_cohort(TAG); table2_accuracy(TAG); table3_classification(TAG)
    print("[paper] Done. Figures in outputs/figs/paper/ ; tables in results/tables/", flush=True)


if __name__ == "__main__":
    main()
