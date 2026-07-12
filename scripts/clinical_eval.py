"""Clinician-facing evaluation of the TiRex-2 impending-hypotension alarm (A/B/C), post-hoc.
Standalone: reads the per-window rows (phase3_ablation.py) for the model's risk scores + labels, and
recomputes event timing / severity from the cached MAP trajectories (vitaldb_loader). NO forecast
rerun. Reuses metric code from hypo_eval.

A. Early-warning value  — lead time (how many min before the event the alarm fires), sensitivity, and
   alarm burden (false alarms / hour) at a deployable operating point (dev-selected, spec>=0.90).
B. Severity gradient    — does the (65-mmHg) risk score still discriminate MORE severe hypotension
   (MAP<55, <50) and SUSTAINED (>=5 min) events? AUROC + lead time by severity.
   [Caveat: risk is thresholded at 65; a threshold-matched P(MAP<thr) needs saved quantiles -> next run.]
C. Decision curve analysis — net benefit of acting on the alarm vs treat-all / treat-none, per horizon.

Run:  PYTHONPATH=scripts <venv>/bin/python scripts/clinical_eval.py n300_s1
Writes results/clinical_eval_<tag>.json + outputs/figs/clinical_{leadtime,severity,decisioncurve}_<tag>.png
"""
import csv, glob, json, sys
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import vitaldb_loader as L
from hypo_eval import (auroc, clustered_boot_ci, op_metrics, pick_threshold,
                       caseid_to_subject, load_rows, split_subjects)

SEVERITIES = [(65, 1, "MAP<65 (≥1min)"), (55, 1, "MAP<55 (≥1min)"),
              (50, 1, "MAP<50 (≥1min)"), (65, 5, "MAP<65 (≥5min, sustained)")]
ALARM_MIN = 15   # alarm score = P(event within this horizon); the longest horizon


def first_sustained_onset(below, min_run):
    """Start index of the first run of >=min_run consecutive True in `below`, else None."""
    run = 0
    for i, b in enumerate(below):
        run = run + 1 if b else 0
        if run >= min_run:
            return i - min_run + 1
    return None


def per_origin_table(rows, cfg, clin, hsteps, dt, min_run_1min):
    """Join model risk (from windows CSV) to recomputed event timing/severity (from cache)."""
    # risk_M1[h_min] per (caseid, t0), and the set of t0 per case
    risk = defaultdict(dict); by_case = defaultdict(set)
    for r in rows:
        if r["risk_M1"] in ("", "nan"):
            continue
        key = (str(r["caseid"]), int(r["t0"])); risk[key][int(r["h_min"])] = float(r["risk_M1"])
        by_case[str(r["caseid"])].add(int(r["t0"]))

    H = max(hsteps)
    table = []   # one row per origin
    for caseid, t0s in by_case.items():
        rec = L.load_case(caseid, cfg, clin)
        if rec is None:
            continue
        truth = rec["target"]
        for t0 in sorted(t0s):
            seg = truth[t0:t0 + H]
            if len(seg) < H:
                seg = np.r_[seg, np.full(H - len(seg), np.nan)]
            cur = truth[t0 - 1] if t0 >= 1 else seg[0]   # last observed MAP (for "normotensive now?")
            row = {"caseid": caseid, "t0": t0, "risk": risk[(caseid, t0)],
                   "cur_map": float(cur) if np.isfinite(cur) else np.nan}
            for thr, sustain_min, _ in SEVERITIES:
                mr = max(1, int(sustain_min * 60 / dt))
                below = np.isfinite(seg) & (seg < thr)
                onset = first_sustained_onset(below, mr)
                row[f"onset_{thr}_{sustain_min}"] = onset * dt / 60 if onset is not None else None
                # per-horizon label: sustained run within first h steps
                for h, hs in zip(HMIN, hsteps):
                    row[f"ev_{thr}_{sustain_min}_{h}"] = int(
                        first_sustained_onset(below[:hs], mr) is not None)
            table.append(row)
    return table


HMIN = [1, 3, 5, 7, 10, 15]


def analysis_A(table, c2s, dev_subjects, tag, n_boot=800):
    """Early warning: lead time + sensitivity + alarm burden at spec>=0.90 (dev-selected)."""
    cid = np.array([r["caseid"] for r in table])
    is_dev = np.array([c2s.get(str(c), str(c)) in dev_subjects for c in cid])
    score = np.array([r["risk"].get(ALARM_MIN, np.nan) for r in table])
    event = np.array([r[f"ev_65_1_{ALARM_MIN}"] for r in table])           # event within 15 min
    onset = np.array([r["onset_65_1"] if r["onset_65_1"] is not None else np.nan for r in table])
    # EARLY-WARNING framing: only origins where the patient is normotensive NOW (deployment scenario:
    # ask "will a stable patient crash?"). Already-hypotensive origins are detection, not prediction.
    curmap = np.array([r["cur_map"] for r in table])
    normo = np.isfinite(curmap) & (curmap >= 65) & np.isfinite(score)

    dev_m, test_m = is_dev & normo, (~is_dev) & normo
    thr = pick_threshold(event[dev_m], score[dev_m], "spec90")
    ev_t, sc_t, on_t, ct = event[test_m], score[test_m], onset[test_m], cid[test_m]
    tmask = test_m
    alarm = sc_t >= thr
    m = op_metrics(ev_t, sc_t, thr)
    # lead time for true positives (event within 15 min AND alarmed): minutes from origin to onset
    tp = (ev_t == 1) & alarm
    leads = on_t[tp]; leads = leads[np.isfinite(leads)]
    # alarm burden: false alarms among event-free origins, expressed per hour at the eval cadence
    spacing = []
    for c in np.unique(ct):
        ts = sorted(int(r["t0"]) for r in table if r["caseid"] == c and c2s.get(c, c) not in dev_subjects)
        spacing += list(np.diff(ts))
    dt_min = np.median(spacing) * DT / 60 if spacing else np.nan     # min between origins
    origins_per_hr = 60 / dt_min if dt_min and np.isfinite(dt_min) else np.nan
    fp_rate = 1 - m["spec"] if m["spec"] is not None else np.nan       # per event-free origin
    out = {"threshold": round(float(thr), 4), "n_test": int(tmask.sum()),
           "n_events_15min": int(ev_t.sum()), "sensitivity": m["sens"], "specificity": m["spec"],
           "ppv": m["ppv"], "auroc": round(auroc(ev_t, sc_t), 4),
           "auroc_CI95": clustered_boot_ci(ct, ev_t, sc_t, auroc, n_boot),
           "lead_time_min_median": round(float(np.median(leads)), 2) if len(leads) else None,
           "lead_time_min_IQR": [round(float(np.percentile(leads, 25)), 2),
                                 round(float(np.percentile(leads, 75)), 2)] if len(leads) else None,
           "pct_detected_ge5min_ahead": round(float(np.mean(leads >= 5)) * 100, 1) if len(leads) else None,
           "pct_detected_ge2min_ahead": round(float(np.mean(leads >= 2)) * 100, 1) if len(leads) else None,
           "origin_cadence_min": round(float(dt_min), 1) if np.isfinite(dt_min) else None,
           "false_alarms_per_hour": round(float(fp_rate * origins_per_hr), 2)
           if np.isfinite(fp_rate) and np.isfinite(origins_per_hr) else None}
    # early-warning YIELD curve: of ALL impending events, the % flagged at least t min ahead
    # (t=0 intercept == sensitivity; the decay is the lead-time distribution). One value per minute.
    ev_total = int(ev_t.sum())
    out["lead_curve"] = {str(t): (round(float(np.sum(leads >= t)) / ev_total * 100, 1) if ev_total else None)
                         for t in range(0, ALARM_MIN + 1)}

    fig, ax = plt.subplots(1, 2, figsize=(12, 4.6))
    if len(leads):
        ax[0].hist(leads, bins=np.arange(0, ALARM_MIN + 1, 1), color="C1", edgecolor="w")
        ax[0].axvline(np.median(leads), color="k", ls="--", lw=1,
                      label=f"median {np.median(leads):.1f} min")
        ax[0].legend(fontsize=9)
    ax[0].set_xlabel("lead time (min before MAP<65 onset)"); ax[0].set_ylabel("alarms")
    ax[0].set_title("A. Warning lead time (true positives)")
    ax[1].axis("off")
    txt = (f"Operating point: dev-selected @ spec≥0.90\n"
           f"  threshold (risk) = {out['threshold']:.3f}\n\n"
           f"Sensitivity  = {100*out['sensitivity']:.1f}%\n"
           f"Specificity  = {100*out['specificity']:.1f}%\n"
           f"PPV          = {100*out['ppv']:.1f}%\n"
           f"AUROC        = {out['auroc']:.3f} {out['auroc_CI95']}\n\n"
           f"Median lead time = {out['lead_time_min_median']} min "
           f"(IQR {out['lead_time_min_IQR']})\n"
           f"Detected ≥5 min ahead = {out['pct_detected_ge5min_ahead']}%\n"
           f"Detected ≥2 min ahead = {out['pct_detected_ge2min_ahead']}%\n\n"
           f"Alarm cadence ≈ every {out['origin_cadence_min']} min\n"
           f"False alarms ≈ {out['false_alarms_per_hour']} / hour\n"
           f"(n_test={out['n_test']} origins, {out['n_events_15min']} with IOH≤15min)")
    ax[1].text(0.02, 0.98, txt, va="top", ha="left", family="monospace", fontsize=10)
    ax[1].set_title("A. Deployable operating point")
    fig.suptitle(f"Early-warning value (origins normotensive at baseline) — {tag}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(f"outputs/figs/clinical_leadtime_{tag}.png", dpi=120)
    plt.close(fig)
    return out


def analysis_B(table, c2s, dev_subjects, tag, n_boot=800):
    """Severity gradient: AUROC of the 65-risk score vs more severe / sustained events, per horizon."""
    cid = np.array([r["caseid"] for r in table])
    is_test = np.array([c2s.get(str(c), str(c)) not in dev_subjects for c in cid])
    ct = cid[is_test]
    res = {}
    for thr, sustain, label in SEVERITIES:
        per_h = {}
        for h in HMIN:
            y = np.array([r[f"ev_{thr}_{sustain}_{h}"] for r in table])[is_test]
            s = np.array([r["risk"].get(h, np.nan) for r in table])[is_test]
            prev = float(y.mean())
            if y.sum() >= 8 and (y == 0).sum() >= 8:
                per_h[h] = {"prevalence": round(prev, 4), "auroc": round(auroc(y, s), 4),
                            "auroc_CI95": clustered_boot_ci(ct, y, s, auroc, n_boot),
                            "n_events": int(y.sum())}
            else:
                per_h[h] = {"prevalence": round(prev, 4), "auroc": None, "n_events": int(y.sum())}
        res[label] = per_h

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 4.8))
    for (thr, sustain, label), c in zip(SEVERITIES, ("C0", "C1", "C3", "C2")):
        hs = [h for h in HMIN if res[label][h]["auroc"] is not None]
        ax[0].plot(hs, [res[label][h]["auroc"] for h in hs], "-o", color=c, label=label)
        ax[1].plot(HMIN, [res[label][h]["prevalence"] for h in HMIN], "-o", color=c, label=label)
    ax[0].set_xlabel("horizon (min)"); ax[0].set_ylabel("AUROC (risk_65 vs severe event)")
    ax[0].set_ylim(0.5, 1.0); ax[0].set_title("B. Discrimination by severity"); ax[0].legend(fontsize=8)
    ax[1].set_xlabel("horizon (min)"); ax[1].set_ylabel("event prevalence")
    ax[1].set_title("B. Event prevalence by severity"); ax[1].legend(fontsize=8)
    fig.suptitle(f"Severity gradient (65-mmHg risk score) — {tag}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95]); fig.savefig(f"outputs/figs/clinical_severity_{tag}.png", dpi=120)
    plt.close(fig)
    return res


def analysis_C(rows, c2s, dev_subjects, tag):
    """Decision curve analysis: net benefit vs treat-all / treat-none, per horizon (held-out test)."""
    horizons = [5, 10, 15]
    pts = np.linspace(0.01, 0.5, 60)
    curves = {}
    fig, ax = plt.subplots(1, len(horizons), figsize=(5 * len(horizons), 4.4), squeeze=False)
    for j, h in enumerate(horizons):
        hr = [r for r in rows if int(r["h_min"]) == h and r["risk_M1"] not in ("", "nan")
              and c2s.get(str(r["caseid"]), str(r["caseid"])) not in dev_subjects]
        y = np.array([float(r["hypo_event"]) for r in hr]); s = np.array([float(r["risk_M1"]) for r in hr])
        n = len(y); prev = y.mean()
        nb_model, nb_all = [], []
        for pt in pts:
            pred = s >= pt
            tp = np.sum(pred & (y == 1)); fp = np.sum(pred & (y == 0))
            nb_model.append(tp / n - fp / n * (pt / (1 - pt)))
            nb_all.append(prev - (1 - prev) * (pt / (1 - pt)))
        curves[h] = {"pt": pts.tolist(), "nb_model": nb_model, "nb_treat_all": nb_all, "prevalence": float(prev)}
        a = ax[0][j]
        a.plot(pts, nb_model, color="C1", lw=2, label="alarm (model)")
        a.plot(pts, nb_all, color="grey", lw=1, ls="--", label="treat all")
        a.axhline(0, color="k", lw=0.8, label="treat none")
        a.set_ylim(min(-0.02, prev * -0.3), max(nb_model) * 1.15 + 1e-3)
        a.set_xlabel("threshold probability"); a.set_ylabel("net benefit")
        a.set_title(f"{h} min (prev {prev:.2f})"); a.legend(fontsize=8)
    fig.suptitle(f"C. Decision curve analysis (held-out test) — {tag}", fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94]); fig.savefig(f"outputs/figs/clinical_decisioncurve_{tag}.png", dpi=120)
    plt.close(fig)
    return curves


DT = None
def main():
    global DT
    tag = sys.argv[1] if len(sys.argv) > 1 else "n300_s1"
    rows, files = load_rows(tag)
    import yaml
    ev = yaml.safe_load(open("configs/eval.yaml"))
    cfg = L.load_config("datasets/vitaldb/configs/data.yaml"); clin = L._clinical_index(cfg["clinical_csv"])
    probe = next((L.load_case(str(r["caseid"]), cfg, clin) for r in rows), None)
    DT = probe["interval_s"]
    hsteps = [int(m * 60 / DT) for m in HMIN]
    min_run_1min = max(1, int(60 / DT))
    c2s = caseid_to_subject()
    dev_subjects = split_subjects([r["caseid"] for r in rows], c2s, seed=0)
    print(f"tag={tag}: {len(files)} shards, {len(rows)} rows, dt={DT}s, dev subj={len(dev_subjects)}",
          flush=True)

    table = per_origin_table(rows, cfg, clin, hsteps, DT, min_run_1min)
    print(f"built {len(table)} origins from cache", flush=True)
    A = analysis_A(table, c2s, dev_subjects, tag)
    print(f"  A: sens={A['sensitivity']:.3f} spec={A['specificity']:.3f} "
          f"median_lead={A['lead_time_min_median']}min  FA/hr={A['false_alarms_per_hour']}", flush=True)
    B = analysis_B(table, c2s, dev_subjects, tag)
    for lbl in B:
        a5 = B[lbl][5]["auroc"]
        print(f"  B: {lbl:28s} 5-min AUROC={a5}  (prev {B[lbl][5]['prevalence']})", flush=True)
    C = analysis_C(rows, c2s, dev_subjects, tag)
    json.dump({"tag": tag, "A_early_warning": A, "B_severity": B,
               "C_decision_curve": {str(k): v for k, v in C.items()}},
              open(f"results/clinical_eval_{tag}.json", "w"), indent=1)
    print(f"\nwrote results/clinical_eval_{tag}.json\n"
          f"wrote outputs/figs/clinical_{{leadtime,severity,decisioncurve}}_{tag}.png", flush=True)


if __name__ == "__main__":
    main()
