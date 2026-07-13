"""Emit per-case MOVER covariates for the external-cohort subgroup (fairness) panel.

Companion to scripts/mover_cohort_table.py. Writes one small CSV keyed by caseid with the
covariates MOVER actually carries (sex, age, BMI from height+weight, anaesthesia duration);
ASA and department are absent from the MOVER SIS and are not emitted. The per-window model
predictions (risk_M1) already live in results/ablation_windows_mover_art.csv on the Mac, so the
subgroup AUROC + case-clustered bootstrap + forest plot are computed locally after joining on
caseid -- this script only exposes the covariates that are not available off-cluster.

STDLIB ONLY (csv + math) -- no numpy -- so it runs with the cluster LOGIN-node python3
directly, no container / compute node required. From the project root:
    PYTHONPATH=scripts:datasets/mover python3 scripts/mover_subgroup_covariates.py

Writes results/mover_subgroup_covariates.csv  (pull to the Mac).
"""
from __future__ import annotations
import csv, math, os, sys

WINDOWS = "results/ablation_windows_mover_art.csv"
CLINICAL = "datasets/mover/clinical_data.csv"
OUT = "results/mover_subgroup_covariates.csv"


def fnum(r, k):
    try:
        return float(r[k])
    except (ValueError, TypeError, KeyError):
        return float("nan")


def norm_caseid(c):
    """Pure-digit ids -> strip zero padding; hex hash ids -> keep verbatim (matches the
    convention in mover_cohort_table.py and naive_baselines.py)."""
    s = str(c).strip()
    d = s[1:] if s[:1] in "+-" else s
    return str(int(s)) if d.isdigit() else s


def windows_caseids(path):
    cids = set()
    for r in csv.DictReader(open(path)):
        cids.add(norm_caseid(r["caseid"]))
    return cids


def main():
    if not os.path.exists(WINDOWS):
        sys.exit(f"missing {WINDOWS} -- run the MOVER ablation first")
    if not os.path.exists(CLINICAL):
        sys.exit(f"missing {CLINICAL} -- build the MOVER cache first")

    keep = windows_caseids(WINDOWS)
    cd = {norm_caseid(r["caseid"]): r
          for r in csv.DictReader(open(CLINICAL, encoding="utf-8-sig"))}

    os.makedirs("results", exist_ok=True)
    n = 0
    with open(OUT, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["caseid", "sex", "age", "bmi", "anes_min"])
        for c in sorted(keep):
            r = cd.get(c)
            if r is None:
                continue
            age = fnum(r, "age")
            ht, wt = fnum(r, "height"), fnum(r, "weight")
            bmi = wt / (ht / 100.0) ** 2 if (ht and math.isfinite(ht) and ht > 0
                                             and math.isfinite(wt)) else float("nan")
            if not (math.isfinite(bmi) and 5 < bmi < 100):
                bmi = float("nan")
            dur = (fnum(r, "aneend") - fnum(r, "anestart")) / 60.0
            if not (math.isfinite(dur) and dur > 0):
                dur = float("nan")
            sex = str(r.get("sex", "")).strip().upper()
            sex = "M" if sex.startswith("M") else ("F" if sex.startswith("F") else "")
            w.writerow([c, sex,
                        "" if math.isnan(age) else round(age, 1),
                        "" if math.isnan(bmi) else round(bmi, 2),
                        "" if math.isnan(dur) else round(dur, 1)])
            n += 1
    print(f"wrote {OUT}  ({n} cases with covariates, of {len(keep)} in windows)")


if __name__ == "__main__":
    main()
