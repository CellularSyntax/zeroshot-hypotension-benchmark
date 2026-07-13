#!/usr/bin/env python3
"""Aggregate results/compute_footprint.json (one row per model, written by compute_footprint.py)
into the paper's compute-footprint table (CSV + Markdown + LaTeX), matching the style of the other
results/tables/* generators. Stdlib only (no pandas/numpy) so it runs on a login node. No model is
loaded here."""
from __future__ import annotations
import json, os

SRC = "results/compute_footprint.json"
ORDER = ["tirex2", "chronos", "timesfm", "moirai", "tft", "patchtst"]
LABEL = {"tirex2": "TiRex-2", "chronos": "Chronos-Bolt", "timesfm": "TimesFM-2.5",
         "moirai": "Moirai-1.1-R", "tft": "TFT", "patchtst": "PatchTST"}
KIND = {"tirex2": "Zero-shot (cov-aware)", "chronos": "Zero-shot", "timesfm": "Zero-shot",
        "moirai": "Zero-shot", "tft": "Task-trained", "patchtst": "Task-trained"}
COLS = [("Model", "Model"), ("Type", "Type"), ("params_M", "Params (M)"),
        ("load_time_s", "Load time (s)"), ("latency_ms_per_window", "Latency (ms/window)"),
        ("throughput_windows_per_s", "Throughput (windows/s)"),
        ("peak_gpu_mem_MB", "Peak GPU mem (MB)")]


def fmt(v):
    if v is None:
        return "--"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


def latex_escape(s):
    return s.replace("&", r"\&").replace("%", r"\%").replace("_", r"\_")


def main():
    if not os.path.exists(SRC):
        raise SystemExit(f"{SRC} not found — run scripts/compute_footprint.py per model first.")
    rows = json.load(open(SRC))
    recs = []
    for m in ORDER:
        if m not in rows:
            continue
        r = rows[m]
        rec = {"Model": LABEL[m], "Type": KIND[m]}
        for key, _ in COLS[2:]:
            rec[key] = r.get(key)
        recs.append(rec)
    if not recs:
        raise SystemExit(f"{SRC} has no recognised model rows.")
    headers = [h for _, h in COLS]
    keys = [k for k, _ in COLS]
    os.makedirs("results/tables", exist_ok=True)

    # CSV
    import csv
    with open("results/tables/TableS_compute_footprint.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(headers)
        for rec in recs:
            w.writerow([fmt(rec[k]) for k in keys])

    # Markdown
    with open("results/tables/TableS_compute_footprint.md", "w") as f:
        f.write("| " + " | ".join(headers) + " |\n")
        f.write("|" + "|".join(["---"] * len(headers)) + "|\n")
        for rec in recs:
            f.write("| " + " | ".join(fmt(rec[k]) for k in keys) + " |\n")

    # LaTeX (booktabs) — matches the other supp tables
    colfmt = "ll" + "r" * (len(headers) - 2)
    with open("results/tables/TableS_compute_footprint.tex", "w") as f:
        f.write("\\begin{tabular}{" + colfmt + "}\n\\toprule\n")
        f.write(" & ".join(latex_escape(h) for h in headers) + " \\\\\n\\midrule\n")
        for rec in recs:
            f.write(" & ".join(latex_escape(fmt(rec[k])) for k in keys) + " \\\\\n")
        f.write("\\bottomrule\n\\end{tabular}\n")

    dev = next((rows[m].get("device") for m in ORDER if m in rows), "?")
    nb = next((rows[m].get("batch_size") for m in ORDER if m in rows), "?")
    print(f"wrote results/tables/TableS_compute_footprint.{{csv,md,tex}}  (device={dev}, batch_size={nb})")
    # plain console table
    widths = [max(len(headers[i]), max(len(fmt(rec[keys[i]])) for rec in recs)) for i in range(len(keys))]
    print("  ".join(h.ljust(widths[i]) for i, h in enumerate(headers)))
    for rec in recs:
        print("  ".join(fmt(rec[keys[i]]).ljust(widths[i]) for i in range(len(keys))))


if __name__ == "__main__":
    main()
