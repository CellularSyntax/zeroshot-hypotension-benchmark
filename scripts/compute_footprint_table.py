#!/usr/bin/env python3
"""Aggregate results/compute_footprint.json (one row per model, written by compute_footprint.py)
into the paper's compute-footprint table (CSV + Markdown + LaTeX), matching the style of the other
results/tables/* generators. No model is loaded here."""
from __future__ import annotations
import json, os
import pandas as pd

SRC = "results/compute_footprint.json"
ORDER = ["tirex2", "chronos", "timesfm", "moirai", "tft", "patchtst"]
LABEL = {"tirex2": "TiRex-2", "chronos": "Chronos-Bolt", "timesfm": "TimesFM-2.5",
         "moirai": "Moirai-1.1-R", "tft": "TFT", "patchtst": "PatchTST"}
KIND = {"tirex2": "Zero-shot (cov-aware)", "chronos": "Zero-shot", "timesfm": "Zero-shot",
        "moirai": "Zero-shot", "tft": "Task-trained", "patchtst": "Task-trained"}


def main():
    if not os.path.exists(SRC):
        raise SystemExit(f"{SRC} not found — run scripts/compute_footprint.py per model first.")
    rows = json.load(open(SRC))
    recs = []
    for m in ORDER:
        if m not in rows:
            continue
        r = rows[m]
        recs.append({
            "Model": LABEL[m],
            "Type": KIND[m],
            "Params (M)": r.get("params_M"),
            "Load time (s)": r.get("load_time_s"),
            "Latency (ms/window)": r.get("latency_ms_per_window"),
            "Throughput (windows/s)": r.get("throughput_windows_per_s"),
            "Peak GPU mem (MB)": r.get("peak_gpu_mem_MB"),
        })
    df = pd.DataFrame(recs)
    os.makedirs("results/tables", exist_ok=True)
    df.to_csv("results/tables/TableS_compute_footprint.csv", index=False)
    with open("results/tables/TableS_compute_footprint.md", "w") as f:
        f.write(df.to_markdown(index=False))
    # LaTeX (booktabs) — matches the other supp tables
    with open("results/tables/TableS_compute_footprint.tex", "w") as f:
        f.write(df.to_latex(index=False, escape=True, na_rep="--",
                            column_format="ll" + "r" * (df.shape[1] - 2)))
    dev = next((rows[m].get("device") for m in ORDER if m in rows), "?")
    nb = next((rows[m].get("batch_size") for m in ORDER if m in rows), "?")
    print(f"wrote results/tables/TableS_compute_footprint.{{csv,md,tex}}  (device={dev}, batch_size={nb})")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
