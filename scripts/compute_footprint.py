#!/usr/bin/env python3
"""Compute-footprint benchmark: load time, inference latency/throughput, peak GPU memory and
parameter count for every model in the paper, measured through their REAL inference paths.

This reuses the exact loaders and forecast calls used to produce the results
(scripts/baselines/zeroshot.py adapters for the univariate foundation models, tirex2.load_model +
phase3_ablation.build_ts/batched_forecast for TiRex-2, baselines.models.build_model +
baselines.train.predict for the task-trained TFT/PatchTST) so the numbers reflect deployment cost,
not a re-implementation. No training and no metric computation happen here.

Run per model on a GPU node (A100), e.g.:

  # foundation models (zero-shot):
  for M in tirex2 chronos timesfm moirai; do
    PYTHONPATH=scripts:datasets/vitaldb:datasets/mover python3 scripts/compute_footprint.py \
      --model $M --match-tirex results/ablation_windows_all2873.csv --device cuda \
      --n-windows 512 --batch-size 64
  done
  # task-trained baselines (need their all-train checkpoint):
  for M in tft patchtst; do
    PYTHONPATH=scripts:datasets/vitaldb:datasets/mover python3 scripts/compute_footprint.py \
      --model $M --match-tirex results/ablation_windows_all2873.csv --device cuda \
      --ckpt results/baseline_ckpt_vitaldb15_$M.pt --n-windows 512 --batch-size 64
  done

Each run appends one row to results/compute_footprint.json (keyed by model). Aggregate into the
paper table with scripts/compute_footprint_table.py.
"""
from __future__ import annotations
import argparse, csv, json, os, time
import numpy as np


def _gpu_reset(device):
    try:
        import torch
        if str(device).startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            return torch
    except Exception:
        pass
    return None


def _gpu_peak_mb(torch, device):
    if torch is not None and str(device).startswith("cuda"):
        torch.cuda.synchronize()
        return torch.cuda.max_memory_allocated() / 1e6
    return float("nan")


def _count_params(obj):
    """Best-effort parameter count over any nn.Module we can find on the loaded object."""
    import torch.nn as nn
    seen = set(); total = 0
    def walk(x, depth=0):
        nonlocal total
        if depth > 3 or id(x) in seen:
            return
        seen.add(id(x))
        if isinstance(x, nn.Module):
            total = max(total, sum(p.numel() for p in x.parameters()))
            return
        for attr in ("model", "module", "pipe", "predictor", "network", "_model"):
            if hasattr(x, attr):
                walk(getattr(x, attr), depth + 1)
    walk(obj)
    return total or float("nan")


def build_contexts(match_tirex, config, eval_config, cov, n_windows, seed):
    """Rebuild the exact windows used in the paper (same cohort, context, horizon)."""
    import yaml
    import phase3_ablation as P
    from baselines import data as D
    ev = yaml.safe_load(open(eval_config))
    L = P.get_loader(config); cfg = L.load_config(config); clin = L._clinical_index(cfg["clinical_csv"])
    preset = P.COV_PRESETS[cov]
    P.FUTURE_COV = list(preset["future"]); P.PRIMARY_COV = preset["primary"]; P.TRANSITION_THR = preset["trans_thr"]
    cases = sorted({r["caseid"] for r in csv.DictReader(open(match_tirex))})
    probe = next((r for r in (L.load_case(c, cfg, clin) for c in cases) if r is not None), None)
    dt = probe["interval_s"]; Lc = int(ev["context_min"] * 60 / dt)
    hsteps = [int(m * 60 / dt) for m in P.HORIZON_STEPS_MIN]; H = max(hsteps)
    stride = int(ev["origin_stride_min"] * 60 / dt); warmup = int(ev["warmup_min"] * 60 / dt)
    min_run = max(1, int(ev.get("hypotension", {}).get("min_sustain_min", 1) * 60 / dt))
    win, past_names, fut_names = D.build_windows(cases, cfg, clin, Lc, H, stride, warmup,
                                                 20, dt, min_run, quiet=True)
    rng = np.random.default_rng(seed)
    if n_windows and len(win) > n_windows:
        idx = rng.choice(len(win), n_windows, replace=False); win = [win[i] for i in idx]
    return win, cfg, clin, L, Lc, H, dt, past_names, fut_names


def time_foundation(model_name, win, Lc, H, bs, device):
    """Time the zero-shot univariate adapters (chronos/timesfm/moirai)."""
    from baselines.zeroshot import build_adapter
    torch = _gpu_reset(device)
    t0 = time.time()
    adapter = build_adapter(model_name, Lc, H).load(device)
    load_s = time.time() - t0
    params = _count_params(adapter)
    contexts = [w["past"][:, 0].astype(np.float32) for w in win]
    # warm-up batch (kernel compile / cudnn autotune) — excluded from timing
    _ = adapter.forecast(contexts[:min(bs, len(contexts))], None, H)
    torch2 = _gpu_reset(device)
    t0 = time.time()
    for i in range(0, len(contexts), bs):
        adapter.forecast(contexts[i:i + bs], None, H)
    if torch2 is not None:
        torch2.cuda.synchronize()
    infer_s = time.time() - t0
    peak = _gpu_peak_mb(torch2, device)
    return load_s, infer_s, len(contexts), peak, params


def build_tirex_items(match_tirex, config, eval_config, cov, n_windows, seed):
    """Build TiRex-2 TimeseriesType items for the covariate-aware (M1: past+future) arm, exactly as
    phase3_ablation does — per case, via load_case/make_windows/build_ts. This is the deployed
    configuration whose cost we want to report."""
    import yaml
    import phase3_ablation as P
    ev = yaml.safe_load(open(eval_config))
    L = P.get_loader(config); cfg = L.load_config(config); clin = L._clinical_index(cfg["clinical_csv"])
    preset = P.COV_PRESETS[cov]
    P.FUTURE_COV = list(preset["future"]); P.PRIMARY_COV = preset["primary"]; P.TRANSITION_THR = preset["trans_thr"]
    cases = sorted({r["caseid"] for r in csv.DictReader(open(match_tirex))})
    probe = next((r for r in (L.load_case(c, cfg, clin) for c in cases) if r is not None), None)
    dt = probe["interval_s"]; Lc = int(ev["context_min"] * 60 / dt)
    hsteps = [int(m * 60 / dt) for m in P.HORIZON_STEPS_MIN]; H = max(hsteps)
    stride = int(ev["origin_stride_min"] * 60 / dt); warmup = int(ev["warmup_min"] * 60 / dt)
    items = []
    for caseid in cases:
        rec = L.load_case(caseid, cfg, clin)
        if rec is None:
            continue
        for t0 in P.make_windows(rec, Lc, H, stride, warmup, 20):
            items.append(P.build_ts(rec, t0, Lc, H, use_past=True, use_future=True))  # M1 arm
        if n_windows and len(items) >= n_windows:
            break
    rng = np.random.default_rng(seed)
    if n_windows and len(items) > n_windows:
        idx = sorted(rng.choice(len(items), n_windows, replace=False))
        items = [items[i] for i in idx]
    return items, Lc, H, dt


def time_tirex2(items, H, bs, device):
    """Time TiRex-2 through its real load + forecast path (covariate-aware M1 items)."""
    from tirex2 import load_model
    _ = _gpu_reset(device)
    t0 = time.time()
    model = load_model("NX-AI/TiRex-2", device=device)
    load_s = time.time() - t0
    params = _count_params(model)
    _ = model.forecast(items[:min(bs, len(items))], prediction_length=H, output_type="numpy")  # warm-up
    torch2 = _gpu_reset(device)
    t0 = time.time()
    for i in range(0, len(items), bs):
        model.forecast(items[i:i + bs], prediction_length=H, output_type="numpy")
    if torch2 is not None:
        torch2.cuda.synchronize()
    infer_s = time.time() - t0
    peak = _gpu_peak_mb(torch2, device)
    return load_s, infer_s, len(items), peak, params


def time_baseline(model_name, ckpt, win, device):
    """Time a task-trained baseline (TFT/PatchTST) through build_model + train.predict."""
    import torch
    from baselines.models import build_model
    from baselines import data as D, train as TR
    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    torch_ = _gpu_reset(device)
    t0 = time.time()
    model = build_model(ck["model"], ck["n_past"], ck["n_fut"], ck["H"],
                        context_len=ck["Lc"], d=ck["d_model"]).to(device)
    arm = "M1" if "M1" in ck["state"] else next(iter(ck["state"]))
    model.load_state_dict(ck["state"][arm]); model.eval()
    load_s = time.time() - t0
    params = sum(p.numel() for p in model.parameters())
    X = D.to_tensors(win, ck["norm"], use_future=(ck["n_fut"] > 0))
    _ = TR.predict(model, X, device, ck["norm"], bs=min(512, len(win)))  # warm-up
    torch2 = _gpu_reset(device)
    t0 = time.time()
    TR.predict(model, X, device, ck["norm"], bs=512)
    if torch2 is not None:
        torch2.cuda.synchronize()
    infer_s = time.time() - t0
    peak = _gpu_peak_mb(torch2, device)
    return load_s, infer_s, len(win), peak, params


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    choices=["tirex2", "chronos", "timesfm", "moirai", "tft", "patchtst"])
    ap.add_argument("--match-tirex", required=True, help="ablation_windows_*.csv fixing the cohort")
    ap.add_argument("--config", default="datasets/vitaldb/configs/data.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--cov", default="ce")
    ap.add_argument("--ckpt", default=None, help="baseline checkpoint (required for tft/patchtst)")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-windows", type=int, default=512, help="windows to time over (subsampled)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/compute_footprint.json")
    args = ap.parse_args()

    if args.model == "tirex2":
        items, Lc, H, dt = build_tirex_items(
            args.match_tirex, args.config, args.eval_config, args.cov, args.n_windows, args.seed)
        print(f"[fp] model=tirex2 device={args.device} windows={len(items)} "
              f"Lc={Lc} H={H} dt={dt}s bs={args.batch_size}", flush=True)
        load_s, infer_s, n, peak, params = time_tirex2(items, H, args.batch_size, args.device)
        kind = "zero-shot foundation (covariate-aware)"
    else:
        win, cfg, clin, L, Lc, H, dt, past_names, fut_names = build_contexts(
            args.match_tirex, args.config, args.eval_config, args.cov, args.n_windows, args.seed)
        print(f"[fp] model={args.model} device={args.device} windows={len(win)} "
              f"Lc={Lc} H={H} dt={dt}s bs={args.batch_size}", flush=True)
        if args.model in ("chronos", "timesfm", "moirai"):
            load_s, infer_s, n, peak, params = time_foundation(args.model, win, Lc, H, args.batch_size, args.device)
            kind = "zero-shot foundation"
        elif args.model in ("tft", "patchtst"):
            if not args.ckpt:
                raise SystemExit(f"--ckpt required for {args.model}")
            load_s, infer_s, n, peak, params = time_baseline(args.model, args.ckpt, win, args.device)
            kind = "task-trained"

    row = {
        "model": args.model, "kind": kind, "device": args.device,
        "n_windows": n, "batch_size": args.batch_size,
        "params_M": None if params != params else round(params / 1e6, 2),   # NaN-safe
        "load_time_s": round(load_s, 2),
        "infer_total_s": round(infer_s, 3),
        "latency_ms_per_window": round(1000 * infer_s / max(n, 1), 3),
        "throughput_windows_per_s": round(n / infer_s, 1) if infer_s > 0 else None,
        "peak_gpu_mem_MB": None if peak != peak else round(peak, 1),
        "context_len": Lc, "horizon": H, "dt_s": dt,
    }
    print("[fp] " + json.dumps(row, indent=1), flush=True)

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    allrows = {}
    if os.path.exists(args.out):
        try:
            allrows = json.load(open(args.out))
        except Exception:
            allrows = {}
    allrows[args.model] = row
    json.dump(allrows, open(args.out, "w"), indent=1)
    print(f"[fp] wrote {args.out} (model '{args.model}')", flush=True)


if __name__ == "__main__":
    main()
