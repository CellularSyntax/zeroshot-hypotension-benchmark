"""Extract per-window encoder embeddings from each zero-shot foundation model.

Representation-explainability layer (supplementary). For a fixed, model-shared, stratified
SUBSAMPLE of forecast windows (identical origins across models), we capture each model's ENCODER
hidden state at the forecast origin and mean-pool it to one vector per window. This is a
representation of the recent hemodynamic trajectory (the model's view at decision time), NOT a
patient-level disease encoding -- these are time-series forecasters. Downstream we UMAP-visualise,
linear-probe, and RSA-compare these vectors (scripts/explain_representations.py).

We subsample because (i) RSA builds an N x N dissimilarity matrix (all ~285k windows is intractable
and unnecessary) and (ii) a few thousand windows give stable UMAP/probe/RSA estimates. Subsampling
is seeded and stratified by (stratum x 5-min hypotension label) so the sample is balanced, and the
SAME (caseid,t0) set is used for every model so the four embeddings are row-aligned for RSA.

Each architecture exposes its embedding differently, so we hook the encoder trunk and mean-pool.
Every model is driven in explicit fixed-size batches (bs windows per forward), so accumulate-then-
concatenate reproduces window order exactly -- including Moirai, whose GluonTS predictor otherwise
hides batch boundaries (we feed it one sub-ListDataset per batch). The hook path actually used is
recorded in the meta JSON for provenance.

Writes results/embeddings_<model>_<tag>.npz: emb [N,D] float32, caseid [N], t0 [N] int,
stratum [N] str, hypo_event_5 [N] int, t_event_65 [N] float, row-aligned across models.

Run (GPU) from project root, one model at a time:
    PYTHONPATH=scripts:datasets/vitaldb python scripts/baselines/extract_embeddings.py \
        --model chronos --match-tirex results/ablation_windows_all2873.csv --device cuda
"""
from __future__ import annotations
import argparse, csv, json, os, time
import numpy as np


def _pool(t):
    """Mean-pool a hidden-state tensor to [B, D]. Accepts [B,S,D] or [B,D], torch or numpy."""
    import torch
    if isinstance(t, (list, tuple)):
        t = t[0]
    if not torch.is_tensor(t):
        t = torch.as_tensor(t)
    t = t.detach().float().cpu()
    if t.dim() == 3:
        return t.mean(dim=1).numpy()
    if t.dim() == 2:
        return t.numpy()
    return t.reshape(t.shape[0], -1).numpy()


class HookCapture:
    """Forward hook that ACCUMULATES each forward's pooled output (one entry per batch)."""
    def __init__(self):
        self.batches = []
        self.handle = None
        self.path = None

    def attach(self, module, path):
        def hook(_m, _inp, out):
            self.batches.append(_pool(out))
        self.handle = module.register_forward_hook(hook)
        self.path = path
        return self

    def reset(self):
        self.batches = []

    def stack(self):
        return np.concatenate(self.batches, axis=0) if self.batches else np.empty((0, 0))

    def detach(self):
        if self.handle is not None:
            self.handle.remove()


def _largest_encoderish(root):
    """Fallback: the encoder/block container with the most parameters."""
    import torch.nn as nn
    best, best_n, best_name = None, -1, None
    for name, m in root.named_modules():
        n = sum(p.numel() for p in m.parameters())
        if n <= 0:
            continue
        if isinstance(m, nn.ModuleList) or "encoder" in name.lower() or "block" in name.lower():
            if n > best_n:
                best, best_n, best_name = m, n, name
    return best, best_name


def _find_encoder(root, suffixes=("encoder",)):
    for name, m in root.named_modules():
        if any(name.endswith(s) for s in suffixes):
            return m, name
    return _largest_encoderish(root)


# ---- per-model runners: driven in fixed-size batches; return emb[N,D], hook_path ----

def embed_tirex2(win, Lc, H, bs, device):
    import phase3_ablation as P
    from tirex2 import load_model
    model = load_model("NX-AI/TiRex-2", device=device)
    core = getattr(model, "model", model)
    target, path = None, None
    for name, m in core.named_modules():
        if name.endswith("blocks") or "bi_xlstm" in name.lower():
            target, path = m, name
    if target is None:
        target, path = _largest_encoderish(core)
    cap = HookCapture().attach(target, path or "tirex2:fallback")
    items = [P.build_ts(w["_rec"], w["t0"], Lc, H, True, True) for w in win]
    for i in range(0, len(items), bs):
        model.forecast(items[i:i+bs], prediction_length=H, output_type="numpy")
    cap.detach()
    return cap.stack(), cap.path


def embed_chronos(win, Lc, H, bs, device):
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base",
                                               device_map=device, torch_dtype=torch.float32)
    inner = getattr(pipe, "model", pipe)
    target, path = _find_encoder(inner)
    cap = HookCapture().attach(target, path or "chronos:fallback")
    contexts = [w["past"][:, 0].astype(np.float32) for w in win]
    for i in range(0, len(contexts), bs):
        ctx = [torch.tensor(c) for c in contexts[i:i+bs]]
        pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=[0.1, 0.5, 0.9])
    cap.detach()
    return cap.stack(), cap.path


def embed_timesfm(win, Lc, H, bs, device):
    import timesfm
    try:
        Cls = timesfm.TimesFM_2p5_200M_torch
    except AttributeError:
        from timesfm.timesfm_2p5.timesfm_2p5_torch import TimesFM_2p5_200M_torch as Cls
    model = Cls.from_pretrained("google/timesfm-2.5-200m-pytorch")
    max_ctx = max(512, (Lc + 31) // 32 * 32)
    model.compile(timesfm.ForecastConfig(max_context=max_ctx, max_horizon=max(320, H),
                                         normalize_inputs=True, use_continuous_quantile_head=True))
    core = getattr(model, "model", model)
    target, path = _largest_encoderish(core)
    cap = HookCapture().attach(target, path or "timesfm:fallback")
    contexts = [np.asarray(w["past"][:, 0], dtype=np.float32) for w in win]
    for i in range(0, len(contexts), bs):
        model.forecast(horizon=H, inputs=contexts[i:i+bs])
    cap.detach()
    return cap.stack(), cap.path


def embed_moirai(win, Lc, H, bs, device):
    import pandas as pd
    from gluonts.dataset.common import ListDataset
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    module = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-large")
    model = MoiraiForecast(module=module, prediction_length=H, context_length=Lc or 512,
                           patch_size="auto", num_samples=100, target_dim=1,
                           feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
    target, path = _find_encoder(module)
    cap = HookCapture().attach(target, path or "moirai:fallback")
    try:
        predictor = model.create_predictor(batch_size=bs, device=device)
    except TypeError:
        predictor = model.to(device).create_predictor(batch_size=bs)
    contexts = [w["past"][:, 0].astype(np.float32) for w in win]
    # Feed ONE fixed-size batch at a time as its own ListDataset, and reduce that batch's
    # (possibly multiple) forward-fires to a single pooled vector per window by averaging the
    # per-forward captures for the block. Because the sampler may fire the encoder once per batch,
    # we exhaust the predictor for each chunk and take the mean over fires, giving [chunk, D].
    embs = []
    for i in range(0, len(contexts), bs):
        chunk = contexts[i:i+bs]
        ds = ListDataset([{"target": c, "start": pd.Period("2020-01-01", freq="s")} for c in chunk],
                         freq="s")
        cap.reset()
        _ = list(predictor.predict(ds))          # exhaust -> fires the hook for this chunk
        b = cap.batches
        if not b:
            raise RuntimeError("moirai: encoder hook never fired; check hook path")
        # concatenate fires; if the encoder fired once, this is [chunk,D]; if multiple fires stacked
        # the same windows (sampling), rows > chunk -> average the fires back to [chunk,D]
        stacked = np.concatenate(b, axis=0)
        if stacked.shape[0] == len(chunk):
            embs.append(stacked)
        elif stacked.shape[0] % len(chunk) == 0:
            k = stacked.shape[0] // len(chunk)
            embs.append(stacked.reshape(k, len(chunk), -1).mean(axis=0))
        else:
            raise RuntimeError(f"moirai: {stacked.shape[0]} pooled rows for chunk {len(chunk)}")
    cap.detach()
    return np.concatenate(embs, axis=0), cap.path


RUNNERS = {"tirex2": embed_tirex2, "chronos": embed_chronos,
           "timesfm": embed_timesfm, "moirai": embed_moirai}


def stratified_subsample(win, max_windows, seed):
    """Seeded stratified subsample by (stratum, 5-min hypo label), preserving proportions."""
    if len(win) <= max_windows:
        return list(range(len(win)))
    rng = np.random.default_rng(seed)
    keys = {}
    for i, w in enumerate(win):
        keys.setdefault((w["stratum"], int(w["_hypo5"])), []).append(i)
    idx = []
    for k, members in keys.items():
        take = max(1, round(max_windows * len(members) / len(win)))
        members = np.array(members)
        idx.extend(members[rng.permutation(len(members))[:take]].tolist())
    rng.shuffle(idx)
    return sorted(idx[:max_windows])


def main():
    import phase3_ablation as P
    from baselines import data as D
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="datasets/vitaldb/configs/data.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--model", required=True, choices=list(RUNNERS))
    ap.add_argument("--cov", default="ce", choices=list(P.COV_PRESETS))
    ap.add_argument("--match-tirex", required=True,
                    help="a TiRex ablation_windows_*.csv; locks cohort + window origins")
    ap.add_argument("--max-windows", type=int, default=4000,
                    help="stratified subsample size (RSA is O(N^2); 4000 is ample)")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-origins", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    import yaml
    ev = yaml.safe_load(open(args.eval_config))
    L = P.get_loader(args.config)
    cfg = L.load_config(args.config); clin = L._clinical_index(cfg["clinical_csv"])
    preset = P.COV_PRESETS[args.cov]
    P.FUTURE_COV = list(preset["future"]); P.PRIMARY_COV = preset["primary"]; P.TRANSITION_THR = preset["trans_thr"]

    cases = sorted({r["caseid"] for r in csv.DictReader(open(args.match_tirex))})
    stem = os.path.basename(args.match_tirex).replace("ablation_windows_", "").replace(".csv", "")
    tag = args.tag or f"{args.model}_{stem}"

    probe = next((r for r in (L.load_case(c, cfg, clin) for c in cases) if r is not None), None)
    dt = probe["interval_s"]; Lc = int(ev["context_min"] * 60 / dt)
    hsteps = [int(m * 60 / dt) for m in P.HORIZON_STEPS_MIN]; H = max(hsteps)
    stride = int(ev["origin_stride_min"] * 60 / dt); warmup = int(ev["warmup_min"] * 60 / dt)
    min_run = max(1, int(ev.get("hypotension", {}).get("min_sustain_min", 1) * 60 / dt))
    h5 = int(5 * 60 / dt)

    t0 = time.time()
    win, _, _ = D.build_windows(cases, cfg, clin, Lc, H, stride, warmup, args.max_origins, dt, min_run,
                                quiet=args.quiet)
    for w in win:
        w["_hypo5"] = P.hypo_event(w["truth"][:h5], min_run, P.HYPO_THR)
    # seeded stratified subsample -- IDENTICAL across models (same cases/origins/seed -> same win order)
    keep = stratified_subsample(win, args.max_windows, args.seed)
    win = [win[i] for i in keep]
    print(f"[emb] model={args.model} kept {len(win)} of subsample windows Lc={Lc} H={H} dt={dt} tag={tag}",
          flush=True)

    if args.model == "tirex2":
        recs = {}
        for w in win:
            if w["caseid"] not in recs:
                recs[w["caseid"]] = L.load_case(w["caseid"], cfg, clin)
            w["_rec"] = recs[w["caseid"]]

    emb, path = RUNNERS[args.model](win, Lc, H, args.batch_size, args.device)
    assert emb.shape[0] == len(win), f"emb rows {emb.shape[0]} != windows {len(win)}"
    print(f"[emb] captured {emb.shape} via '{path}' in {time.time()-t0:.0f}s", flush=True)

    caseid = np.array([w["caseid"] for w in win])
    t0arr = np.array([w["t0"] for w in win], dtype=np.int64)
    stratum = np.array([w["stratum"] for w in win])
    hypo5 = np.array([w["_hypo5"] for w in win], dtype=np.int64)
    tev = np.array([w["t_event_65"] if w["t_event_65"] is not None else np.nan for w in win],
                   dtype=np.float64)

    os.makedirs("results", exist_ok=True)
    out = f"results/embeddings_{tag}.npz"
    np.savez_compressed(out, emb=emb.astype(np.float32), caseid=caseid, t0=t0arr,
                        stratum=stratum, hypo_event_5=hypo5, t_event_65=tev)
    json.dump({"tag": tag, "model": args.model, "hook_path": path, "n_windows": int(len(win)),
               "emb_dim": int(emb.shape[1]), "context_min": ev["context_min"], "cov": args.cov,
               "max_windows": args.max_windows, "seed": args.seed,
               "match_tirex": os.path.basename(args.match_tirex)},
              open(f"results/embeddings_meta_{tag}.json", "w"), indent=1)
    print(f"[emb] wrote {out}  emb_dim={emb.shape[1]}", flush=True)


if __name__ == "__main__":
    main()
