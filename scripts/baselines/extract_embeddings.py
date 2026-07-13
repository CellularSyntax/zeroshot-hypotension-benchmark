"""Extract per-window encoder embeddings from each zero-shot foundation model.

Representation-explainability layer (supplementary). For a fixed, model-shared, stratified
SUBSAMPLE of forecast windows (identical origins across models), we capture each model's internal
hidden state at the forecast origin and mean-pool it to one vector per window. This is a
representation of the recent hemodynamic trajectory (the model's view at decision time), NOT a
patient-level disease encoding -- these are time-series forecasters. Downstream we UMAP-visualise,
linear-probe, and RSA-compare these vectors (scripts/explain_representations.py).

We subsample because RSA builds an N x N dissimilarity matrix (all ~285k windows is intractable
and unnecessary); a few thousand windows give stable UMAP/probe/RSA estimates. Subsampling is
seeded and stratified by (stratum x 5-min hypotension label), and the SAME (caseid,t0) set is used
for every model so the four embeddings are row-aligned for RSA.

AUTODISCOVERY of the embedding module (why -- the four architectures name their internals
differently and can't be introspected off the GPU): on the FIRST batch we attach lightweight
probe hooks to every parameterised submodule, record which ones emit a poolable hidden state
([B,S,D] or [B,D] float, 16<=D<=4096), and pick the best encoder-like module by a name+depth
heuristic. We then attach a single capture hook to that module and run all batches. Hidden states
are unwrapped from HuggingFace ModelOutput objects / dicts / tuples. The chosen module path and
its dim are written to the meta JSON for provenance -- verify these look like real encoder outputs
before trusting the analyses.

Every model is driven in explicit fixed-size batches so accumulate-then-concatenate reproduces
window order exactly, including Moirai (one sub-ListDataset per batch).

Writes results/embeddings_<model>_<tag>.npz: emb [N,D] float32, caseid [N], t0 [N] int,
stratum [N] str, hypo_event_5 [N] int, t_event_65 [N] float, row-aligned across models.

Run (GPU) from project root, one model at a time -- see slurm/extract_embeddings.sbatch.
"""
from __future__ import annotations
import argparse, csv, json, os, time
import numpy as np


def _extract_hidden(out):
    """Unwrap a module output to a float hidden-state tensor, or None."""
    import torch
    if torch.is_tensor(out):
        return out
    for attr in ("last_hidden_state", "hidden_states"):
        v = getattr(out, attr, None)
        if torch.is_tensor(v):
            return v
        if isinstance(v, (list, tuple)) and v and torch.is_tensor(v[-1]):
            return v[-1]
    if isinstance(out, dict):
        for k in ("last_hidden_state", "hidden_states"):
            v = out.get(k)
            if torch.is_tensor(v):
                return v
    if isinstance(out, (list, tuple)):
        for v in out:
            if torch.is_tensor(v) and v.dim() >= 2 and v.is_floating_point():
                return v
    return None


def _pool(h):
    """Mean-pool a hidden-state tensor [B,S,D] or [B,D] to [B,D] numpy."""
    h = h.detach().float().cpu()
    if h.dim() == 3:
        return h.mean(dim=1).numpy()
    if h.dim() == 2:
        return h.numpy()
    return h.reshape(h.shape[0], -1).numpy()


def _poolable(h, min_dim=16, max_dim=4096):
    return (h is not None and h.is_floating_point() and h.dim() in (2, 3)
            and min_dim <= h.shape[-1] <= max_dim)


_BAD_NAME = ("embed", "token", "patch", "input", "rotary", "pos_enc", "positional")


def _choose_target(clean):
    """clean: name -> dict(D, order, fires). Pick the module whose pooled output best represents
    the encoder's final hidden state. Priority: (1) a stack literally named 'encoder'/'encoders'
    (widest, shallowest); (2) else the LATEST-executing candidate whose name is not an input/
    embedding/tokenizer layer (= the last transformer/xLSTM block output before the head);
    (3) else the latest candidate. Returns the chosen name or None."""
    if not clean:
        return None
    enc = {n: v for n, v in clean.items() if n.split(".")[-1] in ("encoder", "encoders")}
    if enc:
        return max(enc, key=lambda n: (clean[n]["D"], -n.count(".")))
    good = {n: v for n, v in clean.items() if not any(b in n.lower() for b in _BAD_NAME)}
    pool = good or clean
    return max(pool, key=lambda n: clean[n]["order"])


class _Capture:
    """Single-module capture. Collects one pooled array per forward-fire; the caller reduces the
    per-batch fires (a module inside a layer/sample loop fires k times per batch -> we average)."""
    def __init__(self):
        self.batches = []; self.handle = None; self.path = None
    def attach(self, module, path):
        def hook(_m, _i, out):
            h = _extract_hidden(out)
            if h is not None:
                self.batches.append(_pool(h))
        self.handle = module.register_forward_hook(hook); self.path = path; return self
    def detach(self):
        if self.handle:
            self.handle.remove()


def run_capture(root, run_batch, n, bs):
    """Discover the embedding module on batch 0, then capture it across all n windows.
    run_batch(lo, hi) triggers a forward over windows [lo:hi]. Returns (emb[N,D], path).

    Discovery probes every PARAMETERISED module (including containers such as the encoder stack,
    whose params live in children -- the earlier recurse=False filter wrongly excluded them). A
    candidate is 'clean' if every one of its fires emitted exactly `expected` rows (one full
    batch); k = number of fires (k>1 = a module inside a per-layer/per-sample loop). The capture
    pass averages the k fires per batch so multi-fire modules (moirai, tirex2) yield [b, D]."""
    import torch
    expected = min(bs, n)
    stats, order = {}, [0]
    handles = []
    def mk_probe(name):
        def probe(_m, _i, out):
            h = _extract_hidden(out)
            if not _poolable(h):
                return
            rows = _pool(h).shape[0]
            s = stats.setdefault(name, dict(D=int(h.shape[-1]), order=order[0], fire_rows=[]))
            s["fire_rows"].append(rows)
            order[0] += 1
        return probe
    for name, m in root.named_modules():
        if name == "" or not any(True for _ in m.parameters()):   # keep containers w/ child params
            continue
        handles.append(m.register_forward_hook(mk_probe(name)))
    run_batch(0, expected)
    for h in handles:
        h.remove()
    clean = {n_: dict(D=s["D"], order=s["order"], fires=len(s["fire_rows"]))
             for n_, s in stats.items()
             if s["fire_rows"] and all(r == expected for r in s["fire_rows"])}
    target = _choose_target(clean)
    if target is None:
        raise RuntimeError(f"autodiscovery found no clean single-batch hidden state "
                           f"(probed {len(stats)} modules, expected rows/fire={expected})")
    k = clean[target]["fires"]
    module = dict(root.named_modules())[target]
    print(f"[emb] discovered {len(clean)} clean modules ({len(stats)} probed); "
          f"target='{target}' dim={clean[target]['D']} fires/batch={k}", flush=True)
    # ---- capture pass: per-batch, averaging the k fires the chosen module makes ----
    cap = _Capture().attach(module, f"{target}[D={clean[target]['D']},k={k}]")
    embs = []
    for lo in range(0, n, bs):
        b = min(lo + bs, n) - lo
        cap.batches = []
        run_batch(lo, lo + b)
        rows = np.concatenate(cap.batches, axis=0) if cap.batches else np.empty((0, clean[target]["D"]))
        if rows.shape[0] == b:
            embs.append(rows)
        elif rows.shape[0] % b == 0:                 # k fires of the full batch -> average them
            embs.append(rows.reshape(rows.shape[0] // b, b, -1).mean(axis=0))
        else:
            cap.detach()
            raise RuntimeError(f"capture: {rows.shape[0]} rows for batch of {b} (not a multiple)")
    cap.detach()
    return np.concatenate(embs, axis=0), cap.path


# ---- per-model runners: build (root, run_batch) and delegate to run_capture ----

def embed_tirex2(win, Lc, H, bs, device):
    import phase3_ablation as P
    from tirex2 import load_model
    model = load_model("NX-AI/TiRex-2", device=device)
    root = getattr(model, "model", model)
    items = [P.build_ts(w["_rec"], w["t0"], Lc, H, True, True) for w in win]
    def run_batch(lo, hi):
        model.forecast(items[lo:hi], prediction_length=H, output_type="numpy")
    return run_capture(root, run_batch, len(win), bs)


def embed_chronos(win, Lc, H, bs, device):
    import torch
    from chronos import BaseChronosPipeline
    pipe = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-base",
                                               device_map=device, torch_dtype=torch.float32)
    root = getattr(pipe, "model", pipe)
    ctx = [w["past"][:, 0].astype(np.float32) for w in win]
    def run_batch(lo, hi):
        pipe.predict_quantiles([torch.tensor(c) for c in ctx[lo:hi]],
                               prediction_length=H, quantile_levels=[0.1, 0.5, 0.9])
    return run_capture(root, run_batch, len(win), bs)


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
    root = getattr(model, "model", model)
    ctx = [np.asarray(w["past"][:, 0], dtype=np.float32) for w in win]
    def run_batch(lo, hi):
        model.forecast(horizon=H, inputs=ctx[lo:hi])
    return run_capture(root, run_batch, len(win), bs)


def embed_moirai(win, Lc, H, bs, device):
    import pandas as pd
    from gluonts.dataset.common import ListDataset
    from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
    module = MoiraiModule.from_pretrained("Salesforce/moirai-1.1-R-large")
    model = MoiraiForecast(module=module, prediction_length=H, context_length=Lc or 512,
                           patch_size="auto", num_samples=100, target_dim=1,
                           feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
    try:
        predictor = model.create_predictor(batch_size=bs, device=device)
    except TypeError:
        predictor = model.to(device).create_predictor(batch_size=bs)
    ctx = [w["past"][:, 0].astype(np.float32) for w in win]
    def run_batch(lo, hi):
        ds = ListDataset([{"target": c, "start": pd.Period("2020-01-01", freq="s")}
                          for c in ctx[lo:hi]], freq="s")
        list(predictor.predict(ds))
    # hook the MoiraiModule (the nn.Module the predictor calls under the hood)
    return run_capture(module, run_batch, len(win), bs)


RUNNERS = {"tirex2": embed_tirex2, "chronos": embed_chronos,
           "timesfm": embed_timesfm, "moirai": embed_moirai}


def stratified_subsample(win, max_windows, seed):
    if len(win) <= max_windows:
        return list(range(len(win)))
    rng = np.random.default_rng(seed)
    keys = {}
    for i, w in enumerate(win):
        keys.setdefault((w["stratum"], int(w["_hypo5"])), []).append(i)
    idx = []
    for members in keys.values():
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
    ap.add_argument("--match-tirex", required=True)
    ap.add_argument("--max-windows", type=int, default=4000)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--max-origins", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    if str(args.device).startswith("cuda"):
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but torch.cuda.is_available()==False. The enroot NVIDIA hook "
                "injected no GPU -- almost always NVIDIA_VISIBLE_DEVICES=void inherited from the "
                "login shell (a CPU-srun workaround). Fix: 'unset NVIDIA_VISIBLE_DEVICES "
                "NVIDIA_DRIVER_CAPABILITIES' before sbatch (the sbatch now also forces =all).")

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
    keep = stratified_subsample(win, args.max_windows, args.seed)
    win = [win[i] for i in keep]
    print(f"[emb] model={args.model} kept {len(win)} subsample windows Lc={Lc} H={H} dt={dt} tag={tag}",
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
