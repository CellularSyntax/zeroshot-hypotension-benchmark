"""Zero-shot foundation-model sweep — other TSFMs as covariate-blind foils for TiRex-2.

Runs a chosen pretrained time-series foundation model (Chronos-Bolt, TimesFM, Moirai, ...)
ZERO-SHOT on the SAME windows / subject-split / metrics as the TiRex-2 evaluation and the
trained baselines, and writes per-window quantile forecasts in the exact phase3 schema
(tagged `baseline-<model>_<stem>`). So `compare.py`, `hypo_eval.py`, and `paper_figures.py`
pick these up with NO changes — the model just becomes another row in the matched head-to-head.

Scientific point: unlike TiRex-2, these models do not natively ingest the known future
drug-infusion trajectory. We therefore benchmark them as *covariate-blind* zero-shot foils —
one forecast per window, reused for the M1 and M0 columns (so their covariate effect X% is 0
by construction, which is exactly the finding: generic TSFMs can't exploit the drug plan).
Covariate-aware adapters can set `supports_future = True` and consume the future block.

Run (cluster, inside the .tsfm-venv):
  PYTHONPATH=scripts:datasets/vitaldb python scripts/baselines/zeroshot.py \
      --model chronos --match-tirex results/ablation_windows_all2873.csv --device cuda
"""
from __future__ import annotations
import argparse, csv, json, os, time
import numpy as np

import phase3_ablation as P
import vitaldb_loader as L
from baselines import data as D
from baselines.splits import subject_split

QLEVELS = list(map(float, P.QLEVELS))          # [0.1 .. 0.9]
MED = P.MED


# --------------------------------------------------------------------------------------
# Model adapters. Each returns, per window, a [9, H] array of quantile forecasts at QLEVELS
# (in mmHg). Foundation models self-normalise, so we feed the raw MAP context directly.
# `supports_future=False` ⇒ univariate: the runner calls forecast() once and reuses it for
# both the M1 and M0 arms.
# --------------------------------------------------------------------------------------
class Adapter:
    supports_future = False
    name = "base"
    def load(self, device):                    # noqa: D401
        raise NotImplementedError
    def forecast(self, contexts, futures, H):  # contexts: list[1D mmHg]; futures: list|None
        raise NotImplementedError


class ChronosAdapter(Adapter):
    """Chronos-Bolt (Ansari et al., 2024) — T5 encoder–decoder, direct multi-step quantiles.
    Apache-2.0, ungated. Univariate. pip: chronos-forecasting."""
    name = "chronos"
    def __init__(self, repo="amazon/chronos-bolt-base"):
        self.repo = repo
    def load(self, device):
        import torch
        from chronos import BaseChronosPipeline
        self.torch = torch
        dt = torch.bfloat16 if str(device).startswith("cuda") else torch.float32
        self.pipe = BaseChronosPipeline.from_pretrained(self.repo, device_map=device, torch_dtype=dt)
        return self
    def forecast(self, contexts, futures, H):
        # pass the context positionally — the param is named `context` in some versions and
        # `inputs` in others; positional binds either. Returns (quantiles[B,H,Q], mean[B,H]).
        ctx = [self.torch.tensor(np.asarray(c, dtype=np.float32)) for c in contexts]
        q, _ = self.pipe.predict_quantiles(ctx, prediction_length=H, quantile_levels=QLEVELS)
        q = q.float().cpu().numpy()             # [B, H, 9]
        return [np.sort(q[i].T, axis=0) for i in range(q.shape[0])]   # each [9, H], monotone


class TimesFMAdapter(Adapter):
    """TimesFM 2.0 (Das et al., 2024) — decoder-only patched TSFM. Apache-2.0, ungated.
    Univariate; 10 experimental quantile heads (col 0 = mean, cols 1..9 = deciles 0.1..0.9).
    pip: timesfm[torch]."""
    name = "timesfm"
    def __init__(self, repo="google/timesfm-2.0-500m-pytorch", context_len=None):
        self.repo = repo; self.context_len = context_len
    def load(self, device):
        import timesfm
        backend = "gpu" if str(device).startswith("cuda") else "cpu"
        cl = self.context_len or 512
        cl = (cl // 32) * 32 or 32               # TimesFM context must be a multiple of 32
        self.model = timesfm.TimesFm(
            hparams=timesfm.TimesFmHparams(backend=backend, per_core_batch_size=64,
                                           horizon_len=self._H_hint or 320, context_len=cl,
                                           num_layers=50, use_positional_embedding=False),
            checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=self.repo))
        return self
    _H_hint = None
    def forecast(self, contexts, futures, H):
        inputs = [np.asarray(c, dtype=np.float32) for c in contexts]
        _, qf = self.model.forecast(inputs, freq=[0] * len(inputs))   # qf: [B, H, 10]
        qf = np.asarray(qf)[:, :H, 1:10]        # drop mean col -> [B, H, 9] deciles
        return [np.sort(qf[i].T, axis=0) for i in range(qf.shape[0])]  # each [9, H]


class MoiraiAdapter(Adapter):
    """Moirai-1.1-R (Woo et al., 2024) — masked-encoder any-variate TSFM. CC-BY-NC-4.0
    (research use OK). Run univariate here (sample forecasts -> empirical quantiles).
    pip: uni2ts. NOTE: verify the per-window path on the cluster before a full sweep."""
    name = "moirai"
    def __init__(self, repo="Salesforce/moirai-1.1-R-large", context_len=None, num_samples=100):
        self.repo = repo; self.context_len = context_len; self.num_samples = num_samples
    def load(self, device):
        import torch
        from uni2ts.model.moirai import MoiraiForecast, MoiraiModule
        self.torch = torch
        self.module = MoiraiModule.from_pretrained(self.repo)
        self._device = device
        return self
    def _predictor(self, H):
        from uni2ts.model.moirai import MoiraiForecast
        m = MoiraiForecast(module=self.module, prediction_length=H,
                           context_length=self.context_len or 512, patch_size="auto",
                           num_samples=self.num_samples, target_dim=1,
                           feat_dynamic_real_dim=0, past_feat_dynamic_real_dim=0)
        return m.to(self._device).eval()
    def forecast(self, contexts, futures, H):
        # Batched sample forecast via the MoiraiForecast module (GluonTS-free tensor path).
        model = self._predictor(H)
        Lc = len(contexts[0])
        past = self.torch.tensor(np.stack(contexts).astype(np.float32)).unsqueeze(-1)  # [B,Lc,1]
        past = past.to(self._device)
        obs = self.torch.ones_like(past, dtype=self.torch.bool)
        pad = self.torch.zeros(past.shape[:2], dtype=self.torch.bool, device=self._device)
        with self.torch.no_grad():
            samples = model(past_target=past, past_observed_target=obs, past_is_pad=pad)  # [B,S,H,1]
        s = samples.float().cpu().numpy()[..., 0]                                          # [B,S,H]
        return [np.quantile(s[i], QLEVELS, axis=0) for i in range(s.shape[0])]             # each [9,H]


ADAPTERS = {"chronos": ChronosAdapter, "timesfm": TimesFMAdapter, "moirai": MoiraiAdapter}


def build_adapter(name, context_len, H):
    if name not in ADAPTERS:
        raise ValueError(f"unknown model '{name}'; have {list(ADAPTERS)}")
    a = ADAPTERS[name](context_len=context_len) if name != "chronos" else ADAPTERS[name]()
    a._H_hint = H
    return a


def batched(adapter, contexts, futures, H, bs):
    out = []
    for i in range(0, len(contexts), bs):
        f = None if futures is None else futures[i:i + bs]
        out.extend(adapter.forecast(contexts[i:i + bs], f, H))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="datasets/vitaldb/configs/data.yaml")
    ap.add_argument("--eval-config", default="configs/eval.yaml")
    ap.add_argument("--model", required=True, choices=list(ADAPTERS))
    ap.add_argument("--cov", default="ce", choices=list(P.COV_PRESETS),
                    help="covariate preset (selects window anchor + future channels; univariate "
                         "models ignore the covariate but the cohort/strata still match TiRex).")
    ap.add_argument("--match-tirex", required=True,
                    help="a TiRex ablation_windows_*.csv; locks the cohort + tag stem so the "
                         "windows and canonical test split are identical to the matched comparison.")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-origins", type=int, default=20)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--tag", default=None)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    import yaml
    ev = yaml.safe_load(open(args.eval_config))
    cfg = L.load_config(args.config); clin = L._clinical_index(cfg["clinical_csv"])
    preset = P.COV_PRESETS[args.cov]
    P.FUTURE_COV = list(preset["future"]); P.PRIMARY_COV = preset["primary"]; P.TRANSITION_THR = preset["trans_thr"]

    cases = sorted({r["caseid"] for r in csv.DictReader(open(args.match_tirex))})
    stem = os.path.basename(args.match_tirex).replace("ablation_windows_", "").replace(".csv", "")
    tag = args.tag or f"baseline-{args.model}_{stem}"

    probe = next((r for r in (L.load_case(c, cfg, clin) for c in cases) if r is not None), None)
    dt = probe["interval_s"]; Lc = int(ev["context_min"] * 60 / dt)
    hsteps = [int(m * 60 / dt) for m in P.HORIZON_STEPS_MIN]; H = max(hsteps)
    stride = int(ev["origin_stride_min"] * 60 / dt); warmup = int(ev["warmup_min"] * 60 / dt)
    min_run = max(1, int(ev.get("hypotension", {}).get("min_sustain_min", 1) * 60 / dt))

    # canonical 60/20/20 subject split -> evaluate on the SAME held-out test cases as the baselines
    from baselines.train import caseid_to_subject
    c2s = caseid_to_subject(cfg["clinical_csv"])
    split = subject_split(cases, c2s, seed=args.seed)
    test_cases = [c for c in cases if split[c] == "test"]
    print(f"[zs] model={args.model} cov={args.cov} dt={dt} Lc={Lc} H={H} "
          f"cases={len(cases)} test={len(test_cases)} tag={tag} device={args.device}", flush=True)

    t0 = time.time()
    win, past_names, fut_names = D.build_windows(test_cases, cfg, clin, Lc, H, stride, warmup,
                                                 args.max_origins, dt, min_run, quiet=args.quiet)
    print(f"[zs] built {len(win)} test windows ({time.time()-t0:.0f}s)", flush=True)

    adapter = build_adapter(args.model, Lc, H).load(args.device)
    print(f"[zs] loaded {args.model}; forecasting ...", flush=True)

    contexts = [w["past"][:, 0].astype(np.float32) for w in win]     # MAP channel (finite-filled)
    tf = time.time()
    if adapter.supports_future:
        futures = [w["future"].astype(np.float32) for w in win]
        q_M1 = batched(adapter, contexts, futures, H, args.batch_size)
        q_M0 = batched(adapter, contexts, [None] * len(contexts), H, args.batch_size)
    else:
        q = batched(adapter, contexts, None, H, args.batch_size)
        q_M1 = q_M0 = q                                              # covariate-blind: M1 == M0
    print(f"[zs] forecast done in {time.time()-tf:.0f}s", flush=True)

    # ---- write per-window rows in the phase3 schema (identical to the trained baselines) ----
    os.makedirs("results", exist_ok=True)
    cols = ["caseid", "t0", "h_min", "stratum", "t_event_65",
            "crps_M1", "mae_M1", "mae_inst_M1", "crps_M0", "mae_M0", "mae_inst_M0",
            "crps_M1_to", "mae_M1_to", "mae_inst_M1_to", "crps_M0_to", "mae_M0_to", "mae_inst_M0_to",
            "crps_persist", "hypo_event", "risk_M1", "risk_M0",
            "hypo_event_55", "risk_M1_55", "risk_M0_55", "hypo_event_50", "risk_M1_50", "risk_M0_50", "split"]
    path = f"results/ablation_windows_{tag}.csv"; n_rows = 0
    with open(path, "w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=cols); wr.writeheader()
        for wi, w in enumerate(win):
            truth = w["truth"]; q = {"M1": q_M1[wi], "M0": q_M0[wi]}     # each [9, H]
            pastMAP = w["past"][:, 0]; fin = pastMAP[np.isfinite(pastMAP)]
            plast = fin[-1] if len(fin) else np.nan
            for h in hsteps:
                tr = truth[:h]; hm = round(h * dt / 60)
                row = {"caseid": w["caseid"], "t0": w["t0"], "h_min": hm, "stratum": w["stratum"],
                       "t_event_65": w["t_event_65"], "split": "test"}
                for c in ("M1", "M0"):
                    cr, ma = P.pinball(tr, q[c][:, :h]); row[f"crps_{c}"] = cr; row[f"mae_{c}"] = ma
                    yl = tr[-1]
                    row[f"mae_inst_{c}"] = float(abs(q[c][MED, h - 1] - yl)) if np.isfinite(yl) else np.nan
                    row[f"crps_{c}_to"] = cr; row[f"mae_{c}_to"] = ma       # no target-only arm
                    row[f"mae_inst_{c}_to"] = row[f"mae_inst_{c}"]
                row["crps_persist"] = float(np.nanmean(np.abs(plast - tr))) if np.isfinite(tr).any() else np.nan
                for thr in P.HYPO_THRS:
                    tk = "" if thr == P.HYPO_THR else f"_{int(thr)}"
                    row[f"hypo_event{tk}"] = P.hypo_event(tr, min_run, thr)
                    row[f"risk_M1{tk}"] = P.hypo_risk(q["M1"][:, :h], thr)
                    row[f"risk_M0{tk}"] = P.hypo_risk(q["M0"][:, :h], thr)
                wr.writerow(row); n_rows += 1
    json.dump({"tag": tag, "model": args.model, "repo": getattr(adapter, "repo", None),
               "zero_shot": True, "supports_future": adapter.supports_future,
               "n_test_windows": len(win), "n_rows": n_rows, "split_seed": args.seed,
               "n_test_cases": len(test_cases)}, open(f"results/baseline_meta_{tag}.json", "w"), indent=1)
    print(f"[zs] wrote {path}  ({n_rows} rows) + results/baseline_meta_{tag}.json", flush=True)
    print(f"[zs] Done in {time.time()-t0:.0f}s. Compare: scripts/baselines/compare.py "
          f"--tirex {stem} --baseline {tag}", flush=True)


if __name__ == "__main__":
    main()
