"""Print the self-describing metadata of the trained baseline checkpoints (H/Lc/channels/d_model
and the arm keys). Diagnostic for the embedding-extraction geometry -- run inside the container."""
import sys, glob, torch

paths = sys.argv[1:] or sorted(glob.glob("results/baseline_ckpt_*.pt"))
for p in paths:
    ck = torch.load(p, map_location="cpu", weights_only=False)
    print(f"{p} | model={ck['model']} H={ck['H']} Lc={ck['Lc']} "
          f"n_past={ck['n_past']} n_fut={ck['n_fut']} d_model={ck['d_model']} "
          f"arms={list(ck['state'].keys())} "
          f"past_names={ck.get('past_names')} fut_names={ck.get('fut_names')} dt={ck.get('dt')}")
