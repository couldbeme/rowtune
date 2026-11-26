"""Quantities 1 to 3 for one file: layer-local errors on held-out text with 16-bit inputs, floor fraction F_l, tuned fraction T_l, and where the
tuned multipliers sit relative to the closed form. usage: analyse.py <fmt> <ckpt of the input> <results dir>"""
import json
import re
import sys

import numpy as np
import torch

from rowtune import floor as FL
from rowtune.upgrade import effective_factors

import common as C

fmt, ckpt, res = sys.argv[1:4]
dev = "cuda:0"
H = C.load_windows(res + "/windows.json")["held"]
teacher, student = C.load_teacher(dev), C.load_ckpt_model(ckpt, dev)
mom = FL.moments_teacher_input(teacher, student, H, dev)
cf = torch.load(f"{res}/{fmt}_c1_factors.pt")
fac = {"C1": {n: v.numpy() for n, v in cf.items()}}
c2p = f"{res}/{fmt}_c2_factors.pt"
import os
if os.path.exists(c2p):
    fac["C2"] = {n: v.numpy() for n, v in torch.load(c2p).items()}
fac["B"] = effective_factors({n: v for n, v in torch.load(C.B_SCALES[fmt]).items()}, torch.bfloat16)
ones = {n: np.ones_like(v) for n, v in fac["C1"].items()}
rows = []
proj = lambda n: re.search(r"(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)", n).group(1)
for n in sorted(mom, key=lambda s: (int(s.split(".layers.")[1].split(".")[0]), s)):
    E1 = FL.layer_error(mom[n], ones[n])
    r = {"name": n, "proj": proj(n), "E1": E1}
    for k, f in fac.items():
        r["E_" + k] = FL.layer_error(mom[n], f[n])
        r["frac_" + k] = r["E_" + k] / E1
    rows.append(r)
frac = lambda k: np.array([r["frac_" + k] for r in rows])
out = {"fmt": fmt, "n_linears": len(rows), "heldout_tokens": len(H) * len(H[0]),
       "median_recovered_C1": float(np.median(1 - frac("C1"))), "median_F_C1": float(np.median(frac("C1"))), "median_T_B": float(np.median(frac("B"))),
       "share_T_gt_F": float(np.mean(frac("B") > frac("C1"))), "n_T_gt_F": int(np.sum(frac("B") > frac("C1"))),
       "min_recovered_C1": float(np.min(1 - frac("C1"))), "max_recovered_C1": float(np.max(1 - frac("C1")))}
if "C2" in fac:
    out["median_recovered_C2"] = float(np.median(1 - frac("C2")))
mb = np.concatenate([fac["B"][r["name"]] for r in rows])
mc = np.concatenate([fac["C1"][r["name"]] for r in rows])
out["alignment_all"] = FL.alignment_m(mb, mc)
out["alignment_by_proj"] = {}
for p in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"):
    ns = [r["name"] for r in rows if r["proj"] == p]
    out["alignment_by_proj"][p] = FL.alignment_m(np.concatenate([fac["B"][n] for n in ns]), np.concatenate([fac["C1"][n] for n in ns]))
out["median_abs_log_c1_positive_rows"] = out["alignment_all"]["median_abs_log_c"]
out["median_abs_log_B"] = out["alignment_all"]["median_abs_log_b"]
out["c1_range"] = [float(mc.min()), float(mc.max())]
out["B_range"] = [float(mb.min()), float(mb.max())]
out["c1_share_nonpositive"] = float((mc <= 0).mean())
out["c1_quantiles_1_5_25_50_75_95_99"] = [float(x) for x in np.quantile(mc, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])]
out["c1_nonpositive_by_proj"] = {p: float(np.mean(np.concatenate([fac["C1"][r["name"]] for r in rows if r["proj"] == p]) <= 0)) for p in ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")}
out["per_linear"] = rows
json.dump(out, open(f"{res}/{fmt}_layer_quantities.json", "w"), indent=1)
print({k: v for k, v in out.items() if k not in ("per_linear", "alignment_by_proj")})
