"""Arms C1 / C2: closed-form per-row multipliers from the tuning windows, written with the tool's writer; the tool's file check must pass
or the arm is void. usage: make_arm.py <fmt> <c1|c2> <input> <ckpt of the input> <output> <results dir>"""
import json
import os
import sys
import time

import numpy as np
import torch

from rowtune import floor as FL
from rowtune.formats import detect
from rowtune.upgrade import effective_factors

import common as C

fmt, arm, src, ckpt, dst, res = sys.argv[1:7]
dev = "cuda:0"
W = C.load_windows(res + "/windows.json")["tune"]
assert len(W) == 600 and all(len(w) == 1024 for w in W)
teacher, student = C.load_teacher(dev), C.load_ckpt_model(ckpt, dev)
torch.cuda.synchronize()
t0 = time.time()
if arm == "c1":
    m = FL.closed_form_from_moments(FL.moments_teacher_input(teacher, student, W, dev))
else:
    m = FL.closed_form_c2(teacher, student, W, dev, torch.bfloat16)
torch.cuda.synchronize()
secs = time.time() - t0
eff = effective_factors(m, torch.bfloat16)  # rounded to bf16, as the tool does before writing
torch.save({n: torch.from_numpy(v) for n, v in eff.items()}, f"{res}/{fmt}_{arm}_factors.pt")
allv = np.concatenate([v for v in eff.values()])
mod = detect(src)
info = {"arm": arm, "fmt": fmt, "input": src, "output": dst, "compute_seconds": round(secs, 1), "n_values": int(allv.size), "c_min": float(allv.min()),
        "c_max": float(allv.max()), "c_mean": float(allv.mean()), "median_abs_log": float(np.median(np.abs(np.log(allv)))),
        "windows": len(W), "tokens": len(W) * 1024}
mod.write(src, dst, {n + ".weight": v for n, v in eff.items()})
chk = mod.verify(src, dst)
info["file_check"] = chk
info["VOID"] = not chk["PASS"]
json.dump(info, open(f"{res}/{fmt}_{arm}_make.json", "w"), indent=1)
print(json.dumps(info))
sys.exit(0 if chk["PASS"] else 5)
