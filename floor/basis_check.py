"""Basis check before any arm: ||W_teacher - W_hat||_F / ||W_teacher||_F < 0.5 for every decoder linear of the dequantised scoring
checkpoint. Exit 4 if any linear fails. usage: basis_check.py <fmt> <ckpt dir> <input file or folder> <out json>"""
import json
import sys

import numpy as np
import torch

from rowtune import floor as FL
from rowtune.core import linears
from rowtune.formats import detect
from rowtune.upgrade import student_from

import common as C

fmt, ckpt, src, out = sys.argv[1:5]
dev = "cuda:0"
teacher = C.load_teacher(dev)
student = C.load_ckpt_model(ckpt, dev)
r = FL.basis_check(teacher, student)
# sanity: the checkpoint's linears are the dequantised file the tool reads (bf16 values)
q = detect(src).read(src)
sl, diffs = linears(student), []
for hf, lin in q.linears.items():
    w = torch.from_numpy(lin.dequant()).to(dev, torch.bfloat16).float()
    diffs.append(float((w - sl[hf.removesuffix(".weight")].weight.float()).abs().max()))
r["ckpt_vs_file_max_abs_diff"] = max(diffs)
r["n_linears_in_file"] = len(q.linears)
r["median_ratio"] = float(np.median(list(r["ratio"].values())))
r["fmt"], r["ckpt"] = fmt, ckpt
json.dump(r, open(out, "w"), indent=1)
print("basis", fmt, "PASS" if r["PASS"] else "FAIL", "max", r["max"], "median", r["median_ratio"], "failed", len(r["failed"]), "ckpt-vs-file max abs diff", r["ckpt_vs_file_max_abs_diff"])
sys.exit(0 if r["PASS"] else 4)
