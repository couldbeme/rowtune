"""Held-out forward KL(teacher || arm) per window for dequantised scoring checkpoints. usage: kl_arms.py <results dir> <out json> name=ckpt [name=ckpt ...]"""
import json
import sys

import numpy as np
import torch

from rowtune import floor as FL

import common as C

res, out = sys.argv[1:3]
dev = "cuda:0"
H = C.load_windows(res + "/windows.json")["held"]
teacher = C.load_teacher(dev)
d = {}
for spec in sys.argv[3:]:
    name, ck = spec.split("=", 1)
    m = C.load_ckpt_model(ck, dev)
    d[name] = FL.heldout_kl(teacher, m, H, dev).tolist()
    print(name, float(np.mean(d[name])), flush=True)
    del m
    torch.cuda.empty_cache()
json.dump({"windows": len(H), "tokens_per_window": len(H[0]), "kl": d}, open(out, "w"))
