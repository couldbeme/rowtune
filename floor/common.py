"""Shared paths and loaders of the baseline experiment scripts.

Set ROWTUNE_BENCH_DIR to a folder that holds the models and the input files named below, and
ROWTUNE_TUNING_PARQUET to the WikiText-103 training shard used for tuning (train-00001-of-00002.parquet)."""
import hashlib
import json
import os

import torch

B = os.environ.get("ROWTUNE_BENCH_DIR", ".")
F = B + "/floor-2026-11-26"
TEACHER = B + "/hf/hub/models--Qwen--Qwen3-1.7B/snapshots/70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"
PARQUET = os.environ.get("ROWTUNE_TUNING_PARQUET", "train-00001-of-00002.parquet")
GGUF_IN = B + "/gguf/src/Qwen3-1.7B-Q4_0-from-Q8_0-ggufpy.gguf"
OV_IN = B + "/ov/q17_sym_channel_awq_se"
B_SCALES = {"gguf": B + "/rowtune-test/qwen3-1.7b-q4_0-rowtune.gguf.row_scales.pt", "openvino": B + "/rowtune-test/q17_awq_se_rowtune.row_scales.pt"}


def sha256(path, chunk=1 << 24):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def load_windows(path=None):
    return json.load(open(path or F + "/results/windows.json"))


def load_teacher(device):
    from rowtune import hf
    return hf.load_causal_lm(TEACHER, torch.bfloat16, device)


@torch.no_grad()
def load_ckpt_model(ckpt_dir, device):
    """The teacher snapshot with the decoder linears replaced by those of a dequantised scoring checkpoint (what the scoring driver's
    ext_* methods do: only the linear layers come from the checkpoint, the rest from the snapshot)."""
    from safetensors import safe_open
    from rowtune.core import linears
    m = load_teacher(device)
    lin = linears(m)
    n = 0
    with safe_open(os.path.join(ckpt_dir, "model.safetensors"), "pt", device="cpu") as f:
        for k in f.keys():
            if k.endswith(".weight") and k[:-7] in lin:
                lin[k[:-7]].weight.copy_(f.get_tensor(k).to(device, lin[k[:-7]].weight.dtype))
                n += 1
    if n != len(lin):
        raise SystemExit(f"{ckpt_dir}: {n} of {len(lin)} decoder linears found")
    for p in m.parameters():
        p.requires_grad_(False)
    return m
