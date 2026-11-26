"""Packed 4-bit checkpoints in safetensors files, in the compressed-tensors "pack-quantized" layout: per linear
`<name>.weight_packed` int32 [rows, cols/8] (eight codes per word, offset binary, element 0 in the lowest nibble) and
`<name>.weight_scale` [rows, 1] or [rows, cols/group] (bfloat16, float16 or float32). A row factor multiplies every scale of the
row in place; headers, packed codes and all other tensors keep their bytes. The layout was written from the library's
documentation and the OpenVINO translator's description of it; it has not been checked against a real release on this machine."""
import glob
import json
import os
import shutil
import struct

import numpy as np

from .base import Linear4, Quantised, check_factors, from_bf16_bits, to_bf16_bits

NAME = "ct"
SCALE_DTYPES = {"BF16": 2, "F16": 2, "F32": 4}


def _shards(d):
    return sorted(glob.glob(os.path.join(d, "*.safetensors")))


def _header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        hdr = json.loads(f.read(n))
    hdr.pop("__metadata__", None)
    return hdr, 8 + n


def is_checkpoint(d):
    return any(k.endswith(".weight_packed") for p in _shards(d) for k in _header(p)[0])


def _index(d):
    """{tensor name: (shard path, dtype, shape, absolute start, absolute end)}"""
    out = {}
    for p in _shards(d):
        hdr, base = _header(p)
        for k, v in hdr.items():
            out[k] = (p, v["dtype"], tuple(v["shape"]), base + v["data_offsets"][0], base + v["data_offsets"][1])
    return out


def _raw(t):
    with open(t[0], "rb") as f:
        f.seek(t[3])
        return f.read(t[4] - t[3])


def _scale_array(t):
    raw = _raw(t)
    if t[1] == "BF16":
        a = from_bf16_bits(np.frombuffer(raw, np.uint16))
    elif t[1] == "F16":
        a = np.frombuffer(raw, np.float16).astype(np.float32)
    else:
        a = np.frombuffer(raw, np.float32).copy()
    return a.reshape(t[2])


def _scale_bytes(a, dtype):
    if dtype == "BF16":
        return to_bf16_bits(a).tobytes()
    return a.astype(np.float16 if dtype == "F16" else np.float32).tobytes()


def _linears(idx):
    ok, skipped = {}, []
    for k, t in idx.items():
        if not k.endswith(".weight_packed"):
            continue
        base = k.removesuffix(".weight_packed")
        s = idx.get(base + ".weight_scale")
        if t[1] == "I32" and s is not None and s[1] in SCALE_DTYPES and (base + ".weight_zero_point") not in idx and len(s[2]) == 2 \
                and (t[2][1] * 8) % s[2][1] == 0:
            ok[base + ".weight"] = (t, s)
        else:
            skipped.append(base + ".weight")
    return ok, skipped


def read(d):
    if not os.path.isdir(d) or not _shards(d) or not is_checkpoint(d):
        raise ValueError(f"{d} is not a folder with a packed 4-bit safetensors checkpoint")
    idx = _index(d)
    lin, skipped = _linears(idx)
    out = {}
    for hf, (t, s) in lin.items():
        w = np.frombuffer(_raw(t), np.int32).reshape(t[2]).view(np.uint32)
        nib = np.stack([(w >> np.uint32(4 * i)) & np.uint32(15) for i in range(8)], axis=2).reshape(t[2][0], -1)
        out[hf] = Linear4((nib.astype(np.int16) - 8).astype(np.int8), _scale_array(s))
    return Quantised(out, {}, skipped)


def write(src, dst, factors):
    idx = _index(src)
    lin, _ = _linears(idx)
    check_factors(factors, {hf: t[2][0] for hf, (t, s) in lin.items()})
    if os.path.exists(dst):
        raise FileExistsError(dst)
    part = dst + ".part"
    shutil.copytree(src, part)
    for hf, c in factors.items():
        s = lin[hf][1]
        new = _scale_array(s) * np.asarray(c, np.float32)[:, None]
        with open(os.path.join(part, os.path.basename(s[0])), "r+b") as f:
            f.seek(s[3])
            f.write(_scale_bytes(new, s[1]))
    os.replace(part, dst)


def _listing(d):
    return sorted((os.path.relpath(os.path.join(r, f), d), os.path.getsize(os.path.join(r, f))) for r, _, fs in os.walk(d) for f in fs)


def verify(src, dst):
    """dst against src: same files and sizes, same safetensors headers, identical bytes of every tensor except the scales."""
    same_size = _listing(src) == _listing(dst)
    a, b = _index(src), _index(dst)
    same_headers = same_size and all(_header(p)[0] == _header(os.path.join(dst, os.path.basename(p)))[0] for p in _shards(src))
    codes_identical, rest_identical, changed, n_scales = True, True, 0, 0
    if same_headers:
        for k, t in a.items():
            u = b[k]
            same = _raw(t) == _raw(u)
            if k.endswith(".weight_scale"):
                n_scales += 1
                changed += int(not same)
            elif k.endswith(".weight_packed"):
                codes_identical &= same
            else:
                rest_identical &= same
    rep = {"format": NAME, "same_size": bool(same_size), "same_structure": bool(same_headers), "codes_identical": bool(codes_identical),
           "all_other_tensors_identical": bool(rest_identical), "scale_tensors": n_scales, "scale_tensors_changed": changed,
           "bytes": sum(s for _, s in _listing(dst))}
    rep["PASS"] = bool(same_headers and codes_identical and rest_identical)
    return rep
