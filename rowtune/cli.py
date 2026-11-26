"""Command line.
rowtune INPUT -o OUTPUT [--teacher MODEL]   upgrade a 4-bit file (GGUF Q4_0, OpenVINO INT4, packed 4-bit safetensors) in its own format
rowtune round --model M --out DIR           round a 16-bit model to 4 bits per row, tune, write a 16-bit checkpoint of the result"""
import argparse
import json
import os
import sys
import time

import torch

from . import DEFAULT_LENGTH, DEFAULT_LR, DEFAULT_STEPS, hf
from .core import copy_linears, rtn_student
from .formats import detect
from .data import read_parquet_texts, read_text_file, windows_from_texts
from .tune import tune
from .upgrade import upgrade

DTYPES = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}


def parse_round_args(argv=None):
    ap = argparse.ArgumentParser(prog="rowtune round", description="Round a 16-bit model to 4 bits, tune the row multipliers, write 16-bit weights.")
    ap.add_argument("--model", required=True, help="16-bit model id or local folder: the teacher, and the source of the weights to round")
    ap.add_argument("--student", default=None, help="local folder (or id) of a 4-bit model stored as dequantised 16-bit weights; default: round --model to 4 bits per row")
    ap.add_argument("--student-compressed", action="store_true", help="--student is a compressed-tensors release; decompress it on load")
    ap.add_argument("--group-size", type=int, default=0, help="rounding of the default student: 0 = one scale per row, else per group of this many columns")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--text-file", help="UTF-8 text; documents split on a line holding only <<<DOC>>> if present, else on lines")
    src.add_argument("--parquet", help="parquet file with a 'text' column")
    ap.add_argument("--out", required=True, help="output folder (must not exist or be empty)")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="tuning windows, one optimiser step each")
    ap.add_argument("--length", type=int, default=DEFAULT_LENGTH, help="tokens per window")
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--seed", type=int, default=1, help="shuffle seed of the documents")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--device", default="cuda:0", help="device of the student")
    ap.add_argument("--teacher-device", default=None, help="device of the teacher (default: --device)")
    return ap.parse_args(argv)


def run_round(a):
    if os.path.isdir(a.out) and os.listdir(a.out):
        raise SystemExit(f"{a.out} exists and is not empty; nothing was written")
    t0 = time.time()
    dt, tdev = DTYPES[a.dtype], a.teacher_device or a.device
    tok = hf.load_tokenizer(a.model)
    texts = read_text_file(a.text_file) if a.text_file else read_parquet_texts(a.parquet)
    windows = windows_from_texts(tok, texts, a.steps, a.length, a.seed)
    if len(windows) < a.steps:
        print(f"warning: text gave {len(windows)} windows, fewer than --steps {a.steps}", flush=True)
    teacher = hf.load_causal_lm(a.model, dt, tdev)
    for p in teacher.parameters():
        p.requires_grad_(False)
    student = hf.load_causal_lm(a.model, dt, a.device)
    if a.student:
        copy_linears(hf.load_causal_lm(a.student, dt, "cpu", compressed=a.student_compressed), student)
        start = f"checkpoint {a.student}"
    else:
        rtn_student(student, 4, a.group_size)
        start = "round-to-nearest, 4 bits, " + ("one scale per row" if a.group_size == 0 else f"one scale per group of {a.group_size}")
    info = tune(student, teacher, windows, lr=a.lr, device=a.device, teacher_device=tdev)
    scales = info.pop("scales")
    info.update(start=start, model=a.model, length=a.length, lr=a.lr, seed=a.seed, dtype=a.dtype, seconds=round(time.time() - t0, 1))
    hf.save_checkpoint(student, tok, a.out)
    torch.save(scales, os.path.join(a.out, "row_scales.pt"))
    with open(os.path.join(a.out, "tuning.json"), "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info), flush=True)
    return 0




def parse_upgrade_args(argv=None):
    ap = argparse.ArgumentParser(prog="rowtune", description="Upgrade a 4-bit model file in its own format: integer codes bit-identical, "
                                 "only the per-row (per-block-row) scales change; output has the same structure and size.")
    ap.add_argument("input", help="GGUF file with Q4_0 weights, OpenVINO IR folder with INT4 weights, or packed 4-bit safetensors folder")
    ap.add_argument("-o", "--output", required=True, help="output file or folder (must not exist)")
    ap.add_argument("--teacher", default=None, help="16-bit model id or folder; default: the model named in the input folder's config.json")
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--text-file", help="UTF-8 text; documents split on a line holding only <<<DOC>>> if present, else on lines")
    src.add_argument("--parquet", help="parquet file with a 'text' column (default: WikiText-103 training shard 2, downloaded)")
    ap.add_argument("--steps", type=int, default=DEFAULT_STEPS, help="tuning windows, one optimiser step each")
    ap.add_argument("--length", type=int, default=DEFAULT_LENGTH, help="tokens per window")
    ap.add_argument("--lr", type=float, default=DEFAULT_LR)
    ap.add_argument("--seed", type=int, default=1, help="shuffle seed of the documents")
    ap.add_argument("--dtype", choices=sorted(DTYPES), default="bf16")
    ap.add_argument("--device", default="cuda:0", help="device of the student")
    ap.add_argument("--teacher-device", default=None, help="device of the teacher (default: --device)")
    return ap.parse_args(argv)


def _teacher_name(a):
    if a.teacher:
        return a.teacher
    cfg = os.path.join(a.input, "config.json")
    if os.path.isfile(cfg):
        with open(cfg) as f:
            name = json.load(f).get("_name_or_path")
        if name:
            return name
    raise SystemExit("pass --teacher: the 16-bit model this file was made from (the input does not name it)")


def run_upgrade(a):
    if os.path.exists(a.output):
        raise SystemExit(f"{a.output} exists; nothing was written")
    try:
        fmt = detect(a.input)
    except ValueError as e:
        raise SystemExit(str(e))
    teacher_name = _teacher_name(a)
    t0 = time.time()
    dt, tdev = DTYPES[a.dtype], a.teacher_device or a.device
    teacher = hf.load_causal_lm(teacher_name, dt, tdev)
    for p in teacher.parameters():
        p.requires_grad_(False)
    tok = hf.load_tokenizer(teacher_name)
    texts = read_text_file(a.text_file) if a.text_file else read_parquet_texts(a.parquet or hf.default_text_parquet())
    windows = windows_from_texts(tok, texts, a.steps, a.length, a.seed)
    if len(windows) < a.steps:
        print(f"warning: text gave {len(windows)} windows, fewer than --steps {a.steps}", flush=True)
    info = upgrade(a.input, a.output, teacher, lambda: hf.load_causal_lm(teacher_name, dt, a.device), windows, lr=a.lr, device=a.device,
                   teacher_device=tdev, dtype=dt)
    scales = info.pop("scales")
    info.update(input=os.path.basename(os.path.normpath(a.input)), teacher=teacher_name, length=a.length, lr=a.lr, seed=a.seed, dtype=a.dtype,
                seconds=round(time.time() - t0, 1))
    torch.save(scales, a.output + ".row_scales.pt")
    with open(a.output + ".rowtune.json", "w") as f:
        json.dump(info, f, indent=1)
    print(json.dumps(info), flush=True)
    return 0


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == "round":
        return run_round(parse_round_args(argv[1:]))
    return run_upgrade(parse_upgrade_args(argv))


if __name__ == "__main__":
    raise SystemExit(main())
