"""Tuning windows (the tool's) and 200 held-out windows from documents the tuning stream never read; stops (exit 3) if they are not disjoint.
usage: prep_windows.py <out dir>"""
import json
import os
import sys

from rowtune import floor as FL
from rowtune import hf
from rowtune.data import read_parquet_texts

import common as C

out = sys.argv[1]
os.makedirs(out, exist_ok=True)
tok = hf.load_tokenizer(C.TEACHER)
texts = read_parquet_texts(C.PARQUET)
w = FL.split_windows(tok, texts, n_tune=600, length=1024, tune_seed=1, n_held=200, held_seed=12345)
try:
    rep = FL.check_disjoint(w)
except FL.DisjointError as e:
    json.dump({"disjoint": False, "error": str(e)}, open(os.path.join(out, "disjoint.json"), "w"))
    print("DISJOINTNESS CHECK FAILED", e)
    sys.exit(3)
rep.update(parquet=C.PARQUET, parquet_sha256=C.sha256(C.PARQUET), tune_seed=1, held_seed=12345, length=1024, n_texts=len(texts))
json.dump(w, open(os.path.join(out, "windows.json"), "w"))
rep["windows_json_sha256"] = C.sha256(os.path.join(out, "windows.json"))
json.dump(rep, open(os.path.join(out, "disjoint.json"), "w"), indent=1)
print(json.dumps(rep))
