"""The tuning windows used for C1/C2 and the held-out split are exactly the windows the tool cuts (windows_from_texts, steps 600, length 1024, seed 1)."""
import json
import sys

from rowtune import hf
from rowtune.data import read_parquet_texts, windows_from_texts

import common as C

W = C.load_windows(sys.argv[1] + "/windows.json")
tok = hf.load_tokenizer(C.TEACHER)
tool = windows_from_texts(tok, read_parquet_texts(C.PARQUET), 600, 1024, 1)
ok = tool == W["tune"]
json.dump({"tune_windows_equal_tool_windows": ok, "n": len(tool)}, open(sys.argv[1] + "/tool_windows_check.json", "w"))
print("tool windows equal:", ok)
sys.exit(0 if ok else 6)
