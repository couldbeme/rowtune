"""Re-scored input file (frozen driver) against the released result of the same file: per-item agreement of the 0/1 correctness and of the
chosen answers, per task. usage: compare_released.py <released json> <released key> <rescored json> <rescored key> <out json>"""
import json
import sys

a = json.load(open(sys.argv[1]))["methods"][sys.argv[2]]["items"]
b = json.load(open(sys.argv[3]))["methods"][sys.argv[4]]["items"]
out = {}
for t in a:
    n = min(len(a[t]["acc"]), len(b[t]["acc"]), 2000)
    same_acc = sum(1 for i in range(n) if a[t]["acc"][i] == b[t]["acc"][i])
    same_pred = sum(1 for i in range(n) if a[t]["pred"][i] == b[t]["pred"][i])
    out[t] = {"items": n, "same_correct": same_acc, "same_pred": same_pred, "mean_released": sum(a[t]["acc"][:n]) / n, "mean_rescored": sum(b[t]["acc"][:n]) / n}
    print(t, out[t])
json.dump(out, open(sys.argv[5], "w"), indent=1)
