"""Paired bootstrap exactly as bench/bootstrap_ci.py: first 2,000 items per task, 2,000 resamples, numpy default_rng(0), one index set per task shared by
every version, percentile 95% intervals on the mean over six tasks, kept share vs 'bf16', differences of named pairs.
usage: paired_ci.py out.json --v name=file:key [--v ...] --pair a,b [--pair ...]   (a version called bf16 must be given)"""
import argparse, json
import numpy as np
ap = argparse.ArgumentParser(); ap.add_argument("out"); ap.add_argument("--v", action="append", required=True); ap.add_argument("--pair", action="append", default=[])
a = ap.parse_args()
TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "lambada_openai"]; REPS = 2000; LIM = 2000
V = {}
for spec in a.v:
    name, fk = spec.split("=", 1); f, k = fk.rsplit(":", 1)
    e = json.load(open(f))["methods"][k]; assert "items" in e and "error" not in e, (f, k)
    V[name] = e["items"]
n = {t: min(LIM, min(len(V[m][t]["acc"]) for m in V)) for t in TASKS}
A = {m: {t: np.array(V[m][t]["acc"][: n[t]], float) for t in TASKS} for m in V}
rng = np.random.default_rng(0); idx = {t: rng.integers(0, n[t], size=(REPS, n[t])) for t in TASKS}
boot = {m: np.mean([x[t][idx[t]].mean(1) for t in TASKS], 0) * 100 for m, x in A.items()}
point = {m: float(np.mean([x[t].mean() for t in TASKS]) * 100) for m, x in A.items()}
out = {"n_items": n, "versions": {}, "pairs": {}}
print("items per task", n, "total", sum(n.values()))
for m in A:
    ci = np.percentile(boot[m], [2.5, 97.5]); r = {"mean": point[m], "ci": ci.tolist()}
    if "bf16" in A:
        kept = boot[m] / boot["bf16"] * 100; kci = np.percentile(kept, [2.5, 97.5]); r.update(kept=point[m] / point["bf16"] * 100, kept_ci=kci.tolist())
    out["versions"][m] = r
    print(f"{m:34s} mean {point[m]:6.2f} [{ci[0]:6.2f},{ci[1]:6.2f}]" + (f"  kept {r['kept']:6.2f}% [{r['kept_ci'][0]:6.2f},{r['kept_ci'][1]:6.2f}]" if "kept" in r else ""))
for p in a.pair:
    x, y = p.split(","); d = boot[x] - boot[y]; ci = np.percentile(d, [2.5, 97.5])
    out["pairs"][p] = {"diff": point[x] - point[y], "ci": ci.tolist()}; print(f"diff {x} - {y}: {point[x]-point[y]:+.2f} points, 95% CI [{ci[0]:+.2f}, {ci[1]:+.2f}]")
json.dump(out, open(a.out, "w"), indent=1)
