"""Quantities 4 to 6: accuracy and kept share with paired bootstrap, held-out KL, and the share of the tuned gain reached.
usage: final_stats.py <fmt gguf|openvino> <results dir> <out json> <bf16 reference json:key> name=levers.json:key [...]   (names: A B C1 C2 D)"""
import json
import sys

from rowtune import floor as FL

fmt, res, out, ref = sys.argv[1:5]
acc = {}
f, k = ref.rsplit(":", 1)
acc["bf16"] = {t: v["acc"] for t, v in json.load(open(f))["methods"][k]["items"].items()}
for spec in sys.argv[5:]:
    name, fk = spec.split("=", 1)
    f, k = fk.rsplit(":", 1)
    e = json.load(open(f))["methods"][k]
    assert "items" in e and "error" not in e, spec
    acc[name] = {t: v["acc"] for t, v in e["items"].items()}
have = lambda *a: all(x in acc for x in a)
pairs = [p for p in [("B", "A"), ("C1", "A"), ("C2", "A"), ("B", "C1"), ("B", "C2"), ("D", "A"), ("B", "D")] if have(*p)]
ratios = [r for r in [("C1", "B", "A"), ("C2", "B", "A"), ("D", "B", "A")] if have(*r)]
o = {"fmt": fmt, "accuracy": FL.accuracy_bootstrap(acc, pairs, ratios)}
kl = json.load(open(f"{res}/{fmt}_kl.json"))["kl"]
kl = {k: v for k, v in kl.items() if k in acc}
klp = [p for p in pairs if have(*p) and all(x in kl for x in p)]
klr = [r for r in ratios if all(x in kl for x in r)]
o["kl"] = FL.kl_bootstrap(kl, klp, klr)
json.dump(o, open(out, "w"), indent=1)
a = o["accuracy"]
for m, v in a["versions"].items():
    print(f"{m:6s} mean {v['mean']:.2f} [{v['ci'][0]:.2f},{v['ci'][1]:.2f}] kept {v['kept']:.2f}% [{v['kept_ci'][0]:.2f},{v['kept_ci'][1]:.2f}]  KL {o['kl']['means'].get(m, float('nan')):.4f}")
for p, v in a["pairs"].items():
    print(f"acc diff {p}: {v['diff']:+.2f} [{v['ci'][0]:+.2f},{v['ci'][1]:+.2f}]")
for p, v in a["ratios"].items():
    print(f"acc R {p}: {v['point']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]")
for p, v in o["kl"]["pairs"].items():
    print(f"KL diff {p}: {v['diff']:+.4f} [{v['ci'][0]:+.4f},{v['ci'][1]:+.4f}]")
for p, v in o["kl"]["ratios"].items():
    print(f"KL R {p}: {v['point']:.3f} [{v['ci'][0]:.3f},{v['ci'][1]:.3f}]")
