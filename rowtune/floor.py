"""Baselines for the per-row multiplier: the published closed-form least-squares multiplier per output row (arms C1 and C2 of the
baseline plan), held-out windows and checks, layer errors, alignment statistics and paired bootstraps. No gradients anywhere here."""
import random

import numpy as np
import torch
import torch.nn.functional as F

from .core import linears
from .data import MIN_CHARS

TASKS = ["arc_easy", "arc_challenge", "hellaswag", "piqa", "winogrande", "lambada_openai"]


class DisjointError(RuntimeError):
    pass


class _Stop(Exception):
    pass


# ---- closed form -----------------------------------------------------------------------------------------------------------------
def _tok(x, device):
    return torch.tensor([x], device=device)


def _acc_add(acc, name, y, yh):
    """acc[name] += per-row sums over tokens of (y*y, y*yh, yh*yh), float64. y, yh: [tokens, rows]."""
    s = torch.stack([(y * y).sum(0), (y * yh).sum(0), (yh * yh).sum(0)]).double()
    acc[name] = s if name not in acc else acc[name] + s


@torch.no_grad()
def moments_teacher_input(teacher, student, windows, device):
    """Per linear and output row i, summed over all tokens of `windows`: [yy, y.yh, yh.yh] with x the 16-bit teacher's input to the
    linear, y = w_i . x (teacher row) and yh = what_i . x (the student's dequantised row). One teacher forward per window."""
    tl, sl, acc, hooks = linears(teacher), linears(student), {}, []
    for n, m in tl.items():
        def pre(mod, a, n=n):
            x = a[0].reshape(-1, a[0].shape[-1]).float()
            _acc_add(acc, n, x @ tl[n].weight.float().T, x @ sl[n].weight.to(x.device).float().T)
        hooks.append(m.register_forward_pre_hook(pre))
    try:
        for w in windows:
            teacher(input_ids=_tok(w, device))
    finally:
        for h in hooks:
            h.remove()
    return {n: v.cpu() for n, v in acc.items()}


def closed_form_from_moments(mom):
    """m_i = sum(y yh) / sum(yh^2); a row whose student output is identically zero gets 1."""
    out = {}
    for n, v in mom.items():
        hh = v[2]
        out[n] = torch.where(hh > 0, v[1] / hh.clamp(min=1e-300), torch.ones_like(hh)).float()
    return out


def layer_error(mom_n, m):
    """E(m) = sum_i ||y_i - m_i yh_i||^2 from the moments of one linear."""
    m = torch.as_tensor(m).double()
    return float((mom_n[0] - 2 * m * mom_n[1] + m * m * mom_n[2]).sum())


def _layer_index(name):
    return int(name.split(".layers.")[1].split(".")[0])


@torch.no_grad()
def closed_form_c2(teacher, student, windows, device, dtype):
    """Arm C2: layers in forward order. For layer L, x~ is the input the linear receives in the student whose layers < L already carry
    their C2 multipliers (the layer's own linears still carry none); y_i = w_i . x (teacher weights, teacher input) and
    y~_i = what_i . x~; m_i = sum(y y~) / sum(y~^2). The multipliers are applied to `student` in place (rounded to `dtype`) as the
    layers are done. Returns {name: float32 multipliers}."""
    tl, sl = linears(teacher), linears(student)
    by_layer = {}
    for n in sl:
        by_layer.setdefault(_layer_index(n), []).append(n)
    out = {}
    for L in sorted(by_layer):
        names = by_layer[L]
        acc, cur_t, cur_s, hooks = {}, {}, {}, []
        for store, lin in ((cur_t, tl), (cur_s, sl)):
            for n in names:
                hooks.append(lin[n].register_forward_pre_hook(lambda mod, a, n=n, store=store: store.__setitem__(n, a[0].reshape(-1, a[0].shape[-1]).float())))

            def stop(mod, a, o):
                raise _Stop

            hooks.append(lin[names[-1]].register_forward_hook(stop))
        try:
            for w in windows:
                for model, store in ((teacher, cur_t), (student, cur_s)):
                    try:
                        model(input_ids=_tok(w, device))
                    except _Stop:
                        pass
                for n in names:
                    _acc_add(acc, n, cur_t[n] @ tl[n].weight.float().T, cur_s[n] @ sl[n].weight.float().T)
        finally:
            for h in hooks:
                h.remove()
        m = closed_form_from_moments({n: v.cpu() for n, v in acc.items()})
        for n in names:
            out[n] = m[n]
            sl[n].weight.mul_(m[n].to(sl[n].weight.device, sl[n].weight.dtype)[:, None])
    return out


# ---- checks and held-out quantities ----------------------------------------------------------------------------------------------
@torch.no_grad()
def basis_check(teacher, student, limit=0.5):
    """||W_teacher - W_student||_F / ||W_teacher||_F for every decoder linear; PASS when all are below `limit`."""
    tl, sl = linears(teacher), linears(student)
    ratio = {}
    for n in tl:
        a, b = tl[n].weight.float(), sl[n].weight.to(tl[n].weight.device).float()
        ratio[n] = float((a - b).norm() / a.norm())
    failed = [n for n, r in ratio.items() if not r < limit]
    return {"ratio": ratio, "max": max(ratio.values()), "limit": limit, "failed": failed, "PASS": not failed}


@torch.no_grad()
def heldout_kl(teacher, model, windows, device):
    """Forward KL(teacher || model), mean over positions of each window (float32 log-softmax): a numpy array, one value per window."""
    out = []
    for w in windows:
        x = _tok(w, device)
        lt = F.log_softmax(teacher(input_ids=x).logits.float(), -1)
        ls = F.log_softmax(model(input_ids=x).logits.float().to(lt.device), -1)
        out.append((lt.exp() * (lt - ls)).sum(-1).mean().item())
    return np.array(out)


def split_windows(tokenizer, texts, n_tune, length, tune_seed, n_held, held_seed):
    """The tool's tuning windows, and held-out windows cut from documents the tuning stream never touched.
    Documents (longer than MIN_CHARS) are shuffled with `tune_seed`; the tuning stream reads them in that order until it has n_tune windows.
    Held-out: every document the tuning stream did not read, shuffled with `held_seed`, cut the same way until n_held windows.
    Returns windows and the (filtered-list) document indices each stream read."""
    kept = [t for t in texts if len(t) > MIN_CHARS]
    order = list(range(len(kept)))
    random.Random(tune_seed).shuffle(order)  # same permutation as shuffling the texts themselves

    def cut(ids, n):
        out, cur, used = [], [], []
        for i in ids:
            used.append(i)
            cur += tokenizer(kept[i], add_special_tokens=False).input_ids
            while len(cur) >= length and len(out) < n:
                out.append(cur[:length])
                cur = cur[length:]
            if len(out) >= n:
                break
        return out, used

    tune, tune_docs = cut(order, n_tune)
    if len(tune) < n_tune:
        raise ValueError(f"text gave {len(tune)} tuning windows, fewer than {n_tune}")
    read = set(tune_docs)
    rest = [i for i in range(len(kept)) if i not in read]
    random.Random(held_seed).shuffle(rest)
    held, held_docs = cut(rest, n_held)
    if len(held) < n_held:
        raise ValueError(f"the documents the tuning windows did not read give {len(held)} windows, fewer than {n_held}")
    return {"tune": tune, "held": held, "tune_docs": tune_docs, "held_docs": held_docs, "length": length, "n_documents": len(kept)}


def check_disjoint(w):
    """Raises DisjointError unless the held-out windows are disjoint from the tuning windows: no shared document (so no shared token
    offset of the source text) and no identical window."""
    shared = set(w["tune_docs"]) & set(w["held_docs"])
    same = {tuple(x) for x in w["tune"]} & {tuple(x) for x in w["held"]}
    if shared or same:
        raise DisjointError(f"{len(shared)} shared documents, {len(same)} identical windows")
    return {"disjoint": True, "shared_docs": 0, "identical_windows": 0, "tune_windows": len(w["tune"]), "held_windows": len(w["held"]),
            "tune_docs_read": len(w["tune_docs"]), "held_docs_read": len(w["held_docs"]), "documents": w["n_documents"],
            "tune_tokens": len(w["tune"]) * w["length"], "held_tokens": len(w["held"]) * w["length"]}


# ---- statistics ------------------------------------------------------------------------------------------------------------------
def _rank(x):
    _, inv, cnt = np.unique(x, return_inverse=True, return_counts=True)
    return (np.cumsum(cnt) - (cnt - 1) / 2)[inv]


def alignment(b, c):
    """Where b (log of the tuned multipliers) sits relative to c (log of the closed form): Spearman and Pearson correlation, share of
    rows with the same sign, median |c|, median |b|, median |b - c|."""
    b, c = np.asarray(b, np.float64), np.asarray(c, np.float64)
    return {"spearman": float(np.corrcoef(_rank(b), _rank(c))[0, 1]), "pearson": float(np.corrcoef(b, c)[0, 1]),
            "same_sign": float((np.sign(b) == np.sign(c)).mean()), "median_abs_c": float(np.median(np.abs(c))),
            "median_abs_b": float(np.median(np.abs(b))), "median_abs_diff": float(np.median(np.abs(b - c))), "n": int(b.size)}


def alignment_m(mb, mc):
    """As alignment, on the multipliers themselves (the closed form can be zero or negative, where log is undefined). Spearman and the share
    of rows with the same sign of (m - 1) use every row (for positive m they equal those of the logs); Pearson of the logs and the
    log medians use the rows where the multipliers involved are positive; the number and share of non-positive closed-form rows is reported."""
    mb, mc = np.asarray(mb, np.float64), np.asarray(mc, np.float64)
    both, cpos = (mb > 0) & (mc > 0), mc > 0
    lb, lc = np.log(mb[both]), np.log(mc[both])
    return {"spearman": float(np.corrcoef(_rank(mb), _rank(mc))[0, 1]), "pearson_log": float(np.corrcoef(lb, lc)[0, 1]),
            "same_sign": float((np.sign(mb - 1) == np.sign(mc - 1)).mean()), "median_abs_log_c": float(np.median(np.abs(np.log(mc[cpos])))),
            "median_abs_log_b": float(np.median(np.abs(np.log(mb[mb > 0])))), "median_abs_log_diff": float(np.median(np.abs(lb - lc))),
            "median_abs_c_minus_1": float(np.median(np.abs(mc - 1))), "median_abs_b_minus_1": float(np.median(np.abs(mb - 1))),
            "n": int(mb.size), "n_both_positive": int(both.sum()), "n_c_nonpositive": int((~cpos).sum()), "share_c_nonpositive": float((~cpos).mean())}


def accuracy_bootstrap(acc, pairs, ratios=(), reps=2000, lim=2000):
    """The released procedure (floor/paired_ci.py): acc {version: {task: [0/1 per item]}}, first `lim` items per task, `reps` resamples,
    numpy default_rng(0), one index set per task shared by every version, percentile 95% intervals on the mean over six tasks.
    pairs: [(x, y)] -> x - y in points. ratios: [(c, b, a)] -> (c - a) / (b - a)."""
    n = {t: min(lim, min(len(acc[m][t]) for m in acc)) for t in TASKS}
    A = {m: {t: np.array(acc[m][t][: n[t]], float) for t in TASKS} for m in acc}
    rng = np.random.default_rng(0)
    idx = {t: rng.integers(0, n[t], size=(reps, n[t])) for t in TASKS}
    boot = {m: np.mean([x[t][idx[t]].mean(1) for t in TASKS], 0) * 100 for m, x in A.items()}
    point = {m: float(np.mean([x[t].mean() for t in TASKS]) * 100) for m, x in A.items()}
    out = {"n_items": n, "versions": {}, "pairs": {}, "ratios": {}}
    for m in A:
        r = {"mean": point[m], "ci": np.percentile(boot[m], [2.5, 97.5]).tolist()}
        if "bf16" in A:
            r.update(kept=point[m] / point["bf16"] * 100, kept_ci=np.percentile(boot[m] / boot["bf16"] * 100, [2.5, 97.5]).tolist())
        out["versions"][m] = r
    for x, y in pairs:
        out["pairs"][f"{x},{y}"] = {"diff": point[x] - point[y], "ci": np.percentile(boot[x] - boot[y], [2.5, 97.5]).tolist()}
    for c, b, a in ratios:
        out["ratios"][f"{c}/{b}/{a}"] = {"point": (point[c] - point[a]) / (point[b] - point[a]),
                                          "ci": np.percentile((boot[c] - boot[a]) / (boot[b] - boot[a]), [2.5, 97.5]).tolist()}
    return out


def kl_bootstrap(kl, pairs, ratios=(), reps=2000, seed=0):
    """Resampling windows: kl {version: array [windows]} (same windows for every version). pairs: [(x, y)] -> mean(x - y).
    ratios: [(c, b, a)] -> share of the KL reduction (a - c) / (a - b)."""
    n = len(next(iter(kl.values())))
    idx = np.random.default_rng(seed).integers(0, n, size=(reps, n))
    K = {m: np.asarray(v, float) for m, v in kl.items()}
    boot = {m: v[idx].mean(1) for m, v in K.items()}
    out = {"means": {m: float(v.mean()) for m, v in K.items()}, "mean_ci": {m: np.percentile(boot[m], [2.5, 97.5]).tolist() for m in K},
           "pairs": {}, "ratios": {}, "windows": n, "reps": reps, "seed": seed}
    for x, y in pairs:
        out["pairs"][f"{x},{y}"] = {"diff": float((K[x] - K[y]).mean()), "ci": np.percentile(boot[x] - boot[y], [2.5, 97.5]).tolist()}
    for c, b, a in ratios:
        out["ratios"][f"{c}/{b}/{a}"] = {"point": float((K[a].mean() - K[c].mean()) / (K[a].mean() - K[b].mean())),
                                          "ci": np.percentile((boot[a] - boot[c]) / (boot[a] - boot[b]), [2.5, 97.5]).tolist()}
    return out
