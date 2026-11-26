"""Closed-form per-row multipliers (arms C1, C2), held-out windows and checks, layer errors, alignment, bootstrap of the baseline plan."""
import copy
import json
import os
import subprocess
import sys
import tempfile
import unittest

import numpy as np
import torch

from rowtune import floor as FL
from rowtune.core import linears
from rowtune.data import windows_from_texts
from rowtune.formats import gguf
from rowtune.upgrade import effective_factors, student_from
from tests.fixtures import write_gguf_from
from tests.tiny import ByteTokenizer, make_model, make_windows

LEN = 16


def quantised_pair(seed=3):
    teacher = make_model(seed)
    d = tempfile.mkdtemp()
    src = os.path.join(d, "m.gguf")
    write_gguf_from(src, teacher)
    student = student_from(gguf.read(src), lambda: make_model(seed), torch.float32)
    return teacher, student, src, d


def capture_inputs(model, windows):
    """{linear name: [tokens, in]} of the inputs each decoder linear receives, over all windows (independent of rowtune.floor)."""
    got, hooks = {n: [] for n in linears(model)}, []
    for n, m in linears(model).items():
        hooks.append(m.register_forward_pre_hook(lambda mod, a, n=n: got[n].append(a[0].detach().reshape(-1, a[0].shape[-1]).double())))
    with torch.no_grad():
        for x in windows:
            model(input_ids=torch.tensor([x]))
    for h in hooks:
        h.remove()
    return {n: torch.cat(v) for n, v in got.items()}


class ClosedFormC1(unittest.TestCase):
    def test_matches_a_direct_float64_computation_and_is_the_least_squares_optimum(self):
        teacher, student, _, _ = quantised_pair()
        wins = make_windows(6, LEN, seed=1)
        mom = FL.moments_teacher_input(teacher, student, wins, "cpu")
        m = FL.closed_form_from_moments(mom)
        X = capture_inputs(teacher, wins)
        for n in linears(teacher):
            x = X[n]
            y = x @ linears(teacher)[n].weight.double().T
            yh = x @ linears(student)[n].weight.double().T
            want = (y * yh).sum(0) / (yh * yh).sum(0)
            self.assertTrue(torch.allclose(m[n].double(), want, rtol=1e-4), n)
            E1 = ((y - yh) ** 2).sum()
            Es = ((y - want * yh) ** 2).sum()
            self.assertLessEqual(Es.item(), E1.item() + 1e-9)
            for eps in (0.01, -0.01):  # least squares: any perturbation is worse
                self.assertGreaterEqual(((y - (want + eps) * yh) ** 2).sum().item(), Es.item() - 1e-9)

    def test_a_student_equal_to_the_teacher_gets_exactly_one(self):
        teacher = make_model(3)
        m = FL.closed_form_from_moments(FL.moments_teacher_input(teacher, copy.deepcopy(teacher), make_windows(4, LEN, seed=1), "cpu"))
        for v in m.values():
            self.assertTrue(torch.allclose(v, torch.ones_like(v), atol=1e-6))

    def test_zero_row_of_the_student_gets_one_not_nan(self):
        teacher, student, _, _ = quantised_pair()
        n = next(iter(linears(student)))
        with torch.no_grad():
            linears(student)[n].weight[0].zero_()
        m = FL.closed_form_from_moments(FL.moments_teacher_input(teacher, student, make_windows(3, LEN, seed=1), "cpu"))
        self.assertEqual(float(m[n][0]), 1.0)
        self.assertTrue(all(torch.isfinite(v).all() for v in m.values()))

    def test_no_gradients_are_needed_or_left(self):
        teacher, student, _, _ = quantised_pair()
        FL.moments_teacher_input(teacher, student, make_windows(2, LEN), "cpu")
        self.assertTrue(all(p.grad is None for p in student.parameters()))


class ClosedFormC2(unittest.TestCase):
    def test_first_layer_equals_c1_for_the_linears_that_read_the_embeddings(self):
        teacher, student, _, _ = quantised_pair()
        wins = make_windows(6, LEN, seed=1)
        c1 = FL.closed_form_from_moments(FL.moments_teacher_input(teacher, student, wins, "cpu"))
        c2 = FL.closed_form_c2(teacher, copy.deepcopy(student), wins, "cpu", torch.float32)
        for n in linears(teacher):
            if "layers.0.self_attn.q_proj" in n or "layers.0.self_attn.k_proj" in n or "layers.0.self_attn.v_proj" in n:
                self.assertTrue(torch.allclose(c1[n], c2[n], rtol=1e-5), n)
        self.assertEqual(set(c2), set(linears(teacher)))

    def test_later_layers_use_the_inputs_of_the_student_whose_earlier_layers_carry_their_multipliers(self):
        teacher, student, _, _ = quantised_pair()
        wins = make_windows(5, LEN, seed=2)
        c2 = FL.closed_form_c2(teacher, copy.deepcopy(student), wins, "cpu", torch.float32)
        ref = copy.deepcopy(student)  # reference: apply layer-0 multipliers, then read layer 1 with direct float64 maths
        with torch.no_grad():
            for n, m in linears(ref).items():
                if ".layers.0." in n:
                    m.weight.mul_(c2[n].to(m.weight.dtype)[:, None])
        Xs, Xt = capture_inputs(ref, wins), capture_inputs(teacher, wins)
        for n in linears(teacher):
            if ".layers.1." not in n:
                continue
            y = Xt[n] @ linears(teacher)[n].weight.double().T
            yt = Xs[n] @ linears(student)[n].weight.double().T  # student weight of layer 1 is still the file's
            want = (y * yt).sum(0) / (yt * yt).sum(0)
            self.assertTrue(torch.allclose(c2[n].double(), want, rtol=1e-3), n)

    def test_differs_from_c1_in_later_layers(self):
        teacher, student, _, _ = quantised_pair()
        wins = make_windows(6, LEN, seed=1)
        c1 = FL.closed_form_from_moments(FL.moments_teacher_input(teacher, student, wins, "cpu"))
        c2 = FL.closed_form_c2(teacher, copy.deepcopy(student), wins, "cpu", torch.float32)
        n = "model.layers.1.mlp.down_proj"
        self.assertFalse(torch.allclose(c1[n], c2[n], rtol=1e-6))


class LayerErrors(unittest.TestCase):
    def test_error_from_moments_equals_direct_sum_and_floor_fraction_is_below_one(self):
        teacher, student, _, _ = quantised_pair()
        wins = make_windows(6, LEN, seed=1)
        mom = FL.moments_teacher_input(teacher, student, wins, "cpu")
        m = FL.closed_form_from_moments(mom)
        X = capture_inputs(teacher, wins)
        for n in linears(teacher):
            y = X[n] @ linears(teacher)[n].weight.double().T
            yh = X[n] @ linears(student)[n].weight.double().T
            self.assertAlmostEqual(FL.layer_error(mom[n], m[n]), ((y - m[n].double() * yh) ** 2).sum().item(), delta=1e-4 * ((y - yh) ** 2).sum().item() + 1e-9)
            self.assertAlmostEqual(FL.layer_error(mom[n], torch.ones_like(m[n])), ((y - yh) ** 2).sum().item(), delta=1e-4 * ((y - yh) ** 2).sum().item() + 1e-9)
            self.assertLessEqual(FL.layer_error(mom[n], m[n]), FL.layer_error(mom[n], torch.ones_like(m[n])) * (1 + 1e-9))


class Alignment(unittest.TestCase):
    def test_spearman_pearson_sign_and_medians(self):
        a = np.array([0.1, -0.2, 0.3, 0.05, -0.4])
        r = FL.alignment(a, a * 2)
        self.assertAlmostEqual(r["spearman"], 1.0)
        self.assertAlmostEqual(r["pearson"], 1.0)
        self.assertEqual(r["same_sign"], 1.0)
        r = FL.alignment(a, -a)
        self.assertAlmostEqual(r["spearman"], -1.0)
        self.assertEqual(r["same_sign"], 0.0)
        self.assertAlmostEqual(r["median_abs_b"], float(np.median(np.abs(a))))
        self.assertAlmostEqual(r["median_abs_diff"], float(np.median(np.abs(2 * a))))

    def test_spearman_with_ties_uses_average_ranks(self):
        a = np.array([1.0, 1.0, 2.0, 3.0])
        b = np.array([1.0, 2.0, 2.0, 5.0])
        # ranks a: 1.5 1.5 3 4; b: 1 2.5 2.5 4 -> pearson of ranks
        ra, rb = np.array([1.5, 1.5, 3, 4]), np.array([1, 2.5, 2.5, 4])
        self.assertAlmostEqual(FL.alignment(a, b)["spearman"], float(np.corrcoef(ra, rb)[0, 1]))

    def test_exact_zero_has_no_sign_and_is_not_counted_as_same(self):
        r = FL.alignment(np.array([0.0, 0.1, -0.3]), np.array([0.1, 0.2, -0.1]))
        self.assertAlmostEqual(r["same_sign"], 2 / 3)


class AlignmentOfMultipliers(unittest.TestCase):
    """The closed form is not constrained to be positive, so log m is undefined for some rows; this version works on the multipliers."""

    def test_identical_positive_multipliers(self):
        m = np.array([0.9, 1.1, 1.3, 0.7])
        r = FL.alignment_m(m, m)
        self.assertAlmostEqual(r["spearman"], 1.0)
        self.assertAlmostEqual(r["pearson_log"], 1.0)
        self.assertEqual(r["same_sign"], 1.0)
        self.assertEqual(r["n_c_nonpositive"], 0)
        self.assertAlmostEqual(r["median_abs_log_diff"], 0.0)

    def test_nonpositive_rows_are_counted_ranked_and_signed_but_left_out_of_log_statistics(self):
        b = np.array([0.9, 1.1, 1.2, 0.8, 1.05])
        c = np.array([-0.5, 1.2, 1.3, 0.0, 0.95])
        r = FL.alignment_m(b, c)
        self.assertEqual(r["n_c_nonpositive"], 2)
        self.assertAlmostEqual(r["share_c_nonpositive"], 0.4)
        self.assertEqual(r["n_both_positive"], 3)
        # sign of (m - 1): b: - + + - + ; c: - + + - -  -> 4 of 5
        self.assertAlmostEqual(r["same_sign"], 0.8)
        # spearman over all five rows equals pearson of average ranks
        ra, rb = np.array([2, 4, 5, 1, 3.0]), np.array([1, 4, 5, 2, 3.0])
        self.assertAlmostEqual(r["spearman"], float(np.corrcoef(ra, rb)[0, 1]))
        lb, lc = np.log(b[[1, 2, 4]]), np.log(c[[1, 2, 4]])
        self.assertAlmostEqual(r["pearson_log"], float(np.corrcoef(lb, lc)[0, 1]))
        self.assertAlmostEqual(r["median_abs_log_c"], float(np.median(np.abs(lc))))
        self.assertAlmostEqual(r["median_abs_log_diff"], float(np.median(np.abs(lb - lc))))
        self.assertAlmostEqual(r["median_abs_c_minus_1"], float(np.median(np.abs(c - 1))))


class HeldOut(unittest.TestCase):
    def texts(self, n=400):
        rng = np.random.default_rng(0)  # distinct random documents, so no two windows coincide by chance
        return ["".join(chr(97 + int(c)) for c in rng.integers(0, 26, size=250 + i % 7)) for i in range(n)] + ["short"] * 5

    def test_tuning_windows_are_exactly_the_tools(self):
        tok, T = ByteTokenizer(), self.texts()
        w = FL.split_windows(tok, T, n_tune=30, length=16, tune_seed=1, n_held=10, held_seed=12345)
        self.assertEqual(w["tune"], windows_from_texts(tok, T, 30, 16, 1))

    def test_held_out_windows_use_only_documents_the_tuning_stream_never_touched(self):
        tok, T = ByteTokenizer(), self.texts()
        w = FL.split_windows(tok, T, n_tune=30, length=16, tune_seed=1, n_held=10, held_seed=12345)
        self.assertEqual(len(w["held"]), 10)
        self.assertFalse(set(w["tune_docs"]) & set(w["held_docs"]))
        self.assertTrue(all(len(x) == 16 for x in w["held"]))
        self.assertFalse({tuple(x) for x in w["tune"]} & {tuple(x) for x in w["held"]})
        rep = FL.check_disjoint(w)
        self.assertTrue(rep["disjoint"])
        self.assertEqual(rep["shared_docs"], 0)

    def test_held_seed_changes_the_held_windows_and_short_documents_are_dropped(self):
        tok, T = ByteTokenizer(), self.texts()
        a = FL.split_windows(tok, T, 30, 16, 1, 10, 12345)
        b = FL.split_windows(tok, T, 30, 16, 1, 10, 7)
        self.assertNotEqual(a["held"], b["held"])
        self.assertEqual(a["tune"], b["tune"])
        self.assertTrue(all(i < len([t for t in T if len(t) > 200]) for i in a["held_docs"]))

    def test_check_fails_loudly_when_the_sets_overlap(self):
        tok, T = ByteTokenizer(), self.texts()
        w = FL.split_windows(tok, T, 30, 16, 1, 10, 12345)
        w["held_docs"] = list(w["held_docs"]) + [w["tune_docs"][0]]
        with self.assertRaises(FL.DisjointError):
            FL.check_disjoint(w)
        w2 = FL.split_windows(tok, T, 30, 16, 1, 10, 12345)
        w2["held"][0] = list(w2["tune"][0])
        with self.assertRaises(FL.DisjointError):
            FL.check_disjoint(w2)

    def test_not_enough_text_for_the_held_windows_is_an_error(self):
        tok = ByteTokenizer()
        with self.assertRaises(ValueError):
            FL.split_windows(tok, self.texts(60), 30, 16, 1, 10000, 12345)


class BasisCheck(unittest.TestCase):
    def test_passes_for_rounded_weights_and_fails_above_half(self):
        teacher, student, _, _ = quantised_pair()
        r = FL.basis_check(teacher, student)
        self.assertTrue(r["PASS"])
        self.assertEqual(set(r["ratio"]), set(linears(teacher)))
        self.assertLess(r["max"], 0.5)
        n = next(iter(linears(student)))
        with torch.no_grad():
            linears(student)[n].weight.mul_(-1.0)  # a basis change (sign flip) is far from the teacher
        r = FL.basis_check(teacher, student)
        self.assertFalse(r["PASS"])
        self.assertEqual(r["failed"], [n])


class HeldoutKL(unittest.TestCase):
    def test_zero_for_identical_models_positive_otherwise_and_matches_formula(self):
        teacher, student, _, _ = quantised_pair()
        wins = make_windows(4, LEN, seed=9)
        self.assertTrue(np.allclose(FL.heldout_kl(teacher, teacher, wins, "cpu"), 0, atol=1e-7))
        k = FL.heldout_kl(teacher, student, wins, "cpu")
        self.assertEqual(k.shape, (4,))
        self.assertTrue((k > 0).all())
        x = torch.tensor([wins[0]])
        with torch.no_grad():
            p = torch.softmax(teacher(input_ids=x).logits.double(), -1)
            q = torch.softmax(student(input_ids=x).logits.double(), -1)
        self.assertAlmostEqual(float(k[0]), float((p * (p.log() - q.log())).sum(-1).mean()), places=5)


class Bootstrap(unittest.TestCase):
    def synth(self):
        rng = np.random.default_rng(5)
        items = {}
        for name, p in (("bf16", 0.8), ("A", 0.6), ("B", 0.7), ("C", 0.62)):
            items[name] = {t: {"acc": (rng.random(n) < p).astype(float).tolist()} for t, n in zip(FL.TASKS, (300, 200, 300, 250, 200, 300))}
        return items

    def test_pair_differences_equal_the_released_script(self):
        items = self.synth()
        d = tempfile.mkdtemp()
        args = [sys.executable, os.path.join(os.path.dirname(FL.__file__), "..", "floor", "paired_ci.py"), os.path.join(d, "o.json")]
        for m, it in items.items():
            p = os.path.join(d, m + ".json")
            json.dump({"methods": {"k": {"items": it}}}, open(p, "w"))
            args += ["--v", f"{m}={p}:k"]
        args += ["--pair", "B,A", "--pair", "C,A", "--pair", "B,C"]
        subprocess.run(args, check=True, capture_output=True)
        ref = json.load(open(os.path.join(d, "o.json")))
        acc = {m: {t: v["acc"] for t, v in it.items()} for m, it in items.items()}
        res = FL.accuracy_bootstrap(acc, [("B", "A"), ("C", "A"), ("B", "C")], ratios=[("C", "B", "A")])
        for k, v in ref["pairs"].items():
            self.assertAlmostEqual(res["pairs"][k]["diff"], v["diff"], places=9)
            self.assertTrue(np.allclose(res["pairs"][k]["ci"], v["ci"], atol=1e-9))
        for m, v in ref["versions"].items():
            self.assertAlmostEqual(res["versions"][m]["kept"], v["kept"], places=9)
        r = res["ratios"]["C/B/A"]
        self.assertAlmostEqual(r["point"], res["pairs"]["C,A"]["diff"] / res["pairs"]["B,A"]["diff"], places=9)
        self.assertLess(r["ci"][0], r["point"])
        self.assertGreater(r["ci"][1], r["point"])

    def test_kl_bootstrap_differences_and_ratio(self):
        rng = np.random.default_rng(1)
        kl = {"A": 0.30 + 0.02 * rng.standard_normal(200), "B": 0.20 + 0.02 * rng.standard_normal(200), "C": 0.29 + 0.02 * rng.standard_normal(200)}
        res = FL.kl_bootstrap(kl, [("B", "A")], ratios=[("C", "B", "A")])
        self.assertAlmostEqual(res["means"]["A"], float(kl["A"].mean()))
        self.assertAlmostEqual(res["pairs"]["B,A"]["diff"], float((kl["B"] - kl["A"]).mean()))
        r = res["ratios"]["C/B/A"]
        self.assertAlmostEqual(r["point"], float((kl["A"].mean() - kl["C"].mean()) / (kl["A"].mean() - kl["B"].mean())))
        self.assertLess(r["ci"][0], r["point"] + 1e-12)
        res2 = FL.kl_bootstrap(kl, [("B", "A")], ratios=[("C", "B", "A")])
        self.assertEqual(res["pairs"]["B,A"]["ci"], res2["pairs"]["B,A"]["ci"])  # seeded


class FileCheck(unittest.TestCase):
    def test_closed_form_factors_written_with_the_tools_writer_keep_every_code_byte(self):
        teacher, student, src, d = quantised_pair()
        m = FL.closed_form_from_moments(FL.moments_teacher_input(teacher, student, make_windows(4, LEN, seed=1), "cpu"))
        f = {n + ".weight": v for n, v in effective_factors(m, torch.bfloat16).items()}
        out = os.path.join(d, "c1.gguf")
        gguf.write(src, out, f)
        rep = gguf.verify(src, out)
        self.assertTrue(rep["PASS"] and rep["codes_identical"], rep)


if __name__ == "__main__":
    unittest.main()
