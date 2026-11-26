import unittest

import torch
import torch.nn.functional as F

from rowtune import DEFAULT_LENGTH, DEFAULT_LR, DEFAULT_STEPS
from rowtune.core import linears, quantize_rtn, rtn_student
from rowtune.tune import tune
from tests.tiny import make_model, make_windows

LEN = 16


def mean_kl(teacher, student, windows):
    out = []
    with torch.no_grad():
        for x in windows:
            x = torch.tensor([x])
            lt = F.log_softmax(teacher(input_ids=x).logits.float(), -1)
            ls = F.log_softmax(student(input_ids=x).logits.float(), -1)
            out.append((lt.exp() * (lt - ls)).sum(-1).mean().item())
    return sum(out) / len(out)


def setup(seed=3):
    teacher = make_model(seed)
    student = make_model(seed)
    rtn_student(student)
    return teacher, student


class Defaults(unittest.TestCase):
    def test_documented_defaults(self):
        self.assertEqual((DEFAULT_STEPS, DEFAULT_LENGTH, DEFAULT_LR), (600, 1024, 1e-3))


class TuningLowersKl(unittest.TestCase):
    def test_kl_to_teacher_falls_on_held_out_windows(self):
        teacher, student = setup()
        held = make_windows(8, LEN, seed=99)
        before = mean_kl(teacher, student, held)
        tune(student, teacher, make_windows(60, LEN, seed=1), lr=1e-2, device="cpu")
        after = mean_kl(teacher, student, held)
        self.assertLess(after, before)

    def test_reported_training_kl_falls(self):
        teacher, student = setup()
        info = tune(student, teacher, make_windows(60, LEN, seed=1), lr=1e-2, device="cpu")
        self.assertLess(info["kl_last10pct"], info["kl_first10pct"])
        self.assertEqual(info["steps"], 60)
        self.assertEqual(info["n_scales"], sum(l.weight.shape[0] for l in linears(student).values()))


class Subset(unittest.TestCase):
    def test_tuning_a_subset_leaves_the_other_linears_bit_identical(self):
        teacher, student = setup()
        chosen = [n for n in linears(student) if "layers.0" in n]
        before = {n: l.weight.detach().clone() for n, l in linears(student).items()}
        info = tune(student, teacher, make_windows(10, LEN, seed=1), lr=1e-2, device="cpu", only=chosen)
        self.assertEqual(set(info["scales"]), set(chosen))
        self.assertEqual(info["n_scales"], sum(linears(student)[n].weight.shape[0] for n in chosen))
        for n, l in linears(student).items():
            if n not in chosen:
                self.assertTrue(torch.equal(l.weight, before[n]), n)


class OnlyTheRowScaleMoves(unittest.TestCase):
    def test_codes_unchanged_only_per_row_scale_moves(self):
        teacher, student = setup()
        deq, codes, scale = {}, {}, {}
        for n, l in linears(student).items():
            deq[n], codes[n], scale[n] = quantize_rtn(teacher.get_submodule(n).weight.detach().clone(), 4, 0)
        w_start = {n: l.weight.detach().clone() for n, l in linears(student).items()}
        info = tune(student, teacher, make_windows(40, LEN, seed=1), lr=1e-2, device="cpu")
        self.assertEqual(info["rows_not_pure_rescale"], 0)
        c = info["scales"]
        moved = 0
        for n, l in linears(student).items():
            W = l.weight.detach()
            self.assertTrue(torch.allclose(W, w_start[n] * c[n][:, None], rtol=1e-6, atol=1e-7))
            # the integer grid under the new per-row scale is the original one
            self.assertTrue(torch.equal(torch.round(W / (scale[n] * c[n][:, None])).clamp(-8, 7).to(torch.int8), codes[n]))
            moved += int((c[n] != 1).sum())
        self.assertGreater(moved, 0)

    def test_structure_and_shapes_unchanged(self):
        teacher, student = setup()
        keys = {k: (tuple(v.shape), v.dtype) for k, v in student.state_dict().items()}
        emb = student.model.embed_tokens.weight.clone()
        head = student.lm_head.weight.clone()
        t_before = {k: v.clone() for k, v in teacher.state_dict().items()}
        tune(student, teacher, make_windows(10, LEN, seed=1), lr=1e-2, device="cpu")
        self.assertEqual({k: (tuple(v.shape), v.dtype) for k, v in student.state_dict().items()}, keys)
        self.assertTrue(torch.equal(student.model.embed_tokens.weight, emb))
        self.assertTrue(torch.equal(student.lm_head.weight, head))
        for k, v in teacher.state_dict().items():
            self.assertTrue(torch.equal(v, t_before[k]))
        self.assertFalse(any(isinstance(m, torch.nn.Module) and hasattr(m, "parametrizations") for m in student.modules()))

    def test_student_weights_stay_frozen(self):
        teacher, student = setup()
        tune(student, teacher, make_windows(3, LEN, seed=1), lr=1e-2, device="cpu")
        self.assertFalse(any(p.requires_grad for p in student.parameters()))


if __name__ == "__main__":
    unittest.main()
