import os
import tempfile
import unittest

import numpy as np
import torch

from rowtune.core import linears
from rowtune.formats import ct, gguf, openvino
from rowtune.upgrade import effective_factors, student_from, upgrade
from tests.fixtures import write_ct_from, write_gguf_from, write_ir_from
from tests.test_tune import mean_kl
from tests.tiny import make_model, make_windows

LEN = 16


def build(kind, path, model):
    return {"gguf": lambda: write_gguf_from(path, model), "openvino": lambda: write_ir_from(path, model),
            "ct": lambda: write_ct_from(path, model, 0, "F16")}[kind]()


MODULE = {"gguf": gguf, "openvino": openvino, "ct": ct}
NAME = {"gguf": "m.gguf", "openvino": "ir", "ct": "ckpt"}


class EffectiveFactors(unittest.TestCase):
    def test_rounded_to_the_student_dtype_like_the_tuned_model_used_them(self):
        c = {"a": torch.tensor([1.00391, 0.99, 1.2])}
        f = effective_factors(c, torch.bfloat16)["a"]
        self.assertTrue(np.array_equal(f, c["a"].to(torch.bfloat16).float().numpy()))
        self.assertTrue(np.array_equal(effective_factors(c, torch.float32)["a"], c["a"].numpy()))


class Upgrade(unittest.TestCase):
    kinds = ("gguf", "openvino", "ct")

    def run_kind(self, kind):
        d = tempfile.mkdtemp()
        teacher = make_model(3)
        src, out = os.path.join(d, NAME[kind]), os.path.join(d, "out_" + NAME[kind])
        build(kind, src, teacher)
        info = upgrade(src, out, teacher, lambda: make_model(3), make_windows(60, LEN, seed=1), lr=1e-2, device="cpu", dtype=torch.float32)
        return d, src, out, teacher, info

    def test_codes_identical_scales_changed_size_same_and_file_loads_again(self):
        for kind in self.kinds:
            with self.subTest(kind):
                _, src, out, _, info = self.run_kind(kind)
                M = MODULE[kind]
                a, b = M.read(src), M.read(out)
                self.assertEqual(set(a.linears), set(b.linears))
                for k in a.linears:
                    self.assertTrue(np.array_equal(a.linears[k].codes, b.linears[k].codes), k)
                self.assertGreater(sum(int((a.linears[k].scale != b.linears[k].scale).any(1).sum()) for k in a.linears), 0)
                rep = info["file_check"]
                self.assertTrue(rep["PASS"], rep)
                self.assertTrue(rep["codes_identical"] and rep["same_size"] and rep["same_structure"])
                self.assertEqual(rep, M.verify(src, out))

    def test_the_upgraded_file_is_closer_to_the_teacher_than_the_input(self):
        held = make_windows(8, LEN, seed=99)
        for kind in self.kinds:
            with self.subTest(kind):
                _, src, out, teacher, _ = self.run_kind(kind)
                M = MODULE[kind]
                s_in = student_from(M.read(src), lambda: make_model(3), torch.float32)
                s_out = student_from(M.read(out), lambda: make_model(3), torch.float32)
                self.assertLess(mean_kl(teacher, s_out, held), mean_kl(teacher, s_in, held))

    def test_student_takes_linears_and_norms_from_the_file_and_the_rest_from_the_teacher_model(self):
        d = tempfile.mkdtemp()
        teacher = make_model(3)
        src = os.path.join(d, "ir")
        write_ir_from(src, teacher)
        other = lambda: make_model(11)  # noqa: E731  a model whose own weights differ from the file's
        s = student_from(openvino.read(src), other, torch.float32)
        q = openvino.read(src)
        for n, l in linears(s).items():
            self.assertTrue(np.allclose(l.weight.detach().numpy(), q.linears[n + ".weight"].dequant()), n)
        self.assertTrue(np.allclose(s.model.layers[0].input_layernorm.weight.detach().numpy(), teacher.model.layers[0].input_layernorm.weight.detach().numpy()))
        self.assertTrue(torch.equal(s.lm_head.weight, other().lm_head.weight))

    def test_missing_norm_in_ir_needs_a_unit_gain_in_the_teacher(self):
        d = tempfile.mkdtemp()
        teacher = make_model(3)  # gains are not 1
        src = os.path.join(d, "ir")
        write_ir_from(src, teacher, norms=False)
        with self.assertRaises(ValueError):
            upgrade(src, os.path.join(d, "o"), teacher, lambda: make_model(3), make_windows(2, LEN), device="cpu", dtype=torch.float32)

    def test_linears_stored_in_another_type_stay_untouched_and_are_not_tuned(self):
        d = tempfile.mkdtemp()
        teacher = make_model(3)
        other = "model.layers.0.mlp.up_proj.weight"
        src, out = os.path.join(d, "m.gguf"), os.path.join(d, "o.gguf")
        write_gguf_from(src, teacher, other_type_for=other)
        info = upgrade(src, out, teacher, lambda: make_model(3), make_windows(6, LEN, seed=1), lr=1e-2, device="cpu", dtype=torch.float32)
        self.assertNotIn(other.removesuffix(".weight"), info["scales"])
        self.assertEqual(info["skipped"], [other])
        self.assertTrue(info["file_check"]["PASS"])


if __name__ == "__main__":
    unittest.main()
