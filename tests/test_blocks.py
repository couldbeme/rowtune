"""Per-block scales (arm D of the baseline plan): one trainable multiplier per 32-weight block, GGUF Q4_0 only."""
import json
import os
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
import torch

from rowtune import cli
from rowtune.core import BlockScale, attach_block_scales, groups_not_pure_rescale, linears, rtn_student
from rowtune.formats import gguf, openvino
from rowtune.tune import tune
from rowtune.upgrade import student_from, upgrade
from tests.fixtures import write_gguf_from, write_ir_from
from tests.test_cli import fake_loader, fake_save
from tests.test_tune import mean_kl
from tests.tiny import ByteTokenizer, make_model, make_windows

LEN = 16


class BlockScaleModule(unittest.TestCase):
    def test_starts_at_one_and_expands_each_value_over_its_32_columns(self):
        rs = BlockScale(3, 64, torch.device("cpu"), 32)
        self.assertEqual(tuple(rs.c.shape), (3, 2))
        self.assertTrue(torch.equal(rs.c, torch.ones(3, 2)))
        W = torch.randn(3, 64)
        self.assertTrue(torch.equal(rs(W), W))
        with torch.no_grad():
            rs.c[1, 1] = 2.0
        out = rs(W)
        self.assertTrue(torch.equal(out[1, 32:], 2 * W[1, 32:]))
        self.assertTrue(torch.equal(out[1, :32], W[1, :32]))
        self.assertTrue(torch.equal(out[0], W[0]))

    def test_columns_not_divisible_by_the_block_are_an_error(self):
        with self.assertRaises(ValueError):
            BlockScale(2, 40, torch.device("cpu"), 32)

    def test_attach_puts_one_per_decoder_linear_and_freezes_the_rest(self):
        m = make_model(1)
        mods = attach_block_scales(m, 32)
        self.assertEqual(set(mods), set(linears(m)))
        self.assertEqual(sum(rs.c.numel() for rs in mods.values()), sum(l.weight.numel() // 32 for l in linears(m).values()))
        self.assertTrue(all(not p.requires_grad for n, p in m.named_parameters() if not n.endswith(".c")))


class TuneBlocks(unittest.TestCase):
    def setup(self, seed=3):
        teacher, student = make_model(seed), make_model(seed)
        rtn_student(student, 4, 32)
        return teacher, student

    def test_trains_one_value_per_block_and_kl_falls(self):
        teacher, student = self.setup()
        held = make_windows(8, LEN, seed=99)
        before = mean_kl(teacher, student, held)
        info = tune(student, teacher, make_windows(60, LEN, seed=1), lr=1e-2, device="cpu", block=32)
        self.assertEqual(info["n_scales"], sum(l.weight.numel() // 32 for l in linears(student).values()))
        for n, c in info["scales"].items():
            self.assertEqual(tuple(c.shape), (linears(student)[n].weight.shape[0], linears(student)[n].weight.shape[1] // 32))
        self.assertLess(info["kl_last10pct"], info["kl_first10pct"])
        self.assertLess(mean_kl(teacher, student, held), before)

    def test_default_is_unchanged_one_value_per_row(self):
        teacher, student = self.setup()
        info = tune(student, teacher, make_windows(5, LEN, seed=1), lr=1e-2, device="cpu")
        self.assertEqual(info["n_scales"], sum(l.weight.shape[0] for l in linears(student).values()))

    def test_integer_grid_is_unchanged_blockwise_but_not_rowwise(self):
        teacher, student = self.setup()
        info = tune(student, teacher, make_windows(30, LEN, seed=1), lr=1e-2, device="cpu", block=32)
        self.assertEqual(info["rows_not_pure_rescale"], 0)  # counted per block in this mode

    def test_groups_not_pure_rescale_counts_a_block_whose_ratio_is_not_constant(self):
        W0 = torch.randn(4, 64) + 3
        W1 = W0.clone()
        W1[:, :32] *= 1.5
        W1[:, 32:] *= 0.7
        self.assertEqual(groups_not_pure_rescale(W1, W0, 32), 0)
        self.assertGreater(groups_not_pure_rescale(W1, W0, 64), 0)
        W1[2, 5] *= 1.3
        self.assertEqual(groups_not_pure_rescale(W1, W0, 32), 1)


class GgufBlockWrite(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.model = make_model(5)
        self.src = os.path.join(self.dir, "m.gguf")
        write_gguf_from(self.src, self.model)
        self.out = os.path.join(self.dir, "o.gguf")

    def test_per_block_factors_multiply_each_block_scale_and_leave_codes(self):
        q = gguf.read(self.src)
        rng = np.random.default_rng(0)
        f = {n: (1 + 0.3 * rng.random(l.scale.shape)).astype(np.float32) for n, l in q.linears.items()}
        gguf.write(self.src, self.out, f)
        rep = gguf.verify(self.src, self.out)
        self.assertTrue(rep["PASS"], rep)
        q2 = gguf.read(self.out)
        for n, l in q.linears.items():
            self.assertTrue(np.array_equal(q2.linears[n].codes, l.codes))
            want = (l.scale.astype(np.float16) * 0 + l.scale).astype(np.float32) * f[n]
            self.assertTrue(np.allclose(q2.linears[n].scale, want, rtol=2e-3), n)
        self.assertGreater(rep["scale_tensors_changed"], 0)

    def test_row_factors_still_work_and_equal_block_factors_repeated(self):
        q = gguf.read(self.src)
        rng = np.random.default_rng(1)
        row = {n: (1 + 0.3 * rng.random(l.scale.shape[0])).astype(np.float32) for n, l in q.linears.items()}
        blk = {n: np.repeat(c[:, None], q.linears[n].scale.shape[1], 1) for n, c in row.items()}
        o2 = os.path.join(self.dir, "o2.gguf")
        gguf.write(self.src, self.out, row)
        gguf.write(self.src, o2, blk)
        self.assertEqual(open(self.out, "rb").read(), open(o2, "rb").read())

    def test_wrong_shapes_are_errors_and_write_nothing(self):
        q = gguf.read(self.src)
        n = next(iter(q.linears))
        rows, groups = q.linears[n].scale.shape
        for bad in (np.ones((rows, groups + 1), np.float32), np.ones((rows + 1, groups), np.float32), np.ones((rows, groups, 1), np.float32)):
            with self.assertRaises(ValueError):
                gguf.write(self.src, self.out, {n: bad})
            self.assertFalse(os.path.exists(self.out))
            self.assertFalse(os.path.exists(self.out + ".part"))

    def test_openvino_rejects_block_factors(self):
        d = os.path.join(self.dir, "ir")
        write_ir_from(d, self.model)
        q = openvino.read(d)
        n = next(iter(q.linears))
        with self.assertRaises(ValueError):
            openvino.write(d, os.path.join(self.dir, "irout"), {n: np.ones((q.linears[n].codes.shape[0], 2), np.float32)})


class UpgradeBlocks(unittest.TestCase):
    def test_gguf_upgrade_per_block_passes_the_code_check_and_beats_the_input(self):
        d = tempfile.mkdtemp()
        teacher = make_model(3)
        src, out = os.path.join(d, "m.gguf"), os.path.join(d, "o.gguf")
        write_gguf_from(src, teacher)
        info = upgrade(src, out, teacher, lambda: make_model(3), make_windows(40, LEN, seed=1), lr=1e-2, device="cpu", dtype=torch.float32, block=32)
        self.assertTrue(info["file_check"]["PASS"], info["file_check"])
        self.assertTrue(info["file_check"]["codes_identical"])
        held = make_windows(8, LEN, seed=99)
        s_in = student_from(gguf.read(src), lambda: make_model(3), torch.float32)
        s_out = student_from(gguf.read(out), lambda: make_model(3), torch.float32)
        self.assertLess(mean_kl(teacher, s_out, held), mean_kl(teacher, s_in, held))
        self.assertEqual(info["n_scales"], sum(l.scale.size for l in gguf.read(src).linears.values()))

    def test_block_mode_on_a_non_gguf_file_is_refused_before_tuning(self):
        d = tempfile.mkdtemp()
        teacher = make_model(3)
        src = os.path.join(d, "ir")
        write_ir_from(src, teacher)
        with self.assertRaises(ValueError):
            upgrade(src, os.path.join(d, "o"), teacher, lambda: make_model(3), make_windows(3, LEN), lr=1e-2, device="cpu", dtype=torch.float32, block=32)
        self.assertFalse(os.path.exists(os.path.join(d, "o")))


class CliPerBlock(unittest.TestCase):
    def test_flag_parses_and_runs_end_to_end(self):
        d = tempfile.mkdtemp()
        text = os.path.join(d, "t.txt")
        with open(text, "w", encoding="utf-8") as f:
            f.write("\n<<<DOC>>>\n".join(("some public text " * 30) + str(i) for i in range(30)))
        teacher = make_model(3)
        src, out = os.path.join(d, "m.gguf"), os.path.join(d, "o.gguf")
        write_gguf_from(src, teacher)
        self.assertEqual(cli.parse_upgrade_args([src, "-o", out]).per_block, 0)
        with patch("rowtune.cli.hf.load_causal_lm", side_effect=fake_loader), patch("rowtune.cli.hf.load_tokenizer", return_value=ByteTokenizer()), \
                patch("rowtune.cli.hf.save_checkpoint", side_effect=fake_save):
            rc = cli.main([src, "-o", out, "--teacher", "t", "--text-file", text, "--steps", "20", "--length", "16", "--device", "cpu", "--lr", "0.01",
                           "--dtype", "fp32", "--per-block", "32"])
        self.assertEqual(rc, 0)
        info = json.load(open(out + ".rowtune.json"))
        self.assertTrue(info["file_check"]["PASS"])
        self.assertEqual(info["per_block"], 32)
        self.assertEqual(info["n_scales"], sum(l.scale.size for l in gguf.read(src).linears.values()))


if __name__ == "__main__":
    unittest.main()
