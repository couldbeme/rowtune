import json
import os
import tempfile
import unittest
from unittest.mock import patch

import torch

from rowtune import cli
from rowtune.formats import ct, gguf, openvino
from rowtune.core import linears, rtn_student
from rowtune.data import read_text_file, windows_from_texts
from rowtune.upgrade import student_from
from tests.fixtures import write_ct_from, write_gguf_from, write_ir_from
from tests.test_tune import mean_kl
from tests.tiny import ByteTokenizer, make_model


def fake_loader(name, dtype=None, device="cpu", compressed=False):
    m = make_model(3)
    if name == "student-ckpt":
        rtn_student(m)
    return m


def fake_save(model, tokenizer, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(out_dir, "weights.pt"))


class RoundCommand(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.text = os.path.join(self.dir, "text.txt")
        with open(self.text, "w", encoding="utf-8") as f:
            f.write("\n<<<DOC>>>\n".join(("some public text " * 30) + str(i) for i in range(30)))
        self.out = os.path.join(self.dir, "out")

    def run_cli(self, *extra):
        argv = ["round", "--model", "teacher", "--text-file", self.text, "--out", self.out, "--steps", "8", "--length", "16", "--device", "cpu",
                "--teacher-device", "cpu", "--lr", "0.01", "--dtype", "fp32", *extra]
        with patch("rowtune.cli.hf.load_causal_lm", side_effect=fake_loader), patch("rowtune.cli.hf.load_tokenizer", return_value=ByteTokenizer()), \
                patch("rowtune.cli.hf.save_checkpoint", side_effect=fake_save):
            return cli.main(argv)

    def read_info(self):
        with open(os.path.join(self.out, "tuning.json")) as f:
            return json.load(f)

    def test_defaults_are_the_documented_ones(self):
        a = cli.parse_round_args(["--model", "m", "--text-file", "t", "--out", "o"])
        self.assertEqual((a.steps, a.length, a.lr, a.seed, a.dtype, a.group_size), (600, 1024, 1e-3, 1, "bf16", 0))

    def test_end_to_end_from_original_weights(self):
        self.assertEqual(self.run_cli(), 0)
        self.assertTrue(os.path.exists(os.path.join(self.out, "weights.pt")))
        info = self.read_info()
        self.assertEqual(info["steps"], 8)
        self.assertEqual(info["rows_not_pure_rescale"], 0)
        self.assertEqual(info["start"], "round-to-nearest, 4 bits, one scale per row")
        scales = torch.load(os.path.join(self.out, "row_scales.pt"))
        self.assertEqual(set(scales), set(linears(make_model(3))))
        w = torch.load(os.path.join(self.out, "weights.pt"))
        rounded = make_model(3)
        rtn_student(rounded)
        for n, l in linears(rounded).items():
            self.assertTrue(torch.allclose(w[n + ".weight"], l.weight * scales[n][:, None], rtol=1e-6, atol=1e-7))
        self.assertTrue(any(not torch.allclose(s, torch.ones_like(s)) for s in scales.values()))

    def test_end_to_end_from_a_local_4bit_checkpoint(self):
        self.assertEqual(self.run_cli("--student", "student-ckpt"), 0)
        info = self.read_info()
        self.assertEqual(info["start"], "checkpoint student-ckpt")
        self.assertEqual(info["rows_not_pure_rescale"], 0)

    def test_does_not_overwrite_existing_output(self):
        os.makedirs(self.out)
        with open(os.path.join(self.out, "keep.txt"), "w") as f:
            f.write("x")
        with self.assertRaises(SystemExit):
            self.run_cli()
        self.assertTrue(os.path.exists(os.path.join(self.out, "keep.txt")))


if __name__ == "__main__":
    unittest.main()


class UpgradeCommand(unittest.TestCase):
    """rowtune <input> -o <output> [--teacher M] on tiny synthetic files of each format; hugging-face calls are replaced by the tiny model."""

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.text = os.path.join(self.dir, "text.txt")
        with open(self.text, "w", encoding="utf-8") as f:
            f.write("\n<<<DOC>>>\n".join(("some public text " * 30) + str(i) for i in range(30)))
        self.teacher = make_model(3)
        self.made = {"gguf": ("m.gguf", lambda p: write_gguf_from(p, self.teacher)), "openvino": ("ir", lambda p: write_ir_from(p, self.teacher)),
                     "ct": ("ckpt", lambda p: write_ct_from(p, self.teacher, 32, "F16"))}

    def run_cli(self, argv):
        with patch("rowtune.cli.hf.load_causal_lm", side_effect=fake_loader) as ll, patch("rowtune.cli.hf.load_tokenizer", return_value=ByteTokenizer()), \
                patch("rowtune.cli.hf.save_checkpoint", side_effect=fake_save):
            rc = cli.main(argv)
        return rc, ll

    def common(self):
        return ["--text-file", self.text, "--steps", "40", "--length", "16", "--device", "cpu", "--lr", "0.01", "--dtype", "fp32"]

    def test_each_format_end_to_end(self):
        held = windows_from_texts(ByteTokenizer(), read_text_file(self.text), 8, 16, 99)  # same kind of text, other windows
        for kind, (name, build) in self.made.items():
            with self.subTest(kind):
                src, out = os.path.join(self.dir, name), os.path.join(self.dir, "out-" + name)
                build(src)
                rc, loads = self.run_cli([src, "-o", out, "--teacher", "tiny-teacher", *self.common()])
                self.assertEqual(rc, 0)
                self.assertEqual(loads.call_args_list[0].args[0], "tiny-teacher")
                M = {"gguf": gguf, "openvino": openvino, "ct": ct}[kind]
                rep = M.verify(src, out)
                self.assertTrue(rep["PASS"], rep)
                self.assertTrue(rep["codes_identical"] and rep["same_size"])
                self.assertGreater(rep["scale_tensors_changed"], 0)
                self.assertTrue(os.path.exists(out))
                with open(out + ".rowtune.json") as f:
                    info = json.load(f)
                self.assertEqual(info["format"], kind)
                self.assertTrue(info["file_check"]["PASS"])
                s_in = student_from(M.read(src), lambda: make_model(3), torch.float32)
                s_out = student_from(M.read(out), lambda: make_model(3), torch.float32)
                self.assertLess(mean_kl(self.teacher, s_out, held), mean_kl(self.teacher, s_in, held))

    def test_teacher_defaults_to_the_model_named_in_the_input_config(self):
        src, out = os.path.join(self.dir, "ir"), os.path.join(self.dir, "out")
        write_ir_from(src, self.teacher)
        rc, loads = self.run_cli([src, "-o", out, *self.common()])
        self.assertEqual(rc, 0)
        self.assertEqual(loads.call_args_list[0].args[0], "tiny-teacher")  # config.json _name_or_path

    def test_gguf_without_teacher_is_an_error_that_writes_nothing(self):
        src, out = os.path.join(self.dir, "m.gguf"), os.path.join(self.dir, "out.gguf")
        write_gguf_from(src, self.teacher)
        with self.assertRaises(SystemExit):
            self.run_cli([src, "-o", out, *self.common()])
        self.assertFalse(os.path.exists(out))

    def test_refuses_an_existing_output_and_an_unknown_input(self):
        src, out = os.path.join(self.dir, "m.gguf"), os.path.join(self.dir, "out.gguf")
        write_gguf_from(src, self.teacher)
        open(out, "wb").write(b"keep")
        with self.assertRaises(SystemExit):
            self.run_cli([src, "-o", out, "--teacher", "t", *self.common()])
        self.assertEqual(open(out, "rb").read(), b"keep")
        junk = os.path.join(self.dir, "junk.bin")
        open(junk, "wb").write(b"junk")
        with self.assertRaises(SystemExit):
            self.run_cli([junk, "-o", os.path.join(self.dir, "o2"), "--teacher", "t", *self.common()])

    def test_defaults_match_the_documented_ones(self):
        a = cli.parse_upgrade_args(["in", "-o", "out"])
        self.assertEqual((a.steps, a.length, a.lr, a.seed, a.dtype, a.teacher), (600, 1024, 1e-3, 1, "bf16", None))
