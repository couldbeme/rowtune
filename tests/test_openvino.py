import os
import re
import tempfile
import unittest

import numpy as np

from rowtune.formats import openvino as O
from tests.fixtures import deq, write_ir_from
from tests.tiny import make_model


def consts(d):
    """{layer name: (element_type, raw bytes)} of every Const of an IR, parsed with regular expressions (independent of rowtune)."""
    xml = open(os.path.join(d, "openvino_model.xml")).read()
    blob = open(os.path.join(d, "openvino_model.bin"), "rb").read()
    out = {}
    for m in re.finditer(r'<layer id="\d+" name="([^"]+)" type="Const".*?<data element_type="(\w+)" shape="[^"]*" offset="(\d+)" size="(\d+)"', xml):
        n, et, off, size = m.group(1), m.group(2), int(m.group(3)), int(m.group(4))
        out[n] = (et, blob[off:off + size])
    return out


class Fixture(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.model = make_model(6)
        self.src = os.path.join(self.dir, "ir")
        self.q = write_ir_from(self.src, self.model)
        self.out = os.path.join(self.dir, "ir_out")


class Read(Fixture):
    def test_codes_scales_and_dequantisation(self):
        m = O.read(self.src)
        self.assertEqual(set(m.linears), set(self.q))
        for hf, (codes, scale) in self.q.items():
            self.assertTrue(np.array_equal(m.linears[hf].codes, codes), hf)
            s16 = scale.astype(np.float16).astype(np.float32)
            self.assertTrue(np.array_equal(m.linears[hf].scale, s16))
            self.assertTrue(np.allclose(m.linears[hf].dequant(), deq(codes, s16)))

    def test_norm_gains_found_through_the_graph(self):
        m = O.read(self.src)
        for l, blk in enumerate(self.model.model.layers):
            for tag in ("input_layernorm", "post_attention_layernorm"):
                self.assertTrue(np.allclose(m.extras[f"model.layers.{l}.{tag}.weight"], getattr(blk, tag).weight.detach().numpy()))

    def test_missing_norm_constant_is_reported_as_none(self):
        d = os.path.join(self.dir, "nonorm")
        write_ir_from(d, self.model, norms=False)
        m = O.read(d)
        self.assertTrue(all(v is None for v in m.extras.values()))
        self.assertEqual(len(m.extras), 2 * 2)

    def test_not_an_ir_folder(self):
        with self.assertRaises(ValueError):
            O.read(self.dir)


class Write(Fixture):
    def factors(self, seed=0):
        rng = np.random.default_rng(seed)
        return {hf: (1 + 0.2 * (rng.random(c.shape[0]) - 0.5)).astype(np.float32) for hf, (c, _) in self.q.items()}

    def test_codes_identical_scales_changed_same_size_and_graph(self):
        F = self.factors()
        O.write(self.src, self.out, F)
        a, b = consts(self.src), consts(self.out)
        self.assertEqual(a.keys(), b.keys())
        n_scales = 0
        for k in a:
            if k.endswith("_openvino_orig_weight/scale") and "layers" in k:
                n_scales += 1
                self.assertNotEqual(a[k][1], b[k][1], k)
            else:
                self.assertEqual(a[k], b[k], k)  # int4 codes, embedding, norms
        self.assertEqual(n_scales, len(self.q))
        for f in ("openvino_model.bin", "openvino_model.xml", "config.json"):
            self.assertEqual(os.path.getsize(os.path.join(self.src, f)), os.path.getsize(os.path.join(self.out, f)))
        self.assertEqual(open(os.path.join(self.src, "openvino_model.xml")).read(), open(os.path.join(self.out, "openvino_model.xml")).read())

    def test_new_scale_is_old_times_factor_in_float16_and_file_loads_again(self):
        F = self.factors(1)
        O.write(self.src, self.out, F)
        a, b = O.read(self.src), O.read(self.out)
        for hf in self.q:
            want = (a.linears[hf].scale * F[hf][:, None]).astype(np.float16).astype(np.float32)
            self.assertTrue(np.array_equal(b.linears[hf].scale, want), hf)
            self.assertTrue(np.array_equal(b.linears[hf].codes, a.linears[hf].codes))

    def test_factor_one_gives_identical_files(self):
        O.write(self.src, self.out, {hf: np.ones(c.shape[0], np.float32) for hf, (c, _) in self.q.items()})
        self.assertEqual(open(os.path.join(self.src, "openvino_model.bin"), "rb").read(), open(os.path.join(self.out, "openvino_model.bin"), "rb").read())

    def test_errors_and_no_overwrite(self):
        with self.assertRaises(ValueError):
            O.write(self.src, self.out, {"model.layers.0.mlp.up_proj.weight": np.ones(3, np.float32)})
        os.makedirs(self.out)
        with self.assertRaises(FileExistsError):
            O.write(self.src, self.out, {})

    def test_verify(self):
        O.write(self.src, self.out, self.factors())
        rep = O.verify(self.src, self.out)
        self.assertTrue(rep["PASS"], rep)
        self.assertTrue(rep["codes_identical"] and rep["same_size"] and rep["same_structure"])
        self.assertEqual(rep["scale_tensors"], len(self.q))
        self.assertEqual(rep["scale_tensors_changed"], len(self.q))
        with open(os.path.join(self.out, "openvino_model.bin"), "r+b") as f:  # flip one code byte
            off = int(re.search(r'name="self\.model\.layers\.0\.self_attn\.q_proj\._openvino_orig_weight" type="Const".*?offset="(\d+)"',
                                open(os.path.join(self.out, "openvino_model.xml")).read(), re.S).group(1))
            f.seek(off)
            f.write(b"\xff")
        self.assertFalse(O.verify(self.src, self.out)["PASS"])


if __name__ == "__main__":
    unittest.main()
