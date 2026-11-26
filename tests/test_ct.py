import json
import os
import struct
import tempfile
import unittest

import numpy as np

from rowtune.formats import ct as C
from tests.fixtures import bf16_bits, deq, from_bf16_bits, write_ct_from
from tests.tiny import make_model


def tensors(path):
    raw = open(path, "rb").read()
    n = struct.unpack_from("<Q", raw, 0)[0]
    hdr = json.loads(raw[8:8 + n])
    return {k: (v["dtype"], v["shape"], raw[8 + n + v["data_offsets"][0]:8 + n + v["data_offsets"][1]]) for k, v in hdr.items() if k != "__metadata__"}, raw[:8 + n]


class Fixture(unittest.TestCase):
    group = 0
    dtype = "BF16"
    shards = 1

    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.model = make_model(7)
        self.src = os.path.join(self.dir, "ckpt")
        self.q = write_ct_from(self.src, self.model, self.group, self.dtype, self.shards)
        self.out = os.path.join(self.dir, "out")

    def roundtrip_scale(self, s):
        return from_bf16_bits(bf16_bits(s)) if self.dtype == "BF16" else s.astype(np.float16).astype(np.float32)

    def factors(self, seed=0):
        rng = np.random.default_rng(seed)
        return {hf: (1 + 0.2 * (rng.random(c.shape[0]) - 0.5)).astype(np.float32) for hf, (c, _) in self.q.items()}


class PerRowBf16(Fixture):
    def test_read_codes_scales(self):
        m = C.read(self.src)
        self.assertEqual(set(m.linears), set(self.q))
        for hf, (codes, scale) in self.q.items():
            self.assertTrue(np.array_equal(m.linears[hf].codes, codes), hf)
            self.assertTrue(np.array_equal(m.linears[hf].scale, self.roundtrip_scale(scale)))
            self.assertTrue(np.allclose(m.linears[hf].dequant(), deq(codes, self.roundtrip_scale(scale))))

    def test_write_changes_only_scales(self):
        F = self.factors()
        C.write(self.src, self.out, F)
        a, ha = tensors(os.path.join(self.src, "model.safetensors"))
        b, hb = tensors(os.path.join(self.out, "model.safetensors"))
        self.assertEqual(ha, hb)  # header (names, dtypes, shapes, offsets) identical
        self.assertEqual(a.keys(), b.keys())
        for k in a:
            if k.endswith(".weight_scale"):
                self.assertNotEqual(a[k][2], b[k][2], k)
            else:
                self.assertEqual(a[k], b[k], k)  # packed codes, shapes, norms, embeddings
        self.assertEqual(os.path.getsize(os.path.join(self.src, "model.safetensors")), os.path.getsize(os.path.join(self.out, "model.safetensors")))
        self.assertEqual(open(os.path.join(self.src, "config.json")).read(), open(os.path.join(self.out, "config.json")).read())

    def test_new_scale_is_old_times_factor_rounded_to_the_scale_dtype(self):
        F = self.factors(2)
        C.write(self.src, self.out, F)
        a, b = C.read(self.src), C.read(self.out)
        for hf in self.q:
            want = self.roundtrip_scale(a.linears[hf].scale * F[hf][:, None])
            self.assertTrue(np.array_equal(b.linears[hf].scale, want), hf)
            self.assertTrue(np.array_equal(b.linears[hf].codes, a.linears[hf].codes))

    def test_verify_and_failure_on_changed_code(self):
        C.write(self.src, self.out, self.factors())
        rep = C.verify(self.src, self.out)
        self.assertTrue(rep["PASS"], rep)
        self.assertEqual(rep["scale_tensors"], len(self.q))
        p = os.path.join(self.out, "model.safetensors")
        t, h = tensors(p)
        raw = bytearray(open(p, "rb").read())
        name = next(k for k in t if k.endswith("weight_packed"))
        hdr = json.loads(raw[8:len(h)])
        raw[len(h) + hdr[name]["data_offsets"][0]] ^= 0x0F
        open(p, "wb").write(bytes(raw))
        self.assertFalse(C.verify(self.src, self.out)["PASS"])

    def test_refuses_to_overwrite_and_checks_names(self):
        with self.assertRaises(ValueError):
            C.write(self.src, self.out, {"model.layers.0.mlp.up_proj.weight": np.ones(2, np.float32)})
        os.makedirs(self.out)
        with self.assertRaises(FileExistsError):
            C.write(self.src, self.out, {})

    def test_not_a_checkpoint(self):
        with self.assertRaises(ValueError):
            C.read(self.dir)


class Grouped(Fixture):
    group = 32
    dtype = "F16"

    def test_every_group_scale_of_a_row_is_multiplied_by_the_row_factor(self):
        F = self.factors(3)
        C.write(self.src, self.out, F)
        a, b = C.read(self.src), C.read(self.out)
        for hf in self.q:
            self.assertEqual(a.linears[hf].scale.shape[1], a.linears[hf].codes.shape[1] // 32)
            want = (a.linears[hf].scale * F[hf][:, None]).astype(np.float16).astype(np.float32)
            self.assertTrue(np.array_equal(b.linears[hf].scale, want), hf)
        self.assertTrue(C.verify(self.src, self.out)["PASS"])


class Sharded(Fixture):
    shards = 3

    def test_write_and_verify_across_shards(self):
        C.write(self.src, self.out, self.factors())
        self.assertTrue(C.verify(self.src, self.out)["PASS"])
        self.assertEqual(sorted(os.listdir(self.src)), sorted(os.listdir(self.out)))


if __name__ == "__main__":
    unittest.main()
