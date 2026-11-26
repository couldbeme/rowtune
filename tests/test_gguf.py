import os
import struct
import tempfile
import unittest

import numpy as np

from rowtune.formats import gguf as G
from tests.fixtures import deq, quantised_linears, write_gguf_from
from tests.tiny import make_model

BLOCK = 18


def tensor_regions(path):
    """(name, absolute offset, nbytes, type) of every tensor of a GGUF, parsed independently of rowtune."""
    raw = open(path, "rb").read()
    assert raw[:4] == b"GGUF"
    ver, nt, nkv = struct.unpack_from("<IQQ", raw, 4)
    pos = 24

    def rstr(p):
        n = struct.unpack_from("<Q", raw, p)[0]
        return raw[p + 8:p + 8 + n].decode(), p + 8 + n

    for _ in range(nkv):
        _, pos = rstr(pos)
        t = struct.unpack_from("<I", raw, pos)[0]
        pos += 4
        if t == 8:
            _, pos = rstr(pos)
        else:
            pos += 4
    infos = []
    for _ in range(nt):
        name, pos = rstr(pos)
        nd = struct.unpack_from("<I", raw, pos)[0]
        dims = struct.unpack_from(f"<{nd}Q", raw, pos + 4)
        typ, off = struct.unpack_from("<IQ", raw, pos + 4 + 8 * nd)
        pos += 4 + 8 * nd + 12
        infos.append((name, dims, typ, off))
    data0 = pos + (-pos) % 32
    size = {0: lambda n: 4 * n, 1: lambda n: 2 * n, 2: lambda n: n // 32 * 18, 8: lambda n: n // 32 * 34}
    return [(n, data0 + off, size[typ](int(np.prod(d))), typ) for n, d, typ, off in infos], raw


class Fixture(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.model = make_model(5)
        self.src = os.path.join(self.dir, "m.gguf")
        self.q = write_gguf_from(self.src, self.model)
        self.out = os.path.join(self.dir, "out.gguf")


class Read(Fixture):
    def test_codes_and_scales_of_every_q4_0_linear(self):
        m = G.read(self.src)
        self.assertEqual(set(m.linears), set(self.q))
        for hf, (codes, scale) in self.q.items():
            self.assertTrue(np.array_equal(m.linears[hf].codes, codes))
            self.assertTrue(np.array_equal(m.linears[hf].scale, scale.astype(np.float16).astype(np.float32)))
            self.assertTrue(np.allclose(m.linears[hf].dequant(), deq(codes, scale.astype(np.float16).astype(np.float32))))
        self.assertEqual(m.extras, {})

    def test_other_types_are_listed_and_left_out(self):
        other = "model.layers.0.mlp.up_proj.weight"
        write_gguf_from(os.path.join(self.dir, "o.gguf"), self.model, other_type_for=other)
        m = G.read(os.path.join(self.dir, "o.gguf"))
        self.assertNotIn(other, m.linears)
        self.assertEqual(m.skipped, [other])

    def test_llama_rows_come_back_in_hf_order(self):
        p = os.path.join(self.dir, "l.gguf")
        q = write_gguf_from(p, self.model, arch="llama", n_head=2, n_head_kv=2)
        m = G.read(p)
        for hf, (codes, _) in q.items():
            self.assertTrue(np.array_equal(m.linears[hf].codes, codes), hf)

    def test_rejects_other_files(self):
        p = os.path.join(self.dir, "x.gguf")
        with open(p, "wb") as f:
            f.write(b"NOPE" + b"\0" * 64)
        with self.assertRaises(ValueError):
            G.read(p)


class Write(Fixture):
    def factors(self, seed=0, scale=0.2):
        rng = np.random.default_rng(seed)
        return {hf: (1 + scale * (rng.random(c.shape[0]) - 0.5)).astype(np.float32) for hf, (c, _) in self.q.items()}

    def test_factor_one_gives_a_byte_identical_file(self):
        G.write(self.src, self.out, {hf: np.ones(c.shape[0], np.float32) for hf, (c, _) in self.q.items()})
        self.assertEqual(open(self.src, "rb").read(), open(self.out, "rb").read())

    def test_only_the_block_scale_bytes_change(self):
        F = self.factors()
        G.write(self.src, self.out, F)
        self.assertEqual(os.path.getsize(self.src), os.path.getsize(self.out))
        regions, a = tensor_regions(self.src)
        _, b = tensor_regions(self.out)
        changed_scale = 0
        for name, off, n, typ in regions:
            ra, rb = a[off:off + n], b[off:off + n]
            if typ == 2:
                A, B = np.frombuffer(ra, np.uint8).reshape(-1, BLOCK), np.frombuffer(rb, np.uint8).reshape(-1, BLOCK)
                self.assertTrue(np.array_equal(A[:, 2:], B[:, 2:]), name)  # the 4-bit codes
                changed_scale += int((A[:, :2] != B[:, :2]).any(1).sum())
            else:
                self.assertEqual(ra, rb, name)
        self.assertGreater(changed_scale, 0)
        outside = bytearray(a)
        for _, off, n, _ in regions:
            outside[off:off + n] = bytes(n)
        outside_b = bytearray(b)
        for _, off, n, _ in regions:
            outside_b[off:off + n] = bytes(n)
        self.assertEqual(bytes(outside), bytes(outside_b))  # header, metadata, padding

    def test_new_scale_is_old_scale_times_row_factor_in_float16(self):
        F = self.factors(1)
        G.write(self.src, self.out, F)
        a, b = G.read(self.src), G.read(self.out)
        for hf in self.q:
            want = (a.linears[hf].scale * F[hf][:, None]).astype(np.float16).astype(np.float32)
            self.assertTrue(np.array_equal(b.linears[hf].scale, want), hf)
            self.assertTrue(np.array_equal(b.linears[hf].codes, a.linears[hf].codes), hf)

    def test_llama_factors_are_given_in_hf_row_order(self):
        p, o = os.path.join(self.dir, "l.gguf"), os.path.join(self.dir, "lo.gguf")
        q = write_gguf_from(p, self.model, arch="llama", n_head=2, n_head_kv=2)
        F = {hf: np.linspace(0.8, 1.2, c.shape[0]).astype(np.float32) for hf, (c, _) in q.items()}
        G.write(p, o, F)
        a, b = G.read(p), G.read(o)  # both read back in HF order
        for hf in q:
            ratio = b.linears[hf].scale[:, 0] / a.linears[hf].scale[:, 0]
            self.assertTrue(np.allclose(ratio, F[hf], rtol=2e-3), hf)

    def test_unknown_factor_name_and_wrong_length_are_errors(self):
        with self.assertRaises(ValueError):
            G.write(self.src, self.out, {"model.layers.9.mlp.up_proj.weight": np.ones(3, np.float32)})
        with self.assertRaises(ValueError):
            G.write(self.src, self.out, {"model.layers.0.mlp.up_proj.weight": np.ones(3, np.float32)})
        self.assertFalse(os.path.exists(self.out))

    def test_refuses_to_overwrite(self):
        open(self.out, "wb").close()
        with self.assertRaises(FileExistsError):
            G.write(self.src, self.out, {})

    def test_verify_reports_pass_and_fails_on_a_changed_code(self):
        G.write(self.src, self.out, self.factors())
        rep = G.verify(self.src, self.out)
        self.assertTrue(rep["PASS"], rep)
        self.assertTrue(rep["codes_identical"] and rep["same_size"] and rep["same_structure"])
        self.assertGreater(rep["scale_rows_changed"], 0)
        bad = os.path.join(self.dir, "bad.gguf")
        raw = bytearray(open(self.out, "rb").read())
        regions, _ = tensor_regions(self.out)
        off = next(o for n, o, _, t in regions if t == 2)
        raw[off + 5] ^= 0xFF
        open(bad, "wb").write(bytes(raw))
        self.assertFalse(G.verify(self.src, bad)["PASS"])


if __name__ == "__main__":
    unittest.main()
