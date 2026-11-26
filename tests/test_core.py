import unittest

import torch

from rowtune.core import RowScale, attach_row_scales, fold_row_scales, linears, quantize_rtn, rows_not_pure_rescale, rtn_student
from tests.tiny import LINEARS_PER_BLOCK, make_model, make_windows


class RowScaleModule(unittest.TestCase):
    def test_starts_at_one_in_float32(self):
        rs = RowScale(7, "cpu")
        self.assertEqual(rs.c.shape, (7,))
        self.assertEqual(rs.c.dtype, torch.float32)
        self.assertTrue(torch.equal(rs.c, torch.ones(7)))

    def test_scales_each_output_row(self):
        rs = RowScale(3, "cpu")
        with torch.no_grad():
            rs.c.copy_(torch.tensor([1.0, 2.0, 0.5]))
        W = torch.arange(12, dtype=torch.float32).view(3, 4)
        self.assertTrue(torch.equal(rs(W), W * torch.tensor([[1.0], [2.0], [0.5]])))

    def test_keeps_weight_dtype(self):
        rs = RowScale(3, "cpu")
        self.assertEqual(rs(torch.ones(3, 4, dtype=torch.bfloat16)).dtype, torch.bfloat16)


class Linears(unittest.TestCase):
    def test_only_decoder_layer_linears(self):
        names = set(linears(make_model()))
        self.assertEqual(len(names), 2 * LINEARS_PER_BLOCK)
        self.assertTrue(all(".layers." in n for n in names))
        self.assertNotIn("lm_head", names)


class Rtn(unittest.TestCase):
    def test_per_row_codes_and_dequantisation(self):
        torch.manual_seed(0)
        W = torch.randn(6, 40)
        deq, codes, scale = quantize_rtn(W, bits=4, group_size=0)
        self.assertEqual(scale.shape, (6, 1))
        self.assertGreaterEqual(codes.min().item(), -8)
        self.assertLessEqual(codes.max().item(), 7)
        self.assertTrue(torch.allclose(deq, codes.float() * scale))
        self.assertTrue(torch.allclose(scale[:, 0], W.abs().amax(1) / 7.5))

    def test_grouped_scales(self):
        W = torch.randn(4, 256)
        deq, codes, scale = quantize_rtn(W, bits=4, group_size=128)
        self.assertEqual(scale.shape, (4, 2))
        self.assertEqual(deq.shape, W.shape)
        self.assertEqual(codes.shape, W.shape)

    def test_error_is_at_most_half_a_step(self):
        W = torch.randn(5, 64)
        deq, _, scale = quantize_rtn(W, bits=4, group_size=0)
        self.assertTrue(((deq - W).abs() <= 0.5 * scale + 1e-6).all())

    def test_student_replaces_only_decoder_linears(self):
        m = make_model(1)
        emb, head = m.model.embed_tokens.weight.clone(), m.lm_head.weight.clone()
        w0 = {n: l.weight.clone() for n, l in linears(m).items()}
        rtn_student(m, bits=4, group_size=0)
        self.assertTrue(torch.equal(m.model.embed_tokens.weight, emb))
        self.assertTrue(torch.equal(m.lm_head.weight, head))
        for n, l in linears(m).items():
            self.assertFalse(torch.equal(l.weight, w0[n]))
            self.assertLessEqual(len(torch.unique(l.weight[0])), 16)  # at most 16 levels per row


class FoldIsParametrised(unittest.TestCase):
    def test_folded_weights_equal_parametrised_ones(self):
        m = make_model(2)
        rtn_student(m)
        scales = attach_row_scales(m)
        with torch.no_grad():
            for rs in scales.values():
                rs.c.copy_(0.5 + torch.rand_like(rs.c))
        x = torch.tensor([make_windows(1, 12)[0]])
        with torch.no_grad():
            before = m(input_ids=x).logits
            param_w = {n: l.weight.detach().clone() for n, l in linears(m).items()}
        fold_row_scales(m)
        with torch.no_grad():
            after = m(input_ids=x).logits
        self.assertTrue(torch.equal(before, after))
        for n, l in linears(m).items():
            self.assertTrue(torch.equal(l.weight, param_w[n]))
            self.assertFalse(hasattr(l, "parametrizations"))
        self.assertEqual(sum(isinstance(p, torch.nn.Parameter) for p in m.parameters()), len(list(make_model().parameters())))


class PureRescaleCheck(unittest.TestCase):
    def test_counts_rows_that_are_not_a_single_factor(self):
        W0 = torch.randn(4, 16)
        W = W0 * torch.tensor([[1.1], [0.9], [1.3], [0.7]])
        self.assertEqual(rows_not_pure_rescale(W, W0), 0)
        W[2, 3] *= 1.5
        self.assertEqual(rows_not_pure_rescale(W, W0), 1)


if __name__ == "__main__":
    unittest.main()


class SubsetOfLinears(unittest.TestCase):
    def test_only_named_linears_get_a_scale_and_the_rest_stay_plain(self):
        m = make_model(4)
        chosen = ["model.layers.0.self_attn.q_proj", "model.layers.1.mlp.down_proj"]
        scales = attach_row_scales(m, only=chosen)
        self.assertEqual(set(scales), set(chosen))
        for n, l in linears(m).items():
            self.assertEqual(hasattr(l, "parametrizations"), n in chosen, n)
        fold_row_scales(m)
        self.assertFalse(any(hasattr(l, "parametrizations") for l in linears(m).values()))
