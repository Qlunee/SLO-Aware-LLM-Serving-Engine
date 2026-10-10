"""Independent validation helpers; CPU tests require torch, not CUDA/model weights."""
import unittest
import torch


class DistributionTests(unittest.TestCase):
    def test_common_logit_shift_does_not_change_distribution(self):
        from benchmarks.v1_validation.metrics import distribution_metrics
        row = distribution_metrics(torch.tensor([1., 2., 3.]) + 100,
                                   torch.tensor([1., 2., 3.]), target=2)
        self.assertAlmostEqual(row["js_divergence"], 0., places=7)
        self.assertAlmostEqual(row["centered_relative_l2"], 0., places=7)
        self.assertAlmostEqual(row["target_nll_delta"], 0., places=6)
        self.assertGreater(row["relative_l2"], 1.)

    def test_nonfinite_values_are_errors_not_success(self):
        from benchmarks.v1_validation.metrics import distribution_metrics
        for value in (float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                distribution_metrics([value, 0.], [1., 0.])

    def test_a_changed_target_distribution_has_loss(self):
        from benchmarks.v1_validation.metrics import distribution_metrics
        row = distribution_metrics([3., 0.], [0., 3.], target=1)
        self.assertAlmostEqual(row["target_nll_delta"], 3., places=6)
        self.assertGreater(row["js_divergence"], .4)
        self.assertFalse(row["top1_match"])

    def test_first_divergence_is_not_hidden_by_later_matches(self):
        from benchmarks.v1_validation.metrics import sequence_metrics
        row = sequence_metrics([1, 9, 3, 4], [1, 2, 3, 4])
        self.assertFalse(row["exact_match"])
        self.assertEqual(row["first_divergence_index"], 1)
        self.assertEqual(row["common_prefix_tokens"], 1)
        self.assertEqual(row["position_match_rate"], .75)


class DenseReferenceTests(unittest.TestCase):
    def test_suffix_mask_is_bottom_right_aligned_and_supports_gqa(self):
        from benchmarks.v1_validation.reference import dense_suffix_attention
        q = torch.zeros(2, 4, 2)
        k = torch.zeros(4, 2, 2)
        v = torch.tensor([0., 3., 6., 90.])[:, None, None].expand(4, 2, 2)
        actual = dense_suffix_attention(q, k, v, scale=1.)
        # First query sees keys 0..2; second query sees keys 0..3.
        expected = torch.tensor([3., 24.75])[:, None, None].expand(2, 4, 2)
        torch.testing.assert_close(actual, expected)

    def test_fp32_norm_preserves_input_and_unnormalized_residual(self):
        from benchmarks.v1_validation.reference import rms_reference
        x, residual = torch.tensor([[3., 4.]]), torch.tensor([[1., 2.]])
        x_before, r_before = x.clone(), residual.clone()
        _, result = rms_reference(x, torch.ones(2), 1e-6, residual)
        torch.testing.assert_close(x, x_before, atol=0, rtol=0)
        torch.testing.assert_close(residual, r_before, atol=0, rtol=0)
        torch.testing.assert_close(result, torch.tensor([[4., 6.]]), atol=0, rtol=0)

    def test_invalid_suffix_and_nonfinite_inputs_are_rejected(self):
        from benchmarks.v1_validation.reference import dense_suffix_attention
        with self.assertRaises(ValueError):
            dense_suffix_attention(torch.zeros(3, 2, 2), torch.zeros(2, 1, 2), torch.zeros(2, 1, 2))
        with self.assertRaises(ValueError):
            dense_suffix_attention(torch.full((1,2,2),float("nan")),torch.zeros(2,1,2),torch.zeros(2,1,2))

    def test_sampled_nll_summary_is_not_full_corpus_perplexity(self):
        import math
        from benchmarks.v1_validation.metrics import distribution_metrics, summarize
        row = distribution_metrics([3., 0.], [0., 3.], target=1)
        result = summarize([row])
        self.assertAlmostEqual(result["sampled_conditional_ppl_ratio"], math.exp(3), places=6)
        self.assertNotIn("perplexity", result)

    def test_empty_measurements_cannot_be_reported_as_success(self):
        from benchmarks.v1_validation.metrics import summarize, sequence_metrics
        with self.assertRaises(ValueError):
            summarize([])
        with self.assertRaises(ValueError):
            sequence_metrics([], [])

    def test_builtin_texts_are_marked_as_sanity_not_real_user_data(self):
        from benchmarks.v1_validation.corpus import load_corpus
        rows, source = load_corpus()
        self.assertEqual(len(rows), 8)
        self.assertIn("not_real_user_data", source["kind"])
        self.assertTrue(all(len(r["text"].split("\n\n")) == 4 for r in rows))


if __name__ == "__main__":
    unittest.main()
