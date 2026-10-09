import math
import unittest

from benchmarks.v0.probe_quant_support import decode_e4m3fn, encode_e4m3fn


class QuantReferenceTests(unittest.TestCase):
    def test_all_finite_codes(self):
        codes = [i for i in range(256) if (i & 127) != 127]
        self.assertEqual(len(codes), 254)
        for code in codes:
            self.assertEqual(encode_e4m3fn(decode_e4m3fn(code)), code)

    def test_ties_even(self):
        self.assertEqual(encode_e4m3fn(1.0625), 56)
        self.assertEqual(encode_e4m3fn(1.1875), 58)

    def test_sign_zero_and_saturation(self):
        self.assertEqual(encode_e4m3fn(-0.), 128)
        self.assertEqual(encode_e4m3fn(1e10), 126)
        self.assertEqual(encode_e4m3fn(-1e10), 254)
        self.assertEqual(decode_e4m3fn(126), 448.)

    def test_nonfinite(self):
        self.assertTrue(math.isnan(decode_e4m3fn(127)))
        for v in [math.nan, math.inf, -math.inf]:
            with self.assertRaises(ValueError):
                encode_e4m3fn(v)


if __name__ == "__main__":
    unittest.main()
