"""Unit tests for voice blending (kokorotts.voice_blend).

Exercises the real torch math on CPU: zero VRAM, no model downloads.
"""

import unittest

import torch

from kokorotts.voice_blend import (
    blend_packs,
    check_blend_compatible,
    parse_blend_weight,
)


def pack(values: list[list[float]]) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float32)


class TestParseBlendWeight(unittest.TestCase):
    def test_accepts_bounds_and_midpoint(self):
        self.assertEqual(parse_blend_weight(0.0), 0.0)
        self.assertEqual(parse_blend_weight(0.5), 0.5)
        self.assertEqual(parse_blend_weight(1.0), 1.0)
        self.assertEqual(parse_blend_weight("0.25"), 0.25)

    def test_rejects_out_of_range(self):
        for bad in (-0.1, 1.1, 2.0):
            with self.assertRaises(ValueError):
                parse_blend_weight(bad)

    def test_rejects_non_numeric(self):
        for bad in ("half", None, [0.5]):
            with self.assertRaises(ValueError):
                parse_blend_weight(bad)


class TestBlendPacks(unittest.TestCase):
    def test_endpoints_reproduce_inputs_exactly(self):
        a = pack([[1.0, 2.0], [3.0, 4.0]])
        b = pack([[5.0, 6.0], [7.0, 8.0]])
        self.assertTrue(torch.equal(blend_packs(a, b, 0.0), a))
        self.assertTrue(torch.equal(blend_packs(a, b, 1.0), b))

    def test_midpoint_averages(self):
        a = pack([[0.0, 0.0]])
        b = pack([[2.0, 4.0]])
        self.assertTrue(
            torch.allclose(blend_packs(a, b, 0.5), pack([[1.0, 2.0]]))
        )
        self.assertTrue(
            torch.allclose(blend_packs(a, b, 0.25), pack([[0.5, 1.0]]))
        )

    def test_preserves_shape_and_dtype(self):
        a = torch.zeros(4, 8)
        b = torch.ones(4, 8)
        blended = blend_packs(a, b, 0.3)
        self.assertEqual(blended.shape, (4, 8))
        self.assertEqual(blended.dtype, torch.float32)

    def test_rejects_shape_mismatch(self):
        with self.assertRaises(ValueError):
            check_blend_compatible(torch.zeros(4, 8), torch.zeros(5, 8))
        with self.assertRaises(ValueError):
            blend_packs(torch.zeros(4, 8), torch.zeros(5, 8), 0.5)

    def test_rejects_dtype_mismatch(self):
        with self.assertRaises(ValueError):
            check_blend_compatible(torch.zeros(4, 8), torch.ones(4, 8, dtype=torch.float64))


if __name__ == "__main__":
    unittest.main()
