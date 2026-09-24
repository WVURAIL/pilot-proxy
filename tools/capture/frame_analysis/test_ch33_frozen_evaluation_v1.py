"""Guard the identities and integer decisions used by the CH33 replay."""
from fractions import Fraction
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import numpy as np

from ch33_frozen_evaluation_v1 import exact_keep, read_psd_window, retention_bounds, unique_join


class FrozenEvaluationTests(unittest.TestCase):
    def test_integer_boundary_survives_uint64_overflow(self):
        values = np.array([2**63, 2**63 + 1], dtype=np.uint64)
        keep = exact_keep(values, [2**63, 2**63], 12734, 12734, (1, 1))
        np.testing.assert_array_equal(keep, [True, False])

    def test_fraction_decisions_and_invalid_reference(self):
        target, reference = [5, 5, 8, 1], [7, 8, 11, 0]
        ratio = (13, 8)
        expected = [r > 0 and Fraction(t * 11, r * 5) <= Fraction(*ratio)
                    for t, r in zip(target, reference)]
        np.testing.assert_array_equal(exact_keep(target, reference, 5, 11, ratio), expected)

    def test_join_preserves_order_and_missing_rows(self):
        np.testing.assert_array_equal(unique_join([("a", 0), ("b", 0), ("a", 1)],
                                                  [("a", 1), ("a", 0)]), [1, -1, 0])
        with self.assertRaises(ValueError):
            unique_join([("a", 0)], [("a", 0), ("a", 0)])

    def test_missingness_bounds_keep_original_denominator(self):
        self.assertEqual(retention_bounds(3, 8, 10), [.3, .5])
        with self.assertRaises(ValueError):
            retention_bounds(9, 8, 10)

    def test_streamed_spectrum_window(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "product.npz"
            data = np.arange(137 * 16, dtype=np.int16).reshape(137, 16)
            np.savez_compressed(path, psd_frame_db_i16=data)
            columns = [15, 0, 3]
            np.testing.assert_array_equal(read_psd_window(path, columns, 137, 16), data[:, columns])
            with self.assertRaises(ValueError):
                read_psd_window(path, columns, 136, 16)

    def test_optimized_python_rejects_tampered_frozen_input(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            release = root / "output/dissertation-implementation-2026-09-19/frame-policies/corrected-ch33-calibration-v1"
            release.mkdir(parents=True)
            (release / "summary.json").write_text('{"tampered": true}\n')
            output = root / "must-not-be-created"
            producer = Path(__file__).with_name("ch33_frozen_evaluation_v1.py")
            result = subprocess.run([sys.executable, "-O", str(producer), "--root", str(root),
                                     "--output", str(output)], text=True, capture_output=True)
            self.assertNotEqual(result.returncode, 0)
            self.assertIn("Calibration summary hash mismatch.", result.stderr)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
