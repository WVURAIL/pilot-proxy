"""Regression checks for frame selection and the limits of aggregate diagnostics."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from frame_policy_v1 import (NFFT, Refusal, align_ids, allocation_bins, complex_stats,
                             exact_keep, threshold_rows, read_detector, sha)
from screen_export_v1 import qualify_row


class FramePolicyTests(unittest.TestCase):
    def test_alignment_uses_id_not_position(self):
        np.testing.assert_array_equal(align_ids(np.array([100, 200]), np.array([200, 100])), [1, 0])

    def test_missing_and_duplicate_id_refuse(self):
        for a, b in [([1, 2], [1]), ([1, 1], [1, 2]), ([1], [1, 1])]:
            with self.assertRaises(Refusal):
                align_ids(np.array(a), np.array(b))

    def test_exact_threshold_boundary_and_uint64_scale(self):
        t = np.array([2**63, 2**63 + 1, 0, 1], dtype=np.uint64)
        r = np.array([2**63, 2**63, 0, 1], dtype=np.uint64)
        np.testing.assert_array_equal(exact_keep(t, r, [True, True, True, False], 7, 7, (1, 1)),
                                      [True, False, False, False])

    def test_quiet_frames_beat_every_identical_dump(self):
        data = np.array([1, 1, 10, 10], dtype=complex)
        ids = np.arange(4, dtype=np.int64) * NFFT
        whole = complex_stats(data, ids)
        quiet = complex_stats(data[:2], ids[:2])
        self.assertAlmostEqual(whole['mean_real'], 5.5)
        self.assertAlmostEqual(quiet['mean_real'], 1.)
        self.assertLess(quiet['cross_frame_real_product'], whole['cross_frame_real_product'])

    def test_constant_phase_is_not_zero_residual_after_mean_removal(self):
        v = np.array([1, 2, 3, 4]) * np.exp(.4j)
        st = complex_stats(v, np.arange(4, dtype=np.int64) * NFFT)
        self.assertGreater(st['var_real'] + st['var_imag'], 0)
        self.assertAlmostEqual(np.angle(v[0] * v[-1].conjugate()), 0)

    def test_lag_one_uses_actual_spacing(self):
        st = complex_stats(np.array([1, 2, 3], complex), np.array([0, NFFT, 3*NFFT]))
        self.assertEqual(st['lag1_pairs'], 1)

    def test_allocation_edges_sum_to_six_mhz(self):
        for ch in range(14, 37):
            self.assertAlmostEqual(sum(allocation_bins(ch).values()), 6.)
        self.assertGreater(len(allocation_bins(30)), 1)

    def test_empty_policy_has_no_fake_zero_residual(self):
        self.assertIsNone(complex_stats(np.array([], complex), np.array([], dtype=int))['mean_abs'])

    def test_missing_frozen_threshold_refuses(self):
        with TemporaryDirectory() as tmp:
            p = Path(tmp) / 'thresholds.json'
            p.write_text(json.dumps(dict(channel=29, policies=[dict(policy='keep_all')])))
            with self.assertRaises(Refusal):
                threshold_rows(p, 29)

    def test_missing_pilot_refuses_before_any_row_alignment(self):
        with TemporaryDirectory() as tmp:
            d = Path(tmp)
            manifest = d / 'input_manifest.json'
            manifest.write_text(json.dumps(dict(datasets=[])))
            (d / 'run_config.json').write_text(json.dumps(dict(frame_size_samples=NFFT,
                absolute_time_used=False, input_manifest_sha256=sha(manifest))))
            with self.assertRaisesRegex(Refusal, 'missing pilot'):
                read_detector(d, 29, d, [])

    def test_screen_failure_never_becomes_exclusion(self):
        for screen in ('excise', 'keep', 'undetermined'):
            r = qualify_row(dict(freq_id='521', channel='35', role='pilot', disposition=screen))
            self.assertEqual(r['screen_disposition'], screen)
            self.assertEqual(r['scientific_ruling'], 'undetermined')
            self.assertFalse(r['physical_exclusion_certified'])


if __name__ == '__main__':
    unittest.main()
