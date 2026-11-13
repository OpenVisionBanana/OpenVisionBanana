"""Small deterministic checks for diagnostic thresholds; no model required."""
import unittest
import numpy as np
from audit_paired_geometry_artifacts import normal_artifacts


class ArtifactTests(unittest.TestCase):
    def setUp(self):
        self.gt = np.zeros((8, 8, 3), dtype=np.float32)
        self.gt[..., 2] = 1
        self.mask = np.ones((8, 8), bool)
        self.rgb = np.full((8, 8, 3), [128, 128, 255], dtype=np.uint8)

    def test_smooth_correct_normals(self):
        metrics, _ = normal_artifacts(self.rgb, self.gt, self.mask)
        self.assertEqual(metrics['near_neutral_fraction'], 0)
        self.assertLess(metrics['smooth_surface_excess_deg'], 0.1)

    def test_neutral_candidate(self):
        self.rgb[:] = 128
        metrics, _ = normal_artifacts(self.rgb, self.gt, self.mask)
        self.assertEqual(metrics['near_neutral_fraction'], 1)
        self.assertEqual(metrics['invalid_fraction'], 1)

    def test_invalid_gt_is_excluded(self):
        self.rgb[:4] = 128
        self.mask[:4] = False
        metrics, _ = normal_artifacts(self.rgb, self.gt, self.mask)
        self.assertEqual(metrics['near_neutral_fraction'], 0)

    def test_no_valid_gt_fails(self):
        with self.assertRaises(ValueError):
            normal_artifacts(self.rgb, self.gt, np.zeros_like(self.mask))


if __name__ == '__main__':
    unittest.main()
