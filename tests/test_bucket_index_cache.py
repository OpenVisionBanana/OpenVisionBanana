import tempfile
import unittest
from pathlib import Path

import numpy as np

from common.training.bucket_index_cache import load_or_build_bucket_indices


class BucketIndexCacheTest(unittest.TestCase):
    def test_reuses_exact_bucket_assignments_without_rebuilding(self):
        metadata = {
            "dataset_specs_sha256": "specs-a",
            "buckets": [[512, 512], [576, 448]],
            "image_column": "image",
        }
        expected = np.array([1, 0, 1, 1, 0], dtype=np.uint16)
        with tempfile.TemporaryDirectory() as temporary:
            calls = 0

            def first_build():
                nonlocal calls
                calls += 1
                return expected

            first, reused = load_or_build_bucket_indices(
                Path(temporary), metadata, len(expected), 2, first_build
            )
            self.assertFalse(reused)
            self.assertEqual(calls, 1)
            self.assertTrue(np.array_equal(first, expected))

            def unexpected_build():
                raise AssertionError("a complete bucket index must be reused")

            second, reused = load_or_build_bucket_indices(
                Path(temporary), metadata, len(expected), 2, unexpected_build
            )
            self.assertTrue(reused)
            self.assertTrue(np.array_equal(second, expected))

    def test_configuration_change_uses_a_distinct_cache_entry(self):
        with tempfile.TemporaryDirectory() as temporary:
            first, first_reused = load_or_build_bucket_indices(
                Path(temporary),
                {"dataset_specs_sha256": "old", "buckets": [[512, 512]]},
                2,
                1,
                lambda: np.array([0, 0], dtype=np.uint16),
            )
            second, second_reused = load_or_build_bucket_indices(
                Path(temporary),
                {"dataset_specs_sha256": "new", "buckets": [[512, 512]]},
                2,
                1,
                lambda: np.array([0, 0], dtype=np.uint16),
            )
            self.assertFalse(first_reused)
            self.assertFalse(second_reused)
            self.assertTrue(np.array_equal(first, second))

    def test_migrates_compatible_legacy_metadata_without_rebuilding(self):
        metadata = {
            "kind": "dreambooth_prepared_bucket_index",
            "buckets": [[512, 512], [576, 448]],
            "image_column": "image",
            "sources": [{"name": "cityscapes", "rows": 3}],
        }
        legacy_metadata = {**metadata, "row_identity_sha256": "old-per-row-digest"}
        expected = np.array([1, 0, 1], dtype=np.uint16)

        def compatible(cached, current):
            cached = dict(cached)
            cached.pop("row_identity_sha256", None)
            return cached == current

        with tempfile.TemporaryDirectory() as temporary:
            load_or_build_bucket_indices(
                Path(temporary), legacy_metadata, len(expected), 2, lambda: expected
            )

            def unexpected_build():
                raise AssertionError("a compatible legacy bucket index must be reused")

            migrated, reused = load_or_build_bucket_indices(
                Path(temporary),
                metadata,
                len(expected),
                2,
                unexpected_build,
                metadata_compatible=compatible,
            )
            self.assertTrue(reused)
            self.assertTrue(np.array_equal(migrated, expected))

            reused_exact, reused = load_or_build_bucket_indices(
                Path(temporary), metadata, len(expected), 2, unexpected_build
            )
            self.assertTrue(reused)
            self.assertTrue(np.array_equal(reused_exact, expected))


if __name__ == "__main__":
    unittest.main()
