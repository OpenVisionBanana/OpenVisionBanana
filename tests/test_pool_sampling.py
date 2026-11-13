import random
import unittest
from dataclasses import dataclass

from common.training.pool_sampling import limit_distinct_per_group, sample_indices


@dataclass
class Row:
    image: str
    expression: str


class PoolSamplingTest(unittest.TestCase):
    def test_refseg_cap_is_distinct_deterministic_and_original_order(self):
        rows = [
            Row("a.png", "left"),
            Row("a.png", "left"),
            Row("a.png", "right"),
            Row("a.png", "middle"),
            Row("b.png", "only"),
        ]
        select = lambda seed: limit_distinct_per_group(
            rows, lambda row: row.image, lambda row: row.expression, 2, seed
        )
        first = select(42)
        self.assertEqual(first, select(42))
        self.assertEqual(first, sorted(first))
        selected_a = [rows[index].expression for index in first if rows[index].image == "a.png"]
        self.assertEqual(len(selected_a), 2)
        self.assertEqual(len(set(selected_a)), 2)
        self.assertIn(4, first)

    def test_with_replacement_can_repeat_below_population_size(self):
        result = sample_indices(random.Random(7), list(range(100)), 80, "with_replacement")
        self.assertEqual(len(result), 80)
        self.assertLess(len(set(result)), len(result))
        self.assertEqual(result, sample_indices(random.Random(7), list(range(100)), 80, "with_replacement"))

    def test_without_replacement_preserves_unique_draw_when_possible(self):
        result = sample_indices(random.Random(7), list(range(100)), 80, "without_replacement")
        self.assertEqual(len(set(result)), 80)


if __name__ == "__main__":
    unittest.main()
