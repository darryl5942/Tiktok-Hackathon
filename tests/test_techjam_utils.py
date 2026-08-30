import unittest
from pathlib import Path

from techjam_utils import (
    env_flag,
    find_best_threshold,
    portable_identifier,
    validate_labeled_csv_rows,
)


class TechJamUtilsTests(unittest.TestCase):
    def test_portable_identifier_prefers_relative_paths(self):
        base = Path("/tmp/project")
        target = Path("/tmp/project/data/sample.png")
        self.assertEqual(portable_identifier(target, base), "data/sample.png")

    def test_portable_identifier_falls_back_to_basename(self):
        self.assertEqual(portable_identifier("/some/other/place/example.png"), "example.png")

    def test_find_best_threshold(self):
        labels = [0, 0, 1, 1]
        scores = [0.1, 0.2, 0.8, 0.9]
        threshold, acc = find_best_threshold(labels, scores)
        self.assertGreaterEqual(acc, 1.0)
        self.assertGreaterEqual(threshold, 0.2)
        self.assertLessEqual(threshold, 0.8)

    def test_validate_labeled_csv_rows_accepts_good_rows(self):
        rows = [{"image_path": "a.png", "label": "0"}, {"image_path": "b.png", "label": "1"}]
        validate_labeled_csv_rows(rows, source="demo")

    def test_validate_labeled_csv_rows_rejects_bad_label(self):
        rows = [{"image_path": "a.png", "label": "2"}]
        with self.assertRaises(ValueError):
            validate_labeled_csv_rows(rows, source="demo")


if __name__ == "__main__":
    unittest.main()
