import unittest
from pathlib import Path

from data_pipeline import _stable_hash_fraction, assign_split


class StableHashFractionTests(unittest.TestCase):
    def test_deterministic_across_calls(self):
        self.assertEqual(_stable_hash_fraction("image_0001.jpg"), _stable_hash_fraction("image_0001.jpg"))

    def test_in_unit_interval(self):
        for key in ["a.jpg", "b.png", "some/nested/name.webp", ""]:
            frac = _stable_hash_fraction(key)
            self.assertGreaterEqual(frac, 0.0)
            self.assertLess(frac, 1.0)

    def test_different_keys_typically_differ(self):
        self.assertNotEqual(_stable_hash_fraction("a.jpg"), _stable_hash_fraction("b.jpg"))


class AssignSplitTests(unittest.TestCase):
    def _paths(self, names):
        return [Path(n) for n in names]

    def test_val_ratio_zero_sends_everything_to_train(self):
        manifest = {}
        train, val, newly_assigned = assign_split(self._paths(["a.jpg", "b.jpg", "c.jpg"]), manifest, val_ratio=0.0)
        self.assertEqual(len(val), 0)
        self.assertEqual(len(train), 3)
        self.assertEqual(newly_assigned, 3)

    def test_val_ratio_one_sends_everything_to_val(self):
        manifest = {}
        train, val, newly_assigned = assign_split(self._paths(["a.jpg", "b.jpg", "c.jpg"]), manifest, val_ratio=1.0)
        self.assertEqual(len(train), 0)
        self.assertEqual(len(val), 3)
        self.assertEqual(newly_assigned, 3)

    def test_existing_assignment_is_never_moved_by_a_new_ratio(self):
        # This is the leak-safety guarantee: once a file has a recorded split,
        # changing val_ratio on a later call must not retroactively move it.
        manifest = {}
        paths = self._paths(["a.jpg", "b.jpg", "c.jpg", "d.jpg", "e.jpg"])
        train_first, val_first, _ = assign_split(paths, manifest, val_ratio=0.5)
        first_assignment = dict(manifest)

        train_second, val_second, newly_assigned_second = assign_split(paths, manifest, val_ratio=0.0)

        self.assertEqual(manifest, first_assignment)
        self.assertEqual(newly_assigned_second, 0)
        self.assertEqual({p.name for p in train_first}, {p.name for p in train_second})
        self.assertEqual({p.name for p in val_first}, {p.name for p in val_second})

    def test_only_new_paths_are_freshly_assigned(self):
        manifest = {"a.jpg": "train", "b.jpg": "val"}
        train, val, newly_assigned = assign_split(self._paths(["a.jpg", "b.jpg", "c.jpg"]), manifest, val_ratio=0.5)
        self.assertEqual(newly_assigned, 1)
        self.assertIn("c.jpg", manifest)
        self.assertEqual(manifest["a.jpg"], "train")
        self.assertEqual(manifest["b.jpg"], "val")

    def test_mutates_manifest_bucket_in_place(self):
        manifest = {}
        assign_split(self._paths(["a.jpg"]), manifest, val_ratio=0.5)
        self.assertIn("a.jpg", manifest)


if __name__ == "__main__":
    unittest.main()
