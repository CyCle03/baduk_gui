import os
import unittest

from engine import star_points

try:
    from train_selfplay import MODEL_DIR, MODEL_PATH, paths_for_size, size_tag
    _HAVE_TRAINER = True
except Exception:  # tensorflow not installed in this environment
    _HAVE_TRAINER = False


class TestStarPoints(unittest.TestCase):
    def test_19_standard(self):
        expected = {(x, y) for x in (3, 9, 15) for y in (3, 9, 15)}
        self.assertEqual(set(star_points(19)), expected)

    def test_13_standard(self):
        expected = {
            (3, 3), (3, 9), (9, 3), (9, 9), (6, 6),
            (3, 6), (9, 6), (6, 3), (6, 9),
        }
        self.assertEqual(set(star_points(13)), expected)

    def test_9_standard(self):
        self.assertEqual(
            set(star_points(9)), {(2, 2), (2, 6), (6, 2), (6, 6), (4, 4)}
        )

    def test_7(self):
        self.assertEqual(
            set(star_points(7)), {(2, 2), (2, 4), (4, 2), (4, 4), (3, 3)}
        )

    def test_in_bounds_and_unique(self):
        for n in range(5, 26):
            pts = star_points(n)
            self.assertEqual(len(pts), len(set(pts)), msg=f"dupes for n={n}")
            for x, y in pts:
                self.assertTrue(0 <= x < n and 0 <= y < n, msg=f"oob for n={n}")


@unittest.skipUnless(_HAVE_TRAINER, "train_selfplay (tensorflow) not importable")
class TestPerSizePaths(unittest.TestCase):
    def test_19_uses_legacy_paths(self):
        p = paths_for_size(19)
        self.assertEqual(p["model_path"], MODEL_PATH)
        self.assertEqual(p["model_dir"], MODEL_DIR)

    def test_other_sizes_are_namespaced(self):
        for n in (7, 9, 13):
            p = paths_for_size(n)
            tag = size_tag(n)
            self.assertEqual(p["model_dir"], os.path.join(MODEL_DIR, tag))
            self.assertEqual(p["model_path"], os.path.join(MODEL_DIR, tag, "latest.keras"))
            self.assertTrue(p["ckpt_dir"].endswith(tag))
            self.assertTrue(p["data_dir"].endswith(tag))
            self.assertIn(tag, p["train_state_path"])
            # No two sizes collide.
        self.assertNotEqual(paths_for_size(9)["model_path"], paths_for_size(13)["model_path"])


if __name__ == "__main__":
    unittest.main()
