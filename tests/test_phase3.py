import unittest

from engine import star_points


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


if __name__ == "__main__":
    unittest.main()
