import tempfile
import unittest


class TestSgfParserErrors(unittest.TestCase):
    def _write_temp_sgf(self, content: str) -> str:
        with tempfile.NamedTemporaryFile("w", suffix=".sgf", delete=False, encoding="utf-8") as f:
            f.write(content)
            return f.name

    def test_invalid_size_raises(self):
        try:
            from baduk_gui import parse_sgf
        except Exception:
            self.skipTest("PyQt6 or baduk_gui not available")
        path = self._write_temp_sgf("(;GM[1]FF[4]SZ[abc];B[aa])")
        with self.assertRaises(ValueError):
            parse_sgf(path)

    def test_invalid_coord_raises(self):
        try:
            from baduk_gui import parse_sgf
        except Exception:
            self.skipTest("PyQt6 or baduk_gui not available")
        path = self._write_temp_sgf("(;GM[1]FF[4]SZ[19];B[a])")
        with self.assertRaises(ValueError):
            parse_sgf(path)


if __name__ == "__main__":
    unittest.main()
