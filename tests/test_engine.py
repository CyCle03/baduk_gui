import unittest

from engine import GoBoard, EMPTY, IllegalMove


class TestEngine(unittest.TestCase):
    def test_capture_single_stone(self):
        board = GoBoard(5)
        board.play(2, 1)  # B
        board.play(2, 2)  # W
        board.play(1, 2)  # B
        board.play(0, 0)  # W
        board.play(3, 2)  # B
        board.play(4, 4)  # W
        board.play(2, 3)  # B captures

        self.assertEqual(board.get(2, 2), EMPTY)
        self.assertEqual(board.prisoners_black, 1)

    def test_suicide_is_illegal(self):
        board = GoBoard(3)
        board.play(0, 0)  # B
        board.play(1, 0)  # W
        board.play(2, 0)  # B
        board.play(0, 1)  # W
        board.play(0, 2)  # B
        board.play(2, 1)  # W
        board.play(2, 2)  # B
        board.play(1, 2)  # W

        with self.assertRaises(IllegalMove):
            board.play(1, 1)

    def test_simple_ko(self):
        board = GoBoard(3)
        board.play(0, 1)  # B
        board.play(0, 0)  # W
        board.play(2, 1)  # B
        board.play(2, 0)  # W
        board.play(1, 2)  # B
        board.play(1, 1)  # W
        board.play(1, 0)  # B captures

        with self.assertRaises(IllegalMove):
            board.play(1, 1)


if __name__ == "__main__":
    unittest.main()
