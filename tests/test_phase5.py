import random
import unittest

from engine import GoBoard, IllegalMove, PASS_MOVE


class TestPositionalSuperko(unittest.TestCase):
    def test_default_is_basic_ko(self):
        # Without superko, only the immediate previous position is forbidden.
        board = GoBoard(3)
        self.assertFalse(board.superko)
        board.play(0, 1)  # B
        board.play(0, 0)  # W
        board.play(2, 1)  # B
        board.play(2, 0)  # W
        board.play(1, 2)  # B
        board.play(1, 1)  # W
        board.play(1, 0)  # B captures (1,1)
        with self.assertRaises(IllegalMove):  # immediate ko recapture
            board.play(1, 1)

    def test_superko_rejects_non_immediate_repeat(self):
        # Play a basic-ko game until the whole-board position repeats. Such a
        # repeat is NOT the immediate previous position (basic ko would have
        # blocked that), so it is a positional-superko-only violation.
        rng = random.Random(7)
        basic = GoBoard(5, superko=False)
        prefix = []
        seen = {basic._zhash}
        repeat_move = None
        while len(prefix) < 400:
            legal = [m for m in basic.legal_moves() if m != PASS_MOVE]
            if not legal:
                break
            mv = rng.choice(legal)
            basic.play(mv[0], mv[1])
            if basic._zhash in seen:
                repeat_move = mv
                break
            seen.add(basic._zhash)
            prefix.append(mv)

        self.assertIsNotNone(repeat_move, "expected a repeated position")

        # Replay the same prefix on a superko board (all distinct positions, so
        # all legal), then the repeating move must be rejected.
        sk = GoBoard(5, superko=True)
        for x, y in prefix:
            sk.play(x, y)
        self.assertFalse(sk.is_legal(repeat_move[0], repeat_move[1]))
        self.assertNotIn(repeat_move, sk.legal_moves())
        with self.assertRaises(IllegalMove):
            sk.play(repeat_move[0], repeat_move[1])

    def test_position_history_tracks_and_undo_rolls_back(self):
        board = GoBoard(5, superko=True)
        # Initial empty position is recorded.
        self.assertIn(board._zhash, board._position_history)
        board.play(2, 2)
        h1 = board._zhash
        self.assertIn(h1, board._position_history)
        board.play(1, 1)
        h2 = board._zhash
        self.assertIn(h2, board._position_history)

        board.undo()  # back to after (2,2)
        self.assertEqual(board._zhash, h1)
        self.assertNotIn(h2, board._position_history)  # rolled back

        board.undo()  # back to empty
        self.assertEqual(board.move_count(), 0)

        # Replaying the same first move is fine (no spurious superko).
        board.play(2, 2)
        self.assertEqual(board._zhash, h1)

    def test_superko_no_false_positive_on_normal_play(self):
        # A normal game with superko on should never wrongly reject ordinary
        # moves: a seeded random game completes without unexpected errors.
        rng = random.Random(123)
        board = GoBoard(7, superko=True)
        for _ in range(120):
            legal = board.legal_moves()
            mv = rng.choice(legal)
            board.play(mv[0], mv[1])  # must not raise (chosen from legal_moves)


if __name__ == "__main__":
    unittest.main()
