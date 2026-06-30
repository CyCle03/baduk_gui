import random
import unittest

from engine import (
    BLACK,
    EMPTY,
    PASS_MOVE,
    WHITE,
    GoBoard,
    IllegalMove,
)


def _random_legal_playout(board: GoBoard, num_moves: int, rng: random.Random):
    """Play up to num_moves random legal (non-pass) moves on board."""
    for _ in range(num_moves):
        legal = [m for m in board.legal_moves() if m != PASS_MOVE]
        if not legal:
            break
        x, y = rng.choice(legal)
        board.play(x, y)


class TestLegalMovesEquivalence(unittest.TestCase):
    def test_shortcut_matches_full(self):
        rng = random.Random(1234)
        for trial in range(60):
            board = GoBoard(rng.choice([5, 7, 9]))
            _random_legal_playout(board, rng.randint(0, 40), rng)
            fast = set(board.legal_moves())
            full = set(board._legal_moves_full())
            self.assertEqual(
                fast,
                full,
                msg=f"legal_moves mismatch on trial {trial}, size {board.size}",
            )


class TestZobristHashing(unittest.TestCase):
    def test_incremental_matches_full_recompute(self):
        rng = random.Random(99)
        for _ in range(80):
            board = GoBoard(rng.choice([5, 7, 9]))
            _random_legal_playout(board, rng.randint(0, 50), rng)
            self.assertEqual(board._zhash, board._zhash_full())

    def test_hash_restored_after_capture(self):
        # A capture mutates many points; the incremental hash must still match.
        board = GoBoard(5)
        board.play(2, 1)  # B
        board.play(2, 2)  # W
        board.play(1, 2)  # B
        board.play(0, 0)  # W
        board.play(3, 2)  # B
        board.play(4, 4)  # W
        board.play(2, 3)  # B captures W at (2,2)
        self.assertEqual(board.get(2, 2), EMPTY)
        self.assertEqual(board._zhash, board._zhash_full())

    def test_empty_board_hash_is_zero(self):
        self.assertEqual(GoBoard(9)._zhash, 0)


class TestCaptureSuicideKo(unittest.TestCase):
    def test_capture(self):
        board = GoBoard(5)
        board.play(2, 1)
        board.play(2, 2)
        board.play(1, 2)
        board.play(0, 0)
        board.play(3, 2)
        board.play(4, 4)
        board.play(2, 3)
        self.assertEqual(board.get(2, 2), EMPTY)
        self.assertEqual(board.prisoners_black, 1)

    def test_suicide_illegal_and_excluded(self):
        board = GoBoard(3)
        board.play(0, 0)  # B
        board.play(1, 0)  # W
        board.play(2, 0)  # B
        board.play(0, 1)  # W
        board.play(0, 2)  # B
        board.play(2, 1)  # W
        board.play(2, 2)  # B
        board.play(1, 2)  # W
        # (1,1) is fully surrounded by White -> suicide for Black.
        self.assertFalse(board._has_empty_neighbor(1, 1))
        self.assertFalse(board.is_legal(1, 1))
        self.assertNotIn((1, 1), board.legal_moves())
        with self.assertRaises(IllegalMove):
            board.play(1, 1)

    def test_simple_ko_point_is_fully_surrounded_and_illegal(self):
        board = GoBoard(3)
        board.play(0, 1)  # B
        board.play(0, 0)  # W
        board.play(2, 1)  # B
        board.play(2, 0)  # W
        board.play(1, 2)  # B
        board.play(1, 1)  # W
        board.play(1, 0)  # B captures W at (1,1)
        # The ko recapture point (1,1) is fully surrounded -> shortcut must NOT
        # admit it; the full legality check rejects it as ko.
        self.assertFalse(board._has_empty_neighbor(1, 1))
        self.assertFalse(board.is_legal(1, 1))
        self.assertNotIn((1, 1), board.legal_moves())
        with self.assertRaises(IllegalMove):
            board.play(1, 1)


class TestLightPlayEquivalence(unittest.TestCase):
    def test_clone_light_play_fast_matches_play(self):
        rng = random.Random(7)
        for trial in range(40):
            size = rng.choice([5, 7, 9])
            seq_rng = random.Random(rng.random())
            normal = GoBoard(size)
            # Build a random mid-game position by replaying legal moves.
            for _ in range(rng.randint(0, 20)):
                legal = [m for m in normal.legal_moves() if m != PASS_MOVE]
                if not legal:
                    break
                mv = seq_rng.choice(legal)
                normal.play(mv[0], mv[1])

            legal = [m for m in normal.legal_moves() if m != PASS_MOVE]
            if not legal:
                continue
            mv = seq_rng.choice(legal)

            # Clone the position BEFORE the next move, then apply the same move
            # via the light path and the normal path; end states must match.
            light = normal.clone_light()
            normal.play(mv[0], mv[1])
            light.play_fast(mv[0], mv[1])

            self.assertEqual(light.grid, normal.grid, msg=f"trial {trial}")
            self.assertEqual(light.to_play, normal.to_play)
            self.assertEqual(light._zhash, normal._zhash)
            self.assertEqual(light._prev_pos_hash, normal._prev_pos_hash)
            self.assertEqual(light._zhash, light._zhash_full())

    def test_legal_moves_on_light_board(self):
        # legal_moves() (and is_legal) must work on a history-free clone, since
        # MCTS expands nodes by masking legal moves on the simulation board.
        rng = random.Random(2024)
        for _ in range(30):
            board = GoBoard(rng.choice([5, 7, 9]))
            _random_legal_playout(board, rng.randint(0, 40), rng)
            light = board.clone_light()
            self.assertEqual(set(light.legal_moves()), set(board.legal_moves()))


class TestScoring(unittest.TestCase):
    def test_empty_board_scores_negative_komi(self):
        # No stones, no territory: black_score - white_score == -komi.
        self.assertAlmostEqual(GoBoard(9).score_area(komi=6.5), -6.5)
        self.assertAlmostEqual(GoBoard(9).score_area(komi=0.0), 0.0)

    def test_alive_black_group_scores_positive(self):
        # Fill a 5x5 board with Black except for two separated true eyes, so the
        # group is unconditionally alive and owns the whole board.
        board = GoBoard(5)
        eyes = {(1, 1), (3, 3)}
        for y in range(5):
            for x in range(5):
                if (x, y) not in eyes:
                    board.set(x, y, BLACK)
        # 23 living stones + 2 territory (the two eyes) - 6.5 komi == 18.5.
        # (Living stones are counted exactly once.)
        self.assertAlmostEqual(board.score_area(komi=6.5), 18.5)

    def test_alive_white_group_scores_symmetric(self):
        # Same shape for White: 23 stones + 2 territory + komi, all for White.
        board = GoBoard(5)
        eyes = {(1, 1), (3, 3)}
        for y in range(5):
            for x in range(5):
                if (x, y) not in eyes:
                    board.set(x, y, WHITE)
        # black_score 0 - white_score (23 + 2 + 6.5) == -31.5
        self.assertAlmostEqual(board.score_area(komi=6.5), -31.5)


if __name__ == "__main__":
    unittest.main()
