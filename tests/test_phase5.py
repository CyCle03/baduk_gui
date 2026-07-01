import random
import unittest

import numpy as np

from engine import GoBoard, IllegalMove, PASS_MOVE
from features import (
    NUM_SYMMETRIES,
    move_to_index,
    safe_choice,
    transform_actions,
    transform_policies,
    transform_states,
)


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


class TestSafeChoice(unittest.TestCase):
    def test_handles_drift_and_degenerate(self):
        # Sum slightly off 1 (would break np.random.choice directly).
        p = np.array([0.2, 0.2, 0.2, 0.2, 0.2000003], dtype=np.float32)
        for _ in range(50):
            self.assertIn(safe_choice(p), range(5))
        # Tiny negative entry gets clipped.
        p2 = np.array([0.5, -1e-9, 0.5], dtype=np.float64)
        self.assertIn(safe_choice(p2), range(3))
        # All-zero -> uniform fallback, never raises.
        self.assertIn(safe_choice(np.zeros(4)), range(4))

    def test_respects_support(self):
        # Zero-probability entries are never selected.
        p = np.array([0.0, 1.0, 0.0, 0.0])
        for _ in range(20):
            self.assertEqual(safe_choice(p), 1)


class TestSymmetry(unittest.TestCase):
    size = 5

    def test_identity(self):
        rng = np.random.RandomState(0)
        s = rng.rand(2, self.size, self.size, 3).astype(np.float32)
        np.testing.assert_array_equal(transform_states(s, 0), s)
        p = rng.rand(2, self.size * self.size + 1).astype(np.float32)
        np.testing.assert_array_equal(transform_policies(p, 0, self.size), p)
        a = np.array([0, 7, self.size * self.size])
        np.testing.assert_array_equal(transform_actions(a, 0, self.size), a)

    def test_state_policy_action_consistency(self):
        n = self.size * self.size
        for (x, y) in [(0, 0), (1, 3), (4, 2), (2, 2), (3, 0)]:
            idx = move_to_index((x, y), self.size)
            state = np.zeros((1, self.size, self.size, 3), dtype=np.float32)
            state[0, y, x, 0] = 1.0
            policy = np.zeros((1, n + 1), dtype=np.float32)
            policy[0, idx] = 1.0
            for t in range(NUM_SYMMETRIES):
                ts = transform_states(state, t)[0]
                tp = transform_policies(policy, t, self.size)[0]
                ta = int(transform_actions(np.array([idx]), t, self.size)[0])
                ys, xs = np.argwhere(ts[:, :, 0] == 1.0)[0]
                stone_idx = move_to_index((int(xs), int(ys)), self.size)
                self.assertEqual(stone_idx, int(np.argmax(tp[:n])))
                self.assertEqual(stone_idx, ta)

    def test_pass_preserved(self):
        n = self.size * self.size
        p = np.zeros((1, n + 1), dtype=np.float32)
        p[0, n] = 1.0
        for t in range(NUM_SYMMETRIES):
            self.assertEqual(int(np.argmax(transform_policies(p, t, self.size)[0])), n)
            self.assertEqual(int(transform_actions(np.array([n]), t, self.size)[0]), n)

    def test_action_map_is_permutation(self):
        n = self.size * self.size
        for t in range(NUM_SYMMETRIES):
            mapped = transform_actions(np.arange(n), t, self.size)
            self.assertEqual(sorted(mapped.tolist()), list(range(n)))


if __name__ == "__main__":
    unittest.main()
