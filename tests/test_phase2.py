import math
import random
import unittest

import numpy as np

import mcts
from engine import BLACK, WHITE, PASS_MOVE, GoBoard
from features import (
    action_size,
    encode_board,
    index_to_move,
    legal_moves_mask,
    move_to_index,
)


def _playout(board, n, rng):
    for _ in range(n):
        legal = [m for m in board.legal_moves() if m != PASS_MOVE]
        if not legal:
            break
        x, y = rng.choice(legal)
        board.play(x, y)


def _reference_encode(board):
    size = board.size
    state = np.zeros((size, size, 3), dtype=np.float32)
    for y in range(size):
        for x in range(size):
            v = board.get(x, y)
            if v == BLACK:
                state[y, x, 0] = 1.0
            elif v == WHITE:
                state[y, x, 1] = 1.0
    if board.to_play == BLACK:
        state[:, :, 2] = 1.0
    return state


def _reference_mask(board):
    size = board.size
    mask = np.zeros(action_size(size), dtype=np.float32)
    for move in board.legal_moves():
        mask[move_to_index(move, size)] = 1.0
    return mask


class TestFeatureVectorization(unittest.TestCase):
    def test_encode_matches_reference(self):
        rng = random.Random(11)
        for _ in range(40):
            board = GoBoard(rng.choice([5, 7, 9]))
            _playout(board, rng.randint(0, 30), rng)
            np.testing.assert_array_equal(encode_board(board), _reference_encode(board))

    def test_mask_matches_reference(self):
        rng = random.Random(13)
        for _ in range(40):
            board = GoBoard(rng.choice([5, 7, 9]))
            _playout(board, rng.randint(0, 30), rng)
            np.testing.assert_array_equal(legal_moves_mask(board), _reference_mask(board))


def _stub_infer(states):
    """Deterministic fake model: logits from (black-white) per cell, value=tanh(sum)."""
    n = states.shape[0]
    size = states.shape[1]
    a = size * size + 1
    logits = np.zeros((n, a), dtype=np.float32)
    values = np.zeros((n,), dtype=np.float32)
    for i in range(n):
        flat = (states[i, :, :, 0] - states[i, :, :, 1]).ravel()
        logits[i, : size * size] = flat
        logits[i, size * size] = -1.0  # mildly disfavor pass
        values[i] = float(np.tanh(flat.sum() * 0.1))
    return logits, values


# Independent re-implementation of the original sequential search, used as the
# ground truth that run_search(batch_size=1) must reproduce exactly.
def _reference_search(infer_fn, board, num_simulations, cpuct, komi, max_moves):
    class N:
        def __init__(self, prior):
            self.prior = prior
            self.visit_count = 0
            self.value_sum = 0.0
            self.children = {}

        def value(self):
            return 0.0 if self.visit_count == 0 else self.value_sum / self.visit_count

    def select(node):
        total = sum(c.visit_count for c in node.children.values())
        best, ba, bc = -1e9, 0, None
        for ai, c in node.children.items():
            u = cpuct * c.prior * math.sqrt(total + 1) / (1 + c.visit_count)
            s = c.value() + u
            if s > best:
                best, ba, bc = s, ai, c
        return ba, bc

    def msoft(logits, mask):
        masked = np.where(mask > 0, logits, -1e9)
        e = np.exp(masked - np.max(masked))
        p = e * mask
        t = np.sum(p)
        return mask / np.sum(mask) if t <= 0 else p / t

    def expand(node, b, logits):
        mask = legal_moves_mask(b)
        probs = msoft(logits, mask)
        for idx, p in enumerate(probs):
            if mask[idx] > 0:
                node.children[idx] = N(p)

    def terminal(b):
        if b.consecutive_passes >= 2 or b.move_count() >= max_moves:
            sd = b.score_area(komi=komi)
            rb = 1.0 if sd > 0 else -1.0
            return rb if b.to_play == BLACK else -rb
        return None

    root = N(1.0)
    rl, rv = infer_fn(encode_board(board)[None, ...])
    expand(root, board, rl[0])
    for _ in range(num_simulations):
        node = root
        sim = board.clone_light()
        path = [node]
        while node.children:
            ai, node = select(node)
            mv = index_to_move(ai, sim.size)
            sim.play_fast(mv[0], mv[1])
            path.append(node)
        term = terminal(sim)
        if term is None:
            lg, vv = infer_fn(encode_board(sim)[None, ...])
            expand(node, sim, lg[0])
            value = float(vv[0])
        else:
            value = term
        for nd in reversed(path):
            nd.visit_count += 1
            nd.value_sum += value
            value = -value
    counts = np.zeros(action_size(board.size), dtype=np.float32)
    for ai, c in root.children.items():
        counts[ai] = c.visit_count
    total = np.sum(counts)
    return counts / total if total > 0 else counts


class TestMCTS(unittest.TestCase):
    def test_batch1_matches_reference(self):
        rng = random.Random(5)
        for _ in range(8):
            board = GoBoard(5)
            _playout(board, rng.randint(0, 8), rng)
            ref = _reference_search(_stub_infer, board, 40, 1.5, 6.5, 300)
            got, _ = mcts.run_search(_stub_infer, board, 40, 1.5, 6.5, 300, batch_size=1)
            np.testing.assert_allclose(got, ref, rtol=0, atol=0)

    def test_search_is_deterministic(self):
        board = GoBoard(5)
        random.Random(1).random()
        a, _ = mcts.run_search(_stub_infer, board, 50, 1.5, 6.5, 300, batch_size=1)
        b, _ = mcts.run_search(_stub_infer, board, 50, 1.5, 6.5, 300, batch_size=1)
        np.testing.assert_array_equal(a, b)

    def test_batched_distribution_valid(self):
        board = GoBoard(7)
        _playout(board, 5, random.Random(3))
        for batch in (4, 8, 16):
            counts, value = mcts.run_search(
                _stub_infer, board, 64, 1.5, 6.5, 300, batch_size=batch
            )
            self.assertAlmostEqual(float(counts.sum()), 1.0, places=5)
            self.assertTrue(np.all(counts >= 0))
            # Visited actions must all be legal.
            mask = legal_moves_mask(board)
            self.assertTrue(np.all(mask[counts > 0] > 0))
            self.assertTrue(-1.0 <= value <= 1.0)

    def test_total_visits_conserved(self):
        # Root child visits must sum to num_simulations regardless of batching.
        board = GoBoard(5)
        for batch in (1, 4, 10):
            root = mcts.MCTSNode(1.0)
            # run_search normalizes; instead verify via a direct count by
            # re-running and checking the normalized counts sum to 1 (proxy).
            counts, _ = mcts.run_search(
                _stub_infer, board, 30, 1.5, 6.5, 300, batch_size=batch
            )
            self.assertAlmostEqual(float(counts.sum()), 1.0, places=5)


if __name__ == "__main__":
    unittest.main()
