"""Board feature helpers shared by the model, MCTS, and evaluation.

Kept free of any TensorFlow import so the encoding/indexing logic can be unit
tested and reused without pulling in the (heavy) ML stack.
"""

from typing import Tuple

import numpy as np

from engine import BLACK, WHITE, PASS_MOVE, GoBoard


def action_size(board_size: int) -> int:
    return board_size * board_size + 1


def move_to_index(move: Tuple[int, int], board_size: int) -> int:
    if move == PASS_MOVE:
        return board_size * board_size
    return move[1] * board_size + move[0]


def index_to_move(idx: int, board_size: int) -> Tuple[int, int]:
    if idx == board_size * board_size:
        return PASS_MOVE
    x = idx % board_size
    y = idx // board_size
    return (x, y)


def encode_board(board: GoBoard) -> np.ndarray:
    """(size, size, 3) planes: black stones, white stones, side-to-move==Black."""
    size = board.size
    grid = np.asarray(board.grid, dtype=np.int8)  # indexed [y, x]
    state = np.empty((size, size, 3), dtype=np.float32)
    state[:, :, 0] = grid == BLACK
    state[:, :, 1] = grid == WHITE
    state[:, :, 2] = 1.0 if board.to_play == BLACK else 0.0
    return state


def legal_moves_mask(board: GoBoard) -> np.ndarray:
    size = board.size
    mask = np.zeros(action_size(size), dtype=np.float32)
    indices = [move_to_index(move, size) for move in board.legal_moves()]
    if indices:
        mask[indices] = 1.0
    return mask


# --- Dihedral (8-fold) symmetry augmentation ---------------------------------
# A board is square, so the policy/value targets are invariant under the 8
# symmetries of the square. Augmenting training data with them improves sample
# efficiency. Transform index t encodes flip bit (t & 4) + rotation (t & 3).
# State planes are [y, x, c]; policy board cells are indexed y*size + x, so both
# share the [y, x] convention and transform identically.

NUM_SYMMETRIES = 8


def _spatial(arr: np.ndarray, t: int, y_axis: int, x_axis: int) -> np.ndarray:
    if t & 4:
        arr = np.flip(arr, axis=x_axis)
    k = t & 3
    if k:
        arr = np.rot90(arr, k=k, axes=(y_axis, x_axis))
    return arr


def transform_states(states: np.ndarray, t: int) -> np.ndarray:
    """Apply symmetry t to a batch of (N, size, size, C) state planes."""
    return np.ascontiguousarray(_spatial(states, t, 1, 2))


def dihedral_action_map(t: int, board_size: int) -> np.ndarray:
    """Permutation of the board-cell action indices under symmetry t (length n).

    `out[old_index] = new_index` (pass is handled separately by callers).
    """
    n = board_size * board_size
    grid = np.arange(n).reshape(board_size, board_size)
    transformed = _spatial(grid, t, 0, 1).reshape(-1)
    new_of_old = np.empty(n, dtype=np.int64)
    new_of_old[transformed] = np.arange(n)
    return new_of_old


def transform_policies(policies: np.ndarray, t: int, board_size: int) -> np.ndarray:
    """Apply symmetry t to a batch of (N, n+1) policy targets (pass kept last)."""
    n = board_size * board_size
    board = policies[:, :n].reshape(-1, board_size, board_size)
    board = _spatial(board, t, 1, 2).reshape(policies.shape[0], n)
    out = np.empty_like(policies)
    out[:, :n] = board
    out[:, n:] = policies[:, n:]
    return out


def transform_actions(actions: np.ndarray, t: int, board_size: int) -> np.ndarray:
    """Apply symmetry t to a batch of scalar action indices (pass unchanged)."""
    n = board_size * board_size
    mapping = dihedral_action_map(t, board_size)
    actions = np.asarray(actions)
    out = actions.copy()
    board_mask = actions < n
    out[board_mask] = mapping[actions[board_mask]]
    return out


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax restricted to legal actions (no temperature)."""
    masked = np.where(mask > 0, logits, -1e9)
    exps = np.exp(masked - np.max(masked))
    probs = exps * mask
    total = np.sum(probs)
    if total <= 0:
        return mask / np.sum(mask)
    return probs / total
