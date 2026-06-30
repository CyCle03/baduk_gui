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


def masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Softmax restricted to legal actions (no temperature)."""
    masked = np.where(mask > 0, logits, -1e9)
    exps = np.exp(masked - np.max(masked))
    probs = exps * mask
    total = np.sum(probs)
    if total <= 0:
        return mask / np.sum(mask)
    return probs / total
