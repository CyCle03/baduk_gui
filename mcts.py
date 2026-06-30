"""Shared Monte-Carlo Tree Search core for self-play and the GUI.

Previously the search was duplicated almost verbatim in train_selfplay.py and
baduk_gui.py. This module is the single implementation. It is free of any
TensorFlow import: inference is injected as ``infer_fn(states) -> (logits,
values)`` operating on numpy arrays, so the same code drives the real Keras
model and the test stubs.

Leaf batching (virtual-visit loss): up to ``batch_size`` leaves are collected
per iteration, evaluated in one ``infer_fn`` call, then expanded and backed up.
While collecting, each selected path gets temporary virtual visits so sibling
leaves are explored instead of re-selecting the same path. With ``batch_size==1``
no virtual loss is ever observed by a selection step, so the search is exactly
the original sequential algorithm.
"""

import math
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from engine import BLACK, GoBoard
from features import action_size, encode_board, index_to_move, legal_moves_mask

InferFn = Callable[[np.ndarray], Tuple[np.ndarray, np.ndarray]]


class MCTSNode:
    __slots__ = ("prior", "visit_count", "value_sum", "children")

    def __init__(self, prior: float):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, "MCTSNode"] = {}

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _select_child(node: MCTSNode, cpuct: float) -> Tuple[int, MCTSNode]:
    total_visits = sum(child.visit_count for child in node.children.values())
    best_score = -1e9
    best_action = 0
    best_child = None
    for action_idx, child in node.children.items():
        q = child.value()
        u = cpuct * child.prior * math.sqrt(total_visits + 1) / (1 + child.visit_count)
        score = q + u
        if score > best_score:
            best_score = score
            best_action = action_idx
            best_child = child
    if best_child is None:
        raise RuntimeError("MCTS selection failed")
    return best_action, best_child


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask > 0, logits, -1e9)
    exps = np.exp(masked - np.max(masked))
    probs = exps * mask
    total = np.sum(probs)
    if total <= 0:
        return mask / np.sum(mask)
    return probs / total


def _terminal_value(board: GoBoard, komi: float, max_moves: int) -> Optional[float]:
    if board.consecutive_passes >= 2 or board.move_count() >= max_moves:
        score_diff = board.score_area(komi=komi)
        result_black = 1.0 if score_diff > 0 else -1.0
        return result_black if board.to_play == BLACK else -result_black
    return None


def _expand(node: MCTSNode, board: GoBoard, logits: np.ndarray) -> None:
    mask = legal_moves_mask(board)
    probs = _masked_softmax(logits, mask)
    for idx, p in enumerate(probs):
        if mask[idx] > 0:
            node.children[idx] = MCTSNode(p)


def run_search(
    infer_fn: InferFn,
    board: GoBoard,
    num_simulations: int,
    cpuct: float,
    komi: float,
    max_moves: int,
    batch_size: int = 1,
    virtual_loss: int = 1,
) -> Tuple[np.ndarray, float]:
    """Run MCTS and return (normalized visit counts over actions, root value)."""
    root = MCTSNode(1.0)
    root_logits, root_values = infer_fn(encode_board(board)[None, ...])
    root_value = float(root_values[0])
    _expand(root, board, root_logits[0])

    batch_size = max(1, batch_size)
    sims_done = 0
    while sims_done < num_simulations:
        n = min(batch_size, num_simulations - sims_done)
        collected: List[Tuple[List[MCTSNode], GoBoard, Optional[float]]] = []
        for _ in range(n):
            node = root
            sim_board = board.clone_light()
            path = [node]
            while node.children:
                action_idx, node = _select_child(node, cpuct)
                move = index_to_move(action_idx, sim_board.size)
                sim_board.play_fast(move[0], move[1])
                path.append(node)
            term = _terminal_value(sim_board, komi, max_moves)
            # Virtual loss: temporary visits discourage re-selecting this path
            # while the rest of the batch is collected.
            for nd in path:
                nd.visit_count += virtual_loss
            collected.append((path, sim_board, term))

        # Batch-evaluate all non-terminal leaves in a single inference call.
        pending = [c for c in collected if c[2] is None]
        if pending:
            states = np.stack([encode_board(c[1]) for c in pending])
            logits_b, values_b = infer_fn(states)
            leaf_logits = {id(c[0][-1]): logits_b[i] for i, c in enumerate(pending)}
            leaf_value = {id(c[0][-1]): float(values_b[i]) for i, c in enumerate(pending)}

        for path, sim_board, term in collected:
            leaf = path[-1]
            # Remove the virtual loss before the real backup.
            for nd in path:
                nd.visit_count -= virtual_loss
            if term is not None:
                value = term
            else:
                if not leaf.children:
                    _expand(leaf, sim_board, leaf_logits[id(leaf)])
                value = leaf_value[id(leaf)]
            for nd in reversed(path):
                nd.visit_count += 1
                nd.value_sum += value
                value = -value

        sims_done += n

    size = board.size
    counts = np.zeros(action_size(size), dtype=np.float32)
    for action_idx, child in root.children.items():
        counts[action_idx] = child.visit_count
    total = np.sum(counts)
    if total <= 0:
        mask = legal_moves_mask(board)
        counts = mask / np.sum(mask)
    else:
        counts = counts / total
    return counts, root_value
