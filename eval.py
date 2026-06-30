"""Evaluation harness: pit two players against each other and report a winrate.

Unlike self-play, evaluation is deterministic: no Dirichlet noise, no opening
temperature. A model player picks the argmax legal move; a ``random`` player
uses the engine's RandomAI. Players swap colors every game for fairness, and an
optional fixed seed makes a run reproducible.

Examples:
    python eval.py --p1 models/latest.keras --p2 random --games 40 --seed 0
    python eval.py --p1 models/latest.keras --p2 models/checkpoint_000100.keras
"""

import argparse
import csv
import os
import random
import time
from typing import Optional, Tuple

import numpy as np

from engine import BLACK, PASS_MOVE, GoBoard, RandomAI, opponent

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_LOG_CSV = os.path.join(BASE_DIR, "logs", "eval_log.csv")
DEFAULT_BOARD_SIZE = 19
DEFAULT_KOMI = 6.5
DEFAULT_GAMES = 40
DEFAULT_MAX_MOVES = 300


class DeterministicPolicyPlayer:
    """Plays the highest-logit legal move (no sampling, no noise)."""

    def __init__(self, model_path: str, board_size: int):
        # Imported lazily so a `random`-only evaluation does not require TF.
        from rl_model import load_or_create_model

        self.model = load_or_create_model(model_path, board_size)
        self.board_size = board_size

    def select_move(self, board: GoBoard) -> Tuple[int, int]:
        from rl_model import encode_board, index_to_move, legal_moves_mask

        state = encode_board(board)
        mask = legal_moves_mask(board)
        logits = self.model(state[None, ...], training=False)[0].numpy()[0]

        masked = np.where(mask > 0, logits, -np.inf)
        pass_idx = board.size * board.size
        # Only pass when no legal stone move exists.
        non_pass = mask.copy()
        non_pass[pass_idx] = 0.0
        if np.any(non_pass > 0):
            masked[pass_idx] = -np.inf
        idx = int(np.argmax(masked))
        return index_to_move(idx, board.size)


def make_player(spec: str, board_size: int):
    if spec == "random":
        return RandomAI(), "random"
    return DeterministicPolicyPlayer(spec, board_size), spec


def play_game(
    p1,
    p2,
    p1_color: int,
    board_size: int,
    komi: float,
    max_moves: int,
) -> float:
    """Play one game; return the score margin from P1's perspective.

    Positive means P1 is ahead. score_area returns (black - white).
    """
    board = GoBoard(board_size)
    players = {p1_color: p1, opponent(p1_color): p2}
    while board.consecutive_passes < 2 and board.move_count() < max_moves:
        mv = players[board.to_play].select_move(board)
        if mv != PASS_MOVE and not board.is_legal(mv[0], mv[1]):
            mv = PASS_MOVE  # defensive: never crash on an unexpected move
        board.play(mv[0], mv[1])
    score_diff = board.score_area(komi=komi)  # black - white
    return score_diff if p1_color == BLACK else -score_diff


def evaluate(
    p1_spec: str,
    p2_spec: str,
    games: int,
    board_size: int,
    komi: float,
    max_moves: int,
    seed: Optional[int],
) -> Tuple[float, float]:
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)

    p1, p1_name = make_player(p1_spec, board_size)
    p2, p2_name = make_player(p2_spec, board_size)

    p1_wins = 0
    margins = []
    for i in range(games):
        # Alternate colors for fairness: P1 is Black on even games.
        p1_color = BLACK if i % 2 == 0 else opponent(BLACK)
        margin = play_game(p1, p2, p1_color, board_size, komi, max_moves)
        margins.append(margin)
        if margin > 0:
            p1_wins += 1
        print(
            f"game {i + 1}/{games} p1={'B' if p1_color == BLACK else 'W'} "
            f"margin={margin:+.1f} {'P1' if margin > 0 else 'P2'} win"
        )

    winrate = p1_wins / games if games > 0 else 0.0
    avg_margin = float(np.mean(margins)) if margins else 0.0
    print(
        f"\n{p1_name} vs {p2_name}: P1 winrate={winrate:.3f} "
        f"({p1_wins}/{games}), avg_margin={avg_margin:+.2f}"
    )
    _log_csv(p1_name, p2_name, games, winrate, avg_margin)
    return winrate, avg_margin


def _log_csv(
    p1_name: str, p2_name: str, games: int, winrate: float, avg_margin: float,
    log_csv: str = DEFAULT_LOG_CSV,
) -> None:
    try:
        os.makedirs(os.path.dirname(log_csv), exist_ok=True)
        write_header = not os.path.exists(log_csv) or os.path.getsize(log_csv) == 0
        with open(log_csv, "a", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "timestamp",
                    "p1",
                    "p2",
                    "games",
                    "p1_winrate",
                    "avg_margin",
                ],
            )
            if write_header:
                writer.writeheader()
            writer.writerow(
                {
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    "p1": p1_name,
                    "p2": p2_name,
                    "games": games,
                    "p1_winrate": f"{winrate:.4f}",
                    "avg_margin": f"{avg_margin:.2f}",
                }
            )
    except OSError as exc:
        print(f"Warning: could not write eval log {log_csv}: {exc}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate two Go players")
    parser.add_argument("--p1", required=True, help="'random' or a path to a .keras model")
    parser.add_argument("--p2", default="random", help="'random' or a path to a .keras model")
    parser.add_argument("--games", type=int, default=DEFAULT_GAMES, help="number of games")
    parser.add_argument("--board-size", type=int, default=DEFAULT_BOARD_SIZE, help="board size")
    parser.add_argument("--komi", type=float, default=DEFAULT_KOMI, help="komi")
    parser.add_argument("--max-moves", type=int, default=DEFAULT_MAX_MOVES, help="max moves per game")
    parser.add_argument("--seed", type=int, default=None, help="random seed (omit for nondeterministic)")
    args = parser.parse_args()

    evaluate(
        args.p1,
        args.p2,
        games=args.games,
        board_size=args.board_size,
        komi=args.komi,
        max_moves=args.max_moves,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
