import argparse
import json
import math
import os
import shutil
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf

from engine import GoBoard, BLACK, WHITE, PASS_MOVE
from rl_model import (
    encode_board,
    index_to_move,
    legal_moves_mask,
    load_or_create_model,
    move_to_index,
)

DEFAULT_BOARD_SIZE = 19
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "latest.keras")
CKPT_DIR = os.path.join(BASE_DIR, "checkpoints")
SGF_DIR = os.path.join(BASE_DIR, "sgf")
DEFAULT_SAVE_EVERY = 10
DEFAULT_MAX_MOVES = 300
DEFAULT_KOMI = 6.5
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_PASS_START = 300
DEFAULT_VALUE_WINDOW = 20
DEFAULT_VALUE_DELTA = 0.05
DEFAULT_VALUE_MARGIN = 0.6
DEFAULT_SLEEP = 0.0
TRAIN_STATE_PATH = os.path.join(BASE_DIR, "train_state.json")
TRAIN_STATE_BACKUP_PATH = os.path.join(MODEL_DIR, "train_state_backup.json")


def _sgf_coord(x: int, y: int) -> str:
    letters = "abcdefghijklmnopqrstuvwxyz"
    return f"{letters[x]}{letters[y]}"


def _save_sgf(
    moves: List[Tuple[int, Tuple[int, int]]],
    score_diff: float,
    episode: int,
    board_size: int,
    komi: float,
    sgf_dir: str,
):
    os.makedirs(sgf_dir, exist_ok=True)
    result = f"B+{abs(score_diff):.1f}" if score_diff > 0 else f"W+{abs(score_diff):.1f}"
    sgf_moves = []
    for player, move in moves:
        color = "B" if player == BLACK else "W"
        if move == PASS_MOVE:
            coord = ""
        else:
            coord = _sgf_coord(move[0], move[1])
        sgf_moves.append(f";{color}[{coord}]")
    header = f"(;GM[1]FF[4]SZ[{board_size}]KM[{komi}]RE[{result}]"
    body = "".join(sgf_moves)
    content = f"{header}{body})"
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(sgf_dir, f"selfplay_{episode:06d}_{timestamp}.sgf")
    with open(path, "w", encoding="utf-8") as f:
        f.write(content)


def _load_train_state(path: str) -> dict:
    if not os.path.exists(path):
        return {"total_episodes": 0}
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "total_episodes" in data:
            return data
    except (OSError, json.JSONDecodeError):
        pass
    return {"total_episodes": 0}


def _save_train_state(path: str, data: dict) -> None:
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=True, indent=2)
    except OSError:
        pass


def _backup_train_state(src_path: str, backup_path: str) -> None:
    if not os.path.exists(src_path):
        return
    try:
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        shutil.copy2(src_path, backup_path)
    except OSError:
        pass


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask > 0, logits, -1e9)
    exps = np.exp(masked - np.max(masked))
    probs = exps * mask
    total = np.sum(probs)
    if total <= 0:
        return mask / np.sum(mask)
    return probs / total


def _clone_board(board: GoBoard) -> GoBoard:
    new_board = GoBoard(board.size)
    new_board.grid = [row[:] for row in board.grid]
    new_board.to_play = board.to_play
    new_board.consecutive_passes = board.consecutive_passes
    new_board.prisoners_black = board.prisoners_black
    new_board.prisoners_white = board.prisoners_white
    new_board._prev_pos_hash = board._prev_pos_hash
    new_board._history = [
        (
            new_board.copy_grid(),
            new_board.to_play,
            new_board.consecutive_passes,
            new_board.prisoners_black,
            new_board.prisoners_white,
            new_board._prev_pos_hash,
        )
    ]
    return new_board


class _MCTSNode:
    def __init__(self, prior: float):
        self.prior = float(prior)
        self.visit_count = 0
        self.value_sum = 0.0
        self.children: Dict[int, "_MCTSNode"] = {}

    def value(self) -> float:
        if self.visit_count == 0:
            return 0.0
        return self.value_sum / self.visit_count


def _select_child(node: _MCTSNode, cpuct: float) -> Tuple[int, _MCTSNode]:
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


def _terminal_value(board: GoBoard, komi: float, max_moves: int) -> Optional[float]:
    if board.consecutive_passes >= 2 or board.move_count() >= max_moves:
        score_diff = board.score_area(komi=komi)
        result_black = 1.0 if score_diff > 0 else -1.0
        return result_black if board.to_play == BLACK else -result_black
    return None


def _expand_node(model: tf.keras.Model, node: _MCTSNode, board: GoBoard) -> float:
    state = encode_board(board)
    mask = legal_moves_mask(board)
    logits, value = model(state[None, ...], training=False)
    logits = logits.numpy()[0]
    value = float(value.numpy()[0][0])
    probs = _masked_softmax(logits, mask)
    for idx, p in enumerate(probs):
        if mask[idx] > 0:
            node.children[idx] = _MCTSNode(p)
    return value


def _mcts_search(
    model: tf.keras.Model,
    board: GoBoard,
    num_simulations: int,
    cpuct: float,
    komi: float,
    max_moves: int,
) -> Tuple[np.ndarray, float]:
    root = _MCTSNode(1.0)
    root_value = _expand_node(model, root, board)

    for _ in range(num_simulations):
        node = root
        sim_board = _clone_board(board)
        path = [node]
        while node.children:
            action_idx, node = _select_child(node, cpuct)
            move = index_to_move(action_idx, sim_board.size)
            sim_board.play(move[0], move[1])
            path.append(node)

        terminal = _terminal_value(sim_board, komi, max_moves)
        if terminal is None:
            value = _expand_node(model, node, sim_board)
        else:
            value = terminal

        for n in reversed(path):
            n.visit_count += 1
            n.value_sum += value
            value = -value

    action_size = board.size * board.size + 1
    counts = np.zeros(action_size, dtype=np.float32)
    for action_idx, child in root.children.items():
        counts[action_idx] = child.visit_count
    total = np.sum(counts)
    if total <= 0:
        mask = legal_moves_mask(board)
        counts = mask / np.sum(mask)
    else:
        counts = counts / total
    return counts, root_value


def sample_action(model: tf.keras.Model, board: GoBoard) -> Tuple[int, float]:
    state = encode_board(board)
    mask = legal_moves_mask(board)
    logits, value = model(state[None, ...], training=False)
    logits = logits.numpy()[0]
    value = float(value.numpy()[0][0])
    probs = _masked_softmax(logits, mask)
    return int(np.random.choice(len(probs), p=probs)), value


def play_episode(
    model: tf.keras.Model,
    board_size: int,
    komi: float,
    max_moves: int = DEFAULT_MAX_MOVES,
    pass_start: int = DEFAULT_PASS_START,
    value_window: int = DEFAULT_VALUE_WINDOW,
    value_delta: float = DEFAULT_VALUE_DELTA,
    value_margin: float = DEFAULT_VALUE_MARGIN,
    use_mcts: bool = False,
    mcts_simulations: int = 0,
    mcts_cpuct: float = 1.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float, List[Tuple[int, Tuple[int, int]]]]:
    board = GoBoard(board_size)
    states: List[np.ndarray] = []
    actions: List[int] = []
    players: List[int] = []
    moves: List[Tuple[int, Tuple[int, int]]] = []
    recent_values: List[float] = []

    move_count = 0
    while board.consecutive_passes < 2 and move_count < max_moves:
        state = encode_board(board)
        if use_mcts and mcts_simulations > 0:
            probs, value = _mcts_search(
                model,
                board,
                mcts_simulations,
                mcts_cpuct,
                komi,
                max_moves,
            )
            action_idx = int(np.random.choice(len(probs), p=probs))
        else:
            action_idx, value = sample_action(model, board)
        recent_values.append(value)
        if len(recent_values) > value_window:
            recent_values.pop(0)

        stable = (
            move_count >= pass_start
            and len(recent_values) >= value_window
            and (max(recent_values) - min(recent_values) <= value_delta)
        )
        should_pass = stable and abs(value) >= value_margin

        if should_pass:
            action_idx = move_to_index(PASS_MOVE, board.size)
        states.append(state)
        actions.append(action_idx)
        players.append(board.to_play)

        move = index_to_move(action_idx, board.size)
        moves.append((board.to_play, move))
        board.play(move[0], move[1])
        move_count += 1

    score_diff = board.score_area(komi=komi)
    result = 1.0 if score_diff > 0 else -1.0
    value_targets = np.array([result if p == BLACK else -result for p in players], dtype=np.float32)
    rewards = value_targets - np.mean(value_targets)
    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.int32),
        rewards,
        value_targets,
        score_diff,
        moves,
    )


def train(
    num_episodes: int = 10000,
    board_size: int = DEFAULT_BOARD_SIZE,
    komi: float = DEFAULT_KOMI,
    save_every: int = DEFAULT_SAVE_EVERY,
    sleep_time: float = DEFAULT_SLEEP,
    mcts_simulations: int = 0,
    mcts_cpuct: float = 1.5,
):
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_state = _load_train_state(TRAIN_STATE_PATH)

    model = load_or_create_model(MODEL_PATH, board_size)
    optimizer = tf.keras.optimizers.Adam(learning_rate=DEFAULT_LEARNING_RATE)
    # Build variables before restoring optimizer state.
    _ = model(tf.zeros((1, board_size, board_size, 3)))
    checkpoint = tf.train.Checkpoint(model=model, optimizer=optimizer)
    manager = tf.train.CheckpointManager(checkpoint, CKPT_DIR, max_to_keep=3)
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint)
    last_episode = 0

    start_time = time.perf_counter()
    recent_times: List[float] = []
    try:
        for episode in range(1, num_episodes + 1):
            episode_start = time.perf_counter()
            last_episode = episode
            train_state["total_episodes"] = train_state.get("total_episodes", 0) + 1
            states, actions, rewards, value_targets, score_diff, moves = play_episode(
                model,
                board_size,
                komi,
                use_mcts=mcts_simulations > 0,
                mcts_simulations=mcts_simulations,
                mcts_cpuct=mcts_cpuct,
            )
            with tf.GradientTape() as tape:
                logits, values = model(states, training=True)
                ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
                policy_loss = tf.reduce_mean(ce * rewards)
                values = tf.squeeze(values, axis=-1)
                value_loss = tf.reduce_mean(tf.square(values - value_targets))
                loss = policy_loss + 0.5 * value_loss

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if episode % save_every == 0:
                model.save(MODEL_PATH)
                checkpoint_path = os.path.join(MODEL_DIR, f"checkpoint_{episode:06d}.keras")
                model.save(checkpoint_path)
                manager.save(checkpoint_number=episode)
                _save_sgf(moves, score_diff, episode, board_size, komi, SGF_DIR)
                _save_train_state(TRAIN_STATE_PATH, train_state)
                _backup_train_state(TRAIN_STATE_PATH, TRAIN_STATE_BACKUP_PATH)

            episode_time = time.perf_counter() - episode_start
            total_time = time.perf_counter() - start_time
            avg_time = total_time / episode
            recent_times.append(episode_time)
            if len(recent_times) > 10:
                recent_times.pop(0)
            recent_avg = sum(recent_times) / len(recent_times)
            print(
                f"episode={episode} loss={loss.numpy():.4f} policy={policy_loss.numpy():.4f} "
                f"value={value_loss.numpy():.4f} score_diff={score_diff:.1f} moves={len(actions)} "
                f"episode_time={episode_time:.2f}s total_time={total_time:.2f}s "
                f"avg_time={avg_time:.2f}s recent10_avg={recent_avg:.2f}s"
            )

            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Interrupted: saving latest model...")
    finally:
        if last_episode > 0:
            model.save(MODEL_PATH)
            checkpoint_path = os.path.join(MODEL_DIR, f"checkpoint_{last_episode:06d}.keras")
            model.save(checkpoint_path)
            manager.save(checkpoint_number=last_episode)
            _save_train_state(TRAIN_STATE_PATH, train_state)
            _backup_train_state(TRAIN_STATE_PATH, TRAIN_STATE_BACKUP_PATH)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-play trainer")
    parser.add_argument("--episodes", type=int, default=1000, help="number of self-play episodes")
    parser.add_argument("--board-size", type=int, default=DEFAULT_BOARD_SIZE, help="board size")
    parser.add_argument("--komi", type=float, default=DEFAULT_KOMI, help="komi")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="save interval")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="sleep seconds per episode")
    parser.add_argument("--mcts-sims", type=int, default=0, help="MCTS simulations per move (0 to disable)")
    parser.add_argument("--mcts-cpuct", type=float, default=1.5, help="MCTS exploration constant")
    args = parser.parse_args()
    train(
        args.episodes,
        board_size=args.board_size,
        komi=args.komi,
        save_every=args.save_every,
        sleep_time=args.sleep,
        mcts_simulations=args.mcts_sims,
        mcts_cpuct=args.mcts_cpuct,
    )
