import argparse
import csv
import json
import math
import os
import shutil
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

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
DEFAULT_PASS_MIN_MOVES = 150
DEFAULT_VALUE_WINDOW = 20
DEFAULT_VALUE_DELTA = 0.05
DEFAULT_VALUE_MARGIN = 0.6
DEFAULT_SLEEP = 0.0
DEFAULT_BUFFER_SIZE = 5000
DEFAULT_BATCH_SIZE = 256
DEFAULT_TRAIN_STEPS = 1
DEFAULT_MCTS_SIMS = 100
DEFAULT_DIRICHLET_ALPHA = 0.03
DEFAULT_DIRICHLET_EPS = 0.30
DEFAULT_MCTS_TEMP = 1.3
DEFAULT_MCTS_TEMP_MOVES = 50
DEFAULT_RESIGN_THRESHOLD = 0.99
DEFAULT_RESIGN_START = 250
DEFAULT_RESIGN_SCORE_CHECK_MOVES = 30
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_STATE_PATH = os.path.join(BASE_DIR, "train_state.json")
TRAIN_STATE_BACKUP_PATH = os.path.join(MODEL_DIR, "train_state_backup.json")
DEFAULT_LOG_CSV = os.path.join(BASE_DIR, "logs", "train_log.csv")
DEFAULT_LOG_CUDA_CSV = os.path.join(BASE_DIR, "logs", "train_log_cuda.csv")
DEFAULT_LOG_CPU_CSV = os.path.join(BASE_DIR, "logs", "train_log_cpu.csv")


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


def _save_episode_data(
    data_dir: str,
    episode: int,
    states: np.ndarray,
    actions: np.ndarray,
    policy_targets: Optional[np.ndarray],
    rewards: np.ndarray,
    value_targets: np.ndarray,
) -> str:
    os.makedirs(data_dir, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(data_dir, f"episode_{episode:06d}_{timestamp}.npz")
    if policy_targets is not None:
        np.savez_compressed(
            path,
            states=states,
            actions=actions,
            policy_targets=policy_targets,
            rewards=rewards,
            value_targets=value_targets,
        )
    else:
        np.savez_compressed(
            path,
            states=states,
            actions=actions,
            rewards=rewards,
            value_targets=value_targets,
        )
    return path


def _load_data_dir(
    buffer: "ReplayBuffer", data_dir: str, max_files: Optional[int] = None
) -> int:
    if not os.path.isdir(data_dir):
        return 0
    files = [f for f in os.listdir(data_dir) if f.endswith(".npz")]
    files.sort()
    if max_files is not None and max_files > 0:
        files = files[-max_files:]
    total = 0
    for name in files:
        path = os.path.join(data_dir, name)
        try:
            data = np.load(path)
        except OSError:
            continue
        states = data["states"]
        actions = data["actions"]
        rewards = data["rewards"]
        value_targets = data["value_targets"]
        policy_targets = data["policy_targets"] if "policy_targets" in data else None
        buffer.add(states, actions, policy_targets, rewards, value_targets)
        total += len(states)
    return total


class ReplayBuffer:
    def __init__(self, capacity: int, use_policy_targets: bool):
        self.capacity = capacity
        self.use_policy_targets = use_policy_targets
        self.states: Deque[np.ndarray] = deque(maxlen=capacity)
        self.actions: Deque[int] = deque(maxlen=capacity)
        self.policy_targets: Deque[Optional[np.ndarray]] = deque(maxlen=capacity)
        self.rewards: Deque[float] = deque(maxlen=capacity)
        self.value_targets: Deque[float] = deque(maxlen=capacity)

    def add(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        policy_targets: Optional[np.ndarray],
        rewards: np.ndarray,
        value_targets: np.ndarray,
    ) -> None:
        for i in range(len(states)):
            self.states.append(states[i])
            self.actions.append(int(actions[i]))
            if self.use_policy_targets:
                self.policy_targets.append(policy_targets[i] if policy_targets is not None else None)
            else:
                self.policy_targets.append(None)
            self.rewards.append(float(rewards[i]))
            self.value_targets.append(float(value_targets[i]))

    def sample(self, batch_size: int):
        idx = np.random.choice(len(self.states), size=batch_size, replace=False)
        states = np.array([self.states[i] for i in idx], dtype=np.float32)
        actions = np.array([self.actions[i] for i in idx], dtype=np.int32)
        rewards = np.array([self.rewards[i] for i in idx], dtype=np.float32)
        value_targets = np.array([self.value_targets[i] for i in idx], dtype=np.float32)
        policy_targets = None
        if self.use_policy_targets:
            policy_targets = np.array([self.policy_targets[i] for i in idx], dtype=np.float32)
        return states, actions, policy_targets, rewards, value_targets

    def __len__(self) -> int:
        return len(self.states)


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask > 0, logits, -1e9)
    exps = np.exp(masked - np.max(masked))
    probs = exps * mask
    total = np.sum(probs)
    if total <= 0:
        return mask / np.sum(mask)
    return probs / total


def _apply_temperature(probs: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        return probs
    if abs(temperature - 1.0) < 1e-6:
        return probs
    scaled = np.power(probs, 1.0 / temperature)
    total = np.sum(scaled)
    if total <= 0:
        return probs
    return scaled / total


def _add_dirichlet_noise(
    probs: np.ndarray, mask: np.ndarray, alpha: float, eps: float
) -> np.ndarray:
    if eps <= 0 or alpha <= 0:
        return probs
    legal_indices = np.flatnonzero(mask > 0)
    if legal_indices.size == 0:
        return probs
    noise = np.random.dirichlet([alpha] * legal_indices.size)
    mixed = probs.copy()
    mixed[legal_indices] = (1.0 - eps) * probs[legal_indices] + eps * noise
    total = np.sum(mixed)
    if total <= 0:
        return probs
    return mixed / total


def _clone_board(board: GoBoard) -> GoBoard:
    new_board = GoBoard(board.size)
    new_board.grid = [row[:] for row in board.grid]
    new_board.to_play = board.to_play
    new_board.consecutive_passes = board.consecutive_passes
    new_board.last_pass_player = board.last_pass_player
    new_board.pass_streak = board.pass_streak
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
            new_board.last_pass_player,
            new_board.pass_streak,
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


def sample_action(model: tf.keras.Model, board: GoBoard, mask: Optional[np.ndarray] = None) -> Tuple[int, float]:
    state = encode_board(board)
    if mask is None:
        mask = legal_moves_mask(board)
    logits, value = model(state[None, ...], training=False)
    logits = logits.numpy()[0]
    value = float(value.numpy()[0][0])
    probs = _masked_softmax(logits, mask)
    return int(np.random.choice(len(probs), p=probs)), value


def _render_progress_bar(current: int, total: int, width: int = 20) -> str:
    if total <= 0:
        total = 1
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(ratio * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def play_episode(
    model: tf.keras.Model,
    board_size: int,
    komi: float,
    max_moves: int = DEFAULT_MAX_MOVES,
    pass_start: int = DEFAULT_PASS_START,
    pass_min_moves: int = DEFAULT_PASS_MIN_MOVES,
    value_window: int = DEFAULT_VALUE_WINDOW,
    value_delta: float = DEFAULT_VALUE_DELTA,
    value_margin: float = DEFAULT_VALUE_MARGIN,
    use_mcts: bool = False,
    mcts_simulations: int = 0,
    mcts_cpuct: float = 1.5,
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
    dirichlet_eps: float = DEFAULT_DIRICHLET_EPS,
    mcts_temperature: float = DEFAULT_MCTS_TEMP,
    mcts_temperature_moves: int = DEFAULT_MCTS_TEMP_MOVES,
    resign_threshold: float = DEFAULT_RESIGN_THRESHOLD,
    resign_start: int = DEFAULT_RESIGN_START,
    resign_score_check_moves: int = DEFAULT_RESIGN_SCORE_CHECK_MOVES,
    show_progress: bool = False,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    np.ndarray,
    np.ndarray,
    float,
    List[Tuple[int, Tuple[int, int]]],
]:
    board = GoBoard(board_size)
    states: List[np.ndarray] = []
    actions: List[int] = []
    players: List[int] = []
    moves: List[Tuple[int, Tuple[int, int]]] = []
    policy_targets: List[np.ndarray] = []
    recent_values: List[float] = []

    move_count = 0
    resigned = False
    resign_player = None
    while board.consecutive_passes < 2 and move_count < max_moves:
        state = encode_board(board)
        pass_allowed = move_count >= pass_min_moves
        maybe_policy = None
        if use_mcts and mcts_simulations > 0:
            probs, value = _mcts_search(
                model,
                board,
                mcts_simulations,
                mcts_cpuct,
                komi,
                max_moves,
            )
            mask = legal_moves_mask(board)
            if not pass_allowed:
                mask[move_to_index(PASS_MOVE, board.size)] = 0.0
            probs = _add_dirichlet_noise(probs, mask, dirichlet_alpha, dirichlet_eps)
            if move_count < mcts_temperature_moves:
                probs = _apply_temperature(probs, mcts_temperature)
            probs = probs * mask
            total = np.sum(probs)
            if total > 0:
                probs = probs / total
            maybe_policy = probs
            action_idx = int(np.random.choice(len(probs), p=probs))
        else:
            mask = legal_moves_mask(board)
            if not pass_allowed:
                mask[move_to_index(PASS_MOVE, board.size)] = 0.0
            action_idx, value = sample_action(model, board, mask)

        if resign_threshold > 0 and move_count >= resign_start and value <= -resign_threshold:
            skip_resign = False
            if move_count >= max_moves - resign_score_check_moves:
                score_diff = board.score_area(komi=komi)
                result_black = 1.0 if score_diff > 0 else -1.0
                if (board.to_play == BLACK and result_black > 0) or (
                    board.to_play == WHITE and result_black < 0
                ):
                    skip_resign = True
            if not skip_resign:
                resigned = True
                resign_player = board.to_play
                break
        recent_values.append(value)
        if len(recent_values) > value_window:
            recent_values.pop(0)

        stable = (
            move_count >= pass_start
            and len(recent_values) >= value_window
            and (max(recent_values) - min(recent_values) <= value_delta)
        )
        should_pass = stable and abs(value) >= value_margin

        if should_pass and pass_allowed:
            action_idx = move_to_index(PASS_MOVE, board.size)
        if maybe_policy is not None:
            policy_targets.append(maybe_policy)
        states.append(state)
        actions.append(action_idx)
        players.append(board.to_play)

        move = index_to_move(action_idx, board.size)
        moves.append((board.to_play, move))
        board.play(move[0], move[1])
        move_count += 1
        if board.pass_streak >= 3 and board.last_pass_player is not None:
            resigned = True
            resign_player = board.last_pass_player
            break
        if show_progress:
            bar = _render_progress_bar(move_count, max_moves)
            pct = int((move_count / max_moves) * 100)
            print(f"progress {bar} {pct:3d}% (moves {move_count}/{max_moves})", end="\r", flush=True)

    if resigned:
        score_diff = -1.0 if resign_player == BLACK else 1.0
    else:
        score_diff = board.score_area(komi=komi)
    if show_progress:
        print(" " * 80, end="\r")
    result = 1.0 if score_diff > 0 else -1.0
    value_targets = np.array([result if p == BLACK else -result for p in players], dtype=np.float32)
    rewards = value_targets - np.mean(value_targets)
    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.int32),
        np.array(policy_targets, dtype=np.float32) if policy_targets else None,
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
    mcts_simulations: int = DEFAULT_MCTS_SIMS,
    mcts_cpuct: float = 1.5,
    pass_min_moves: int = DEFAULT_PASS_MIN_MOVES,
    dirichlet_alpha: float = DEFAULT_DIRICHLET_ALPHA,
    dirichlet_eps: float = DEFAULT_DIRICHLET_EPS,
    mcts_temperature: float = DEFAULT_MCTS_TEMP,
    mcts_temperature_moves: int = DEFAULT_MCTS_TEMP_MOVES,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    train_steps: int = DEFAULT_TRAIN_STEPS,
    resign_threshold: float = DEFAULT_RESIGN_THRESHOLD,
    resign_start: int = DEFAULT_RESIGN_START,
    resign_score_check_moves: int = DEFAULT_RESIGN_SCORE_CHECK_MOVES,
    data_dir: str = DEFAULT_DATA_DIR,
    selfplay_only: bool = False,
    train_only: bool = False,
    save_selfplay: bool = False,
    max_data_files: Optional[int] = None,
    show_progress: bool = True,
    log_csv: Optional[str] = DEFAULT_LOG_CSV,
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
    use_policy_targets = mcts_simulations > 0
    buffer = ReplayBuffer(buffer_size, use_policy_targets=use_policy_targets)
    csv_writer = None
    csv_file = None
    cuda_enabled = bool(tf.config.list_physical_devices("GPU"))
    if log_csv == DEFAULT_LOG_CSV:
        log_csv = DEFAULT_LOG_CUDA_CSV if cuda_enabled else DEFAULT_LOG_CPU_CSV

    if log_csv:
        try:
            os.makedirs(os.path.dirname(log_csv), exist_ok=True)
            csv_file = open(log_csv, "a", encoding="utf-8", newline="")
            csv_writer = csv.DictWriter(
                csv_file,
                fieldnames=[
                    "episode",
                    "loss",
                    "policy",
                    "value",
                    "score_diff",
                    "moves",
                    "episode_time",
                    "total_time",
                    "avg_time",
                    "recent10_avg",
                    "cuda_enabled",
                ],
            )
            if csv_file.tell() == 0:
                csv_writer.writeheader()
                csv_file.flush()
        except OSError:
            csv_writer = None
            if csv_file is not None:
                csv_file.close()
                csv_file = None

    if selfplay_only and train_only:
        print("selfplay-only and train-only cannot be used together.")
        return

    if train_only:
        loaded = _load_data_dir(buffer, data_dir, max_files=max_data_files)
        if loaded == 0:
            print("No training data found; skipping train-only mode.")
            return

    try:
        for episode in range(1, num_episodes + 1):
            episode_start = time.perf_counter()
            last_episode = episode
            if train_only:
                score_diff = 0.0
                moves = []
                actions = np.array([], dtype=np.int32)
            else:
                train_state["total_episodes"] = train_state.get("total_episodes", 0) + 1
                states, actions, policy_targets, rewards, value_targets, score_diff, moves = play_episode(
                    model,
                    board_size,
                    komi,
                    use_mcts=mcts_simulations > 0,
                    mcts_simulations=mcts_simulations,
                    mcts_cpuct=mcts_cpuct,
                    pass_min_moves=pass_min_moves,
                    dirichlet_alpha=dirichlet_alpha,
                    dirichlet_eps=dirichlet_eps,
                    mcts_temperature=mcts_temperature,
                    mcts_temperature_moves=mcts_temperature_moves,
                    resign_threshold=resign_threshold,
                    resign_start=resign_start,
                    resign_score_check_moves=resign_score_check_moves,
                    show_progress=show_progress,
                )
                buffer.add(states, actions, policy_targets, rewards, value_targets)

                if save_selfplay or selfplay_only:
                    _save_episode_data(
                        data_dir, episode, states, actions, policy_targets, rewards, value_targets
                    )

            if not selfplay_only:
                if len(buffer) >= batch_size:
                    for _ in range(train_steps):
                        b_states, b_actions, b_policy_targets, b_rewards, b_value_targets = buffer.sample(
                            batch_size
                        )
                        with tf.GradientTape() as tape:
                            logits, values = model(b_states, training=True)
                            if b_policy_targets is not None:
                                targets = tf.convert_to_tensor(b_policy_targets, dtype=tf.float32)
                                log_probs = tf.nn.log_softmax(logits)
                                policy_loss = -tf.reduce_mean(tf.reduce_sum(targets * log_probs, axis=-1))
                            else:
                                ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                    labels=b_actions, logits=logits
                                )
                                policy_loss = tf.reduce_mean(ce * b_rewards)
                            values = tf.squeeze(values, axis=-1)
                            value_loss = tf.reduce_mean(tf.square(values - b_value_targets))
                            loss = policy_loss + 0.5 * value_loss

                        grads = tape.gradient(loss, model.trainable_variables)
                        optimizer.apply_gradients(zip(grads, model.trainable_variables))
                elif not train_only:
                    with tf.GradientTape() as tape:
                        logits, values = model(states, training=True)
                        if policy_targets is not None:
                            targets = tf.convert_to_tensor(policy_targets, dtype=tf.float32)
                            log_probs = tf.nn.log_softmax(logits)
                            policy_loss = -tf.reduce_mean(tf.reduce_sum(targets * log_probs, axis=-1))
                        else:
                            ce = tf.nn.sparse_softmax_cross_entropy_with_logits(
                                labels=actions, logits=logits
                            )
                            policy_loss = tf.reduce_mean(ce * rewards)
                        values = tf.squeeze(values, axis=-1)
                        value_loss = tf.reduce_mean(tf.square(values - value_targets))
                        loss = policy_loss + 0.5 * value_loss

                    grads = tape.gradient(loss, model.trainable_variables)
                    optimizer.apply_gradients(zip(grads, model.trainable_variables))
                else:
                    loss = tf.constant(0.0)
                    policy_loss = tf.constant(0.0)
                    value_loss = tf.constant(0.0)
            else:
                loss = tf.constant(0.0)
                policy_loss = tf.constant(0.0)
                value_loss = tf.constant(0.0)

            if not selfplay_only and episode % save_every == 0:
                model.save(MODEL_PATH)
                checkpoint_path = os.path.join(MODEL_DIR, f"checkpoint_{episode:06d}.keras")
                model.save(checkpoint_path)
                manager.save(checkpoint_number=episode)
                if not train_only:
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
            if csv_writer is not None:
                csv_writer.writerow(
                    {
                        "episode": episode,
                        "loss": f"{loss.numpy():.4f}",
                        "policy": f"{policy_loss.numpy():.4f}",
                        "value": f"{value_loss.numpy():.4f}",
                        "score_diff": f"{score_diff:.1f}",
                        "moves": len(actions),
                        "episode_time": f"{episode_time:.2f}",
                        "total_time": f"{total_time:.2f}",
                        "avg_time": f"{avg_time:.2f}",
                        "recent10_avg": f"{recent_avg:.2f}",
                        "cuda_enabled": int(cuda_enabled),
                    }
                )
                csv_file.flush()

            if sleep_time > 0:
                time.sleep(sleep_time)
    except KeyboardInterrupt:
        print("Interrupted: saving latest model...")
    finally:
        if csv_file is not None:
            csv_file.close()
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
    parser.add_argument(
        "--mcts-sims",
        type=int,
        default=DEFAULT_MCTS_SIMS,
        help="MCTS simulations per move (0 to disable)",
    )
    parser.add_argument("--mcts-cpuct", type=float, default=1.5, help="MCTS exploration constant")
    parser.add_argument(
        "--pass-min-moves",
        type=int,
        default=DEFAULT_PASS_MIN_MOVES,
        help="minimum moves before pass is allowed",
    )
    parser.add_argument("--dirichlet-alpha", type=float, default=DEFAULT_DIRICHLET_ALPHA, help="Dirichlet alpha")
    parser.add_argument("--dirichlet-eps", type=float, default=DEFAULT_DIRICHLET_EPS, help="Dirichlet epsilon")
    parser.add_argument("--mcts-temp", type=float, default=DEFAULT_MCTS_TEMP, help="MCTS temperature")
    parser.add_argument(
        "--mcts-temp-moves",
        type=int,
        default=DEFAULT_MCTS_TEMP_MOVES,
        help="number of opening moves using temperature",
    )
    parser.add_argument("--buffer-size", type=int, default=DEFAULT_BUFFER_SIZE, help="replay buffer size")
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE, help="replay batch size")
    parser.add_argument("--train-steps", type=int, default=DEFAULT_TRAIN_STEPS, help="train steps per episode")
    parser.add_argument("--resign-threshold", type=float, default=DEFAULT_RESIGN_THRESHOLD, help="resign threshold")
    parser.add_argument("--resign-start", type=int, default=DEFAULT_RESIGN_START, help="min moves before resign")
    parser.add_argument(
        "--resign-score-check-moves",
        type=int,
        default=DEFAULT_RESIGN_SCORE_CHECK_MOVES,
        help="skip resign check for last N moves when area score favors current player",
    )
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR, help="self-play data directory")
    parser.add_argument("--selfplay-only", action="store_true", help="generate self-play data only")
    parser.add_argument("--train-only", action="store_true", help="train from data directory only")
    parser.add_argument("--save-selfplay", action="store_true", help="save self-play data while training")
    parser.add_argument("--max-data-files", type=int, default=0, help="max data files to load")
    parser.add_argument("--log-csv", type=str, default=DEFAULT_LOG_CSV, help="CSV log path (empty to disable)")
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show per-episode progress bar",
    )
    args = parser.parse_args()
    train(
        args.episodes,
        board_size=args.board_size,
        komi=args.komi,
        save_every=args.save_every,
        sleep_time=args.sleep,
        mcts_simulations=args.mcts_sims,
        mcts_cpuct=args.mcts_cpuct,
        pass_min_moves=args.pass_min_moves,
        dirichlet_alpha=args.dirichlet_alpha,
        dirichlet_eps=args.dirichlet_eps,
        mcts_temperature=args.mcts_temp,
        mcts_temperature_moves=args.mcts_temp_moves,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        train_steps=args.train_steps,
        resign_threshold=args.resign_threshold,
        resign_start=args.resign_start,
        resign_score_check_moves=args.resign_score_check_moves,
        data_dir=args.data_dir,
        selfplay_only=args.selfplay_only,
        train_only=args.train_only,
        save_selfplay=args.save_selfplay,
        max_data_files=args.max_data_files if args.max_data_files > 0 else None,
        show_progress=args.progress,
        log_csv=args.log_csv if args.log_csv else None,
    )
