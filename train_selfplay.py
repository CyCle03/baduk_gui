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
import torch
import torch.nn.functional as F

import mcts
from engine import GoBoard, BLACK, WHITE, PASS_MOVE
from features import (
    NUM_SYMMETRIES,
    safe_choice,
    transform_actions,
    transform_policies,
    transform_states,
)
from rl_model import (
    DEVICE,
    PolicyValueNet,
    encode_board,
    forward_numpy,
    index_to_move,
    legal_moves_mask,
    load_or_create_model,
    make_infer_fn,
    move_to_index,
    save_model,
)


def _augment_batch(states, actions, policy_targets, board_size):
    """Apply one random dihedral symmetry to a training batch (data augmentation).

    States, policy targets, and sparse actions are transformed consistently.
    """
    t = int(np.random.randint(0, NUM_SYMMETRIES))
    if t == 0:
        return states, actions, policy_targets
    states = transform_states(states, t)
    if policy_targets is not None:
        policy_targets = transform_policies(policy_targets, t, board_size)
    if actions is not None and len(actions) > 0:
        actions = transform_actions(actions, t, board_size)
    return states, actions, policy_targets

DEFAULT_BOARD_SIZE = 19
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "latest.pt")
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
DEFAULT_MCTS_BATCH = 1
DEFAULT_CHANNELS = 64
DEFAULT_BLOCKS = 4
DEFAULT_EVAL_GAMES = 20
DEFAULT_RESIGN_THRESHOLD = 0.99
DEFAULT_RESIGN_START = 250
DEFAULT_RESIGN_SCORE_CHECK_MOVES = 30
DEFAULT_DATA_DIR = os.path.join(BASE_DIR, "data")
TRAIN_STATE_PATH = os.path.join(BASE_DIR, "train_state.json")
TRAIN_STATE_BACKUP_PATH = os.path.join(MODEL_DIR, "train_state_backup.json")
DEFAULT_LOG_CSV = os.path.join(BASE_DIR, "logs", "train_log.csv")
DEFAULT_LOG_CUDA_CSV = os.path.join(BASE_DIR, "logs", "train_log_cuda.csv")
DEFAULT_LOG_CPU_CSV = os.path.join(BASE_DIR, "logs", "train_log_cpu.csv")


def size_tag(board_size: int) -> str:
    return f"{board_size}x{board_size}"


def paths_for_size(board_size: int) -> dict:
    """Model/checkpoint/sgf/data/state paths for a board size.

    19x19 keeps the historical (un-namespaced) locations for backward
    compatibility with existing checkpoints and the committed
    models/latest.pt; other sizes live under a `{size}x{size}` subfolder so
    they never clobber each other.
    """
    if board_size == DEFAULT_BOARD_SIZE:
        return {
            "model_dir": MODEL_DIR,
            "model_path": MODEL_PATH,
            "ckpt_dir": CKPT_DIR,
            "sgf_dir": SGF_DIR,
            "data_dir": DEFAULT_DATA_DIR,
            "train_state_path": TRAIN_STATE_PATH,
            "train_state_backup_path": TRAIN_STATE_BACKUP_PATH,
        }
    tag = size_tag(board_size)
    model_dir = os.path.join(MODEL_DIR, tag)
    return {
        "model_dir": model_dir,
        "model_path": os.path.join(model_dir, "latest.pt"),
        "ckpt_dir": os.path.join(CKPT_DIR, tag),
        "sgf_dir": os.path.join(SGF_DIR, tag),
        "data_dir": os.path.join(DEFAULT_DATA_DIR, tag),
        "train_state_path": os.path.join(BASE_DIR, f"train_state_{tag}.json"),
        "train_state_backup_path": os.path.join(model_dir, "train_state_backup.json"),
    }


def _save_training_checkpoint(ckpt_dir, model, optimizer, episode, max_to_keep=3):
    """Save model+optimizer for resuming, keeping only the newest `max_to_keep`.

    Replaces tf.train.CheckpointManager: the optimizer state lives here so an
    interrupted run resumes mid-training, while the eval-able model snapshots
    (`checkpoint_XXXXXX.pt`) are written separately via save_model.
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    path = os.path.join(ckpt_dir, f"ckpt_{episode:06d}.pt")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "episode": episode,
            "arch": {
                "board_size": model.board_size,
                "channels": model.channels,
                "blocks": model.blocks,
            },
        },
        path,
    )
    existing = sorted(
        f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_") and f.endswith(".pt")
    )
    for stale in existing[:-max_to_keep]:
        try:
            os.remove(os.path.join(ckpt_dir, stale))
        except OSError:
            pass
    return path


def _restore_training_checkpoint(ckpt_dir, model, optimizer):
    """Restore model+optimizer from the latest checkpoint if compatible.

    A checkpoint written for a different board size or architecture is skipped
    (with a message) rather than crashing training, matching the old behaviour.
    """
    if not os.path.isdir(ckpt_dir):
        return
    ckpts = sorted(
        f for f in os.listdir(ckpt_dir) if f.startswith("ckpt_") and f.endswith(".pt")
    )
    if not ckpts:
        return
    path = os.path.join(ckpt_dir, ckpts[-1])
    try:
        ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
        arch = ckpt.get("arch", {})
        # The net is fully convolutional, so board_size doesn't affect weights;
        # only channels/blocks must match. Validate before load_state_dict so a
        # shape mismatch doesn't half-overwrite the freshly-built model before
        # raising.
        if (
            arch.get("channels", model.channels) != model.channels
            or arch.get("blocks", model.blocks) != model.blocks
        ):
            raise ValueError("architecture mismatch")
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
    except Exception as exc:
        print(
            f"Could not restore checkpoint {path} "
            f"(likely a channels/blocks architecture change): {exc}. "
            "Starting from a fresh model."
        )


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
    # PID + sub-second token keeps filenames unique when several self-play
    # workers write to the same data dir in parallel (second-resolution
    # timestamps alone would collide and overwrite each other's episodes).
    unique = f"{os.getpid()}_{time.time_ns() % 1_000_000:06d}"
    path = os.path.join(data_dir, f"episode_{episode:06d}_{timestamp}_{unique}.npz")
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
        except Exception:
            # Skip truncated/corrupt .npz (e.g. a parallel worker killed
            # mid-write) instead of crashing the trainer.
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


# Accumulates wall time spent inside neural-network inference for the current
# episode. Reset by play_episode(); read back to split nn_time vs board_time.
_NN_TIME_ACC = [0.0]


def _timed_infer(infer_fn):
    """Wrap a batched infer_fn so NN time is folded into _NN_TIME_ACC."""

    def wrapped(states):
        _t0 = time.perf_counter()
        out = infer_fn(states)
        _NN_TIME_ACC[0] += time.perf_counter() - _t0
        return out

    return wrapped


def _optimize_batch(model, optimizer, states, actions, policy_targets, rewards, value_targets):
    """One gradient step; returns (loss, policy_loss, value_loss) as floats.

    Shared by the replay-buffer and single-episode training paths. Uses soft
    policy targets (MCTS visit distributions) when available, otherwise a
    reward-weighted cross-entropy on the sampled actions.
    """
    device = next(model.parameters()).device
    model.train()
    states_t = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
    logits, values = model(states_t)

    if policy_targets is not None:
        targets = torch.as_tensor(np.asarray(policy_targets), dtype=torch.float32, device=device)
        log_probs = F.log_softmax(logits, dim=-1)
        policy_loss = -(targets * log_probs).sum(dim=-1).mean()
    else:
        actions_t = torch.as_tensor(np.asarray(actions), dtype=torch.long, device=device)
        rewards_t = torch.as_tensor(np.asarray(rewards), dtype=torch.float32, device=device)
        ce = F.cross_entropy(logits, actions_t, reduction="none")
        policy_loss = (ce * rewards_t).mean()

    values = values.squeeze(-1)
    value_targets_t = torch.as_tensor(np.asarray(value_targets), dtype=torch.float32, device=device)
    value_loss = ((values - value_targets_t) ** 2).mean()
    loss = policy_loss + 0.5 * value_loss

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    return float(loss.item()), float(policy_loss.item()), float(value_loss.item())


def sample_action(model: PolicyValueNet, board: GoBoard, mask: Optional[np.ndarray] = None) -> Tuple[int, float]:
    state = encode_board(board)
    if mask is None:
        mask = legal_moves_mask(board)
    _t0 = time.perf_counter()
    logits, values = forward_numpy(model, state[None, ...])
    logits = logits[0]
    value = float(values[0][0])
    _NN_TIME_ACC[0] += time.perf_counter() - _t0
    probs = _masked_softmax(logits, mask)
    return safe_choice(probs), value


def _render_progress_bar(current: int, total: int, width: int = 20) -> str:
    if total <= 0:
        total = 1
    ratio = min(max(current / total, 0.0), 1.0)
    filled = int(ratio * width)
    return "[" + "#" * filled + "-" * (width - filled) + "]"


def play_episode(
    model: PolicyValueNet,
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
    infer_fn=None,
    mcts_batch: int = 1,
    superko: bool = True,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    Optional[np.ndarray],
    np.ndarray,
    np.ndarray,
    float,
    List[Tuple[int, Tuple[int, int]]],
    float,
    float,
]:
    if infer_fn is None:
        infer_fn = _timed_infer(make_infer_fn(model))
    board = GoBoard(board_size, superko=superko)
    _NN_TIME_ACC[0] = 0.0
    episode_t0 = time.perf_counter()
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
            probs, value = mcts.run_search(
                infer_fn,
                board,
                mcts_simulations,
                mcts_cpuct,
                komi,
                max_moves,
                batch_size=mcts_batch,
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
            else:
                # All MCTS weight fell on now-masked actions (e.g. the search
                # concentrated on pass while pass is still disallowed). Fall back
                # to a uniform distribution over the legal moves so sampling
                # stays on-support instead of picking an illegal move. If nothing
                # is legal, allow pass.
                if np.sum(mask) <= 0:
                    mask[move_to_index(PASS_MOVE, board.size)] = 1.0
                probs = mask / np.sum(mask)
            maybe_policy = probs
            action_idx = safe_choice(probs)
        else:
            mask = legal_moves_mask(board)
            if not pass_allowed:
                mask[move_to_index(PASS_MOVE, board.size)] = 0.0
            # If disallowing pass left no legal move (e.g. a small board filled
            # up before pass_min_moves), re-allow pass so the mask isn't all
            # zero. Otherwise _masked_softmax degenerates and safe_choice falls
            # back to a uniform pick over illegal cells, crashing board.play().
            # Mirrors the MCTS branch's empty-mask fallback above.
            if np.sum(mask) <= 0:
                mask[move_to_index(PASS_MOVE, board.size)] = 1.0
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
    nn_time = _NN_TIME_ACC[0]
    board_time = max(0.0, time.perf_counter() - episode_t0 - nn_time)
    return (
        np.array(states, dtype=np.float32),
        np.array(actions, dtype=np.int32),
        np.array(policy_targets, dtype=np.float32) if policy_targets else None,
        rewards,
        value_targets,
        score_diff,
        moves,
        nn_time,
        board_time,
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
    mcts_batch: int = DEFAULT_MCTS_BATCH,
    superko: bool = True,
    augment: bool = False,
    buffer_size: int = DEFAULT_BUFFER_SIZE,
    batch_size: int = DEFAULT_BATCH_SIZE,
    train_steps: int = DEFAULT_TRAIN_STEPS,
    resign_threshold: float = DEFAULT_RESIGN_THRESHOLD,
    resign_start: int = DEFAULT_RESIGN_START,
    resign_score_check_moves: int = DEFAULT_RESIGN_SCORE_CHECK_MOVES,
    data_dir: Optional[str] = None,
    selfplay_only: bool = False,
    train_only: bool = False,
    save_selfplay: bool = False,
    max_data_files: Optional[int] = None,
    show_progress: bool = True,
    log_csv: Optional[str] = DEFAULT_LOG_CSV,
    channels: int = DEFAULT_CHANNELS,
    blocks: int = DEFAULT_BLOCKS,
    eval_every: int = 0,
    eval_games: int = DEFAULT_EVAL_GAMES,
    init_from: Optional[str] = None,
):
    # Per-board-size paths so different sizes don't overwrite each other.
    paths = paths_for_size(board_size)
    model_dir = paths["model_dir"]
    model_path = paths["model_path"]
    ckpt_dir = paths["ckpt_dir"]
    sgf_dir = paths["sgf_dir"]
    train_state_path = paths["train_state_path"]
    train_state_backup_path = paths["train_state_backup_path"]
    if data_dir is None:
        data_dir = paths["data_dir"]

    os.makedirs(model_dir, exist_ok=True)

    train_state = _load_train_state(train_state_path)

    # Warm-start: if no model exists yet for this board size and --init-from
    # points at another model (e.g. a 9x9 net), load its weights instead of
    # starting fresh. The fully-convolutional net makes these weights board-size
    # independent. An existing checkpoint below still takes precedence (resume).
    if not os.path.exists(model_path) and init_from and os.path.exists(init_from):
        print(f"Warm-starting from {init_from}")
        model = load_or_create_model(init_from, board_size, channels, blocks)
    else:
        model = load_or_create_model(model_path, board_size, channels, blocks)
    optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_LEARNING_RATE)
    infer_fn = _timed_infer(make_infer_fn(model))
    # Restore model+optimizer from the latest checkpoint. A checkpoint written
    # for a different board size / architecture is skipped instead of crashing.
    _restore_training_checkpoint(ckpt_dir, model, optimizer)
    last_episode = 0

    start_time = time.perf_counter()
    recent_times: List[float] = []
    use_policy_targets = mcts_simulations > 0
    buffer = ReplayBuffer(buffer_size, use_policy_targets=use_policy_targets)
    csv_writer = None
    csv_file = None
    cuda_enabled = torch.cuda.is_available()
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
                    "nn_time",
                    "board_time",
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
                nn_time = 0.0
                board_time = 0.0
            else:
                train_state["total_episodes"] = train_state.get("total_episodes", 0) + 1
                states, actions, policy_targets, rewards, value_targets, score_diff, moves, nn_time, board_time = play_episode(
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
                    infer_fn=infer_fn,
                    mcts_batch=mcts_batch,
                    superko=superko,
                )
                buffer.add(states, actions, policy_targets, rewards, value_targets)

                if save_selfplay or selfplay_only:
                    _save_episode_data(
                        data_dir, episode, states, actions, policy_targets, rewards, value_targets
                    )

            loss = policy_loss = value_loss = 0.0
            if not selfplay_only:
                if len(buffer) >= batch_size:
                    for _ in range(train_steps):
                        b_states, b_actions, b_policy_targets, b_rewards, b_value_targets = buffer.sample(
                            batch_size
                        )
                        if augment:
                            b_states, b_actions, b_policy_targets = _augment_batch(
                                b_states, b_actions, b_policy_targets, board_size
                            )
                        loss, policy_loss, value_loss = _optimize_batch(
                            model, optimizer, b_states, b_actions,
                            b_policy_targets, b_rewards, b_value_targets,
                        )
                elif not train_only:
                    if augment:
                        states, actions, policy_targets = _augment_batch(
                            states, actions, policy_targets, board_size
                        )
                    loss, policy_loss, value_loss = _optimize_batch(
                        model, optimizer, states, actions,
                        policy_targets, rewards, value_targets,
                    )

            if not selfplay_only and episode % save_every == 0:
                save_model(model, model_path)
                checkpoint_path = os.path.join(model_dir, f"checkpoint_{episode:06d}.pt")
                save_model(model, checkpoint_path)
                _save_training_checkpoint(ckpt_dir, model, optimizer, episode)
                if not train_only:
                    _save_sgf(moves, score_diff, episode, board_size, komi, sgf_dir)
                _save_train_state(train_state_path, train_state)
                _backup_train_state(train_state_path, train_state_backup_path)

            if (
                eval_every > 0
                and not selfplay_only
                and not train_only
                and episode % eval_every == 0
            ):
                try:
                    from eval import evaluate_model

                    evaluate_model(
                        model,
                        board_size,
                        eval_games,
                        komi,
                        DEFAULT_MAX_MOVES,
                        seed=0,
                        episode=episode,
                    )
                except Exception as exc:  # eval must never crash training
                    print(f"eval failed: {exc}")

            episode_time = time.perf_counter() - episode_start
            total_time = time.perf_counter() - start_time
            avg_time = total_time / episode
            recent_times.append(episode_time)
            if len(recent_times) > 10:
                recent_times.pop(0)
            recent_avg = sum(recent_times) / len(recent_times)
            print(
                f"episode={episode} loss={loss:.4f} policy={policy_loss:.4f} "
                f"value={value_loss:.4f} score_diff={score_diff:.1f} moves={len(actions)} "
                f"episode_time={episode_time:.2f}s nn_time={nn_time:.2f}s board_time={board_time:.2f}s "
                f"total_time={total_time:.2f}s "
                f"avg_time={avg_time:.2f}s recent10_avg={recent_avg:.2f}s"
            )
            if csv_writer is not None:
                csv_writer.writerow(
                    {
                        "episode": episode,
                        "loss": f"{loss:.4f}",
                        "policy": f"{policy_loss:.4f}",
                        "value": f"{value_loss:.4f}",
                        "score_diff": f"{score_diff:.1f}",
                        "moves": len(actions),
                        "episode_time": f"{episode_time:.2f}",
                        "nn_time": f"{nn_time:.2f}",
                        "board_time": f"{board_time:.2f}",
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
        # selfplay-only workers are pure data producers: they must never write
        # the model / checkpoints / train_state, or a parallel worker would
        # clobber the trainer's latest.pt with its own stale (unchanged) weights.
        if last_episode > 0 and not selfplay_only:
            save_model(model, model_path)
            checkpoint_path = os.path.join(model_dir, f"checkpoint_{last_episode:06d}.pt")
            save_model(model, checkpoint_path)
            _save_training_checkpoint(ckpt_dir, model, optimizer, last_episode)
            _save_train_state(train_state_path, train_state)
            _backup_train_state(train_state_path, train_state_backup_path)


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
    parser.add_argument(
        "--superko",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use positional superko in self-play (forbid recreating any prior position)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="augment training batches with random 8-fold board symmetries",
    )
    parser.add_argument("--channels", type=int, default=DEFAULT_CHANNELS, help="conv channels (new models)")
    parser.add_argument("--blocks", type=int, default=DEFAULT_BLOCKS, help="residual blocks (new models)")
    parser.add_argument(
        "--eval-every",
        type=int,
        default=0,
        help="run a deterministic vs-Random eval every N episodes (0 disables)",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=DEFAULT_EVAL_GAMES,
        help="games per --eval-every evaluation",
    )
    parser.add_argument(
        "--mcts-batch",
        type=int,
        default=DEFAULT_MCTS_BATCH,
        help="MCTS leaves evaluated per batched inference (1 = exact sequential; "
        ">1 enables virtual-loss leaf batching so the GPU is actually used)",
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
    parser.add_argument(
        "--data-dir",
        type=str,
        default=None,
        help="self-play data directory (default: per-board-size under ./data)",
    )
    parser.add_argument("--selfplay-only", action="store_true", help="generate self-play data only")
    parser.add_argument("--train-only", action="store_true", help="train from data directory only")
    parser.add_argument("--save-selfplay", action="store_true", help="save self-play data while training")
    parser.add_argument("--max-data-files", type=int, default=0, help="max data files to load")
    parser.add_argument("--log-csv", type=str, default=DEFAULT_LOG_CSV, help="CSV log path (empty to disable)")
    parser.add_argument(
        "--init-from",
        type=str,
        default=None,
        help="warm-start weights from another model (.pt), e.g. a 9x9 net to "
        "seed 19x19 training. Ignored if a model already exists for this size.",
    )
    parser.add_argument(
        "--progress",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="show per-episode progress bar",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="run only 2 episodes under cProfile and print the top 25 functions by cumulative time",
    )
    args = parser.parse_args()

    episodes = args.episodes
    save_every = args.save_every
    log_csv = args.log_csv if args.log_csv else None
    if args.profile:
        # Short, no-side-effect run focused on the hot path.
        episodes = 2
        save_every = 10 ** 9  # effectively never save during profiling
        log_csv = None

    def _run():
        train(
            episodes,
            board_size=args.board_size,
            komi=args.komi,
            save_every=save_every,
            sleep_time=args.sleep,
            mcts_simulations=args.mcts_sims,
            mcts_cpuct=args.mcts_cpuct,
            pass_min_moves=args.pass_min_moves,
            dirichlet_alpha=args.dirichlet_alpha,
            dirichlet_eps=args.dirichlet_eps,
            mcts_temperature=args.mcts_temp,
            mcts_temperature_moves=args.mcts_temp_moves,
            mcts_batch=args.mcts_batch,
            superko=args.superko,
            augment=args.augment,
            channels=args.channels,
            blocks=args.blocks,
            eval_every=args.eval_every,
            eval_games=args.eval_games,
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
            log_csv=log_csv,
            init_from=args.init_from,
        )

    if args.profile:
        import cProfile
        import pstats

        profiler = cProfile.Profile()
        profiler.runcall(_run)
        stats = pstats.Stats(profiler)
        stats.sort_stats("cumulative")
        print("\n===== cProfile: top 25 by cumulative time =====")
        stats.print_stats(25)
    else:
        _run()
