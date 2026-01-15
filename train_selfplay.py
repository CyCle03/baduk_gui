import argparse
import os
import time
from typing import List, Tuple

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
DEFAULT_MAX_MOVES = 400
DEFAULT_KOMI = 6.5
DEFAULT_LEARNING_RATE = 1e-4
DEFAULT_PASS_START = 300
DEFAULT_VALUE_WINDOW = 20
DEFAULT_VALUE_DELTA = 0.05
DEFAULT_VALUE_MARGIN = 0.6
DEFAULT_SLEEP = 0.0


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


def sample_action(model: tf.keras.Model, board: GoBoard) -> Tuple[int, float]:
    state = encode_board(board)
    mask = legal_moves_mask(board)
    logits, value = model(state[None, ...], training=False)
    logits = logits.numpy()[0]
    value = float(value.numpy()[0][0])
    masked = np.where(mask > 0, logits, -1e9)
    exps = np.exp(masked - np.max(masked))
    probs = exps * mask
    total = np.sum(probs)
    if total <= 0:
        probs = mask / np.sum(mask)
    else:
        probs = probs / total
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
):
    os.makedirs(MODEL_DIR, exist_ok=True)

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
            states, actions, rewards, value_targets, score_diff, moves = play_episode(
                model,
                board_size,
                komi,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-play trainer")
    parser.add_argument("--episodes", type=int, default=1000, help="number of self-play episodes")
    parser.add_argument("--board-size", type=int, default=DEFAULT_BOARD_SIZE, help="board size")
    parser.add_argument("--komi", type=float, default=DEFAULT_KOMI, help="komi")
    parser.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help="save interval")
    parser.add_argument("--sleep", type=float, default=DEFAULT_SLEEP, help="sleep seconds per episode")
    args = parser.parse_args()
    train(
        args.episodes,
        board_size=args.board_size,
        komi=args.komi,
        save_every=args.save_every,
        sleep_time=args.sleep,
    )
