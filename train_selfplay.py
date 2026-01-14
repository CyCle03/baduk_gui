import argparse
import os
import time
from typing import List, Tuple

import numpy as np
import tensorflow as tf

from engine import GoBoard, BLACK, WHITE
from rl_model import encode_board, index_to_move, legal_moves_mask, load_or_create_model

BOARD_SIZE = 19
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "latest.keras")
SAVE_EVERY = 10
MAX_MOVES = 400
KOMI = 6.5
LEARNING_RATE = 1e-4


def sample_action(model: tf.keras.Model, board: GoBoard) -> int:
    state = encode_board(board)
    mask = legal_moves_mask(board)
    logits = model(state[None, ...], training=False).numpy()[0]
    masked = np.where(mask > 0, logits, -1e9)
    exps = np.exp(masked - np.max(masked))
    probs = exps * mask
    total = np.sum(probs)
    if total <= 0:
        probs = mask / np.sum(mask)
    else:
        probs = probs / total
    return int(np.random.choice(len(probs), p=probs))


def play_episode(model: tf.keras.Model) -> Tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    board = GoBoard(BOARD_SIZE)
    states: List[np.ndarray] = []
    actions: List[int] = []
    players: List[int] = []

    move_count = 0
    while board.consecutive_passes < 2 and move_count < MAX_MOVES:
        state = encode_board(board)
        action_idx = sample_action(model, board)
        states.append(state)
        actions.append(action_idx)
        players.append(board.to_play)

        move = index_to_move(action_idx, board.size)
        board.play(move[0], move[1])
        move_count += 1

    score_diff = board.score_area(komi=KOMI)
    result = 1.0 if score_diff > 0 else -1.0
    rewards = np.array([result if p == BLACK else -result for p in players], dtype=np.float32)
    rewards -= np.mean(rewards)
    return np.array(states, dtype=np.float32), np.array(actions, dtype=np.int32), rewards, score_diff


def train(num_episodes: int = 10000):
    os.makedirs(MODEL_DIR, exist_ok=True)

    model = load_or_create_model(MODEL_PATH, BOARD_SIZE)
    optimizer = tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE)
    last_episode = 0

    try:
        for episode in range(1, num_episodes + 1):
            last_episode = episode
            states, actions, rewards, score_diff = play_episode(model)
            with tf.GradientTape() as tape:
                logits = model(states, training=True)
                ce = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=actions, logits=logits)
                loss = tf.reduce_mean(ce * rewards)

            grads = tape.gradient(loss, model.trainable_variables)
            optimizer.apply_gradients(zip(grads, model.trainable_variables))

            if episode % SAVE_EVERY == 0:
                model.save(MODEL_PATH)
                checkpoint_path = os.path.join(MODEL_DIR, f"checkpoint_{episode:06d}.keras")
                model.save(checkpoint_path)

            print(
                f"episode={episode} loss={loss.numpy():.4f} score_diff={score_diff:.1f} "
                f"moves={len(actions)}"
            )

            time.sleep(0.01)
    except KeyboardInterrupt:
        print("Interrupted: saving latest model...")
    finally:
        if last_episode > 0:
            model.save(MODEL_PATH)
            checkpoint_path = os.path.join(MODEL_DIR, f"checkpoint_{last_episode:06d}.keras")
            model.save(checkpoint_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Self-play trainer")
    parser.add_argument("--episodes", type=int, default=1000, help="number of self-play episodes")
    args = parser.parse_args()
    train(args.episodes)
