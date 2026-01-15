import os
from typing import Tuple

import numpy as np
import tensorflow as tf

from engine import BLACK, WHITE, EMPTY, PASS_MOVE, GoBoard


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


def legal_moves_mask(board: GoBoard) -> np.ndarray:
    size = board.size
    mask = np.zeros(action_size(size), dtype=np.float32)
    for move in board.legal_moves():
        mask[move_to_index(move, size)] = 1.0
    return mask


def build_policy_model(board_size: int) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(board_size, board_size, 3))
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(inputs)

    def residual_block(t):
        skip = t
        t = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(t)
        t = tf.keras.layers.Conv2D(64, 3, padding="same")(t)
        t = tf.keras.layers.Add()([t, skip])
        return tf.keras.layers.Activation("relu")(t)

    for _ in range(4):
        x = residual_block(x)

    # Policy head
    p = tf.keras.layers.Conv2D(2, 1, padding="same", activation="relu")(x)
    p = tf.keras.layers.Flatten()(p)
    logits = tf.keras.layers.Dense(action_size(board_size), name="policy_logits")(p)

    # Value head
    v = tf.keras.layers.Conv2D(1, 1, padding="same", activation="relu")(x)
    v = tf.keras.layers.Flatten()(v)
    v = tf.keras.layers.Dense(64, activation="relu")(v)
    value = tf.keras.layers.Dense(1, activation="tanh", name="value")(v)
    return tf.keras.Model(inputs, [logits, value])


def masked_softmax(logits: np.ndarray, mask: np.ndarray, temperature: float) -> np.ndarray:
    if temperature <= 0:
        temperature = 1.0
    scaled = logits / temperature
    masked = np.where(mask > 0, scaled, -1e9)
    exps = np.exp(masked - np.max(masked))
    probs = exps * mask
    total = np.sum(probs)
    if total <= 0:
        probs = mask / np.sum(mask)
    else:
        probs = probs / total
    return probs


def load_or_create_model(path: str, board_size: int) -> tf.keras.Model:
    if os.path.exists(path):
        model = tf.keras.models.load_model(path)
        if len(model.outputs) == 2:
            input_shape = model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            if input_shape[1] == board_size and input_shape[2] == board_size:
                return model
            print("Model board size mismatch; creating new policy+value model.")
            return build_policy_model(board_size)
        print("Loaded legacy policy-only model; creating new policy+value model.")
        return build_policy_model(board_size)
    return build_policy_model(board_size)


class PolicyAI:
    def __init__(self, model_path: str, board_size: int = 19, temperature: float = 0.8):
        self.model_path = model_path
        self.board_size = board_size
        self.temperature = temperature
        self.model = load_or_create_model(model_path, board_size)

    def reload(self):
        self.model = load_or_create_model(self.model_path, self.board_size)

    def select_move(self, board: GoBoard) -> Tuple[int, int]:
        state = encode_board(board)
        mask = legal_moves_mask(board)
        outputs = self.model(state[None, ...], training=False)
        logits = outputs[0].numpy()[0]
        probs = masked_softmax(logits, mask, self.temperature)
        idx = int(np.random.choice(len(probs), p=probs))
        return index_to_move(idx, board.size)
