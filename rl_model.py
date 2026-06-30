import os
from typing import Tuple

import numpy as np
import tensorflow as tf

from engine import BLACK, WHITE, EMPTY, PASS_MOVE, GoBoard

# TF-free feature helpers live in features.py; re-exported here so existing
# imports (`from rl_model import encode_board, ...`) keep working.
from features import (
    action_size,
    move_to_index,
    index_to_move,
    encode_board,
    legal_moves_mask,
)


def build_policy_model(board_size: int, channels: int = 64, blocks: int = 4) -> tf.keras.Model:
    inputs = tf.keras.Input(shape=(board_size, board_size, 3))
    x = tf.keras.layers.Conv2D(channels, 3, padding="same", activation="relu")(inputs)

    def residual_block(t):
        skip = t
        t = tf.keras.layers.Conv2D(channels, 3, padding="same", activation="relu")(t)
        t = tf.keras.layers.Conv2D(channels, 3, padding="same")(t)
        t = tf.keras.layers.Add()([t, skip])
        return tf.keras.layers.Activation("relu")(t)

    for _ in range(blocks):
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


def make_infer_fn(model: tf.keras.Model):
    """Return ``infer(states) -> (logits, values)`` for batched NN inference.

    ``states`` is an (N, size, size, 3) float array; the result is a tuple of
    numpy arrays: logits (N, action_size) and values (N,). The inner call is
    wrapped in a ``tf.function`` with retracing reduction so repeated calls with
    varying batch sizes avoid the per-call eager dispatch overhead. This single
    entry point is shared by self-play and the GUI so MCTS can evaluate many
    leaves in one model call (the only way the GPU earns its keep here).
    """

    input_shape = model.input_shape
    if isinstance(input_shape, list):
        input_shape = input_shape[0]
    # Fixed signature with a free batch dimension: any leaf-batch size reuses the
    # same concrete function instead of triggering a retrace.
    signature = tf.TensorSpec(
        [None, input_shape[1], input_shape[2], input_shape[3]], tf.float32
    )

    @tf.function(input_signature=[signature])
    def _forward(x):
        return model(x, training=False)

    def infer(states: np.ndarray):
        logits, values = _forward(tf.convert_to_tensor(states, dtype=tf.float32))
        return logits.numpy(), values.numpy()[:, 0]

    return infer


def load_or_create_model(
    path: str, board_size: int, channels: int = 64, blocks: int = 4
) -> tf.keras.Model:
    if os.path.exists(path):
        try:
            model = tf.keras.models.load_model(path)
        except (OSError, ValueError) as exc:
            print(f"Failed to load model at {path}: {exc}. Creating a new model.")
            return build_policy_model(board_size, channels, blocks)
        if len(model.outputs) == 2:
            input_shape = model.input_shape
            if isinstance(input_shape, list):
                input_shape = input_shape[0]
            if input_shape[1] == board_size and input_shape[2] == board_size:
                return model
            print("Model board size mismatch; creating new policy+value model.")
            return build_policy_model(board_size, channels, blocks)
        print("Loaded legacy policy-only model; creating new policy+value model.")
        return build_policy_model(board_size, channels, blocks)
    return build_policy_model(board_size, channels, blocks)


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
