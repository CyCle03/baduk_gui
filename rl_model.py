import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from engine import BLACK, WHITE, EMPTY, PASS_MOVE, GoBoard

# TF-free feature helpers live in features.py; re-exported here so existing
# imports (`from rl_model import encode_board, ...`) keep working.
from features import (
    action_size,
    move_to_index,
    index_to_move,
    encode_board,
    legal_moves_mask,
    safe_choice,
)

# One device for the whole process: models and input tensors live here.
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PolicyValueNet(nn.Module):
    """Fully-convolutional ResNet policy+value net.

    Input is NHWC ``(N, size, size, 3)`` (the layout produced by
    ``features.encode_board``); it is permuted to NCHW inside ``forward`` so the
    feature/encoding code stays framework-agnostic. Returns ``(logits, value)``
    with logits ``(N, size*size + 1)`` and value ``(N, 1)``.

    No layer depends on the board size: board-point logits come from a 1x1 conv
    (one logit per point), the pass logit and the value come from a
    global-average-pooled trunk feature. So one set of weights runs on any board
    size, and a model trained on 9x9 can warm-start 19x19. ``board_size`` is kept
    only as metadata (the size the net was primarily trained on).
    """

    def __init__(self, board_size: int, channels: int = 64, blocks: int = 4):
        super().__init__()
        self.board_size = board_size
        self.channels = channels
        self.blocks = blocks

        self.stem = nn.Conv2d(3, channels, 3, padding=1)
        self.res_blocks = nn.ModuleList(
            nn.ModuleList(
                [
                    nn.Conv2d(channels, channels, 3, padding=1),
                    nn.Conv2d(channels, channels, 3, padding=1),
                ]
            )
            for _ in range(blocks)
        )

        # Policy head (size-independent): per-point logits via 1x1 convs, plus a
        # single pass logit from the global-pooled trunk feature.
        self.policy_conv = nn.Conv2d(channels, 32, 1)
        self.policy_point = nn.Conv2d(32, 1, 1)
        self.policy_pass = nn.Linear(channels, 1)

        # Value head (size-independent): global average pool -> MLP.
        self.value_fc1 = nn.Linear(channels, 64)
        self.value_fc2 = nn.Linear(64, 1)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        # NHWC -> NCHW
        x = x.permute(0, 3, 1, 2).contiguous()
        x = F.relu(self.stem(x))
        for conv1, conv2 in self.res_blocks:
            skip = x
            t = F.relu(conv1(x))
            t = conv2(t)
            x = F.relu(t + skip)

        pooled = x.mean(dim=(2, 3))  # (N, channels), global average pool

        # Policy: board-point logits (flattened row-major == y*W + x, matching
        # features.move_to_index) followed by the pass logit at the last index.
        p = F.relu(self.policy_conv(x))
        board_logits = torch.flatten(self.policy_point(p), 1)  # (N, H*W)
        pass_logit = self.policy_pass(pooled)                  # (N, 1)
        logits = torch.cat([board_logits, pass_logit], dim=1)  # (N, H*W + 1)

        # Value
        v = F.relu(self.value_fc1(pooled))
        value = torch.tanh(self.value_fc2(v))
        return logits, value


def build_policy_model(board_size: int, channels: int = 64, blocks: int = 4) -> PolicyValueNet:
    return PolicyValueNet(board_size, channels, blocks).to(DEVICE)


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


def forward_numpy(model: PolicyValueNet, states: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """no_grad forward that keeps the batch dimension.

    ``states`` is an (N, size, size, 3) float array; returns ``(logits, values)``
    numpy arrays shaped (N, action_size) and (N, 1). This is the single entry
    point the GUI/eval/PolicyAI consumers share so no TensorFlow/torch call sites
    are duplicated across the codebase.
    """
    device = next(model.parameters()).device
    model.eval()
    with torch.no_grad():
        x = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
        logits, values = model(x)
    return logits.cpu().numpy(), values.cpu().numpy()


def make_infer_fn(model: PolicyValueNet):
    """Return ``infer(states) -> (logits, values)`` for batched NN inference.

    ``states`` is an (N, size, size, 3) float array; the result is a tuple of
    numpy arrays: logits (N, action_size) and values (N,). This single entry
    point is shared by self-play and the GUI so MCTS can evaluate many leaves in
    one model call (the only way the GPU earns its keep here).
    """
    device = next(model.parameters()).device

    def infer(states: np.ndarray):
        model.eval()
        with torch.no_grad():
            x = torch.as_tensor(np.asarray(states), dtype=torch.float32, device=device)
            logits, values = model(x)
        return logits.cpu().numpy(), values.cpu().numpy()[:, 0]

    return infer


def save_model(model: PolicyValueNet, path: str) -> None:
    """Persist weights plus the architecture metadata needed to rebuild them."""
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)
    torch.save(
        {
            "model": model.state_dict(),
            "arch": {
                "board_size": model.board_size,
                "channels": model.channels,
                "blocks": model.blocks,
            },
        },
        path,
    )


def load_or_create_model(
    path: str, board_size: int, channels: int = 64, blocks: int = 4
) -> PolicyValueNet:
    if os.path.exists(path):
        try:
            # weights_only=True: the payload is only tensors + a metadata dict of
            # primitives, so we avoid arbitrary-code execution when loading a
            # model file supplied on the command line (e.g. eval.py --p1 <path>).
            ckpt = torch.load(path, map_location=DEVICE, weights_only=True)
        except Exception as exc:
            print(f"Failed to load model at {path}: {exc}. Creating a new model.")
            return build_policy_model(board_size, channels, blocks)

        if isinstance(ckpt, dict) and "model" in ckpt:
            arch = ckpt.get("arch", {})
            state_dict = ckpt["model"]
        else:
            arch = {}
            state_dict = ckpt

        # The net is fully convolutional, so weights are board-size independent:
        # a checkpoint trained on any size loads here (only channels/blocks must
        # match, which load_state_dict enforces via shapes). This is what lets a
        # 9x9 model warm-start 19x19. `board_size` is carried as metadata only.
        model = build_policy_model(
            board_size, arch.get("channels", channels), arch.get("blocks", blocks)
        )
        try:
            model.load_state_dict(state_dict)
        except Exception as exc:
            print(f"Failed to load model at {path}: {exc}. Creating a new model.")
            return build_policy_model(board_size, channels, blocks)
        return model
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
        logits, _values = forward_numpy(self.model, state[None, ...])
        probs = masked_softmax(logits[0], mask, self.temperature)
        idx = safe_choice(probs)
        return index_to_move(idx, board.size)
