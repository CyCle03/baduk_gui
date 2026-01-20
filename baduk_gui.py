import csv
import math
import random
import os
import re
import sys
import time
from collections import deque
from typing import Deque, Dict, List, Optional, Tuple

import numpy as np
from PyQt6.QtCore import Qt, QPointF, QRectF, QTimer, QProcess
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QFontDatabase
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox, QFileDialog, QSlider
)

from engine import (
    GoBoard,
    RandomAI,
    IllegalMove,
    BLACK,
    WHITE,
    EMPTY,
    PASS_MOVE,
    in_bounds,
    neighbors,
    opponent,
)

try:
    from rl_model import PolicyAI, encode_board, index_to_move, legal_moves_mask
except Exception:
    PolicyAI = None
    encode_board = None
    index_to_move = None
    legal_moves_mask = None

GUI_KOMI = 6.5
GUI_MAX_MOVES = 300
GUI_PASS_WIN_THRESHOLD = 0.7
GUI_PASS_MIN_MOVES = 150
GUI_MCTS_DIRICHLET_ALPHA = 0.03
GUI_MCTS_DIRICHLET_EPS = 0.30
GUI_MCTS_TEMP = 1.3
GUI_MCTS_TEMP_MOVES = 50
GUI_PASS_START = 300
GUI_VALUE_WINDOW = 20
GUI_VALUE_DELTA = 0.05
GUI_VALUE_MARGIN = 0.6
GUI_RESIGN_THRESHOLD = 0.99
GUI_RESIGN_START = 250
GUI_LOG_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "gui_log.csv")

# ----------------------------
# GUI (PyQt6)
# ----------------------------

class BoardWidget(QWidget):
    def __init__(self, board: GoBoard, on_human_move, parent=None):
        super().__init__(parent)
        self.board = board
        self.on_human_move = on_human_move
        self.setMinimumSize(760, 760)
        self.margin = 30
        self.star_points = self._star_points(board.size)
        self.territory_owner: Optional[Dict[Tuple[int, int], int]] = None
        self.show_territory = False

        self.hover_xy: Optional[Tuple[int, int]] = None
        self.last_move: Optional[Tuple[int, int]] = None
        self._territory_owner: Optional[Dict[Tuple[int, int], int]] = None
        self.setMouseTracking(True)

    def _star_points(self, n: int) -> List[Tuple[int, int]]:
        # Standard 19x19 hoshi: (3,3) etc 0-indexed => (3,3), (3,9), (3,15) ...
        if n == 19:
            pts = [3, 9, 15]
            return [(x, y) for x in pts for y in pts]
        # fallback: simple center
        c = n // 2
        return [(c, c)]

    def _cell(self) -> float:
        w = self.width() - 2 * self.margin
        h = self.height() - 2 * self.margin
        return min(w, h) / (self.board.size - 1)

    def _board_rect(self) -> QRectF:
        cell = self._cell()
        size = cell * (self.board.size - 1)
        return QRectF(self.margin, self.margin, size, size)

    def _xy_to_point(self, x: int, y: int) -> QPointF:
        cell = self._cell()
        r = self._board_rect()
        return QPointF(r.left() + x * cell, r.top() + y * cell)

    def _pos_to_xy(self, px: float, py: float) -> Optional[Tuple[int, int]]:
        r = self._board_rect()
        cell = self._cell()
        if px < r.left() - cell * 0.5 or px > r.right() + cell * 0.5:
            return None
        if py < r.top() - cell * 0.5 or py > r.bottom() + cell * 0.5:
            return None
        x = int(round((px - r.left()) / cell))
        y = int(round((py - r.top()) / cell))
        if not in_bounds(self.board.size, x, y):
            return None
        return (x, y)

    def mouseMoveEvent(self, e):
        xy = self._pos_to_xy(e.position().x(), e.position().y())
        self.hover_xy = xy
        self.update()

    def leaveEvent(self, e):
        self.hover_xy = None
        self.update()

    def mousePressEvent(self, e):
        if e.button() != Qt.MouseButton.LeftButton:
            return
        xy = self._pos_to_xy(e.position().x(), e.position().y())
        if xy is None:
            return
        self.on_human_move(xy[0], xy[1])

    def paintEvent(self, e):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # background
        painter.fillRect(self.rect(), QColor(235, 200, 120))

        # board lines
        r = self._board_rect()
        cell = self._cell()

        pen = QPen(QColor(60, 40, 20))
        pen.setWidth(2)
        painter.setPen(pen)

        for i in range(self.board.size):
            # horizontal
            y = r.top() + i * cell
            painter.drawLine(QPointF(r.left(), y), QPointF(r.right(), y))
            # vertical
            x = r.left() + i * cell
            painter.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))

        # star points
        painter.setBrush(QBrush(QColor(60, 40, 20)))
        painter.setPen(Qt.PenStyle.NoPen)
        for sx, sy in self.star_points:
            p = self._xy_to_point(sx, sy)
            painter.drawEllipse(p, 4, 4)

        # stones
        stone_r = cell * 0.45
        for y in range(self.board.size):
            for x in range(self.board.size):
                c = self.board.get(x, y)
                if c == EMPTY:
                    continue
                p = self._xy_to_point(x, y)
                if c == BLACK:
                    painter.setBrush(QBrush(QColor(20, 20, 20)))
                    painter.setPen(QPen(QColor(10, 10, 10), 1))
                else:
                    painter.setBrush(QBrush(QColor(245, 245, 245)))
                    painter.setPen(QPen(QColor(160, 160, 160), 1))
                painter.drawEllipse(p, stone_r, stone_r)

        # territory overlay (empty points only)
        if self.show_territory and self.territory_owner:
            overlay_r = stone_r * 0.6
            for y in range(self.board.size):
                for x in range(self.board.size):
                    if self.board.get(x, y) != EMPTY:
                        continue
                    owner = self.territory_owner.get((x, y), EMPTY)
                    if owner == BLACK:
                        painter.setBrush(QBrush(QColor(30, 30, 30, 90)))
                    elif owner == WHITE:
                        painter.setBrush(QBrush(QColor(230, 230, 230, 140)))
                    else:
                        continue
                    painter.setPen(Qt.PenStyle.NoPen)
                    p = self._xy_to_point(x, y)
                    painter.drawEllipse(p, overlay_r, overlay_r)

        # last move highlight
        if self.last_move is not None:
            lx, ly = self.last_move
            if in_bounds(self.board.size, lx, ly) and self.board.get(lx, ly) != EMPTY:
                p = self._xy_to_point(lx, ly)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(220, 60, 60), 2))
                painter.drawEllipse(p, stone_r * 0.6, stone_r * 0.6)

        # hover hint
        if self.hover_xy is not None:
            hx, hy = self.hover_xy
            if self.board.get(hx, hy) == EMPTY and self.board.is_legal(hx, hy) and self.board.to_play == BLACK:
                p = self._xy_to_point(hx, hy)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(0, 80, 200), 2, Qt.PenStyle.DashLine))
                painter.drawEllipse(p, stone_r, stone_r)

def parse_sgf(path: str) -> Tuple[int, List[Tuple[int, Tuple[int, int]]]]:
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    size = 19
    size_idx = content.find("SZ[")
    if size_idx != -1:
        end = content.find("]", size_idx)
        if end != -1:
            try:
                size = int(content[size_idx + 3 : end])
            except ValueError:
                raise ValueError("SGF 크기 파싱 실패")

    moves: List[Tuple[int, Tuple[int, int]]] = []
    i = 0
    while i < len(content):
        if content[i] == ";" and i + 3 < len(content) and content[i + 1] in ("B", "W"):
            player = BLACK if content[i + 1] == "B" else WHITE
            if content[i + 2] == "[":
                end = content.find("]", i + 3)
                if end == -1:
                    break
                coord = content[i + 3 : end]
                if coord == "":
                    move = PASS_MOVE
                elif len(coord) >= 2:
                    x = ord(coord[0]) - ord("a")
                    y = ord(coord[1]) - ord("a")
                    move = (x, y)
                else:
                    raise ValueError("SGF 좌표 파싱 실패")
                moves.append((player, move))
                i = end
        i += 1
    return size, moves


def _masked_softmax(logits: np.ndarray, mask: np.ndarray) -> np.ndarray:
    masked = np.where(mask > 0, logits, -1e9)
    exps = np.exp(masked - np.max(masked))
    probs = exps * mask
    total = np.sum(probs)
    if total <= 0:
        return mask / np.sum(mask)
    return probs / total


def _is_true_eye_region(board: GoBoard, region: List[Tuple[int, int]], color: int) -> bool:
    size = board.size
    opp = opponent(color)
    for x, y in region:
        diagonals = [
            (x - 1, y - 1),
            (x + 1, y - 1),
            (x - 1, y + 1),
            (x + 1, y + 1),
        ]
        for dx, dy in diagonals:
            if not in_bounds(size, dx, dy):
                continue
            if board.get(dx, dy) == opp:
                return False
    return True


def _score_area_with_dead(
    board: GoBoard, komi: float
) -> Tuple[float, int, int, int, int, Dict[Tuple[int, int], int]]:
    size = board.size
    territory_owner: Dict[Tuple[int, int], int] = {}
    region_id_map: Dict[Tuple[int, int], int] = {}
    region_bordering: Dict[int, set] = {}
    region_cells: Dict[int, List[Tuple[int, int]]] = {}
    region_id = 0
    visited = set()
    black_territory = 0
    white_territory = 0

    for y in range(size):
        for x in range(size):
            if board.get(x, y) != EMPTY:
                continue
            if (x, y) in visited:
                continue
            region = []
            bordering = set()
            stack = [(x, y)]
            while stack:
                cx, cy = stack.pop()
                if (cx, cy) in visited:
                    continue
                visited.add((cx, cy))
                region.append((cx, cy))
                for nx, ny in neighbors(size, cx, cy):
                    nv = board.get(nx, ny)
                    if nv == EMPTY:
                        if (nx, ny) not in visited:
                            stack.append((nx, ny))
                    else:
                        bordering.add(nv)
            owner = EMPTY
            if bordering == {BLACK}:
                owner = BLACK
                black_territory += len(region)
            elif bordering == {WHITE}:
                owner = WHITE
                white_territory += len(region)
            for pos in region:
                territory_owner[pos] = owner
                region_id_map[pos] = region_id
            region_bordering[region_id] = bordering
            region_cells[region_id] = region
            region_id += 1

    stone_visited = set()
    black_stones = 0
    white_stones = 0
    black_dead = 0
    white_dead = 0

    for y in range(size):
        for x in range(size):
            color = board.get(x, y)
            if color == EMPTY or (x, y) in stone_visited:
                continue
            group = []
            stack = [(x, y)]
            stone_visited.add((x, y))
            while stack:
                cx, cy = stack.pop()
                group.append((cx, cy))
                for nx, ny in neighbors(size, cx, cy):
                    if board.get(nx, ny) == color and (nx, ny) not in stone_visited:
                        stone_visited.add((nx, ny))
                        stack.append((nx, ny))

            opp = opponent(color)
            eye_regions = set()
            dead = True
            for gx, gy in group:
                for nx, ny in neighbors(size, gx, gy):
                    if board.get(nx, ny) != EMPTY:
                        continue
                    owner = territory_owner.get((nx, ny), EMPTY)
                    if owner != opp:
                        dead = False
                    rid = region_id_map.get((nx, ny))
                    if (
                        rid is not None
                        and region_bordering.get(rid) == {color}
                        and _is_true_eye_region(board, region_cells.get(rid, []), color)
                    ):
                        eye_regions.add(rid)
                if not dead and len(eye_regions) >= 2:
                    break
            if len(eye_regions) >= 2:
                dead = False
            if dead:
                if color == BLACK:
                    black_dead += len(group)
                else:
                    white_dead += len(group)
            else:
                if color == BLACK:
                    black_stones += len(group)
                else:
                    white_stones += len(group)

    black_score = black_stones + black_territory + white_dead
    white_score = white_stones + white_territory + black_dead + komi
    return (
        black_score - white_score,
        black_dead,
        white_dead,
        black_territory,
        white_territory,
        territory_owner,
    )


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


def _expand_node(model, node: _MCTSNode, board: GoBoard) -> float:
    if encode_board is None or legal_moves_mask is None:
        raise RuntimeError("rl_model helpers not available")
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


def _mcts_pick_move(
    model,
    board: GoBoard,
    num_simulations: int,
    cpuct: float,
    use_exploration: bool,
    pass_allowed: bool,
) -> Tuple[Tuple[int, int], float]:
    if index_to_move is None:
        raise RuntimeError("rl_model helpers not available")
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

        terminal = _terminal_value(sim_board, GUI_KOMI, GUI_MAX_MOVES)
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
    if not pass_allowed:
        counts[board.size * board.size] = 0.0
    if not use_exploration:
        items = root.children.items()
        if not pass_allowed:
            items = [(k, v) for k, v in items if k != board.size * board.size]
        best_action = max(items, key=lambda kv: kv[1].visit_count)[0]
        return index_to_move(best_action, board.size), root_value
    total = np.sum(counts)
    if total <= 0:
        mask = np.zeros_like(counts)
        for action_idx in root.children.keys():
            mask[action_idx] = 1.0
        probs = mask / np.sum(mask)
    else:
        probs = counts / total
        mask = (counts > 0).astype(np.float32)
    probs = _add_dirichlet_noise(probs, mask, GUI_MCTS_DIRICHLET_ALPHA, GUI_MCTS_DIRICHLET_EPS)
    if board.move_count() < GUI_MCTS_TEMP_MOVES:
        probs = _apply_temperature(probs, GUI_MCTS_TEMP)
    action_idx = int(np.random.choice(len(probs), p=probs))
    return index_to_move(action_idx, board.size), root_value


def _estimate_win_prob(model, board: GoBoard) -> Optional[float]:
    if encode_board is None:
        return None
    state = encode_board(board)
    outputs = model(state[None, ...], training=False)
    if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
        return None
    value = float(outputs[1].numpy()[0][0])
    return 0.5 * (value + 1.0)


def _estimate_value(model, board: GoBoard) -> Optional[float]:
    if encode_board is None:
        return None
    state = encode_board(board)
    outputs = model(state[None, ...], training=False)
    if not isinstance(outputs, (list, tuple)) or len(outputs) < 2:
        return None
    return float(outputs[1].numpy()[0][0])


def _pick_non_pass_move(ai, board: GoBoard) -> Tuple[int, int]:
    if isinstance(ai, PolicyAI) and encode_board is not None and legal_moves_mask is not None:
        state = encode_board(board)
        mask = legal_moves_mask(board)
        mask[board.size * board.size] = 0.0
        logits, _value = ai.model(state[None, ...], training=False)
        probs = _masked_softmax(logits.numpy()[0], mask)
        idx = int(np.random.choice(len(probs), p=probs))
        return index_to_move(idx, board.size)
    legal_non_pass = [m for m in board.legal_moves() if m != PASS_MOVE]
    if legal_non_pass:
        return random.choice(legal_non_pass)
    return PASS_MOVE


class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Baduk 19x19 (Korean-rules-ish MVP) - Human(B) vs RandomAI(W)")

        self.board = GoBoard(19)
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(base_dir, "models", "latest.keras")
        self.train_script = os.path.join(base_dir, "train_selfplay.py")
        self.ai = self._make_ai()
        self.ai_vs_ai = False
        self.use_mcts = True
        self.mcts_simulations = 100
        self.mcts_cpuct = 1.5
        self.train_running = False
        self.game_over_shown = False
        self.last_train_status = "대기"
        self.model_status = "대기"
        self.sgf_mode = False
        self.sgf_moves: List[Tuple[int, Tuple[int, int]]] = []
        self.sgf_index = 0
        self._recent_values: Deque[float] = deque(maxlen=GUI_VALUE_WINDOW)
        self._gui_log = None
        self._gui_log_writer = None

        self.status = QLabel()
        self.status.setFont(pick_korean_font(12))
        self.status.setWordWrap(True)
        self.status.setMinimumWidth(260)
        self._update_status()

        self.board_widget = BoardWidget(self.board, self.on_human_move)
        self.last_move: Optional[Tuple[int, int]] = None

        self.btn_pass = QPushButton("PASS (흑)")
        self.btn_undo = QPushButton("UNDO (한 턴)")
        self.btn_new = QPushButton("NEW GAME")
        self.btn_reload = QPushButton("MODEL RELOAD")
        self.btn_selfplay = QPushButton()
        self.btn_train = QPushButton()
        self.btn_mcts = QPushButton()
        self.mcts_sims = QSlider(Qt.Orientation.Horizontal)
        self.mcts_sims.setRange(10, 200)
        self.mcts_sims.setValue(self.mcts_simulations)
        self.mcts_sims.setSingleStep(10)
        self.mcts_sims.setPageStep(10)
        self.btn_load_sgf = QPushButton("LOAD SGF")
        self.btn_sgf_prev = QPushButton("SGF ◀")
        self.btn_sgf_next = QPushButton("SGF ▶")
        self.btn_sgf_exit = QPushButton("EXIT SGF")
        self.btn_sgf_play = QPushButton("SGF PLAY")
        self.btn_sgf_stats = QPushButton("SGF STATS")
        self.sgf_speed = QSlider(Qt.Orientation.Horizontal)
        self.sgf_speed.setRange(100, 2000)
        self.sgf_speed.setValue(400)
        self.sgf_speed.setSingleStep(50)
        self.sgf_speed.setPageStep(100)
        self._update_selfplay_label()
        self._update_train_label()
        self._update_mcts_label()
        self._update_sgf_controls()

        self.btn_pass.clicked.connect(self.on_pass)
        self.btn_undo.clicked.connect(self.on_undo_turn)
        self.btn_new.clicked.connect(self.on_new_game)
        self.btn_reload.clicked.connect(self.on_reload_model)
        self.btn_selfplay.clicked.connect(self.on_toggle_selfplay)
        self.btn_train.clicked.connect(self.on_toggle_train)
        self.btn_mcts.clicked.connect(self.on_toggle_mcts)
        self.mcts_sims.valueChanged.connect(self.on_mcts_sims_changed)
        self.btn_load_sgf.clicked.connect(self.on_load_sgf)
        self.btn_sgf_prev.clicked.connect(self.on_sgf_prev)
        self.btn_sgf_next.clicked.connect(self.on_sgf_next)
        self.btn_sgf_exit.clicked.connect(self.on_exit_sgf)
        self.btn_sgf_play.clicked.connect(self.on_toggle_sgf_play)
        self.btn_sgf_stats.clicked.connect(self.on_sgf_stats)
        self.sgf_speed.valueChanged.connect(self.on_sgf_speed_changed)

        side = QVBoxLayout()
        side.addWidget(self.status)
        side.addSpacing(10)
        side.addWidget(self.btn_pass)
        side.addWidget(self.btn_undo)
        side.addWidget(self.btn_new)
        side.addSpacing(10)
        side.addWidget(self.btn_reload)
        side.addWidget(self.btn_selfplay)
        side.addWidget(self.btn_train)
        side.addWidget(self.btn_mcts)
        side.addWidget(QLabel("MCTS Sims"))
        side.addWidget(self.mcts_sims)
        side.addSpacing(10)
        side.addWidget(self.btn_load_sgf)
        side.addWidget(self.btn_sgf_prev)
        side.addWidget(self.btn_sgf_next)
        side.addWidget(self.btn_sgf_play)
        side.addWidget(QLabel("SGF Speed (ms)"))
        side.addWidget(self.sgf_speed)
        side.addWidget(self.btn_sgf_stats)
        side.addWidget(self.btn_sgf_exit)
        side.addStretch(1)

        root = QHBoxLayout()
        root.addWidget(self.board_widget, 1)
        root.addLayout(side)

        self.setLayout(root)
        self.setMinimumSize(980, 820)

        # to avoid re-entrancy, queue AI move after human move
        self._ai_timer = QTimer(self)
        self._ai_timer.setSingleShot(True)
        self._ai_timer.timeout.connect(self._do_ai_move)

        # background training process + model auto-reload
        self._train_process = QProcess(self)
        self._train_process.setWorkingDirectory(base_dir)
        self._train_process.finished.connect(self._on_train_finished)
        self._train_process.readyReadStandardOutput.connect(self._on_train_output)

        self._model_reload_timer = QTimer(self)
        self._model_reload_timer.setInterval(10000)
        self._model_reload_timer.timeout.connect(self.on_reload_model)
        # Always auto-reload to reflect external CLI training.
        self._model_reload_timer.start()

        self._sgf_play_timer = QTimer(self)
        self._sgf_play_timer.setInterval(self.sgf_speed.value())
        self._sgf_play_timer.timeout.connect(self._sgf_play_step)
        self._setup_gui_log()
        

    def _make_ai(self):
        if PolicyAI is not None and os.path.exists(self.model_path):
            return PolicyAI(self.model_path, board_size=self.board.size)
        return RandomAI(pass_prob=0.03)
    def _update_status(self):
        to_play = "흑(사람)" if self.board.to_play == BLACK else "백(AI)"
        train_state = "ON" if self.train_running else "OFF"
        mcts_state = f"ON ({self.mcts_simulations})" if self.use_mcts else "OFF"
        sgf_state = ""
        if self.sgf_mode:
            sgf_state = f"\nSGF: {self.sgf_index}/{len(self.sgf_moves)}"
        self.status.setText(
            f"차례: {to_play}\n"
            f"진행 수: {self.board.move_count()}\n"
            f"학습: {train_state}\n"
            f"학습 상태: {self.last_train_status}\n"
            f"MCTS: {mcts_state}\n"
            f"모델: {self.model_status}\n"
            f"포로(흑이 딴 수 / 백이 딴 수): {self.board.prisoners_black} / {self.board.prisoners_white}\n"
            f"연속 패스: {self.board.consecutive_passes}"
            f"{sgf_state}"
        )

    def _update_selfplay_label(self):
        state = "ON" if self.ai_vs_ai else "OFF"
        self.btn_selfplay.setText(f"AI vs AI (자가대국): {state}")

    def _update_train_label(self):
        state = "ON" if self.train_running else "OFF"
        self.btn_train.setText(f"TRAIN (GUI): {state}")

    def _update_mcts_label(self):
        state = "ON" if self.use_mcts else "OFF"
        self.btn_mcts.setText(f"MCTS (AI): {state}")

    def _update_sgf_controls(self):
        active = self.sgf_mode
        self.btn_sgf_prev.setEnabled(active)
        self.btn_sgf_next.setEnabled(active)
        self.btn_sgf_exit.setEnabled(active)
        self.btn_sgf_play.setEnabled(active)
        self.sgf_speed.setEnabled(active)

    def _ensure_policy_ai(self) -> bool:
        if PolicyAI is None:
            QMessageBox.information(self, "MCTS 불가", "TensorFlow를 사용할 수 없습니다.")
            return False
        if not os.path.exists(self.model_path):
            QMessageBox.information(self, "MCTS 불가", "models/latest.keras 파일이 없습니다.")
            return False
        if not isinstance(self.ai, PolicyAI):
            self.ai = self._make_ai()
        return isinstance(self.ai, PolicyAI)

    def _should_pass_after_opponent(self) -> bool:
        if self.board.consecutive_passes != 1:
            return False
        if self.ai_vs_ai:
            return False
        if PolicyAI is None:
            return False
        if not isinstance(self.ai, PolicyAI):
            return False
        try:
            win_prob = _estimate_win_prob(self.ai.model, self.board)
        except Exception:
            return False
        if win_prob is None:
            return False
        return win_prob >= GUI_PASS_WIN_THRESHOLD

    def _should_resign_selfplay(self, value: float) -> bool:
        if not self.ai_vs_ai:
            return False
        if self.board.move_count() < GUI_RESIGN_START:
            return False
        return value <= -GUI_RESIGN_THRESHOLD

    def _should_pass_selfplay(self, value: float, pass_allowed: bool) -> bool:
        if not self.ai_vs_ai or not pass_allowed:
            return False
        if self.board.move_count() < GUI_PASS_START:
            return False
        if len(self._recent_values) < GUI_VALUE_WINDOW:
            return False
        stable = max(self._recent_values) - min(self._recent_values) <= GUI_VALUE_DELTA
        return stable and abs(value) >= GUI_VALUE_MARGIN

    def _setup_gui_log(self):
        try:
            os.makedirs(os.path.dirname(GUI_LOG_CSV), exist_ok=True)
            self._gui_log = open(GUI_LOG_CSV, "a", encoding="utf-8", newline="")
            self._gui_log_writer = csv.DictWriter(
                self._gui_log,
                fieldnames=[
                    "timestamp",
                    "mode",
                    "mcts",
                    "mcts_sims",
                    "moves",
                    "score_diff",
                    "reason",
                ],
            )
            if self._gui_log.tell() == 0:
                self._gui_log_writer.writeheader()
                self._gui_log.flush()
        except OSError:
            self._gui_log = None
            self._gui_log_writer = None

    def _log_gui_game(self, score_diff: float, reason: str):
        if self._gui_log_writer is None:
            return
        self._gui_log_writer.writerow(
            {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "mode": "AI vs AI" if self.ai_vs_ai else "Human vs AI",
                "mcts": "on" if self.use_mcts else "off",
                "mcts_sims": self.mcts_simulations,
                "moves": self.board.move_count(),
                "score_diff": f"{score_diff:.1f}",
                "reason": reason,
            }
        )
        self._gui_log.flush()

    def _show_resign(self, resign_player: int):
        winner = "백" if resign_player == BLACK else "흑"
        self.game_over_shown = True
        QMessageBox.information(
            self,
            "게임 종료(임시)",
            f"기권으로 {winner} 승.\n"
            "사석 추정 포함 자동 계산입니다.",
        )
        score_diff = -1.0 if resign_player == BLACK else 1.0
        self._log_gui_game(score_diff, "기권")

    def _check_three_pass_resign(self) -> bool:
        if self.board.pass_streak >= 3 and self.board.last_pass_player is not None:
            self._show_resign(self.board.last_pass_player)
            return True
        return False

    def _maybe_game_end(self):
        is_pass_end = self.board.consecutive_passes >= 2
        is_move_end = self.board.move_count() >= GUI_MAX_MOVES
        if (is_pass_end or is_move_end) and not self.game_over_shown:
            (
                score_diff,
                black_dead,
                white_dead,
                black_territory,
                white_territory,
                territory_owner,
            ) = _score_area_with_dead(self.board, GUI_KOMI)
            winner = "흑" if score_diff > 0 else "백"
            result = f"{winner} 승 ({abs(score_diff):.1f}점 차)"
            reason = "연속 2패스" if is_pass_end else f"{GUI_MAX_MOVES}수 도달"
            black_houses = black_territory + white_dead
            white_houses = white_territory + black_dead
            dead_info = f"사석(추정): 흑 {black_dead} / 백 {white_dead}"
            house_info = (
                f"집(추정): 흑 {black_houses} (영역 {black_territory} + 사석 {white_dead}) / "
                f"백 {white_houses} (영역 {white_territory} + 사석 {black_dead})"
            )
            self.game_over_shown = True
            QMessageBox.information(
                self, "게임 종료(임시)",
                f"{reason}. (MVP) 여기서 종료로 간주합니다.\n"
                f"계가 결과: {result} (코미 {GUI_KOMI})\n"
                f"{dead_info}\n"
                f"{house_info}\n"
                "사석 추정 포함 자동 계산입니다."
            )
            self._territory_owner = territory_owner
            self.board_widget.territory_owner = territory_owner
            self.board_widget.show_territory = True
            self.board_widget.update()
            self._log_gui_game(score_diff, reason)

    def on_human_move(self, x: int, y: int):
        # Human is always black in this MVP
        if self.sgf_mode:
            return
        if self.ai_vs_ai:
            return
        if self.board.to_play != BLACK:
            return
        try:
            self.board.play(x, y)
        except IllegalMove:
            return

        self.last_move = (x, y)
        self.board_widget.last_move = self.last_move
        if self.board.consecutive_passes < 2:
            self.game_over_shown = False
        self._update_status()
        self.board_widget.update()
        self._maybe_game_end()

        # If game not ended, let AI play
        if self.board.consecutive_passes < 2 and self.board.move_count() < GUI_MAX_MOVES:
            self._ai_timer.start(50)

    def on_pass(self):
        if self.sgf_mode:
            return
        if self.ai_vs_ai:
            return
        if self.board.to_play != BLACK:
            return
        self.board.play(*PASS_MOVE)
        self.last_move = None
        self.board_widget.last_move = None
        if self.board.consecutive_passes < 2:
            self.game_over_shown = False
        self._update_status()
        self.board_widget.update()
        if self._check_three_pass_resign():
            return
        self._maybe_game_end()

        if self.board.consecutive_passes < 2 and self.board.move_count() < GUI_MAX_MOVES:
            self._ai_timer.start(50)

    def _do_ai_move(self):
        if self.board.consecutive_passes >= 2 or self.board.move_count() >= GUI_MAX_MOVES:
            return
        if self.sgf_mode:
            return
        if self.board.to_play == BLACK and not self.ai_vs_ai:
            return
        pass_allowed = not self.ai_vs_ai or self.board.move_count() >= GUI_PASS_MIN_MOVES
        mv = None
        value = None
        if self.use_mcts and self._ensure_policy_ai():
            try:
                mv, value = _mcts_pick_move(
                    self.ai.model,
                    self.board,
                    self.mcts_simulations,
                    self.mcts_cpuct,
                    self.ai_vs_ai,
                    pass_allowed,
                )
            except Exception:
                mv = None
        if mv is None:
            if self.ai_vs_ai and isinstance(self.ai, PolicyAI):
                value = _estimate_value(self.ai.model, self.board)
            mv = self.ai.select_move(self.board)
        if self.ai_vs_ai and value is not None:
            if self._should_resign_selfplay(value):
                self._show_resign(self.board.to_play)
                return
            self._recent_values.append(value)
            if self._should_pass_selfplay(value, pass_allowed):
                mv = PASS_MOVE
        if self._should_pass_after_opponent():
            mv = PASS_MOVE
        if not pass_allowed and mv == PASS_MOVE:
            mv = _pick_non_pass_move(self.ai, self.board)
        self.board.play(mv[0], mv[1])
        self.last_move = None if mv == PASS_MOVE else (mv[0], mv[1])
        self.board_widget.last_move = self.last_move
        if self.board.consecutive_passes < 2:
            self.game_over_shown = False
        self._update_status()
        self.board_widget.update()
        if self._check_three_pass_resign():
            return
        self._maybe_game_end()
        if (
            self.board.consecutive_passes < 2
            and self.board.move_count() < GUI_MAX_MOVES
            and (self.ai_vs_ai or self.board.to_play == WHITE)
        ):
            self._ai_timer.start(50)

    def on_undo_turn(self):
        # Undo "one full turn": typically two plies (human + AI)
        # If it's AI to play, undo one ply (AI's pending), then one more if possible.
        undone_any = False

        # Always try to undo at least one move (or pass)
        if self.board.undo():
            undone_any = True

        # Try to undo second ply so user gets back to their turn
        # After one undo, if to_play is WHITE, it means we reverted to a state where AI would play,
        # so undo one more.
        if self.board.to_play == WHITE:
            if self.board.undo():
                undone_any = True

        if not undone_any:
            return

        self.last_move = None
        self.board_widget.last_move = None
        self.game_over_shown = False
        self._recent_values.clear()
        self._territory_owner = None
        self.board_widget.territory_owner = None
        self.board_widget.show_territory = False
        self._update_status()
        self.board_widget.update()

    def on_new_game(self):
        if self.sgf_mode:
            return
        self.board = GoBoard(19)
        self.board_widget.board = self.board
        self.last_move = None
        self.board_widget.last_move = None
        self.game_over_shown = False
        self._recent_values.clear()
        self._territory_owner = None
        self.board_widget.territory_owner = None
        self.board_widget.show_territory = False
        self._update_status()
        self.board_widget.update()
        if self.ai_vs_ai:
            self._ai_timer.start(50)

    def on_reload_model(self):
        if PolicyAI is None:
            self.model_status = "TensorFlow 없음"
            QMessageBox.information(self, "모델 로드 실패", "TensorFlow를 사용할 수 없습니다.")
            self._update_status()
            return
        if not os.path.exists(self.model_path):
            self.model_status = "모델 없음"
            QMessageBox.information(self, "모델 없음", "models/latest.keras 파일이 없습니다.")
            self._update_status()
            return
        if isinstance(self.ai, PolicyAI):
            self.ai.reload()
        else:
            self.ai = self._make_ai()
        self.model_status = "모델 로드됨"
        self._update_status()

    def on_toggle_selfplay(self):
        self.ai_vs_ai = not self.ai_vs_ai
        self._update_selfplay_label()
        if self.ai_vs_ai and self.use_mcts:
            if not self._ensure_policy_ai():
                self.use_mcts = False
                self._update_mcts_label()
                self._update_status()
        if (
            self.ai_vs_ai
            and self.board.consecutive_passes < 2
            and self.board.move_count() < GUI_MAX_MOVES
        ):
            self._ai_timer.start(50)

    def on_toggle_train(self):
        if self.train_running:
            self._train_process.terminate()
            self._model_reload_timer.stop()
            self.train_running = False
            self.last_train_status = "중지됨"
            self._update_train_label()
            self._update_status()
            return

        if PolicyAI is None:
            QMessageBox.information(self, "학습 불가", "TensorFlow를 사용할 수 없습니다.")
            return
        if not os.path.exists(self.train_script):
            QMessageBox.information(self, "학습 스크립트 없음", "train_selfplay.py를 찾을 수 없습니다.")
            return

        self._train_process.start(sys.executable, [self.train_script])
        if not self._train_process.waitForStarted(3000):
            QMessageBox.information(self, "학습 시작 실패", "학습 프로세스를 시작하지 못했습니다.")
            return

        self.train_running = True
        self.last_train_status = "시작됨"
        self._model_reload_timer.start()
        self._update_train_label()
        self._update_status()

    def on_toggle_mcts(self):
        if self.use_mcts:
            self.use_mcts = False
            self._update_mcts_label()
            self._update_status()
            return
        if not self._ensure_policy_ai():
            return
        self.use_mcts = True
        self._update_mcts_label()
        self._update_status()

    def on_mcts_sims_changed(self, value: int):
        self.mcts_simulations = value
        self._update_status()

    def _on_train_finished(self):
        if self.train_running:
            self.train_running = False
            self.last_train_status = "중지됨"
            self._model_reload_timer.stop()
            self._update_train_label()
            self._update_status()

    def _on_train_output(self):
        data = bytes(self._train_process.readAllStandardOutput()).decode("utf-8", errors="ignore")
        lines = [line.strip() for line in data.splitlines() if line.strip()]
        if not lines:
            return
        self.last_train_status = lines[-1]
        self._update_status()

    def on_load_sgf(self):
        path, _ = QFileDialog.getOpenFileName(self, "SGF 열기", "", "SGF Files (*.sgf)")
        if not path:
            return
        try:
            size, moves = parse_sgf(path)
        except ValueError as e:
            QMessageBox.information(self, "SGF 오류", str(e))
            return
        if size != self.board.size:
            QMessageBox.information(self, "SGF 크기 불일치", f"SGF 보드 크기 {size}는 지원하지 않습니다.")
            return
        self.sgf_mode = True
        self.sgf_moves = moves
        self.sgf_index = 0
        self._apply_sgf_index()
        self._update_sgf_controls()

    def on_sgf_prev(self):
        if not self.sgf_mode:
            return
        if self.sgf_index <= 0:
            return
        self.sgf_index -= 1
        self._apply_sgf_index()

    def on_sgf_next(self):
        if not self.sgf_mode:
            return
        if self.sgf_index >= len(self.sgf_moves):
            return
        self.sgf_index += 1
        self._apply_sgf_index()

    def on_toggle_sgf_play(self):
        if not self.sgf_mode:
            return
        if self._sgf_play_timer.isActive():
            self._sgf_play_timer.stop()
            self.btn_sgf_play.setText("SGF PLAY")
            return
        if self.sgf_index >= len(self.sgf_moves):
            return
        self._sgf_play_timer.start()
        self.btn_sgf_play.setText("SGF PAUSE")

    def on_sgf_speed_changed(self, value: int):
        self._sgf_play_timer.setInterval(value)

    def on_exit_sgf(self):
        if not self.sgf_mode:
            return
        self._sgf_play_timer.stop()
        self.btn_sgf_play.setText("SGF PLAY")
        self.sgf_mode = False
        self.sgf_moves = []
        self.sgf_index = 0
        self.board = GoBoard(19)
        self.board_widget.board = self.board
        self.last_move = None
        self.board_widget.last_move = None
        self.game_over_shown = False
        self._recent_values.clear()
        self._territory_owner = None
        self.board_widget.territory_owner = None
        self.board_widget.show_territory = False
        self._update_sgf_controls()
        self._update_status()
        self.board_widget.update()

    def on_sgf_stats(self):
        base_dir = os.path.dirname(os.path.abspath(__file__))
        sgf_dir = os.path.join(base_dir, "sgf")
        if not os.path.isdir(sgf_dir):
            QMessageBox.information(self, "SGF 통계", "sgf 폴더가 없습니다.")
            return
        files = [f for f in os.listdir(sgf_dir) if f.lower().endswith(".sgf")]
        if not files:
            QMessageBox.information(self, "SGF 통계", "SGF 파일이 없습니다.")
            return

        re_result = re.compile(r"RE\[([^\]]+)\]")
        re_moves = re.compile(r";[BW]\[")
        b_wins = 0
        w_wins = 0
        total_moves = 0
        for name in files:
            path = os.path.join(sgf_dir, name)
            try:
                text = open(path, "r", encoding="utf-8").read()
            except OSError:
                continue
            m = re_result.search(text)
            result = m.group(1) if m else ""
            if result.startswith("B+"):
                b_wins += 1
            elif result.startswith("W+"):
                w_wins += 1
            total_moves += len(re_moves.findall(text))

        count = b_wins + w_wins if (b_wins + w_wins) > 0 else len(files)
        avg_moves = total_moves / count if count > 0 else 0
        QMessageBox.information(
            self,
            "SGF 통계",
            f"총 {len(files)}판\\n"
            f"흑 승: {b_wins} / 백 승: {w_wins}\\n"
            f"평균 수: {avg_moves:.1f}",
        )

    def _sgf_play_step(self):
        if not self.sgf_mode:
            return
        if self.sgf_index >= len(self.sgf_moves):
            self._sgf_play_timer.stop()
            self.btn_sgf_play.setText("SGF PLAY")
            return
        self.sgf_index += 1
        self._apply_sgf_index()

    def _apply_sgf_index(self):
        self.board = GoBoard(self.board.size)
        for i in range(self.sgf_index):
            player, move = self.sgf_moves[i]
            # Ensure to_play matches SGF sequence.
            if self.board.to_play != player:
                self.board.to_play = player
            try:
                self.board.play(move[0], move[1])
            except IllegalMove:
                self._sgf_play_timer.stop()
                self.btn_sgf_play.setText("SGF PLAY")
                QMessageBox.information(
                    self,
                    "SGF 오류",
                    f"{i + 1}번째 수에서 IllegalMove가 발생했습니다. SGF 재생을 종료합니다.",
                )
                self.sgf_mode = False
                self.sgf_moves = []
                self.sgf_index = 0
                self.board = GoBoard(19)
                self.board_widget.board = self.board
                self.last_move = None
                self.board_widget.last_move = None
                self.game_over_shown = False
                self._update_sgf_controls()
                self._update_status()
                self.board_widget.update()
                return
        self.board_widget.board = self.board
        if self.sgf_index > 0:
            _, move = self.sgf_moves[self.sgf_index - 1]
            self.last_move = None if move == PASS_MOVE else move
        else:
            self.last_move = None
        self.board_widget.last_move = self.last_move
        self.game_over_shown = False
        self._update_status()
        self.board_widget.update()

def pick_korean_font(size: int) -> QFont:
    if sys.platform.startswith("win"):
        return QFont("Malgun Gothic", size)

    preferred = [
        "Noto Sans CJK KR",
        "Noto Sans KR",
        "NanumGothic",
        "UnDotum",
        "AppleGothic",
    ]
    available = set(QFontDatabase.families())
    for name in preferred:
        if name in available:
            return QFont(name, size)

    for name in QFontDatabase.families():
        if QFontDatabase.WritingSystem.Korean in QFontDatabase.writingSystems(name):
            return QFont(name, size)

    return QFont("Sans Serif", size)

def main():
    app = QApplication(sys.argv)
    app.setFont(pick_korean_font(12))
    w = MainWindow()
    w.show()
    rc = app.exec()
    if w._gui_log is not None:
        w._gui_log.close()
    sys.exit(rc)

if __name__ == "__main__":
    main()
