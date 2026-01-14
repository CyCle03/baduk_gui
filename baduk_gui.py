import os
import sys
from typing import List, Optional, Tuple

from PyQt6.QtCore import Qt, QPointF, QRectF, QTimer, QProcess
from PyQt6.QtGui import QPainter, QPen, QBrush, QColor, QFont, QFontDatabase
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QMessageBox
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
)

try:
    from rl_model import PolicyAI
except Exception:
    PolicyAI = None

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

        self.hover_xy: Optional[Tuple[int, int]] = None
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

        # hover hint
        if self.hover_xy is not None:
            hx, hy = self.hover_xy
            if self.board.get(hx, hy) == EMPTY and self.board.is_legal(hx, hy) and self.board.to_play == BLACK:
                p = self._xy_to_point(hx, hy)
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.setPen(QPen(QColor(0, 80, 200), 2, Qt.PenStyle.DashLine))
                painter.drawEllipse(p, stone_r, stone_r)

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
        self.train_running = False
        self.last_train_status = "대기"

        self.status = QLabel()
        self.status.setFont(pick_korean_font(12))
        self._update_status()

        self.board_widget = BoardWidget(self.board, self.on_human_move)

        self.btn_pass = QPushButton("PASS (흑)")
        self.btn_undo = QPushButton("UNDO (한 턴)")
        self.btn_new = QPushButton("NEW GAME")
        self.btn_reload = QPushButton("MODEL RELOAD")
        self.btn_selfplay = QPushButton()
        self.btn_train = QPushButton()
        self._update_selfplay_label()
        self._update_train_label()

        self.btn_pass.clicked.connect(self.on_pass)
        self.btn_undo.clicked.connect(self.on_undo_turn)
        self.btn_new.clicked.connect(self.on_new_game)
        self.btn_reload.clicked.connect(self.on_reload_model)
        self.btn_selfplay.clicked.connect(self.on_toggle_selfplay)
        self.btn_train.clicked.connect(self.on_toggle_train)

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
        self._model_reload_timer.setInterval(5000)
        self._model_reload_timer.timeout.connect(self.on_reload_model)
        

    def _make_ai(self):
        if PolicyAI is not None and os.path.exists(self.model_path):
            return PolicyAI(self.model_path, board_size=self.board.size)
        return RandomAI(pass_prob=0.03)
    def _update_status(self):
        to_play = "흑(사람)" if self.board.to_play == BLACK else "백(AI)"
        train_state = "ON" if self.train_running else "OFF"
        self.status.setText(
            f"차례: {to_play}\n"
            f"진행 수: {self.board.move_count()}\n"
            f"학습: {train_state}\n"
            f"학습 상태: {self.last_train_status}\n"
            f"포로(흑이 딴 수 / 백이 딴 수): {self.board.prisoners_black} / {self.board.prisoners_white}\n"
            f"연속 패스: {self.board.consecutive_passes}"
        )

    def _update_selfplay_label(self):
        state = "ON" if self.ai_vs_ai else "OFF"
        self.btn_selfplay.setText(f"AI vs AI (자가대국): {state}")

    def _update_train_label(self):
        state = "ON" if self.train_running else "OFF"
        self.btn_train.setText(f"TRAIN (GUI): {state}")

    def _maybe_game_end(self):
        if self.board.consecutive_passes >= 2:
            QMessageBox.information(
                self, "게임 종료(임시)",
                "연속 2패스. (MVP) 여기서 종료로 간주합니다.\n"
                "계가(사석 마킹)는 다음 단계에서 추가하는 걸 추천해요."
            )

    def on_human_move(self, x: int, y: int):
        # Human is always black in this MVP
        if self.ai_vs_ai:
            return
        if self.board.to_play != BLACK:
            return
        try:
            self.board.play(x, y)
        except IllegalMove:
            return

        self._update_status()
        self.board_widget.update()
        self._maybe_game_end()

        # If game not ended, let AI play
        if self.board.consecutive_passes < 2:
            self._ai_timer.start(50)

    def on_pass(self):
        if self.ai_vs_ai:
            return
        if self.board.to_play != BLACK:
            return
        self.board.play(*PASS_MOVE)
        self._update_status()
        self.board_widget.update()
        self._maybe_game_end()

        if self.board.consecutive_passes < 2:
            self._ai_timer.start(50)

    def _do_ai_move(self):
        if self.board.consecutive_passes >= 2:
            return
        if self.board.to_play == BLACK and not self.ai_vs_ai:
            return
        mv = self.ai.select_move(self.board)
        self.board.play(mv[0], mv[1])
        self._update_status()
        self.board_widget.update()
        self._maybe_game_end()
        if self.board.consecutive_passes < 2 and (self.ai_vs_ai or self.board.to_play == WHITE):
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

        self._update_status()
        self.board_widget.update()

    def on_new_game(self):
        self.board = GoBoard(19)
        self.board_widget.board = self.board
        self._update_status()
        self.board_widget.update()

    def on_reload_model(self):
        if PolicyAI is None:
            QMessageBox.information(self, "모델 로드 실패", "TensorFlow를 사용할 수 없습니다.")
            return
        if not os.path.exists(self.model_path):
            QMessageBox.information(self, "모델 없음", "models/latest.keras 파일이 없습니다.")
            return
        if isinstance(self.ai, PolicyAI):
            self.ai.reload()
        else:
            self.ai = self._make_ai()

    def on_toggle_selfplay(self):
        self.ai_vs_ai = not self.ai_vs_ai
        self._update_selfplay_label()
        if self.ai_vs_ai and self.board.consecutive_passes < 2:
            self._ai_timer.start(50)

    def on_toggle_train(self):
        if self.train_running:
            self._train_process.terminate()
            self._model_reload_timer.stop()
            self.train_running = False
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
    sys.exit(app.exec())

if __name__ == "__main__":
    main()
