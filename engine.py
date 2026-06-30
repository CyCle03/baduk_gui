import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

EMPTY = 0
BLACK = 1
WHITE = 2
PASS_MOVE = (-1, -1)


# Zobrist hashing tables, shared per board size so that hashes are comparable
# across cloned boards. EMPTY contributes 0 so the empty board hashes to 0 and
# incremental XOR updates stay consistent with a from-scratch recomputation.
_ZOBRIST_CACHE: dict = {}


def _get_zobrist(size: int):
    table = _ZOBRIST_CACHE.get(size)
    if table is None:
        rng = random.Random(0x9E3779B97F4A7C15 ^ size)
        table = [
            (0, rng.getrandbits(64), rng.getrandbits(64))
            for _ in range(size * size)
        ]
        _ZOBRIST_CACHE[size] = table
    return table


def opponent(c: int) -> int:
    return BLACK if c == WHITE else WHITE


@dataclass(frozen=True)
class Move:
    x: int
    y: int


def in_bounds(n: int, x: int, y: int) -> bool:
    return 0 <= x < n and 0 <= y < n


def neighbors(n: int, x: int, y: int):
    if x > 0:
        yield (x - 1, y)
    if x < n - 1:
        yield (x + 1, y)
    if y > 0:
        yield (x, y - 1)
    if y < n - 1:
        yield (x, y + 1)


def star_points(n: int) -> List[Tuple[int, int]]:
    """Hoshi (star point) coordinates for an n x n board, 0-indexed.

    Generic rule that reproduces the standard layouts: corner points sit `edge`
    in from each side (3 for boards >= 13, else 2); odd boards add a tengen
    (center) and, for boards >= 13, the four side midpoints. This yields the
    canonical 9 points on 19x19 and 13x13, and 5 points on 9x9 and 7x7.
    """
    if n < 7:
        return [(n // 2, n // 2)] if n % 2 == 1 else []
    edge = 3 if n >= 13 else 2
    last = n - 1 - edge
    pts = [(edge, edge), (edge, last), (last, edge), (last, last)]
    if n % 2 == 1:
        c = n // 2
        pts.append((c, c))
        if n >= 13:
            pts += [(edge, c), (last, c), (c, edge), (c, last)]
    seen = set()
    out = []
    for p in pts:
        if p not in seen:
            seen.add(p)
            out.append(p)
    return out


def _is_true_eye_region(board: "GoBoard", region: List[Tuple[int, int]], color: int) -> bool:
    opp = opponent(color)
    for x, y in region:
        diagonals = [
            (x - 1, y - 1),
            (x + 1, y - 1),
            (x - 1, y + 1),
            (x + 1, y + 1),
        ]
        for dx, dy in diagonals:
            if not in_bounds(board.size, dx, dy):
                continue
            if board.get(dx, dy) == opp:
                return False
    return True


class IllegalMove(Exception):
    pass


class GoBoard:
    """
    19x19 Go-like board with:
      - capture
      - suicide forbidden (unless capturing makes it legal)
      - basic ko: forbid recreating the immediate previous position
      - pass
      - undo via snapshot stack
    """

    def __init__(self, size: int = 19):
        self.size = size
        self._zobrist = _get_zobrist(size)
        self._zhash = 0  # empty board => 0; updated incrementally via set()
        self.grid = [[EMPTY] * size for _ in range(size)]
        self.to_play = BLACK
        self.consecutive_passes = 0
        self.last_pass_player: Optional[int] = None
        self.pass_streak = 0
        # Explicit move counter so move_count() works for light (history-free)
        # boards too. Kept in lockstep with play()/play_fast()/undo().
        self.num_moves = 0

        self.prisoners_black = 0  # stones captured by Black (i.e., White prisoners)
        self.prisoners_white = 0  # stones captured by White (i.e., Black prisoners)

        # for basic ko: store previous board position hash (just one-step)
        self._prev_pos_hash: Optional[int] = None

        # Undo stack: list of snapshots
        self._history: List[Tuple[List[List[int]], int, int, int, int, Optional[int], Optional[int], int]] = []

        # Start snapshot (optional) so undo works gracefully
        self._push_snapshot()

    def copy_grid(self) -> List[List[int]]:
        return [row[:] for row in self.grid]

    def _hash_position(self) -> int:
        # O(1) incremental Zobrist hash of the board position (player not
        # included, matching the basic-ko semantics of comparing board shape).
        return self._zhash

    def _zhash_full(self) -> int:
        # Recompute the Zobrist hash from scratch (validation / undo recovery).
        h = 0
        table = self._zobrist
        size = self.size
        for y in range(size):
            row = self.grid[y]
            base = y * size
            for x in range(size):
                c = row[x]
                if c:
                    h ^= table[base + x][c]
        return h

    def _push_snapshot(self):
        self._history.append(
            (
                self.copy_grid(),
                self.to_play,
                self.consecutive_passes,
                self.prisoners_black,
                self.prisoners_white,
                self._prev_pos_hash,
                self.last_pass_player,
                self.pass_streak,
            )
        )

    def undo(self) -> bool:
        # Need at least 2 snapshots to pop back
        if len(self._history) <= 1:
            return False
        self._history.pop()
        grid, to_play, passes, pb, pw, prev_hash, last_pass_player, pass_streak = self._history[-1]
        self.grid = [row[:] for row in grid]
        self.to_play = to_play
        self.consecutive_passes = passes
        self.prisoners_black = pb
        self.prisoners_white = pw
        self._prev_pos_hash = prev_hash
        self.last_pass_player = last_pass_player
        self.pass_streak = pass_streak
        self.num_moves = max(0, self.num_moves - 1)
        self._zhash = self._zhash_full()
        return True

    def move_count(self) -> int:
        # Number of moves played so far (excluding the initial snapshot).
        return self.num_moves

    def get(self, x: int, y: int) -> int:
        return self.grid[y][x]

    def set(self, x: int, y: int, c: int):
        old = self.grid[y][x]
        if old != c:
            cell = self._zobrist[y * self.size + x]
            self._zhash ^= cell[old] ^ cell[c]
            self.grid[y][x] = c

    def _collect_group_and_liberties(self, x: int, y: int) -> Tuple[List[Tuple[int, int]], int]:
        """
        Returns (stones_in_group, liberties_count)
        """
        color = self.get(x, y)
        if color == EMPTY:
            return ([], 0)
        visited = set()
        stack = [(x, y)]
        group = []
        liberties = set()

        while stack:
            cx, cy = stack.pop()
            if (cx, cy) in visited:
                continue
            visited.add((cx, cy))
            group.append((cx, cy))
            for nx, ny in neighbors(self.size, cx, cy):
                v = self.get(nx, ny)
                if v == EMPTY:
                    liberties.add((nx, ny))
                elif v == color and (nx, ny) not in visited:
                    stack.append((nx, ny))

        return (group, len(liberties))

    def _remove_stones(self, stones: List[Tuple[int, int]]) -> int:
        for x, y in stones:
            self.set(x, y, EMPTY)
        return len(stones)

    def _has_empty_neighbor(self, x: int, y: int) -> bool:
        for nx, ny in neighbors(self.size, x, y):
            if self.get(nx, ny) == EMPTY:
                return True
        return False

    def legal_moves(self) -> List[Tuple[int, int]]:
        moves = []
        # include pass always
        moves.append(PASS_MOVE)
        for y in range(self.size):
            for x in range(self.size):
                if self.get(x, y) != EMPTY:
                    continue
                # Shortcut: a point with at least one empty neighbor gives the
                # placed stone its own liberty, so it can be neither suicide nor
                # a ko recapture (a ko point is fully surrounded by definition).
                # Only fully-enclosed points need the full legality check.
                if self._has_empty_neighbor(x, y):
                    moves.append((x, y))
                elif self.is_legal(x, y):
                    moves.append((x, y))
        return moves

    def _legal_moves_full(self) -> List[Tuple[int, int]]:
        # Reference implementation: runs the full is_legal() check on every
        # empty point. Kept for equivalence testing against legal_moves().
        moves = [PASS_MOVE]
        for y in range(self.size):
            for x in range(self.size):
                if self.get(x, y) != EMPTY:
                    continue
                if self.is_legal(x, y):
                    moves.append((x, y))
        return moves

    def is_legal(self, x: int, y: int) -> bool:
        if not in_bounds(self.size, x, y):
            return False
        if self.get(x, y) != EMPTY:
            return False

        # simulate move quickly (copy-on-write minimal)
        color = self.to_play
        opp = opponent(color)

        # Save state for revert (no dependency on the history stack so this
        # works for light/history-free boards too).
        saved_grid = self.copy_grid()
        saved_zhash = self._zhash

        # Place stone
        self.set(x, y, color)

        captured_groups = []

        # Capture adjacent opponent groups that now have 0 liberties
        for nx, ny in neighbors(self.size, x, y):
            if self.get(nx, ny) == opp:
                g, libs = self._collect_group_and_liberties(nx, ny)
                if libs == 0:
                    captured_groups.append(g)

        for g in captured_groups:
            self._remove_stones(g)

        # Suicide check: our group must have at least 1 liberty after captures
        _, our_libs = self._collect_group_and_liberties(x, y)
        legal = our_libs > 0

        # Ko check: forbid recreating immediate previous position
        # basic ko compares resulting position hash with previous position hash
        new_hash = self._hash_position()
        if self._prev_pos_hash is not None and new_hash == self._prev_pos_hash:
            legal = False

        # revert simulation
        self.grid = saved_grid
        self._zhash = saved_zhash

        return legal

    def play(self, x: int, y: int):
        # PASS
        if (x, y) == PASS_MOVE:
            if self.last_pass_player == self.to_play:
                self.pass_streak += 1
            else:
                self.pass_streak = 1
                self.last_pass_player = self.to_play
            # Update ko baseline: after a real move, ko checks against previous position.
            # For pass, we still update prev_pos_hash to current position to keep ko consistent.
            self._prev_pos_hash = self._hash_position()
            self.consecutive_passes += 1
            self.to_play = opponent(self.to_play)
            self.num_moves += 1
            self._push_snapshot()
            return

        if not in_bounds(self.size, x, y):
            raise IllegalMove("out of bounds")
        if self.get(x, y) != EMPTY:
            raise IllegalMove("occupied")

        # Remember current position hash for ko comparison (immediate previous position)
        current_hash = self._hash_position()
        # Save state for revert on illegal move.
        saved_grid = self.copy_grid()
        saved_zhash = self._zhash

        color = self.to_play
        opp = opponent(color)

        # Place
        self.set(x, y, color)

        # Capture
        total_captured = 0
        captured_groups = []
        for nx, ny in neighbors(self.size, x, y):
            if self.get(nx, ny) == opp:
                g, libs = self._collect_group_and_liberties(nx, ny)
                if libs == 0:
                    captured_groups.append(g)
        for g in captured_groups:
            total_captured += self._remove_stones(g)

        # Suicide check
        _, our_libs = self._collect_group_and_liberties(x, y)
        if our_libs == 0:
            # revert
            self.grid = saved_grid
            self._zhash = saved_zhash
            raise IllegalMove("suicide")

        # Ko check: resulting position must not equal immediate previous position
        new_hash = self._hash_position()
        if self._prev_pos_hash is not None and new_hash == self._prev_pos_hash:
            # revert
            self.grid = saved_grid
            self._zhash = saved_zhash
            raise IllegalMove("ko")

        # Update prisoners
        if total_captured > 0:
            if color == BLACK:
                self.prisoners_black += total_captured
            else:
                self.prisoners_white += total_captured

        # Update turn / passes / ko baseline
        self.consecutive_passes = 0
        self.last_pass_player = None
        self.pass_streak = 0
        self._prev_pos_hash = current_hash
        self.to_play = opp
        self.num_moves += 1

        self._push_snapshot()

    def clone_light(self) -> "GoBoard":
        """Clone for MCTS simulations: shares the Zobrist table, copies position
        state, but keeps no undo history. ``move_count()`` restarts at 0 to match
        the previous standalone ``_clone_board`` behaviour used by the searches.
        Use with :meth:`play_fast`; undo is not supported on a light board.
        """
        nb = GoBoard.__new__(GoBoard)
        nb.size = self.size
        nb._zobrist = self._zobrist
        nb._zhash = self._zhash
        nb.grid = [row[:] for row in self.grid]
        nb.to_play = self.to_play
        nb.consecutive_passes = self.consecutive_passes
        nb.last_pass_player = self.last_pass_player
        nb.pass_streak = self.pass_streak
        nb.num_moves = 0
        nb.prisoners_black = self.prisoners_black
        nb.prisoners_white = self.prisoners_white
        nb._prev_pos_hash = self._prev_pos_hash
        nb._history = []
        return nb

    def play_fast(self, x: int, y: int) -> None:
        """Lightweight play for MCTS simulations: no snapshot/history bookkeeping.

        The caller is expected to only pass moves already known to be legal
        (e.g. expanded from a legal-move mask), so suicide/ko are not re-checked.
        Produces the same grid / to_play / hash state as :meth:`play`.
        """
        if (x, y) == PASS_MOVE:
            if self.last_pass_player == self.to_play:
                self.pass_streak += 1
            else:
                self.pass_streak = 1
                self.last_pass_player = self.to_play
            self._prev_pos_hash = self._zhash
            self.consecutive_passes += 1
            self.to_play = opponent(self.to_play)
            self.num_moves += 1
            return

        color = self.to_play
        opp = opponent(color)
        current_hash = self._zhash

        self.set(x, y, color)

        total_captured = 0
        captured_groups = []
        for nx, ny in neighbors(self.size, x, y):
            if self.get(nx, ny) == opp:
                g, libs = self._collect_group_and_liberties(nx, ny)
                if libs == 0:
                    captured_groups.append(g)
        for g in captured_groups:
            total_captured += self._remove_stones(g)

        if total_captured > 0:
            if color == BLACK:
                self.prisoners_black += total_captured
            else:
                self.prisoners_white += total_captured

        self.consecutive_passes = 0
        self.last_pass_player = None
        self.pass_streak = 0
        self._prev_pos_hash = current_hash
        self.to_play = opp
        self.num_moves += 1

    def score_area(self, komi: float = 6.5) -> float:
        black_stones = 0
        white_stones = 0
        black_territory = 0
        white_territory = 0
        region_id_map = {}
        region_bordering = {}
        region_cells = {}
        region_id = 0
        visited = set()

        for y in range(self.size):
            for x in range(self.size):
                v = self.get(x, y)
                # Stones are tallied in the second (life/death) pass below; here
                # we only skip them so they don't seed empty-territory regions.
                if v != EMPTY:
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
                    for nx, ny in neighbors(self.size, cx, cy):
                        nv = self.get(nx, ny)
                        if nv == EMPTY:
                            if (nx, ny) not in visited:
                                stack.append((nx, ny))
                        else:
                            bordering.add(nv)

                if bordering == {BLACK}:
                    black_territory += len(region)
                elif bordering == {WHITE}:
                    white_territory += len(region)
                for pos in region:
                    region_id_map[pos] = region_id
                region_bordering[region_id] = bordering
                region_cells[region_id] = region
                region_id += 1

        stone_visited = set()
        black_dead = 0
        white_dead = 0
        for y in range(self.size):
            for x in range(self.size):
                color = self.get(x, y)
                if color == EMPTY or (x, y) in stone_visited:
                    continue
                group = []
                stack = [(x, y)]
                stone_visited.add((x, y))
                while stack:
                    cx, cy = stack.pop()
                    group.append((cx, cy))
                    for nx, ny in neighbors(self.size, cx, cy):
                        if self.get(nx, ny) == color and (nx, ny) not in stone_visited:
                            stone_visited.add((nx, ny))
                            stack.append((nx, ny))

                opp = opponent(color)
                eye_regions = set()
                dead = True
                for gx, gy in group:
                    for nx, ny in neighbors(self.size, gx, gy):
                        if self.get(nx, ny) != EMPTY:
                            continue
                        if region_bordering.get(region_id_map.get((nx, ny))) != {color}:
                            dead = False
                        rid = region_id_map.get((nx, ny))
                        if (
                            rid is not None
                            and region_bordering.get(rid) == {color}
                            and _is_true_eye_region(self, region_cells.get(rid, []), color)
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
        return black_score - white_score


class RandomAI:
    def __init__(self, pass_prob: float = 0.03):
        self.pass_prob = pass_prob

    def select_move(self, board: GoBoard) -> Tuple[int, int]:
        moves = board.legal_moves()
        # PASS_MOVE is included; we want it rare unless no moves
        legal_non_pass = [m for m in moves if m != PASS_MOVE]
        if not legal_non_pass:
            return PASS_MOVE
        if random.random() < self.pass_prob:
            return PASS_MOVE
        return random.choice(legal_non_pass)
