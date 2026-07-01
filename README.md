# baduk_gui

Baduk (Go) GUI with self-play reinforcement learning.

## Requirements

- Python 3.10+
- PyQt6
- PyTorch (for training/policy model)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install PyQt6 torch
```

## Run GUI

```bash
.venv/bin/python baduk_gui.py
```

GUI controls:
- `MODEL RELOAD`: reloads `models/latest.pt`
- `AI vs AI`: toggles self-play viewing
- `TRAIN (GUI)`: starts/stops background training
- `MCTS (AI)`: toggles MCTS move selection (PolicyAI only)
- `MCTS Sims`: number of MCTS simulations per move
- `LOAD SGF`: open SGF file and view moves
- `SGF ◀ / SGF ▶`: step through moves
- `SGF PLAY / PAUSE`: auto-play SGF
- `SGF Speed (ms)`: adjust auto-play speed

Game end conditions (GUI):
- Two consecutive passes
- 300 moves reached (`GUI_MAX_MOVES`)

AI pass behavior (GUI):
- After the opponent passes, the AI will pass to end the game when its win
  probability is high enough (default threshold: `0.7`).

MCTS + self-play note:
- If MCTS is enabled, starting `AI vs AI` will keep MCTS active as long as the
  policy model is available.
- In self-play mode, MCTS adds Dirichlet noise and an opening temperature
  (defaults: alpha `0.03`, eps `0.30`, temp `1.3`, first `50` moves).
- In self-play mode, resign checks start at move 150 and PASS is blocked before
  move 150 to match CLI defaults.

The GUI auto-reloads the model every 10 seconds to reflect CLI training.
GUI writes per-game results to `logs/gui_log.csv`.
After a game ends, the GUI shows estimated territory counts (area + dead stones)
and overlays territory on the board.
If the same player passes 3 times in a row, the game is treated as a resignation.
GUI score estimation uses area scoring plus a simple dead-stone heuristic:
- A group must have at least two true eyes (diagonal checks) to be considered alive.

## Train (CLI)

```bash
.venv/bin/python train_selfplay.py --episodes 1000
```

Optional flags:
- `--board-size 19`
- `--komi 6.5`
- `--save-every 10`
- `--sleep 0.0`
- `--mcts-sims 100`
- `--mcts-cpuct 1.5`
- `--mcts-batch 1` (MCTS leaves evaluated per batched inference; `1` is the exact
  sequential search, `>1` enables virtual-loss leaf batching so the GPU is used)
- `--channels 64` / `--blocks 4` (size of newly created models)
- `--superko` / `--no-superko` (positional superko in self-play; on by default —
  forbids recreating any prior board position, preventing infinite repetition)
- `--augment` (augment training batches with random 8-fold board symmetries)
- `--eval-every 0` / `--eval-games 20` (every N episodes, run a deterministic
  vs-Random evaluation and append the winrate to `logs/eval_log.csv`)
- `--profile` (run 2 episodes under cProfile and print the top 25 functions)
- `--buffer-size 5000`
- `--batch-size 256`
- `--train-steps 1`
- `--resign-threshold 0.99`
- `--resign-start 250`
- `--resign-score-check-moves 30` (skip resign near the end when area score favors current player)
- `--data-dir ./data`
- `--selfplay-only`
- `--train-only`
- `--save-selfplay`
- `--max-data-files 0`
- `--log-csv logs/train_log.csv` (CSV log file; use empty string to disable). When using the default path, logs are split automatically:
  - GPU detected → `logs/train_log_cuda.csv`
  - CPU only → `logs/train_log_cpu.csv`
- `--progress` / `--no-progress`

Models are saved to:
- `models/latest.pt` (latest)
- `models/checkpoint_XXXXXX.pt` (every N episodes)
Optimizer checkpoints are saved to:
- `checkpoints/` (PyTorch model+optimizer checkpoint, auto-resume)

Logs include per-episode time, total time, average time, and recent 10-episode average.
Log fields:
- `episode`: self-play game index
- `loss`: total loss (policy + value)
- `policy`: policy loss
- `value`: value loss
- `score_diff`: final score (black - white)
- `moves`: number of moves (including passes)
- `episode_time`: time for the episode
- `total_time`: cumulative time
- `avg_time`: average time per episode
- `recent10_avg`: moving average of last 10 episodes

Total episode counter:
- `train_state.json` stores cumulative `total_episodes` across runs.

Self-play data modes:
- `--selfplay-only`: generate self-play data to `--data-dir` only
- `--train-only`: train from data in `--data-dir` only
- `--save-selfplay`: save self-play data while training

## Evaluation

Pit two players (a `.pt` model or `random`) against each other with
deterministic play (argmax legal move, no Dirichlet/temperature), alternating
colors for fairness:

```bash
python eval.py --p1 models/latest.pt --p2 random --games 40 --seed 0
python eval.py --p1 models/latest.pt --p2 models/checkpoint_000100.pt --games 40
```

Results (P1 winrate, average margin) are appended to `logs/eval_log.csv`. With a
fixed `--seed` a run is reproducible. During training, `--eval-every N` records
the same vs-Random curve automatically.

## Smaller boards

Both training and the GUI take `--board-size` (e.g. `9` or `13`), which is the
fastest way to iterate on experiments:

```bash
python train_selfplay.py --board-size 9 --channels 32 --blocks 3 --eval-every 50
python baduk_gui.py --board-size 9
```

Star points (hoshi) adapt to the board size automatically.

The policy/value net is fully convolutional, so its weights are board-size
independent: a model trained on a small board can warm-start a larger one with
`--init-from` (much faster than starting 19x19 from scratch):

```bash
# train 9x9 first, then seed 19x19 from it
python train_selfplay.py --board-size 9 --episodes 500
python train_selfplay.py --board-size 19 --init-from models/9x9/latest.pt --mcts-sims 100
```

`--init-from` is ignored once a model already exists for the target size (a
resume checkpoint always takes precedence).

## Parallel self-play

Self-play (MCTS on CPU) is the bottleneck, and a single trainer only uses one
core. `parallel_train.py` runs several self-play workers plus one trainer to use
the whole machine (an actor/learner split built on `--selfplay-only` /
`--train-only`):

```bash
python parallel_train.py --board-size 9 --workers 8
# warm-start a bigger board from a smaller one:
python parallel_train.py --board-size 19 --workers 8 --init-from models/9x9/latest.pt
```

Workers restart in short batches so they pick up the trainer's newest
`latest.pt`, and default to CPU inference so the GPU is reserved for the trainer
(`--gpu-workers` to share it). Per-worker logs go to
`data/parallel_<size>/worker_logs/`; the trainer's loss stays on the console.
Stop with Ctrl+C. Use `--workers`, `--mcts-sims`, `--worker-episodes`, and
`--keep-data` to tune throughput and disk use.

## Architecture notes

- `features.py` — framework-free board encoding / move indexing (vectorized).
- `rl_model.py` — fully-convolutional ResNet: per-point policy logits from a 1x1
  conv, plus a pass logit and the value from a global-average-pooled trunk, so
  one set of weights runs on any board size.
- `mcts.py` — the single MCTS implementation shared by self-play and the GUI;
  inference is injected as a function so the GPU model and test stubs share code.
- `engine.py` — board logic with incremental Zobrist hashing and a light
  (history-free) simulation path (`clone_light` / `play_fast`).

## Notes

- Training uses the GPU automatically when CUDA (`torch.cuda.is_available()`) is
  detected, otherwise CPU.
- Default max moves per game is 300.
- Model files are ignored by git (`models/`, `*.keras`, `*.pt`).

---

## 한국어 안내

Baduk (바둑) GUI와 자가대국 강화학습 예제입니다.

### 요구 사항

- Python 3.10+
- PyQt6
- PyTorch (학습/정책 모델용)

### 설치

```bash
python3 -m venv .venv
.venv/bin/pip install PyQt6 torch
```

### GUI 실행

```bash
.venv/bin/python baduk_gui.py
```

버튼 안내:
- `MODEL RELOAD`: `models/latest.pt` 다시 불러오기
- `AI vs AI`: 자가대국 보기 토글
- `TRAIN (GUI)`: GUI에서 학습 시작/중지
- `MCTS (AI)`: MCTS 수 선택 토글(PolicyAI 필요)
- `MCTS Sims`: 수당 MCTS 시뮬레이션 수
- `LOAD SGF`: SGF 파일 열기 및 기보 보기
- `SGF ◀ / SGF ▶`: 한 수씩 이동
- `SGF PLAY / PAUSE`: 자동 재생
- `SGF Speed (ms)`: 재생 속도 조절

GUI는 10초마다 모델을 자동 리로드해서 CLI 학습 결과를 반영합니다.
GUI 대국 결과는 `logs/gui_log.csv`에 기록됩니다.
대국 종료 시 추정 집(영역+사석) 수치를 표시하고, 보드에 영역 오버레이를
표시합니다.
같은 플레이어가 3번 연속 패스하면 기권 처리됩니다.
GUI 계가 추정은 면적 계산에 간단한 사석 휴리스틱을 적용합니다:
- 두 눈(대각선 검사 포함)을 만족해야 생존으로 판단합니다.

### CLI 학습

```bash
.venv/bin/python train_selfplay.py --episodes 1000
```

옵션:
- `--board-size 19`
- `--komi 6.5`
- `--save-every 10`
- `--sleep 0.0` (에피소드 간 대기 시간, 0이면 없음)
- `--mcts-sims 100`
- `--mcts-cpuct 1.5`
- `--buffer-size 5000`
- `--batch-size 256`
- `--train-steps 1`
- `--resign-threshold 0.99`
- `--resign-start 250`
- `--resign-score-check-moves 30` (막판 계가가 유리하면 기권을 건너뜀)
- `--data-dir ./data`
- `--selfplay-only`
- `--train-only`
- `--save-selfplay`
- `--max-data-files 0`
- `--log-csv logs/train_log.csv` (CSV 로그 파일, 비활성화하려면 빈 문자열). 기본 경로를 쓰면 자동 분리됩니다:
  - GPU 감지됨 → `logs/train_log_cuda.csv`
  - CPU만 사용 → `logs/train_log_cpu.csv`
- `--progress` / `--no-progress`

모델 저장 위치:
- `models/latest.pt` (최신)
- `models/checkpoint_XXXXXX.pt` (N 에피소드마다)
옵티마이저 체크포인트:
- `checkpoints/` (PyTorch 체크포인트, 재시작 시 이어짐)

로그에는 판당 시간, 누적 시간, 평균 시간, 최근 10판 평균 시간이 포함됩니다.
로그 항목:
- `episode`: 자가대국 판 번호
- `loss`: 전체 손실(정책+가치)
- `policy`: 정책 손실
- `value`: 가치 손실
- `score_diff`: 최종 점수(흑-백)
- `moves`: 진행 수(패스 포함)
- `episode_time`: 해당 판 소요 시간
- `total_time`: 누적 시간
- `avg_time`: 판당 평균 시간
- `recent10_avg`: 최근 10판 평균

GUI 종료 조건:
- 연속 2패스
- 300수 도달(`GUI_MAX_MOVES`)

AI 패스 동작(GUI):
- 상대가 패스한 직후, 승리 확률이 충분히 높으면 패스해서 바로 종료합니다
  (기본 임계값: `0.7`).

MCTS + 자가대국 참고:
- MCTS가 켜져 있으면 `AI vs AI`를 시작할 때 정책 모델이 로드된 상태에서
  계속 MCTS가 유지됩니다.
- 자가대국 모드에서는 MCTS 루트 확률에 Dirichlet 노이즈와 초반 온도를
  적용합니다(기본값: alpha `0.03`, eps `0.30`, temp `1.3`, 첫 `50`수).
- 자가대국 모드에서는 150수 이전 패스를 막고, 150수부터 기권 판단을
  시작하도록 CLI 기본값에 맞췄습니다.

기본 최대 수: 300

누적 판수:
- `train_state.json`에 전체 `total_episodes`가 누적 저장됩니다.

자가대국 데이터 모드:
- `--selfplay-only`: `--data-dir`에 자가대국 데이터만 생성
- `--train-only`: `--data-dir` 데이터를 불러 학습만 수행
- `--save-selfplay`: 학습 중 자가대국 데이터를 저장
