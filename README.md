# baduk_gui

Baduk (Go) GUI with self-play reinforcement learning.

## Requirements

- Python 3.10+
- PyQt6
- TensorFlow (for training/policy model)

## Setup

```bash
python3 -m venv .venv
.venv/bin/pip install PyQt6 tensorflow
```

## Run GUI

```bash
.venv/bin/python baduk_gui.py
```

GUI controls:
- `MODEL RELOAD`: reloads `models/latest.keras`
- `AI vs AI`: toggles self-play viewing
- `TRAIN (GUI)`: starts/stops background training
- `MCTS (AI)`: toggles MCTS move selection (PolicyAI only)
- `MCTS Sims`: number of MCTS simulations per move
- `LOAD SGF`: open SGF file and view moves
- `SGF ◀ / SGF ▶`: step through moves
- `SGF PLAY / PAUSE`: auto-play SGF
- `SGF Speed (ms)`: adjust auto-play speed

The GUI auto-reloads the model every 10 seconds to reflect CLI training.

## Train (CLI)

```bash
.venv/bin/python train_selfplay.py --episodes 1000
```

Optional flags:
- `--board-size 19`
- `--komi 6.5`
- `--save-every 10`
- `--sleep 0.0`
- `--mcts-sims 0`
- `--mcts-cpuct 1.5`

Models are saved to:
- `models/latest.keras` (latest)
- `models/checkpoint_XXXXXX.keras` (every N episodes)
Optimizer checkpoints are saved to:
- `checkpoints/` (TensorFlow checkpoint, auto-resume)

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

## Notes

- Training is CPU-only unless TensorFlow detects a GPU.
- Default max moves per game is 300.
- Model files are ignored by git (`models/`, `*.keras`).

---

## 한국어 안내

Baduk (바둑) GUI와 자가대국 강화학습 예제입니다.

### 요구 사항

- Python 3.10+
- PyQt6
- TensorFlow (학습/정책 모델용)

### 설치

```bash
python3 -m venv .venv
.venv/bin/pip install PyQt6 tensorflow
```

### GUI 실행

```bash
.venv/bin/python baduk_gui.py
```

버튼 안내:
- `MODEL RELOAD`: `models/latest.keras` 다시 불러오기
- `AI vs AI`: 자가대국 보기 토글
- `TRAIN (GUI)`: GUI에서 학습 시작/중지
- `MCTS (AI)`: MCTS 수 선택 토글(PolicyAI 필요)
- `MCTS Sims`: 수당 MCTS 시뮬레이션 수
- `LOAD SGF`: SGF 파일 열기 및 기보 보기
- `SGF ◀ / SGF ▶`: 한 수씩 이동
- `SGF PLAY / PAUSE`: 자동 재생
- `SGF Speed (ms)`: 재생 속도 조절

GUI는 10초마다 모델을 자동 리로드해서 CLI 학습 결과를 반영합니다.

### CLI 학습

```bash
.venv/bin/python train_selfplay.py --episodes 1000
```

옵션:
- `--board-size 19`
- `--komi 6.5`
- `--save-every 10`
- `--sleep 0.0` (에피소드 간 대기 시간, 0이면 없음)
- `--mcts-sims 0`
- `--mcts-cpuct 1.5`

모델 저장 위치:
- `models/latest.keras` (최신)
- `models/checkpoint_XXXXXX.keras` (N 에피소드마다)
옵티마이저 체크포인트:
- `checkpoints/` (TensorFlow 체크포인트, 재시작 시 이어짐)

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

기본 최대 수: 300

누적 판수:
- `train_state.json`에 전체 `total_episodes`가 누적 저장됩니다.
