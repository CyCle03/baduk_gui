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

The GUI auto-reloads the model every 10 seconds to reflect CLI training.

## Train (CLI)

```bash
.venv/bin/python train_selfplay.py --episodes 1000
```

Models are saved to:
- `models/latest.keras` (latest)
- `models/checkpoint_XXXXXX.keras` (every N episodes)

Logs include per-episode time, total time, average time, and recent 10-episode average.

## Notes

- Training is CPU-only unless TensorFlow detects a GPU.
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

GUI는 10초마다 모델을 자동 리로드해서 CLI 학습 결과를 반영합니다.

### CLI 학습

```bash
.venv/bin/python train_selfplay.py --episodes 1000
```

모델 저장 위치:
- `models/latest.keras` (최신)
- `models/checkpoint_XXXXXX.keras` (N 에피소드마다)

로그에는 판당 시간, 누적 시간, 평균 시간, 최근 10판 평균 시간이 포함됩니다.
