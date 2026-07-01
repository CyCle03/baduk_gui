# Session Log — 2026-07-01

TensorFlow → PyTorch migration, board-size-independent network, parallel
self-play, and self-play collapse hardening for `baduk_gui`.

---

## English

### Goal
Migrate the Go engine's neural network stack from TensorFlow/Keras to PyTorch,
then make training scale better (board-size-independent weights, warm-starting,
and parallel self-play), and finally diagnose and harden a self-play training
collapse.

### Environment
- Windows 11, Python 3.11, GTX 1660 (6 GB), 12 logical cores.
- PyTorch 2.5.1 + CUDA 12.1 (`torch.cuda.is_available() == True`).

### Timeline

**1. TensorFlow vs PyTorch decision.**
Chose PyTorch. Key reason: TensorFlow dropped native Windows GPU support after
2.10 (needs WSL2), while PyTorch ships native Windows CUDA wheels. Also better
reference code / ecosystem for AlphaZero-style Go.

**2. TF/Keras → PyTorch migration (PR #4, merged).**
- `rl_model.py`: `PolicyValueNet(nn.Module)`; inference centralized in
  `make_infer_fn` + new `forward_numpy`; `.pt` save/load with arch metadata.
- `train_selfplay.py`: `GradientTape` → shared `_optimize_batch`
  (backward/step), Adam, `torch.save` checkpoints with `max_to_keep=3` rotation
  and resume, `torch.cuda.is_available()`, `loss.item()` logging.
- `eval.py` / `baduk_gui.py`: route model calls through `forward_numpy`;
  `.keras` → `.pt`.
- Fixed a **pre-existing** non-MCTS crash: when passing was disallowed and the
  board filled up, the legal mask went all-zero → `safe_choice` picked an
  illegal move → `IllegalMove("occupied")`. Re-allow pass on an empty mask.
- Code-review hardening: `torch.load(weights_only=True)` (eval.py loads
  CLI-supplied paths), `model.eval()` in inference helpers, full-arch validation
  before `load_state_dict`.
- GUI: show trained-episode count; reset a stale 1077 counter left from the old
  TF model; fix an empty-mask GUI crash; scale pass/resign move thresholds by
  board size (small boards used to never pass/resign).

**3. WSL2 disk cleanup.**
Removed the old TF/CUDA/pip-cache from the Ubuntu WSL2 distro (home 27 GB → 7.4
GB, ~19.6 GB freed inside; `rknn_env` preserved). The `ext4.vhdx` does not
auto-shrink, so ran `fstrim` then an elevated `diskpart … compact vdisk`:
**31.4 GB → 12.8 GB (~18.6 GB reclaimed on the Windows drive).**

**4. Fully-convolutional network (PR #5, merged).**
The old policy/value heads used `Linear(board_size²·…)`, so weights were
board-size specific and a 9×9 model could not be used for 19×19. Rewrote the
heads to be fully convolutional (per-point 1×1-conv policy logits + a pass logit
and value from a global-average-pooled trunk). Now one set of weights runs on
any board size, and `--init-from` warm-starts a bigger board from a smaller one.
NOTE: this changed the architecture, so old `.pt` weights are incompatible and
require retraining.

**5. Parallel self-play launcher (PR #5, merged).**
`parallel_train.py` runs N self-play workers + 1 trainer (actor/learner split on
top of the existing `--selfplay-only` / `--train-only`). Fixes needed to make it
correct:
- Workers pinned to 1 BLAS thread + conservative worker default — 8 workers had
  OOM'd with `WinError 1455` ("paging file too small") because each spawned
  `cpu_count` threads.
- **CPU workers now load the trainer's GPU-saved model.** `CUDA_VISIBLE_DEVICES=""`
  didn't reliably hide the GPU (`is_available()==True` but `device_count()==0`),
  so `DEVICE` became `"cuda"` and `torch.load` failed — workers silently played
  with a fresh RANDOM model, so the whole loop never improved. Fix: `DEVICE`
  also requires `device_count() > 0`; workers hide the GPU with `"-1"`.
- Unbuffered worker stdout (`PYTHONUNBUFFERED=1`) so per-game log lines appear
  live instead of in 20-game bursts.
- Robustness: atomic `save_model` (temp + `os.replace`), unique per-episode data
  filenames, skip corrupt `.npz` on load.

**6. Branch cleanup.** Deleted all merged local + remote branches; repo down to
`main`.

**7. Self-play collapse — diagnosis.**
A 9×9 parallel run evaluated at **winrate 0.125 vs random (worse than random)**.
Diagnostics:
- A fresh untrained model scores 0.500 vs random → harness/orientation are fine.
- The trained model's `value ≈ -0.9997` on every position → the value head
  collapsed to the trivial "side-to-move → lose" shortcut (it just reads the
  side-to-move input plane), and the policy piled onto a corner.
- Mechanism: one-sided/weak self-play → value takes the shortcut → MCTS is blind
  → policy targets are garbage → policy degenerates → snowballs.
This is a training-dynamics/hyperparameter failure, not a code bug.

**8. Collapse guards (PR #6, open).**
Added standard regularizers, active by default and tunable:
- AdamW weight decay (`--weight-decay`, 1e-4) — discourages the extreme weights
  behind the value shortcut.
- Policy entropy bonus (`--entropy-coef`, 0.01) — stops the policy collapsing
  onto one move; keeps self-play diverse.
- Gradient-norm clipping (`--grad-clip`, 5.0) — caps runaway updates.
Recommended together with higher `--mcts-sims` (100+).

### Pull requests
- **#4** TF/Keras → PyTorch migration — merged.
- **#5** Fully-convolutional net + warm-start + parallel self-play — merged.
- **#6** Collapse guards (weight decay + entropy + grad clip) — open.

### Key commands
```powershell
# quick 9x9 training
python train_selfplay.py --board-size 9 --episodes 20 --mcts-sims 0 --pass-min-moves 10
# parallel self-play (N workers + trainer)
python parallel_train.py --board-size 9 --workers 8 --channels 32 --blocks 3 --mcts-sims 100
# warm-start a bigger board from a smaller one
python train_selfplay.py --board-size 19 --init-from models/9x9/latest.pt --mcts-sims 100
# evaluate vs random
python eval.py --p1 models/9x9/latest.pt --p2 random --games 40 --board-size 9 --komi 2.5 --seed 0
```

### Open items / recommendations
- Merge PR #6, then retrain 9×9 fresh with the guards + `--mcts-sims 100`; watch
  that vs-random winrate climbs above 0.5 and value does not re-collapse to −1.
- Self-play RL is finicky; may need further tuning (higher sims, entropy coef,
  komi) — no guaranteed one-shot fix.
- 19×19 from scratch is impractical on this hardware (~15–25× slower per game
  than 9×9); use 9×9 → 19×19 warm-start; consider `--gpu-workers` for 19×19.
- The non-MCTS (`--mcts-sims 0`) path is a REINFORCE objective that can diverge
  on long runs; prefer MCTS for real training.

---

## 한국어

### 목표
바둑 엔진의 신경망 스택을 TensorFlow/Keras에서 PyTorch로 이식하고, 학습이 더 잘
확장되도록(보드 크기 무관 가중치, 워밍업, 병렬 자기대국) 개선한 뒤, 마지막으로
자기대국 학습 붕괴를 진단하고 방어 코드를 추가했다.

### 환경
- Windows 11, Python 3.11, GTX 1660 (6 GB), 논리 코어 12개.
- PyTorch 2.5.1 + CUDA 12.1 (`torch.cuda.is_available() == True`).

### 진행 순서

**1. TensorFlow vs PyTorch 결정.**
PyTorch 선택. 핵심 이유: TensorFlow는 2.10 이후 Windows 네이티브 GPU 지원을 중단
(WSL2 필요)한 반면, PyTorch는 Windows 네이티브 CUDA 휠을 제공. AlphaZero류 참고
코드·생태계도 PyTorch가 유리.

**2. TF/Keras → PyTorch 이식 (PR #4, 병합).**
- `rl_model.py`: `PolicyValueNet(nn.Module)`; 추론을 `make_infer_fn` + 신규
  `forward_numpy`로 집약; arch 메타 포함 `.pt` 저장/로드.
- `train_selfplay.py`: `GradientTape` → 공유 `_optimize_batch`, Adam,
  `torch.save` 체크포인트(`max_to_keep=3` 회전·재개), `torch.cuda.is_available()`,
  `loss.item()` 로깅.
- `eval.py` / `baduk_gui.py`: 모델 호출을 `forward_numpy`로 통일; `.keras`→`.pt`.
- **기존에 있던** 비-MCTS 크래시 수정: pass가 금지된 상태에서 판이 꽉 차면 합법
  마스크가 전부 0 → `safe_choice`가 착수 불가 지점 선택 → `IllegalMove("occupied")`.
  빈 마스크일 때 pass 재허용.
- 코드리뷰 보강: `torch.load(weights_only=True)`(eval.py가 CLI 경로 로드),
  추론 헬퍼에 `model.eval()`, `load_state_dict` 전 전체 아키텍처 검증.
- GUI: 학습 판수 표시; 구 TF 모델의 stale 카운터(1077) 리셋; GUI 빈-마스크
  크래시 수정; pass/기권 임계값을 보드 크기 비례로 스케일(작은 판은 원래 영영
  패스/기권을 안 했음).

**3. WSL2 디스크 정리.**
Ubuntu WSL2에서 구 TF/CUDA/pip 캐시 제거(홈 27 GB → 7.4 GB, 내부 ~19.6 GB 확보;
`rknn_env`는 보존). `ext4.vhdx`는 자동으로 안 줄어들어 `fstrim` 후 관리자
`diskpart … compact vdisk` 실행: **31.4 GB → 12.8 GB (Windows 드라이브에서
~18.6 GB 반환).**

**4. 완전 합성곱(fully-conv) 네트워크 (PR #5, 병합).**
기존 policy/value 헤드는 `Linear(board_size²·…)`라 가중치가 보드 크기 종속이었고
9×9 모델을 19×19에 못 썼다. 헤드를 완전 합성곱(1×1 conv 점별 정책 로짓 +
global-average-pool에서 pass 로짓·value)으로 재작성. 이제 한 벌의 가중치가
아무 보드 크기에서나 돌아가고, `--init-from`으로 작은 판 모델이 큰 판을 워밍업.
주의: 아키텍처가 바뀌어 기존 `.pt`는 호환 불가 → 재학습 필요.

**5. 병렬 자기대국 런처 (PR #5, 병합).**
`parallel_train.py`가 워커 N개 + 트레이너 1개(기존 `--selfplay-only` /
`--train-only` 위의 액터/러너)를 구동. 올바른 동작을 위한 수정:
- 워커를 BLAS 1스레드로 고정 + 보수적 기본 워커 수 — 8워커가 `WinError 1455`
  ("페이징 파일 부족")로 OOM. 각 워커가 `cpu_count`만큼 스레드를 띄운 게 원인.
- **CPU 워커가 GPU 저장 모델을 로드.** `CUDA_VISIBLE_DEVICES=""`가 GPU를 확실히
  안 숨겨(`is_available()==True`인데 `device_count()==0`) `DEVICE`가 `"cuda"`가
  되어 `torch.load` 실패 → 워커가 조용히 랜덤 새 모델로 대국 → 학습 루프가 헛돎.
  수정: `DEVICE`가 `device_count() > 0`까지 요구; 워커는 `"-1"`로 GPU 숨김.
- 워커 stdout 언버퍼링(`PYTHONUNBUFFERED=1`) → 판 단위 로그가 20판 뭉텅이가
  아니라 실시간으로.
- 견고성: 원자적 `save_model`(temp + `os.replace`), 에피소드 파일명 고유화,
  손상 `.npz` 로드 시 건너뛰기.

**6. 브랜치 정리.** 병합된 로컬·원격 브랜치 전부 삭제; 저장소를 `main` 하나로.

**7. 자기대국 붕괴 — 진단.**
9×9 병렬 학습 결과가 **랜덤 상대 승률 0.125(랜덤보다 나쁨)**. 진단:
- fresh(미학습) 모델은 랜덤 상대 0.500 → 하베스/방향은 정상.
- 학습된 모델은 모든 위치에서 `value ≈ -0.9997` → value 헤드가 "이번 차례 → 패배"
  라는 껍데기 지름길(차례 평면만 읽음)로 붕괴, 정책은 구석에 몰아둠.
- 메커니즘: 한쪽으로 쏠린 약한 자기대국 → value 지름길 → MCTS가 눈멀음 → 정책
  타깃이 엉망 → 정책 퇴화 → 악순환.
코드 버그가 아니라 학습 동역학/하이퍼파라미터 문제.

**8. 붕괴 방지 가드 (PR #6, 진행 중).**
표준 정규화 추가(기본 활성·조정 가능):
- AdamW weight decay(`--weight-decay`, 1e-4) — value 지름길의 극단 가중치 억제.
- 정책 엔트로피 보너스(`--entropy-coef`, 0.01) — 정책이 한 점으로 붕괴하는 것 방지,
  자기대국 다양성 유지.
- 그래디언트 클리핑(`--grad-clip`, 5.0) — 폭주 업데이트 차단.
`--mcts-sims`(100 이상)와 함께 쓰기를 권장.

### 풀 리퀘스트
- **#4** TF/Keras → PyTorch 이식 — 병합.
- **#5** 완전 합성곱망 + 워밍업 + 병렬 자기대국 — 병합.
- **#6** 붕괴 방지 가드(weight decay + entropy + grad clip) — 진행 중.

### 주요 명령
```powershell
# 빠른 9x9 학습
python train_selfplay.py --board-size 9 --episodes 20 --mcts-sims 0 --pass-min-moves 10
# 병렬 자기대국 (워커 N개 + 트레이너)
python parallel_train.py --board-size 9 --workers 8 --channels 32 --blocks 3 --mcts-sims 100
# 작은 판으로 큰 판 워밍업
python train_selfplay.py --board-size 19 --init-from models/9x9/latest.pt --mcts-sims 100
# 랜덤 상대 평가
python eval.py --p1 models/9x9/latest.pt --p2 random --games 40 --board-size 9 --komi 2.5 --seed 0
```

### 남은 일 / 권장
- PR #6 병합 후, 가드 + `--mcts-sims 100`으로 9×9 fresh 재학습; 랜덤 상대 승률이
  0.5를 넘어 오르고 value가 −1로 재붕괴하지 않는지 확인.
- 자기대국 RL은 까다로움 — 추가 튜닝(sims↑, 엔트로피 계수, komi)이 필요할 수
  있고 한 번에 되는 보장은 없음.
- 19×19 바닥부터는 이 하드웨어로 비현실적(9×9 대비 판당 ~15–25배 느림) → 9×9 →
  19×19 워밍업; 19×19엔 `--gpu-workers` 고려.
- 비-MCTS(`--mcts-sims 0`) 경로는 REINFORCE 목적함수라 긴 학습에서 발산 가능 →
  실학습엔 MCTS 권장.
