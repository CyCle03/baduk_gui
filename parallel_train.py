"""Parallel self-play launcher: N self-play workers + 1 trainer.

The single-process trainer only uses one CPU core, but the bottleneck is
self-play (MCTS tree search on CPU). This runs several worker processes that
generate self-play games into a shared data dir, plus one trainer process that
learns from that data and writes ``latest.pt``.

Actor/learner design (no changes to the algorithm, just orchestration):
  * Workers run ``train_selfplay.py --selfplay-only`` in short batches. Each
    batch reloads ``latest.pt`` on start, so workers keep up with the trainer.
    They are pure data producers -- they never write the model.
  * The trainer runs ``train_selfplay.py --train-only`` in a loop, each batch
    re-reading the newest self-play data and saving an improved ``latest.pt``.
  * A pruner keeps the data dir bounded.

Workers default to CPU inference (``CUDA_VISIBLE_DEVICES=""``) so the GPU is
reserved for the trainer; pass ``--gpu-workers`` to share the GPU instead.

Example:
    python parallel_train.py --board-size 9 --workers 8
    python parallel_train.py --board-size 19 --workers 8 --init-from models/9x9/latest.pt

Stop with Ctrl+C; the trainer finishes its current batch and saves cleanly.
"""

import argparse
import glob
import os
import signal
import subprocess
import sys
import threading
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
TRAIN_SCRIPT = os.path.join(BASE_DIR, "train_selfplay.py")
IS_WINDOWS = os.name == "nt"
DEFAULT_BOARD_SIZE = 19

_stop = threading.Event()


def _model_path_for_size(board_size: int) -> str:
    # Mirrors train_selfplay.paths_for_size: 19x19 keeps the un-namespaced path.
    if board_size == DEFAULT_BOARD_SIZE:
        return os.path.join(BASE_DIR, "models", "latest.pt")
    return os.path.join(BASE_DIR, "models", f"{board_size}x{board_size}", "latest.pt")


def _scaled_pass_min(board_size: int) -> int:
    # 150 moves before pass is allowed on 19x19, scaled down by area so small
    # boards (which fill up sooner) still let self-play pass at a sensible point.
    return max(5, round(150 * board_size * board_size / (DEFAULT_BOARD_SIZE ** 2)))


def _popen(cmd, env, log_path=None):
    kwargs = {"env": env}
    if IS_WINDOWS:
        kwargs["creationflags"] = subprocess.CREATE_NEW_PROCESS_GROUP
    else:
        kwargs["start_new_session"] = True
    if log_path is not None:
        logf = open(log_path, "a", encoding="utf-8", buffering=1)
        kwargs["stdout"] = logf
        kwargs["stderr"] = subprocess.STDOUT
        return subprocess.Popen(cmd, **kwargs), logf
    return subprocess.Popen(cmd, **kwargs), None


def _signal_graceful(p):
    """Ask a child to stop cleanly (KeyboardInterrupt -> its finally: saves)."""
    try:
        if IS_WINDOWS:
            os.kill(p.pid, signal.CTRL_BREAK_EVENT)
        else:
            os.killpg(os.getpgid(p.pid), signal.SIGINT)
    except (ProcessLookupError, OSError):
        pass


def _terminate(p, grace=12.0):
    if p.poll() is not None:
        return
    _signal_graceful(p)
    deadline = time.time() + grace
    while p.poll() is None and time.time() < deadline:
        time.sleep(0.2)
    if p.poll() is None:
        try:
            p.kill()
        except OSError:
            pass


def _supervise(build_cmd, env, log_path=None, ready=None):
    """Run build_cmd() as a subprocess repeatedly until _stop is set.

    ``ready`` is an optional predicate; while it returns False the loop idles
    (used by the trainer to wait for the first self-play data).
    """
    while not _stop.is_set():
        if ready is not None and not ready():
            if _stop.wait(5.0):
                return
            continue
        p, logf = _popen(build_cmd(), env, log_path)
        try:
            while p.poll() is None:
                if _stop.wait(0.5):
                    _terminate(p)
                    return
        finally:
            if logf is not None:
                logf.close()
        _stop.wait(0.5)  # brief gap before respawn


def _prune_loop(data_dir, keep, interval=30.0):
    while not _stop.is_set():
        try:
            files = sorted(glob.glob(os.path.join(data_dir, "*.npz")), key=os.path.getmtime)
            for f in files[: max(0, len(files) - keep)]:
                try:
                    os.remove(f)
                except OSError:
                    pass
        except OSError:
            pass
        if _stop.wait(interval):
            return


def main():
    ap = argparse.ArgumentParser(description="Parallel self-play launcher (N workers + 1 trainer)")
    ap.add_argument("--board-size", type=int, default=9)
    # Memory (not core count) is usually the limit on Windows: every worker is a
    # separate Python+torch process. Start conservative; raise --workers if RAM
    # allows, lower it if you hit "paging file too small" (WinError 1455).
    ap.add_argument("--workers", type=int, default=min(4, max(1, (os.cpu_count() or 4) - 2)),
                    help="number of self-play worker processes (raise if RAM allows)")
    ap.add_argument("--worker-episodes", type=int, default=20,
                    help="self-play games per worker batch; smaller = fresher model, more restarts")
    ap.add_argument("--train-episodes", type=int, default=40,
                    help="train steps per trainer batch before it reloads data")
    ap.add_argument("--channels", type=int, default=64)
    ap.add_argument("--blocks", type=int, default=4)
    ap.add_argument("--mcts-sims", type=int, default=100)
    ap.add_argument("--mcts-batch", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--pass-min-moves", type=int, default=None,
                    help="moves before self-play may pass (default: scaled by board size)")
    ap.add_argument("--data-dir", type=str, default=None)
    ap.add_argument("--max-data-files", type=int, default=2000,
                    help="newest N episode files the trainer loads each batch")
    ap.add_argument("--keep-data", type=int, default=6000,
                    help="prune the shared data dir to this many newest files")
    ap.add_argument("--gpu-workers", action="store_true",
                    help="let workers use the GPU too (default: workers on CPU, GPU for trainer)")
    ap.add_argument("--init-from", type=str, default=None,
                    help="warm-start the trainer's model from another .pt (e.g. a smaller board)")
    args = ap.parse_args()

    size = args.board_size
    data_dir = args.data_dir or os.path.join(BASE_DIR, "data", f"parallel_{size}x{size}")
    log_dir = os.path.join(data_dir, "worker_logs")
    os.makedirs(log_dir, exist_ok=True)
    pass_min = args.pass_min_moves if args.pass_min_moves is not None else _scaled_pass_min(size)

    common = [
        "--board-size", str(size),
        "--channels", str(args.channels), "--blocks", str(args.blocks),
        "--data-dir", data_dir, "--no-progress",
    ]

    worker_env = os.environ.copy()
    if not args.gpu_workers:
        # "-1" reliably hides the GPU (empty string is treated as "all visible"
        # by some torch builds), so workers run on CPU and load the trainer's
        # GPU-saved latest.pt via map_location="cpu".
        worker_env["CUDA_VISIBLE_DEVICES"] = "-1"
    # Each worker plays one game at a time, so pin it to a single BLAS/torch
    # thread. Otherwise every worker spawns cpu_count intra-op threads, which
    # oversubscribes the cores and multiplies committed memory across N workers
    # (the usual cause of WinError 1455 "paging file too small"). One thread per
    # worker also maps N workers cleanly onto N cores.
    for _var in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        worker_env[_var] = "1"

    def worker_cmd():
        return [
            sys.executable, TRAIN_SCRIPT, *common,
            "--selfplay-only",
            "--episodes", str(args.worker_episodes),
            "--mcts-sims", str(args.mcts_sims), "--mcts-batch", str(args.mcts_batch),
            "--pass-min-moves", str(pass_min),
            "--log-csv", "",
        ]

    def trainer_cmd():
        cmd = [
            sys.executable, TRAIN_SCRIPT, *common,
            "--train-only",
            # Must match the workers: the trainer builds its replay buffer for
            # policy targets iff mcts-sims > 0, so it has to agree with how the
            # workers generated the data (with/without MCTS visit distributions).
            "--mcts-sims", str(args.mcts_sims),
            "--episodes", str(args.train_episodes),
            "--batch-size", str(args.batch_size),
            "--max-data-files", str(args.max_data_files),
            "--save-every", "5",
        ]
        if args.init_from:
            cmd += ["--init-from", args.init_from]
        return cmd

    def have_data():
        try:
            return any(f.endswith(".npz") for f in os.listdir(data_dir))
        except OSError:
            return False

    threads = []
    for i in range(args.workers):
        log_path = os.path.join(log_dir, f"worker{i}.log")
        t = threading.Thread(target=_supervise, args=(worker_cmd, worker_env, log_path), daemon=True)
        t.start()
        threads.append(t)
    # Trainer stays on the console (so its loss log is visible) and waits for data.
    t = threading.Thread(target=_supervise, args=(trainer_cmd, os.environ.copy(), None, have_data), daemon=True)
    t.start()
    threads.append(t)
    t = threading.Thread(target=_prune_loop, args=(data_dir, args.keep_data), daemon=True)
    t.start()
    threads.append(t)

    model_path = _model_path_for_size(size)
    print(f"[parallel] {size}x{size} | workers={args.workers}x1-thread "
          f"({'GPU' if args.gpu_workers else 'CPU'}) | trainer=GPU | pass_min={pass_min}")
    print(f"[parallel] data: {data_dir}  (worker logs in {log_dir})")
    print("[parallel] If workers die with WinError 1455 (paging file too small): "
          "lower --workers or raise the Windows paging file.")
    print("[parallel] Ctrl+C to stop.\n")

    def _on_sigint(signum, frame):
        if not _stop.is_set():
            print("\n[parallel] stopping... (letting the trainer finish its batch)")
        _stop.set()

    signal.signal(signal.SIGINT, _on_sigint)

    last_count, last_t = 0, time.time()
    try:
        while not _stop.is_set():
            if _stop.wait(30.0):
                break
            try:
                count = len(glob.glob(os.path.join(data_dir, "*.npz")))
            except OSError:
                count = last_count
            now = time.time()
            rate = (count - last_count) / max(1e-6, (now - last_t)) * 60.0
            age = (now - os.path.getmtime(model_path)) if os.path.exists(model_path) else None
            age_s = f"{age:.0f}s ago" if age is not None else "not yet"
            print(f"[parallel] data files={count} (+{rate:.0f}/min) | latest.pt: {age_s}")
            last_count, last_t = count, now
    finally:
        _stop.set()
        for t in threads:
            t.join(timeout=20)
    print("[parallel] stopped.")


if __name__ == "__main__":
    main()
