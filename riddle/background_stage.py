from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from copy import deepcopy
from datetime import datetime, timezone
import os
from pathlib import Path
import socket
import threading
import time

from .storage import write_json

NATIVE_METHODS = ("riddle", "iad", "supervised")
BARRIER_POLL_SECONDS = 0.5


@contextmanager
def log_background_preparation(args, contract):
    from .worker_progress import emit_message

    host = socket.gethostname()
    node = os.environ.get("RIDDLE_NODE_NAME") or (
        "unavailable" if os.environ.get("KUBERNETES_SERVICE_HOST") else host
    )
    gpu = "CPU" if str(args.device) == "cpu" else contract.get("environment", {}).get("gpu") or "unavailable"
    injection = (contract.get("inputs", {}).get("injection_scan") or {}).get("signal_events")
    identity = (f"method={args.method}; seed={args.seed}; scenario={args.scenario}; "
                f"signal_events={injection if injection is not None else 'nominal'}; "
                f"gpu={gpu}; node={node}; host={host}; "
                f"bg_workers={getattr(args, 'background_concurrency', 1)}; "
                f"resume_requested={bool(getattr(args, 'resume', False))}")
    started_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    started = time.monotonic()
    emit_message(f"BG stages START; start_utc={started_utc}; {identity}", level=0, durable=True)
    status, error_type = "completed", "none"
    try:
        yield
    except BaseException as error:
        status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        error_type = type(error).__name__
        raise
    finally:
        elapsed = time.monotonic() - started
        ended_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        emit_message(
            f"BG stages END; start_utc={started_utc}; end_utc={ended_utc}; "
            f"elapsed_seconds={elapsed:.3f}; status={status}; error_type={error_type}; {identity}",
            kind="INFO" if status == "completed" else "WARNING", level=0, durable=True,
        )


def record_timing(output, name, seconds, **values):
    if os.environ.get("RIDDLE_BG_BENCHMARK") == "1":
        write_json(Path(output) / ".resume" / f"timing_{name}.json",
                   {"seconds": seconds, **values})


def record_background_stage(output, preparation_seconds):
    if os.environ.get("RIDDLE_BG_BENCHMARK") == "1":
        write_json(Path(output) / ".resume" / "background_stage.json",
                   {"preparation_seconds": preparation_seconds, "checkpoint_seconds": 0.0})


def barrier_paths(output):
    root = Path(output) / ".resume"
    return root / "background_stage.ready", root / "background_stage.release"


def task_output(task):
    if any(len(getattr(task, name)) != 1 for name in ("methods", "scenarios", "seeds")):
        raise ValueError("Background tasks require one method, scenario and seed")
    return (Path(task.output).resolve() / task.methods[0] / task.scenarios[0]
            / f"seed_{task.seeds[0]:03d}")


def clear_stage_checkpoint(output):
    root = Path(output) / ".resume"
    for path in (root / "background_stage.json", root / "background_stage.pt"):
        path.unlink(missing_ok=True)


def clear_barrier(output):
    for path in barrier_paths(output):
        path.unlink(missing_ok=True)


def wait_for_stage_release(output):
    from .worker_progress import emit_message

    ready, release = barrier_paths(output)
    ready.parent.mkdir(parents=True, exist_ok=True)
    write_json(ready, {"pid": os.getpid(), "ready": True})
    emit_message("Background preparation complete; waiting for fitting slot", level=0)
    while not release.exists():
        time.sleep(BARRIER_POLL_SECONDS)
    clear_barrier(output)
    emit_message("Background fitting slot released; continuing to fits", level=0)


def release_device_cache(device):
    import gc
    import torch

    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()
    gc.collect()
    if str(device).startswith("cuda") and torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_staged(tasks, workers, run_one, *, background_only=False):
    tasks = list(tasks)
    if not tasks:
        return
    cancel = threading.Event()
    concurrency = min(workers, len(tasks))
    semaphore = threading.Semaphore(concurrency)
    outputs = [task_output(task) for task in tasks]
    if len(set(outputs)) != len(outputs):
        raise ValueError("Background tasks must have distinct result directories")
    paths = [barrier_paths(output) for output in outputs]
    for output in outputs:
        clear_barrier(output)

    if background_only:
        pool = ThreadPoolExecutor(max_workers=concurrency)
        futures = []
        try:
            for task in tasks:
                options = deepcopy(task)
                options.background_phase = "prepare"
                options.background_concurrency = concurrency
                futures.append(pool.submit(run_one, options, cancel_event=cancel))
            for future in futures:
                future.result()
        except BaseException:
            cancel.set()
            for future in futures:
                future.cancel()
            raise
        finally:
            pool.shutdown(wait=True, cancel_futures=True)
            for output in outputs:
                clear_barrier(output)
        return

    def execute(task):
        options = deepcopy(task)
        options.background_phase = "staged"
        options.background_concurrency = concurrency
        while not semaphore.acquire(timeout=BARRIER_POLL_SECONDS):
            if cancel.is_set():
                return
        if cancel.is_set():
            semaphore.release()
            return
        ready_path, _ = barrier_paths(task_output(options))
        finished = threading.Event()
        released = threading.Event()

        def release_slot():
            while not finished.is_set():
                if ready_path.exists():
                    semaphore.release()
                    released.set()
                    return
                finished.wait(BARRIER_POLL_SECONDS)
            if not released.is_set():
                semaphore.release()

        watcher = threading.Thread(target=release_slot, daemon=True)
        watcher.start()
        try:
            run_one(options, cancel_event=cancel)
        finally:
            finished.set()
            watcher.join()

    pool = ThreadPoolExecutor(max_workers=len(tasks))
    futures = []
    try:
        futures = [pool.submit(execute, task) for task in tasks]
        pending = set(range(len(tasks)))
        skipped = set()
        while pending:
            for future in futures:
                if future.done():
                    future.result()
            for index in tuple(pending):
                future = futures[index]
                if future.done() and not paths[index][0].exists():
                    future.result()
                    skipped.add(index)
                    pending.remove(index)
                elif paths[index][0].is_file():
                    pending.remove(index)
            if pending:
                time.sleep(BARRIER_POLL_SECONDS)
        for index, future in enumerate(futures):
            if index in skipped:
                continue
            paths[index][1].parent.mkdir(parents=True, exist_ok=True)
            paths[index][1].touch()
            while not future.done():
                for later in range(index + 1, len(futures)):
                    if later not in skipped and futures[later].done():
                        futures[later].result()
                        raise RuntimeError("Background worker exited before its in-memory stage was released")
                time.sleep(BARRIER_POLL_SECONDS)
            future.result()
    except BaseException:
        cancel.set()
        for future in futures:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
        for output in outputs:
            clear_barrier(output)


def run_seed_backgrounds(args, run_one):
    from .worker_progress import emit_message

    for scenario in args.scenarios:
        for method in args.methods:
            tasks = []
            for seed in args.seeds:
                options = deepcopy(args)
                options.methods, options.scenarios, options.seeds = [method], [scenario], [seed]
                tasks.append(options)
            if method in NATIVE_METHODS:
                run_staged(tasks, args.scan_bg_workers, run_one,
                           background_only=getattr(args, "background_only", False))
            else:
                emit_message(f"{method}: background concurrency applies to RIDDLE, IAD and Supervised; retaining the external scheduler")
                for options in tasks:
                    options.scan_bg_workers = 1
                    run_one(options)
