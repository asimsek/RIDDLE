from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import os
from pathlib import Path
import threading
import time

from .storage import write_json

NATIVE_METHODS = ("riddle", "iad", "supervised")
BARRIER_POLL_SECONDS = 0.5


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


def clear_stage_checkpoint(output):
    root = Path(output) / ".resume"
    for path in (root / "background_stage.json", root / "background_stage.pt"):
        path.unlink(missing_ok=True)


def clear_barrier(output):
    for path in barrier_paths(output):
        path.unlink(missing_ok=True)


def wait_for_stage_release(output):
    ready, release = barrier_paths(output)
    ready.parent.mkdir(parents=True, exist_ok=True)
    write_json(ready, {"pid": os.getpid(), "ready": True})
    while not release.exists():
        time.sleep(BARRIER_POLL_SECONDS)
    clear_barrier(output)


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
    paths = [barrier_paths(Path(task.output).resolve()) for task in tasks]
    for task in tasks:
        clear_barrier(Path(task.output).resolve())

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
            for task in tasks:
                clear_barrier(Path(task.output).resolve())
        return

    def execute(task):
        options = deepcopy(task)
        options.background_phase = "staged"
        options.background_concurrency = concurrency
        semaphore.acquire()
        ready_path, _ = barrier_paths(Path(options.output).resolve())
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
        for task in tasks:
            clear_barrier(Path(task.output).resolve())


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
