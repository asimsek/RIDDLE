import heapq
import json
import multiprocessing
import os
from pathlib import Path
import queue
import signal
import time
import traceback
from riddle.progress import _duration, _short_value
from riddle.worker_progress import _LOCAL_SINK, emit_message


def _fit_worker(events, job, rows, options):
    from contextlib import redirect_stderr, redirect_stdout
    import numpy as np
    import torch
    from riddle.storage import save_npz, verify_artifacts
    from riddle.storage import atomic_write, locked, write_json
    from .campaign import member_split
    from .runtime import fingerprints
    from .training import train_residual
    from riddle.acceleration import install_tensor_batches

    directory = Path(job["directory"])
    directory.mkdir(parents=True, exist_ok=True)
    recovery = directory / ".resume"
    recovery.mkdir(exist_ok=True)
    fit = job["fit"]
    started = time.monotonic()
    label = f"RIDDLE fit {fit}/{options['total']}"
    with (recovery / "worker.log").open("a") as log, redirect_stdout(log), redirect_stderr(log):

        def publish(event):
            log.write(json.dumps(event) + "\n")
            log.flush()
            events.put((fit, "progress", event))

        token = _LOCAL_SINK.set(publish)
        try:
            torch.set_num_threads(options["io_workers"])
            install_tensor_batches()
            torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
            with locked(recovery / "fit.lock"):
                a, b, seed = member_split(
                    len(rows),
                    options["seed"],
                    job["index"],
                    batch_size=options["settings"]["training"]["batch_size"],
                )
                split_path = directory / "split.npz"
                if split_path.exists():
                    with np.load(split_path) as split:
                        if not np.array_equal(split["training"], a) or not np.array_equal(
                            split["validation"], b
                        ):
                            raise ValueError("Saved residual split changed")
                else:
                    atomic_write(split_path, lambda p: save_npz(p, training=a, validation=b))
                checkpoint_path = recovery / "latest.pt"
                checkpoint = (
                    torch.load(checkpoint_path, map_location=options["device"], weights_only=False)
                    if checkpoint_path.exists()
                    else None
                )
                initial = checkpoint["epoch"] + 1 if checkpoint is not None else 0
                if checkpoint is not None:
                    verify_artifacts(directory, checkpoint["files"], f"{label} | Verify epoch checkpoints")
                clock = {}
                try:
                    order, history = train_residual(
                        rows[a],
                        rows[b],
                        directory,
                        epochs=options["epochs"],
                        seed=seed,
                        device=options["device"],
                        checkpoint=checkpoint,
                        fraction=job["fraction"],
                        initialization=options["initialization"],
                        progress_label=f"{label} | Train residual mixture",
                        on_training_start=lambda: clock.update(start=time.monotonic()),
                        settings=options["settings"],
                    )
                    training_finished = time.monotonic()
                    saved = {
                        "status": "completed",
                        "epochs": order,
                        "signal_fractions": [history[i]["signal_fraction"] for i in order],
                    }
                except FloatingPointError as error:
                    training_finished = time.monotonic()
                    saved = {
                        "status": "numerical_failure",
                        "error": str(error),
                        "error_type": type(error).__name__,
                    }
                paths = [p for p in directory.iterdir() if p.is_file() and p.name != "fit.json"]
                saved.update(
                    directory=job["relative"],
                    seed=seed,
                    artifacts_sha256={
                        p.name: v for p, v in fingerprints(paths, options["io_workers"]).items()
                    },
                )
                write_json(directory / "fit.json", saved)
                result = {
                    "status": saved["status"],
                    "initial_epoch": initial,
                    "new_epochs": options["epochs"] - initial if saved["status"] == "completed" else 0,
                    "training_seconds": training_finished - clock.get("start", training_finished),
                    "finalization_seconds": time.monotonic() - training_finished,
                    "worker_started": started,
                    "worker_finished": time.monotonic(),
                    "pid": os.getpid(),
                }
                write_json(recovery / "execution.json", result)
                events.put((fit, "result", result))
        except BaseException as error:
            traceback.print_exc()
            events.put((fit, "error", f"{type(error).__name__}: {error}"))
        finally:
            _LOCAL_SINK.reset(token)


def fitting_eta(mean_fit, epochs, active, queued, workers):
    if not active and (not queued):
        return 0.0
    if mean_fit is None:
        return None
    loads = [mean_fit * max(0, epochs - state.get("epoch", 0)) / epochs for state in active]
    loads.extend([0.0] * max(0, workers - len(loads)))
    heapq.heapify(loads)
    for _ in range(queued):
        heapq.heappush(loads, heapq.heappop(loads) + mean_fit)
    return max(loads, default=0.0)


def run_fits(
    rows, jobs, *, epochs, seed, device, initialization, workers, io_workers, total, started=None, settings
):
    if not jobs:
        return
    context = multiprocessing.get_context("spawn")
    events = context.Queue()
    waiting, active = (list(jobs), {})
    workers = min(workers, len(waiting))
    started = time.monotonic() if started is None else started
    completed = total - len(waiting)
    seconds = measured_epochs = finalization = measured_fits = 0
    options = dict(
        epochs=epochs,
        seed=seed,
        device=str(device),
        initialization=initialization,
        io_workers=io_workers,
        total=total,
        settings=settings,
    )
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        while waiting or active:
            while waiting and len(active) < workers:
                job = waiting.pop(0)
                process = context.Process(target=_fit_worker, args=(events, job, rows, options))
                process.start()
                active[job["fit"]] = {
                    "process": process,
                    "started": time.monotonic(),
                    "epoch": 0,
                    "initial": 0,
                    "printed_epoch": -1,
                    "metrics": {},
                }
                emit_message(
                    f"RIDDLE fit {job['fit']}/{total}: started; active fits={len(active)}/{workers}",
                    kind="WORK",
                )
            try:
                fit, kind, event = events.get(timeout=1)
            except queue.Empty:
                for fit, state in active.items():
                    if state["process"].exitcode is not None:
                        raise RuntimeError(
                            f"RIDDLE fit {fit}/{total}: worker exited without a result; resume to retry"
                        )
                fit, kind, event = (None, None, None)
            if kind == "error":
                raise RuntimeError(f"RIDDLE fit {fit}/{total} failed: {event}")
            if kind == "result":
                state = active.pop(fit)
                state["process"].join(timeout=10)
                if state["process"].is_alive():
                    state["process"].terminate()
                    state["process"].join(timeout=5)
                if event["new_epochs"]:
                    seconds += event["training_seconds"]
                    measured_epochs += event["new_epochs"]
                    finalization += event["finalization_seconds"]
                    measured_fits += 1
                completed += 1
                if event["status"] == "completed" and state["printed_epoch"] < epochs:
                    emit_message(
                        f"RIDDLE fit {fit}/{total}: {epochs}/{epochs} epoch; fit_elapsed={_duration(time.monotonic() - state['started'])}",
                        kind="WORK",
                    )
                emit_message(
                    f"RIDDLE fit {fit}/{total}: {event['status']}; completed fits={completed}/{total}",
                    kind="PASS" if event["status"] == "completed" else "WARNING",
                )
                mean = (
                    epochs * seconds / measured_epochs + finalization / measured_fits
                    if measured_epochs
                    else None
                )
                eta = fitting_eta(mean, epochs, list(active.values()), len(waiting), workers)
                elapsed = time.monotonic() - started
                duration = lambda v: "unknown" if v is None else _duration(v)
                emit_message(
                    f"RIDDLE fitting estimate after completed fit {completed}/{total}: elapsed={_duration(elapsed)}; mean_fit={duration(mean)}; estimated_total={duration(elapsed + eta if eta is not None else None)}; fitting_ETA={duration(eta)}; active fits={len(active)}/{workers}"
                )
            elif kind == "progress" and fit in active:
                state = active[fit]
                if event.get("unit") == "epoch":
                    state["epoch"] = event.get("completed") or state["epoch"]
                    state["initial"] = event.get("initial", 0)
                    state["metrics"] = event.get("metrics", {})
                    if "training_started" not in state:
                        state["training_started"] = time.monotonic()
            now = time.monotonic()
            for fit, state in active.items():
                if state["epoch"] <= max(state["initial"], state["printed_epoch"]):
                    continue
                elapsed = now - state.get("training_started", state["started"])
                processed = state["epoch"] - state["initial"]
                eta = _duration(elapsed * (epochs - state["epoch"]) / processed) if processed else "unknown"
                metrics = "; ".join(
                    (
                        f"{k}={_short_value(v)}"
                        for k, v in state["metrics"].items()
                        if k in {"operation", "minibatch", "train_nll", "validation_nll", "fraction"}
                    )
                )
                emit_message(
                    f"RIDDLE fit {fit}/{total}: {state['epoch']}/{epochs} epoch; fit_elapsed={_duration(elapsed)}; fit_ETA={eta}"
                    + (f"; {metrics}" if metrics else ""),
                    kind="PROGRESS",
                )
                state["printed_epoch"] = state["epoch"]
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        for state in active.values():
            if state["process"].is_alive():
                state["process"].terminate()
        for state in active.values():
            state["process"].join(timeout=5)
            if state["process"].is_alive():
                state["process"].kill()
                state["process"].join(timeout=5)
        events.close()
        events.cancel_join_thread()
