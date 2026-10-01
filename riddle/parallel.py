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
from riddle.worker_progress import _LOCAL_SINK, emit_message, BufferedLog, durable_progress_event


def _fit_worker(events, job, rows, validation, options):
    from contextlib import redirect_stderr, redirect_stdout
    import torch
    from .storage import write_json
    from .campaign import train_member
    from riddle.acceleration import install_tensor_batches

    directory = Path(job["directory"])
    recovery = directory / ".resume"
    recovery.mkdir(parents=True, exist_ok=True)
    fit, started = job["fit"], time.monotonic()
    label = f"RIDDLE fit {fit}/{options['total']}"
    epoch_progress = {}
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    with (recovery / "worker.log").open("a") as raw_log:
        log = BufferedLog(raw_log)
        with redirect_stdout(log), redirect_stderr(log):
            def publish(event):
                if event.get("phase") == "residual_training" and event.get("completed") is not None:
                    epoch_progress[event["label"]] = max(0, event["completed"] - event.get("initial", 0))
                log.write(json.dumps(event) + "\n")
                if durable_progress_event(event):
                    log.force_flush()
                events.put((fit, "progress", event))
            token = _LOCAL_SINK.set(publish)
            try:
                torch.set_num_threads(options["torch_threads"])
                torch.set_num_interop_threads(1)
                install_tensor_batches(options["device"])
                torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
                saved = train_member(rows, validation, directory.parents[1],
                    relative=job["relative"], index=job["index"], fraction=job["fraction"],
                    epochs=options["epochs"], seed=options["seed"], device=options["device"],
                    initialization=options["initialization"], settings=options["settings"],
                    label=label, normalization_tests=options["normalization_tests"],
                    member_split_indices=job.get("member_split_indices"), source_ids=options.get("source_ids"),
                    background_correction=options.get("background_correction"))
                result = dict(status=saved["status"], initial_epoch=0,
                              new_epochs=sum(epoch_progress.values()),
                              trained_epochs=saved.get("trained_epochs", options["epochs"]),
                              training_seconds=time.monotonic() - started, finalization_seconds=0,
                              worker_started=started, worker_finished=time.monotonic(), pid=os.getpid())
                write_json(recovery / "execution.json", result)
                events.put((fit, "result", result))
            except BaseException as error:
                traceback.print_exc()
                log.force_flush()
                events.put((fit, "error", f"{type(error).__name__}: {error}"))
            finally:
                log.force_flush()
                _LOCAL_SINK.reset(token)
                signal.signal(signal.SIGTERM, previous_handler)


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
    rows, jobs, *, validation, epochs, seed, device, initialization, workers, io_workers, total, started=None,
    settings, normalization_tests, source_ids=None, background_correction=None, torch_threads=2
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
        torch_threads=torch_threads,
        total=total,
        settings=settings,
        normalization_tests=normalization_tests,
        source_ids=source_ids,
        background_correction=background_correction,
    )
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        while waiting or active:
            while waiting and len(active) < workers:
                job = waiting.pop(0)
                process = context.Process(target=_fit_worker, args=(events, job, rows, validation, options))
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
                        f"RIDDLE fit {fit}/{total}: {event.get('trained_epochs', epochs)}/{epochs} epoch; fit_elapsed={_duration(time.monotonic() - state['started'])}",
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
                if "message" in event:
                    emit_message(event["message"], kind=event.get("kind", "INFO"), level=event.get("level", 1))
                state = active[fit]
                if event.get("unit") == "epoch":
                    if state.get("attempt_label") != event.get("label"):
                        state.update(epoch=0, initial=0, printed_epoch=-1,
                                     training_started=time.monotonic(), attempt_label=event.get("label"))
                    state["epoch"] = event.get("completed") if event.get("completed") is not None else state["epoch"]
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

def completed_classifier_fit(root, fit_index):
    from .storage import file_digest

    path = Path(root) / f"fit_{fit_index:03d}" / "fit.json"
    if not path.is_file():
        return None
    saved = json.loads(path.read_text())
    if saved.get("fit_index") != fit_index or saved.get("status") not in ("completed", "excluded"):
        raise ValueError("Invalid AD baseline fit receipt")
    if saved["status"] == "completed":
        attempt = path.parent / saved["attempt_directory"]
        for candidate in saved["selected_checkpoints"]:
            checkpoint = attempt / candidate["filename"]
            if not checkpoint.is_file() or file_digest(checkpoint) != candidate["sha256"]:
                raise ValueError("AD baseline fit checkpoint changed")
    return saved


def _classifier_fit_worker(events, job, options):
    from contextlib import redirect_stderr, redirect_stdout
    import torch
    from .training import train_classifier_fit

    root = Path(job["fit_root"])
    log_root = root / ".resume"
    log_root.mkdir(parents=True, exist_ok=True)
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    started = time.monotonic()
    with (log_root / "worker.log").open("a") as raw_log:
        log = BufferedLog(raw_log)
        with redirect_stdout(log), redirect_stderr(log):
            def publish(event):
                log.write(json.dumps(event) + "\n")
                if durable_progress_event(event):
                    log.force_flush()
                events.put((job["fit_index"], "progress", event))
            token = _LOCAL_SINK.set(publish)
            try:
                torch.set_num_threads(options["torch_threads"])
                torch.set_num_interop_threads(1)
                torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
                receipt = train_classifier_fit(
                    options["method"],
                    options["cache"],
                    root,
                    options["classifier"],
                    options["scientific_seed"],
                    options["run_index"],
                    job["fit_index"],
                    options["device"],
                    options["resume"],
                )
                events.put((job["fit_index"], "result", {**receipt, "worker_seconds": time.monotonic() - started, "pid": os.getpid()}))
            except BaseException as error:
                traceback.print_exc()
                log.force_flush()
                events.put((job["fit_index"], "error", f"{type(error).__name__}: {error}"))
            finally:
                log.force_flush()
                _LOCAL_SINK.reset(token)
                signal.signal(signal.SIGTERM, previous_handler)


def run_classifier_fits(method, cache, training_root, classifier, scientific_seed, run_index, device, workers, torch_threads, resume):
    training_root = Path(training_root)
    receipts = {}
    pending = []
    for fit_index in range(classifier["fits"]):
        saved = completed_classifier_fit(training_root, fit_index) if resume else None
        if saved is None:
            pending.append({"fit_index": fit_index, "fit_root": str(training_root / f"fit_{fit_index:03d}")})
        else:
            receipts[fit_index] = saved
            emit_message(f"{method} fit {fit_index + 1}/{classifier['fits']}: verified completed fit reused", kind="PASS")
    if not pending:
        return [receipts[index] for index in range(classifier["fits"])]
    context = multiprocessing.get_context("spawn")
    events = context.Queue()
    waiting = list(pending)
    active = {}
    workers = min(max(1, int(workers)), len(waiting))
    options = {
        "method": method,
        "cache": str(cache),
        "classifier": classifier,
        "scientific_seed": int(scientific_seed),
        "run_index": int(run_index),
        "device": str(device),
        "torch_threads": int(torch_threads),
        "resume": bool(resume),
    }
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    try:
        while waiting or active:
            while waiting and len(active) < workers:
                job = waiting.pop(0)
                process = context.Process(target=_classifier_fit_worker, args=(events, job, options))
                process.start()
                active[job["fit_index"]] = {"process": process, "started": time.monotonic(), "epoch": 0, "printed": -1, "metrics": {}}
                emit_message(f"{method} fit {job['fit_index'] + 1}/{classifier['fits']}: started; active fits={len(active)}/{workers}", kind="WORK")
            try:
                fit_index, kind, event = events.get(timeout=1)
            except queue.Empty:
                for fit_index, state in active.items():
                    if state["process"].exitcode is not None:
                        raise RuntimeError(f"{method} fit {fit_index + 1}/{classifier['fits']}: worker exited without result; resume to retry")
                fit_index, kind, event = None, None, None
            if kind == "error":
                raise RuntimeError(f"{method} fit {fit_index + 1}/{classifier['fits']} failed: {event}")
            if kind == "result":
                state = active.pop(fit_index)
                state["process"].join(timeout=10)
                receipts[fit_index] = event
                status = event["status"]
                emit_message(
                    f"{method} fit {fit_index + 1}/{classifier['fits']}: {status}; completed fits={len(receipts)}/{classifier['fits']}; active fits={len(active)}/{workers}",
                    kind="PASS" if status == "completed" else "WARNING",
                )
            elif kind == "progress" and fit_index in active:
                if "message" in event:
                    emit_message(event["message"], kind=event.get("kind", "INFO"), level=event.get("level", 1))
                if event.get("unit") == "epoch" and event.get("completed") is not None:
                    state = active[fit_index]
                    state["epoch"] = int(event["completed"])
                    state["metrics"] = event.get("metrics", {})
            now = time.monotonic()
            for fit_index, state in active.items():
                if state["epoch"] <= state["printed"]:
                    continue
                elapsed = now - state["started"]
                epoch = state["epoch"]
                eta = _duration(elapsed * max(0, classifier["epochs"] - epoch) / max(1, epoch))
                metrics = "; ".join(f"{name}={_short_value(value)}" for name, value in state["metrics"].items() if name in {"train_bce", "validation_bce"})
                emit_message(
                    f"{method} fit {fit_index + 1}/{classifier['fits']}: {epoch}/{classifier['epochs']} epoch; fit_elapsed={_duration(elapsed)}; fit_ETA={eta}; active fits={len(active)}/{workers}" + (f"; {metrics}" if metrics else ""),
                    kind="PROGRESS",
                )
                state["printed"] = epoch
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
    return [receipts[index] for index in range(classifier["fits"])]
