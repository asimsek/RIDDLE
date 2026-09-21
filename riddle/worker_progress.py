from __future__ import annotations
from collections import deque
import hashlib
import json
import os
import queue
import subprocess
import threading
import time
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from .progress import colored_status, training_progress, verbosity, _duration, _short_value
from .storage import IO_PERSIST_EVERY, WORKER_LOG_FLUSH_SECONDS

EVENT_PREFIX = "[RIDDLE_WORKER_PROGRESS] "
_LOCAL_SINK = ContextVar("riddle_progress_sink", default=None)
_NO_DIGEST = object()


class BufferedLog:
    """Keep small progress writes in userspace; persist periodically or at durable boundaries."""
    def __init__(self, stream, interval=WORKER_LOG_FLUSH_SECONDS):
        self.stream = stream
        self.interval = float(interval)
        self.last_flush = time.monotonic()

    def write(self, value):
        result = self.stream.write(value)
        self.flush_due()
        return result

    def flush(self):
        self.flush_due()

    def flush_due(self):
        now = time.monotonic()
        if now - self.last_flush >= self.interval:
            self.force_flush(now)

    def force_flush(self, now=None):
        self.stream.flush()
        self.last_flush = time.monotonic() if now is None else now


def durable_progress_event(event):
    if event.get("unit") != "epoch" or event.get("completed") is None:
        return False
    completed = int(event["completed"]); total = int(event.get("total", 0) or 0)
    return completed > 0 and (completed % IO_PERSIST_EVERY == 0 or completed == total)


def durable_progress_line(line):
    if not line.startswith(EVENT_PREFIX):
        return False
    try:
        return durable_progress_event(json.loads(line[len(EVENT_PREFIX):]))
    except (TypeError, ValueError, json.JSONDecodeError):
        return False


def emit_progress(
    phase, label, *, total=1, unit="step", completed=None, initial=None, report_every=None,
    stream=None, **metrics
):
    event = {
        "phase": phase,
        "label": label,
        "total": total,
        "unit": unit,
        "completed": completed,
        "metrics": metrics,
    }
    if initial is not None:
        event["initial"] = initial
    if report_every is not None:
        event["report_every"] = report_every
    if stream is not None:
        event["stream"] = stream
    _emit_event(event)


def emit_message(message, *, kind="INFO", level=1):
    _emit_event({"message": message, "kind": kind, "level": level})


def _emit_event(event):
    sink = _LOCAL_SINK.get()
    if sink is not None:
        sink(event)
    elif os.environ.get("RIDDLE_WORKER_PROGRESS") == "1":
        print(EVENT_PREFIX + json.dumps(event), flush=True)


class ProgressStage:
    def __init__(
        self, phase, label, total=1, unit="step", *, initial=0, enabled=True, report_every=None, **metrics
    ):
        self.phase, self.label, self.total, self.unit = (phase, label, int(total), unit)
        self.initial = self.completed = int(initial)
        self.enabled, self.report_every = (enabled, report_every)
        self.metrics, self.last_emit, self.sub_operation = ({}, 0.0, None)
        self.update(initial, force=True, **metrics)

    def update(self, completed=None, *, force=False, **metrics):
        if completed is not None:
            if not self.completed <= completed <= self.total:
                raise ValueError("Invalid progress counter")
            self.completed = int(completed)
        self.metrics.update(metrics)
        now = time.monotonic()
        if self.enabled and (force or self.completed == self.total or now - self.last_emit >= 2):
            emit_progress(
                self.phase,
                self.label,
                total=self.total,
                unit=self.unit,
                completed=self.completed,
                initial=self.initial,
                report_every=self.report_every,
                **self.metrics,
            )
            self.last_emit = now

    def substep(self, operation, completed, total, **metrics):
        changed = operation != self.sub_operation or completed == 0
        now = time.monotonic()
        if changed:
            self.sub_operation, self.sub_started = (operation, now)
        eta = (
            _duration((now - self.sub_started) * (total - completed) / completed) if completed else "unknown"
        )
        self.update(
            force=changed or completed == total,
            operation=operation,
            minibatch=f"{completed}/{total}",
            substage_eta=eta,
            **metrics,
        )

    def digest(self, path, *, expected=_NO_DIGEST, mismatch=None):
        path = Path(path)
        size, read, started = (path.stat().st_size, 0, time.monotonic())
        self.update(force=True, file=str(path), file_bytes=f"0/{size}", file_eta="unknown")
        digest = hashlib.sha256()
        with path.open("rb") as stream:
            while block := stream.read(4 * 1024 * 1024):
                digest.update(block)
                read += len(block)
                eta = (time.monotonic() - started) * max(0, size - read) / read
                self.update(file_bytes=f"{read}/{size}", file_eta=_duration(eta))
        value = digest.hexdigest()
        if expected is not _NO_DIGEST and value != expected:
            raise ValueError(mismatch or f"Checksum mismatch: {path}")
        self.update(self.completed + 1, force=True, file_bytes=f"{read}/{size}", file_eta="0:00")
        return value

    def __enter__(self):
        return self

    def __exit__(self, kind, value, traceback):
        if kind is None and self.completed < self.total:
            self.update(self.total, force=True)


@contextmanager
def local_progress(label, *, display=None):
    with ExitStack() as stack:
        display = WorkerDisplays(stack, label, startup=False) if display is None else display
        guard, stop = (threading.RLock(), threading.Event())

        def publish(event):
            with guard:
                display.event(event)

        def heartbeat():
            while not stop.wait(1):
                with guard:
                    display.tick()

        token = _LOCAL_SINK.set(publish)
        thread = threading.Thread(target=heartbeat, name="worker-local-progress", daemon=True)
        thread.start()
        try:
            yield
        finally:
            stop.set()
            thread.join(timeout=2)
            _LOCAL_SINK.reset(token)


def progress_items(items, label, *, unit="file"):
    items = list(items)
    with training_progress(label, len(items), unit=unit) as progress:
        for item in items:
            detail = item[0] if isinstance(item, tuple) else getattr(item, "name", item)
            colored_status(f"{label}: {detail}", kind="WORK", level=2)
            yield item
            progress.update(item=detail)


class WorkerDisplays:
    """Keep independent phase counters and clocks for concurrent subprocesses."""

    def __init__(self, stack, label, *, startup=True):
        self.stack, self.label = stack, label
        self.displays = {None: WorkerDisplay(stack.enter_context(ExitStack()), label, startup=startup)}

    def event(self, event):
        stream = event.get("stream")
        if stream not in self.displays:
            self.displays[stream] = WorkerDisplay(
                self.stack.enter_context(ExitStack()), f"{self.label} | {stream}", startup=False
            )
        self.displays[stream].event(event)

    def line(self, line):
        if line.startswith(EVENT_PREFIX):
            self.event(json.loads(line[len(EVENT_PREFIX):]))
        else:
            self.displays[None].line(line)

    def tick(self):
        for display in self.displays.values():
            display.tick()


class WorkerDisplay:
    def __init__(self, stack, label, *, heartbeat_seconds=30, startup=True):
        self.stack, self.label = (stack, label)
        self.phase = None
        self.activity = None
        self.completed = 0
        self.metrics = {}
        self._file_bytes = None
        self.finished = False
        self.heartbeat_seconds = heartbeat_seconds
        self.started = self.last_heartbeat = time.monotonic()
        self.last_advance = self.started
        self.last_refresh = self.started
        if startup:
            self.event({"phase": "startup", "label": "Start Python and load dependencies"})

    def event(self, event):
        if "message" in event:
            colored_status(
                event["message"],
                kind=event.get("kind", "INFO"),
                label=self.label,
                level=event.get("level", 1),
            )
            return
        phase = event["phase"]
        raw_metrics = event.get("metrics", {})
        metrics = {
            key: value
            for key, value in raw_metrics.items()
            if key not in {"file", "filename", "path", "file_bytes"}
        }
        new_phase = phase != self.phase or (
            self.finished and event.get("completed") == event.get("initial", 0)
        )
        detail_changed = any(
            (key in metrics and self.metrics.get(key) != metrics[key] for key in ("operation", "array"))
        )
        resumed = (
            not new_phase and self.activity is not None
            and self.completed == self.activity.initial
            and event.get("initial", 0) > self.completed
            and event.get("completed") == event.get("initial")
        )
        if new_phase:
            if self.phase == "startup" and (not self.finished):
                self.activity.update(self.activity.total - self.completed)
                self.completed = self.activity.total
                self.finish()
            self.stack.close()
            self.phase, self.completed, self.metrics = (phase, 0, {})
            self._file_bytes = None
            self.finished = False
            self.phase_label = event["label"]
            self.started = self.last_heartbeat = time.monotonic()
            self.last_advance = self.started
            initial = int(event.get("initial", 0))
            self.activity = self.stack.enter_context(
                training_progress(
                    f"{self.label} | {self.phase_label}",
                    int(event.get("total", 1)),
                    unit=event.get("unit", "step"),
                    initial=initial,
                )
            )
            self.completed = initial
            if self.activity.bar is not None:
                self.activity.bar.set_description_str(f"  {self.phase_label}", refresh=False)
        elif resumed:
            initial = int(event["initial"])
            if initial > self.activity.total:
                raise ValueError("Invalid subprocess resume count")
            self.completed = self.activity.initial = self.activity._completed = initial
            self.phase_label = event["label"]
            self.activity.label = f"{self.label} | {self.phase_label}"
            self.started = self.last_advance = self.last_heartbeat = time.monotonic()
            self.activity._started = self.activity._last_update = self.started
            if self.activity.bar is not None:
                self.activity.bar.reset()
                self.activity.bar.set_description_str(f"  {self.phase_label}", refresh=False)
                self.activity.bar.update(initial)
        # Epoch completions are the durable log records; heartbeats must not
        # repeat their counters or recalculate an ETA mid-epoch.
        if self.activity.unit == "epoch":
            self.activity.report_every = 1
        elif "report_every" in event:
            self.activity.report_every = event["report_every"]
        if "file_bytes" in raw_metrics and self._file_bytes != raw_metrics["file_bytes"]:
            self._file_bytes = raw_metrics["file_bytes"]
            self.last_advance = time.monotonic()
        if "minibatch" in metrics and self.metrics.get("minibatch") != metrics["minibatch"]:
            self.last_advance = time.monotonic()
        self.metrics.update(metrics)
        previous = self.completed
        completed = event.get("completed")
        if completed is not None:
            completed = int(completed)
            if not self.completed <= completed <= self.activity.total:
                raise ValueError("Invalid subprocess progress count")
            self.activity.update(completed - self.completed, **self.metrics)
            if completed > self.completed:
                self.last_advance = time.monotonic()
            self.completed = completed
            if completed == self.activity.total and (not self.finished):
                self.finish()
        if not self.finished and (
            previous == self.completed
            and (new_phase or resumed or (detail_changed and self.activity.unit != "epoch"))
        ):
            colored_status(self.snapshot(), kind="WORK", label=self.label, level=1)

    def finish(self):
        self.finished = True
        elapsed = _duration(time.monotonic() - self.started)
        if self.activity.unit != "epoch" or self.completed == self.activity.initial:
            colored_status(
                f"{self.phase_label}: {self.completed}/{self.activity.total} {self.activity.unit}; elapsed={elapsed}",
                kind="WORK",
                label=self.label,
                level=1,
            )
        self.stack.close()

    def snapshot(self):
        elapsed = time.monotonic() - self.started
        processed = self.completed - self.activity.initial
        remaining = self.activity.total - self.completed
        eta = _duration(elapsed * remaining / processed) if processed else "unknown"
        details = "; ".join((f"{key}={_short_value(value)}" for key, value in self.metrics.items()))
        return (
            f"{self.phase_label}: {self.completed}/{self.activity.total} {self.activity.unit}; elapsed={_duration(elapsed)}; ETA={eta}; last_advance={_duration(time.monotonic() - self.last_advance)} ago"
            + (f"; {details}" if details else "")
        )

    def line(self, line):
        text = line.rstrip("\r\n")
        if text.startswith(EVENT_PREFIX):
            self.event(json.loads(text[len(EVENT_PREFIX) :]))
            return
        warning = "[WARNING] resume: "
        if text.startswith(warning):
            colored_status(text[len(warning) :], kind="WARNING", label=self.label, level=0)
            return
        for prefix, key in (
            ("train_loss =", "train_loss"),
            ("val_loss =", "validation_loss"),
            ("training loss:", "train_loss"),
            ("validation loss:", "validation_loss"),
        ):
            if text.strip().startswith(prefix):
                self.metrics[key] = text.strip()[len(prefix) :].strip()
        if text:
            colored_status(text, label=self.label, level=2)

    def tick(self):
        if self.activity is None or self.finished:
            return
        now = time.monotonic()
        if now - self.last_refresh >= 1:
            self.activity.refresh(**self.metrics)
            self.last_refresh = now
        if self.activity.unit == "epoch":
            return
        if now - self.last_heartbeat >= self.heartbeat_seconds:
            if self.activity.bar is None or verbosity() >= 2:
                colored_status(self.snapshot(), kind="WORK", label=self.label, level=1)
            self.last_heartbeat = now


def monitor_worker(command, env, log, label, *, resume=False):
    messages = queue.Queue()
    recent_output = deque(maxlen=30)
    with (
        log.open("a" if resume else "w") as raw_stream,
        subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            errors="replace",
        ) as process,
    ):
        stream = BufferedLog(raw_stream)

        def read_output():
            try:
                for line in process.stdout:
                    messages.put(line)
            except Exception as exc:
                messages.put(exc)
            finally:
                messages.put(None)

        reader = threading.Thread(target=read_output, name="worker-worker-output", daemon=True)
        reader.start()
        try:
            with ExitStack() as phases:
                display = WorkerDisplays(phases, label)
                while True:
                    try:
                        line = messages.get(timeout=1)
                    except queue.Empty:
                        stream.flush_due()
                        display.tick()
                        continue
                    if line is None:
                        break
                    if isinstance(line, Exception):
                        raise line
                    stream.write(line)
                    if durable_progress_line(line):
                        stream.force_flush()
                    if line.strip() and not line.startswith(EVENT_PREFIX):
                        recent_output.append(line.rstrip())
                    display.line(line)
                    display.tick()
                stream.force_flush()
                if process.wait():
                    details = "\n".join(recent_output)[-8000:]
                    raise RuntimeError(
                        f"Worker failed (exit {process.returncode}); inspect {log}"
                        + (f"\nRecent worker output:\n{details}" if details else "")
                    )
        except BaseException:
            stream.force_flush()
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            raise
        finally:
            stream.force_flush()
            reader.join(timeout=2)
