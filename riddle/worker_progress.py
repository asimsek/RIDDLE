from __future__ import annotations
from collections import Counter, deque
import hashlib
import json
import os
import queue
import re
import signal
import subprocess
import threading
import time
from contextlib import ExitStack, contextmanager
from contextvars import ContextVar
from pathlib import Path
from .progress import colored_status, training_progress, verbosity, _duration, _short_value
from .storage import IO_PERSIST_EVERY, WORKER_LOG_FLUSH_SECONDS, timed_io

EVENT_PREFIX = "[RIDDLE_WORKER_PROGRESS] "
_LOCAL_SINK = ContextVar("riddle_progress_sink", default=None)
_OUTPUT_LOCK = threading.RLock()
_NO_DIGEST = object()


class ConsoleMessages:
    io_report_seconds = 30
    slow_io_seconds = 5
    fit_pattern = re.compile(r"(?:RIDDLE fit|Classifier fit) (\d+)/(\d+): (\d+)/(\d+) epoch;")
    runtime_seen = set()
    runtime_guard = threading.Lock()

    def __init__(self, label, *, status=None, get_verbosity=None):
        self.label = label
        self.status = status or colored_status
        self.verbosity = get_verbosity or verbosity
        self.active_io, self.fit_updates = {}, {}
        self.io_counts = Counter()
        self.io_seconds = 0.0
        self.io_started = self.io_start_utc = self.io_end_utc = None
        self.last_io = 0.0

    def event(self, event):
        text = event["message"]
        if self.verbosity() >= 2:
            self.status(text, kind=event.get("kind", "INFO"), label=self.label,
                        level=event.get("level", 1))
            return
        task = event.get("task")
        if task and self.label:
            text = text.replace(f"{task}; pid={event.get('pid')} | ", "", 1)
            if text.startswith("BG stages "):
                for field in task.split("; "):
                    if field.split("=", 1)[0] in {"method", "seed", "scenario", "signal_events"}:
                        text = text.replace("; " + field, "", 1)
        if text.startswith(("BG stages START", "Runtime; ")):
            fields = dict(part.split("=", 1) for part in text.split("; ") if "=" in part)
            runtime = tuple(fields.get(key, "unknown") for key in ("gpu", "node", "host"))
            with self.runtime_guard:
                announce = runtime not in self.runtime_seen
                self.runtime_seen.add(runtime)
            if announce:
                self.status(f"Runtime | GPU {runtime[0]} | node {runtime[1]}", label=self.label, level=1)
            if text.startswith("Runtime; "):
                return
        io_match = re.search(r"\bI/O (START|END); (.*)", text)
        if io_match:
            fields = dict(part.split("=", 1) for part in io_match[2].split("; ") if "=" in part)
            if all(key in fields for key in ("io_id", "operation", "path", "start_utc")):
                try:
                    self.io_event(io_match[1], fields, event)
                    return
                except (TypeError, ValueError):
                    pass
        match = self.fit_pattern.search(text) if event.get("kind") == "PROGRESS" else None
        if match:
            fit, _, epoch, total = map(int, match.groups())
            now = time.monotonic()
            previous, last = self.fit_updates.get(fit, (0, 0.0))
            if fit in self.fit_updates:
                if epoch == previous:
                    return
                if epoch > previous and epoch < total and epoch // 5 == previous // 5 and now - last < 60:
                    return
            self.fit_updates[fit] = epoch, now
            text = text.replace("RIDDLE fit", "Fit", 1)
            text = text.replace("Classifier fit", "Fit", 1)
            if "; operation=Epoch complete" in text:
                text = re.sub(r"; minibatch=0/\d+", "", text.replace("; operation=Epoch complete", ""))
        if event.get("kind") in {"WARNING", "ERROR"} or text.startswith("BG stages END"):
            self.flush()
        self.status(text, kind=event.get("kind", "INFO"), label=self.label,
                    level=event.get("level", 1))

    def io_event(self, state, fields, event):
        now = time.monotonic()
        elapsed = float(fields.get("elapsed_seconds", 0))
        if self.io_started is None:
            self.io_started = now if state == "START" else self.active_io.get(fields["io_id"], (now - elapsed,))[0]
            self.io_start_utc = fields["start_utc"]
            if state == "START":
                self.status(f"I/O START; start_utc={fields['start_utc']}; operation={fields['operation']}",
                            label=self.label, level=event.get("level", 0))
        self.last_io = now
        self.io_start_utc = min(self.io_start_utc, fields["start_utc"])
        if state == "START":
            self.active_io[fields["io_id"]] = (now, fields, now + self.slow_io_seconds)
        else:
            self.active_io.pop(fields["io_id"], None)
            self.io_counts[fields["operation"]] += 1
            self.io_seconds += elapsed
            end_utc = fields.get("end_utc", fields["start_utc"])
            self.io_end_utc = max(self.io_end_utc or end_utc, end_utc)
            failed = fields.get("status", "completed") != "completed" or event.get("kind") in {"WARNING", "ERROR"}
            if failed or elapsed >= self.slow_io_seconds:
                self.status(
                    f"I/O END; operation={fields['operation']}; path={fields['path']}; "
                    f"start_utc={fields['start_utc']}; end_utc={end_utc}; "
                    f"elapsed_seconds={elapsed:.3f}; status={fields.get('status', 'completed')}; "
                    f"io_id={fields['io_id']}"
                    + (f"; error_type={fields['error_type']}" if fields.get("error_type", "none") != "none" else ""),
                    kind=event.get("kind", "INFO"), label=self.label, level=event.get("level", 0),
                )
        self.tick()

    def tick(self):
        if self.verbosity() >= 2:
            return
        now = time.monotonic()
        for identity, (started, fields, next_report) in tuple(self.active_io.items()):
            if now >= next_report:
                self.status(
                    f"I/O RUNNING; operation={fields['operation']}; path={fields['path']}; "
                    f"start_utc={fields['start_utc']}; elapsed_seconds={now - started:.1f}; io_id={identity}",
                    kind="WORK", label=self.label, level=0,
                )
                self.active_io[identity] = started, fields, now + self.io_report_seconds
        if self.io_started is not None and self.io_counts and (
            now - self.io_started >= self.io_report_seconds or (not self.active_io and now - self.last_io >= 1)
        ):
            self.flush()

    def flush(self):
        if not self.io_counts:
            return
        operations = ", ".join(f"{operation}={count}" for operation, count in self.io_counts.items())
        self.status(
            f"I/O SUMMARY; start_utc={self.io_start_utc}; end_utc={self.io_end_utc}; "
            f"elapsed_seconds={self.last_io - self.io_started:.3f}; "
            f"operations={sum(self.io_counts.values())}; summed_io_seconds={self.io_seconds:.3f}; "
            f"pending={len(self.active_io)}; {operations}",
            label=self.label, level=0,
        )
        self.io_counts.clear()
        self.io_seconds = 0.0
        self.io_started = self.io_start_utc = self.io_end_utc = None


_CONSOLE_MESSAGES = ConsoleMessages("Framework")


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
    if event.get("durable") is True:
        return True
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


def emit_message(message, *, kind="INFO", level=1, durable=False, console=False):
    event = {"message": message, "kind": kind, "level": level}
    if durable:
        event["durable"] = True
    _emit_event(event, console=console)


def emit_runtime(gpu):
    host = os.uname().nodename
    node = os.environ.get("RIDDLE_NODE_NAME") or ("unavailable" if os.environ.get("KUBERNETES_SERVICE_HOST") else host)
    emit_message(f"Runtime; gpu={gpu}; node={node}; host={host}; job={os.environ.get('RIDDLE_JOB_NAME', 'local')}",
                 level=1, durable=True)


def _emit_event(event, *, console=False):
    task = event.get("task") or os.environ.get("RIDDLE_LOG_TASK")
    if task:
        event = {**event, "task": task, "pid": event.get("pid", os.getpid())}
        prefix = f"{task}; pid={event['pid']} | "
        if "message" in event and prefix not in event["message"]:
            event["message"] = prefix + event["message"]
    sink = _LOCAL_SINK.get()
    if sink is not None:
        sink(event)
    elif os.environ.get("RIDDLE_WORKER_PROGRESS") == "1":
        line = EVENT_PREFIX + json.dumps(event)
        with _OUTPUT_LOCK:
            print(line, flush=True)
    elif console:
        with _OUTPUT_LOCK:
            _CONSOLE_MESSAGES.event(event)


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
        with timed_io("Verify checksum", path):
            return self._digest(path, expected=expected, mismatch=mismatch)

    def _digest(self, path, *, expected=_NO_DIGEST, mismatch=None):
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
            if isinstance(display, WorkerDisplays):
                display.flush()
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
        with _OUTPUT_LOCK:
            _CONSOLE_MESSAGES.tick()

    def flush(self):
        for display in self.displays.values():
            display.messages.flush()
        with _OUTPUT_LOCK:
            _CONSOLE_MESSAGES.flush()


class WorkerDisplay:
    def __init__(self, stack, label, *, heartbeat_seconds=30, startup=True):
        self.stack, self.label = (stack, label)
        self.messages = ConsoleMessages(label)
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
            self.messages.event(event)
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
            self.messages.flush()
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
        # Heartbeats must not advance durable epoch counters.

        if self.activity.unit == "epoch":
            self.activity.report_every = 1 if verbosity() >= 2 else 5
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
        self.messages.flush()
        self.finished = True
        elapsed = _duration(time.monotonic() - self.started)
        if self.activity.unit != "epoch" or self.completed == self.activity.initial:
            colored_status(
                f"{self.phase_label}: {self.completed}/{self.activity.total} {self.activity.unit}; elapsed={elapsed}",
                kind="WORK",
                label=self.label,
                level=2,
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
        self.messages.tick()
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


def monitor_worker(command, env, log, label, *, resume=False, cancel_event=None):
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
            start_new_session=cancel_event is not None and os.name == "posix",
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
                phases.callback(display.flush)
                while True:
                    if cancel_event is not None and cancel_event.is_set():
                        raise RuntimeError("Background preparation cancelled after another worker failed")
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
                if cancel_event is not None and os.name == "posix":
                    os.killpg(process.pid, signal.SIGTERM)
                else:
                    process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if cancel_event is not None and os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            raise
        finally:
            stream.force_flush()
            reader.join(timeout=2)
