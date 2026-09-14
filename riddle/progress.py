from __future__ import annotations
import os
import sys
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from typing import Any, Iterator

try:
    from tqdm.auto import tqdm as _tqdm
except ImportError:
    from tqdm import tqdm as _tqdm


def tqdm(*args: Any, **kwargs: Any) -> Any:
    try:
        return _tqdm(*args, **kwargs)
    except Exception as exc:
        if "colour" not in kwargs or "colour" not in str(exc).lower():
            raise
        fallback = dict(kwargs)
        fallback.pop("colour", None)
        return _tqdm(*args, **fallback)


_CURRENT_REPORTER: ContextVar["ProgressReporter | None"] = ContextVar(
    "riddle_progress_reporter", default=None
)
_PROGRESS_CONTEXT: ContextVar[str | None] = ContextVar("riddle_progress_context", default=None)
_DETACHED_WORKER: ContextVar[bool] = ContextVar("riddle_detached_progress_worker", default=False)
_VERBOSITY = int(os.environ.get("RIDDLE_VERBOSE", "-1"))
_STATUS_LOCK = threading.Lock()
_ANSI = {
    "INFO": "\x1b[1;96m",
    "START": "\x1b[1;94m",
    "WORK": "\x1b[1;95m",
    "PROGRESS": "\x1b[1;95m",
    "PASS": "\x1b[1;38;5;34m",
    "SKIP": "\x1b[1;93m",
    "WARNING": "\x1b[1;38;5;208m",
    "ERROR": "\x1b[1;91m",
}
_ANSI_RESET = "\x1b[0m"


def _supports_live_progress(stream: Any) -> bool:
    try:
        return bool(stream.isatty())
    except (AttributeError, OSError, ValueError):
        return False


def set_verbosity(value: int) -> None:
    global _VERBOSITY
    if value not in {0, 1, 2}:
        raise ValueError("--verbose must be 0, 1, or 2")
    _VERBOSITY = int(value)
    os.environ["RIDDLE_VERBOSE"] = str(value)


def set_progress_context(value: str | None) -> None:
    _PROGRESS_CONTEXT.set(None if value is None else str(value))


def verbosity() -> int:
    return _VERBOSITY


def _colored_line(message: str, *, kind: str, label: str | None) -> str:
    normalized = kind.upper()
    prefix = f"[{normalized}]"
    if label:
        prefix += f" [{label}]"
    color = "" if os.environ.get("NO_COLOR") is not None else _ANSI.get(normalized, _ANSI["INFO"])
    reset = "" if not color else _ANSI_RESET
    return f"{color}{prefix}{reset} {message}"


def colored_status(message: str, *, kind: str = "INFO", label: str | None = None, level: int = 0) -> None:
    if _VERBOSITY < level:
        return
    resolved_label = _PROGRESS_CONTEXT.get() if label is None else label
    line = _colored_line(str(message), kind=kind, label=resolved_label)
    reporter = _CURRENT_REPORTER.get()
    with _STATUS_LOCK:
        live_bar = None
        if reporter is not None:
            live_bar = reporter._activity if reporter._activity is not None else reporter._overall
        if live_bar is not None:
            live_bar.write(line, file=sys.stderr)
            return
        terminal_prefix = (
            "\r\x1b[2K" if _DETACHED_WORKER.get() and _supports_live_progress(sys.stderr) else ""
        )
        payload = (terminal_prefix + line + "\n").encode("utf-8", errors="replace")
        try:
            os.write(sys.stderr.fileno(), payload)
        except (AttributeError, OSError, ValueError):
            print(line, file=sys.stderr, flush=True)


def _duration(seconds: float) -> str:
    seconds = max(0, int(round(seconds)))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:d}:{minutes:02d}:{seconds:02d}" if hours else f"{minutes:d}:{seconds:02d}"


def _short_value(value: Any) -> str:
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


@dataclass
class ActivityProgress:
    bar: Any | None
    label: str
    total: int
    unit: str
    initial: int = 0
    report_every: int | None = None
    _completed: int = field(init=False)
    _started: float = field(default_factory=time.monotonic, init=False)
    _last_update: float = field(init=False)

    def __post_init__(self) -> None:
        self._completed = int(self.initial)
        self._last_update = self._started

    def announce_start(self, *, level: int | None = None) -> None:
        detail = f" from {self.initial}/{self.total}" if self.initial else f" ({self.total} {self.unit}s)"
        colored_status(
            f"Starting {self.label}{detail}",
            kind="START",
            level=(1 if self.bar is None else 2) if level is None else int(level),
        )

    def announce_finish(self, *, success: bool, level: int | None = None, show_count: bool = True) -> None:
        elapsed = time.monotonic() - self._started
        count = (
            f" at {self._completed}/{self.total} {self.unit}"
            if show_count and (self.unit != "epoch" or not success)
            else ""
        )
        colored_status(
            f"{('Finished' if success else 'Failed')} {self.label}{count}; elapsed={_duration(elapsed)}",
            kind="PASS" if success else "ERROR",
            level=(1 if self.bar is None else 2) if level is None else int(level),
        )

    def _should_emit(self, previous: int) -> bool:
        if _VERBOSITY >= 2:
            return True
        interval = self.report_every or max(1, self.total // (10 if _VERBOSITY == 1 else 4))
        return (
            self._completed == self.initial + 1
            or self._completed >= self.total
            or self._completed // interval > previous // interval
        )

    def update(self, amount: int = 1, **metrics: Any) -> None:
        previous = self._completed
        now = time.monotonic()
        if self.bar is not None:
            if metrics:
                self.bar.set_postfix(
                    {name: _short_value(value) for name, value in metrics.items()}, refresh=False
                )
            self.bar.update(amount)
        self._completed = min(self.total, self._completed + int(amount))
        if amount > 0 and (self.bar is None or _VERBOSITY >= 2) and self._should_emit(previous):
            processed = max(1, self._completed - self.initial)
            elapsed = now - self._started
            remaining = max(0, self.total - self._completed)
            eta = elapsed * remaining / processed
            detail = "; ".join((f"{name}={_short_value(value)}" for name, value in metrics.items()))
            suffix = f"; {detail}" if detail else ""
            colored_status(
                f"{self.label}: {self._completed}/{self.total} {self.unit}; elapsed={_duration(elapsed)}; last={_duration(now - self._last_update)}; ETA={_duration(eta)}{suffix}",
                kind="PROGRESS",
            )
        if amount > 0:
            self._last_update = now

    def note(self, message: str) -> None:
        if self.bar is not None:
            self.bar.write(message, file=sys.stderr)
        else:
            colored_status(message, kind="INFO", level=1)

    def refresh(self, **metrics: Any) -> None:
        if self.bar is None:
            return
        if metrics:
            self.bar.set_postfix(
                {name: _short_value(value) for name, value in metrics.items()}, refresh=False
            )
        self.bar.refresh()


class ProgressReporter:
    def __init__(
        self,
        total: int | None = None,
        *,
        enabled: bool = True,
        name: str = "Campaign",
        installer_style: bool = False,
        compact_threshold: int = 8,
    ) -> None:
        self.total = None if total is None else int(total)
        self.requested = bool(enabled)
        self.enabled = self.requested and _supports_live_progress(sys.stderr)
        self.name = str(name)
        self.status_label = self.name.lower().replace(" ", "_")
        self.installer_style = bool(installer_style)
        self.compact_threshold = max(2, int(compact_threshold))
        self._overall: Any | None = None
        self._activity: Any | None = None
        self._token: Any | None = None
        self._completed = 0
        self._stage_label = "campaign"
        self._aggregate_stage = False
        self._stage_started = time.monotonic()

    def __enter__(self) -> "ProgressReporter":
        self._token = _CURRENT_REPORTER.set(self)
        if self.total is not None:
            self.start_stage("campaign", self.total)
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self._close_activity()
        if self._overall is not None:
            self._overall.close()
        if self._token is not None:
            _CURRENT_REPORTER.reset(self._token)

    def start_stage(self, label: str, total: int, *, aggregate: bool | None = None) -> None:
        self._close_activity()
        if self._overall is not None:
            self._overall.close()
            self._overall = None
        self.total = int(total)
        self._completed = 0
        self._stage_label = str(label)
        self._stage_started = time.monotonic()
        count = f" ({self.total} items)" if self.installer_style and self.total > 1 else ""
        message = (
            f"Starting {label}{count}" if self.installer_style else f"Starting {label}: {self.total} task(s)"
        )
        colored_status(message, kind="START", label=self.status_label)
        use_aggregate = (
            not self.installer_style
            or bool(aggregate)
            or (aggregate is None and self.total >= self.compact_threshold)
        )
        self._aggregate_stage = use_aggregate
        if not self.enabled:
            return
        if not use_aggregate:
            return
        self._overall = tqdm(
            total=self.total,
            desc=f"  {label}",
            unit="item",
            dynamic_ncols=True,
            position=0,
            leave=True,
            file=sys.stderr,
            colour="cyan" if self.installer_style else None,
            bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} elapsed, ETA {remaining}, {rate_fmt}{postfix}]",
        )

    def start_task(self, label: str) -> None:
        if self.installer_style:
            if self._overall is not None:
                self._overall.set_postfix_str(f"working: {label}", refresh=True)
                if _VERBOSITY >= 2:
                    colored_status(label, kind="WORK", label=self.status_label)
                return
            if self._aggregate_stage and _VERBOSITY < 2:
                return
            colored_status(label, kind="WORK", label=self.status_label, level=1)
            return
        if self._overall is None:
            colored_status(f"Starting {label}", kind="START", label=self.status_label, level=2)
            return
        self._overall.set_description_str(f"{self.name} | {label}", refresh=True)
        if _VERBOSITY >= 2:
            colored_status(f"Starting {label}", kind="START", label=self.status_label)

    def advance(self, label: str, *, amount: int = 1) -> None:
        self._completed = min(int(self.total or 0), self._completed + int(amount))
        if self.installer_style and self._overall is None:
            if self._aggregate_stage:
                total = max(1, int(self.total or 0))
                interval = max(1, total // (10 if _VERBOSITY >= 1 else 4))
                previous = max(0, self._completed - int(amount))
                should_emit = self._completed >= total or self._completed // interval > previous // interval
                if not should_emit:
                    return
            kind = (
                "PROGRESS"
                if self._aggregate_stage and self._completed < int(self.total or 0)
                else "SKIP"
                if label.startswith("resumed")
                else "PASS"
            )
            elapsed = time.monotonic() - self._stage_started
            processed = max(1, self._completed)
            remaining = max(0, int(self.total or 0) - self._completed)
            eta = elapsed * remaining / processed
            position = f"[{self._completed}/{self.total}] " if int(self.total or 0) > 1 else ""
            eta_text = f"; ETA={_duration(eta)}" if remaining else ""
            colored_status(
                f"{position}{label}; elapsed={_duration(elapsed)}{eta_text}",
                kind=kind,
                label=self.status_label,
            )
            return
        if self._overall is None:
            kind = "SKIP" if label.startswith("resumed") else "PASS"
            elapsed = time.monotonic() - self._stage_started
            processed = max(1, self._completed)
            remaining = max(0, int(self.total or 0) - self._completed)
            eta = elapsed * remaining / processed
            colored_status(
                f"{label}; stage={self._completed}/{self.total}; elapsed={_duration(elapsed)}; ETA={_duration(eta)}",
                kind=kind,
                label=self.status_label,
            )
            return
        self._overall.set_postfix_str(label, refresh=False)
        self._overall.update(amount)
        if _VERBOSITY >= 2:
            kind = "SKIP" if label.startswith("resumed") else "PASS"
            colored_status(label, kind=kind, label=self.status_label)
        if self.installer_style and self._completed >= int(self.total or 0):
            elapsed = time.monotonic() - self._stage_started
            self._overall.close()
            self._overall = None
            colored_status(
                f"Finished {self._stage_label}: {self._completed}/{self.total} items; elapsed={_duration(elapsed)}",
                kind="PASS",
                label=self.status_label,
            )

    def note(self, message: str) -> None:
        if self._overall is not None:
            self._overall.write(message, file=sys.stderr)
        elif self._activity is not None:
            self._activity.write(message, file=sys.stderr)

    def detail(self, message: str) -> None:
        if self._overall is not None:
            self._overall.set_postfix_str(str(message), refresh=True)
        elif _VERBOSITY >= 2 or (self.requested and _VERBOSITY >= 1):
            colored_status(str(message), kind="PROGRESS", label=self.status_label)

    @contextmanager
    def activity(
        self, label: str, total: int, *, unit: str = "epoch", initial: int = 0
    ) -> Iterator[ActivityProgress]:
        self._close_activity()
        if not 0 <= int(initial) <= int(total):
            raise ValueError("Progress initial value must lie between zero and total")
        if self.installer_style and self.enabled:
            position = 1 if self._overall is not None else 0
            if int(total) == 1:
                self._activity = tqdm(
                    total=None,
                    desc=f"  {label}",
                    unit=unit,
                    dynamic_ncols=True,
                    position=position,
                    leave=False,
                    file=sys.stderr,
                    bar_format="{desc} | {elapsed} elapsed{postfix}",
                )
            else:
                self._activity = tqdm(
                    total=int(total),
                    initial=int(initial),
                    desc=f"  {label}",
                    unit=unit,
                    dynamic_ncols=True,
                    position=position,
                    leave=False,
                    file=sys.stderr,
                    colour="cyan",
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} elapsed, ETA {remaining}, {rate_fmt}{postfix}]",
                )
        elif self._overall is not None:
            self._activity = tqdm(
                total=int(total),
                initial=int(initial),
                desc=label,
                unit=unit,
                dynamic_ncols=True,
                position=1,
                leave=False,
                file=sys.stderr,
                bar_format="  {l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed} elapsed, ETA {remaining}, {rate_fmt}{postfix}]",
            )
        activity = ActivityProgress(self._activity, label, int(total), unit, int(initial))
        activity.announce_start(level=2 if self.installer_style else None)
        try:
            yield activity
        except BaseException:
            if self.installer_style:
                self._close_activity()
            activity.announce_finish(
                success=False, level=0 if self.installer_style else None, show_count=not self.installer_style
            )
            raise
        else:
            if self.installer_style:
                self._close_activity()
            activity.announce_finish(
                success=True, level=1 if self.installer_style else None, show_count=not self.installer_style
            )
        finally:
            self._close_activity()

    def _close_activity(self) -> None:
        if self._activity is not None:
            self._activity.close()
            self._activity = None


CampaignProgress = ProgressReporter


@contextmanager
def campaign_progress(
    total: int | None = None, *, enabled: bool = True, name: str = "Campaign", installer_style: bool = False
) -> Iterator[ProgressReporter]:
    with progress_session(total, enabled=enabled, name=name, installer_style=installer_style) as reporter:
        yield reporter


@contextmanager
def progress_session(
    total: int | None = None,
    *,
    enabled: bool = True,
    name: str = "Campaign",
    installer_style: bool = True,
    compact_threshold: int = 8,
) -> Iterator[ProgressReporter]:
    with ProgressReporter(
        total,
        enabled=enabled,
        name=name,
        installer_style=installer_style,
        compact_threshold=compact_threshold,
    ) as reporter:
        yield reporter


@contextmanager
def training_progress(
    label: str, total: int, *, unit: str = "epoch", initial: int = 0
) -> Iterator[ActivityProgress]:
    reporter = _CURRENT_REPORTER.get()
    if reporter is None:
        activity = ActivityProgress(None, label, int(total), unit, int(initial))
        activity.announce_start()
        try:
            yield activity
        except BaseException:
            activity.announce_finish(success=False)
            raise
        else:
            activity.announce_finish(success=True)
        return
    with reporter.activity(label, total, unit=unit, initial=initial) as activity:
        yield activity


@contextmanager
def operation_progress(
    label: str, *, unit: str = "step", refresh_seconds: float = 1.0
) -> Iterator[ActivityProgress]:
    with training_progress(label, 1, unit=unit) as activity:
        stop = threading.Event()
        thread: threading.Thread | None = None
        if activity.bar is not None:

            def heartbeat() -> None:
                frames = ("⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏")
                frame = 0
                while not stop.wait(max(0.2, float(refresh_seconds))):
                    activity.refresh(status=f"{frames[frame % len(frames)]} working")
                    frame += 1

            thread = threading.Thread(target=heartbeat, name="progress-heartbeat", daemon=True)
            thread.start()
        try:
            yield activity
        except BaseException:
            raise
        else:
            activity.update()
        finally:
            stop.set()
            if thread is not None:
                thread.join(timeout=max(0.2, float(refresh_seconds)) + 0.5)


def disable_inherited_progress() -> None:
    _CURRENT_REPORTER.set(None)
    _DETACHED_WORKER.set(True)


__all__ = [
    "ActivityProgress",
    "CampaignProgress",
    "ProgressReporter",
    "campaign_progress",
    "colored_status",
    "disable_inherited_progress",
    "operation_progress",
    "progress_session",
    "set_progress_context",
    "set_verbosity",
    "training_progress",
    "verbosity",
]
