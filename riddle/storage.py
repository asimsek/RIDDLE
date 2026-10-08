from contextlib import contextmanager
from contextvars import ContextVar, copy_context
from concurrent.futures import ThreadPoolExecutor, as_completed, wait
from datetime import datetime, timezone
from functools import wraps
from itertools import count
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import shutil
import tempfile
import threading
import time

IO_PERSIST_EVERY = 10
WORKER_LOG_FLUSH_SECONDS = 30.0

_DIGEST_CACHE = {}
_DIGEST_LOCKS = {}
_DIGEST_GUARD = threading.Lock()
_VERIFY_POOLS = {}
_VERIFY_GUARD = threading.Lock()
_IO_DEPTH = ContextVar("riddle_io_depth", default=0)
_IO_SEQUENCE = count(1)

import numpy as np


@contextmanager
def timed_io(operation, path, *, durable=True):
    if _IO_DEPTH.get():
        yield
        return
    from .worker_progress import emit_message

    identity = f"operation={operation}; path={path}; io_id={os.getpid()}:{next(_IO_SEQUENCE)}"
    started_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
    started = time.monotonic()
    emit_message(f"I/O START; start_utc={started_utc}; {identity}", level=0, durable=durable, console=True)
    token = _IO_DEPTH.set(1)
    status, error_type = "completed", "none"
    try:
        yield
    except BaseException as error:
        status = "interrupted" if isinstance(error, (KeyboardInterrupt, SystemExit)) else "failed"
        error_type = type(error).__name__
        raise
    finally:
        _IO_DEPTH.reset(token)
        elapsed = time.monotonic() - started
        ended_utc = datetime.now(timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")
        emit_message(
            f"I/O END; start_utc={started_utc}; end_utc={ended_utc}; elapsed_seconds={elapsed:.3f}; "
            f"status={status}; error_type={error_type}; {identity}",
            kind="INFO" if status == "completed" else "WARNING", level=0,
            durable=durable or status != "completed", console=True,
        )


def _timed_file_operation(operation):
    def decorate(function):
        @wraps(function)
        def wrapped(*args, **kwargs):
            path = args[0] if args else kwargs[function.__code__.co_varnames[0]]
            with timed_io(operation, path):
                return function(*args, **kwargs)
        return wrapped
    return decorate


def load_array(path, *args, **kwargs):
    operation = "Open memory-mapped array" if kwargs.get("mmap_mode") else "Read NumPy array"
    with timed_io(operation, path):
        return np.load(path, *args, **kwargs)


@contextmanager
def open_npz(path, **kwargs):
    with timed_io("Read NumPy archive", path):
        with np.load(path, **kwargs) as archive:
            yield archive


@_timed_file_operation("Read checkpoint")
def load_checkpoint(path, *args, **kwargs):
    import torch

    return torch.load(path, *args, **kwargs)


@_timed_file_operation("Read JSON")
def read_json(path):
    return json.loads(Path(path).read_text())


def copy_file(source, destination, **kwargs):
    with timed_io("Copy file", f"{source} -> {destination}"):
        return shutil.copy2(source, destination, **kwargs)


@_timed_file_operation("Write artifact")
def atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        writer(temporary)
        with temporary.open("rb") as stream:
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _cache_digest(path, value):
    path = Path(path)
    stat = path.stat()
    key = str(path.resolve())
    with _DIGEST_GUARD:
        _DIGEST_CACHE[key] = (*_digest_identity(stat), value)
    return value


def _digest_identity(stat):
    return stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns


def configure_verification(workers=8):
    if type(workers) is not int or workers < 1:
        raise ValueError("Verification workers must be positive")
    os.environ["RIDDLE_VERIFY_WORKERS"] = str(workers)


def verification_workers():
    workers = int(os.environ.get("RIDDLE_VERIFY_WORKERS", "8"))
    if workers < 1:
        raise ValueError("Verification workers must be positive")
    return workers


def _verification_pool(workers):
    key = os.getpid(), workers
    with _VERIFY_GUARD:
        if key not in _VERIFY_POOLS:
            _VERIFY_POOLS[key] = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="verify")
        return _VERIFY_POOLS[key]


def persist_boundary(epoch, total=None, *, interval=IO_PERSIST_EVERY):
    completed = int(epoch) + 1
    return completed % int(interval) == 0 or (total is not None and completed == int(total))


class _HashingWriter:
    def __init__(self, stream):
        self.stream = stream
        self.digest = hashlib.sha256()

    def write(self, data):
        self.digest.update(data)
        return self.stream.write(data)

    def flush(self):
        return self.stream.flush()

    def tell(self):
        return self.stream.tell()

    def seek(self, *args):
        return self.stream.seek(*args)

    def fileno(self):
        return self.stream.fileno()

    def writable(self):
        return True


@_timed_file_operation("Write checkpoint")
def atomic_torch_save(path, value):
    """Atomically torch.save and compute SHA256 during the same write pass."""
    import torch

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        with temporary.open("wb") as raw:
            stream = _HashingWriter(raw)
            torch.save(value, stream)
            stream.flush()
            os.fsync(raw.fileno())
            value_digest = stream.digest.hexdigest()
        os.replace(temporary, path)
        return _cache_digest(path, value_digest)
    finally:
        temporary.unlink(missing_ok=True)


def write_json(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    payload = text.encode("utf-8")
    atomic_write(path, lambda p: p.write_bytes(payload))
    _cache_digest(path, hashlib.sha256(payload).hexdigest())


@_timed_file_operation("Write NumPy archive")
def save_npz(path, **arrays):
    with Path(path).open("wb") as stream:
        np.savez(stream, **arrays)


@_timed_file_operation("Write NumPy array")
def save_array(path, array):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix="." + path.name + "-", dir=path.parent)
    os.close(fd)
    temporary = Path(name)
    try:
        with temporary.open("wb") as raw:
            stream = _HashingWriter(raw)
            np.save(stream, array, allow_pickle=False)
            stream.flush()
            os.fsync(raw.fileno())
            value_digest = stream.digest.hexdigest()
        os.replace(temporary, path)
        _cache_digest(path, value_digest)
    finally:
        temporary.unlink(missing_ok=True)


def file_digest(path, *, durable=True):
    path = Path(path).resolve()
    key = str(path)
    with _DIGEST_GUARD:
        guard = _DIGEST_LOCKS.setdefault(key, threading.Lock())
    with guard:
        before = _digest_identity(path.stat())
        with _DIGEST_GUARD:
            cached = _DIGEST_CACHE.get(key)
        if cached is not None and cached[:-1] == before:
            return cached[-1]
        with timed_io("Checksum file", path, durable=durable):
            with path.open("rb") as stream:
                value = hashlib.file_digest(stream, "sha256").hexdigest()
                if _digest_identity(os.fstat(stream.fileno())) != before or _digest_identity(path.stat()) != before:
                    raise ValueError(f"File changed during checksum: {path}")
            with _DIGEST_GUARD:
                _DIGEST_CACHE[key] = (*before, value)
        return value


def fingerprint_files(root, paths, workers=1):
    from .worker_progress import ProgressStage

    paths = list(paths)
    with ProgressStage("output_artifacts", "Fingerprint output artifacts", len(paths), "file") as progress:
        with ThreadPoolExecutor(max_workers=max(1, min(workers, len(paths)))) as pool:
            result = {}
            for index, (path, value) in enumerate(zip(paths, pool.map(file_digest, paths)), 1):
                result[str(path.relative_to(root))] = value
                progress.update(index)
    return result


def digest(value):
    array = np.ascontiguousarray(value)
    if array.dtype.hasobject:
        raise TypeError("Object arrays cannot be fingerprinted")
    h = hashlib.sha256()
    h.update(str(array.dtype).encode())
    h.update(str(array.shape).encode())
    if array.size:
        h.update(memoryview(array).cast("B"))
    return h.hexdigest()


@contextmanager
def locked(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError(
                "Another process is using this output; use a distinct method/seed destination"
            ) from None
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def verify_artifacts(root, files, label="Verify saved artifacts", *, workers=None, progress_class=None):
    from .worker_progress import ProgressStage

    root = Path(root).resolve()
    workers = verification_workers() if workers is None else workers
    progress_class = ProgressStage if progress_class is None else progress_class
    if type(workers) is not int or workers < 1:
        raise ValueError("Verification workers must be positive")

    def verify_one(name, checksum):
        started = time.monotonic()
        candidate = root / name
        with timed_io("Verify checksum", candidate, durable=False):
            path = candidate.resolve()
            if Path(name).is_absolute() or ".." in Path(name).parts or candidate.is_symlink() or not path.is_relative_to(root) or not path.is_file():
                raise ValueError(f"Missing or invalid artifact: {name}")
            value = file_digest(path, durable=False)
            if value != checksum:
                raise ValueError(f"Checksum mismatch: {path}")
        return name, time.monotonic() - started

    futures = []
    with progress_class("verify_" + label, label, len(files), "file", verify_workers=workers) as progress:
        pool = _verification_pool(workers)
        try:
            for name, checksum in files.items():
                futures.append(pool.submit(copy_context().run, verify_one, name, checksum))
            for completed, future in enumerate(as_completed(futures), 1):
                name, elapsed = future.result()
                progress.update(completed, force=completed == 1, file=name, file_elapsed_seconds=elapsed)
        finally:
            for future in futures:
                future.cancel()
            wait(futures)


def rng_state():
    import torch

    return {
        "numpy": np.random.get_state(),
        "python": random.getstate(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    import torch

    np.random.set_state(state["numpy"])
    random.setstate(state["python"])
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state(state["cuda"][0].cpu(), device=0)


def seed_start(seed):
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def environment():
    from importlib.metadata import version
    import torch

    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {p: version(p) for p in ("numpy", "torch", "scipy", "scikit-learn", "nflows", "PyYAML")},
        "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "threads": torch.get_num_threads(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
    }


@_timed_file_operation("Fingerprint framework")
def code_hashes(root):
    root = Path(root)
    return {
        str(p.relative_to(root)): file_digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in (".py", ".yml", ".yaml", ".json")
    }
