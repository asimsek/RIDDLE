from contextlib import contextmanager
from concurrent.futures import ThreadPoolExecutor
import fcntl
import hashlib
import json
import os
from pathlib import Path
import platform
import random
import tempfile

IO_PERSIST_EVERY = 10
WORKER_LOG_FLUSH_SECONDS = 30.0

_DIGEST_CACHE = {}

import numpy as np


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
    _DIGEST_CACHE[str(path.resolve())] = (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns, value)
    return value


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


def save_npz(path, **arrays):
    with Path(path).open("wb") as stream:
        np.savez(stream, **arrays)


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


def file_digest(path):
    path = Path(path)
    stat = path.stat()
    cached = _DIGEST_CACHE.get(str(path.resolve()))
    if cached is not None and cached[:4] == (stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns):
        return cached[4]
    with path.open("rb") as stream:
        value = hashlib.file_digest(stream, "sha256").hexdigest()
    return _cache_digest(path, value)


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


def verify_artifacts(root, files, label="Verify saved artifacts"):
    from .worker_progress import ProgressStage

    root = Path(root).resolve()
    with ProgressStage("verify_" + label, label, len(files), "file") as progress:
        for name, checksum in files.items():
            path = (root / name).resolve()
            if not path.is_relative_to(root) or not path.is_file():
                raise ValueError("Missing or invalid artifact")
            progress.digest(path, expected=checksum)


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


def code_hashes(root):
    root = Path(root)
    return {
        str(p.relative_to(root)): file_digest(p)
        for p in sorted(root.rglob("*"))
        if p.is_file() and p.suffix in (".py", ".yml", ".yaml", ".json")
    }
