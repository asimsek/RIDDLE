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


def write_json(path, value):
    text = json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"
    atomic_write(path, lambda p: p.write_text(text))


def save_npz(path, **arrays):
    with Path(path).open("wb") as stream:
        np.savez(stream, **arrays)


def save_array(path, array):
    def write(p):
        with p.open("wb") as stream:
            np.save(stream, array, allow_pickle=False)

    atomic_write(path, write)


def file_digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


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
