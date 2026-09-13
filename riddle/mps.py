from __future__ import annotations
from dataclasses import asdict, dataclass
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time


@dataclass(frozen=True)
class MPSStatus:
    requested: bool
    active: bool
    started: bool
    pipe_directory: str | None
    log_directory: str | None
    reason: str

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _runtime_directories() -> tuple[Path, Path]:
    user_id = os.getuid() if hasattr(os, "getuid") else os.getpid()
    root = Path(tempfile.gettempdir()) / f"riddle-mps-{user_id}"
    return (root / "pipe", root / "log")


def _control_socket(pipe_directory: Path) -> Path:
    return pipe_directory / "control"


def _active_pipe_directory() -> Path | None:
    configured = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
    candidates = [Path(configured)] if configured else []
    managed, _ = _runtime_directories()
    candidates.extend((managed, Path("/tmp/nvidia-mps")))
    for candidate in candidates:
        if _control_socket(candidate).exists():
            return candidate
    return None


def configure_mps(mode: str, device: str, workers: int) -> MPSStatus:
    if mode not in {"auto", "on", "off"}:
        raise ValueError("MPS mode must be auto, on, or off")
    requested = mode != "off" and workers > 1 and (not str(device).lower().startswith("cpu"))
    if not requested:
        return MPSStatus(False, False, False, None, None, "disabled_or_single_worker")
    if os.name != "posix" or not sys_platform_linux():
        reason = "mps_requires_linux"
        if mode == "on":
            raise RuntimeError(reason)
        return MPSStatus(True, False, False, None, None, reason)
    active = _active_pipe_directory()
    if active is not None:
        if str(active) != "/tmp/nvidia-mps":
            os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(active)
        return MPSStatus(
            True,
            True,
            False,
            str(active),
            os.environ.get("CUDA_MPS_LOG_DIRECTORY"),
            "attached_to_existing_daemon",
        )
    control = shutil.which("nvidia-cuda-mps-control")
    if control is None:
        reason = "nvidia_cuda_mps_control_not_found"
        if mode == "on":
            raise RuntimeError(reason)
        return MPSStatus(True, False, False, None, None, reason)
    pipe_directory, log_directory = _runtime_directories()
    pipe_directory.mkdir(parents=True, exist_ok=True, mode=448)
    log_directory.mkdir(parents=True, exist_ok=True, mode=448)
    os.chmod(pipe_directory, 448)
    os.chmod(log_directory, 448)
    environment = os.environ.copy()
    environment["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe_directory)
    environment["CUDA_MPS_LOG_DIRECTORY"] = str(log_directory)
    try:
        completed = subprocess.run(
            [control, "-d"], check=False, capture_output=True, text=True, timeout=10, env=environment
        )
    except (OSError, subprocess.SubprocessError) as exc:
        reason = f"mps_start_failed:{type(exc).__name__}"
        if mode == "on":
            raise RuntimeError(reason) from exc
        return MPSStatus(True, False, False, str(pipe_directory), str(log_directory), reason)
    if completed.returncode != 0:
        detail = (completed.stderr or completed.stdout).strip().splitlines()
        reason = f"mps_start_failed:{(detail[0] if detail else completed.returncode)}"
        if mode == "on":
            raise RuntimeError(reason)
        return MPSStatus(True, False, False, str(pipe_directory), str(log_directory), reason)
    deadline = time.monotonic() + 5.0
    while time.monotonic() < deadline and (not _control_socket(pipe_directory).exists()):
        time.sleep(0.05)
    if not _control_socket(pipe_directory).exists():
        reason = "mps_control_socket_did_not_appear"
        if mode == "on":
            raise RuntimeError(reason)
        return MPSStatus(True, False, False, str(pipe_directory), str(log_directory), reason)
    os.environ["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe_directory)
    os.environ["CUDA_MPS_LOG_DIRECTORY"] = str(log_directory)
    return MPSStatus(True, True, True, str(pipe_directory), str(log_directory), "started_user_daemon")


def sys_platform_linux() -> bool:
    import sys

    return sys.platform.startswith("linux")


__all__ = ["MPSStatus", "configure_mps"]
