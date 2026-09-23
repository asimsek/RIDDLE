import argparse
import importlib
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path


def gpu_uuid():
    """Resolve the first visible CUDA device in a disposable process."""
    import ctypes as ct
    import uuid

    driver = ct.CDLL("libcuda.so.1")
    driver.cuInit.argtypes = [ct.c_uint]
    driver.cuDeviceGet.argtypes = [ct.POINTER(ct.c_int), ct.c_int]

    def check(code):
        if code:
            raise RuntimeError(f"CUDA device identification failed ({code})")

    check(driver.cuInit(0))
    device = ct.c_int()
    check(driver.cuDeviceGet(ct.byref(device), 0))
    values = []
    for function in (driver.cuDeviceGetUuid, getattr(driver, "cuDeviceGetUuid_v2", driver.cuDeviceGetUuid)):
        function.argtypes = [ct.c_void_p, ct.c_int]
        value = (ct.c_ubyte * 16)()
        check(function(ct.byref(value), device.value))
        values.append(bytes(value))
    return ("MIG-" if values[0] != values[1] else "GPU-") + str(uuid.UUID(bytes=values[1]))


def concurrent_environment(mode, device):
    """Optional MPS for stage subprocesses; no model settings or parent CUDA remapping."""
    if mode not in ("auto", "on", "off"):
        raise ValueError("MPS must be auto, on, or off")
    if mode == "off" or device == "cpu":
        return {}, {"active": False, "reason": "disabled_or_cpu"}
    try:
        if not sys.platform.startswith("linux"):
            raise RuntimeError("MPS requires Linux")
        visible = subprocess.check_output(
            [sys.executable, "-m", "external.ranode_utils.runtime", "--gpu-uuid"],
            text=True, stderr=subprocess.PIPE, timeout=15,
        ).strip()
        import re
        if not re.fullmatch(r"(?:GPU|MIG)-[0-9a-fA-F-]{36}", visible):
            raise RuntimeError("Cannot verify the allocated GPU identity for MPS")

        env = {"CUDA_VISIBLE_DEVICES": visible}
        root = Path(tempfile.gettempdir()) / f"ranode-mps-{os.getuid()}-{visible}"
        configured = os.environ.get("CUDA_MPS_PIPE_DIRECTORY")
        candidates = [Path(configured)] if configured else [root / "pipe", Path("/tmp/nvidia-mps")]
        for pipe in candidates:
            if (pipe / "control").exists():
                env["CUDA_MPS_PIPE_DIRECTORY"] = str(pipe)
                return env, {"active": True, "reason": "attached_to_existing_daemon", "pipe": str(pipe), "gpu": visible}
        control = shutil.which("nvidia-cuda-mps-control")
        if control is None:
            raise RuntimeError("nvidia-cuda-mps-control is unavailable")
        pipe = candidates[0]
        log = Path(os.environ.get("CUDA_MPS_LOG_DIRECTORY", str(root / "log")))
        for path in (pipe, log):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        env.update(CUDA_MPS_PIPE_DIRECTORY=str(pipe), CUDA_MPS_LOG_DIRECTORY=str(log))
        subprocess.run([control, "-d"], env={**os.environ, **env}, check=True,
                       capture_output=True, text=True, timeout=15)
        deadline = time.monotonic() + 5
        while not (pipe / "control").exists() and time.monotonic() < deadline:
            time.sleep(.05)
        if not (pipe / "control").exists():
            raise RuntimeError("MPS control socket did not appear")
        return env, {"active": True, "reason": "started_user_daemon", "pipe": str(pipe), "gpu": visible}
    except (OSError, RuntimeError, subprocess.SubprocessError) as error:
        if mode == "on":
            raise RuntimeError(f"Required MPS is unavailable: {error}") from error
        return {}, {"active": False, "reason": f"ordinary_concurrency: {error}"}


def normalized(text):
    return sorted(
        x.strip()
        for x in text.splitlines()
        if x.strip() and not x.lstrip().startswith("#")
    )


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Verify the shared runtime for the independent R-ANODE pipeline"
    )
    parser.add_argument("--require-cuda", action="store_true")
    parser.add_argument("--gpu-uuid", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.gpu_uuid:
        print(gpu_uuid())
        return
    image = Path("/opt/riddle-image")
    root = Path(__file__).resolve().parents[2]
    try:
        if normalized((image / "requirements.txt").read_text()) != normalized(
            (root / "requirements.txt").read_text()
        ):
            raise RuntimeError(
                "Rebuild and pin riddle-runtime after changing its dependencies"
            )
        packages = subprocess.check_output(
            [
                sys.executable,
                "-m",
                "pip",
                "--disable-pip-version-check",
                "--no-cache-dir",
                "freeze",
            ],
            text=True,
        )
        if normalized(packages) != normalized((image / "packages.lock").read_text()):
            raise RuntimeError(
                "R-ANODE container packages changed; recreate it from the pinned image"
            )
        for name in (
            "numpy",
            "scipy",
            "sklearn",
            "yaml",
            "matplotlib",
            "torch",
            "nflows",
            "wandb",
        ):
            importlib.import_module(name)
        import torch

        if args.require_cuda:
            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable for R-ANODE")
            torch.ones(1, device="cuda:0").sum().item()
        print("[PASS] Independent R-ANODE container runtime verified", flush=True)
    except (ImportError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"[ERROR] {error}; use the updated shared riddle-runtime image\n")


if __name__ == "__main__":
    main()
