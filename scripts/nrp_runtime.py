#!/usr/bin/env python3
import argparse
import importlib
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
IMAGE_ROOT = Path("/opt/riddle-image")
IMPORTS = (
    "h5py", "matplotlib", "mplhep", "numpy", "pandas", "yaml", "sklearn",
    "scipy", "tables", "torch", "tqdm", "nflows.flows.base", "vector", "wandb",
)


def normalized(text):
    return sorted(line.strip() for line in text.splitlines() if line.strip() and not line.lstrip().startswith("#"))


def verify(root=ROOT, *, require_cuda=False, image_root=IMAGE_ROOT):
    if Path(sys.prefix).resolve() != Path("/opt/conda").resolve():
        raise RuntimeError("Use the runtime image's /opt/conda/bin/python")
    root, image_root = Path(root), Path(image_root)
    lock = image_root / "packages.lock"
    requirements = image_root / "requirements.txt"
    if not lock.is_file() or not requirements.is_file():
        raise RuntimeError("Missing runtime image metadata; use the pre-built RIDDLE image")
    if normalized((root / "requirements.txt").read_text()) != normalized(requirements.read_text()):
        raise RuntimeError("Project dependencies differ from the image; rebuild and pin a new runtime image")
    frozen = subprocess.check_output(
        [sys.executable, "-m", "pip", "--disable-pip-version-check", "--no-cache-dir", "freeze"],
        text=True,
    )
    if normalized(frozen) != normalized(lock.read_text()):
        raise RuntimeError("Image packages changed; recreate the pod using the pinned runtime image")
    for name in IMPORTS:
        importlib.import_module(name)
    import torch

    if torch.version.cuda is None:
        raise RuntimeError("The runtime must include CUDA-enabled PyTorch")
    if require_cuda:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; check the GPU allocation and container runtime")
        torch.ones(1, device="cuda:0").sum().item()
    print(f"[PASS] Container runtime verified: PyTorch {torch.__version__}, CUDA {torch.version.cuda}", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Verify the pre-built RIDDLE container runtime")
    parser.add_argument("--require-cuda", action="store_true", help="Also test access to the allocated GPU")
    args = parser.parse_args(argv)
    try:
        verify(require_cuda=args.require_cuda)
    except (ImportError, ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"[ERROR] {error}\n")


if __name__ == "__main__":
    main()
