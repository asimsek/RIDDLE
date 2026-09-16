import argparse
import importlib
import subprocess
import sys
from pathlib import Path


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
    args = parser.parse_args(argv)
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
