#!/usr/bin/env python3
import argparse
import fcntl
import hashlib
import json
from pathlib import Path
import platform
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def inspect_environment(root):
    if Path(sys.prefix).resolve() != (root / ".venv").resolve():
        raise RuntimeError("Activate the shared project .venv before running this command")
    frozen = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
    packages = "\n".join(sorted(frozen.splitlines())) + "\n"
    identity = {
        "python": platform.python_version(),
        "machine": platform.machine(),
        "lock_sha256": hashlib.sha256(packages.encode()).hexdigest(),
    }
    return packages, identity


def verify(root, *, freeze=False):
    root = Path(root).resolve()
    lock, identity_path = root / "requirements-nrp.lock", root / "environment-nrp.json"
    packages, identity = inspect_environment(root)
    if freeze:
        with (root / ".environment.lock").open("a") as guard:
            fcntl.flock(guard, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if lock.exists() or identity_path.exists():
                raise FileExistsError("Environment already frozen; do not change it during a campaign")
            subprocess.run([sys.executable, "-m", "pip", "check"], check=True)
            with lock.open("x") as stream:
                stream.write(packages)
            with identity_path.open("x") as stream:
                json.dump(identity, stream, indent=2)
        print("[PASS] Shared environment frozen", flush=True)
    else:
        if not lock.is_file() or not identity_path.is_file():
            raise RuntimeError("Freeze the shared environment once before submitting jobs")
        if packages != lock.read_text() or identity != json.loads(identity_path.read_text()):
            raise RuntimeError("Shared Python environment changed; restore it or use a new campaign area")
        print("[PASS] Shared environment verified", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Freeze or verify the persistent shared Python environment")
    parser.add_argument("--freeze", action="store_true")
    args = parser.parse_args(argv)
    try:
        verify(ROOT, freeze=args.freeze)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"[ERROR] {error}\n")


if __name__ == "__main__":
    main()
