import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace

from .storage import code_hashes, environment, file_digest, write_json, verify_artifacts, fingerprint_files
from .data import validate
from .worker_progress import emit_message


def runtime_code(method, root=None):
    package = Path(root) if root is not None else Path(__file__).parent
    if method == "riddle":
        return {
            f"riddle/{p.name}": file_digest(p)
            for p in sorted(package.glob("*.py"))
            if p.name not in ("figures.py", "plotting.py")
        }
    if method != "lacathode":
        raise ValueError("Unknown method")
    framework = (
        "cli.py",
        "worker.py",
        "data.py",
        "data_spec.py",
        "features.py",
        "controls.py",
        "datasets.py",
        "storage.py",
        "worker_progress.py",
        "progress.py",
    )
    code = {f"framework/{name}": file_digest(package / name) for name in framework}
    code.update(
        {f"lacathode/{k}": v for k, v in code_hashes(package.parent / "external/lacathode_utils").items()}
    )
    return code


def main():
    args = SimpleNamespace(**json.loads(sys.argv[1]))
    for name in ("output", "data", "sources"):
        setattr(args, name, Path(getattr(args, name)))

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    if args.method == "riddle":
        from .mps import configure_mps

        if (
            args.device != "cpu"
            and args.workers > 1
            and args.mps != "off"
            and not os.environ.get("CUDA_VISIBLE_DEVICES", "").startswith(("GPU-", "MIG-"))
        ):
            try:
                identity = subprocess.check_output(
                    [sys.executable, "-m", "riddle.gpu_identity"], text=True, timeout=10
                ).strip()
                if not identity.startswith(("GPU-", "MIG-")) or "\n" in identity:
                    raise ValueError("Invalid GPU identity")
                os.environ["CUDA_VISIBLE_DEVICES"] = identity
            except (OSError, ValueError, subprocess.SubprocessError):
                if args.mps == "on":
                    raise RuntimeError("Cannot verify GPU identity for MPS; use --mps off") from None
                args.mps = "off"
                emit_message("GPU identity unavailable; using ordinary concurrent fits", kind="WARNING")
        status = configure_mps(args.mps, args.device, args.workers)
        if status.requested and not status.active:
            emit_message("MPS unavailable; ordinary concurrent fits remain enabled", kind="WARNING")
    import torch

    torch.set_num_threads(args.io_workers)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    inputs = validate(args.data)
    if args.method == "riddle":
        from .settings import input_features

        input_features(args.settings, inputs)
    code = runtime_code(args.method)
    settings = {"seed": args.seed, "scenario": args.scenario, "device": args.device}
    if args.method == "riddle":
        settings.update(args.settings)
    contract = {
        "schema": 1,
        "method": args.method,
        "inputs": inputs,
        "environment": environment(),
        "code": code,
        "settings": settings,
    }
    if args.method == "lacathode":
        from external.lacathode_utils.source import verify, COMMIT

        contract.update(source_sha256=verify(args.sources), source_commit=COMMIT)
    path = args.output / "result.json"
    if path.exists():
        saved = json.loads(path.read_text())
        if not args.resume or saved["contract"] != contract:
            raise ValueError("Resume contract changed; retain these results and use a new output")
        if saved["completed"]:
            verify_artifacts(args.output, saved["artifacts_sha256"])
            emit_message("Reuse verified completed result", kind="PASS")
            return
    report = {
        "schema": 1,
        "method": args.method,
        "seed": args.seed,
        "scenario": args.scenario,
        "variant": inputs.get("variant", "default"),
        "contract": contract,
        "completed": False,
    }
    write_json(path, report)
    if args.method == "riddle":
        from .pipeline import run
    else:
        from external.lacathode_utils.pipeline import run
    run(args, contract)
    validate(args.data)
    artifacts = [
        p
        for p in args.output.rglob("*")
        if p.is_file()
        and ".resume" not in p.relative_to(args.output).parts
        and p != path
        and p.name != "training.log"
    ]
    report.update(completed=True, artifacts_sha256=fingerprint_files(args.output, artifacts, args.io_workers))
    write_json(path, report)
    emit_message("Verified result saved", kind="PASS")


if __name__ == "__main__":
    main()
