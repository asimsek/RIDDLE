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
from .resume import inspect_resume, record_transition, resume_policy
from .integrity import SCIENTIFIC_VERSION
from .production import validate_result_scores


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
        "integrity.py",
        "production.py",
        "metrics.py",
        "scan.py",
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
        "resume.py",
        "mps.py",
        "gpu_identity.py",
    )
    code = {f"framework/{name}": file_digest(package / name) for name in framework}
    code.update(
        {f"lacathode/{k}": v for k, v in code_hashes(package.parent / "external/lacathode_utils").items()}
    )
    return code


def main():
    args = SimpleNamespace(**json.loads(sys.argv[1]))
    policy = resume_policy(args)
    for name in ("output", "data", "sources"):
        setattr(args, name, Path(getattr(args, name)))

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    concurrent = args.method == "riddle"
    workers = getattr(args, "workers", 1)
    if args.method == "lacathode" and not getattr(args, "lacathode_replica", False):
        from external.lacathode_utils.pipeline import run_settings

        settings = run_settings(getattr(args, "runs", None), getattr(args, "epochs", None),
                                getattr(args, "lacathode_background", "independent"))
        workers = min(workers, settings["pipeline_runs"])
        concurrent = getattr(args, "lacathode_background", "independent") == "independent" and workers > 1
    if concurrent:
        from .mps import configure_mps

        if (
            args.device != "cpu"
            and workers > 1
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
        status = configure_mps(args.mps, args.device, workers)
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
    else:
        from external.lacathode_utils.resume import contract_settings

        settings.update(contract_settings(getattr(args, "runs", None), getattr(args, "epochs", None),
                                          getattr(args, "lacathode_background", "independent")))
    contract = {
        "scientific_version": SCIENTIFIC_VERSION if args.method == "riddle" else "pinned_upstream",
        "flow_checkpoint_prefix": args.method + "_model",
        "schema": 1,
        "method": args.method,
        "inputs": inputs,
        "environment": environment(),
        "code": code,
        "settings": settings,
    }
    if args.method == "riddle":
        from .production import PRODUCTION_POLICY

        contract["riddle_production_policy"] = PRODUCTION_POLICY
    if args.method == "lacathode":
        from external.lacathode_utils.source import verify, COMMIT
        from external.lacathode_utils.pipeline import RUN_LAYOUT, FIXED_RUN_LAYOUT

        contract.update(source_sha256=verify(args.sources), source_commit=COMMIT,
                        lacathode_run_layout=(FIXED_RUN_LAYOUT
                            if getattr(args, "lacathode_background", "independent") == "fixed" else RUN_LAYOUT))
        if getattr(args, "lacathode_replica", False):
            contract["settings"].update(campaign_seed=args.campaign_seed, run_index=args.run_index)
    path = args.output / "result.json"
    saved, changes = inspect_resume(args.output, contract, resume=args.resume, **policy)
    completed = saved is not None and saved["completed"]
    if completed:
        verify_artifacts(args.output, saved["artifacts_sha256"])
        validate_result_scores(args.output, args.method)
    if saved is not None:
        record_transition(
            args.output / ".resume/resume_history.json", saved["contract"], contract, changes,
            action="reuse_completed_result" if completed else "resume_requested",
        )
    if changes:
        kinds = "/".join(sorted({c["kind"] for c in changes}))
        emit_message(
            f"Permitted resume across {kinds} changes recorded; bitwise reproducibility is not guaranteed",
            kind="WARNING",
        )
    if completed:
        emit_message("Reuse verified completed result; original provenance retained", kind="PASS")
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
    if args.method == "lacathode" and getattr(args, "lacathode_replica", False):
        report.update(campaign_seed=args.campaign_seed, run_index=args.run_index)
    if args.method == "riddle":
        report.update(campaign_seed=getattr(args, "campaign_seed", args.seed),
                      run_index=getattr(args, "run_index", 0),
                      independent_run_count=getattr(args, "independent_run_count", 1),
                      ensemble_fits=args.runs)
    if saved is not None and (changes or "initial_contract" in saved):
        report["initial_contract"] = saved.get("initial_contract", saved["contract"])
    write_json(path, report)
    if args.method == "riddle":
        from .pipeline import run
    else:
        if getattr(args, "lacathode_replica", False):
            from external.lacathode_utils.pipeline import run_single as run
        else:
            from external.lacathode_utils.pipeline import run
    run(args, contract)
    validate(args.data)
    write_json(args.output / "score_health.json", validate_result_scores(args.output, args.method))
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
