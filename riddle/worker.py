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
from .integrity import SCIENTIFIC_VERSION, RIDDLE_BENCHMARK_LABELS, RIDDLE_BENCHMARK_SCIENTIFIC_VERSION
from .production import validate_result_scores, NumericalFitError


def runtime_code(method, root=None):
    package = Path(root) if root is not None else Path(__file__).parent
    if method in ("riddle", "iad", "supervised"):
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
        "populations.py",
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
    native_method = args.method in ("riddle", "iad", "supervised")
    concurrent = native_method
    workers = getattr(args, "workers", 1)
    if native_method:
        workers = min(workers, int(args.runs))
    if args.method == "lacathode" and not getattr(args, "lacathode_replica", False):
        from external.lacathode_utils.pipeline import run_settings

        settings = run_settings(getattr(args, "runs", None), getattr(args, "epochs", None),
                                getattr(args, "lacathode_background", "independent"))
        background_mode = getattr(args, "lacathode_background", "independent")
        limit = settings["classifier_runs"] if background_mode == "fixed" else settings["pipeline_runs"]
        workers = min(workers, limit)
        concurrent = workers > 1
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
        args.runtime_mps = status.to_dict()
        if status.requested and not status.active:
            emit_message("MPS unavailable; ordinary concurrent fits remain enabled", kind="WARNING")
    else:
        args.runtime_mps = {"requested": False, "active": False, "started": False,
                            "pipe_directory": None, "log_directory": None,
                            "reason": "disabled_or_single_worker"}
    import torch

    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    mapping_experiment = getattr(args, "mapping_experiment", None)
    if mapping_experiment is not None:
        from .mapping_experiment import validate_worker
        validate_worker(args)
        from .mapping_experiment import environment as experiment_environment, read as experiment_read
        numerical = experiment_environment(args.output / '.resume/mapping_experiment_environment.json')
        if numerical != experiment_read(mapping_experiment['manifest'])['numerical_environment']:
            raise ValueError('Mapping experiment is not running in its pinned numerical environment')
    audit_environment = os.environ.get("RIDDLE_PROTOCOL_EXECUTION_RECORD")
    if audit_environment:
        if (args.method != "riddle" or args.device != "cpu" or
                args.settings["riddle"].get("data_policy") != "study_replay_v1"):
            raise ValueError("Paired protocol environment recording is restricted to CPU study replay")
        from scripts.riddle_protocol_environment import record
        record(audit_environment)
    if args.device != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable; refusing CPU fallback")
    oracle_method = args.method in ("iad", "supervised")
    inputs = validate(args.data, require_event_ids=native_method, require_oracle=oracle_method,
                      require_supervised=args.method == "supervised")
    if native_method:
        from .settings import input_features

        input_features(args.settings, inputs)
        from .score_selection import validate_population_contract
        validate_population_contract(args.settings["riddle"], inputs, args.method)
    code = runtime_code(args.method)
    mass_pilot = args.method == "riddle" and args.settings["riddle"].get("mass_conditioning", False)
    if mass_pilot:
        pilot_script = Path(__file__).parents[1] / "scripts/cpu_mass_dependence.py"
        if pilot_script.is_file():
            code["scripts/cpu_mass_dependence.py"] = file_digest(pilot_script)
    settings = {"seed": args.seed, "scenario": args.scenario, "device": args.device}
    if native_method:
        settings.update(args.settings)
        if oracle_method:
            settings.update(oracle_benchmark=args.method, run_index=getattr(args, "run_index", 0),
                            campaign_seed=getattr(args, "campaign_seed", args.seed))
    else:
        from external.lacathode_utils.resume import contract_settings

        settings.update(contract_settings(getattr(args, "runs", None), getattr(args, "epochs", None),
                                          getattr(args, "lacathode_background", "independent")))
    contract = {
        "scientific_version": (RIDDLE_BENCHMARK_SCIENTIFIC_VERSION if oracle_method else
                               SCIENTIFIC_VERSION if args.method == "riddle" else "pinned_upstream"),
        "flow_checkpoint_prefix": args.method + "_model",
        "schema": 1,
        "method": args.method,
        "inputs": inputs,
        "environment": environment(),
        "code": code,
        "settings": settings,
    }
    if native_method:
        from .production import PRODUCTION_POLICY

        contract["riddle_production_policy"] = PRODUCTION_POLICY
        contract["riddle_score_scope"] = ("signal_region" if args.settings["riddle"].get("mass_conditioning")
                                          else "full_region")
        if oracle_method:
            contract["oracle_benchmark"] = {
                "method": args.method,
                "public_label": RIDDLE_BENCHMARK_LABELS[args.method],
                "p": "signal_region_data_mixture" if args.method == "iad" else "pure_signal",
                "q": "pure_background",
                "core": "riddle_stein_witness",
            }
        if mapping_experiment is not None:
            contract["mapping_experiment"] = mapping_experiment
    if args.method == "lacathode":
        from external.lacathode_utils.source import verify, COMMIT
        from external.lacathode_utils.pipeline import RUN_LAYOUT, FIXED_RUN_LAYOUT

        fixed_background = getattr(args, "lacathode_background", "independent") == "fixed"
        contract.update(fit_failure_policy=("fail_closed_fixed_background_v1" if fixed_background
                                           else "exclude_numerically_invalid_independent_runs_v1"),
                        source_sha256=verify(args.sources), source_commit=COMMIT,
                        lacathode_run_layout=FIXED_RUN_LAYOUT if fixed_background else RUN_LAYOUT)
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
    if changes and completed:
        emit_message("Completed-result reuse differences recorded: "
                     + ", ".join(c["field"] for c in changes))
    elif changes:
        kinds = "/".join(sorted({c["kind"] for c in changes}))
        emit_message(
            f"Permitted resume across {kinds} changes recorded; bitwise reproducibility is not guaranteed",
            kind="WARNING",
        )
    if completed:
        if oracle_method:
            history_path = args.output / ".resume" / "resume_history.json"
            saved["resume_history"] = json.loads(history_path.read_text()) if history_path.is_file() else {"schema": 1, "transitions": []}
            write_json(path, saved)
        emit_message("Reuse verified completed result; original provenance retained", kind="PASS")
        return
    report = {
        "schema": 1,
        "method": ("riddlev3" if args.settings["riddle"].get("input_space") == "physical" else "riddlev2") if mass_pilot else args.method,
        "seed": args.seed,
        "scenario": args.scenario,
        "variant": inputs.get("variant", "default"),
        "contract": contract,
        "completed": False,
    }
    if args.method == "lacathode" and getattr(args, "lacathode_replica", False):
        report.update(campaign_seed=args.campaign_seed, run_index=args.run_index)
    if oracle_method:
        report.update(public_label=RIDDLE_BENCHMARK_LABELS[args.method], scientific_version=RIDDLE_BENCHMARK_SCIENTIFIC_VERSION,
                      benchmark_protocol="riddle_oracle", score_scope=contract["riddle_score_scope"],
                      campaign_seed=getattr(args, "campaign_seed", args.seed),
                      run_index=getattr(args, "run_index", 0),
                      independent_run_count=getattr(args, "independent_run_count", 1),
                      ensemble_fits=args.runs, epochs=args.epochs,
                      device=args.device, workers=workers, io_workers=args.io_workers,
                      torch_threads=args.torch_threads, mps=args.runtime_mps)
    if args.method == "riddle":
        report.update(score_scope=contract["riddle_score_scope"],
                      campaign_seed=getattr(args, "campaign_seed", args.seed),
                      run_index=getattr(args, "run_index", 0),
                      independent_run_count=getattr(args, "independent_run_count", 1),
                      ensemble_fits=args.runs)
    if saved is not None and (changes or "initial_contract" in saved):
        report["initial_contract"] = saved.get("initial_contract", saved["contract"])
    write_json(path, report)
    if native_method:
        from .pipeline import run
    else:
        if getattr(args, "lacathode_replica", False):
            from external.lacathode_utils.pipeline import run_single as run
        else:
            from external.lacathode_utils.pipeline import run
    try:
        run(args, contract)
        if oracle_method:
            protocol_path = args.output / "protocol.json"
            if protocol_path.is_file():
                protocol = json.loads(protocol_path.read_text())
                report.update(oracle_benchmark=protocol.get("oracle_benchmark"),
                              score_scope=protocol.get("score_scope"),
                              objective=protocol.get("objective"),
                              selected_checkpoints=protocol.get("selected_checkpoints"),
                              accepted_fits=protocol.get("accepted_fits"),
                              ensemble_fits=protocol.get("ensemble_fits"))
                write_json(path, report)
        validate(args.data, require_oracle=oracle_method, require_supervised=args.method == "supervised")
        write_json(args.output / "score_health.json", validate_result_scores(args.output, args.method))
    except (FloatingPointError, NumericalFitError) as error:
        report["failure"] = dict(kind="numerical", error=str(error), error_type=type(error).__name__,
                                 execution_token=getattr(args, "execution_token", None))
        write_json(path, report)
        raise
    artifacts = [
        p
        for p in args.output.rglob("*")
        if p.is_file()
        and ".resume" not in p.relative_to(args.output).parts
        and p != path
        and p.name != "training.log"
    ]
    history_path = args.output / ".resume" / "resume_history.json"
    if oracle_method:
        report["resume_history"] = json.loads(history_path.read_text()) if history_path.is_file() else {"schema": 1, "transitions": []}
    report.update(completed=True, artifacts_sha256=fingerprint_files(args.output, artifacts, args.io_workers))
    write_json(path, report)
    emit_message("Verified result saved", kind="PASS")


if __name__ == "__main__":
    main()
