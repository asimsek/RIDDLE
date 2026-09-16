import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from .resume import add_resume_options, resume_policy

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("lacathode", "riddle", "ranode")
DEFAULT_METHODS = ("lacathode", "riddle")
SCENARIOS = ("signal_injection", "background_only")


def seeds(value):
    result = []
    for token in value.split(","):
        match = re.fullmatch(r"(\d+)(?:-(\d+))?", token)
        if not match:
            raise argparse.ArgumentTypeError(
                "Seeds must be nonnegative integers, ranges or comma-separated lists"
            )
        a, b = int(match[1]), int(match[2] or match[1])
        if b < a or b - a > 10000 or b >= 2**32:
            raise argparse.ArgumentTypeError("Invalid seed range")
        result.extend(range(a, b + 1))
    if len(result) != len(set(result)):
        raise argparse.ArgumentTypeError("Duplicate seeds")
    return result


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return value


def parser():
    p = argparse.ArgumentParser(description="Independent LaCathode, RIDDLE and R-ANODE LHCO pipelines")
    subs = p.add_subparsers(dest="command", required=True)
    setup = subs.add_parser("setup", help="Verify manually cloned, pinned upstream checkouts")
    setup.add_argument("--sources", type=Path, default=ROOT / "external/lacathode")
    setup.add_argument("--methods", nargs="+", choices=("lacathode", "ranode"), default=["lacathode"])
    setup.add_argument("--ranode-sources", type=Path, default=ROOT / "external/ranode")
    prep = subs.add_parser("prepare", help="Download, verify and prepare LHCO data")
    prep.add_argument("--dataset", choices=["lhco"], default="lhco")
    prep.add_argument("--catalog", type=Path, default=ROOT / "config/datasets.yaml")
    prep.add_argument("--output", type=Path, default=Path("data/lhco"))
    prep.add_argument("--variant", choices=["default", "shifted", "deltaR"], default="default")
    prep.add_argument("--io-workers", type=positive, default=4)
    prep.add_argument("--resume", action="store_true")
    prep.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    prep_scan = subs.add_parser("prepare-scan", parents=[deepcopy(prep)], add_help=False,
                                help="Prepare independently partitioned injection strengths")
    prep_scan.set_defaults(output=Path("data/injection_scan"))
    prep_scan.add_argument("--config", type=Path, default=ROOT / "config/settings.yaml")
    prep_scan.add_argument("--signal-events", type=seeds, help="Subset of configured total signal counts")
    prep_scan.add_argument("--replicas", type=seeds, help="Zero-based replica indices, e.g. 0-9")
    for command in ("run", "scan"):
        run = subs.add_parser(command, help="Run independent methods" if command == "run" else "Run the optional injection scan")
        run.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
        run.add_argument("--data", type=Path, default=Path("data/lhco"))
        run.add_argument("--output", type=Path, default=Path("results"))
        run.add_argument("--sources", type=Path, default=ROOT / "external/lacathode")
        run.add_argument("--ranode-sources", type=Path, default=ROOT / "external/ranode")
        run.add_argument("--ranode-config", type=Path, default=ROOT / "external/ranode_utils/ranode.yaml",
                         help="R-ANODE settings (default: external/ranode_utils/ranode.yaml)")
        run.add_argument("--config", type=Path, default=ROOT / "config/settings.yaml")
        run.add_argument("--device", default="cpu")
        run.add_argument(
            "--workers",
            type=positive,
            default=1,
            help="Concurrent RIDDLE fits; method/seed jobs remain isolated subprocesses",
        )
        run.add_argument("--io-workers", type=positive, default=2)
        run.add_argument("--mps", choices=["auto", "on", "off"], default="auto")
        run.add_argument("--runs", type=positive, help="Override RIDDLE/R-ANODE fit count and LaCathode independent flow-plus-classifier run count")
        run.add_argument(
            "--epochs", type=positive, help="Override RIDDLE/R-ANODE signal-fit and LaCathode classifier epochs; background stages are unchanged"
        )
        run.add_argument("--fractions", nargs="+", help="Override YAML mixture-fraction configurations")
        add_resume_options(run)
        run.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
        if command == "run":
            run.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["signal_injection"])
            run.add_argument("--seeds", type=seeds, default=[42])
        else:
            run.set_defaults(data=Path("data/injection_scan"), output=Path("results/injection_scan"))
            run.add_argument("--signal-events", type=seeds, help="Subset of configured total signal counts")
            run.add_argument("--replicas", type=seeds, help="Zero-based replica indices, e.g. 0-9")
    return p


def run_campaign(args):
    resume_policy(args)
    fit_overrides = {key: getattr(args, key, None) for key in ("runs", "epochs")}
    if "lacathode" in args.methods:
        from external.lacathode_utils.pipeline import run_settings

        run_settings(**fit_overrides)
    from .storage import locked
    from .worker_progress import monitor_worker

    if "riddle" in args.methods:
        from .settings import resolve, input_features

        resolve(args)
    else:
        args.settings = None
    if args.device != "cpu" and not re.fullmatch(r"cuda:\d+", args.device):
        raise ValueError("--device must be cpu or cuda:<index>")
    if len(args.methods) != len(set(args.methods)) or len(args.scenarios) != len(set(args.scenarios)):
        raise ValueError("Duplicate methods/scenarios")
    args.data, args.output, args.sources = (p.resolve() for p in (args.data, args.output, args.sources))
    if args.output.is_relative_to(args.data) or args.data.is_relative_to(args.output):
        raise ValueError("Keep data and results in separate directories")
    for scenario in args.scenarios:
        if args.methods == ["ranode"]:
            from external.ranode_utils.data import validate as validate_ranode
            manifest, _ = validate_ranode(args.data / scenario)
        else:
            from .data import validate
            manifest = validate(args.data / scenario)
        if "riddle" in args.methods:
            input_features(args.settings, manifest)
        if "ranode" in args.methods:
            from external.ranode_utils.data import validate_schema

            validate_schema(manifest)
    if "lacathode" in args.methods:
        from external.lacathode_utils.source import verify

        verify(args.sources)
    if "ranode" in args.methods:
        from external.ranode_utils.source import verify
        from external.ranode_utils.runner import settings

        args.ranode_sources = args.ranode_sources.resolve()
        args.ranode_config = args.ranode_config.resolve()
        verify(args.ranode_sources)
        settings(args.ranode_config, **fit_overrides)
        if args.methods == ["ranode"] and getattr(args, "fractions", None) is not None:
            raise ValueError("R-ANODE retains its upstream learned fraction; --fractions is RIDDLE-only")
    for scenario in args.scenarios:
        for seed in args.seeds:
            for method in args.methods:
                output = args.output / method / scenario / f"seed_{seed:03d}"
                if output.exists() and not args.resume:
                    raise FileExistsError("Result exists; use --resume or a new output")
                output.mkdir(parents=True, exist_ok=True)
                options = {
                    **vars(args),
                    "method": method,
                    "seed": seed,
                    "scenario": scenario,
                    "output": str(output),
                    "data": str(args.data / scenario),
                    "sources": str(args.sources),
                }
                if method == "lacathode":
                    options.update(fit_overrides)
                env = os.environ.copy()
                if args.device == "cpu":
                    env["CUDA_VISIBLE_DEVICES"] = ""
                else:
                    index = int(args.device.split(":")[1])
                    visible = env.get("CUDA_VISIBLE_DEVICES")
                    tokens = visible.split(",") if visible is not None else None
                    if tokens is not None and (
                        index >= len(tokens) or not tokens[index].strip() or tokens[index].strip() == "-1"
                    ):
                        raise ValueError("Requested CUDA device is outside the visible device set")
                    env["CUDA_VISIBLE_DEVICES"] = tokens[index].strip() if tokens else str(index)
                    options["device"] = "cuda:0"
                env.update(
                    PYTHONHASHSEED=str(seed),
                    PYTHONUNBUFFERED="1",
                    PYTHONDONTWRITEBYTECODE="1",
                    MPLBACKEND="Agg",
                    RIDDLE_WORKER_PROGRESS="1",
                    PYTHONPATH=os.pathsep.join(filter(None, (str(ROOT), env.get("PYTHONPATH")))),
                )
                for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                    env[key] = str(args.io_workers)
                if method == "ranode":
                    command = [sys.executable, "-m", "external.ranode_utils.runner",
                               "--sources", str(args.ranode_sources), "--data", options["data"],
                               "--output", str(output), "--config", str(args.ranode_config),
                               "--scenario", scenario, "--seed", str(seed), "--device", options["device"],
                               "--io-workers", str(args.io_workers)]
                    for key, value in fit_overrides.items():
                        if value is not None:
                            command.extend(["--" + key, str(value)])
                    if args.resume:
                        command.append("--resume")
                    for key in ("resume_across_code_change", "resume_across_device_change"):
                        if getattr(args, key, False):
                            command.append("--" + key.replace("_", "-"))
                else:
                    command = [sys.executable, "-m", "riddle.worker", json.dumps(options, default=str)]
                with locked(output / ".resume/command.lock"):
                    monitor_worker(
                        command,
                        env,
                        output / "training.log",
                        f"{method} | {scenario}",
                        resume=args.resume,
                    )


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    if args.command in ("run", "scan"):
        try:
            resume_policy(args)
        except ValueError as error:
            p.error(str(error))
    from .progress import set_verbosity, progress_session, colored_status
    from .worker_progress import local_progress

    set_verbosity(getattr(args, "verbose", 1))
    try:
        with progress_session(enabled=True, name="RIDDLE"), local_progress("Framework"):
            if args.command == "setup":
                if "lacathode" in args.methods:
                    from external.lacathode_utils.source import setup
                    setup(args.sources)
                if "ranode" in args.methods:
                    from external.ranode_utils.source import verify
                    verify(args.ranode_sources)
            elif args.command == "prepare":
                from .data import prepare

                prepare(args)
            elif args.command == "prepare-scan":
                from .scan import prepare_scan
                prepare_scan(args)
            elif args.command == "scan":
                from .scan import run_scan
                run_scan(args)
            else:
                run_campaign(args)
        colored_status("Completed", kind="PASS")
        return 0
    except (ValueError, OSError, RuntimeError, FloatingPointError, subprocess.SubprocessError) as error:
        p.exit(1, f"[ERROR] {error}\n")
    except KeyboardInterrupt:
        p.exit(130, "[WARNING] Interrupted; completed checkpoints retained\n")
