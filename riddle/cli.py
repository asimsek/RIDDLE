import argparse
import json
import os
from pathlib import Path
import re
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("lacathode", "riddle")
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
    p = argparse.ArgumentParser(description="Independent LaCathode and RIDDLE LHCO pipelines")
    subs = p.add_subparsers(dest="command", required=True)
    setup = subs.add_parser("setup", help="Verify the manually cloned, pinned LaCathode checkout")
    setup.add_argument("--sources", type=Path, default=ROOT / "external/lacathode")
    prep = subs.add_parser("prepare", help="Download, verify and prepare LHCO data")
    prep.add_argument("--dataset", choices=["lhco"], default="lhco")
    prep.add_argument("--catalog", type=Path, default=ROOT / "config/datasets.yaml")
    prep.add_argument("--output", type=Path, default=Path("data/lhco"))
    prep.add_argument("--variant", choices=["default", "shifted", "deltaR"], default="default")
    prep.add_argument("--io-workers", type=positive, default=4)
    prep.add_argument("--resume", action="store_true")
    prep.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    run = subs.add_parser("run", help="Run either or both independent methods")
    run.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    run.add_argument("--data", type=Path, default=Path("data/lhco"))
    run.add_argument("--output", type=Path, default=Path("results"))
    run.add_argument("--sources", type=Path, default=ROOT / "external/lacathode")
    run.add_argument("--config", type=Path, default=ROOT / "config/settings.yaml")
    run.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["signal_injection"])
    run.add_argument("--seeds", type=seeds, default=[42])
    run.add_argument("--device", default="cpu")
    run.add_argument(
        "--workers",
        type=positive,
        default=1,
        help="Concurrent RIDDLE fits; method/seed jobs remain isolated subprocesses",
    )
    run.add_argument("--io-workers", type=positive, default=2)
    run.add_argument("--mps", choices=["auto", "on", "off"], default="auto")
    run.add_argument("--runs", type=positive, help="Override YAML RIDDLE fits per fraction")
    run.add_argument(
        "--epochs", type=positive, help="Override YAML RIDDLE epochs; LaCathode remains fixed at 100"
    )
    run.add_argument("--fractions", nargs="+", help="Override YAML mixture-fraction configurations")
    run.add_argument("--resume", action="store_true")
    run.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    return p


def run_campaign(args):
    from .storage import locked
    from .worker_progress import monitor_worker
    from .data import validate

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
        manifest = validate(args.data / scenario)
        if "riddle" in args.methods:
            input_features(args.settings, manifest)
    if "lacathode" in args.methods:
        from external.lacathode_utils.source import verify

        verify(args.sources)
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
                with locked(output / ".resume/command.lock"):
                    monitor_worker(
                        [sys.executable, "-m", "riddle.worker", json.dumps(options, default=str)],
                        env,
                        output / "training.log",
                        f"{method} | {scenario}",
                        resume=args.resume,
                    )


def main(argv=None):
    p = parser()
    args = p.parse_args(argv)
    from .progress import set_verbosity, progress_session, colored_status
    from .worker_progress import local_progress

    set_verbosity(getattr(args, "verbose", 1))
    try:
        with progress_session(enabled=True, name="RIDDLE"), local_progress("Framework"):
            if args.command == "setup":
                from external.lacathode_utils.source import setup

                setup(args.sources)
            elif args.command == "prepare":
                from .data import prepare

                prepare(args)
            else:
                run_campaign(args)
        colored_status("Completed", kind="PASS")
        return 0
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        p.exit(1, f"[ERROR] {error}\n")
    except KeyboardInterrupt:
        p.exit(130, "[WARNING] Interrupted; completed checkpoints retained\n")
