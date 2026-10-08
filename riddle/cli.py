import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import subprocess
import sys
from .resume import add_resume_options, resume_policy
from .settings import default_config_path

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("lacathode", "riddle", "ranode", "iad", "supervised")
DEFAULT_METHODS = ("lacathode", "riddle")
METHOD_ALIASES = {"idealized": "iad", "idealized_ad": "iad", "supervised_ad": "supervised"}
SCENARIOS = ("signal_injection", "background_only")



def method_name(value):
    return METHOD_ALIASES.get(value, value)

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


class SeedList(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        result = [seed for group in values for seed in group]
        if len(result) != len(set(result)):
            raise argparse.ArgumentError(self, "Duplicate seeds")
        setattr(namespace, self.dest, result)


def add_seed_argument(parser, default=(42,)):
    parser.add_argument("--seed", "--seeds", dest="seeds", nargs="+", type=seeds,
                        action=SeedList, default=None if default is None else list(default),
                        help="Explicit seeds; one complete run per seed"
                             + (" (scan default: settings.yaml; run default: 42)" if default is None else " (default: 42)"))


def positive(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("Must be positive")
    return value


def parser():
    p = argparse.ArgumentParser(description="Independent LaCathode, RIDDLE, R-ANODE, Idealized RIDDLE (IAD), and Supervised RIDDLE LHCO pipelines")
    subs = p.add_subparsers(dest="command", required=True)
    setup = subs.add_parser("setup", help="Verify manually cloned, pinned upstream checkouts")
    setup.add_argument("--sources", type=Path, default=ROOT / "external/lacathode")
    setup.add_argument("--methods", nargs="+", choices=("lacathode", "ranode"), default=["lacathode"])
    setup.add_argument("--ranode-sources", type=Path, default=ROOT / "external/ranode")
    prep = subs.add_parser("prepare", help="Download, verify and prepare LHCO data")
    prep.add_argument("--dataset", choices=["lhco"], default="lhco")
    prep.add_argument("--catalog", type=Path, default=default_config_path("datasets.yaml"))
    prep.add_argument("--output", type=Path, default=Path("data/lhco"))
    prep.add_argument("--config", type=Path, default=default_config_path("settings.yaml"))
    prep.add_argument("--population-config", type=Path, default=default_config_path("populations.yaml"),
                      help="Shared BG/signal percentages and independent validation reserves")
    prep.add_argument("--variant", choices=["default", "shifted", "deltaR"], default="default")
    prep.add_argument("--io-workers", type=positive, default=4)
    prep.add_argument("--resume", action="store_true")
    prep.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
    prep_scan = subs.add_parser("prepare-scan", parents=[deepcopy(prep)], add_help=False,
                                help="Prepare injection strengths with the ordinary shared populations")
    prep_scan.set_defaults(output=Path("data/injection_scan"))
    prep_scan.add_argument("--signal-events", type=seeds, help="Subset of configured total signal counts")
    for command in ("run", "scan"):
        run = subs.add_parser(command, allow_abbrev=False,
                              help="Run independent methods" if command == "run" else "Run the optional injection scan")
        run.add_argument("--methods", nargs="+", type=method_name, choices=METHODS, default=list(DEFAULT_METHODS))
        run.add_argument("--data", type=Path, default=Path("data/lhco"))
        run.add_argument("--output", type=Path, default=Path("results"))
        run.add_argument("--sources", type=Path, default=ROOT / "external/lacathode")
        run.add_argument("--ranode-sources", type=Path, default=ROOT / "external/ranode")
        run.add_argument("--ranode-config", type=Path, default=ROOT / "external/ranode_utils/ranode.yaml",
                         help="R-ANODE settings (default: external/ranode_utils/ranode.yaml)")
        run.add_argument("--config", type=Path, default=default_config_path("settings.yaml"))
        run.add_argument("--device", default="cpu")
        run.add_argument(
            "--workers",
            type=positive,
            default=1,
            help="Concurrent RIDDLE/Idealized-RIDDLE/Supervised-RIDDLE/R-ANODE fits, independent LaCathode runs, or fixed-background LaCathode classifier fits",
        )
        run.add_argument("--io-workers", type=positive, default=2,
                         help="Filesystem/host I/O concurrency; independent of PyTorch compute threads")
        run.add_argument("--scan-bg-workers", type=positive, default=1,
                         help="Concurrent native background preparations: strengths within each retraining scan seed, or seeds in run mode")
        run.add_argument("--torch-threads", type=positive, default=2,
                         help="Intra-op CPU threads per training process (default: 2)")
        run.add_argument("--mps", choices=["auto", "on", "off"], default="auto")
        run.add_argument("--fits", type=positive,
                         help="Ensemble fits per RIDDLE/Idealized-RIDDLE/Supervised-RIDDLE/R-ANODE run (default: method settings); does not change LaCathode")
        run.add_argument("--lacathode-background", choices=("independent", "fixed"), default="independent",
                         help="Retrain each LaCathode background flow (default), or share one flow across classifier fits")
        run.add_argument(
            "--epochs", type=positive, help="Override RIDDLE/Idealized-RIDDLE/Supervised-RIDDLE/R-ANODE fit epochs and LaCathode classifier epochs; background stages are unchanged"
        )
        run.add_argument("--fractions", nargs="+", help="Override YAML mixture-fraction configurations (residual only)")
        from .roles import POLICIES
        run.add_argument("--data-policy", choices=tuple(POLICIES), help="Source-role policy; default production_v2 uses HC study mapping + production residual roles")
        run.add_argument("--ensemble-completion", choices=("strict", "partial"), help="Require every requested fit, or explicitly retain a partial ensemble")
        run.add_argument("--mass-conditioning", action=argparse.BooleanOptionalAction, default=None,
                         help="Condition the RIDDLE latent background/model on mjj inside the signal region")
        run.add_argument("--background-correction", choices=("none", "bgcorr_40_reguide"), default=None,
                         help="Optional latent background correction; bgcorr_40_reguide trains one shared 40-epoch q_phi(z|m) used by either RIDDLE core")
        from .options import add_feature_arguments
        add_feature_arguments(run)
        add_resume_options(run)
        run.add_argument("--verbose", type=int, choices=[0, 1, 2], default=1)
        if command == "run":
            run.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["signal_injection"])
            add_seed_argument(run)
        else:
            run.set_defaults(data=Path("data/injection_scan"), output=Path("results/injection_scan"))
            run.add_argument("--signal-events", type=seeds, help="Subset of configured total signal counts")
            add_seed_argument(run, default=None)
            run.add_argument("--population-config", type=Path, default=default_config_path("populations.yaml"))
            run.add_argument("--scan-reuse", action=argparse.BooleanOptionalAction, default=True)
            run.add_argument("--reuse-results", nargs="+", type=Path, help="Search these result roots for compatible completed work")
    return p


def independent_run_seeds(seed, count):
    """Stable complete-run seeds; run zero preserves existing result locations."""
    import numpy as np

    if type(seed) is not int or not 0 <= seed < 2**32 or type(count) is not int or not 1 <= count < 2**32:
        raise ValueError("Independent runs require a 32-bit seed and a positive run count")
    values = [seed] + [int(np.random.SeedSequence([seed, i]).generate_state(1)[0]) for i in range(1, count)]
    if len(set(values)) != len(values):
        raise ValueError("Independent run seeds collided; choose another base seed")
    return values


def campaign_runs(args):
    requests, used = [], set()
    for seed in args.seeds:
        for method in args.methods:
            identity = (method, seed)
            if identity in used:
                raise ValueError("Duplicate method/seed combination")
            used.add(identity)
            requests.append((method, seed, 0, seed))
    return requests


def run_campaign(args, *, cancel_event=None):
    resume_policy(args)
    if len(args.methods) != len(set(args.methods)) or len(args.scenarios) != len(set(args.scenarios)):
        raise ValueError("Duplicate methods/scenarios")
    campaign_runs(args)
    if cancel_event is not None and cancel_event.is_set():
        raise RuntimeError("Background preparation cancelled")
    if (getattr(args, "scan_bg_workers", 1) > 1 and not getattr(args, "background_phase", None)
            and getattr(args, "command", "run") != "scan"):
        from .background_stage import run_seed_backgrounds
        return run_seed_backgrounds(args, run_campaign)
    resume_policy(args)
    run_overrides = {"runs": 1, "epochs": getattr(args, "epochs", None)}
    fit_overrides = dict(runs=getattr(args, "fits", None), epochs=getattr(args, "epochs", None))
    requests = campaign_runs(args)
    if "lacathode" in args.methods:
        from external.lacathode_utils.pipeline import run_settings

        run_settings(**run_overrides, background=getattr(args, "lacathode_background", "independent"))
    from .storage import locked
    from .worker_progress import monitor_worker

    if any(method in args.methods for method in ("riddle", "iad", "supervised")):
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
    native_requested = any(method in args.methods for method in ("riddle", "iad", "supervised"))
    oracle_requested = any(method in args.methods for method in ("iad", "supervised"))
    manifests = {}
    for scenario in args.scenarios:
        if args.methods == ["ranode"]:
            from external.ranode_utils.data import validate as validate_ranode
            manifest, _ = validate_ranode(args.data / scenario)
        else:
            from .data import validate
            manifest = validate(args.data / scenario, require_event_ids=native_requested, require_oracle=oracle_requested,
                                require_supervised="supervised" in args.methods)
        manifests[scenario] = manifest
        if native_requested:
            input_features(args.settings, manifest)
            from .score_selection import validate_population_contract
            for method in args.methods:
                if method in ("riddle", "iad", "supervised"):
                    validate_population_contract(args.settings["riddle"], manifest, method)
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
        for method, base_seed, run_index, seed in requests:
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
                "campaign_seed": base_seed,
                "run_index": run_index,
                "independent_run_count": run_overrides["runs"] or 1,
            }
            if method == "lacathode":
                options.update(run_overrides)
            elif method == "riddle":
                options["runs"] = args.fits
            elif method in ("iad", "supervised"):
                options["runs"] = args.fits
            env = os.environ.copy()
            manifest = manifests[scenario]
            injection = (manifest.get("injection_scan") or {}).get(
                "signal_events", manifest.get("preparation", {}).get("injected_signal_rows", "nominal"))
            if scenario == "background_only":
                injection = 0
            task = (f"method={method}; seed={seed}; scenario={scenario}; "
                    f"signal_events={injection}; variant={manifest.get('variant', 'default')}")
            if env.get("RIDDLE_JOB_NAME"):
                task = f"job={env['RIDDLE_JOB_NAME']}; {task}"
            env["RIDDLE_LOG_TASK"] = task
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
            if getattr(args, "background_benchmark", None):
                env["RIDDLE_BG_BENCHMARK"] = "1"
            for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
                env[key] = str(args.torch_threads)
            if method == "ranode":
                command = [sys.executable, "-m", "external.ranode_utils.runner",
                           "--sources", str(args.ranode_sources), "--data", options["data"],
                           "--output", str(output), "--config", str(args.ranode_config),
                           "--scenario", scenario, "--seed", str(seed), "--device", options["device"],
                           "--io-workers", str(args.io_workers), "--torch-threads", str(args.torch_threads),
                           "--workers", str(args.workers), "--mps", args.mps]
                for key, value in fit_overrides.items():
                    if value is not None:
                        command.extend(["--fits" if key == "runs" else "--" + key, str(value)])
                command.extend(["--campaign-seed", str(base_seed), "--run-index", str(run_index),
                                "--independent-run-count", str(run_overrides["runs"] or 1)])
                reuse_policy = getattr(args, "ranode_scan_background_reuse_policy", None)
                if reuse_policy is not None:
                    command.extend(["--scan-background-reuse-policy", reuse_policy])
                for candidate in getattr(args, "ranode_background_reuse_candidates", None) or ():
                    command.extend(["--background-reuse-candidate", str(candidate)])
                for candidate in getattr(args, "scan_result_candidates", None) or ():
                    command.extend(["--scan-result-candidate", str(candidate)])
                for key in ("scan_background_mode", "scan_background_baseline", "scan_background_data"):
                    value = getattr(args, key, None)
                    if value is not None:
                        command.extend(["--" + key.replace("_", "-"), str(value)])
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
                    f"{method} | {scenario} | seed {seed} | {args.data.name}",
                    resume=args.resume,
                    cancel_event=cancel_event,
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
