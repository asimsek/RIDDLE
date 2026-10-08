from copy import copy
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile

from .data import read_sources, export_roles, validate, supervised_sources, validate_source_configuration, preparation_spec
from .features import build_dataset_roles
from . import controls
from .populations import load_config
from .settings import load_settings
from .storage import locked, write_json, file_digest

SCAN_SCHEMA = 2


def points(args):
    config = load_settings(args.config)["injection_scan"]
    population = load_config(getattr(args, "population_config", None))
    counts = getattr(args, "signal_events", None)
    counts = config["signal_events"] if counts is None else counts
    if not counts or len(set(counts)) != len(counts):
        raise ValueError("Empty or duplicate scan points")
    if any(type(n) is not int or n not in config["signal_events"] for n in counts):
        raise ValueError("Requested signal counts must be configured in settings.yaml")
    return [dict(schema=SCAN_SCHEMA, signal_events=n, preparation_seed=population["preparation_seed"])
            for n in counts]


def point_name(point):
    name = f"signal_{point['signal_events']:06d}"
    return name if point.get("schema", 1) == SCAN_SCHEMA else f"{name}/replica_{point['replica']:03d}"


def require_current_layout(root):
    if next(root.glob("signal_*/replica_*"), None) is not None:
        raise ValueError("Legacy scan replicas use different populations; use a new scan directory")


def verify_point(manifest, point, population, settings):
    if (manifest.get("injection_scan") != point
            or manifest.get("shared_population", {}).get("configuration") != population
            or manifest.get("preparation") != asdict(preparation_spec(population, settings, signal_events=point["signal_events"]))):
        raise ValueError("Prepared scan point differs from the shared population settings; prepare a new scan directory")


def reuse_candidates(args, output_root, variant, seed):
    suffix = "" if variant == "default" else "_" + variant
    roots = getattr(args, "reuse_results", None)
    roots = [Path("results" + suffix), output_root.parent / ("results" + suffix)] if roots is None else roots
    roots = list(dict.fromkeys(Path(root).resolve() for root in [*roots, output_root]))
    candidates = []
    for root in roots:
        for method in ("riddle", "iad", "supervised", "lacathode", "ranode"):
            relative = Path(method) / "signal_injection" / f"seed_{seed:03d}" / "result.json"
            paths = [root / relative, *sorted(root.glob("signal_*/" + str(relative)))]
            for path in paths:
                try:
                    report = json.loads(path.read_text())
                except (OSError, ValueError):
                    continue
                if (report.get("completed") is True and report.get("seed") == seed
                        and report.get("variant", report.get("contract", {}).get("inputs", {}).get("variant", "default")) == variant):
                    candidates.append(str(path.parent))
    return list(dict.fromkeys(candidates))


def prepare_scan(args):
    population = load_config(getattr(args, "population_config", None))
    settings = load_settings(args.config)
    root = args.output.resolve()
    plan = points(args)
    controls.columns(args.variant)
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / ".prepare.lock"):
        require_current_layout(root)
        pending = []
        for point in plan:
            destination = root / point_name(point)
            if destination.exists():
                if not args.resume:
                    raise FileExistsError("Prepared scan point exists; use --resume or a new output")
                manifest = validate(destination / "signal_injection")
                validate_source_configuration(manifest, args.catalog)
                verify_point(manifest, point, population, settings)
                if manifest.get("variant") != args.variant:
                    raise ValueError("Prepared scan variant changed")
            else:
                pending.append(point)
        if pending:
            primary, extra, by_source, provenance = read_sources(args, root, args.variant)
        for point in pending:
            spec = preparation_spec(population, settings, signal_events=point["signal_events"])
            roles = build_dataset_roles(
                primary, spec, sic_background_arrays=extra, enforce_expected_counts=True, population=population,
                supervised_signal_arrays=supervised_sources(by_source, provenance),
            )
            if args.variant == "deltaR":
                controls.attach_delta_r(roles, by_source)
            destination = root / point_name(point)
            stage = Path(tempfile.mkdtemp(prefix=".prepare-", dir=root))
            export_roles(roles, stage, spec, provenance, variant=args.variant, scan=point,
                         scenarios=("signal_injection",), population=population)
            validate(stage / "signal_injection")
            os.rename(stage, destination)
        write_json(root / "scan_inputs.json", {
            "schema": SCAN_SCHEMA,
            "variant": args.variant,
            "points": {str(p.parent.relative_to(root)): file_digest(p)
                       for p in sorted(root.glob("signal_*/signal_injection/inputs.json"))},
        })


def run_scan(args):
    from .cli import run_campaign

    settings = load_settings(args.config)
    population = load_config(getattr(args, "population_config", None))
    selected_seeds = getattr(args, "seeds", None)
    selected_seeds = settings["injection_scan"]["seeds"] if selected_seeds is None else selected_seeds
    if (not selected_seeds or len(set(selected_seeds)) != len(selected_seeds)
            or any(type(seed) is not int or not 0 <= seed < 2**32 for seed in selected_seeds)):
        raise ValueError("Provide distinct 32-bit scan training seeds")
    plan = points(args)
    data_root, output_root = args.data.resolve(), args.output.resolve()
    baseline = population["injected_signal_events"]
    background_mode = settings["injection_scan"]["background_mode"] if getattr(args, "scan_reuse", True) else "retrain"
    if background_mode == "reuse":
        plan.sort(key=lambda point: point["signal_events"] != baseline)
    require_current_layout(data_root)
    require_current_layout(output_root)
    native_requested = any(method in args.methods for method in ("riddle", "iad", "supervised"))
    oracle_requested = any(method in args.methods for method in ("iad", "supervised"))
    manifests = {}
    for point in plan:
        manifest = validate(
            data_root / point_name(point) / "signal_injection",
            require_event_ids=native_requested,
            require_oracle=oracle_requested,
            require_supervised="supervised" in args.methods,
        )
        verify_point(manifest, point, population, settings)
        manifests[point["signal_events"]] = manifest
    from .scan_cache import background_population, require_background_population
    nominal_data = data_root / f"signal_{baseline:06d}" / "signal_injection"
    nominal_manifest = manifests.get(baseline)
    if nominal_manifest is None:
        nominal_manifest = validate(nominal_data, require_event_ids=True)
        verify_point(nominal_manifest, dict(schema=SCAN_SCHEMA, signal_events=baseline,
                     preparation_seed=population["preparation_seed"]), population, settings)
    nominal_background = background_population(nominal_data, nominal_manifest)
    for point in plan:
        data = data_root / point_name(point) / "signal_injection"
        manifest = manifests[point["signal_events"]]
        if manifest.get("variant", "default") != nominal_manifest.get("variant", "default"):
            raise ValueError("Injection-scan strengths use different data variants")
        require_background_population(nominal_background, background_population(data, manifest), data)
    from .background_stage import NATIVE_METHODS, run_staged
    workers = getattr(args, "scan_bg_workers", 1)
    if workers > 1 and background_mode == "reuse":
        from .worker_progress import emit_message
        emit_message("Background reuse is active; --scan-bg-workers applies to retraining scans")
    for seed in selected_seeds:
        for method in args.methods:
            tasks = []
            for point in plan:
                options = copy(args)
                options.data = data_root / point_name(point)
                options.output = output_root / point_name(point)
                options.scenarios = ["signal_injection"]
                options.seeds, options.methods = [seed], [method]
                options.command = "scan"
                variant = manifests[point["signal_events"]].get("variant", "default")
                candidates = reuse_candidates(args, output_root, variant, seed) if getattr(args, "scan_reuse", True) else []
                options.scan_result_candidates = candidates
                options.scan_background_mode = background_mode
                options.scan_background_baseline = baseline
                options.scan_background_data = str(nominal_data)
                options.riddle_background_reuse_candidates = []
                options.oracle_background_reuse_candidates = []
                options.supervised_ensemble_reuse_candidates = candidates
                tasks.append(options)
            if background_mode == "retrain" and method in NATIVE_METHODS and (
                    workers > 1 or getattr(args, "background_only", False)):
                def execute(options, *, cancel_event=None):
                    candidates = reuse_candidates(args, output_root, variant, seed) if getattr(args, "scan_reuse", True) else []
                    options.scan_result_candidates = candidates
                    options.supervised_ensemble_reuse_candidates = candidates
                    run_campaign(options, cancel_event=cancel_event)

                run_staged(tasks, workers, execute,
                           background_only=getattr(args, "background_only", False))
            else:
                for options in tasks:
                    candidates = reuse_candidates(args, output_root, variant, seed) if getattr(args, "scan_reuse", True) else []
                    options.scan_result_candidates = candidates
                    options.supervised_ensemble_reuse_candidates = candidates
                    run_campaign(options)
