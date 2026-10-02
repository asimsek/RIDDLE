"""Optional signal-strength study with paired methods and independent partition seeds."""

from copy import copy
from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile

from .data import read_sources, export_roles, validate
from .data_spec import DatasetSpec
from .features import build_dataset_roles
from . import controls
from .settings import load_settings
from .storage import locked, write_json, file_digest

SCAN_SCHEMA = 1


def points(args):
    config = load_settings(args.config)["injection_scan"]
    counts = getattr(args, "signal_events", None) or config["signal_events"]
    replicas = getattr(args, "replicas", None)
    replicas = list(range(config["replicas"])) if replicas is None else replicas
    if (
        not counts
        or not replicas
        or len(set(counts)) != len(counts)
        or len(set(replicas)) != len(replicas)
    ):
        raise ValueError("Empty or duplicate scan points")
    if any(n not in config["signal_events"] for n in counts):
        raise ValueError("Requested signal counts must be configured in settings.yaml")
    if any(type(r) is not int or not 0 <= r < config["replicas"] for r in replicas):
        raise ValueError("Replica index outside configured scan")
    return [
        dict(
            schema=SCAN_SCHEMA,
            signal_events=n,
            replica=r,
            preparation_seed=config["preparation_seed"] + r,
            training_seed=config["training_seed"] + r,
        )
        for n in counts
        for r in replicas
    ]


def point_name(point):
    return f"signal_{point['signal_events']:06d}/replica_{point['replica']:03d}"


def prepare_scan(args):
    root = args.output.resolve()
    plan = points(args)
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / ".prepare.lock"):
        pending = []
        for point in plan:
            destination = root / point_name(point)
            if destination.exists():
                if not args.resume:
                    raise FileExistsError("Prepared scan point exists; use --resume or a new output")
                for scenario in ("signal_injection",):
                    manifest = validate(destination / scenario)
                    if (
                        manifest.get("injection_scan") != point
                        or manifest.get("variant") != args.variant
                    ):
                        raise ValueError("Prepared scan identity changed")
            else:
                pending.append(point)
        if not pending:
            return
        primary, extra, by_source, provenance = read_sources(args, root, args.variant)
        prepared_settings = load_settings(args.config)
        reservoir_rows = max(prepared_settings["injection_scan"]["signal_events"])
        for point in pending:
            spec = replace(
                DatasetSpec(),
                injected_signal_rows=point["signal_events"],
                injection_reservoir_rows=reservoir_rows,
                preparation_seed=point["preparation_seed"],
            )
            roles = build_dataset_roles(
                primary, spec, sic_background_arrays=extra, independent_partition=True
            )
            if args.variant == "deltaR":
                controls.attach_delta_r(roles, by_source)
            destination = root / point_name(point)
            destination.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=".prepare-", dir=destination.parent))
            export_roles(
                roles,
                stage,
                spec,
                provenance,
                variant=args.variant,
                scan=point,
                scenarios=("signal_injection",),
            )
            for scenario in ("signal_injection",):
                validate(stage / scenario)
            os.rename(stage, destination)

        write_json(
            root / "scan_inputs.json",
            {
                "schema": SCAN_SCHEMA,
                "variant": args.variant,
                "points": {
                    str(p.parent.relative_to(root)): file_digest(p)
                    for p in sorted(root.glob("signal_*/replica_*/signal_injection/inputs.json"))
                },
            },
        )




def _candidate_roots(output_root):
    roots = []
    for root in (Path("results").resolve(), output_root.parent / "results", output_root):
        root = root.resolve()
        if root.exists() and root not in roots:
            roots.append(root)
    return roots


def _source_priority(report):
    scan = report.get("contract", {}).get("inputs", {}).get("injection_scan")
    return 0 if scan is None else 1 if scan.get("replica") == 0 else 2


def _matching_prepared_data(report, data_root):
    inputs = report.get("contract", {}).get("inputs", {})
    expected = inputs.get("files", {})
    if not expected:
        return None
    candidates = []
    scan = inputs.get("injection_scan")
    if scan is not None:
        candidates.append(data_root / point_name(scan) / "signal_injection")
    candidates.extend((Path("data/lhco/signal_injection"), data_root.parent / "lhco/signal_injection"))
    seen = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen or not candidate.is_dir():
            continue
        seen.add(candidate)
        manifest_path = candidate / "inputs.json"
        if not manifest_path.is_file():
            continue
        try:
            manifest = json.loads(manifest_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("files") == expected and manifest.get("variant", "default") == inputs.get("variant", "default"):
            return str(candidate)
    return None


def _lacathode_background_candidates(output_root, data_root, variant, current_point):
    candidates = []
    seen = set()
    for root in _candidate_roots(output_root):
        for report_path in root.rglob("result.json"):
            source = report_path.parent.resolve()
            if source in seen or source == current_point.resolve() or current_point.resolve() in source.parents:
                continue
            try:
                report = json.loads(report_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if report.get("method") != "lacathode" or report.get("completed") is not True or report.get("scenario") != "signal_injection":
                continue
            source_variant = report.get("variant", report.get("contract", {}).get("inputs", {}).get("variant", "default"))
            if source_variant != variant:
                continue
            training = source / "training"
            losses = training / f"lacathode_model_val_losses.npy"
            checkpoints = sorted(training.glob("lacathode_model_epoch_*.par"))
            data = _matching_prepared_data(report, data_root)
            if not losses.is_file() or len(checkpoints) < 10 or data is None:
                continue
            seen.add(source)
            candidates.append((
                _source_priority(report),
                str(source),
                data,
                report.get("run_index"),
                report.get("seed"),
            ))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [
        {"result": source, "data": data, "run_index": run_index, "seed": seed}
        for _, source, data, run_index, seed in candidates
    ]


def _ranode_background_candidates(output_root, variant, current_point):
    candidates = []
    seen = set()
    for root in _candidate_roots(output_root):
        for report_path in root.rglob("result.json"):
            source = report_path.parent.resolve()
            if source in seen or source == current_point.resolve() or current_point.resolve() in source.parents:
                continue
            try:
                report = json.loads(report_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if report.get("method") != "ranode" or report.get("completed") is not True or report.get("scenario") != "signal_injection":
                continue
            source_variant = report.get("variant", report.get("contract", {}).get("inputs", {}).get("variant", "default"))
            if source_variant != variant or not (source / "background_complete.json").is_file():
                continue
            seen.add(source)
            candidates.append((_source_priority(report), str(source)))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [source for _, source in candidates]


def _riddle_background_candidates(output_root, point, variant, current_point):
    candidates = []
    seen = set()
    for root in _candidate_roots(output_root):
        for report_path in root.rglob("result.json"):
            source = report_path.parent.resolve()
            if source in seen or source == current_point.resolve() or current_point.resolve() in source.parents:
                continue
            try:
                report = json.loads(report_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            method = str(report.get("method", ""))
            if not (method.startswith("riddle") or method in ("iad", "supervised")) or report.get("completed") is not True:
                continue
            if report.get("scenario") != "signal_injection":
                continue
            source_variant = report.get("variant", report.get("contract", {}).get("inputs", {}).get("variant", "default"))
            if source_variant != variant:
                continue
            scan = report.get("contract", {}).get("inputs", {}).get("injection_scan")
            required = (
                source / "background" / "model.pt",
                source / "background" / "preprocessing.pt",
                source / "background" / "flow_selection.json",
                source / "background" / "mapping_settings.json",
            )
            if not all(path.is_file() for path in required):
                continue
            seen.add(source)
            priority = 0 if scan is None else 1 if scan.get("replica") == 0 else 2
            candidates.append((priority, str(source)))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [source for _, source in candidates]




def _oracle_background_candidates(output_root, point, variant, current_point):
    candidates = []
    seen = set()
    for root in _candidate_roots(output_root):
        for report_path in root.rglob("result.json"):
            source = report_path.parent.resolve()
            if source in seen or source == current_point.resolve() or current_point.resolve() in source.parents:
                continue
            try:
                report = json.loads(report_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if report.get("method") not in ("iad", "supervised") or report.get("completed") is not True or report.get("scenario") != "signal_injection":
                continue
            source_variant = report.get("variant", report.get("contract", {}).get("inputs", {}).get("variant", "default"))
            if source_variant != variant:
                continue
            scan = report.get("contract", {}).get("inputs", {}).get("injection_scan")
            if isinstance(scan, dict) and scan.get("replica") != point["replica"]:
                continue
            required = (
                source / "density" / "background_correction" / "contract.json",
                source / "density" / "background_correction" / "model.pt",
                source / "density" / "background_correction" / "selection.json",
            )
            if not all(path.is_file() for path in required):
                continue
            seen.add(source)
            priority = 0 if scan is None else abs(int(scan.get("signal_events", 0)) - int(point["signal_events"])) + 1
            candidates.append((priority, str(source)))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [source for _, source in candidates]


def _supervised_ensemble_candidates(output_root, point, variant, current_point):
    candidates = []
    seen = set()
    for root in _candidate_roots(output_root):
        for report_path in root.rglob("result.json"):
            source = report_path.parent.resolve()
            if source in seen or source == current_point.resolve() or current_point.resolve() in source.parents:
                continue
            try:
                report = json.loads(report_path.read_text())
            except (OSError, json.JSONDecodeError):
                continue
            if report.get("method") != "supervised" or report.get("completed") is not True or report.get("scenario") != "signal_injection":
                continue
            source_variant = report.get("variant", report.get("contract", {}).get("inputs", {}).get("variant", "default"))
            if source_variant != variant:
                continue
            scan = report.get("contract", {}).get("inputs", {}).get("injection_scan")
            if not isinstance(scan, dict) or scan.get("replica") != point["replica"]:
                continue
            required = (source / "density" / "ensemble_inputs.json", source / "density" / "ensemble_selection.json")
            if not all(path.is_file() for path in required):
                continue
            seen.add(source)
            priority = abs(int(scan.get("signal_events", 0)) - int(point["signal_events"]))
            candidates.append((priority, str(source)))
    candidates.sort(key=lambda item: (item[0], item[1]))
    return [source for _, source in candidates]


def run_scan(args):
    from .cli import run_campaign

    plan = points(args)
    data_root, output_root = args.data.resolve(), args.output.resolve()

    native_requested = any(method in args.methods for method in ("riddle", "iad", "supervised"))
    oracle_requested = any(method in args.methods for method in ("iad", "supervised"))
    for point in plan:
        manifest = validate(
            data_root / point_name(point) / "signal_injection",
            require_event_ids=native_requested,
            require_oracle=oracle_requested,
        )
        if manifest.get("injection_scan") != point:
            raise ValueError(
                "Prepared point differs from configured scan; prepare the matching scan first"
            )
    for point in plan:
        options = copy(args)
        options.data = data_root / point_name(point)
        options.output = output_root / point_name(point)
        options.scenarios = ["signal_injection"]
        options.seeds = [point["training_seed"]]
        manifest = validate(options.data / "signal_injection", require_event_ids=native_requested, require_oracle=oracle_requested)
        variant = manifest.get("variant", "default")
        if any(method in args.methods for method in ("riddle", "iad", "supervised")):
            options.scan_background_reuse_policy = "shared_fixed_background_v1"
            options.riddle_background_reuse_candidates = _riddle_background_candidates(
                output_root, point, variant, options.output
            )
        if "lacathode" in args.methods:
            options.lacathode_scan_background_reuse_policy = "shared_fixed_background_v1"
            options.lacathode_background_reuse_candidates = _lacathode_background_candidates(
                output_root, data_root, variant, options.output
            )
        if "ranode" in args.methods:
            options.ranode_scan_background_reuse_policy = "shared_fixed_background_v1"
            options.ranode_background_reuse_candidates = _ranode_background_candidates(
                output_root, variant, options.output
            )
        if any(method in args.methods for method in ("iad", "supervised")):
            options.oracle_background_reuse_candidates = _oracle_background_candidates(
                output_root, point, variant, options.output
            )
        if "supervised" in args.methods:
            options.supervised_ensemble_reuse_candidates = _supervised_ensemble_candidates(
                output_root, point, variant, options.output
            )
        run_campaign(options)
