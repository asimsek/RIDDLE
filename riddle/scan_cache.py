from copy import deepcopy
import json
from pathlib import Path

from .resume import check_contract
from .storage import file_digest, fingerprint_files, write_json, read_json, load_array, open_npz
from .worker_progress import emit_message

_BACKGROUND_POPULATIONS = {}


def background_population(directory, manifest=None):
    import numpy as np
    from .storage import digest

    directory = Path(directory).resolve()
    if manifest is None:
        manifest = read_json(directory / "inputs.json")
    names = [f"{region}data_{part}.npy" for region in ("inner", "outer") for part in ("train", "val", "test")]
    names += [f"innerdata_extrabkg_{part}.npy" for part in ("train", "val", "test")]
    expected = {name: manifest.get("files", {}).get(name) for name in names}
    expected["event_ids.npz"] = manifest.get("event_ids_sha256")
    if any(value is None for value in expected.values()):
        raise ValueError(f"Missing background population provenance in {directory}")
    stamps = []
    for name, value in expected.items():
        stat = (directory / name).stat()
        stamps.append((name, value, stat.st_dev, stat.st_ino, stat.st_size, stat.st_mtime_ns))
    key = (str(directory), tuple(stamps))
    if key in _BACKGROUND_POPULATIONS:
        return deepcopy(_BACKGROUND_POPULATIONS[key])
    for name, checksum in expected.items():
        if file_digest(directory / name) != checksum:
            raise ValueError(f"Background population artifact changed: {directory / name}")
    result = {}
    with open_npz(directory / "event_ids.npz", allow_pickle=False) as archive:
        for name in names:
            rows = load_array(directory / name, mmap_mode="r", allow_pickle=False)
            ids = archive[name]
            if ids.shape != (len(rows), 2) or ids.dtype != np.uint64:
                raise ValueError(f"Invalid background event identities: {directory / name}")
            selected = ids[rows[:, -1] == 0]
            ordered = selected[np.lexsort((selected[:, 1], selected[:, 0]))]
            if len(ordered) > 1 and np.any(np.all(ordered[1:] == ordered[:-1], axis=1)):
                raise ValueError(f"Duplicate background event identities: {directory / name}")
            result[name] = {"events": len(ordered), "event_ids_sha256": digest(ordered)}
    if len(_BACKGROUND_POPULATIONS) >= 32:
        _BACKGROUND_POPULATIONS.clear()
    _BACKGROUND_POPULATIONS[key] = result
    return deepcopy(result)


def require_background_population(reference, current, context):
    changed = sorted(name for name in set(reference) | set(current) if reference.get(name) != current.get(name))
    if changed:
        raise ValueError(f"Background event population differs from nominal for {context}: {', '.join(changed)}")


def scan_background_policy(report, nominal=None):
    contract = report.get("contract", {})
    inputs = contract.get("inputs", {})
    point = inputs.get("injection_scan") or {}
    frozen = contract.get("scan_background")
    if frozen:
        return {key: frozen.get(key) for key in ("policy", "signal_events", "sideband_roles")}
    baseline = inputs.get("shared_population", {}).get("configuration", {}).get("injected_signal_events", nominal)
    if point.get("schema") == 2 and point.get("signal_events") != baseline:
        return {"policy": "retrain"}
    return None


def aggregation_signature(method, report, *, include_background=True):
    from .evaluation import riddle_score_scope

    contract = report.get("contract", {})
    settings = contract.get("settings", {})
    volatile = {"seed", "scenario", "device", "run_index", "campaign_seed", "injection_scan", "io_workers"}
    payload = {key: value for key, value in contract.items()
               if key not in ("inputs", "settings", "environment", "code", "scan_background")}
    payload["settings"] = {key: value for key, value in settings.items() if key not in volatile}
    payload["fit_identity"] = {key: report[key] for key in ("ensemble_fits", "epochs") if key in report}
    payload["score_scope"] = (riddle_score_scope(report) if method in ("riddle", "iad", "supervised") else
                              "signal_region" if method == "ranode" else
                              report.get("score_scope", report.get("plotting", {}).get("score_scope", "full_region")))
    inputs = contract.get("inputs", {})
    if (inputs.get("injection_scan") or {}).get("schema") == 2:
        payload["population"] = background_contract({"inputs": inputs})["inputs"]
    if include_background:
        payload["scan_background"] = scan_background_policy(report)
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def require_scan_compatibility(method, records):
    reports = [report for _, report in records]
    if len({aggregation_signature(method, report, include_background=False) for report in reports}) > 1:
        raise ValueError(f"Incompatible scientific configurations across injection-scan results for {method}")
    baselines = {value for report in reports for value in (
        report.get("contract", {}).get("inputs", {}).get("shared_population", {}).get("configuration", {}).get("injected_signal_events"),
        (report.get("contract", {}).get("scan_background") or {}).get("signal_events")) if value is not None}
    if len(baselines) > 1:
        raise ValueError(f"Incompatible nominal injection strengths for {method}")
    baseline = next(iter(baselines), None)
    policies, sources, nominal_reports = set(), {}, {}
    for root, report in records:
        policy = scan_background_policy(report, baseline)
        if policy is not None:
            policies.add(json.dumps(policy, sort_keys=True))
        seed = report.get("seed")
        frozen = report.get("contract", {}).get("scan_background")
        if frozen:
            identity = {key: frozen.get(key) for key in ("source_result", "artifacts_sha256")}
            if seed in sources and sources[seed] != identity:
                raise ValueError(f"Incompatible nominal background sources for {method}, seed {seed}")
            sources[seed] = identity
        elif (report.get("contract", {}).get("inputs", {}).get("injection_scan") or {}).get("signal_events") == baseline:
            nominal_reports[seed] = (Path(root).resolve(), report)
    if len(policies) > 1:
        raise ValueError(f"Incompatible background policies across injection-scan results for {method}")
    for seed, source in sources.items():
        if seed not in nominal_reports:
            continue
        root, report = nominal_reports[seed]
        producer = Path((report.get("scan_reuse") or {}).get("source_result", root)).resolve()
        if (Path(source["source_result"]).resolve() != producer
                or any(report.get("artifacts_sha256", {}).get(name) != checksum
                       for name, checksum in (source["artifacts_sha256"] or {}).items())):
            raise ValueError(f"Nominal background provenance disagrees with the plotted nominal result for {method}, seed {seed}")


def comparison_contract(contract):
    value = deepcopy(contract)
    value.get("inputs", {}).pop("injection_scan", None)
    value.get("settings", {}).pop("injection_scan", None)
    return value


def reference_content(report):
    value = deepcopy(report)
    value.pop("scan_reuse", None)
    value.pop("resume_history", None)
    value["contract"] = comparison_contract(value["contract"])
    return value


def resolve_reference(root, report):
    reuse = report.get("scan_reuse")
    if reuse is None:
        return Path(root), report
    if reuse.get("schema") != 1 or report.get("completed") is not True:
        raise ValueError("Invalid completed scan reference")
    source = Path(reuse["source_result"])
    if not source.is_absolute() or source.resolve() == Path(root).resolve():
        raise ValueError("Invalid scan reference source")
    path = source / "result.json"
    original = read_json(path)
    if original.get("completed") is not True or original.get("scan_reuse") is not None:
        raise ValueError("Scan references require an original completed result")
    if reference_content(report) != reference_content(original):
        raise ValueError("Scan reference differs from its original producer")
    point = report["contract"]["inputs"].get("injection_scan", {})
    if point.get("schema") != 2 or point.get("signal_events") != report["contract"]["inputs"].get("preparation", {}).get("injected_signal_rows"):
        raise ValueError("Invalid scan reference injection strength")
    return source, report


def runtime_compatible(previous, current, *, allow_code_change=False, allow_device_change=False):
    if current is None:
        return False
    fields = ("environment", "code")
    old = {key: previous.get(key) for key in fields}
    new = {key: current.get(key) for key in fields}
    for target, source in ((old, previous), (new, current)):
        target["settings"] = {key: source.get("settings", {}).get(key) for key in ("seed", "device")}
    try:
        check_contract(old, new, allow_code_change=allow_code_change,
                       allow_device_change=allow_device_change, reuse_completed=True)
    except ValueError:
        return False
    return True


def background_contract(contract):
    value = comparison_contract(contract)
    value.pop("scan_background", None)
    inputs = value["inputs"]
    inputs.pop("event_ids_sha256", None)
    inputs.get("preparation", {}).pop("injected_signal_rows", None)
    for region in ("innerdata", "outerdata"):
        for partition in ("train", "val", "test"):
            inputs.get("files", {}).pop(f"{region}_{partition}.npy", None)
    for key in ("evaluation_events", "evaluation_event_ids_sha256"):
        inputs.get("shared_population", {}).pop(key, None)
    for key in ("signal", "signal_to_background", "nominal_significance"):
        inputs.get("uncut_signal_region", {}).pop(key, None)
    return value


def configure_background_reuse(args, contract):
    point = contract["inputs"].get("injection_scan") or {}
    baseline = getattr(args, "scan_background_baseline", None)
    if (point.get("schema") != 2 or getattr(args, "scan_background_mode", "retrain") != "reuse"
            or point.get("signal_events") == baseline):
        return
    if type(baseline) is not int or baseline <= 0:
        raise ValueError("Missing nominal injection count for background reuse")
    candidates = getattr(args, "scan_result_candidates", ()) or ()
    last_error = "no completed nominal result found"
    for candidate in candidates:
        source = Path(candidate).resolve()
        try:
            report = read_json(source / "result.json")
            source, report = resolve_reference(source, report)
            if getattr(args, "lacathode_replica", False):
                source = source / "runs" / f"run_{args.run_index:03d}"
                report = read_json(source / "result.json")
            previous = report["contract"]
            if (report.get("completed") is not True or previous.get("method") != contract["method"]
                    or report.get("seed") != args.seed or report.get("scenario") != "signal_injection"
                    or previous.get("scan_background") is not None
                    or previous["inputs"]["preparation"]["injected_signal_rows"] != baseline):
                continue
            check_contract(background_contract(previous), background_contract(contract), reuse_completed=True,
                           allow_code_change=getattr(args, "resume_across_code_change", False),
                           allow_device_change=getattr(args, "resume_across_device_change", False))
            nominal_data = Path(args.scan_background_data)
            nominal_inputs = read_json(nominal_data / "inputs.json")
            if comparison_contract({"inputs": previous["inputs"]}) != comparison_contract({"inputs": nominal_inputs}):
                raise ValueError("Nominal prepared data differ from the background model's original inputs")
            require_background_population(background_population(nominal_data, nominal_inputs),
                                          background_population(args.data, contract["inputs"]), args.data)
        except (OSError, ValueError, KeyError) as error:
            last_error = str(error)
            continue
        contract["scan_background"] = {
            "policy": "nominal_background_v1", "signal_events": baseline,
            "sideband_roles": "nominal_event_id_intersections_v1",
            "source_result": str(source),
            "artifacts_sha256": {name: checksum for name, checksum in report.get("artifacts_sha256", {}).items()
                                 if name.startswith(("background/", "density/background_correction/", "upstream_runs/background/"))
                                 or (name.startswith("training/") and "lacathode_model" in name)
                                 or name == "background_complete.json"},
        }
        args.riddle_background_reuse_candidates = [str(source)]
        args.oracle_background_reuse_candidates = [str(source)]
        args.scan_background_reuse_policy = "shared_fixed_background_v1"
        args.background_reuse_candidate = [str(source)]
        args.lacathode_scan_background_reuse_policy = "shared_fixed_background_v1"
        args.lacathode_background_reuse_candidates = [dict(
            result=str(source), data=args.scan_background_data,
            seed=args.seed, run_index=getattr(args, "run_index", 0))]
        emit_message(f"Injection scan: freeze seed {args.seed} background at {baseline} injected signals from {source}", kind="PASS", level=0)
        return
    raise ValueError(f"No compatible nominal {contract['method']} background for seed {args.seed}: {last_error}. "
                     "Complete the nominal point first, supply its --reuse-results directory, or set "
                     "injection_scan.background_mode: retrain in a new output directory")


def reuse_completed(output, contract, candidates=(), *, resume=False, allow_code_change=False, allow_device_change=False,
                    io_workers=1):
    point = contract.get("inputs", {}).get("injection_scan") or {}
    if point.get("schema") != 2:
        return False
    output = Path(output)
    destination = output / "result.json"
    existing = read_json(destination) if destination.exists() else None
    if existing is not None and existing.get("scan_reuse") is None:
        return False
    if existing is not None:
        if not resume:
            raise FileExistsError("Scan reference exists; use --resume")
        source, _ = resolve_reference(output, existing)
        candidates = [source]
    elif any(path.is_file() for directory in ("background", "density", "training", "fits", "upstream_runs")
             for path in (output / directory).rglob("*")):
        return False
    for candidate in candidates:
        source = Path(candidate).resolve()
        if source == output.resolve():
            continue
        try:
            original = read_json(source / "result.json")
            source, original = resolve_reference(source, original)
            if original.get("completed") is not True:
                continue
            changes = check_contract(comparison_contract(original["contract"]), comparison_contract(contract),
                                     allow_code_change=allow_code_change, allow_device_change=allow_device_change,
                                     reuse_completed=True)
        except (OSError, ValueError, KeyError):
            if existing is not None:
                raise
            continue
        artifacts = original.get("artifacts_sha256", {})
        if not artifacts or any(Path(name).is_absolute() or ".." in Path(name).parts for name in artifacts):
            raise ValueError("Invalid reusable result artifact inventory")
        paths = [source / name for name in artifacts]
        if any(not path.resolve().is_relative_to(source) or not path.is_file() for path in paths):
            raise ValueError("Missing or invalid reusable scan artifact")
        if fingerprint_files(source, paths, io_workers) != artifacts:
            raise ValueError("Reusable scan artifact checksum mismatch")
        from .production import validate_result_scores
        validate_result_scores(source, contract["method"])
        if contract["method"] == "ranode":
            from external.ranode_utils.runner import validate_result_normalization
            validate_result_normalization(source, original)
        report = deepcopy(original)
        report["contract"]["inputs"] = deepcopy(contract["inputs"])
        report["scan_reuse"] = {
            "schema": 1,
            "source_result": str(source),
            "source_result_sha256": file_digest(source / "result.json"),
            "requested_contract": contract,
            "permitted_changes": changes,
        }
        write_json(destination, report)
        emit_message(f"Reuse completed {contract['method']} seed {original['seed']} from {source}; no training or artifact copies", kind="PASS", level=0)
        return True
    if existing is not None:
        raise ValueError("Saved scan reference no longer matches this request")
    return False
