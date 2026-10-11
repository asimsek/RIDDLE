from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from copy import deepcopy
import errno
import hashlib
import json
import multiprocessing
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import time

import numpy as np

from .mass_spectrum import CUTS, MASS_CLOSURE_PROTOCOL, background_closure_fit, event_keys, match_ids, mass_edges
from .storage import atomic_write, copy_file, digest, environment, file_digest, fingerprint_files, locked, read_json, save_array, save_npz, seed_start, verify_artifacts, write_json
from .worker_progress import emit_message


PROTOCOL = "frozen_native_full_mass_v1"
DERIVED_DIRECTORY = "full_mass"
REFERENCE_SEEDS = {"contexts": 96001, "background": 96002, "split": 96003, "ties": 96004}


def _export_process(args, report):
    if os.name == "posix":
        os.setsid()
    import torch
    torch.set_num_threads(args.torch_threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    export_full_mass(args, report)


def score_sidebands(args, report=None):
    if not getattr(args, "sideband_scoring", True):
        return
    report = read_json(Path(args.output) / "result.json") if report is None else report
    scientific_settings = report["contract"]["settings"]["riddle"]
    if scientific_settings.get("core") != "stein_witness" or scientific_settings.get("input_space") == "physical":
        if getattr(args, "score_sidebands_only", False):
            raise ValueError("Frozen sideband backfill requires the latent Stein-witness core")
        emit_message("Sideband export requires the latent Stein-witness core; legacy scoring retained", kind="WARNING")
        return
    if str(args.device).startswith("cuda"):
        import torch
        torch.cuda.empty_cache()
    previous = os.environ.get("CUBLAS_WORKSPACE_CONFIG")
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    process = multiprocessing.get_context("spawn").Process(target=_export_process, args=(args, report))
    try:
        process.start()
    finally:
        if previous is None:
            os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
        else:
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = previous
    try:
        while process.is_alive():
            process.join(timeout=10)
        if process.exitcode != 0:
            raise RuntimeError("Frozen sideband inference failed; completed training is preserved. Resume with --score-sidebands-only")
    finally:
        if process.is_alive():
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    process.terminate()
            else:
                process.terminate()
            process.join(timeout=5)
            if os.name == "posix":
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif process.is_alive():
                process.kill()
            process.join(timeout=5)


def identity_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def threshold_rules(scores):
    scores = np.asarray(scores, dtype=float)
    if scores.ndim != 1 or len(scores) < 1024 or not np.isfinite(scores).all():
        raise ValueError("Invalid independent background threshold reference")
    ordered = np.sort(scores, kind="stable")
    rules = []
    for alpha in CUTS:
        target = alpha * len(scores)
        threshold = float(ordered[-int(np.ceil(target))])
        above = int((scores > threshold).sum())
        tied = int((scores == threshold).sum())
        rules.append(dict(background_acceptance=alpha, threshold=threshold,
                          tie_probability=float(np.clip((target - above) / tied, 0, 1)),
                          reference_events=len(scores), reference_above=above, reference_tied=tied))
    return rules


def tie_uniform(ids, seed):
    event_keys(ids)
    ids = np.asarray(ids, dtype=np.uint64)
    with np.errstate(over="ignore"):
        values = ids[:, 0] * np.uint64(0x9E3779B97F4A7C15) + ids[:, 1] + np.uint64(seed)
        values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
        values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
        values = values ^ (values >> np.uint64(31))
    return (values >> np.uint64(11)).astype(np.float64) / float(2**53)


def apply_rules(scores, mask, ids, rules, seed):
    scores, mask = np.asarray(scores), np.asarray(mask, dtype=bool)
    if scores.shape != mask.shape or len(ids) != len(scores):
        raise ValueError("Misaligned frozen score-cut inputs")
    if not np.isfinite(scores[mask]).all():
        raise FloatingPointError("Nonfinite accepted full-mass scores")
    ties = tie_uniform(ids, seed)
    return np.asarray([mask & ((scores > rule["threshold"]) |
                       ((scores == rule["threshold"]) & (ties < rule["tie_probability"])))
                       for rule in rules], dtype=bool)


def _verified(root, report, names, workers):
    artifacts = report.get("artifacts_sha256", {})
    selected = {}
    for name in sorted(set(map(str, names))):
        path = Path(name)
        if path.is_absolute() or ".." in path.parts or name not in artifacts:
            raise ValueError(f"Frozen artifact is missing from the result provenance: {name}")
        selected[name] = artifacts[name]
    verify_artifacts(root, selected, "Verify frozen scoring artifacts", workers=workers)
    return selected


def frozen_contract(root, report, workers):
    from .production import require_complete_ensemble
    from .stein import _checkpoint_weights
    from .stein_scoring import SCORING_PROTOCOL, _background_identity, _scoring

    root = Path(root)
    names = ["density/ensemble_selection.json", "density/ensemble_inputs.json",
             "density/stein_scoring_reference.json", "density/stein_scoring_reference.npz",
             "signal_region_scores.npz", "background/flow_selection.json"]
    _verified(root, report, names, workers)
    ensemble = read_json(root / "density/ensemble_selection.json")
    require_complete_ensemble(ensemble)
    ensemble_inputs = read_json(root / "density/ensemble_inputs.json")
    settings = deepcopy(ensemble_inputs["settings"])
    if settings.get("core") != "stein_witness" or settings.get("input_space") == "physical":
        raise ValueError("Frozen full-mass scoring currently requires the native latent Stein-witness core")
    cfg = _scoring(settings)
    mode = cfg["mode"]
    if "density/score_selection.json" in report["artifacts_sha256"]:
        names.append("density/score_selection.json")
        _verified(root, report, [names[-1]], workers)
        mode = read_json(root / names[-1])["selected_mode"]
    with np.load(root / "signal_region_scores.npz", allow_pickle=False) as archive:
        if "selected_scoring_mode" in archive and str(archive["selected_scoring_mode"].item()) != mode:
            raise ValueError("The frozen selector and saved SR scoring modes disagree")
        expected_kind = "stein_" + mode
        if not str(archive["score_kind"].item()).startswith(expected_kind):
            raise ValueError("The saved SR score kind differs from the frozen scoring mode")
    settings["stein"]["scoring"]["mode"] = mode
    if (root / "background/mapping_settings.json").is_file():
        mapping_kind = "enhanced"
        names.extend("background/" + name for name in
                     ("mapping_settings.json", "model.pt", "preprocessing.pt", "background_settings.json"))
    else:
        mapping_kind = "legacy"
        epoch = read_json(root / "background/flow_selection.json")["inference_mapping_epoch"]
        names.extend([f"background/riddle_model_epoch_{epoch}.par", "background/background_settings.json"])
    selected_directories = {member["directory"] for member in ensemble["members"]}
    members = [member for member in ensemble.get("accepted_members", ensemble["members"])
               if member["directory"] in selected_directories]
    if len(members) != len(selected_directories):
        raise ValueError("Invalid frozen fit aggregation inventory")
    reference = read_json(root / "density/stein_scoring_reference.json")
    if reference.get("scoring_protocol") != SCORING_PROTOCOL:
        raise ValueError("Unsupported frozen Stein calibration protocol")
    descriptions = []
    for member in members:
        prefix = "density/" + member["directory"] + "/"
        path = prefix + "residual_training_inputs.json"
        _verified(root, report, [path], workers)
        inputs = read_json(root / path)
        if inputs.get("core") != "stein_witness" or _background_identity(inputs) != reference["background_model_hash"]:
            raise ValueError("Selected fits do not share the frozen background reference")
        names.extend([path, prefix + "residual_selection.json", prefix + "stein_scoring_calibration.json",
                      prefix + "stein_scoring_calibration.npz"])
        names.extend(prefix + f"residual_epoch_{int(epoch)}.pt" for epoch in member["epochs"])
        if inputs.get("background_correction"):
            names.append(prefix + "background_correction.pt")
        descriptions.append(dict(fit_index=member["fit_index"], directory=member["directory"],
                                 epochs=member["epochs"], background=inputs.get("background_correction")))
    artifacts = _verified(root, report, names, workers)
    for member, description in zip(members, descriptions):
        prefix = "density/" + member["directory"] + "/"
        metadata = read_json(root / (prefix + "stein_scoring_calibration.json"))
        saved = metadata["contract"]
        expected_checkpoints = {str(int(epoch)): artifacts[prefix + f"residual_epoch_{int(epoch)}.pt"]
                                for epoch in member["epochs"]}
        if (saved.get("epochs") != member["epochs"] or saved.get("checkpoint_sha256") != expected_checkpoints
                or saved.get("reference_A_sha256") != reference["reference_A_sha256"]
                or saved.get("background_model_hash") != reference["background_model_hash"]
                or saved.get("mapping_hash") != reference["mapping_hash"]
                or saved.get("scoring_protocol") != SCORING_PROTOCOL
                or metadata.get("calibration_file_sha256") != artifacts[prefix + "stein_scoring_calibration.npz"]):
            raise ValueError("Frozen member calibration provenance differs from the selected ensemble")
        description["checkpoint_weights"] = _checkpoint_weights(root / "density" / member["directory"], member["epochs"]).tolist()
    return dict(scoring_protocol=SCORING_PROTOCOL, settings=settings, mode=mode,
                mapping_kind=mapping_kind, members=descriptions, artifacts_sha256=artifacts,
                reference=reference, aggregation="saved_accepted_member_order_arithmetic_mean"), ensemble


def _save_array(path, array):
    atomic_write(path, lambda temporary: save_array(temporary, array))


def _inference_worker_init(threads):
    import torch
    torch.set_num_threads(threads)
    torch.set_num_interop_threads(1)
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def _member_prediction(job):
    from .enhancements import deterministic_spline_sums
    from .stein_scoring import member_scores

    root, work = Path(job["root"]), Path(job["work"])
    inputs = np.load(work / "latents.npy", mmap_mode="c", allow_pickle=False)
    qscore = np.load(work / "qscore.npy", mmap_mode="c", allow_pickle=False) if job["derivatives"] else None
    member = job["member"]
    with np.load(root / "density" / member["directory"] / "stein_scoring_calibration.npz", allow_pickle=False) as archive:
        calibration = {name: archive[name] for name in archive.files}
    with deterministic_spline_sums():
        values = member_scores(root / "density" / member["directory"], member["epochs"], inputs,
                               job["device"], mode=job["mode"], scoring_root=root / "density",
                               scoring_settings=job["settings"], background_scores=qscore, calibration_arrays=calibration,
                               strict_batch_size=True)
    path = work / f"fit_{int(member['fit_index']):03d}.npy"
    _save_array(path, values)
    write_json(path.with_suffix(".json"), dict(identity=job["identity"], sha256=file_digest(path),
                                             fit_index=member["fit_index"]))
    return member["fit_index"]


def _member_cache(work, member, identity, events):
    path = work / f"fit_{int(member['fit_index']):03d}.npy"
    receipt = path.with_suffix(".json")
    if not path.exists() or not receipt.exists():
        return False
    metadata = read_json(receipt)
    if metadata.get("identity") != identity or metadata.get("sha256") != file_digest(path):
        raise ValueError("Frozen fit-score cache failed verification")
    values = np.load(path, mmap_mode="r", allow_pickle=False)
    if values.shape != (events,) or values.dtype != np.float64 or not np.isfinite(values).all():
        raise ValueError("Invalid frozen fit-score cache")
    return True


def _predict_members(root, work, frozen, identity, args):
    events = len(np.load(work / "latents.npy", mmap_mode="r", allow_pickle=False))
    members = frozen["members"]
    with ThreadPoolExecutor(max_workers=args.io_workers) as pool:
        available = list(pool.map(lambda member: _member_cache(work, member, identity, events), members))
    jobs = [dict(root=str(root), work=str(work), member=member, identity=identity,
                 settings=frozen["settings"], mode=frozen["mode"], device=args.device,
                 derivatives=frozen["mode"] in ("local_qnorm", "hybrid", "hybrid_gated", "sic_preserving", "tail_focus"),
                 torch_threads=args.torch_threads) for member, exists in zip(members, available) if not exists]
    workers = min(args.workers, len(jobs))
    emit_message(f"Frozen sideband inference: {sum(available)}/{len(members)} fits cached; workers={max(1, workers)}")
    if jobs:
        started = time.monotonic()
        with ProcessPoolExecutor(max_workers=max(1, workers), mp_context=multiprocessing.get_context("spawn"),
                                 initializer=_inference_worker_init, initargs=(args.torch_threads,)) as pool:
            queued = iter(jobs)
            pending = {pool.submit(_member_prediction, next(queued)) for _ in range(workers)}
            completed = sum(available)
            while pending:
                done, pending = wait(pending, timeout=10, return_when=FIRST_COMPLETED)
                fits = [future.result() for future in done]
                for fit in fits:
                    completed += 1
                    emit_message(f"Frozen scoring: fit {fit:03d}; {completed}/{len(members)} complete")
                    job = next(queued, None)
                    if job is not None:
                        pending.add(pool.submit(_member_prediction, job))
                if not done:
                    emit_message(f"Frozen scoring: {completed}/{len(members)} fits; elapsed={time.monotonic() - started:.0f}s", level=2)
    combined = None
    for member in members:
        values = np.load(work / f"fit_{int(member['fit_index']):03d}.npy", mmap_mode="r", allow_pickle=False)
        combined = values.copy() if combined is None else combined + values
    return combined / len(members)


def diagnostic_validation(record, assessment_scores, rules):
    from scipy.stats import binomtest

    mass, labels = record["mass"], record["labels"]
    regions = {"signal_region": record["is_signal_region"], "sidebands": ~record["is_signal_region"]}
    edges = mass_edges(mass, split_sr=True)
    widths = np.diff(edges)
    result = dict(schema=1, thresholds_use_evaluation_labels=False,
                  closure_uses_simulation_background_labels=True, cuts=[], physics_certified=False,
                  calibration_domain="SR; sidebands use the frozen edge-clamped SR score and support calibration",
                  fit_model="diagnostic exp(a+b*log(m)+c*log(m)^2), integrated over mass bins",
                  fit_is_publication_search_model=False)
    if not np.any(labels == 0):
        result.update(status="inconclusive", reason="No labelled held-out background for closure diagnostics")
        return update_mass_closure(result)
    assessment_ids = np.column_stack((np.full(len(assessment_scores), 2**64 - 1, dtype=np.uint64),
                                      np.arange(len(assessment_scores), dtype=np.uint64)))
    assessment_masks = apply_rules(assessment_scores, np.ones(len(assessment_scores), bool),
                                   assessment_ids, rules, REFERENCE_SEEDS["ties"])
    for i, rule in enumerate(rules):
        alpha = rule["background_acceptance"]
        independent_count = int(assessment_masks[i].sum())
        independent_test = binomtest(independent_count, len(assessment_scores), alpha)
        independent_ci = independent_test.proportion_ci(confidence_level=0.99)
        row = dict(background_acceptance=alpha, independent_reference_assessment=dict(
            events=len(assessment_scores), selected=independent_count,
            acceptance=independent_count / len(assessment_scores), p_value=float(independent_test.pvalue),
            confidence_interval_99=[float(independent_ci.low), float(independent_ci.high)]), regions={})
        for name, region in regions.items():
            bg = region & (labels == 0)
            sig = region & (labels == 1)
            selected = record["cut_masks"][i]
            trials, passed = int(bg.sum()), int((bg & selected).sum())
            test = binomtest(passed, trials, alpha) if trials else None
            row["regions"][name] = dict(background_events=trials, selected_background=passed,
                background_acceptance=passed / trials if trials else None,
                nominal_acceptance_p_value=float(test.pvalue) if test else None,
                signal_events=int(sig.sum()), selected_signal=int((sig & selected).sum()),
                signal_retention=float((sig & selected).sum() / sig.sum()) if sig.any() else None)
        bg = labels == 0
        total = np.histogram(mass[bg], edges)[0]
        chosen = np.histogram(mass[bg & record["cut_masks"][i]], edges)[0]
        row["mass_acceptance"] = [dict(low_TeV=float(edges[j]), high_TeV=float(edges[j + 1]),
                                     background=int(total[j]), selected=int(chosen[j]),
                                     acceptance=float(chosen[j] / total[j]) if total[j] else None)
                                   for j in range(len(widths))]
        left = (mass >= 3.2) & (mass < 3.3) & bg
        right = (mass > 3.7) & (mass <= 3.8) & bg
        inner_left = (mass > 3.3) & (mass <= 3.4) & bg
        inner_right = (mass >= 3.6) & (mass < 3.7) & bg
        from scipy.stats import fisher_exact
        row["SR_boundaries"] = []
        for name, outer, inner in (("lower", left, inner_left), ("upper", right, inner_right)):
            a, b = int((outer & record["cut_masks"][i]).sum()), int((inner & record["cut_masks"][i]).sum())
            if min(outer.sum(), inner.sum()) >= 50:
                p = float(fisher_exact([[a, int(outer.sum()) - a], [b, int(inner.sum()) - b]]).pvalue)
                status = "failed" if p < 0.001 else "passed"
            else:
                p, status = None, "inconclusive"
            row["SR_boundaries"].append(dict(boundary=name, status=status, p_value=p,
                                            outer_events=int(outer.sum()), outer_selected=a,
                                            inner_events=int(inner.sum()), inner_selected=b))
        result["cuts"].append(row)
    return update_mass_closure(result)


def update_mass_closure(validation):
    result = deepcopy(validation)
    result.update(schema=2, diagnostic_protocol=MASS_CLOSURE_PROTOCOL,
                  fit_model="diagnostic exp(a+b*log(m)+c*log(m)^2), integrated over mass bins",
                  fit_is_publication_search_model=False, physics_certified=False,
                  fit_range_policy="Fixed local SR window; merge narrow bins next to SR boundaries",
                  p_value_method="Asymptotic Poisson deviance; exploratory diagnostic")
    cuts = result.get("cuts", [])
    if not cuts:
        result.update(status="inconclusive", baseline_status="inconclusive",
                      mass_fit_status="inconclusive", boundary_status="inconclusive")
        return result
    first = cuts[0]["mass_acceptance"]
    if not first or any(row["high_TeV"] != first[i + 1]["low_TeV"] for i, row in enumerate(first[:-1])):
        raise ValueError("Invalid saved mass-bin edges")
    edges = np.array([row["low_TeV"] for row in first] + [first[-1]["high_TeV"]])
    total = np.array([row["background"] for row in first])
    baseline = background_closure_fit(edges, total)
    result["uncut_background_fit"] = baseline
    result["baseline_status"] = baseline["status"]
    for cut in cuts:
        bins = cut["mass_acceptance"]
        if (len(bins) != len(first) or any(row["low_TeV"] != first[i]["low_TeV"]
                or row["high_TeV"] != first[i]["high_TeV"] or row["background"] != total[i]
                for i, row in enumerate(bins))
                or any(row["selected"] < 0 or row["selected"] > row["background"] for row in bins)):
            raise ValueError("Inconsistent saved mass-closure histograms")
        fit = background_closure_fit(edges, [row["selected"] for row in bins])
        if baseline["status"] != "passed":
            fit.update(conditional_status=fit["status"], status="inconclusive",
                       reason="Uncut background does not validate the diagnostic model")
        cut["background_fit_closure"] = fit

    def combined(statuses):
        return "failed" if "failed" in statuses else "inconclusive" if not statuses or "inconclusive" in statuses else "passed"

    result["mass_fit_status"] = combined([cut["background_fit_closure"]["status"] for cut in cuts])
    result["boundary_status"] = combined([boundary["status"] for cut in cuts for boundary in cut["SR_boundaries"]])
    result["status"] = combined([result["mass_fit_status"], result["boundary_status"]])
    result["failed_checks"] = [name for name in ("mass_fit", "boundary") if result[name + "_status"] == "failed"]
    if baseline["status"] != "passed":
        result["reason"] = "Uncut baseline model is not validated; selected mass-fit closure is inconclusive"
    else:
        result.pop("reason", None)
    return result


def emit_closure_status(validation):
    emit_message(f"Full-mass diagnostic: {validation['status']}; uncut model {validation['baseline_status']}",
                 kind="PASS" if validation["status"] == "passed" else "WARNING")
    emit_message(f"Mass-fit check {validation['mass_fit_status']}; SR-boundary check {validation['boundary_status']}",
                 kind="PASS" if validation["status"] == "passed" else "WARNING")
    for cut in validation.get("cuts", []):
        fit = cut["background_fit_closure"]
        if fit["status"] == "failed":
            checks = ", ".join(fit["failed_checks"])
            emit_message(f"Mass-fit failure at epsilon_B={100 * cut['background_acceptance']:g}%: {checks}", kind="WARNING")


def _diagnostic_json_digest(value):
    return hashlib.sha256((json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()).hexdigest()


def _diagnostic_code():
    return {path.name: file_digest(path) for path in (Path(__file__), Path(__file__).with_name("mass_spectrum.py"))}


def _recover_mass_diagnostic(root):
    root = Path(root)
    journal = root / ".resume/full_mass_diagnostics/pending.json"
    if not journal.is_file():
        return
    state = read_json(journal)
    if state.get("protocol") != MASS_CLOSURE_PROTOCOL or set(state.get("updates", {})) != {"validation.json", "manifest.json"}:
        raise ValueError("Invalid pending full-mass diagnostic update")
    export = root / DERIVED_DIRECTORY
    for name, update in state["updates"].items():
        if (update["new_sha256"] != _diagnostic_json_digest(update["value"])
                or file_digest(export / name) not in (update["old_sha256"], update["new_sha256"])):
            raise ValueError("Full-mass diagnostic metadata changed during publication")
    if state["updates"]["manifest.json"]["value"]["artifacts_sha256"]["validation.json"] != state["updates"]["validation.json"]["new_sha256"]:
        raise ValueError("Pending full-mass diagnostic checksum is inconsistent")
    for name in ("validation.json", "manifest.json"):
        update = state["updates"][name]
        if file_digest(export / name) != update["new_sha256"]:
            write_json(export / name, update["value"])
    journal.unlink()


def _validate_full_mass_manifest(manifest, report):
    if not report.get("completed") or manifest.get("protocol") != PROTOCOL or not manifest.get("completed"):
        raise ValueError("Full-mass score export is incomplete or incompatible")
    expected = report["contract"]["inputs"]
    if (manifest["identity"]["inputs"]["files"] != expected["files"]
            or manifest["identity"]["inputs"]["event_ids_sha256"] != expected.get("event_ids_sha256")
            or manifest["identity"]["seed"] != report["seed"]):
        raise ValueError("Full-mass export uses a different prepared population or seed")
    for name, checksum in manifest["identity"]["frozen"]["artifacts_sha256"].items():
        if report["artifacts_sha256"].get(name) != checksum:
            raise ValueError("Full-mass export uses different frozen training artifacts")


def refresh_mass_diagnostic(root):
    root = Path(root)
    with locked(root / ".resume/command.lock"), locked(root / ".resume/full_mass_diagnostics.lock"):
        rescore_state = root / ".resume/rescore_in_place.json"
        if rescore_state.is_file() and read_json(rescore_state).get("phase") != "completed":
            raise ValueError("Finish the pending in-place re-scoring before refreshing diagnostics")
        _recover_mass_diagnostic(root)
        export = root / DERIVED_DIRECTORY
        manifest = read_json(export / "manifest.json")
        _validate_full_mass_manifest(manifest, read_json(root / "result.json"))
        original_sha256 = file_digest(export / "validation.json")
        if manifest["artifacts_sha256"].get("validation.json") != original_sha256:
            raise ValueError("Saved full-mass diagnostic checksum mismatch")
        original = read_json(export / "validation.json")
        validation = update_mass_closure(original)
        code = _diagnostic_code()
        if (validation == original and manifest.get("diagnostic_protocol") == MASS_CLOSURE_PROTOCOL
                and manifest.get("diagnostic_code_sha256") == code):
            return validation
        updated = deepcopy(manifest)
        updated.update(validation_status=validation["status"], diagnostic_protocol=MASS_CLOSURE_PROTOCOL,
                       diagnostic_code_sha256=code,
                       diagnostic_updated_utc=datetime.now(timezone.utc).isoformat(timespec="seconds"),
                       diagnostic_scores_recomputed=False)
        updated["artifacts_sha256"]["validation.json"] = _diagnostic_json_digest(validation)
        metadata = root / ".resume/full_mass_diagnostics"
        backup = metadata / identity_digest(dict(validation=original_sha256, manifest=file_digest(export / "manifest.json")))
        for name in ("validation.json", "manifest.json"):
            if not (backup / name).exists():
                atomic_write(backup / name, lambda path, name=name: copy_file(export / name, path))
        updates = {name: dict(old_sha256=file_digest(export / name), new_sha256=_diagnostic_json_digest(value), value=value)
                   for name, value in (("validation.json", validation), ("manifest.json", updated))}
        write_json(metadata / "pending.json", dict(protocol=MASS_CLOSURE_PROTOCOL, updates=updates))
        _recover_mass_diagnostic(root)
        return validation


def refresh_mass_diagnostics(args):
    roots = list(dict.fromkeys(path.resolve() for path in args.results))
    jobs = []
    for root in roots:
        if not root.is_dir():
            raise ValueError(f"Results folder is missing: {root}")
        for method in args.methods:
            for scenario in args.scenarios:
                for path in sorted((root / method / scenario).glob("seed_*/full_mass/manifest.json")):
                    seed = int(path.parent.parent.name.removeprefix("seed_"))
                    if args.seeds is None or seed in args.seeds:
                        jobs.append(path.parent.parent)
    jobs = list(dict.fromkeys(jobs))
    if not jobs:
        raise ValueError("No completed full-mass exports match the requested methods, scenarios and seeds")
    from threadpoolctl import threadpool_limits
    from .progress import colored_status, training_progress
    from contextvars import copy_context

    started = time.monotonic()
    statuses = []
    with threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=min(args.io_workers, len(jobs))) as pool:
        futures = {pool.submit(copy_context().run, refresh_mass_diagnostic, root): root for root in jobs}
        with training_progress("Refresh full-mass diagnostics", len(jobs), unit="result") as progress:
            from concurrent.futures import as_completed
            for index, future in enumerate(as_completed(futures), 1):
                root = futures[future]
                validation = future.result()
                statuses.append(validation["status"])
                colored_status(f"{root.parents[2].name}/{root.parent.parent.name} {root.parent.name} {root.name}: {validation['status']}",
                               kind="PASS" if validation["status"] == "passed" else "WARNING",
                               level=2 if validation["status"] == "passed" else 0)
                if validation["status"] != "passed":
                    colored_status(f"Uncut model {validation['baseline_status']}; mass fit {validation['mass_fit_status']}; boundaries {validation['boundary_status']}", kind="WARNING")
                progress.update(index)
    counts = ", ".join(f"{statuses.count(status)} {status}" for status in ("passed", "failed", "inconclusive") if status in statuses)
    emit_message(f"Refreshed {len(jobs)} diagnostics in {time.monotonic() - started:.1f}s; {counts}",
                 kind="PASS" if all(status == "passed" for status in statuses) else "WARNING")
    emit_message("Training, calibration, scores and thresholds unchanged", kind="PASS")


def load_full_mass(root, report, *, workers=1):
    root = Path(root)
    path = root / DERIVED_DIRECTORY / "manifest.json"
    if not path.is_file():
        return None
    with locked(root / ".resume/full_mass_diagnostics.lock"):
        _recover_mass_diagnostic(root)
        manifest = read_json(path)
        _validate_full_mass_manifest(manifest, report)
        verify_artifacts(root / DERIVED_DIRECTORY, manifest["artifacts_sha256"], "Verify full-mass scores", workers=workers)
    with np.load(root / DERIVED_DIRECTORY / "scores.npz", allow_pickle=False) as archive:
        record = {name: archive[name] for name in archive.files}
    if (record["scores"].shape != record["mass"].shape
            or record["cut_masks"].shape != (len(CUTS), len(record["mass"]))
            or not np.isfinite(record["scores"][record["mask"]]).all()
            or digest(record["event_ids"]) != manifest["population_event_ids_sha256"]):
        raise ValueError("Invalid full-mass score export")
    event_keys(record["event_ids"])
    record["manifest"] = manifest
    return record


def export_full_mass(args, report=None):
    from .enhancements import deterministic_spline_sums

    with deterministic_spline_sums():
        return _export_full_mass(args, report)


def _export_full_mass(args, report=None):
    import torch
    from .enhancements import SPLINE_SUM_PROTOCOL as spline_protocol
    from .scan_cache import resolve_reference
    from .stein import _background_score_array
    from .stein_scoring import SUPPORT_CONTEXT_PROTOCOL, _load_background, _load_reference_b, _support_context, final_transform
    from .training import residual_background_sample

    destination = Path(args.output)
    report = read_json(destination / "result.json") if report is None else report
    if not report.get("completed"):
        raise ValueError("Sideband backfill requires a completed result; training has not been started")
    with locked(destination / ".resume/full_mass_diagnostics.lock"):
        _recover_mass_diagnostic(destination)
    root, source_report = resolve_reference(destination, report)
    data = Path(args.data)
    inputs = report["contract"]["inputs"]
    current = read_json(data / "inputs.json")
    if current["files"] != inputs["files"] or current.get("event_ids_sha256") != inputs.get("event_ids_sha256"):
        raise ValueError("Prepared data changed since training; refusing frozen sideband backfill")
    torch.use_deterministic_algorithms(True)
    frozen, ensemble = frozen_contract(root, source_report, getattr(args, "verify_workers", 8))
    count = int(getattr(args, "sideband_reference_events", 65536))
    if count < 32768:
        raise ValueError("Independent sideband threshold reference requires at least 32768 events")
    identity = dict(protocol=PROTOCOL, frozen=frozen, inputs=dict(files=inputs["files"],
        event_ids_sha256=inputs["event_ids_sha256"]), seed=int(report["seed"]),
        reference_events=count, reference_seeds=REFERENCE_SEEDS,
        execution=dict(device=args.device, environment=environment(), torch_threads=args.torch_threads,
                       deterministic_algorithms=True, spline_cumsum=spline_protocol,
                       cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                       code={path.name: file_digest(path)
                        for path in sorted(Path(__file__).parent.glob("*.py"))
                        if path.name not in ("plotting.py", "figures.py")}))
    key = identity_digest(identity)
    export = destination / DERIVED_DIRECTORY
    manifest_path = export / "manifest.json"
    if manifest_path.is_file() and read_json(manifest_path).get("identity") == identity:
        load_full_mass(destination, report, workers=getattr(args, "verify_workers", 8))
        emit_message("Reuse verified frozen SR + sideband export", kind="PASS")
        return
    work = destination / ".resume" / "sidebands" / key
    work.mkdir(parents=True, exist_ok=True)
    started = time.monotonic()
    started_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    emit_message(f"Frozen sideband scoring START; seed={report['seed']}; mode={frozen['mode']}; utc={started_utc}")
    names = ("innerdata_test.npy", "outerdata_test.npy")
    with ThreadPoolExecutor(max_workers=args.io_workers) as pool:
        rows = list(pool.map(lambda name: np.load(data / name, allow_pickle=False).astype(np.float32), names))
    verify_artifacts(data, {name: inputs["files"][name] for name in names}, workers=getattr(args, "verify_workers", 8))
    verify_artifacts(data, {"event_ids.npz": inputs["event_ids_sha256"]}, workers=getattr(args, "verify_workers", 8))
    with np.load(data / "event_ids.npz", allow_pickle=False) as archive:
        ids = [archive[name] for name in names]
        test_ids = np.concatenate(ids)
        development_ids = np.concatenate([archive[name] for name in archive.files
            if name.endswith(("_train.npy", "_val.npy"))])
    if len(np.intersect1d(event_keys(test_ids), event_keys(development_ids))):
        raise ValueError("Full-mass test population overlaps training/validation events")
    sr_rows, sb_rows = rows
    sr_ids, sb_ids = ids
    if (not np.all((sr_rows[:, 0] > 3.3) & (sr_rows[:, 0] < 3.7))
            or np.any((sb_rows[:, 0] > 3.3) & (sb_rows[:, 0] < 3.7))):
        raise ValueError("Prepared primary test SR/sideband membership is inconsistent")
    with np.load(root / "signal_region_scores.npz", allow_pickle=False) as archive:
        sr_index = match_ids(archive["event_ids"], sr_ids)
        sr_scores = archive["scores"][sr_index]
        sr_mask = archive["mask"][sr_index]
        if not np.array_equal(archive["mass"][sr_index], sr_rows[:, 0]):
            raise ValueError("Saved primary SR masses differ from the prepared events")
        if not np.array_equal(archive["physical"][sr_index], sr_rows[:, 1:-1]):
            raise ValueError("Saved primary SR features differ from the prepared events")
    bundle_path = work / "mapped.npz"
    bundle_meta = work / "mapped.json"
    if bundle_path.is_file() and bundle_meta.is_file():
        if read_json(bundle_meta).get("sha256") != file_digest(bundle_path):
            raise ValueError("Frozen mapped sideband cache changed")
        with np.load(bundle_path, allow_pickle=False) as archive:
            sb_mask, z, reference_d = archive["mask"], archive["latents"], archive["reference_D"]
    else:
        if frozen["mapping_kind"] == "enhanced":
            from .mapping import Mapper
            mapper = Mapper(root / "background", args.device)
        else:
            from .latent import Mapper
            epoch = read_json(root / "background/flow_selection.json")["inference_mapping_epoch"]
            mapper = Mapper(data, root / "background", epoch, args.device)
        clean = sb_rows.astype(np.float32, copy=True)
        clean[:, -1] = 0
        z, sb_mask = mapper.map(clean)
        if (str(args.device).startswith("cuda") and getattr(mapper, "last_inference_batch_size", None) is not None
                and mapper.last_inference_batch_size != mapper.inference_batch_size):
            raise RuntimeError("Frozen sideband mapping reduced its inference batch after GPU OOM; resume with more free GPU memory")
        del mapper
        from .model import with_mass_context
        z = with_mass_context(z, sb_rows[sb_mask, 0])
        if not len(z) or not np.isfinite(z).all():
            raise FloatingPointError("Sideband mapping produced no finite accepted events")
        reference_b, _ = _load_reference_b(root / "density")
        context_rng = np.random.default_rng([report["seed"], REFERENCE_SEEDS["contexts"]])
        contexts = reference_b[context_rng.integers(len(reference_b), size=count), -1]
        seed = int(np.random.SeedSequence([report["seed"], REFERENCE_SEEDS["background"]]).generate_state(1)[0])
        reference_d = residual_background_sample(root / "density" / frozen["members"][0]["directory"],
                                                contexts, count, seed, args.device,
                                                batch_size=frozen["settings"]["stein"]["scoring"]["inference_batch_size"],
                                                strict_batch_size=True)
        atomic_write(bundle_path, lambda path: save_npz(path, mask=sb_mask, latents=z, reference_D=reference_d))
        write_json(bundle_meta, dict(sha256=file_digest(bundle_path), truth_labels_used=False))
    latents = np.concatenate((z, reference_d))
    if not (work / "latents.npy").is_file():
        _save_array(work / "latents.npy", latents)
    elif digest(np.load(work / "latents.npy", mmap_mode="r", allow_pickle=False)) != digest(latents):
        raise ValueError("Frozen inference input cache changed")
    derivative = frozen["mode"] in ("local_qnorm", "hybrid", "hybrid_gated", "sic_preserving", "tail_focus")
    if derivative:
        q_path, q_meta = work / "qscore.npy", work / "qscore.json"
        if q_path.is_file() and q_meta.is_file():
            if read_json(q_meta).get("sha256") != file_digest(q_path):
                raise ValueError("Frozen background-gradient cache changed")
        else:
            first = root / "density" / frozen["members"][0]["directory"]
            model = _load_background(first, read_json(first / "residual_training_inputs.json"), args.device)
            runtime = {}
            qscore = _background_score_array(_support_context(latents), model, args.device, mass_conditioning=True,
                batch_size=frozen["settings"]["stein"]["scoring"]["qscore_batch_size"], runtime_metadata=runtime)
            if runtime["qscore_batch_size_used"] not in (0, runtime["qscore_batch_size_requested"]):
                raise RuntimeError("Frozen background scoring reduced its batch after GPU OOM; resume with more free GPU memory")
            _save_array(q_path, qscore)
            write_json(q_meta, dict(sha256=file_digest(q_path), runtime=runtime))
            del model, qscore
    if str(args.device).startswith("cuda"):
        torch.cuda.empty_cache()
    raw = _predict_members(root, work, frozen, key, args)
    scores, guarded_raw, _ = final_transform(root / "density", ensemble["members"], raw, latents,
                                           args.device, frozen["settings"])
    aligned_sb = np.full(len(sb_rows), np.nan, dtype=scores.dtype)
    aligned_sb[sb_mask] = scores[:len(z)]
    split = np.random.default_rng([report["seed"], REFERENCE_SEEDS["split"]]).permutation(count)
    threshold_indices, assessment_indices = split[:count // 2], split[count // 2:]
    reference_scores = scores[len(z):]
    rules = threshold_rules(reference_scores[threshold_indices])
    full_rows = np.concatenate(rows)
    record = dict(mass=full_rows[:, 0], physical=full_rows[:, 1:-1], labels=full_rows[:, -1].astype(np.int8),
                  event_ids=test_ids, mask=np.r_[sr_mask, sb_mask], scores=np.r_[sr_scores, aligned_sb],
                  is_signal_region=np.r_[np.ones(len(sr_rows), bool), np.zeros(len(sb_rows), bool)])
    record["cut_masks"] = apply_rules(record["scores"], record["mask"], test_ids, rules, REFERENCE_SEEDS["ties"])
    validation = diagnostic_validation(record, reference_scores[assessment_indices], rules)
    export.mkdir(parents=True, exist_ok=True)
    atomic_write(export / "scores.npz", lambda path: save_npz(path, **record))
    atomic_write(export / "sideband_scores.npz", lambda path: save_npz(path, mass=sb_rows[:, 0],
        event_ids=sb_ids, scores=aligned_sb, mask=sb_mask,
        cut_masks=record["cut_masks"][:, len(sr_rows):], score_scope=np.array("held_out_sidebands")))
    atomic_write(export / "reference_D.npz", lambda path: save_npz(path, scores=reference_scores,
        threshold_indices=threshold_indices, assessment_indices=assessment_indices))
    write_json(export / "validation.json", validation)
    artifacts = {name: file_digest(export / name) for name in
                 ("scores.npz", "sideband_scores.npz", "reference_D.npz", "validation.json")}
    manifest = dict(protocol=PROTOCOL, completed=True, identity=identity, artifacts_sha256=artifacts,
                    diagnostic_protocol=MASS_CLOSURE_PROTOCOL, diagnostic_code_sha256=_diagnostic_code(),
                    population="primary_held_out_innerdata_test_plus_outerdata_test",
                    population_event_ids_sha256=digest(test_ids), thresholds=rules,
                    thresholds_source="independent_generated_reference_D_threshold_half",
                    threshold_truth_labels_used=False, SR_scores_preserved=True,
                    SR_source_sha256=source_report["artifacts_sha256"]["signal_region_scores.npz"],
                    reference_D_sha256=digest(reference_d), validation_status=validation["status"],
                    execution_resources=dict(workers=args.workers, io_workers=args.io_workers,
                                             verify_workers=getattr(args, "verify_workers", 8)),
                    background_correction_domain="SR only" if report["contract"]["method"] in ("iad", "supervised") else "sidebands",
                    background_gradient_context_protocol=SUPPORT_CONTEXT_PROTOCOL,
                    physics_certified=False, elapsed_seconds=time.monotonic() - started)
    write_json(manifest_path, manifest)
    emit_closure_status(validation)
    ended_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    emit_message(f"Frozen sideband scoring END; utc={ended_utc}; elapsed={time.monotonic() - started:.1f}s", kind="PASS")


RESCORE_PROTOCOL = "frozen_native_rescoring_v3_sr_support_context"
RESCORE_PARTITIONS = ("signal_region", "test", "validation")
RESCORE_METHODS = ("riddle", "iad", "supervised")


def _rescore_link(source, destination):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_file() or file_digest(destination) != file_digest(source):
            raise ValueError(f"Unexpected frozen re-scoring artifact: {destination}")
    else:
        try:
            os.link(source.resolve(), destination)
        except OSError as error:
            if error.errno not in (errno.EXDEV, errno.EPERM, errno.EOPNOTSUPP):
                raise
            def verified_copy(temporary):
                copy_file(source, temporary)
                if file_digest(temporary) != file_digest(source):
                    raise ValueError(f"Copied frozen re-scoring artifact changed: {destination}")
            atomic_write(destination, verified_copy)


def _rescore_stage(source, output, report, workers):
    mutable = {"protocol.json", "score_health.json", "score_comparison.json",
               "density/ensemble_inputs.json", "density/stein_scoring_calibration.json",
               "density/selected_scoring_calibration.json", "density/score_selection.json"}
    mutable.update(f"{name}_scores.npz" for name in RESCORE_PARTITIONS)
    names = [name for name in report["artifacts_sha256"] if name not in mutable]
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda name: _rescore_link(source / name, output / name), names))
    for name in ("stein_scoring", "stein_qscore_cache"):
        directory = source / "density/.resume" / name
        for path in directory.rglob("*") if directory.is_dir() else ():
            if path.is_file():
                _rescore_link(path, output / "density/.resume" / name / path.relative_to(directory))


def _oracle_selector_roles(args, source, mapper, saved):
    from .model import with_mass_context
    from .pipeline import _map_oracle_rows, _oracle_role_seed
    from .populations import prepared_config, role_indices

    population = prepared_config(args.data)
    if population is None:
        raise ValueError("Frozen oracle re-scoring requires the original shared population configuration")
    receipt = read_json(source / "oracle_roles.json")
    if receipt.get("method") != args.method:
        raise ValueError("Frozen oracle population belongs to another method")
    with np.load(args.data / "event_ids.npz", allow_pickle=False) as archive:
        ids = {name: archive[name] for name in archive.files}
    background = np.load(args.data / "innerdata_extrabkg_val.npy", allow_pickle=False)
    if len(background) != len(ids["innerdata_extrabkg_val.npy"]):
        raise ValueError("Frozen oracle background rows and event identities are misaligned")
    qparts = role_indices(len(background), population["validation_roles"]["oracle_background"],
                          ("fit", "closure", "selector"), _oracle_role_seed(args.seed, "oracle_background_validation"))
    _, qz, qm = _map_oracle_rows(mapper, background, 0)
    selector = dict(q=with_mass_context(qz[qparts["selector"]], qm[qparts["selector"]]),
                    q_ids=ids["innerdata_extrabkg_val.npy"][qparts["selector"]])
    if args.method == "supervised":
        signal = np.load(args.data / "innerdata_extrasig_val.npy", allow_pickle=False)
        if len(signal) != len(ids["innerdata_extrasig_val.npy"]):
            raise ValueError("Frozen oracle signal rows and event identities are misaligned")
        pparts = role_indices(len(signal), population["validation_roles"]["supervised_signal"],
                              ("assessment", "selector"), _oracle_role_seed(args.seed, "oracle_signal_validation"))
        _, pz, pm = _map_oracle_rows(mapper, signal, 1)
        selector.update(p=with_mass_context(pz[pparts["selector"]], pm[pparts["selector"]]),
                        p_ids=ids["innerdata_extrasig_val.npy"][pparts["selector"]])
        p_train_ids = ids["innerdata_extrasig_train.npy"]
        p_validation_ids = ids["innerdata_extrasig_val.npy"][pparts["assessment"]]
    else:
        p_train_ids, p_validation_ids = saved["residual_train__ids"], saved["evidence__ids"]
    for name in ("p", "q"):
        if name in selector and (len(selector[name]) != receipt["selector_counts"][name]
                                 or digest(selector[name + "_ids"]) != receipt["selector_reserves"][name + "_ids"]):
            raise ValueError("Frozen oracle selector event population changed")
    selector["used_ids"] = np.concatenate([p_train_ids, p_validation_ids, ids["innerdata_extrabkg_train.npy"],
                                           ids["innerdata_extrabkg_val.npy"][qparts["fit"]],
                                           ids["innerdata_extrabkg_val.npy"][qparts["closure"]]])
    return dict(selector=selector)


def _selector_data(args, source, report):
    from .mapping import Mapper
    from .score_selection import selector_populations

    roles = read_json(source / "background/data_roles.json")
    with np.load(source / "background/event_roles.npz", allow_pickle=False) as archive:
        saved = {key: archive[key] for key in archive.files}
    mixture = np.load(source / "background/mixture_validation_latents.npy", allow_pickle=False)
    p_ids = saved["mixture_validation__ids"]
    indices = saved["closure__source_indices"]
    if digest(indices) != roles["closure"]["indices_sha256"]:
        raise ValueError("Frozen closure role indices changed")
    rows = np.load(args.data / roles["closure"]["source"], mmap_mode="r", allow_pickle=False)[indices]
    mapper = Mapper(source / "background", args.device)
    z, mask = mapper.map(rows)
    if not np.array_equal(mask, saved["closure__mask"]):
        raise ValueError("Frozen sideband control acceptance changed")
    with np.load(source / "signal_region_scores.npz", allow_pickle=False) as archive:
        evaluation_ids = archive["event_ids"]
    if len(mixture) != len(p_ids) or len(z) != len(saved["closure__ids"]):
        raise ValueError("Frozen selector rows and event identities are misaligned")
    development = {name: dict(ids=saved[name + "__ids"]) for name in roles}
    development["mixture_validation"].update(z=mixture[:, 1:-2], mass=mixture[:, 0])
    development["closure"].update(z=z, mass=rows[mask, 0])
    oracle = _oracle_selector_roles(args, source, mapper, saved) if args.method in ("iad", "supervised") else None
    populations = selector_populations(development, oracle, args.method, evaluation_ids)
    original = read_json(source / "density/score_selection.json")["identity"]
    if (digest(populations["p_ids"]) != original["p_event_ids_sha256"]
            or digest(populations["q_ids"]) != original["q_event_ids_sha256"]
            or ("background_control" in original and digest(populations["control_ids"])
                != original["background_control"]["event_ids_sha256"])):
        raise ValueError("Frozen re-scoring selector event identities changed")
    return populations


class FrozenPredictor:
    def __init__(self, args, source, selection, settings):
        self.args, self.source, self.selection, self.settings = args, source, selection, settings
        self.members = selection.get("accepted_members", selection["members"])
        self.environment = environment()

    def values(self, z, mode, previous=None):
        from .stein_scoring import _background_identity, _load_background, _qscore, _support_context, final_transform

        current = deepcopy(self.settings)
        current["stein"]["scoring"]["mode"] = mode
        cfg = current["stein"]["scoring"]
        if previous is not None and str(previous["selected_scoring_mode"].item()) == mode:
            mask = previous["mask"].astype(bool)
            if list(previous["accepted_fit_directories"]) != [m["directory"] for m in self.members]:
                raise ValueError("Saved fit ordering differs from the frozen ensemble")
            member_values = previous["accepted_fit_scores"][:, mask]
        else:
            identity = identity_digest(dict(protocol=RESCORE_PROTOCOL, inputs=digest(z), mode=mode,
                                            scoring=cfg, members=self.members, environment=self.environment,
                                            source_report_sha256=file_digest(self.source / "result.json")))
            work = self.args.output / ".resume/rescore" / identity
            work.mkdir(parents=True, exist_ok=True)
            if not (work / "latents.npy").exists():
                atomic_write(work / "latents.npy", lambda p: save_array(p, z))
            elif digest(np.load(work / "latents.npy", allow_pickle=False)) != digest(z):
                raise ValueError("Re-scoring input cache changed")
            derivatives = mode == "tail_focus"
            if derivatives:
                directory = self.args.output / "density" / self.members[0]["directory"]
                inputs = read_json(directory / "residual_training_inputs.json")
                background = _load_background(directory, inputs, self.args.device)
                gradients = _qscore(_support_context(z), background, self.args.device, cfg,
                                    self.args.output / "density/.resume/stein_qscore_cache", _background_identity(inputs))
                if not (work / "qscore.npy").exists():
                    atomic_write(work / "qscore.npy", lambda p: save_array(p, gradients))
                elif digest(np.load(work / "qscore.npy", allow_pickle=False)) != digest(gradients):
                    raise ValueError("Re-scoring gradient cache changed")
            frozen = dict(members=self.members, settings=current, mode=mode)
            _predict_members(self.args.output, work, frozen, identity, self.args)
            member_values = np.stack([np.load(work / f"fit_{int(m['fit_index']):03d}.npy", allow_pickle=False)
                                      for m in self.members])
        selected = [next(i for i, m in enumerate(self.members) if m["directory"] == member["directory"])
                    for member in self.selection["members"]]
        table = member_values[selected]
        scores, raw, metadata = final_transform(self.args.output / "density", self.selection["members"],
                                               table.mean(axis=0), z, self.args.device, current)
        return scores, raw, table, member_values, metadata

    def __call__(self, z, mode):
        return self.values(z, mode)[0]


def _validate_rescore_source(args, inputs, source_report):
    from .settings import validate_residual
    from .stein import _training_contract

    if not source_report.get("completed") or (source_report["seed"], source_report["scenario"], source_report["contract"]["method"]) != (args.seed, args.scenario, args.method):
        raise ValueError("Re-scoring source must be a completed result for the requested method, scenario and seed")
    old = source_report["contract"]["inputs"]
    if any(old.get(key) != inputs.get(key) for key in ("files", "event_ids_sha256", "shared_population", "variant")):
        raise ValueError("Re-scoring requires the identical prepared-data population")
    original_settings = source_report["contract"]["settings"]
    if original_settings["background"] != args.settings["background"]:
        raise ValueError("Re-scoring must preserve the saved background configuration")
    if "density/score_selection.json" not in source_report["artifacts_sha256"]:
        raise ValueError("Re-scoring requires a completed shared-population native result with a frozen selector")
    settings = validate_residual(deepcopy(args.settings["riddle"]))
    original = validate_residual(deepcopy(original_settings["riddle"]))
    if _training_contract(original) != _training_contract(settings):
        raise ValueError("Re-scoring may change scoring settings only; model/training settings must match")
    original_scoring = deepcopy(original["stein"]["scoring"])
    current_scoring = deepcopy(settings["stein"]["scoring"])
    for config in (original_scoring, current_scoring):
        config.pop("support_guard")
        config.pop("auto_switch")
    if original_scoring != current_scoring:
        raise ValueError("Re-scoring preserves member calibration and reference settings; change only support_guard or auto_switch")
    return settings


def _rescore(args, inputs, *, source=None, source_name=None):
    import torch
    from .enhancements import deterministic_spline_sums
    from .metrics import paired_score_metrics
    from .production import validate_result_scores
    from .score_selection import MODES, freeze
    from .stein_scoring import write_root_provenance

    source = (Path(args.rescore_from).resolve() / args.method / args.scenario / f"seed_{args.seed:03d}") if source is None else Path(source)
    output = args.output.resolve()
    if args.method not in RESCORE_METHODS or source == output or (source_name is None and (output.is_relative_to(source) or source.is_relative_to(output))):
        raise ValueError("Frozen re-scoring requires RIDDLE/IAD/Supervised and separate source/output directories")
    source_report = read_json(source / "result.json")
    settings = _validate_rescore_source(args, inputs, source_report)
    identity = dict(protocol=RESCORE_PROTOCOL, source=str(source if source_name is None else source_name), source_report_sha256=file_digest(source / "result.json"),
                    scoring=settings["stein"]["scoring"], input_files=inputs["files"])
    identity_hash = identity_digest(identity)
    receipt = output / ".resume/rescore_identity.json"
    if receipt.exists() and read_json(receipt) != identity:
        raise ValueError("Re-scoring source or requested scoring settings changed; use another output root")
    result_path = output / "result.json"
    if result_path.exists():
        saved = read_json(result_path)
        if saved.get("rescoring", {}).get("identity_sha256") != identity_hash:
            raise ValueError("Output contains a result from another re-scoring request")
        if saved.get("completed"):
            verify_artifacts(output, saved["artifacts_sha256"], workers=args.verify_workers)
            emit_message("Verified completed re-scoring result; fitted models remain unchanged")
            return saved
    elif not receipt.exists() and any(p.name not in ("training.log", ".resume") for p in output.iterdir()):
        raise ValueError("Re-scoring requires a new output directory")
    write_json(receipt, identity)
    verify_artifacts(source, source_report["artifacts_sha256"], "Verify frozen re-scoring source", workers=args.verify_workers)
    frozen, selection = frozen_contract(source, source_report, args.verify_workers)
    if settings.get("core") != "stein_witness" or settings.get("input_space") == "physical":
        raise ValueError("Re-scoring requires the mass-conditioned latent Stein-witness core")
    torch.use_deterministic_algorithms(True)
    seed_start(args.seed)
    started = time.monotonic()
    emit_message(f"Frozen re-scoring START; seed={args.seed}; trained models reused")
    _rescore_stage(source, output, source_report, args.io_workers)
    ensemble_inputs = read_json(source / "density/ensemble_inputs.json")
    ensemble_inputs["settings"] = settings
    write_json(output / "density/ensemble_inputs.json", ensemble_inputs)
    mapping_identity = ensemble_inputs["mapping_identity"]
    write_root_provenance(output / "density", selection["members"], selection.get("accepted_members", selection["members"]),
                          settings, mapping_identity)
    predictor = FrozenPredictor(args, source, selection, settings)
    with deterministic_spline_sums():
        populations = _selector_data(args, source, source_report)
        decision = freeze(output / "density", settings, args.method, populations,
                          inputs["shared_population"], args.device, predictor=predictor)
        selected_mode = decision["selected_mode"]
        effective = deepcopy(settings["stein"]["scoring"])
        effective["mode"] = selected_mode
        selection_hash = file_digest(output / "density/score_selection.json")
        for partition in RESCORE_PARTITIONS:
            emit_message(f"Re-score {partition}; selected mode={selected_mode}")
            with np.load(source / f"{partition}_scores.npz", allow_pickle=False) as archive:
                arrays = {key: archive[key] for key in archive.files}
            mask = arrays["mask"].astype(bool)
            endpoints = {}
            for mode in MODES:
                result = predictor.values(arrays["latent"], mode, previous=arrays)
                endpoints[mode] = result[0]
                if mode == selected_mode:
                    selected_prediction = result
            scores, raw, table, accepted, metadata = selected_prediction
            for name, values in (("scores", scores), ("raw_scores", raw), ("pew_scores", endpoints["tail_focus"]),
                                 ("potential_qnorm_scores", endpoints["potential_qnorm"])):
                arrays[name] = np.full(len(mask), np.nan)
                arrays[name][mask] = values
            for name, values in (("fit_scores", table), ("accepted_fit_scores", accepted)):
                arrays[name] = np.full((len(values), len(mask)), np.nan)
                arrays[name][:, mask] = values
            arrays["selected_scoring_mode"] = np.array(selected_mode)
            arrays["score_selection_sha256"] = np.array(selection_hash)
            arrays["auto_switch_enabled"] = np.array(decision["enabled"])
            suffix = "_support_guard" if effective["support_guard"]["enabled"] else ""
            arrays["score_kind"] = np.array(f"stein_{selected_mode}{suffix}_{effective['final_transform']}")
            arrays["fit_score_kind"] = arrays["accepted_fit_score_kind"] = np.array(f"stein_{selected_mode}")
            for prefix, members in (("fit", selection["members"]), ("accepted_fit", predictor.members)):
                arrays[prefix + "_indices"] = np.array([m["fit_index"] for m in members], dtype=np.int64)
                arrays[prefix + "_seeds"] = np.array([m["seed"] for m in members], dtype=np.uint32)
                arrays[prefix + "_directories"] = np.array([m["directory"] for m in members])
            atomic_write(output / f"{partition}_scores.npz", lambda p: save_npz(p, **arrays))
            if partition == "signal_region":
                write_json(output / "score_comparison.json", paired_score_metrics(arrays, effective["auto_switch"]["efficiencies"]))
    calibration = dict(schema=2, mode=selected_mode, settings=effective, auto_switch=decision,
                       configured_candidate_provenance=read_json(output / "density/stein_scoring_calibration.json"),
                       final_reference="full q-reference B; identical for both candidates",
                       final_reference_sha256=decision["identity"]["reference_B_sha256"],
                       final_reference_events=decision["identity"]["reference_B_events"],
                       support_guard=effective["support_guard"], final_transform=effective["final_transform"],
                       final_mass_bins=effective["final_mass_bins"], final_power=effective["final_power"],
                       truth_labels_used_by_selector=False)
    write_json(output / "density/selected_scoring_calibration.json", calibration)
    protocol = read_json(source / "protocol.json")
    protocol.update(settings={**protocol["settings"], "riddle": settings}, stein=settings["stein"],
                    score_selection=decision, stein_scoring=calibration,
                    score=f"Stein {selected_mode} with frozen {effective['support_guard']['statistic']} support and {effective['final_transform']} calibration",
                    raw_score=f"Stein {selected_mode} after frozen support correction, before final calibration",
                    unguarded_ensemble_score=f"arithmetic mean of selected per-fit {selected_mode} Stein scores before support correction",
                    fit_score_note=f"per-fit {selected_mode} Stein scores; support correction is applied only to the ensemble",
                    support_guard=f"q-reference-B {effective['support_guard']['statistic']}; label-free and truth-blind",
                    inputs="Frozen mapped SR latents and mass context; q-reference-B support and conditional calibration",
                    rescoring=identity)
    write_json(output / "protocol.json", protocol)
    write_json(output / "score_health.json", validate_result_scores(output, args.method))
    report = deepcopy(source_report)
    report["contract"]["settings"]["riddle"] = settings
    report["rescoring"] = dict(identity_sha256=identity_hash, identity=identity, trained_models_changed=False,
                               elapsed_seconds=time.monotonic() - started, inference_environment=environment(),
                               deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
                               cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
                               inference_code={path.name: file_digest(path) for path in Path(__file__).parent.glob("*.py")
                                               if path.name not in ("plotting.py", "figures.py")})
    artifacts = [path for path in output.rglob("*") if path.is_file() and ".resume" not in path.relative_to(output).parts
                 and "full_mass" not in path.relative_to(output).parts and path.name not in ("result.json", "training.log")]
    report.update(completed=True, artifacts_sha256=fingerprint_files(output, artifacts, args.io_workers))
    write_json(result_path, report)
    emit_message(f"Frozen re-scoring END; seed={args.seed}; elapsed={time.monotonic() - started:.1f}s")
    return report


def _rescore_snapshot(source, backup, report, workers):
    names = set(report["artifacts_sha256"]) | {"result.json"}
    if (source / DERIVED_DIRECTORY).is_dir():
        names.update(str(path.relative_to(source)) for path in (source / DERIVED_DIRECTORY).rglob("*")
                     if path.is_file())
    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(lambda name: _rescore_link(source / name, backup / name), sorted(names)))
    for directory_name in ("stein_scoring", "stein_qscore_cache"):
        directory = source / "density/.resume" / directory_name
        for path in directory.rglob("*") if directory.is_dir() else ():
            if path.is_file():
                _rescore_link(path, backup / path.relative_to(source))


def _publish_rescore(output, stage, backup, state, state_path, workers):
    report = read_json(stage / "result.json")
    if not report.get("completed"):
        raise ValueError("The staged re-scoring result is incomplete")
    verify_artifacts(stage, report["artifacts_sha256"], "Verify replacement scores", workers=workers)
    files = dict(report["artifacts_sha256"])
    if (stage / DERIVED_DIRECTORY / "manifest.json").is_file():
        load_full_mass(stage, report, workers=workers)
        files.update({str(path.relative_to(stage)): file_digest(path)
                      for path in (stage / DERIVED_DIRECTORY).rglob("*") if path.is_file()})
    old = read_json(backup / "result.json")
    if state["phase"] != "publishing":
        if file_digest(output / "result.json") != state["source_report_sha256"]:
            raise ValueError("The original result changed before re-scoring publication")
        state["phase"] = "publishing"
        write_json(state_path, state)
    pending = deepcopy(old)
    pending.update(completed=False, rescoring_update=dict(backup=state["backup"], stage=state["stage"]))
    write_json(output / "result.json", pending)
    obsolete = set(old["artifacts_sha256"]) - set(report["artifacts_sha256"])
    if (output / DERIVED_DIRECTORY).is_dir():
        obsolete.update(str(path.relative_to(output)) for path in (output / DERIVED_DIRECTORY).rglob("*")
                        if path.is_file() and str(path.relative_to(output)) not in files)
    for name in sorted(files, key=lambda name: (name.endswith("/manifest.json"), name)):
        destination = output / name
        checksum = files[name]
        if destination.is_file() and not destination.is_symlink() and file_digest(destination) == checksum:
            continue
        def publish(temporary, name=name, checksum=checksum):
            temporary.unlink()
            _rescore_link(stage / name, temporary)
            if file_digest(temporary) != checksum:
                raise ValueError(f"Replacement artifact changed during publication: {name}")
        atomic_write(destination, publish)
    for name in sorted(obsolete):
        (output / name).unlink(missing_ok=True)
    verify_artifacts(output, files, "Verify published re-scoring artifacts", workers=workers)
    report["rescoring"].update(in_place=True, request_sha256=state["request_sha256"], backup=state["backup"])
    write_json(output / "result.json", report)
    state["phase"] = "completed"
    write_json(state_path, state)
    emit_message("Updated scores in the original result folder; previous artifacts preserved under .resume", kind="PASS")
    return report


def _rescore_in_place(args, inputs):
    output = Path(args.output).resolve()
    request = dict(protocol=RESCORE_PROTOCOL, scoring=args.settings["riddle"]["stein"]["scoring"], inputs=inputs,
                   background=args.settings["background"],
                   residual_training=args.settings["riddle"],
                   sideband_scoring=getattr(args, "sideband_scoring", True),
                   sideband_reference_events=getattr(args, "sideband_reference_events", 65536))
    request_hash = identity_digest(request)
    state_path = output / ".resume/rescore_in_place.json"
    state = read_json(state_path) if state_path.is_file() else None
    if state is not None and state["phase"] != "completed" and state["request_sha256"] != request_hash:
        legacy_request = {**request, "protocol": "frozen_riddle_rescoring_v2_deterministic"}
        if (not getattr(args, "resume_across_code_change", False)
                or state["request_sha256"] != identity_digest(legacy_request)):
            raise ValueError("An unfinished in-place re-scoring request exists; resume with the same scoring settings and prepared population")
        if state["phase"] == "publishing":
            _publish_rescore(output, output / state["stage"], output / state["backup"],
                             state, state_path, args.verify_workers)
        elif state["phase"] not in ("snapshot", "scoring", "ready") or file_digest(output / "result.json") != state["source_report_sha256"]:
            raise ValueError("The original result changed during the scoring-calibration upgrade")
        emit_message("Upgrade frozen scoring calibration; previous stages and trained models preserved")
        state = None
    current = read_json(output / "result.json")
    validation_report = current
    if state is not None and state["phase"] in ("scoring", "ready", "publishing"):
        validation_report = read_json(output / state["backup"] / "result.json")
    _validate_rescore_source(args, inputs, validation_report)
    if current.get("completed") and current.get("rescoring", {}).get("request_sha256") == request_hash:
        verify_artifacts(output, current["artifacts_sha256"], "Verify completed replacement scores", workers=args.verify_workers)
        score_sidebands(args, current)
        if state is not None and state["phase"] != "completed":
            state["phase"] = "completed"
            write_json(state_path, state)
        emit_message("Reuse verified in-place re-scoring result", kind="PASS")
        return current
    if state is None or state["phase"] == "completed":
        if not current.get("completed"):
            raise ValueError("In-place re-scoring requires a completed original result")
        source_hash = file_digest(output / "result.json")
        generation = identity_digest(dict(source_report_sha256=source_hash, request_sha256=request_hash))
        state = dict(phase="snapshot", request_sha256=request_hash, source_report_sha256=source_hash,
                     backup=f".resume/rescore_backups/{generation}", stage=f".resume/rescore_stages/{generation}")
        write_json(state_path, state)
    backup, stage = output / state["backup"], output / state["stage"]
    if state["phase"] == "snapshot":
        if file_digest(output / "result.json") != state["source_report_sha256"]:
            raise ValueError("The completed original result changed during backup")
        verify_artifacts(output, current["artifacts_sha256"], "Verify original re-scoring source", workers=args.verify_workers)
        _rescore_snapshot(output, backup, current, args.io_workers)
        verify_artifacts(backup, current["artifacts_sha256"], "Verify recoverable score backup", workers=args.verify_workers)
        if file_digest(backup / "result.json") != state["source_report_sha256"]:
            raise ValueError("The recoverable original report changed")
        state["phase"] = "scoring"
        write_json(state_path, state)
    if state["phase"] == "scoring":
        from types import SimpleNamespace
        stage.mkdir(parents=True, exist_ok=True)
        staged_args = SimpleNamespace(**{**vars(args), "output": stage})
        staged_report = _rescore(staged_args, inputs, source=backup, source_name=output)
        score_sidebands(staged_args, staged_report)
        state["phase"] = "ready"
        write_json(state_path, state)
    return _publish_rescore(output, stage, backup, state, state_path, args.verify_workers)


def rescore(args, inputs):
    source = Path(args.rescore_from).resolve() / args.method / args.scenario / f"seed_{args.seed:03d}"
    output = Path(args.output).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"Completed re-scoring source is missing: {source}")
    if args.method not in RESCORE_METHODS or (source != output and (source.is_relative_to(output) or output.is_relative_to(source))):
        raise ValueError("Frozen re-scoring requires RIDDLE/IAD/Supervised and matching or nonoverlapping source/output folders")
    if not getattr(args, "rescore_output_locked", False):
        with locked(output / ".resume/command.lock"):
            from types import SimpleNamespace
            locked_args = SimpleNamespace(**{**vars(args), "rescore_output_locked": True})
            return rescore(locked_args, inputs)
    if source == output:
        return _rescore_in_place(args, inputs)
    with locked(source / ".resume/command.lock"):
        report = _rescore(args, inputs)
        score_sidebands(args, report)
        return report
