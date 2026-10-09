from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from copy import deepcopy
import hashlib
import json
import multiprocessing
from datetime import datetime, timezone
import os
from pathlib import Path
import signal
import time

import numpy as np

from .mass_spectrum import CUTS, event_keys, match_ids, mass_edges
from .storage import atomic_write, digest, environment, file_digest, read_json, save_array, save_npz, verify_artifacts, write_json
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
    from .stein_scoring import member_scores

    root, work = Path(job["root"]), Path(job["work"])
    inputs = np.load(work / "latents.npy", mmap_mode="c", allow_pickle=False)
    qscore = np.load(work / "qscore.npy", mmap_mode="c", allow_pickle=False) if job["derivatives"] else None
    member = job["member"]
    with np.load(root / "density" / member["directory"] / "stein_scoring_calibration.npz", allow_pickle=False) as archive:
        calibration = {name: archive[name] for name in archive.files}
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
    from scipy.optimize import minimize
    from scipy.special import xlogy
    from scipy.stats import binomtest, chi2

    mass, labels = record["mass"], record["labels"]
    regions = {"signal_region": record["is_signal_region"], "sidebands": ~record["is_signal_region"]}
    edges = mass_edges(mass, split_sr=True)
    widths = np.diff(edges)
    nodes, quadrature = np.polynomial.legendre.leggauss(8)
    centers = (edges[:-1] + edges[1:]) / 2
    log_nodes = np.log(centers[:, None] + widths[:, None] * nodes / 2)
    design = np.stack([np.ones_like(log_nodes), log_nodes, log_nodes ** 2], axis=-1)
    bins_sr = (edges[:-1] >= 3.3) & (edges[1:] <= 3.7)
    result = dict(schema=1, thresholds_use_evaluation_labels=False,
                  closure_uses_simulation_background_labels=True, cuts=[], physics_certified=False,
                  calibration_domain="SR; sidebands use the unchanged edge-clamped SR calibration",
                  fit_model="diagnostic exp(a+b*log(m)+c*log(m)^2), integrated over mass bins",
                  fit_is_publication_search_model=False)
    if not np.any(labels == 0):
        result.update(status="inconclusive", reason="No labelled held-out background for closure diagnostics")
        return result
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
        fit_bins = (~bins_sr) & (total > 0)
        fit = None
        if fit_bins.sum() >= 5 and chosen[fit_bins].sum() >= 50:
            initial = np.array([np.log(max(1.0, chosen[fit_bins].sum() / widths[fit_bins].sum())), 0, 0])

            def expectation(parameters):
                density = np.exp(np.clip(design @ parameters, -100, 100))
                return (density * quadrature).sum(axis=1) * widths / 2

            def loss(parameters):
                mean = expectation(parameters)[fit_bins]
                return float(np.sum(mean - xlogy(chosen[fit_bins], mean)))

            optimized = minimize(loss, initial, method="L-BFGS-B", bounds=[(-100, 100)] * 3)
            expected = expectation(optimized.x)
            if optimized.success and np.isfinite(expected).all() and np.all(expected > 0):
                counts, fitted = chosen[fit_bins], expected[fit_bins]
                deviance = float(2 * np.sum(xlogy(counts, counts / fitted) - counts + fitted))
                observed_sr, expected_sr = int(chosen[bins_sr].sum()), float(expected[bins_sr].sum())
                hessian = np.asarray(optimized.hess_inv.todense())
                gradient = ((np.exp(np.clip(design @ optimized.x, -100, 100)) * quadrature)[..., None]
                            * design * widths[:, None, None] / 2).sum(axis=1)[bins_sr].sum(axis=0)
                fit_variance = max(0.0, float(gradient @ hessian @ gradient))
                z = (observed_sr - expected_sr) / np.sqrt(max(1.0, expected_sr + fit_variance))
                fit = dict(status="passed" if abs(z) < 5 and chi2.sf(deviance, fit_bins.sum() - 3) > 0.001 else "failed",
                           sideband_deviance=deviance, degrees_of_freedom=int(fit_bins.sum() - 3),
                           approximate_p_value=float(chi2.sf(deviance, fit_bins.sum() - 3)),
                           observed_SR_background=observed_sr, predicted_SR_background=expected_sr,
                           approximate_fit_variance=fit_variance, approximate_SR_pull=float(z),
                           covariance="L-BFGS inverse Hessian; exploratory diagnostic")
        row["background_fit_closure"] = fit or dict(status="inconclusive", reason="Insufficient statistics or fit convergence")
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
    statuses = [row["background_fit_closure"]["status"] for row in result["cuts"]]
    statuses.extend(boundary["status"] for row in result["cuts"] for boundary in row["SR_boundaries"])
    result["status"] = "failed" if "failed" in statuses else "inconclusive" if "inconclusive" in statuses else "passed"
    return result


def load_full_mass(root, report, *, workers=1):
    root = Path(root)
    path = root / DERIVED_DIRECTORY / "manifest.json"
    if not path.is_file():
        return None
    manifest = read_json(path)
    if manifest.get("protocol") != PROTOCOL or not manifest.get("completed"):
        raise ValueError("Full-mass score export is incomplete or incompatible")
    expected = report["contract"]["inputs"]
    if (manifest["identity"]["inputs"]["files"] != expected["files"]
            or manifest["identity"]["inputs"]["event_ids_sha256"] != expected.get("event_ids_sha256")):
        raise ValueError("Full-mass export uses a different prepared population")
    frozen = manifest["identity"]["frozen"]
    for name, checksum in frozen["artifacts_sha256"].items():
        if report["artifacts_sha256"].get(name) != checksum:
            raise ValueError("Full-mass export uses different frozen training artifacts")
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
    import torch
    from .scan_cache import resolve_reference
    from .stein import _background_score_array
    from .stein_scoring import _load_background, _load_reference_b, final_transform
    from .training import residual_background_sample

    destination = Path(args.output)
    report = read_json(destination / "result.json") if report is None else report
    if not report.get("completed"):
        raise ValueError("Sideband backfill requires a completed result; training has not been started")
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
                       deterministic_algorithms=True, cublas_workspace=os.environ.get("CUBLAS_WORKSPACE_CONFIG"),
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
            qscore = _background_score_array(latents, model, args.device, mass_conditioning=True,
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
                    population="primary_held_out_innerdata_test_plus_outerdata_test",
                    population_event_ids_sha256=digest(test_ids), thresholds=rules,
                    thresholds_source="independent_generated_reference_D_threshold_half",
                    threshold_truth_labels_used=False, SR_scores_preserved=True,
                    SR_source_sha256=source_report["artifacts_sha256"]["signal_region_scores.npz"],
                    reference_D_sha256=digest(reference_d), validation_status=validation["status"],
                    execution_resources=dict(workers=args.workers, io_workers=args.io_workers,
                                             verify_workers=getattr(args, "verify_workers", 8)),
                    background_correction_domain="SR only" if report["contract"]["method"] in ("iad", "supervised") else "sidebands",
                    physics_certified=False, elapsed_seconds=time.monotonic() - started)
    write_json(manifest_path, manifest)
    emit_message(f"Frozen full-mass closure: {validation['status']}; diagnostic export saved",
                 kind="WARNING" if validation["status"] != "passed" else "PASS")
    ended_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    emit_message(f"Frozen sideband scoring END; utc={ended_utc}; elapsed={time.monotonic() - started:.1f}s", kind="PASS")
