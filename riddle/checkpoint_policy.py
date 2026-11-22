import fcntl
import multiprocessing
import os
import tempfile
from concurrent.futures import ProcessPoolExecutor
from copy import copy, deepcopy
from pathlib import Path

import numpy as np

from .storage import atomic_torch_save, file_digest, load_checkpoint, read_json, verify_artifacts, write_json
from .worker_progress import emit_message


METHODS = ("riddle", "iad", "supervised")
DIRECTORY = "checkpoint_scores"
POLICY_VERSION = 1


def source_identity(report):
    from .full_mass import identity_digest

    return identity_digest({key: report[key] for key in ("method", "seed", "scenario", "contract", "artifacts_sha256")})


def verified(root, report, relative):
    name = str(relative)
    if name not in report["artifacts_sha256"]:
        raise ValueError(f"Missing registered artifact {root / name}")
    verify_artifacts(root, {name: report["artifacts_sha256"][name]}, workers=1)
    return root / name


def shared_artifact(name):
    path = Path(name)
    return (name != "density/ensemble_selection.json" and
            path.name not in ("residual_selection.json", "stein_scoring_calibration.json", "stein_scoring_calibration.npz")
            and not path.match("residual_epoch_*.pt"))


def fit_origin(source, report, directory):
    prefix = Path("density") / directory
    visited = {source.resolve()}
    while True:
        if "density/ensemble_reuse.json" in report["artifacts_sha256"]:
            receipt = read_json(verified(source, report, "density/ensemble_reuse.json"))
            if receipt["policy"] != "exact_supervised_ensemble_inputs_v1":
                raise ValueError(f"Unrecognized witness ensemble reuse policy: {source}")
            original = Path(receipt["source_result"]).resolve()
        else:
            ancestor = report.get("rescoring", {}).get("identity", {}).get("source")
            if (source / prefix / ".resume/latest.pt").is_file() or not ancestor or Path(ancestor).resolve() == source:
                break
            original = Path(ancestor).resolve()
        if original in visited:
            raise ValueError(f"Invalid witness ensemble reuse chain: {source}")
        visited.add(original)
        original_report = read_json(original / "result.json")
        for path in (prefix / "residual_selection.json", prefix / "residual_losses.json",
                     prefix / "residual_training_inputs.json", prefix.parents[1] / "split.npz"):
            if original_report["artifacts_sha256"].get(str(path)) != report["artifacts_sha256"][str(path)]:
                raise ValueError(f"Reused fit differs from its original producer: {directory}")
        source, report = original, original_report
    return source, report


def _recovery(source, report, directory, info, history):
    from .stein import _validate_recovery

    prefix = Path("density") / directory
    origin, _ = fit_origin(source, report, directory)
    path = origin / prefix / ".resume/latest.pt"
    if not path.is_file():
        raise FileNotFoundError(f"No saved recovery state for {source / prefix}")
    state = load_checkpoint(path, map_location="cpu", weights_only=False)
    _validate_recovery(state, info["settings"]["epochs"], info["settings"],
                       (info.get("background_correction") or {}).get("source_sha256"))
    if state["history"] != history or state["epoch"] + 1 != len(history):
        raise ValueError(f"Recovery endpoint differs from the registered fit history: {path}")
    for name, checksum in state["files"].items():
        if report["artifacts_sha256"].get(str(prefix / name)) != checksum:
            raise ValueError(f"Recovery state differs from the registered checkpoints: {path}")
    return state, file_digest(path)


def archive_weights(archives, source, report, members):
    result = {m["directory"]: {} for m in members}
    producer = file_digest(source / "result.json")
    for archive in map(Path, archives or ()):
        legacy = archive / "runner/training_protocol.json"
        if legacy.is_file():
            case = read_json(legacy)["cases"].get(str(report["seed"]), {})
            if case.get("producer") != producer:
                continue
            base = archive / "trained" / f"seed_{report['seed']:03d}"
        else:
            cases = read_json(archive / "runner/plan.json")["cases"]
            case = next((c for c in cases if c["producer"] == producer), None)
            if case is None:
                continue
            base = archive / "training" / case["key"]
            registry = {row["fit"]: row for row in read_json(archive / "registry.json")["cases"][case["key"]]}
        for member in members:
            index = member["fit_index"]
            directory = member["directory"]
            folder = base / f"fit_{index:02d}"
            if not (folder / "complete.json").exists():
                continue
            receipt = read_json(folder / "complete.json")
            if legacy.is_file():
                valid = (receipt["producer_sha256"] == producer and receipt["seed"] == report["seed"]
                         and receipt["source_sha256"] == case["last_weights"][str(index)])
                weights = {int(Path(n).stem.split("_")[-1]) + 1: h for n, h in receipt["snapshots"].items()}
            else:
                valid = (receipt["case"] == case["key"] and receipt["producer"] == producer
                         and registry[index]["directory"] == directory and receipt["origin"] == registry[index]["start_sha"])
                weights = {int(e): h for e, h in receipt["weights"].items()}
            if not valid or receipt["fit"] != index:
                raise ValueError(f"Checkpoint archive has inconsistent fit provenance: {folder}")
            for epoch, checksum in weights.items():
                if epoch in result[directory] and result[directory][epoch]["sha256"] != checksum:
                    raise ValueError(f"Archives disagree on fit {index} at epoch {epoch}")
                path = folder / "epochs" / f"residual_epoch_{epoch - 1}.pt"
                result[directory][epoch] = dict(epoch=epoch, path=str(path), sha256=checksum)
    return result


def include_scored_weights(output, identity, inventory):
    index_path = output / "index.json"
    if not index_path.is_file():
        return
    index = read_json(index_path)
    if index.get("source_identity") != identity:
        return
    for tag, entry in index["results"].items():
        folder = output / "last" / tag / entry["revision"]
        path = folder / "result.json"
        if file_digest(path) != entry["report_sha256"]:
            raise ValueError(f"Checkpoint policy report changed: {folder}")
        saved = read_json(path)
        policy = saved["checkpoint_policy"]
        if policy["source_identity"] != identity or policy["epoch_by_fit"] != entry["epoch_by_fit"]:
            raise ValueError(f"Checkpoint policy provenance differs from its index: {folder}")
        for name, epoch in policy["epoch_by_fit"].items():
            relative = Path("density") / name / f"residual_epoch_{epoch - 1}.pt"
            checksum = saved["artifacts_sha256"][str(relative)]
            supplied = inventory[name].get(epoch)
            if supplied and supplied["sha256"] != checksum:
                raise ValueError(f"Saved and supplied checkpoint weights differ: {name}, epoch {epoch}")
            inventory[name][epoch] = dict(epoch=epoch, path=str(folder / relative), sha256=checksum)


def _recovered_weight(job):
    import torch
    from .enhancements import deterministic_spline_sums
    from .model import real_sr_latents
    from .stein import _background_score_array, _validation_loss, build_potential
    from .stein_scoring import _load_background
    from .storage import digest

    source, name, epoch, output, rows_file, device = job
    report = read_json(source / "result.json")
    prefix = Path("density") / name
    info = read_json(verified(source, report, prefix / "residual_training_inputs.json"))
    history = read_json(verified(source, report, prefix / "residual_losses.json"))["history"]
    state, recovery_hash = _recovery(source, report, name, info, history)
    if len(history) != epoch:
        raise ValueError(f"Fit endpoint and published history disagree: {name}")
    path = output / "weights" / name / f"residual_epoch_{epoch - 1}.pt"
    stamp = path.with_suffix(".json")
    if stamp.exists():
        record = read_json(stamp)
        if record["recovery_sha256"] != recovery_hash or record["sha256"] != file_digest(path):
            raise ValueError(f"Recovered checkpoint changed: {path}")
    else:
        settings = info["settings"]
        rows = np.load(rows_file, mmap_mode="r", allow_pickle=False)
        with np.load(verified(source, report, prefix.parents[1] / "split.npz"), allow_pickle=False) as split:
            values = real_sr_latents(rows[split["validation"]], mass_conditioning=settings["mass_conditioning"])
        exact = digest(values) == info["validation_latents_sha256"]
        if not exact and report["contract"]["method"] != "supervised":
            raise ValueError(f"Legacy checkpoint validation population differs: {name}")
        with deterministic_spline_sums():
            background = _load_background(source / prefix, info, device)
            gradient = _background_score_array(values, background, device, mass_conditioning=settings["mass_conditioning"])
            model = build_potential(device, info["features"], settings)
            model.load_state_dict(state["model"])
            batch = torch.utils.data.TensorDataset(torch.from_numpy(values), torch.from_numpy(gradient))
            loader = torch.utils.data.DataLoader(batch, batch_size=settings["training"]["validation_batch_size"])
            loss = float(_validation_loss(model, loader, settings))
        if not np.isclose(loss, history[-1]["validation_nll"], atol=2e-5, rtol=3e-4):
            raise ValueError(f"Legacy checkpoint fails its recorded validation loss: {name}, epoch {epoch}")
        checksum = atomic_torch_save(path, {k: state[k] for k in ("model", "epoch", "scientific_version", "core", "stein_protocol")})
        record = dict(recovery_sha256=recovery_hash, sha256=checksum, replayed_validation_loss=loss,
                      identical_validation_latent_bytes=exact)
        write_json(stamp, record)
    return name, dict(epoch=epoch, path=str(path), sha256=record["sha256"])


def _inventory(source, report, selection, output, archived, args, requested):
    inventory, pending = {}, []
    for member in selection.get("accepted_members", selection["members"]):
        name = member["directory"]
        prefix = Path("density") / name
        metadata = read_json(verified(source, report, prefix / "residual_selection.json"))
        end = metadata["trained_epochs"]
        entries = {}
        for key, checksum in report["artifacts_sha256"].items():
            path = Path(key)
            if path.parent == prefix and path.match("residual_epoch_*.pt"):
                epoch = int(path.stem.split("_")[-1]) + 1
                if 0 < epoch <= end:
                    entries[epoch] = dict(epoch=epoch, path=str(source / path), sha256=checksum)
        entries.update(archived[name])
        extension_dir = output / "extensions" / name
        receipt_path = extension_dir / "weights.json"
        if receipt_path.is_file():
            receipt = read_json(receipt_path)
            if receipt["source_identity"] != source_identity(report):
                raise ValueError(f"Continuation belongs to a different fit: {name}")
            for epoch, checksum in receipt["weights"].items():
                path = extension_dir / f"residual_epoch_{int(epoch) - 1}.pt"
                entries[int(epoch)] = dict(epoch=int(epoch), path=str(path), sha256=checksum)
        inventory[name] = entries
        if end not in entries and (getattr(args, "checkpoint_trends", False) or requested == end
                                   or requested is None and max(entries, default=0) < end):
            pending.append((name, end))
    if pending:
        rows_file = training_rows(args, source, report, output)
        jobs = [(source, name, epoch, output, rows_file, args.device) for name, epoch in pending]
        for name, item in fit_tasks(_recovered_weight, jobs, args):
            inventory[name][item["epoch"]] = item
    return inventory, {name: max(entries, default=0) for name, entries in inventory.items()}


def _calibrate(job):
    from .enhancements import deterministic_spline_sums
    from .stein_scoring import _calibration

    directory, epoch, root, cfg, device = job
    metadata = directory / "stein_scoring_calibration.json"
    arrays = directory / "stein_scoring_calibration.npz"
    if arrays.is_file() and not metadata.exists():
        arrays.unlink()
    with deterministic_spline_sums():
        _calibration(directory, [epoch], device, cfg, root)


def fit_tasks(function, tasks, args):
    from .full_mass import _inference_worker_init

    with ProcessPoolExecutor(max_workers=min(args.workers, len(tasks)), mp_context=multiprocessing.get_context("spawn"),
                             initializer=_inference_worker_init, initargs=(args.torch_threads,)) as pool:
        return list(pool.map(function, tasks))


def _continue_fit(job):
    import torch
    from .acceleration import install_tensor_batches
    from .background_correction import descriptor
    from .enhancements import deterministic_spline_sums
    from .full_mass import _rescore_link
    from .model import real_sr_latents
    from .stein import _validate_recovery, train_stein_witness
    from .storage import digest

    source, member, rows_file, output, target, device, identity = job
    report = read_json(source / "result.json")
    name = member["directory"]
    prefix = Path("density") / name
    fit = source / prefix
    info = read_json(verified(source, report, prefix / "residual_training_inputs.json"))
    history = read_json(verified(source, report, prefix / "residual_losses.json"))["history"]
    if len(history) >= target:
        return
    initial, checksum = _recovery(source, report, name, info, history)
    if initial["stopped_early"]:
        raise ValueError(f"Cannot continue a fit stopped by its training policy: {fit}")
    folder = output / "extensions" / name
    folder.mkdir(parents=True, exist_ok=True)
    receipt_path = folder / "weights.json"
    if receipt_path.exists():
        receipt = read_json(receipt_path)
        if receipt["source_identity"] != identity or receipt["anchor_sha256"] != checksum:
            raise ValueError(f"Continuation anchor changed: {folder}")
        if str(target) in receipt["weights"]:
            return
    with np.load(verified(source, report, prefix.parents[1] / "split.npz"), allow_pickle=False) as split:
        rows = np.load(rows_file, mmap_mode="r", allow_pickle=False)
        training, validation = rows[split["training"]], rows[split["validation"]]
    settings = info["settings"]
    for data, role in ((training, "train"), (validation, "validation")):
        if digest(real_sr_latents(data, mass_conditioning=settings["mass_conditioning"])) != info[role + "_latents_sha256"]:
            raise ValueError(f"Continuation {role} population differs from the fitted witness: {fit}")
    background = descriptor(source / "density/background_correction") if info.get("background_correction") else None
    checkpoint = folder / ".resume/latest.pt"
    if checkpoint.is_file():
        state = load_checkpoint(checkpoint, map_location="cpu", weights_only=False)
        _validate_recovery(state, state["settings"]["epochs"], state["settings"], initial["background_correction_sha256"])
        if state["history"][:len(history)] != history:
            raise ValueError(f"Continuation history differs from its published anchor: {folder}")
        if state["epoch"] + 1 > target:
            raise ValueError(f"Epoch {target} is absent from the completed continuation: {folder}")
        verify_artifacts(folder, state["files"])
    else:
        state = initial
        for filename in state["files"]:
            _rescore_link(verified(source, report, prefix / filename), folder / filename)
    settings = dict(settings, epochs=target)
    state = dict(state, settings=dict(state["settings"], epochs=target))
    torch.use_deterministic_algorithms(initial.get("deterministic_algorithms", False))
    workspace = initial.get("cublas_workspace")
    if workspace is None:
        os.environ.pop("CUBLAS_WORKSPACE_CONFIG", None)
    else:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = workspace
    install_tensor_batches(device)
    with deterministic_spline_sums():
        _, extended = train_stein_witness(training, validation, folder, epochs=target, seed=member["seed"],
            device=device, initialization=settings["initialization"], settings=settings, checkpoint=state,
            background_correction=background, truth_labels_used=info["truth_labels_used"])
    if extended[:len(history)] != history or len(extended) != target:
        raise ValueError(f"Witness continuation stopped at epoch {len(extended)}; requested {target}")
    weights = read_json(receipt_path)["weights"] if receipt_path.exists() else {}
    for epoch in range(len(history) + 1, target + 1):
        if epoch % 10 == 0 or epoch == target:
            weights[str(epoch)] = file_digest(folder / f"residual_epoch_{epoch - 1}.pt")
    write_json(receipt_path, dict(source_identity=identity, anchor_sha256=checksum, weights=weights))


def training_rows(args, source, report, output):
    from .storage import atomic_write, digest, save_array

    identity = source_identity(report)
    if args.method != "supervised":
        return verified(source, report, "background/training_latents.npy")
    registered = "supervised_witness_training.npy"
    if registered in report["artifacts_sha256"]:
        return verified(source, report, registered)
    rows_file = output / ".resume" / f"training_{identity}_{str(args.device).replace(':', '_')}.npy"
    stamp = rows_file.with_suffix(".json")
    if stamp.exists() and rows_file.is_file() and read_json(stamp)["sha256"] == file_digest(rows_file):
        return rows_file
    from .mapping import Mapper
    from .pipeline import _map_oracle_rows
    import torch

    roles = read_json(verified(source, report, "oracle_roles.json"))
    filename = "innerdata_extrasig_train.npy"
    inputs = report["contract"]["inputs"]
    verify_artifacts(args.data, {filename: inputs["files"][filename], "event_ids.npz": inputs["event_ids_sha256"]})
    with np.load(args.data / "event_ids.npz", allow_pickle=False) as events:
        if digest(events[filename]) != roles["p_training_event_ids_sha256"]:
            raise ValueError("Supervised witness training event identities changed")
    for name in ("mapping_settings.json", "model.pt", "preprocessing.pt", "flow_selection.json", "background_settings.json"):
        verified(source, report, Path("background") / name)
    deterministic = torch.are_deterministic_algorithms_enabled()
    try:
        torch.use_deterministic_algorithms(False)
        rows, _, _ = _map_oracle_rows(Mapper(source / "background", args.device),
                                      np.load(args.data / filename, allow_pickle=False), 1)
    finally:
        torch.use_deterministic_algorithms(deterministic)
    if len(rows) != roles["p_training_events"]:
        raise ValueError("Supervised witness training event count changed")
    atomic_write(rows_file, lambda path: save_array(path, rows))
    write_json(stamp, dict(sha256=file_digest(rows_file), exact_training_inputs=digest(rows) == roles["p_training_sha256"],
                           event_ids_sha256=roles["p_training_event_ids_sha256"]))
    return rows_file


def extend_fits(args, source, report, selection, output, target, archived):
    members = selection.get("accepted_members", selection["members"])
    origin, original_report = fit_origin(source, report, members[0]["directory"])
    owner = origin / DIRECTORY
    owner.mkdir(exist_ok=True)
    identity = source_identity(original_report)
    with (owner / ".extensions.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        pending = []
        for member in members:
            name = member["directory"]
            if source != origin and fit_origin(source, report, name)[0] != origin:
                raise ValueError("Reused witness fits have different producers")
            selection_file = verified(source, report, Path("density") / name / "residual_selection.json")
            if read_json(selection_file)["trained_epochs"] >= target or target in archived[name]:
                continue
            receipt = owner / "extensions" / name / "weights.json"
            if receipt.exists():
                saved = read_json(receipt)
                if saved["source_identity"] == identity and str(target) in saved["weights"]:
                    continue
            pending.append(member)
        if pending:
            rows_file = training_rows(args, source, report, output)
            fit_tasks(_continue_fit, [(origin, member, rows_file, owner, target, args.device, identity)
                                      for member in pending], args)
        if source != origin:
            for member in members:
                name = member["directory"]
                folder = owner / "extensions" / name
                receipt = folder / "weights.json"
                if receipt.exists():
                    saved = read_json(receipt)
                    if saved["source_identity"] != identity:
                        raise ValueError(f"Stale witness continuation: {folder}")
                    for epoch, checksum in saved["weights"].items():
                        e = int(epoch)
                        archived[name][e] = dict(epoch=e, path=str(folder / f"residual_epoch_{e-1}.pt"), sha256=checksum)


def stage_selection(args, source, selection, plan, settings):
    from .full_mass import _rescore_link
    from .production import require_complete_ensemble
    from .stein_scoring import _background_identity, _load_background, _qscore

    require_complete_ensemble(selection)
    selection = deepcopy(selection)
    root = args.output / "density"
    cfg = settings["stein"]["scoring"]
    jobs = []
    inventories = {}
    for name, item in plan.items():
        path = Path(item["path"])
        epoch = item["epoch"] - 1
        if file_digest(path) != item["sha256"]:
            raise ValueError(f"Checkpoint checksum mismatch: {path}")
        weights = load_checkpoint(path, map_location="cpu", weights_only=True)
        if weights.get("epoch") != epoch or weights.get("core") != "stein_witness":
            raise ValueError(f"Checkpoint identity mismatch: {path}")
        directory = root / name
        _rescore_link(path, directory / f"residual_epoch_{epoch}.pt")
        metadata = read_json(source / "density" / name / "residual_selection.json")
        metadata.update(epochs=[epoch], criterion="fixed completed witness epoch", checkpoint_weights=[1.0],
                        signal_fractions=[0.0], checkpoint_weighting="uniform", witness_ensemble="single checkpoint",
                        trained_epochs=max(metadata["trained_epochs"], epoch + 1),
                        maximum_epochs=max(metadata.get("maximum_epochs", metadata["trained_epochs"]), epoch + 1))
        metadata.pop("validation_objective", None)
        write_json(directory / "residual_selection.json", metadata)
        jobs.append((directory, epoch, root, cfg, args.device))
    directory = jobs[0][0]
    inputs = read_json(directory / "residual_training_inputs.json")
    with np.load(root / "stein_scoring_reference.npz", allow_pickle=False) as archive:
        _qscore(archive["reference_A"], _load_background(directory, inputs, args.device), args.device, cfg,
                root / ".resume/stein_qscore_cache", _background_identity(inputs))
    fit_tasks(_calibrate, jobs, args)
    for directory, _, _, _, _ in jobs:
        inventories[str(directory.relative_to(root))] = {p.name: file_digest(p) for p in directory.iterdir() if p.is_file()}
    selection["configurations"] = [c for c in selection["configurations"] if c["name"] == selection["selected_configuration"]]
    for block in [selection, *selection["configurations"]]:
        for group in ("members", "accepted_members", "unselected_members"):
            for member in block.get(group, []):
                epoch = plan[member["directory"]]["epoch"] - 1
                member.update(epochs=[epoch], signal_fractions=[0.0],
                              artifacts_sha256=inventories[member["directory"]])
        block["checkpoints_per_run"] = 1
        block["selected_checkpoints"] = len(block["members"])
        block["accepted_checkpoints"] = len(block.get("accepted_members", block["members"]))
    require_complete_ensemble(selection)
    write_json(root / "ensemble_selection.json", selection)
    return selection


def trend_metrics(labels, mask, scores, report, efficiencies):
    from .mass_spectrum import CUTS
    from .metrics import oracle_metrics
    from .plotting import exact_background_selection

    record = dict(labels=labels, mask=mask, scores=scores)
    performance = oracle_metrics(labels, scores, mask, min_background=10, min_efficiency=1e-4)
    sic = performance.get("full_pipeline_max_sic")
    counts = report["contract"]["inputs"].get("uncut_signal_region", {})
    nominal = counts["signal"] / np.sqrt(counts["background"]) if counts.get("background") else None
    retention = {}
    for efficiency in sorted(set(efficiencies).union(CUTS)):
        result = exact_background_selection(record, efficiency)
        retention[str(efficiency)] = result["signal_efficiency"] if result else None
    return dict(signal_efficiency=retention, maximum_sic=sic, maximum_counting_sensitivity=(
        float(sic * nominal) if sic is not None and nominal is not None else None))


def checkpoint_result(root, report, epoch=None):
    from .evaluation import riddle_score_scope

    index_file = Path(root) / DIRECTORY / "index.json"
    if not index_file.is_file():
        return None
    index = read_json(index_file)
    tag = index["last"] if epoch is None else f"epoch_{epoch:04d}"
    entry = index["results"].get(tag)
    if entry is None:
        return None
    destination = index_file.parent / "last" / tag / entry["revision"]
    if file_digest(destination / "result.json") != entry["report_sha256"]:
        raise ValueError(f"Checkpoint policy report changed: {destination}")
    derived = read_json(destination / "result.json")
    producer = read_json(Path(root) / "result.json") if report.get("scan_reuse") else report
    if derived["checkpoint_policy"]["source_identity"] != source_identity(producer):
        raise ValueError(f"Checkpoint scores are stale for {root}; run --backfill-checkpoint-scores")
    if epoch is not None and set(derived["checkpoint_policy"]["epoch_by_fit"].values()) != {epoch}:
        raise ValueError(f"Checkpoint policy epoch mismatch: {destination}")
    if report.get("scan_reuse"):
        derived["contract"]["inputs"] = report["contract"]["inputs"]
    derived["protocol"] = read_json(verified(destination, derived, "protocol.json"))
    derived["score_scope"] = riddle_score_scope(derived)
    derived["_full_mass_root"] = str(destination)
    return destination, derived


def _scoring_process(args, inputs):
    from .full_mass import _inference_worker_init

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    _inference_worker_init(args.torch_threads)
    backfill(args, inputs)


def score_checkpoints(args, inputs):
    if str(args.device).startswith("cuda") and os.environ.get("CUBLAS_WORKSPACE_CONFIG") != ":4096:8":
        with ProcessPoolExecutor(max_workers=1, mp_context=multiprocessing.get_context("spawn")) as pool:
            pool.submit(_scoring_process, args, inputs).result()
    else:
        backfill(args, inputs)


def backfill(args, inputs):
    from .full_mass import _rescore, identity_digest, score_sidebands
    from .production import require_complete_ensemble
    from .scan_cache import resolve_reference

    base_report = read_json(args.output / "result.json")
    source, _ = resolve_reference(args.output, base_report)
    source = source.resolve()
    report = read_json(source / "result.json")
    if (not report.get("completed") or report["contract"]["method"] != args.method
            or report["seed"] != args.seed or report["scenario"] != args.scenario):
        raise ValueError(f"Expected a completed {args.method} seed {args.seed} {args.scenario}: {source}")
    if (report["contract"]["inputs"]["files"] != inputs["files"]
            or report["contract"]["inputs"]["event_ids_sha256"] != inputs.get("event_ids_sha256")):
        raise ValueError(f"Prepared evaluation population changed: {source}")
    settings = report["contract"]["settings"]
    if settings["riddle"].get("core") != "stein_witness" or settings["riddle"].get("input_space") == "physical":
        if getattr(args, "backfill_checkpoint_scores", False):
            raise ValueError("Checkpoint scoring requires a latent Stein witness")
        return
    out = source / DIRECTORY
    out.mkdir(exist_ok=True)
    with (out / ".lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        selection = read_json(verified(source, report, "density/ensemble_selection.json"))
        require_complete_ensemble(selection)
        options = copy(args)
        options.settings = deepcopy(settings)
        target = getattr(args, "extend_witness_to", None)
        archived = archive_weights(getattr(args, "checkpoint_archive", None), source, report,
                                   selection.get("accepted_members", selection["members"]))
        include_scored_weights(out, source_identity(report), archived)
        if target:
            extend_fits(options, source, report, selection, out, target, archived)
        requested = getattr(args, "last_epoch", None) or target
        inventory, last = _inventory(source, report, selection, out, archived, args, requested)
        epochs = {name: requested or epoch for name, epoch in last.items()}
        missing = [name for name, epoch in epochs.items() if epoch not in inventory[name]]
        if missing:
            raise ValueError(f"Epoch {requested} missing for {len(missing)} fits: {', '.join(missing)}")
        plan = {name: inventory[name][epoch] for name, epoch in epochs.items()}
        levels = set(epochs.values())
        tag = f"epoch_{min(levels):04d}" if len(levels) == 1 else "final_" + "_".join(map(str, epochs.values()))
        revision = identity_digest(dict(protocol=POLICY_VERSION, source=source_identity(report),
                                        weights={n: v["sha256"] for n, v in plan.items()}))
        options.output = out / "last" / tag / revision
        options.output.mkdir(parents=True, exist_ok=True)
        scoring_cache = {}
        derived = _rescore(options, inputs, source=source, source_name=source, checkpoints=plan, scoring_cache=scoring_cache)
        score_sidebands(options, derived)
        if source_identity(read_json(source / "result.json")) != source_identity(report):
            raise ValueError(f"The producer changed while scoring its witness checkpoints: {source}")
        index_path = out / "index.json"
        index = read_json(index_path) if index_path.is_file() else dict(schema=1, last=tag, results={})
        if index.get("source_identity") != source_identity(report):
            index = dict(schema=1, source_identity=source_identity(report), last=tag, results={})
        index["results"][tag] = dict(revision=revision, report_sha256=file_digest(options.output / "result.json"), epoch_by_fit=epochs)
        previous = index["results"][index["last"]]["epoch_by_fit"]
        if all(epoch >= previous[name] for name, epoch in epochs.items()):
            index["last"] = tag
        write_json(index_path, index)
        if args.output.resolve() != source:
            write_json(args.output / DIRECTORY / "reference.json",
                       dict(source=str(source), source_identity=source_identity(report), index=str(index_path)))
        if getattr(args, "checkpoint_trends", False):
            common = set.intersection(*(set(entries) for entries in inventory.values()))
            rows_path = out / "trends.json"
            identity = source_identity(report)
            table = read_json(rows_path) if rows_path.is_file() else {}
            if table.get("source_identity") != identity:
                table = dict(schema=1, source_identity=identity, method=args.method, seed=args.seed, scenario=args.scenario,
                             minimum_background=10, minimum_background_efficiency=1e-4, fits=len(selection["members"]), epochs={})
            scheduled = set(range(10, max(last.values()) + 1, 10)) | levels
            table["missing_epochs"] = sorted(scheduled - common)
            for epoch in sorted(scheduled & common):
                step = {name: entries[epoch] for name, entries in inventory.items()}
                hashes = {name: item["sha256"] for name, item in step.items()}
                prior = table["epochs"].get(str(epoch))
                if prior and prior["checkpoint_sha256"] == hashes:
                    continue
                saved = checkpoint_result(source, report, epoch)
                if saved:
                    with np.load(saved[0] / "signal_region_scores.npz", allow_pickle=False) as data:
                        metrics = trend_metrics(data["labels"], data["mask"], data["scores"], report,
                                                settings["riddle"]["stein"]["scoring"]["auto_switch"]["efficiencies"])
                else:
                    with tempfile.TemporaryDirectory(prefix=".trend-", dir=out) as work:
                        options.output = Path(work)
                        metrics = _rescore(options, inputs, source=source, source_name=source, checkpoints=step,
                                           trend_only=True, scoring_cache=scoring_cache)
                table["epochs"][str(epoch)] = dict(**metrics, checkpoint_sha256=hashes)
                write_json(rows_path, table)
            write_json(rows_path, table)
            emit_message(f"Checkpoint trends: {len(table['epochs'])} measured epochs; unavailable {table['missing_epochs']}")
