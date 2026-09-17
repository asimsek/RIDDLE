import json
from pathlib import Path
import numpy as np
import torch
from copy import deepcopy
from .settings import DEFAULTS, validate_residual
from riddle.storage import verify_artifacts
from riddle.worker_progress import emit_message
from riddle.storage import atomic_write, write_json
from riddle.storage import digest, file_digest
from .model import PROTOCOL as SINGLE_PROTOCOL, background_log_prob, real_sr_latents
from .training import train_residual, residual_scores
from .integrity import SCIENTIFIC_VERSION
from .production import PRODUCTION_POLICY, require_complete_members, require_complete_ensemble

PROTOCOL = {
    **SINGLE_PROTOCOL,
    "name": "RIDDLE",
    "campaign_version": SCIENTIFIC_VERSION,
    "ensemble": "equal-weight mean signal density over accepted fits and ten validation-selected epochs per fit",
    "splits": "80/20 resamples of development training rows; reserved validation reserved for configuration/cuts",
    "failures": "bounded fresh retries for numerical or reserved-validation failures; exclude exhausted fits",
}


class MemberScoreError(FloatingPointError):
    def __init__(self, member, error):
        self.member = member
        super().__init__(str(error))


def reject_scoring_member(output, member, error):
    """A numerical export failure spends the same persisted fit attempt budget."""
    from .storage import locked

    output = Path(output)
    root = (output / member["directory"]).parents[1]
    with locked(root / ".resume/fit.lock"):
        state = json.loads((root / "attempts.json").read_text())
        attempt = state["attempts"][member["attempt"]]
        if attempt["status"] != "completed" or attempt["directory"] != member["directory"]:
            raise ValueError("Cannot revoke an unrecognized accepted fit")
        attempt.update(status="numerical_failure", error=str(error), stage="score_export",
                       error_type=type(error).__name__)
        state["invalidated_receipt"] = True
        write_json(root / "attempts.json", state)
        # The ledger is authoritative if a process is interrupted between these writes.
        (root / "fit.json").unlink(missing_ok=True)
    selection = json.loads((output / "ensemble_selection.json").read_text())
    selection["status"] = "incomplete"
    write_json(output / "ensemble_selection.json", selection)
    emit_message(f"RIDDLE fit {member['fit_index']:03d}: numerical scoring failure; "
                 "retry within the original attempt budget", kind="WARNING", level=0)


def fractions(values):
    result = []
    for value in values:
        fraction = None if str(value) == "learned" else float(value)
        if fraction is not None and (not np.isfinite(fraction) or not 0 < fraction < 1):
            raise ValueError("Fractions must be 'learned' or numbers in (0,1)")
        if fraction in result:
            raise ValueError("Duplicate fraction configuration")
        result.append(fraction)
    if not result:
        raise ValueError("Provide at least one signal-fraction configuration")
    return result


def member_split(n, seed, index, *, batch_size=256):
    seeds = np.random.SeedSequence([int(seed), 1380533316, index]).generate_state(2).tolist()
    order = np.random.default_rng(seeds[0]).permutation(n)
    count = int(0.8 * n)
    if count < 2 or n - count < 2:
        raise ValueError("Too few development rows for an 80/20 split")
    if count % batch_size == 1:
        count -= 1
    return (order[:count], order[count:], seeds[1])


def assess_fit(directory, epochs, fractions, validation, device, *, sigma, normalization_tests):
    """Validate selected checkpoints before any physical/test score is exported."""
    from .production import validate_density_ratio, validation_improvement

    reference = np.random.default_rng(3407).standard_normal((8192, validation.shape[1])).astype(np.float32)
    rows = np.concatenate((validation, reference))
    mixture, reference_sum, checks = None, None, []
    with torch.random.fork_rng(devices=[] if str(device) == "cpu" else None):
        for epoch, weight in zip(epochs, fractions):
            ratio = residual_scores(directory, [epoch], rows, device)
            valid, normal = ratio[:len(validation)], ratio[len(validation):]
            checks.append(dict(epoch=epoch, **validate_density_ratio(
                normal, stage=f"{directory} checkpoint {epoch}", tests=normalization_tests)))
            reference_sum = normal if reference_sum is None else np.logaddexp(reference_sum, normal)
            weighted = np.logaddexp(np.log1p(-weight) if weight < 1 else -np.inf,
                                    (np.log(weight) if weight > 0 else -np.inf) + valid)
            mixture = weighted if mixture is None else np.logaddexp(mixture, weighted)
    mixture -= np.log(len(epochs))
    normalization = dict(status="passed", reference="independent standard normal",
                         reference_seed=3407, reference_samples=len(reference), checkpoints=checks,
                         ensemble=validate_density_ratio(reference_sum - np.log(len(epochs)),
                             stage=f"{directory} ensemble", tests=normalization_tests))
    quality = validation_improvement(mixture, sigma=sigma)
    write_json(directory / "fit_health.json", dict(quality=quality, normalization=normalization))
    return quality, normalization, mixture


def train_member(rows, validation, output, *, relative, index, fraction, epochs, seed,
                 device, initialization, settings, label, normalization_tests):
    """One persisted attempt budget shared by sequential and spawned workers."""
    from .production import NumericalFitError
    from .storage import locked, save_npz

    root = Path(output) / relative
    root.mkdir(parents=True, exist_ok=True)
    policy = settings["fit_recovery"]
    a, b, first_seed = member_split(len(rows), seed, index,
                                   batch_size=settings["training"]["batch_size"])
    seeds = [first_seed] + [int(np.random.SeedSequence([seed, index, 1380275289, i]).generate_state(1)[0])
                           for i in range(1, policy["max_retries"] + 1)]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Fit retry seeds collided")
    identity = dict(schema=1, scientific_version=SCIENTIFIC_VERSION, seed=seed, fit_index=index,
                    attempt_seeds=seeds, settings=settings, fraction=fraction,
                    training_sha256=digest(rows), validation_sha256=digest(validation),
                    normalization_tests=normalization_tests)
    state_path = root / "attempts.json"
    with locked(root / ".resume/fit.lock"):
        if state_path.exists():
            state = json.loads(state_path.read_text())
            if state["identity"] != identity:
                raise ValueError("Fit inputs or retry policy changed; use a new output directory")
            attempts = state.get("attempts")
            if (not isinstance(attempts, list) or len(attempts) > len(seeds)
                    or any(a.get("attempt") != i or a.get("seed") != seeds[i]
                           or a.get("directory") != str(Path(relative) / "attempts" / f"attempt_{i:03d}")
                           or a.get("status") not in ("running", "completed", "numerical_failure", "validation_failure")
                           or (i < len(attempts) - 1 and a.get("status") in ("running", "completed"))
                           for i, a in enumerate(attempts))):
                raise ValueError("Invalid persisted fit attempt inventory")
        else:
            if any(root.glob("residual_epoch_*.pt")) or (root / "fit.json").exists():
                raise ValueError("Legacy fit cannot acquire a new selection policy through resume")
            state = dict(identity=identity, attempts=[])
            write_json(state_path, state)
        split_path = root / "split.npz"
        if split_path.exists():
            with np.load(split_path, allow_pickle=False) as saved:
                if not np.array_equal(saved["training"], a) or not np.array_equal(saved["validation"], b):
                    raise ValueError("Saved residual split changed")
        else:
            atomic_write(split_path, lambda p: save_npz(p, training=a, validation=b))
        terminal = root / "fit.json"
        if state.pop("invalidated_receipt", False):
            terminal.unlink(missing_ok=True)
            write_json(state_path, state)
        # Verify every recorded attempt, including rejected evidence, on resume.
        for attempt in state["attempts"]:
            directory = Path(output) / attempt["directory"]
            if attempt.get("artifacts_sha256") is not None:
                verify_artifacts(directory, attempt["artifacts_sha256"], f"{label} | Verify attempt")
        if terminal.exists():
            saved = json.loads(terminal.read_text())
            accepted = next((a for a in state["attempts"] if a["status"] == "completed"), None)
            expected = ({**accepted, "fit_index": index, "attempts": state["attempts"]}
                        if accepted else dict(status="excluded", fit_index=index, directory=str(relative),
                                              reason="retry_budget_exhausted", attempts=state["attempts"]))
            if saved != expected:
                raise ValueError("Fit receipt disagrees with the persisted attempt inventory")
            return saved
        for number, training_seed in enumerate(seeds):
            attempt_relative = Path(relative) / "attempts" / f"attempt_{number:03d}"
            directory = Path(output) / attempt_relative
            if number < len(state["attempts"]):
                attempt = state["attempts"][number]
                if attempt["status"] == "completed":
                    break
                if attempt["status"] != "running":
                    continue
            else:
                attempt = dict(attempt=number, seed=training_seed, status="running",
                               directory=str(attempt_relative))
                state["attempts"].append(attempt)
                write_json(state_path, state)
            directory.mkdir(parents=True, exist_ok=True)
            emit_message(f"{label} | Attempt {number + 1}/{len(seeds)}; seed={training_seed}", kind="WORK")
            checkpoint_path = directory / ".resume/latest.pt"
            checkpoint = (torch.load(checkpoint_path, map_location=device, weights_only=False)
                          if checkpoint_path.exists() else None)
            if checkpoint is not None:
                verify_artifacts(directory, checkpoint["files"], f"{label} | Verify epoch checkpoints")
            try:
                order, history = train_residual(
                    rows[a], rows[b], directory, epochs=epochs, seed=training_seed, device=device,
                    checkpoint=checkpoint, fraction=fraction, initialization=initialization,
                    progress_label=f"{label} | Attempt {number + 1}/{len(seeds)}", settings=settings)
                weights = [history[e]["signal_fraction"] for e in order]
                quality, normalization, mixture = assess_fit(
                    directory, order, weights, validation, device,
                    sigma=policy["validation_sigma"], normalization_tests=normalization_tests)
                # Kept alongside the selected density for configuration selection, never test labels.
                from .storage import save_array
                atomic_write(directory / "validation_log_mixture_ratio.npy", lambda p: save_array(p, mixture))
                attempt.update(status="completed" if quality["status"] == "passed" else "validation_failure",
                               epochs=order, signal_fractions=weights, quality=quality,
                               normalization=normalization)
                if attempt["status"] != "completed":
                    attempt["error"] = "Insufficient reserved-validation improvement over background"
            except (FloatingPointError, NumericalFitError) as error:
                attempt.update(status="numerical_failure", error=str(error), error_type=type(error).__name__)
            attempt["artifacts_sha256"] = {
                p.name: file_digest(p) for p in directory.iterdir() if p.is_file()
            }
            write_json(state_path, state)
            if attempt["status"] == "completed":
                break
            emit_message(f"{label} | Attempt {number + 1}: {attempt['status']}: {attempt['error']}; "
                         + ("restart this fit from epoch zero" if number + 1 < len(seeds)
                            else "retry budget exhausted; exclude this fit"), kind="WARNING", level=0)
        accepted = next((a for a in state["attempts"] if a["status"] == "completed"), None)
        if accepted:
            saved = {**accepted, "fit_index": index, "attempts": state["attempts"]}
        else:
            saved = dict(status="excluded", fit_index=index, directory=str(relative),
                         reason="retry_budget_exhausted", attempts=state["attempts"])
        write_json(terminal, saved)
        return saved


def train_campaign(
    train, validation, output, *, epochs=DEFAULTS["riddle"]["epochs"],
    runs=DEFAULTS["riddle"]["runs"], seed=0, device="cpu", fraction_values=(None,),
    initialization="background", started=None, workers=1, io_workers=2, settings=None,
):
    settings = deepcopy(DEFAULTS["riddle"] if settings is None else settings)
    settings.update(epochs=epochs, runs=runs, initialization=initialization)
    settings = validate_residual(settings)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if min(runs, workers, io_workers) < 1:
        raise ValueError("Require positive runs/workers/io_workers")
    ztrain, zval = real_sr_latents(train), real_sr_latents(validation)
    rows = np.column_stack((np.full(len(ztrain), 3.5), ztrain,
                            np.ones(len(ztrain)), np.zeros(len(ztrain)))).astype(np.float32)
    identity = dict(scientific_version=SCIENTIFIC_VERSION, production_policy=PRODUCTION_POLICY,
                    training_sha256=digest(ztrain), selection_sha256=digest(zval),
                    epochs=epochs, runs=runs, seed=seed, fractions=list(fraction_values),
                    initialization=initialization, settings=settings)
    identity_path = output / "ensemble_inputs.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Residual ensemble inputs/settings changed; use a new output")
    write_json(identity_path, identity)
    tags = ["learned" if f is None else "fraction_" + str(f).replace(".", "p") for f in fraction_values]
    total = runs * len(tags)
    checkpoints = settings["training"]["selected_checkpoints"]
    tests = total * (settings["fit_recovery"]["max_retries"] + 1) * (checkpoints + 1)
    jobs = [dict(fit=c * runs + i + 1, index=i, fraction=fraction_values[c],
                 relative=str(Path(tag) / f"run_{i:03d}"),
                 directory=str(output / tag / f"run_{i:03d}"))
            for c, tag in enumerate(tags) for i in range(runs)]
    if workers > 1:
        from .parallel import run_fits
        run_fits(rows, jobs, validation=zval, epochs=epochs, seed=seed, device=device,
                 initialization=initialization, workers=workers, io_workers=io_workers,
                 total=total, started=started, settings=settings, normalization_tests=tests)
    else:
        for job in jobs:
            train_member(rows, zval, output, relative=job["relative"], index=job["index"],
                         fraction=job["fraction"], epochs=epochs, seed=seed, device=device,
                         initialization=initialization, settings=settings,
                         label=f"RIDDLE fit {job['fit']}/{total}", normalization_tests=tests)
    configs, failures = [], []
    for tag in tags:
        receipts = [json.loads((output / tag / f"run_{i:03d}" / "fit.json").read_text()) for i in range(runs)]
        members = [m for m in receipts if m["status"] == "completed"]
        excluded = [m for m in receipts if m["status"] == "excluded"]
        failures.extend(dict(configuration=tag, fit_index=m["fit_index"], **attempt)
                        for m in receipts for attempt in m["attempts"] if attempt["status"] != "completed")
        config = dict(name=tag, fraction=fraction_values[tags.index(tag)], members=members,
                      excluded_fits=excluded, valid_runs=len(members))
        if members:
            require_complete_members(members, runs, checkpoints, configuration=tag, allow_excluded=True)
            mixtures = [np.load(output / m["directory"] / "validation_log_mixture_ratio.npy") for m in members]
            combined = np.logaddexp.reduce(mixtures, axis=0) - np.log(len(members))
            config["selection_nll"] = -float(np.mean(
                background_log_prob(torch.from_numpy(zval)).numpy().astype(np.float64) + combined))
            histories = [json.loads((output / m["directory"] / "residual_losses.json").read_text())["history"]
                         for m in members]
            config["history"] = [dict(epoch=e, **{key: float(np.mean([h[e][key] for h in histories]))
                for key in ("train_nll", "validation_nll", "signal_fraction")}) for e in range(epochs)]
        configs.append(config)
    usable = [c for c in configs if c["members"]]
    result = dict(status="completed" if usable else "no_accepted_fits", production_policy=PRODUCTION_POLICY,
                  requested_runs=runs, checkpoints_per_run=checkpoints, configurations=configs,
                  failures=failures, fit_recovery=settings["fit_recovery"],
                  selection="reserved-validation mixture likelihood; no truth or test-score selection")
    write_json(output / "numerical_failures.json", {"failures": failures})
    if not usable:
        write_json(output / "ensemble_selection.json", result)
        from .production import IncompleteEnsembleError
        raise IncompleteEnsembleError("RIDDLE: no fits passed after bounded retries; no physics result was produced. "
                                      "Inspect density/ensemble_selection.json and the fit attempt records.")
    chosen = min(usable, key=lambda c: c["selection_nll"])
    result.update(selected_configuration=chosen["name"], members=chosen["members"],
                  valid_runs=len(chosen["members"]), selected_checkpoints=len(chosen["members"]) * checkpoints)
    require_complete_ensemble(result)
    write_json(output / "ensemble_selection.json", result)
    write_json(output / "residual_losses.json", dict(history=chosen["history"], aggregation="mean over accepted fits"))
    emit_message(f"RIDDLE: {result['valid_runs']}/{runs} fits accepted; saved ensemble excludes exhausted fits", level=0)
    return result


def validate_normalization(output, features, device):
    """Check every selected density before evaluating data; preserve training RNG."""
    import torch
    from .production import validate_density_ratio

    root = Path(output)
    selection = json.loads((root / "ensemble_selection.json").read_text())
    require_complete_ensemble(selection)
    if selection["production_policy"] == PRODUCTION_POLICY:
        members = []
        for member in selection["members"]:
            directory = root / member["directory"]
            verify_artifacts(directory, member["artifacts_sha256"], "Verify accepted RIDDLE density")
            health = json.loads((directory / "fit_health.json").read_text())
            if health != dict(quality=member["quality"], normalization=member["normalization"]):
                raise ValueError("Selected RIDDLE fit health disagrees with its acceptance receipt")
            members.append(dict(directory=member["directory"], **health["normalization"]))
        audit = dict(schema=1, status="passed", features=features, members=members,
                     source="per-attempt checks before acceptance; verified artifact hashes")
        write_json(root / "normalization_check.json", audit)
        return audit
    seed, count = 3407, 8192
    reference = np.random.default_rng(seed).standard_normal((count, features)).astype(np.float32)
    members = selection["members"]
    tests = sum(len(m["epochs"]) + 1 for m in members)
    audit = dict(schema=1, status="checking", reference="independent standard normal",
                 reference_seed=seed, reference_samples=count, features=features,
                 familywise_false_rejection_bound=1e-12, members=[])
    try:
        with torch.random.fork_rng(devices=[] if str(device) == "cpu" else None):
            for member in members:
                checkpoints = []
                scores = residual_scores(root / member["directory"], member["epochs"], reference, device,
                                         normalization_checks=checkpoints, normalization_tests=tests)
                audit["members"].append(dict(directory=member["directory"], checkpoints=checkpoints,
                    ensemble=validate_density_ratio(scores, stage=f"RIDDLE {member['directory']} ensemble", tests=tests)))
    except (ValueError, FloatingPointError) as error:
        audit.update(status="failed", error=str(error))
        write_json(root / "normalization_check.json", audit)
        raise
    audit["status"] = "passed"
    write_json(root / "normalization_check.json", audit)
    return audit


def ensemble_predict(output, z, device, *, return_members=False):
    root = Path(output)
    selection = json.loads((root / "ensemble_selection.json").read_text())
    require_complete_ensemble(selection)
    combined = None
    predictions = []
    for member in selection["members"]:
        from .production import NumericalFitError
        try:
            ratio = residual_scores(root / member["directory"], member["epochs"], z, device)
        except (FloatingPointError, NumericalFitError) as error:
            raise MemberScoreError(member, error) from error
        combined = ratio if combined is None else np.logaddexp(combined, ratio)
        if return_members:
            predictions.append(ratio)
    scores = combined - np.log(len(selection["members"]))
    return (scores, np.stack(predictions)) if return_members else scores
