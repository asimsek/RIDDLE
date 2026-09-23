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
from .training import train_residual, residual_scores, residual_background_log_prob, residual_background_sample
from .integrity import SCIENTIFIC_VERSION
from .production import (PRODUCTION_POLICY, EVIDENCE_GATED_POLICY, fit_acceptance,
                         require_complete_members, require_complete_ensemble)
from .options import effective_features

PROTOCOL = {
    **SINGLE_PROTOCOL,
    "name": "RIDDLE",
    "campaign_version": SCIENTIFIC_VERSION,
    "ensemble": "equal-weight mean signal density over accepted fits and ten validation-selected epochs per fit",
    "splits": "80/20 resamples of development training rows; reserved validation reserved for configuration/cuts",
    "failures": "bounded fresh retries for numerical failures only; retain inconclusive evidence without retries",
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


def assess_fit(directory, epochs, fractions, validation, device, *, sigma, normalization_tests,
               mass_conditioning=False, background_reference=None, profile_validation=None,
               fixed_coherent_fraction=None):
    """Validate selected checkpoints before any physical/test score is exported."""
    from .production import validate_density_ratio, validation_improvement

    correction_inputs = json.loads((Path(directory) / "residual_training_inputs.json").read_text())
    correction = correction_inputs.get("background_correction")
    if background_reference is not None:
        reference = np.asarray(background_reference)
    elif correction is not None:
        # Match the validated bgcorr study: sample the actual q_phi denominator
        # across the full SR context range, not a Gaussian proxy.
        contexts = np.random.default_rng(3407).uniform(-1.0, 1.0, 8192).astype(np.float32)
        reference = residual_background_sample(directory, contexts, len(contexts), 3408, device)
    else:
        reference = np.random.default_rng(3407).standard_normal((8192, validation.shape[1])).astype(np.float32)
        if mass_conditioning:
            # Gaussian latents at reserved-data masses, independent of the latents.
            # The conditional denominator has no Gaussian density factor for mass.
            reference[:, -1] = np.random.default_rng(3408).choice(validation[:, -1], len(reference))
    if reference.shape != (8192, validation.shape[1]) or not np.isfinite(reference).all():
        raise ValueError("Invalid background normalization reference")
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
    if profile_validation is not None:
        from .enhancements import profile_fraction, mixture_gain
        internal_ratio = residual_scores(directory, epochs, profile_validation, device)
        coherent_fraction = (profile_fraction(internal_ratio) if fixed_coherent_fraction is None
                             else fixed_coherent_fraction)
        ratio = residual_scores(directory, epochs, validation, device)
        mixture = mixture_gain(ratio, coherent_fraction)
        write_json(directory / "coherent_mixture.json", dict(fraction=coherent_fraction,
                   density="checkpoint density mean with weights recorded in residual_selection.json", truth_labels_used=False,
                   profiling_population="internal fit validation; not reserved evidence",
                   validation_sha256=digest(profile_validation), fixed_fraction=fixed_coherent_fraction is not None))
    ensemble_reference_ratio = residual_scores(directory, epochs, reference, device)
    normalization = dict(status="passed", reference="independent standard normal",
                         reference_seed=3407, reference_samples=len(reference), checkpoints=checks,
                         ensemble=validate_density_ratio(ensemble_reference_ratio,
                             stage=f"{directory} ensemble", tests=normalization_tests))
    if correction is not None:
        normalization.update(reference="q_phi(z|m) samples across the signal-region mass context",
                             reference_seed=3408, mass_context_seed=3407, conditional_density=True,
                             background_correction=correction["mode"],
                             background_sha256=correction["source_sha256"])
    elif mass_conditioning:
        normalization.update(reference="independent standard-normal latents at reserved-validation masses",
                             mass_context_seed=3408, conditional_density=True)
    if background_reference is not None:
        normalization.update(reference="frozen physical background flow sampled at reserved-validation masses",
                             reference_sha256=digest(reference), conditional_density=True)
    quality = validation_improvement(mixture, sigma=sigma)
    if background_reference is not None:
        quality["baseline"] = "frozen conditional background density in preprocessed physical coordinates"
    health = dict(quality=quality, normalization=normalization)
    health.update(fit_acceptance(health))
    write_json(directory / "fit_health.json", health)
    return quality, normalization, mixture


def train_member(rows, validation, output, *, relative, index, fraction, epochs, seed,
                 device, initialization, settings, label, normalization_tests, background_reference=None,
                 background_correction=None, member_split_indices=None, source_ids=None):
    """One persisted attempt budget shared by sequential and spawned workers."""
    from .production import NumericalFitError
    from .storage import locked, save_npz

    if (np.asarray(rows).ndim != 2 or np.asarray(validation).ndim != 2
            or not np.isfinite(rows).all() or not np.isfinite(validation).all()):
        raise ValueError("Invalid fit inputs; repair the input data before training, not through fit retries")
    root = Path(output) / relative
    root.mkdir(parents=True, exist_ok=True)
    policy = settings["fit_recovery"]
    a, b, first_seed = member_split(len(rows), seed, index,
                                   batch_size=settings["training"]["batch_size"])
    if member_split_indices is not None:
        from .roles import validate_member_split
        a, b = validate_member_split(member_split_indices, len(rows))
    if source_ids is not None:
        if len(source_ids) != len(rows): raise ValueError("Member source identities are misaligned")
    seeds = [first_seed] + [int(np.random.SeedSequence([seed, index, 1380275289, i]).generate_state(1)[0])
                           for i in range(1, policy["max_retries"] + 1)]
    if len(set(seeds)) != len(seeds):
        raise ValueError("Fit retry seeds collided")
    identity = dict(schema=2, scientific_version=SCIENTIFIC_VERSION, production_policy=PRODUCTION_POLICY,
                    seed=seed, fit_index=index,
                    attempt_seeds=seeds, settings=settings, fraction=fraction,
                    training_sha256=digest(rows), validation_sha256=digest(validation),
                    normalization_tests=normalization_tests)
    if background_reference is not None:
        identity["background_reference_sha256"] = digest(background_reference)
    if background_correction is not None:
        identity["background_correction"] = {
            "mode": background_correction["mode"],
            "protocol": background_correction["protocol"],
            "epochs": background_correction["epochs"],
            "selected_epoch": background_correction["selected_epoch"],
            "model_sha256": background_correction["model_sha256"],
        }
    if member_split_indices is not None:
        identity["explicit_split"] = dict(train=digest(a), validation=digest(b))
    if source_ids is not None: identity["source_ids_sha256"] = digest(source_ids)
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
                           or a.get("status") not in ("running", "completed", "numerical_failure")
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
        if source_ids is not None:
            atomic_write(root / "event_roles.npz", lambda p: save_npz(p, train_ids=source_ids[a], validation_ids=source_ids[b], train_indices=a, validation_indices=b))
        terminal = root / "fit.json"
        if state.pop("invalidated_receipt", False):
            terminal.unlink(missing_ok=True)
            write_json(state_path, state)
        # Verify every recorded attempt, including numerical failures, on resume.
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
                    progress_label=f"{label} | Attempt {number + 1}/{len(seeds)}", settings=settings,
                    background_correction=background_correction)
                weights = [history[e]["signal_fraction"] for e in order]
                quality, normalization, mixture = assess_fit(
                    directory, order, weights, validation, device,
                    sigma=policy["validation_sigma"], normalization_tests=normalization_tests,
                    mass_conditioning=settings.get("mass_conditioning", False),
                    **({"profile_validation": real_sr_latents(
                            rows[b], mass_conditioning=settings.get("mass_conditioning", False),
                            physical_inputs=settings.get("input_space") == "physical"),
                        "fixed_coherent_fraction": fraction}
                       if effective_features(settings)["coherent_mixture"] else {}),
                    **({"background_reference": background_reference} if background_reference is not None else {}))
                # Kept alongside the selected density for configuration selection, never test labels.
                from .storage import save_array
                atomic_write(directory / "validation_log_mixture_ratio.npy", lambda p: save_array(p, mixture))
                assessment = fit_acceptance(dict(quality=quality, normalization=normalization))
                if not assessment["fit_valid"]:
                    raise NumericalFitError("Incomplete or invalid fit health assessment")
                attempt.update(status="completed", **assessment,
                               epochs=order, signal_fractions=weights, quality=quality,
                               normalization=normalization)
                if "optimization" in settings:
                    attempt["trained_epochs"] = len(history)
                if assessment["evidence_status"] != "improved":
                    emit_message(f"{label} | Numerically valid; evidence {assessment['evidence_status']}; "
                                 "retained without retry", kind="INFO", level=0)
            except (FloatingPointError, NumericalFitError) as error:
                attempt.update(status="numerical_failure", error=str(error), error_type=type(error).__name__)
            attempt["artifacts_sha256"] = {
                str(p.relative_to(directory)): file_digest(p) for p in directory.rglob("*")
                if p.is_file() and ".resume" not in p.relative_to(directory).parts
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
    initialization="background", started=None, workers=1, io_workers=2, torch_threads=2, settings=None, background_reference=None,
    background_correction=None, selection_validation=None, member_splits=None, source_ids=None,
):
    settings = deepcopy(DEFAULTS["riddle"] if settings is None else settings)
    settings.update(epochs=epochs, runs=runs, initialization=initialization)
    settings = validate_residual(settings)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if min(runs, workers, io_workers, torch_threads) < 1:
        raise ValueError("Require positive runs/workers/io_workers/torch_threads")
    mass_conditioning = settings.get("mass_conditioning", False)
    physical_inputs = settings.get("input_space") == "physical"
    if physical_inputs and (background_reference is None or workers != 1):
        raise ValueError("Physical pilot requires a frozen-background reference and one fit worker")
    if background_reference is not None and not physical_inputs:
        raise ValueError("Custom background reference is only supported by the physical pilot")
    if background_correction is not None:
        if settings.get("background_correction") != "bgcorr_40_reguide":
            raise ValueError("Background-correction descriptor supplied for a disabled profile")
        if not mass_conditioning or physical_inputs:
            raise ValueError("bgcorr_40_reguide requires mapped latent inputs with mass conditioning")
    ztrain, zval = (real_sr_latents(rows, mass_conditioning=mass_conditioning,
                                  physical_inputs=physical_inputs) for rows in (train, validation))
    zselection = (None if selection_validation is None else real_sr_latents(
        selection_validation, mass_conditioning=mass_conditioning, physical_inputs=physical_inputs))
    coherent = effective_features(settings)["coherent_mixture"]
    if coherent and zselection is None:
        raise ValueError("Coherent mixture requires separate internal selection and reserved evidence populations")
    if mass_conditioning:
        rows = train[train[:, -2] == 1].copy()
        rows[:, -1] = 0  # Truth is never used by the density fit.
    else:
        rows = np.column_stack((np.full(len(ztrain), 3.5), ztrain,
                                np.ones(len(ztrain)), np.zeros(len(ztrain)))).astype(np.float32)
    identity = dict(scientific_version=SCIENTIFIC_VERSION, production_policy=PRODUCTION_POLICY,
                    training_sha256=digest(ztrain), selection_sha256=digest(zval),
                    epochs=epochs, runs=runs, seed=seed, fractions=list(fraction_values),
                    initialization=initialization, settings=settings)
    if background_reference is not None:
        identity["background_reference_sha256"] = digest(background_reference)
    if background_correction is not None:
        identity["background_correction"] = {
            "mode": background_correction["mode"], "protocol": background_correction["protocol"],
            "epochs": background_correction["epochs"], "selected_epoch": background_correction["selected_epoch"],
            "model_sha256": background_correction["model_sha256"],
        }
    if zselection is not None:
        identity["internal_selection_sha256"] = digest(zselection)
    if member_splits is not None:
        if set(member_splits) != set(range(runs)): raise ValueError("Explicit splits required for every member")
        identity["explicit_splits"] = {str(i): {k:digest(v) for k,v in split.items()} for i,split in member_splits.items()}
    if source_ids is not None: identity["source_ids_sha256"] = digest(source_ids)
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
    for job in jobs:
        job["member_split_indices"] = None if member_splits is None else member_splits[job["index"]]
    if workers > 1:
        from .parallel import run_fits
        run_fits(rows, jobs, validation=zval, epochs=epochs, seed=seed, device=device,
                 initialization=initialization, workers=workers, io_workers=io_workers, torch_threads=torch_threads,
                 total=total, started=started, settings=settings, normalization_tests=tests, source_ids=source_ids,
                 background_correction=background_correction)
    else:
        for job in jobs:
            train_member(rows, zval, output, relative=job["relative"], index=job["index"],
                         fraction=job["fraction"], epochs=epochs, seed=seed, device=device,
                         initialization=initialization, settings=settings,
                         label=f"RIDDLE fit {job['fit']}/{total}", normalization_tests=tests,
                         member_split_indices=job["member_split_indices"], source_ids=source_ids,
                         **({"background_reference": background_reference} if background_reference is not None else {}),
                         **({"background_correction": background_correction} if background_correction is not None else {}))
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
            if background_correction is not None:
                recorded = []
                for member in members:
                    inputs = json.loads((output / member["directory"] / "residual_training_inputs.json").read_text())
                    info = inputs.get("background_correction")
                    if info is None:
                        raise ValueError("Accepted corrected-background member is missing its denominator identity")
                    recorded.append(info["source_sha256"])
                if set(recorded) != {background_correction["model_sha256"]}:
                    raise ValueError("Accepted RIDDLE members do not share one coherent q_phi denominator")
            mixtures = [np.load(output / m["directory"] / "validation_log_mixture_ratio.npy") for m in members]
            combined = np.logaddexp.reduce(mixtures, axis=0) - np.log(len(members))
            selection_gain = None
            if zselection is not None:
                # A single ensemble density and coefficient, profiled on a common
                # sample withheld from every member, never on evidence/test rows.
                from .enhancements import profile_fraction, mixture_gain
                ratios = [residual_scores(output / m["directory"], m["epochs"], zselection, device)
                          for m in members]
                internal = np.logaddexp.reduce(ratios, axis=0)-np.log(len(members))
                if coherent:
                    weight = profile_fraction(internal) if config["fraction"] is None else config["fraction"]
                    config["coherent_mixture"] = dict(fraction=weight, density="equal mean over fits; per-fit checkpoint weights recorded by each residual selection",
                        validation_sha256=digest(zselection), selection_events=len(zselection), truth_labels_used=False)
                    evidence_ratio = np.logaddexp.reduce([
                        residual_scores(output / m["directory"], m["epochs"], zval, device) for m in members], axis=0)-np.log(len(members))
                    combined = mixture_gain(evidence_ratio, weight)
                    selection_gain = mixture_gain(internal, weight)
                else:
                    # Preserve checkpoint-specific fractions when coherence is disabled.
                    terms = [mixture_gain(residual_scores(output / m["directory"], [e], zselection, device), w)
                             for m in members for e, w in zip(m["epochs"], m["signal_fractions"])]
                    selection_gain = np.logaddexp.reduce(terms, axis=0)-np.log(len(terms))
            from .production import validation_improvement
            config["health"] = dict(normalization_status="passed",
                quality=validation_improvement(combined, sigma=settings["fit_recovery"]["validation_sigma"]))
            config["health"].update(fit_acceptance(config["health"]))
            selection_rows = zselection if zselection is not None else zval
            selected_background = residual_background_log_prob(
                output / members[0]["directory"], selection_rows, device)
            config["selection_nll"] = -float(np.mean(
                selected_background + (selection_gain if selection_gain is not None else combined)))
            histories = [json.loads((output / m["directory"] / "residual_losses.json").read_text())["history"]
                         for m in members]
            config["history"] = [dict(epoch=e, **{key: float(np.mean([h[e][key] for h in histories if e < len(h)]))
                for key in ("train_nll", "validation_nll", "signal_fraction")},
                **(dict(contributing_fits=sum(e < len(h) for h in histories)) if "optimization" in settings else {}))
                for e in range(max(map(len, histories)))]
        configs.append(config)
    usable = [c for c in configs if c["members"]]
    result = dict(status="completed" if usable else "no_accepted_fits", production_policy=PRODUCTION_POLICY,
                  requested_runs=runs, checkpoints_per_run=checkpoints, configurations=configs,
                  background_correction=(None if background_correction is None else {
                      "mode": background_correction["mode"], "protocol": background_correction["protocol"],
                      "epochs": background_correction["epochs"], "selected_epoch": background_correction["selected_epoch"],
                      "model_sha256": background_correction["model_sha256"]}),
                  failures=failures, fit_recovery=settings["fit_recovery"],
                  selection=("internal selection likelihood; reserved evidence never selects configurations"
                             if zselection is not None else "reserved-validation mixture likelihood; no truth or test-score selection"))
    result['ensemble_completion'] = settings.get('ensemble_completion', 'partial')
    write_json(output / "numerical_failures.json", {"failures": failures})
    if not usable:
        write_json(output / "ensemble_selection.json", result)
        from .production import IncompleteEnsembleError
        raise IncompleteEnsembleError("RIDDLE: no fits passed after bounded retries; no physics result was produced. "
                                      "Inspect density/ensemble_selection.json and the fit attempt records.")
    if settings.get("ensemble_completion", "partial") == "strict" and any(len(c["members"]) != runs for c in configs):
        from .production import IncompleteEnsembleError
        result["status"] = "incomplete_strict_ensemble"
        write_json(output / "ensemble_selection.json", result)
        raise IncompleteEnsembleError("Strict ensemble incomplete after bounded retries; artifacts preserved")
    chosen = min(usable, key=lambda c: c["selection_nll"])
    result.update(selected_configuration=chosen["name"], members=chosen["members"],
                  health=chosen["health"], production_ready=False,
                  valid_runs=len(chosen["members"]), selected_checkpoints=len(chosen["members"]) * checkpoints)
    if coherent:
        result["coherent_mixture"] = chosen["coherent_mixture"]
        write_json(output / "coherent_mixture.json", chosen["coherent_mixture"])
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
    if selection["production_policy"] in (PRODUCTION_POLICY, EVIDENCE_GATED_POLICY):
        members = []
        for member in selection["members"]:
            directory = root / member["directory"]
            verify_artifacts(directory, member["artifacts_sha256"], "Verify accepted RIDDLE density")
            health = json.loads((directory / "fit_health.json").read_text())
            expected = dict(quality=member["quality"], normalization=member["normalization"])
            if selection["production_policy"] == PRODUCTION_POLICY:
                expected.update(fit_acceptance(expected))
            if health != expected:
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
