"""Production validation without clipping, repairing, or selecting on test scores."""

import numpy as np

SR_BOUNDS = (3.3, 3.7)
PRODUCTION_POLICY = {
    "ensemble": "numerically_valid_fits_after_bounded_retries_v2",
    "signal_region": "strict_prepared_input_membership",
    "signal_region_bounds": list(SR_BOUNDS),
}
EVIDENCE_GATED_POLICY = {**PRODUCTION_POLICY, "ensemble": "accepted_fits_after_bounded_retries_v1"}
FIT_ACCEPTANCE_POLICY = "numerical_validity_with_diagnostic_evidence_v2"


class IncompleteEnsembleError(RuntimeError):
    pass


class NumericalFitError(ValueError):
    """A model output is unusable; input corruption is not a fit failure."""


def validation_improvement(log_mixture_ratios, *, sigma):
    """Paired validation gain against the fixed background, without truth labels.

    This is a label-free model-selection diagnostic, not a fit-retry criterion or calibrated discovery test.
    """
    values = np.asarray(log_mixture_ratios, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2:
        raise ValueError("Fit validation requires at least two reserved events")
    if not np.isfinite(values).all():
        raise NumericalFitError("Nonfinite reserved-validation mixture likelihood")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Invalid validation improvement threshold")
    gain = float(values.mean())
    error = float(values.std(ddof=1) / np.sqrt(len(values)))
    margin = gain - sigma * error
    if not np.isfinite([gain, error, margin]).all():
        raise NumericalFitError("Nonfinite validation improvement statistics")
    tolerance = 10 * np.finfo(np.float32).eps
    evidence = ("improved" if margin > tolerance else
                "deteriorated" if gain + sigma * error < -tolerance else "inconclusive")
    return dict(status="passed" if margin > tolerance else "insufficient_validation_improvement",
                evidence_status=evidence, upper_margin=gain + sigma * error,
                events=len(values), mean_log_likelihood_gain=gain,
                standard_error=error, sigma=float(sigma), lower_margin=margin, numerical_tolerance=float(tolerance),
                baseline="standard-normal background", truth_labels_used=False,
                interpretation="label-free validation metric; may rank ensemble fits; not evidence of a physical signal")




def validation_witness_improvement(data_scores, reference_scores, *, sigma):
    data = np.asarray(data_scores, dtype=np.float64)
    reference = np.asarray(reference_scores, dtype=np.float64)
    if data.ndim != 1 or reference.ndim != 1 or len(data) < 2 or len(reference) < 2:
        raise ValueError("Stein validation requires data and reference score samples")
    if not np.isfinite(data).all() or not np.isfinite(reference).all():
        raise NumericalFitError("Nonfinite reserved-validation Stein witness scores")
    if not np.isfinite(sigma) or sigma < 0:
        raise ValueError("Invalid validation improvement threshold")
    gain = float(data.mean() - reference.mean())
    error = float(np.sqrt(data.var(ddof=1) / len(data) + reference.var(ddof=1) / len(reference)))
    margin = gain - sigma * error
    if not np.isfinite([gain, error, margin]).all():
        raise NumericalFitError("Nonfinite Stein validation statistics")
    tolerance = 10 * np.finfo(np.float32).eps
    evidence = ("improved" if margin > tolerance else
                "deteriorated" if gain + sigma * error < -tolerance else "inconclusive")
    return dict(
        status="passed" if margin > tolerance else "insufficient_validation_improvement",
        evidence_status=evidence, upper_margin=gain + sigma * error,
        events=len(data), reference_events=len(reference), mean_witness_gain=gain,
        standard_error=error, sigma=float(sigma), lower_margin=margin,
        numerical_tolerance=float(tolerance), baseline="fixed latent background q_B(z|m)",
        truth_labels_used=False,
        interpretation="label-free held-out Stein witness discrepancy; may rank ensemble fits; not evidence of a physical signal",
    )

def select_ensemble_members(members, *, mode, fit_count):
    if mode not in ("validation-best", "all", "all-valid"):
        raise ValueError("ensemble fit selection must be validation-best, all-valid, or all")
    if type(fit_count) is not int or fit_count < 1:
        raise ValueError("ensemble fit count must be a positive integer")
    if not isinstance(members, list):
        raise ValueError("ensemble members must be a list")
    ranking = []
    for member in members:
        quality = member.get("quality", {}) or {}
        metric = "mean_witness_gain" if "mean_witness_gain" in quality else "mean_log_likelihood_gain"
        gain = quality.get(metric)
        if type(member.get("fit_index")) is not int or type(gain) not in (int, float) or not np.isfinite(gain):
            raise ValueError("ensemble fit selection requires a finite label-free validation metric for every fit")
        error = quality.get("standard_error")
        if type(error) not in (int, float) or not np.isfinite(error):
            raise ValueError("ensemble fit selection requires finite validation uncertainty for every fit")
        ranking.append(dict(
            fit_index=member["fit_index"],
            selection_metric=metric,
            selection_gain=float(gain),
            standard_error=float(error),
            directory=member.get("directory"),
        ))
    metrics = {row["selection_metric"] for row in ranking}
    if len(metrics) != 1:
        raise ValueError("Cannot rank ensemble fits with mixed validation metrics")
    metric = next(iter(metrics))
    ordered = sorted(ranking, key=lambda row: (-row["selection_gain"], row["fit_index"]))
    if mode in ("all", "all-valid"):
        chosen = {row["fit_index"] for row in ranking}
    else:
        chosen = {row["fit_index"] for row in ordered[:min(fit_count, len(ordered))]}
    selected = [member for member in members if member["fit_index"] in chosen]
    unselected = [member for member in members if member["fit_index"] not in chosen]
    metadata = dict(
        mode=mode,
        fit_count=int(fit_count),
        candidate_fits=len(members),
        selected_fits=len(selected),
        ranking_metric="reserved_validation_" + metric,
        ranking_direction="descending",
        truth_labels_used=False,
        selected_fit_indices=[member["fit_index"] for member in selected],
        selected_fit_indices_by_rank=[row["fit_index"] for row in ordered if row["fit_index"] in chosen],
        unselected_fit_indices=[member["fit_index"] for member in unselected],
        ranking=ordered,
    )
    return selected, unselected, metadata


def fit_acceptance(health):
    """Separate usable densities from evidence and deployment decisions.

    Derive the current assessment from numerical/quality receipts, including old
    receipts, without trusting their combined production_guard_would_accept flag.
    Reading an old receipt does not restore previously discarded fits or change
    any historical training selection. Calibration must be assessed separately.
    """
    normal = health.get("normalization_status", health.get("normalization", {}).get("status"))
    quality = health.get("quality", {}) or {}
    evidence = "unavailable"
    try:
        metric = "mean_witness_gain" if "mean_witness_gain" in quality else "mean_log_likelihood_gain"
        gain, error, sigma, tolerance = (float(quality[k]) for k in
            (metric, "standard_error", "sigma", "numerical_tolerance"))
        if np.isfinite([gain, error, sigma, tolerance]).all() and min(error, sigma, tolerance) >= 0:
            evidence = ("improved" if gain - sigma * error > tolerance else
                        "deteriorated" if gain + sigma * error < -tolerance else "inconclusive")
    except (KeyError, TypeError, ValueError):
        pass
    valid = (normal == "passed" and health.get("normalization", {}).get("status", "passed") == "passed"
             and not health.get("failures") and evidence != "unavailable")
    calibration = health.get("calibration_status", "unassessed")
    deployment = ("blocked_numerical_or_incomplete" if not valid else
                  "blocked_validation_deterioration" if evidence == "deteriorated" else
                  "blocked_calibration" if calibration == "failed" else
                  "requires_search_validation" if calibration == "passed" else
                  "requires_calibration_and_search_validation")
    return dict(acceptance_policy=FIT_ACCEPTANCE_POLICY, fit_valid=valid,
                fit_status="valid_" + evidence if valid else "invalid_or_incomplete",
                evidence_status=evidence, calibration_status=calibration,
                deployment_status=deployment, production_ready=False,
                # This alias records numerical acceptance, not production certification.
                production_guard_would_accept=valid)


def fit_status_label(health):
    assessment = fit_acceptance(health)
    return ("valid / " + assessment["evidence_status"] if assessment["fit_valid"]
            else "invalid or incomplete")


def score_diagnostics(scores, *, stage, probability=False, summarize=True):
    """Check numerical validity, not discrimination; a weak classifier is valid."""
    scores = np.asarray(scores)
    if scores.ndim != 1 or not scores.size:
        raise ValueError(f"{stage}: empty or malformed accepted scores")
    if not np.isfinite(scores).all():
        raise NumericalFitError(f"{stage}: nonfinite accepted scores")
    if probability and ((scores < 0).any() or (scores > 1).any()):
        raise NumericalFitError(f"{stage}: classifier probabilities outside [0, 1]")
    if not summarize:
        return None
    summary = dict(events=len(scores), minimum=float(scores.min()), maximum=float(scores.max()),
                   quantiles=np.quantile(scores, [.01, .16, .5, .84, .99]).tolist())
    summary["warnings"] = ["All accepted scores are identical"] if summary["minimum"] == summary["maximum"] else []
    if probability:
        summary["saturated_fraction"] = float(np.mean((scores == 0) | (scores == 1)))
    return summary


def scoring_masks(preprocessing_mask, region, score_scope):
    """Separate mapping acceptance from intentional score-domain eligibility."""
    preprocessing_mask = np.asarray(preprocessing_mask)
    if preprocessing_mask.ndim != 1 or preprocessing_mask.dtype != bool:
        raise ValueError("Preprocessing acceptance must be a one-dimensional boolean mask")
    region = validate_region(region, len(preprocessing_mask))
    if score_scope not in ("signal_region", "full_region"):
        raise ValueError("Unrecognized score scope")
    domain = region.copy() if score_scope == "signal_region" else np.ones_like(region)
    return preprocessing_mask.copy(), domain, preprocessing_mask & domain


def validate_score_record(record, *, method, stage):
    """Shared producer/consumer check, including members before any averaging."""
    mask, scores = np.asarray(record["mask"]), np.asarray(record["scores"])
    if mask.dtype != bool or mask.ndim != 1 or scores.shape != mask.shape:
        raise ValueError(f"{stage}: invalid score/mapping shapes")
    n = len(mask)
    scope_keys = ("preprocessing_mask", "score_domain_mask", "score_scope")
    if any(key in record for key in scope_keys):
        if not all(key in record for key in (*scope_keys, "is_signal_region")):
            raise ValueError(f"{stage}: incomplete preprocessing/score-domain metadata")
        scope_value = np.asarray(record["score_scope"])
        if scope_value.ndim != 0:
            raise ValueError(f"{stage}: score scope must be scalar")
        pre, domain, expected = scoring_masks(record["preprocessing_mask"],
                                             record["is_signal_region"], str(scope_value.item()))
        saved_domain = np.asarray(record["score_domain_mask"])
        if (pre.shape != mask.shape or saved_domain.shape != mask.shape
                or saved_domain.dtype != bool or not np.array_equal(saved_domain, domain)
                or not np.array_equal(mask, expected)):
            raise ValueError(f"{stage}: score mask must equal preprocessing acceptance AND score domain")
    if "is_signal_region" in record:
        validate_region(record["is_signal_region"], n)
    if "event_ids" in record:
        event_ids = np.asarray(record["event_ids"])
        valid_ids = ((event_ids.shape == (n,) and event_ids.dtype.kind in "uiUS")
                     or (event_ids.shape == (n, 2) and event_ids.dtype == np.uint64))
        if not valid_ids or len(np.unique(event_ids, axis=0)) != n:
            raise ValueError(f"{stage}: invalid or duplicate evaluation event identities")
    for key in ("mass", "labels"):
        if np.asarray(record[key]).shape != (n,):
            raise ValueError(f"{stage}: misaligned {key}")
    if not np.isin(record["labels"], [0, 1]).all():
        raise ValueError(f"{stage}: invalid labels")
    for key in ("mass", "physical", "latent", "fit_latents", "density_inputs", "background_log_density"):
        if key in record and not np.isfinite(record[key]).all():
            raise ValueError(f"{stage}: nonfinite {key}")
    if "physical" in record and (record["physical"].ndim != 2 or len(record["physical"]) != n):
        raise ValueError(f"{stage}: misaligned physical features")
    if "latent" in record and (record["latent"].ndim != 2 or len(record["latent"]) != mask.sum()):
        raise ValueError(f"{stage}: misaligned accepted latents")
    if "density_inputs" in record and (record["density_inputs"].ndim != 2 or len(record["density_inputs"]) != mask.sum()):
        raise ValueError(f"{stage}: misaligned accepted density inputs")
    if "background_log_density" in record and record["background_log_density"].shape != (int(mask.sum()),):
        raise ValueError(f"{stage}: misaligned accepted background density")
    if not np.isnan(scores[~mask]).all():
        raise ValueError(f"{stage}: rejected-event scores must be NaN")
    score_kind = str(np.asarray(record["score_kind"]).item()) if "score_kind" in record else None
    if method in ("iad", "supervised"):
        if score_kind != "classifier_probability":
            raise ValueError(f"{stage}: AD baseline requires classifier_probability scores")
        if str(np.asarray(record.get("fit_score_kind", "")).item()) != "classifier_probability":
            raise ValueError(f"{stage}: AD baseline requires classifier_probability fit scores")
    elif score_kind == "classifier_probability":
        raise ValueError(f"{stage}: classifier_probability is reserved for AD baselines")
    probability = method in ("lacathode", "iad", "supervised")
    summary = score_diagnostics(scores[mask], stage=stage, probability=probability)
    if score_kind is not None:
        if score_kind not in ("log_density_ratio", "background_percentile_logit", "stein_witness", "classifier_probability") \
                and not score_kind.startswith("stein_"):
            raise ValueError(f"{stage}: unrecognized score coordinate")
    if "raw_scores" in record:
        raw = np.asarray(record["raw_scores"])
        if raw.shape != (n,) or not np.isnan(raw[~mask]).all():
            raise ValueError(f"{stage}: invalid raw density-score alignment")
        score_diagnostics(raw[mask], stage=stage+" raw score")
    summary["rejected_events"] = int((~mask).sum())
    if "preprocessing_mask" in record:
        pre = np.asarray(record["preprocessing_mask"])
        domain = np.asarray(record["score_domain_mask"])
        summary.update(preprocessing_rejected_events=int((~pre).sum()),
                       outside_score_domain_events=int((~domain).sum()),
                       mapped_but_outside_score_domain_events=int((pre & ~domain).sum()),
                       score_scope=str(np.asarray(record["score_scope"]).item()))
    if "fit_scores" in record:
        fits = record["fit_scores"]
        if fits.ndim != 2 or fits.shape[1:] != (n,) or not len(fits) or not np.isnan(fits[:, ~mask]).all():
            raise ValueError(f"{stage}: invalid saved member scores")
        summary["members"] = [score_diagnostics(fit[mask], stage=f"{stage} member {i}",
                                               probability=probability)
                              for i, fit in enumerate(fits)]
        if method in ("iad", "supervised"):
            for key in ("accepted_fit_indices", "accepted_fit_seeds"):
                if key not in record or np.asarray(record[key]).shape != (len(fits),):
                    raise ValueError(f"{stage}: invalid {key}")
            expected = np.mean(fits[:, mask], axis=0, dtype=np.float64)
            if not np.allclose(scores[mask], expected, rtol=1e-6, atol=1e-7):
                raise ValueError(f"{stage}: ensemble score disagrees with mean fit probability")
    if "accepted_fit_scores" in record:
        accepted = record["accepted_fit_scores"]
        if (accepted.ndim != 2 or accepted.shape[1:] != (n,) or not len(accepted)
                or not np.isfinite(accepted[:, mask]).all() or not np.isnan(accepted[:, ~mask]).all()):
            raise ValueError(f"{stage}: invalid saved safeguard-valid fit scores")
        for key in ("accepted_fit_indices", "accepted_fit_seeds", "accepted_fit_directories"):
            if key in record and np.asarray(record[key]).shape != (len(accepted),):
                raise ValueError(f"{stage}: invalid {key}")
        summary["accepted_members"] = len(accepted)
    return summary


def validate_result_scores(root, method):
    from pathlib import Path

    partitions = {}
    for name in ("validation", "test", "signal_region"):
        with np.load(Path(root) / f"{name}_scores.npz", allow_pickle=False) as archive:
            record = {key: archive[key] for key in archive.files}
            partitions[name] = validate_score_record(record, method=method, stage=f"{root}/{name}")
    return {"schema": 1, "method": method, "status": "passed", "partitions": partitions}


def validate_density_ratio(log_ratios, *, stage, tests=1):
    """One-sided normalization test on independent samples from the denominator.

    For normalized p/q, E_q[p/q]=1, so P_q(p/q >= t) <= 1/t.
    The binomial survival bound detects inflated finite ratios without assuming
    a finite variance or rejecting a valid, concentrated signal density. This
    must only be called on reference draws, never on observed data or labels.
    """
    from scipy.stats import binom

    log_ratios = np.asarray(log_ratios)
    summary = score_diagnostics(log_ratios, stage=stage)
    thresholds = (100., 1000., 10000., 1000000.)
    if type(tests) is not int or tests < 1:
        raise ValueError("Normalization multiplicity must be a positive integer")
    cutoff = np.log(1e-12 / (len(thresholds) * tests))
    checks = []
    for threshold in thresholds:
        count = int(np.count_nonzero(log_ratios >= np.log(threshold)))
        log_p = float(binom.logsf(count - 1, len(log_ratios), 1 / threshold))
        checks.append(dict(ratio_threshold=threshold, exceedances=count,
                           log_p_upper_bound=log_p if np.isfinite(log_p) else None))
        if log_p < cutoff:
            raise NumericalFitError(f"{stage}: density-ratio normalization failed on independent denominator reference "
                             f"draws ({count}/{len(log_ratios)} ratios >= {threshold:g}). "
                             "Inspect the flow/checkpoint; this density cannot be used for scoring.")
    summary["normalization_checks"] = checks
    return summary


def strict_signal_region(mass):
    mass = np.asarray(mass)
    return (mass > SR_BOUNDS[0]) & (mass < SR_BOUNDS[1])


def evaluation_rows(arrays, names):
    """Keep source membership; downcast only the model's numerical inputs."""
    if not arrays or len(arrays) != len(names):
        raise ValueError("Missing or misaligned RIDDLE evaluation sources")
    regions = []
    for array, name in zip(arrays, names):
        if not (name.startswith("innerdata_") or name.startswith("outerdata_")):
            raise ValueError("Unknown RIDDLE evaluation region")
        region = np.full(len(array), name.startswith("innerdata_"), dtype=bool)
        if not np.array_equal(region, strict_signal_region(array[:, 0])):
            raise ValueError("Prepared RIDDLE SR membership disagrees with the original masses")
        regions.append(region)
    return np.vstack(arrays).astype("float32"), np.concatenate(regions)


def validate_region(region, n):
    region = np.asarray(region)
    if region.shape != (n,) or region.dtype != bool:
        raise ValueError("RIDDLE requires one boolean SR-membership flag per original event")
    return region


def region_acceptance(labels, mask, region):
    from .metrics import acceptance_report
    labels, mask = np.asarray(labels), np.asarray(mask)
    region = validate_region(region, len(labels))
    return {
        "full": acceptance_report(labels, mask)["full"],
        "signal_region": acceptance_report(labels[region], mask[region])["full"],
    }


def require_complete_members(members, requested, checkpoints, *, configuration, allow_excluded=False):
    if type(requested) is not int or requested < 1 or type(checkpoints) is not int or checkpoints < 1:
        raise IncompleteEnsembleError("Missing RIDDLE ensemble size/checkpoint requirements")
    directories = [m.get("directory") for m in members]
    if (not 0 < len(members) <= requested or (not allow_excluded and len(members) != requested)
            or any(not isinstance(d, str) or not d for d in directories)
            or len(set(directories)) != len(members)):
        raise IncompleteEnsembleError(
            f"RIDDLE configuration {configuration}: {len(members)}/{requested} valid fits; "
            "no usable ensemble can be finalized. Checkpoints and failure records are preserved; "
            "diagnose failed fits before finalizing production."
        )
    for member in members:
        epochs, fractions = member.get("epochs", []), member.get("signal_fractions", [])
        if (member.get("status") != "completed" or len(epochs) != checkpoints
                or len(fractions) != checkpoints or len(set(epochs)) != checkpoints
                or any(type(e) is not int or e < 0 for e in epochs)
                or not np.isfinite(fractions).all()
                or any(f < 0 or f > 1 for f in fractions)):
            raise IncompleteEnsembleError(
                f"RIDDLE configuration {configuration}: incomplete or invalid member {member.get('directory')}"
            )


def require_complete_ensemble(selection):
    completion = selection.get("ensemble_completion", "partial")
    if completion not in ("strict", "partial"):
        raise IncompleteEnsembleError("Unrecognized ensemble completion policy")
    policy = selection.get("production_policy")
    legacy = policy == {**PRODUCTION_POLICY, "ensemble": "all_requested_runs"}
    evidence_gated = policy == EVIDENCE_GATED_POLICY
    if selection.get("status") != "completed" or (not legacy and not evidence_gated and policy != PRODUCTION_POLICY):
        raise IncompleteEnsembleError("RIDDLE ensemble is not finalized under a recognized production policy")
    requested = selection.get("requested_runs")
    checkpoints = selection.get("checkpoints_per_run")
    configs = selection.get("configurations", [])
    if not configs or (legacy and selection.get("failures")):
        raise IncompleteEnsembleError("RIDDLE ensemble has failed or missing configurations")

    fit_selection_enabled = isinstance(selection.get("ensemble_fit_selection"), dict) or any(
        "accepted_members" in config for config in configs
    )
    if not fit_selection_enabled:
        for config in configs:
            members = config.get("members", [])
            if completion == "strict" and len(members) != requested:
                raise IncompleteEnsembleError("Strict ensemble has missing requested fits")
            if members or legacy:
                require_complete_members(members, requested, checkpoints,
                                         configuration=config.get("name"), allow_excluded=not legacy)
            if config.get("valid_runs") != len(members):
                raise IncompleteEnsembleError("RIDDLE configuration has an inconsistent accepted-fit count")
            if not legacy:
                excluded = config.get("excluded_fits", [])
                indices = [m.get("fit_index") for m in [*members, *excluded]]
                if (type(requested) is not int or any(type(i) is not int for i in indices)
                        or sorted(indices) != list(range(requested))
                        or any(m.get("status") != "excluded" for m in excluded)
                        or (evidence_gated and any(m.get("quality", {}).get("status") != "passed" for m in members))
                        or (not evidence_gated and any(not fit_acceptance(m)["fit_valid"] for m in members))
                        or any(m.get("normalization", {}).get("status") != "passed" for m in members)):
                    raise IncompleteEnsembleError("RIDDLE fit acceptance/exclusion inventory is inconsistent")
        chosen = [c for c in configs if c.get("name") == selection.get("selected_configuration")]
        members = selection.get("members", [])
        if (len(chosen) != 1 or not members or members != chosen[0]["members"]
                or selection.get("valid_runs") != len(members)
                or selection.get("selected_checkpoints") != len(members) * checkpoints):
            raise IncompleteEnsembleError("RIDDLE ensemble selection is incomplete or inconsistent")
        return

    if type(requested) is not int or requested < 1 or type(checkpoints) is not int or checkpoints < 1:
        raise IncompleteEnsembleError("RIDDLE ensemble size/checkpoint requirements are invalid")
    for config in configs:
        selected = config.get("members", [])
        accepted = config.get("accepted_members", selected)
        unselected = config.get("unselected_members", [])
        excluded = config.get("excluded_fits", [])
        policy_info = config.get("ensemble_fit_selection")
        if not isinstance(policy_info, dict):
            raise IncompleteEnsembleError("RIDDLE ensemble fit selection metadata is missing")
        if completion == "strict" and len(accepted) != requested:
            raise IncompleteEnsembleError("Strict ensemble has missing requested fits")
        if accepted:
            require_complete_members(accepted, requested, checkpoints,
                                     configuration=config.get("name"), allow_excluded=not legacy)
        accepted_indices = [m.get("fit_index") for m in accepted]
        selected_indices = [m.get("fit_index") for m in selected]
        unselected_indices = [m.get("fit_index") for m in unselected]
        excluded_indices = [m.get("fit_index") for m in excluded]
        if (any(type(i) is not int for i in [*accepted_indices, *excluded_indices])
                or sorted([*accepted_indices, *excluded_indices]) != list(range(requested))
                or any(m.get("status") != "excluded" for m in excluded)
                or config.get("accepted_runs") != len(accepted)
                or config.get("valid_runs") != len(selected)
                or sorted([*selected_indices, *unselected_indices]) != sorted(accepted_indices)
                or len(set(selected_indices).intersection(unselected_indices)) != 0
                or any(not fit_acceptance(m)["fit_valid"] for m in accepted)
                or any(m.get("normalization", {}).get("status") != "passed" for m in accepted)):
            raise IncompleteEnsembleError("RIDDLE fit acceptance/selection inventory is inconsistent")
        expected_selected, expected_unselected, expected_info = select_ensemble_members(
            accepted, mode=policy_info.get("mode"), fit_count=policy_info.get("fit_count")
        )
        if ([m["fit_index"] for m in selected] != [m["fit_index"] for m in expected_selected]
                or [m["fit_index"] for m in unselected] != [m["fit_index"] for m in expected_unselected]
                or policy_info.get("selected_fit_indices") != expected_info["selected_fit_indices"]
                or policy_info.get("unselected_fit_indices") != expected_info["unselected_fit_indices"]
                or policy_info.get("ranking_metric") != expected_info["ranking_metric"]
                or policy_info.get("truth_labels_used") is not False):
            raise IncompleteEnsembleError("RIDDLE label-free ensemble fit selection is inconsistent")

    chosen = [c for c in configs if c.get("name") == selection.get("selected_configuration")]
    members = selection.get("members", [])
    accepted = selection.get("accepted_members", [])
    unselected = selection.get("unselected_members", [])
    if (len(chosen) != 1 or not members or members != chosen[0]["members"]
            or accepted != chosen[0]["accepted_members"] or unselected != chosen[0]["unselected_members"]
            or selection.get("ensemble_fit_selection") != chosen[0]["ensemble_fit_selection"]
            or selection.get("accepted_runs") != len(accepted)
            or selection.get("valid_runs") != len(members)
            or selection.get("accepted_checkpoints") != len(accepted) * checkpoints
            or selection.get("selected_checkpoints") != len(members) * checkpoints):
        raise IncompleteEnsembleError("RIDDLE ensemble fit selection is incomplete or inconsistent")

CLASSIFIER_SCORE_PARTITIONS = {
    "validation": ("innerdata_val.npy", "outerdata_val.npy"),
    "test": ("innerdata_test.npy", "outerdata_test.npy"),
    "signal_region": ("innerdata_test.npy", "innerdata_extrabkg_test.npy", "innerdata_extrasig.npy"),
}


def load_classifier_preprocessing(training_root):
    from pathlib import Path

    with np.load(Path(training_root) / "preprocessing.npz", allow_pickle=False) as archive:
        mean = archive["mean"]
        std = archive["std"]
        order = archive["feature_order"].astype(str).tolist()
        include_mass = bool(archive["include_mass"].item())
        input_dimension = int(archive["input_dimension"].item())
    if len(order) != input_dimension or len(mean) != input_dimension or len(std) != input_dimension:
        raise ValueError("Invalid AD baseline preprocessing artifact")
    return mean, std, order, include_mass


def classifier_evaluation_population(data_root, names, event_ids):
    from pathlib import Path

    arrays = [np.load(Path(data_root) / name, mmap_mode="r", allow_pickle=False) for name in names]
    event_rows, region = evaluation_rows(arrays, names)
    identities = np.concatenate([event_ids[name] for name in names])
    return event_rows, region, identities


def classifier_standardized_features(event_rows, mean, std, include_mass):
    from .data import baseline_feature_matrix

    values = baseline_feature_matrix(event_rows, include_mass)
    values = (values - mean) / std
    if not np.isfinite(values).all():
        raise ValueError("AD baseline evaluation preprocessing produced nonfinite features")
    return np.asarray(values, dtype=np.float32)


def predict_classifier_checkpoint(model, features, batch_size, device):
    import torch

    outputs = []
    model.eval()
    with torch.no_grad():
        for start in range(0, len(features), batch_size):
            batch = torch.from_numpy(features[start:start + batch_size]).to(device)
            outputs.append(torch.sigmoid(model(batch).reshape(-1)).cpu().numpy())
    result = np.concatenate(outputs) if outputs else np.empty(0, dtype=np.float32)
    if not np.isfinite(result).all() or (result < 0).any() or (result > 1).any():
        raise ValueError("AD baseline classifier produced invalid probabilities")
    return result


def classifier_fit_predictions(training_root, receipt, features, classifier, device):
    from pathlib import Path
    import torch
    from .storage import file_digest
    from .training import build_binary_classifier

    fit_root = Path(training_root) / f"fit_{receipt['fit_index']:03d}" / receipt["attempt_directory"]
    predictions = np.zeros(len(features), dtype=np.float64)
    selected = receipt["selected_checkpoints"]
    for candidate in selected:
        checkpoint = fit_root / candidate["filename"]
        if file_digest(checkpoint) != candidate["sha256"]:
            raise ValueError("AD baseline selected checkpoint changed before scoring")
        payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
        if int(payload["epoch"]) != int(candidate["epoch"]) or float(payload["validation_bce"]) != float(candidate["validation_bce"]):
            raise ValueError("AD baseline checkpoint metadata changed")
        model = build_binary_classifier(features.shape[1], classifier).to(device)
        model.load_state_dict(payload["model"])
        predictions += predict_classifier_checkpoint(model, features, classifier["inference_batch_size"], device)
        del model
    predictions /= len(selected)
    return predictions.astype(np.float32)


def classifier_checkpoint_signature(receipt):
    return [
        {
            "epoch": int(candidate["epoch"]),
            "validation_bce": float(candidate["validation_bce"]),
            "filename": candidate["filename"],
            "sha256": candidate["sha256"],
        }
        for candidate in receipt["selected_checkpoints"]
    ]


def classifier_member_signature(receipt):
    return {
        "fit_index": int(receipt["fit_index"]),
        "fit_seed": int(receipt["fit_seed"]),
        "attempt": int(receipt["attempt"]),
        "attempt_directory": receipt["attempt_directory"],
        "selected_checkpoints": classifier_checkpoint_signature(receipt),
    }


def load_or_predict_classifier_member(scoring_root, training_root, receipt, features, classifier, device, resume):
    import json
    from pathlib import Path
    from .storage import file_digest, save_array, write_json

    signature = classifier_member_signature(receipt)
    path = Path(scoring_root) / f"fit_{int(receipt['fit_index']):03d}.npy"
    receipt_path = path.with_suffix(".json")
    if resume and path.is_file() and receipt_path.is_file():
        saved = json.loads(receipt_path.read_text())
        if saved.get("member") == signature and saved.get("prediction_sha256") == file_digest(path):
            predictions = np.load(path, allow_pickle=False)
            if predictions.shape == (len(features),) and np.isfinite(predictions).all() and (predictions >= 0).all() and (predictions <= 1).all():
                return np.asarray(predictions, dtype=np.float32)
    predictions = classifier_fit_predictions(training_root, receipt, features, classifier, device)
    save_array(path, predictions)
    write_json(receipt_path, {
        "schema": 1,
        "member": signature,
        "prediction_sha256": file_digest(path),
        "events": int(len(predictions)),
    })
    return predictions


def score_classifier_partitions(method, data_root, output, training_root, classifier, accepted_receipts, device, resume=False):
    import json
    from pathlib import Path
    import torch
    from .storage import atomic_write, digest, file_digest, save_npz, write_json

    data_root = Path(data_root)
    output = Path(output)
    with np.load(data_root / "event_ids.npz", allow_pickle=False) as archive:
        event_ids = {name: archive[name] for name in archive.files}
    mean, std, feature_order, include_mass = load_classifier_preprocessing(training_root)
    prepared = {}
    lengths = []
    population_signatures = {}
    for name, files in CLASSIFIER_SCORE_PARTITIONS.items():
        event_rows, region, identities = classifier_evaluation_population(data_root, files, event_ids)
        features = classifier_standardized_features(event_rows, mean, std, include_mass)
        prepared[name] = (event_rows, region, identities, features)
        lengths.append(len(features))
        population_signatures[name] = {
            "files": list(files),
            "events": int(len(event_rows)),
            "event_ids_sha256": digest(identities),
            "physical_population_sha256": digest(event_rows),
        }
    combined = np.concatenate([prepared[name][3] for name in CLASSIFIER_SCORE_PARTITIONS], axis=0)
    scoring_root = output / ".resume" / "baseline_scoring"
    scoring_root.mkdir(parents=True, exist_ok=True)
    scoring_contract = {
        "schema": 1,
        "method": method,
        "classifier": classifier,
        "preprocessing_sha256": file_digest(Path(training_root) / "preprocessing.npz"),
        "members": [classifier_member_signature(receipt) for receipt in accepted_receipts],
        "populations": population_signatures,
    }
    contract_path = scoring_root / "contract.json"
    if contract_path.is_file() and resume:
        saved_contract = json.loads(contract_path.read_text())
        if saved_contract != scoring_contract:
            raise ValueError("AD baseline scoring recovery contract changed")
    write_json(contract_path, scoring_contract)
    member_predictions = [
        load_or_predict_classifier_member(scoring_root, training_root, receipt, combined, classifier, torch.device(device), resume)
        for receipt in accepted_receipts
    ]
    members = np.asarray(member_predictions, dtype=np.float32)
    if members.ndim != 2 or members.shape[1] != len(combined):
        raise ValueError("Invalid AD baseline fit prediction shape")
    ensemble = members.mean(axis=0, dtype=np.float64).astype(np.float32)
    offsets = np.cumsum([0, *lengths])
    acceptance = {}
    evaluation_hashes = {}
    for index, name in enumerate(CLASSIFIER_SCORE_PARTITIONS):
        event_rows, region, identities, features = prepared[name]
        fit_scores = members[:, offsets[index]:offsets[index + 1]].copy()
        scores = ensemble[offsets[index]:offsets[index + 1]].copy()
        preprocessing_mask = np.ones(len(event_rows), dtype=bool)
        preprocessing_mask, score_domain_mask, mask = scoring_masks(preprocessing_mask, region, classifier["score_scope"])
        scores[~mask] = np.nan
        fit_scores[:, ~mask] = np.nan
        arrays = {
            "mass": event_rows[:, 0].astype(np.float32),
            "labels": event_rows[:, -1].astype(np.int64),
            "mask": mask,
            "scores": scores,
            "physical": event_rows[:, 1:-1].astype(np.float32),
            "event_ids": identities,
            "is_signal_region": region,
            "preprocessing_mask": preprocessing_mask,
            "score_domain_mask": score_domain_mask,
            "fit_scores": fit_scores,
            "accepted_fit_indices": np.asarray([receipt["fit_index"] for receipt in accepted_receipts], dtype=np.int64),
            "accepted_fit_seeds": np.asarray([receipt["fit_seed"] for receipt in accepted_receipts], dtype=np.uint64),
            "score_scope": np.asarray(classifier["score_scope"]),
            "score_kind": np.asarray("classifier_probability"),
            "fit_score_kind": np.asarray("classifier_probability"),
        }
        atomic_write(output / f"{name}_scores.npz", lambda path, values=arrays: save_npz(path, **values))
        acceptance[name] = {
            "events": int(len(event_rows)),
            "scored_events": int(mask.sum()),
            "outside_score_domain_events": int((~score_domain_mask).sum()),
        }
        evaluation_hashes[name] = {
            "event_ids_sha256": digest(identities),
            "physical_population_sha256": digest(event_rows),
            "score_artifact_sha256": file_digest(output / f"{name}_scores.npz"),
        }
    write_json(output / "mapping_acceptance.json", acceptance)
    return {
        "feature_order": feature_order,
        "evaluation_population_hashes": evaluation_hashes,
        "accepted_fit_indices": [int(receipt["fit_index"]) for receipt in accepted_receipts],
        "accepted_fit_seeds": [int(receipt["fit_seed"]) for receipt in accepted_receipts],
    }
