"""Production validation without clipping, repairing, or selecting on test scores."""

import numpy as np

SR_BOUNDS = (3.3, 3.7)
PRODUCTION_POLICY = {
    "ensemble": "accepted_fits_after_bounded_retries_v1",
    "signal_region": "strict_prepared_input_membership",
    "signal_region_bounds": list(SR_BOUNDS),
}


class IncompleteEnsembleError(RuntimeError):
    pass


class NumericalFitError(ValueError):
    """A model output is unusable; input corruption is not a fit failure."""


def validation_improvement(log_mixture_ratios, *, sigma):
    """Paired validation gain against the fixed background, without truth labels.

    This is an optimization/selection rule, not a calibrated discovery test.
    The threshold and retry budget are part of the scientific run contract.
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
    return dict(status="passed" if margin > tolerance else "insufficient_validation_improvement",
                events=len(values), mean_log_likelihood_gain=gain,
                standard_error=error, sigma=float(sigma), lower_margin=margin, numerical_tolerance=float(tolerance),
                baseline="standard-normal background", truth_labels_used=False,
                interpretation="validation selection only; not evidence of a physical signal")


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


def validate_score_record(record, *, method, stage):
    """Shared producer/consumer check, including members before any averaging."""
    mask, scores = np.asarray(record["mask"]), np.asarray(record["scores"])
    if mask.dtype != bool or mask.ndim != 1 or scores.shape != mask.shape:
        raise ValueError(f"{stage}: invalid score/mapping shapes")
    n = len(mask)
    for key in ("mass", "labels"):
        if np.asarray(record[key]).shape != (n,):
            raise ValueError(f"{stage}: misaligned {key}")
    if not np.isin(record["labels"], [0, 1]).all():
        raise ValueError(f"{stage}: invalid labels")
    for key in ("mass", "physical", "latent", "fit_latents"):
        if key in record and not np.isfinite(record[key]).all():
            raise ValueError(f"{stage}: nonfinite {key}")
    if "physical" in record and (record["physical"].ndim != 2 or len(record["physical"]) != n):
        raise ValueError(f"{stage}: misaligned physical features")
    if "latent" in record and (record["latent"].ndim != 2 or len(record["latent"]) != mask.sum()):
        raise ValueError(f"{stage}: misaligned accepted latents")
    if not np.isnan(scores[~mask]).all():
        raise ValueError(f"{stage}: rejected-event scores must be NaN")
    summary = score_diagnostics(scores[mask], stage=stage, probability=method == "lacathode")
    summary["rejected_events"] = int((~mask).sum())
    if "fit_scores" in record:
        fits = record["fit_scores"]
        if fits.ndim != 2 or fits.shape[1:] != (n,) or not len(fits) or not np.isnan(fits[:, ~mask]).all():
            raise ValueError(f"{stage}: invalid saved member scores")
        summary["members"] = [score_diagnostics(fit[mask], stage=f"{stage} member {i}",
                                               probability=method == "lacathode")
                              for i, fit in enumerate(fits)]
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
    cutoff = np.log(1e-12 / (len(thresholds) * tests))
    checks = []
    for threshold in thresholds:
        count = int(np.count_nonzero(log_ratios >= np.log(threshold)))
        log_p = float(binom.logsf(count - 1, len(log_ratios), 1 / threshold))
        checks.append(dict(ratio_threshold=threshold, exceedances=count,
                           log_p_upper_bound=log_p if np.isfinite(log_p) else None))
        if log_p < cutoff:
            raise NumericalFitError(f"{stage}: density-ratio normalization failed on independent Gaussian reference "
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
    policy = selection.get("production_policy")
    legacy = policy == {**PRODUCTION_POLICY, "ensemble": "all_requested_runs"}
    if selection.get("status") != "completed" or (not legacy and policy != PRODUCTION_POLICY):
        raise IncompleteEnsembleError("RIDDLE ensemble is not finalized under a recognized production policy")
    requested = selection.get("requested_runs")
    checkpoints = selection.get("checkpoints_per_run")
    configs = selection.get("configurations", [])
    if not configs or (legacy and selection.get("failures")):
        raise IncompleteEnsembleError("RIDDLE ensemble has failed or missing configurations")
    for config in configs:
        members = config.get("members", [])
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
                    or any(m.get("quality", {}).get("status") != "passed" for m in members)
                    or any(m.get("normalization", {}).get("status") != "passed" for m in members)):
                raise IncompleteEnsembleError("RIDDLE fit acceptance/exclusion inventory is inconsistent")
    chosen = [c for c in configs if c.get("name") == selection.get("selected_configuration")]
    members = selection.get("members", [])
    if (len(chosen) != 1 or not members or members != chosen[0]["members"]
            or selection.get("valid_runs") != len(members)
            or selection.get("selected_checkpoints") != len(members) * checkpoints):
        raise IncompleteEnsembleError("RIDDLE ensemble selection is incomplete or inconsistent")
