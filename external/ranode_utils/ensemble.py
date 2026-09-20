"""Combine completed upstream fits without modifying their training or score export."""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np
from riddle.production import NumericalFitError


MASS_BINS = np.linspace(3.3, 3.7, 50)  # Pinned scripts/r_anode.py mass marginal.


def validate_upstream_likelihood(namespace):
    raw = np.asarray(namespace["likelihood_"])
    if raw.ndim != 1 or not raw.size:
        raise ValueError("Malformed upstream R-ANODE likelihood")
    if not np.isfinite(raw).all():
        raise NumericalFitError("Nonfinite upstream R-ANODE likelihood before nan_to_num; refusing sanitized scores")
    if not np.array_equal(raw, namespace["likelihood"]):
        raise NumericalFitError("Upstream R-ANODE likelihood was modified by sanitization")


def validate_mass_normalization(attempt):
    """The sampled mass denominator must have support throughout the scored SR.

    An empty bin invokes upstream's 1e-31 floor: finite but unphysical ratios
    which can dominate every other fit. Reject that fit; never repair its scores
    by smoothing or clipping the denominator.
    """
    attempt = Path(attempt)
    path = attempt / "results/upstream/signal/fit/samples.npy"
    samples = np.load(path, mmap_mode="r", allow_pickle=False)
    if samples.ndim != 2 or samples.shape[1] < 1 or not len(samples):
        raise ValueError(f"R-ANODE {attempt}: invalid generated signal samples")
    if not np.isfinite(samples).all():
        raise NumericalFitError(f"R-ANODE {attempt}: nonfinite generated signal samples")
    counts, _ = np.histogram(samples[:, 0], bins=MASS_BINS)
    empty = np.flatnonzero(counts == 0)
    if len(empty):
        raise NumericalFitError(
            f"R-ANODE {attempt}: unsupported signal-mass normalization: "
            f"{len(empty)}/{len(counts)} empty SR bins, {counts.sum()}/{len(samples)} samples in SR. "
            "The upstream density floor would create artificial likelihood ratios. "
            "Result rejected; inspect this fit's samples and training before rerunning. "
            "This fit cannot contribute to a saved ensemble."
        )
    return dict(status="passed", samples=len(samples), samples_in_sr=int(counts.sum()),
                edges=MASS_BINS.tolist(), counts=counts.tolist(),
                minimum_bin_count=int(counts.min()))


def validate_result_normalization(root, report):
    """Recheck saved samples too, including results made before this safeguard."""
    import json
    from riddle.storage import file_digest

    root = Path(root).resolve()

    def verified(name):
        path = (root / name).resolve()
        expected = report.get("artifacts_sha256", {}).get(name)
        if not path.is_relative_to(root) or not expected or file_digest(path) != expected:
            raise ValueError(f"R-ANODE normalization input missing or changed: {path}")
        return path

    protocol = json.loads(verified("protocol.json").read_text())
    attempts = protocol.get("signal_attempts", [])
    # Older single-fit adapter releases used signal_attempt.
    if not attempts and protocol.get("signal_attempt"):
        attempts = [protocol["signal_attempt"]]
    requested = protocol.get("valid_runs", protocol.get("requested_runs", 1))
    if len(attempts) != requested or not attempts or len(set(attempts)) != len(attempts):
        raise ValueError(f"R-ANODE {root}: incomplete signal-fit normalization evidence")
    diagnostics = {}
    for name in attempts:
        verified(str(Path(name) / "results/upstream/signal/fit/samples.npy"))
        diagnostics[name] = validate_mass_normalization(root / name)
    return diagnostics


def mass_normalization_check(attempt, *, safeguards=True):
    """The explicit pilot control records failures without excluding the fit."""
    try:
        health = validate_mass_normalization(attempt)
    except NumericalFitError as error:
        if safeguards:
            raise
        return dict(status="disabled", would_reject=True, diagnostic=str(error))
    return health if safeguards else dict(health, status="disabled", would_reject=False)


def signal_fit_health(attempt, namespace, *, safeguards=True):
    health = mass_normalization_check(attempt, safeguards=safeguards)
    try:
        validate_upstream_likelihood(namespace)
    except NumericalFitError as error:
        if safeguards:
            raise
        health["likelihood_diagnostic"] = str(error)
    return health


def selected_epochs(attempt, epochs, *, safeguards=True):
    root = Path(attempt) / "results/upstream/signal/fit"
    losses = np.load(root / "valloss.npy", allow_pickle=False)
    if losses.shape != (epochs,) or (safeguards and not np.isfinite(losses).all()) or epochs < 10:
        raise ValueError("Incomplete R-ANODE validation losses")
    selected = np.argsort(losses).flatten()[:10].tolist()
    if any(not (root / f"model_S_{epoch}.pt").is_file() for epoch in selected):
        raise ValueError("Missing validation-selected R-ANODE checkpoint")
    return selected


def combine_fits(attempts, output, *, requested_runs, safeguards=True):
    if not attempts or len(attempts) != requested_runs or len(set(attempts)) != requested_runs:
        raise ValueError("All requested R-ANODE fits must complete before ensembling")
    # Validate every denominator before writing any ensemble partition.
    for attempt in attempts:
        mass_normalization_check(attempt, safeguards=safeguards)
    fields = ("mass", "physical", "labels", "scores", "mask", "is_signal_region")
    for name in ("validation", "test", "signal_region"):
        base, combined = None, None
        for attempt in attempts:
            with np.load(Path(attempt) / (name + "_scores.npz"), allow_pickle=False) as data:
                if set(data.files) != set(fields):
                    raise ValueError("Incomplete R-ANODE score artifact")
                record = {key: data[key] for key in fields}
            mask, scores = record["mask"], record["scores"]
            if mask.dtype != bool or mask.shape != scores.shape or mask.ndim != 1:
                raise ValueError("Invalid R-ANODE score mask")
            if not np.isfinite(scores[mask]).all() or not np.isnan(scores[~mask]).all():
                raise ValueError("Invalid R-ANODE scores or rejected-event values")
            if base is None:
                base = record
                combined = scores[mask].astype(np.float64)
            else:
                if any(not np.array_equal(base[key], record[key]) for key in fields if key != "scores"):
                    raise ValueError("R-ANODE fits disagree on event alignment or preprocessing acceptance")
                combined = np.logaddexp(combined, scores[mask])
        destination = Path(output) / (name + "_scores.npz")
        with tempfile.NamedTemporaryFile(dir=output, suffix=".npz", delete=False) as stream:
            temporary = Path(stream.name)
        try:
            if requested_runs == 1:
                shutil.copyfile(Path(attempts[0]) / (name + "_scores.npz"), temporary)
            else:
                base["scores"] = base["scores"].copy()
                base["scores"][base["mask"]] = combined - np.log(requested_runs)
                np.savez_compressed(temporary, **base)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
