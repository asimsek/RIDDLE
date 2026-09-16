"""Combine completed upstream fits without modifying their training or score export."""

import os
import shutil
import tempfile
from pathlib import Path

import numpy as np


def selected_epochs(attempt, epochs):
    root = Path(attempt) / "results/upstream/signal/fit"
    losses = np.load(root / "valloss.npy", allow_pickle=False)
    if losses.shape != (epochs,) or not np.isfinite(losses).all() or epochs < 10:
        raise ValueError("Incomplete R-ANODE validation losses")
    selected = np.argsort(losses).flatten()[:10].tolist()
    if any(not (root / f"model_S_{epoch}.pt").is_file() for epoch in selected):
        raise ValueError("Missing validation-selected R-ANODE checkpoint")
    return selected


def combine_fits(attempts, output, *, requested_runs):
    if not attempts or len(attempts) != requested_runs or len(set(attempts)) != requested_runs:
        raise ValueError("All requested R-ANODE fits must complete before ensembling")
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
