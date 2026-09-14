"""RIDDLE-only production requirements; no LaCathode numerical overrides."""

import numpy as np

SR_BOUNDS = (3.3, 3.7)
PRODUCTION_POLICY = {
    "ensemble": "all_requested_runs",
    "signal_region": "strict_prepared_input_membership",
    "signal_region_bounds": list(SR_BOUNDS),
}


class IncompleteEnsembleError(RuntimeError):
    pass


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


def require_complete_members(members, requested, checkpoints, *, configuration):
    if type(requested) is not int or requested < 1 or type(checkpoints) is not int or checkpoints < 1:
        raise IncompleteEnsembleError("Missing RIDDLE ensemble size/checkpoint requirements")
    directories = [m.get("directory") for m in members]
    if (len(members) != requested or any(not isinstance(d, str) or not d for d in directories)
            or len(set(directories)) != requested):
        raise IncompleteEnsembleError(
            f"RIDDLE configuration {configuration}: {len(members)}/{requested} valid fits; "
            "all requested fits are required. Checkpoints and failure records are preserved; "
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
    if selection.get("status") != "completed" or selection.get("production_policy") != PRODUCTION_POLICY:
        raise IncompleteEnsembleError("RIDDLE ensemble is not finalized under the all-fits production policy")
    requested = selection.get("requested_runs")
    checkpoints = selection.get("checkpoints_per_run")
    configs = selection.get("configurations", [])
    if not configs or selection.get("failures"):
        raise IncompleteEnsembleError("RIDDLE ensemble has failed or missing configurations")
    for config in configs:
        require_complete_members(config.get("members", []), requested, checkpoints,
                                 configuration=config.get("name"))
        if config.get("valid_runs") != requested:
            raise IncompleteEnsembleError("RIDDLE configuration has an inconsistent valid-fit count")
    chosen = [c for c in configs if c.get("name") == selection.get("selected_configuration")]
    if (len(chosen) != 1 or selection.get("members") != chosen[0]["members"]
            or selection.get("valid_runs") != requested
            or selection.get("selected_checkpoints") != requested * checkpoints):
        raise IncompleteEnsembleError("RIDDLE ensemble selection is incomplete or inconsistent")
