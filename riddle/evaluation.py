"""Evaluation-only population identity and denominator provenance (never training input)."""
import hashlib
import numpy as np


def _canonical_array(values, *, event_ids=False):
    array = np.asarray(values)
    if array.dtype.hasobject:
        raise ValueError("Population provenance does not accept object arrays")
    if event_ids:
        if array.dtype.kind not in "iu" or np.any(array < 0):
            raise ValueError("Event identities must be nonnegative integers")
        return np.ascontiguousarray(array, dtype="<u8")
    if array.dtype.kind == "f":
        return np.ascontiguousarray(array, dtype="<f4")
    if array.dtype.kind == "u":
        return np.ascontiguousarray(array, dtype="<u8")
    if array.dtype.kind == "i":
        return np.ascontiguousarray(array, dtype="<i8")
    if array.dtype.kind == "b":
        return np.ascontiguousarray(array, dtype=np.uint8)
    return np.ascontiguousarray(array)


def _ordered_digest(arrays):
    h = hashlib.sha256()
    for name, values in arrays:
        array = _canonical_array(values, event_ids=name == "event_ids")
        h.update(name.encode("utf-8") + b"\0")
        h.update(str(array.dtype).encode("ascii") + b"\0")
        h.update(str(array.shape).encode("ascii") + b"\0")
        h.update(array.tobytes())
    return h.hexdigest()


def population_metadata(record, partition, scope, *, threshold_source=None):
    if partition not in ("signal_region", "validation", "test"):
        raise ValueError("Unknown score partition")
    if scope not in ("signal_region", "full_region"):
        raise ValueError("Unknown evaluation scope")
    labels, mass, mask = (np.asarray(record[k]) for k in ("labels", "mass", "mask"))
    if (labels.ndim != 1 or mass.shape != labels.shape or mask.shape != labels.shape
            or mask.dtype != bool or not np.isin(labels, (0, 1)).all()):
        raise ValueError("Misaligned evaluation population")
    if scope == "signal_region":
        region = np.asarray(record.get("is_signal_region", (mass > 3.3) & (mass < 3.7)))
        if region.shape != mass.shape or region.dtype != bool:
            raise ValueError("Invalid population SR membership")
    else:
        region = np.ones(len(labels), dtype=bool)
    rows = [("mass", mass[region]), ("labels", labels[region])]
    if "physical" in record:
        physical = np.asarray(record["physical"])
        if physical.ndim != 2 or len(physical) != len(labels):
            raise ValueError("Misaligned population features")
        rows.append(("physical", physical[region]))
    checksum = _ordered_digest(rows)
    identity_digest = None
    if "event_ids" in record:
        ids = np.asarray(record["event_ids"])
        if ids.ndim not in (1, 2) or len(ids) != len(labels):
            raise ValueError("Misaligned population event IDs")
        identity_digest = _ordered_digest([("event_ids", ids[region])])
    population_identity = identity_digest or checksum
    selected_labels, accepted = labels[region], mask[region]
    result = dict(
        evaluation_population_id=f"{partition}:{scope}:{population_identity[:16]}",
        score_partition=partition, score_artifact=f"{partition}_scores.npz", scope=scope,
        population_sha256=checksum, event_ids_sha256=identity_digest,
        identity_basis=("ordered event IDs" if identity_digest is not None
                        else "canonical ordered mass/label/physical rows"),
        counts=dict(total=int(region.sum()), background=int((selected_labels == 0).sum()),
                    signal=int((selected_labels == 1).sum()), accepted_total=int(accepted.sum()),
                    accepted_background=int(((selected_labels == 0) & accepted).sum()),
                    accepted_signal=int(((selected_labels == 1) & accepted).sum())),
        efficiency_denominator="all original class events in this population, including unscored events",
        auc_population="accepted events only",
        acceptance_definition=("preprocessing_mask AND score_domain_mask" if "preprocessing_mask" in record
                               else "saved effective score mask; legacy preprocessing/domain split unavailable"),
    )
    if threshold_source is not None:
        result["threshold_source"] = threshold_source
    return result


def common_acceptance_auc(records, region=None):
    if not records:
        raise ValueError("Common-population AUC requires at least one method")
    base = next(iter(records.values()))
    labels = np.asarray(base["labels"])
    if labels.ndim != 1 or not np.isin(labels, (0, 1)).all():
        raise ValueError("Invalid common AUC labels")
    common = np.ones(len(labels), dtype=bool) if region is None else np.asarray(region, dtype=bool).copy()
    if common.shape != labels.shape:
        raise ValueError("Invalid common AUC region")
    for record in records.values():
        other_labels = np.asarray(record["labels"])
        mask = np.asarray(record["mask"])
        scores = np.asarray(record["scores"])
        if (not np.array_equal(other_labels, labels) or mask.shape != labels.shape
                or mask.dtype != bool or scores.shape != labels.shape):
            raise ValueError("Misaligned common AUC population")
        common &= mask & np.isfinite(scores)
    selected = labels[common]
    counts = dict(total=int(common.sum()), background=int((selected == 0).sum()), signal=int((selected == 1).sum()))
    if len(np.unique(selected)) != 2:
        return dict(status="unavailable", counts=counts, methods={})
    from sklearn.metrics import roc_auc_score
    return dict(
        status="available",
        counts=counts,
        population="intersection of accepted physical events across compared methods",
        methods={name: float(roc_auc_score(selected, np.asarray(record["scores"])[common]))
                 for name, record in records.items()},
    )

def riddle_score_scope(report):
    """Resolve scope per result, never from the last-discovered global method.

    Older mass-conditioned RIDDLE archives used riddlev2/riddlev3 and sometimes
    recorded only protocol.scope.  They are SR-only even though their test NPZ
    also stores unscored sideband rows.  Conflicting declarations fail closed.
    """
    contract = report.get("contract", {})
    settings = contract.get("settings", {}).get("riddle", {})
    conditional = settings.get("mass_conditioning", report.get("method") in ("riddlev2", "riddlev3"))
    protocol = report.get("protocol", {})
    claims = [report.get("score_scope"), report.get("plotting", {}).get("score_scope"),
              contract.get("riddle_score_scope"), protocol.get("score_scope"), protocol.get("scope")]
    claims = [value for value in claims if value is not None]
    if any(value not in ("signal_region", "full_region") for value in claims):
        raise ValueError("Unrecognized RIDDLE score scope")
    if len(set(claims)) > 1 or (conditional and "full_region" in claims):
        raise ValueError("Conflicting RIDDLE score-domain declarations")
    return claims[0] if claims else "signal_region" if conditional else "full_region"
