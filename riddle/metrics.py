"""Truth-assisted evaluation only. Never imported into model optimization."""

import numpy as np
from sklearn.metrics import roc_curve, roc_auc_score


def acceptance_report(labels, mask, mass=None):
    labels, mask = np.asarray(labels), np.asarray(mask)
    if labels.shape != mask.shape or mask.dtype != bool or not np.isin(labels, [0, 1]).all():
        raise ValueError("Invalid acceptance population")
    scopes = {"full": np.ones(len(labels), dtype=bool)}
    if mass is not None:
        mass = np.asarray(mass)
        if mass.shape != labels.shape:
            raise ValueError("Misaligned acceptance masses")
        scopes["signal_region"] = (mass > 3.3) & (mass < 3.7)
    report = {}
    for scope, region in scopes.items():
        report[scope] = {}
        for name, label in (("background", 0), ("signal", 1)):
            pop = region & (labels == label)
            total, mapped = int(pop.sum()), int((pop & mask).sum())
            report[scope][name] = dict(
                total=total,
                mapped=mapped,
                rejected=total - mapped,
                acceptance=mapped / total if total else None,
            )
    return report


def efficiency_curve(labels, scores, mask, *, full_pipeline=False):
    """Unmapped events never pass; no fake score or forced (1,1) endpoint."""
    labels, scores, mask = np.asarray(labels), np.asarray(scores), np.asarray(mask)
    acceptance = acceptance_report(labels, mask)["full"]
    if len(np.unique(labels[mask])) != 2 or not np.isfinite(scores[mask]).all():
        raise ValueError("ROC requires finite mapped scores from both classes")
    b, s, cuts = roc_curve(labels[mask], scores[mask], drop_intermediate=False)
    if full_pipeline:
        b *= acceptance["background"]["acceptance"]
        s *= acceptance["signal"]["acceptance"]
    return b, s, cuts


def oracle_metrics(labels, scores, mask, *, min_background=10, min_efficiency=1e-4):
    report = acceptance_report(labels, mask)["full"]
    result = {
        "acceptance": report,
        "selection": "oracle maximum on truth-labelled test sample; not a deployable cut",
    }
    if len(np.unique(np.asarray(labels)[mask])) != 2:
        return {
            **result,
            "conditional_auc": None,
            "conditional_max_sic": None,
            "full_pipeline_max_sic": None,
        }
    result["conditional_auc"] = float(roc_auc_score(np.asarray(labels)[mask], np.asarray(scores)[mask]))
    for full in (False, True):
        b, s, _ = efficiency_curve(labels, scores, mask, full_pipeline=full)
        n_b = report["background"]["total" if full else "mapped"]
        supported = (b >= min_efficiency) & (np.rint(b * n_b) >= min_background)
        name = "full_pipeline" if full else "conditional"
        result[name + "_max_sic"] = (
            float(np.max(s[supported] / np.sqrt(b[supported]))) if supported.any() else None
        )
    result["minimum_background_count"] = min_background
    result["minimum_background_efficiency"] = min_efficiency
    return result
