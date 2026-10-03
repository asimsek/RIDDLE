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


def paired_score_metrics(record, efficiencies):
    """Post-freeze MC evaluation; never call this to choose a scoring mode."""
    labels, mask = np.asarray(record["labels"]), np.asarray(record["mask"])
    candidates = {}
    for name, key in (("PEW", "pew_scores"), ("potential_qnorm", "potential_qnorm_scores"), ("selected", "scores")):
        scores = np.asarray(record[key])
        report = oracle_metrics(labels, scores, mask)
        rows = []
        bg, signal = scores[mask & (labels == 0)], scores[mask & (labels == 1)]
        for efficiency in efficiencies:
            threshold = float(np.quantile(bg, 1 - efficiency, method="higher")) if len(bg) else None
            eb = float(np.mean(bg > threshold)) if len(bg) else None
            es = float(np.mean(signal > threshold)) if len(signal) and threshold is not None else None
            rows.append({"target_background_efficiency": efficiency, "background_efficiency": eb,
                         "signal_efficiency": es, "sic": es / np.sqrt(eb) if es is not None and eb else None,
                         "threshold": threshold})
        candidates[name] = {**report, "tail_points": rows}
    return {"schema": 1, "purpose": "truth-assisted evaluation after score selection was frozen",
            "used_for_score_selection": False, "selected_mode": str(record["selected_scoring_mode"].item()),
            "auto_switch_enabled": bool(record["auto_switch_enabled"].item()),
            "event_ids_sha256": str(record["shared_evaluation_event_ids_sha256"].item()),
            "score_selection_sha256": str(record["score_selection_sha256"].item()),
            "candidates": candidates}
