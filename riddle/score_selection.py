"""Select PEW/potential scores from reserved p/q events without evaluation data.
Oracle pool definitions use truth; RIDDLE uses a model-dependent q proxy."""
import json
from pathlib import Path

import numpy as np

from .storage import atomic_write, digest, file_digest, save_npz, write_json
from .stein_scoring import _json_digest, _load_reference_b

MODES = ("tail_focus", "potential_qnorm")
PROTOCOL = "reserved_ensemble_score_selection_v2_single_comparison"


def validate_population_contract(settings, manifest, method):
    cfg = settings["stein"]["scoring"]
    enabled = cfg["auto_switch"]["enabled"][method]
    shared = manifest.get("schema") in (5, 6)
    if enabled and not shared:
        raise ValueError("Auto-switch requires shared schema-5/6 data. Prepare a new directory with config/populations.yaml.")
    if shared and settings.get("core") == "stein_witness":
        from .roles import PRODUCTION_POLICIES, DEFAULT_POLICY
        from .options import effective_features
        if (not settings.get("mass_conditioning") or settings.get("input_space") == "physical"
                or settings.get("data_policy", DEFAULT_POLICY) not in PRODUCTION_POLICIES
                or cfg["mode"] not in MODES):
            raise ValueError("Shared Stein comparison requires mapped mass-conditioned production roles and PEW/potential-qnorm")
        if not (any(effective_features(settings).values()) or settings.get("background_correction") == "bgcorr_40_reguide"):
            raise ValueError("Shared score selection requires the mapping pipeline with reserved development roles")
        return True
    if enabled:
        raise ValueError("Auto-switch requires a shared Stein comparison")
    return False


def assert_disjoint(*populations):
    views = []
    for values in populations:
        a = np.asarray(values)
        if a.ndim != 2 or a.shape[1] != 2 or a.dtype != np.uint64:
            raise ValueError("Selector provenance requires uint64 source/event identity pairs")
        v = np.ascontiguousarray(a).view("V16").ravel()
        if len(np.unique(v)) != len(v):
            raise ValueError("Duplicate selector event identities")
        if any(np.intersect1d(v, previous).size for previous in views):
            raise ValueError("Score selector overlaps fitting, assessment, or evaluation events")
        views.append(v)


def utility(p, thresholds, q, cfg):
    cuts = np.quantile(thresholds, 1 - np.asarray(cfg["efficiencies"]), axis=0, method="higher")
    p_accept = (p[:, None, :] > cuts[None, :, :]).mean(axis=0).T
    q_accept = (q[:, None, :] > cuts[None, :, :]).mean(axis=0).T
    return (p_accept - q_accept) @ np.asarray(cfg["weights"]), p_accept, q_accept


def bootstrap_gain(p, thresholds, q, cfg, alternative, baseline, seed):
    """Paired event bootstrap with threshold uncertainty; intervals are exploratory."""
    pools = (p, thresholds, q)
    orders = [np.argsort(x, axis=0, kind="stable") for x in pools]
    sorted_values = [np.take_along_axis(x, order, axis=0) for x, order in zip(pools, orders)]
    rng = np.random.default_rng(seed)
    boot = np.empty((cfg["bootstrap_replicas"], 2))
    target = np.ceil((len(thresholds) - 1) * (1 - np.asarray(cfg["efficiencies"]))).astype(int) + 1
    for b in range(len(boot)):
        cumulative = []
        for x, order in zip(pools, orders):
            counts = np.bincount(rng.integers(0, len(x), len(x)), minlength=len(x))
            cumulative.append(np.vstack((np.zeros(2), np.cumsum(counts[order], axis=0))))
        for j in range(2):
            pos = np.searchsorted(cumulative[1][1:, j], target, side="left")
            cuts = sorted_values[1][pos, j]
            fractions = []
            for k in (0, 2):
                at = np.searchsorted(sorted_values[k][:, j], cuts, side="right")
                fractions.append(1 - cumulative[k][at, j] / len(pools[k]))
            boot[b, j] = (fractions[0] - fractions[1]) @ np.asarray(cfg["weights"])
    delta = boot[:, alternative] - boot[:, baseline]
    return {"lower": float(np.quantile(delta, 1 - cfg["confidence"])),
            "upper": float(np.quantile(delta, cfg["confidence"])),
            "bootstrap_standard_deviation": float(delta.std(ddof=1)),
            "one_sided_confidence": cfg["confidence"]}


def threshold_split(count, seed):
    indices = np.random.default_rng(seed).permutation(count)
    return np.split(indices, [count // 2])


def choose(p, q, cfg, baseline="tail_focus", enabled=True):
    """Pure numerical selection interface: only paired scores, never labels."""
    if baseline not in MODES:
        raise ValueError("Unknown auto-switch baseline")
    decision = {"enabled": bool(enabled), "baseline_mode": baseline, "selected_mode": baseline,
                "switched": False, "candidate_order": list(MODES), "reason": "disabled",
                "selection_basis": "disabled_fixed_fallback", "comparison_count": 0}
    if not enabled:
        return decision
    p, q = np.asarray(p), np.asarray(q)
    for a in (p, q):
        if a.ndim != 2 or a.shape[1] != 2 or not np.isfinite(a).all():
            raise ValueError("Selector requires finite paired candidate scores")
    qt, qa = threshold_split(len(q), cfg["seed"])
    decision.update(selection_basis="inconclusive_fallback",
                    counts=dict(p_assessment=len(p), q_threshold=len(qt), q_assessment=len(qa)),
                    q_threshold_indices_sha256=digest(qt), q_assessment_indices_sha256=digest(qa))
    if (len(p) < 2 or
            min(len(qt), len(qa)) * min(cfg["efficiencies"]) < cfg["min_tail_events"]):
        decision["reason"] = "insufficient_independent_tail_statistics"
        return decision
    cuts = np.quantile(q[qt], 1 - np.asarray(cfg["efficiencies"]), axis=0, method="higher")
    tail_counts = (q[qt, None, :] > cuts[None, :, :]).sum(axis=0).T
    decision["q_threshold_tail_counts"] = tail_counts.tolist()
    if np.any(tail_counts < cfg["min_tail_events"]):
        decision["reason"] = "insufficient_resolved_tail_statistics"
        return decision
    base = MODES.index(baseline)
    alternative = 1 - base
    u, pfrac, qfrac = utility(p, q[qt], q[qa], cfg)
    report = {"utility": u.tolist(), "p_acceptance": pfrac.tolist(), "q_acceptance": qfrac.tolist(),
              "gain": float(u[alternative] - u[base]),
              **bootstrap_gain(p, q[qt], q[qa], cfg, alternative, base, cfg["seed"] + 901)}
    decision.update(comparison=report, comparison_count=1)
    if report["lower"] > 0 and report["gain"] > 0:
        decision.update(selected_mode=MODES[alternative], switched=True,
                        reason="reserved_comparison_supports_alternative", selection_basis="evidence_supported_switch")
    elif report["upper"] < 0 and report["gain"] < 0:
        decision.update(reason="reserved_comparison_supports_baseline", selection_basis="evidence_supported_baseline")
    else:
        decision["reason"] = "inconclusive_reserved_comparison"
    return decision


def selector_populations(development, oracle_roles, method, evaluation_ids):
    from .model import with_mass_context
    oracle = oracle_roles["selector"] if oracle_roles is not None else {}
    if method == "supervised":
        p, p_ids = oracle["p"], oracle["p_ids"]
        p_source = "reserved_pure_signal_validation"
    else:
        mixture = development["mixture_validation"]
        p, p_ids = with_mass_context(mixture["z"], mixture["mass"]), mixture["ids"]
        p_source = "reserved_signal_region_mixture_validation"
    used = [v["ids"] for k, v in development.items()
            if k not in ("member_splits", "mixture_validation") and "ids" in v]
    if "used_ids" in oracle:
        used.append(oracle["used_ids"])
    # Union overlapping mapping roles before checking selector isolation.
    used_ids = np.unique(np.concatenate(used), axis=0)
    q_ids = oracle.get("q_ids", np.empty((0, 2), dtype=np.uint64))
    assert_disjoint(p_ids, q_ids, used_ids, evaluation_ids)
    return {"p": p, "p_ids": p_ids, "q": oracle.get("q"), "q_ids": q_ids,
            "p_source": p_source, "used_ids_sha256": digest(used_ids),
            "q_source": "reserved_pure_background_validation" if oracle else "independent_generated_q_reference_C"}


def freeze(root, settings, method, populations, shared_population, device):
    """Freeze the decision before scoring any evaluation events; resumable by hash."""
    from .campaign import ensemble_predict
    from .production import require_complete_ensemble
    from .stein_scoring import prepare_selector_reference
    from .worker_progress import emit_message
    root = Path(root)
    cfg = settings["stein"]["scoring"]["auto_switch"]
    enabled = cfg["enabled"][method]
    reference, reference_meta = _load_reference_b(root)
    selection = json.loads((root / "ensemble_selection.json").read_text())
    require_complete_ensemble(selection)
    if method not in cfg["fallback"] or (populations["q"] is None) != (method == "riddle"):
        raise ValueError("Selector method and background population disagree")
    p = populations["p"]
    base_provenance = json.loads((root / "stein_scoring_calibration.json").read_text())
    checkpoints = {str(m["fit_index"]): {str(int(e)): file_digest(
        root / m["directory"] / f"residual_epoch_{int(e)}.pt") for e in m["epochs"]}
        for m in selection["members"]}
    if checkpoints != base_provenance["scoring_identity"]["selected_checkpoint_sha256"]:
        raise ValueError("Score selector checkpoint identity changed")
    q = populations["q"]
    c_meta = None
    if enabled and method == "riddle":
        q, c_meta = prepare_selector_reference(root, selection["members"], settings, device)
    identity = {"protocol": PROTOCOL, "method": method, "settings": settings,
                "shared_population": shared_population,
                "model_scoring_identity": base_provenance["scoring_identity"],
                "reference_B_sha256": reference_meta["reference_B_sha256"],
                "reference_B_events": len(reference), "selector_reference_C": c_meta,
                "p_sha256": digest(p), "q_sha256": None if q is None else digest(q),
                "p_event_ids_sha256": digest(populations["p_ids"]),
                "q_event_ids_sha256": digest(populations["q_ids"]),
                "used_event_ids_sha256": populations["used_ids_sha256"]}
    cache = root / ".resume" / "score_selection" / _json_digest(identity)
    cache.mkdir(parents=True, exist_ok=True)
    saved_path, scores_path = cache / "decision.json", cache / "selector_scores.npz"
    if saved_path.exists():
        decision = json.loads(saved_path.read_text())
        if (decision["identity"] != identity or not scores_path.is_file()
                or file_digest(scores_path) != decision["selector_scores_sha256"]):
            raise ValueError("Frozen score-selection artifacts changed")
    else:
        if enabled:
            scores = []
            for mode in MODES:
                emit_message(f"Score selector: {method} {mode} on reserved validation events")
                scores.append(ensemble_predict(root, np.concatenate((p, q)), device, stein_mode=mode,
                                               scoring_settings=settings))
            paired = np.column_stack(scores)
            ps, qs = paired[:len(p)], paired[len(p):]
        else:
            ps, qs = np.empty((0, 2)), np.empty((0, 2))
        decision = choose(ps, qs, cfg, cfg["fallback"][method], enabled)
        qt, qa = threshold_split(len(qs), cfg["seed"])
        atomic_write(scores_path, lambda path: save_npz(path, p_scores=ps, q_scores=qs,
                     p_event_ids=populations["p_ids"] if enabled else np.empty((0, 2), dtype=np.uint64),
                     q_event_ids=populations["q_ids"] if enabled else np.empty((0, 2), dtype=np.uint64),
                     q_threshold_indices=qt, q_assessment_indices=qa))
        decision.update(identity=identity, schema=2, selector_scores_sha256=file_digest(scores_path),
                        p_source=populations["p_source"], q_source=populations["q_source"],
                        evaluation_labels_used=False, evaluation_scores_used=False,
                        event_labels_used_by_selector=False, oracle_pool_definitions_use_truth=method != "riddle",
                        utility="weighted held-out p acceptance minus q acceptance at q-defined tail thresholds",
                        uncertainty_note="Paired bootstrap intervals are exploratory finite-sample evidence, not coverage guarantees.",
                        q_proxy_limitation="RIDDLE decisions depend on generated-background accuracy; this does not certify data closure.")
        write_json(saved_path, decision)
    decision["selector_scores_file"] = str(scores_path.relative_to(root))
    write_json(root / "score_selection.json", decision)
    emit_message(f"Score selector: {method} chose {decision['selected_mode']} ({decision['reason']})")
    return decision


def predict(root, z, device, settings, decision):
    from .campaign import ensemble_predict
    predictions = {}
    endpoints = {}
    for mode in MODES:
        selected = mode == decision["selected_mode"]
        value = ensemble_predict(root, z, device, stein_mode=mode, scoring_settings=settings,
                                 return_raw=selected,
                                 return_members=selected, return_accepted_members=selected)
        if selected:
            predictions[mode] = value
            endpoints[mode] = value[0]
        else:
            endpoints[mode] = value
    return predictions[decision["selected_mode"]], endpoints
