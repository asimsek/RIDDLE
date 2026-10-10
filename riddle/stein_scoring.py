import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtri

from .integrity import SCIENTIFIC_VERSION, require_finite
from .storage import atomic_write, digest, file_digest, save_array, save_npz, write_json

SCORING_PROTOCOL = "stein_scoring_v6_full_b_reserved_selection"
REFERENCE_SEEDS = {"mass_context": 93001, "background_sample": 93002, "split": 93003}
SELECTOR_REFERENCE_SEEDS = {"mass_context": 94001, "background_sample": 94002}


def _json_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _background_identity(inputs):
    info = inputs.get("background_correction")
    return "standard_normal" if info is None else info["source_sha256"]


def _scoring(settings):
    from .settings import validate_residual

    return validate_residual(settings)["stein"]["scoring"]


def _mode_identity(cfg):
    value = {
        "mode": cfg["mode"],
        "reference_samples": int(cfg["reference_samples"]),
        "reference_split": float(cfg["reference_split"]),
        "mass_bins": int(cfg["mass_bins"]),
    }
    if cfg["mode"] in ("sic_preserving", "tail_focus"):
        value.update(
            energy_weight=float(cfg["energy_weight"]),
            operator_weight=float(cfg["operator_weight"]),
            operator_gate_z=float(cfg["operator_gate_z"]),
            operator_temperature=float(cfg["operator_temperature"]),
        )
    elif cfg["mode"] == "hybrid":
        value["beta"] = float(cfg["beta"])
    elif cfg["mode"] == "hybrid_gated":
        value.update(
            beta=float(cfg["beta"]),
            local_gate_z=float(cfg["local_gate_z"]),
            local_temperature=float(cfg["local_temperature"]),
        )
    return value


def _support_identity(cfg):
    guard = cfg["support_guard"]
    return {
        "enabled": bool(guard["enabled"]),
        "statistic": guard["statistic"],
        "mass_bins": int(guard["mass_bins"]),
        "gate_quantile": float(guard["gate_quantile"]),
        "gate_z": float(ndtri(float(guard["gate_quantile"]))),
        "weight": float(guard["weight"]),
        "temperature": float(guard["temperature"]),
        "radius_excludes_mass": True,
    }


def _scoring_identity(cfg):
    return {
        "mode": _mode_identity(cfg),
        "support_guard": _support_identity(cfg),
        "final_mass_bins": int(cfg["final_mass_bins"]),
        "final_transform": cfg["final_transform"],
        "final_power": float(cfg["final_power"]),
    }


def _ensemble_identity(root, selected, cfg, reference_meta):
    root = Path(root)
    return {
        "scoring_protocol": SCORING_PROTOCOL,
        "precalibration_scoring": _mode_identity(cfg),
        "reference_A_sha256": reference_meta["reference_A_sha256"],
        "reference_B_sha256": reference_meta["reference_B_sha256"],
        "background_model_hash": reference_meta["background_model_hash"],
        "mapping_hash": reference_meta["mapping_hash"],
        "selected_fits": [{
            "fit_index": int(m["fit_index"]),
            "directory": m["directory"],
            "epochs": [int(e) for e in m["epochs"]],
            "checkpoint_sha256": {
                str(int(e)): file_digest(root / m["directory"] / f"residual_epoch_{int(e)}.pt")
                for e in m["epochs"]
            },
        } for m in selected],
    }


def _mass_cdf(reference_scores, reference_mass, values, mass, bins):
    reference_scores = np.asarray(reference_scores, dtype=np.float64)
    reference_mass = np.asarray(reference_mass, dtype=np.float64)
    values = np.asarray(values, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    if (reference_scores.ndim != 1 or reference_mass.shape != reference_scores.shape
            or values.ndim != 1 or mass.shape != values.shape or len(reference_scores) < 2
            or not np.isfinite(reference_scores).all() or not np.isfinite(reference_mass).all()
            or not np.isfinite(values).all() or not np.isfinite(mass).all()):
        raise ValueError("Invalid conditional background calibration arrays")
    count = min(int(bins), len(reference_scores))
    order = np.argsort(reference_mass, kind="stable")
    groups = np.array_split(order, count)
    centers = np.asarray([reference_mass[g].mean() for g in groups], dtype=np.float64)
    sorted_scores = [np.sort(reference_scores[g], kind="stable") for g in groups]
    epsilon = max(np.finfo(np.float64).eps, 0.5 / (min(map(len, groups)) + 1.0))

    def percentile(sample, query):
        left = np.searchsorted(sample, query, side="left")
        right = np.searchsorted(sample, query, side="right")
        return (left + right) / (2.0 * len(sample))

    if count == 1:
        result = percentile(sorted_scores[0], values)
    else:
        upper = np.searchsorted(centers, mass, side="right")
        upper = np.clip(upper, 1, count - 1)
        lower = upper - 1
        below = mass <= centers[0]
        above = mass >= centers[-1]
        lower[below] = upper[below] = 0
        lower[above] = upper[above] = count - 1
        result = np.empty(len(values), dtype=np.float64)
        for lo in np.unique(lower):
            lo_mask = lower == lo
            hi_values = np.unique(upper[lo_mask])
            for hi in hi_values:
                select = lo_mask & (upper == hi)
                p0 = percentile(sorted_scores[int(lo)], values[select])
                if lo == hi:
                    result[select] = p0
                else:
                    p1 = percentile(sorted_scores[int(hi)], values[select])
                    span = centers[int(hi)] - centers[int(lo)]
                    if span <= 0.0:
                        result[select] = 0.5 * (p0 + p1)
                    else:
                        weight = (mass[select] - centers[int(lo)]) / span
                        result[select] = p0 + np.clip(weight, 0.0, 1.0) * (p1 - p0)
    return np.clip(result, epsilon, 1.0 - epsilon), {
        "mass_bins": count,
        "mass_bin_centers": centers.tolist(),
        "bin_counts": [int(len(g)) for g in groups],
        "quantile_epsilon": float(epsilon),
        "tie_handling": "midrank_searchsorted_left_right",
        "mass_interpolation": "linear_between_equal_occupancy_bin_centers",
    }


def conditional_gaussianize(reference_scores, reference_mass, values, mass, bins):
    percentile, metadata = _mass_cdf(reference_scores, reference_mass, values, mass, bins)
    result = ndtri(percentile)
    if not np.isfinite(result).all():
        raise FloatingPointError("Nonfinite Gaussianized Stein score")
    return result, metadata


def conditional_percentile(reference_scores, reference_mass, values, mass, bins):
    return _mass_cdf(reference_scores, reference_mass, values, mass, bins)


def _pew_score(zp, ze, zw, cfg):
    boost = np.logaddexp(
        0.0, (zw - float(cfg["operator_gate_z"])) / float(cfg["operator_temperature"])
    )
    return zp + float(cfg["energy_weight"]) * ze + float(cfg["operator_weight"]) * boost


def _latent_radius(z):
    z = np.asarray(z)
    if z.ndim != 2 or z.shape[1] < 2 or not np.isfinite(z).all():
        raise ValueError("Invalid mapped latents for Stein support radius")
    radius = np.linalg.norm(np.asarray(z[:, :-1], dtype=np.float64), axis=1)
    if radius.shape != (len(z),) or not np.isfinite(radius).all():
        raise FloatingPointError("Nonfinite Stein support radius")
    return radius


def _support_penalties(reference, z, cfg, *, include_reference,
                       background_scores=None, reference_background_scores=None):
    reference = np.asarray(reference)
    z = np.asarray(z)
    if reference.ndim != 2 or z.ndim != 2 or reference.shape[1] != z.shape[1] or len(reference) < 2:
        raise ValueError("Stein support calibration latents are misaligned")
    if not np.isfinite(reference).all() or not np.isfinite(z).all():
        raise ValueError("Nonfinite Stein support calibration latents")
    guard = cfg["support_guard"]
    reference_radius = _latent_radius(reference)
    radius = _latent_radius(z)
    reference_mass = np.asarray(reference[:, -1], dtype=np.float64)
    mass = np.asarray(z[:, -1], dtype=np.float64)
    query_radius = np.concatenate((radius, reference_radius)) if include_reference else radius
    query_mass = np.concatenate((mass, reference_mass)) if include_reference else mass
    zr, metadata = conditional_gaussianize(
        reference_radius, reference_mass, query_radius, query_mass, int(guard["mass_bins"])
    )
    gate_z = float(ndtri(float(guard["gate_quantile"])))
    penalties = float(guard["weight"]) * np.logaddexp(
        0.0, (zr - gate_z) / float(guard["temperature"])
    )
    if guard["statistic"] == "radius_and_qscore":
        norms = []
        for values, points in ((background_scores, z), (reference_background_scores, reference)):
            values = np.asarray(values)
            if values.shape != (len(points), points.shape[1] - 1) or not np.isfinite(values).all():
                raise ValueError("Gradient support requires aligned finite frozen background scores")
            norms.append(np.linalg.norm(values.astype(np.float64), axis=1))
        query_norm = np.concatenate(norms) if include_reference else norms[0]
        zq, qmeta = conditional_gaussianize(
            norms[1], reference_mass, query_norm, query_mass, int(guard["mass_bins"])
        )
        gradient_penalties = float(guard["weight"]) * np.logaddexp(
            0.0, (zq - gate_z) / float(guard["temperature"])
        )
        penalties = np.maximum(penalties, gradient_penalties)
        metadata = {**metadata, "background_gradient_calibration": qmeta,
                    "background_gradient_statistic": "norm_gradient_z_log_q",
                    "combination": "maximum_radius_and_gradient_penalty",
                    "truth_labels_used": False}
    penalty = penalties[:len(z)]
    reference_penalty = penalties[len(z):] if include_reference else None
    if penalty.shape != (len(z),) or not np.isfinite(penalty).all():
        raise FloatingPointError("Nonfinite Stein support penalty")
    if include_reference and (reference_penalty.shape != (len(reference),)
                              or not np.isfinite(reference_penalty).all()):
        raise FloatingPointError("Nonfinite Stein reference support penalty")
    return penalty, reference_penalty, metadata


def _support_scores(root, selected, reference, z, device, cfg):
    if cfg["support_guard"]["statistic"] == "radius":
        return {}
    if not selected:
        raise ValueError("Gradient support requires a frozen selected ensemble")
    directory = Path(root) / selected[0]["directory"]
    inputs = json.loads((directory / "residual_training_inputs.json").read_text())
    identity = _background_identity(inputs)
    metadata = json.loads((Path(root) / "stein_scoring_reference.json").read_text())
    if identity != metadata["background_model_hash"]:
        raise ValueError("Gradient support background differs from reference B")
    model = _load_background(directory, inputs, device)
    cache = Path(root) / ".resume/stein_qscore_cache"
    return dict(background_scores=_qscore(z, model, device, cfg, cache, identity),
                reference_background_scores=_qscore(reference, model, device, cfg, cache, identity))


def _support_penalty(reference, z, cfg, **scores):
    penalty, _, metadata = _support_penalties(reference, z, cfg, include_reference=False, **scores)
    return penalty, metadata


def _load_reference_b(root):
    root = Path(root)
    reference_meta = json.loads((root / "stein_scoring_reference.json").read_text())
    if reference_meta.get("scoring_protocol") != SCORING_PROTOCOL:
        raise ValueError("Stein reference B uses a different scoring protocol")
    with np.load(root / "stein_scoring_reference.npz", allow_pickle=False) as archive:
        reference = np.ascontiguousarray(archive["reference_B"], dtype=np.float32)
    if reference.ndim != 2 or len(reference) < 2 or not np.isfinite(reference).all():
        raise ValueError("Invalid Stein reference B array")
    if reference_meta.get("reference_B_sha256") != digest(reference):
        raise ValueError("Stein reference B changed")
    return reference, reference_meta


def prepare_reference(root, members, validation, device, settings, mapping_identity, *, allow_device_change=False):
    root = Path(root)
    if not members:
        raise ValueError("Stein scoring reference requires at least one valid fit")
    cfg = _scoring(settings)
    validation = np.asarray(validation, dtype=np.float32)
    if validation.ndim != 2 or len(validation) < 2 or not np.isfinite(validation).all():
        raise ValueError("Invalid reserved validation latents for Stein scoring")
    member_inputs = []
    for member in members:
        path = root / member["directory"] / "residual_training_inputs.json"
        inputs = json.loads(path.read_text())
        if inputs.get("core") != "stein_witness":
            raise ValueError("Stein scoring reference received a non-Stein fit")
        member_inputs.append(inputs)
    backgrounds = {_background_identity(inputs) for inputs in member_inputs}
    features = {inputs.get("features") for inputs in member_inputs}
    if len(backgrounds) != 1 or len(features) != 1 or next(iter(features)) != validation.shape[1]:
        raise ValueError("Stein fits do not share one background/mapping feature identity")
    background_hash = next(iter(backgrounds))
    mapping_hash = _json_digest(mapping_identity)
    reference_settings = {
        "reference_samples": int(cfg["reference_samples"]),
        "reference_split": float(cfg["reference_split"]),
    }
    sampling_contract = {
        "protocol": SCORING_PROTOCOL,
        **reference_settings,
        "reference_settings_hash": _json_digest(reference_settings),
        "seeds": REFERENCE_SEEDS,
        "background_model_hash": background_hash,
        "mapping_hash": mapping_hash,
        "mass_context_source_sha256": digest(validation[:, -1]),
    }
    meta_path = root / "stein_scoring_reference.json"
    npz_path = root / "stein_scoring_reference.npz"
    if meta_path.exists() or npz_path.exists():
        if not (meta_path.exists() and npz_path.exists()):
            raise ValueError("Incomplete persisted Stein scoring reference")
        saved = json.loads(meta_path.read_text())
        from .resume import check_derived_contract
        check_derived_contract(saved.get("sampling_contract"), sampling_contract,
                               allow_device_change=allow_device_change,
                               hash_paths={("mapping_hash",)},
                               history_path=root / ".resume/reference_device_history.json")
        if saved.get("reference_file_sha256") != file_digest(npz_path):
            raise ValueError("Persisted Stein scoring reference file changed")
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != {"reference_A", "reference_B"}:
                raise ValueError("Invalid persisted Stein scoring reference arrays")
            a, b = archive["reference_A"], archive["reference_B"]
        expected_split = int(round(int(cfg["reference_samples"]) * float(cfg["reference_split"])))
        if (a.shape != (expected_split, validation.shape[1])
                or b.shape != (int(cfg["reference_samples"]) - expected_split, validation.shape[1])
                or a.dtype != np.float32 or b.dtype != np.float32
                or not np.isfinite(a).all() or not np.isfinite(b).all()
                or saved.get("reference_A_events") != len(a) or saved.get("reference_B_events") != len(b)
                or saved.get("reference_A_sha256") != digest(a) or saved.get("reference_B_sha256") != digest(b)):
            raise ValueError("Persisted Stein scoring reference changed")
        return saved

    count = int(cfg["reference_samples"])
    contexts = np.random.default_rng(REFERENCE_SEEDS["mass_context"]).choice(
        validation[:, -1], count, replace=True
    ).astype(np.float32)
    from .training import residual_background_sample

    first = root / members[0]["directory"]
    reference = residual_background_sample(
        first, contexts, count, REFERENCE_SEEDS["background_sample"], device,
        batch_size=int(cfg["inference_batch_size"]),
    )
    permutation = np.random.default_rng(REFERENCE_SEEDS["split"]).permutation(count)
    split = int(round(count * float(cfg["reference_split"])))
    if min(split, count - split) < 512:
        raise ValueError("Stein scoring reference split is too small")
    a = np.ascontiguousarray(reference[permutation[:split]], dtype=np.float32)
    b = np.ascontiguousarray(reference[permutation[split:]], dtype=np.float32)
    atomic_write(npz_path, lambda p: save_npz(p, reference_A=a, reference_B=b))
    metadata = {
        "schema": 1,
        "scoring_protocol": SCORING_PROTOCOL,
        "sampling_contract": sampling_contract,
        "reference_A_events": len(a),
        "reference_B_events": len(b),
        "reference_A_sha256": digest(a),
        "reference_B_sha256": digest(b),
        "reference_file_sha256": file_digest(npz_path),
        "background_model_hash": background_hash,
        "mapping_hash": mapping_hash,
        "reference_settings_hash": _json_digest(reference_settings),
        "truth_labels_used": False,
    }
    write_json(meta_path, metadata)
    return metadata


def prepare_selector_reference(root, selected, settings, device):
    root = Path(root)
    reference, reference_meta = _load_reference_b(root)
    cfg = _scoring(settings)
    count = cfg["auto_switch"]["reference_c_samples"]
    if not selected:
        raise ValueError("Selector reference C requires selected Stein fits")
    for member in selected:
        inputs = json.loads((root / member["directory"] / "residual_training_inputs.json").read_text())
        if (inputs.get("core") != "stein_witness" or inputs.get("features") != reference.shape[1]
                or _background_identity(inputs) != reference_meta["background_model_hash"]):
            raise ValueError("Selector reference C background identity differs from A/B")
        correction = inputs.get("background_correction")
        if correction is not None and file_digest(root / member["directory"] / "background_correction.pt") != correction["local_sha256"]:
            raise ValueError("Selector reference C fit-local background changed")
    contract = {
        "scoring_protocol": SCORING_PROTOCOL,
        "reference_C_events": count,
        "seeds": SELECTOR_REFERENCE_SEEDS,
        "background_model_hash": reference_meta["background_model_hash"],
        "mapping_hash": reference_meta["mapping_hash"],
        "mass_context_source": "reference_B_mass_empirical_distribution",
        "mass_context_source_sha256": digest(reference[:, -1]),
        "reference_B_sha256": reference_meta["reference_B_sha256"],
    }
    directory = root / "stein_scoring_reference_C" / _json_digest(contract)
    directory.mkdir(parents=True, exist_ok=True)
    meta_path, npz_path = directory / "reference.json", directory / "reference.npz"
    if meta_path.exists():
        saved = json.loads(meta_path.read_text())
        if (saved.get("sampling_contract") != contract or not npz_path.is_file()
                or saved.get("reference_file_sha256") != file_digest(npz_path)):
            raise ValueError("Persisted selector reference C changed")
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != {"reference_C"}:
                raise ValueError("Invalid selector reference C arrays")
            c = archive["reference_C"]
        if (c.shape != (count, reference.shape[1]) or c.dtype != np.float32
                or not np.isfinite(c).all() or digest(c) != saved.get("reference_C_sha256")):
            raise ValueError("Persisted selector reference C changed")
        return c, saved
    from .training import residual_background_sample

    contexts = np.random.default_rng(SELECTOR_REFERENCE_SEEDS["mass_context"]).choice(
        reference[:, -1], count, replace=True
    ).astype(np.float32)
    c = np.ascontiguousarray(residual_background_sample(
        root / selected[0]["directory"], contexts, count,
        SELECTOR_REFERENCE_SEEDS["background_sample"], device,
        batch_size=cfg["inference_batch_size"],
    ), dtype=np.float32)
    if c.shape != (count, reference.shape[1]) or not np.isfinite(c).all():
        raise ValueError("Invalid generated selector reference C")
    atomic_write(npz_path, lambda path: save_npz(path, reference_C=c))
    saved = {"schema": 1, "sampling_contract": contract, "reference_C_sha256": digest(c),
             "reference_file": str(npz_path.relative_to(root)),
             "reference_file_sha256": file_digest(npz_path), "truth_labels_used": False}
    write_json(meta_path, saved)
    return c, saved


def _load_background(output, inputs, device):
    info = inputs.get("background_correction")
    if info is None:
        return None
    from .background_correction import load_local

    loaded = load_local(output, inputs["settings"], inputs["features"] - 1, device)
    if loaded is None:
        raise ValueError("Missing fit-local corrected background for Stein scoring")
    model, saved = loaded
    if saved.get("source_sha256") != info.get("source_sha256"):
        raise ValueError("Stein scoring background identity changed")
    return model


def _qscore(inputs, background_model, device, cfg, cache_root, background_hash):
    from .stein import _background_score_array
    from .enhancements import deterministic_spline_sums

    array = np.asarray(inputs, dtype=np.float32)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    key = _json_digest({"protocol": SCORING_PROTOCOL, "input": digest(array), "background": background_hash})
    path = cache_root / f"qscore_{key}.npy"
    metadata_path = cache_root / f"qscore_{key}.json"
    identity = {
        "scoring_protocol": SCORING_PROTOCOL,
        "input_sha256": digest(array),
        "background_model_hash": background_hash,
    }
    if path.exists() or metadata_path.exists():
        if not (path.exists() and metadata_path.exists()):
            raise ValueError("Incomplete cached Stein background score")
        value = np.load(path, allow_pickle=False)
        metadata = json.loads(metadata_path.read_text())
        if (metadata.get("identity") != identity
                or value.shape != (len(array), array.shape[1] - 1)
                or value.dtype != np.float32 or not np.isfinite(value).all()
                or metadata.get("array_sha256") != digest(value)):
            raise ValueError("Invalid cached Stein background score")
        return value
    runtime = {}
    with deterministic_spline_sums():
        value = _background_score_array(
            array, background_model, device, mass_conditioning=True,
            batch_size=int(cfg["qscore_batch_size"]), runtime_metadata=runtime,
        )
    atomic_write(path, lambda p: save_array(p, value))
    write_json(metadata_path, {
        "identity": identity, "events": int(len(array)), "features": int(array.shape[1] - 1),
        "array_sha256": digest(value), **runtime,
    })
    return value


def _load_checkpoint(output, model, epoch, device):
    from .stein import PROTOCOL

    checkpoint = torch.load(Path(output) / f"residual_epoch_{epoch}.pt", map_location=device, weights_only=True)
    if (checkpoint.get("scientific_version") != SCIENTIFIC_VERSION
            or checkpoint.get("core") != "stein_witness"
            or checkpoint.get("stein_protocol") != PROTOCOL
            or checkpoint.get("epoch") != epoch):
        raise ValueError("Stein checkpoint identity differs from requested selection")
    model.load_state_dict(checkpoint["model"])


def _checkpoint_values(model, inputs, qscore, regularization, device, batch_size, need_derivatives):
    from .stein import _split_inputs, _stein_operator_values

    inputs = np.asarray(inputs, dtype=np.float32)
    qscore = None if qscore is None else np.asarray(qscore, dtype=np.float32)
    if qscore is not None and qscore.shape != (len(inputs), inputs.shape[1] - 1):
        raise ValueError("Stein q-score is misaligned")
    if need_derivatives and qscore is None:
        raise ValueError("Stein derivative scoring requires the background q-score")
    potential_parts, local_parts, operator_parts, energy_parts = [], [], [], []
    size = int(batch_size)
    offset = 0
    while offset < len(inputs):
        stop = min(offset + size, len(inputs))
        try:
            x = torch.as_tensor(inputs[offset:stop], dtype=torch.float32, device=device)
            if need_derivatives:
                qs = torch.as_tensor(qscore[offset:stop], dtype=torch.float32, device=device)
                with torch.enable_grad():
                    operator, energy, potential = _stein_operator_values(model, x, qs, training=False)
                local = operator - 0.5 * float(regularization) * energy
                local_parts.append(local.detach().cpu().numpy())
                operator_parts.append(operator.detach().cpu().numpy())
                energy_parts.append(energy.detach().cpu().numpy())
                potential_parts.append((-potential).detach().cpu().numpy())
            else:
                with torch.inference_mode():
                    latent, context = _split_inputs(x, model.mass_conditioning)
                    potential_parts.append((-model(latent, context)).cpu().numpy())
            offset = stop
        except torch.cuda.OutOfMemoryError:
            if torch.device(device).type != "cuda" or size <= 256:
                raise
            torch.cuda.empty_cache()
            size = max(256, size // 2)
    potential = np.concatenate(potential_parts).astype(np.float64)
    if not np.isfinite(potential).all():
        raise FloatingPointError("Nonfinite Stein potential score")
    if not need_derivatives:
        return potential, None, None, None, size
    local = np.concatenate(local_parts).astype(np.float64)
    operator = np.concatenate(operator_parts).astype(np.float64)
    energy = np.concatenate(energy_parts).astype(np.float64)
    if not np.isfinite(local).all() or not np.isfinite(operator).all() or not np.isfinite(energy).all():
        raise FloatingPointError("Nonfinite Stein derivative score")
    return potential, local, operator, energy, size


def _validate_calibration_archive(path, saved, order, reference):
    if saved.get("calibration_file_sha256") != file_digest(path):
        raise ValueError("Persisted Stein member scoring calibration changed")
    expected = {"mass"}
    for epoch in order:
        expected.update({f"p_{int(epoch)}", f"h_{int(epoch)}", f"w_{int(epoch)}", f"e_{int(epoch)}"})
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected:
            raise ValueError("Incomplete Stein member scoring calibration arrays")
        for name in expected:
            value = archive[name]
            if value.shape != (len(reference),) or value.dtype != np.float32 or not np.isfinite(value).all():
                raise ValueError("Invalid Stein member scoring calibration arrays")
        if not np.array_equal(archive["mass"], reference[:, -1].astype(np.float32)):
            raise ValueError("Stein member calibration mass is misaligned with reference A")


def _calibration(output, order, device, cfg, scoring_root):
    from .stein import build_potential

    output = Path(output)
    scoring_root = Path(scoring_root)
    inputs = json.loads((output / "residual_training_inputs.json").read_text())
    reference_meta = json.loads((scoring_root / "stein_scoring_reference.json").read_text())
    if reference_meta.get("scoring_protocol") != SCORING_PROTOCOL:
        raise ValueError("Stein calibration reference uses a different scoring protocol")
    with np.load(scoring_root / "stein_scoring_reference.npz", allow_pickle=False) as archive:
        reference = np.ascontiguousarray(archive["reference_A"], dtype=np.float32)
    if reference_meta.get("reference_A_sha256") != digest(reference):
        raise ValueError("Stein calibration reference A changed")
    background_hash = _background_identity(inputs)
    if background_hash != reference_meta["background_model_hash"]:
        raise ValueError("Stein calibration background differs from shared q reference")
    metadata_path = output / "stein_scoring_calibration.json"
    arrays_path = output / "stein_scoring_calibration.npz"
    contract = {
        "scoring_protocol": SCORING_PROTOCOL,
        "training_protocol": inputs.get("stein_protocol"),
        "reference_A_sha256": reference_meta["reference_A_sha256"],
        "background_model_hash": background_hash,
        "mapping_hash": reference_meta["mapping_hash"],
        "features": int(inputs["features"]),
        "epochs": [int(e) for e in order],
        "checkpoint_sha256": {str(int(e)): file_digest(output / f"residual_epoch_{int(e)}.pt") for e in order},
        "witness_regularization": float(inputs["settings"]["stein"]["witness_regularization"]),
    }
    if metadata_path.exists() or arrays_path.exists():
        if not (metadata_path.exists() and arrays_path.exists()):
            raise ValueError("Incomplete Stein member scoring calibration")
        saved = json.loads(metadata_path.read_text())
        if saved.get("contract") != contract:
            raise ValueError("Stein member scoring calibration provenance changed")
        _validate_calibration_archive(arrays_path, saved, order, reference)
        return saved, arrays_path

    background = _load_background(output, inputs, device)
    qscore = _qscore(
        reference, background, device, cfg, scoring_root / ".resume/stein_qscore_cache", background_hash
    )
    model = build_potential(device, reference.shape[1], inputs["settings"], initialization="random").eval()
    model.requires_grad_(False)
    payload = {"mass": reference[:, -1].astype(np.float32)}
    effective = int(cfg["inference_batch_size"])
    for epoch in order:
        _load_checkpoint(output, model, int(epoch), device)
        p, h, w, e, used = _checkpoint_values(
            model, reference, qscore, contract["witness_regularization"], device,
            int(cfg["inference_batch_size"]), True,
        )
        payload[f"p_{int(epoch)}"] = p.astype(np.float32)
        payload[f"h_{int(epoch)}"] = h.astype(np.float32)
        payload[f"w_{int(epoch)}"] = w.astype(np.float32)
        payload[f"e_{int(epoch)}"] = e.astype(np.float32)
        effective = min(effective, used)
    atomic_write(arrays_path, lambda p: save_npz(p, **payload))
    metadata = {
        "schema": 1,
        "contract": contract,
        "effective_inference_batch_size": int(effective),
        "calibration_file_sha256": file_digest(arrays_path),
        "truth_labels_used": False,
    }
    write_json(metadata_path, metadata)
    _validate_calibration_archive(arrays_path, metadata, order, reference)
    return metadata, arrays_path


def member_scores(output, order, z, device, *, mode=None, scoring_root=None, normalization_checks=None,
                  scoring_settings=None, background_scores=None, calibration_arrays=None, strict_batch_size=False):
    from .stein import build_potential, _checkpoint_weights

    output = Path(output)
    z = np.asarray(z, dtype=np.float32)
    inputs = json.loads((output / "residual_training_inputs.json").read_text())
    if inputs.get("core") != "stein_witness" or inputs.get("features") != z.shape[1] or not np.isfinite(z).all():
        raise ValueError("Invalid Stein scoring inputs")
    cfg = _scoring(inputs["settings"] if scoring_settings is None else scoring_settings)
    mode = cfg["mode"] if mode is None else mode
    if mode not in ("potential_raw", "potential_qnorm", "local_qnorm", "hybrid", "hybrid_gated", "sic_preserving", "tail_focus"):
        raise ValueError("Unknown Stein scoring mode")
    if mode != cfg["mode"]:
        cfg = dict(cfg)
        cfg["mode"] = mode
    model = build_potential(device, z.shape[1], inputs["settings"], initialization="random").eval()
    model.requires_grad_(False)
    weights = _checkpoint_weights(output, order)
    need_derivatives = mode in ("local_qnorm", "hybrid", "hybrid_gated", "sic_preserving", "tail_focus")
    background_hash = _background_identity(inputs)
    qscore = None
    calibration = None
    if mode != "potential_raw":
        if scoring_root is None:
            raise ValueError("q-normalized Stein scoring requires the shared scoring reference")
        if calibration_arrays is None:
            _, calibration_path = _calibration(output, order, device, cfg, scoring_root)
            calibration = np.load(calibration_path, allow_pickle=False)
        else:
            calibration = calibration_arrays
            keys = {"mass"} | {f"{field}_{int(epoch)}" for field in ("p", "h", "w", "e") for epoch in order}
            if (set(calibration) != keys or np.asarray(calibration["mass"]).ndim != 1
                    or any(np.asarray(calibration[key]).shape != np.asarray(calibration["mass"]).shape
                           or not np.isfinite(calibration[key]).all() for key in keys)):
                raise ValueError("Invalid preloaded frozen member calibration")
        if need_derivatives:
            if background_scores is None:
                background = _load_background(output, inputs, device)
                qscore = _qscore(
                    z, background, device, cfg, Path(scoring_root) / ".resume/stein_qscore_cache", background_hash
                )
            else:
                qscore = np.asarray(background_scores)
                if (qscore.shape != (len(z), z.shape[1] - 1) or qscore.dtype != np.float32
                        or not np.isfinite(qscore).all()):
                    raise ValueError("Invalid precomputed frozen background scores")
    combined = np.zeros(len(z), dtype=np.float64)
    mass = z[:, -1]
    regularization = float(inputs["settings"]["stein"]["witness_regularization"])
    for epoch, weight in zip(order, weights):
        _load_checkpoint(output, model, int(epoch), device)
        p, h, w, e, used = _checkpoint_values(
            model, z, qscore, regularization, device, int(cfg["inference_batch_size"]), need_derivatives
        )
        if strict_batch_size and used != int(cfg["inference_batch_size"]):
            raise RuntimeError("Frozen scoring required a smaller inference batch after GPU OOM; reduce --workers and resume")
        if mode == "potential_raw":
            values = p
        else:
            ref_mass = calibration["mass"]
            zp, _ = conditional_gaussianize(calibration[f"p_{int(epoch)}"], ref_mass, p, mass, cfg["mass_bins"])
            if mode == "potential_qnorm":
                values = zp
            elif mode in ("sic_preserving", "tail_focus"):
                ze, _ = conditional_gaussianize(calibration[f"e_{int(epoch)}"], ref_mass, e, mass, cfg["mass_bins"])
                zw, _ = conditional_gaussianize(calibration[f"w_{int(epoch)}"], ref_mass, w, mass, cfg["mass_bins"])
                values = _pew_score(zp, ze, zw, cfg)
            else:
                zh, _ = conditional_gaussianize(calibration[f"h_{int(epoch)}"], ref_mass, h, mass, cfg["mass_bins"])
                if mode == "local_qnorm":
                    values = zh
                elif mode == "hybrid":
                    values = zp + float(cfg["beta"]) * zh
                else:
                    boost = np.logaddexp(0.0, (zh - float(cfg["local_gate_z"])) / float(cfg["local_temperature"]))
                    values = zp + float(cfg["beta"]) * boost
        if normalization_checks is not None:
            from .production import score_diagnostics
            normalization_checks.append(dict(
                epoch=int(epoch), kind=f"stein_{mode}_reference",
                **score_diagnostics(values, stage=f"RIDDLE {output}, Stein checkpoint {epoch} {mode}"),
            ))
        combined += float(weight) * values
    if calibration is not None and hasattr(calibration, "close"):
        calibration.close()
    if not np.isfinite(combined).all():
        raise FloatingPointError("Nonfinite Stein checkpoint ensemble score")
    return combined


def _ensemble_reference(root, selected, device, cfg, settings, reference=None, reference_meta=None):
    root = Path(root)
    if reference is None or reference_meta is None:
        reference, reference_meta = _load_reference_b(root)
    else:
        reference = np.asarray(reference, dtype=np.float32)
        if (reference.ndim != 2 or len(reference) < 2 or not np.isfinite(reference).all()
                or reference_meta.get("scoring_protocol") != SCORING_PROTOCOL
                or reference_meta.get("reference_B_sha256") != digest(reference)):
            raise ValueError("Invalid preloaded Stein ensemble reference B")
    identity = _ensemble_identity(root, selected, cfg, reference_meta)
    cache = root / ".resume/stein_scoring"
    cache.mkdir(parents=True, exist_ok=True)
    key = _json_digest(identity)[:24]
    meta_path = cache / f"ensemble_{key}.json"
    npz_path = cache / f"ensemble_{key}.npz"
    if meta_path.exists() or npz_path.exists():
        if not (meta_path.exists() and npz_path.exists()):
            raise ValueError("Incomplete Stein ensemble scoring calibration")
        saved = json.loads(meta_path.read_text())
        if saved.get("identity") != identity:
            raise ValueError("Stein ensemble scoring calibration cache identity changed")
        if saved.get("reference_file_sha256") != file_digest(npz_path):
            raise ValueError("Persisted Stein ensemble scoring calibration changed")
        with np.load(npz_path, allow_pickle=False) as archive:
            if set(archive.files) != {"raw", "mass"}:
                raise ValueError("Invalid Stein ensemble scoring calibration arrays")
            raw = archive["raw"]
            mass = archive["mass"]
            if (raw.shape != (len(reference),) or mass.shape != (len(reference),)
                    or raw.dtype != np.float32 or mass.dtype != np.float32
                    or not np.isfinite(raw).all() or not np.isfinite(mass).all()
                    or not np.array_equal(mass, reference[:, -1].astype(np.float32))):
                raise ValueError("Invalid Stein ensemble scoring calibration arrays")
            return raw.astype(np.float64), reference, saved
    rows = [member_scores(root / m["directory"], m["epochs"], reference, device,
                          mode=cfg["mode"], scoring_root=root, scoring_settings=settings) for m in selected]
    raw = np.mean(np.stack(rows), axis=0)
    if not np.isfinite(raw).all():
        raise FloatingPointError("Nonfinite Stein reference-B ensemble score")
    atomic_write(npz_path, lambda p: save_npz(p, raw=raw.astype(np.float32), mass=reference[:, -1].astype(np.float32)))
    metadata = {
        "schema": 1,
        "identity": identity,
        "reference_file_sha256": file_digest(npz_path),
        "truth_labels_used": False,
    }
    write_json(meta_path, metadata)
    return raw.astype(np.float32).astype(np.float64), reference, metadata


def final_transform(root, selected, raw, z, device, settings):
    cfg = _scoring(settings)
    raw = np.asarray(raw, dtype=np.float64)
    z = np.asarray(z)
    if z.ndim != 2 or len(z) != len(raw) or z.shape[1] < 2 or not np.isfinite(z).all():
        raise ValueError("Invalid mapped latents for Stein final transform")
    if raw.shape != (len(z),) or not np.isfinite(raw).all():
        raise ValueError("Invalid Stein ensemble raw score")
    mass = np.asarray(z[:, -1], dtype=np.float64)
    support_metadata = _support_identity(cfg)
    guarded_raw = raw.copy()
    reference = None
    reference_meta = None
    reference_penalty = None
    transform = cfg["final_transform"]
    if cfg["support_guard"]["enabled"]:
        if reference is None:
            reference, reference_meta = _load_reference_b(root)
        penalty, reference_penalty, radius_metadata = _support_penalties(
            reference, z, cfg, include_reference=transform != "identity",
            **_support_scores(root, selected, reference, z, device, cfg),
        )
        guarded_raw = raw - penalty
        if not np.isfinite(guarded_raw).all():
            raise FloatingPointError("Nonfinite support-guarded Stein raw score")
        support_metadata = {**support_metadata, "radius_calibration": radius_metadata}
    if transform == "identity":
        return guarded_raw.copy(), guarded_raw.copy(), {
            "final_transform": "identity",
            "final_mass_bins": int(cfg["final_mass_bins"]),
            "support_guard": support_metadata,
        }
    reference_raw, reference_latents, reference_score_meta = _ensemble_reference(
        root, selected, device, cfg, settings, reference=reference, reference_meta=reference_meta
    )
    if reference is None:
        reference = reference_latents
    elif not np.array_equal(reference, reference_latents):
        raise ValueError("Stein support and ensemble reference B arrays disagree")
    reference_guarded = reference_raw.copy()
    if cfg["support_guard"]["enabled"]:
        reference_guarded = reference_raw - reference_penalty
        if not np.isfinite(reference_guarded).all():
            raise FloatingPointError("Nonfinite support-guarded Stein reference-B raw score")
        support_metadata = {**support_metadata, "reference_radius_calibration": support_metadata["radius_calibration"]}
    reference_mass = np.asarray(reference[:, -1], dtype=np.float64)
    uniform, cdf_meta = conditional_percentile(
        reference_guarded, reference_mass, guarded_raw, mass, int(cfg["final_mass_bins"])
    )
    if transform == "background_cdf":
        final = uniform
    elif transform == "background_cdf_power":
        final = uniform ** float(cfg["final_power"])
    else:
        raise ValueError("Unknown Stein final transform")
    if (np.any(uniform <= 0) or np.any(uniform >= 1) or not np.isfinite(uniform).all()
            or np.any(final < 0) or np.any(final > 1) or not np.isfinite(final).all()):
        raise FloatingPointError("Invalid CDF-based Stein final score")
    return final, guarded_raw.copy(), {
        "final_transform": transform,
        "final_mass_bins": int(cfg["final_mass_bins"]),
        "final_power": float(cfg["final_power"]),
        "cdf": cdf_meta,
        "support_guard": support_metadata,
        "ensemble_reference": reference_score_meta["identity"],
    }


def write_root_provenance(root, selected, accepted, settings, mapping_identity, final_metadata=None):
    root = Path(root)
    cfg = _scoring(settings)
    reference = json.loads((root / "stein_scoring_reference.json").read_text())
    if reference.get("scoring_protocol") != SCORING_PROTOCOL:
        raise ValueError("Stein root provenance reference uses a different scoring protocol")
    mapping_hash = _json_digest(mapping_identity)
    if reference.get("mapping_hash") != mapping_hash:
        raise ValueError("Stein root provenance mapping identity changed")
    checkpoint_hashes = {
        str(m["fit_index"]): {
            str(int(e)): file_digest(root / m["directory"] / f"residual_epoch_{int(e)}.pt")
            for e in m["epochs"]
        } for m in selected
    }
    selected_checkpoints = {str(m["fit_index"]): [int(e) for e in m["epochs"]] for m in selected}
    scoring_identity = {
        "scoring_protocol": SCORING_PROTOCOL,
        "scoring": _scoring_identity(cfg),
        "reference_A_sha256": reference["reference_A_sha256"],
        "reference_B_sha256": reference["reference_B_sha256"],
        "background_model_hash": reference["background_model_hash"],
        "mapping_hash": mapping_hash,
        "selected_fits": [int(m["fit_index"]) for m in selected],
        "selected_checkpoints": selected_checkpoints,
        "selected_checkpoint_sha256": checkpoint_hashes,
    }
    support = _support_identity(cfg)
    score_kind = (
        f"stein_{cfg['mode']}_support_guard_{cfg['final_transform']}"
        if support["enabled"] else f"stein_{cfg['mode']}_{cfg['final_transform']}"
    )
    path = root / "stein_scoring_calibration.json"
    existing = json.loads(path.read_text()) if path.exists() else None
    if existing is not None and existing.get("scoring_identity") != scoring_identity:
        raise ValueError("Stein scoring calibration provenance changed; use a new output")
    value = {
        "schema": 1,
        "scoring_protocol": SCORING_PROTOCOL,
        "scoring_identity": scoring_identity,
        "mode": cfg["mode"],
        "reference_samples": int(cfg["reference_samples"]),
        "reference_split": float(cfg["reference_split"]),
        "mass_bins": int(cfg["mass_bins"]),
        "energy_weight": float(cfg["energy_weight"]),
        "operator_weight": float(cfg["operator_weight"]),
        "operator_gate_z": float(cfg["operator_gate_z"]),
        "operator_temperature": float(cfg["operator_temperature"]),
        "beta": float(cfg["beta"]),
        "local_gate_z": float(cfg["local_gate_z"]),
        "local_temperature": float(cfg["local_temperature"]),
        "support_guard": support,
        "final_mass_bins": int(cfg["final_mass_bins"]),
        "final_transform": cfg["final_transform"],
        "final_power": float(cfg["final_power"]),
        "gamma": float(cfg["final_power"]),
        "q_reference_hashes": {"A": reference["reference_A_sha256"], "B": reference["reference_B_sha256"]},
        "q_reference_seeds": REFERENCE_SEEDS,
        "background_model_hash": reference["background_model_hash"],
        "mapping_hash": mapping_hash,
        "selected_fits": [int(m["fit_index"]) for m in selected],
        "accepted_fits": [int(m["fit_index"]) for m in accepted],
        "selected_checkpoints": selected_checkpoints,
        "selected_checkpoint_sha256": checkpoint_hashes,
        "conditional_calibration": "equal-occupancy mass bins with interpolated empirical midrank CDF",
        "precalibration_ensemble_score": f"stein_{cfg['mode']}",
        "discriminating_raw_score": (f"stein_{cfg['mode']}_support_guard" if support["enabled"]
                                     else f"stein_{cfg['mode']}"),
        "final_monotonic_score": score_kind,
        "support_reference": f"q-reference B {support['statistic']} excluding mass coordinate",
        "truth_labels_used": False,
        "settings": cfg,
    }
    if final_metadata is not None:
        value["final_calibration"] = final_metadata
    elif existing is not None and "final_calibration" in existing:
        value["final_calibration"] = existing["final_calibration"]
    write_json(path, value)
    return value
