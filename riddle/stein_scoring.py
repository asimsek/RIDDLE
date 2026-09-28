import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy.special import ndtri

from .integrity import SCIENTIFIC_VERSION, require_finite
from .storage import atomic_write, digest, file_digest, save_array, save_npz, write_json

SCORING_PROTOCOL = "stein_scoring_v3_qanchored_local_hybrid"
REFERENCE_SEEDS = {"mass_context": 93001, "background_sample": 93002, "split": 93003}


def _json_digest(value):
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()
    return hashlib.sha256(payload).hexdigest()


def _background_identity(inputs):
    info = inputs.get("background_correction")
    return "standard_normal" if info is None else info["source_sha256"]


def _scoring(settings):
    from .settings import validate_residual

    return validate_residual(settings)["stein"]["scoring"]


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


def prepare_reference(root, members, validation, device, settings, mapping_identity):
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
        if saved.get("sampling_contract") != sampling_contract:
            raise ValueError("Stein scoring reference provenance changed; use a new output")
        with np.load(npz_path, allow_pickle=False) as archive:
            a, b = archive["reference_A"], archive["reference_B"]
        if saved.get("reference_A_sha256") != digest(a) or saved.get("reference_B_sha256") != digest(b):
            raise ValueError("Persisted Stein scoring reference changed")
        return saved

    count = int(cfg["reference_samples"])
    contexts = np.random.default_rng(REFERENCE_SEEDS["mass_context"]).choice(
        validation[:, -1], count, replace=True
    ).astype(np.float32)
    from .training import residual_background_sample

    first = root / members[0]["directory"]
    reference = residual_background_sample(
        first, contexts, count, REFERENCE_SEEDS["background_sample"], device
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
        "settings_hash": _json_digest(cfg),
        "truth_labels_used": False,
    }
    write_json(meta_path, metadata)
    return metadata


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

    array = np.asarray(inputs, dtype=np.float32)
    cache_root = Path(cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)
    key = _json_digest({"input": digest(array), "background": background_hash})
    path = cache_root / f"qscore_{key}.npy"
    metadata_path = cache_root / f"qscore_{key}.json"
    identity = {"input_sha256": digest(array), "background_model_hash": background_hash}
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


def _checkpoint_values(model, inputs, qscore, regularization, device, batch_size, need_local):
    from .stein import _split_inputs, _stein_operator_values

    inputs = np.asarray(inputs, dtype=np.float32)
    qscore = None if qscore is None else np.asarray(qscore, dtype=np.float32)
    if qscore is not None and qscore.shape != (len(inputs), inputs.shape[1] - 1):
        raise ValueError("Stein q-score is misaligned")
    potential_parts, local_parts = [], []
    size = int(batch_size)
    offset = 0
    while offset < len(inputs):
        stop = min(offset + size, len(inputs))
        try:
            x = torch.as_tensor(inputs[offset:stop], dtype=torch.float32, device=device)
            if need_local:
                qs = torch.as_tensor(qscore[offset:stop], dtype=torch.float32, device=device)
                with torch.enable_grad():
                    operator, energy, potential = _stein_operator_values(model, x, qs, training=False)
                local = operator - 0.5 * float(regularization) * energy
                local_parts.append(local.detach().cpu().numpy())
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
    if not need_local:
        return potential, None, size
    local = np.concatenate(local_parts).astype(np.float64)
    if not np.isfinite(local).all():
        raise FloatingPointError("Nonfinite local Stein score")
    return potential, local, size


def _calibration(output, order, device, cfg, scoring_root):
    from .stein import build_potential

    output = Path(output)
    scoring_root = Path(scoring_root)
    inputs = json.loads((output / "residual_training_inputs.json").read_text())
    reference_meta = json.loads((scoring_root / "stein_scoring_reference.json").read_text())
    with np.load(scoring_root / "stein_scoring_reference.npz", allow_pickle=False) as archive:
        reference = np.ascontiguousarray(archive["reference_A"], dtype=np.float32)
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
        return saved, arrays_path

    background = _load_background(output, inputs, device)
    need_local = True
    qscore = _qscore(
        reference, background, device, cfg, scoring_root / ".resume/stein_qscore_cache", background_hash
    )
    model = build_potential(device, reference.shape[1], inputs["settings"], initialization="random").eval()
    model.requires_grad_(False)
    payload = {"mass": reference[:, -1].astype(np.float32)}
    effective = int(cfg["inference_batch_size"])
    for epoch in order:
        _load_checkpoint(output, model, int(epoch), device)
        p, h, used = _checkpoint_values(
            model, reference, qscore, contract["witness_regularization"], device,
            int(cfg["inference_batch_size"]), need_local,
        )
        payload[f"p_{int(epoch)}"] = p.astype(np.float32)
        payload[f"h_{int(epoch)}"] = h.astype(np.float32)
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
    return metadata, arrays_path


def member_scores(output, order, z, device, *, mode=None, scoring_root=None, normalization_checks=None,
                  scoring_settings=None):
    from .stein import build_potential, _checkpoint_weights

    output = Path(output)
    z = np.asarray(z, dtype=np.float32)
    inputs = json.loads((output / "residual_training_inputs.json").read_text())
    if inputs.get("core") != "stein_witness" or inputs.get("features") != z.shape[1] or not np.isfinite(z).all():
        raise ValueError("Invalid Stein scoring inputs")
    cfg = _scoring(inputs["settings"] if scoring_settings is None else scoring_settings)
    mode = cfg["mode"] if mode is None else mode
    if mode not in ("potential_raw", "potential_qnorm", "local_qnorm", "hybrid", "hybrid_gated"):
        raise ValueError("Unknown Stein scoring mode")
    model = build_potential(device, z.shape[1], inputs["settings"], initialization="random").eval()
    model.requires_grad_(False)
    weights = _checkpoint_weights(output, order)
    need_local = mode in ("local_qnorm", "hybrid", "hybrid_gated")
    background_hash = _background_identity(inputs)
    qscore = None
    calibration = None
    if mode != "potential_raw":
        if scoring_root is None:
            raise ValueError("q-normalized Stein scoring requires the shared scoring reference")
        _, calibration_path = _calibration(output, order, device, cfg, scoring_root)
        calibration = np.load(calibration_path, allow_pickle=False)
        if need_local:
            background = _load_background(output, inputs, device)
            qscore = _qscore(
                z, background, device, cfg, Path(scoring_root) / ".resume/stein_qscore_cache", background_hash
            )
    combined = np.zeros(len(z), dtype=np.float64)
    mass = z[:, -1]
    regularization = float(inputs["settings"]["stein"]["witness_regularization"])
    for epoch, weight in zip(order, weights):
        _load_checkpoint(output, model, int(epoch), device)
        p, h, _ = _checkpoint_values(
            model, z, qscore, regularization, device, int(cfg["inference_batch_size"]), need_local
        )
        if mode == "potential_raw":
            values = p
        else:
            ref_mass = calibration["mass"]
            zp, _ = conditional_gaussianize(calibration[f"p_{int(epoch)}"], ref_mass, p, mass, cfg["mass_bins"])
            if mode == "potential_qnorm":
                values = zp
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
    if calibration is not None:
        calibration.close()
    if not np.isfinite(combined).all():
        raise FloatingPointError("Nonfinite Stein checkpoint ensemble score")
    return combined


def _ensemble_reference(root, selected, device, cfg):
    root = Path(root)
    reference_meta = json.loads((root / "stein_scoring_reference.json").read_text())
    with np.load(root / "stein_scoring_reference.npz", allow_pickle=False) as archive:
        reference = np.ascontiguousarray(archive["reference_B"], dtype=np.float32)
    identity = {
        "scoring_protocol": SCORING_PROTOCOL,
        "reference_B_sha256": reference_meta["reference_B_sha256"],
        "mode": cfg["mode"],
        "beta": float(cfg["beta"]),
        "local_gate_z": float(cfg["local_gate_z"]),
        "local_temperature": float(cfg["local_temperature"]),
        "mass_bins": int(cfg["mass_bins"]),
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
        with np.load(npz_path, allow_pickle=False) as archive:
            return archive["raw"].astype(np.float64), archive["mass"].astype(np.float64), saved
    settings = json.loads((root / "ensemble_inputs.json").read_text()).get("settings")
    rows = [member_scores(root / m["directory"], m["epochs"], reference, device,
                          mode=cfg["mode"], scoring_root=root, scoring_settings=settings) for m in selected]
    raw = np.mean(np.stack(rows), axis=0)
    atomic_write(npz_path, lambda p: save_npz(p, raw=raw.astype(np.float32), mass=reference[:, -1].astype(np.float32)))
    metadata = {
        "schema": 1,
        "identity": identity,
        "reference_file_sha256": file_digest(npz_path),
        "truth_labels_used": False,
    }
    write_json(meta_path, metadata)
    return raw, reference[:, -1].astype(np.float64), metadata


def final_transform(root, selected, raw, mass, device, settings):
    cfg = _scoring(settings)
    raw = np.asarray(raw, dtype=np.float64)
    mass = np.asarray(mass, dtype=np.float64)
    transform = cfg["final_transform"]
    if transform == "identity":
        return raw.copy(), raw.copy(), {"final_transform": "identity"}
    reference_raw, reference_mass, reference_meta = _ensemble_reference(root, selected, device, cfg)
    uniform, cdf_meta = conditional_percentile(reference_raw, reference_mass, raw, mass, cfg["mass_bins"])
    if transform == "background_cdf":
        final = uniform
    elif transform == "background_cdf_power":
        final = uniform ** float(cfg["final_power"])
    else:
        raise ValueError("Unknown Stein final transform")
    if (np.any(uniform <= 0) or np.any(uniform >= 1) or not np.isfinite(uniform).all()
            or np.any(final < 0) or np.any(final > 1) or not np.isfinite(final).all()):
        raise FloatingPointError("Invalid CDF-based Stein final score")
    return final, uniform, {
        "final_transform": transform,
        "final_power": float(cfg["final_power"]),
        "cdf": cdf_meta,
        "ensemble_reference": reference_meta["identity"],
    }


def write_root_provenance(root, selected, accepted, settings, mapping_identity, final_metadata=None):
    root = Path(root)
    cfg = _scoring(settings)
    reference = json.loads((root / "stein_scoring_reference.json").read_text())
    value = {
        "schema": 1,
        "scoring_protocol": SCORING_PROTOCOL,
        "mode": cfg["mode"],
        "beta": float(cfg["beta"]),
        "local_gate_z": float(cfg["local_gate_z"]),
        "local_temperature": float(cfg["local_temperature"]),
        "gamma": float(cfg["final_power"]),
        "final_transform": cfg["final_transform"],
        "mass_bins": int(cfg["mass_bins"]),
        "q_reference_hashes": {"A": reference["reference_A_sha256"], "B": reference["reference_B_sha256"]},
        "q_reference_seeds": REFERENCE_SEEDS,
        "background_model_hash": reference["background_model_hash"],
        "mapping_hash": _json_digest(mapping_identity),
        "selected_fits": [int(m["fit_index"]) for m in selected],
        "accepted_fits": [int(m["fit_index"]) for m in accepted],
        "selected_checkpoints": {str(m["fit_index"]): [int(e) for e in m["epochs"]] for m in selected},
        "selected_checkpoint_sha256": {
            str(m["fit_index"]): {
                str(int(e)): file_digest(root / m["directory"] / f"residual_epoch_{int(e)}.pt")
                for e in m["epochs"]
            } for m in selected
        },
        "conditional_calibration": "equal-occupancy mass bins with interpolated empirical midrank CDF",
        "precalibration_ensemble_score": cfg["mode"],
        "discriminating_raw_score": (cfg["mode"] if cfg["final_transform"] == "identity"
                                     else "conditional_background_cdf"),
        "final_monotonic_score": cfg["final_transform"],
        "truth_labels_used": False,
        "settings": cfg,
    }
    if final_metadata is not None:
        value["final_calibration"] = final_metadata
    write_json(root / "stein_scoring_calibration.json", value)
    return value
