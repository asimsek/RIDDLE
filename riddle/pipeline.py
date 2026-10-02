import json
from pathlib import Path
import numpy as np

from riddle.storage import atomic_write, save_npz, write_json, digest, file_digest
from riddle.recovery import EpochRecovery
from riddle.acceleration import install_tensor_batches, execution_report
from .latent import prepare, Mapper
from .campaign import (train_campaign, ensemble_predict, fractions, PROTOCOL, validate_normalization,
                       MemberScoreError, reject_scoring_member)
from .runtime import ordered_map
from .settings import input_features
from .resume import resume_policy
from .production import evaluation_rows, region_acceptance, scoring_masks, PRODUCTION_POLICY
from .roles import DEFAULT_POLICY
from .integrity import require_finite, RIDDLE_BENCHMARK_LABELS, RIDDLE_BENCHMARK_SCIENTIFIC_VERSION
from .options import effective_features, feature_options


def _oracle_role_seed(seed, role):
    import hashlib

    token = int.from_bytes(hashlib.sha256(role.encode()).digest()[:4], "little")
    return int(np.random.SeedSequence([int(seed), token]).generate_state(1)[0])


def _oracle_half_split(length, seed, role):
    if length < 4:
        raise ValueError("RIDDLE oracle validation pool is too small")
    order = np.random.default_rng(_oracle_role_seed(seed, role)).permutation(length)
    cut = length // 2
    if cut < 2 or length - cut < 2:
        raise ValueError("RIDDLE oracle validation split is too small")
    return order[:cut], order[cut:]


def _map_oracle_rows(mapper, rows, expected_label):
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or len(rows) < 2 or np.any(rows[:, -1] != expected_label):
        raise ValueError("Invalid RIDDLE oracle population")
    if not np.all((rows[:, 0] > 3.3) & (rows[:, 0] < 3.7)):
        raise ValueError("RIDDLE oracle populations must be signal-region events")
    latent, mask = mapper.map(rows)
    if mask.shape != (len(rows),) or not bool(mask.all()) or len(latent) != len(rows):
        raise ValueError("RIDDLE oracle mapping changed event acceptance")
    latent_rows = np.column_stack((rows[:, 0], latent, np.ones(len(rows)), np.zeros(len(rows)))).astype(np.float32)
    return latent_rows, np.asarray(latent, dtype=np.float32), np.asarray(rows[:, 0], dtype=np.float32)


def _prepare_oracle_roles(args, mapper, training, validation, development):
    from .storage import digest

    data = Path(args.data)
    with np.load(data / "event_ids.npz", allow_pickle=False) as archive:
        ids = {name: archive[name] for name in archive.files}
    background_train = np.load(data / "innerdata_extrabkg_train.npy", allow_pickle=False)
    background_val = np.load(data / "innerdata_extrabkg_val.npy", allow_pickle=False)
    qval_indices, qclosure_indices = _oracle_half_split(len(background_val), args.seed, "oracle_background_validation")
    _, qtrain_z, qtrain_mass = _map_oracle_rows(mapper, background_train, 0)
    _, qval_all_z, qval_all_mass = _map_oracle_rows(mapper, background_val, 0)
    qval_z, qval_mass = qval_all_z[qval_indices], qval_all_mass[qval_indices]
    qclosure_z, qclosure_mass = qval_all_z[qclosure_indices], qval_all_mass[qclosure_indices]
    if args.method == "iad":
        ptrain = training
        pvalidation = validation
        member_splits = development.get("member_splits") if development is not None else None
        source_ids = development["residual_train"].get("ids") if development is not None else None
        p_definition = "RIDDLE signal-region data mixture"
        p_truth_selection = False
        p_train_ids = development["residual_train"].get("ids") if development is not None else ids["innerdata_train.npy"]
        p_validation_ids = development["evidence"].get("ids") if development is not None else ids["innerdata_val.npy"]
    elif args.method == "supervised":
        signal_train = np.load(data / "innerdata_extrasig_train.npy", allow_pickle=False)
        signal_val = np.load(data / "innerdata_extrasig_val.npy", allow_pickle=False)
        ptrain, _, _ = _map_oracle_rows(mapper, signal_train, 1)
        pvalidation, _, _ = _map_oracle_rows(mapper, signal_val, 1)
        member_splits = None
        source_ids = ids["innerdata_extrasig_train.npy"]
        p_definition = "pure simulated signal in the signal region"
        p_truth_selection = True
        p_train_ids = source_ids
        p_validation_ids = ids["innerdata_extrasig_val.npy"]
    else:
        raise ValueError("Unknown RIDDLE oracle method")
    receipt = {
        "schema": 2,
        "method": args.method,
        "p_definition": p_definition,
        "q_definition": "pure simulated background in the signal region",
        "p_truth_role_selection": p_truth_selection,
        "q_truth_role_selection": True,
        "p_training_events": int(len(ptrain)),
        "p_validation_events": int(len(pvalidation)),
        "q_training_events": int(len(qtrain_z)),
        "q_validation_events": int(len(qval_z)),
        "q_closure_events": int(len(qclosure_z)),
        "p_training_sha256": digest(ptrain),
        "p_validation_sha256": digest(pvalidation),
        "p_training_event_ids_sha256": digest(p_train_ids),
        "p_validation_event_ids_sha256": digest(p_validation_ids),
        "q_training_latents_sha256": digest(qtrain_z),
        "q_validation_latents_sha256": digest(qval_z),
        "q_closure_latents_sha256": digest(qclosure_z),
        "q_training_event_ids_sha256": digest(ids["innerdata_extrabkg_train.npy"]),
        "q_validation_event_ids_sha256": digest(ids["innerdata_extrabkg_val.npy"][qval_indices]),
        "q_closure_event_ids_sha256": digest(ids["innerdata_extrabkg_val.npy"][qclosure_indices]),
    }
    return {
        "training": ptrain,
        "validation": pvalidation,
        "member_splits": member_splits,
        "source_ids": source_ids,
        "qtrain_z": qtrain_z,
        "qtrain_mass": qtrain_mass,
        "qval_z": qval_z,
        "qval_mass": qval_mass,
        "qclosure_z": qclosure_z,
        "qclosure_mass": qclosure_mass,
        "receipt": receipt,
    }


def run(args, contract):
    fraction_values = fractions(args.fractions)
    acceleration = install_tensor_batches(args.device)
    output = args.output
    latent_root = output / "background"
    recovery = EpochRecovery(latent_root, contract, args.resume, **resume_policy(args))
    settings = args.settings
    oracle_method = args.method in RIDDLE_BENCHMARK_LABELS
    active = effective_features(settings["riddle"])
    core = settings["riddle"].get("core", "residual")
    enhanced = any(active.values()) or settings["riddle"].get("background_correction") == "bgcorr_40_reguide"
    development = None
    mass_conditioning = settings["riddle"].get("mass_conditioning", False)
    physical_inputs = settings["riddle"].get("input_space") == "physical"
    score_scope = "signal_region" if mass_conditioning else "full_region"
    mapping_reuse = None
    if enhanced:
        from .mapping import prepare as prepare_mapping
        if getattr(args, "mapping_experiment", None) is not None:
            from .mapping_experiment import prepare_frozen
            selection, mapper, development = prepare_frozen(
                args.data, latent_root, args.seed, args.device,
                background=settings["background"], options=feature_options(settings["riddle"]),
                data_policy=settings["riddle"].get("data_policy", DEFAULT_POLICY),
                residual_batch_size=settings["riddle"]["training"]["batch_size"],
                experiment=args.mapping_experiment,
            )
        else:
            selection, mapper, development, mapping_reuse = prepare_mapping(
                args.data, latent_root, args.seed, args.device,
                background=settings["background"], options=feature_options(settings["riddle"]),
                data_policy=settings["riddle"].get("data_policy", DEFAULT_POLICY),
                residual_batch_size=settings["riddle"]["training"]["batch_size"],
                reuse_candidates=getattr(args, "riddle_background_reuse_candidates", None),
            )
    else:
        selection = prepare(args.data, latent_root, args.seed, args.device, recovery, settings=settings["background"])
    training, validation = ordered_map(
        np.load,
        [latent_root / name for name in ("training_latents.npy", "validation_latents.npy")],
        args.io_workers,
    )
    if not enhanced:
        mapper = Mapper(args.data, latent_root, selection["inference_mapping_epoch"], args.device)
    mapping_identity = {
        "training_latents_sha256": digest(training),
        "validation_latents_sha256": digest(validation),
        "selection": selection,
    }
    selection_validation = np.load(latent_root / "mixture_validation_latents.npy") if enhanced else None
    member_splits = development.get("member_splits") if development is not None else None
    source_ids = development["residual_train"].get("ids") if development is not None else None
    oracle_roles = None
    for name in ("model.pt", "preprocessing.pt", "mapping_settings.json"):
        path = latent_root / name
        if path.is_file():
            mapping_identity[name] = file_digest(path)
    if mapping_reuse is not None:
        mapping_identity["background_reuse"] = mapping_reuse
    background_reference = None
    if physical_inputs:
        training, validation = mapper.physical_development(args.data, settings["background"]["reference_samples"])
        background_reference = mapper.physical_reference(validation)
    dimensions = training.shape[1] - 3 - int(physical_inputs)
    background_correction = None
    background_correction_decision = None
    if oracle_method:
        if core != "stein_witness" or not mass_conditioning or physical_inputs:
            raise ValueError("Idealized RIDDLE and Supervised RIDDLE require the active mapped, mass-conditioned Stein-witness flow")
        if settings["riddle"].get("background_correction", "none") != "bgcorr_40_reguide":
            raise ValueError("Idealized RIDDLE and Supervised RIDDLE require the active RIDDLE q_phi background stage")
        oracle_roles = _prepare_oracle_roles(args, mapper, training, validation, development)
        training = oracle_roles["training"]
        validation = oracle_roles["validation"]
        selection_validation = None
        member_splits = oracle_roles["member_splits"]
        source_ids = oracle_roles["source_ids"]
        mapping_identity["oracle_roles"] = oracle_roles["receipt"]
        mapping_identity["p_training_sha256"] = oracle_roles["receipt"]["p_training_sha256"]
        mapping_identity["p_validation_sha256"] = oracle_roles["receipt"]["p_validation_sha256"]
        if args.method == "supervised":
            mapping_identity.pop("training_latents_sha256", None)
            mapping_identity.pop("validation_latents_sha256", None)
            mapping_identity.pop("background_reuse", None)
        write_json(output / "oracle_roles.json", oracle_roles["receipt"])
        from .background_correction import train_oracle
        background_correction_decision = train_oracle(
            output / "density" / "background_correction",
            oracle_roles["qtrain_z"], oracle_roles["qtrain_mass"],
            oracle_roles["qval_z"], oracle_roles["qval_mass"],
            oracle_roles["qclosure_z"], oracle_roles["qclosure_mass"],
            settings=settings["riddle"], seed=(int(args.seed)+91000) % 2**32, device=args.device,
            reuse_candidates=getattr(args, "oracle_background_reuse_candidates", None),
            current_code=contract.get("code", {}),
            allow_code_change=resume_policy(args)["allow_code_change"],
        )
        background_correction = background_correction_decision["descriptor"]
    elif settings["riddle"].get("background_correction", "none") == "bgcorr_40_reguide":
        if development is None:
            raise ValueError("bgcorr_40_reguide requires the enhanced mapping pipeline and reserved correction roles")
        for role in ("correction_train", "correction_val", "closure"):
            if role not in development:
                raise ValueError(f"Missing reserved {role} role required by bgcorr_40_reguide")
        from .background_correction import train as train_background_correction, reuse as reuse_background_correction
        background_correction_decision = None
        if mapping_reuse is not None:
            background_correction_decision = reuse_background_correction(
                output / "density" / "background_correction", mapping_reuse["source_result"],
                settings=settings["riddle"], seed=(int(args.seed)+91000) % 2**32, device=args.device,
            )
        if background_correction_decision is None:
            background_correction_decision = train_background_correction(
                output / "density" / "background_correction",
                development["correction_train"]["z"], development["correction_train"]["mass"],
                development["correction_val"]["z"], development["correction_val"]["mass"],
                settings=settings["riddle"], seed=(int(args.seed)+91000) % 2**32, device=args.device,
                closure_z=development["closure"]["z"], closure_mass=development["closure"]["mass"])
        background_correction = background_correction_decision["descriptor"]
    acceptance, mapped = {}, {}
    evaluation_ids = {}
    ids_path = args.data / "event_ids.npz"
    if ids_path.exists():
        with np.load(ids_path, allow_pickle=False) as archive:
            source_event_ids = {k: archive[k] for k in archive.files}
    else:
        source_event_ids = None
    for partition, suffix in (("validation", "val"), ("test", "test"), ("signal_region", None)):
        names = (
            (f"innerdata_{suffix}.npy", f"outerdata_{suffix}.npy")
            if suffix
            else ("innerdata_test.npy", "innerdata_extrabkg_test.npy", "innerdata_extrasig.npy")
        )
        if enhanced and partition == "validation":
            rows, region = evaluation_rows([development["evidence"]["rows"], development["closure"]["rows"]],
                                          ("innerdata_val.npy", "outerdata_val.npy"))
        else:
            rows, region = evaluation_rows(
                ordered_map(np.load, [args.data / name for name in names], args.io_workers), names)
        if source_event_ids is not None:
            evaluation_ids[partition] = (np.concatenate([development[k]["source_ids"] for k in ("evidence", "closure")])
                if enhanced and partition == "validation" else np.concatenate([source_event_ids[n] for n in names]))
        z, preprocessing_mask = mapper.physical(rows) if physical_inputs else mapper.map(rows)
        preprocessing_mask, score_domain_mask, mask = scoring_masks(preprocessing_mask, region, score_scope)
        if mass_conditioning:
            from .model import with_mass_context
            z = z[score_domain_mask[preprocessing_mask]]
            z = (np.column_stack((with_mass_context(z[:, :-1], rows[mask, 0]), z[:, -1])).astype(np.float32)
                 if physical_inputs else with_mass_context(z, rows[mask, 0]))
        mapped[partition] = (rows, region, z, mask, preprocessing_mask, score_domain_mask)
    while True:
        result = train_campaign(
            training,
            validation,
            output / "density",
            epochs=args.epochs,
            runs=args.runs,
            seed=args.seed,
            device=args.device,
            fraction_values=fraction_values,
            initialization=settings["riddle"]["initialization"],
            workers=args.workers,
            io_workers=args.io_workers,
            torch_threads=args.torch_threads,
            settings=settings["riddle"],
            **({"selection_validation": selection_validation} if selection_validation is not None else {}),
            **({"background_reference": background_reference} if physical_inputs else {}),
            **({"background_correction": background_correction} if background_correction is not None else {}),
            **({"member_splits": member_splits} if member_splits is not None else {}),
            **({"source_ids": source_ids} if source_ids is not None else {}),
            mapping_identity=mapping_identity,
            truth_labels_used=(oracle_roles["receipt"]["p_truth_role_selection"] if oracle_method else False),
            **({
                "ensemble_reuse_candidates": getattr(args, "supervised_ensemble_reuse_candidates", None),
                "ensemble_reuse_contract": contract,
                "allow_ensemble_reuse_code_change": resume_policy(args)["allow_code_change"],
            } if args.method == "supervised" else {}),
        )
        validate_normalization(output / "density", dimensions + int(mass_conditioning) + int(physical_inputs), args.device)
        exports = {}
        calibration_raw = {}
        try:
            for partition, (rows, region, z, mask, preprocessing_mask, score_domain_mask) in mapped.items():
                prediction = ensemble_predict(
                    output / "density", z, args.device, return_members=True, return_accepted_members=True,
                    return_raw=core == "stein_witness", scoring_settings=settings["riddle"],
                )
                if core == "stein_witness":
                    scores, raw_scores, fit_scores, accepted_fit_scores = prediction
                else:
                    scores, fit_scores, accepted_fit_scores = prediction
                    raw_scores = scores
                require_finite(scores, "RIDDLE ensemble scores")
                require_finite(raw_scores, "RIDDLE discriminating raw scores")
                acceptance[partition] = {
                    **region_acceptance(rows[:, -1], mask, region),
                    "preprocessing": region_acceptance(rows[:, -1], preprocessing_mask, region),
                    "score_scope": score_scope,
                    "legacy_fields": "full and signal_region count effective scored events, not preprocessing alone",
                }
                aligned = np.full(len(rows), np.nan, dtype=scores.dtype)
                aligned[mask] = scores
                aligned_raw = np.full(len(rows), np.nan, dtype=raw_scores.dtype)
                aligned_raw[mask] = raw_scores
                aligned_fits = np.full((len(fit_scores), len(rows)), np.nan, dtype=fit_scores.dtype)
                aligned_fits[:, mask] = fit_scores
                aligned_accepted_fits = np.full((len(accepted_fit_scores), len(rows)), np.nan, dtype=accepted_fit_scores.dtype)
                aligned_accepted_fits[:, mask] = accepted_fit_scores
                arrays = dict(
                    mass=rows[:, 0],
                    is_signal_region=region,
                    labels=rows[:, -1].astype(np.int8),
                    mask=mask,
                    preprocessing_mask=preprocessing_mask,
                    score_domain_mask=score_domain_mask,
                    score_scope=np.array(score_scope),
                    scores=aligned,
                    raw_scores=aligned_raw,
                    physical=rows[:, 1:-1],
                    **({"density_inputs": z[:, :-1], "background_log_density": z[:, -1]}
                       if physical_inputs else {"latent": z}),
                    fit_scores=aligned_fits,
                    fit_indices=np.array([m["fit_index"] for m in result["members"]], dtype=np.int64),
                    fit_seeds=np.array([m["seed"] for m in result["members"]], dtype=np.uint32),
                    fit_directories=np.array([m["directory"] for m in result["members"]]),
                    accepted_fit_scores=aligned_accepted_fits,
                    accepted_fit_indices=np.array([m["fit_index"] for m in result["accepted_members"]], dtype=np.int64),
                    accepted_fit_seeds=np.array([m["seed"] for m in result["accepted_members"]], dtype=np.uint32),
                    accepted_fit_directories=np.array([m["directory"] for m in result["accepted_members"]]),
                    accepted_fit_score_kind=np.array(
                        f"stein_{settings['riddle']['stein']['scoring']['mode']}" if core == "stein_witness"
                        else "log_density_ratio"
                    ),
                )
                if core == "stein_witness":
                    scoring_cfg = settings["riddle"]["stein"]["scoring"]
                    support_suffix = "_support_guard" if scoring_cfg["support_guard"]["enabled"] else ""
                    arrays["score_kind"] = np.array(
                        f"stein_{scoring_cfg['mode']}{support_suffix}_{scoring_cfg['final_transform']}"
                    )
                    arrays["fit_score_kind"] = np.array(
                        f"stein_{scoring_cfg['mode']}"
                    )
                if partition in evaluation_ids: arrays["event_ids"] = evaluation_ids[partition]
                exports[partition] = arrays
            if active["score_flow"]:
                # Apply the same bounded retry budget to required density evaluations.

                for role in ("calibration_train", "calibration_val", "closure"):
                    calibration_raw[role] = ensemble_predict(output / "density", development[role]["z"], args.device)
        except MemberScoreError as error:
            reject_scoring_member(output / "density", error.member, error)
            continue
        break
    calibration_status = "unassessed"
    if active["score_flow"]:
        from .enhancements import train_score_flow, chunks
        import torch
        ct, cv = development["calibration_train"], development["calibration_val"]
        calibrator = train_score_flow(output / "calibration",
            calibration_raw["calibration_train"], ct["mass"],
            calibration_raw["calibration_val"], cv["mass"],
            seed=(args.seed+7000) % 2**32, epochs=feature_options(settings["riddle"])["score_flow_epochs"], device=args.device)
        for arrays in exports.values():
            mask = arrays["mask"]
            arrays["raw_scores"] = arrays["scores"].copy()
            arrays["raw_fit_scores"] = arrays["fit_scores"].copy()
            mass = torch.as_tensor(arrays["mass"][mask], dtype=torch.float32)
            arrays["scores"][mask] = chunks(calibrator, torch.as_tensor(arrays["raw_scores"][mask], dtype=torch.float32), mass,
                                           device=args.device).numpy()
            for fit, raw in zip(arrays["fit_scores"], arrays["raw_fit_scores"]):
                fit[mask] = chunks(calibrator, torch.as_tensor(raw[mask], dtype=torch.float32), mass, device=args.device).numpy()
            arrays["score_kind"] = np.array("background_percentile_logit")
            arrays["fit_score_kind"] = np.array("member_ratio_mapped_by_frozen_ensemble_calibrator")
        closure = development["closure"]
        raw = calibration_raw["closure"]
        score = chunks(calibrator, torch.as_tensor(raw, dtype=torch.float32),
                       torch.as_tensor(closure["mass"], dtype=torch.float32), device=args.device).numpy()
        from .calibration import assess_closure
        closure_report = assess_closure(score, closure["mass"])
        calibration_status = closure_report["status"]
        write_json(output / "calibration/closure.json", closure_report)
        atomic_write(output / "calibration/closure_scores.npz", lambda p: save_npz(p, mass=closure["mass"], scores=score, raw_scores=raw))
    else:
        for arrays in exports.values():
            if core != "stein_witness":
                arrays["raw_scores"] = arrays["scores"].copy()
                arrays["score_kind"] = np.array("log_density_ratio")
    from .production import fit_acceptance
    health = dict(result["health"], calibration_status=calibration_status)
    health.update(fit_acceptance(health))
    write_json(output / "method_health.json", health)
    for partition, arrays in exports.items():
        atomic_write(output / f"{partition}_scores.npz", lambda p: save_npz(p, **arrays))
    write_json(output / "mapping_acceptance.json", acceptance)
    stein_scoring = (json.loads((output / "density/stein_scoring_calibration.json").read_text())
                     if core == "stein_witness" else None)
    scoring_cfg = settings["riddle"]["stein"]["scoring"] if core == "stein_witness" else None
    support_enabled = bool(scoring_cfg["support_guard"]["enabled"]) if scoring_cfg is not None else False
    support_text = " with q-reference-B latent-radius support guard" if support_enabled else ""
    write_json(
        output / "protocol.json",
        {
            **PROTOCOL,
            "production_policy": PRODUCTION_POLICY,
            "score_scope": score_scope,
            "public_label": RIDDLE_BENCHMARK_LABELS.get(args.method, "RIDDLE"),
            "oracle_benchmark": None if not oracle_method else oracle_roles["receipt"],
            "supervised_ensemble_reuse": result.get("ensemble_reuse") if args.method == "supervised" else None,
            "score_mask_definition": "preprocessing_mask AND score_domain_mask",
            "input_features": input_features(settings, contract["inputs"]),
            "features": dimensions,
            "core": core,
            "mixture_loss": (None if core == "stein_witness" else PROTOCOL.get("mixture_loss")),
            "fraction_interpretation": ("not used by stein_witness" if core == "stein_witness" else PROTOCOL.get("fraction_interpretation")),
            "scan_score": ("configured q-anchored Stein event score; CDF-based modes are already in [0,1]"
                           if core == "stein_witness" else PROTOCOL.get("scan_score")),
            "objective": ("neural Stein witness objective against the fixed latent background score field" if core == "stein_witness" else "residual mixture likelihood"),
            "layers": (settings["riddle"]["stein"]["hidden_layers"] if core == "stein_witness" else settings["riddle"]["flow"]["layers"]),
            "blocks": (None if core == "stein_witness" else settings["riddle"]["flow"]["num_blocks"]),
            "hidden_features": (settings["riddle"]["stein"]["hidden_features"] if core == "stein_witness" else settings["riddle"]["flow"]["hidden_features"]),
            **settings["riddle"]["training"],
            "gradient_clip": f"{'witness' if core == 'stein_witness' else 'flow'} parameters only; norm {settings['riddle']['training']['gradient_clip_norm']}",
            "ensemble": (
                (f"arithmetic mean over {result['valid_runs']} selected Stein-witness fits from {result['accepted_runs']} safeguard-valid fits; "
                 f"{settings['riddle']['training']['selected_checkpoints']} validation-selected epochs per fit with equal checkpoint weights")
                if core == "stein_witness" else
                (f"equal-weight mean over {result['valid_runs']} selected fits from {result['accepted_runs']} safeguard-valid fits; "
                 f"{settings['riddle']['training']['selected_checkpoints']} validation-selected epochs per fit with "
                 + ("validation-likelihood checkpoint weights" if feature_options(settings["riddle"]).get("checkpoint_weighting") == "validation_likelihood" else "equal checkpoint weights"))
            ),
            "ensemble_fit_selection": result["ensemble_fit_selection"],
            "settings": settings,
            "implementation": ((f"riddle_stein_witness_v6_5_{scoring_cfg['mode']}"
                                f"{'_support_guard' if support_enabled else ''}_scoring") if core == "stein_witness" else
                               "riddle_v5_3_smooth_fm_bgcorr_40_reguide" if background_correction is not None
                               else "riddle_v5_3_smooth_fm_gaussian_fallback" if background_correction_decision is not None
                               else "riddle_v5_3_smooth_fm"),
            "data_policy": ("mapping_component_diagnostic_v1" if getattr(args,"mapping_experiment",None)
                            else settings["riddle"].get("data_policy", DEFAULT_POLICY)),
            "requested_features": {k: feature_options(settings["riddle"])[k] for k in active},
            "effective_features": active,
            **({"splits": (("RIDDLE mapping roles unchanged; p uses the RIDDLE signal-region data mixture and q uses disjoint pure-background SR train/validation/closure roles; final test untouched"
                              if args.method == "iad" else
                              "RIDDLE mapping roles unchanged; p uses disjoint pure-signal SR train/validation roles and q uses disjoint pure-background SR train/validation/closure roles; final test untouched")
                             if oracle_method else
                             "Explicit historical study roles; mixture profiling reuses residual validation; final test untouched"
                             if settings['riddle'].get('data_policy') == 'study_replay_v1' else
                             "Disjoint mapping, residual development, common mixture-selection, reserved evidence, calibration training/validation, and sideband closure roles; final test untouched")} if enhanced else {}),
            "feature_dependencies": {"hard_bg": "inactive when guided_fit is disabled"},
            "score": (f"Stein {scoring_cfg['mode']}{support_text} with "
                      f"{scoring_cfg['final_transform']} final transform"
                      if core == "stein_witness" else
                      "log p_signal(z|mjj) ensemble - log q_phi(z|mjj)" if background_correction is not None else
                      "logit conditional background percentile" if active["score_flow"] else "log mean residual/background density ratio"),
            "raw_score": ((f"Stein {scoring_cfg['mode']} ensemble score"
                           f"{' after q-reference-B latent-radius support correction' if support_enabled else ''} before final calibration")
                          if core == "stein_witness" else
                          "log p_signal(z|mjj) ensemble - log q_phi(z|mjj)" if background_correction is not None else
                          "log mean residual/background density ratio"),
            **({
                "unguarded_ensemble_score": f"arithmetic mean of selected per-fit {scoring_cfg['mode']} Stein scores before support correction",
                "support_guard": ("q-reference-B latent Euclidean radius excluding mass; label-free and truth-blind"
                                  if support_enabled else "disabled"),
            } if core == "stein_witness" else {}),
            "background_correction": (None if background_correction is None else {
                "mode": background_correction["mode"], "protocol": background_correction["protocol"],
                "epochs": background_correction["epochs"], "selected_epoch": background_correction["selected_epoch"],
                "model_sha256": background_correction["model_sha256"],
                "validation_gate": background_correction["validation_gate"],
                "pseudo_sr_closure": background_correction["pseudo_sr_closure"],
                "independent_closure_diagnostic": background_correction.get("independent_closure_diagnostic"),
                "full_search_closure_status": "not_evaluated",
                "training_roles": (["pure_background_sr_train", "pure_background_sr_validation"] if oracle_method else ["correction_train", "correction_val"]),
                "shared_across_residual_fits": True,
                "guide": ("not used by stein_witness" if core == "stein_witness" else "retrained against q_phi(z|m) samples at matched masses"),
                "residual_initialization": ("zero Stein witness" if core == "stein_witness" else "q_phi"),
                "denominator": ("q_phi_oracle(z|m) from pure signal-region background" if oracle_method else "q_phi(z|m)"),
                "reuse": background_correction_decision.get("reuse"),
            }),
            "background_correction_decision": (None if background_correction_decision is None else
                                                background_correction_decision["selection"]),
            "scan_background_reuse": (None if mapping_reuse is None else {
                "policy": mapping_reuse["policy"],
                "mapping": mapping_reuse,
                "background_correction": background_correction_decision.get("reuse")
                    if background_correction_decision is not None else None,
            }),
            "fit_score_note": ((f"per-fit {scoring_cfg['mode']} Stein scores after per-checkpoint q-normalization and checkpoint averaging; "
                                "support correction is ensemble-level and not included in fit_scores")
                               if core == "stein_witness" else
                               "Individual density scores through the frozen ensemble calibrator; not independently calibrated member models" if active["score_flow"] else "individual density ratios"),
            "coherent_mixture": result.get("coherent_mixture"),
            "calibration_status": calibration_status,
            "flow_selection": selection,
            "background_epochs": settings["background"]["epochs"],
            "background_configuration": {
                **settings["background"]["configuration"],
                "num_inputs": dimensions,
            },
            "reference_samples": settings["background"]["reference_samples"],
            "epochs": args.epochs,
            "fits": args.runs,
            "accepted_fits": result["accepted_runs"],
            "ensemble_fits": result["valid_runs"],
            "unselected_fits": result["accepted_runs"] - result["valid_runs"],
            "excluded_fits": args.runs - result["accepted_runs"],
            "fit_recovery": settings["riddle"]["fit_recovery"],
            "initialization": settings["riddle"]["initialization"],
            "fractions": args.fractions,
            "mass_fraction": ({
                **settings["riddle"]["mass_fraction"],
                "role": "training-only smooth f(mjj) gate learned from latent residual responsibilities",
                "uses_mjj_event_count_density": False,
                "included_in_final_score": False,
            } if settings["riddle"].get("mass_fraction", {}).get("enabled", False) else {"enabled": False}),
            "stein": (settings["riddle"]["stein"] if core == "stein_witness" else None),
            "stein_scoring": stein_scoring,
            "selected_checkpoints": result["selected_checkpoints"],
            "acceleration": execution_report(acceleration),
            **({"name": (RIDDLE_BENCHMARK_LABELS[args.method] if oracle_method else
                         "RIDDLE bgcorr_40_reguide" if background_correction is not None else
                         "RIDDLE bgcorr Gaussian fallback" if background_correction_decision is not None else
                         "RIDDLE + mass-conditioned residual"),
                "mass_conditioning": True,
                "inputs": ((("p: RIDDLE SR mapped data mixture; q: mapped pure-background SR reference"
                              if args.method == "iad" else
                              "p: mapped pure-signal SR reference; q: mapped pure-background SR reference")
                             + " with mjj context for conditional q-normalization"
                             + (", q-reference-B latent-radius support calibration" if support_enabled else "")
                             + ", and final conditional calibration")
                            if oracle_method and core == "stein_witness" else
                            ("SR mapped latents with mjj context for conditional q-normalization"
                             + (", q-reference-B latent-radius support calibration" if support_enabled else "")
                             + ", and final conditional calibration; no truth labels")
                            if core == "stein_witness" else
                            "SR latents plus mass context (mjj - 3.5 TeV) / 0.2 TeV; no truth labels"),
                "score": (f"{scoring_cfg['mode']} Stein score{support_text} with "
                          f"{scoring_cfg['final_transform']} final calibration"
                          if core == "stein_witness" else
                          "log(mean signal density p(z|mjj)) - log q_phi(z|mjj)" if background_correction is not None else
                          "log(mean signal density p(z|mjj)) - log standard-normal latent density"),
                "background": ("shared pure-background-SR-trained q_phi(z|mjj); no mass PDF factor"
                               if oracle_method and background_correction is not None else
                               "shared sideband-trained q_phi(z|mjj); no mass PDF factor" if background_correction is not None else
                               "standard-normal latent density conditional on mass; no mass PDF factor"),
                "mixture_fraction": ("not used" if core == "stein_witness" else
                                     "smooth f(mjj) learned from latent responsibilities; training only; excluded from score"
                                     if settings["riddle"].get("mass_fraction", {}).get("enabled", False) else "global fraction"),
                "context_features": ["mjj"], "scope": "signal_region"} if mass_conditioning else {}),
            **({"name": "RIDDLE physical + mass (CPU pilot)", "input_space": "physical",
                "inputs": "Logit-standardized physical features plus mass context; no learned latent transform in signal inputs",
                "score": "log(mean p_signal(x|mjj)) - log frozen p_background(x|mjj)",
                "background": "Same frozen sideband flow; evaluated in preprocessed physical coordinates; no mass PDF factor",
                "initialization_note": "Identity splines give Gaussian signal density in physical coordinates, not a background-matched density",
                "normalization_reference": "Independent samples from the frozen background at reserved-validation masses",
                "bookkeeping": "Cached background log density is never passed to the signal network"} if physical_inputs else {}),
        },
    )
