import numpy as np

from riddle.storage import atomic_write, save_npz, write_json
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
from .integrity import require_finite
from .options import effective_features, feature_options


def run(args, contract):
    fraction_values = fractions(args.fractions)
    acceleration = install_tensor_batches(args.device)
    output = args.output
    latent_root = output / "background"
    recovery = EpochRecovery(latent_root, contract, args.resume, **resume_policy(args))
    settings = args.settings
    active = effective_features(settings["riddle"])
    enhanced = any(active.values())
    development = None
    mass_conditioning = settings["riddle"].get("mass_conditioning", False)
    physical_inputs = settings["riddle"].get("input_space") == "physical"
    score_scope = "signal_region" if mass_conditioning else "full_region"
    if enhanced:
        from .mapping import prepare as prepare_mapping
        if getattr(args, "mapping_experiment", None) is not None:
            from .mapping_experiment import prepare_frozen
            prepare_mapping = lambda *a, **kw: prepare_frozen(*a, **kw, experiment=args.mapping_experiment)
        selection, mapper, development = prepare_mapping(args.data, latent_root, args.seed, args.device,
            background=settings["background"], options=feature_options(settings["riddle"]),
            data_policy=settings["riddle"].get("data_policy", DEFAULT_POLICY),
            residual_batch_size=settings["riddle"]["training"]["batch_size"])
    else:
        selection = prepare(args.data, latent_root, args.seed, args.device, recovery, settings=settings["background"])
    training, validation = ordered_map(
        np.load,
        [latent_root / name for name in ("training_latents.npy", "validation_latents.npy")],
        args.io_workers,
    )
    if not enhanced:
        mapper = Mapper(args.data, latent_root, selection["inference_mapping_epoch"], args.device)
    background_reference = None
    if physical_inputs:
        training, validation = mapper.physical_development(args.data, settings["background"]["reference_samples"])
        background_reference = mapper.physical_reference(validation)
    dimensions = training.shape[1] - 3 - int(physical_inputs)
    background_correction = None
    background_correction_decision = None
    if settings["riddle"].get("background_correction", "none") == "bgcorr_40_reguide":
        if development is None:
            raise ValueError("bgcorr_40_reguide requires the enhanced mapping pipeline and reserved correction roles")
        for role in ("correction_train", "correction_val", "closure"):
            if role not in development:
                raise ValueError(f"Missing reserved {role} role required by bgcorr_40_reguide")
        from .background_correction import train as train_background_correction
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
            **({"selection_validation": np.load(latent_root / "mixture_validation_latents.npy")} if enhanced else {}),
            **({"background_reference": background_reference} if physical_inputs else {}),
            **({"background_correction": background_correction} if background_correction is not None else {}),
            **({"member_splits": development.get("member_splits"),
                "source_ids": development["residual_train"].get("ids")} if enhanced else {}),
        )
        validate_normalization(output / "density", dimensions + int(mass_conditioning) + int(physical_inputs), args.device)
        exports = {}
        calibration_raw = {}
        try:
            for partition, (rows, region, z, mask, preprocessing_mask, score_domain_mask) in mapped.items():
                scores, fit_scores = ensemble_predict(output / "density", z, args.device, return_members=True)
                require_finite(scores, "RIDDLE ensemble scores")
                acceptance[partition] = {
                    **region_acceptance(rows[:, -1], mask, region),
                    "preprocessing": region_acceptance(rows[:, -1], preprocessing_mask, region),
                    "score_scope": score_scope,
                    "legacy_fields": "full and signal_region count effective scored events, not preprocessing alone",
                }
                aligned = np.full(len(rows), np.nan, dtype=scores.dtype)
                aligned[mask] = scores
                aligned_fits = np.full((len(fit_scores), len(rows)), np.nan, dtype=fit_scores.dtype)
                aligned_fits[:, mask] = fit_scores
                arrays = dict(
                    mass=rows[:, 0],
                    is_signal_region=region,
                    labels=rows[:, -1].astype(np.int8),
                    mask=mask,
                    preprocessing_mask=preprocessing_mask,
                    score_domain_mask=score_domain_mask,
                    score_scope=np.array(score_scope),
                    scores=aligned,
                    physical=rows[:, 1:-1],
                    **({"density_inputs": z[:, :-1], "background_log_density": z[:, -1]}
                       if physical_inputs else {"latent": z}),
                    fit_scores=aligned_fits,
                    fit_indices=np.array([m["fit_index"] for m in result["members"]], dtype=np.int64),
                    fit_seeds=np.array([m["seed"] for m in result["members"]], dtype=np.uint32),
                    fit_directories=np.array([m["directory"] for m in result["members"]]),
                )
                if partition in evaluation_ids: arrays["event_ids"] = evaluation_ids[partition]
                exports[partition] = arrays
            if active["score_flow"]:
                # A failed density evaluation in calibration/closure spends the
                # same bounded fit retry budget as any other required export.
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
            arrays["raw_scores"] = arrays["scores"].copy()
            arrays["score_kind"] = np.array("log_density_ratio")
    from .production import fit_acceptance
    health = dict(result["health"], calibration_status=calibration_status)
    health.update(fit_acceptance(health))
    write_json(output / "method_health.json", health)
    for partition, arrays in exports.items():
        atomic_write(output / f"{partition}_scores.npz", lambda p: save_npz(p, **arrays))
    write_json(output / "mapping_acceptance.json", acceptance)
    write_json(
        output / "protocol.json",
        {
            **PROTOCOL,
            "production_policy": PRODUCTION_POLICY,
            "score_scope": score_scope,
            "score_mask_definition": "preprocessing_mask AND score_domain_mask",
            "input_features": input_features(settings, contract["inputs"]),
            "features": dimensions,
            "layers": settings["riddle"]["flow"]["layers"],
            "blocks": settings["riddle"]["flow"]["num_blocks"],
            "hidden_features": settings["riddle"]["flow"]["hidden_features"],
            **settings["riddle"]["training"],
            "gradient_clip": f"flow parameters only; norm {settings['riddle']['training']['gradient_clip_norm']}",
            "ensemble": (
                f"equal-weight mean over accepted fits; {settings['riddle']['training']['selected_checkpoints']} validation-selected epochs per fit with "
                + ("validation-likelihood checkpoint weights" if feature_options(settings["riddle"]).get("checkpoint_weighting") == "validation_likelihood" else "equal checkpoint weights")
            ),
            "settings": settings,
            "implementation": ("riddle_bgcorr_40_reguide_v3_multiwindow_closure" if background_correction is not None
                               else "riddle_bgcorr_gaussian_fallback_v3_multiwindow_closure" if background_correction_decision is not None
                               else "riddle_default_six_features_v2_hc_baseline"),
            "data_policy": ("mapping_component_diagnostic_v1" if getattr(args,"mapping_experiment",None)
                            else settings["riddle"].get("data_policy", DEFAULT_POLICY)),
            "requested_features": {k: feature_options(settings["riddle"])[k] for k in active},
            "effective_features": active,
            **({"splits": ("Explicit historical study roles; mixture profiling reuses residual validation; final test untouched"
                if settings['riddle'].get('data_policy') == 'study_replay_v1' else
                "Disjoint mapping, residual development, common mixture-selection, reserved evidence, calibration training/validation, and sideband closure roles; final test untouched")} if enhanced else {}),
            "feature_dependencies": {"hard_bg": "inactive when guided_fit is disabled"},
            "score": ("log p_signal(z|mjj) ensemble - log q_phi(z|mjj)" if background_correction is not None else
                      "logit conditional background percentile" if active["score_flow"] else "log mean residual/background density ratio"),
            "raw_score": ("log p_signal(z|mjj) ensemble - log q_phi(z|mjj)" if background_correction is not None else
                          "log mean residual/background density ratio"),
            "background_correction": (None if background_correction is None else {
                "mode": background_correction["mode"], "protocol": background_correction["protocol"],
                "epochs": background_correction["epochs"], "selected_epoch": background_correction["selected_epoch"],
                "model_sha256": background_correction["model_sha256"],
                "validation_gate": background_correction["validation_gate"],
                "pseudo_sr_closure": background_correction["pseudo_sr_closure"],
                "independent_closure_diagnostic": background_correction.get("independent_closure_diagnostic"),
                "full_search_closure_status": "not_evaluated",
                "training_roles": ["correction_train", "correction_val"],
                "shared_across_residual_fits": True,
                "guide": "retrained against q_phi(z|m) samples at matched masses",
                "residual_initialization": "q_phi",
                "denominator": "q_phi(z|m)",
            }),
            "background_correction_decision": (None if background_correction_decision is None else
                                                background_correction_decision["selection"]),
            "fit_score_note": "Individual density scores through the frozen ensemble calibrator; not independently calibrated member models" if active["score_flow"] else "individual density ratios",
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
            "accepted_fits": result["valid_runs"],
            "excluded_fits": args.runs - result["valid_runs"],
            "fit_recovery": settings["riddle"]["fit_recovery"],
            "initialization": settings["riddle"]["initialization"],
            "fractions": args.fractions,
            "selected_checkpoints": result["selected_checkpoints"],
            "acceleration": execution_report(acceleration),
            **({"name": ("RIDDLE bgcorr_40_reguide" if background_correction is not None else
                         "RIDDLE bgcorr Gaussian fallback" if background_correction_decision is not None else
                         "RIDDLE + mass-conditioned residual"),
                "mass_conditioning": True,
                "inputs": "SR latents plus mass context (mjj - 3.5 TeV) / 0.2 TeV; no truth labels",
                "score": ("log(mean signal density p(z|mjj)) - log q_phi(z|mjj)" if background_correction is not None else
                          "log(mean signal density p(z|mjj)) - log standard-normal latent density"),
                "background": ("shared sideband-trained q_phi(z|mjj); no mass PDF factor" if background_correction is not None else
                               "standard-normal latent density conditional on mass; no mass PDF factor"),
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
