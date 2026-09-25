import json
import math
from pathlib import Path
import random
import numpy as np
import torch
from copy import deepcopy
from .settings import DEFAULTS, validate_residual
from riddle.worker_progress import ProgressStage
from riddle.storage import atomic_torch_save, write_json, digest, file_digest, rng_state, restore_rng, persist_boundary
from .model import (
    build_signal_flow,
    initial_fraction_logit,
    real_sr_latents,
    train_epoch,
    background_log_prob,
    signal_log_prob,
    residual_optimizer,
)
from .integrity import SCIENTIFIC_VERSION, ordered_epochs
from .options import feature_options, effective_features

ENHANCED_TRAINING_PROTOCOL = "riddle_v5_3_smooth_mass_fraction_no_guide_init_v1"


def enhanced_epoch(model, logit, ztrain, optimizer, options, additions, *, epoch, seed, guide,
                   fixed_fraction=False, progress=None, background_model=None,
                   mass_fraction_state=None, mass_fraction_settings=None):
    """Guided EM/mixture fitting with optional smooth f(m) and q_phi(z|m)."""
    from .enhancements import (chunks, contrastive_loss, clip_gradients, standard_normal_log_prob,
                               sharpen_responsibilities, tail_ranking_loss)
    from .integrity import mixture_log_density, require_finite
    device = logit.device
    if (np.asarray(ztrain).ndim != 2 or not len(ztrain)
            or not np.isfinite(ztrain).all()):
        raise ValueError("Enhanced residual fitting requires finite, nonempty two-dimensional inputs")
    z = torch.from_numpy(ztrain)
    conditional = bool(getattr(model, "mass_conditioning", False))
    latent = z[:, :-1] if conditional else z
    if background_model is not None:
        if not conditional:
            raise ValueError("Corrected latent background requires mass-conditioned residual inputs")
        from .background_correction import log_prob as corrected_log_prob
        logb = chunks(lambda x, c: corrected_log_prob(background_model, x, c),
                      latent, z[:, -1], device=device)
    else:
        logb = standard_normal_log_prob(latent)
    model.eval()
    guided = additions["guided_fit"]
    warm = guided and epoch < additions["guide_warmup_epochs"]
    mass_fraction_active = mass_fraction_state is not None
    if mass_fraction_active:
        if not conditional or fixed_fraction or mass_fraction_settings is None:
            raise ValueError("Smooth f(m) requires learned, mass-conditioned residual fitting")
        from .mass_fraction import logits as mass_fraction_logits

    if warm:
        if guide is None:
            raise ValueError("Guided warm-up requires frozen event responsibilities")
        weights = torch.as_tensor(guide).detach().cpu()
    else:
        logs = chunks(lambda x: signal_log_prob(model, x), z, device=device)
        if mass_fraction_active:
            gate_logit = torch.from_numpy(
                mass_fraction_logits(mass_fraction_state, ztrain[:, -1], mass_fraction_settings)
            ).to(dtype=torch.float64)
        else:
            gate_logit = logit.detach().cpu()
        weights = (torch.nn.functional.logsigmoid(gate_logit) + logs
                   - mixture_log_density(logs, logb, gate_logit)).exp().detach().float()
        if guided and not fixed_fraction:
            if mass_fraction_active:
                from .mass_fraction import fit as fit_mass_fraction, state_summary
                mass_fraction_state = fit_mass_fraction(
                    ztrain[:, -1], weights.numpy(), mass_fraction_state, mass_fraction_settings,
                    source="alternating_latent_responsibilities",
                )
                summary = state_summary(mass_fraction_state, ztrain[:, -1], mass_fraction_settings)
                fraction = float(summary["mean"])
            else:
                fraction = float(weights.double().mean().clamp(1e-8, 1-1e-8))
            with torch.no_grad():
                logit.fill_(np.log(fraction/(1-fraction)))
    if (not warm) and float(additions.get("responsibility_temperature", 1.0)) < 1.0:
        # Fit f(m) from unsharpened responsibilities; sharpening only guides the density update.

        weights = sharpen_responsibilities(weights, float(additions["responsibility_temperature"]))
    if weights.shape != (len(z),):
        raise ValueError("Residual responsibilities must have one entry per training event")
    require_finite(weights, "Residual responsibilities")
    if bool(((weights < 0) | (weights > 1)).any()):
        raise ValueError("Residual responsibilities must lie in [0,1]")
    weights = weights.detach()
    mean_weight = float(weights.double().mean())
    if mean_weight <= 0 or not np.isfinite(mean_weight):
        raise FloatingPointError("Invalid residual responsibilities")
    loader_generator = (torch.Generator().manual_seed(int(seed)+22000+epoch)
                        if background_model is not None else None)
    loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(z, logb, weights),
                                         batch_size=options["batch_size"], shuffle=True,
                                         generator=loader_generator)
    total = contrast = tail_total = 0.
    for step, (xb, lb, w) in enumerate(loader):
        xb, lb, w = xb.to(device), lb.to(device), w.to(device)
        model.train(); optimizer.zero_grad()
        log_signal = signal_log_prob(model, xb)
        loss = (-(w*log_signal).mean()/mean_weight if guided else
                -mixture_log_density(log_signal, lb, logit).mean())
        nc = torch.zeros((), device=device)
        if additions["contrastive_fit"]:
            # Sample references from the same denominator while keeping mass as context only.

            model.eval()
            if background_model is not None:
                from .background_correction import sample as sample_background, log_prob as corrected_log_prob
                ref_latent = sample_background(background_model, xb[:, -1], xb.shape[1]-1,
                                               int(seed)+600000+epoch*10000+step, device)
                ref = torch.cat((ref_latent, xb[:, -1:]), dim=1)
                with torch.no_grad():
                    ref_logb = corrected_log_prob(background_model, ref_latent, xb[:, -1])
            else:
                gen = torch.Generator().manual_seed(int(seed)+600000+epoch*10000+step)
                if conditional:
                    ref_latent = torch.randn(xb[:, :-1].shape, generator=gen).to(device)
                    ref = torch.cat((ref_latent, xb[:, -1:]), dim=1)
                    ref_logb = standard_normal_log_prob(ref_latent)
                else:
                    ref = torch.randn(xb.shape, generator=gen).to(device)
                    ref_logb = standard_normal_log_prob(ref)
            both = signal_log_prob(model, torch.cat((xb, ref)))
            nc = contrastive_loss(both[:len(xb)]-lb, both[len(xb):]-ref_logb, w, mean_weight)
            loss = loss + additions["contrastive_strength"]*nc
            model.train()
        tail = torch.zeros((), device=device)
        if additions.get("tail_rank", False) and not warm:
            multiplier = int(additions.get("tail_candidate_multiplier", 4))
            candidate_count = max(len(xb), multiplier * len(xb))
            if conditional:
                candidate_context = xb[:, -1].repeat_interleave(multiplier)[:candidate_count]
                if len(candidate_context) < candidate_count:
                    candidate_context = candidate_context.repeat(
                        int(np.ceil(candidate_count / max(1, len(candidate_context))))
                    )[:candidate_count]
            else:
                candidate_context = None
            model.eval()
            if background_model is not None:
                from .background_correction import sample as sample_background, log_prob as corrected_log_prob
                candidate_latent = sample_background(
                    background_model, candidate_context, xb.shape[1]-1,
                    int(seed)+700000+epoch*10000+step, device
                )
                candidate = torch.cat((candidate_latent, candidate_context[:, None]), dim=1)
                with torch.no_grad():
                    candidate_logb = corrected_log_prob(background_model, candidate_latent, candidate_context)
                    candidate_ratio = signal_log_prob(model, candidate) - candidate_logb
            else:
                gen = torch.Generator().manual_seed(int(seed)+700000+epoch*10000+step)
                if conditional:
                    candidate_latent = torch.randn((candidate_count, xb.shape[1]-1), generator=gen).to(device)
                    candidate = torch.cat((candidate_latent, candidate_context[:, None]), dim=1)
                    candidate_logb = standard_normal_log_prob(candidate_latent)
                else:
                    candidate = torch.randn((candidate_count, xb.shape[1]), generator=gen).to(device)
                    candidate_logb = standard_normal_log_prob(candidate)
                with torch.no_grad():
                    candidate_ratio = signal_log_prob(model, candidate) - candidate_logb
            hard_count = max(1, int(math.ceil(candidate_count * float(additions.get("tail_hard_fraction", .10)))))
            hard_indices = torch.argsort(candidate_ratio, descending=True, stable=True)[:hard_count]
            hard_candidate = candidate[hard_indices]
            hard_logb = candidate_logb[hard_indices]
            model.train()
            hard_ratio = signal_log_prob(model, hard_candidate) - hard_logb
            positive_ratio = log_signal - lb
            tail = tail_ranking_loss(
                positive_ratio, hard_ratio, w, mean_weight,
                margin=float(additions.get("tail_margin", 0.0)),
                temperature=float(additions.get("tail_temperature", 1.0)),
            )
            loss = loss + float(additions.get("tail_strength", .20)) * tail
        require_finite(loss, "Residual objective")
        loss.backward()
        if logit.requires_grad:
            if logit.grad is None:
                raise FloatingPointError("Missing mixture-fraction gradient; optimizer not updated")
            require_finite(logit.grad, "Mixture-fraction gradient")
        clip_gradients(model.parameters(), options["gradient_clip_norm"])
        optimizer.step(); total += float(loss.detach())*len(xb); contrast += float(nc.detach())*len(xb); tail_total += float(tail.detach())*len(xb)
        if progress:
            progress(step+1, len(loader))
    model.eval()
    signal_values = chunks(lambda x: signal_log_prob(model, x), z, device=device)
    if mass_fraction_active:
        gate_logit = torch.from_numpy(
            mass_fraction_logits(mass_fraction_state, ztrain[:, -1], mass_fraction_settings)
        ).to(dtype=torch.float64)
    else:
        gate_logit = logit.detach().cpu()
    nll = -float(mixture_log_density(signal_values, logb, gate_logit).double().mean())
    diagnostics = dict(train_objective=total/len(z), contrastive_loss=contrast/len(z),
                       tail_rank_loss=tail_total/len(z),
                       responsibility_temperature=float(additions.get("responsibility_temperature", 1.0)),
                       phase="guided_initialization" if warm else "alternating_mass_fraction" if mass_fraction_active else "alternating_mixture" if guided else "mixture_likelihood",
                       effective_residual_events=float(weights.sum().square()/weights.square().sum()))
    if mass_fraction_active:
        from .mass_fraction import state_summary
        summary = state_summary(mass_fraction_state, ztrain[:, -1], mass_fraction_settings)
        diagnostics.update(mass_fraction_mean=summary["mean"], mass_fraction_min=summary["minimum"],
                           mass_fraction_max=summary["maximum"], mass_fraction_roughness=summary["roughness"],
                           mass_fraction_updates=summary["updates"])
    return nll, diagnostics, mass_fraction_state


def _cpu_residual_payload(model, logit, epoch, mass_fraction_state=None):
    return {
        "model": {
            key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
            for key, value in model.state_dict().items()
        },
        "fraction_logit": logit.detach().cpu().clone(),
        "epoch": int(epoch),
        "scientific_version": SCIENTIFIC_VERSION,
        "mass_fraction_state": deepcopy(mass_fraction_state),
    }


def _offer_candidate(candidates, count, epoch, validation_nll, model, logit, mass_fraction_state=None):
    key = (float(validation_nll), int(epoch))
    if len(candidates) >= count:
        worst = max((c["validation_nll"], c["epoch"]) for c in candidates)
        if key >= worst:
            return
    candidates.append({
        "epoch": int(epoch),
        "validation_nll": float(validation_nll),
        "filename": f"residual_epoch_{epoch}.pt",
        "sha256": None,
        "payload": _cpu_residual_payload(model, logit, epoch, mass_fraction_state),
    })
    candidates.sort(key=lambda c: (c["validation_nll"], c["epoch"]))
    if len(candidates) > count:
        candidates.pop()


def _persist_candidates(output, candidates):
    for candidate in candidates:
        if candidate.get("sha256") is None:
            candidate["sha256"] = atomic_torch_save(
                Path(output) / candidate["filename"], candidate.pop("payload")
            )
    return {candidate["filename"]: candidate["sha256"] for candidate in candidates}


def _candidate_metadata(candidates):
    return [
        {k: candidate[k] for k in ("epoch", "validation_nll", "filename", "sha256")}
        for candidate in candidates
    ]


def validate_checkpoint(checkpoint, epochs, warmup):
    if checkpoint.get("scientific_version") != SCIENTIFIC_VERSION:
        raise ValueError("Residual checkpoint uses a legacy scientific protocol; retrain in a new output")
    epoch, history = checkpoint.get("epoch"), checkpoint.get("history")
    if (
        type(epoch) is not int
        or not 0 <= epoch < epochs
        or not isinstance(history, list)
        or len(history) != epoch + 1
    ):
        raise ValueError("Invalid residual recovery checkpoint epoch/history")
    for i, row in enumerate(history):
        try:
            valid = (
                type(row["epoch"]) is int
                and row["epoch"] == i
                and np.isfinite([row["train_nll"], row["validation_nll"], row["signal_fraction"]]).all()
                and 0 <= row["signal_fraction"] <= 1
            )
        except (KeyError, TypeError, ValueError):
            valid = False
        if not valid:
            raise ValueError("Invalid residual recovery checkpoint loss history")
    if checkpoint.get("selection_warmup") != warmup:
        raise ValueError("Residual checkpoint selection warm-up changed")
    candidates = checkpoint.get("candidates")
    files = checkpoint.get("files")
    if not isinstance(candidates, list) or not isinstance(files, dict):
        raise ValueError("Residual recovery lacks validation-selected candidate inventory")
    count = checkpoint.get("settings", {}).get("training", {}).get("selected_checkpoints")
    if type(count) is not int or count < 1:
        raise ValueError("Residual recovery has invalid checkpoint-selection count")
    expected_epochs = [
        i + warmup for i in sorted(
            range(max(0, len(history) - warmup)),
            key=lambda i: (history[i + warmup]["validation_nll"], i),
        )[:count]
    ]
    if [c.get("epoch") for c in candidates] != expected_epochs:
        raise ValueError("Residual recovery Top-N candidates do not match saved validation history")
    expected_files = {}
    for candidate in candidates:
        e = candidate.get("epoch")
        name = candidate.get("filename")
        sha = candidate.get("sha256")
        if (name != f"residual_epoch_{e}.pt" or candidate.get("validation_nll") != history[e]["validation_nll"]
                or not isinstance(sha, str) or len(sha) != 64):
            raise ValueError("Invalid residual recovery candidate metadata")
        expected_files[name] = sha
    if files != expected_files:
        raise ValueError("Residual recovery candidate file inventory is inconsistent")


def train_residual(
    train,
    validation,
    output,
    *,
    epochs,
    seed,
    device,
    checkpoint=None,
    after_epoch=None,
    fraction=None,
    initialization="background",
    progress_label="Train residual mixture",
    on_training_start=None,
    settings=None,
    background_correction=None,
):
    settings = deepcopy(DEFAULTS["riddle"] if settings is None else settings)
    settings.update(epochs=epochs, initialization=initialization)
    settings = validate_residual(settings)
    options = settings["training"]
    additions = feature_options(settings)
    active = effective_features(settings)
    enhanced = active["guided_fit"] or active["contrastive_fit"]
    if (checkpoint is not None and enhanced
            and checkpoint.get("enhanced_training_protocol") != ENHANCED_TRAINING_PROTOCOL):
        raise ValueError("Enhanced training protocol changed; keep the original runtime or start a new output")
    control = settings.get("optimization", {})
    warmup = (additions["guide_warmup_epochs"] if active["guided_fit"] else
              control.get("fraction_warmup_epochs", 0) if fraction is None else 0)
    if checkpoint is not None:
        validate_checkpoint(checkpoint, epochs, warmup)
        if checkpoint.get("settings") != settings:
            raise ValueError("Residual training settings changed; use a new output")
        expected_background = None if background_correction is None else background_correction["model_sha256"]
        if checkpoint.get("background_correction_sha256") != expected_background:
            raise ValueError("Residual background correction changed; use a new output")
        inputs = Path(output) / "residual_training_inputs.json"
        if (
            inputs.is_file()
            and json.loads(inputs.read_text()).get("initialization", "random") != initialization
        ):
            raise ValueError("Residual initialization changed; use a new output or the saved initialization")
    start = checkpoint["epoch"] + 1 if checkpoint is not None else 0
    progress = ProgressStage(
        "residual_training", progress_label, epochs, "epoch", initial=start, report_every=1
    )
    mass_conditioning = settings.get("mass_conditioning", False)
    physical_inputs = settings.get("input_space") == "physical"
    ztrain, zval = (real_sr_latents(rows, mass_conditioning=mass_conditioning,
                                  physical_inputs=physical_inputs) for rows in (train, validation))
    if len(ztrain) % options["batch_size"] == 1 and settings["flow"]["use_batch_norm"]:
        raise ValueError("BatchNorm cannot train a singleton final batch; no rows were dropped")
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(ztrain)),
        batch_size=options["batch_size"],
        shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(zval)), batch_size=options["validation_batch_size"]
    )
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    model, logit = (
        build_signal_flow(device, features=ztrain.shape[1], settings=settings),
        initial_fraction_logit(seed, device),
    )
    corrected_background = None
    corrected_background_digest = None
    if background_correction is not None:
        from .background_correction import load as load_background_correction
        corrected_background = load_background_correction(
            background_correction, settings, ztrain.shape[1]-1, device
        )
        model.load_state_dict(corrected_background.state_dict())
    if corrected_background is None:
        if initialization in ("background", "identity"):
            from .model import match_background
            match_background(model)
        elif initialization != "random":
            raise ValueError("Unknown residual initialization")
    elif initialization not in ("background", "identity"):
        raise ValueError("bgcorr_40_reguide fixes residual initialization to q_phi; use background initialization")
    if fraction is not None:
        if not np.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError("Fixed signal fraction must lie in (0,1)")
        logit = torch.tensor(np.log(fraction / (1 - fraction)), dtype=torch.float64, device=device)
    elif control.get("initial_fraction") is not None:
        initial = control["initial_fraction"]
        with torch.no_grad():
            logit.fill_(np.log(initial / (1 - initial)))
    initial_fraction = float(logit.sigmoid().detach())
    if active["guided_fit"]:
        logit.requires_grad_(False)
    from .mass_fraction import is_enabled as mass_fraction_enabled, initial_state as initial_mass_fraction_state
    mass_fraction_active = mass_fraction_enabled(settings, fraction)
    mass_fraction_state = (initial_mass_fraction_state(initial_fraction, settings)
                           if mass_fraction_active else None)
    optimizer = residual_optimizer(model, logit, options)
    scheduler = None
    if control.get("lr_factor", 1) < 1:

        minimum = [control["min_lr"]] + [options["learning_rate"]] * (len(optimizer.param_groups) - 1)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=control["lr_factor"], patience=control["lr_patience"],
            threshold=control["min_delta"], threshold_mode="abs", min_lr=minimum)
    history, files, candidates, start = [], {}, [], 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        with torch.no_grad():
            logit.copy_(checkpoint["fraction_logit"].to(device))
        saved_mass_fraction = checkpoint.get("mass_fraction_state")
        if mass_fraction_active and saved_mass_fraction is None:
            raise ValueError("v5.3 recovery checkpoint is missing its smooth f(m) state")
        if not mass_fraction_active and saved_mass_fraction is not None:
            raise ValueError("Recovery checkpoint has an unexpected smooth f(m) state")
        mass_fraction_state = deepcopy(saved_mass_fraction)
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        history, files, start = checkpoint["history"], checkpoint["files"], checkpoint["epoch"] + 1
        candidates = [dict(candidate) for candidate in checkpoint["candidates"]]
        restore_rng(checkpoint["rng"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if corrected_background is not None:
        from .background_correction import save_local
        corrected_background_digest = save_local(output, corrected_background, background_correction)
    guide = None
    if active["guided_fit"]:
        before = rng_state()
        guide_inputs = ztrain[:, :-1] if mass_conditioning else ztrain
        if corrected_background is not None:
            from .enhancements import corrected_teacher_weights
            guide = corrected_teacher_weights(
                output / "guide", guide_inputs, ztrain[:, -1], initial_fraction,
                seed=(int(seed)+93000) % 2**32, options={**additions, **active},
                batch_size=options["batch_size"], background_model=corrected_background,
                background_sha256=background_correction["model_sha256"], device=device)
        else:
            from .enhancements import teacher_weights
            guide = teacher_weights(output / "guide", guide_inputs, initial_fraction,
                                    seed=(int(seed)+2000) % 2**32, options={**additions, **active},
                                    batch_size=options["batch_size"],
                                    context=(ztrain[:, -1] if mass_conditioning else None), device=device)
        restore_rng(before)
    write_json(
        output / "residual_training_inputs.json",
        {
            "scientific_version": SCIENTIFIC_VERSION,
            "train_events": len(ztrain),
            "validation_events": len(zval),
            "train_latents_sha256": digest(ztrain),
            "validation_latents_sha256": digest(zval),
            "data_reference_label": 1,
            "truth_labels_used": False,
            "mass_input_used": mass_conditioning,
            "initial_fraction": initial_fraction,
            "fraction_mode": ("learned_smooth_f(m)" if mass_fraction_active else
                              "learned_global" if fraction is None else "fixed_global"),
            "initialization": initialization,
            "settings": settings,
            "effective_features": active,
            "fraction_estimator": ("five-control-point smooth logistic f(m) from latent responsibilities; excluded from final score"
                                   if mass_fraction_active else
                                   "guided responsibilities" if active["guided_fit"] else "gradient likelihood"),
            "mass_fraction": (None if not mass_fraction_active else {
                "enabled": True, "settings": settings["mass_fraction"],
                "training_evidence": "latent residual responsibilities only; no mjj count/bump density",
                "final_score_uses_fraction": False,
            }),
            "enhanced_training_protocol": ENHANCED_TRAINING_PROTOCOL if enhanced else None,
            "features": ztrain.shape[1],
            "background_correction": (None if corrected_background is None else {
                "mode": background_correction["mode"],
                "protocol": background_correction["protocol"],
                "source_sha256": background_correction["model_sha256"],
                "local_sha256": corrected_background_digest,
                "selected_epoch": background_correction["selected_epoch"],
                "epochs": background_correction["epochs"],
            }),
        },
    )
    if on_training_start is not None:
        on_training_start()
    best, stale = float("inf"), 0
    for row in history[warmup:]:
        if row["validation_nll"] < best - control.get("min_delta", 0):
            best, stale = row["validation_nll"], 0
        else:
            stale += 1
    stopped = bool(checkpoint and checkpoint.get("stopped_early", False))
    durable_history = list(history)
    if stopped:
        start = epochs
    try:
        for epoch in range(start, epochs):
            if fraction is None and not active["guided_fit"]:
                # Freeze fraction gradients during warm-up.

                logit.requires_grad_(epoch >= warmup)
            losses = {}
            flow_lr = optimizer.param_groups[0]["lr"]
            diagnostics = {}
            for part, loader, opt in (("train", train_loader, optimizer), ("validation", val_loader, None)):
                progress.substep(part.title(), 0, len(loader))
                if part == "train" and enhanced:
                    losses["train_nll"], diagnostics, mass_fraction_state = enhanced_epoch(
                        model, logit, ztrain, optimizer, options, additions, epoch=epoch, seed=seed, guide=guide,
                        fixed_fraction=fraction is not None, background_model=corrected_background,
                        mass_fraction_state=mass_fraction_state, mass_fraction_settings=settings,
                        progress=lambda done, total: progress.substep("Train", done, total))
                    continue
                if part == "validation" and enhanced:
                    # Preserve training RNG state while computing validation losses.


                    from .enhancements import chunks, standard_normal_log_prob
                    from .integrity import mixture_log_density
                    model.eval()
                    valid = torch.from_numpy(zval)
                    valid_latent = valid[:, :-1] if mass_conditioning else valid
                    if corrected_background is not None:
                        from .background_correction import log_prob as corrected_log_prob
                        log_background = chunks(
                            lambda x, c: corrected_log_prob(corrected_background, x, c),
                            valid_latent, valid[:, -1], device=device)
                    else:
                        log_background = standard_normal_log_prob(valid_latent)
                    if mass_fraction_active:
                        from .mass_fraction import logits as mass_fraction_logits
                        validation_logit = torch.from_numpy(
                            mass_fraction_logits(mass_fraction_state, zval[:, -1], settings)
                        ).to(dtype=torch.float64)
                    else:
                        validation_logit = logit.detach().cpu()
                    losses["validation_nll"] = -float(mixture_log_density(
                        chunks(lambda x: signal_log_prob(model, x), valid, device=device), log_background,
                        validation_logit).double().mean())
                    continue
                losses[part + "_nll"] = train_epoch(
                    model,
                    logit,
                    loader,
                    opt,
                    lambda done, total: progress.substep(part.title(), done, total),
                    gradient_clip_norm=options["gradient_clip_norm"],
                )
            if mass_fraction_active:
                from .mass_fraction import state_summary
                fitted_fraction = float(state_summary(mass_fraction_state, ztrain[:, -1], settings)["mean"])
                with torch.no_grad():
                    logit.fill_(np.log(fitted_fraction/(1-fitted_fraction)))
            else:
                fitted_fraction = float(logit.sigmoid().detach())
            if not np.isfinite([*losses.values(), fitted_fraction]).all():
                raise FloatingPointError("Nonfinite residual training state")
            row = {"epoch": epoch, **losses, "signal_fraction": fitted_fraction, **diagnostics}
            if control:
                row.update(flow_learning_rate=flow_lr, fraction_warmup=epoch < warmup)
            history.append(row)
            if epoch >= warmup:
                if scheduler is not None:
                    scheduler.step(losses["validation_nll"])
                if losses["validation_nll"] < best - control.get("min_delta", 0):
                    best, stale = losses["validation_nll"], 0
                else:
                    stale += 1
            stopped = bool(control.get("early_stopping_patience", 0)
                           and epoch + 1 >= control["minimum_epochs"]
                           and stale >= control["early_stopping_patience"])
            if epoch >= warmup:
                _offer_candidate(
                    candidates, options["selected_checkpoints"], epoch, losses["validation_nll"], model, logit,
                    mass_fraction_state
                )
            durable = persist_boundary(epoch, epochs) or stopped
            if durable:
                progress.update(force=True, operation="Persist 10-epoch recovery boundary", minibatch="-")
                files = _persist_candidates(output, candidates)
                state = {
                    "scientific_version": SCIENTIFIC_VERSION,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "fraction_logit": logit.detach(),
                    "mass_fraction_state": deepcopy(mass_fraction_state),
                    "optimizer": optimizer.state_dict(),
                    "rng": rng_state(),
                    "history": history,
                    "files": files,
                    "candidates": _candidate_metadata(candidates),
                    "selection_warmup": warmup,
                    "settings": settings,
                    "enhanced_training_protocol": ENHANCED_TRAINING_PROTOCOL if enhanced else None,
                    "background_correction_sha256": (None if background_correction is None else background_correction["model_sha256"]),
                }
                if control:
                    state.update(stopped_early=stopped, scheduler=scheduler.state_dict() if scheduler else None)
                atomic_torch_save(output / ".resume/latest.pt", state)
                write_json(output / "residual_losses.json", {"history": history})
                durable_history = list(history)
            progress.update(epoch + 1, force=True, operation="Early stop" if stopped else "Epoch complete",
                            **losses, fraction=fitted_fraction)
            if after_epoch is not None:
                after_epoch(epoch)
            if stopped:
                break
    except BaseException:
        # Publish history only through the last durable recovery point.

        write_json(output / "residual_losses.json", {"history": durable_history})
        raise
    write_json(output / "residual_losses.json", {"history": history})
    order = [i + warmup for i in ordered_epochs(
        [r["validation_nll"] for r in history[warmup:]], options["selected_checkpoints"])]
    if [candidate["epoch"] for candidate in candidates] != order:
        raise ValueError("Buffered residual Top-N checkpoint inventory disagrees with final selection")
    if set(files) != {f"residual_epoch_{epoch}.pt" for epoch in order}:
        raise ValueError("Final residual checkpoint files disagree with validation selection")
    weighting = additions.get("checkpoint_weighting", "uniform")
    selected_nll = np.asarray([history[i]["validation_nll"] for i in order], dtype=np.float64)
    if weighting == "validation_likelihood":
        logw = -len(zval) * (selected_nll - float(selected_nll.min()))
        logw = np.maximum(logw, -60.0)
        checkpoint_weights = np.exp(logw - np.logaddexp.reduce(logw))
    else:
        checkpoint_weights = np.full(len(order), 1.0 / len(order), dtype=np.float64)
    write_json(
        output / "residual_selection.json",
        {
            "epochs": order,
            "criterion": f"{options['selected_checkpoints']} lowest validation mixture NLL epochs",
            "density_ensemble": ("validation-likelihood weighted mean" if weighting == "validation_likelihood" else "arithmetic mean"),
            "checkpoint_weighting": weighting,
            "checkpoint_weights": [float(x) for x in checkpoint_weights],
            "validation_nll": [float(x) for x in selected_nll],
            "signal_fractions": [history[i]["signal_fraction"] for i in order],
            "mass_fraction": ({
                "enabled": True,
                "model": "five-control-point natural-cubic logistic spline",
                "training_evidence": "latent residual responsibilities only",
                "final_score_uses_fraction": False,
                "selected_states": [
                    torch.load(output / f"residual_epoch_{epoch}.pt", map_location="cpu", weights_only=True).get("mass_fraction_state")
                    for epoch in order
                ],
            } if mass_fraction_active else {"enabled": False}),
            **(dict(trained_epochs=len(history), maximum_epochs=epochs, stopped_early=stopped,
                    excluded_warmup_epochs=warmup) if control or enhanced else {}),
        },
    )
    if mass_fraction_active:
        from .mass_fraction import probabilities as mass_fraction_probabilities
        selected = json.loads((output / "residual_selection.json").read_text())["mass_fraction"]["selected_states"]
        context_grid = np.linspace(-1.0, 1.0, 101, dtype=np.float64)
        curves = np.stack([mass_fraction_probabilities(state, context_grid, settings) for state in selected])
        ensemble_curve = np.average(curves, axis=0, weights=checkpoint_weights)
        write_json(output / "mass_fraction_curve.json", {
            "schema": 1,
            "model": "smooth training-only f(mjj)",
            "context_definition": "(mjj - 3.5 TeV) / 0.2 TeV",
            "context": context_grid.tolist(),
            "mjj_tev": (3.5 + 0.2*context_grid).tolist(),
            "ensemble_fraction": ensemble_curve.tolist(),
            "checkpoint_epochs": order,
            "checkpoint_weights": [float(x) for x in checkpoint_weights],
            "checkpoint_fractions": curves.tolist(),
            "truth_labels_used": False,
            "mjj_count_density_used": False,
            "included_in_final_score": False,
        })
    return order, history


def residual_scores(output, order, z, device, *, normalization_checks=None, normalization_tests=1):
    if not len(order):
        raise ValueError("Residual scoring requires at least one checkpoint")
    inputs = json.loads((output / "residual_training_inputs.json").read_text())
    if inputs.get("scientific_version") != SCIENTIFIC_VERSION:
        raise ValueError("Legacy residual inputs; use saved legacy scores or retrain")
    if inputs["features"] != z.shape[1]:
        raise ValueError("Scoring feature count differs from training")
    model = build_signal_flow(device, features=z.shape[1], settings=inputs["settings"]).eval()
    selection_path = output / "residual_selection.json"
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        saved_epochs = list(selection.get("epochs", []))
        saved_weights = np.asarray(selection.get("checkpoint_weights", []), dtype=np.float64)
        if saved_weights.shape == (len(saved_epochs),) and all(epoch in saved_epochs for epoch in order):
            checkpoint_weights = np.asarray([saved_weights[saved_epochs.index(epoch)] for epoch in order], dtype=np.float64)
            checkpoint_weights = checkpoint_weights / checkpoint_weights.sum()
        else:
            checkpoint_weights = np.array([], dtype=np.float64)
    else:
        checkpoint_weights = np.array([], dtype=np.float64)
    if checkpoint_weights.shape != (len(order),):
        checkpoint_weights = np.full(len(order), 1.0 / len(order), dtype=np.float64)
    if (not np.isfinite(checkpoint_weights).all() or np.any(checkpoint_weights <= 0)
            or not np.isclose(checkpoint_weights.sum(), 1.0, rtol=1e-8, atol=1e-10)):
        raise ValueError("Invalid residual checkpoint ensemble weights")
    corrected_background = None
    correction_info = inputs.get("background_correction")
    if correction_info is not None:
        from .background_correction import load_local
        loaded = load_local(output, inputs["settings"], z.shape[1]-1, device)
        if loaded is None:
            raise ValueError("Missing fit-local corrected background")
        corrected_background, saved_correction = loaded
        if file_digest(Path(output) / "background_correction.pt") != correction_info["local_sha256"]:
            raise ValueError("Fit-local corrected background artifact changed")
        if saved_correction.get("source_sha256") != correction_info["source_sha256"]:
            raise ValueError("Fit-local corrected background source changed")
    from .roles import DEFAULT_POLICY, REPLAY_SCORING_POLICIES
    policy = inputs['settings'].get('data_policy', DEFAULT_POLICY)
    replay = policy in REPLAY_SCORING_POLICIES
    if corrected_background is not None:
        if not model.mass_conditioning:
            raise ValueError("Corrected-background scoring requires mass-conditioned inputs")
        from .background_correction import log_prob as corrected_log_prob
        from .enhancements import chunks as infer_chunks
        values = torch.from_numpy(z)
        log_background = infer_chunks(
            lambda x, c: corrected_log_prob(corrected_background, x, c),
            values[:, :-1], values[:, -1], device=device, size=8192).numpy()
    elif replay:




        from .enhancements import standard_normal_log_prob
        log_background = standard_normal_log_prob(torch.from_numpy(z).double()).numpy()
    else:
        log_background = background_log_prob(torch.from_numpy(z), mass_conditioning=model.mass_conditioning,
                                             physical_inputs=model.physical_inputs).numpy()
    log_sum = None
    columns = []
    batch = 4096 if replay else 8192
    with ProgressStage(
        "residual_scoring", "Score residual density ensemble", len(order) * len(z), "prediction"
    ) as progress:
        with torch.no_grad():
            for i, epoch in enumerate(order):
                checkpoint = torch.load(
                    output / f"residual_epoch_{epoch}.pt", map_location=device, weights_only=True
                )
                if checkpoint.get("scientific_version") != SCIENTIFIC_VERSION:
                    raise ValueError("Cannot mix scientific checkpoint versions")
                if checkpoint.get("epoch") != epoch:
                    raise ValueError("Residual checkpoint epoch differs from requested selection")
                model.load_state_dict(checkpoint["model"])
                chunks = []
                for offset in range(0, len(z), batch):
                    x = torch.as_tensor(z[offset : offset + batch], device=device)
                    chunks.append(signal_log_prob(model, x).cpu().numpy())
                    progress.update(i * len(z) + min(offset + batch, len(z)))
                log_density = np.concatenate(chunks).astype(np.float64)
                if not np.isfinite(log_density).all():
                    raise FloatingPointError("Nonfinite residual checkpoint density")
                if normalization_checks is not None:
                    from .production import validate_density_ratio

                    normalization_checks.append(dict(epoch=epoch, **validate_density_ratio(
                        log_density - log_background.astype(np.float64),
                        stage=f"RIDDLE {output}, checkpoint {epoch}", tests=normalization_tests)))
                weighted = log_density + math.log(float(checkpoint_weights[i]))
                log_sum = weighted if log_sum is None else np.logaddexp(log_sum, weighted)
                if replay: columns.append(log_density)
    log_signal = log_sum
    if replay:
        from scipy.special import logsumexp
        log_signal = logsumexp(np.column_stack(columns) + np.log(checkpoint_weights)[None, :], axis=1)
    scores = log_signal.astype(np.float64) - log_background.astype(np.float64)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite residual ensemble scores")
    return scores


def residual_fraction_probabilities(output, order, z, *, return_checkpoints=False):
    """Evaluate the training-only mixture gate for saved residual checkpoints.

    This helper is used for validation/configuration likelihood accounting only.
    The exported RIDDLE anomaly score never multiplies by f(m).
    """
    output = Path(output)
    z = np.asarray(z, dtype=np.float32)
    if z.ndim != 2 or not len(z) or not np.isfinite(z).all():
        raise ValueError("Mass-fraction evaluation requires finite residual inputs")
    inputs = json.loads((output / "residual_training_inputs.json").read_text())
    settings = inputs["settings"]
    selection_path = output / "residual_selection.json"
    checkpoint_weights = np.full(len(order), 1.0 / len(order), dtype=np.float64)
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        saved_epochs = list(selection.get("epochs", []))
        saved_weights = np.asarray(selection.get("checkpoint_weights", []), dtype=np.float64)
        if saved_weights.shape == (len(saved_epochs),) and all(epoch in saved_epochs for epoch in order):
            checkpoint_weights = np.asarray([saved_weights[saved_epochs.index(epoch)] for epoch in order], dtype=np.float64)
            checkpoint_weights /= checkpoint_weights.sum()
    curves = []
    for epoch in order:
        checkpoint = torch.load(output / f"residual_epoch_{epoch}.pt", map_location="cpu", weights_only=True)
        if checkpoint.get("scientific_version") != SCIENTIFIC_VERSION or checkpoint.get("epoch") != epoch:
            raise ValueError("Mass-fraction checkpoint identity differs from selection")
        state = checkpoint.get("mass_fraction_state")
        if state is not None:
            if not settings.get("mass_conditioning", False):
                raise ValueError("Saved f(m) gate requires mass-conditioned inputs")
            from .mass_fraction import probabilities
            curve = probabilities(state, z[:, -1], settings)
        else:
            scalar = float(torch.sigmoid(checkpoint["fraction_logit"].double()).item())
            curve = np.full(len(z), scalar, dtype=np.float64)
        curves.append(np.asarray(curve, dtype=np.float64))
    curves = np.stack(curves)
    if return_checkpoints:
        return curves, checkpoint_weights
    result = np.average(curves, axis=0, weights=checkpoint_weights)
    if not np.isfinite(result).all() or np.any(result <= 0) or np.any(result >= 1):
        raise FloatingPointError("Invalid saved residual mixture fraction")
    return result



def residual_background_log_prob(output, z, device):
    """Return the exact denominator log density recorded for a residual fit."""
    output = Path(output)
    inputs = json.loads((output / "residual_training_inputs.json").read_text())
    z = np.asarray(z, dtype=np.float32)
    if inputs["features"] != z.shape[1]:
        raise ValueError("Background scoring feature count differs from training")
    info = inputs.get("background_correction")
    if info is not None:
        from .background_correction import load_local, log_prob as corrected_log_prob
        from .enhancements import chunks
        loaded = load_local(output, inputs["settings"], z.shape[1]-1, device)
        if loaded is None:
            raise ValueError("Missing fit-local corrected background")
        q, saved = loaded
        if saved.get("source_sha256") != info["source_sha256"]:
            raise ValueError("Corrected-background source identity changed")
        t = torch.from_numpy(z)
        return chunks(lambda x, c: corrected_log_prob(q, x, c), t[:, :-1], t[:, -1],
                      device=device, size=8192).numpy().astype(np.float64)
    model = build_signal_flow(device, features=z.shape[1], settings=inputs["settings"])
    return background_log_prob(torch.from_numpy(z), mass_conditioning=model.mass_conditioning,
                               physical_inputs=model.physical_inputs).numpy().astype(np.float64)


def residual_background_sample(output, contexts, count, seed, device):
    """Draw full residual inputs from the recorded denominator at supplied contexts."""
    output = Path(output)
    inputs = json.loads((output / "residual_training_inputs.json").read_text())
    info = inputs.get("background_correction")
    contexts = np.asarray(contexts, dtype=np.float32)
    if contexts.shape != (count,):
        raise ValueError("Background sample contexts are misaligned")
    features = inputs["features"]
    if info is not None:
        from .background_correction import load_local, sample as sample_background
        loaded = load_local(output, inputs["settings"], features-1, device)
        if loaded is None:
            raise ValueError("Missing fit-local corrected background")
        q, saved = loaded
        if saved.get("source_sha256") != info["source_sha256"]:
            raise ValueError("Corrected-background source identity changed")
        latent = sample_background(q, torch.from_numpy(contexts), features-1, seed, device).detach().cpu().numpy()
        return np.column_stack((latent, contexts)).astype(np.float32)
    rng = np.random.default_rng(seed)
    latent = rng.standard_normal((count, features-1)).astype(np.float32)
    return np.column_stack((latent, contexts)).astype(np.float32)
