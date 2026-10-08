import json
import math
import random
import time
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from torch import nn

from .integrity import SCIENTIFIC_VERSION, ordered_epochs, require_finite
from .model import real_sr_latents
from .options import feature_options
from .settings import validate_residual
from .stein_scoring import SCORING_PROTOCOL
from .storage import atomic_torch_save, digest, persist_boundary, restore_rng, rng_state, write_json
from .worker_progress import ProgressStage

PROTOCOL = "stein_witness_v2_exact_laplacian_mass_blind_qscore_cached_tail_rank_closure_gated"


class SteinPotential(nn.Module):
    def __init__(self, latent_features, settings):
        super().__init__()
        cfg = settings["stein"]
        self.latent_features = int(latent_features)
        self.mass_conditioning = bool(settings.get("mass_conditioning", False))
        inputs = self.latent_features
        layers = []
        width = int(cfg["hidden_features"])
        for index in range(int(cfg["hidden_layers"])):
            layers.append(nn.Linear(inputs if index == 0 else width, width))
            layers.append(nn.SiLU())
        layers.append(nn.Linear(width, 1))
        self.network = nn.Sequential(*layers)

    def forward(self, latent, context=None):
        if latent.ndim != 2 or latent.shape[1] != self.latent_features:
            raise ValueError("Invalid Stein latent tensor")
        if context is not None and context.shape != (len(latent),):
            raise ValueError("Stein mass context is misaligned")
        return self.network(latent).squeeze(-1)


def build_potential(device, features, settings, initialization="background"):
    if settings.get("input_space") == "physical":
        raise ValueError("stein_witness requires mapped latent inputs")
    latent_features = int(features) - int(settings.get("mass_conditioning", False))
    if latent_features not in (4, 5):
        raise ValueError("Stein witness accepts four configured latents, or five for DeltaR")
    model = SteinPotential(latent_features, settings).to(device)
    if initialization == "background":
        last = model.network[-1]
        with torch.no_grad():
            last.weight.zero_()
            last.bias.zero_()
    elif initialization != "random":
        raise ValueError("Unknown Stein witness initialization")
    return model


def _split_inputs(inputs, mass_conditioning):
    if mass_conditioning:
        return inputs[:, :-1], inputs[:, -1]
    return inputs, None


def _background_score_array(inputs, background_model, device, *, mass_conditioning, batch_size=8192,
                            runtime_metadata=None):
    inputs = np.asarray(inputs, dtype=np.float32)
    latent = inputs[:, :-1] if mass_conditioning else inputs
    if background_model is None:
        if runtime_metadata is not None:
            runtime_metadata["qscore_batch_size_requested"] = int(batch_size)
            runtime_metadata["qscore_batch_size_used"] = 0
            runtime_metadata["qscore_implementation"] = "analytic_standard_normal"
        return np.ascontiguousarray(-latent, dtype=np.float32)
    if not mass_conditioning:
        raise ValueError("Corrected Stein background requires mass-conditioned inputs")
    from .background_correction import log_prob as corrected_log_prob

    background_model.eval()
    background_model.requires_grad_(False)
    target = torch.device(device)
    chunk_size = int(batch_size)
    if chunk_size <= 0:
        raise ValueError("Stein background score batch size must be positive")
    result = []
    offset = 0
    while offset < len(inputs):
        stop = min(offset + chunk_size, len(inputs))
        try:
            chunk = torch.as_tensor(inputs[offset:stop], dtype=torch.float32, device=target)
            z = chunk[:, :-1].detach().requires_grad_(True)
            context = chunk[:, -1]
            logq = corrected_log_prob(background_model, z, context)
            score = torch.autograd.grad(logq.sum(), z, create_graph=False, retain_graph=False)[0]
            require_finite(score, "Stein background score")
            result.append(score.detach().cpu())
            del score, logq, context, z, chunk
            offset = stop
        except torch.cuda.OutOfMemoryError:
            if target.type != "cuda" or chunk_size <= 256:
                raise
            torch.cuda.empty_cache()
            chunk_size = max(256, chunk_size // 2)
    if runtime_metadata is not None:
        runtime_metadata["qscore_batch_size_requested"] = int(batch_size)
        runtime_metadata["qscore_batch_size_used"] = int(chunk_size)
    return torch.cat(result).numpy().astype(np.float32, copy=False)


def _training_contract(settings):
    contract = deepcopy(settings)
    contract.pop("ensemble_fit_selection", None)
    contract.pop("ensemble_fit_count", None)
    contract.pop("ensemble_completion", None)
    contract.pop("fits", None)
    if isinstance(contract.get("stein"), dict):
        contract["stein"].pop("scoring", None)
        contract["stein"].pop("ensemble_fit_selection", None)
    return contract


def _stein_operator_values(model, inputs, background_score, *, training):
    latent, context = _split_inputs(inputs, model.mass_conditioning)
    z = latent.detach().requires_grad_(True)
    potential = model(z, context)
    gradient = torch.autograd.grad(potential.sum(), z, create_graph=True, retain_graph=True)[0]
    laplacian = torch.zeros(len(z), dtype=z.dtype, device=z.device)
    for dimension in range(z.shape[1]):
        component = gradient[:, dimension]
        if component.requires_grad:
            second = torch.autograd.grad(
                component.sum(), z, create_graph=training, retain_graph=True, allow_unused=True
            )[0]
            diagonal = (torch.zeros(len(z), dtype=z.dtype, device=z.device)
                        if second is None else second[:, dimension])
        else:
            diagonal = torch.zeros(len(z), dtype=z.dtype, device=z.device)
        laplacian = laplacian + diagonal
    operator = laplacian + (gradient * background_score).sum(dim=1)
    energy = gradient.square().sum(dim=1)
    require_finite(operator, "Stein operator")
    require_finite(energy, "Stein gradient energy")
    require_finite(potential, "Stein potential")
    return operator, energy, potential


def _stein_terms(model, inputs, background_score, settings, *, training):
    operator, energy, potential = _stein_operator_values(
        model, inputs, background_score, training=training
    )
    cfg = settings["stein"]
    objective = -operator.mean() + 0.5 * float(cfg["witness_regularization"]) * energy.mean()
    center = potential.mean()
    center_strength = float(cfg["potential_center_strength"])
    if center_strength:
        objective = objective + center_strength * center.square()
    require_finite(objective, "Stein witness objective")
    return objective, operator.mean(), energy.mean(), center


def _equal_count_groups(values, bins):
    count = min(int(bins), len(values))
    if count <= 1:
        return torch.zeros(len(values), dtype=torch.long, device=values.device)
    order = torch.argsort(values.detach(), stable=True)
    ranks = torch.arange(len(values), device=values.device)
    groups = torch.empty(len(values), dtype=torch.long, device=values.device)
    groups[order] = torch.div(ranks * count, len(values), rounding_mode="floor")
    return groups.clamp_max(count - 1)


def _hard_indices(scores, fraction, groups=None):
    if not 0 < fraction <= 1:
        raise ValueError("Stein tail fraction must lie in (0,1]")
    if groups is None:
        count = max(1, int(math.ceil(len(scores) * fraction)))
        return torch.argsort(scores, descending=True, stable=True)[:count]
    selected = []
    for group in torch.unique(groups, sorted=True):
        indices = torch.nonzero(groups == group, as_tuple=False).flatten()
        count = max(1, int(math.ceil(len(indices) * fraction)))
        selected.append(indices[torch.argsort(scores[indices], descending=True, stable=True)[:count]])
    return torch.cat(selected)


def _sample_background(background_model, context, dimensions, count, seed, device):
    if background_model is not None:
        from .background_correction import sample as sample_background

        latent = sample_background(background_model, context, dimensions, seed, device)
    else:
        target = torch.device(device)
        generator = torch.Generator(device=target if target.type == "cuda" else "cpu")
        generator.manual_seed(int(seed))
        latent = torch.randn((count, dimensions), generator=generator, device=target)
    return latent


def _tail_loss(model, batch, background_model, additions, settings, *, epoch, step, seed):
    if not additions.get("tail_rank", False) or epoch < int(additions.get("hard_start_epoch", 0)):
        return batch.new_zeros(())
    from .enhancements import tail_ranking_loss

    latent, context = _split_inputs(batch, model.mass_conditioning)
    multiplier = int(additions.get("tail_candidate_multiplier", 8))
    count = max(len(batch), multiplier * len(batch))
    groups = (_equal_count_groups(context, int(additions.get("tail_mass_bins", 8)))
              if context is not None else None)
    if context is not None:
        candidate_context = context.repeat_interleave(multiplier)[:count]
        if len(candidate_context) < count:
            candidate_context = candidate_context.repeat(int(math.ceil(count / len(candidate_context))))[:count]
        candidate_groups = groups.repeat_interleave(multiplier)[:count]
        if len(candidate_groups) < count:
            candidate_groups = candidate_groups.repeat(int(math.ceil(count / len(candidate_groups))))[:count]
    else:
        candidate_context = None
        candidate_groups = None
    candidate_latent = _sample_background(
        background_model, candidate_context, latent.shape[1], count,
        int(seed) + 700000 + epoch * 10000 + step, batch.device,
    )
    with torch.no_grad():
        data_selection_scores = -model(latent, context)
        candidate_selection_scores = -model(candidate_latent, candidate_context)
        positive_indices = _hard_indices(
            data_selection_scores, float(settings["stein"]["data_tail_fraction"]), groups
        )
        negative_indices = _hard_indices(
            candidate_selection_scores, float(additions.get("tail_hard_fraction", 0.01)), candidate_groups
        )
    positive = -model(latent[positive_indices], None if context is None else context[positive_indices])
    negative = -model(candidate_latent[negative_indices],
                      None if candidate_context is None else candidate_context[negative_indices])
    weights = torch.ones_like(positive)
    positive_groups = None if groups is None else groups[positive_indices]
    negative_groups = None if candidate_groups is None else candidate_groups[negative_indices]
    return tail_ranking_loss(
        positive, negative, weights, 1.0,
        margin=float(additions.get("tail_margin", 0.0)),
        temperature=float(additions.get("tail_temperature", 1.0)),
        positive_groups=positive_groups,
        negative_groups=negative_groups,
    )


def _epoch(model, loader, optimizer, settings, additions, background_model, *, epoch, seed, progress=None):
    from .enhancements import clip_gradients

    model.train()
    total = witness = energy = center = tail_total = 0.0
    events = 0
    for step, (inputs, background_score) in enumerate(loader):
        device = next(model.parameters()).device
        inputs = inputs.to(device, non_blocking=True)
        background_score = background_score.to(device, non_blocking=True)
        optimizer.zero_grad(set_to_none=True)
        loss, op, en, ce = _stein_terms(model, inputs, background_score, settings, training=True)
        tail = _tail_loss(model, inputs, background_model, additions, settings, epoch=epoch, step=step, seed=seed)
        loss = loss + float(additions.get("tail_strength", 0.0)) * tail
        require_finite(loss, "Stein training loss")
        loss.backward()
        clip_gradients(model.parameters(), settings["training"]["gradient_clip_norm"])
        optimizer.step()
        n = len(inputs)
        total += float(loss.detach()) * n
        witness += float(op.detach()) * n
        energy += float(en.detach()) * n
        center += float(ce.detach()) * n
        tail_total += float(tail.detach()) * n
        events += n
        if progress is not None:
            progress(step + 1, len(loader))
    if events != len(loader.dataset):
        raise ValueError("Stein epoch did not include every training event")
    return total / events, dict(
        stein_witness_mean=witness / events,
        stein_gradient_energy=energy / events,
        stein_potential_mean=center / events,
        tail_rank_loss=tail_total / events,
        phase="stein_witness",
    )


def _validation_loss(model, loader, settings):
    model.eval()
    total = 0.0
    events = 0
    for inputs, background_score in loader:
        device = next(model.parameters()).device
        inputs = inputs.to(device, non_blocking=True)
        background_score = background_score.to(device, non_blocking=True)
        loss, _, _, _ = _stein_terms(model, inputs, background_score, settings, training=False)
        total += float(loss.detach()) * len(inputs)
        events += len(inputs)
    if events != len(loader.dataset):
        raise ValueError("Stein validation did not include every expected event")
    return total / events


def _closure_summary(values, contexts, *, sigma, mass_bins):
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) < 2 or not np.isfinite(values).all():
        raise ValueError("Invalid Stein closure operator sample")
    tolerance = 10 * np.finfo(np.float32).eps

    def one(sample):
        mean = float(np.mean(sample))
        error = float(np.std(sample, ddof=1) / np.sqrt(len(sample))) if len(sample) > 1 else 0.0
        passed = abs(mean) <= float(sigma) * error + tolerance
        zscore = 0.0 if error == 0 and abs(mean) <= tolerance else (None if error == 0 else mean / error)
        return dict(events=len(sample), mean_operator=mean, standard_error=error,
                    z_score=zscore, passed=bool(passed))

    overall = one(values)
    bins = []
    if contexts is not None and int(mass_bins) > 1:
        contexts = np.asarray(contexts, dtype=np.float64)
        if contexts.shape != values.shape or not np.isfinite(contexts).all():
            raise ValueError("Invalid Stein closure mass contexts")
        order = np.argsort(contexts, kind="stable")
        groups = np.array_split(order, min(int(mass_bins), len(values)))
        for index, group in enumerate(groups):
            if len(group) < 2:
                continue
            result = one(values[group])
            result.update(index=index, context_min=float(np.min(contexts[group])),
                          context_max=float(np.max(contexts[group])))
            bins.append(result)
    status = "passed" if overall["passed"] and all(row["passed"] for row in bins) else "failed"
    return dict(status=status, sigma_threshold=float(sigma), numerical_tolerance=float(tolerance),
                overall=overall, mass_bins=bins)


def stein_identity_diagnostics(output, order, reference, device):
    output = Path(output)
    if not len(order):
        raise ValueError("Stein closure requires at least one checkpoint")
    reference = np.asarray(reference, dtype=np.float32)
    inputs = json.loads((output / "residual_training_inputs.json").read_text())
    if inputs.get("core") != "stein_witness" or inputs.get("stein_protocol") != PROTOCOL:
        raise ValueError("Stein closure protocol differs from training")
    if reference.ndim != 2 or reference.shape[1] != inputs["features"] or len(reference) < 2:
        raise ValueError("Invalid Stein closure reference")
    if not np.isfinite(reference).all():
        raise ValueError("Nonfinite Stein closure reference")
    settings = inputs["settings"]
    model = build_potential(device, reference.shape[1], settings, initialization="random").eval()
    model.requires_grad_(False)
    corrected_background = None
    info = inputs.get("background_correction")
    if info is not None:
        from .background_correction import load_local
        loaded = load_local(output, settings, reference.shape[1] - 1, device)
        if loaded is None:
            raise ValueError("Missing fit-local corrected background for Stein closure")
        corrected_background, saved = loaded
        if saved.get("source_sha256") != info.get("source_sha256"):
            raise ValueError("Stein closure background identity changed")
    background_score = _background_score_array(
        reference, corrected_background, device,
        mass_conditioning=bool(settings.get("mass_conditioning", False)),
    )
    weights = _checkpoint_weights(output, order)
    combined = np.zeros(len(reference), dtype=np.float64)
    checkpoint_summaries = []
    batch = 2048
    reference_tensor = torch.as_tensor(reference, dtype=torch.float32, device=device)
    background_tensor = torch.as_tensor(background_score, dtype=torch.float32, device=device)
    for epoch, weight in zip(order, weights):
        checkpoint = torch.load(output / f"residual_epoch_{epoch}.pt", map_location=device, weights_only=True)
        if (checkpoint.get("scientific_version") != SCIENTIFIC_VERSION
                or checkpoint.get("core") != "stein_witness"
                or checkpoint.get("stein_protocol") != PROTOCOL
                or checkpoint.get("epoch") != epoch):
            raise ValueError("Stein closure checkpoint identity differs from selection")
        model.load_state_dict(checkpoint["model"])
        pieces = []
        for offset in range(0, len(reference), batch):
            x = reference_tensor[offset:offset + batch]
            qscore = background_tensor[offset:offset + batch]
            with torch.enable_grad():
                operator, _, _ = _stein_operator_values(model, x, qscore, training=False)
            pieces.append(operator.detach().cpu().numpy())
        values = np.concatenate(pieces).astype(np.float64)
        if not np.isfinite(values).all():
            raise FloatingPointError("Nonfinite Stein closure operator")
        contexts = reference[:, -1] if model.mass_conditioning else None
        summary = _closure_summary(
            values, contexts, sigma=settings["stein"]["closure_sigma"],
            mass_bins=settings["stein"]["closure_mass_bins"],
        )
        checkpoint_summaries.append(dict(epoch=int(epoch), weight=float(weight), **summary))
        combined += float(weight) * values
    contexts = reference[:, -1] if model.mass_conditioning else None
    ensemble = _closure_summary(
        combined, contexts, sigma=settings["stein"]["closure_sigma"],
        mass_bins=settings["stein"]["closure_mass_bins"],
    )
    return dict(
        status=ensemble["status"],
        identity="E_q[T_q g]=0",
        reference_samples=len(reference),
        reference_sha256=digest(reference),
        truth_labels_used=False,
        conditional_mass_matching=bool(model.mass_conditioning),
        ensemble=ensemble,
        checkpoints=checkpoint_summaries,
    )


def _payload(model, epoch):
    return {
        "model": {key: value.detach().cpu().clone() for key, value in model.state_dict().items()},
        "epoch": int(epoch),
        "scientific_version": SCIENTIFIC_VERSION,
        "core": "stein_witness",
        "stein_protocol": PROTOCOL,
    }


def _offer(candidates, count, epoch, validation_loss, model):
    key = (float(validation_loss), int(epoch))
    if len(candidates) >= count and key >= max((c["validation_nll"], c["epoch"]) for c in candidates):
        return
    candidates.append({
        "epoch": int(epoch),
        "validation_nll": float(validation_loss),
        "filename": f"residual_epoch_{epoch}.pt",
        "sha256": None,
        "payload": _payload(model, epoch),
    })
    candidates.sort(key=lambda c: (c["validation_nll"], c["epoch"]))
    if len(candidates) > count:
        candidates.pop()


def _persist(output, candidates):
    for candidate in candidates:
        if candidate.get("sha256") is None:
            candidate["sha256"] = atomic_torch_save(Path(output) / candidate["filename"], candidate.pop("payload"))
    return {candidate["filename"]: candidate["sha256"] for candidate in candidates}


def _candidate_metadata(candidates):
    return [{k: candidate[k] for k in ("epoch", "validation_nll", "filename", "sha256")} for candidate in candidates]


def _validate_recovery(checkpoint, epochs, settings, background_sha256):
    if checkpoint.get("scientific_version") != SCIENTIFIC_VERSION or checkpoint.get("core") != "stein_witness":
        raise ValueError("Stein recovery checkpoint uses a different scientific protocol")
    if (checkpoint.get("stein_protocol") != PROTOCOL
            or _training_contract(checkpoint.get("settings", {})) != _training_contract(settings)):
        raise ValueError("Stein recovery settings or protocol changed; use a new output")
    if checkpoint.get("background_correction_sha256") != background_sha256:
        raise ValueError("Stein background correction changed; use a new output")
    epoch = checkpoint.get("epoch")
    history = checkpoint.get("history")
    if type(epoch) is not int or not 0 <= epoch < epochs or not isinstance(history, list) or len(history) != epoch + 1:
        raise ValueError("Invalid Stein recovery history")
    if any(row.get("epoch") != index or not np.isfinite([row.get("train_nll"), row.get("validation_nll")]).all()
           for index, row in enumerate(history)):
        raise ValueError("Invalid Stein recovery losses")
    candidates = checkpoint.get("candidates")
    files = checkpoint.get("files")
    if not isinstance(candidates, list) or not isinstance(files, dict):
        raise ValueError("Stein recovery lacks checkpoint inventory")
    count = settings["training"]["selected_checkpoints"]
    expected = sorted(range(len(history)), key=lambda i: (history[i]["validation_nll"], i))[:count]
    if [candidate.get("epoch") for candidate in candidates] != expected:
        raise ValueError("Stein recovery candidates disagree with validation history")
    expected_files = {}
    for candidate in candidates:
        epoch = candidate.get("epoch")
        name = candidate.get("filename")
        sha = candidate.get("sha256")
        if name != f"residual_epoch_{epoch}.pt" or candidate.get("validation_nll") != history[epoch]["validation_nll"]:
            raise ValueError("Invalid Stein recovery checkpoint metadata")
        if not isinstance(sha, str) or len(sha) != 64:
            raise ValueError("Invalid Stein recovery checkpoint digest")
        expected_files[name] = sha
    if files != expected_files:
        raise ValueError("Stein recovery checkpoint inventory is inconsistent")


def train_stein_witness(train, validation, output, *, epochs, seed, device, checkpoint=None,
                         fraction=None, initialization="background", progress_label="Train Stein witness",
                         on_training_start=None, settings=None, background_correction=None, after_epoch=None,
                         truth_labels_used=False):
    if type(truth_labels_used) is not bool:
        raise ValueError("truth_labels_used must be boolean")
    settings = deepcopy(settings)
    settings.update(epochs=epochs, initialization=initialization)
    settings = validate_residual(settings)
    if settings["core"] != "stein_witness":
        raise ValueError("Stein trainer received a non-Stein core")
    if fraction is not None:
        raise ValueError("stein_witness does not use a mixture fraction")
    options = settings["training"]
    additions = feature_options(settings)
    mass_conditioning = settings.get("mass_conditioning", False)
    ztrain, zval = (real_sr_latents(rows, mass_conditioning=mass_conditioning, physical_inputs=False)
                    for rows in (train, validation))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if str(device).startswith("cuda"):
        torch.cuda.manual_seed_all(seed)
    model = build_potential(device, ztrain.shape[1], settings, initialization=initialization)
    corrected_background = None
    corrected_background_digest = None
    if background_correction is not None:
        from .background_correction import load as load_background_correction
        corrected_background = load_background_correction(background_correction, settings, ztrain.shape[1] - 1, device)
        corrected_background.eval()
        corrected_background.requires_grad_(False)
    background_sha256 = None if background_correction is None else background_correction["model_sha256"]
    if checkpoint is not None:
        _validate_recovery(checkpoint, epochs, settings, background_sha256)
    qtrain = _background_score_array(ztrain, corrected_background, device, mass_conditioning=mass_conditioning)
    qval = _background_score_array(zval, corrected_background, device, mass_conditioning=mass_conditioning)
    train_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(ztrain), torch.from_numpy(qtrain)),
        batch_size=options["batch_size"], shuffle=True,
    )
    val_loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.from_numpy(zval), torch.from_numpy(qval)),
        batch_size=options["validation_batch_size"], shuffle=False,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=options["learning_rate"], weight_decay=options["weight_decay"])
    control = settings.get("optimization", {})
    scheduler = None
    if control.get("lr_factor", 1) < 1:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=control["lr_factor"], patience=control["lr_patience"],
            threshold=control["min_delta"], threshold_mode="abs", min_lr=control["min_lr"],
        )
    history, files, candidates, start = [], {}, [], 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(checkpoint["scheduler"])
        history = checkpoint["history"]
        files = checkpoint["files"]
        candidates = [dict(candidate) for candidate in checkpoint["candidates"]]
        start = checkpoint["epoch"] + 1
        restore_rng(checkpoint["rng"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if corrected_background is not None:
        from .background_correction import save_local
        corrected_background_digest = save_local(output, corrected_background, background_correction)
    write_json(output / "residual_training_inputs.json", {
        "scientific_version": SCIENTIFIC_VERSION,
        "core": "stein_witness",
        "stein_protocol": PROTOCOL,
        "stein_scoring_protocol": SCORING_PROTOCOL,
        "train_events": len(ztrain),
        "validation_events": len(zval),
        "train_latents_sha256": digest(ztrain),
        "validation_latents_sha256": digest(zval),
        "background_score_train_sha256": digest(qtrain),
        "background_score_validation_sha256": digest(qval),
        "data_reference_label": 1,
        "truth_labels_used": truth_labels_used,
        "mass_input_used": False,
        "witness_mass_input_used": False,
        "background_mass_conditioning": mass_conditioning,
        "initialization": initialization,
        "settings": settings,
        "features": ztrain.shape[1],
        "score": "negative learned Stein potential baseline; post-training scoring is independently configurable",
        "background_correction": (None if corrected_background is None else {
            "mode": background_correction["mode"],
            "protocol": background_correction["protocol"],
            "source_sha256": background_correction["model_sha256"],
            "local_sha256": corrected_background_digest,
            "selected_epoch": background_correction["selected_epoch"],
            "epochs": background_correction["epochs"],
        }),
    })
    if on_training_start is not None:
        on_training_start()
    progress = ProgressStage("residual_training", progress_label, epochs, "epoch", initial=start, report_every=1)
    best = min((row["validation_nll"] for row in history), default=float("inf"))
    stale = 0
    if history:
        best = float("inf")
        for row in history:
            if row["validation_nll"] < best - control.get("min_delta", 0):
                best, stale = row["validation_nll"], 0
            else:
                stale += 1
    stopped = bool(checkpoint and checkpoint.get("stopped_early", False))
    if stopped:
        start = epochs
    durable_history = list(history)
    epoch_seconds = []
    try:
        for epoch in range(start, epochs):
            epoch_started = time.monotonic()
            flow_lr = optimizer.param_groups[0]["lr"]
            progress.substep("Train", 0, len(train_loader))
            train_loss, diagnostics = _epoch(
                model, train_loader, optimizer, settings, additions, corrected_background,
                epoch=epoch, seed=seed, progress=lambda done, total: progress.substep("Train", done, total),
            )
            progress.substep("Validation", 0, len(val_loader))
            validation_loss = _validation_loss(model, val_loader, settings)
            row = {
                "epoch": epoch,
                "train_nll": float(train_loss),
                "validation_nll": float(validation_loss),
                "signal_fraction": 0.0,
                "train_objective": float(train_loss),
                "validation_objective": float(validation_loss),
                **diagnostics,
            }
            if control:
                row.update(flow_learning_rate=flow_lr, fraction_warmup=False)
            if not np.isfinite([train_loss, validation_loss]).all():
                raise FloatingPointError("Nonfinite Stein training state")
            history.append(row)
            if scheduler is not None:
                scheduler.step(validation_loss)
            if validation_loss < best - control.get("min_delta", 0):
                best, stale = validation_loss, 0
            else:
                stale += 1
            stopped = bool(control.get("early_stopping_patience", 0)
                           and epoch + 1 >= control.get("minimum_epochs", 1)
                           and stale >= control["early_stopping_patience"])
            _offer(candidates, options["selected_checkpoints"], epoch, validation_loss, model)
            epoch_seconds.append(time.monotonic() - epoch_started)
            durable = persist_boundary(epoch, epochs) or stopped
            if durable:
                progress.update(force=True, operation="Persist 10-epoch recovery boundary", minibatch="-")
                files = _persist(output, candidates)
                state = {
                    "scientific_version": SCIENTIFIC_VERSION,
                    "core": "stein_witness",
                    "stein_protocol": PROTOCOL,
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler else None,
                    "rng": rng_state(),
                    "history": history,
                    "files": files,
                    "candidates": _candidate_metadata(candidates),
                    "settings": settings,
                    "background_correction_sha256": background_sha256,
                    "stopped_early": stopped,
                }
                atomic_torch_save(output / ".resume/latest.pt", state)
                write_json(output / "residual_losses.json", {"history": history})
                durable_history = list(history)
            progress.update(epoch + 1, force=True, operation="Early stop" if stopped else "Epoch complete",
                            train_objective=train_loss, validation_objective=validation_loss)
            if after_epoch is not None:
                after_epoch(epoch)
            if stopped:
                break
    except BaseException:
        write_json(output / "residual_losses.json", {"history": durable_history})
        raise
    write_json(output / "residual_losses.json", {"history": history})
    from .background_stage import record_timing
    record_timing(output, "epochs", sum(epoch_seconds), epoch_seconds=epoch_seconds)
    order = ordered_epochs([row["validation_nll"] for row in history], options["selected_checkpoints"])
    if [candidate["epoch"] for candidate in candidates] != order:
        raise ValueError("Buffered Stein Top-N checkpoint inventory disagrees with final selection")
    if set(files) != {f"residual_epoch_{epoch}.pt" for epoch in order}:
        raise ValueError("Final Stein checkpoint files disagree with validation selection")
    weighting = additions.get("checkpoint_weighting", "uniform")
    if weighting != "uniform":
        raise ValueError("Stein checkpoint weighting must be uniform")
    selected = np.asarray([history[index]["validation_nll"] for index in order], dtype=np.float64)
    checkpoint_weights = np.full(len(order), 1.0 / len(order), dtype=np.float64)
    write_json(output / "residual_selection.json", {
        "core": "stein_witness",
        "stein_protocol": PROTOCOL,
        "epochs": order,
        "criterion": f"{options['selected_checkpoints']} lowest validation Stein objectives",
        "witness_ensemble": "uniform arithmetic mean of selected Stein witness checkpoint scores",
        "checkpoint_weighting": weighting,
        "checkpoint_weights": [float(value) for value in checkpoint_weights],
        "validation_objective": [float(value) for value in selected],
        "signal_fractions": [0.0 for _ in order],
        "mass_fraction": {"enabled": False},
        "trained_epochs": len(history),
        "maximum_epochs": epochs,
        "stopped_early": stopped,
    })
    return order, history


def _checkpoint_weights(output, order):
    selection_path = Path(output) / "residual_selection.json"
    weights = np.array([], dtype=np.float64)
    if selection_path.exists():
        selection = json.loads(selection_path.read_text())
        saved_epochs = list(selection.get("epochs", []))
        saved_weights = np.asarray(selection.get("checkpoint_weights", []), dtype=np.float64)
        if saved_weights.shape == (len(saved_epochs),) and all(epoch in saved_epochs for epoch in order):
            weights = np.asarray([saved_weights[saved_epochs.index(epoch)] for epoch in order], dtype=np.float64)
            weights /= weights.sum()
    if weights.shape != (len(order),):
        weights = np.full(len(order), 1.0 / len(order), dtype=np.float64)
    if not np.isfinite(weights).all() or np.any(weights <= 0) or not np.isclose(weights.sum(), 1.0):
        raise ValueError("Invalid Stein checkpoint weights")
    return weights


def stein_scores(output, order, z, device, *, normalization_checks=None, normalization_tests=1,
                 mode=None, scoring_root=None, scoring_settings=None):
    from .stein_scoring import member_scores

    return member_scores(
        output, order, z, device, mode=mode, scoring_root=scoring_root,
        normalization_checks=normalization_checks, scoring_settings=scoring_settings,
    )
