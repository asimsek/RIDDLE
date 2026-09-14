import json
from pathlib import Path
import random
import numpy as np
import torch
from copy import deepcopy
from .settings import DEFAULTS, validate_residual
from riddle.worker_progress import ProgressStage
from riddle.storage import atomic_write, write_json, digest, file_digest, rng_state, restore_rng
from .model import (
    build_signal_flow,
    initial_fraction_logit,
    real_sr_latents,
    train_epoch,
    background_log_prob,
    residual_optimizer,
)
from .integrity import SCIENTIFIC_VERSION, ordered_epochs


def validate_checkpoint(checkpoint, epochs):
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
    expected = {f"residual_epoch_{i}.pt" for i in range(epoch + 1)}
    if not isinstance(checkpoint.get("files"), dict) or set(checkpoint["files"]) != expected:
        raise ValueError("Incomplete residual recovery checkpoint file inventory")
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
):
    settings = deepcopy(DEFAULTS["riddle"] if settings is None else settings)
    settings.update(epochs=epochs, initialization=initialization)
    settings = validate_residual(settings)
    options = settings["training"]
    if checkpoint is not None:
        validate_checkpoint(checkpoint, epochs)
        if checkpoint.get("settings") != settings:
            raise ValueError("Residual training settings changed; use a new output")
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
    ztrain, zval = real_sr_latents(train), real_sr_latents(validation)
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
    if initialization == "background":
        from .model import match_background

        match_background(model)
    elif initialization != "random":
        raise ValueError("Unknown residual initialization")
    if fraction is not None:
        if not np.isfinite(fraction) or not 0 < fraction < 1:
            raise ValueError("Fixed signal fraction must lie in (0,1)")
        logit = torch.tensor(np.log(fraction / (1 - fraction)), dtype=torch.float64, device=device)
    optimizer = residual_optimizer(model, logit, options)
    history, files, start = [], {}, 0
    if checkpoint is not None:
        model.load_state_dict(checkpoint["model"])
        with torch.no_grad():
            logit.copy_(checkpoint["fraction_logit"].to(device))
        optimizer.load_state_dict(checkpoint["optimizer"])
        history, files, start = checkpoint["history"], checkpoint["files"], checkpoint["epoch"] + 1
        restore_rng(checkpoint["rng"])
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
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
            "mass_input_used": False,
            "initial_fraction": float(initial_fraction_logit(seed).sigmoid().detach())
            if fraction is None
            else fraction,
            "fraction_mode": "learned" if fraction is None else "fixed",
            "initialization": initialization,
            "settings": settings,
            "features": ztrain.shape[1],
        },
    )
    if on_training_start is not None:
        on_training_start()
    for epoch in range(start, epochs):
        losses = {}
        for part, loader, opt in (("train", train_loader, optimizer), ("validation", val_loader, None)):
            progress.substep(part.title(), 0, len(loader))
            losses[part + "_nll"] = train_epoch(
                model,
                logit,
                loader,
                opt,
                lambda done, total: progress.substep(part.title(), done, total),
                gradient_clip_norm=options["gradient_clip_norm"],
            )
        fraction = float(logit.sigmoid().detach())
        if not np.isfinite([*losses.values(), fraction]).all():
            raise FloatingPointError("Nonfinite residual training state")
        row = {"epoch": epoch, **losses, "signal_fraction": fraction}
        history.append(row)
        name = f"residual_epoch_{epoch}.pt"
        progress.update(force=True, operation="Save residual checkpoint", minibatch="-")
        atomic_write(
            output / name,
            lambda p: torch.save(
                {"model": model.state_dict(), "fraction_logit": logit.detach(), "epoch": epoch,
                 "scientific_version": SCIENTIFIC_VERSION}, p
            ),
        )
        files[name] = file_digest(output / name)
        state = {
            "scientific_version": SCIENTIFIC_VERSION,
            "epoch": epoch,
            "model": model.state_dict(),
            "fraction_logit": logit.detach(),
            "optimizer": optimizer.state_dict(),
            "rng": rng_state(),
            "history": history,
            "files": files,
            "settings": settings,
        }
        atomic_write(output / ".resume/latest.pt", lambda p: torch.save(state, p))
        write_json(output / "residual_losses.json", {"history": history})
        progress.update(epoch + 1, force=True, operation="Epoch complete", **losses, fraction=fraction)
        if after_epoch is not None:
            after_epoch(epoch)
    write_json(output / "residual_losses.json", {"history": history})
    order = ordered_epochs([r["validation_nll"] for r in history], options["selected_checkpoints"])
    write_json(
        output / "residual_selection.json",
        {
            "epochs": order,
            "criterion": f"{options['selected_checkpoints']} lowest validation mixture NLL epochs",
            "density_ensemble": "arithmetic mean",
            "signal_fractions": [history[i]["signal_fraction"] for i in order],
        },
    )
    return order, history


def residual_scores(output, order, z, device):
    if not len(order):
        raise ValueError("Residual scoring requires at least one checkpoint")
    inputs = json.loads((output / "residual_training_inputs.json").read_text())
    if inputs.get("scientific_version") != SCIENTIFIC_VERSION:
        raise ValueError("Legacy residual inputs; use saved legacy scores or retrain")
    if inputs["features"] != z.shape[1]:
        raise ValueError("Scoring feature count differs from training")
    model = build_signal_flow(device, features=z.shape[1], settings=inputs["settings"]).eval()
    log_sum = None
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
                for offset in range(0, len(z), 8192):
                    x = torch.as_tensor(z[offset : offset + 8192], device=device)
                    chunks.append(model.log_prob(x).cpu().numpy())
                    progress.update(i * len(z) + min(offset + 8192, len(z)))
                log_density = np.concatenate(chunks).astype(np.float64)
                if not np.isfinite(log_density).all():
                    raise FloatingPointError("Nonfinite residual checkpoint density")
                log_sum = log_density if log_sum is None else np.logaddexp(log_sum, log_density)
    log_signal = log_sum - np.log(len(order))
    log_background = background_log_prob(torch.from_numpy(z)).numpy()
    scores = log_signal.astype(np.float64) - log_background.astype(np.float64)
    if not np.isfinite(scores).all():
        raise FloatingPointError("Nonfinite residual ensemble scores")
    return scores
