"""Latent background correction used by the bgcorr_40_reguide RIDDLE profile.

The correction q_phi(z|m) is trained once per RIDDLE method run on the reserved
correction sidebands.  Every residual fit in that run reuses the same frozen
q_phi for initialization, guide construction, likelihood ratios and scoring.
"""
from copy import deepcopy
from pathlib import Path
import json
import math

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .integrity import SCIENTIFIC_VERSION, require_finite
from .model import build_signal_flow, match_background
from .storage import (atomic_write, digest, file_digest, rng_state, restore_rng,
                      write_json)
from .worker_progress import emit_message

MODE = "bgcorr_40_reguide"
EPOCHS = 40
PROTOCOL = "shared_sideband_qphi_reguide_v1"


def enabled(settings):
    return settings.get("background_correction", "none") == MODE


def _model(settings, dimensions, device):
    cfg = deepcopy(settings)
    cfg["mass_conditioning"] = True
    cfg["enhancements"] = deepcopy(cfg.get("enhancements", {}))
    cfg["enhancements"]["score_flow"] = False
    model = build_signal_flow(device, features=dimensions + 1, settings=cfg)
    match_background(model)
    return model


def log_prob(model, latent, context):
    """log q_phi(z|m), with context already equal to (mjj-3.5)/0.2."""
    return model.log_prob(latent, context=context.reshape(-1, 1))


def sample(model, context, dimensions, seed, device):
    """Sample q_phi(z|m) at matched mass contexts without perturbing caller RNG."""
    context = torch.as_tensor(context, dtype=torch.float32, device=device).reshape(-1, 1)
    devices = [] if str(device) == "cpu" else [torch.device(device).index or 0]
    with torch.random.fork_rng(devices=devices):
        torch.manual_seed(int(seed))
        if str(device).startswith("cuda"):
            torch.cuda.manual_seed_all(int(seed))
        values = model.sample(1, context=context)
    if values.ndim == 3 and values.shape[1] == 1:
        values = values[:, 0, :]
    elif values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if tuple(values.shape) != (len(context), dimensions):
        raise ValueError(f"Unexpected conditional background sample shape {tuple(values.shape)}")
    require_finite(values, "Corrected-background samples")
    return values


def _contract(settings, train_z, train_mass, val_z, val_mass, seed, device):
    return dict(
        schema=1,
        scientific_version=SCIENTIFIC_VERSION,
        protocol=PROTOCOL,
        mode=MODE,
        epochs=EPOCHS,
        seed=int(seed),
        device=str(device),
        flow=settings["flow"],
        learning_rate=settings["training"]["learning_rate"],
        hashes={
            "train_z": digest(train_z), "train_mass": digest(train_mass),
            "validation_z": digest(val_z), "validation_mass": digest(val_mass),
        },
    )


def train(directory, train_z, train_mass, val_z, val_mass, *, settings, seed, device):
    """Fit the one shared q_phi(z|m) using exactly 40 reserved-sideband epochs."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    train_z = np.ascontiguousarray(train_z, dtype=np.float32)
    val_z = np.ascontiguousarray(val_z, dtype=np.float32)
    train_mass = np.ascontiguousarray(train_mass, dtype=np.float32)
    val_mass = np.ascontiguousarray(val_mass, dtype=np.float32)
    if train_z.ndim != 2 or val_z.shape[1:] != train_z.shape[1:] or min(len(train_z), len(val_z)) < 30:
        raise ValueError("Background correction requires aligned sideband latent samples")
    for z, m in ((train_z, train_mass), (val_z, val_mass)):
        if m.shape != (len(z),) or not np.isfinite(z).all() or not np.isfinite(m).all():
            raise ValueError("Invalid background-correction inputs")
        if ((m > 3.3) & (m < 3.7)).any():
            raise ValueError("Background correction is sideband-only; SR events are forbidden")
    contract = _contract(settings, train_z, train_mass, val_z, val_mass, seed, device)
    contract_path = directory / "contract.json"
    if contract_path.exists() and json.loads(contract_path.read_text()) != contract:
        raise ValueError("Background-correction inputs/settings changed; use a new output directory")
    write_json(contract_path, contract)

    torch.manual_seed(int(seed)); np.random.seed(int(seed) % 2**32)
    model = _model(settings, train_z.shape[1], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["training"]["learning_rate"], weight_decay=1e-4)
    history, start, best, best_epoch, best_model = [], 0, float("inf"), None, None
    latest = directory / ".resume/latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        if state.get("contract") != contract:
            raise ValueError("Background-correction recovery contract changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, start = state["history"], state["epoch"] + 1
        best, best_epoch, best_model = state["best"], state["best_epoch"], state["best_model"]
        restore_rng(state["rng"])

    zt = torch.from_numpy(train_z)
    mt = torch.from_numpy(((train_mass - 3.5) / 0.2).astype(np.float32))
    zv = torch.from_numpy(val_z).to(device)
    mv = torch.from_numpy(((val_mass - 3.5) / 0.2).astype(np.float32)).to(device)
    dataset = TensorDataset(zt, mt)
    for epoch in range(start, EPOCHS):
        generator = torch.Generator().manual_seed(int(seed) + 11000 + epoch)
        loader = DataLoader(dataset, batch_size=512, shuffle=True, generator=generator)
        model.train(); total = 0.0
        for z, m in loader:
            z, m = z.to(device), m.to(device)
            optimizer.zero_grad()
            loss = -log_prob(model, z, m).mean()
            require_finite(loss, "Latent-background correction loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach()) * len(z)
        model.eval()
        with torch.no_grad():
            validation_nll = -float(log_prob(model, zv, mv).double().mean().cpu())
            gaussian_nll = -float((-.5 * (zv.square() + math.log(2 * math.pi)).sum(-1)).double().mean().cpu())
        row = dict(epoch=epoch, train_nll=total / len(train_z), validation_nll=validation_nll,
                   gaussian_validation_nll=gaussian_nll, improvement=gaussian_nll-validation_nll)
        history.append(row)
        if (validation_nll, epoch) < (best, best_epoch if best_epoch is not None else math.inf):
            best, best_epoch, best_model = validation_nll, epoch, deepcopy(model.state_dict())
        state = dict(contract=contract, epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                     history=history, best=best, best_epoch=best_epoch, best_model=best_model, rng=rng_state())
        latest.parent.mkdir(parents=True, exist_ok=True)
        atomic_write(latest, lambda p, value=state: torch.save(value, p))
        write_json(directory / "history.json", history)
        emit_message(f"Background correction {epoch+1}/{EPOCHS}: validation NLL={validation_nll:.6g}")

    if best_model is None:
        raise FloatingPointError("Background correction produced no valid checkpoint")
    selected = {"model": best_model, "selected_epoch": best_epoch, "contract": contract}
    atomic_write(directory / "model.pt", lambda p: torch.save(selected, p))
    write_json(directory / "selection.json", dict(mode=MODE, selected_epoch=best_epoch, epochs=EPOCHS,
               criterion="lowest reserved correction-validation NLL", truth_labels_used=False,
               training_region="reserved sidebands", denominator="q_phi(z|m)"))
    return descriptor(directory)


def descriptor(directory):
    directory = Path(directory)
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    contract = state["contract"]
    if contract.get("mode") != MODE or contract.get("epochs") != EPOCHS:
        raise ValueError("Invalid bgcorr_40_reguide model")
    return dict(mode=MODE, protocol=PROTOCOL, epochs=EPOCHS,
                selected_epoch=int(state["selected_epoch"]), model_path=str((directory / "model.pt").resolve()),
                model_sha256=file_digest(directory / "model.pt"), contract=contract)


def load(description, settings, features, device):
    """Load and verify the shared q_phi descriptor on a fit/scoring worker."""
    if not description or description.get("mode") != MODE:
        return None
    path = Path(description["model_path"])
    if not path.is_file() or file_digest(path) != description.get("model_sha256"):
        raise ValueError("Shared background-correction artifact changed or is missing")
    state = torch.load(path, map_location=device, weights_only=False)
    if state.get("contract") != description.get("contract"):
        raise ValueError("Shared background-correction contract changed")
    model = _model(settings, features, device)
    model.load_state_dict(state["model"])
    return model.eval().requires_grad_(False)


def save_local(directory, model, description):
    """Persist the shared denominator state inside one fit for self-contained scoring."""
    directory = Path(directory)
    path = directory / "background_correction.pt"
    payload = {"model": model.state_dict(), "mode": MODE, "protocol": PROTOCOL,
               "source_sha256": description["model_sha256"],
               "selected_epoch": description["selected_epoch"],
               "contract": description["contract"]}
    if path.exists():
        saved = torch.load(path, map_location="cpu", weights_only=False)
        if saved.get("mode") != MODE or saved.get("source_sha256") != description["model_sha256"] or saved.get("contract") != description["contract"]:
            raise ValueError("Persisted fit background correction changed")
    else:
        atomic_write(path, lambda p: torch.save(payload, p))
    return file_digest(path)


def load_local(directory, settings, features, device):
    path = Path(directory) / "background_correction.pt"
    if not path.exists():
        return None
    saved = torch.load(path, map_location=device, weights_only=False)
    if saved.get("mode") != MODE or saved.get("protocol") != PROTOCOL:
        raise ValueError("Invalid fit-local background correction")
    model = _model(settings, features, device)
    model.load_state_dict(saved["model"])
    return model.eval().requires_grad_(False), saved
