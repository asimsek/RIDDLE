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
from .storage import (atomic_torch_save, digest, file_digest, rng_state, restore_rng,
                      write_json, persist_boundary)
from .worker_progress import emit_message
from .options import feature_options

MODE = "bgcorr_40_reguide"
EPOCHS = 40
PROTOCOL = "shared_sideband_qphi_reguide_independent_diagnostic"
CLOSURE_SCOPE = "correction_only_interpolation_on_fixed_upstream_map; not_full_search_closure"
MIN_VALIDATION_IMPROVEMENT = 0.0
# Three localized interpolation probes per sideband.  Quantile windows are
# deliberately separated and leave training support on both sides of every gap.
PSEUDO_WINDOW_QUANTILES = ((0.15, 0.30), (0.425, 0.575), (0.70, 0.85))
MIN_PSEUDO_WINDOW_EVENTS = 30
MIN_POSITIVE_WINDOWS_PER_SIDE = 2
PSEUDO_COMPATIBILITY_SIGMA = 1.96


def enabled(settings):
    return settings.get("background_correction", "none") == MODE


def _qphi_options(settings):
    options = feature_options(settings)
    epochs = int(options.get("qphi_epochs", EPOCHS))
    bins = int(options.get("qphi_mass_bins", 1))
    if bins > 1 and bins % 2:
        raise ValueError("qphi_mass_bins must be 1 or an even number")
    return epochs, bins


def _balanced_indices(mass, bins, seed):
    """Equalize lower/upper sidebands and mass-quantile strata without labels."""
    if bins <= 1:
        return None
    rng = np.random.default_rng(int(seed))
    mass = np.asarray(mass)
    groups = []
    per_side = bins // 2
    for mask in (mass < 3.3, mass > 3.7):
        indices = np.flatnonzero(mask)
        if len(indices) < per_side:
            raise ValueError("Insufficient sideband events for mass-balanced q_phi batches")
        ordered = indices[np.argsort(mass[indices], kind="stable")]
        groups.extend(np.array_split(ordered, per_side))
    if len(groups) != bins or any(len(g) == 0 for g in groups):
        raise ValueError("Could not construct balanced q_phi mass strata")
    target = int(math.ceil(len(mass) / bins))
    chosen = [rng.choice(g, size=target, replace=len(g) < target) for g in groups]
    merged = np.concatenate(chosen)
    rng.shuffle(merged)
    return merged.astype(np.int64, copy=False)


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
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Corrected-background sampling supports CPU and CUDA devices")
    index = (torch.cuda.current_device() if target.index is None else target.index) if target.type == "cuda" else None
    devices = [] if index is None else [index]
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.random.default_generator.manual_seed(int(seed))
        if index is not None:
            torch.cuda.default_generators[index].manual_seed(int(seed))
        values = model.sample(1, context=context)
    if values.ndim == 3 and values.shape[1] == 1:
        values = values[:, 0, :]
    elif values.ndim == 3 and values.shape[0] == 1:
        values = values[0]
    if tuple(values.shape) != (len(context), dimensions):
        raise ValueError(f"Unexpected conditional background sample shape {tuple(values.shape)}")
    require_finite(values, "Corrected-background samples")
    return values


def _gaussian_log_prob(latent):
    return -.5 * (latent.square() + math.log(2 * math.pi)).sum(-1)


def _validation_metrics(model, latent, mass, device):
    """Compare q_phi with Gaussian on caller-specified rows (not necessarily independent)."""
    z = torch.as_tensor(latent, dtype=torch.float32, device=device)
    m = torch.as_tensor(((np.asarray(mass, dtype=np.float32) - 3.5) / 0.2),
                        dtype=torch.float32, device=device)
    with torch.no_grad():
        qlog = log_prob(model, z, m).double()
        glog = _gaussian_log_prob(z).double()
    gain = (qlog - glog).cpu().numpy()
    require_finite(gain, "Background-correction validation gain")
    return dict(
        events=int(len(gain)),
        qphi_nll=-float(qlog.mean().cpu()),
        gaussian_nll=-float(glog.mean().cpu()),
        improvement=float(gain.mean()),
        improvement_standard_error=(float(gain.std(ddof=1) / math.sqrt(len(gain)))
                                    if len(gain) > 1 else None),
    )


def _fit_probe(train_z, train_mass, val_z, val_mass, *, settings, seed, device):
    """Deterministic auxiliary fit used only for masked-mass interpolation closure."""
    torch.manual_seed(int(seed)); np.random.seed(int(seed) % 2**32)
    model = _model(settings, train_z.shape[1], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["training"]["learning_rate"],
                                  weight_decay=1e-4)
    dataset = TensorDataset(torch.from_numpy(np.ascontiguousarray(train_z, dtype=np.float32)),
                            torch.from_numpy(((np.asarray(train_mass, dtype=np.float32)-3.5)/0.2).astype(np.float32)))
    zv = torch.as_tensor(val_z, dtype=torch.float32, device=device)
    mv = torch.as_tensor(((np.asarray(val_mass, dtype=np.float32)-3.5)/0.2), dtype=torch.float32, device=device)
    epochs, mass_bins = _qphi_options(settings)
    best, best_epoch, best_model = float("inf"), None, None
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(int(seed) + 21000 + epoch)
        balanced = _balanced_indices(train_mass, mass_bins, int(seed) + 21500 + epoch)
        epoch_dataset = dataset if balanced is None else TensorDataset(dataset.tensors[0][balanced], dataset.tensors[1][balanced])
        loader = DataLoader(epoch_dataset, batch_size=512, shuffle=balanced is None, generator=generator)
        model.train()
        for z, m in loader:
            z, m = z.to(device), m.to(device)
            optimizer.zero_grad()
            loss = -log_prob(model, z, m).mean()
            require_finite(loss, "Pseudo-SR background-correction loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_nll = -float(log_prob(model, zv, mv).double().mean().cpu())
        require_finite(validation_nll, "Background-correction checkpoint validation NLL")
        if (validation_nll, epoch) < (best, best_epoch if best_epoch is not None else math.inf):
            best, best_epoch, best_model = validation_nll, epoch, deepcopy(model.state_dict())
    if best_model is None:
        raise FloatingPointError("Pseudo-SR closure produced no valid checkpoint")
    model.load_state_dict(best_model)
    return model.eval().requires_grad_(False), int(best_epoch), float(best)


def _pseudo_sr_closure(train_z, train_mass, val_z, val_mass, *, settings, seed, device):
    """Test q_phi interpolation in three artificial gaps on each side of the SR.

    Each probe masks one localized mass window from both correction-training and
    checkpoint-selection rows, fits a fresh auxiliary q_phi, and evaluates only
    the held-out validation rows in that window.  The six windows are defined by
    mass quantiles and use no truth labels.

    A sideband passes when all three windows are evaluable, at least two have a
    positive Gaussian-relative NLL gain, their event-weighted mean gain is
    positive, and no individual window is significantly worse than Gaussian at
    the configured compatibility threshold.  Both sidebands must pass.
    """
    all_mass = np.concatenate((train_mass, val_mass)).astype(np.float64, copy=False)
    reports = []
    sides = (
        ("lower", all_mass < 3.3, lambda x: x < 3.3),
        ("upper", all_mass > 3.7, lambda x: x > 3.7),
    )
    position_names = {
        "lower": ("outer", "middle", "inner"),
        "upper": ("inner", "middle", "outer"),
    }
    windows_per_side = len(PSEUDO_WINDOW_QUANTILES)

    for side_index, (side_name, combined_side, side_selector) in enumerate(sides):
        values = all_mass[combined_side]
        minimum_side_events = 4 * MIN_PSEUDO_WINDOW_EVENTS
        if len(values) < minimum_side_events:
            for window_index, quantiles in enumerate(PSEUDO_WINDOW_QUANTILES):
                reports.append(dict(
                    side=side_name, window=window_index + 1,
                    position=position_names[side_name][window_index],
                    quantiles=list(quantiles), status="insufficient_events",
                    events=int(len(values)),
                ))
            continue

        train_side = side_selector(train_mass)
        val_side = side_selector(val_mass)
        for window_index, quantiles in enumerate(PSEUDO_WINDOW_QUANTILES):
            low, high = (float(x) for x in np.quantile(values, quantiles))
            train_gap = train_side & (train_mass >= low) & (train_mass <= high)
            val_gap = val_side & (val_mass >= low) & (val_mass <= high)
            train_keep = ~train_gap
            val_keep = ~val_gap
            same_side_left = int(np.sum(train_side & (train_mass < low)))
            same_side_right = int(np.sum(train_side & (train_mass > high)))
            counts = dict(
                train_gap=int(train_gap.sum()), validation_gap=int(val_gap.sum()),
                train_outside=int(train_keep.sum()), validation_outside=int(val_keep.sum()),
                same_side_left=same_side_left, same_side_right=same_side_right,
            )
            base = dict(
                side=side_name, window=window_index + 1,
                position=position_names[side_name][window_index],
                quantiles=list(quantiles), bounds=[low, high],
            )
            if (counts["validation_gap"] < MIN_PSEUDO_WINDOW_EVENTS
                    or counts["train_outside"] < MIN_PSEUDO_WINDOW_EVENTS
                    or counts["validation_outside"] < MIN_PSEUDO_WINDOW_EVENTS
                    or same_side_left < MIN_PSEUDO_WINDOW_EVENTS
                    or same_side_right < MIN_PSEUDO_WINDOW_EVENTS):
                reports.append(dict(**base, status="insufficient_events", **counts))
                continue

            probe_index = side_index * windows_per_side + window_index
            probe, selected_epoch, selection_nll = _fit_probe(
                train_z[train_keep], train_mass[train_keep],
                val_z[val_keep], val_mass[val_keep],
                settings=settings, seed=(int(seed) + 3000 + probe_index) % 2**32,
                device=device,
            )
            metrics = _validation_metrics(probe, val_z[val_gap], val_mass[val_gap], device)
            improvement = metrics["improvement"]
            standard_error = metrics["improvement_standard_error"]
            positive = improvement > MIN_VALIDATION_IMPROVEMENT
            if positive:
                compatible = True
            elif standard_error is None or not np.isfinite(standard_error) or standard_error <= 0:
                compatible = False
            else:
                compatible = (improvement + PSEUDO_COMPATIBILITY_SIGMA * standard_error
                              >= MIN_VALIDATION_IMPROVEMENT)
            status = "passed" if positive else ("compatible" if compatible else "failed")
            reports.append(dict(
                **base, status=status, positive=bool(positive),
                gaussian_compatible=bool(compatible), selected_epoch=selected_epoch,
                selection_nll=selection_nll, **counts, **metrics,
            ))
            emit_message(
                f"Background pseudo-SR closure {side_name} {base['position']} "
                f"[{window_index+1}/{windows_per_side}]: improvement={improvement:.6g} "
                f"({status})"
            )

    side_reports = []
    for side_name, _, _ in sides:
        windows = [r for r in reports if r["side"] == side_name]
        evaluable = [r for r in windows if r.get("status") != "insufficient_events"]
        positive_count = sum(bool(r.get("positive", False)) for r in evaluable)
        incompatible_count = sum(not bool(r.get("gaussian_compatible", False)) for r in evaluable)
        if evaluable:
            weights = np.asarray([r["events"] for r in evaluable], dtype=np.float64)
            gains = np.asarray([r["improvement"] for r in evaluable], dtype=np.float64)
            weighted_improvement = float(np.average(gains, weights=weights))
        else:
            weighted_improvement = None
        side_passed = (
            len(evaluable) == windows_per_side
            and positive_count >= MIN_POSITIVE_WINDOWS_PER_SIDE
            and incompatible_count == 0
            and weighted_improvement is not None
            and weighted_improvement > MIN_VALIDATION_IMPROVEMENT
        )
        side_reports.append(dict(
            side=side_name, status="passed" if side_passed else "failed",
            windows_evaluable=len(evaluable), windows_total=windows_per_side,
            positive_windows=positive_count,
            minimum_positive_windows=MIN_POSITIVE_WINDOWS_PER_SIDE,
            significantly_worse_windows=incompatible_count,
            event_weighted_improvement=weighted_improvement,
        ))
        emit_message(
            f"Background pseudo-SR closure {side_name} summary: "
            f"positive={positive_count}/{windows_per_side}, "
            f"significantly_worse={incompatible_count}, "
            f"weighted_improvement={weighted_improvement if weighted_improvement is not None else 'n/a'} "
            f"({'passed' if side_passed else 'failed'})"
        )

    passed = len(side_reports) == 2 and all(r["status"] == "passed" for r in side_reports)
    return dict(
        status="passed" if passed else "failed",
        protocol="masked_sideband_multiwindow_interpolation_v2",
        scope=CLOSURE_SCOPE,
        upstream_map_refitted=False,
        evaluation_role="correction_val",
        reserved_closure_role_used=False,
        evaluated_events=sum(int(r.get("events", 0)) for r in reports if "improvement" in r),
        evaluated_windows=sum("improvement" in r for r in reports),
        truth_labels_used=False,
        gap_quantiles=[list(x) for x in PSEUDO_WINDOW_QUANTILES],
        windows_per_side=windows_per_side,
        minimum_gap_events=MIN_PSEUDO_WINDOW_EVENTS,
        minimum_positive_windows_per_side=MIN_POSITIVE_WINDOWS_PER_SIDE,
        compatibility_sigma=PSEUDO_COMPATIBILITY_SIGMA,
        criterion=(
            "both sidebands must pass; per side all three windows must be evaluable, "
            "at least two must improve on Gaussian, the event-weighted mean improvement "
            "must be positive, and no window may be significantly worse than Gaussian"
        ),
        sides=side_reports,
        windows=reports,
    )


def _contract(settings, train_z, train_mass, val_z, val_mass, seed, device,
              closure_z=None, closure_mass=None):
    return dict(
        schema=4,
        scientific_version=SCIENTIFIC_VERSION,
        protocol=PROTOCOL,
        mode=MODE,
        epochs=_qphi_options(settings)[0],
        mass_bins=_qphi_options(settings)[1],
        activation_gate=dict(
            min_validation_improvement=MIN_VALIDATION_IMPROVEMENT,
            pseudo_window_quantiles=[list(x) for x in PSEUDO_WINDOW_QUANTILES],
            minimum_pseudo_window_events=MIN_PSEUDO_WINDOW_EVENTS,
            minimum_positive_windows_per_side=MIN_POSITIVE_WINDOWS_PER_SIDE,
            compatibility_sigma=PSEUDO_COMPATIBILITY_SIGMA,
        ),
        seed=int(seed),
        device=str(device),
        flow=settings["flow"],
        learning_rate=settings["training"]["learning_rate"],
        hashes={
            "train_z": digest(train_z), "train_mass": digest(train_mass),
            "validation_z": digest(val_z), "validation_mass": digest(val_mass),
            "independent_closure_z": None if closure_z is None else digest(closure_z),
            "independent_closure_mass": None if closure_mass is None else digest(closure_mass),
        },
    )


def train(directory, train_z, train_mass, val_z, val_mass, *, settings, seed, device,
          closure_z=None, closure_mass=None):
    """Fit q_phi, apply selection gates, and audit reserved sideband rows."""
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
    if (closure_z is None) != (closure_mass is None):
        raise ValueError("Provide both reserved closure latents and masses, or neither")
    if closure_z is not None:
        closure_z = np.ascontiguousarray(closure_z, dtype=np.float32)
        closure_mass = np.ascontiguousarray(closure_mass, dtype=np.float32)
        if (closure_z.ndim != 2 or closure_z.shape[1:] != train_z.shape[1:]
                or len(closure_z) < 2 or closure_mass.shape != (len(closure_z),)
                or not np.isfinite(closure_z).all() or not np.isfinite(closure_mass).all()
                or ((closure_mass > 3.3) & (closure_mass < 3.7)).any()):
            raise ValueError("Invalid reserved sideband closure inputs")
    epochs, mass_bins = _qphi_options(settings)
    contract = _contract(settings, train_z, train_mass, val_z, val_mass, seed, device,
                         closure_z, closure_mass)
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
    for epoch in range(start, epochs):
        generator = torch.Generator().manual_seed(int(seed) + 11000 + epoch)
        balanced = _balanced_indices(train_mass, mass_bins, int(seed) + 11500 + epoch)
        epoch_dataset = dataset if balanced is None else TensorDataset(zt[balanced], mt[balanced])
        loader = DataLoader(epoch_dataset, batch_size=512, shuffle=balanced is None, generator=generator)
        model.train(); total = 0.0; trained_events = 0
        for z, m in loader:
            z, m = z.to(device), m.to(device)
            optimizer.zero_grad()
            loss = -log_prob(model, z, m).mean()
            require_finite(loss, "Latent-background correction loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach()) * len(z); trained_events += len(z)
        model.eval()
        with torch.no_grad():
            validation_nll = -float(log_prob(model, zv, mv).double().mean().cpu())
            gaussian_nll = -float(_gaussian_log_prob(zv).double().mean().cpu())
        row = dict(epoch=epoch, train_nll=total / trained_events, validation_nll=validation_nll,
                   gaussian_validation_nll=gaussian_nll, improvement=gaussian_nll-validation_nll)
        history.append(row)
        require_finite(validation_nll, "Background-correction checkpoint validation NLL")
        if (validation_nll, epoch) < (best, best_epoch if best_epoch is not None else math.inf):
            best, best_epoch, best_model = validation_nll, epoch, deepcopy(model.state_dict())
        if persist_boundary(epoch, epochs):
            state = dict(contract=contract, epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                         history=history, best=best, best_epoch=best_epoch, best_model=best_model, rng=rng_state())
            latest.parent.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(latest, state)
            write_json(directory / "history.json", history)
        emit_message(f"Background correction {epoch+1}/{epochs}: validation NLL={validation_nll:.6g}")

    if best_model is None:
        raise FloatingPointError("Background correction produced no valid checkpoint")
    selected = {"model": best_model, "selected_epoch": best_epoch, "contract": contract}
    atomic_torch_save(directory / "model.pt", selected)
    model.load_state_dict(best_model); model.eval().requires_grad_(False)
    validation = _validation_metrics(model, val_z, val_mass, device)
    validation_passed = validation["improvement"] > MIN_VALIDATION_IMPROVEMENT
    if validation_passed:
        closure = _pseudo_sr_closure(train_z, train_mass, val_z, val_mass,
                                     settings=settings, seed=(int(seed)+40000) % 2**32, device=device)
    else:
        closure = dict(status="not_run", reason="correction-selection validation did not beat Gaussian",
                       truth_labels_used=False, protocol="masked_sideband_multiwindow_interpolation_v2",
                       scope=CLOSURE_SCOPE, upstream_map_refitted=False,
                       evaluation_role="correction_val", reserved_closure_role_used=False,
                       evaluated_events=0, evaluated_windows=0)
    active = validation_passed and closure.get("status") == "passed"
    reasons = []
    if not validation_passed:
        reasons.append("q_phi did not improve reserved validation NLL over Gaussian")
    if validation_passed and closure.get("status") != "passed":
        reasons.append("q_phi failed masked-sideband pseudo-SR interpolation closure")
    independent = independent_closure_diagnostic(model, closure_z, closure_mass, device,
                                                selected_denominator="q_phi(z|m)" if active else "standard_normal(z)")
    decision = dict(
        mode=MODE, protocol=PROTOCOL, selected_epoch=int(best_epoch), epochs=epochs,
        status="activated" if active else "gaussian_fallback", active=bool(active),
        criterion="lowest reserved correction-validation NLL, then positive Gaussian-relative validation gain and multi-window pseudo-SR closure",
        validation_gate={**validation, "minimum_improvement": MIN_VALIDATION_IMPROVEMENT,
                         "status": "passed" if validation_passed else "failed"},
        pseudo_sr_closure=closure,
        independent_closure_diagnostic=independent,
        validation_scope="correction-validation is also used for checkpoint selection",
        full_search_closure_status="not_evaluated",
        truth_labels_used=False, training_region="reserved sidebands",
        denominator="q_phi(z|m)" if active else "standard_normal(z)",
        fallback_reason="; ".join(reasons) if reasons else None,
    )
    write_json(directory / "selection.json", decision)
    emit_message("Background correction activated" if active else
                 "Background correction failed safety gates; using Gaussian denominator", level=0)
    return dict(requested_mode=MODE, active=bool(active), descriptor=descriptor(directory) if active else None,
                selection=decision)


def independent_closure_diagnostic(model, latent, mass, device, *, selected_denominator):
    """No-retuning audit of the already selected q_phi candidate on reserved rows."""
    info = dict(role="closure", scope="fixed_map_sideband_density_diagnostic_not_full_search",
                used_for_training=False, used_for_checkpoint_selection=False,
                used_for_activation_gate=False, truth_labels_used=False,
                selected_denominator=selected_denominator,
                interpretation="Gaussian-relative NLL diagnostic; not a tail-selection or discovery test")
    if latent is None:
        return dict(info, status="unavailable", events=0, reason="reserved closure role not supplied")
    metrics = _validation_metrics(model, latent, mass, device)
    sides = {}
    for name, keep in (("lower", mass <= 3.3), ("upper", mass >= 3.7)):
        if keep.any():
            sides[name] = _validation_metrics(model, latent[keep], mass[keep], device)
    return dict(info, status="evaluated", events=len(latent), candidate_qphi=metrics, sides=sides)


def descriptor(directory):
    directory = Path(directory)
    state = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
    contract = state["contract"]
    if contract.get("mode") != MODE or type(contract.get("epochs")) is not int or contract["epochs"] < 10:
        raise ValueError("Invalid bgcorr_40_reguide model")
    selection = json.loads((directory / "selection.json").read_text())
    if selection.get("status") != "activated" or not selection.get("active"):
        raise ValueError("Inactive background correction cannot be used as the RIDDLE denominator")
    return dict(mode=MODE, protocol=PROTOCOL, epochs=int(contract["epochs"]),
                selected_epoch=int(state["selected_epoch"]), model_path=str((directory / "model.pt").resolve()),
                model_sha256=file_digest(directory / "model.pt"), contract=contract,
                validation_gate=selection["validation_gate"], pseudo_sr_closure=selection["pseudo_sr_closure"],
                independent_closure_diagnostic=selection.get("independent_closure_diagnostic"))


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
        atomic_torch_save(path, payload)
    return file_digest(path)


def load_local(directory, settings, features, device):
    path = Path(directory) / "background_correction.pt"
    if not path.exists():
        return None
    saved = torch.load(path, map_location=device, weights_only=False)
    readable = (PROTOCOL, "shared_sideband_qphi_reguide_v3_multiwindow_closure")
    if saved.get("mode") != MODE or saved.get("protocol") not in readable:
        raise ValueError("Invalid fit-local background correction")
    model = _model(settings, features, device)
    model.load_state_dict(saved["model"])
    return model.eval().requires_grad_(False), saved
