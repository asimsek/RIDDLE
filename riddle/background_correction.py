"""Latent background correction used by the bgcorr_40_reguide RIDDLE family."""
from concurrent.futures import ProcessPoolExecutor, as_completed
from copy import deepcopy
from pathlib import Path
import json
import shutil
import math
import multiprocessing as mp
import time

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .integrity import SCIENTIFIC_VERSION, RIDDLE_BENCHMARK_SCIENTIFIC_VERSION, require_finite
from .model import build_signal_flow, match_background
from .storage import (atomic_torch_save, digest, file_digest, rng_state, restore_rng,
                      write_json, persist_boundary)
from .worker_progress import emit_message
from .options import feature_options
from .settings import sr_closure_mode

MODE = "bgcorr_40_reguide"
EPOCHS = 40
PROTOCOL = "shared_sideband_qphi_balanced_validation_compatible_closure_reguide_decoupled_width"
ORACLE_PROTOCOL = "riddle_oracle_sr_qphi_v2_gaussian_gate"
READABLE_PROTOCOLS = (
    PROTOCOL, ORACLE_PROTOCOL,
    "shared_sideband_qphi_balanced_validation_uncertainty_gated_mean_supported_closure_reguide_decoupled_width",
    "shared_sideband_qphi_reguide_v3_multiwindow_closure",
)
CLOSURE_SCOPE = "correction_only_interpolation_on_fixed_upstream_map; not_full_search_closure"
MIN_VALIDATION_IMPROVEMENT = 0.0
# Keep training support on both sides of each gap.

PSEUDO_WINDOW_QUANTILES = ((0.15, 0.30), (0.425, 0.575), (0.70, 0.85))
MIN_PSEUDO_WINDOW_EVENTS = 30
PSEUDO_COMPATIBILITY_SIGMA = 1.96


def enabled(settings):
    return (settings.get("background_correction", "none") == MODE
            and sr_closure_mode(settings.get("sr_closure", "auto")) != "off")


def _training_batches(dataset, *, shuffle, generator):
    if len(dataset) < 2:
        raise ValueError("Background-correction training requires at least two events")
    loader = DataLoader(dataset, batch_size=512, shuffle=shuffle, generator=generator)
    merge_at = len(loader) - 2 if len(dataset) % loader.batch_size == 1 else -1
    batches = iter(loader)
    for index, batch in enumerate(batches):
        if index == merge_at:
            batch = tuple(torch.cat((values, tail), dim=0)
                          for values, tail in zip(batch, next(batches)))
        yield batch


def _qphi_options(settings):
    options = feature_options(settings)
    epochs = int(options.get("qphi_epochs", EPOCHS))
    bins = int(options.get("qphi_mass_bins", 1))
    if bins > 1 and bins % 2:
        raise ValueError("qphi_mass_bins must be 1 or an even number")
    return epochs, bins


def _mass_strata(mass, bins):
    if bins <= 1:
        return [np.arange(len(mass), dtype=np.int64)]
    mass = np.asarray(mass)
    groups = []
    per_side = bins // 2
    for mask in (mass < 3.3, mass > 3.7):
        indices = np.flatnonzero(mask)
        if len(indices) < per_side:
            raise ValueError("Insufficient sideband events for mass-balanced q_phi strata")
        ordered = indices[np.argsort(mass[indices], kind="stable")]
        groups.extend(np.array_split(ordered, per_side))
    if len(groups) != bins or any(len(g) == 0 for g in groups):
        raise ValueError("Could not construct balanced q_phi mass strata")
    return [np.asarray(group, dtype=np.int64) for group in groups]


def _balanced_indices(mass, bins, seed):
    """Equalize lower/upper sidebands and mass-quantile strata without labels."""
    if bins <= 1:
        return None
    rng = np.random.default_rng(int(seed))
    groups = _mass_strata(mass, bins)
    target = int(math.ceil(len(mass) / bins))
    chosen = [rng.choice(g, size=target, replace=len(g) < target) for g in groups]
    merged = np.concatenate(chosen)
    rng.shuffle(merged)
    return merged.astype(np.int64, copy=False)


def _equal_stratum_mean(values, mass, bins):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(mass),) or not np.isfinite(values).all():
        raise ValueError("Invalid q_phi validation values")
    groups = _mass_strata(mass, bins)
    return float(np.mean([values[group].mean() for group in groups]))


def _qphi_flow(settings):
    flow = deepcopy(settings["flow"])
    flow["hidden_features"] = int(feature_options(settings)["qphi_hidden_features"])
    return flow


def _model(settings, dimensions, device):
    cfg = deepcopy(settings)
    cfg["mass_conditioning"] = True
    cfg["enhancements"] = deepcopy(cfg.get("enhancements", {}))
    cfg["enhancements"]["score_flow"] = False
    cfg["flow"] = _qphi_flow(settings)
    model = build_signal_flow(device, features=dimensions + 1, settings=cfg)
    match_background(model)
    return model


def log_prob(model, latent, context):
    """log q_phi(z|m), with context already equal to (mjj-3.5)/0.2."""
    return model.log_prob(latent, context=context.reshape(-1, 1))


def sample(model, context, dimensions, seed, device, batch_size=None):
    """Sample q_phi(z|m) at matched mass contexts without perturbing caller RNG."""
    context = torch.as_tensor(context, dtype=torch.float32).reshape(-1, 1)
    target = torch.device(device)
    if target.type not in ("cpu", "cuda"):
        raise ValueError("Corrected-background sampling supports CPU and CUDA devices")
    if batch_size is None:
        index = (torch.cuda.current_device() if target.index is None else target.index) if target.type == "cuda" else None
        devices = [] if index is None else [index]
        with torch.random.fork_rng(devices=devices), torch.no_grad():
            torch.random.default_generator.manual_seed(int(seed))
            if index is not None:
                torch.cuda.default_generators[index].manual_seed(int(seed))
            values = model.sample(1, context=context.to(target))
        if values.ndim == 3 and values.shape[1] == 1:
            values = values[:, 0, :]
        elif values.ndim == 3 and values.shape[0] == 1:
            values = values[0]
        if tuple(values.shape) != (len(context), dimensions):
            raise ValueError(f"Unexpected conditional background sample shape {tuple(values.shape)}")
        require_finite(values, "Corrected-background samples")
        return values
    if type(batch_size) is not int or batch_size < 1:
        raise ValueError("Corrected-background sampling batch size must be a positive integer")
    index = (torch.cuda.current_device() if target.index is None else target.index) if target.type == "cuda" else None
    devices = [] if index is None else [index]
    parts = []
    size = min(int(batch_size), max(1, len(context)))
    offset = 0
    with torch.random.fork_rng(devices=devices), torch.no_grad():
        torch.random.default_generator.manual_seed(int(seed))
        if index is not None:
            torch.cuda.default_generators[index].manual_seed(int(seed))
        noise = torch.randn((len(context), dimensions), dtype=torch.float32, device=target)
        while offset < len(context):
            stop = min(offset + size, len(context))
            try:
                c = context[offset:stop].to(target)
                embedded = model._embedding_net(c)
                values, _ = model._transform.inverse(noise[offset:stop], context=embedded)
                if tuple(values.shape) != (stop - offset, dimensions):
                    raise ValueError(f"Unexpected conditional background sample shape {tuple(values.shape)}")
                require_finite(values, "Corrected-background samples")
                parts.append(values.detach().cpu())
                offset = stop
            except torch.cuda.OutOfMemoryError:
                if target.type != "cuda" or size <= 256:
                    raise
                torch.cuda.empty_cache()
                size = max(256, size // 2)
    return torch.cat(parts, dim=0)


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


def _gaussian_gate(metrics):
    improvement = float(metrics["improvement"])
    standard_error = metrics.get("improvement_standard_error")
    positive = improvement > MIN_VALIDATION_IMPROVEMENT
    if positive:
        compatible = True
    elif standard_error is None or not np.isfinite(standard_error) or standard_error <= 0:
        compatible = False
    else:
        compatible = (improvement + PSEUDO_COMPATIBILITY_SIGMA * float(standard_error)
                      >= MIN_VALIDATION_IMPROVEMENT)
    status = "passed" if positive else ("compatible" if compatible else "failed")
    return dict(
        **metrics,
        minimum_improvement=MIN_VALIDATION_IMPROVEMENT,
        compatibility_sigma=PSEUDO_COMPATIBILITY_SIGMA,
        positive=bool(positive),
        gaussian_compatible=bool(compatible),
        status=status,
    )


def _validation_side_gates(model, latent, mass, device):
    sides = {}
    for name, keep in (("lower", mass <= 3.3), ("upper", mass >= 3.7)):
        if not np.any(keep):
            sides[name] = dict(status="failed", events=0, gaussian_compatible=False,
                               positive=False, reason="no events")
        else:
            sides[name] = _gaussian_gate(_validation_metrics(model, latent[keep], mass[keep], device))
    return sides


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
    with torch.no_grad():
        gaussian_log_prob = _gaussian_log_prob(zv).double().cpu().numpy()
    best_any, best_any_epoch, best_any_model = float("inf"), None, None
    best_eligible, best_eligible_epoch, best_eligible_model = float("inf"), None, None
    for epoch in range(epochs):
        generator = torch.Generator().manual_seed(int(seed) + 21000 + epoch)
        balanced = _balanced_indices(train_mass, mass_bins, int(seed) + 21500 + epoch)
        epoch_dataset = dataset if balanced is None else TensorDataset(dataset.tensors[0][balanced], dataset.tensors[1][balanced])
        batches = _training_batches(epoch_dataset, shuffle=balanced is None, generator=generator)
        model.train()
        for z, m in batches:
            z, m = z.to(device), m.to(device)
            optimizer.zero_grad()
            loss = -log_prob(model, z, m).mean()
            require_finite(loss, "Pseudo-SR background-correction loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
        model.eval()
        with torch.no_grad():
            validation_log_prob = log_prob(model, zv, mv).double().cpu().numpy()
        validation_nll = -float(validation_log_prob.mean())
        selection_nll = -_equal_stratum_mean(validation_log_prob, val_mass, mass_bins)
        gain = validation_log_prob - gaussian_log_prob
        gate = _gaussian_gate(dict(
            events=int(len(gain)), qphi_nll=validation_nll,
            gaussian_nll=-float(gaussian_log_prob.mean()),
            improvement=float(gain.mean()),
            improvement_standard_error=(float(gain.std(ddof=1) / math.sqrt(len(gain)))
                                        if len(gain) > 1 else None),
        ))
        require_finite(validation_nll, "Background-correction validation NLL")
        require_finite(selection_nll, "Background-correction checkpoint selection NLL")
        if (selection_nll, epoch) < (best_any, best_any_epoch if best_any_epoch is not None else math.inf):
            best_any, best_any_epoch, best_any_model = selection_nll, epoch, deepcopy(model.state_dict())
        if gate["gaussian_compatible"] and (
                (selection_nll, epoch)
                < (best_eligible, best_eligible_epoch if best_eligible_epoch is not None else math.inf)):
            best_eligible = selection_nll
            best_eligible_epoch = epoch
            best_eligible_model = deepcopy(model.state_dict())
    use_eligible = best_eligible_model is not None
    best = best_eligible if use_eligible else best_any
    best_epoch = best_eligible_epoch if use_eligible else best_any_epoch
    best_model = best_eligible_model if use_eligible else best_any_model
    if best_model is None:
        raise FloatingPointError("Pseudo-SR closure produced no valid checkpoint")
    model.load_state_dict(best_model)
    return model.eval().requires_grad_(False), int(best_epoch), float(best), bool(use_eligible)


def _pseudo_probe_worker(task):
    threads = task.get("torch_threads")
    if threads is not None:
        torch.set_num_threads(int(threads))
    probe, selected_epoch, selection_nll, checkpoint_eligible = _fit_probe(
        task["train_z"], task["train_mass"], task["val_z"], task["val_mass"],
        settings=task["settings"], seed=task["seed"], device=task["device"],
    )
    metrics = _validation_metrics(probe, task["gap_z"], task["gap_mass"], task["device"])
    gate = _gaussian_gate(metrics)
    return dict(
        **task["base"], selected_epoch=selected_epoch,
        selection_nll=selection_nll,
        checkpoint_gaussian_compatible=bool(checkpoint_eligible),
        **task["counts"], **gate,
    )


def _pseudo_sr_closure(train_z, train_mass, val_z, val_mass, *, settings, seed, device):
    """Require all six held-out sideband gaps to be Gaussian-compatible."""
    all_mass = np.concatenate((train_mass, val_mass)).astype(np.float64, copy=False)
    reports = []
    tasks = []
    sides = (
        ("lower", all_mass < 3.3, lambda x: x < 3.3),
        ("upper", all_mass > 3.7, lambda x: x > 3.7),
    )
    position_names = {
        "lower": ("outer", "middle", "inner"),
        "upper": ("inner", "middle", "outer"),
    }
    windows_per_side = len(PSEUDO_WINDOW_QUANTILES)
    parallelism = int(feature_options(settings).get("pseudo_sr_parallel_probes", 3))

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
            tasks.append(dict(
                base=base,
                counts=counts,
                train_z=np.ascontiguousarray(train_z[train_keep], dtype=np.float32),
                train_mass=np.ascontiguousarray(train_mass[train_keep], dtype=np.float32),
                val_z=np.ascontiguousarray(val_z[val_keep], dtype=np.float32),
                val_mass=np.ascontiguousarray(val_mass[val_keep], dtype=np.float32),
                gap_z=np.ascontiguousarray(val_z[val_gap], dtype=np.float32),
                gap_mass=np.ascontiguousarray(val_mass[val_gap], dtype=np.float32),
                settings=settings,
                seed=(int(seed) + 3000 + probe_index) % 2**32,
                device=str(device),
                torch_threads=None,
            ))

    if tasks:
        workers = min(parallelism, len(tasks))
        if workers == 1:
            completed = [_pseudo_probe_worker(task) for task in tasks]
            for report in completed:
                reports.append(report)
                emit_message(
                    f"Background pseudo-SR closure {report['side']} {report['position']} "
                    f"[{report['window']}/{windows_per_side}]: improvement={report['improvement']:.6g} "
                    f"({report['status']})"
                )
        else:
            child_threads = max(1, int(torch.get_num_threads()) // workers)
            for task in tasks:
                task["torch_threads"] = child_threads
            context = mp.get_context("spawn")
            with ProcessPoolExecutor(max_workers=workers, mp_context=context) as executor:
                futures = [executor.submit(_pseudo_probe_worker, task) for task in tasks]
                for future in as_completed(futures):
                    report = future.result()
                    reports.append(report)
                    emit_message(
                        f"Background pseudo-SR closure {report['side']} {report['position']} "
                        f"[{report['window']}/{windows_per_side}]: improvement={report['improvement']:.6g} "
                        f"({report['status']})"
                    )

    side_order = {"lower": 0, "upper": 1}
    reports.sort(key=lambda r: (side_order[r["side"]], int(r["window"])))
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
            and incompatible_count == 0
            and weighted_improvement is not None
            and np.isfinite(weighted_improvement)
        )
        side_reports.append(dict(
            side=side_name, status="passed" if side_passed else "failed",
            windows_evaluable=len(evaluable), windows_total=windows_per_side,
            positive_windows=positive_count,
            minimum_positive_windows=0,
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
        protocol="masked_sideband_multiwindow_interpolation_v4_compatibility",
        scope=CLOSURE_SCOPE,
        upstream_map_refitted=False,
        evaluation_role="correction_val",
        reserved_closure_role_used=False,
        evaluated_events=sum(int(r.get("events", 0)) for r in reports if "improvement" in r),
        evaluated_windows=sum("improvement" in r for r in reports),
        truth_labels_used=False,
        gap_quantiles=[list(x) for x in PSEUDO_WINDOW_QUANTILES],
        windows_per_side=windows_per_side,
        parallel_probes=parallelism,
        minimum_gap_events=MIN_PSEUDO_WINDOW_EVENTS,
        minimum_positive_windows_per_side=0,
        positive_weighted_mean_required=False,
        compatibility_sigma=PSEUDO_COMPATIBILITY_SIGMA,
        criterion=(
            "both sidebands must pass; all three windows per side must be evaluable "
            "and Gaussian-compatible (passed or compatible); positive gains are not required"
        ),
        sides=side_reports,
        windows=reports,
    )

def _contract(settings, train_z, train_mass, val_z, val_mass, seed, device,
              closure_z=None, closure_mass=None):
    policy = sr_closure_mode(settings.get("sr_closure", "auto"))
    return dict(
        schema=9,
        scientific_version=SCIENTIFIC_VERSION,
        protocol=PROTOCOL,
        mode=MODE,
        activation_policy=policy,
        epochs=_qphi_options(settings)[0],
        mass_bins=_qphi_options(settings)[1],
        activation_gate=dict(
            min_validation_improvement=MIN_VALIDATION_IMPROVEMENT,
            pseudo_window_quantiles=[list(x) for x in PSEUDO_WINDOW_QUANTILES],
            minimum_pseudo_window_events=MIN_PSEUDO_WINDOW_EVENTS,
            minimum_positive_windows_per_side=0,
            compatibility_sigma=PSEUDO_COMPATIBILITY_SIGMA,
            validation_requires_side_compatibility=policy == "auto",
            checkpoint_requires_natural_validation_compatibility=True,
            pseudo_sr_always_evaluated=policy == "auto",
            compatible_windows_allowed=True,
            positive_weighted_mean_required=False,
        ),
        seed=int(seed),
        device=str(device),
        flow=_qphi_flow(settings),
        learning_rate=settings["training"]["learning_rate"],
        hashes={
            "train_z": digest(train_z), "train_mass": digest(train_mass),
            "validation_z": digest(val_z), "validation_mass": digest(val_mass),
            "independent_closure_z": None if closure_z is None else digest(closure_z),
            "independent_closure_mass": None if closure_mass is None else digest(closure_mass),
        },
    )


def _check_mapped_contract(previous, current, allow_device_change, history_path=None):
    from .resume import check_derived_contract

    return check_derived_contract(
        previous, current, allow_device_change=allow_device_change,
        hash_paths={("hashes", name) for name in
                    ("train_z", "validation_z", "closure_z", "independent_closure_z")},
        device_paths={("device",)}, history_path=history_path,
    )


def train(directory, train_z, train_mass, val_z, val_mass, *, settings, seed, device,
          closure_z=None, closure_mass=None, allow_device_change=False):
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
    if contract_path.exists():
        previous = json.loads(contract_path.read_text())
        changes = _check_mapped_contract(previous, contract, allow_device_change,
                                         directory / ".resume/device_history.json")
        if changes and (directory / "model.pt").is_file() and (directory / "selection.json").is_file():
            state = torch.load(directory / "model.pt", map_location="cpu", weights_only=False)
            if state.get("contract") != previous:
                raise ValueError("Completed background-correction contract changed")
            decision = json.loads((directory / "selection.json").read_text())
            active = decision.get("status") == "activated" and bool(decision.get("active"))
            emit_message("Reuse completed background correction after approved GPU migration", level=0)
            return dict(requested_mode=MODE, active=active,
                        descriptor=descriptor(directory) if active else None, selection=decision)
    write_json(contract_path, contract)

    policy = contract["activation_policy"]
    if policy == "off":
        decision = dict(
            mode=MODE, protocol=PROTOCOL, activation_policy=policy,
            status="disabled", active=False, epochs=0, selected_epoch=None,
            denominator="standard_normal(z)", truth_labels_used=False,
            validation_gate={"status": "not_evaluated"},
            pseudo_sr_closure={"status": "not_evaluated", "reason": "disabled_by_configuration"},
            fallback_reason=None,
        )
        write_json(directory / "selection.json", decision)
        emit_message("Background correction disabled by sr_closure=off; using Gaussian denominator", level=0)
        return dict(requested_mode=MODE, active=False, descriptor=None, selection=decision)

    torch.manual_seed(int(seed)); np.random.seed(int(seed) % 2**32)
    model = _model(settings, train_z.shape[1], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["training"]["learning_rate"], weight_decay=1e-4)
    history, start = [], 0
    best_any, best_any_epoch, best_any_model = float("inf"), None, None
    best_eligible, best_eligible_epoch, best_eligible_model = float("inf"), None, None
    latest = directory / ".resume/latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        _check_mapped_contract(state.get("contract"), contract, allow_device_change)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, start = state["history"], state["epoch"] + 1
        best_any = state["best_any"]
        best_any_epoch = state["best_any_epoch"]
        best_any_model = state["best_any_model"]
        best_eligible = state["best_eligible"]
        best_eligible_epoch = state["best_eligible_epoch"]
        best_eligible_model = state["best_eligible_model"]
        restore_rng(state["rng"])

    zt = torch.from_numpy(train_z)
    mt = torch.from_numpy(((train_mass - 3.5) / 0.2).astype(np.float32))
    zv = torch.from_numpy(val_z).to(device)
    mv = torch.from_numpy(((val_mass - 3.5) / 0.2).astype(np.float32)).to(device)
    dataset = TensorDataset(zt, mt)
    epoch_seconds = []
    for epoch in range(start, epochs):
        epoch_started = time.monotonic()
        generator = torch.Generator().manual_seed(int(seed) + 11000 + epoch)
        balanced = _balanced_indices(train_mass, mass_bins, int(seed) + 11500 + epoch)
        epoch_dataset = dataset if balanced is None else TensorDataset(zt[balanced], mt[balanced])
        batches = _training_batches(epoch_dataset, shuffle=balanced is None, generator=generator)
        model.train(); total = 0.0; trained_events = 0
        for z, m in batches:
            z, m = z.to(device), m.to(device)
            optimizer.zero_grad()
            loss = -log_prob(model, z, m).mean()
            require_finite(loss, "Latent-background correction loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step(); total += float(loss.detach()) * len(z); trained_events += len(z)
        model.eval()
        with torch.no_grad():
            validation_log_prob = log_prob(model, zv, mv).double().cpu().numpy()
            gaussian_log_prob = _gaussian_log_prob(zv).double().cpu().numpy()
        validation_nll = -float(validation_log_prob.mean())
        gaussian_nll = -float(gaussian_log_prob.mean())
        balanced_validation_nll = -_equal_stratum_mean(validation_log_prob, val_mass, mass_bins)
        balanced_gaussian_nll = -_equal_stratum_mean(gaussian_log_prob, val_mass, mass_bins)
        gain = validation_log_prob - gaussian_log_prob
        improvement = float(gain.mean())
        improvement_standard_error = (float(gain.std(ddof=1) / math.sqrt(len(gain)))
                                      if len(gain) > 1 else None)
        epoch_gate = _gaussian_gate(dict(
            events=int(len(gain)), qphi_nll=validation_nll, gaussian_nll=gaussian_nll,
            improvement=improvement, improvement_standard_error=improvement_standard_error,
        ))
        row = dict(epoch=epoch, train_nll=total / trained_events, validation_nll=validation_nll,
                   balanced_validation_nll=balanced_validation_nll,
                   gaussian_validation_nll=gaussian_nll,
                   balanced_gaussian_validation_nll=balanced_gaussian_nll,
                   improvement=improvement,
                   improvement_standard_error=improvement_standard_error,
                   gaussian_compatible=epoch_gate["gaussian_compatible"],
                   balanced_improvement=balanced_gaussian_nll-balanced_validation_nll)
        history.append(row)
        require_finite(validation_nll, "Background-correction validation NLL")
        require_finite(balanced_validation_nll, "Background-correction checkpoint selection NLL")
        if (balanced_validation_nll, epoch) < (best_any, best_any_epoch if best_any_epoch is not None else math.inf):
            best_any, best_any_epoch, best_any_model = balanced_validation_nll, epoch, deepcopy(model.state_dict())
        if epoch_gate["gaussian_compatible"] and (
                (balanced_validation_nll, epoch)
                < (best_eligible, best_eligible_epoch if best_eligible_epoch is not None else math.inf)):
            best_eligible = balanced_validation_nll
            best_eligible_epoch = epoch
            best_eligible_model = deepcopy(model.state_dict())
        epoch_seconds.append(time.monotonic() - epoch_started)
        if persist_boundary(epoch, epochs):
            state = dict(
                contract=contract, epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                history=history, best_any=best_any, best_any_epoch=best_any_epoch,
                best_any_model=best_any_model, best_eligible=best_eligible,
                best_eligible_epoch=best_eligible_epoch, best_eligible_model=best_eligible_model,
                rng=rng_state(),
            )
            latest.parent.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(latest, state)
            write_json(directory / "history.json", history)
        emit_message(
            f"Background correction {epoch+1}/{epochs}: balanced validation NLL={balanced_validation_nll:.6g}; "
            f"natural validation NLL={validation_nll:.6g}; natural gate={epoch_gate['status']}"
        )

    from .background_stage import record_timing
    record_timing(directory, "epochs", sum(epoch_seconds), epoch_seconds=epoch_seconds)
    use_eligible = best_eligible_model is not None
    best = best_eligible if use_eligible else best_any
    best_epoch = best_eligible_epoch if use_eligible else best_any_epoch
    best_model = best_eligible_model if use_eligible else best_any_model
    if best_model is None:
        raise FloatingPointError("Background correction produced no valid checkpoint")
    selected = {"model": best_model, "selected_epoch": best_epoch, "contract": contract}
    atomic_torch_save(directory / "model.pt", selected)
    model.load_state_dict(best_model); model.eval().requires_grad_(False)

    validation = _gaussian_gate(_validation_metrics(model, val_z, val_mass, device))
    validation_sides = _validation_side_gates(model, val_z, val_mass, device)
    overall_compatible = bool(validation["gaussian_compatible"])
    sides_compatible = all(v.get("gaussian_compatible", False) for v in validation_sides.values())
    validation_passed = overall_compatible and sides_compatible

    closure_started = time.monotonic()
    closure = _pseudo_sr_closure(
        train_z, train_mass, val_z, val_mass,
        settings=settings, seed=(int(seed)+40000) % 2**32, device=device,
    ) if policy == "auto" else dict(status="not_evaluated", reason="forced_on_by_configuration")

    record_timing(directory, "closure", time.monotonic() - closure_started)
    active = policy == "on" or (validation_passed and closure.get("status") == "passed")
    reasons = []
    if not overall_compatible:
        reasons.append("q_phi is significantly worse than Gaussian on reserved validation")
    failed_sides = [name for name, gate in validation_sides.items()
                    if not gate.get("gaussian_compatible", False)]
    if failed_sides:
        reasons.append("q_phi is significantly worse than Gaussian on validation sideband(s): "
                       + ", ".join(failed_sides))
    if policy == "auto" and closure.get("status") != "passed":
        reasons.append("q_phi failed masked-sideband pseudo-SR interpolation closure")
    independent = independent_closure_diagnostic(
        model, closure_z, closure_mass, device,
        selected_denominator="q_phi(z|m)" if active else "standard_normal(z)",
    )
    decision = dict(
        mode=MODE, protocol=PROTOCOL, selected_epoch=int(best_epoch), epochs=epochs,
        activation_policy=policy, activation_forced=policy == "on",
        status="activated" if active else "gaussian_fallback", active=bool(active),
        checkpoint_selection=dict(
            metric="equal_mass_stratum_validation_nll",
            eligibility="Gaussian-compatible natural validation",
            eligible_checkpoint_found=bool(use_eligible),
            mass_bins=mass_bins, selected_nll=float(best),
        ),
        criterion=(
            "best equal-mass-stratum checkpoint among Gaussian-compatible natural-validation epochs; "
            "selected checkpoint must be Gaussian-compatible globally and on each sideband, and both "
            "sidebands must have three evaluable, Gaussian-compatible pseudo-SR windows "
            "(passed or compatible; positive gains are not required)"
            if policy == "auto" else
            "sr_closure=on forces the selected finite q_phi checkpoint; validation is diagnostic and pseudo-SR probes are skipped"
        ),
        validation_gate=dict(
            **validation,
            sides=validation_sides,
            side_compatibility_required=True,
            activation_eligible=bool(validation_passed),
        ),
        pseudo_sr_closure=closure,
        independent_closure_diagnostic=independent,
        validation_scope=(
            "correction-validation selects checkpoints with equal mass-stratum weighting; "
            "Gaussian compatibility is evaluated with natural event weighting globally and per sideband"
        ),
        full_search_closure_status="not_evaluated",
        truth_labels_used=False, training_region="reserved sidebands",
        denominator="q_phi(z|m)" if active else "standard_normal(z)",
        fallback_reason="; ".join(reasons) if reasons and not active else None,
        validation_warnings=reasons if policy == "on" else [],
    )
    write_json(directory / "selection.json", decision)
    emit_message("Background correction forced on by sr_closure=on" if policy == "on" else
                 "Background correction activated" if active else
                 "Background correction failed safety gates; using Gaussian denominator", level=0)
    return dict(requested_mode=MODE, active=bool(active), descriptor=descriptor(directory) if active else None,
                selection=decision)


def _oracle_mass_strata(mass, bins):
    mass = np.asarray(mass)
    if mass.ndim != 1 or len(mass) < 2 or not np.isfinite(mass).all():
        raise ValueError("Invalid oracle-background mass sample")
    count = min(int(bins), len(mass))
    if count <= 1:
        return [np.arange(len(mass), dtype=np.int64)]
    ordered = np.argsort(mass, kind="stable")
    groups = [np.asarray(group, dtype=np.int64) for group in np.array_split(ordered, count)]
    if any(len(group) == 0 for group in groups):
        raise ValueError("Could not construct oracle-background mass strata")
    return groups


def _oracle_balanced_indices(mass, bins, seed):
    if bins <= 1:
        return None
    rng = np.random.default_rng(int(seed))
    groups = _oracle_mass_strata(mass, bins)
    target = int(math.ceil(len(mass) / len(groups)))
    chosen = [rng.choice(group, size=target, replace=len(group) < target) for group in groups]
    merged = np.concatenate(chosen)
    rng.shuffle(merged)
    return merged.astype(np.int64, copy=False)


def _oracle_equal_stratum_mean(values, mass, bins):
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (len(mass),) or not np.isfinite(values).all():
        raise ValueError("Invalid oracle-background validation values")
    return float(np.mean([values[group].mean() for group in _oracle_mass_strata(mass, bins)]))


def _oracle_contract(settings, train_z, train_mass, val_z, val_mass, closure_z, closure_mass, seed, device):
    epochs, mass_bins = _qphi_options(settings)
    return {
        "schema": 1,
        "scientific_version": RIDDLE_BENCHMARK_SCIENTIFIC_VERSION,
        "mode": MODE,
        "protocol": ORACLE_PROTOCOL,
        "epochs": epochs,
        "mass_bins": mass_bins,
        "flow": _qphi_flow(settings),
        "learning_rate": settings["training"]["learning_rate"],
        "weight_decay": 1e-4,
        "gradient_clip_norm": 1.0,
        "seed": int(seed),
        "training_region": "signal_region",
        "truth_role_selection": "pure_background",
        "hashes": {
            "train_z": digest(train_z),
            "train_mass": digest(train_mass),
            "validation_z": digest(val_z),
            "validation_mass": digest(val_mass),
            "closure_z": digest(closure_z),
            "closure_mass": digest(closure_mass),
        },
    }


def _activate_oracle_reuse(directory, candidates, contract, current_code, allow_code_change,
                           production_contract=None, allow_device_change=False):
    directory = Path(directory)
    for candidate in candidates or ():
        source = Path(candidate).resolve()
        result_path = source / "result.json"
        if not result_path.is_file():
            continue
        try:
            report = json.loads(result_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if report.get("completed") is not True or report.get("method") not in ("iad", "supervised"):
            continue
        from .scan_cache import runtime_compatible
        if not runtime_compatible(report.get("contract", {}), production_contract,
                                  allow_code_change=allow_code_change, allow_device_change=allow_device_change):
            continue
        source_code = report.get("contract", {}).get("code", {})
        changed_code = sorted(name for name in set(source_code) | set(current_code) if source_code.get(name) != current_code.get(name))
        if changed_code and not allow_code_change:
            continue
        base = source / "density" / "background_correction"
        names = ("contract.json", "model.pt", "selection.json")
        verified = {}
        valid = True
        for name in names:
            relative = f"density/background_correction/{name}"
            path = base / name
            expected = report.get("artifacts_sha256", {}).get(relative)
            if expected is None or not path.is_file() or file_digest(path) != expected:
                valid = False
                break
            verified[name] = path
        if not valid:
            continue
        try:
            source_contract = json.loads(verified["contract.json"].read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if source_contract != contract:
            continue
        directory.mkdir(parents=True, exist_ok=True)
        for name, path in verified.items():
            shutil.copy2(path, directory / name)
        history = base / "history.json"
        relative = "density/background_correction/history.json"
        expected = report.get("artifacts_sha256", {}).get(relative)
        if expected is not None and history.is_file() and file_digest(history) == expected:
            shutil.copy2(history, directory / "history.json")
        reuse = {
            "policy": "shared_oracle_sr_background_v1",
            "source_result": str(source),
            "source_result_sha256": file_digest(result_path),
            "contract_sha256": file_digest(directory / "contract.json"),
            "model_sha256": file_digest(directory / "model.pt"),
            "changed_code": changed_code,
            "code_change_approved": bool(changed_code),
        }
        write_json(directory / "reuse.json", reuse)
        return reuse
    return None


def train_oracle(directory, train_z, train_mass, val_z, val_mass, closure_z, closure_mass, *, settings, seed, device, reuse_candidates=None, current_code=None, allow_code_change=False, allow_device_change=False, production_contract=None):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    arrays = [np.ascontiguousarray(value, dtype=np.float32) for value in (train_z, train_mass, val_z, val_mass, closure_z, closure_mass)]
    train_z, train_mass, val_z, val_mass, closure_z, closure_mass = arrays
    if train_z.ndim != 2 or val_z.shape[1:] != train_z.shape[1:] or closure_z.shape[1:] != train_z.shape[1:] or min(len(train_z), len(val_z), len(closure_z)) < 2:
        raise ValueError("Oracle background requires aligned pure-background latent samples")
    for z, mass in ((train_z, train_mass), (val_z, val_mass), (closure_z, closure_mass)):
        if mass.shape != (len(z),) or not np.isfinite(z).all() or not np.isfinite(mass).all():
            raise ValueError("Invalid oracle-background inputs")
        if not np.all((mass > 3.3) & (mass < 3.7)):
            raise ValueError("Oracle background requires pure signal-region background events")
    epochs, mass_bins = _qphi_options(settings)
    contract = _oracle_contract(settings, train_z, train_mass, val_z, val_mass, closure_z, closure_mass, seed, device)
    if (production_contract or {}).get("scan_background"):
        decision = reuse(directory, production_contract["scan_background"]["source_result"],
                         expected_contract=contract, frozen=True, allow_device_change=allow_device_change)
        if decision is None or not decision["active"]:
            raise ValueError("Nominal oracle background is incompatible, inactive, or failed artifact verification")
        return decision
    current_code = {} if current_code is None else dict(current_code)
    contract_path = directory / "contract.json"
    if not contract_path.exists() and not (directory / ".resume/latest.pt").exists() and not (directory / "model.pt").exists():
        _activate_oracle_reuse(directory, reuse_candidates, contract, current_code, allow_code_change,
                              production_contract, allow_device_change)
    if contract_path.exists():
        _check_mapped_contract(json.loads(contract_path.read_text()), contract, allow_device_change,
                               directory / ".resume/device_history.json")
    if (directory / "model.pt").is_file() and (directory / "selection.json").is_file():
        selection = json.loads((directory / "selection.json").read_text())
        active = selection.get("status") == "activated" and bool(selection.get("active"))
        if not active:
            raise ValueError("Oracle background q_phi failed Gaussian validation; refusing a Gaussian denominator fallback")
        return {"requested_mode": MODE, "active": active, "descriptor": descriptor(directory), "selection": selection, "reuse": json.loads((directory / "reuse.json").read_text()) if (directory / "reuse.json").is_file() else None}
    write_json(contract_path, contract)
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % 2**32)
    model = _model(settings, train_z.shape[1], device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings["training"]["learning_rate"], weight_decay=1e-4)
    history, start = [], 0
    best_any, best_any_epoch, best_any_model = float("inf"), None, None
    best_eligible, best_eligible_epoch, best_eligible_model = float("inf"), None, None
    latest = directory / ".resume/latest.pt"
    if latest.exists():
        state = torch.load(latest, map_location=device, weights_only=False)
        _check_mapped_contract(state.get("contract"), contract, allow_device_change)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        history, start = state["history"], state["epoch"] + 1
        best_any, best_any_epoch, best_any_model = state["best_any"], state["best_any_epoch"], state["best_any_model"]
        best_eligible, best_eligible_epoch, best_eligible_model = state["best_eligible"], state["best_eligible_epoch"], state["best_eligible_model"]
        restore_rng(state["rng"])
    zt = torch.from_numpy(train_z)
    mt = torch.from_numpy(((train_mass - 3.5) / 0.2).astype(np.float32))
    zv = torch.from_numpy(val_z).to(device)
    mv = torch.from_numpy(((val_mass - 3.5) / 0.2).astype(np.float32)).to(device)
    dataset = TensorDataset(zt, mt)
    epoch_seconds = []
    for epoch in range(start, epochs):
        epoch_started = time.monotonic()
        generator = torch.Generator().manual_seed(int(seed) + 11000 + epoch)
        balanced = _oracle_balanced_indices(train_mass, mass_bins, int(seed) + 11500 + epoch)
        epoch_dataset = dataset if balanced is None else TensorDataset(zt[balanced], mt[balanced])
        batches = _training_batches(epoch_dataset, shuffle=balanced is None, generator=generator)
        model.train()
        total = 0.0
        trained_events = 0
        for z, mass in batches:
            z, mass = z.to(device), mass.to(device)
            optimizer.zero_grad()
            loss = -log_prob(model, z, mass).mean()
            require_finite(loss, "Oracle background loss")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0, error_if_nonfinite=True)
            optimizer.step()
            total += float(loss.detach()) * len(z)
            trained_events += len(z)
        model.eval()
        with torch.no_grad():
            validation_log_prob = log_prob(model, zv, mv).double().cpu().numpy()
            gaussian_log_prob = _gaussian_log_prob(zv).double().cpu().numpy()
        validation_nll = -float(validation_log_prob.mean())
        gaussian_nll = -float(gaussian_log_prob.mean())
        selection_nll = -_oracle_equal_stratum_mean(validation_log_prob, val_mass, mass_bins)
        gain = validation_log_prob - gaussian_log_prob
        improvement = float(gain.mean())
        improvement_standard_error = float(gain.std(ddof=1) / math.sqrt(len(gain))) if len(gain) > 1 else None
        epoch_gate = _gaussian_gate({"events": int(len(gain)), "qphi_nll": validation_nll, "gaussian_nll": gaussian_nll, "improvement": improvement, "improvement_standard_error": improvement_standard_error})
        row = {"epoch": epoch, "train_nll": total / trained_events, "validation_nll": validation_nll, "balanced_validation_nll": selection_nll, "gaussian_validation_nll": gaussian_nll, "improvement": improvement, "improvement_standard_error": improvement_standard_error, "gaussian_compatible": epoch_gate["gaussian_compatible"]}
        history.append(row)
        require_finite(validation_nll, "Oracle background validation NLL")
        require_finite(selection_nll, "Oracle background checkpoint selection NLL")
        if (selection_nll, epoch) < (best_any, best_any_epoch if best_any_epoch is not None else math.inf):
            best_any, best_any_epoch, best_any_model = selection_nll, epoch, deepcopy(model.state_dict())
        if epoch_gate["gaussian_compatible"] and (selection_nll, epoch) < (best_eligible, best_eligible_epoch if best_eligible_epoch is not None else math.inf):
            best_eligible, best_eligible_epoch, best_eligible_model = selection_nll, epoch, deepcopy(model.state_dict())
        epoch_seconds.append(time.monotonic() - epoch_started)
        if persist_boundary(epoch, epochs):
            latest.parent.mkdir(parents=True, exist_ok=True)
            atomic_torch_save(latest, {"contract": contract, "epoch": epoch, "model": model.state_dict(), "optimizer": optimizer.state_dict(), "history": history, "best_any": best_any, "best_any_epoch": best_any_epoch, "best_any_model": best_any_model, "best_eligible": best_eligible, "best_eligible_epoch": best_eligible_epoch, "best_eligible_model": best_eligible_model, "rng": rng_state()})
            write_json(directory / "history.json", history)
        emit_message(f"Oracle background {epoch+1}/{epochs}: validation NLL={validation_nll:.6g}; selection NLL={selection_nll:.6g}; Gaussian gate={epoch_gate['status']}")
    from .background_stage import record_timing
    record_timing(directory, "epochs", sum(epoch_seconds), epoch_seconds=epoch_seconds)
    use_eligible = best_eligible_model is not None
    best = best_eligible if use_eligible else best_any
    best_epoch = best_eligible_epoch if use_eligible else best_any_epoch
    best_model = best_eligible_model if use_eligible else best_any_model
    if best_model is None:
        raise FloatingPointError("Oracle background produced no valid checkpoint")
    atomic_torch_save(directory / "model.pt", {"model": best_model, "selected_epoch": best_epoch, "contract": contract})
    model.load_state_dict(best_model)
    model.eval().requires_grad_(False)
    validation = _gaussian_gate(_validation_metrics(model, val_z, val_mass, device))
    closure = _gaussian_gate(_validation_metrics(model, closure_z, closure_mass, device))
    active = bool(validation["gaussian_compatible"])
    reasons = [] if active else ["q_phi is significantly worse than Gaussian on reserved pure-background signal-region validation"]
    decision = {
        "mode": MODE,
        "protocol": ORACLE_PROTOCOL,
        "selected_epoch": int(best_epoch),
        "epochs": epochs,
        "status": "activated" if active else "validation_failed",
        "active": active,
        "checkpoint_selection": {"metric": "equal_mass_stratum_validation_nll", "eligibility": "Gaussian-compatible natural validation", "eligible_checkpoint_found": bool(use_eligible), "mass_bins": mass_bins, "selected_nll": float(best)},
        "criterion": "best equal-mass-stratum checkpoint among Gaussian-compatible reserved pure-background signal-region validation epochs",
        "validation_gate": {**validation, "activation_eligible": active, "direct_oracle_background": True},
        "pseudo_sr_closure": {"status": "not_applicable", "reason": "direct pure-background signal-region oracle"},
        "independent_closure_diagnostic": {"status": "evaluated", "events": int(len(closure_z)), "candidate_qphi": closure, "role": "pure_background_signal_region_closure", "used_for_training": False, "used_for_checkpoint_selection": False, "used_for_activation": False},
        "validation_scope": "reserved pure-background signal-region validation with Gaussian compatibility gate",
        "full_search_closure_status": "not_applicable",
        "truth_labels_used": True,
        "training_region": "pure background signal region",
        "denominator": "q_phi_oracle(z|m)" if active else None,
        "failure_reason": "; ".join(reasons) if reasons else None,
    }
    write_json(directory / "selection.json", decision)
    write_json(directory / "history.json", history)
    if not active:
        emit_message("Oracle background correction failed Gaussian validation gate; refusing Gaussian denominator fallback", level=0)
        raise ValueError("Oracle background q_phi failed Gaussian validation; refusing a Gaussian denominator fallback")
    emit_message("Oracle background correction activated", level=0)
    return {"requested_mode": MODE, "active": True, "descriptor": descriptor(directory), "selection": decision, "reuse": json.loads((directory / "reuse.json").read_text()) if (directory / "reuse.json").is_file() else None}


def reuse(directory, source_result, *, expected_contract, frozen=False, allow_device_change=False):
    if expected_contract.get("activation_policy") == "off":
        return None
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    manifest_path = directory / "reuse.json"
    expected_signature = expected_contract
    policy = "nominal_background_v1" if frozen else "exact_background_correction_inputs_v1"
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("policy") != policy or manifest.get("target_signature") != expected_signature:
            raise ValueError("Background-correction reuse settings changed; use a new output")
        for name, expected in manifest.get("local_artifacts_sha256", {}).items():
            path = directory / name
            if not path.is_file() or file_digest(path) != expected:
                raise ValueError("Reused background-correction artifact changed or is missing")
        selection = json.loads((directory / "selection.json").read_text())
        active = selection.get("status") == "activated" and bool(selection.get("active"))
        return dict(requested_mode=MODE, active=active, descriptor=descriptor(directory) if active else None,
                    selection=selection, reuse=manifest)
    if (directory / "contract.json").exists() or (directory / ".resume/latest.pt").exists():
        return None
    source = Path(source_result).resolve()
    report_path = source / "result.json"
    if not report_path.is_file():
        return None
    try:
        report = json.loads(report_path.read_text())
    except json.JSONDecodeError:
        return None
    native = str(report.get("method", "")).startswith("riddle") or (frozen and report.get("method") in ("iad", "supervised"))
    if report.get("completed") is not True or not native:
        return None
    base = source / "density" / "background_correction"
    relative_names = ("contract.json", "model.pt", "selection.json")
    source_paths = {}
    for name in relative_names:
        relative = f"density/background_correction/{name}"
        path = base / name
        expected = report.get("artifacts_sha256", {}).get(relative)
        if expected is None or not path.is_file() or file_digest(path) != expected:
            return None
        source_paths[name] = path
    source_contract = json.loads(source_paths["contract.json"].read_text())
    previous, current = dict(source_contract), dict(expected_signature)
    if frozen:
        previous.pop("hashes", None)
        current.pop("hashes", None)
        if allow_device_change:
            previous.pop("device", None)
            current.pop("device", None)
    if previous != current:
        return None
    optional = base / "history.json"
    optional_relative = "density/background_correction/history.json"
    optional_expected = report.get("artifacts_sha256", {}).get(optional_relative)
    if optional_expected is not None and optional.is_file() and file_digest(optional) == optional_expected:
        source_paths["history.json"] = optional
    copied = {}
    for name, source_path in source_paths.items():
        target = directory / name
        shutil.copy2(source_path, target)
        copied[name] = file_digest(target)
    selection = json.loads((directory / "selection.json").read_text())
    active = selection.get("status") == "activated" and bool(selection.get("active"))
    manifest = {
        "schema": 1,
        "policy": policy,
        "source_result": str(source),
        "source_result_sha256": file_digest(report_path),
        "source_model_sha256": file_digest(source_paths["model.pt"]),
        "target_signature": expected_signature,
        "local_artifacts_sha256": copied,
        "training_reused": True,
        "truth_labels_used": source_contract.get("truth_role_selection") == "pure_background",
    }
    write_json(manifest_path, manifest)
    emit_message(f"Reuse matching RIDDLE background correction from {source}", kind="PASS", level=0)
    return dict(requested_mode=MODE, active=active, descriptor=descriptor(directory) if active else None,
                selection=selection, reuse=manifest)


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
    protocol = contract.get("protocol", PROTOCOL)
    if contract.get("mode") != MODE or protocol not in READABLE_PROTOCOLS or type(contract.get("epochs")) is not int or contract["epochs"] < 10:
        raise ValueError("Invalid RIDDLE background model")
    selection = json.loads((directory / "selection.json").read_text())
    if selection.get("status") != "activated" or not selection.get("active"):
        raise ValueError("Inactive background correction cannot be used as the RIDDLE denominator")
    return dict(mode=MODE, protocol=protocol, epochs=int(contract["epochs"]),
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
    payload = {"model": model.state_dict(), "mode": MODE, "protocol": description["protocol"],
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
    if saved.get("mode") != MODE or saved.get("protocol") not in READABLE_PROTOCOLS:
        raise ValueError("Invalid fit-local background correction")
    model = _model(settings, features, device)
    model.load_state_dict(saved["model"])
    return model.eval().requires_grad_(False), saved
