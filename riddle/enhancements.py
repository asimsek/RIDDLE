"""Label-free algorithms used by the default RIDDLE pipeline.

Rosenblatt: doi:10.1214/aoms/1177729394; spline maps: arXiv:1906.04032.
Hard sampling adapts arXiv:1604.03540 with importance-corrected reference loss.
Contrastive fitting adapts proceedings.mlr.press/v9/gutmann10a.html.
Conditional score calibration follows the motivation of arXiv:2211.02486.
No API in this module receives signal labels or dataset-variation identifiers.
"""
from copy import deepcopy
import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .integrity import require_finite
from .storage import atomic_torch_save, digest, write_json, rng_state, restore_rng, seed_start, persist_boundary
from .worker_progress import emit_message


def save_torch(path, value):
    return atomic_torch_save(Path(path), value)


def load_torch(path, device="cpu"):
    # Recovery files include optimizer and RNG state.
    return torch.load(path, map_location=device, weights_only=False)


def chunks(function, *arrays, device="cpu", size=4096):
    if not arrays or not len(arrays[0]) or any(len(a) != len(arrays[0]) for a in arrays):
        raise ValueError("Inference requires aligned, nonempty arrays")
    with torch.no_grad():
        result = torch.cat([function(*(a[i:i+size].to(device) for a in arrays)).detach().cpu()
                            for i in range(0, len(arrays[0]), size)])
    require_finite(result, "RIDDLE inference")
    return result


class Rosenblatt(nn.Module):
    """Fixed-order triangular scalar splines; appending x does not rotate z."""
    def __init__(self, dimensions, seed, bins=12, hidden=64, bound=8.):
        super().__init__()
        self.bins, self.bound = bins, bound
        self.heads = nn.ModuleList()
        for j in range(dimensions):
            torch.manual_seed((int(seed) + 137*j) % 2**32)
            head = nn.Sequential(nn.Linear(j+1, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                 nn.Tanh(), nn.Linear(hidden, 3*bins-1))
            with torch.no_grad():
                head[-1].weight.zero_(); head[-1].bias.zero_()
                head[-1].bias[2*bins:] = math.log(math.expm1(1-1e-3))
            self.heads.append(head)

    def coordinate_log_probs(self, x, mass):
        from nflows.transforms.splines.rational_quadratic import unconstrained_rational_quadratic_spline
        zs, logs = [], []
        for j, head in enumerate(self.heads):
            p = head(torch.cat((mass, x[:, :j]), 1))
            z, ld = unconstrained_rational_quadratic_spline(
                x[:, j], p[:, :self.bins], p[:, self.bins:2*self.bins], p[:, 2*self.bins:],
                tails="linear", tail_bound=self.bound)
            zs.append(z); logs.append(-.5*(z.square()+math.log(2*math.pi))+ld)
        return torch.stack(zs, 1), torch.stack(logs, 1)

    def forward(self, x, mass):
        return self.coordinate_log_probs(x, mass)[0]

    def log_probs(self, x, mass):
        return self.coordinate_log_probs(x, mass)[1].sum(1)


def proposal(logits, pool_fraction=.25, sampling_fraction=.5):
    if not 0 < pool_fraction <= 1 or not 0 <= sampling_fraction < 1:
        raise ValueError("Invalid hard-background proposal")
    logits = logits.detach().cpu().flatten().double()
    require_finite(logits, "Hard-background pool logits")
    n = len(logits)
    if not n:
        raise ValueError("Empty hard-background pool")
    k = max(1, math.ceil(n*pool_fraction))
    hard = torch.zeros(n, dtype=torch.bool)
    hard[torch.argsort(logits, descending=True, stable=True)[:k]] = True
    q = torch.full((n,), (1-sampling_fraction)/n, dtype=torch.float64)
    q[hard] += sampling_fraction/k
    return q, 1/(n*q), hard


def _legacy_teacher_weights(directory, z, fraction, *, seed, options, batch_size, device="cpu"):
    """Two-fold out-of-fold data/reference guide with resumable hard mining."""
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    z = torch.as_tensor(z, dtype=torch.float32).cpu()
    require_finite(z, "Guide inputs")
    if z.ndim != 2 or len(z) < 4 or not 0 < fraction < 1:
        raise ValueError("Guide requires at least four events and an interior fraction")
    contract = dict(z=digest(z.numpy()), seed=seed, options=options,
                    batch_size=batch_size, fraction=fraction, device=str(device))
    saved = directory / "weights.pt"
    if saved.exists():
        result = load_torch(saved)
        if result["contract"] != contract:
            raise ValueError("Guide inputs/settings changed")
        require_finite(result["weights"], "Saved guide weights")
        write_json(directory / "guidance.json", result["guidance"])
        return result["weights"]
    folds = np.array_split(np.random.default_rng(seed).permutation(len(z)), 2)
    logits, reports = torch.zeros(len(z)), []
    for f in range(2):
        seed_start((seed+f) % 2**32)
        model = nn.Sequential(nn.Linear(z.shape[1], 64), nn.LeakyReLU(), nn.Linear(64, 64),
                              nn.LeakyReLU(), nn.Linear(64, 1)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        train, hold = folds[1-f], folds[f]
        gen = torch.Generator().manual_seed(seed+f+1000)
        ref = torch.randn((len(train), z.shape[1]), generator=gen)
        n = len(ref); origin = torch.cat((torch.ones(n), torch.zeros(n)))
        latest = directory / f"fold_{f}.pt"
        start, history = 0, []
        if latest.exists():
            state = load_torch(latest, device)
            if state["contract"] != contract:
                raise ValueError("Guide recovery contract changed")
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            start, history = state["epoch"]+1, state["history"]
            restore_rng(state["rng"])
        for epoch in range(start, options["guide_epochs"]):
            model.eval()
            if options["hard_bg"] and epoch >= options["hard_start_epoch"]:
                pool = chunks(lambda x: model(x).flatten(), ref, device=device)
                q, importance, hard = proposal(pool, options["hard_pool_fraction"], options["hard_sampling_fraction"])
                draw = torch.Generator().manual_seed(seed+50000+1000*f+epoch)
                chosen = torch.multinomial(q, n, replacement=True, generator=draw)
                iw = importance[chosen].float()
                info = dict(phase="hard_background", hard_pool_events=int(hard.sum()),
                            selected_hard_fraction=float(hard[chosen].double().mean()),
                            importance_mean=float(iw.mean()), proposal_sha256=digest(q.numpy()))
            else:
                chosen, iw = torch.arange(n), torch.ones(n)
                info = dict(phase="uniform_reference")
            loader = DataLoader(TensorDataset(torch.cat((z[train], ref[chosen])), origin,
                                             torch.cat((torch.ones(n), iw))),
                                batch_size=batch_size, shuffle=True)
            model.train(); total = 0.
            for x, y, w in loader:
                x, y, w = x.to(device), y.to(device), w.to(device)
                optimizer.zero_grad()
                loss = (w*nn.functional.binary_cross_entropy_with_logits(model(x).flatten(), y, reduction="none")).mean()
                require_finite(loss, "Guide weighted BCE")
                loss.backward(); clip_gradients(model.parameters(), 1.)
                optimizer.step(); total += float(loss.detach())*len(x)
            history.append(dict(epoch=epoch, loss=total/(2*n), **info))
            if persist_boundary(epoch, options["guide_epochs"]):
                save_torch(latest, dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                       history=history, rng=rng_state(), contract=contract))
            emit_message(f"Guide fold {f+1}/2 epoch {epoch+1}/{options['guide_epochs']}: {info['phase']}")
        model.eval()
        logits[hold] = chunks(lambda x: model(x).flatten(), z[hold], device=device)
        reports.append(dict(fold=f, train_indices_sha256=digest(train), holdout_indices_sha256=digest(hold),
                            training_events=len(train), holdout_events=len(hold), history=history))
    weights = -torch.expm1(torch.minimum(torch.zeros_like(logits), math.log1p(-fraction)-logits))
    fallback = float(weights.sum()) < 1e-8
    if fallback:
        weights.fill_(fraction)
    report = dict(truth_labels_used=False, folds=2, hard_bg=options["hard_bg"], fold_reports=reports,
                  initial_fraction=fraction, mean_weight=float(weights.mean()), uniform_fallback=fallback,
                  reference="independent standard normal; classifier inputs are latent coordinates only")
    save_torch(saved, dict(contract=contract, weights=weights, logits=logits, guidance=report))
    write_json(directory / "guidance.json", report)
    return weights


def _legacy_corrected_teacher_weights(directory, z, context, fraction, *, seed, options, batch_size,
                                      background_model, background_sha256, device="cpu"):
    """Two-fold mass-blind guide using q_phi(z|m) samples at matched SR masses.

    This is the production form of the validated bgcorr_40_reguide control.  The
    classifier never receives mass; mass is used only to sample the denominator
    at the same contexts as the data events.
    """
    from .background_correction import sample as sample_background
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    z = torch.as_tensor(z, dtype=torch.float32).cpu()
    context = torch.as_tensor(context, dtype=torch.float32).flatten().cpu()
    require_finite(z, "Corrected-guide inputs"); require_finite(context, "Corrected-guide mass contexts")
    if z.ndim != 2 or context.shape != (len(z),) or len(z) < 4 or not 0 < fraction < 1:
        raise ValueError("Corrected guide requires aligned latent/context events and an interior fraction")
    contract = dict(z=digest(z.numpy()), context=digest(context.numpy()), seed=seed, options=options,
                    batch_size=batch_size, fraction=fraction, device=str(device),
                    reference="q_phi(z|m)", background_sha256=background_sha256)
    saved = directory / "weights.pt"
    if saved.exists():
        result = load_torch(saved)
        if result["contract"] != contract:
            raise ValueError("Corrected-guide inputs/settings changed")
        require_finite(result["weights"], "Saved corrected-guide weights")
        write_json(directory / "guidance.json", result["guidance"])
        return result["weights"]
    folds = np.array_split(np.random.default_rng(seed).permutation(len(z)), 2)
    logits, reports = torch.zeros(len(z)), []
    for f in range(2):
        seed_start((seed + f) % 2**32)
        model = nn.Sequential(nn.Linear(z.shape[1], 64), nn.LeakyReLU(), nn.Linear(64, 64),
                              nn.LeakyReLU(), nn.Linear(64, 1)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        train, hold = folds[1-f], folds[f]
        ref = sample_background(background_model, context[train].to(device), z.shape[1],
                                seed + 71000 + f, device).detach().cpu()
        n = len(train); origin = torch.cat((torch.ones(n), torch.zeros(n)))
        latest = directory / f"fold_{f}.pt"
        start, history = 0, []
        if latest.exists():
            state = load_torch(latest, device)
            if state["contract"] != contract:
                raise ValueError("Corrected-guide recovery contract changed")
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            start, history = state["epoch"] + 1, state["history"]
            restore_rng(state["rng"])
        for epoch in range(start, options["guide_epochs"]):
            model.eval()
            if options["hard_bg"] and epoch >= options["hard_start_epoch"]:
                pool = chunks(lambda x: model(x).flatten(), ref, device=device)
                q, importance, hard = proposal(pool, options["hard_pool_fraction"], options["hard_sampling_fraction"])
                draw = torch.Generator().manual_seed(seed + 72000 + 1000*f + epoch)
                chosen = torch.multinomial(q, n, replacement=True, generator=draw)
                iw = importance[chosen].float()
                info = dict(phase="hard_background", hard_pool_events=int(hard.sum()),
                            selected_hard_fraction=float(hard[chosen].double().mean()),
                            importance_mean=float(iw.mean()), proposal_sha256=digest(q.numpy()))
            else:
                chosen, iw = torch.arange(n), torch.ones(n)
                info = dict(phase="uniform_reference")
            loader_gen = torch.Generator().manual_seed(seed + 73000 + 1000*f + epoch)
            loader = DataLoader(TensorDataset(torch.cat((z[train], ref[chosen])), origin,
                                             torch.cat((torch.ones(n), iw))),
                                batch_size=batch_size, shuffle=True, generator=loader_gen)
            model.train(); total = 0.
            for x, y, w in loader:
                x, y, w = x.to(device), y.to(device), w.to(device)
                optimizer.zero_grad()
                loss = (w*nn.functional.binary_cross_entropy_with_logits(model(x).flatten(), y,
                                                                          reduction="none")).mean()
                require_finite(loss, "Corrected-guide weighted BCE")
                loss.backward(); clip_gradients(model.parameters(), 1.)
                optimizer.step(); total += float(loss.detach())*len(x)
            history.append(dict(epoch=epoch, loss=total/(2*n), **info))
            if persist_boundary(epoch, options["guide_epochs"]):
                save_torch(latest, dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                       history=history, rng=rng_state(), contract=contract))
            emit_message(f"Corrected guide fold {f+1}/2 epoch {epoch+1}/{options['guide_epochs']}: {info['phase']}")
        model.eval()
        logits[hold] = chunks(lambda x: model(x).flatten(), z[hold], device=device)
        reports.append(dict(fold=f, train_indices_sha256=digest(train), holdout_indices_sha256=digest(hold),
                            training_events=len(train), holdout_events=len(hold), history=history))
    weights = -torch.expm1(torch.minimum(torch.zeros_like(logits), math.log1p(-fraction)-logits))
    fallback = float(weights.sum()) < 1e-8
    if fallback:
        weights.fill_(fraction)
    report = dict(truth_labels_used=False, folds=2, hard_bg=options["hard_bg"], fold_reports=reports,
                  initial_fraction=fraction, mean_weight=float(weights.mean()), uniform_fallback=fallback,
                  reference="q_phi(z|m) samples at matched masses; classifier inputs are latent coordinates only",
                  background_sha256=background_sha256)
    save_torch(saved, dict(contract=contract, weights=weights, logits=logits, guidance=report))
    write_json(directory / "guidance.json", report)
    return weights



def _guide_plus_enabled(options):
    return bool(
        options.get("guide_folds", 2) != 2
        or options.get("guide_mass_conditioning", False)
        or options.get("guide_reference_multiplier", 1) != 1
        or options.get("guide_refresh_reference", False)
        or options.get("guide_ratio_calibration", False)
    )


def _logmeanexp(values):
    values = torch.as_tensor(values, dtype=torch.float64)
    return torch.logsumexp(values, dim=0) - math.log(len(values))


def _guide_input(z, context, use_mass):
    if not use_mass:
        return z
    if context is None:
        raise ValueError("Mass-conditioned guide requires matched mass contexts")
    return torch.cat((z, context.reshape(-1, 1)), dim=1)


def _folds(length, count, seed):
    if count > length // 2:
        raise ValueError("Too many guide folds for the available SR events")
    return np.array_split(np.random.default_rng(seed).permutation(length), count)


def _train_indices(folds, hold_index):
    parts = [part for i, part in enumerate(folds) if i != hold_index]
    return np.concatenate(parts) if len(parts) > 1 else parts[0]


def _standard_reference(context, dimensions, count, *, seed):
    gen = torch.Generator().manual_seed(int(seed))
    if context is None:
        chosen_context = None
    else:
        draw = torch.randint(0, len(context), (count,), generator=gen)
        chosen_context = context[draw]
    latent = torch.randn((count, dimensions), generator=gen)
    return latent, chosen_context


def _corrected_reference(background_model, context, dimensions, count, *, seed, device):
    from .background_correction import sample as sample_background
    gen = torch.Generator().manual_seed(int(seed))
    draw = torch.randint(0, len(context), (count,), generator=gen)
    chosen_context = context[draw]
    latent = sample_background(background_model, chosen_context.to(device), dimensions, int(seed) + 17, device)
    return latent.detach().cpu(), chosen_context


def _teacher_weights_plus(directory, z, context, fraction, *, seed, options, batch_size,
                          reference_kind, background_model=None, background_sha256=None, device="cpu"):
    """Configurable OOF guide used only by the tail-sensitivity ablation study.

    The default production path stays byte-for-byte on the legacy two-fold guide.
    Guide+ can use more OOF folds, matched mass as a classifier context, a larger
    refreshable reference reservoir, and an independent density-ratio
    normalization E_q[exp(log r)]=1.  No truth labels enter this routine.
    """
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    z = torch.as_tensor(z, dtype=torch.float32).cpu()
    context = None if context is None else torch.as_tensor(context, dtype=torch.float32).flatten().cpu()
    require_finite(z, "Guide+ inputs")
    if context is not None:
        require_finite(context, "Guide+ mass contexts")
    if z.ndim != 2 or len(z) < 4 or not 0 < fraction < 1:
        raise ValueError("Guide+ requires at least four events and an interior fraction")
    if context is not None and context.shape != (len(z),):
        raise ValueError("Guide+ contexts are misaligned")
    use_mass = bool(options.get("guide_mass_conditioning", False))
    folds_count = int(options.get("guide_folds", 2))
    multiplier = int(options.get("guide_reference_multiplier", 1))
    refresh = bool(options.get("guide_refresh_reference", False))
    calibrate = bool(options.get("guide_ratio_calibration", False))
    if use_mass and context is None:
        raise ValueError("Guide+ mass conditioning requested without mass contexts")
    if reference_kind not in ("gaussian", "qphi"):
        raise ValueError("Unknown Guide+ reference")
    contract = dict(
        z=digest(z.numpy()), context=None if context is None else digest(context.numpy()), seed=seed,
        options=options, batch_size=batch_size, fraction=fraction, device=str(device),
        reference=reference_kind, background_sha256=background_sha256,
        protocol="oof_guide_plus_v1",
    )
    saved = directory / "weights.pt"
    if saved.exists():
        result = load_torch(saved)
        if result["contract"] != contract:
            raise ValueError("Guide+ inputs/settings changed")
        require_finite(result["weights"], "Saved Guide+ weights")
        write_json(directory / "guidance.json", result["guidance"])
        return result["weights"]

    folds = _folds(len(z), folds_count, seed)
    logits, reports = torch.zeros(len(z)), []
    input_features = z.shape[1] + int(use_mass)

    def make_reference(base_context, count, ref_seed):
        if reference_kind == "qphi":
            if background_model is None or base_context is None:
                raise ValueError("q_phi Guide+ requires a background model and contexts")
            return _corrected_reference(background_model, base_context, z.shape[1], count,
                                        seed=ref_seed, device=device)
        return _standard_reference(base_context, z.shape[1], count, seed=ref_seed)

    for f in range(folds_count):
        seed_start((seed + f) % 2**32)
        model = nn.Sequential(nn.Linear(input_features, 64), nn.LeakyReLU(), nn.Linear(64, 64),
                              nn.LeakyReLU(), nn.Linear(64, 1)).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=.01)
        train, hold = _train_indices(folds, f), folds[f]
        n = len(train)
        latest = directory / f"fold_{f}.pt"
        start, history = 0, []
        if latest.exists():
            state = load_torch(latest, device)
            if state["contract"] != contract:
                raise ValueError("Guide+ recovery contract changed")
            model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
            start, history = state["epoch"] + 1, state["history"]
            restore_rng(state["rng"])

        fixed_ref = None
        if not refresh:
            fixed_ref = make_reference(None if context is None else context[train], multiplier*n,
                                       seed + 81000 + 1000*f)

        for epoch in range(start, options["guide_epochs"]):
            if refresh or fixed_ref is None:
                ref, ref_context = make_reference(None if context is None else context[train], multiplier*n,
                                                  seed + 82000 + 100000*f + epoch)
            else:
                ref, ref_context = fixed_ref
            model.eval()
            ref_input = _guide_input(ref, ref_context, use_mass)
            if options["hard_bg"] and epoch >= options["hard_start_epoch"]:
                pool = chunks(lambda x: model(x).flatten(), ref_input, device=device)
                q, importance, hard = proposal(pool, options["hard_pool_fraction"], options["hard_sampling_fraction"])
                draw = torch.Generator().manual_seed(seed + 83000 + 1000*f + epoch)
                chosen = torch.multinomial(q, n, replacement=True, generator=draw)
                iw = importance[chosen].float()
                info = dict(phase="hard_background", reference_pool_events=len(ref),
                            hard_pool_events=int(hard.sum()),
                            selected_hard_fraction=float(hard[chosen].double().mean()),
                            importance_mean=float(iw.mean()), proposal_sha256=digest(q.numpy()))
            else:
                draw = torch.Generator().manual_seed(seed + 84000 + 1000*f + epoch)
                chosen = (torch.randperm(len(ref), generator=draw)[:n] if len(ref) >= n
                          else torch.randint(0, len(ref), (n,), generator=draw))
                iw = torch.ones(n)
                info = dict(phase="uniform_reference", reference_pool_events=len(ref))
            positive = _guide_input(z[train], None if context is None else context[train], use_mass)
            negative = ref_input[chosen]
            origin = torch.cat((torch.ones(n), torch.zeros(n)))
            weights = torch.cat((torch.ones(n), iw))
            loader_gen = torch.Generator().manual_seed(seed + 85000 + 1000*f + epoch)
            loader = DataLoader(TensorDataset(torch.cat((positive, negative)), origin, weights),
                                batch_size=batch_size, shuffle=True, generator=loader_gen)
            model.train(); total = 0.
            for x, y, w in loader:
                x, y, w = x.to(device), y.to(device), w.to(device)
                optimizer.zero_grad()
                loss = (w*nn.functional.binary_cross_entropy_with_logits(
                    model(x).flatten(), y, reduction="none")).mean()
                require_finite(loss, "Guide+ weighted BCE")
                loss.backward(); clip_gradients(model.parameters(), 1.)
                optimizer.step(); total += float(loss.detach())*len(x)
            history.append(dict(epoch=epoch, loss=total/(2*n), **info))
            if persist_boundary(epoch, options["guide_epochs"]):
                save_torch(latest, dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                       history=history, rng=rng_state(), contract=contract))
            emit_message(f"Guide+ fold {f+1}/{folds_count} epoch {epoch+1}/{options['guide_epochs']}: {info['phase']}")

        model.eval()
        hold_input = _guide_input(z[hold], None if context is None else context[hold], use_mass)
        hold_logits = chunks(lambda x: model(x).flatten(), hold_input, device=device)
        calibration_shift = 0.0
        if calibrate:
            cal_ref, cal_context = make_reference(None if context is None else context[hold], max(len(hold), 256),
                                                  seed + 86000 + f)
            cal_input = _guide_input(cal_ref, cal_context, use_mass)
            cal_logits = chunks(lambda x: model(x).flatten(), cal_input, device=device)
            calibration_shift = float(_logmeanexp(cal_logits))
            hold_logits = hold_logits - calibration_shift
        logits[hold] = hold_logits
        reports.append(dict(
            fold=f, train_indices_sha256=digest(train), holdout_indices_sha256=digest(hold),
            training_events=len(train), holdout_events=len(hold), history=history,
            ratio_calibration_shift=calibration_shift,
        ))

    weights = -torch.expm1(torch.minimum(torch.zeros_like(logits), math.log1p(-fraction)-logits))
    fallback = float(weights.sum()) < 1e-8
    if fallback:
        weights.fill_(fraction)
    report = dict(
        truth_labels_used=False, folds=folds_count, hard_bg=options["hard_bg"], fold_reports=reports,
        initial_fraction=fraction, mean_weight=float(weights.mean()), uniform_fallback=fallback,
        reference=("q_phi(z|m)" if reference_kind == "qphi" else "standard normal") +
                  ("; classifier receives matched mass context" if use_mass else "; mass-blind classifier"),
        reference_multiplier=multiplier, refreshed_each_epoch=refresh,
        ratio_calibration=calibrate, background_sha256=background_sha256,
    )
    save_torch(saved, dict(contract=contract, weights=weights, logits=logits, guidance=report))
    write_json(directory / "guidance.json", report)
    return weights


def teacher_weights(directory, z, fraction, *, seed, options, batch_size, context=None, device="cpu"):
    if not _guide_plus_enabled(options):
        return _legacy_teacher_weights(directory, z, fraction, seed=seed, options=options,
                                       batch_size=batch_size, device=device)
    return _teacher_weights_plus(directory, z, context, fraction, seed=seed, options=options,
                                 batch_size=batch_size, reference_kind="gaussian", device=device)


def corrected_teacher_weights(directory, z, context, fraction, *, seed, options, batch_size,
                              background_model, background_sha256, device="cpu"):
    if not _guide_plus_enabled(options):
        return _legacy_corrected_teacher_weights(
            directory, z, context, fraction, seed=seed, options=options, batch_size=batch_size,
            background_model=background_model, background_sha256=background_sha256, device=device)
    return _teacher_weights_plus(
        directory, z, context, fraction, seed=seed, options=options, batch_size=batch_size,
        reference_kind="qphi", background_model=background_model,
        background_sha256=background_sha256, device=device)

def clip_gradients(parameters, limit):
    try:
        nn.utils.clip_grad_norm_(parameters, limit, error_if_nonfinite=True)
    except RuntimeError as error:
        raise FloatingPointError("Nonfinite gradient; optimizer not updated") from error



def sharpen_responsibilities(weights, temperature):
    """Sharpen soft responsibilities while preserving their mean exactly enough for training."""
    if temperature >= 1.0 - 1e-12:
        return weights
    if not 0 < temperature < 1:
        raise ValueError("Responsibility temperature must lie in (0,1] for sharpening")
    original = weights.detach().double().clamp(1e-7, 1-1e-7)
    target = original.mean()
    logits = torch.logit(original) / float(temperature)
    lo, hi = -40.0, 40.0
    for _ in range(80):
        mid = 0.5*(lo+hi)
        mean = torch.sigmoid(logits + mid).mean()
        if mean < target:
            lo = mid
        else:
            hi = mid
    sharpened = torch.sigmoid(logits + 0.5*(lo+hi)).to(dtype=weights.dtype)
    require_finite(sharpened, "Sharpened residual responsibilities")
    return sharpened


def tail_ranking_loss(positive_ratio, negative_ratio, weights, mean_weight, *, margin=0.0, temperature=1.0):
    """Weighted pairwise ranking loss against final-score hard background samples."""
    if temperature <= 0:
        raise ValueError("Tail ranking temperature must be positive")

    difference = (positive_ratio[:, None] - negative_ratio[None, :] - float(margin)) / float(temperature)
    per_positive = nn.functional.softplus(-difference).mean(dim=1)
    return (weights * per_positive).mean() / mean_weight

def contrastive_loss(positive, negative, weights, mean_weight):
    return .5*((weights*nn.functional.softplus(-positive)).mean()/mean_weight
               + nn.functional.softplus(negative).mean())


def standard_normal_log_prob(z):
    # Keep the validated float32 reduction because alternatives change EM weights.

    return -.5 * (z.square() + math.log(2 * math.pi)).sum(-1)


def mixture_gain(ratio, fraction):
    ratio = np.asarray(ratio, dtype=np.float64)
    fraction = np.asarray(fraction, dtype=np.float64)
    if not np.isfinite(ratio).all() or not np.isfinite(fraction).all():
        raise ValueError("Invalid density ratio/fraction")
    try:
        fraction = np.broadcast_to(fraction, ratio.shape)
    except ValueError as error:
        raise ValueError("Density ratio and fraction are not broadcast-compatible") from error
    if np.any(fraction < 0) or np.any(fraction > 1):
        raise ValueError("Invalid density ratio/fraction")
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.logaddexp(np.log1p(-fraction), np.log(fraction) + ratio)


def profile_fraction(ratio):
    from scipy.optimize import minimize_scalar
    ratio = np.asarray(ratio, dtype=np.float64)
    if ratio.ndim != 1 or len(ratio) < 2 or not np.isfinite(ratio).all():
        raise ValueError("Fraction profiling requires finite internal-validation ratios")
    result = minimize_scalar(lambda f: -mixture_gain(ratio, f).mean(), bounds=(0., 1.),
                             method="bounded", options={"xatol": 1e-12})
    if not result.success:
        raise FloatingPointError("Coherent fraction optimization failed")
    candidates = [0., float(result.x), 1.]
    values = [float(mixture_gain(ratio, f).mean()) for f in candidates]
    return next(f for f, v in zip(candidates, values) if v >= max(values)-1e-12)


class ScoreFlow(nn.Module):
    def __init__(self, seed, parameters):
        super().__init__()
        self.transform = Rosenblatt(1, seed, bins=8, hidden=16, bound=8.)
        self.register_buffer("parameters_", torch.tensor(parameters, dtype=torch.float32))

    def gaussian(self, score, mass):
        sm, ss, mm, ms = self.parameters_
        return self.transform(((score-sm)/ss).reshape(-1, 1), ((mass-mm)/ms).reshape(-1, 1)).flatten()

    def log_prob(self, score, mass):
        sm, ss, mm, ms = self.parameters_
        return self.transform.log_probs(((score-sm)/ss).reshape(-1, 1), ((mass-mm)/ms).reshape(-1, 1))-ss.log()

    def forward(self, score, mass):
        z = self.gaussian(score, mass).double()
        return torch.special.log_ndtr(z)-torch.special.log_ndtr(-z)


def train_score_flow(directory, train_score, train_mass, val_score, val_mass, *, seed, epochs, device="cpu"):
    directory = Path(directory); directory.mkdir(parents=True, exist_ok=True)
    arrays = [torch.as_tensor(a, dtype=torch.float32).flatten().cpu()
              for a in (train_score, train_mass, val_score, val_mass)]
    st, mt, sv, mv = arrays
    for s, m in ((st, mt), (sv, mv)):
        require_finite(s, "Score calibration input"); require_finite(m, "Calibration masses")
        if len(s) < 30 or len(s) != len(m) or ((m > 3.3) & (m < 3.7)).any():
            raise ValueError("Score calibration needs >=30 aligned sideband events; SR is excluded")
    parameters = [float(st.mean()), max(float(st.std()), 1e-4), float(mt.mean()), float(mt.std())]
    if parameters[-1] <= 0:
        raise ValueError("Calibration requires a mass span")
    contract = dict(seed=seed, epochs=epochs, parameters=parameters, device=str(device),
                    hashes=[digest(a.numpy()) for a in arrays])
    seed_start(seed); model = ScoreFlow(seed, parameters).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=3e-4, weight_decay=1e-4)
    loader = DataLoader(TensorDataset(st, mt), batch_size=256, shuffle=True)
    start, history, best, best_model = 0, [], float("inf"), None
    latest = directory / ".resume/latest.pt"
    if latest.exists():
        state = load_torch(latest, device)
        if state["contract"] != contract:
            raise ValueError("Score-flow inputs/settings changed")
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        start, history, best, best_model = state["epoch"]+1, state["history"], state["best"], state["best_model"]
        restore_rng(state["rng"])
    for epoch in range(start, epochs):
        model.train()
        for s, m in loader:
            optimizer.zero_grad(); loss = -model.log_prob(s.to(device), m.to(device)).mean()
            require_finite(loss, "Score-flow NLL")
            loss.backward(); clip_gradients(model.parameters(), 1.)
            optimizer.step()
        model.eval(); value = -float(chunks(model.log_prob, sv, mv, device=device).double().mean())
        history.append(dict(epoch=epoch, validation_nll=value))
        if value < best:
            best, best_model = value, deepcopy(model.state_dict())
        if persist_boundary(epoch, epochs):
            save_torch(latest, dict(epoch=epoch, model=model.state_dict(), optimizer=optimizer.state_dict(),
                                   contract=contract, rng=rng_state(), history=history, best=best, best_model=best_model))
            write_json(directory / "history.json", history)
        emit_message(f"Score flow {epoch+1}/{epochs}: sideband validation NLL={value:.6g}")
    model.load_state_dict(best_model); model.eval().requires_grad_(False)
    save_torch(directory / "model.pt", dict(model=best_model, contract=contract))
    write_json(directory / "selection.json", dict(contract=contract, truth_labels_used=False,
               selected_epoch=min(history, key=lambda h: h["validation_nll"])["epoch"],
               score="logit of conditional background percentile", training_region="independent sidebands"))
    return model


def load_score_flow(directory, device="cpu"):
    state = load_torch(Path(directory) / "model.pt", device)
    model = ScoreFlow(state["contract"]["seed"], state["contract"]["parameters"]).to(device)
    model.load_state_dict(state["model"])
    return model.eval().requires_grad_(False)
