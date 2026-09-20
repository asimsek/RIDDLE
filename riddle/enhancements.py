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
from .storage import atomic_write, digest, write_json, rng_state, restore_rng, seed_start
from .worker_progress import emit_message


def save_torch(path, value):
    atomic_write(Path(path), lambda p: torch.save(value, p))


def load_torch(path, device="cpu"):
    # Recovery files are produced locally and include optimizer/RNG state.
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


def teacher_weights(directory, z, fraction, *, seed, options, batch_size, device="cpu"):
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


def clip_gradients(parameters, limit):
    try:
        nn.utils.clip_grad_norm_(parameters, limit, error_if_nonfinite=True)
    except RuntimeError as error:
        raise FloatingPointError("Nonfinite gradient; optimizer not updated") from error


def contrastive_loss(positive, negative, weights, mean_weight):
    return .5*((weights*nn.functional.softplus(-positive)).mean()/mean_weight
               + nn.functional.softplus(negative).mean())


def standard_normal_log_prob(z):
    # Keep the per-coordinate float32 reduction used by the validated study.
    # Algebraically equivalent reductions can change EM weights/checkpoints.
    return -.5 * (z.square() + math.log(2 * math.pi)).sum(-1)


def mixture_gain(ratio, fraction):
    ratio = np.asarray(ratio, dtype=np.float64)
    if not np.isfinite(ratio).all() or not 0 <= fraction <= 1:
        raise ValueError("Invalid density ratio/fraction")
    if fraction == 0:
        return np.zeros_like(ratio)
    if fraction == 1:
        return ratio.copy()
    return np.logaddexp(math.log1p(-fraction), math.log(fraction)+ratio)


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
