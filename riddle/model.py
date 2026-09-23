from importlib.metadata import version
import math
import numpy as np
import torch
from .settings import DEFAULTS
from .integrity import SCIENTIFIC_VERSION, mixture_log_density

PROTOCOL = {
    "scientific_version": SCIENTIFIC_VERSION,
    "name": "Residual",
    "features": 4,
    "layers": 6,
    "blocks": 2,
    "hidden_features": 64,
    "batch_size": 256,
    "validation_batch_size": 1280,
    "default_epochs": DEFAULTS["riddle"]["epochs"],
    "optimizer": "AdamW",
    "learning_rate": 0.0003,
    "weight_decay": 0.01,
    "fraction_weight_decay": 0.0,
    "mixture_loss": "logaddexp with log-sigmoid weights; no density floor",
    "fraction_interpretation": "fitted mixture parameter, not an unbiased contamination measurement",
    "gradient_clip": "flow parameters only; norm 1",
    "selected_checkpoints": 10,
    "inputs": "saved real-data SR latents; no mass or truth labels",
    "background": "standard-normal latent base; no mass PDF",
    "ensemble": "arithmetic mean of ten signal densities; validation mixture NLL selection",
    "score": "log(mean signal density) - log standard-normal density",
    "scan_score": "sigmoid(log density ratio); monotone display coordinate, not signal probability",
}


def build_signal_flow(device="cpu", *, features=4, settings=None):
    mass_conditioning = bool((settings or {}).get("mass_conditioning", False))
    physical_inputs = (settings or {}).get("input_space") == "physical"
    features -= int(mass_conditioning) + int(physical_inputs)
    if features not in (4, 5):
        raise ValueError("Residual flow accepts four configured latents, or five for DeltaR")
    if version("nflows") != "0.14":
        raise RuntimeError("RIDDLE requires nflows==0.14")
    from nflows.distributions.normal import StandardNormal
    from nflows.flows.base import Flow
    from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform
    from nflows.transforms.base import CompositeTransform
    from nflows.transforms.permutations import RandomPermutation

    config = (DEFAULTS["riddle"] if settings is None else settings)["flow"]
    transforms = []
    for _ in range(config["layers"]):
        transforms.append(
            MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
                features=features,
                context_features=1 if mass_conditioning else None,
                hidden_features=config["hidden_features"],
                num_blocks=config["num_blocks"],
                use_residual_blocks=config["use_residual_blocks"],
                use_batch_norm=config["use_batch_norm"],
                dropout_probability=config["dropout_probability"],
                activation=torch.nn.functional.leaky_relu,
                random_mask=config["random_mask"],
                num_bins=config["num_bins"],
                tails=config["tails"],
                tail_bound=config["tail_bound"],
                min_bin_width=config["min_bin_width"],
                min_bin_height=config["min_bin_height"],
                min_derivative=config["min_derivative"],
            )
        )
        transforms.append(RandomPermutation(features))
    model = Flow(CompositeTransform(transforms), StandardNormal([features])).to(device)
    model.mass_conditioning = mass_conditioning
    model.physical_inputs = physical_inputs
    return model


def match_background(model):
    from nflows.transforms.autoregressive import MaskedPiecewiseRationalQuadraticAutoregressiveTransform

    with torch.no_grad():
        for transform in model._transform._transforms:
            if isinstance(transform, MaskedPiecewiseRationalQuadraticAutoregressiveTransform):
                layer = transform.autoregressive_net.final_layer
                layer.weight.zero_()
                bias = layer.bias.reshape(model._distribution._shape[0], -1)
                bias.zero_()
                bias[:, 2 * transform.num_bins :] = math.log(math.expm1(1 - transform.min_derivative))


def initial_fraction_logit(seed, device="cpu"):
    fraction = np.random.RandomState(seed).uniform(0.01, 0.001)
    return torch.tensor(
        np.log(fraction / (1 - fraction)), dtype=torch.float64, device=device, requires_grad=True
    )


def background_log_prob(z, *, mass_conditioning=False, physical_inputs=False):
    if physical_inputs:
        # Cached frozen p_B in exactly the same preprocessed coordinates as p_S.
        # This column is bookkeeping and is never passed to the signal network.
        return z[..., -1]
    if mass_conditioning:
        z = z[..., :-1]
    return -0.5 * (z.square().sum(-1) + z.shape[-1] * math.log(2 * math.pi))


def signal_log_prob(model, inputs):
    if getattr(model, "physical_inputs", False):
        inputs = inputs[:, :-1]
    if getattr(model, "mass_conditioning", False):
        return model.log_prob(inputs[:, :-1], context=inputs[:, -1:])
    return model.log_prob(inputs)


def with_mass_context(z, mass):
    """Fixed SR coordinate; no fitted transform, truth labels, or test statistics."""
    mass = np.asarray(mass)
    if mass.shape != (len(z),) or not np.isfinite(mass).all():
        raise ValueError("Invalid mass context")
    return np.column_stack((z, (mass - 3.5) / .2)).astype(np.float32)


def residual_loss(signal_log_prob, background_log_density, fraction_logit):
    log_density = mixture_log_density(signal_log_prob, background_log_density, fraction_logit)
    if not torch.isfinite(log_density).all():
        raise FloatingPointError("Nonfinite residual likelihood; last completed epoch retained")
    return -log_density.mean()


def residual_optimizer(model, logit, options):
    groups = [{"params": list(model.parameters()), "weight_decay": options["weight_decay"]}]
    if logit.requires_grad:
        groups.append({"params": [logit], "weight_decay": 0.0})
    return torch.optim.AdamW(groups, lr=options["learning_rate"])


def real_sr_latents(rows, *, mass_conditioning=False, physical_inputs=False):
    widths = (8, 9) if physical_inputs else (7, 8)
    if rows.ndim != 2 or rows.shape[1] not in widths or rows.dtype != np.float32:
        raise ValueError("Expected float32 latent rows with four or five features")
    if not np.all(np.isfinite(rows[:, :-1])) or not np.all(np.isin(rows[:, -2], (0, 1))):
        raise ValueError("Invalid data/reference rows")
    selected = rows[:, -2] == 1
    real = rows[selected]
    if len(real) < 2 or not np.all((real[:, 0] > 3.3) & (real[:, 0] < 3.7)):
        raise ValueError("Expected at least two real signal-region training rows")
    z = np.ascontiguousarray(real[:, 1:-2])
    if physical_inputs:
        if not mass_conditioning:
            raise ValueError("Physical-input pilot requires mass conditioning")
        return np.column_stack((with_mass_context(z[:, :-1], real[:, 0]), z[:, -1])).astype(np.float32)
    return with_mass_context(z, real[:, 0]) if mass_conditioning else z


def train_epoch(model, logit, loader, optimizer=None, progress=None, *, gradient_clip_norm=1):
    model.train(optimizer is not None)
    total = 0.0
    events = 0
    with torch.set_grad_enabled(optimizer is not None):
        for index, (z,) in enumerate(loader):
            z = z.to(logit.device)
            if optimizer is not None:
                optimizer.zero_grad()
            loss = residual_loss(signal_log_prob(model, z),
                                 background_log_prob(z, mass_conditioning=getattr(model, "mass_conditioning", False),
                                                     physical_inputs=getattr(model, "physical_inputs", False)),
                                 logit)
            total += loss.item() * len(z)
            events += len(z)
            if optimizer is not None:
                loss.backward()
                if logit.requires_grad and (logit.grad is None or not torch.isfinite(logit.grad).all()):
                    raise FloatingPointError(
                        "Nonfinite residual mixture-fraction gradient; optimizer not updated"
                    )
                try:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), gradient_clip_norm, error_if_nonfinite=True
                    )
                except RuntimeError as exc:
                    raise FloatingPointError("Invalid residual flow gradient; optimizer not updated") from exc
                optimizer.step()
            if progress is not None:
                progress(index + 1, len(loader))
    if not events or events != len(loader.dataset):
        raise ValueError("Residual epoch must include every expected event")
    return total / events
