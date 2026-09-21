"""Shared numerical protocol for the standalone and pinned-upstream adapters."""

from pathlib import Path
import json
import numpy as np
import torch

SCIENTIFIC_VERSION = 4


def require_finite(value, stage):
    if torch.is_tensor(value):
        bad = ~torch.isfinite(value)
        count = int(torch.count_nonzero(bad).item())
        if not count:
            return
        rows = (
            int(torch.count_nonzero(bad.reshape(len(value), -1).any(1)).item()) if value.ndim else count
        )
    else:
        value = np.asarray(value)
        bad = ~np.isfinite(value)
        count = int(np.count_nonzero(bad))
        if not count:
            return
        rows = (
            int(np.count_nonzero(bad.reshape(len(value), -1).any(1)))
            if value.ndim and len(value)
            else count
        )
    if count:
        raise FloatingPointError(
            f"{stage}: {count} nonfinite values in {rows} events; no events removed"
        )


def ordered_epochs(losses, count, *, initial_entry=False):
    losses = np.asarray(losses)
    if losses.ndim != 1 or type(count) is not int or count < 1:
        raise ValueError("Invalid checkpoint selection request")
    require_finite(losses, "Checkpoint validation losses")
    # Entry zero is an untrained diagnostic, never a saved checkpoint.
    trained = losses[1:] if initial_entry else losses
    if len(trained) < count:
        raise ValueError("Not enough trained checkpoints for selection")
    return np.argsort(trained, kind="stable")[:count].tolist()


def mixture_log_density(log_signal, log_background, logit):
    require_finite(log_signal, "Residual signal log density")
    require_finite(log_background, "Residual background log density")
    require_finite(logit, "Residual fraction logit")
    return torch.logaddexp(
        torch.nn.functional.logsigmoid(logit) + log_signal,
        torch.nn.functional.logsigmoid(-logit) + log_background,
    )


def flow_loss(model, data_loader, device, correct_logit=None, *, diagnostics=None, role="validation"):
    """All-event NLL. Large finite log likelihoods are retained, not clipped."""
    model.eval()
    counts = dict(
        role=role,
        expected_events=len(data_loader.dataset),
        total_events=0,
        finite_events=0,
        nan_events=0,
        inf_events=0,
        large_finite_events=0,
        rejected_events=0,
        policy="fail_on_nonfinite; retain_all_finite",
    )
    total = corrected_total = 0.0
    with torch.no_grad():
        for batch in data_loader:
            data, condition = batch[0].to(device), batch[1].float().to(device)
            values = model.log_probs(data, condition).reshape(-1)
            if len(values) != len(data):
                raise ValueError("Expected one background log likelihood per event")
            counts["total_events"] += len(values)
            counts["finite_events"] += int(torch.isfinite(values).sum().item())
            counts["nan_events"] += int(torch.isnan(values).sum().item())
            counts["inf_events"] += int(torch.isinf(values).sum().item())
            counts["large_finite_events"] += int(
                (torch.isfinite(values) & (values.abs() >= 1000)).sum().item()
            )
            total -= values.double().sum().item()
            if correct_logit is not None:
                if not bool(((data > 0) & (data < 1)).all()):
                    raise FloatingPointError("Logit Jacobian outside (0,1); no validation rows removed")
                corrected = values + torch.log(correct_logit * data * (1 - data)).sum(1)
                require_finite(corrected, "Background validation Jacobian")
                corrected_total -= corrected.double().sum().item()
    counts["rejected_events"] = counts["nan_events"] + counts["inf_events"]
    counts["valid"] = bool(
        counts["rejected_events"] == 0
        and counts["total_events"] == counts["expected_events"]
        and counts["total_events"] > 0
        and np.isfinite(total)
    )
    if diagnostics is not None:
        from .storage import write_json

        path = Path(diagnostics)
        history = (
            json.loads(path.read_text())
            if path.exists()
            else {"scientific_version": SCIENTIFIC_VERSION, "evaluations": []}
        )
        history["evaluations"].append(counts)
        write_json(path, history)
    if not counts["valid"]:
        raise FloatingPointError(f"Invalid background {role}; checkpoint rejected: {counts}")
    mean = total / counts["total_events"]
    return (corrected_total / counts["total_events"], mean) if correct_logit is not None else (mean,)


def install_flow_validation(module, diagnostics=None):
    roles = {}

    def compute(model, data_loader, device, correct_logit=None):
        roles.setdefault(id(data_loader), "training" if not roles else "validation")
        return flow_loss(
            model,
            data_loader,
            device,
            correct_logit,
            diagnostics=diagnostics,
            role=roles[id(data_loader)],
        )

    module.compute_loss_over_batches = compute


def train_flow_epoch(
    model, optimizer, data_loader, device, *, batch_norm_class, verbose=True, data_std=None
):
    model.train()
    total = corrected_total = 0.0
    count = 0
    for batch in data_loader:
        data = batch[0].to(device)
        condition = batch[1].float().to(device) if len(batch) > 1 else None
        require_finite(data, "Background training inputs")
        optimizer.zero_grad()
        losses = -model.log_probs(data, condition).reshape(-1)
        require_finite(losses, "Background training NLL")
        total += losses.detach().double().sum().item()
        count += len(data)
        if data_std is not None:
            if not bool(((data > 0) & (data < 1)).all()):
                raise FloatingPointError("Background training Jacobian outside (0,1); no rows removed")
            corrected = losses - torch.log(data_std * data * (1 - data)).sum(1)
            require_finite(corrected, "Background training Jacobian")
            corrected_total += corrected.detach().double().sum().item()
        losses.mean().backward()
        checks = [torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None]
        if checks and not bool(torch.stack(checks).all()):
            raise FloatingPointError("Nonfinite background training gradient; optimizer not updated")
        optimizer.step()
    if count != len(data_loader.dataset) or not count:
        raise ValueError("Background training loader did not visit every event")
    modules = [m for m in model.modules() if isinstance(m, batch_norm_class)]
    for module in modules:
        module.momentum = 0
    if modules:
        with torch.no_grad():
            model(
                data_loader.dataset.tensors[0].to(device),
                data_loader.dataset.tensors[1].to(device).float(),
            )
        for module in modules:
            module.momentum = 1
    return (corrected_total / count, total / count) if data_std is not None else (total / count,)
