import json
from pathlib import Path
import time
import numpy as np
import torch
from copy import deepcopy
from .settings import DEFAULTS, validate_residual
from riddle.storage import verify_artifacts
from riddle.progress import _duration
from riddle.worker_progress import ProgressStage, emit_message
from riddle.storage import atomic_write, write_json
from riddle.storage import digest, file_digest
from .model import PROTOCOL as SINGLE_PROTOCOL, background_log_prob, real_sr_latents
from .training import train_residual, residual_scores
from .integrity import SCIENTIFIC_VERSION
from .production import PRODUCTION_POLICY, require_complete_members, require_complete_ensemble

PROTOCOL = {
    **SINGLE_PROTOCOL,
    "name": "RIDDLE",
    "campaign_version": SCIENTIFIC_VERSION,
    "ensemble": "equal-weight mean signal density over all requested runs and ten validation-selected epochs per run",
    "splits": "80/20 resamples of development training rows; reserved validation reserved for configuration/cuts",
    "failures": "record numerical failures and reject incomplete production; retain finite near-zero-fraction fits",
}


class FitTiming:
    def __init__(self, epochs, *, started=None):
        self.epochs = epochs
        self.started = time.monotonic() if started is None else started
        self.seconds = self.measured_epochs = 0
        self.finalization_seconds = self.fits = 0
        self.fit_started = None

    def start_fit(self):
        self.fit_started = time.monotonic()

    def finish_fit(self, new_epochs, *, training_finished=None):
        if self.fit_started is not None and new_epochs > 0:
            finished = time.monotonic()
            training_finished = finished if training_finished is None else training_finished
            self.seconds += training_finished - self.fit_started
            self.finalization_seconds += finished - training_finished
            self.measured_epochs += new_epochs
            self.fits += 1
        self.fit_started = None

    def report(self, fit, total, pending):
        elapsed = time.monotonic() - self.started
        mean = (
            self.epochs * self.seconds / self.measured_epochs + self.finalization_seconds / self.fits
            if self.measured_epochs
            else None
        )
        eta = pending * mean if mean is not None else 0 if not pending else None
        formatted = lambda value: "unknown" if value is None else _duration(value)
        emit_message(
            f"RIDDLE fitting estimate after fit {fit}/{total}: elapsed={_duration(elapsed)}; mean_fit={formatted(mean)}; estimated_total={formatted(elapsed + eta if eta is not None else None)}; ETA={formatted(eta)}"
        )


def fractions(values):
    result = []
    for value in values:
        fraction = None if str(value) == "learned" else float(value)
        if fraction is not None and (not np.isfinite(fraction) or not 0 < fraction < 1):
            raise ValueError("Fractions must be 'learned' or numbers in (0,1)")
        if fraction in result:
            raise ValueError("Duplicate fraction configuration")
        result.append(fraction)
    if not result:
        raise ValueError("Provide at least one signal-fraction configuration")
    return result


def member_split(n, seed, index, *, batch_size=256):
    seeds = np.random.SeedSequence([int(seed), 1380533316, index]).generate_state(2).tolist()
    order = np.random.default_rng(seeds[0]).permutation(n)
    count = int(0.8 * n)
    if count < 2 or n - count < 2:
        raise ValueError("Too few development rows for an 80/20 split")
    if count % batch_size == 1:
        count -= 1
    return (order[:count], order[count:], seeds[1])


def heldout_nll(root, members, z, device, failures, configuration):
    background = background_log_prob(torch.from_numpy(z)).numpy().astype(np.float64)
    combined = None
    count = 0
    for member in list(members):
        densities = []
        try:
            for epoch, weight in zip(member["epochs"], member["signal_fractions"]):
                ratio = residual_scores(root / member["directory"], [epoch], z, device)
                log_weight = np.log(weight) if weight > 0 else -np.inf
                log_rest = np.log1p(-weight) if weight < 1 else -np.inf
                density = background + np.logaddexp(log_rest, log_weight + ratio)
                if not np.isfinite(density).all():
                    raise FloatingPointError("Nonfinite member validation density")
                densities.append(density)
        except FloatingPointError as error:
            member.update(
                status="numerical_failure",
                stage="configuration_validation",
                error=str(error),
                error_type=type(error).__name__,
            )
            write_json(root / member["directory"] / "fit.json", member)
            failures.append({"configuration": configuration, **member})
            write_json(root / "numerical_failures.json", {"failures": failures})
            members.remove(member)
            continue
        for density in densities:
            combined = density if combined is None else np.logaddexp(combined, density)
            count += 1
    if not count:
        return None
    value = -float(np.mean(combined - np.log(count)))
    if not np.isfinite(value):
        raise FloatingPointError("Nonfinite configuration validation likelihood")
    return value


def train_campaign(
    train,
    validation,
    output,
    *,
    epochs=DEFAULTS["riddle"]["epochs"],
    runs=DEFAULTS["riddle"]["runs"],
    seed=0,
    device="cpu",
    fraction_values=(None,),
    initialization="background",
    started=None,
    workers=1,
    io_workers=2,
    settings=None,
):
    settings = deepcopy(DEFAULTS["riddle"] if settings is None else settings)
    settings.update(epochs=epochs, runs=runs, initialization=initialization)
    settings = validate_residual(settings)
    timing = FitTiming(epochs, started=started)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    if min(runs, workers, io_workers) < 1:
        raise ValueError("Require positive runs/workers/io_workers")
    ztrain, zval = (real_sr_latents(train), real_sr_latents(validation))
    rows = np.column_stack(
        (np.full(len(ztrain), 3.5), ztrain, np.ones(len(ztrain)), np.zeros(len(ztrain)))
    ).astype(np.float32)
    identity = {
        "scientific_version": SCIENTIFIC_VERSION,
        "production_policy": PRODUCTION_POLICY,
        "training_sha256": digest(ztrain),
        "selection_sha256": digest(zval),
        "epochs": epochs,
        "runs": runs,
        "seed": seed,
        "fractions": list(fraction_values),
        "initialization": initialization,
        "settings": settings,
    }
    identity_path = output / "ensemble_inputs.json"
    if identity_path.exists() and json.loads(identity_path.read_text()) != identity:
        raise ValueError("Residual ensemble inputs/settings changed; use a new output")
    write_json(identity_path, identity)
    configs, failures = ([], [])
    total_fits = runs * len(fraction_values)
    tags = ["learned" if f is None else "fraction_" + str(f).replace(".", "p") for f in fraction_values]
    pending = {
        (c, i)
        for c, tag in enumerate(tags)
        for i in range(runs)
        if not (output / tag / f"run_{i:03d}" / "fit.json").exists()
    }
    if workers > 1:
        from .parallel import run_fits

        for c, tag in enumerate(tags):
            for i in range(runs):
                if (c, i) not in pending:
                    directory = output / tag / f"run_{i:03d}"
                    saved = json.loads((directory / "fit.json").read_text())
                    verify_artifacts(
                        directory,
                        saved["artifacts_sha256"],
                        f"RIDDLE fit {c * runs + i + 1}/{total_fits} | Verify saved fit",
                    )
        jobs = [
            {
                "fit": c * runs + i + 1,
                "index": i,
                "fraction": fraction_values[c],
                "relative": str(Path(tag) / f"run_{i:03d}"),
                "directory": str(output / tag / f"run_{i:03d}"),
            }
            for c, tag in enumerate(tags)
            for i in range(runs)
            if (c, i) in pending
        ]
        run_fits(
            rows,
            jobs,
            epochs=epochs,
            seed=seed,
            device=device,
            initialization=initialization,
            workers=workers,
            io_workers=io_workers,
            total=total_fits,
            started=timing.started,
            settings=settings,
        )
    for config_index, fraction in enumerate(fraction_values):
        tag = tags[config_index]
        members, histories = ([], [])
        for index in range(runs):
            fit_label = f"RIDDLE fit {config_index * runs + index + 1}/{total_fits}"
            with ProgressStage("residual_member_prepare", f"{fit_label} | Prepare"):
                relative = Path(tag) / f"run_{index:03d}"
                directory = output / relative
                directory.mkdir(parents=True, exist_ok=True)
                a, b, training_seed = member_split(
                    len(rows), seed, index, batch_size=settings["training"]["batch_size"]
                )
                split = {"training": a, "validation": b}
                split_path = directory / "split.npz"
                if split_path.exists():
                    with np.load(split_path) as saved:
                        if any((not np.array_equal(saved[k], v) for k, v in split.items())):
                            raise ValueError("Saved residual split changed")
                else:
                    from riddle.storage import save_npz

                    atomic_write(split_path, lambda p: save_npz(p, **split))
                status_path = directory / "fit.json"
                saved = json.loads(status_path.read_text()) if status_path.exists() else None
            if saved:
                verify_artifacts(directory, saved["artifacts_sha256"], f"{fit_label} | Verify saved fit")
            trained = saved is None
            if saved is None:
                try:
                    checkpoint_path = directory / ".resume/latest.pt"
                    checkpoint = (
                        torch.load(checkpoint_path, map_location=device, weights_only=False)
                        if checkpoint_path.exists()
                        else None
                    )
                    if checkpoint is not None:
                        verify_artifacts(
                            directory, checkpoint["files"], f"{fit_label} | Verify epoch checkpoints"
                        )
                    order, history = train_residual(
                        rows[a],
                        rows[b],
                        directory,
                        epochs=epochs,
                        seed=training_seed,
                        device=device,
                        checkpoint=checkpoint,
                        fraction=fraction,
                        initialization=initialization,
                        progress_label=f"{fit_label} | Train residual mixture",
                        on_training_start=timing.start_fit,
                        settings=settings,
                    )
                    training_finished = time.monotonic()
                    saved = {
                        "status": "completed",
                        "epochs": order,
                        "signal_fractions": [history[i]["signal_fraction"] for i in order],
                    }
                except FloatingPointError as error:
                    timing.finish_fit(0)
                    saved = {
                        "status": "numerical_failure",
                        "error": str(error),
                        "error_type": type(error).__name__,
                    }
                saved.update(
                    directory=str(relative),
                    seed=training_seed,
                    artifacts_sha256={
                        p.name: file_digest(p)
                        for p in directory.iterdir()
                        if p.is_file() and p != status_path
                    },
                )
                write_json(status_path, saved)
            if saved["status"] == "completed":
                members.append(saved)
                histories.append(json.loads((directory / "residual_losses.json").read_text())["history"])
            else:
                failures.append({"configuration": tag, **saved})
                write_json(output / "numerical_failures.json", {"failures": failures})
            pending.discard((config_index, index))
            if trained:
                if saved["status"] == "completed":
                    timing.finish_fit(
                        epochs - (checkpoint["epoch"] + 1 if checkpoint is not None else 0),
                        training_finished=training_finished,
                    )
                timing.report(config_index * runs + index + 1, total_fits, len(pending))
        checkpoints = settings["training"]["selected_checkpoints"]
        current = {"name": tag, "fraction": fraction, "members": members, "valid_runs": len(members)}
        pending_selection = {
            "status": "incomplete", "production_policy": PRODUCTION_POLICY,
            "requested_runs": runs, "checkpoints_per_run": checkpoints,
            "configurations": [*configs, current], "failures": failures,
        }
        write_json(output / "ensemble_selection.json", pending_selection)
        require_complete_members(members, runs, checkpoints, configuration=tag)
        histories = {member["directory"]: history for member, history in zip(members, histories)}
        nll = heldout_nll(output, members, zval, device, failures, tag)
        current["valid_runs"] = len(members)
        write_json(output / "ensemble_selection.json", pending_selection)
        require_complete_members(members, runs, checkpoints, configuration=tag)
        histories = [histories[member["directory"]] for member in members]
        config = {"name": tag, "fraction": fraction, "members": members, "valid_runs": len(members)}
        if members:
            config["selection_nll"] = nll
            config["history"] = [
                {
                    "epoch": e,
                    **{
                        key: float(np.mean([h[e][key] for h in histories]))
                        for key in ("train_nll", "validation_nll", "signal_fraction")
                    },
                }
                for e in range(epochs)
            ]
        configs.append(config)
        write_json(output / "ensemble_selection.json", {
            **pending_selection, "status": "in_progress", "configurations": configs,
        })
    chosen = min(configs, key=lambda c: c["selection_nll"])
    result = {
        "status": "completed",
        "production_policy": PRODUCTION_POLICY,
        "checkpoints_per_run": settings["training"]["selected_checkpoints"],
        "configurations": configs,
        "selected_configuration": chosen["name"],
        "members": chosen["members"],
        "failures": failures,
        "requested_runs": runs,
        "valid_runs": chosen["valid_runs"],
        "selected_checkpoints": sum((len(m["epochs"]) for m in chosen["members"])),
        "selection": "held-out reserved validation mixture NLL; no signal truth",
    }
    require_complete_ensemble(result)
    write_json(output / "ensemble_selection.json", result)
    write_json(output / "numerical_failures.json", {"failures": failures})
    write_json(
        output / "residual_losses.json", {"history": chosen["history"], "aggregation": "mean over valid runs"}
    )
    return result


def validate_normalization(output, features, device):
    """Check every selected density before evaluating data; preserve training RNG."""
    import torch
    from .production import validate_density_ratio

    root = Path(output)
    selection = json.loads((root / "ensemble_selection.json").read_text())
    require_complete_ensemble(selection)
    seed, count = 3407, 8192
    reference = np.random.default_rng(seed).standard_normal((count, features)).astype(np.float32)
    members = selection["members"]
    tests = sum(len(m["epochs"]) + 1 for m in members)
    audit = dict(schema=1, status="checking", reference="independent standard normal",
                 reference_seed=seed, reference_samples=count, features=features,
                 familywise_false_rejection_bound=1e-12, members=[])
    try:
        with torch.random.fork_rng(devices=[] if str(device) == "cpu" else None):
            for member in members:
                checkpoints = []
                scores = residual_scores(root / member["directory"], member["epochs"], reference, device,
                                         normalization_checks=checkpoints, normalization_tests=tests)
                audit["members"].append(dict(directory=member["directory"], checkpoints=checkpoints,
                    ensemble=validate_density_ratio(scores, stage=f"RIDDLE {member['directory']} ensemble", tests=tests)))
    except (ValueError, FloatingPointError) as error:
        audit.update(status="failed", error=str(error))
        write_json(root / "normalization_check.json", audit)
        raise
    audit["status"] = "passed"
    write_json(root / "normalization_check.json", audit)
    return audit


def ensemble_predict(output, z, device):
    root = Path(output)
    selection = json.loads((root / "ensemble_selection.json").read_text())
    require_complete_ensemble(selection)
    combined = None
    for member in selection["members"]:
        ratio = residual_scores(root / member["directory"], member["epochs"], z, device)
        combined = ratio if combined is None else np.logaddexp(combined, ratio)
    return combined - np.log(len(selection["members"]))
