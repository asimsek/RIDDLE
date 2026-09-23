import functools
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from .storage import atomic_write, write_json, rng_state, restore_rng, file_digest, verify_artifacts
from .worker_progress import emit_progress
from .resume import check_contract, record_transition


class EpochRecovery:
    def __init__(self, root, contract, resume=False, *, allow_code_change=False, allow_device_change=False):
        if (allow_code_change or allow_device_change) and not resume:
            raise ValueError("Resume override options require --resume")
        self.root = Path(root)
        self.path = self.root / ".resume/stages.pt"
        self.manifest = self.root / ".resume/contract.json"
        self.root.mkdir(parents=True, exist_ok=True)
        self.save_torch = torch.save
        self.done, self.files, self.next_epochs = [], {}, {}
        self.state = None
        self.resume = resume
        self.classifier_run = None
        if self.manifest.exists():
            if not resume:
                raise FileExistsError("Work already started; use --resume")
            previous = json.loads(self.manifest.read_text())
            changes = check_contract(
                previous, contract, allow_code_change=allow_code_change,
                allow_device_change=allow_device_change,
            )
            if self.path.exists():
                self.state = torch.load(self.path, map_location="cpu", weights_only=False)
                self.done = self.state["done"][:]
                self.files = dict(self.state["files"])
                verify_artifacts(self.root, self.files, "Verify training recovery")
            if changes:
                record_transition(
                    self.root / ".resume/resume_history.json", previous, contract, changes,
                    action="epoch_recovery_accepted",
                )
                write_json(self.manifest, contract)
        else:
            if any(p.name != ".resume" for p in self.root.iterdir()):
                raise ValueError("Unrecognized nonempty training directory")
            write_json(self.manifest, contract)

    def save(self, phase, kind, **extra):
        state = {
            "phase": phase,
            "kind": kind,
            "done": self.done[:],
            "files": self.files.copy(),
            "rng": rng_state(),
            **extra,
        }
        atomic_write(self.path, lambda p: self.save_torch(state, p))

    def stage(self, phase, function):
        if phase in self.done:
            if self.state and self.state["phase"] == phase:
                restore_rng(self.state["rng"])
            return
        if self.state and self.state["phase"] == phase and self.state["kind"] == "start":
            restore_rng(self.state["rng"])
        if not self.state or self.state["phase"] != phase:
            self.save(phase, "start")
        function()
        self.done.append(phase)
        self.files = {
            p.name: file_digest(p)
            for p in sorted(self.root.iterdir())
            if p.is_file()
            and p.suffix in (".npy", ".par", ".p")
            or p.is_file()
            and p.name.startswith("model_run")
        }
        self.save(phase, "complete")

    def classifier_fits(self, function):
        signature = inspect.signature(function)

        @functools.wraps(function)
        def train(*args, **kwargs):
            values = signature.bind(*args, **kwargs)
            values.apply_defaults()
            prefix = Path(values.arguments["save_model"])
            if prefix.parent.resolve() != self.root.resolve() or not prefix.name.startswith("model_run"):
                raise ValueError("Unexpected upstream classifier checkpoint prefix")
            index = int(prefix.name.removeprefix("model_run"))
            path = self.root / ".resume" / f"classifier_run{index}.pt"
            previous = self.path, self.state, self.next_epochs, self.classifier_run
            try:
                state = None
                if path.exists():
                    if not self.resume:
                        raise FileExistsError("Classifier fit already exists; use --resume")
                    state = torch.load(path, map_location="cpu", weights_only=False)
                    verify_artifacts(self.root, state["files"], "Verify classifier fit recovery")
                    self.files.update(state["files"])
                elif index == 0 and self.state and self.state["phase"] == "classifier":
                    state = self.state
                self.path, self.state, self.next_epochs, self.classifier_run = path, state, {}, index
                if state and state["kind"] == "complete":
                    restore_rng(state["rng"])
                    return tuple(np.array(loss, copy=True) for loss in state["losses"])
                if state and state["kind"] == "start":
                    restore_rng(state["rng"])
                elif not state:
                    self.save("classifier", "start")
                phase, label = self.progress_identity("classifier")
                emit_progress(phase, label, total=values.arguments["epochs"], unit="epoch", completed=0)
                losses = function(*args, **kwargs)
                self.save("classifier", "complete", losses=losses)
                return losses
            finally:
                self.path, self.state, self.next_epochs, self.classifier_run = previous

        return train

    def progress_identity(self, phase, *, resume=False):
        action = "Resume" if resume else "Train"
        if phase == "flow":
            return phase, action + " background flow"
        if self.classifier_run is None:
            return phase, action + " classifier"
        return f"classifier_run{self.classifier_run}", f"{action} classifier fit {self.classifier_run + 1}"

    @staticmethod
    def loaders(phase, values):
        names = (
            ("dataloader_train", "dataloader_test")
            if phase == "flow"
            else ("train_dataloader", "val_dataloader")
        )
        return [values[name] for name in names]

    def save_epoch(self, phase, epoch, values):
        model, optimizer = values["model"], values["optimizer"]
        losses = [
            values[n]
            for n in (("train_losses", "val_losses") if phase == "flow" else ("train_loss", "val_loss"))
        ]
        loaders = [
            {
                "verified": getattr(loader, "_runtime_gather_verified", None),
                "generator": loader.generator.get_state() if loader.generator is not None else None,
            }
            for loader in self.loaders(phase, values)
        ]
        attributes = {
            name: {
                k: getattr(module, k)
                for k in ("training", "momentum", "batch_mean", "batch_var", "num_inputs")
                if hasattr(module, k)
            }
            for name, module in model.named_modules()
        }
        checkpoint = (
            f"{values['model_file_name']}_epoch_{epoch}.par"
            if phase == "flow" else f"{Path(values['save_model']).name}_ep{epoch}"
        )
        self.files[checkpoint] = file_digest(self.root / checkpoint)
        self.save(
            phase,
            "epoch",
            next_epoch=epoch + 1,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            attributes=attributes,
            losses=losses,
            loaders=loaders,
        )
        emit_progress(
            *self.progress_identity(phase),
            total=values["epochs"],
            completed=epoch + 1,
            unit="epoch",
            report_every=1,
            train_loss=float(losses[0][epoch + 1 if phase == "flow" else epoch]),
            validation_loss=float(losses[1][epoch + 1 if phase == "flow" else epoch]),
        )

    def restore_epoch(self, phase, values):
        state = self.state
        if not state or state["phase"] != phase or state["kind"] != "epoch":
            return None
        model = values["model"]
        device = next(model.parameters()).device
        model.load_state_dict(state["model"])
        values["optimizer"].load_state_dict(state["optimizer"])
        for name, module in model.named_modules():
            for key, value in state["attributes"][name].items():
                setattr(module, key, value.to(device) if isinstance(value, torch.Tensor) else value)
        for loader, saved in zip(self.loaders(phase, values), state["loaders"]):
            loader._runtime_gather_verified = saved["verified"]
            if saved["generator"] is not None:
                loader.generator.set_state(saved["generator"])
        restore_rng(state["rng"])
        self.next_epochs[phase] = state["next_epoch"]
        emit_progress(
            *self.progress_identity(phase, resume=True),
            total=values["epochs"],
            completed=state["next_epoch"],
            initial=state["next_epoch"],
            unit="epoch",
        )
        return [np.array(loss, copy=True) for loss in state["losses"]]

    def next_epoch(self, phase):
        return self.next_epochs.get(phase, 0)
