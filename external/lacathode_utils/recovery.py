import copy
import functools
import inspect
import json
from pathlib import Path

import numpy as np
import torch

from .storage import atomic_write, write_json, rng_state, restore_rng, file_digest, verify_artifacts, save_array
from riddle.storage import persist_boundary
from .worker_progress import emit_progress
from .resume import check_contract, record_transition


PROTOCOL = "buffered_all_epoch_selection_v1"


def cpu_snapshot(value):
    if isinstance(value, torch.nn.Module):
        return copy.deepcopy(value).cpu()
    result = copy.copy(value)
    for key, item in value.items():
        result[key] = item.detach().cpu().clone() if isinstance(item, torch.Tensor) else copy.deepcopy(item)
    if hasattr(value, "_metadata"):
        result._metadata = copy.deepcopy(value._metadata)
    return result


def save_checkpoint(path, value):
    def write(temporary):
        with temporary.open("wb") as stream:
            torch.save(value, stream)

    atomic_write(path, write)


class CheckpointBuffer:
    def __init__(self, root, phase, prefix, epochs, state=None):
        self.root = Path(root).resolve()
        self.phase, self.prefix, self.epochs = phase, str(prefix), int(epochs)
        if phase not in ("flow", "classifier") or self.epochs <= 10:
            raise ValueError("LaCATHODE checkpoint selection requires more than ten epochs")
        if Path(self.prefix).name != self.prefix or not self.prefix:
            raise ValueError("Unexpected LaCATHODE checkpoint prefix")
        self.candidates, self.last, self.validation_losses = {}, None, []
        self.finished = False
        if state is not None:
            saved = state.get("checkpoint_buffer")
            if saved is None:
                self.restore_legacy(state)
            else:
                expected = (PROTOCOL, phase, self.prefix, self.epochs)
                if tuple(saved.get(key) for key in ("protocol", "phase", "prefix", "epochs")) != expected:
                    raise ValueError("LaCATHODE checkpoint buffer contract changed")
                self.candidates = dict(saved["candidates"])
                self.last = saved["last"]
                self.validation_losses = list(saved["validation_losses"])
                self.finished = saved["finished"]
            if len(self.validation_losses) != state["next_epoch"]:
                raise ValueError("LaCATHODE checkpoint buffer epoch differs from recovery")

    def filename(self, epoch):
        return f"{self.prefix}_epoch_{epoch}.par" if self.phase == "flow" else f"{self.prefix}_ep{epoch}"

    def owns(self, name):
        if self.phase == "flow":
            return name.startswith(self.prefix + "_epoch_") and name.endswith(".par")
        return name.startswith(self.prefix + "_ep")

    def eligible(self, loss):
        if len(self.validation_losses) < 10 or not np.isfinite(loss):
            return True
        threshold = np.sort(self.validation_losses)[9]
        return np.isnan(threshold) or loss <= threshold

    def offer(self, value, path, epoch, loss):
        epoch, loss = int(epoch), float(loss)
        if self.finished or epoch != len(self.validation_losses) or not 0 <= epoch < self.epochs:
            raise ValueError("LaCATHODE checkpoint epoch is out of sequence")
        if Path(path).resolve() != self.root / self.filename(epoch):
            raise ValueError("Unexpected LaCATHODE checkpoint path")
        eligible = self.eligible(loss)
        candidate = {"epoch": epoch, "model": cpu_snapshot(value)} if eligible or persist_boundary(epoch, self.epochs) else None
        if eligible:
            self.candidates[epoch] = candidate
        self.last = candidate
        self.validation_losses.append(loss)
        if len(self.validation_losses) >= 10:
            threshold = np.sort(self.validation_losses)[9]
            if not np.isnan(threshold):
                self.candidates = {
                    index: item for index, item in self.candidates.items()
                    if self.validation_losses[index] <= threshold or not np.isfinite(self.validation_losses[index])
                }

    def restore_legacy(self, state):
        count = int(state["next_epoch"])
        losses = np.asarray(state["losses"][1])
        self.validation_losses = losses[1:count + 1].tolist() if self.phase == "flow" else losses[:count].tolist()
        if not 0 < count <= self.epochs or len(self.validation_losses) != count:
            raise ValueError("Invalid legacy LaCATHODE recovery epoch")
        threshold = np.sort(self.validation_losses)[min(9, count - 1)]
        required = {
            index for index, loss in enumerate(self.validation_losses)
            if np.isnan(threshold) or loss <= threshold or not np.isfinite(loss)
        } | {count - 1}
        for index in sorted(required):
            name = self.filename(index)
            if name not in state["files"]:
                raise ValueError("Legacy LaCATHODE recovery is missing a selection candidate")
            candidate = {
                "epoch": index,
                "model": torch.load(self.root / name, map_location="cpu", weights_only=False),
                "sha256": state["files"][name],
            }
            self.candidates[index] = candidate
            if index == count - 1:
                self.last = candidate

    def state(self):
        return {
            "protocol": PROTOCOL,
            "phase": self.phase,
            "prefix": self.prefix,
            "epochs": self.epochs,
            "candidates": self.candidates,
            "last": self.last,
            "validation_losses": self.validation_losses,
            "finished": self.finished,
        }

    def finish(self, losses):
        if self.finished:
            return {}
        if len(self.validation_losses) != self.epochs or self.last is None:
            raise ValueError("LaCATHODE checkpoint buffer has incomplete training")
        losses = np.asarray(losses)
        trained = losses[1:] if self.phase == "flow" else losses
        if not np.array_equal(trained, self.validation_losses, equal_nan=True):
            raise ValueError("LaCATHODE checkpoint selection losses changed")
        offset = 1 if self.phase == "flow" else 0
        required = set((np.argpartition(losses, 10)[:10] - offset).tolist()) | {self.epochs - 1}
        if self.phase == "flow":
            required.add(int(np.argpartition(losses, 1)[0]) - 1)
        files = {}
        for index in sorted(required - {-1}):
            candidate = self.last if index == self.epochs - 1 else self.candidates.get(index)
            if candidate is None or candidate["epoch"] != index:
                raise ValueError("Missing all-epoch LaCATHODE selection candidate")
            path = self.root / self.filename(index)
            if candidate.get("sha256") is not None:
                if file_digest(path) != candidate["sha256"]:
                    raise ValueError("Legacy LaCATHODE selection candidate changed")
            else:
                save_checkpoint(path, candidate["model"])
            files[path.name] = file_digest(path)
        self.candidates.clear()
        self.last = None
        self.finished = True
        return files


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
        self.checkpoint_buffer = None
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
            previous = self.path, self.state, self.next_epochs, self.classifier_run, self.checkpoint_buffer
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
                self.path, self.state, self.next_epochs, self.classifier_run, self.checkpoint_buffer = previous

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

    def start_checkpoint_buffer(self, phase, values):
        prefix = values["model_file_name"] if phase == "flow" else Path(values["save_model"]).name
        state = self.state if self.next_epoch(phase) else None
        self.checkpoint_buffer = CheckpointBuffer(self.root, phase, prefix, values["epochs"], state)
        self.files = {name: checksum for name, checksum in self.files.items() if not self.checkpoint_buffer.owns(name)}
        if self.checkpoint_buffer.finished:
            self.files.update({name: checksum for name, checksum in state["files"].items() if self.checkpoint_buffer.owns(name)})

    def buffer_checkpoint(self, value, path, phase, epoch, values):
        loss = values["val_losses"][epoch + 1] if phase == "flow" else values["val_loss"][epoch]
        self.checkpoint_buffer.offer(value, path, epoch, loss)

    @staticmethod
    def save_loss_array(path, losses, epoch, epochs):
        if persist_boundary(epoch, epochs):
            save_array(path, losses)

    def finish_checkpoint_buffer(self, phase, values):
        losses = values["val_losses"] if phase == "flow" else values["val_loss"]
        self.files.update(self.checkpoint_buffer.finish(losses))
        self.checkpoint_buffer = None

    def save_epoch(self, phase, epoch, values):
        losses = [
            values[n]
            for n in (("train_losses", "val_losses") if phase == "flow" else ("train_loss", "val_loss"))
        ]
        if persist_boundary(epoch, values["epochs"]):
            if epoch + 1 == values["epochs"]:
                self.files.update(self.checkpoint_buffer.finish(losses[1]))
            self.save_epoch_state(phase, epoch, values, losses)
        emit_progress(
            *self.progress_identity(phase),
            total=values["epochs"],
            completed=epoch + 1,
            unit="epoch",
            report_every=1,
            train_loss=float(losses[0][epoch + 1 if phase == "flow" else epoch]),
            validation_loss=float(losses[1][epoch + 1 if phase == "flow" else epoch]),
        )

    def save_epoch_state(self, phase, epoch, values, losses):
        model, optimizer = values["model"], values["optimizer"]
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
        self.save(
            phase,
            "epoch",
            next_epoch=epoch + 1,
            model=model.state_dict(),
            optimizer=optimizer.state_dict(),
            attributes=attributes,
            losses=losses,
            loaders=loaders,
            checkpoint_buffer=self.checkpoint_buffer.state(),
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
