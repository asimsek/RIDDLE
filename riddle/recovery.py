import json
from pathlib import Path

import numpy as np
import torch

from .storage import (
    atomic_torch_save,
    write_json,
    rng_state,
    restore_rng,
    file_digest,
    verify_artifacts,
    persist_boundary,
    save_array,
)
from .worker_progress import emit_progress
from .resume import check_contract, record_transition


FLOW_CANDIDATES = 10


def _cpu_state_dict(model):
    return {
        key: value.detach().cpu().clone() if isinstance(value, torch.Tensor) else value
        for key, value in model.state_dict().items()
    }


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
        self.flow_candidates = []
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
        atomic_torch_save(self.path, state)
        self.state = state

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
            if (p.is_file() and p.suffix in (".npy", ".par", ".p"))
            or (p.is_file() and p.name.startswith("model_run"))
        }
        self.save(phase, "complete")

    @staticmethod
    def loaders(phase, values):
        names = (
            ("dataloader_train", "dataloader_test")
            if phase == "flow"
            else ("train_dataloader", "val_dataloader")
        )
        return [values[name] for name in names]

    def _offer_flow_candidate(self, epoch, validation_loss, model, model_file_name):
        key = (float(validation_loss), int(epoch))
        if len(self.flow_candidates) >= FLOW_CANDIDATES:
            worst = max((c["validation_loss"], c["epoch"]) for c in self.flow_candidates)
            if key >= worst:
                return
        self.flow_candidates.append({
            "epoch": int(epoch),
            "validation_loss": float(validation_loss),
            "filename": f"{model_file_name}_epoch_{epoch}.par",
            "sha256": None,
            "state": _cpu_state_dict(model),
        })
        self.flow_candidates.sort(key=lambda c: (c["validation_loss"], c["epoch"]))
        if len(self.flow_candidates) > FLOW_CANDIDATES:
            self.flow_candidates.pop()

    def _persist_flow_candidates(self):
        for candidate in self.flow_candidates:
            if candidate.get("sha256") is None:
                candidate["sha256"] = atomic_torch_save(
                    self.root / candidate["filename"], candidate.pop("state")
                )
        self.files = {c["filename"]: c["sha256"] for c in self.flow_candidates}
        return [
            {k: c[k] for k in ("epoch", "validation_loss", "filename", "sha256")}
            for c in self.flow_candidates
        ]

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

        if phase == "flow":
            validation_loss = float(losses[1][epoch + 1])
            self._offer_flow_candidate(epoch, validation_loss, model, values["model_file_name"])

        durable = persist_boundary(epoch, values["epochs"])
        if durable:
            candidates = None
            if phase == "flow":
                candidates = self._persist_flow_candidates()
                save_array(self.root / f"{values['model_file_name']}_train_losses.npy", losses[0])
                save_array(self.root / f"{values['model_file_name']}_val_losses.npy", losses[1])
            else:
                checkpoint = f"model_run0_ep{epoch}"
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
                **({"candidates": candidates} if candidates is not None else {}),
            )

        emit_progress(
            phase,
            "Train background flow" if phase == "flow" else "Train classifier",
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
        if phase == "flow":
            candidates = state.get("candidates")
            if not isinstance(candidates, list):
                raise ValueError("Flow recovery lacks validation-selected candidate inventory")
            self.flow_candidates = [dict(candidate) for candidate in candidates]
        restore_rng(state["rng"])
        self.next_epochs[phase] = state["next_epoch"]
        emit_progress(
            phase,
            "Resume background flow" if phase == "flow" else "Resume classifier",
            total=values["epochs"],
            completed=state["next_epoch"],
            initial=state["next_epoch"],
            unit="epoch",
        )
        return [np.array(loss, copy=True) for loss in state["losses"]]

    def next_epoch(self, phase):
        return self.next_epochs.get(phase, 0)

def _optimizer_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


class ClassifierFitRecovery:
    def __init__(self, root, identity, resume):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.resume_root = self.root / ".resume"
        self.resume_root.mkdir(parents=True, exist_ok=True)
        self.identity_path = self.resume_root / "identity.json"
        self.latest_path = self.resume_root / "latest.pt"
        self.identity = identity
        if self.identity_path.exists():
            if not resume:
                raise FileExistsError("AD baseline fit recovery exists; use --resume")
            previous = json.loads(self.identity_path.read_text())
            if previous != identity:
                raise ValueError("AD baseline fit recovery identity changed")
        else:
            write_json(self.identity_path, identity)

    def load(self, model, optimizer, train_loader, val_loader):
        if not self.latest_path.exists():
            return 0, [], [], []
        state = torch.load(self.latest_path, map_location="cpu", weights_only=False)
        if state.get("identity") != self.identity:
            raise ValueError("AD baseline recovery state identity changed")
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        _optimizer_to_device(optimizer, next(model.parameters()).device)
        if train_loader.generator is not None:
            train_loader.generator.set_state(state["train_generator"])
        if val_loader.generator is not None and state.get("val_generator") is not None:
            val_loader.generator.set_state(state["val_generator"])
        restore_rng(state["rng"])
        candidates = list(state.get("candidates", []))
        for candidate in candidates:
            path = self.root / candidate["filename"]
            if not path.is_file() or file_digest(path) != candidate["sha256"]:
                raise ValueError("AD baseline selected checkpoint recovery artifact changed")
        return int(state["next_epoch"]), list(state["train_bce"]), list(state["validation_bce"]), candidates

    def offer_candidate(self, epoch, validation_bce, model, candidates, selected):
        key = (float(validation_bce), int(epoch))
        ordered = sorted(candidates, key=lambda item: (item["validation_bce"], item["epoch"]))
        if len(ordered) >= selected and key >= (ordered[-1]["validation_bce"], ordered[-1]["epoch"]):
            return ordered
        filename = f"checkpoints/epoch_{epoch:03d}.pt"
        payload = {
            "epoch": int(epoch),
            "validation_bce": float(validation_bce),
            "model": _cpu_state_dict(model),
        }
        checksum = atomic_torch_save(self.root / filename, payload)
        ordered.append({"epoch": int(epoch), "validation_bce": float(validation_bce), "filename": filename, "sha256": checksum})
        ordered.sort(key=lambda item: (item["validation_bce"], item["epoch"]))
        while len(ordered) > selected:
            removed = ordered.pop()
            (self.root / removed["filename"]).unlink(missing_ok=True)
        return ordered

    def save_epoch(self, next_epoch, model, optimizer, train_loader, val_loader, train_bce, validation_bce, candidates):
        state = {
            "identity": self.identity,
            "next_epoch": int(next_epoch),
            "model": _cpu_state_dict(model),
            "optimizer": optimizer.state_dict(),
            "train_generator": train_loader.generator.get_state() if train_loader.generator is not None else None,
            "val_generator": val_loader.generator.get_state() if val_loader.generator is not None else None,
            "rng": rng_state(),
            "train_bce": list(train_bce),
            "validation_bce": list(validation_bce),
            "candidates": list(candidates),
        }
        atomic_torch_save(self.latest_path, state)

    def complete(self, train_bce, validation_bce, candidates, receipt):
        save_array(self.root / "train_bce.npy", np.asarray(train_bce, dtype=np.float64))
        save_array(self.root / "validation_bce.npy", np.asarray(validation_bce, dtype=np.float64))
        receipt = {
            **receipt,
            "train_bce_sha256": file_digest(self.root / "train_bce.npy"),
            "validation_bce_sha256": file_digest(self.root / "validation_bce.npy"),
            "selected_checkpoints": list(candidates),
        }
        write_json(self.root / "fit.json", receipt)
        return receipt
