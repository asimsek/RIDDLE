from pathlib import Path
from copy import deepcopy

import numpy as np
import torch

from riddle.storage import seed_start, save_array, write_json
from riddle.epoch_hook import install_epoch_recovery
from riddle.acceleration import install_validation_counts
from riddle.worker_progress import ProgressStage
from .preprocessing import LHCORD_data_handler, load_dataset, stack_data
from .density_estimator import DensityEstimator
from . import flow_training
from .settings import DEFAULTS
from .integrity import ordered_epochs, require_finite


def require_common_mapping(selection):
    training = selection.get("training_mapping_epoch")
    inference = selection.get("inference_mapping_epoch")
    if type(inference) is not int or not 0 <= inference < selection.get("trained_epochs", 100):
        raise ValueError("Invalid frozen flow epoch")
    if type(training) is not int or training != inference:
        raise ValueError(
            "RIDDLE requires identical training and inference flow checkpoints; "
            f"selected training={training}, inference={inference}. Selection rules are unchanged."
        )


def flow_selection(output):
    losses = np.load(output / "riddle_model_val_losses.npy")
    if losses.dtype != np.float32 or not np.isfinite(losses).all() or len(losses) <= 10:
        raise ValueError("Invalid initial-plus-trained flow validation loss array")
    mapping = ordered_epochs(losses, 10, initial_entry=True)
    inference = mapping[0]
    result = {
        "training_mapping_epoch": mapping[0],
        "inference_mapping_epoch": inference,
        "ordered_mapping_epochs": mapping,
        "trained_epochs": len(losses) - 1,
        "criterion": "lowest all-event validation NLL among trained checkpoints; stable epoch tie-break",
    }
    write_json(output / "flow_selection.json", result)
    require_common_mapping(result)
    return result


def make_data(data, device, batch_size=256):
    return LHCORD_data_handler(
        *(
            str(data / name)
            for name in (
                "innerdata_train.npy",
                "innerdata_val.npy",
                "outerdata_train.npy",
                "outerdata_val.npy",
            )
        ),
        None,
        batch_size=batch_size,
        device=device,
    )


def prepare(data, output, seed, device, recovery, *, settings=None):
    data, output = Path(data), Path(output)
    settings = deepcopy(DEFAULTS["background"] if settings is None else settings)
    configuration = settings["configuration"]
    configuration["num_inputs"] = np.load(data / "outerdata_train.npy", mmap_mode="r").shape[1] - 2
    write_json(output / "background_settings.json", settings)
    install_validation_counts(flow_training, output / "validation_diagnostics.json")
    install_epoch_recovery(flow_training, "train_ANODE", "flow", recovery)
    seed_start(seed)

    def train():
        with ProgressStage("flow_setup", "Prepare background flow"):
            handler = make_data(data, device, settings["batch_size"])
            handler.preprocess_ANODE_data(no_logit=False, no_mean_shift=False)
            estimator = DensityEstimator(configuration, device=device, verbose=False, bound=False)
        with ProgressStage("flow", "Train background flow", settings["epochs"], "epoch"):
            flow_training.train_ANODE(
                estimator.model,
                estimator.optimizer,
                handler.outer_ANODE_datadict_train["loader"],
                handler.outer_ANODE_datadict_test["loader"],
                "riddle_model",
                settings["epochs"],
                savedir=str(output),
                device=device,
                verbose=False,
                no_logit=False,
                data_std=handler.outer_ANODE_datadict_train["std2_logit_fix"],
            )

    recovery.stage("flow", train)
    selection = flow_selection(output)

    def create():
        with torch.no_grad(), ProgressStage("latents", "Build development latents"):
            handler = make_data(data, device, settings["batch_size"])
            handler.preprocess_ANODE_data(fiducial_cut=False, no_logit=False, no_mean_shift=False)
            models = [
                DensityEstimator(
                    configuration,
                    eval_mode=True,
                    load_path=str(output / f"riddle_model_epoch_{e}.par"),
                    device=device,
                    verbose=False,
                    bound=False,
                ).model
                for e in selection["ordered_mapping_epochs"]
            ]
            gaussian = np.random.normal(
                0, 1, (settings["reference_samples"], handler.inner_ANODE_datadict_train["tensor"].shape[1])
            )
            model = models[0]
            train, validation = development_rows(handler, model, gaussian)
            save_array(output / "training_latents.npy", train)
            save_array(output / "validation_latents.npy", validation)

    recovery.stage("latents", create)
    return selection


def development_rows(handler, model, gaussian):
    real = []
    with torch.no_grad():
        for d in (handler.inner_ANODE_datadict_train, handler.inner_ANODE_datadict_test):
            z = model(d["tensor2"], d["labels"])[0].detach().cpu().numpy()
            real.append(
                stack_data(z, d["labels"].cpu().numpy(), sig_labels=d["sigorbg"].cpu().numpy(), samples=False)
            )
    samples = stack_data(gaussian, np.ones(len(gaussian)), sig_labels=None, samples=True)
    fraction = len(real[0]) / (len(real[0]) + len(real[1]))
    count = int(fraction * len(samples))
    train = np.concatenate((samples[:count], real[0])).astype("float32")
    test = np.concatenate((samples[count:], real[1])).astype("float32")
    np.random.seed(42)
    np.random.shuffle(train)
    np.random.shuffle(test)
    validation = test[: int(2.0 / 5 * len(test))]
    return train, validation


class Mapper:
    def __init__(self, data, output, epoch, device):
        import json

        selection = json.loads((Path(output) / "flow_selection.json").read_text())
        require_common_mapping(selection)
        if epoch != selection["inference_mapping_epoch"]:
            raise ValueError("Requested inference checkpoint differs from the verified flow selection")
        self.device = device
        self.reference = load_dataset(np.load(Path(data) / "outerdata_train.npy").astype("float32"))
        settings = json.loads((Path(output) / "background_settings.json").read_text())
        self.model = DensityEstimator(settings["configuration"], eval_mode=True).model
        weights = torch.load(
            Path(output) / f"riddle_model_epoch_{epoch}.par", map_location="cpu", weights_only=True
        )
        self.model.load_state_dict(weights)
        self.model.to(device).eval().requires_grad_(False)

    def physical(self, rows):
        """Preprocessed physical features and frozen log p_B in that coordinate system."""
        prepared = load_dataset(rows.astype("float32"), external_datadict=self.reference)
        x, m = prepared["tensor2"], prepared["labels"]
        densities = []
        with torch.no_grad():
            for offset in range(0, len(x), 8192):
                densities.append(self.model.log_probs(x[offset:offset+8192].to(self.device),
                    m[offset:offset+8192].to(self.device)).flatten().cpu().numpy())
        if not densities:
            raise ValueError("No physical events remain inside the background domain")
        result = np.column_stack((x.cpu().numpy(), np.concatenate(densities))).astype(np.float32)
        require_finite(result, "Physical background densities")
        return result, prepared["mask"].cpu().numpy()

    def physical_development(self, data, reference_samples):
        """Reproduce development_rows membership/order, replacing z with x and log p_B."""
        real = []
        for suffix in ("train", "val"):
            original = np.load(Path(data) / f"innerdata_{suffix}.npy").astype(np.float32)
            features, mask = self.physical(original)
            real.append(np.column_stack((original[mask, 0], features, np.ones(mask.sum()),
                                         original[mask, -1])).astype(np.float32))
        # Keep reference-row positions fixed so compared protocols reserve identical real events.

        placeholders = np.zeros((reference_samples, real[0].shape[1]), dtype=np.float32)
        count = int(len(real[0]) / (len(real[0]) + len(real[1])) * reference_samples)
        train = np.concatenate((placeholders[:count], real[0]))
        validation = np.concatenate((placeholders[count:], real[1]))
        rng = np.random.RandomState(42)
        rng.shuffle(train)
        rng.shuffle(validation)
        return train, validation[:int(2.0 / 5 * len(validation))]

    def physical_reference(self, validation, count=8192):
        """Independent samples from frozen p_B(x|m), at reserved-validation masses."""
        from .model import with_mass_context
        masses = np.random.default_rng(3408).choice(validation[validation[:, -2] == 1, 0], count)
        dimensions = validation.shape[1] - 4
        noise = np.random.default_rng(3407).standard_normal((count, dimensions)).astype(np.float32)
        outputs = []
        with torch.no_grad():
            for offset in range(0, count, 1024):
                m = torch.from_numpy(masses[offset:offset+1024, None]).to(self.device)
                x = self.model.sample(noise=torch.from_numpy(noise[offset:offset+1024]).to(self.device), cond_inputs=m)
                log_b = self.model.log_probs(x, m).flatten().cpu().numpy()
                outputs.append(np.column_stack((with_mass_context(x.cpu().numpy(), m.cpu().numpy().ravel()), log_b)))
        result = np.concatenate(outputs).astype(np.float32)
        require_finite(result, "Frozen physical background reference")
        return result

    def map(self, rows):
        prepared = load_dataset(rows.astype("float32"), external_datadict=self.reference)
        x, m = prepared["tensor2"], prepared["labels"]
        outputs = []
        with (
            torch.no_grad(),
            ProgressStage("map", "Map events through frozen flow", len(x), "event") as progress,
        ):
            for offset in range(0, len(x), 8192):
                outputs.append(
                    self.model(
                        x[offset : offset + 8192].to(self.device), m[offset : offset + 8192].to(self.device)
                    )[0]
                    .cpu()
                    .numpy()
                )
                progress.update(min(offset + 8192, len(x)))
        if not outputs:
            raise ValueError("No events remain inside the flow domain")
        result = np.concatenate(outputs)
        require_finite(result, "Frozen background mapping")
        return result, prepared["mask"].cpu().numpy()
