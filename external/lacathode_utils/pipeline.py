import ast
import inspect
import json
import os
from pathlib import Path
import pickle
import sys

import numpy as np
import torch

from .storage import atomic_write, save_npz, seed_start, write_json
from .recovery import EpochRecovery
from .epoch_hook import install_epoch_recovery
from .acceleration import install_validation_counts, install_tensor_batches, execution_report
from .worker_progress import ProgressStage, emit_progress
from .source import verify, COMMIT


def device_masks(module):
    original = module.load_dataset
    tree = ast.parse(inspect.getsource(original))
    for name, label in (("sigmask", 1), ("bgmask", 0)):
        expected = ast.parse(f"datadict[{name!r}] = torch.from_numpy(input_data[:, -1] == {label})").body[0]
        found = [n for n in ast.walk(tree) if isinstance(n, ast.Assign) and ast.dump(n) == ast.dump(expected)]
        if len(found) != 1:
            raise ValueError("Unexpected preprocessing kernel; refusing device compatibility patch")
        found[0].value = ast.Call(
            func=ast.Attribute(value=found[0].value, attr="to", ctx=ast.Load()),
            args=[ast.Name(id="device", ctx=ast.Load())],
            keywords=[],
        )
    ast.fix_missing_locations(tree)
    namespace = {}
    exec(compile(tree, "<device masks>", "exec"), original.__globals__, namespace)
    module.load_dataset = namespace["load_dataset"]


def run(args, contract):
    config_file = (
        "DE_MAF_model_deltaR.yml" if contract["inputs"].get("variant") == "deltaR" else "DE_MAF_model.yml"
    )
    verify(args.sources)
    sys.path.insert(0, str(args.sources))
    os.chdir(args.sources)
    import data_handler
    import run_all
    import run_ANODE_training
    import ANODE_training_utils
    import classifier_training_utils

    device_masks(data_handler)
    install_validation_counts(ANODE_training_utils)
    original_load = torch.load

    def trusted_load(*a, **kw):
        kw.setdefault("weights_only", False)
        return original_load(*a, **kw)

    torch.load = trusted_load
    root = args.output / "training"
    recovery = EpochRecovery(root, contract, args.resume)
    run_ANODE_training.train_ANODE = install_epoch_recovery(
        ANODE_training_utils, "train_ANODE", "flow", recovery
    )
    install_epoch_recovery(classifier_training_utils, "train_model", "classifier", recovery)
    acceleration = install_tensor_batches()
    arguments = [
        "--data_dir",
        str(args.data),
        "--save_dir",
        str(root),
        "--mode",
        "CATHODE",
        "--cf_separate_val_set",
        "--no_extra_signal",
        "--cf_n_samples",
        "267000",
        "--cf_realistic_conditional",
        "--cf_oversampling",
        "--cf_no_logit",
        "--cf_use_class_weights",
        "--cf_save_model",
        "--cf_n_runs",
        "1",
        "--DE_epochs",
        "100",
        "--cf_epochs",
        "100",
        "--DE_config_file",
        config_file,
    ]
    parsed = run_all.parser.parse_args(arguments)
    de = run_all.create_namespace_DE_training(parsed)
    seed_start(args.seed)

    def flow():
        emit_progress("flow", "Train background flow", total=100, unit="epoch", completed=0)
        run_all.train_DE(de)

    recovery.stage("flow", flow)

    def create():
        with ProgressStage("creation", "Build latent/reference datasets"):
            creation = run_all.create_namespace_classifier_creation(parsed)
            creation.ANODE_models = run_all.find_best_epochs(de, 10)
            if any(Path(p).name.endswith("_epoch_-1.par") for p in creation.ANODE_models):
                raise ValueError("Upstream selected the untrained flow entry")
            run_all.create_data(creation)

    recovery.stage("creation", create)

    def classify():
        emit_progress("classifier", "Train classifier", total=100, unit="epoch", completed=0)
        run_all.train_classifier(run_all.create_namespace_classifier_training(parsed))

    recovery.stage("classifier", classify)
    rows = np.load(root / "X_test.npy")
    if len(np.unique(rows[rows[:, -2] == 1, -1])) > 1:
        run_all.full_single_evaluation(
            str(root),
            str(root),
            n_ensemble_epochs=10,
            extra_signal=False,
            sic_range=(0, 20),
            savefig=str(root / "internal_sic"),
        )
    evaluate(args, root, data_handler, config_file=config_file)
    write_json(
        args.output / "protocol.json",
        {
            "name": "LaCathode",
            "commit": COMMIT,
            "flow_epochs": 100,
            "classifier_epochs": 100,
            "classifier_runs": 1,
            "reference_samples": 267000,
            "selected_checkpoints": 10,
            "acceleration": execution_report(acceleration),
            "configuration": {
                name: (args.sources / name).read_text() for name in (config_file, "classifier.yml")
            },
        },
    )
    verify(args.sources)


def evaluate(args, root, data_handler, *, config_file="DE_MAF_model.yml"):
    from density_estimator import DensityEstimator
    from evaluation_utils import minimum_validation_loss_models
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    captured = {}

    def capture(labels, scores):
        captured.update(labels=labels, scores=scores)
        return roc_curve(labels, scores)

    notebook = json.loads((args.sources / "bkg_sculpting_study.ipynb").read_text())
    text = next(
        "".join(c.get("source", []))
        for c in notebook["cells"]
        if "def make_ROCs(" in "".join(c.get("source", []))
    )
    function = next(
        n for n in ast.parse(text).body if isinstance(n, ast.FunctionDef) and n.name == "make_ROCs"
    )
    namespace = {
        "np": np,
        "torch": torch,
        "plt": plt,
        "join": os.path.join,
        "pickle": pickle,
        "load_dataset": data_handler.load_dataset,
        "DensityEstimator": DensityEstimator,
        "minimum_validation_loss_models": minimum_validation_loss_models,
        "roc_curve": capture,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<pinned SR evaluation>", "exec"), namespace)
    seed_start(args.seed)
    with ProgressStage("evaluation", "Evaluate signal-region ensemble"):
        namespace["make_ROCs"](
            str(root),
            str(args.data),
            [1],
            str(root / "sr_evaluation.pdf"),
            str(root / "sr_evaluation.pkl"),
            True,
            config_file=str(args.sources / config_file),
            num_DE_models=1,
            num_clsf_models=10,
            multirun=True,
        )
    losses = np.load(root / "my_ANODE_model_val_losses.npy")
    epoch = int(np.argpartition(losses, 1)[0]) - 1
    if epoch < 0:
        raise ValueError("Inference selected the untrained flow entry")
    reference = data_handler.load_dataset(np.load(args.data / "outerdata_train.npy").astype("float32"))
    model = DensityEstimator(str(args.sources / config_file), eval_mode=True).model
    model.load_state_dict(
        torch.load(root / f"my_ANODE_model_epoch_{epoch}.par", map_location="cpu", weights_only=True)
    )
    model.eval().requires_grad_(False)
    paths = minimum_validation_loss_models(str(root), n_epochs=10)[0]
    write_json(root / "classifier_selection.json", {"ordered_checkpoints": [Path(p).name for p in paths]})
    for partition, suffix in (("validation", "val"), ("test", "test"), ("signal_region", None)):
        names = (
            (f"innerdata_{suffix}.npy", f"outerdata_{suffix}.npy")
            if suffix
            else ("innerdata_test.npy", "innerdata_extrabkg_test.npy", "innerdata_extrasig.npy")
        )
        rows = np.vstack([np.load(args.data / name) for name in names]).astype("float32")
        prepared = data_handler.load_dataset(rows, external_datadict=reference)
        x, m = prepared["tensor2"], prepared["labels"]
        with torch.no_grad():
            z = np.concatenate(
                [model(x[o : o + 8192], m[o : o + 8192])[0].numpy() for o in range(0, len(x), 8192)]
            )
        mask = prepared["mask"].numpy()
        if suffix:
            predictions = []
            with (
                ProgressStage(
                    "scores_" + partition, "Score " + partition, len(paths), "checkpoint"
                ) as progress,
                torch.no_grad(),
            ):
                for i, path in enumerate(paths):
                    classifier = torch.load(path, map_location="cpu", weights_only=False).eval()
                    predictions.append(
                        np.concatenate(
                            [
                                classifier(torch.as_tensor(z[o : o + 8192])).numpy().ravel()
                                for o in range(0, len(z), 8192)
                            ]
                        )
                    )
                    progress.update(i + 1)
            scores = np.mean(np.stack(predictions), axis=0)
        else:
            if not np.array_equal(captured["labels"], rows[mask, -1]):
                raise ValueError("Pinned SR evaluation event alignment changed")
            scores = captured["scores"]
        aligned = np.full(len(rows), np.nan, dtype=scores.dtype)
        aligned[mask] = scores
        atomic_write(
            args.output / f"{partition}_scores.npz",
            lambda p: save_npz(
                p,
                mass=rows[:, 0],
                labels=rows[:, -1].astype(np.int8),
                mask=mask,
                scores=aligned,
                physical=rows[:, 1:-1],
                latent=z,
            ),
        )
