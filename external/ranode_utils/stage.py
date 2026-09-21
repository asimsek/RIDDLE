import ast
import importlib
import json
import os
import random
import runpy
import sys
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch

from .data import BACKGROUND_PROTOCOL, MatchedPartitions, evaluation, export_scores, input_features, latent_inputs, region, validate
from .source import digest, verify
from .ensemble import signal_fit_health
from .variants import background_diagnostics, extend_delta_r, model_config
from riddle.worker_progress import emit_progress as stage_progress
from riddle.production import NumericalFitError


@contextmanager
def matched_inputs(arrays, stage, *, resample_training=False):
    import sklearn.model_selection as selection
    import src.generate_data_lhc as generate
    import src.utils as utils

    adapter = MatchedPartitions(arrays, stage)
    original_split, original_resample = selection.ShuffleSplit, generate.resample_split
    original_transform = utils.preprocess_params_transform

    class FixedSplit:
        def __init__(self, n_splits=20, **kwargs):
            if stage == "background" and (
                kwargs.get("test_size") != 0.5 or kwargs.get("random_state") != 22
            ):
                raise ValueError("R-ANODE background requires the upstream 50/50 ShuffleSplit")
            self.n_splits = n_splits
            self.resampler = original_split(n_splits=n_splits, **kwargs)

        def split(self, rows, *args, **kwargs):
            indices = adapter.split(rows)
            if stage == "background":
                for train, val in self.resampler.split(rows, *args, **kwargs):
                    adapter.split_indices = train, val
                    adapter.split_audit = {
                        "protocol": BACKGROUND_PROTOCOL,
                        "mode": "upstream 50/50 ShuffleSplit of sideband development rows",
                        "source_files": ["outerdata_train.npy", "outerdata_val.npy"],
                        "preprocessing_rows": sum(len(arrays[name]) for name in (
                            "outerdata_train.npy", "outerdata_val.npy"
                        )),
                        "accepted_rows": len(rows),
                        "training_rows": len(train),
                        "validation_rows": len(val),
                        "signal_region_rows": 0,
                        "heldout_test_rows": 0,
                        "validation_fraction": 0.5,
                        "split_random_state": 22,
                    }
                    yield train, val
                return
            if stage == "signal" and resample_training:
                pool, reserved = indices
                for train, val in self.resampler.split(pool):
                    chosen = pool[train], pool[val]
                    adapter.split_audit = {
                        "mode": "80/20 resamples of prepared training rows",
                        "training_rows": len(train),
                        "validation_rows": len(val),
                        "reserved_validation_rows_excluded": len(reserved),
                    }
                    adapter.split_indices = chosen
                    yield chosen
                return
            adapter.split_indices = indices
            for _ in range(self.n_splits):
                yield indices

    def transform(rows, params):
        result = original_transform(rows, params)
        if not adapter.membership:
            adapter.register(original_transform, params)
        return result

    selection.ShuffleSplit = FixedSplit
    generate.resample_split = adapter.resample_split
    utils.preprocess_params_transform = transform
    try:
        yield adapter
    finally:
        selection.ShuffleSplit = original_split
        generate.resample_split = original_resample
        utils.preprocess_params_transform = original_transform


def execute_script(path, *, cpu_background=False, variant="default"):
    sideband_background = path.name == "nflows_CR.py"
    if not cpu_background and variant != "deltaR" and not sideband_background:
        return runpy.run_path(str(path), run_name="__main__")
    tree = ast.parse(path.read_text(), filename=str(path))
    if sideband_background:
        tree = background_diagnostics(tree, len(input_features(variant)))
    elif variant == "deltaR":
        tree = extend_delta_r(tree, path.name)
    if cpu_background:
        # The upstream background launcher hardcodes CUDA; only its device switch changes on CPU.
        assignments = [
            node
            for node in tree.body
            if isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and node.targets[0].id == "CUDA"
        ]
        if (
            len(assignments) != 1
            or not isinstance(assignments[0].value, ast.Constant)
            or assignments[0].value.value is not True
        ):
            raise ValueError("Unrecognized upstream background device switch")
        assignments[0].value = ast.copy_location(ast.Constant(False), assignments[0].value)
    namespace = {"__name__": "__main__", "__file__": str(path)}
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), namespace)  # noqa: S102 -- verified pinned source
    return namespace


def initial_rng(path):
    path = Path(path)
    checksum = path.with_suffix(".sha256")
    if path.exists():
        if not checksum.is_file() or checksum.read_text().strip() != digest(path):
            raise ValueError("R-ANODE restart RNG state failed verification")
        state = torch.load(path, map_location="cpu", weights_only=False)
        random.setstate(state["python"])
        np.random.set_state(state["numpy"])
        torch.set_rng_state(state["torch"])
        if state["cuda"] is not None:
            torch.cuda.set_rng_state_all(state["cuda"])
    else:
        state = {
            "python": random.getstate(),
            "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all()
            if torch.cuda.is_available()
            else None,
        }
        torch.save(state, path)
        checksum.write_text(digest(path) + "\n")


def main():
    options = json.loads(sys.argv[1])
    sources, data, attempt = (
        Path(options[k]).resolve() for k in ("sources", "data", "attempt")
    )
    stage_progress("inputs", "Verify upstream sources and load input arrays")
    verify(sources)
    inputs, arrays = validate(data)
    originals, mapping_masks = None, None
    if options.get("latent_inputs") is not None:
        if options["device"] != "cpu" or not options.get("safeguards", True):
            raise ValueError("riddlev4 requires the guarded CPU pilot path")
        originals = arrays
        _, arrays, mapping_masks = latent_inputs(options["latent_inputs"], inputs, originals,
                                                expected_digest=options["latent_manifest_sha256"])
    safeguards = options.get("safeguards", True)
    if not safeguards:
        from riddle.data import diagnostic_profile
        if diagnostic_profile(inputs) is None or options["device"] != "cpu":
            raise ValueError("Disabling R-ANODE safeguards is restricted to the CPU pilot")
    variant = inputs.get("variant", "default")
    sys.path.insert(0, str(sources))
    for name in ("src.nflow_utils", "src.utils", "wandb"):
        importlib.import_module(name)

    if options["device"] != "cpu" and not torch.cuda.is_available():
        raise RuntimeError("R-ANODE requested CUDA, but it is unavailable")
    torch.set_num_threads(int(options.get("torch_threads", 2)))
    torch.set_num_interop_threads(1)
    from riddle.acceleration import install_tensor_batches, execution_report
    tensor_acceleration = install_tensor_batches(options["device"])
    stage_progress("prepare", "Prepare physical features and restore stage RNG")
    initial_rng(options["rng"])
    records = evaluation(arrays)
    upstream_data = attempt / "inputs"
    upstream_data.mkdir()
    background_config = model_config(sources, upstream_data / "DE_MAF_model_deltaR.yml", variant)
    np.save(
        upstream_data / "x_test.npy",
        np.concatenate([r[region(r[:, 0])] for r in records.values()]),
    )
    os.chdir(attempt)
    stage = options["stage"]
    script = (
        sources
        / "scripts"
        / ("nflows_CR.py" if stage == "background" else "r_anode.py")
    )
    argv = [
        str(script),
        "--data_dir",
        str(upstream_data),
        "--config_file",
        str(background_config),
        "--epochs",
        str(options["epochs"]),
        "--batch_size",
        "256",
        "--shuffle_split",
        "--split",
        str(options["fit_index"]),
        "--seed",
        str(options["seed"]),
        "--wandb_group",
        "upstream",
        "--wandb_job_type",
        stage,
        "--wandb_run_name",
        "fit",
    ]
    if stage == "signal":
        argv += [
            "--CR_path",
            options["background"],
            "--gpu",
            options["device"],
            "--mini_batch",
            "256",
            "--mode_background",
            "freeze",
            "--validation_fraction",
            "0.2",
            "--resample",
            "--random_w",
            "--w_train",
            "--data_loss_expr",
            "true_likelihood",
        ]
        if options["scenario"] == "background_only":
            argv.append("--no_signal_fit")
    else:
        argv += ["--n_sig", "2000", "--try_", "1"]
    (attempt / "command.json").write_text(json.dumps(argv, indent=2) + "\n")
    sys.argv = argv
    stage_progress("setup", "Prepare upstream model and data split")
    with matched_inputs(
        arrays, stage, resample_training=options.get("resample_training", False)
    ) as adapter:
        namespace = execute_script(
            script, cpu_background=stage == "background" and options["device"] == "cpu",
            variant=variant,
        )
    stage_progress("losses", "Validate losses and record data partitions")
    output = attempt / "results/upstream" / stage / "fit"
    loss_names = (
        ("trainloss_list", "valloss_list")
        if stage == "background"
        else ("trainloss", "valloss")
    )
    nonfinite_losses = []
    for name in loss_names:
        loss = np.load(output / (name + ".npy"), allow_pickle=False)
        if loss.shape != (options["epochs"],):
            raise ValueError(
                "Incomplete upstream R-ANODE losses; refusing this result"
            )
        if not np.isfinite(loss).all():
            if safeguards:
                raise NumericalFitError("Nonfinite upstream R-ANODE losses")
            nonfinite_losses.append(name)
    if adapter.split_audit is None:
        raise ValueError("The pinned script did not use the matched partitions")
    (attempt / "partition_audit.json").write_text(
        json.dumps(adapter.split_audit, indent=2) + "\n"
    )
    np.savez_compressed(
        attempt / "split_indices.npz",
        training=adapter.split_indices[0], validation=adapter.split_indices[1],
    )
    if stage == "signal":
        stage_progress("normalization", "Validate sampled signal-mass normalization")
        health = signal_fit_health(attempt, namespace, safeguards=safeguards)
        health["nonfinite_loss_arrays"] = nonfinite_losses
        (attempt / "mass_normalization.json").write_text(json.dumps(health, indent=2) + "\n")
        stage_progress("export", "Export validation, test and signal-region scores")
        export_scores(namespace, arrays, attempt, **(
            dict(original_arrays=originals, mapping_masks=mapping_masks) if originals is not None else {}))
        np.save(attempt / "upstream_likelihood.npy", namespace["likelihood"])
    stage_progress("sources", "Verify upstream sources are unchanged")
    verify(sources)
    from riddle.storage import write_json
    write_json(attempt / "tensor_batch_acceleration.json", execution_report(tensor_acceleration))
    stage_progress("sources", "Verify upstream sources are unchanged", completed=1)


if __name__ == "__main__":
    from riddle.production import NumericalFitError
    from riddle.storage import write_json
    request = json.loads(sys.argv[1])
    try:
        main()
    except (NumericalFitError, FloatingPointError) as error:
        write_json(Path(request["attempt"]) / "fit_failure.json", dict(
            kind="numerical", stage=request["stage"], fit_index=request["fit_index"],
            error=str(error), error_type=type(error).__name__))
        raise SystemExit(86) from error
