import argparse
import fcntl
import importlib.metadata
import json
import os
import signal
import subprocess
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import yaml

from .data import BACKGROUND_PROTOCOL, input_features, scientific_version, validate
from .ensemble import combine_fits, selected_epochs
from .resume import check_contract, policy, transition_history
from .source import COMMIT, REPOSITORY, digest, verify

ROOT = Path(__file__).resolve().parents[2]


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", dir=path.parent, delete=False) as stream:
        temporary = Path(stream.name)
        json.dump(value, stream, indent=2, allow_nan=False)
        stream.write("\n")
    os.replace(temporary, path)


@contextmanager
def lock(path):
    with Path(path).open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError("Another R-ANODE process owns this output") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def settings(path, *, runs=None, epochs=None):
    value = yaml.safe_load(Path(path).read_text())
    required = {"background_epochs", "signal_epochs", "fit_index"}
    if (
        not isinstance(value, dict)
        or not required <= set(value)
        or set(value) - required - {"runs", "split_mode"}
    ):
        raise ValueError(
            "R-ANODE config requires background_epochs, signal_epochs, fit_index and optional runs/split_mode"
        )
    value.setdefault("runs", 1)  # Older single-fit configuration files retain their meaning.
    value.setdefault("split_mode", "fixed")
    if value["split_mode"] not in ("fixed", "resample_training"):
        raise ValueError("R-ANODE split_mode must be fixed or resample_training")
    if runs is not None:
        value["runs"] = runs
    if epochs is not None:
        value["signal_epochs"] = epochs
    if (
        any(type(value[key]) is not int for key in required | {"runs"})
        or min(value["background_epochs"], value["signal_epochs"]) < 10
    ):
        raise ValueError(
            "R-ANODE needs at least ten epochs for its original ten-checkpoint selection"
        )
    if not 0 <= value["fit_index"] < 20:
        raise ValueError("R-ANODE fit_index must be 0–19, as in the upstream launcher")
    if value["runs"] < 1 or value["fit_index"] + value["runs"] > 20:
        raise ValueError("R-ANODE requires runs >= 1 and fit_index + runs <= 20")
    return value


def runtime():
    import torch

    packages = {
        name: importlib.metadata.version(name)
        for name in (
            "numpy",
            "scipy",
            "scikit-learn",
            "torch",
            "nflows",
            "wandb",
            "matplotlib",
            "PyYAML",
        )
    }
    return {
        "python": sys.version,
        "packages": packages,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "torch_threads": torch.get_num_threads(),
        "matmul_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_tf32": torch.backends.cudnn.allow_tf32,
    }


def check_files(root, hashes):
    if not hashes:
        raise ValueError("Empty R-ANODE artifact receipt")
    for name, expected in hashes.items():
        path = Path(root) / name
        if (
            Path(name).is_absolute()
            or ".." in Path(name).parts
            or path.is_symlink()
            or digest(path) != expected
        ):
            raise ValueError("R-ANODE artifact changed: " + name)


def fingerprint(root):
    return {
        str(p.relative_to(root)): digest(p)
        for p in sorted(Path(root).rglob("*"))
        if p.is_file()
    }


def execute(options, env):
    command = [sys.executable, "-m", "external.ranode_utils.stage", json.dumps(options)]
    process = subprocess.Popen(command, env=env, start_new_session=True)
    try:
        if process.wait() != 0:
            raise RuntimeError(
                f"Upstream R-ANODE {options['stage']} stage failed; artifacts retained in {options['attempt']}"
            )
    finally:
        if process.poll() is None:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()


def stage_root(args, stage, config):
    if stage == "background":
        return args.output
    return args.output / "fits" / f"fit_{config['fit_index']:03d}"


def run_stage(args, stage, config, background=None):
    root = stage_root(args, stage, config)
    receipt = root / (stage + "_complete.json")
    if receipt.exists():
        saved = json.loads(receipt.read_text())
        check_files(args.output, saved["files"])
        print(f"[PASS] Reuse verified R-ANODE {stage} stage", flush=True)
        return args.output / saved["attempt"]
    attempts = root / "upstream_runs" / stage
    attempts.mkdir(parents=True, exist_ok=True)
    attempt = Path(tempfile.mkdtemp(prefix="attempt_", dir=attempts))
    options = {
        "stage": stage,
        "data": str(args.data),
        "sources": str(args.sources),
        "attempt": str(attempt),
        "rng": str(attempts / "initial_rng.pt"),
        "scenario": args.scenario,
        "device": args.device,
        "seed": args.seed,
        "fit_index": config["fit_index"],
        "epochs": config[stage + "_epochs"],
        "resample_training": config["split_mode"] == "resample_training",
    }
    if background is not None:
        options["background"] = str(background / "results/upstream/background/fit")
    env = os.environ.copy()
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        MPLBACKEND="Agg",
        WANDB_MODE="disabled",
        PYTHONPATH=os.pathsep.join(filter(None, (str(ROOT), env.get("PYTHONPATH")))),
    )
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = str(args.io_workers)
    write_json(attempt / "producer_contract.json", args.production_contract)
    print(f"[WORK] Original R-ANODE {stage}: {options['epochs']} epochs", flush=True)
    execute(options, env)
    relative = attempt.relative_to(args.output)
    hashes = {
        str(relative / name): value for name, value in fingerprint(attempt).items()
    }
    write_json(receipt, {"attempt": str(relative), "files": hashes})
    return attempt


def run(args):
    resume_options = policy(args)
    for name in ("sources", "data", "output", "config"):
        setattr(args, name, Path(getattr(args, name)).resolve())
    if args.device not in ("cpu", "cuda:0"):
        raise ValueError(
            "Independent R-ANODE accepts cpu or cuda:0; select other GPUs with CUDA_VISIBLE_DEVICES"
        )
    if (
        args.output.is_relative_to(args.data)
        or args.data.is_relative_to(args.output)
        or args.output.is_relative_to(args.sources)
    ):
        raise ValueError("Keep R-ANODE data, upstream sources and results separate")
    config = settings(
        args.config, runs=getattr(args, "runs", None), epochs=getattr(args, "epochs", None)
    )
    inputs, _ = validate(args.data)
    if inputs["scenario"] != args.scenario:
        raise ValueError("R-ANODE scenario disagrees with prepared inputs")
    source = verify(args.sources)
    try:
        environment = runtime()
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "R-ANODE runtime is incomplete; install requirements.txt locally or use the updated riddle-runtime image"
        ) from error
    contract = {
        "schema": 1,
        "method": "ranode",
        "scientific_version": scientific_version(inputs.get("variant", "default")),
        "source_commit": COMMIT,
        "source_sha256": source,
        "inputs": inputs,
        "environment": environment,
        "settings": {
            **config,
            "seed": args.seed,
            "scenario": args.scenario,
            "device": args.device,
            "io_workers": args.io_workers,
            "ensemble": "mean_upstream_ratios_v1",
            "background_protocol": BACKGROUND_PROTOCOL,
        },
        "code": {p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
    }
    args.output.mkdir(parents=True, exist_ok=True)
    with lock(args.output / ".ranode.lock"):
        manifest = args.output / "result.json"
        saved = json.loads(manifest.read_text()) if manifest.exists() else None
        if saved is None and any(
            (args.output / name).exists()
            for name in (
                "upstream_runs",
                "background_complete.json",
                "signal_complete.json",
                "fits",
            )
        ):
            raise ValueError(
                "R-ANODE artifacts exist without their result contract; use a new output"
            )
        if saved:
            if not args.resume:
                raise FileExistsError(
                    "R-ANODE result exists; use --resume or a new output"
                )
            changes = check_contract(saved["contract"], contract, **resume_options)
            if saved["completed"]:
                check_files(args.output, saved["artifacts_sha256"])
            else:
                receipts = [args.output / "background_complete.json"] + [
                    stage_root(args, "signal", {"fit_index": index}) / "signal_complete.json"
                    for index in range(config["fit_index"], config["fit_index"] + config["runs"])
                ]
                for receipt in receipts:
                    if receipt.exists():
                        check_files(args.output, json.loads(receipt.read_text())["files"])
            if changes:
                history_path = args.output / ".resume/transitions.json"
                write_json(history_path, transition_history(
                    history_path, saved["contract"], contract, changes,
                    action="reuse_completed" if saved["completed"] else "resume_incomplete",
                ))
                print("[WARNING] Approved R-ANODE resume changes: "
                      + ", ".join(change["field"] for change in changes), flush=True)
            if saved["completed"]:
                print("[PASS] Reuse verified completed R-ANODE result", flush=True)
                return
        report = {
            "schema": 1,
            "method": "ranode",
            "seed": args.seed,
            "scenario": args.scenario,
            "variant": inputs.get("variant", "default"),
            "completed": False,
            "contract": contract,
        }
        if saved:
            report["initial_contract"] = saved.get("initial_contract", saved["contract"])
        write_json(manifest, report)
        args.production_contract = contract
        background = run_stage(args, "background", config)
        fits, members = [], []
        for index in range(config["fit_index"], config["fit_index"] + config["runs"]):
            print(f"[WORK] R-ANODE fit {len(fits) + 1}/{config['runs']}", flush=True)
            member_config = {**config, "fit_index": index}
            fit = run_stage(args, "signal", member_config, background)
            epochs = selected_epochs(fit, config["signal_epochs"])
            fits.append(fit)
            members.append({
                "fit_index": index,
                "attempt": str(fit.relative_to(args.output)),
                "epochs": epochs,
            })
        combine_fits(fits, args.output, requested_runs=config["runs"])
        protocol = {
            "repository": REPOSITORY,
            "commit": COMMIT,
            "benchmark": "matched physical LHCO partitions, not an exact paper reproduction",
            "features": list(input_features(inputs.get("variant", "default"))),
            "latent_transformation": None,
            "background_script": "scripts/nflows_CR.py",
            "signal_script": "scripts/r_anode.py",
            "background_scope": "sideband development data only; no signal-region or held-out test rows",
            "background_protocol": BACKGROUND_PROTOCOL,
            "background_preprocessing": "Original preprocessing fitted on outerdata_train.npy + outerdata_val.npy only",
            "background_split": "Original 50/50 ShuffleSplit on accepted sideband development rows; random_state=22",
            "background_diagnostics": "Align original diagnostic sample indices with physical features; no training changes",
            "signal_scope": "strict 3.3 < mjj < 3.7",
            "evaluation_scope": "signal region only",
            "input_adapter": "Sideband-only background pool; signal fits use upstream 80/20 ShuffleSplit within prepared training rows only" if config["split_mode"] == "resample_training" else "Sideband-only background pool; signal fits retain prepared train/validation membership",
            "optimizer": "Upstream AdamW unchanged, including decay on the learned fraction logit",
            "ensemble": "Equal-weight arithmetic mean of upstream per-fit density ratios; ten original validation-selected checkpoints per fit; all requested fits required; no manual exclusions",
            "score": "Each fit retains scripts/r_anode.py likelihood, including its sampled signal-mass normalization and nan_to_num; final score is log(mean(exp(per-fit log ratio)))",
            "seed": "Upstream seed controls data ordering; fit_index selects the split and fraction initialization. Upstream does not seed Torch",
            "restart": "Reuse completed stages; restart interrupted stage from saved initial RNG, without changing upstream checkpoint format",
            "background_attempt": str(background.relative_to(args.output)),
            "signal_attempts": [member["attempt"] for member in members],
            "members": members,
            "requested_runs": config["runs"],
            "valid_runs": len(members),
            "selected_checkpoints": sum(len(member["epochs"]) for member in members),
            "partitions": {
                "background": json.loads((background / "partition_audit.json").read_text()),
                "signal": [json.loads((path / "partition_audit.json").read_text()) for path in fits],
            },
            "configuration": config,
        }
        if inputs.get("variant") == "deltaR":
            protocol["dimensional_extension"] = {
                "background_num_inputs": 5,
                "background_num_cond_inputs": 1,
                "signal_num_features": 6,
                "changes": "Add physical deltaR; adjust input dimensions, sample reshapes and diagnostic feature loops only",
                "unchanged": "Upstream flow classes, preprocessing, likelihood, optimization, split rules, checkpoint selection and score definition",
            }
        write_json(args.output / "protocol.json", protocol)
        validate(args.data)
        verify(args.sources)
        # Only successful attempts enter the published receipt; partial attempts remain recoverable.
        artifacts = [
            p
            for p in args.output.iterdir()
            if p.is_file()
            and p.name not in ("result.json", ".ranode.lock", "training.log")
        ]
        hashes = {p.name: digest(p) for p in artifacts}
        receipts = [args.output / "background_complete.json"] + [
            stage_root(args, "signal", {"fit_index": member["fit_index"]}) / "signal_complete.json"
            for member in members
        ]
        for receipt in receipts:
            hashes[str(receipt.relative_to(args.output))] = digest(receipt)
            hashes.update(
                json.loads(receipt.read_text())["files"]
            )
        report.update(completed=True, artifacts_sha256=hashes)
        write_json(manifest, report)
        print("[PASS] Independent upstream R-ANODE result verified", flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run original R-ANODE on matched physical LHCO partitions"
    )
    parser.add_argument("--sources", type=Path, default=ROOT / "external/ranode")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "external/ranode_utils/ranode.yaml")
    parser.add_argument("--runs", type=int, help="Override YAML signal-fit count")
    parser.add_argument("--epochs", type=int, help="Override YAML signal epochs; background_epochs is unchanged")
    parser.add_argument(
        "--scenario", choices=("signal_injection", "background_only"), required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--io-workers", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--resume-across-code-change", action="store_true",
        help="With --resume, permit recorded helper-code changes; upstream sources stay pinned",
    )
    parser.add_argument(
        "--resume-across-device-change", action="store_true",
        help="With --resume, permit recorded CUDA GPU changes; software and precision stay strict",
    )
    args = parser.parse_args(argv)
    if args.io_workers < 1 or not 0 <= args.seed < 2**32:
        parser.error("Positive CPU threads and a nonnegative 32-bit seed are required")

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.io_workers)
    try:
        run(args)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"[ERROR] {error}\n")


if __name__ == "__main__":
    main()
