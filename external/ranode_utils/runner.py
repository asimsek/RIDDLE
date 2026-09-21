import argparse
from copy import copy
import fcntl
import importlib.metadata
import json
import os
import re
import signal
import subprocess
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path

import yaml

from .data import BACKGROUND_PROTOCOL, input_features, latent_inputs, scientific_version, validate
from .ensemble import combine_fits, selected_epochs, mass_normalization_check, validate_result_normalization
from .resume import check_contract, policy, transition_history
from .source import COMMIT, REPOSITORY, digest, verify
from riddle.worker_progress import EVENT_PREFIX, ProgressStage, emit_message, emit_progress, BufferedLog
from riddle.production import NumericalFitError

ROOT = Path(__file__).resolve().parents[2]


def safeguards_active(args):
    return not getattr(args, "pilot_no_safeguards", False)


def paired_rng_contract(args, inputs, config, source, environment):
    """Pair the diagnostic control with saved stage RNGs, without altering the source run."""
    root = getattr(args, "pilot_rng_source", None)
    if root is None:
        return dict(paired=False, reason="No saved guarded R-ANODE run supplied; independent initialization")
    root = Path(root).resolve()
    report = json.loads((root / "result.json").read_text())
    original = report["contract"]
    expected = {**config, "seed": args.seed, "scenario": args.scenario,
                "device": args.device, "io_workers": args.io_workers, "torch_threads": args.torch_threads}
    if (report.get("method") != "ranode" or original["inputs"] != inputs or original["source_sha256"] != source
            or any(original["settings"].get(k) != v for k, v in expected.items())
            or original["environment"] != environment):
        raise ValueError("R-ANODE RNG pairing requires identical data, upstream sources, settings and runtime")
    if report.get("completed"):
        check_files(root, report["artifacts_sha256"])
    names = ["upstream_runs/background/initial_rng.pt"] + [
        f"fits/fit_{i:03d}/upstream_runs/signal/initial_rng.pt"
        for i in range(config["fit_index"], config["fit_index"] + config["runs"])]
    hashes = {}
    for name in names:
        path = root / name
        if not path.exists():
            raise ValueError(f"Missing paired RNG state: {path}; guarded job must finish its stages first")
        checksum = digest(path)
        if path.with_suffix(".sha256").read_text().strip() != checksum:
            raise ValueError(f"Changed paired RNG state: {path}")
        hashes[name] = checksum
    return dict(paired=True, source=str(root), files=hashes,
                note="Same initial stage RNGs; equality also requires deterministic upstream CPU operations")


class StageReporter:
    """Translate existing upstream stdout without touching its training loop."""

    epoch_pattern = re.compile(r"^epoch:\s*(\d+)\s+trainloss:\s*(\S+)\s+valloss:\s*(\S+)\s*$")

    def __init__(self, options):
        self.options = options
        self.stream = "Background" if options["stage"] == "background" else f"Fit {options['fit_index']:03d}"
        self.current = None
        self.phase("imports", "Start stage process and load dependencies")

    def publish(self, **updates):
        self.current.update(updates)
        emit_progress(**self.current, stream=self.stream)

    def phase(self, phase, label, *, total=1, unit="step"):
        if self.current and self.current["phase"] == phase:
            return
        self.finish()
        self.current = dict(phase=phase, label=label, total=total, unit=unit, completed=0)
        self.publish()

    def finish(self):
        if self.current and self.current["unit"] != "epoch":
            self.publish(completed=self.current["total"])

    def line(self, line):
        text = line.strip()
        if text.startswith(EVENT_PREFIX):
            event = json.loads(text[len(EVENT_PREFIX):])
            self.phase(event["phase"], event["label"])
            if event.get("completed") is not None:
                self.publish(completed=event["completed"])
            return
        prefix = "" if self.options["stage"] == "background" else f"[R-ANODE fit {self.options['fit_index']}] "
        if text:
            if os.environ.get("RIDDLE_WORKER_PROGRESS") == "1":
                print(prefix + line.rstrip(), flush=True)
            else:
                from riddle.progress import colored_status

                colored_status(prefix + line.rstrip(), level=2)
        match = self.epoch_pattern.fullmatch(text)
        if text.startswith("X_test shape") or match:
            total = self.options.get("epochs", 1)
            if self.current["phase"] not in ("training", "evaluation"):
                self.phase("training", f"Train {self.options['stage']} flow", total=total, unit="epoch")
            if match and self.current["phase"] == "training":
                completed = int(match[1]) + 1
                if self.current["completed"] < completed <= total:
                    self.publish(completed=completed, train_loss=match[2], validation_loss=match[3])
                    if completed == total:
                        self.phase("evaluation", "Upstream checkpoint selection and evaluation")


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
    if isinstance(value, dict) and "fits" in value:
        if "runs" in value:
            raise ValueError("Use fits, or the legacy runs key, not both")
        value["runs"] = value.pop("fits")
    required = {"background_epochs", "signal_epochs", "fit_index"}
    if (
        not isinstance(value, dict)
        or not required <= set(value)
        or set(value) - required - {"runs", "split_mode"}
    ):
        raise ValueError(
            "R-ANODE config requires background_epochs, signal_epochs, fit_index and optional fits/split_mode"
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
        raise ValueError("R-ANODE requires fits >= 1 and fit_index + fits <= 20")
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


def stage_command(options):
    return [sys.executable, "-m", "external.ranode_utils.stage", json.dumps(options)]


def stop_processes(processes):
    running = [process for process in processes if process.poll() is None]
    for process in running:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 10
    for process in running:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def stage_failure(options):
    path = Path(options["attempt"]) / "fit_failure.json"
    if not path.exists():
        return None
    report = json.loads(path.read_text())
    if (report.get("kind") != "numerical" or report.get("stage") != options["stage"]
            or report.get("fit_index") != options["fit_index"]):
        raise ValueError("Invalid upstream numerical failure receipt")
    error = NumericalFitError(report["error"])
    error.attempt = options["attempt"]
    return error


def reject_signal_fit(args, index, error):
    if not safeguards_active(args):
        # The control must never quietly become a filtered ensemble.
        raise error
    attempt = Path(error.attempt)
    relative = attempt.relative_to(args.output)
    receipt = dict(fit_index=index, status="excluded", check="numerical_validity",
                   attempt=str(relative), error=str(error),
                   files={str(relative / name): value for name, value in fingerprint(attempt).items()})
    write_json(stage_root(args, "signal", {"fit_index": index}) / "fit_failure.json", receipt)
    emit_message(f"R-ANODE fit {index:03d}: excluded numerical failure: {error}", kind="WARNING", level=0)


def execute(options, env):
    reporter = StageReporter(options)
    log_path = Path(options["attempt"]) / "stage.log"
    with log_path.open("w") as raw_stream:
        stream = BufferedLog(raw_stream)
        process = subprocess.Popen(
            stage_command(options), env=env, start_new_session=True,
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
            encoding="utf-8", errors="replace", bufsize=1,
        )
        try:
            for line in process.stdout:
                stream.write(line)
                reporter.line(line)
            if process.wait() != 0:
                failure = stage_failure(options)
                if failure is not None:
                    raise failure
                raise RuntimeError(
                    f"Upstream R-ANODE {options['stage']} stage failed; artifacts retained in {options['attempt']}; see {log_path}"
                )
        finally:
            stream.force_flush()
            stop_processes([process])
            process.stdout.close()


def stage_root(args, stage, config):
    if stage == "background":
        return args.output
    return args.output / "fits" / f"fit_{config['fit_index']:03d}"


def prepare_stage(args, stage, config, background=None):
    root = stage_root(args, stage, config)
    receipt = root / (stage + "_complete.json")
    if receipt.exists():
        label = "Background" if stage == "background" else f"Fit {config['fit_index']:03d}"
        emit_progress("reuse", "Verify saved stage artifacts", stream=label, completed=0)
        saved = json.loads(receipt.read_text())
        check_files(args.output, saved["files"])
        if stage == "signal":
            try:
                mass_normalization_check(args.output / saved["attempt"], safeguards=safeguards_active(args))
            except NumericalFitError as error:
                error.attempt = str(args.output / saved["attempt"])
                raise
        emit_progress("reuse", "Verify saved stage artifacts", stream=label, completed=1)
        emit_message(f"{label}: reuse verified R-ANODE {stage} stage", kind="PASS")
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
        "torch_threads": args.torch_threads,
        "seed": args.seed,
        "fit_index": config["fit_index"],
        "epochs": config[stage + "_epochs"],
        "resample_training": config["split_mode"] == "resample_training",
    }
    if not safeguards_active(args):
        options["safeguards"] = False
    if getattr(args, "pilot_latent_inputs", None) is not None:
        options.update(latent_inputs=str(args.pilot_latent_inputs),
                       latent_manifest_sha256=args.production_contract["settings"]["latent_manifest_sha256"])
    if "rng_pairing" in args.production_contract["settings"]:
        pairing = args.production_contract["settings"]["rng_pairing"]
        if pairing["paired"]:
            import shutil
            destination = Path(options["rng"])
            name = str(destination.relative_to(args.output))
            source = Path(pairing["source"]) / name
            checksum = pairing["files"][name]
            if digest(source) != checksum:
                raise ValueError("Paired RNG source changed during execution")
            if destination.exists() and digest(destination) != checksum:
                raise ValueError("Paired RNG destination changed during execution")
            if not destination.exists():
                shutil.copyfile(source, destination)
                destination.with_suffix(".sha256").write_text(checksum + "\n")
    if background is not None:
        options["background"] = str(background / "results/upstream/background/fit")
    env = os.environ.copy()
    env.update(
        PYTHONDONTWRITEBYTECODE="1",
        PYTHONUNBUFFERED="1",
        MPLBACKEND="Agg",
        WANDB_MODE="disabled",
        RIDDLE_WORKER_PROGRESS="1",
        PYTHONPATH=os.pathsep.join(filter(None, (str(ROOT), env.get("PYTHONPATH")))),
    )
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        env[key] = str(args.torch_threads)
    write_json(attempt / "producer_contract.json", args.production_contract)
    return options, env


def complete_stage(args, options):
    label = "Background" if options["stage"] == "background" else f"Fit {options['fit_index']:03d}"
    emit_progress("receipt", "Verify and record stage artifacts", stream=label, completed=0)
    attempt = Path(options["attempt"])
    if options["stage"] == "signal":
        try:
            mass_normalization_check(attempt, safeguards=safeguards_active(args))
        except NumericalFitError as error:
            error.attempt = str(attempt)
            raise
    relative = attempt.relative_to(args.output)
    hashes = {
        str(relative / name): value for name, value in fingerprint(attempt).items()
    }
    receipt = stage_root(args, options["stage"], options) / (options["stage"] + "_complete.json")
    write_json(receipt, {"attempt": str(relative), "files": hashes})
    emit_progress("receipt", "Verify and record stage artifacts", stream=label, completed=1)
    return attempt


def run_stage(args, stage, config, background=None):
    request = prepare_stage(args, stage, config, background)
    if isinstance(request, Path):
        return request
    options, env = request
    execute(options, env)
    return complete_stage(args, options)


def fit_log(state, *, final=False):
    text = state["pending_log"] + state["reader"].read()
    lines = text.splitlines(keepends=True)
    state["pending_log"] = ""
    for line in lines:
        if not final and not line.endswith(("\n", "\r")):
            state["pending_log"] = line
        elif line.strip():
            state["reporter"].line(line)


def run_signal_fits(args, config, background):
    """Schedule isolated upstream processes; retain fit-index ensemble order."""
    indices = list(range(config["fit_index"], config["fit_index"] + config["runs"]))
    workers = min(getattr(args, "workers", 1), len(indices))
    progress = ProgressStage("ranode_fits", "Train and evaluate signal fits", len(indices), "fit", report_every=1)
    previously_excluded = {}
    for index in indices:
        receipt = stage_root(args, "signal", {"fit_index": index}) / "fit_failure.json"
        if receipt.exists():
            if not safeguards_active(args):
                raise ValueError("Unguarded pilot cannot reuse a fit exclusion receipt")
            saved = json.loads(receipt.read_text())
            if saved.get("fit_index") != index or saved.get("status") != "excluded":
                raise ValueError("Invalid R-ANODE fit exclusion receipt")
            check_files(args.output, saved["files"])
            previously_excluded[index] = None
    if workers == 1:
        fits = []
        for index in indices:
            emit_message(f"Fit {len(fits) + 1}/{len(indices)} (fit_{index:03d}): starting")
            try:
                fit = (None if index in previously_excluded else
                       run_stage(args, "signal", {**config, "fit_index": index}, background))
            except NumericalFitError as error:
                reject_signal_fit(args, index, error)
                fit = None
            fits.append(fit)
            progress.update(len(fits), force=True)
        return fits
    from .runtime import concurrent_environment

    extra_env, mps = concurrent_environment(getattr(args, "mps", "auto"), args.device)
    emit_message(f"R-ANODE: up to {workers} concurrent fits on {args.device}; MPS: {mps['reason']}")
    execution = {"workers": workers, "device": args.device, "mps": mps, "fits": [], "completed": False}
    execution_path = args.output / ".resume" / f"concurrency_{time.time_ns()}.json"
    write_json(execution_path, execution)
    active, completed, cursor = {}, dict(previously_excluded), 0
    try:
        while cursor < len(indices) or active:
            while cursor < len(indices) and len(active) < workers:
                index = indices[cursor]
                cursor += 1
                if index in previously_excluded:
                    continue
                try:
                    request = prepare_stage(args, "signal", {**config, "fit_index": index}, background)
                except NumericalFitError as error:
                    reject_signal_fit(args, index, error)
                    completed[index] = None
                    progress.update(len(completed), force=True)
                    continue
                if isinstance(request, Path):
                    completed[index] = request
                    execution["fits"].append({"fit_index": index, "reused": True})
                    progress.update(len(completed), force=True)
                    continue
                options, env = request
                reporter = StageReporter(options)
                log_path = Path(options["attempt"]) / "stage.log"
                stream = log_path.open("w")
                reader = log_path.open(encoding="utf-8", errors="replace")
                try:
                    process = subprocess.Popen(stage_command(options), env={**env, **extra_env},
                                               stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
                except BaseException:
                    stream.close()
                    reader.close()
                    raise
                timing = {"fit_index": index, "pid": process.pid, "started": time.time(), "reused": False}
                execution["fits"].append(timing)
                active[index] = dict(process=process, options=options, stream=stream, reader=reader,
                                     pending_log="", timing=timing, reporter=reporter)
                emit_message(f"Fit {cursor}/{len(indices)} (fit_{index:03d}): started; log: {log_path}")
            finished = []
            for index, state in active.items():
                fit_log(state)
                code = state["process"].poll()
                if code is None:
                    continue
                state["timing"].update(finished=time.time(), returncode=code)
                fit_log(state, final=True)
                if code and stage_failure(state["options"]) is None:
                    raise RuntimeError(f"R-ANODE fit {index} failed (exit {code}); see {state['options']['attempt']}/stage.log")
                state["stream"].close()
                state["reader"].close()
                try:
                    if code:
                        raise stage_failure(state["options"])
                    completed[index] = complete_stage(args, state["options"])
                except NumericalFitError as error:
                    reject_signal_fit(args, index, error)
                    completed[index] = None
                finished.append(index)
                emit_message(f"Fit {index:03d}: " + ("completed" if completed[index] is not None else "excluded"),
                             kind="PASS" if completed[index] is not None else "WARNING")
                progress.update(len(completed), force=True)
            for index in finished:
                del active[index]
            if active and not finished:
                time.sleep(.2)
        execution["completed"] = True
    finally:
        stop_processes([state["process"] for state in active.values()])
        for state in active.values():
            state["timing"].update(finished=time.time(), returncode=state["process"].returncode)
            if not state["reader"].closed:
                fit_log(state, final=True)
            state["stream"].close()
            state["reader"].close()
        write_json(execution_path, execution)
    return [completed[index] for index in indices]


def run(args):
    resume_options = policy(args)
    if type(getattr(args, "workers", 1)) is not int or getattr(args, "workers", 1) < 1:
        raise ValueError("R-ANODE workers must be a positive integer")
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
        args.config, runs=getattr(args, "fits", None), epochs=getattr(args, "epochs", None)
    )
    checks = ProgressStage("ranode_checks", "Verify inputs, upstream sources and runtime", 3)
    inputs, original_arrays = validate(args.data)
    guarded = safeguards_active(args)
    latent = getattr(args, "pilot_latent_inputs", None)
    if latent is not None:
        latent = args.pilot_latent_inputs = Path(latent).resolve()
        if not guarded or args.device != "cpu" or not latent.is_relative_to(args.output):
            raise ValueError("riddlev4 requires guarded CPU execution with latent inputs stored inside its output")
        latent_meta, _, _ = latent_inputs(latent, inputs, original_arrays)
    if not guarded or getattr(args, "pilot_rng_source", None) is not None:
        from riddle.data import diagnostic_profile
        if (guarded and latent is None) or args.device != "cpu" or diagnostic_profile(inputs) is None:
            raise ValueError("R-ANODE controls/RNG pairing are restricted to the explicit CPU pilot")
    method = "riddlev4" if latent is not None else ("ranode" if guarded else "ranodev2")
    checks.update(1, force=True)
    if inputs["scenario"] != args.scenario:
        raise ValueError("R-ANODE scenario disagrees with prepared inputs")
    source = verify(args.sources)
    checks.update(2, force=True)
    try:
        environment = runtime()
    except importlib.metadata.PackageNotFoundError as error:
        raise RuntimeError(
            "R-ANODE runtime is incomplete; install requirements.txt locally or use the updated riddle-runtime image"
        ) from error
    checks.update(3, force=True)
    emit_message(
        f"R-ANODE on {args.device}: shared background {config['background_epochs']} epochs; "
        f"{config['runs']} signal fits × {config['signal_epochs']} epochs; "
        f"workers={min(getattr(args, 'workers', 1), config['runs'])}. Epochs are displayed starting at 1."
    )
    contract = {
        "schema": 1,
        "method": method,
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
            "torch_threads": args.torch_threads,
            "ensemble": "mean_upstream_ratios_v1",
            "background_protocol": BACKGROUND_PROTOCOL,
            "fit_failure_policy": "exclude_numerically_invalid_fits_v1",
        },
        "code": {**{p.name: digest(p) for p in sorted(Path(__file__).parent.glob("*.py"))},
                 "framework/production.py": digest(ROOT / "riddle/production.py"),
                 "framework/resume.py": digest(ROOT / "riddle/resume.py"),
                 "framework/acceleration.py": digest(ROOT / "riddle/acceleration.py")},
    }
    if not guarded:
        contract["settings"].update(fit_failure_policy="disabled_cpu_pilot_control",
            rng_pairing=paired_rng_contract(args, inputs, config, source, environment))
    if latent is not None:
        contract["scientific_version"] += "_latent_inputs_v1"
        contract["settings"].update(input_representation=latent_meta,
            latent_manifest_sha256=digest(latent / "manifest.json"),
            rng_pairing=paired_rng_contract(args, inputs, config, source, environment))
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
            resume_progress = ProgressStage("ranode_resume", "Verify resume contract and saved artifacts")
            if not args.resume:
                raise FileExistsError(
                    "R-ANODE result exists; use --resume or a new output"
                )
            changes = check_contract(saved["contract"], contract,
                                     reuse_completed=saved.get("completed") is True, **resume_options)
            if saved["completed"]:
                check_files(args.output, saved["artifacts_sha256"])
                if guarded:
                    validate_result_normalization(args.output, saved)
                from riddle.production import validate_result_scores

                validate_result_scores(args.output, "ranode")
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
                emit_message("Approved R-ANODE resume changes: "
                             + ", ".join(change["field"] for change in changes), kind="WARNING", level=0)
            resume_progress.update(1, force=True)
            if saved["completed"]:
                emit_message("Reuse verified completed R-ANODE result", kind="PASS")
                return
        report = {
            "schema": 1,
            "method": method,
            "seed": args.seed,
            "campaign_seed": getattr(args, "campaign_seed", None) if getattr(args, "campaign_seed", None) is not None else args.seed,
            "run_index": getattr(args, "run_index", 0),
            "independent_run_count": getattr(args, "independent_run_count", 1),
            "ensemble_fits": config["runs"],
            "scenario": args.scenario,
            "variant": inputs.get("variant", "default"),
            "completed": False,
            "contract": contract,
        }
        if saved:
            report["initial_contract"] = saved.get("initial_contract", saved["contract"])
        write_json(manifest, report)
        args.production_contract = contract
        with ProgressStage("ranode_background", "Train and evaluate shared background"):
            background = run_stage(args, "background", config)
        fits = run_signal_fits(args, config, background)
        members, excluded = [], []
        ensemble_progress = ProgressStage("ranode_ensemble", "Select checkpoints and combine fit scores", config["runs"] + 1)
        for index, fit in zip(range(config["fit_index"], config["fit_index"] + config["runs"]), fits):
            if fit is None:
                receipt = stage_root(args, "signal", {"fit_index": index}) / "fit_failure.json"
                excluded.append(json.loads(receipt.read_text()))
                continue
            epochs = selected_epochs(fit, config["signal_epochs"], safeguards=guarded)
            members.append({
                "fit_index": index,
                "attempt": str(fit.relative_to(args.output)),
                "epochs": epochs,
            })
            ensemble_progress.update(len(members), force=True)
        write_json(args.output / "fit_selection.json", dict(
            status="completed" if members else "no_accepted_fits", requested_fits=config["runs"],
            accepted_fits=len(members), members=members, excluded_fits=excluded,
            selection=("numerical validity only; no AUC, SIC, truth or validation-performance selection"
                       if guarded else "all completed fits; helper safeguards disabled; no fit exclusions")))
        fits = [fit for fit in fits if fit is not None]
        if not fits:
            raise RuntimeError("R-ANODE: no numerically valid fits; no physics result was produced")
        combine_fits(fits, args.output, requested_runs=len(fits), safeguards=guarded)
        ensemble_progress.update(config["runs"] + 1, force=True)
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
            "ensemble": "Equal-weight arithmetic mean of upstream per-fit density ratios; ten original validation-selected checkpoints per accepted fit; numerical failures excluded and recorded",
            "score": "Unmodified upstream likelihood; require sampled signal-mass support in every SR bin and finite likelihood before nan_to_num; final score is log(mean(exp(per-fit log ratio)))",
            "seed": "Upstream seed controls data ordering; fit_index selects the split and fraction initialization. Upstream does not seed Torch",
            "restart": "Reuse completed stages; restart interrupted stage from saved initial RNG, without changing upstream checkpoint format",
            "background_attempt": str(background.relative_to(args.output)),
            "signal_attempts": [member["attempt"] for member in members],
            "members": members,
            "excluded_fits": excluded,
            "requested_runs": config["runs"],
            "valid_runs": len(members),
            "selected_checkpoints": sum(len(member["epochs"]) for member in members),
            "partitions": {
                "background": json.loads((background / "partition_audit.json").read_text()),
                "signal": [json.loads((path / "partition_audit.json").read_text()) for path in fits],
            },
            "configuration": config,
        }
        if not guarded:
            protocol.update(method="ranodev2", safeguards=False,
                rng_pairing=contract["settings"]["rng_pairing"],
                ensemble="Equal-weight upstream ratios from every requested fit; no helper fit filtering",
                score="Original upstream likelihood including its density floors and nan_to_num; no helper repair",
                integrity_checks="Event alignment, complete artifacts, hashes, and finite exported scores remain required")
        if latent is not None:
            protocol.update(method="riddlev4", safeguards=True, benchmark="Pinned R-ANODE applied to frozen RIDDLE latent features",
                features=[f"z{i+1}" for i in range(len(input_features(inputs.get("variant", "default"))))],
                latent_transformation=latent_meta, rng_pairing=contract["settings"]["rng_pairing"],
                background_preprocessing="Original upstream preprocessing fitted on mapped sideband development rows",
                signal_inputs="Latent features plus original mass as a joint density coordinate",
                background_density="Original upstream learned conditional background on latent inputs, not an imposed Gaussian",
                coordinate_ratio="Signal and background densities use the same latent coordinates; mapping Jacobians cancel",
                export="All original physical event rows retained; both mapping and upstream-domain rejections recorded in mask",
                riddle_residual_training=False)
        if inputs.get("variant") == "deltaR":
            protocol["dimensional_extension"] = {
                "background_num_inputs": 5,
                "background_num_cond_inputs": 1,
                "signal_num_features": 6,
                "changes": ("Map all five physical features, including deltaR, to five latent coordinates; "
                            "adjust upstream input dimensions, sample reshapes and diagnostic feature loops only"
                            if latent is not None else
                            "Add physical deltaR; adjust input dimensions, sample reshapes and diagnostic feature loops only"),
                "unchanged": "Upstream flow classes, preprocessing, likelihood, optimization, split rules, checkpoint selection and score definition",
            }
        write_json(args.output / "protocol.json", protocol)
        from riddle.production import validate_result_scores

        health = validate_result_scores(args.output, "ranode")
        health["mass_normalization"] = {
            member["attempt"]: mass_normalization_check(fit, safeguards=guarded) for member, fit in zip(members, fits)
        }
        write_json(args.output / "score_health.json", health)
        final_progress = ProgressStage("ranode_final", "Verify inputs, sources and final result artifacts", 3)
        validate(args.data)
        final_progress.update(1, force=True)
        verify(args.sources)
        final_progress.update(2, force=True)
        # Accepted predictions and exclusion evidence are fingerprinted separately.
        artifacts = [
            p
            for p in args.output.iterdir()
            if p.is_file()
            and p.name not in ("result.json", ".ranode.lock", "training.log")
        ]
        hashes = {p.name: digest(p) for p in artifacts}
        if latent is not None:
            # The result remains auditable after the source RIDDLE run is moved.
            latent_inputs(latent, inputs, original_arrays,
                          expected_digest=contract["settings"]["latent_manifest_sha256"])
            hashes.update({str(p.relative_to(args.output)): digest(p) for p in latent.rglob("*") if p.is_file()})
        receipts = [args.output / "background_complete.json"] + [
            stage_root(args, "signal", {"fit_index": member["fit_index"]}) / "signal_complete.json"
            for member in members
        ]
        receipts += [stage_root(args, "signal", {"fit_index": m["fit_index"]}) / "fit_failure.json"
                     for m in excluded]
        for receipt in receipts:
            hashes[str(receipt.relative_to(args.output))] = digest(receipt)
            hashes.update(
                json.loads(receipt.read_text())["files"]
            )
        report.update(completed=True, artifacts_sha256=hashes)
        write_json(manifest, report)
        final_progress.update(3, force=True)
        emit_message("Independent upstream R-ANODE result verified", kind="PASS")


def run_repetitions(args):
    """The direct adapter CLI also distinguishes whole runs from signal fits."""
    from riddle.cli import independent_run_seeds

    count = getattr(args, "runs", None) or 1
    for index, seed in enumerate(independent_run_seeds(args.seed, count)):
        options = copy(args)
        if count > 1:
            options.output = args.output / f"seed_{seed:03d}"
            options.campaign_seed, options.run_index = args.seed, index
            options.independent_run_count = count
        options.seed = seed
        run(options)


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Run original R-ANODE on matched physical LHCO partitions", allow_abbrev=False
    )
    parser.add_argument("--sources", type=Path, default=ROOT / "external/ranode")
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=ROOT / "external/ranode_utils/ranode.yaml")
    from riddle.cli import positive

    parser.add_argument("--fits", type=positive, help="Override YAML signal-fit count per complete run")
    parser.add_argument("--runs", type=positive, default=1,
                        help="Complete independent runs including background retraining (default: 1)")
    parser.add_argument("--campaign-seed", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--run-index", type=int, default=0, help=argparse.SUPPRESS)
    parser.add_argument("--independent-run-count", type=positive, default=1, help=argparse.SUPPRESS)
    parser.add_argument("--epochs", type=int, help="Override YAML signal epochs; background_epochs is unchanged")
    parser.add_argument(
        "--scenario", choices=("signal_injection", "background_only"), required=True
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--workers", type=int, default=1, help="Concurrent signal fits after the shared background stage")
    parser.add_argument("--mps", choices=("auto", "on", "off"), default="auto")
    parser.add_argument("--io-workers", type=int, default=2)
    parser.add_argument("--torch-threads", type=int, default=2)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--pilot-no-safeguards", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--pilot-rng-source", type=Path, help=argparse.SUPPRESS)
    parser.add_argument("--pilot-latent-inputs", type=Path, help=argparse.SUPPRESS)
    parser.add_argument(
        "--resume-across-code-change", action="store_true",
        help="With --resume, permit recorded helper-code changes; upstream sources stay pinned",
    )
    parser.add_argument(
        "--resume-across-device-change", action="store_true",
        help="With --resume, permit recorded CUDA GPU changes; software and precision stay strict",
    )
    args = parser.parse_args(argv)
    if args.workers < 1 or args.io_workers < 1 or args.torch_threads < 1 or not 0 <= args.seed < 2**32:
        parser.error("Positive workers/I/O workers/PyTorch threads and a nonnegative 32-bit seed are required")

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    if args.device == "cpu":
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
    for key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
        os.environ[key] = str(args.torch_threads)
    try:
        if os.environ.get("RIDDLE_WORKER_PROGRESS") == "1":
            run_repetitions(args)
        else:
            from riddle.progress import set_verbosity
            from riddle.worker_progress import local_progress

            set_verbosity(int(os.environ.get("RIDDLE_VERBOSE", "1")))
            with local_progress(f"ranode | {args.scenario}"):
                run_repetitions(args)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as error:
        parser.exit(1, f"[ERROR] {error}\n")


if __name__ == "__main__":
    main()
