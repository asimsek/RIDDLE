import ast
from contextlib import contextmanager, redirect_stderr, redirect_stdout
import inspect
import json
import multiprocessing
import os
from pathlib import Path
import pickle
import queue
import signal
import shutil
import subprocess
import sys
import time
import traceback
import uuid

FLOW_PREFIX = "lacathode_model"
RUN_LAYOUT = "independent_background_classifier_v1"
FIXED_RUN_LAYOUT = "fixed_background_classifiers_v1"
DEFAULTS = {"pipeline_runs": 1, "classifier_epochs": 100}


def run_settings(runs=None, epochs=None, background="independent"):
    if background not in ("independent", "fixed"):
        raise ValueError("LaCathode background must be independent or fixed")
    values = {
        "pipeline_runs": DEFAULTS["pipeline_runs"] if runs is None else runs,
        "classifier_runs": 1,
        "classifier_epochs": DEFAULTS["classifier_epochs"] if epochs is None else epochs,
    }
    if type(values["pipeline_runs"]) is not int or not 1 <= values["pipeline_runs"] < 2**32:
        raise ValueError("LaCathode --runs must be a positive integer")
    if type(values["classifier_epochs"]) is not int or values["classifier_epochs"] < 11:
        raise ValueError("LaCathode --epochs must be at least 11 for its upstream ten-checkpoint selector")
    if background == "fixed":
        values["classifier_runs"], values["pipeline_runs"] = values["pipeline_runs"], 1
    return values


def run_seeds(seed, count):
    import numpy as np

    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid LaCathode campaign seed")
    seeds = [seed] + [int(np.random.SeedSequence([seed, i]).generate_state(1)[0]) for i in range(1, count)]
    if len(set(seeds)) != count:
        raise ValueError("LaCathode run seeds collided; choose another campaign seed")
    return seeds


def run_command(options):
    options["execution_token"] = uuid.uuid4().hex
    output = Path(options["output"])
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    framework = str(Path(__file__).resolve().parents[2])
    env.update(PYTHONHASHSEED=str(options["seed"]), PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1", RIDDLE_WORKER_PROGRESS="1",
               PYTHONPATH=os.pathsep.join(filter(None, (framework, env.get("PYTHONPATH")))))
    command = [sys.executable, "-m", "riddle.worker", json.dumps(options, default=str)]
    return command, env


def numerical_run_failure(options):
    """Only a failure receipt from this worker invocation can justify exclusion."""
    path = Path(options["output"]) / "result.json"
    if not path.exists():
        return False
    report = json.loads(path.read_text())
    failure = report.get("failure", {})
    return (report.get("completed") is False and report.get("method") == "lacathode"
            and report.get("seed") == options["seed"]
            and report.get("run_index") == options["run_index"]
            and failure.get("kind") == "numerical"
            and options.get("execution_token") is not None
            and failure.get("execution_token") == options["execution_token"])


def launch_run(options, index, count):
    from .worker_progress import EVENT_PREFIX
    from riddle.worker_progress import BufferedLog, durable_progress_event

    output = Path(options["output"])
    command, env = run_command(options)
    with (output / "training.log").open("a" if options["resume"] else "w") as raw_log:
        log = BufferedLog(raw_log)
        with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1, errors="replace") as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    if line.startswith(EVENT_PREFIX):
                        event = json.loads(line[len(EVENT_PREFIX):])
                        if "phase" in event:
                            event["phase"] = f"run_{index:03d}/" + event["phase"]
                            event["label"] = f"Run {index + 1}/{count} | " + event["label"]
                        elif "message" in event:
                            event["message"] = f"Run {index + 1}/{count} | " + event["message"]
                        if durable_progress_event(event):
                            log.force_flush()
                        line = EVENT_PREFIX + json.dumps(event) + "\n"
                    print(line, end="", flush=True)
                log.force_flush()
                if process.wait() and not numerical_run_failure(options):
                    raise RuntimeError(f"LaCathode run {index} failed; inspect {output / 'training.log'}")
            except BaseException:
                log.force_flush()
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise


def concurrent_log(state, count, *, final=False):
    from .worker_progress import EVENT_PREFIX, _emit_event

    text = state["pending_log"] + state["reader"].read()
    state["pending_log"] = ""
    for line in text.splitlines(keepends=True):
        if not final and not line.endswith(("\n", "\r")):
            state["pending_log"] = line
            continue
        if line.startswith(EVENT_PREFIX):
            event = json.loads(line[len(EVENT_PREFIX):])
            event["stream"] = f"Run {state['index'] + 1}/{count}"
            _emit_event(event)
        elif line.strip():
            print(f"[LaCathode run {state['index']:03d}] {line.rstrip()}", flush=True)


def stop_runs(active):
    running = [state["process"] for state in active.values() if state["process"].poll() is None]
    for process in running:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + 5
    for process in running:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def launch_runs(args, requests):
    """Bounded, isolated complete pipelines; collection remains in run-index order."""
    from .storage import write_json
    from .worker_progress import ProgressStage, emit_message

    count = len(requests)
    workers = min(getattr(args, "workers", 1), count)
    active, cursor, completed = {}, 0, 0
    execution = {"workers": workers, "device": args.device, "runs": [], "completed": False}
    audit = args.output / ".resume" / f"concurrency_{time.time_ns()}.json"
    write_json(audit, execution)
    emit_message(f"LaCathode independent mode: up to {workers} concurrent complete runs on {args.device}")
    try:
        with ProgressStage("independent_runs", "Independent LaCathode runs", count, "run") as progress:
            while cursor < count or active:
                while cursor < count and len(active) < workers:
                    options = requests[cursor]
                    index = options["run_index"]
                    command, env = run_command(options)
                    log_path = Path(options["output"]) / "training.log"
                    stream = log_path.open("a" if options["resume"] else "w")
                    reader = log_path.open(encoding="utf-8", errors="replace")
                    if options["resume"]:
                        reader.seek(0, os.SEEK_END)
                    try:
                        process = subprocess.Popen(command, env=env, stdout=stream,
                                                   stderr=subprocess.STDOUT, start_new_session=True)
                    except BaseException:
                        stream.close()
                        reader.close()
                        raise
                    timing = {"run": index, "seed": options["seed"], "pid": process.pid, "started": time.time()}
                    active[index] = dict(process=process, stream=stream, reader=reader, index=index,
                                         pending_log="", timing=timing)
                    execution["runs"].append(timing)
                    cursor += 1
                    emit_message(f"Run {index + 1}/{count}: started; log: {log_path}", kind="WORK")
                finished = []
                for index, state in active.items():
                    concurrent_log(state, count)
                    code = state["process"].poll()
                    if code is None:
                        continue
                    state["timing"].update(finished=time.time(), returncode=code)
                    concurrent_log(state, count, final=True)
                    if code and not numerical_run_failure(requests[index]):
                        raise RuntimeError(f"LaCathode run {index} failed (exit {code}); inspect {requests[index]['output']}/training.log")
                    completed += 1
                    progress.update(completed, force=True)
                    finished.append(index)
                for index in finished:
                    state = active.pop(index)
                    state["stream"].close()
                    state["reader"].close()
                if active and not finished:
                    time.sleep(.2)
        execution["completed"] = True
    finally:
        stop_runs(active)
        for state in active.values():
            state["timing"].update(finished=time.time(), returncode=state["process"].returncode)
            concurrent_log(state, count, final=True)
            state["stream"].close()
            state["reader"].close()
        write_json(audit, execution)



def fixed_classifier_seeds(seed, count):
    import numpy as np

    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("Invalid LaCathode classifier seed")
    values = [int(np.random.SeedSequence([seed, 0x4C4346, i]).generate_state(1)[0]) for i in range(count)]
    if len(set(values)) != count:
        raise ValueError("LaCathode classifier fit seeds collided; choose another campaign seed")
    return values


def _fixed_classifier_fit(job, options):
    import numpy as np
    import torch

    from .acceleration import execution_report, install_tensor_batches
    from .epoch_hook import install_epoch_recovery
    from .recovery import EpochRecovery
    from .source import verify
    from .storage import seed_start

    root = Path(options["root"])
    sources = Path(options["sources"])
    verify(sources)
    sys.path.insert(0, str(sources))
    os.chdir(sources)
    import classifier_training_utils

    recovery = EpochRecovery(
        root,
        options["contract"],
        True,
        allow_code_change=options["allow_code_change"],
        allow_device_change=options["allow_device_change"],
    )
    classifier_training_utils.train_model = recovery.classifier_fits(
        install_epoch_recovery(classifier_training_utils, "train_model", "classifier", recovery)
    )
    acceleration = install_tensor_batches(options["device"])
    seed_start(job["seed"])
    X_train = np.load(root / "X_train.npy")
    X_test = np.load(root / "X_test.npy")
    y_train = np.load(root / "y_train.npy")
    y_test = np.load(root / "y_test.npy")
    X_extrasig = None if options["no_extra_signal"] or options["supervised"] else np.load(root / "X_extrasig.npy")
    X_val = np.load(root / "X_validation.npy") if options["supervised"] or options["separate_val_set"] else None
    losses = classifier_training_utils.train_model(
        options["config_file"],
        options["epochs"],
        X_train,
        y_train,
        X_test,
        y_test,
        X_extrasig=X_extrasig,
        X_val=X_val,
        use_mjj=options["use_mjj"],
        batch_size=options["batch_size"],
        supervised=options["supervised"],
        use_class_weights=options["use_class_weights"],
        CWoLa=options["CWoLa"],
        SR_center=options["SR_center"],
        save_model=str(root / f"model_run{job['index']}"),
        verbose=options["verbose"],
    )
    return {
        "fit": job["index"],
        "seed": job["seed"],
        "train_loss": np.asarray(losses[0]),
        "validation_loss": np.asarray(losses[1]),
        "acceleration": execution_report(acceleration),
    }


def _fixed_classifier_worker(events, job, options):
    import torch
    from riddle.worker_progress import BufferedLog, durable_progress_event
    from .worker_progress import _LOCAL_SINK

    index = job["index"]
    root = Path(options["root"])
    log_root = root / ".resume"
    log_root.mkdir(parents=True, exist_ok=True)
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    with (log_root / f"classifier_worker_{index:03d}.log").open("a") as raw_log:
        log = BufferedLog(raw_log)
        with redirect_stdout(log), redirect_stderr(log):
            def publish(event):
                log.write(json.dumps(event) + "\n")
                if durable_progress_event(event):
                    log.force_flush()
                events.put((index, "progress", event))

            token = _LOCAL_SINK.set(publish)
            try:
                torch.set_num_threads(options["torch_threads"])
                torch.set_num_interop_threads(1)
                torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
                events.put((index, "result", _fixed_classifier_fit(job, options)))
            except BaseException as error:
                traceback.print_exc()
                log.force_flush()
                events.put((index, "error", f"{type(error).__name__}: {error}"))
            finally:
                log.force_flush()
                _LOCAL_SINK.reset(token)
                signal.signal(signal.SIGTERM, previous_handler)


def _stop_classifier_workers(active):
    for state in active.values():
        process = state["process"]
        if process.is_alive():
            process.terminate()
    for state in active.values():
        process = state["process"]
        process.join(timeout=5)
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


def _finalize_parallel_classifiers(root, training, results):
    import numpy as np
    import classifier_training_utils
    from evaluation_utils import minimum_val_loss_model_evaluation
    from .storage import save_array

    ordered = [results[index] for index in range(training.n_runs)]
    loss_matrix = np.stack([item["train_loss"] for item in ordered])
    val_loss_matrix = np.stack([item["validation_loss"] for item in ordered])
    save_array(root / "loss_matris.npy", loss_matrix)
    save_array(root / "val_loss_matris.npy", val_loss_matrix)
    minimum_val_loss_model_evaluation(
        str(root),
        str(root),
        n_epochs=10,
        use_mjj=training.use_mjj,
        extra_signal=not training.no_extra_signal,
    )
    for index in range(training.n_runs):
        classifier_training_utils.plot_classifier_losses(
            loss_matrix[index],
            val_loss_matrix[index],
            savefig=str(root / f"model_run{index}_loss_plot"),
            suppress_show=True,
        )
    return ordered


def train_fixed_classifiers(args, contract, root, training):
    from .progress import _duration, _short_value
    from .worker_progress import ProgressStage, emit_message

    count = int(training.n_runs)
    workers = min(max(1, int(getattr(args, "workers", 1))), count)
    if workers == 1 or count == 1:
        import run_all

        run_all.train_classifier(training)
        return {
            "mode": "sequential",
            "workers": 1,
            "fits": count,
            "fit_seeds": None,
            "mps": getattr(args, "runtime_mps", None),
        }
    seeds = fixed_classifier_seeds(args.seed, count)
    training_options = {
        "config_file": training.config_file,
        "epochs": training.epochs,
        "batch_size": training.batch_size,
        "no_extra_signal": training.no_extra_signal,
        "use_mjj": training.use_mjj,
        "supervised": training.supervised,
        "use_class_weights": training.use_class_weights,
        "CWoLa": training.CWoLa,
        "SR_center": training.SR_center,
        "separate_val_set": training.separate_val_set,
        "verbose": training.verbose,
    }
    options = {
        **training_options,
        "root": str(root),
        "sources": str(args.sources),
        "contract": contract,
        "device": str(args.device),
        "torch_threads": int(args.torch_threads),
        "allow_code_change": bool(getattr(args, "resume_across_code_change", False)),
        "allow_device_change": bool(getattr(args, "resume_across_device_change", False)),
    }
    import gc
    import torch

    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    context = multiprocessing.get_context("spawn")
    events = context.Queue()
    waiting = [{"index": index, "seed": seeds[index]} for index in range(count)]
    active = {}
    results = {}
    previous_handler = signal.getsignal(signal.SIGTERM)

    def terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, terminate)
    emit_message(f"LaCathode fixed-background classifiers: up to {workers} concurrent fits on {args.device}")
    try:
        with ProgressStage("fixed_classifier_runs", "Fixed-background classifier fits", count, "fit") as progress:
            while waiting or active:
                while waiting and len(active) < workers:
                    job = waiting.pop(0)
                    process = context.Process(target=_fixed_classifier_worker, args=(events, job, options))
                    process.start()
                    active[job["index"]] = {
                        "process": process,
                        "started": time.monotonic(),
                        "epoch": 0,
                        "printed": -1,
                        "metrics": {},
                    }
                    emit_message(
                        f"Classifier fit {job['index'] + 1}/{count}: started; active fits={len(active)}/{workers}",
                        kind="WORK",
                    )
                try:
                    index, kind, event = events.get(timeout=1)
                except queue.Empty:
                    now = time.monotonic()
                    for index, state in active.items():
                        code = state["process"].exitcode
                        if code is None:
                            state.pop("exit_seen", None)
                            continue
                        if code != 0:
                            raise RuntimeError(
                                f"LaCathode classifier fit {index + 1}/{count}: worker exited with code {code}; resume to retry"
                            )
                        state.setdefault("exit_seen", now)
                        if now - state["exit_seen"] > 5:
                            raise RuntimeError(
                                f"LaCathode classifier fit {index + 1}/{count}: worker exited without a result; resume to retry"
                            )
                    index, kind, event = None, None, None
                if kind == "error":
                    raise RuntimeError(f"LaCathode classifier fit {index + 1}/{count} failed: {event}")
                if kind == "result":
                    state = active.pop(index)
                    state["process"].join(timeout=10)
                    if state["process"].is_alive():
                        state["process"].terminate()
                        state["process"].join(timeout=5)
                    if state["process"].exitcode not in (0, None):
                        raise RuntimeError(
                            f"LaCathode classifier fit {index + 1}/{count}: worker exited with code {state['process'].exitcode}; resume to retry"
                        )
                    results[index] = event
                    progress.update(len(results), force=True)
                    emit_message(
                        f"Classifier fit {index + 1}/{count}: completed; completed fits={len(results)}/{count}; active fits={len(active)}/{workers}",
                        kind="PASS",
                    )
                elif kind == "progress" and index in active:
                    state = active[index]
                    if event.get("unit") == "epoch" and event.get("completed") is not None:
                        state["epoch"] = int(event["completed"])
                        state["metrics"] = event.get("metrics", {})
                now = time.monotonic()
                for index, state in active.items():
                    if state["epoch"] <= state["printed"]:
                        continue
                    epoch = state["epoch"]
                    elapsed = now - state["started"]
                    eta = _duration(elapsed * max(0, training.epochs - epoch) / max(1, epoch))
                    metrics = "; ".join(
                        f"{name}={_short_value(value)}"
                        for name, value in state["metrics"].items()
                        if name in {"train_loss", "validation_loss"}
                    )
                    emit_message(
                        f"Classifier fit {index + 1}/{count}: {epoch}/{training.epochs} epoch; fit_elapsed={_duration(elapsed)}; fit_ETA={eta}; active fits={len(active)}/{workers}"
                        + (f"; {metrics}" if metrics else ""),
                        kind="PROGRESS",
                    )
                    state["printed"] = epoch
    finally:
        signal.signal(signal.SIGTERM, previous_handler)
        _stop_classifier_workers(active)
        events.close()
        events.cancel_join_thread()
    ordered = _finalize_parallel_classifiers(root, training, results)
    return {
        "mode": "parallel",
        "workers": workers,
        "fits": count,
        "fit_seeds": seeds,
        "mps": getattr(args, "runtime_mps", None),
        "worker_acceleration": [item["acceleration"] for item in ordered],
    }

def collect_runs(args, members):
    import numpy as np
    from .storage import atomic_write, save_npz, save_array, write_json, verify_artifacts
    from riddle.production import validate_result_scores

    requested = len(members)
    roots = [args.output / member["directory"] for member in members]
    reports = [json.loads((root / "result.json").read_text()) for root in roots]
    accepted, excluded = [], []
    for root, report, member in zip(roots, reports, members):
        if (report.get("method") != "lacathode"
                or report.get("seed") != member["seed"]
                or report.get("run_index") != member["run"]):
            raise ValueError("Missing or misidentified independent LaCathode run")
        if not report.get("completed"):
            failure = report.get("failure", {})
            if failure.get("kind") != "numerical":
                raise ValueError("Incomplete LaCathode run without a numerical failure receipt")
            excluded.append(dict(**member, failure=failure))
            continue
        verify_artifacts(root, report["artifacts_sha256"])
        validate_result_scores(root, "lacathode")
        accepted.append(member)
    write_json(args.output / "run_selection.json", dict(
        status="completed" if accepted else "no_accepted_runs", requested_runs=requested,
        accepted_runs=len(accepted), members=accepted, excluded_runs=excluded,
        selection="numerical validity only; weak but valid classifiers retained"))
    if not accepted:
        raise RuntimeError("LaCathode: no numerically valid runs; no physics result was produced")
    members = accepted
    roots = [args.output / member["directory"] for member in members]
    inventory = json.loads((args.output / "runs.json").read_text())
    inventory.update(runs=members, requested_runs=requested, excluded_runs=excluded)
    write_json(args.output / "runs.json", inventory)
    for partition in ("validation", "test", "signal_region"):
        records = []
        for root in roots:
            with np.load(root / f"{partition}_scores.npz", allow_pickle=False) as archive:
                records.append({key: archive[key] for key in archive.files})
        first = records[0]
        for other in records[1:]:
            if any(not np.array_equal(first[key], other[key])
                   for key in ("mass", "labels", "mask", "physical")):
                raise ValueError("Independent LaCathode runs have different event populations or masks")
        fits = np.stack([record["scores"] for record in records])
        if not np.isfinite(fits[:, first["mask"]]).all():
            raise ValueError("Nonfinite independent LaCathode scores")
        atomic_write(args.output / f"{partition}_scores.npz", lambda p: save_npz(
            p, **{key: first[key] for key in ("mass", "labels", "mask", "physical", "scores", "latent")},
            fit_scores=fits, fit_latents=np.stack([record["latent"] for record in records]),
            run_seeds=np.array([member["seed"] for member in members], dtype=np.uint32)))
    summary = args.output / "training"
    summary.mkdir(exist_ok=True)
    for name in (f"{FLOW_PREFIX}_train_losses.npy", f"{FLOW_PREFIX}_val_losses.npy",
                 "loss_matris.npy", "val_loss_matris.npy"):
        histories = [np.load(root / "training" / name).reshape(-1) for root in roots]
        save_array(summary / name, np.stack(histories))
    checks = [json.loads((root / "training/flow_checkpoint_selection.json").read_text()) for root in roots]
    write_json(args.output / "flow_checkpoint_selection.json", {
        "runs": [{**member, **check} for member, check in zip(members, checks)],
        "mismatched_runs": [m["run"] for m, check in zip(members, checks) if check["mismatch"]],
    })
    protocol = json.loads((roots[0] / "protocol.json").read_text())
    protocol.update(run_layout=RUN_LAYOUT, pipeline_runs=len(members), requested_runs=requested,
                    excluded_runs=excluded, runs=members,
                    classifier_runs_per_pipeline=1,
                    score="Ten validation-selected classifier checkpoints per independent run; no cross-run score averaging",
                    fit_scores="One score row per independent background-flow-plus-classifier run",
                    primary_classifier_fit=0)
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "mapping_acceptance.json",
               json.loads((roots[0] / "mapping_acceptance.json").read_text()))


def run(args, contract):
    from .storage import write_json
    from .worker_progress import emit_message

    if type(getattr(args, "workers", 1)) is not int or getattr(args, "workers", 1) < 1:
        raise ValueError("LaCathode workers must be a positive integer")
    if getattr(args, "lacathode_background", "independent") == "fixed":
        return run_single(args, contract)
    count = run_settings(getattr(args, "runs", None), getattr(args, "epochs", None))["pipeline_runs"]
    members = [dict(run=i, seed=seed, directory=f"runs/run_{i:03d}")
               for i, seed in enumerate(run_seeds(args.seed, count))]
    write_json(args.output / "runs.json", {
        "schema": 1, "run_layout": RUN_LAYOUT, "campaign_seed": args.seed,
        "seed_rule": "run 0: campaign seed; run i>0: SeedSequence([campaign_seed, i])",
        "runs": members,
    })
    requests = [{**vars(args), "output": str(args.output / member["directory"]),
                   "seed": member["seed"], "runs": 1, "lacathode_replica": True,
                   "campaign_seed": args.seed, "run_index": member["run"], "workers": 1}
                for member in members]
    if getattr(args, "workers", 1) > 1 and count > 1:
        launch_runs(args, requests)
    else:
        for member, options in zip(members, requests):
            emit_message(f"Independent LaCathode run {member['run'] + 1}/{count}; seed={member['seed']}")
            launch_run(options, member["run"], count)
    collect_runs(args, members)


@contextmanager
def classifier_prediction_device(classifier_type):
    """Match prediction inputs to the loaded model, only during SR evaluation."""
    import torch
    from riddle.production import score_diagnostics

    original = classifier_type.predict

    def predict(self, x):
        device = next(self.parameters()).device
        with torch.no_grad():
            self.eval()
            x = torch.tensor(x, device=device)
            result = self.forward(x).detach().cpu().numpy()
            score_diagnostics(result.ravel(), stage="LaCathode SR classifier checkpoint", probability=True, summarize=False)
            return result

    classifier_type.predict = predict
    try:
        yield
    finally:
        classifier_type.predict = original


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


def _background_reuse_signature(contract, flow_epochs, config_file):
    inputs = contract["inputs"]
    return {
        "scientific_version": contract["scientific_version"],
        "source_commit": contract["source_commit"],
        "source_sha256": contract["source_sha256"],
        "variant": inputs.get("variant", "default"),
        "source": inputs.get("source"),
        "flow_epochs": int(flow_epochs),
        "config_file": config_file,
    }


def _select_background_reuse(args, contract, flow_epochs, config_file):
    if getattr(args, "lacathode_scan_background_reuse_policy", None) != "shared_fixed_background_v1":
        return None
    expected = _background_reuse_signature(contract, flow_epochs, config_file)
    independent = getattr(args, "lacathode_background", "independent") == "independent"
    current_run = getattr(args, "run_index", None)
    for item in getattr(args, "lacathode_background_reuse_candidates", None) or ():
        if independent and current_run is not None and (
            item.get("run_index") != current_run or item.get("seed") != args.seed
        ):
            continue
        source = Path(item["result"]).resolve()
        data = Path(item["data"]).resolve()
        report_path = source / "result.json"
        protocol_path = source / "protocol.json"
        if not report_path.is_file() or not protocol_path.is_file() or not data.is_dir():
            continue
        try:
            report = json.loads(report_path.read_text())
            protocol = json.loads(protocol_path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        source_contract = report.get("contract", {})
        signature = {
            "scientific_version": source_contract.get("scientific_version"),
            "source_commit": source_contract.get("source_commit"),
            "source_sha256": source_contract.get("source_sha256"),
            "variant": source_contract.get("inputs", {}).get("variant", "default"),
            "source": source_contract.get("inputs", {}).get("source"),
            "flow_epochs": protocol.get("flow_epochs"),
            "config_file": "DE_MAF_model_deltaR.yml" if source_contract.get("inputs", {}).get("variant") == "deltaR" else "DE_MAF_model.yml",
        }
        if report.get("completed") is not True or report.get("method") != "lacathode" or report.get("scenario") != "signal_injection" or signature != expected:
            continue
        artifacts = report.get("artifacts_sha256", {})
        relative = [name for name in artifacts if name.startswith("training/lacathode_model_epoch_") and name.endswith(".par")]
        relative += ["training/lacathode_model_train_losses.npy", "training/lacathode_model_val_losses.npy"]
        if len([name for name in relative if name.endswith(".par")]) < 10 or any(name not in artifacts for name in relative):
            continue
        if any(not (source / name).is_file() for name in relative):
            continue
        from riddle.storage import file_digest
        outer = data / "outerdata_train.npy"
        if not outer.is_file() or file_digest(outer) != source_contract.get("inputs", {}).get("files", {}).get(outer.name):
            raise ValueError("Nominal LaCathode preprocessing data are missing or differ from the background training data")
        return {"source": source, "data": data, "report": report, "files": relative, "signature": expected}
    return None


def _flow_recovery_state(root):
    path = Path(root) / ".resume/stages.pt"
    if not path.is_file():
        return None
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def _prepare_reuse_root(root, request):
    root = Path(root)
    if request is None or not root.exists() or (root / "background_reuse.json").is_file():
        return request
    state = _flow_recovery_state(root)
    if state is None:
        return None if any(root.iterdir()) else request
    if "flow" in state.get("done", ()) or state.get("phase") not in (None, "flow"):
        return None
    shutil.rmtree(root)
    return request


def _activate_background_reuse(root, request, data_handler, device):
    if request is None:
        return None, None, None
    import torch
    from .storage import atomic_write, file_digest, write_json

    root = Path(root)
    manifest_path = root / "background_reuse.json"
    stats_path = root / "background_preprocessing.pt"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("policy") != "shared_fixed_background_v1" or manifest.get("signature") != request["signature"]:
            raise ValueError("LaCathode shared-background reuse contract changed; use a new output")
        for name, expected in manifest.get("local_artifacts_sha256", {}).items():
            path = root / name
            if not path.is_file() or file_digest(path) != expected:
                raise ValueError("Reused LaCathode background artifact changed or is missing")
    else:
        artifacts = request["report"]["artifacts_sha256"]
        copied = {}
        for relative in request["files"]:
            source_path = request["source"] / relative
            if file_digest(source_path) != artifacts[relative]:
                raise ValueError("LaCathode background source artifact changed")
            target = root / Path(relative).name
            shutil.copy2(source_path, target)
            copied[target.name] = file_digest(target)
        rows = __import__("numpy").load(request["data"] / "outerdata_train.npy").astype("float32")
        reference = data_handler.load_dataset(rows, shuffle_loader=False, device=torch.device("cpu"))
        stats = {name: reference[name].detach().cpu() for name in ("max", "min", "mean2", "std2", "std2_logit_fix")}
        atomic_write(stats_path, lambda path: torch.save(stats, path))
        copied[stats_path.name] = file_digest(stats_path)
        manifest = {
            "schema": 1,
            "policy": "shared_fixed_background_v1",
            "signature": request["signature"],
            "source_result": str(request["source"]),
            "source_result_sha256": file_digest(request["source"] / "result.json"),
            "source_data": str(request["data"]),
            "local_artifacts_sha256": copied,
            "training_reused": True,
            "preprocessing_reused": True,
        }
        write_json(manifest_path, manifest)
    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    creation_reference = {name: value.to(device) for name, value in stats.items()}
    evaluation_reference = {name: value.cpu() for name, value in stats.items()}
    return manifest, creation_reference, evaluation_reference


def _install_fixed_preprocessing(data_handler, reference):
    if reference is None:
        return None
    original = data_handler.LHCORD_data_handler.preprocess_ANODE_data

    def preprocess(self, *args, **kwargs):
        positional = list(args)
        if len(positional) >= 4:
            positional[3] = reference
            kwargs.pop("external_param", None)
        else:
            kwargs["external_param"] = reference
        return original(self, *positional, **kwargs)

    data_handler.LHCORD_data_handler.preprocess_ANODE_data = preprocess
    return original


def _install_fixed_load_dataset(data_handler, reference):
    if reference is None:
        return None
    original = data_handler.load_dataset

    def load_dataset(data, *args, **kwargs):
        positional = list(args)
        if len(positional) >= 2:
            positional[1] = reference
            kwargs.pop("external_datadict", None)
        else:
            kwargs["external_datadict"] = reference
        return original(data, *positional, **kwargs)

    data_handler.load_dataset = load_dataset
    return original


def run_single(args, contract):
    import numpy as np
    import torch

    from .storage import seed_start, write_json
    from .recovery import EpochRecovery
    from .epoch_hook import install_epoch_recovery
    from .acceleration import install_validation_counts, install_tensor_batches, execution_report
    from .worker_progress import ProgressStage, emit_progress
    from .source import verify, COMMIT
    from .resume import resume_policy
    from riddle.data import diagnostic_profile

    settings = run_settings(getattr(args, "runs", None), getattr(args, "epochs", None),
                            getattr(args, "lacathode_background", "independent"))
    if settings["pipeline_runs"] != 1:
        raise ValueError("A single LaCathode pipeline must contain exactly one background flow")
    pilot = diagnostic_profile(contract["inputs"])
    background_epochs = pilot["background_epochs"] if pilot else 100
    reference_samples = pilot["reference_samples"] if pilot else 267000
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
    background_request = _select_background_reuse(args, contract, background_epochs, config_file)
    background_request = _prepare_reuse_root(root, background_request)
    if contract.get("scan_background") and background_request is None:
        raise ValueError("Nominal LaCathode background is incompatible, missing, or conflicts with existing training")
    recovery = EpochRecovery(root, contract, args.resume, **resume_policy(args))
    background_reuse, creation_reference, evaluation_reference = _activate_background_reuse(
        root, background_request, data_handler, torch.device(args.device)
    )
    original_preprocess = _install_fixed_preprocessing(data_handler, creation_reference)
    run_ANODE_training.train_ANODE = install_epoch_recovery(
        ANODE_training_utils, "train_ANODE", "flow", recovery
    )
    classifier_training_utils.train_model = recovery.classifier_fits(
        install_epoch_recovery(classifier_training_utils, "train_model", "classifier", recovery)
    )
    acceleration = install_tensor_batches(args.device)
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
        str(reference_samples),
        "--cf_realistic_conditional",
        "--cf_oversampling",
        "--cf_no_logit",
        "--cf_use_class_weights",
        "--cf_save_model",
        "--cf_n_runs",
        str(settings["classifier_runs"]),
        "--DE_epochs",
        str(background_epochs),
        "--cf_epochs",
        str(settings["classifier_epochs"]),
        "--DE_file_name",
        FLOW_PREFIX,
        "--DE_config_file",
        config_file,
    ]
    parsed = run_all.parser.parse_args(arguments)
    de = run_all.create_namespace_DE_training(parsed)
    seed_start(args.seed)

    def flow():
        emit_progress("flow", "Train background flow", total=background_epochs, unit="epoch", completed=0)
        run_all.train_DE(de)

    if background_reuse is None:
        recovery.stage("flow", flow)
    else:
        emit_progress("flow", "Reuse shared background flow", total=1, unit="step", completed=1)
    for name in (f"{FLOW_PREFIX}_train_losses.npy", f"{FLOW_PREFIX}_val_losses.npy"):
        losses = np.load(root / name, allow_pickle=False)
        if not losses.size:
            raise ValueError(f"LaCathode: missing flow losses in {root / name}")
        if not np.isfinite(losses).all():
            from riddle.production import NumericalFitError
            raise NumericalFitError(f"LaCathode: nonfinite flow losses in {root / name}")

    def create():
        with ProgressStage("creation", "Build latent/reference datasets"):
            creation = run_all.create_namespace_classifier_creation(parsed)
            creation.ANODE_models = run_all.find_best_epochs(de, 10)
            if any(Path(p).name.endswith("_epoch_-1.par") for p in creation.ANODE_models):
                raise ValueError("Upstream selected the untrained flow entry")
            from .storage import file_digest

            write_json(root / "flow_checkpoint_selection.json", {
                "training_checkpoint": Path(creation.ANODE_models[0]).name,
                "training_sha256": file_digest(creation.ANODE_models[0]),
                "ordered_training_candidates": [Path(p).name for p in creation.ANODE_models],
                "training_selection": "upstream first entry of argpartition best-ten list",
            })
            run_all.create_data(creation)

    recovery.stage("creation", create)

    training = run_all.create_namespace_classifier_training(parsed)

    def classify():
        execution = train_fixed_classifiers(args, contract, root, training)
        write_json(root / "classifier_execution.json", execution)

    recovery.stage("classifier", classify)
    classifier_execution_path = root / "classifier_execution.json"
    classifier_execution = (
        json.loads(classifier_execution_path.read_text())
        if classifier_execution_path.is_file()
        else {
            "mode": "sequential",
            "workers": 1,
            "fits": parsed.cf_n_runs,
            "fit_seeds": None,
            "mps": getattr(args, "runtime_mps", None),
        }
    )
    for name in ("loss_matris.npy", "val_loss_matris.npy"):
        losses = np.load(root / name, allow_pickle=False)
        if not losses.size:
            raise ValueError(f"LaCathode: missing classifier losses in {root / name}")
        if not np.isfinite(losses).all():
            from riddle.production import NumericalFitError
            raise NumericalFitError(f"LaCathode: nonfinite classifier losses in {root / name}")
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
    if original_preprocess is not None:
        data_handler.LHCORD_data_handler.preprocess_ANODE_data = original_preprocess
    evaluate(args, root, data_handler, config_file=config_file, classifier_runs=parsed.cf_n_runs,
             background_reference=evaluation_reference)
    write_json(
        args.output / "protocol.json",
        {
            "name": "LaCathode",
            "scientific_version": "pinned_upstream",
            "flow_checkpoint_prefix": FLOW_PREFIX,
            "commit": COMMIT,
            "flow_epochs": background_epochs,
            "classifier_epochs": parsed.cf_epochs,
            "classifier_runs": parsed.cf_n_runs,
            "run_layout": contract["lacathode_run_layout"],
            "background_mode": getattr(args, "lacathode_background", "independent"),
            "scan_background_reuse": background_reuse,
            "reference_samples": reference_samples,
            "selected_checkpoints": 10,
            "score": "Upstream ten-validation-checkpoint mean per classifier fit; no averaging across fits",
            "primary_classifier_fit": 0,
            "fit_scores": "All classifier fits, in upstream run order, when classifier_runs > 1",
            "acceleration": execution_report(acceleration),
            "classifier_execution": classifier_execution,
            "configuration": {
                name: (args.sources / name).read_text() for name in (config_file, "classifier.yml")
            },
        },
    )
    verify(args.sources)


def evaluate(args, root, data_handler, *, config_file="DE_MAF_model.yml", classifier_runs=1,
             background_reference=None):
    from riddle.production import score_diagnostics
    import numpy as np
    import torch

    from .storage import atomic_write, save_npz, seed_start, write_json, file_digest
    from .worker_progress import ProgressStage, emit_message
    from riddle.metrics import acceptance_report

    from classifier import Classifier
    from density_estimator import DensityEstimator
    from evaluation_utils import minimum_validation_loss_models
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve

    captured = []
    evaluation_checkpoints = []
    original_load_dataset = _install_fixed_load_dataset(data_handler, background_reference)

    def evaluation_estimator(*a, **kw):
        if kw.get("load_path") is not None:
            evaluation_checkpoints.append(Path(kw["load_path"]))
        return DensityEstimator(*a, **kw)

    def capture(labels, scores):
        score_diagnostics(np.asarray(scores), stage="LaCathode SR classifier ensemble", probability=True, summarize=False)
        captured.append(dict(labels=labels, scores=scores))
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
        "DensityEstimator": evaluation_estimator,
        "minimum_validation_loss_models": minimum_validation_loss_models,
        "roc_curve": capture,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), "<pinned SR evaluation>", "exec"), namespace)
    seed_start(args.seed)
    with (
        ProgressStage("evaluation", "Evaluate signal-region ensemble"),
        classifier_prediction_device(Classifier),
    ):
        namespace["make_ROCs"](
            str(root),
            str(args.data),
            list(range(1, classifier_runs + 1)),
            str(root / "sr_evaluation.pdf"),
            str(root / "sr_evaluation.pkl"),
            True,
            config_file=str(args.sources / config_file),
            model_file_name=FLOW_PREFIX,
            num_DE_models=1,
            num_clsf_models=10,
            multirun=True,
        )
    if original_load_dataset is not None:
        data_handler.load_dataset = original_load_dataset
    selection_path = root / "flow_checkpoint_selection.json"
    flow_selection = json.loads(selection_path.read_text())
    if len(evaluation_checkpoints) != classifier_runs or len(set(evaluation_checkpoints)) != 1:
        raise ValueError("Unexpected upstream evaluation checkpoint selection")
    evaluation_checkpoint = evaluation_checkpoints[0]
    flow_selection.update(
        evaluation_checkpoint=evaluation_checkpoint.name,
        evaluation_sha256=file_digest(evaluation_checkpoint),
        evaluation_selection="upstream single minimum-validation-loss checkpoint",
        mismatch=flow_selection["training_checkpoint"] != evaluation_checkpoint.name,
        upstream_selection_unchanged=True,
    )
    write_json(selection_path, flow_selection)
    if flow_selection["mismatch"]:
        emit_message(
            f"Background-flow checkpoint mismatch: training={flow_selection['training_checkpoint']}; "
            f"evaluation={evaluation_checkpoint.name}. Original upstream selections are unchanged.",
            kind="WARNING", level=0,
        )
    losses = np.load(root / f"{FLOW_PREFIX}_val_losses.npy")
    epoch = int(np.argpartition(losses, 1)[0]) - 1
    if epoch < 0:
        raise ValueError("Inference selected the untrained flow entry")
    reference = (background_reference if background_reference is not None else
                 data_handler.load_dataset(np.load(args.data / "outerdata_train.npy").astype("float32")))
    model = DensityEstimator(str(args.sources / config_file), eval_mode=True).model
    model.load_state_dict(
        torch.load(root / f"{FLOW_PREFIX}_epoch_{epoch}.par", map_location="cpu", weights_only=True)
    )
    model.eval().requires_grad_(False)
    paths_by_fit = minimum_validation_loss_models(str(root), n_epochs=10)
    if len(paths_by_fit) != classifier_runs or len(captured) != classifier_runs:
        raise ValueError("Incomplete upstream LaCathode classifier fits")
    selection = {"ordered_checkpoints": [Path(p).name for p in paths_by_fit[0]]}
    if classifier_runs > 1:
        selection["fits"] = [
            {"fit": i, "ordered_checkpoints": [Path(p).name for p in paths]}
            for i, paths in enumerate(paths_by_fit)
        ]
    write_json(root / "classifier_selection.json", selection)
    acceptance = {}
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
            fit_predictions = []
            with (
                ProgressStage(
                    "scores_" + partition, "Score " + partition,
                    sum(map(len, paths_by_fit)), "checkpoint"
                ) as progress,
                torch.no_grad(),
            ):
                completed = 0
                for paths in paths_by_fit:
                    predictions = []
                    for path in paths:
                        classifier = torch.load(path, map_location="cpu", weights_only=False).eval()
                        predictions.append(
                            np.concatenate(
                                [
                                    classifier(torch.as_tensor(z[o : o + 8192])).numpy().ravel()
                                    for o in range(0, len(z), 8192)
                                ]
                            )
                        )
                        score_diagnostics(predictions[-1], stage=f"LaCathode {partition} checkpoint {path}",
                                          probability=True, summarize=False)
                        completed += 1
                        progress.update(completed)
                    fit_predictions.append(np.mean(np.stack(predictions), axis=0))
        else:
            if any(not np.array_equal(fit["labels"], rows[mask, -1]) for fit in captured):
                raise ValueError("Pinned SR evaluation event alignment changed")
            fit_predictions = [fit["scores"] for fit in captured]
        scores = fit_predictions[0]
        aligned = np.full(len(rows), np.nan, dtype=scores.dtype)
        aligned[mask] = scores
        extra = {}
        if classifier_runs > 1:
            fit_scores = np.full((classifier_runs, len(rows)), np.nan, dtype=scores.dtype)
            fit_scores[:, mask] = np.stack(fit_predictions)
            extra["fit_scores"] = fit_scores
        acceptance[partition] = acceptance_report(rows[:, -1], mask, rows[:, 0])
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
                **extra,
            ),
        )
    write_json(args.output / "mapping_acceptance.json", acceptance)
