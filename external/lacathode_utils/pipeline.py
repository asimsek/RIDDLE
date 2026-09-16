import ast
from contextlib import contextmanager
import inspect
import json
import os
from pathlib import Path
import pickle
import signal
import subprocess
import sys
import time

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
    # The pinned selector uses argpartition(losses, 10), requiring more than ten epochs.
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
    output = Path(options["output"])
    output.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    framework = str(Path(__file__).resolve().parents[2])
    env.update(PYTHONHASHSEED=str(options["seed"]), PYTHONUNBUFFERED="1",
               PYTHONDONTWRITEBYTECODE="1", RIDDLE_WORKER_PROGRESS="1",
               PYTHONPATH=os.pathsep.join(filter(None, (framework, env.get("PYTHONPATH")))))
    command = [sys.executable, "-m", "riddle.worker", json.dumps(options, default=str)]
    return command, env


def launch_run(options, index, count):
    from .worker_progress import EVENT_PREFIX

    output = Path(options["output"])
    command, env = run_command(options)
    with (output / "training.log").open("a" if options["resume"] else "w") as log:
        with subprocess.Popen(command, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                              text=True, bufsize=1, errors="replace") as process:
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    if line.startswith(EVENT_PREFIX):
                        event = json.loads(line[len(EVENT_PREFIX):])
                        if "phase" in event:
                            event["phase"] = f"run_{index:03d}/" + event["phase"]
                            event["label"] = f"Run {index + 1}/{count} | " + event["label"]
                        elif "message" in event:
                            event["message"] = f"Run {index + 1}/{count} | " + event["message"]
                        line = EVENT_PREFIX + json.dumps(event) + "\n"
                    print(line, end="", flush=True)
                if process.wait():
                    raise RuntimeError(f"LaCathode run {index} failed; inspect {output / 'training.log'}")
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise


def concurrent_log(state, count, *, final=False):
    from .worker_progress import EVENT_PREFIX, emit_message

    text = state["pending_log"] + state["reader"].read()
    state["pending_log"] = ""
    prefix = f"Run {state['index'] + 1}/{count} | "
    for line in text.splitlines(keepends=True):
        if not final and not line.endswith(("\n", "\r")):
            state["pending_log"] = line
            continue
        if line.startswith(EVENT_PREFIX):
            event = json.loads(line[len(EVENT_PREFIX):])
            if "message" in event:
                emit_message(prefix + event["message"], kind=event.get("kind", "INFO"),
                             level=event.get("level", 1))
            else:
                phase, completed = event["phase"], event.get("completed")
                if completed is not None and state["progress"].get(phase) != completed:
                    state["progress"][phase] = completed
                    emit_message(f"{prefix}{event['label']}: {completed}/{event['total']} {event['unit']}", kind="WORK")
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
                                         pending_log="", progress={}, timing=timing)
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
                    if code:
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


def collect_runs(args, members):
    import numpy as np
    from .storage import atomic_write, save_npz, save_array, write_json, verify_artifacts

    roots = [args.output / member["directory"] for member in members]
    reports = [json.loads((root / "result.json").read_text()) for root in roots]
    for root, report, member in zip(roots, reports, members):
        if (not report.get("completed") or report.get("method") != "lacathode"
                or report.get("seed") != member["seed"]
                or report.get("run_index") != member["run"]):
            raise ValueError("Missing or misidentified independent LaCathode run")
        verify_artifacts(root, report["artifacts_sha256"])
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
    protocol.update(run_layout=RUN_LAYOUT, pipeline_runs=len(members), runs=members,
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

    if getattr(args, "lacathode_background", "independent") == "fixed":
        if getattr(args, "workers", 1) > 1:
            emit_message("LaCathode fixed-background mode remains sequential; --workers does not parallelize its classifiers")
        return run_single(args, contract)
    if type(getattr(args, "workers", 1)) is not int or getattr(args, "workers", 1) < 1:
        raise ValueError("LaCathode workers must be a positive integer")
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

    original = classifier_type.predict

    def predict(self, x):
        device = next(self.parameters()).device
        with torch.no_grad():
            self.eval()
            x = torch.tensor(x, device=device)
            return self.forward(x).detach().cpu().numpy()

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

    settings = run_settings(getattr(args, "runs", None), getattr(args, "epochs", None),
                            getattr(args, "lacathode_background", "independent"))
    if settings["pipeline_runs"] != 1:
        raise ValueError("A single LaCathode pipeline must contain exactly one background flow")
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
    recovery = EpochRecovery(root, contract, args.resume, **resume_policy(args))
    run_ANODE_training.train_ANODE = install_epoch_recovery(
        ANODE_training_utils, "train_ANODE", "flow", recovery
    )
    classifier_training_utils.train_model = recovery.classifier_fits(
        install_epoch_recovery(classifier_training_utils, "train_model", "classifier", recovery)
    )
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
        str(settings["classifier_runs"]),
        "--DE_epochs",
        "100",
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
        emit_progress("flow", "Train background flow", total=100, unit="epoch", completed=0)
        run_all.train_DE(de)

    recovery.stage("flow", flow)

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

    def classify():
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
    evaluate(args, root, data_handler, config_file=config_file, classifier_runs=parsed.cf_n_runs)
    write_json(
        args.output / "protocol.json",
        {
            "name": "LaCathode",
            "scientific_version": "pinned_upstream",
            "flow_checkpoint_prefix": FLOW_PREFIX,
            "commit": COMMIT,
            "flow_epochs": 100,
            "classifier_epochs": parsed.cf_epochs,
            "classifier_runs": parsed.cf_n_runs,
            "run_layout": contract["lacathode_run_layout"],
            "background_mode": getattr(args, "lacathode_background", "independent"),
            "reference_samples": 267000,
            "selected_checkpoints": 10,
            "score": "Upstream ten-validation-checkpoint mean per classifier fit; no averaging across fits",
            "primary_classifier_fit": 0,
            "fit_scores": "All classifier fits, in upstream run order, when classifier_runs > 1",
            "acceleration": execution_report(acceleration),
            "configuration": {
                name: (args.sources / name).read_text() for name in (config_file, "classifier.yml")
            },
        },
    )
    verify(args.sources)


def evaluate(args, root, data_handler, *, config_file="DE_MAF_model.yml", classifier_runs=1):
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

    def evaluation_estimator(*a, **kw):
        if kw.get("load_path") is not None:
            evaluation_checkpoints.append(Path(kw["load_path"]))
        return DensityEstimator(*a, **kw)

    def capture(labels, scores):
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
    reference = data_handler.load_dataset(np.load(args.data / "outerdata_train.npy").astype("float32"))
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
