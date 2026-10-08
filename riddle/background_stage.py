from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import os
from pathlib import Path
import threading
import time

from .resume import check_contract, resume_policy
from .storage import atomic_torch_save, file_digest, fingerprint_files, rng_state, restore_rng, write_json

NATIVE_METHODS = ("riddle", "iad", "supervised")


def record_timing(output, name, seconds, **values):
    if os.environ.get("RIDDLE_BG_BENCHMARK") == "1":
        write_json(Path(output) / ".resume" / f"timing_{name}.json",
                   {"seconds": seconds, **values})


def stage_paths(output):
    root = Path(output) / ".resume"
    return root / "background_stage.json", root / "background_stage.pt"


def save_stage(args, contract, state, elapsed):
    receipt_path, state_path = stage_paths(args.output)
    started = time.monotonic()
    checksum = atomic_torch_save(state_path, {"state": state, "rng": rng_state()})
    artifacts = [p for directory in (args.output / "background", args.output / "density/background_correction")
                 if directory.exists() for p in directory.rglob("*")
                 if p.is_file() and ".resume" not in p.relative_to(args.output).parts]
    oracle = args.output / "oracle_roles.json"
    if oracle.is_file():
        artifacts.append(oracle)
    write_json(receipt_path, {
        "schema": 1, "completed": True, "contract": contract,
        "state_sha256": checksum,
        "artifacts_sha256": fingerprint_files(args.output, artifacts, args.io_workers),
        "preparation_seconds": elapsed, "checkpoint_seconds": time.monotonic() - started,
    })


def load_stage(args, contract):
    import torch

    receipt_path, state_path = stage_paths(args.output)
    if not receipt_path.is_file():
        return None
    if not args.resume:
        raise FileExistsError("Background preparation exists; use --resume")
    receipt = json.loads(receipt_path.read_text())
    if receipt.get("schema") != 1 or receipt.get("completed") is not True:
        raise ValueError("Invalid background-stage checkpoint")
    check_contract(receipt["contract"], contract, **resume_policy(args))
    for name, checksum in receipt["artifacts_sha256"].items():
        path = (args.output / name).resolve()
        if not path.is_relative_to(args.output.resolve()) or file_digest(path) != checksum:
            raise ValueError(f"Background-stage artifact changed: {name}")
    if file_digest(state_path) != receipt["state_sha256"]:
        raise ValueError("Background-stage state checksum mismatch")
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    restore_rng(payload["rng"])
    return payload["state"]


def run_staged(tasks, workers, run_one, *, background_only=False):
    tasks = list(tasks)
    if not tasks:
        return
    cancel = threading.Event()

    def prepare(task):
        options = deepcopy(task)
        options.background_phase = "prepare"
        options.background_concurrency = min(workers, len(tasks))
        run_one(options, cancel_event=cancel)

    pool = ThreadPoolExecutor(max_workers=min(workers, len(tasks)))
    futures = []
    try:
        futures = [pool.submit(prepare, task) for task in tasks]
        for future in as_completed(futures):
            future.result()
    except BaseException:
        cancel.set()
        for future in futures:
            future.cancel()
        raise
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    if not background_only:
        for task in tasks:
            options = deepcopy(task)
            options.background_phase = "finish"
            options.resume = True
            run_one(options)


def run_seed_backgrounds(args, run_one):
    from .worker_progress import emit_message

    for scenario in args.scenarios:
        for method in args.methods:
            tasks = []
            for seed in args.seeds:
                options = deepcopy(args)
                options.methods, options.scenarios, options.seeds = [method], [scenario], [seed]
                tasks.append(options)
            if method in NATIVE_METHODS:
                run_staged(tasks, args.scan_bg_workers, run_one,
                           background_only=getattr(args, "background_only", False))
            else:
                emit_message(f"{method}: background concurrency applies to RIDDLE, IAD and Supervised; retaining the external scheduler")
                for options in tasks:
                    options.scan_bg_workers = 1
                    run_one(options)
