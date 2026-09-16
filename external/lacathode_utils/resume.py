"""Explicit, narrowly scoped exceptions to the saved resume contract."""

from datetime import datetime, timezone
import json
from pathlib import Path
import re


def contract_settings(runs=None, epochs=None, background="independent"):
    from .pipeline import run_settings

    return run_settings(runs, epochs, background)


def add_resume_options(parser, *, always_resume=False):
    parser.add_argument(
        "--resume", action="store_true", default=always_resume,
        help="Resume verified checkpoints (always enabled for generated jobs)"
        if always_resume else "Resume verified checkpoints",
    )
    parser.add_argument(
        "--resume-across-code-change", action="store_true",
        help="With --resume, permit implementation changes and record them; settings stay strict",
    )
    parser.add_argument(
        "--resume-across-device-change", action="store_true",
        help="With --resume, permit CUDA GPU identity changes; software and precision stay strict",
    )


def resume_policy(args):
    policy = {
        "allow_code_change": bool(getattr(args, "resume_across_code_change", False)),
        "allow_device_change": bool(getattr(args, "resume_across_device_change", False)),
    }
    if any(policy.values()) and not getattr(args, "resume", False):
        raise ValueError("Resume override options require --resume")
    return policy


def _differences(previous, current, path=()):
    if isinstance(previous, dict) and isinstance(current, dict):
        for key in sorted(previous.keys() | current.keys()):
            if key not in previous or key not in current:
                yield path + (key,), {
                    "previous": previous.get(key), "current": current.get(key),
                    "previous_present": key in previous, "current_present": key in current,
                }
            else:
                yield from _differences(previous[key], current[key], path + (key,))
    elif previous != current:
        yield path, {"previous": previous, "current": current}


def check_contract(previous, current, *, allow_code_change=False, allow_device_change=False):
    """Validate first, returning only explicitly permitted differences."""
    if not isinstance(previous, dict) or not isinstance(current, dict):
        raise ValueError("Invalid resume contract")
    cuda_only = all(
        re.fullmatch(r"cuda:\d+", str(c.get("settings", {}).get("device", "")))
        and isinstance(c.get("environment", {}).get("gpu"), str)
        and bool(c["environment"]["gpu"])
        for c in (previous, current)
    )
    changes, protected, required = [], [], set()
    for path, values in _differences(previous, current):
        field = ".".join(path)
        if len(path) == 2 and path[0] == "code" and path[1].endswith(".py"):
            kind, allowed = "code", allow_code_change
        elif cuda_only and path in {("environment", "gpu"), ("settings", "device")}:
            kind, allowed = "device", allow_device_change
        else:
            protected.append(field)
            continue
        changes.append({"field": field, "kind": kind, **values})
        if not allowed:
            required.add(f"--resume-across-{kind}-change")
    if protected:
        raise ValueError(
            "Resume contract has protected changes: " + ", ".join(protected)
            + ". Data, scientific settings, pinned sources, software, platform and precision "
            "must match; device migration is CUDA-to-CUDA only. Use a new output."
        )
    if required:
        raise ValueError(
            "Resume contract changed; add " + " and ".join(sorted(required))
            + " to --resume only if this migration is intentional"
        )
    return changes


def record_transition(path, previous, current, changes, *, action):
    """Keep before/after contracts without relabelling previously produced artifacts."""
    if not changes:
        return
    from .storage import write_json

    path = Path(path)
    history = json.loads(path.read_text()) if path.exists() else {"schema": 1, "transitions": []}
    if history.get("schema") != 1 or not isinstance(history.get("transitions"), list):
        raise ValueError("Invalid resume history; refusing to overwrite it")
    history["transitions"].append({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "changes": changes,
        "approved_options": sorted({f"--resume-across-{c['kind']}-change" for c in changes}),
        "previous_contract": previous,
        "current_contract": current,
    })
    write_json(path, history)


def inspect_resume(output, contract, *, resume=False, **policy):
    """Preflight both result and stage contracts before updating either manifest."""
    output = Path(output)
    result_path = output / "result.json"
    saved = json.loads(result_path.read_text()) if result_path.exists() else None
    changes = []
    if saved is not None:
        if not resume:
            raise FileExistsError("Result exists; use --resume or a new output")
        changes = check_contract(saved["contract"], contract, **policy)
    stage = "background" if contract["method"] == "riddle" else "training"
    stage_path = output / stage / ".resume/contract.json"
    if stage_path.exists():
        if not resume:
            raise FileExistsError("Training recovery exists; use --resume or a new output")
        check_contract(json.loads(stage_path.read_text()), contract, **policy)
    return saved, changes
