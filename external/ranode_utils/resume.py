from datetime import datetime, timezone
import json
from pathlib import Path
import re


def policy(args):
    options = {
        "allow_code_change": bool(getattr(args, "resume_across_code_change", False)),
        "allow_device_change": bool(getattr(args, "resume_across_device_change", False)),
    }
    if any(options.values()) and not getattr(args, "resume", False):
        raise ValueError("R-ANODE resume override options require --resume")
    return options


def differences(previous, current, path=()):
    if isinstance(previous, dict) and isinstance(current, dict):
        for key in sorted(previous.keys() | current.keys()):
            if key not in previous or key not in current:
                yield path + (key,), {
                    "previous": previous.get(key), "current": current.get(key),
                    "previous_present": key in previous, "current_present": key in current,
                }
            else:
                yield from differences(previous[key], current[key], path + (key,))
    elif previous != current:
        yield path, {"previous": previous, "current": current}


def check_contract(previous, current, *, allow_code_change=False, allow_device_change=False):
    if not isinstance(previous, dict) or not isinstance(current, dict):
        raise ValueError("Invalid R-ANODE resume contract")
    cuda_only = all(
        re.fullmatch(r"cuda:\d+", str(c.get("settings", {}).get("device", "")))
        and isinstance(c.get("environment", {}).get("gpu"), str)
        and bool(c["environment"]["gpu"])
        for c in (previous, current)
    )
    changes, protected, required = [], [], set()
    for path, values in differences(previous, current):
        if len(path) == 2 and path[0] == "code" and path[1].endswith(".py"):
            kind, allowed = "code", allow_code_change
        elif cuda_only and path in {("environment", "gpu"), ("settings", "device")}:
            kind, allowed = "device", allow_device_change
        else:
            protected.append(".".join(path))
            continue
        changes.append({"field": ".".join(path), "kind": kind, **values})
        if not allowed:
            required.add("--resume-across-" + kind + "-change")
    if protected:
        raise ValueError(
            "R-ANODE resume contract changed in protected fields: " + ", ".join(protected)
            + ". Inputs, scientific settings, pinned upstream sources, software, CPU threads "
            "and precision must match; device migration is CUDA-to-CUDA only. Use a new output."
        )
    if required:
        raise ValueError(
            "R-ANODE resume contract changed; add " + " and ".join(sorted(required))
            + " to --resume only if this migration is intentional"
        )
    return changes


def transition_history(path, previous, current, changes, *, action):
    path = Path(path)
    history = json.loads(path.read_text()) if path.exists() else {"schema": 1, "transitions": []}
    if history.get("schema") != 1 or not isinstance(history.get("transitions"), list):
        raise ValueError("Invalid R-ANODE resume history; refusing to overwrite it")
    history["transitions"].append({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "changes": changes,
        "approved_options": sorted({"--resume-across-" + c["kind"] + "-change" for c in changes}),
        "previous_contract": previous,
        "current_contract": current,
    })
    return history
