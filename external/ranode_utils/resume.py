from datetime import datetime, timezone
import json
from pathlib import Path

from riddle.resume import check_contract

__all__ = ["check_contract", "policy", "transition_history"]


def policy(args):
    options = {
        "allow_code_change": bool(getattr(args, "resume_across_code_change", False)),
        "allow_device_change": bool(getattr(args, "resume_across_device_change", False)),
    }
    if any(options.values()) and not getattr(args, "resume", False):
        raise ValueError("R-ANODE resume override options require --resume")
    return options


def transition_history(path, previous, current, changes, *, action):
    path = Path(path)
    history = json.loads(path.read_text()) if path.exists() else {"schema": 1, "transitions": []}
    if history.get("schema") != 1 or not isinstance(history.get("transitions"), list):
        raise ValueError("Invalid R-ANODE resume history; refusing to overwrite it")
    history["transitions"].append({
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "action": action,
        "changes": changes,
        "approved_options": sorted({"--resume-across-" + c["kind"] + "-change" for c in changes
                                    if c["kind"] in {"code", "device"}}),
        "previous_contract": previous,
        "current_contract": current,
    })
    return history
