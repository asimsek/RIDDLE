import re
import textwrap
from pathlib import PurePosixPath


METHOD_NAMES = {"riddle": "RIDDLE", "iad": "IAD", "supervised": "Supervised",
                "lacathode": "LaCATHODE", "ranode": "R-ANODE"}


def compact_label(label):
    if not label:
        return label
    parts = []
    for part in str(label).split(" | "):
        lower = part.lower()
        if lower in METHOD_NAMES:
            part = METHOD_NAMES[lower]
        elif lower == "signal_injection":
            continue
        elif lower == "background_only":
            part = "BG"
        elif match := re.fullmatch(r"seed\s+(\d+)", part):
            part = f"s{int(match[1])}"
        elif match := re.fullmatch(r"signal_(\d+)", part):
            part = f"N{int(match[1])}"
        elif match := re.fullmatch(r"Fit\s+(\d+)", part):
            part = f"fit{int(match[1]):02d}"
        elif lower in {"lhco", "lhco_default"}:
            continue
        elif lower.startswith("lhco_"):
            part = part[5:]
        parts.append(part)
    return " ".join(parts)


def _fields(text):
    return dict(part.split("=", 1) for part in text.split("; ") if "=" in part)


def _clock(value):
    return value.split("T")[-1].split(".")[0].rstrip("Z") + "Z"


def compact_message(message, kind):
    text = str(message)
    io = re.fullmatch(r"I/O (START|END|RUNNING|SUMMARY); (.*)", text)
    if io:
        state, fields = io[1], _fields(io[2])
        try:
            if state == "SUMMARY":
                text = (f"I/O {fields['operations']} ops | {_clock(fields['start_utc'])}–{_clock(fields['end_utc'])}"
                        f" | wall {float(fields['elapsed_seconds']):.1f}s"
                        f" | sum I/O {float(fields['summed_io_seconds']):.1f}s")
                if int(fields.get("pending", 0)):
                    text += f" | pending {fields['pending']}"
            else:
                text = f"I/O {state.lower()} | {fields['operation']}"
                clock = fields.get("end_utc", fields.get("start_utc"))
                if clock:
                    text += f" | {_clock(clock)}"
                if "elapsed_seconds" in fields:
                    text += f" | elapsed {float(fields['elapsed_seconds']):.1f}s"
                if "path" in fields:
                    text += " | " + "/".join(PurePosixPath(fields["path"]).parts[-2:])
                if "io_id" in fields:
                    text += f" | id {fields['io_id']}"
                if fields.get("status", "completed") != "completed":
                    text += f" | {fields['status']}"
                if fields.get("error_type", "none") != "none":
                    text += f" | {fields['error_type']}"
        except (KeyError, TypeError, ValueError):
            text = str(message)
    elif text.startswith("BG stages "):
        fields = _fields(text)
        state = text.split(";", 1)[0].removeprefix("BG stages ").lower()
        text = f"BG stages {state}"
        for key, name in (("start_utc", "start"), ("end_utc", "end")):
            if fields.get(key):
                text += f" | {name} {_clock(fields[key])}"
        if "elapsed_seconds" in fields:
            text += f" | elapsed {float(fields['elapsed_seconds']):.1f}s"
        if "bg_workers" in fields:
            text += f" | workers {fields['bg_workers']}"
        if fields.get("resume_requested") == "True" and state == "start":
            text += " | resume"
        if fields.get("status", "completed") != "completed":
            text += f" | {fields['status']}"
        if fields.get("error_type", "none") != "none":
            text += f" | {fields['error_type']}"
    if kind.upper() not in {"WARNING", "ERROR"}:
        text = re.sub(r"((?:riddle|iad|supervised|lacathode|ranode) \| [^:;]+)",
                      lambda match: compact_label(match[1]), text, flags=re.IGNORECASE)
        text = re.sub(r"; last=[^;]+", "", text)
        text = re.sub(r"; minibatch=-(?=;|$)", "", text)
        text = text.replace("; elapsed=", " | elapsed ").replace("; ETA=", " | ETA ")
        text = text.replace("; fit_elapsed=", " | elapsed ").replace("; fit_ETA=", " | ETA ")
        text = re.sub(r"; file_elapsed_seconds=([\d.eE+-]+)",
                      lambda match: f" | recent file {float(match[1]):.3f}s", text)
        text = text.replace("; verify_workers=", " | verify workers ")
        text = re.sub(r"; last_advance=([^;]+?) ago", r" | idle \1", text)
        if kind.upper() == "PROGRESS":
            text = re.sub(r" \| verify workers \d+", "", text)
        for original, shortened in (
            ("Verify R-ANODE artifacts", "Verify artifacts"),
            ("Verify saved artifacts", "Verify artifacts"),
            ("Verify prepared LHCO arrays", "Verify inputs"),
            ("Verify resume contract and saved artifacts", "Verify resume"),
            ("Start Python and load dependencies", "Load dependencies"),
        ):
            text = text.replace(original, shortened)
    return text


def _embedded_task(message):
    match = re.match(r"(?:(Starting|Finished|Failed) )?(.*)", message)
    heading, rest = match[1], match[2]
    parts = rest.split(" | ")
    if len(parts) < 2 or parts[0].lower() not in METHOD_NAMES:
        return None, message
    index = 1
    while index < len(parts) and re.fullmatch(
        r"signal_injection|background_only|seed\s+\d+|signal_\d+|lhco(?:_\w+)?|deltaR|shifted|Fit\s+\d+|Background",
        parts[index], flags=re.IGNORECASE,
    ):
        index += 1
    if index == len(parts):
        return None, message
    return " | ".join(parts[:index]), (heading + " " if heading else "") + " | ".join(parts[index:])


def status_lines(message, *, kind, label, verbose, width=120):
    prefix = f"[{kind.upper()}]"
    text = str(message)
    if verbose < 2:
        if label is None:
            label, text = _embedded_task(text)
        label = compact_label(label)
        text = compact_message(text, kind)
    if label:
        prefix += f" [{label}]"
    if verbose >= 2:
        return [(prefix, text)]
    available = max(20, width - len(prefix) - 1)
    rows = []
    for line in text.splitlines() or [""]:
        rows.extend(textwrap.wrap(line, width=available, break_long_words=True, break_on_hyphens=False) or [""])
    return [(prefix, row) for row in rows]
