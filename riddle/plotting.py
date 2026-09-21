import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import threading
import time

import numpy as np
import yaml
from matplotlib.colors import is_color_like
from scipy.special import expit
from sklearn.metrics import roc_curve

from . import figures as f
from .storage import atomic_write, file_digest, locked, write_json
from .progress import set_verbosity, colored_status, verbosity, _duration
from .worker_progress import ProgressStage, local_progress
from .metrics import acceptance_report, efficiency_curve, oracle_metrics
from .production import validate_score_record


# Physical comparisons, per-method figures and summaries share numerical work.
efficiency_curve = f.cached_plot_calculation(efficiency_curve)
oracle_metrics = f.cached_plot_calculation(oracle_metrics)
roc_curve = f.cached_plot_calculation(roc_curve)


# Presentation metadata; result discovery, not this registry, defines the cohort.
#
# New methods use result.json plus checksum-protected signal_region, validation
# and test score NPZs (mass, labels, physical, mask, scores, saved SR membership).
# Higher scores must mean more signal-like; unscored rows contain NaN. Latents are
# optional. Optional fit_scores retain the same fit order across partitions.
# Optional result.json['plotting'] sets label, color, linestyle, signal_color and
# score_transform ('identity' or 'sigmoid', for display only), and score_scope
# ('full_region' by default, or 'signal_region'). No method list in
# the comparison renderers needs editing to include a new compatible result.
@dataclass(frozen=True)
class PlotMethod:
    label: str
    color: str
    linestyle: str
    signal_color: str
    score_transform: str = "identity"
    score_scope: str = "full_region"

    @property
    def style(self):
        return self.label, self.color, self.linestyle, self.signal_color


BUILTINS = {
    "lacathode": PlotMethod("LaCathode", "#0072B2", "-", "#CC79A7"),
    "riddle": PlotMethod("RIDDLE", "#D55E00", "--", "#009E73", "sigmoid"),
    "ranode": PlotMethod("R-ANODE", "#8B1A1A", "-.", "#56B4E9", "sigmoid", "signal_region"),
}
METHOD_SPECS = dict(BUILTINS)


def register_method(report):
    """New methods need only standard score artifacts and optional plot metadata."""
    method = report.get("method")
    if not isinstance(method, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", method):
        raise ValueError("Result method must be a safe lowercase identifier")
    if method in BUILTINS:
        return BUILTINS[method]
    meta = report.get("plotting", {})
    if not isinstance(meta, dict):
        raise ValueError(f"Plotting metadata for {method} must be an object")
    palette = ("#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#332288", "#44AA99")
    index = int.from_bytes(hashlib.sha256(method.encode()).digest()[:4], "big")
    spec = PlotMethod(
        meta.get("label", method), meta.get("color", palette[index % len(palette)]),
        meta.get("linestyle", ("-", "--", "-.", ":")[(index // len(palette)) % 4]),
        meta.get("signal_color", palette[(index + 1) % len(palette)]),
        meta.get("score_transform", "identity"),
        meta.get("score_scope", "full_region"),
    )
    if (not isinstance(spec.label, str) or not spec.label.strip()
            or spec.label.casefold() in (".", "..", "comparison", "full_mass", "injection_scan", "background_only", "signal_injection")
            or "/" in spec.label or "\\" in spec.label
            or not is_color_like(spec.color) or not is_color_like(spec.signal_color)
            or spec.linestyle not in ("-", "--", "-.", ":")
            or spec.score_transform not in ("identity", "sigmoid")
            or spec.score_scope not in ("signal_region", "full_region")):
        raise ValueError(f"Invalid plotting metadata for {method}")
    if method in METHOD_SPECS and METHOD_SPECS[method] != spec:
        raise ValueError(f"Inconsistent plotting metadata for {method}")
    if any(m != method and s.label.casefold() == spec.label.casefold() for m, s in METHOD_SPECS.items()):
        raise ValueError(f"Duplicate method plot folder: {spec.label}")
    METHOD_SPECS[method] = spec
    return spec


KEYS = {"lacathode": "raw", "riddle": "residual", "ranode": "ranode"}
STYLES = dict(f.METHODS)
STYLES["ranode"] = ("R-ANODE", "#8B1A1A", "-.")


PHYSICAL_STYLES = {method: spec.style for method, spec in METHOD_SPECS.items()}


class PopulationMismatch(ValueError):
    pass


def plot_group_label(scenario, seed, group):
    names = ", ".join(STYLES[KEYS[m]][0] for m in KEYS if m in group)
    return f"{f.SCENARIO_LABELS[scenario]} | seed {seed} | {names}"


def plot_task_count(groups, scan_groups):
    return (1 + bool(scan_groups) + bool(summary_groups(groups))
            + sum(1 + len(group) for group in groups.values()))


def summary_groups(groups, *, method=None, scope="signal_region"):
    """Only combine scenarios with multiple result directories in this view."""
    eligible = {}
    counts = {}
    for identity, group in groups.items():
        cohort = scope_group(group, scope)
        if method is not None:
            cohort = {method: cohort[method]} if method in cohort else {}
        if cohort:
            eligible[identity] = cohort
            scenario = identity[0]
            counts[scenario] = counts.get(scenario, 0) + 1
    return {identity: group for identity, group in eligible.items() if counts[identity[0]] > 1}


class PlotProgress:
    """Count plotting task groups; keep nested inference quiet at verbosity 1."""

    def __init__(self, total, *, heartbeat_seconds=30):
        self.total, self.index = total, 0
        self.heartbeat_seconds = heartbeat_seconds
        self.lock = threading.RLock()
        self.label = self.detail_label = self.phase = None
        self.started = self.last_message = time.monotonic()

    def status(self, message, *, kind="WORK", level=1):
        colored_status(message, kind=kind, label=f"Plots {self.index}/{self.total}", level=level)
        self.last_message = time.monotonic()

    @contextmanager
    def task(self, label):
        with self.lock:
            if self.label is not None or self.index >= self.total:
                raise RuntimeError("Invalid plotting task plan")
            self.index += 1
            self.label, self.phase = label, None
            self.started = time.monotonic()
            self.status(label, kind="START")
        try:
            yield
        except BaseException:
            with self.lock:
                self.status(f"{label} | failed; elapsed={_duration(time.monotonic() - self.started)}",
                            kind="ERROR", level=0)
            raise
        else:
            with self.lock:
                self.status(f"{label} | done; elapsed={_duration(time.monotonic() - self.started)}",
                            kind="PASS")
        finally:
            with self.lock:
                self.label = self.detail_label = self.phase = None

    @contextmanager
    def detail(self, label):
        started = time.monotonic()
        with self.lock:
            previous = self.detail_label
            self.detail_label, self.phase = label, None
        try:
            yield
        except BaseException:
            with self.lock:
                self.status(f"{self.label} | {label} | failed", kind="ERROR", level=0)
            raise
        else:
            if verbosity() < 2:
                with self.lock:
                    self.status(f"{self.label} | {label} | done in {_duration(time.monotonic() - started)}",
                                kind="PROGRESS")
        finally:
            with self.lock:
                self.detail_label, self.phase = previous, None

    def event(self, event):
        with self.lock:
            if "message" in event:
                self.status(event["message"], kind=event.get("kind", "INFO"), level=event.get("level", 1))
                return
            phase = (event["phase"], event["label"])
            if self.phase != phase:
                self.phase = phase
                self.phase_started = time.monotonic()
                if self.label is not None and self.detail_label is None:
                    self.status(f"{self.label} | {event['label']}")
            self.completed = event.get("completed") or 0
            self.phase_total = event.get("total", 1)
            self.unit = event.get("unit", "step")

    def tick(self):
        with self.lock:
            now = time.monotonic()
            if self.label is None or now - self.last_message < self.heartbeat_seconds:
                return
            detail = self.detail_label or (self.phase[1] if self.phase else "Working")
            counter = ""
            if self.phase and self.completed < self.phase_total:
                counter = f" | {self.completed}/{self.phase_total} {self.unit}"
                if self.completed:
                    eta = (now - self.phase_started) * (self.phase_total - self.completed) / self.completed
                    counter += f"; stage ETA={_duration(eta)}"
            elif self.phase and self.detail_label is None:
                detail = "Finalizing task"
            self.status(f"{self.label} | {detail}{counter}; elapsed={_duration(now - self.started)}")


@contextmanager
def methods(keys, *, view=None):
    previous = f.METHODS, f.VIEWS
    f.METHODS = {k: STYLES[k] for k in keys}
    f.VIEWS = {view or ("comparison" if len(keys) > 1 else STYLES[keys[0]][0]): tuple(keys)}
    try:
        yield
    finally:
        f.METHODS, f.VIEWS = previous


def discover(root, requested=None, *, scan=False):
    paths = [root / "result.json"] if (root / "result.json").is_file() else sorted(root.rglob("result.json"))
    groups = {}
    variants = set()
    versions = {}
    for path in paths:
        report = json.loads(path.read_text())
        method = report.get("method")
        if (requested is not None and method not in requested) or not report.get("completed"):
            continue
        if method == "lacathode" and "run_index" in report and path.parent != root:
            continue
        point = report.get("contract", {}).get("inputs", {}).get("injection_scan")
        if bool(point) != scan:
            continue
        spec = register_method(report)
        KEYS.setdefault(method, "method_" + method)
        STYLES[KEYS[method]] = spec.style[:3]
        PHYSICAL_STYLES[method] = spec.style
        method_versions = versions.setdefault(method, set())
        method_versions.add(report.get("contract", {}).get("scientific_version", 1))
        if len(method_versions) > 1:
            raise ValueError(f"Do not combine different scientific protocols for {method}")
        variants.add(
            report.get("variant", report.get("contract", {}).get("inputs", {}).get("variant", "default"))
        )
        if len(variants) > 1:
            raise ValueError("Plot each dataset variant separately; do not mix controls in one comparison")
        scenario, seed = report["scenario"], report["seed"]
        if scenario not in f.SCENARIOS or type(seed) is not int:
            raise ValueError("Invalid result identity")
        identity = (point["signal_events"], point["replica"], report.get("run_index", 0)) if scan else (scenario, seed)
        group = groups.setdefault(identity, {})
        if method in group:
            raise ValueError("Duplicate method/scenario/seed results; choose a narrower input directory")
        group[method] = (path.parent, report)
    return groups


class InvalidFitScores(ValueError):
    """Numerically unusable predictions, distinct from damaged/misaligned files."""


class NoValidFits(ValueError):
    """A result cannot be plotted, but other method/run results can continue."""


def load_scores(root, report, name, *, attempt=None, for_rebuild=False):
    relative = Path(name + "_scores.npz")
    if attempt is not None:
        relative = Path(attempt) / relative
    path = verify_plot_input(root, report, str(relative))
    with np.load(path, allow_pickle=False) as archive:
        fields = ("mass", "labels", "mask", "scores", "physical")
        if report["method"] in ("riddle", "lacathode") or "latent" in archive:
            fields += ("latent",)
        data = {k: archive[k] for k in fields}
        for key in ("raw_scores", "score_kind", "fit_score_kind"):
            if key in archive:
                data[key] = archive[key]
        if "fit_scores" in archive:
            data["fit_scores"] = archive["fit_scores"]
            if report.get("contract", {}).get("lacathode_run_layout") == "independent_background_classifier_v1":
                data["fit_latents"] = archive["fit_latents"]
                data["run_seeds"] = archive["run_seeds"]
        if "is_signal_region" in archive:
            data["is_signal_region"] = archive["is_signal_region"]
            region = data["is_signal_region"]
            if region.shape != data["mass"].shape or region.dtype != bool:
                raise ValueError("Invalid saved signal-region membership")
            if name == "signal_region" and not region.all():
                raise ValueError("Signal-region evaluation contains non-SR events")
        if report["method"] == "ranode":
            data["is_signal_region"] = archive["is_signal_region"]
            region = data["is_signal_region"]
            if region.shape != data["mass"].shape or region.dtype != bool or (data["mask"] & ~region).any():
                raise ValueError("Invalid R-ANODE signal-region membership")
            if name == "signal_region" and not region.all():
                raise ValueError("R-ANODE evaluation contains non-SR events")
        if report["method"] == "riddle":
            from .production import validate_region

            if "is_signal_region" not in archive:
                raise ValueError("RIDDLE SR membership is missing; regenerate its score artifacts")
            data["is_signal_region"] = validate_region(archive["is_signal_region"], len(data["mass"]))
            if name == "signal_region" and not data["is_signal_region"].all():
                raise ValueError("RIDDLE signal-region evaluation contains non-SR events")
    n = len(data["mass"])
    variant = report.get("variant", report.get("contract", {}).get("inputs", {}).get("variant", "default"))
    dimensions = 5 if variant == "deltaR" else 4
    if any(data[k].shape != (n,) for k in ("mass", "labels", "scores", "mask")):
        raise ValueError("Misaligned score arrays")
    if (
        data["mask"].dtype != bool
        or data["physical"].shape != (n, dimensions)
        or ("latent" in data and data["latent"].shape != (data["mask"].sum(), dimensions))
    ):
        raise ValueError("Invalid feature/mapping shapes")
    if not np.isin(data["labels"], [0, 1]).all() or not all(
        np.isfinite(data[k]).all() for k in ("mass", "physical", "latent") if k in data
    ):
        raise ValueError("Invalid event features or labels")
    if for_rebuild:
        if report["method"] != "riddle":
            raise ValueError("Only RIDDLE checkpoint reconstruction may defer score validation")
        # Validate all event identities/features, even when an old ensemble's
        # predictions are unusable. Actual rebuilt predictions are checked below.
        evidence = {**data, "scores": np.where(data["mask"], 0., np.nan)}
        evidence.pop("fit_scores", None)
        validate_score_record(evidence, method="riddle", stage=str(path))
        data.pop("fit_scores", None)
        return data
    if (not data["mask"].any() or not np.isfinite(data["scores"][data["mask"]]).all()
            or not np.isnan(data["scores"][~data["mask"]]).all()):
        raise InvalidFitScores(f"{path}: empty/nonfinite accepted scores or invalid rejected-event scores")
    validate_score_record(data, method=report["method"], stage=str(path))
    if report["method"] == "riddle" and "method_health.json" in report.get("artifacts_sha256", {}):
        data["plot_saved_ensemble"] = True
    if report["method"] == "lacathode":
        settings = report.get("contract", {}).get("settings", {})
        independent = report.get("contract", {}).get("lacathode_run_layout") == "independent_background_classifier_v1"
        runs = settings.get("pipeline_runs") if independent else settings.get("classifier_runs")
        if independent and "run_selection.json" in report.get("artifacts_sha256", {}):
            selection = read_metadata(root, report, "run_selection.json")
            if (selection.get("status") != "completed" or selection.get("requested_runs") != runs
                    or len(selection["members"]) + len(selection["excluded_runs"]) != runs
                    or selection.get("accepted_runs") != len(selection["members"])):
                raise ValueError("Invalid LaCathode run exclusion inventory")
            runs = selection["accepted_runs"]
        fits = data.get("fit_scores", data["scores"][None, :])
        if (
            fits.ndim != 2 or fits.shape[1:] != (n,) or len(fits) < 1
            or (runs is not None and (type(runs) is not int or len(fits) != runs))
            or not np.isfinite(fits[:, data["mask"]]).all()
            or not np.isnan(fits[:, ~data["mask"]]).all()
            or not np.array_equal(fits[0], data["scores"], equal_nan=True)
        ):
            raise ValueError("Invalid or incomplete LaCathode per-fit scores")
        data["independent_runs"] = independent
        if independent and "run_index" not in report:
            members = read_metadata(root, report, "runs.json")["runs"]
            if (data["fit_latents"].shape != (len(fits), *data["latent"].shape)
                    or not np.isfinite(data["fit_latents"]).all()
                    or not np.array_equal(data["fit_latents"][0], data["latent"])
                    or not np.array_equal(data["run_seeds"], [m["seed"] for m in members])
                    or len(set(data["run_seeds"].tolist())) != len(fits)):
                raise ValueError("Invalid independent LaCathode run identities or latents")
    if report["method"] in ("riddle", "ranode"):
        data.pop("fit_scores", None)
    return data


def ranode_plot_ensemble(root, report):
    """Recover a plotting ensemble from valid saved fits, without editing results.

    One cohort is used for every partition. Exclusions depend only on numerical
    validity; weak but valid fits stay. Missing/changed files and event alignment
    errors still stop plotting rather than being mistaken for failed training.
    """
    from external.ranode_utils.ensemble import validate_mass_normalization
    from .production import NumericalFitError

    protocol = read_metadata(root, report, "protocol.json")
    attempts = protocol.get("signal_attempts") or [protocol.get("signal_attempt")]
    requested = protocol.get("requested_runs", 1)
    producer_excluded = protocol.get("excluded_fits", [])
    if (not isinstance(attempts, list) or type(requested) is not int or requested < 1
            or len(attempts) + len(producer_excluded) != requested
            or any(not isinstance(a, str) or not a for a in attempts)
            or len(set(attempts)) != len(attempts)):
        raise ValueError(f"R-ANODE {root}: incomplete signal-fit normalization evidence")
    indices = {m["attempt"]: m["fit_index"] for m in protocol.get("members", [])}
    partitions = ("validation", "test", "signal_region")
    fields = ("mass", "physical", "labels", "mask", "is_signal_region")
    accepted, excluded, base, combined = [], [dict(
        fit_index=m["fit_index"], attempt=m["attempt"], reason=m["error"], check="producer_numerical_validity")
        for m in producer_excluded], {}, {}
    additional_exclusions = False
    for index, attempt in enumerate(attempts):
        member = {"fit_index": indices.get(attempt, index), "attempt": attempt}
        # Integrity errors are intentionally outside the numerical-failure catch.
        verify_plot_input(root, report, str(Path(attempt) / "results/upstream/signal/fit/samples.npy"))
        try:
            validate_mass_normalization(root / attempt)
        except NumericalFitError as error:
            excluded.append({**member, "reason": str(error), "check": "mass_normalization"})
            additional_exclusions = True
            continue
        try:
            records = {p: load_scores(root, report, p, attempt=attempt) for p in partitions}
        except InvalidFitScores as error:
            excluded.append({**member, "reason": str(error), "check": "score_validity"})
            additional_exclusions = True
            continue
        for partition, record in records.items():
            mask = record["mask"]
            if partition not in base:
                base[partition] = record
                combined[partition] = record["scores"][mask].astype(np.float64)
            else:
                if any(not np.array_equal(base[partition][key], record[key]) for key in fields):
                    raise ValueError(f"R-ANODE {root}: fit {member['fit_index']} has misaligned {partition} events")
                combined[partition] = np.logaddexp(combined[partition], record["scores"][mask])
        accepted.append(member)
    audit = {
        "method": "ranode", "source": str(root.resolve()), "scenario": report["scenario"], "seed": report["seed"],
        "requested_fits": requested, "used_fits": len(accepted), "used_members": accepted, "excluded_members": excluded,
        "status": "rebuilt_from_valid_fits" if additional_exclusions and accepted else "saved_ensemble" if accepted else "no_valid_fits",
        "selection": "Numerical validity only; no selection on AUC, SIC or signal labels",
        "ensemble": "log(mean(exp(per-fit log ratio))) with equal weights over used fits",
        "partitions": list(partitions), "independent_runs": 1,
        "source_artifacts_modified": False,
        "safeguard_filtering": True,
    }
    if additional_exclusions:
        for partition, record in base.items():
            if len(accepted) > 1:
                record["scores"] = record["scores"].copy()
                record["scores"][record["mask"]] = combined[partition] - np.log(len(accepted))
            validate_score_record(record, method="ranode", stage=f"Rebuilt {root}/{partition}")
    else:
        # Preserve the exact saved predictions when every fit passes.
        base = {p: load_scores(root, report, p) for p in partitions}
    for record in base.values():
        record["plot_ensemble"] = audit
    return base, audit


def plot_device_argument(value):
    if value not in ("cpu", "auto") and not re.fullmatch(r"cuda:\d+", value):
        raise argparse.ArgumentTypeError("--device must be cpu, auto, or cuda:<index>")
    return value


def resolve_plot_device(requested):
    """Resolve only when checkpoint inference is needed; other plots need no CUDA."""
    import torch

    plot_device_argument(requested)
    if requested == "auto":
        requested = "cuda:0" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda:"):
        if not torch.cuda.is_available():
            raise RuntimeError(f"Plot safeguard inference requested {requested}, but CUDA is unavailable; "
                               "use --device cpu or --device auto")
        index = int(requested.split(":")[1])
        if index >= torch.cuda.device_count():
            raise ValueError(f"Plot device {requested} is outside the visible CUDA device set")
        requested = f"cuda:{index}"
    return requested


def riddle_plot_predictions(root, report, member, groups, *, device="cpu"):
    """Read-only v2/v3 inference; production training/resume version rules stay strict."""
    import torch
    from .model import build_signal_flow, background_log_prob, signal_log_prob

    relative = Path("density") / member["directory"]
    inputs = read_metadata(root, report, str(relative / "residual_training_inputs.json"))
    version = report["contract"]["scientific_version"]
    if version not in (2, 3) or inputs.get("scientific_version") != version:
        raise ValueError("Unsupported or inconsistent RIDDLE checkpoint protocol")
    paths = [verify_plot_input(root, report, str(relative / f"residual_epoch_{e}.pt"))
             for e in member["epochs"]]
    for z in groups.values():
        if z.ndim != 2 or z.shape[1] != inputs["features"] or not len(z) or not np.isfinite(z).all():
            raise ValueError("Invalid RIDDLE inference latents")
    model = build_signal_flow(device, features=inputs["features"], settings=inputs["settings"]).eval()
    correction = inputs.get("background_correction")
    corrected_background = None
    if correction is not None:
        from .background_correction import load_local, log_prob as corrected_log_prob
        corrected_background, saved_correction = load_local(root / relative, inputs["settings"], inputs["features"]-1, device)
        if saved_correction.get("source_sha256") != correction["source_sha256"]:
            raise ValueError("RIDDLE plot denominator identity changed")
        from .enhancements import chunks as inference_chunks
        backgrounds = {}
        for key, z in groups.items():
            t = torch.from_numpy(z)
            backgrounds[key] = inference_chunks(
                lambda x, c: corrected_log_prob(corrected_background, x, c),
                t[:, :-1], t[:, -1], device=device, size=8192).numpy().astype(np.float64)
    else:
        backgrounds = {key: background_log_prob(
            torch.from_numpy(z), mass_conditioning=model.mass_conditioning,
            physical_inputs=model.physical_inputs).numpy().astype(np.float64)
            for key, z in groups.items()}
    sums, mixture = {}, None
    total = len(paths) * sum(len(z) for z in groups.values())
    completed = 0
    with torch.no_grad(), ProgressStage("plot_riddle_inference", "Evaluate saved RIDDLE fit", total, "prediction") as progress:
        for path, epoch, weight in zip(paths, member["epochs"], member["signal_fractions"]):
            # Stage checkpoint tensors on CPU, then copy into the model on the
            # selected device. Keep only one inference batch in GPU memory.
            checkpoint = torch.load(path, map_location="cpu", weights_only=True)
            if checkpoint.get("scientific_version") != version or checkpoint.get("epoch") != epoch:
                raise ValueError("RIDDLE checkpoint identity differs from saved selection")
            model.load_state_dict(checkpoint["model"])
            for key, z in groups.items():
                chunks = []
                for offset in range(0, len(z), 8192):
                    batch = z[offset:offset + 8192]
                    x = torch.from_numpy(batch).to(device)
                    chunks.append(signal_log_prob(model, x).cpu().numpy())
                    completed += len(batch)
                    progress.update(completed)
                ratio = np.concatenate(chunks).astype(np.float64) - backgrounds[key]
                if not np.isfinite(ratio).all():
                    raise FloatingPointError(f"Nonfinite RIDDLE {key} scores at checkpoint {epoch}")
                if key == "reference":
                    from .production import validate_density_ratio
                    validate_density_ratio(ratio, stage=f"{member['directory']} checkpoint {epoch}",
                                           tests=member["normalization_tests"])
                sums[key] = ratio if key not in sums else np.logaddexp(sums[key], ratio)
                if key == "reserved_validation":
                    weighted = np.logaddexp(np.log1p(-weight) if weight < 1 else -np.inf,
                                            (np.log(weight) if weight > 0 else -np.inf) + ratio)
                    mixture = weighted if mixture is None else np.logaddexp(mixture, weighted)
    count = len(paths)
    return {key: value - np.log(count) for key, value in sums.items()}, (
        None if mixture is None else mixture - np.log(count))


def riddle_plot_ensemble(root, report, *, io_workers=2, device="cpu"):
    """Filter numerical failures only; record reserved-validation evidence separately."""
    import torch
    from .model import real_sr_latents
    from .production import (NumericalFitError, require_complete_ensemble,
                             validate_density_ratio, validation_improvement, fit_acceptance)
    from .storage import digest

    device = resolve_plot_device(device)
    cuda_devices = [int(device.split(":")[1])] if device.startswith("cuda:") else []
    colored_status(f"RIDDLE safeguard checkpoint inference | device={device}", kind="INFO", level=1)
    selection = read_metadata(root, report, "density/ensemble_selection.json")
    require_complete_ensemble(selection)
    if "method_health.json" in report.get("artifacts_sha256", {}):
        # A calibrated score is tied to its exact frozen ensemble. Removing fits
        # after calibration would silently change the method and invalidate u.
        health = read_metadata(root, report, "method_health.json")
        if not fit_acceptance(health)["fit_valid"]:
            raise NumericalFitError("Invalid saved RIDDLE ensemble; rerun training rather than recalibrating while plotting")
        base = {p: load_scores(root, report, p) for p in ("validation", "test", "signal_region")}
        audit = dict(method="riddle", source=str(root.resolve()), scenario=report["scenario"], seed=report["seed"],
                     status="saved_ensemble", requested_fits=selection["requested_runs"], used_fits=len(selection["members"]),
                     used_members=selection["members"], excluded_members=[], independent_runs=1,
                     source_artifacts_modified=False, safeguard_filtering=True, checkpoint_inference=False,
                     health=health, ensemble="immutable production ensemble with its saved score calibration")
        for record in base.values():
            record["plot_ensemble"] = audit
        return base, audit
    inputs = read_metadata(root, report, "density/ensemble_inputs.json")
    path = verify_plot_input(root, report, "background/validation_latents.npy")
    validation = real_sr_latents(np.load(path, allow_pickle=False))
    if digest(validation) != inputs.get("selection_sha256"):
        raise ValueError("Reserved RIDDLE validation events differ from the training selection")
    reference = np.random.default_rng(3407).standard_normal((8192, validation.shape[1])).astype(np.float32)
    # Match the production validation rule; legacy runs predate this setting.
    policy = inputs.get("settings", {}).get("fit_recovery", {})
    sigma = policy.get("validation_sigma", 2.0)
    members = [dict(m, fit_index=m.get("fit_index", i)) for i, m in enumerate(selection["members"])]
    tests = sum(len(m["epochs"]) + 1 for m in members)
    for member in members:
        member["normalization_tests"] = tests
    accepted, excluded, combined, base = [], [], {}, {}
    audit = dict(method="riddle", source=str(root.resolve()), scenario=report["scenario"], seed=report["seed"],
                 requested_fits=selection["requested_runs"], used_fits=0, used_members=[], excluded_members=excluded,
                 status="no_valid_fits", selection="Numerical validity only; reserved-validation evidence is diagnostic; historical production exclusions retained",
                 validation_sigma=sigma, reference_seed=3407, reference_samples=len(reference),
                 ensemble="log(mean(exp(per-fit log ratio))) with equal weights over used fits",
                 partitions=["validation", "test", "signal_region"], independent_runs=1,
                 source_artifacts_modified=False, safeguard_filtering=True, checkpoint_inference=True,
                 inference_device=device)
    previous_threads = torch.get_num_threads()
    previous_tf32 = torch.backends.cuda.matmul.allow_tf32
    try:
        torch.set_num_threads(io_workers)
        if cuda_devices:
            # Keep full float32 model evaluation; likelihood aggregation and
            # safeguard statistics remain float64 on CPU as before.
            torch.backends.cuda.matmul.allow_tf32 = False
        with torch.random.fork_rng(devices=cuda_devices):
            for member in members:
                colored_status(f"RIDDLE fit {member['fit_index']:03d}: check saved checkpoints on reserved validation",
                               kind="WORK", level=1)
                try:
                    scores, mixture = riddle_plot_predictions(root, report, member,
                        {"reserved_validation": validation, "reference": reference}, device=device)
                    validate_density_ratio(scores["reference"], stage=f"{member['directory']} ensemble", tests=tests)
                    quality = validation_improvement(mixture, sigma=sigma)
                except (NumericalFitError, FloatingPointError) as error:
                    excluded.append(dict(member, check="numerical_validity", reason=str(error)))
                    continue
                accepted.append(dict(member, quality=quality, **fit_acceptance(
                    dict(normalization_status="passed", quality=quality))))
            if accepted:
                base = {p: load_scores(root, report, p, for_rebuild=True) for p in audit["partitions"]}
                # Keep the saved prediction exactly when no fit was rejected and it is usable.
                valid_saved_scores = all(np.isfinite(r["scores"][r["mask"]]).all()
                                         and np.isnan(r["scores"][~r["mask"]]).all() for r in base.values())
                if excluded or not valid_saved_scores:
                    surviving = []
                    for member in accepted:
                        colored_status(f"RIDDLE fit {member['fit_index']:03d}: rebuild scores for all plot regions",
                                       kind="WORK", level=1)
                        try:
                            predictions, _ = riddle_plot_predictions(root, report, member,
                                {p: r["latent"] for p, r in base.items()}, device=device)
                        except (NumericalFitError, FloatingPointError) as error:
                            excluded.append(dict(member, check="score_validity", reason=str(error)))
                            continue
                        for p, values in predictions.items():
                            combined[p] = values if p not in combined else np.logaddexp(combined[p], values)
                        surviving.append(member)
                    accepted = surviving
                    for p, record in base.items():
                        if accepted:
                            record["scores"] = np.full(len(record["mask"]), np.nan)
                            record["scores"][record["mask"]] = combined[p] - np.log(len(accepted))
                if accepted:
                    audit["status"] = "rebuilt_from_valid_fits" if excluded or not valid_saved_scores else "saved_ensemble"
    finally:
        torch.set_num_threads(previous_threads)
        if cuda_devices:
            torch.backends.cuda.matmul.allow_tf32 = previous_tf32
    audit.update(used_fits=len(accepted), used_members=accepted)
    if not accepted:
        return {}, audit
    for p, record in base.items():
        validate_score_record(record, method="riddle", stage=f"RIDDLE plot ensemble {root}/{p}")
        record["plot_ensemble"] = audit
    return base, audit


def unfiltered_plot_ensemble(root, report):
    """Opt out of plot-time selection, without undoing production exclusions."""
    if report["method"] == "riddle":
        selection = read_metadata(root, report, "density/ensemble_selection.json")
        members = [dict(m, fit_index=m.get("fit_index", i)) for i, m in enumerate(selection["members"])]
        config = next(c for c in selection["configurations"] if c["name"] == selection["selected_configuration"])
        excluded = config.get("excluded_fits", [])
        requested = selection["requested_runs"]
    else:
        selection = read_metadata(root, report, "protocol.json")
        attempts = selection.get("signal_attempts") or [selection["signal_attempt"]]
        members = selection.get("members", [dict(fit_index=i, attempt=a) for i, a in enumerate(attempts)])
        excluded, requested = selection.get("excluded_fits", []), selection.get("requested_runs", 1)
    records = {p: load_scores(root, report, p) for p in ("validation", "test", "signal_region")}
    audit = dict(method=report["method"], source=str(root.resolve()), scenario=report["scenario"], seed=report["seed"],
                 status="filtering_disabled", safeguard_filtering=False, checkpoint_inference=False,
                 requested_fits=requested, used_fits=len(members), used_members=members, excluded_members=excluded,
                 selection="Exact saved ensemble; plot-time filtering disabled; production exclusions retained",
                 independent_runs=1, source_artifacts_modified=False)
    for record in records.values():
        record["plot_ensemble"] = audit
    return records, audit


class ScoreLoader:
    """Load saved method predictions once; ensemble members are not repeat runs."""

    def __init__(self, *, safeguard_filtering=True, io_workers=2, device="cpu"):
        self.cache = {}
        self.validated = set()
        self.fit_audit = {}
        self.safeguard_filtering = safeguard_filtering
        self.io_workers = io_workers
        self.device = plot_device_argument(device)

    def __call__(self, root, report, partition):
        identity = str(root.resolve())
        if identity not in self.validated:
            method = report["method"]
            legacy_riddle = method == "riddle" and report.get("contract", {}).get("scientific_version", 1) < 3
            if method in ("riddle", "ranode") and (not self.safeguard_filtering or legacy_riddle or method == "ranode"):
                if not self.safeguard_filtering:
                    records, audit = unfiltered_plot_ensemble(root, report)
                elif legacy_riddle:
                    records, audit = riddle_plot_ensemble(root, report, io_workers=self.io_workers, device=self.device)
                else:
                    records, audit = ranode_plot_ensemble(root, report)
                self.fit_audit[identity] = audit
                if not records:
                    raise NoValidFits(f"{BUILTINS[method].label} {root}: no saved fits passed safeguards; skipping this result")
                self.cache.update({(identity, p): record for p, record in records.items()})
                if self.safeguard_filtering and audit["excluded_members"]:
                    indices = ", ".join(f"{m['fit_index']:03d}" for m in audit["excluded_members"])
                    colored_status(f"{BUILTINS[method].label} {report['scenario']}/seed_{report['seed']:03d}: "
                                   f"using {audit['used_fits']}/{audit['requested_fits']} saved fits; "
                                   f"excluded fits {indices}; accepted ensemble verified", kind="INFO", level=0)
            if self.safeguard_filtering and method == "riddle" and not legacy_riddle:
                from .production import require_complete_ensemble
                selection = read_metadata(root, report, "density/ensemble_selection.json")
                require_complete_ensemble(selection)
                chosen = next(c for c in selection["configurations"] if c["name"] == selection["selected_configuration"])
                self.fit_audit[identity] = dict(method="riddle", source=identity, status="producer_selection",
                    scenario=report["scenario"], seed=report["seed"], source_artifacts_modified=False,
                    requested_fits=selection["requested_runs"], used_fits=selection["valid_runs"],
                    used_members=selection["members"], excluded_members=chosen["excluded_fits"],
                    fit_recovery=selection["fit_recovery"], selection=selection["selection"], independent_runs=1,
                    safeguard_filtering=True, checkpoint_inference=False)
            if report["method"] == "lacathode" and "run_selection.json" in report.get("artifacts_sha256", {}):
                selection = read_metadata(root, report, "run_selection.json")
                self.fit_audit[identity] = dict(method="lacathode", source=identity, status="producer_selection",
                    requested_fits=selection["requested_runs"], used_fits=selection["accepted_runs"],
                    used_members=selection["members"], excluded_members=selection["excluded_runs"],
                    selection=selection["selection"], independent_runs=selection["accepted_runs"])
            if (self.safeguard_filtering and not legacy_riddle
                    and "density/normalization_check.json" in report.get("artifacts_sha256", {})):
                audit = read_metadata(root, report, "density/normalization_check.json")
                if audit.get("status") != "passed":
                    raise ValueError(f"{root}: failed density normalization check")
            self.validated.add(identity)
        key = (str(root.resolve()), partition)
        if key not in self.cache:
            self.cache[key] = load_scores(root, report, partition)
        if identity in self.fit_audit:
            self.cache[key]["plot_ensemble"] = self.fit_audit[identity]
        return self.cache[key]


def preflight_scores(groups, loader, *, allow_smoke=False):
    """Finish scientific input checks before --overwrite removes any figures."""
    seen, skipped = set(), set()
    for group in groups:
        for method, (root, report) in list(group.items()):
            identity = str(root.resolve())
            if identity in skipped:
                del group[method]
                continue
            if identity in seen:
                continue
            seen.add(identity)
            if report.get("contract", {}).get("inputs", {}).get("synthetic_smoke_fixture") and not allow_smoke:
                raise ValueError("Synthetic fixtures require --allow-smoke for QA")
            try:
                for partition in ("validation", "test", "signal_region"):
                    loader(root, report, partition)
            except NoValidFits as error:
                skipped.add(identity)
                del group[method]
                colored_status(str(error), kind="WARNING")


def run_scores(record):
    """One ensemble prediction per full run, or saved independent LaCathode runs."""
    return fit_scores(record) if record.get("independent_runs", False) else record["scores"][None, :]


def run_score_groups(method, record):
    if method == "lacathode" and not record.get("independent_runs", False):
        return [fit_scores(record)]
    return [[score] for score in run_scores(record)]


def sic_on_grid(background, signal):
    unique, inverse = np.unique(background, return_inverse=True)
    if len(unique) < 2:
        return np.full(len(f.GRID), np.nan)
    maximum = np.zeros(len(unique))
    np.maximum.at(maximum, inverse, signal)
    return np.interp(f.GRID, unique, maximum / np.sqrt(unique), left=np.nan, right=np.nan)


def fit_scores(record):
    return f.fit_scores(record)


def fit_identity(bundle, key, fit):
    record = bundle["evaluation"]["signal_region"][key]
    seed = bundle["report"]["seed"]
    return dict(seed=seed, fit=fit,
                run_seed=int(record["run_seeds"][fit]) if "run_seeds" in record else seed,
                independent=True, source="independent_complete_method_runs")


def fit_uncertainty(identities):
    sources = {item["source"] for item in identities if "source" in item}
    if sources:
        return next(iter(sources)) if len(sources) == 1 else "mixed_run_structures"
    if identities and all(item.get("independent", False) for item in identities):
        return "independent_background_flow_and_classifier_runs"
    return "classifier_fit_variation_with_shared_background_flow_per_seed"


def require_independent_run_ids(identities):
    seeds = [item.get("run_seed", item["seed"]) for item in identities]
    if len(seeds) != len(set(seeds)):
        raise ValueError("Repeated training seeds cannot be counted as independent uncertainty runs")


def make_bundle(group, confidence, score_loader=None):
    score_loader = load_scores if score_loader is None else score_loader
    keys = [KEYS[m] for m in group]
    bundle = {"samples": {}, "curves": {}, "warnings": [], "sources": group, "latents_by_method": {},
              "evaluation": {}, "acceptance": {}}
    for partition in ("validation", "test", "signal_region"):
        records = {KEYS[m]: score_loader(root, report, partition) for m, (root, report) in group.items()}
        bundle["evaluation"][partition] = records
        from .production import region_acceptance
        bundle["acceptance"][partition] = {
            k: (region_acceptance(r["labels"], r["mask"], r["is_signal_region"])
                if k == "residual" else acceptance_report(r["labels"], r["mask"], r["mass"]))
            for k, r in records.items()
        }
        base = records[keys[0]]
        for record in records.values():
            if any(not np.array_equal(base[k], record[k]) for k in ("mass", "labels", "mask", "physical")):
                raise PopulationMismatch(
                    "Methods have different evaluation populations or mapping masks; plot them separately"
                )
        sr_masks = {k: r["is_signal_region"] for k, r in records.items() if k == "residual"}
        if partition != "signal_region" and "residual" in sr_masks and "raw" in records:
            if not np.array_equal(sr_masks["residual"], f.sr(records["raw"]["mass"])):
                raise PopulationMismatch(
                    "Methods have different SR memberships; separate plots preserve LaCathode's unchanged definition"
                )
        sample = {k: base[k] for k in ("mass", "labels", "mask")}
        sample["sr_masks"] = sr_masks
        sample.update({k + "_scores": r["scores"] for k, r in records.items()})
        sample.update({k + "_fit_scores": f.fit_scores(r) for k, r in records.items() if "fit_scores" in r})
        if partition == "signal_region":
            bundle["curves"]["signal_region"] = (
                base["labels"][base["mask"]],
                {k: r["scores"][base["mask"]] for k, r in records.items()},
            )
        else:
            bundle["samples"][partition] = sample
        if partition == "test":
            bundle["feature_values"] = np.column_stack((base["mass"], base["physical"]))
            bundle["latents_by_method"] = {k: r.get("fit_latents", r["latent"]) for k, r in records.items()}
    report = next(iter(group.values()))[1]
    if "lacathode" in group:
        bundle["classifier_fit_count"] = f.validate_fit_counts(
            [records["raw"] for records in bundle["evaluation"].values()]
        )
    bundle["report"] = report
    bundle["variant"] = report.get(
        "variant", report.get("contract", {}).get("inputs", {}).get("variant", "default")
    )
    with methods(keys):
        f.build_metrics(bundle, confidence)
    if "classifier_fit_count" in bundle:
        bundle["metrics"]["lacathode_fit_count"] = bundle["classifier_fit_count"]
        bundle["metrics"]["lacathode_histograms"] = "Mean counts per fit; separate validation cut per fit; no score averaging or event pooling"
    return bundle


def render_bundle(bundle, target, args):
    group = bundle["sources"]
    keys = [KEYS[m] for m in group]
    with methods(keys, view="."):
        for renderer in (
            f.render_roc,
            f.render_scores,
            f.render_efficiency,
            f.render_mass,
            f.render_mass_scan,
            f.render_features,
        ):
            if renderer is f.render_mass_scan:
                renderer(bundle, target)  # This renderer reports its own 70-cut progress.
                continue
            with ProgressStage(renderer.__name__, renderer.__name__.replace("render_", "Plot ")):
                renderer(bundle, target, args) if renderer in (
                    f.render_roc,
                    f.render_efficiency,
                ) else renderer(bundle, target)
    for key in keys:
        with methods([key], view="."), ProgressStage("representation", "Plot input and latent distributions"):
            f.render_representation({**bundle, "latents": bundle["latents_by_method"][key]}, target)
    with ProgressStage("evaluation_plots", "Plot signal-region performance"):
        render_full_pipeline(bundle, target, args)
    bundle["metrics"]["mapping_acceptance"] = bundle["acceptance"]
    write_json(target / "metrics.json", json_safe(bundle["metrics"]))


def render_full_pipeline(bundle, output, args):
    records = bundle["evaluation"]["signal_region"]
    audit = {}
    for metric in ("roc", "sic"):
        fig, ax, _ = f.canvas("Signal efficiency" if metric == "roc" else "Significance improvement",
                             "Background efficiency")
        drawn = False
        for key, record in records.items():
            if len(np.unique(record["labels"][record["mask"]])) != 2:
                continue
            if len(fit_scores(record)) > 1:
                curves, metrics = [], []
                for score in fit_scores(record):
                    b, s, _ = efficiency_curve(record["labels"], score, record["mask"], full_pipeline=True)
                    use = b > 0 if metric == "roc" else (b >= 1e-4) & (np.rint(b * (record["labels"] == 0).sum()) >= args.min_background)
                    curves.append((b[use], s[use]))
                    metrics.append(oracle_metrics(record["labels"], score, record["mask"], min_background=args.min_background))
                label, color, ls = STYLES[key]
                summary = f.draw_fit_curves(ax, curves, metric, label, color, ls,
                                            uncertainty_source=f.fit_uncertainty(record),
                                            band=record.get("independent_runs", False),
                    run_groups=None if record.get("independent_runs", False) else [0] * len(curves))
                audit.setdefault(label, {}).update(f.aggregate_fit_metrics(metrics))
                audit[label][metric] = summary
                drawn = True
                continue
            b, s, _ = efficiency_curve(record["labels"], record["scores"], record["mask"], full_pipeline=True)
            supported = (b >= 1e-4) & (np.rint(b * (record["labels"] == 0).sum()) >= args.min_background)
            keep = b > 0 if metric == "roc" else supported
            label, color, ls = STYLES[key]
            if keep.any():
                ax.plot(b[keep], s[keep] if metric == "roc" else s[keep]/np.sqrt(b[keep]),
                        color=color, ls=ls, label=label)
                drawn = True
            audit[label] = oracle_metrics(record["labels"], record["scores"], record["mask"],
                                          min_background=args.min_background)
        if drawn:
            ax.set(xscale="log", xlim=(1e-4, 1), ylim=(0, None))
            f.legend(fig, title=f.SCENARIO_LABELS[bundle["report"]["scenario"]])
            f.save(fig, output / "full_pipeline" / ("signal_region_" + metric))
        else:
            f.plt.close(fig)
    bundle["metrics"]["oracle_benchmarks"] = audit


def render_injection_scan(groups, output, args, score_loader=None):
    score_loader = load_scores if score_loader is None else score_loader
    rows, cohorts, classifier_fits = [], {}, []
    for identity, group in sorted(groups.items()):
        count, replica = identity[:2]
        reference = None
        for method, (root, report) in group.items():
            inputs = report["contract"]["inputs"]
            if inputs.get("synthetic_smoke_fixture") and not args.allow_smoke:
                raise ValueError("Synthetic scan requires --allow-smoke for QA")
            expected_protocol = "pinned_upstream"
            if method == "ranode":
                from external.ranode_utils.data import scientific_version

                expected_protocol = scientific_version(inputs.get("variant", "default"))
            allowed_protocols = (2, 3) if method == "riddle" else (expected_protocol,)
            if method in BUILTINS and report["contract"].get("scientific_version") not in allowed_protocols:
                raise ValueError(f"Injection scans require protocol in {allowed_protocols!r} for {method}")
            point = inputs["injection_scan"]
            from .cli import independent_run_seeds

            run_index = report.get("run_index", 0)
            expected_seed = independent_run_seeds(point["training_seed"], run_index + 1)[-1]
            if report["seed"] != expected_seed:
                raise ValueError("Scan result has an incompatible training seed")
            if reference is not None and inputs != reference:
                raise ValueError("Scan methods must share identical prepared data at each point")
            reference = inputs
            data = score_loader(root, report, "signal_region")
            values = oracle_metrics(data["labels"], data["scores"], data["mask"],
                                    min_background=args.min_background)
            if "fit_scores" in data:
                fit_values = [
                    oracle_metrics(data["labels"], score, data["mask"], min_background=args.min_background)
                    for score in fit_scores(data)
                ]
                classifier_fits.append(dict(method=method, signal_events=count, replica=replica,
                                            training_seed=report["seed"], fits=fit_values))
                for field in ("conditional_auc", "conditional_max_sic", "full_pipeline_max_sic"):
                    finite = [v[field] for v in fit_values if v[field] is not None and np.isfinite(v[field])]
                    values[field] = float(np.median(finite)) if finite else None
            initial = inputs["uncut_signal_region"]
            b0, s0 = initial["background"], initial["signal"]
            if b0 <= 0 or s0 <= 0:
                raise ValueError("Positive realized SR background and signal counts required for a scan")
            maximum = values["full_pipeline_max_sic"]
            fixed = [sic_at_background(data, score, 1e-3, args.min_background) for score in fit_scores(data)]
            fixed_sic = float(np.median(fixed)) if all(v is not None for v in fixed) else None
            row = dict(method=method, signal_events=count, replica=replica, run_index=run_index,
                       training_seed=report["seed"], preparation_seed=point["preparation_seed"],
                       sr_background=b0, sr_signal=s0, signal_to_background_percent=100*s0/b0,
                       uncut_nominal_significance=s0/np.sqrt(b0),
                       conditional_mapped_auc=values["conditional_auc"],
                       oracle_conditional_max_sic=values["conditional_max_sic"],
                       oracle_full_pipeline_max_sic=maximum,
                       oracle_max_nominal_significance=maximum*s0/np.sqrt(b0) if maximum is not None else None,
                       sic_at_background_1e3=fixed_sic,
                       nominal_significance_at_background_1e3=fixed_sic*s0/np.sqrt(b0) if fixed_sic is not None else None,
                       background_mapping_acceptance=values["acceptance"]["background"]["acceptance"],
                       signal_mapping_acceptance=values["acceptance"]["signal"]["acceptance"])
            rows.append(row)
            cohorts.setdefault((method, count), []).append(row)
    csv_write(output / "injection_scan.csv", rows)
    audit = dict(
        metric="Oracle maximum on truth-labelled SR test sample; mapping failures never pass",
        nominal_significance="Per replica: maximum SIC times realized uncut SR S/sqrt(B), then aggregate",
        uncertainty="16/50/84 percentiles across independently seeded partitions and training; finite source pool is reused, not independent collision datasets",
        minimum_background_count=args.min_background, minimum_background_efficiency=1e-4,
        fixed_efficiency="SIC at physical background efficiency 0.001; linear ROC interpolation within supported points only",
        significance_caveat="Nominal S/sqrt(B); no systematics, background fit, Poisson calibration or trials correction",
        points=[], plotted_metrics={},
    )
    for field, title, filename in (
        ("oracle_full_pipeline_max_sic", "Maximum significance improvement", "maximum_sic_vs_injection"),
        ("oracle_max_nominal_significance", "Maximum nominal significance", "maximum_nominal_significance_vs_injection"),
        ("sic_at_background_1e3", r"SIC at $\epsilon_B=10^{-3}$", "sic_at_fixed_background_vs_injection"),
        ("nominal_significance_at_background_1e3", r"Nominal significance at $\epsilon_B=10^{-3}$", "significance_at_fixed_background_vs_injection"),
    ):
        fig, ax, _ = f.canvas(title, "Injected SR S/B [%]")
        any_drawn = False
        plotted = {}
        for method in KEYS:
            values = []
            for (name, count), cohort in cohorts.items():
                if name != method:
                    continue
                finite = [r for r in cohort if r[field] is not None and np.isfinite(r[field])]
                if not finite:
                    continue
                x = float(np.median([r["signal_to_background_percent"] for r in finite]))
                low, median, high = np.quantile([r[field] for r in finite], [.16, .5, .84])
                values.append((x, low, median, high, len(finite), count))
            if not values:
                continue
            values.sort()
            matrix = np.asarray(values)
            x, low, median, high = matrix[:, :4].T
            name, color, ls = STYLES[KEYS[method]]
            ax.plot(x, median, marker="x", color=color, ls=ls, label=name)
            # A single replica has no uncertainty band, even beside multi-replica points.
            supported = matrix[:, 4] >= 2
            if supported.any():
                ax.fill_between(x, low, high, where=supported, color=color, alpha=.22)
            if not supported.all():
                colored_status(f"{name}: scan band unavailable at single-replica points", kind="WARNING")
            plotted[method] = [dict(x=float(a), low=float(l), median=float(m), high=float(h),
                                   replicas=int(n), signal_events=int(c)) for a,l,m,h,n,c in values]
            any_drawn = True
        if any_drawn:
            if field == "oracle_max_nominal_significance":
                for level in (3, 5):
                    ax.axhline(level, ls=":", color=".5", lw=1)
            ax.set_ylim(bottom=0)
            ax.invert_xaxis()
            backgrounds = np.asarray([r["sr_background"] for r in rows])
            if len(backgrounds) and np.all(backgrounds == backgrounds[0]):
                factor = np.sqrt(backgrounds[0])/100
                top = ax.secondary_xaxis("top", functions=(lambda x: x*factor, lambda x: x/factor))
                top.set_xlabel(r"Uncut $S/\sqrt{B}$")
            f.legend(fig, title="Signal-Injected")
            f.save(fig, output / filename)
        else:
            f.plt.close(fig)
        audit["plotted_metrics"][field] = plotted
    audit["points"] = rows
    if classifier_fits:
        audit["lacathode_classifier_fits"] = classifier_fits
        audit["lacathode_aggregation"] = "Median of per-fit metrics within each replica; bands remain across replicas, not shared-flow fits"
    write_json(output / "injection_scan.json", json_safe(audit))
    return audit


def sic_at_background(record, scores, efficiency, minimum):
    if len(np.unique(record["labels"][record["mask"]])) != 2:
        return None
    b, s, _ = efficiency_curve(record["labels"], scores, record["mask"], full_pipeline=True)
    use = (b >= 1e-4) & (np.rint(b * (record["labels"] == 0).sum()) >= minimum)
    if not use.any() or efficiency < b[use].min() or efficiency > b[use].max():
        return None
    unique, inverse = np.unique(b[use], return_inverse=True)
    maxima = np.zeros(len(unique))
    np.maximum.at(maxima, inverse, s[use])
    return float(np.interp(efficiency, unique, maxima) / np.sqrt(efficiency))


def comparison_rows(audit, scenario, seed, *, scope="signal_region"):
    """Long-form tables have no fixed method columns or privileged pair."""
    rows = []
    def add(method, metric, value, selection="all"):
        if value is None or isinstance(value, (int, float, np.number)):
            rows.append(dict(scenario=scenario, seed=seed, scope="common_physical_" + scope,
                             method=method, selection=selection, metric=metric, value=f.scalar(value)))
    for method, metrics in audit.get("methods", {}).items():
        for metric, value in metrics.items():
            add(method, metric, value)
        for truth, values in metrics["acceptance"].items():
            for field, value in values.items():
                add(method, truth + "_" + field, value)
    for point in audit.get("working_points", []):
        selection = "validation_background_" + str(point["background_budget"])
        for metric in ("background_efficiency", "signal_efficiency", "fit_count"):
            add(point["method"], metric, point[metric], selection)
        b, signal = point["background_efficiency"], point["signal_efficiency"]
        add(point["method"], "sic", signal / np.sqrt(b) if b and signal is not None else None, selection)
    return rows


def csv_write(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))

    def writer(temp):
        with temp.open("w", newline="") as stream:
            out = csv.DictWriter(stream, fieldnames=fields)
            out.writeheader()
            out.writerows(rows)

    atomic_write(path, writer)


def settings_rows(group, score_loader=None):
    rows = []

    def flatten(value, prefix=""):
        if isinstance(value, dict):
            for key, item in value.items():
                yield from flatten(item, (prefix + "." if prefix else "") + str(key))
        else:
            yield prefix, json.dumps(value) if isinstance(value, (list, tuple)) else value

    for method, (root, report) in group.items():
        protocol = (read_metadata(root, report, "protocol.json")
                    if method in BUILTINS or "protocol.json" in report.get("artifacts_sha256", {}) else {})
        if method == "lacathode":
            protocol["configuration"] = {
                k: yaml.safe_load(v) for k, v in protocol.get("configuration", {}).items()
            }
        payload = {
            "settings": report["contract"].get("settings", {}),
            "environment": report["contract"].get("environment", {}),
            "protocol": protocol,
            "preparation": report["contract"].get("inputs", {}).get("preparation", {}),
        }
        if score_loader is not None and str(root.resolve()) in score_loader.fit_audit:
            payload["plot_ensemble"] = score_loader.fit_audit[str(root.resolve())]
        if method == "riddle":
            result = read_metadata(root, report, "density/ensemble_selection.json")
            payload["ensemble"] = {
                k: result[k]
                for k in (
                    "selected_configuration",
                    "requested_runs",
                    "valid_runs",
                    "selected_checkpoints",
                    "failures",
                    "members",
                )
            }
        for key, value in flatten(payload):
            rows.append(
                {
                    "method": STYLES[KEYS[method]][0],
                    "scenario": report["scenario"],
                    "variant": report.get("variant", "default"),
                    "seed": report["seed"],
                    "setting": key,
                    "value": value,
                }
            )
    return rows


def training_figures(group, output, score_loader=None):
    for method, (root, report) in group.items():
        if method == "ranode":
            audit = score_loader.fit_audit.get(str(root.resolve())) if score_loader is not None else None
            attempts = [m["attempt"] for m in audit["used_members"]] if audit else None
            render_ranode_training((root, report), output, signal_attempts=attempts)
            continue
        destination = output / STYLES[KEYS[method]][0] / "02_training"
        if method == "riddle":
            members = read_metadata(root, report, "density/ensemble_selection.json")["members"]
            audit = score_loader.fit_audit.get(str(root.resolve())) if score_loader is not None else None
            if audit:
                members = audit["used_members"]
            history = read_metadata(root, report, "density/residual_losses.json")["history"]
            histories = [read_metadata(root, report, "density/" + m["directory"] + "/residual_losses.json")["history"]
                         for m in members]
            if audit and audit["status"] == "rebuilt_from_valid_fits":
                if not histories or any(len(h) != len(histories[0]) for h in histories):
                    raise ValueError("Inconsistent accepted RIDDLE training histories")
                history = [dict(epoch=e, **{key: float(np.mean([h[e][key] for h in histories]))
                    for key in ("train_nll", "validation_nll", "signal_fraction")}) for e in range(len(histories[0]))]
            for kind, ylabel in (
                ("nll", "Mixture negative log likelihood"),
                ("fraction", "Fitted mixture fraction"),
            ):
                fig, ax, _ = f.canvas(ylabel, "Epoch")
                x = [r["epoch"] + 1 for r in history]
                fields = (
                    (("train_nll", "Train"), ("validation_nll", "Validation"))
                    if kind == "nll"
                    else (("signal_fraction", "RIDDLE"),)
                )
                for field, label in fields:
                    ax.plot(x, [r[field] for r in history], label=label)
                f.legend(fig)
                f.save(fig, destination / kind)
            if members:
                positions = [m.get("fit_index", i) + 1 for i, m in enumerate(members)]
                fig, ax, _ = f.canvas("Fit", "Selected epoch")
                for i, member in zip(positions, members):
                    epochs = np.asarray(member["epochs"]) + 1
                    ax.scatter(
                        epochs,
                        np.full(len(epochs), i),
                        s=12,
                        color=STYLES["residual"][1],
                        label="RIDDLE" if i == positions[0] else None,
                    )
                ax.set(yticks=positions, ylim=(min(positions) - .7, max(positions) + .7))
                f.legend(fig)
                f.save(fig, destination / "selected_epochs_by_fit")
                fig, ax, _ = f.canvas("Validation mixture NLL", "Fit")
                for i, member, history in zip(positions, members, histories):
                    ax.plot(
                        [i],
                        [history[-1]["validation_nll"]],
                        "x",
                        color=".6",
                        label="Final epoch" if i == positions[0] else None,
                    )
                    ax.scatter(
                        np.full(len(member["epochs"]), i),
                        [history[e]["validation_nll"] for e in member["epochs"]],
                        s=12,
                        color=STYLES["residual"][1],
                        label="Selected" if i == positions[0] else None,
                    )
                ax.set(xticks=positions)
                f.legend(fig)
                f.save(fig, destination / "selected_vs_final_validation")
        for stage in ("background", "classifier") if method == "lacathode" else ("background",):
            directory = root / ("training" if method == "lacathode" else "background")
            names = (
                (f"{method}_model_train_losses.npy", f"{method}_model_val_losses.npy")
                if stage == "background"
                else ("loss_matris.npy", "val_loss_matris.npy")
            )
            if stage == "background" and not all((directory / name).is_file() for name in names):
                # Read legacy artifacts without renaming or modifying saved results.
                names = ("my_ANODE_model_train_losses.npy", "my_ANODE_model_val_losses.npy")
            if not all((directory / name).is_file() for name in names):
                continue
            fig, ax, _ = f.canvas(
                "Negative log likelihood" if stage == "background" else "Classification loss", "Epoch"
            )
            for name, label in zip(names, ("Train", "Validation")):
                verify_plot_input(root, report, str((directory / name).relative_to(root)))
                values = np.load(directory / name)
                histories = np.atleast_2d(values)
                epochs = np.arange(histories.shape[1])
                line, = ax.plot(epochs, histories.mean(axis=0), label=label)
                if (len(histories) > 1 and method == "lacathode"
                        and report.get("contract", {}).get("lacathode_run_layout") == "independent_background_classifier_v1"):
                    low, high = np.quantile(histories, [.16, .84], axis=0)
                    ax.fill_between(epochs, low, high, color=line.get_color(), alpha=.18)
            f.legend(fig)
            f.save(fig, destination / stage)


def verify_plot_input(root, report, name):
    path = (root / name).resolve()
    checksum = report.get("artifacts_sha256", {}).get(name)
    if not path.is_relative_to(root.resolve()) or not checksum or file_digest(path) != checksum:
        raise ValueError("Plot input is missing or changed")
    return path


def read_metadata(root, report, name):
    return json.loads(verify_plot_input(root, report, name).read_text())


def method_band(ax, x, values, key, scenario, metric, seeds, classifier_fits=None):
    name, color, ls = STYLES[key]
    summary = f.draw_band(ax, x, values, name, color, ls)
    summary.update(uncertainty_source="training_seed_variation_on_fixed_data_partition", seeds=seeds)
    summary.update(percentiles=[16, 50, 84], aggregation="median of individual metric curves")
    if classifier_fits:
        summary.update(
            uncertainty_source=fit_uncertainty(classifier_fits),
            classifier_fits=classifier_fits,
            fit_count=len(classifier_fits),
            independent_runs=len({fit.get("run_seed", fit["seed"]) for fit in classifier_fits}),
        )
    if not summary["band_drawn"]:
        colored_status(
            f"{name} | {f.SCENARIO_LABELS[scenario]} | {metric}: "
            "uncertainty band unavailable; requires multiple independent runs with common statistical support.",
            kind="WARNING",
        )
    return summary


def summary_figures(bundles, output, args, *, view="comparison"):
    audit = {}
    for scenario in f.SCENARIOS:
        cohort = [b for b in bundles if b["report"]["scenario"] == scenario]
        if not cohort:
            continue
        partitions = {
            json.dumps(report.get("contract", {}).get("inputs", {}).get("files"), sort_keys=True)
            for bundle in cohort for _, report in bundle["sources"].values()
        }
        if len(partitions) > 1:
            raise ValueError("Ordinary seed bands require a fixed prepared partition; use the injection-scan workflow for varied partitions")
        fig, ax, _ = f.canvas("Significance improvement", "Background efficiency")
        any_curve = False
        for key in STYLES:
            values = []
            native_curves = []
            curve_groups = []
            seeds = []
            classifier_fits = []
            for bundle in cohort:
                labels, scores = bundle["curves"]["signal_region"]
                if key not in scores or len(np.unique(labels)) != 2:
                    continue
                record = bundle["evaluation"]["signal_region"][key]
                method = "lacathode" if key == "raw" else "riddle"
                for fit, score_group in enumerate(run_score_groups(method, record)):
                    per_fit = []
                    for score in score_group:
                        b, s, _ = roc_curve(labels, score[record["mask"]])
                        supported = (b >= 1e-4) & (np.rint(b * (labels == 0).sum()) >= args.min_background)
                        native_curves.append((b[supported], s[supported]))
                        curve_groups.append(len(classifier_fits))
                        per_fit.append(sic_on_grid(b[supported], s[supported]))
                    values.append(np.median(per_fit, axis=0))
                    seeds.append(bundle["report"]["seed"])
                    classifier_fits.append(fit_identity(bundle, key, fit))
            if values:
                require_independent_run_ids(classifier_fits)
                name = STYLES[key][0]
                if key == "raw" and len(native_curves) > 1:
                    summary = f.draw_fit_curves(ax, native_curves, "sic", *STYLES[key],
                                                uncertainty_source=fit_uncertainty(classifier_fits), run_groups=curve_groups)
                    summary.update(classifier_fits=classifier_fits, seeds=seeds)
                else:
                    summary = method_band(ax, f.GRID, values, key, scenario, "SIC", seeds, classifier_fits)
                audit[scenario + "/" + name + "/sic"] = summary
                any_curve = True
        if any_curve:
            ax.plot(f.GRID, np.sqrt(f.GRID), color=".5", ls=":", label="Random")
            f.summary_axes(ax, "sic")
            f.legend(fig, title=f.SCENARIO_LABELS[scenario])
            f.save(fig, output / scenario / "summary" / view / "full_mass" / "signal_region_sic")
        else:
            f.plt.close(fig)
        if scenario != "background_only":
            continue
        fig, ax, _ = f.canvas(r"$\chi^2/n_{\mathrm{dof}}$", "SR selection efficiency")
        found = False
        for key in STYLES:
            curves = []
            seeds = []
            classifier_fits = []
            for bundle in cohort:
                sample = bundle["samples"]["test"]
                if key + "_scores" not in sample:
                    continue
                pop = sample["labels"] == 0
                mass, score = sample["mass"][pop], sample[key + "_scores"][pop]
                mask = sample["mask"][pop]
                region = f.sample_sr(sample, key)[pop] & mask
                if len(mass) < 300 or not region.any():
                    continue
                edges = f.equal_occupancy(mass)
                full = np.histogram(mass, edges)[0]
                if np.any(full == 0):
                    continue
                record = bundle["evaluation"]["test"][key]
                method = "lacathode" if key == "raw" else "riddle"
                for fit, score_group in enumerate(run_score_groups(method, record)):
                    per_fit = []
                    for scores in score_group:
                        score = scores[pop]
                        per_fit.append([
                            f.shape_chi2(full, np.histogram(mass[mask & (score > np.quantile(score[region], 1 - e))], edges)[0], e)
                            for e in f.EFFICIENCIES
                        ])
                    curves.append(np.median(np.asarray(per_fit, dtype=float), axis=0))
                    seeds.append(bundle["report"]["seed"])
                    classifier_fits.append(fit_identity(bundle, key, fit))
            if curves:
                require_independent_run_ids(classifier_fits)
                name = STYLES[key][0]
                audit[scenario + "/" + name + "/mass_flatness"] = method_band(
                    ax, f.EFFICIENCIES, curves, key, scenario, "mass flatness", seeds, classifier_fits
                )
                found = True
        if found:
            sample = cohort[0]["samples"]["test"]
            mass = sample["mass"][sample["labels"] == 0]
            random_summary = f.draw_band(
                ax, f.EFFICIENCIES, random_reference(mass, mode=args.random_reference),
                "Random", ".5", ":"
            )
            random_summary["uncertainty_source"] = "random_" + args.random_reference + "_trials"
            random_summary["sampling_with_replacement"] = args.random_reference == "bootstrap"
            audit[scenario + "/random/mass_flatness"] = random_summary
            f.summary_axes(ax, "mass_flatness")
            f.legend(fig, title="BG-Only")
            f.save(fig, output / scenario / "summary" / view / "full_mass" / "mass_flatness_vs_selection")
        else:
            f.plt.close(fig)
    return audit


@f.cached_plot_calculation
def random_reference(mass, trials=100, *, mode="bootstrap"):
    if mode not in ("bootstrap", "subset"):
        raise ValueError("Random reference must be bootstrap or subset")
    edges = f.equal_occupancy(mass)
    full = np.histogram(mass, edges)[0]
    bin_id = np.minimum(np.searchsorted(edges, mass, side="right") - 1, len(full) - 1)
    result = []
    for efficiency in f.EFFICIENCIES:
        rng = np.random.RandomState(42)
        values = []
        for _ in range(trials):
            indices = rng.choice(len(mass), size=int(efficiency * len(mass)), replace=mode == "bootstrap")
            values.append(f.shape_chi2(full, np.bincount(bin_id[indices], minlength=len(full)), efficiency))
        result.append(values)
    return np.asarray(result).T


def require_same_physical_population(records):
    base = next(iter(records.values()))
    if any(
        any(not np.array_equal(base[k], r[k]) for k in ("mass", "labels", "physical"))
        for r in records.values()
    ):
        raise ValueError(
            "Comparison requires identical physical evaluation events in the same order"
        )


def physical_score_coordinate(method, values):
    return expit(values) if METHOD_SPECS[method].score_transform == "sigmoid" else values


def score_variation_source(method, record):
    return (f.fit_uncertainty(record) if record.get("independent_runs", False)
            else "independent_complete_method_runs")


def common_signal_region(records):
    """Use saved membership, and reject disagreement instead of silently slicing it away."""
    require_same_physical_population(records)
    saved = [r["is_signal_region"] for r in records.values() if "is_signal_region" in r]
    mass = next(iter(records.values()))["mass"]
    region = saved[0] if saved else (mass > 3.3) & (mass < 3.7)
    if any(not np.array_equal(region, other) for other in saved[1:]):
        raise ValueError("Methods disagree on saved signal-region membership")
    return region


def scope_label(scope):
    return {"signal_region": "Signal region", "full_region": "Full region"}[scope]


def scope_group(group, scope):
    return {method: source for method, source in group.items()
            if scope == "signal_region" or METHOD_SPECS[method].score_scope == "full_region"}


def scope_region(records, scope):
    if scope == "signal_region":
        return common_signal_region(records)
    return np.ones(len(next(iter(records.values()))["mass"]), dtype=bool)


def render_physical_curves(records, output, scenario, minimum, *, scope="signal_region"):
    require_same_physical_population(records)
    audit = {m: f.aggregate_fit_metrics([
        oracle_metrics(r["labels"], s, r["mask"], min_background=minimum) for s in fit_scores(r)
    ]) for m, r in records.items()}
    for metric in ("roc", "sic", "sic_signal", "background_rejection"):
        ylabel = {
            "roc": "Signal efficiency",
            "sic": "Significance improvement",
            "sic_signal": "Significance improvement",
            "background_rejection": "Background rejection",
        }[metric]
        xlabel = (
            "Signal efficiency"
            if metric in ("background_rejection", "sic_signal")
            else "Background efficiency"
        )
        fig, ax, _ = f.canvas(ylabel, xlabel)
        drawn = False
        for method, record in records.items():
            if len(np.unique(record["labels"][record["mask"]])) != 2:
                continue
            score_curves = fit_scores(record)
            if len(score_curves) > 1:
                curves = []
                for score in score_curves:
                    b, s, _ = efficiency_curve(record["labels"], score, record["mask"], full_pipeline=True)
                    use = b > 0 if metric == "roc" else (b >= 1e-4) & (np.rint(b * (record["labels"] == 0).sum()) >= minimum)
                    curves.append((b[use], s[use]))
                audit[method][metric + "_fit_band"] = f.draw_fit_curves(
                    ax, curves, metric, *PHYSICAL_STYLES[method][:3],
                    uncertainty_source=score_variation_source(method, record),
                    band=record.get("independent_runs", False),
                    run_groups=None if record.get("independent_runs", False) else [0] * len(curves))
                drawn = True
                continue
            b, s, _ = efficiency_curve(
                record["labels"], record["scores"], record["mask"], full_pipeline=True
            )
            supported = (b >= 1e-4) & (
                np.rint(b * (record["labels"] == 0).sum()) >= minimum
            )
            use = (b > 0) if metric == "roc" else supported
            if not use.any():
                continue
            label, color, ls, _ = PHYSICAL_STYLES[method]
            x, y = (
                (s, 1 / np.maximum(b, 1e-300))
                if metric == "background_rejection"
                else (s if metric == "sic_signal" else b,
                      s if metric == "roc" else s / np.sqrt(np.maximum(b, 1e-300)))
            )
            ax.plot(x[use], y[use], label=label, color=color, ls=ls)
            audit.setdefault(method, {}).update(oracle_metrics(
                record["labels"],
                record["scores"],
                record["mask"],
                min_background=minimum,
            ))
            drawn = True
        if not drawn:
            f.plt.close(fig)
            continue
        grid = np.geomspace(1e-4, 1, 300)
        if metric == "background_rejection":
            ax.plot(grid, 1 / grid, color=".5", ls=":", label="Random")
            ax.set(yscale="log", xlim=(0, 1), ylim=(1, 1e4))
        elif metric == "sic_signal":
            ax.plot(grid, np.sqrt(grid), color=".5", ls=":", label="Random")
            ax.set(xlim=(0, 1), ylim=(0, None))
        else:
            ax.plot(
                grid,
                grid if metric == "roc" else np.sqrt(grid),
                color=".5",
                ls=":",
                label="Random",
            )
            ax.set(
                xscale="log",
                xlim=(1e-4, 1),
                ylim=(0, 1.02 if metric == "roc" else None),
            )
        f.legend(fig, title=f.SCENARIO_LABELS[scenario] + " | " + scope_label(scope), ncols=1)
        f.save(fig, output / (scope + "_" + metric))
    return audit


def render_physical_scores(records, output, scenario, *, scope="signal_region"):
    displayed = {m: physical_score_coordinate(m, f.fit_scores(r)) for m, r in records.items()}
    finite = [v[:, records[m]["mask"]] for m, v in displayed.items() if records[m]["mask"].any()]
    low = min([0.] + [float(v.min()) for v in finite])
    high = max([1.] + [float(v.max()) for v in finite])
    edges = np.linspace(low, high, 51)
    for density in (False, True):
        fig, ax, _ = f.canvas("Density" if density else "Events / bin", "Score")
        fig.set_figwidth(max(9, 2.5 * len(records)))
        columns = []
        for method, record in records.items():
            label, bg, _, sig = PHYSICAL_STYLES[method]
            entries = []
            for truth, color, ls in ((0, bg, "-"), (1, sig, "--")):
                selected = record["mask"] & (record["labels"] == truth)
                values = displayed[method]
                hist = f.fit_histogram(values, edges, selected).astype(float)
                if density and hist.sum():
                    hist /= hist.sum() * np.diff(edges)
                line = ax.stairs(hist, edges, color=color, ls=ls, baseline=None)
                entries.append((line, "Background" if truth == 0 else "Signal"))
            columns.append((label, entries))
        ax.set_xlim(low, high)
        if not density:
            ax.set_yscale("symlog", linthresh=1)
        f.population_legend(
            fig, columns, title=f.SCENARIO_LABELS[scenario] + " | " + scope_label(scope)
        )
        f.save(
            fig,
            output
            / (
                scope + "_score_density"
                if density
                else scope + "_score_counts"
            ),
        )


def physical_sr_record(record, sr=None):
    sr = record.get("is_signal_region") if sr is None else sr
    if sr is None:
        sr = (record["mass"] > 3.3) & (record["mass"] < 3.7)
    result = {k: record[k][sr] for k in ("mass", "labels", "physical", "scores", "mask")}
    if "fit_scores" in record:
        result["fit_scores"] = record["fit_scores"][:, sr]
    for key in ("independent_runs", "run_seeds", "score_kind", "fit_score_kind", "plot_saved_ensemble"):
        if key in record:
            result[key] = record[key]
    return result


def background_cut(record, budget):
    """Strict threshold with an uncut BG denominator, including mapping failures."""
    return background_cuts(record, [budget])[0]


def background_cuts(record, budgets):
    """Sort once for all requested working points; retain strict tie handling."""
    n = int((record["labels"] == 0).sum())
    scores = np.sort(fit_scores(record)[:, (record["labels"] == 0) & record["mask"]], axis=1)
    cuts = []
    for budget in budgets:
        allowed = int(np.floor(n * budget))
        cuts.append(None if not scores.shape[1] or allowed < 1 else
                    scores[:, -allowed - 1] if allowed < scores.shape[1] else
                    np.nextafter(scores[:, 0], -np.inf))
    return cuts


def render_physical_working_points(validation, test, output, *, scenario=None, scope="signal_region"):
    require_same_physical_population(test)
    # Calibration roles differ across methods; evaluation populations may not.
    # Select each validation region independently, without aligning unrelated
    # held-out rows or requiring them to match another method's training roles.
    validation = {m: physical_sr_record(r, scope_region({m: r}, scope)) for m, r in validation.items()}
    test = {m: physical_sr_record(r, scope_region(test, scope)) for m, r in test.items()}
    require_same_physical_population(test)
    base = next(iter(test.values()))
    edges = (np.linspace(3.3, 3.7, 21) if scope == "signal_region" else
             np.linspace(min(1., base["mass"].min()), max(9., base["mass"].max()), 81))
    inclusive = np.histogram(base["mass"][base["labels"] == 0], edges)[0]
    audit = []
    budgets = (0.10, 0.05, 0.01, 0.004)
    cuts = {method: ([np.array([np.log((1-b)/b)]) for b in budgets]
                     if str(record.get("score_kind", "")) == "background_percentile_logit"
                     else background_cuts(record, budgets)) for method, record in validation.items()}
    for index, budget in enumerate(budgets):
        selected, retention = {}, {}
        for method, val in validation.items():
            f.validate_fit_counts([val, test[method]])
            cut = cuts[method][index]
            if cut is None:
                continue
            record = test[method]
            keep = record["mask"] & (f.fit_scores(record) > cut[:, None])
            selected[method] = keep
            retention[method] = []
            for truth in (0, 1):
                population = record["labels"] == truth
                total, passed = int(population.sum()), float((population & keep).sum(axis=1).mean())
                retention[method].append(passed / total if total else None)
            audit.append(
                dict(
                    method=method,
                    background_budget=budget,
                    cut=float(cut[0]) if len(cut) == 1 else cut.tolist(),
                    fit_count=len(cut),
                    threshold_source=("analytic conditional-percentile threshold"
                        if str(val.get("score_kind", "")) == "background_percentile_logit"
                        else "MC validation-background quantile (diagnostic)"),
                    background_efficiency=retention[method][0],
                    signal_efficiency=retention[method][1],
                )
            )
        if not selected:
            continue
        for shape in (False, True):
            fig, ax, lower = f.canvas(
                "Normalized background / bin" if shape else "Events / bin",
                r"$m_{jj}$ [TeV]",
                ratio="Selected / inclusive" if shape else "BG retention",
            )
            fig.set_figwidth(max(10, 2.5 * (len(selected) + 1)))
            columns = []
            norm = (
                inclusive / inclusive.sum() if shape and inclusive.sum() else inclusive
            )
            handle = ax.stairs(norm, edges, color=".65", baseline=None)
            columns.append(("No cut", [(handle, "Background")]))
            if not shape:
                signal_hist = np.histogram(base["mass"][base["labels"] == 1], edges)[0]
                handle = ax.stairs(
                    signal_hist, edges, color=".45", ls="--", baseline=None
                )
                columns[0][1].append((
                    handle, "Signal" if scenario != "background_only" or signal_hist.sum() else None
                ))
            for method, keep in selected.items():
                label, color, _, signal_color = PHYSICAL_STYLES[method]
                record = test[method]
                bg_hist = f.fit_histogram(record["mass"], edges, keep & (record["labels"] == 0))
                hist = bg_hist / bg_hist.sum() if shape and bg_hist.sum() else bg_hist
                b, s = retention[method]
                signal_text = f"{100 * s:.3g}%" if s is not None else "n/a"
                handle = ax.stairs(hist, edges, color=color, baseline=None)
                entries = [
                    (
                        handle,
                        f"B: {100 * b:.3g}% | S: {signal_text}"
                        if b is not None
                        else f"S: {signal_text}",
                    )
                ]
                if scenario == "background_only":
                    entries = [(handle, f"B: {100 * b:.3g}%" if b is not None else "Background")]
                if not shape:
                    shist = f.fit_histogram(record["mass"], edges, keep & (record["labels"] == 1))
                    handle = ax.stairs(
                        shist, edges, color=signal_color, ls="--", baseline=None
                    )
                    signal_label = (
                        f.retention_label(1, s, int((record["labels"] == 1).sum()), scenario=scenario)
                        if scenario == "background_only" else f"S: {signal_text}"
                    )
                    entries.append((handle, signal_label))
                columns.append((label, entries))
                ratio = np.divide(
                    hist, norm, out=np.full(len(hist), np.nan), where=norm > 0
                )
                lower.stairs(ratio, edges, color=color, baseline=None)
            if shape:
                lower.axhline(1, color=".5", ls=":")
            else:
                ax.set_yscale("symlog", linthresh=1)
            ax.set_xlim(edges[0], edges[-1])
            f.population_legend(
                fig, columns, title=f"{scope_label(scope)} | B ≤ {100 * budget:g}%"
            )
            f.save(
                fig,
                output
                / "04_mass_cuts"
                / (
                    ("background_mass_shape_" if shape else "mass_counts_")
                    + f"{100 * budget:g}".replace(".", "p")
                    + "pct"
                ),
            )
        if selected:
            render_physical_efficiency(test, selected, edges, budget, output, scope=scope)
            for density in (False, True):
                render_physical_features(test, selected, retention, budget, output, scenario=scenario, density=density, scope=scope)
    return audit


def render_physical_efficiency(records, selected, edges, budget, output, *, scope="signal_region"):
    fig, ax, _ = f.canvas("Background efficiency", r"$m_{jj}$ [TeV]")
    for method, keep in selected.items():
        record = records[method]
        bg = record["labels"] == 0
        full = np.histogram(record["mass"][bg], edges)[0]
        passed = f.fit_histogram(record["mass"], edges, keep & bg)
        values = np.divide(passed, full, out=np.full(len(full), np.nan), where=full > 0)
        label, color, ls, _ = PHYSICAL_STYLES[method]
        ax.stairs(values, edges, label=label, color=color, ls=ls, baseline=None)
    ax.axhline(budget, color=".5", ls=":", label="Validation target")
    ax.set(xlim=(edges[0], edges[-1]), ylim=(0, None))
    f.legend(fig, title=f"{scope_label(scope)} | B ≤ {100 * budget:g}%")
    tag = f"{100 * budget:g}".replace(".", "p") + "pct"
    f.save(fig, output / "03_mass_sculpting" / ("background_efficiency_" + tag))


def render_physical_features(records, selected, retention, budget, output, *, scenario=None, density=False, scope="signal_region"):
    fields = (
        ("m1", r"$m_{J1}$ [TeV]"),
        ("delta_m", r"$\Delta m_J$ [TeV]"),
        ("tau21_j1", r"$\tau_{21,J1}$"),
        ("tau21_j2", r"$\tau_{21,J2}$"),
    )
    base = next(iter(records.values()))
    if base["physical"].shape[1] == 5:
        fields += (("deltaR", r"$\Delta R_{jj}$"),)
    for index, (name, xlabel) in enumerate(fields):
        low, high = base["physical"][:, index].min(), base["physical"][:, index].max()
        if low == high:
            high = low + 1
        edges = np.linspace(low, high, 31)
        fig, ax, _ = f.canvas("Density" if density else "Events / bin", xlabel)
        fig.set_figwidth(max(10, 2.5 * len(selected)))
        columns = []
        for method, keep in selected.items():
            record = records[method]
            label, bg, _, sig = PHYSICAL_STYLES[method]
            entries = []
            for truth, color, ls in ((0, bg, "-"), (1, sig, "--")):
                hist = f.fit_histogram(record["physical"][:, index], edges, keep & (record["labels"] == truth))
                if density and hist.sum():
                    hist = hist / (hist.sum() * np.diff(edges))
                handle = ax.stairs(hist, edges, color=color, ls=ls, baseline=None)
                efficiency = retention[method][truth]
                percent = (
                    f"{100 * efficiency:.3g}%" if efficiency is not None else "n/a"
                )
                text = ("B: " if truth == 0 else "S: ") + percent
                if truth == 1 and scenario == "background_only":
                    text = f.retention_label(
                        truth, efficiency, int((record["labels"] == truth).sum()), scenario=scenario
                    )
                entries.append((handle, text))
            columns.append((label, entries))
        if not density:
            ax.set_yscale("symlog", linthresh=1)
        f.population_legend(
            fig, columns, title=f"{scope_label(scope)} | B ≤ {100 * budget:g}%"
        )
        tag = f"{100 * budget:g}".replace(".", "p") + "pct"
        f.save(fig, output / "05_features" / f"{name}_{'density' if density else 'counts'}_background_{tag}")


def render_ranode_training(source, output, *, signal_attempts=None):
    root, report = source
    protocol = read_metadata(root, report, "protocol.json")
    for stage in ("background", "signal"):
        attempts = (
            protocol["signal_attempts"] if stage == "signal" and "signal_attempts" in protocol
            else [protocol[stage + "_attempt"]]
        )
        if stage == "signal" and signal_attempts is not None:
            attempts = signal_attempts
        suffix = "_list" if stage == "background" else ""
        fig, ax, _ = f.canvas("Negative log likelihood", "Epoch")
        for kind, color, ls in (
            ("trainloss", "#8B1A1A", "-"),
            ("valloss", "#56B4E9", "--"),
        ):
            values = []
            for attempt in attempts:
                path = attempt + f"/results/upstream/{stage}/fit/" + kind + suffix + ".npy"
                values.append(np.load(
                    verify_plot_input(root, report, path), allow_pickle=False,
                ))
            values = np.asarray(values)
            epochs = np.arange(1, values.shape[1] + 1)
            ax.plot(
                epochs,
                values.mean(axis=0),
                color=color,
                ls=ls,
                label="Train" if kind == "trainloss" else "Validation",
            )
        f.legend(
            fig,
            title="R-ANODE | "
            + ("Background flow" if stage == "background" else "Signal-mixture flow"),
        )
        f.save(fig, output / "R-ANODE" / "02_training" / (stage + "_nll"))


def render_physical_comparison(group, output, args, load_scores, *, scope="signal_region"):
    excluded = {m: "Saved method scores cover the signal region only" for m in group if m not in scope_group(group, scope)}
    group = scope_group(group, scope)
    if not group:
        return {"scope": scope, "status": "no_methods_with_scores_in_scope", "excluded_methods": excluded}
    label = scope_label(scope)
    partition = "signal_region" if scope == "signal_region" else "test"
    reference_inputs = next(iter(group.values()))[1]["contract"]["inputs"]
    if any(
        report["contract"]["inputs"].get("files") != reference_inputs.get("files")
        for _, report in group.values()
    ):
        raise ValueError("Comparisons require the same prepared input files")
    records = {
        m: load_scores(root, report, partition)
        for m, (root, report) in group.items()
    }
    scenario = next(iter(group.values()))[1]["scenario"]
    with ProgressStage("physical_curves", f"Plot {label.lower()} ROC, SIC and rejection"):
        audit = {
            "scope": scope,
            "population": "Common physical events, independent acceptance and uncut denominators",
            "excluded_methods": excluded,
            "score_transforms": {m: METHOD_SPECS[m].score_transform for m in group},
            "score_ensembles": {m: r["plot_ensemble"] for m, r in records.items() if "plot_ensemble" in r},
            "methods": render_physical_curves(records, output / "01_performance", scenario, args.min_background, scope=scope),
        }
    with ProgressStage("physical_scores", f"Plot {label.lower()} scores"):
        render_physical_scores(records, output / "01_performance", scenario, scope=scope)
    validation = {
        m: load_scores(root, report, "validation")
        for m, (root, report) in group.items()
    }
    test = {m: load_scores(root, report, "test") for m, (root, report) in group.items()}
    for method in group:
        f.validate_fit_counts([records[method], validation[method], test[method]])
    if "lacathode" in group:
        audit["lacathode_fit_count"] = f.validate_fit_counts(
            [records["lacathode"], validation["lacathode"], test["lacathode"]]
        )
        audit["lacathode_histograms"] = "Mean per-fit counts; separate validation cut per fit; no score averaging"
    with ProgressStage("physical_cuts", f"Plot {label.lower()} mass cuts and features"):
        audit["working_points"] = render_physical_working_points(validation, test, output, scenario=scenario, scope=scope)
    with ProgressStage("mass_flatness", f"Plot {label.lower()} background chi-squared"):
        seed = next(iter(group.values()))[1]["seed"]
        audit["mass_flatness"] = physical_mass_summary(
            {(scenario, seed): group}, output / "03_mass_sculpting", args, load_scores, scenario=scenario, scope=scope)
    write_json(output / "metrics.json", json_safe(audit))
    return audit


def physical_comparison_summary(groups, output, args, load_scores, *, scope="signal_region", view="comparison"):
    groups = {identity: scope_group(group, scope) for identity, group in groups.items() if scope_group(group, scope)}
    partition = "signal_region" if scope == "signal_region" else "test"
    audits = {}
    for scenario in f.SCENARIOS:
        destination = output / scenario / "summary" / view
        if scope == "full_region":
            destination /= "full_region"
        partitions = {
            json.dumps(report["contract"]["inputs"].get("files"), sort_keys=True)
            for (name, _), group in groups.items()
            if name == scenario
            for _, report in group.values()
        }
        if len(partitions) > 1:
            raise ValueError(
                "Seed bands require a fixed common prepared partition; use scan plots for varied inputs"
            )
        curves = {metric: {m: [] for m in PHYSICAL_STYLES}
                  for metric in ("roc", "sic", "sic_signal", "background_rejection")}
        identities = {metric: {m: [] for m in PHYSICAL_STYLES} for metric in curves}
        native_curves = {metric: {m: [] for m in PHYSICAL_STYLES} for metric in curves}
        curve_groups = {metric: {m: [] for m in PHYSICAL_STYLES} for metric in curves}
        reference_records = None
        grid = np.geomspace(1e-4, 1, 300)
        for (name, seed), group in groups.items():
            if name != scenario:
                continue
            records = {
                m: load_scores(root, report, partition)
                for m, (root, report) in group.items()
            }
            require_same_physical_population(records)
            if reference_records is not None:
                require_same_physical_population({"reference": reference_records, **records})
            reference_records = next(iter(records.values()))
            for method, record in records.items():
                if len(np.unique(record["labels"][record["mask"]])) != 2:
                    continue
                for metric in curves:
                    for run, score_group in enumerate(run_score_groups(method, record)):
                        fixed_background_curves = []
                        for score in score_group:
                            b, s, _ = efficiency_curve(record["labels"], score, record["mask"], full_pipeline=True)
                            use = b > 0 if metric == "roc" else (b >= 1e-4) & (
                                np.rint(b * (record["labels"] == 0).sum()) >= args.min_background)
                            native_curves[metric][method].append((b[use], s[use]))
                            curve_groups[metric][method].append(len(identities[metric][method]))
                            unique, inverse = np.unique(b, return_inverse=True)
                            maxima = np.zeros(len(unique))
                            np.maximum.at(maxima, inverse, s)
                            values = np.interp(grid, unique, maxima, left=np.nan, right=np.nan)
                            supported = grid >= (0 if metric == "roc" else args.min_background / (record["labels"] == 0).sum())
                            values[~supported] = np.nan
                            fixed_background_curves.append(values)
                        curves[metric][method].append(np.median(fixed_background_curves, axis=0))
                        identities[metric][method].append(dict(
                            seed=seed, fit=run, source=score_variation_source(method, record),
                            member_id=run,
                            run_seed=int(record["run_seeds"][run]) if "run_seeds" in record else seed,
                        ))
        for metric in curves:
            fig, ax, _ = f.canvas(
                "Signal efficiency" if metric == "roc" else
                "Background rejection" if metric == "background_rejection" else "Significance improvement",
                "Signal efficiency" if metric in ("sic_signal", "background_rejection") else "Background efficiency",
            )
            audit = {}
            for method, cohort in curves[metric].items():
                if not cohort:
                    continue
                require_independent_run_ids(identities[metric][method])
                label, color, ls, _ = PHYSICAL_STYLES[method]
                values = np.asarray(cohort) / (1 if metric == "roc" else np.sqrt(grid))
                audit[method] = (
                    f.draw_fit_curves(ax, native_curves[metric][method], metric, label, color, ls,
                                      run_groups=curve_groups[metric][method])
                    if (method == "lacathode" and len(native_curves[metric][method]) > 1) or metric in ("sic_signal", "background_rejection")
                    else f.draw_band(ax, grid, values, label, color, ls)
                )
                audit[method].update(uncertainty_source=fit_uncertainty(identities[metric][method]),
                                     percentiles=[16, 50, 84], fit_count=len(cohort), run_count=len(cohort),
                                     members=identities[metric][method])
                audit[method].pop("independent_runs", None)
            if not audit:
                f.plt.close(fig)
                continue
            ax.plot(
                grid,
                grid if metric == "roc" else 1 / grid if metric == "background_rejection" else np.sqrt(grid),
                color=".5",
                ls=":",
                label="Random",
            )
            ax.set(
                xscale="linear" if metric in ("sic_signal", "background_rejection") else "log",
                xlim=(0 if metric in ("sic_signal", "background_rejection") else 1e-4, 1),
                yscale="log" if metric == "background_rejection" else "linear",
                ylim=(1 if metric == "background_rejection" else 0, 1.02 if metric == "roc" else None),
            )
            f.legend(
                fig, title=f.SCENARIO_LABELS[scenario] + " | " + scope_label(scope), ncols=1
            )
            f.save(fig, destination / "01_performance" / (scope + "_" + metric))
            audits[scenario + "/" + metric] = audit
        if any(name == scenario for name, _ in groups):
            audits[scenario + "/mass_flatness"] = physical_mass_summary(
                groups, destination / "03_mass_sculpting", args, load_scores, scenario=scenario, scope=scope)
    return audits


@f.cached_plot_calculation
def physical_mass_values(mass, mask, score, edges, full, efficiencies):
    """One sort and one cumulative histogram for all mass-flatness thresholds."""
    cuts = background_cuts(dict(labels=np.zeros(len(mass)), mask=mask, scores=score), efficiencies)
    thresholds = np.asarray([cut[0] if cut is not None else np.inf for cut in cuts])
    counts = f.strict_cut_histograms(mass, edges, score, mask, thresholds)
    achieved = [float(count.sum() / len(mass)) for count in counts]
    values = [f.shape_chi2(full, count, actual) if actual > 0 else np.nan
              for count, actual in zip(counts, achieved)]
    return values, achieved


def physical_mass_summary(groups, output, args, score_loader, *, scenario="background_only", scope="signal_region"):
    """Background sculpting across independent full runs on a common population."""
    groups = {identity: scope_group(group, scope) for identity, group in groups.items() if scope_group(group, scope)}
    curves = {m: [] for m in PHYSICAL_STYLES}
    identities = {m: [] for m in PHYSICAL_STYLES}
    efficiencies = {m: [] for m in PHYSICAL_STYLES}
    reference_mass = None
    for (name, seed), group in groups.items():
        if name != scenario:
            continue
        records = {m: score_loader(root, report, "test") for m, (root, report) in group.items()}
        require_same_physical_population(records)
        region = scope_region(records, scope)
        base = next(iter(records.values()))
        pop = region & (base["labels"] == 0)
        mass = base["mass"][pop]
        if len(mass) < 300:
            colored_status(f"{scope_label(scope)} mass flatness needs at least 300 background events", kind="WARNING")
            continue
        edges = f.equal_occupancy(mass)
        full = np.histogram(mass, edges)[0]
        if reference_mass is not None and not np.array_equal(reference_mass, mass):
            raise ValueError("Mass-flatness seed bands require identical background events")
        reference_mass = mass
        for method, record in records.items():
            mask = record["mask"][pop]
            if not mask.any():
                continue
            for i, score_group in enumerate(run_score_groups(method, record)):
                per_fit = [physical_mass_values(mass, mask, scores[pop], edges, full, f.EFFICIENCIES)
                           for scores in score_group]
                # Reduce shared-background classifier fits within each complete
                # run; the outer bands then contain only independent runs.
                curves[method].append(np.median([v for v, _ in per_fit], axis=0).tolist())
                efficiencies[method].append(np.median([e for _, e in per_fit], axis=0).tolist())
                source = score_variation_source(method, record)
                identities[method].append(dict(
                    seed=seed, fit=i, source=source,
                    member_id=i,
                    run_seed=int(record["run_seeds"][i]) if "run_seeds" in record else seed,
                ))
    if not any(curves.values()):
        return {"scope": scope, "status": "insufficient_background_events"}
    fig, ax, _ = f.canvas(r"$\chi^2/n_{\mathrm{dof}}$", "Target " + ("SR " if scope == "signal_region" else "full-region ") + "background efficiency")
    audit = {"scope": scope, "bins": 300, "methods": {},
             "selection": "Truth-assisted test thresholds in " + scope + "; strict cut, common uncut denominator",
             "normalization": "Achieved background efficiency, including acceptance and ties"}
    for method, values in curves.items():
        if not values:
            continue
        require_independent_run_ids(identities[method])
        summary = f.draw_band(ax, f.EFFICIENCIES, values, *PHYSICAL_STYLES[method][:3])
        summary.update(uncertainty_source=fit_uncertainty(identities[method]), percentiles=[16, 50, 84],
                       fit_count=len(values), run_count=len(values), members=identities[method],
                       target_efficiencies=f.EFFICIENCIES.tolist(), achieved_efficiencies=efficiencies[method])
        summary.pop("independent_runs", None)
        audit["methods"][method] = summary
    audit["random"] = f.draw_band(ax, f.EFFICIENCIES, random_reference(reference_mass, mode=args.random_reference),
                                  "Random", ".5", ":")
    ax.set(xlim=(.20, .01), yscale="log")
    f.legend(fig, title=f.SCENARIO_LABELS[scenario] + " | Background in " + scope_label(scope).lower())
    f.save(fig, output / "mass_flatness_vs_selection")
    return audit


def refresh_plot_scopes(output, groups, scan_groups):
    """Replace generated figures in requested scopes, preserving notes and other runs."""
    manifest = output / "plot_manifest.json"
    if not manifest.is_file() or json.loads(manifest.read_text()).get("schema") not in (1, 2, 3):
        return
    scopes = set()
    for (scenario, seed), group in groups.items():
        target = output / scenario / f"seed_{seed:03d}"
        scopes.update((target / "comparison", target / "comparison_with_ranode",
                       output / scenario / "summary" / "comparison",
                       output / "comparison" / scenario, output / "comparison_with_ranode" / scenario))
        for method in group:
            label = METHOD_SPECS[method].label
            scopes.update((target / label, output / scenario / "summary" / label,
                           output / label / scenario, output / label / "full_mass" / scenario))
        for filename in ("signal_region_sic", "mass_flatness_vs_selection"):
            for suffix in (".pdf", ".png"):
                (output / scenario / (filename + suffix)).unlink(missing_ok=True)
    if scan_groups:
        scopes.add(output / "comparison" / "injection_scan")
        scopes.add(output / "injection_scan" / "comparison")
        # Older scan files lived directly here; do not recurse into other methods.
        for path in (output / "injection_scan").glob("*"):
            if path.is_file() and (path.suffix in (".pdf", ".png") or path.name in (
                    "injection_scan.json", "injection_scan.csv")):
                path.unlink()
        for group in scan_groups.values():
            scopes.update(output / METHOD_SPECS[m].label / "injection_scan" for m in group)
            scopes.update(output / "injection_scan" / METHOD_SPECS[m].label for m in group)
    for scope in sorted(scopes):
        if not scope.is_dir():
            continue
        for path in scope.rglob("*"):
            if path.is_file() and (path.suffix in (".pdf", ".png") or path.name in (
                    "metrics.json", "injection_scan.json", "injection_scan.csv")):
                path.unlink()
        for path in sorted((p for p in scope.rglob("*") if p.is_dir()), reverse=True):
            if not any(path.iterdir()):
                path.rmdir()
        if not any(scope.iterdir()):
            scope.rmdir()
        # Remove empty containers from the obsolete method-first layout as well.
        for parent in scope.parents:
            if parent == output or not parent.is_relative_to(output):
                break
            if parent.is_dir() and not any(parent.iterdir()):
                parent.rmdir()
    legacy = output / "comparison_with_ranode"
    if legacy.is_dir() and not any(legacy.iterdir()):
        legacy.rmdir()
    (output / "ranode_comparison.json").unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Comparison and individual figures from all completed methods; no fitting")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("plots"))
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace generated figures in requested scopes, including obsolete layouts; preserve notes and other runs",
    )
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Optional method IDs to include; default: discover every completed method")
    parser.add_argument("--io-workers", type=int, default=2, help="CPU threads for numerical calculations and legacy RIDDLE checkpoint inference")
    parser.add_argument("--device", type=plot_device_argument, default="cpu",
                        help="RIDDLE safeguard/checkpoint inference device: cpu (default), cuda:<index>, "
                             "or auto (CUDA when available, otherwise CPU); other plotting stays on CPU")
    parser.add_argument("--no-safeguard-filtering", action="store_true",
                        help="Disable plot-time RIDDLE/R-ANODE fit filtering and use saved ensemble scores; "
                             "does not undo production exclusions or disable artifact/finite-score validation")
    parser.add_argument("--plot-workers", type=int, default=min(8, max(1, (os.cpu_count() or 1) // 2)),
                        help="Parallel PDF/PNG export processes (default: half the CPUs, capped at 8); use 1 for serial export")
    parser.add_argument("--plot-cache-mb", type=int, default=1024,
                        help="Memory limit in MiB for repeated numerical calculations (default: 1024); 0 disables caching")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--min-background", type=int, default=10)
    parser.add_argument("--random-reference", choices=("bootstrap", "subset"), default="bootstrap")
    parser.add_argument("--allow-smoke", action="store_true", help="Permit synthetic QA fixtures")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1,
                        help="0: warnings and result; 1: compact task progress; 2: detailed plotting progress")
    args = parser.parse_args(argv)
    from threadpoolctl import threadpool_limits
    args.population_summary = True
    set_verbosity(args.verbose)
    try:
        if not 0 < args.confidence < 1 or args.min_background < 1 or args.io_workers < 1:
            raise ValueError("Invalid confidence level or background support")
        if args.plot_workers < 1 or args.plot_cache_mb < 0:
            raise ValueError("Plot workers must be positive and plot cache size nonnegative")
        groups = discover(args.results, args.methods)
        scan_groups = discover(args.results, args.methods, scan=True)
        if not groups and not scan_groups:
            colored_status("No completed results for the requested methods", kind="WARNING")
            return 0
        args.results, args.output = args.results.resolve(), args.output.resolve()
        if args.output.is_relative_to(args.results) or args.results.is_relative_to(args.output):
            raise ValueError("Keep plots outside the result input tree")
        if args.output.exists():
            if not args.output.is_dir():
                raise NotADirectoryError("Plot output must be a directory")
            if not args.overwrite:
                raise FileExistsError("Plot output exists; use --overwrite or choose a new output directory")
            if any(path.is_symlink() for path in args.output.rglob("*")):
                raise ValueError("Cannot overwrite a plot directory containing symbolic links")
        score_loader = ScoreLoader(safeguard_filtering=not args.no_safeguard_filtering,
                                   io_workers=args.io_workers, device=args.device)
        # Retain requested scopes so obsolete figures for skipped results are
        # removed too; do not leave an old invalid curve behind on --overwrite.
        requested_groups = {key: dict(group) for key, group in groups.items()}
        requested_scan_groups = {key: dict(group) for key, group in scan_groups.items()}
        colored_status("Validate plot inputs | RIDDLE/R-ANODE safeguard filtering "
                       + ("disabled: use saved ensembles" if args.no_safeguard_filtering
                          else "active: check fits before plotting"), kind="INFO", level=1)
        with threadpool_limits(limits=args.io_workers):
            preflight_scores([*groups.values(), *scan_groups.values()], score_loader, allow_smoke=args.allow_smoke)
        groups = {key: group for key, group in groups.items() if group}
        scan_groups = {key: group for key, group in scan_groups.items() if group}
        args.output.mkdir(parents=True, exist_ok=args.overwrite)
        rows, settings, bundles = [], [], []
        comparisons = {}
        available = set().union(*(set(g) for g in [*groups.values(), *scan_groups.values()]))
        progress = PlotProgress(plot_task_count(groups, scan_groups))
        colored_status(f"Plot plan: {progress.total} task groups | {len(groups)} scenario/seed groups | "
                       f"{len(scan_groups)} injection-scan points | {args.plot_workers} export workers | "
                       f"{args.plot_cache_mb} MiB calculation cache | saved results only, no training",
                       kind="INFO", level=1)
        with local_progress("Plots", display=progress if args.verbose < 2 else None), \
                f.plt.style.context(f.STYLE), threadpool_limits(limits=args.io_workers), locked(args.output / ".plot.lock"), \
                f.plot_resources(workers=args.plot_workers, cache_mb=args.plot_cache_mb) as exporter:
            if args.overwrite:
                refresh_plot_scopes(args.output, requested_groups, requested_scan_groups)
            scan_audit = None
            if scan_groups:
                with progress.task("Injection scan | discrimination and significance"):
                    # Scans use saved ensemble scores, not member re-inference.
                    scan_loader = score_loader
                    scan_audit = render_injection_scan(scan_groups, args.output / "injection_scan" / "comparison", args, scan_loader)
                    for method in sorted(available):
                        cohort = {identity: {method: group[method]} for identity, group in scan_groups.items() if method in group}
                        if cohort:
                            render_injection_scan(cohort, args.output / "injection_scan" / METHOD_SPECS[method].label, args, scan_loader)
                    del scan_loader
            for (scenario, seed), group in groups.items():
                if (
                    any(
                        r["contract"].get("inputs", {}).get("synthetic_smoke_fixture")
                        for _, r in group.values()
                    )
                    and not args.allow_smoke
                ):
                    raise ValueError("Synthetic fixtures require --allow-smoke for QA")
                missing = set(args.methods or available) - set(group)
                if missing:
                    names = ", ".join(METHOD_SPECS[m].label if m in METHOD_SPECS else m for m in sorted(missing))
                    colored_status(f"{plot_group_label(scenario, seed, group)} | "
                                   f"no completed results for: {names}; plotting available methods",
                                   kind="INFO", level=1)
                target = args.output / scenario / f"seed_{seed:03d}"
                label = plot_group_label(scenario, seed, group)
                with progress.task(f"{label} | shared comparison plots"):
                    comparison = render_physical_comparison(group, target / "comparison", args, score_loader)
                    comparison["full_region"] = render_physical_comparison(
                        group, target / "comparison" / "full_region", args, score_loader, scope="full_region")
                comparisons[f"{scenario}/{seed}"] = comparison
                settings.extend(settings_rows(group, score_loader))
                rows.extend(comparison_rows(comparison, scenario, seed))
                rows.extend(comparison_rows(comparison["full_region"], scenario, seed, scope="full_region"))
                for method, source in group.items():
                    with progress.task(f"{label} | {METHOD_SPECS[method].label} individual plots"):
                        individual = {method: source}
                        destination = target / METHOD_SPECS[method].label
                        render_physical_comparison(individual, destination, args, score_loader)
                        render_physical_comparison(individual, destination / "full_region", args, score_loader, scope="full_region")
                        training_figures(individual, target, score_loader)
                        if method in ("lacathode", "riddle"):
                            bundle = make_bundle(individual, args.confidence, score_loader)
                            bundle["individual"] = method
                            render_bundle(bundle, destination / "full_mass", args)
                            bundles.append(bundle)
            audit = {}
            if summary_groups(groups):
                with progress.task("Summary plots | SIC and mass-flatness uncertainty bands"):
                    audit["comparison"] = physical_comparison_summary(
                        summary_groups(groups), args.output, args, score_loader)
                    audit["comparison"]["full_region"] = physical_comparison_summary(
                        summary_groups(groups, scope="full_region"), args.output, args, score_loader, scope="full_region")
                    for method in sorted(available):
                        cohort = summary_groups(groups, method=method)
                        if not cohort:
                            continue
                        with ProgressStage("individual_summary", f"Summarize {METHOD_SPECS[method].label}"):
                            view = METHOD_SPECS[method].label
                            audit[method] = physical_comparison_summary(
                                cohort, args.output, args, score_loader, view=view)
                            audit[method]["full_region"] = physical_comparison_summary(
                                summary_groups(cohort, scope="full_region"), args.output, args, score_loader,
                                scope="full_region", view=view)
                            individual = [b for b in bundles if b.get("individual") == method
                                          and (b["report"]["scenario"], b["report"]["seed"]) in cohort]
                            if individual:
                                audit[method]["full_mass"] = summary_figures(
                                    individual, args.output, args, view=view)
            with progress.task("Export comparison tables, configuration and plot manifest"):
                exporter.flush()
                for filename, table in (("comparison.csv", rows), ("configuration.csv", settings)):
                    if table:
                        csv_write(args.output / filename, table)
                    else:
                        (args.output / filename).unlink(missing_ok=True)
                write_json(args.output / "comparison.json", json_safe(comparisons))
                write_json(
                    args.output / "plot_manifest.json",
                    json_safe(
                        {
                            "schema": 3,
                            "methods": {m: dict(label=METHOD_SPECS[m].label, score_transform=METHOD_SPECS[m].score_transform, score_scope=METHOD_SPECS[m].score_scope)
                                        for m in sorted(available)},
                            "comparison": "All available methods in their supported scope: signal region and full region; independent acceptance and uncut denominators",
                            "layout": "<scenario>/seed_<seed>/{comparison,<method label>}/; summaries in <scenario>/summary/{comparison,<method label>}/ only when combining multiple result directories; full-region panels in full_region/; injection_scan/{comparison,<method label>}/",
                            "uncertainty": audit,
                            "cuts": "Calibrated RIDDLE: analytic percentile thresholds. Other scores: truth-assisted MC validation thresholds in the plotted region. Frozen cuts evaluated on independent physical test rows; each cut records its origin.",
                            "injection_scan": scan_audit,
                            "mass_cut_scan": {
                                "thresholds": list(f.SCORE_CUTS),
                                "comparison": "strict score > threshold",
                                "score_coordinate": "LaCathode classifier score; RIDDLE background percentile when score_kind=background_percentile_logit, otherwise sigmoid(log density ratio); neither density coordinate is a signal probability",
                                "scope": "full physical test mass range",
                                "retention_denominator": "all physical test events of the corresponding class, before cuts and mapping rejection",
                                "panels_per_page": 6,
                                "individual_directory": "04_mass_cuts/individual_cuts",
                                "histograms": "unweighted counts per fit, averaged over LaCathode fits; identical bins, no smoothing or pooled events; B + S is the sum of background and signal counts",
                            },
                            "mass_summary": "Common background in each plotted region: 300 equal-occupancy bins, test-derived cuts with uncut denominators; shape chi2 normalized by achieved efficiency.",
                            "uncertainty_scope": "16/50/84 percentiles across independent full method runs on fixed evaluation events; one saved ensemble score per RIDDLE/R-ANODE run; no within-ensemble fit bands",
                            "score_inputs": ("Legacy RIDDLE checkpoints evaluated on reserved validation; accepted scores reconstructed when needed. No training or modification of results."
                                if any(a.get("checkpoint_inference") for a in score_loader.fit_audit.values())
                                else "Saved predictions only; no checkpoint inference or modification of results"),
                            "safeguard_filtering": not args.no_safeguard_filtering,
                            "requested_inference_device": args.device,
                            "checkpoint_inference_devices": sorted({a["inference_device"] for a in
                                score_loader.fit_audit.values() if a.get("checkpoint_inference")}),
                            "score_ensembles": list(score_loader.fit_audit.values()),
                            "invalid_fit_policy": ("RIDDLE: numerical validity and reserved-validation improvement, including legacy checkpoint evaluation. R-ANODE: numerical validity and mass support. Rebuild one accepted-fit ensemble across all partitions; skip results with no accepted fits. Production exclusions retained; no plot-time training/retries; no test-truth/AUC/SIC selection."
                                if not args.no_safeguard_filtering else "Plot-time RIDDLE/R-ANODE safeguards disabled: use exact saved ensembles, retaining production exclusions. Artifact integrity, event alignment and finite-score checks remain active."),
                            "lacathode_event_plots": "Dynamically detected saved fits; mean per-fit histograms with matching per-fit validation cuts; shapes normalize those mean counts; no cross-fit score averaging",
                            "lacathode_roc_sic": "Median rejection and SIC interpolated at 1000 common signal efficiencies, as upstream, after statistical-support cuts. Bands are parametric 16/84-percentile ribbons at fixed signal efficiency; background-efficiency display axes retained.",
                            "lacathode_fit_counts": [{"scenario": b["report"]["scenario"], "seed": b["report"]["seed"], "fits": b["classifier_fit_count"]} for b in bundles if "classifier_fit_count" in b],
                            "summary_axes": f.SUMMARY_AXES,
                            "publication_y_ranges": f.PUBLICATION_Y_RANGES,
                            "warnings": [w for b in bundles for w in b["warnings"]],
                        }
                    ),
                )
        colored_status(f"Completed {progress.index}/{progress.total} plotting task groups | {args.output}", kind="PASS")
        return 0
    except (ValueError, OSError, KeyError, RuntimeError) as error:
        parser.exit(1, f"[ERROR] {error}\n")


def json_safe(value):
    if isinstance(value, dict):
        return {k: json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_safe(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return f.scalar(value)
    if isinstance(value, np.integer):
        return int(value)
    return value
