import argparse
from contextlib import contextmanager
from dataclasses import dataclass
import csv
import hashlib
import json
import math
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
from .production import validate_score_record, region_acceptance
from .evaluation import common_acceptance_auc, population_metadata, riddle_score_scope



efficiency_curve = f.cached_plot_calculation(efficiency_curve)
oracle_metrics = f.cached_plot_calculation(oracle_metrics)
roc_curve = f.cached_plot_calculation(roc_curve)












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
# Canonicalize mass-conditioned RIDDLE IDs onto the shared plotting path.



RIDDLE_METHOD_IDS = frozenset(("riddle", "riddlev2", "riddlev3"))
METHOD_SPECS = dict(BUILTINS)


def method_family(method):
    return "riddle" if method in RIDDLE_METHOD_IDS else method


def result_variant(report):
    return report.get("variant", report.get("contract", {}).get("inputs", {}).get("variant", "default"))


def scientific_protocol(report):
    value = report.get("contract", {}).get("scientific_version", 1)
    return f"v{value}" if isinstance(value, (int, float)) and not isinstance(value, bool) else str(value)


def protocol_rank(report):
    value = report.get("contract", {}).get("scientific_version", 1)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return (1, float(value))
    return (0, str(value))


def identity_parts(identity):
    if len(identity) == 2:
        return identity[0], identity[1], "default"
    if len(identity) >= 3:
        return identity[0], identity[1], identity[2]
    raise ValueError("Invalid plotting identity")


def variant_component(variant):
    if variant == "default":
        return None
    if not isinstance(variant, str) or not re.fullmatch(r"[A-Za-z0-9_.-]+", variant):
        raise ValueError(f"Unsafe dataset variant {variant!r}")
    return "variant_" + variant


def scope_root(output, scope):
    return output / ("SR-Only" if scope == "signal_region" else "Full-Range")


def scoped_target(output, scope, scenario, seed=None, variant="default"):
    target = scope_root(output, scope) / scenario
    component = variant_component(variant)
    if component:
        target /= component
    return target if seed is None else target / f"seed_{seed:03d}"


def is_riddle_report(report):
    return method_family(report.get("method")) == "riddle"


def riddle_plot_spec(report):
    # Treat NaNs outside the trained SR as intentionally unscored.

    base = BUILTINS["riddle"]
    return PlotMethod(base.label, base.color, base.linestyle, base.signal_color,
                      base.score_transform, riddle_score_scope(report))


def register_method(report):
    """New methods need only standard score artifacts and optional plot metadata."""
    method = report.get("method")
    if not isinstance(method, str) or not re.fullmatch(r"[a-z][a-z0-9_-]*", method):
        raise ValueError("Result method must be a safe lowercase identifier")
    family = method_family(method)
    if family == "riddle":
        return riddle_plot_spec(report)
    if family in BUILTINS:
        return BUILTINS[family]
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




POPULATION_STYLES = dict(f.POPULATIONS)
POPULATION_STYLES["ranode"] = {
    0: (BUILTINS["ranode"].color, "-"),
    1: (BUILTINS["ranode"].signal_color, "--"),
}


PHYSICAL_STYLES = {method: spec.style for method, spec in METHOD_SPECS.items()}


class PopulationMismatch(ValueError):
    pass


def plot_group_label(scenario, seed, group, variant="default"):
    names = ", ".join(STYLES[KEYS[m]][0] for m in KEYS if m in group)
    control = "" if variant == "default" else f" | {variant}"
    return f"{f.SCENARIO_LABELS[scenario]}{control} | seed {seed} | {names}"


def plot_task_count(groups, scan_groups):
    return (1 + bool(scan_groups) + bool(summary_groups(groups))
            + sum(1 + len(group) for group in groups.values()))


def summary_groups(groups, *, method=None, scope="signal_region"):
    """Combine only statistically repeatable runs from one scientific protocol per method.

    Per-seed comparisons may legitimately contain newer and older RIDDLE protocols in
    the same results tree.  Uncertainty bands must never treat those protocols as
    exchangeable repetitions, so each method is restricted to its highest numeric
    protocol (or most common named protocol) before summary aggregation.
    """
    eligible = {}
    for identity, group in groups.items():
        cohort = scope_group(group, scope)
        if method is not None:
            cohort = {method: cohort[method]} if method in cohort else {}
        if cohort:
            eligible[identity] = cohort
    if not eligible:
        return {}
    chosen = {}
    for name in set().union(*(set(g) for g in eligible.values())):
        reports = [report for group in eligible.values() if name in group for _, report in [group[name]]]
        numeric = [r for r in reports if isinstance(r.get("contract", {}).get("scientific_version"), (int, float))
                   and not isinstance(r.get("contract", {}).get("scientific_version"), bool)]
        if numeric:
            chosen[name] = scientific_protocol(max(numeric, key=protocol_rank))
        else:
            counts = {}
            for report in reports:
                tag = scientific_protocol(report)
                counts[tag] = counts.get(tag, 0) + 1
            chosen[name] = max(counts, key=lambda tag: (counts[tag], tag))
    filtered = {}
    counts = {}
    for identity, group in eligible.items():
        cohort = {name: source for name, source in group.items()
                  if scientific_protocol(source[1]) == chosen[name]}
        if not cohort:
            continue
        filtered[identity] = cohort
        scenario, _, variant = identity_parts(identity)
        key = (scenario, variant)
        counts[key] = counts.get(key, 0) + 1
    return {identity: group for identity, group in filtered.items()
            if counts[(identity_parts(identity)[0], identity_parts(identity)[2])] > 1}


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
    previous = f.METHODS, f.VIEWS, f.POPULATIONS
    f.METHODS = {k: STYLES[k] for k in keys}


    missing = [k for k in keys if k not in POPULATION_STYLES]
    if missing:
        reverse_keys = {plot_key: method for method, plot_key in KEYS.items()}
        for key in missing:
            method = reverse_keys.get(key)
            spec = METHOD_SPECS.get(method) if method is not None else None
            if spec is None:
                raise KeyError(f"Missing population style for plotting key {key!r}")
            POPULATION_STYLES[key] = {0: (spec.color, "-"), 1: (spec.signal_color, "--")}
    f.POPULATIONS = {k: POPULATION_STYLES[k] for k in keys}
    f.VIEWS = {view or ("comparison" if len(keys) > 1 else STYLES[keys[0]][0]): tuple(keys)}
    try:
        yield
    finally:
        f.METHODS, f.VIEWS, f.POPULATIONS = previous


def discover(root, requested=None, *, scan=False):
    paths = [root / "result.json"] if (root / "result.json").is_file() else sorted(root.rglob("result.json"))
    groups = {}
    for path in paths:
        report = json.loads(path.read_text())
        stored_method = report.get("method")
        method = method_family(stored_method)
        if (requested is not None and method not in requested) or not report.get("completed"):
            continue
        if method == "lacathode" and "run_index" in report and path.parent != root:
            continue
        point = report.get("contract", {}).get("inputs", {}).get("injection_scan")
        if bool(point) != scan:
            continue
        if method == "riddle" and "protocol.json" in report.get("artifacts_sha256", {}):
            protocol = read_metadata(path.parent, report, "protocol.json")

            report = {**report, "score_scope": riddle_score_scope({**report, "protocol": protocol})}
        spec = register_method(report)
        METHOD_SPECS[method] = spec
        KEYS.setdefault(method, "method_" + method)
        STYLES[KEYS[method]] = spec.style[:3]
        PHYSICAL_STYLES[method] = spec.style
        variant = result_variant(report)
        scenario, seed = report["scenario"], report["seed"]
        if scenario not in f.SCENARIOS or type(seed) is not int:
            raise ValueError("Invalid result identity")
        identity = ((point["signal_events"], point["replica"], report.get("run_index", 0), variant)
                    if scan else (scenario, seed, variant))
        group = groups.setdefault(identity, {})
        if method in group:
            old_root, old_report = group[method]
            if method == "riddle" and scientific_protocol(old_report) != scientific_protocol(report):
                preferred = (path.parent, report) if protocol_rank(report) > protocol_rank(old_report) else (old_root, old_report)
                discarded = old_report if preferred[1] is report else report
                group[method] = preferred
                colored_status(
                    f"Multiple RIDDLE protocols for {scenario}/seed_{seed:03d}/{variant}; "
                    f"using {scientific_protocol(preferred[1])} and ignoring {scientific_protocol(discarded)} for this plot cohort",
                    kind="WARNING", level=0,
                )
                continue
            raise ValueError(
                f"Duplicate {METHOD_SPECS[method].label}/scenario/seed/variant results; choose a narrower input directory"
            )
        group[method] = (path.parent, report)
    return groups


class InvalidFitScores(ValueError):
    """Numerically unusable predictions, distinct from damaged/misaligned files."""


class NoValidFits(ValueError):
    """A result cannot be plotted, but other method/run results can continue."""


def load_scores(root, report, name, *, attempt=None, for_rebuild=False):
    method = method_family(report.get("method"))
    relative = Path(name + "_scores.npz")
    if attempt is not None:
        relative = Path(attempt) / relative
    path = verify_plot_input(root, report, str(relative))
    with np.load(path, allow_pickle=False) as archive:
        fields = ("mass", "labels", "mask", "scores", "physical")
        if method == "lacathode" or "latent" in archive:
            fields += ("latent",)
        data = {k: archive[k] for k in fields}



        if method == "riddle" and "density_inputs" in archive:
            data["density_inputs"] = archive["density_inputs"]
            if "background_log_density" in archive:
                data["background_log_density"] = archive["background_log_density"]
        for key in ("raw_scores", "score_kind", "fit_score_kind", "event_ids",
                    "preprocessing_mask", "score_domain_mask", "score_scope"):
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
        if method == "ranode":
            data["is_signal_region"] = archive["is_signal_region"]
            region = data["is_signal_region"]
            if region.shape != data["mass"].shape or region.dtype != bool or (data["mask"] & ~region).any():
                raise ValueError("Invalid R-ANODE signal-region membership")
            if name == "signal_region" and not region.all():
                raise ValueError("R-ANODE evaluation contains non-SR events")
        if method == "riddle":
            from .production import validate_region

            if "is_signal_region" not in archive:
                raise ValueError("RIDDLE SR membership is missing; regenerate its score artifacts")
            data["is_signal_region"] = validate_region(archive["is_signal_region"], len(data["mass"]))
            if name == "signal_region" and not data["is_signal_region"].all():
                raise ValueError("RIDDLE signal-region evaluation contains non-SR events")
            expected_scope = riddle_score_scope(report)
            if "score_scope" in data and str(np.asarray(data["score_scope"]).item()) != expected_scope:
                raise ValueError("Score artifact scope disagrees with the RIDDLE result contract")
            if expected_scope == "signal_region" and (data["mask"] & ~data["is_signal_region"]).any():
                raise ValueError("SR-only RIDDLE result contains scored sideband events")
    n = len(data["mass"])
    contract = report.get("contract", {})
    inputs = contract.get("inputs", {})
    columns = inputs.get("columns")
    variant = report.get("variant", inputs.get("variant", "default"))
    physical_dimensions = (len(columns) - 2 if isinstance(columns, list) and len(columns) >= 3
                           else 5 if variant == "deltaR" else 4)
    riddle_settings = contract.get("settings", {}).get("riddle", {}) if method == "riddle" else {}
    mass_context = int(bool(riddle_settings.get("mass_conditioning", False)))
    latent_dimensions = physical_dimensions + mass_context if method == "riddle" else physical_dimensions
    if any(data[k].shape != (n,) for k in ("mass", "labels", "scores", "mask")):
        raise ValueError("Misaligned score arrays")
    mapped_rows = int(data["mask"].sum())
    invalid_mapping_shape = (
        ("latent" in data and data["latent"].shape != (mapped_rows, latent_dimensions))
        or ("density_inputs" in data and data["density_inputs"].shape != (mapped_rows, latent_dimensions))
        or ("background_log_density" in data and data["background_log_density"].shape != (mapped_rows,))
    )
    if data["mask"].dtype != bool or data["physical"].shape != (n, physical_dimensions) or invalid_mapping_shape:
        details = (f"physical={data['physical'].shape}, expected={(n, physical_dimensions)}; "
                   f"mask accepted={mapped_rows}; latent={data.get('latent', np.empty((0,))).shape if 'latent' in data else 'absent'}, "
                   f"expected latent width={latent_dimensions}")
        raise ValueError(f"Invalid feature/mapping shapes ({details})")
    if not np.isin(data["labels"], [0, 1]).all() or not all(
        np.isfinite(data[k]).all() for k in ("mass", "physical", "latent", "density_inputs", "background_log_density") if k in data
    ):
        raise ValueError("Invalid event features or labels")
    if for_rebuild:
        if method != "riddle":
            raise ValueError("Only RIDDLE checkpoint reconstruction may defer score validation")


        evidence = {**data, "scores": np.where(data["mask"], 0., np.nan)}
        evidence.pop("fit_scores", None)
        validate_score_record(evidence, method="riddle", stage=str(path))
        data.pop("fit_scores", None)
        return data
    if (not data["mask"].any() or not np.isfinite(data["scores"][data["mask"]]).all()
            or not np.isnan(data["scores"][~data["mask"]]).all()):
        raise InvalidFitScores(f"{path}: empty/nonfinite accepted scores or invalid rejected-event scores")
    validate_score_record(data, method=method, stage=str(path))
    if method == "riddle" and "method_health.json" in report.get("artifacts_sha256", {}):
        data["plot_saved_ensemble"] = True
    if method == "lacathode":
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
    if method in ("riddle", "ranode"):
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
        # Let integrity errors propagate outside numerical-failure handling.
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
    """Read saved checkpoints without modifying production artifacts."""
    import torch
    from .model import build_signal_flow, background_log_prob, signal_log_prob

    relative = Path("density") / member["directory"]
    inputs = read_metadata(root, report, str(relative / "residual_training_inputs.json"))
    version = report["contract"]["scientific_version"]
    if version not in (2, 3, 4, 5) or inputs.get("scientific_version") != version:
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
            # Stage checkpoints on CPU and move one inference batch at a time to GPU.

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
                    if checkpoint.get("mass_fraction_state") is not None:
                        from .mass_fraction import probabilities as mass_fraction_probabilities
                        mixture_weight = mass_fraction_probabilities(
                            checkpoint["mass_fraction_state"], z[:, -1], inputs["settings"])
                    else:
                        mixture_weight = weight
                    from .enhancements import mixture_gain
                    weighted = mixture_gain(ratio, mixture_weight)
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
        # Calibration is valid only for the exact frozen ensemble.

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
    if is_riddle_report(report):
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
    audit = dict(method=method_family(report["method"]), source=str(root.resolve()), scenario=report["scenario"], seed=report["seed"],
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
            method = method_family(report["method"])
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
                selection = (read_metadata(root, report, "density/ensemble_selection.json")
                             if "density/ensemble_selection.json" in report.get("artifacts_sha256", {}) else {})
                require_complete_ensemble(selection)
                chosen = next(c for c in selection["configurations"] if c["name"] == selection["selected_configuration"])
                self.fit_audit[identity] = dict(method="riddle", source=identity, status="producer_selection",
                    scenario=report["scenario"], seed=report["seed"], source_artifacts_modified=False,
                    requested_fits=selection["requested_runs"], used_fits=selection["valid_runs"],
                    used_members=selection["members"], excluded_members=chosen["excluded_fits"],
                    fit_recovery=selection["fit_recovery"], selection=selection["selection"], independent_runs=1,
                    safeguard_filtering=True, checkpoint_inference=False)
            if method == "lacathode" and "run_selection.json" in report.get("artifacts_sha256", {}):
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
        sr_masks = {k: r["is_signal_region"] for k, r in records.items() if "is_signal_region" in r}
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
            latents = {}
            for method, (_, method_report) in group.items():
                key, record = KEYS[method], records[KEYS[method]]
                if "fit_latents" in record:
                    values = record["fit_latents"]
                elif "latent" in record:
                    values = record["latent"]
                else:
                    continue
                # Treat mass as context, not as an extra latent coordinate.


                if (method == "riddle" and
                        method_report.get("contract", {}).get("settings", {}).get("riddle", {}).get("mass_conditioning", False)):
                    values = values[..., :-1]
                latents[key] = values
            bundle["latents_by_method"] = latents
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
                renderer(bundle, target)
                continue
            with ProgressStage(renderer.__name__, renderer.__name__.replace("render_", "Plot ")):
                renderer(bundle, target, args) if renderer in (
                    f.render_roc,
                    f.render_efficiency,
                ) else renderer(bundle, target)
    for key in keys:
        with methods([key], view="."), ProgressStage("representation", "Plot input and latent distributions"):
            representation = dict(bundle)
            if key in bundle["latents_by_method"]:
                representation["latents"] = bundle["latents_by_method"][key]
            f.render_representation(representation, target)
    with ProgressStage("evaluation_plots", "Plot signal-region performance"):
        render_full_pipeline(bundle, target, args)
    bundle["metrics"]["scoring_acceptance"] = bundle["acceptance"]
    keys_to_methods = {KEYS[method]: method for method in group}
    bundle["metrics"]["mapping_acceptance"] = {
        partition: {key: (region_acceptance(record["labels"], record["preprocessing_mask"], record["is_signal_region"])
                          if "preprocessing_mask" in record else
                          bundle["acceptance"][partition][key] if keys_to_methods[key] != "riddle"
                          or riddle_plot_spec(group["riddle"][1]).score_scope == "full_region" else None)
                    for key, record in records.items()}
        for partition, records in bundle["evaluation"].items()
    }
    bundle["metrics"]["mapping_acceptance_note"] = "null means the preprocessing/domain split is unavailable for legacy SR-only RIDDLE scores"
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
    """Render injection scans versus both S/B and injected signal-event count.

    Every figure is saved independently.  The upper x axis shows the realized
    uncut nominal ``S/sqrt(B)`` when all scan points share one SR background
    population, matching the convention used in the reference studies.
    """
    score_loader = load_scores if score_loader is None else score_loader
    rows, cohorts, classifier_fits = [], {}, []
    variants = {identity[3] if len(identity) > 3 else "default" for identity in groups}
    if len(variants) > 1:
        raise ValueError("Render each injection-scan dataset variant separately")
    variant = next(iter(variants), "default")
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
            allowed_protocols = (2, 3, 4, 5) if method == "riddle" else (expected_protocol,)
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
                fit_values = [oracle_metrics(data["labels"], score, data["mask"],
                                             min_background=args.min_background)
                              for score in fit_scores(data)]
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
            row = dict(method=method, variant=variant, signal_events=count, replica=replica, run_index=run_index,
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
        variant=variant,
        metric="Oracle maximum on truth-labelled SR test sample; mapping failures never pass",
        nominal_significance="Per replica: SIC times realized uncut SR S/sqrt(B), then aggregate",
        uncertainty="16/50/84 percentiles across independently seeded partitions and training; finite source pool is reused, not independent collision datasets",
        minimum_background_count=args.min_background, minimum_background_efficiency=1e-4,
        fixed_efficiency="SIC at physical background efficiency 0.001; linear ROC interpolation within supported points only",
        significance_caveat="Nominal S/sqrt(B); no systematics, background fit, Poisson calibration or trials correction",
        points=[], plotted_metrics={},
    )

    metric_specs = (
        ("oracle_full_pipeline_max_sic", "Maximum significance improvement", "maximum_significance_improvement"),
        ("oracle_max_nominal_significance", "Maximum achieved nominal significance", "maximum_achieved_significance"),
        ("sic_at_background_1e3", r"SIC at $\epsilon_B=10^{-3}$", "sic_at_background_1e3"),
        ("nominal_significance_at_background_1e3", r"Significance at $\epsilon_B=10^{-3}$", "significance_at_background_1e3"),
    )
    x_specs = (
        ("signal_to_background_percent", "Injected SR S/B [%]", "s_over_b", True),
        ("signal_events", "Injected signal events", "signal_events", False),
    )
    backgrounds = np.asarray([r["sr_background"] for r in rows], dtype=float)
    common_background = len(backgrounds) and np.all(backgrounds == backgrounds[0])
    for field, title, stem in metric_specs:
        for x_field, xlabel, x_stem, invert in x_specs:
            fig, ax, _ = f.canvas(title, xlabel)
            any_drawn, plotted = False, {}
            for method in KEYS:
                values = []
                for (name, count), cohort in cohorts.items():
                    if name != method:
                        continue
                    finite = [r for r in cohort if r[field] is not None and np.isfinite(r[field])]
                    if not finite:
                        continue
                    x = float(np.median([r[x_field] for r in finite]))
                    low, median, high = np.quantile([r[field] for r in finite], [.16, .5, .84])
                    values.append((x, low, median, high, len(finite), count))
                if not values:
                    continue
                values.sort()
                matrix = np.asarray(values, dtype=float)
                x, low, median, high = matrix[:, :4].T
                name, color, ls = STYLES[KEYS[method]]
                ax.plot(x, median, marker="x", ms=6, mew=1.2, color=color, ls=ls, label=name)
                supported = matrix[:, 4] >= 2
                if supported.any():
                    ax.fill_between(x, low, high, where=supported, color=color, alpha=.18, linewidth=0)
                if not supported.all():
                    colored_status(f"{name}: scan band unavailable at single-replica points", kind="WARNING")
                plotted[method] = [dict(x=float(a), low=float(l), median=float(m), high=float(h),
                                       replicas=int(n), signal_events=int(c))
                                   for a, l, m, h, n, c in values]
                any_drawn = True
            if not any_drawn:
                f.plt.close(fig)
                audit["plotted_metrics"][field + "/" + x_field] = plotted
                continue
            if field in ("oracle_max_nominal_significance", "nominal_significance_at_background_1e3"):
                for level in (3, 5):
                    ax.axhline(level, ls=":", color=".5", lw=1)
            ax.set_ylim(bottom=0)
            if invert:
                ax.invert_xaxis()
            if common_background:
                background = float(backgrounds[0])
                if x_field == "signal_to_background_percent":
                    factor = np.sqrt(background) / 100.0
                    top = ax.secondary_xaxis("top", functions=(lambda x, a=factor: x*a,
                                                                lambda x, a=factor: x/a))
                else:
                    factor = np.sqrt(background)
                    top = ax.secondary_xaxis("top", functions=(lambda x, a=factor: x/a,
                                                                lambda x, a=factor: x*a))
                top.set_xlabel(r"Uncut $S/\sqrt{B}$")
            title_text = "Signal-Injected" if variant == "default" else f"Signal-Injected | {variant}"
            f.legend(fig, title=title_text)
            f.save(fig, output / f"{stem}_vs_{x_stem}")
            audit["plotted_metrics"][field + "/" + x_field] = plotted
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


def comparison_rows(audit, scenario, seed, *, scope="signal_region", variant="default"):
    """Long-form tables have no fixed method columns or privileged pair."""
    rows = []
    def add(method, metric, value, selection="all"):
        if value is None or isinstance(value, (int, float, np.number)):
            rows.append(dict(scenario=scenario, variant=variant, seed=seed, scope="common_physical_" + scope,
                             method=method, selection=selection, metric=metric, value=f.scalar(value)))
    for method, metrics in audit.get("methods", {}).items():
        for metric, value in metrics.items():
            add(method, metric, value)
        for truth, values in metrics["acceptance"].items():
            for field, value in values.items():
                add(method, truth + "_" + field, value)
    for point in audit.get("working_points", []):
        selection = "exact_test_background_" + str(point["background_budget"])
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



def _sic_signal_curve(record, minimum_background):
    """SIC parameterized by signal efficiency on supported physical SR points."""
    if len(np.unique(record["labels"][record["mask"]])) != 2:
        return None
    b, sig, _ = efficiency_curve(record["labels"], record["scores"], record["mask"], full_pipeline=True)
    support = (b >= 1e-4) & (np.rint(b * (record["labels"] == 0).sum()) >= minimum_background) & (sig > 0)
    if support.sum() < 2:
        return None
    x, y = sig[support], sig[support] / np.sqrt(b[support])
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique, inverse = np.unique(x, return_inverse=True)
    maxima = np.full(len(unique), -np.inf)
    np.maximum.at(maxima, inverse, y)
    return unique, maxima


def render_variant_sic_ratio(groups, output, args, score_loader):
    """Compare non-default dataset/control variants with default at fixed signal efficiency.

    This figure is emitted only when a method has matching seed and scientific
    protocol in both the default and alternate dataset.  It therefore cannot
    accidentally compare a code/protocol change with a dataset shift.
    """
    indexed = {}
    for identity, group in groups.items():
        scenario, seed, variant = identity_parts(identity)
        if scenario != "signal_injection":
            continue
        for method, source in group.items():
            indexed[(method, seed, variant)] = source
    variants = sorted({key[2] for key in indexed if key[2] != "default"})
    audit = {}
    for variant in variants:
        per_method = {}
        for method in sorted({key[0] for key in indexed}):
            ratios = []
            members = []
            seeds = sorted({key[1] for key in indexed if key[0] == method and key[2] == variant})
            for seed in seeds:
                default = indexed.get((method, seed, "default"))
                alternate = indexed.get((method, seed, variant))
                if default is None or alternate is None:
                    continue
                if scientific_protocol(default[1]) != scientific_protocol(alternate[1]):
                    colored_status(
                        f"{METHOD_SPECS[method].label} seed {seed}: skip {variant}/default ratio across different scientific protocols",
                        kind="WARNING", level=1,
                    )
                    continue
                base = score_loader(*default, "signal_region")
                shifted = score_loader(*alternate, "signal_region")


                if not np.array_equal(base["labels"], shifted["labels"]):
                    colored_status(
                        f"{METHOD_SPECS[method].label} seed {seed}: skip {variant}/default ratio with different SR labels",
                        kind="WARNING", level=1,
                    )
                    continue
                base_curve = _sic_signal_curve(base, args.min_background)
                alt_curve = _sic_signal_curve(shifted, args.min_background)
                if base_curve is None or alt_curve is None:
                    continue
                lo = max(base_curve[0].min(), alt_curve[0].min())
                hi = min(base_curve[0].max(), alt_curve[0].max())
                if not hi > lo:
                    continue
                grid = np.linspace(lo, hi, 350)
                denominator = np.interp(grid, base_curve[0], base_curve[1])
                numerator = np.interp(grid, alt_curve[0], alt_curve[1])
                valid = np.isfinite(numerator) & np.isfinite(denominator) & (denominator > 0)
                if valid.sum() < 2:
                    continue
                ratios.append((grid[valid], numerator[valid] / denominator[valid]))
                members.append(dict(seed=seed, protocol=scientific_protocol(default[1])))
            if ratios:
                per_method[method] = (ratios, members)
        if not per_method:
            continue
        fig, ax, _ = f.canvas("Ratio of significance improvements", "Signal efficiency")
        variant_audit = {}
        for method, (curves, members) in per_method.items():
            lower = max(curve[0].min() for curve in curves)
            upper = min(curve[0].max() for curve in curves)
            if upper <= lower:
                continue
            grid = np.linspace(lower, upper, 350)
            matrix = np.asarray([np.interp(grid, x, y) for x, y in curves])
            low, median, high = np.quantile(matrix, [.16, .5, .84], axis=0)
            label, color, ls, _ = PHYSICAL_STYLES[method]
            ax.plot(grid, median, color=color, ls=ls, label=label)
            if len(matrix) > 1:
                ax.fill_between(grid, low, high, color=color, alpha=.18, linewidth=0)
            variant_audit[method] = dict(seeds=members, signal_efficiency=[float(grid[0]), float(grid[-1])],
                                         median_ratio=[float(v) for v in median])
        if not variant_audit:
            f.plt.close(fig)
            continue
        ax.axhline(1.0, color=".25", ls=":", lw=1.2)
        ax.set(xlim=(0, 1))
        f.legend(fig, title=f"Signal region | {variant} / default")
        target = scope_root(output, "signal_region") / "signal_injection" / "dataset_controls" / variant
        f.save(fig, target / "significance_improvement_ratio_vs_signal_efficiency")
        write_json(target / "significance_improvement_ratio.json", json_safe(variant_audit))
        audit[variant] = variant_audit
    return audit


def _prepared_data_directory(data_root, report):
    """Locate the immutable prepared directory by its manifest hashes only."""
    if data_root is None:
        return None
    data_root = Path(data_root)
    if not data_root.exists():
        return None
    inputs = report.get("contract", {}).get("inputs", {})
    expected = inputs.get("files")
    scenario = report.get("scenario")
    variant = result_variant(report)
    candidates = [data_root / scenario, data_root / variant / scenario, data_root]
    seen = set()
    for candidate in candidates:
        manifest = candidate / "inputs.json"
        if manifest.is_file():
            seen.add(manifest.resolve())
            try:
                payload = json.loads(manifest.read_text())
            except (OSError, ValueError):
                continue
            if payload.get("files") == expected:
                return candidate
    for manifest in data_root.rglob("inputs.json"):
        if manifest.resolve() in seen:
            continue
        try:
            payload = json.loads(manifest.read_text())
        except (OSError, ValueError):
            continue
        if payload.get("scenario") == scenario and payload.get("files") == expected:
            return manifest.parent
    return None


def _array_rows(directory, name):
    path = None if directory is None else Path(directory) / name
    if path is None or not path.is_file():
        return None
    try:
        return int(np.load(path, mmap_mode="r", allow_pickle=False).shape[0])
    except (OSError, ValueError):
        return None


def _event_eval_counts(root, report, score_loader):
    record = score_loader(root, report, "signal_region")
    return int((record["labels"] == 0).sum()), int((record["labels"] == 1).sum())


def _first_numeric(mapping, key):
    if isinstance(mapping, dict):
        if key in mapping and isinstance(mapping[key], (int, float)) and not isinstance(mapping[key], bool):
            return int(mapping[key])
        for value in mapping.values():
            found = _first_numeric(value, key)
            if found is not None:
                return found
    elif isinstance(mapping, list):
        values = [_first_numeric(value, key) for value in mapping]
        values = [value for value in values if value is not None]
        if values:
            return int(round(float(np.median(values))))
    return None


def event_size_rows(groups, score_loader, *, data_root=None):
    """Build a paper-friendly event accounting table from saved provenance.

    Counts come only from checksum-protected result metadata, score populations,
    or the exact prepared-data manifest used by that result.  Missing counts are
    left blank rather than inferred from a paper or hard-coded benchmark.
    """
    rows = []
    for identity, group in sorted(groups.items()):
        scenario, seed, variant = identity_parts(identity)
        for method, (root, report) in sorted(group.items()):
            protocol = scientific_protocol(report)
            evaluation_background, evaluation_signal = _event_eval_counts(root, report, score_loader)
            base = dict(
                method=METHOD_SPECS[method].label,
                scenario=scenario,
                variant=variant,
                scientific_protocol=protocol,
                seed=seed,
                evaluation_background=evaluation_background,
                evaluation_signal=evaluation_signal,
                evaluation_total=evaluation_background + evaluation_signal,
                evaluation_sample="independent signal-region evaluation",
            )
            data_dir = _prepared_data_directory(data_root, report)
            if method == "riddle":
                roles = (read_metadata(root, report, "background/data_roles.json")
                         if "background/data_roles.json" in report.get("artifacts_sha256", {}) else {})
                rows.append({**base, "component": "Background map", "model_type": "conditional density estimator",
                             "train_events": roles.get("map_train", {}).get("events"),
                             "train_sample": "sideband data (" + roles.get("map_train", {}).get("source", "prepared") + ")",
                             "validation_events": roles.get("map_val", {}).get("events"),
                             "validation_sample": "sideband data (" + roles.get("map_val", {}).get("source", "prepared") + ")",
                             "generated_reference_samples": "", "notes": "Background transformation/map"})
                if "correction_train" in roles:
                    correction_path = "density/background_correction/selection.json"
                    correction = (read_metadata(root, report, correction_path)
                                  if correction_path in report.get("artifacts_sha256", {}) else {})
                    gap = correction.get("pseudo_sr_closure", {})
                    gap_events = sum(int(w.get("events", 0)) for w in gap.get("windows", [])
                                     if "improvement" in w)
                    gap_note = (f"masked-gap evaluation: {gap_events} correction_val events; fixed upstream map; "
                                "not full-search closure")
                    independent = correction.get("independent_closure_diagnostic", {})
                    rows.append({**base, "component": "Background correction", "model_type": "conditional density estimator",
                                 "train_events": roles["correction_train"].get("events"),
                                 "train_sample": "sideband data (" + roles["correction_train"].get("source", "prepared") + ")",
                                 "validation_events": roles.get("correction_val", {}).get("events"),
                                 "validation_sample": "sideband data (" + roles.get("correction_val", {}).get("source", "prepared") + ")",
                                 "generated_reference_samples": "",
                                 "notes": gap_note})
                    rows.append({**base, "component": "Reserved sideband closure role",
                                 "model_type": "diagnostic only", "train_events": 0,
                                 "train_sample": "not used for training",
                                 "validation_events": independent.get("events") if independent.get("status") == "evaluated" else None,
                                 "validation_sample": "reserved closure role; no checkpoint or activation selection",
                                 "generated_reference_samples": "",
                                 "notes": (f"Reserved role size={roles.get('closure', {}).get('events', '')}; "
                                           f"independent q_phi diagnostic status={independent.get('status', 'not recorded')}; "
                                           "not the masked-gap gate or a full-search closure test")})
                selection = (read_metadata(root, report, "density/ensemble_selection.json")
                             if "density/ensemble_selection.json" in report.get("artifacts_sha256", {}) else {})
                members = selection.get("members", [])
                train_events = roles.get("residual_train", {}).get("events")
                validation_events = None
                if members:
                    name = "density/" + members[0]["directory"] + "/residual_training_inputs.json"
                    try:
                        inputs = read_metadata(root, report, name)
                        train_events = inputs.get("train_events", train_events)
                        validation_events = inputs.get("validation_events")
                    except (OSError, ValueError, KeyError):
                        pass
                validation_parts = []
                if validation_events is not None:
                    validation_parts.append("internal residual validation")
                for role, label in (("mixture_validation", "mixture selection"), ("evidence", "reserved evidence")):
                    count = roles.get(role, {}).get("events")
                    if count is not None:
                        validation_parts.append(f"{int(count):,} {label}")
                rows.append({**base, "component": "Residual signal density", "model_type": "residual density estimator",
                             "train_events": train_events, "train_sample": "signal-region data (unlabelled)",
                             "validation_events": validation_events,
                             "validation_sample": "; ".join(validation_parts) or "signal-region validation",
                             "generated_reference_samples": "",
                             "notes": f"{selection.get('valid_runs', '')}/{selection.get('requested_runs', '')} accepted fits; "
                                      f"{selection.get('selected_checkpoints', '')} selected checkpoints"})
            elif method == "lacathode":
                metadata = read_metadata(root, report, "protocol.json")
                rows.append({**base, "component": "Background flow", "model_type": "density estimator",
                             "train_events": _array_rows(data_dir, "outerdata_train.npy"),
                             "train_sample": "sideband data (outerdata_train)",
                             "validation_events": _array_rows(data_dir, "outerdata_val.npy"),
                             "validation_sample": "sideband data (outerdata_val)",
                             "generated_reference_samples": "", "notes": "Pinned upstream LaCATHODE flow"})
                rows.append({**base, "component": "Classifier", "model_type": "classifier",
                             "train_events": _array_rows(data_dir, "innerdata_train.npy"),
                             "train_sample": "signal-region data + generated background reference",
                             "validation_events": _array_rows(data_dir, "innerdata_val.npy"),
                             "validation_sample": "signal-region validation + generated background reference",
                             "generated_reference_samples": metadata.get("reference_samples", ""),
                             "notes": f"{metadata.get('classifier_runs', '')} classifier fits; generated reference count is configured pool size"})
            elif method == "ranode":
                metadata = read_metadata(root, report, "protocol.json")
                partitions = metadata.get("partitions", {})
                background = partitions.get("background", {})
                signal = partitions.get("signal", [])
                rows.append({**base, "component": "Background density", "model_type": "density estimator",
                             "train_events": _first_numeric(background, "training_rows"),
                             "train_sample": "sideband development data",
                             "validation_events": _first_numeric(background, "validation_rows"),
                             "validation_sample": "sideband validation",
                             "generated_reference_samples": "", "notes": metadata.get("background_split", "")})
                rows.append({**base, "component": "Signal-mixture density", "model_type": "density estimator",
                             "train_events": _first_numeric(signal, "training_rows"),
                             "train_sample": "signal-region training data",
                             "validation_events": _first_numeric(signal, "validation_rows"),
                             "validation_sample": "signal-region validation",
                             "generated_reference_samples": "",
                             "notes": f"{metadata.get('valid_runs', '')}/{metadata.get('requested_runs', '')} accepted fits"})
            else:
                rows.append({**base, "component": "Method", "model_type": "model",
                             "train_events": "", "train_sample": "",
                             "validation_events": "", "validation_sample": "",
                             "generated_reference_samples": "", "notes": "No method-specific event ledger available"})


    collapsed = {}
    for row in rows:
        key = tuple((k, str(v)) for k, v in row.items() if k != "seed")
        collapsed.setdefault(key, {**row, "seeds": []})["seeds"].append(row["seed"])
    result = []
    for item in collapsed.values():
        seeds = sorted(set(item.pop("seeds")))
        item["seeds"] = ";".join(map(str, seeds))
        item["seed_count"] = len(seeds)
        item.pop("seed", None)
        def describe(count, sample):
            return (f"{int(count):,} {sample}" if isinstance(count, (int, np.integer))
                    else (str(sample) if sample else ""))
        item["paper_train"] = describe(item.get("train_events"), item.get("train_sample"))
        item["paper_validation"] = describe(item.get("validation_events"), item.get("validation_sample"))
        item["paper_evaluation"] = (
            f"{int(item['evaluation_background']):,} SR background + {int(item['evaluation_signal']):,} SR signal"
            if isinstance(item.get("evaluation_background"), (int, np.integer))
            and isinstance(item.get("evaluation_signal"), (int, np.integer)) else ""
        )
        result.append(item)
    result.sort(key=lambda row: (row["scenario"], row["variant"], row["method"], row["component"]))
    return result

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
                    "variant": result_variant(report),
                    "seed": report["seed"],
                    "setting": key,
                    "value": value,
                }
            )
    return rows



def plot_history_series(ax, x, values, label, *, window=5, color=None):
    """Plot the moving average first, then the per-epoch curve on top."""
    x = np.asarray(x)
    values = np.asarray(values, dtype=float)
    average_line = None
    if len(values) >= window:
        smooth = np.convolve(values, np.ones(window) / window, mode="valid")
        average_line, = ax.plot(
            x[window - 1:], smooth, color=color, lw=2.0, ls="-", zorder=2,
            label=f"{label} | {window}-epoch average",
        )
        color = average_line.get_color()
    per_epoch, = ax.plot(
        x, values, color=color, ls=":", lw=1.25, alpha=.80, zorder=3,
        label=f"{label} | per epoch",
    )
    return average_line if average_line is not None else per_epoch


def _history_rows(root, report, relative):
    payload = read_metadata(root, report, relative)
    rows = payload.get("history") if isinstance(payload, dict) else payload
    if (not isinstance(rows, list) or not rows
            or any(not isinstance(row, dict) for row in rows)):
        return None
    return rows


def _background_nll_history(method, source):
    """Return train/validation NLL for the sideband-trained background model."""
    root, report = source
    if method == "riddle":
        try:
            rows = _history_rows(root, report, "background/history.json")
        except (OSError, ValueError, KeyError):
            return None
        if rows is None or any(k not in row for row in rows for k in ("train_nll", "validation_nll")):
            return None
        return {
            "train": np.asarray([row["train_nll"] for row in rows], dtype=float),
            "validation": np.asarray([row["validation_nll"] for row in rows], dtype=float),
        }

    if method == "ranode":
        protocol = read_metadata(root, report, "protocol.json")
        attempt = protocol.get("background_attempt")
        if not isinstance(attempt, str) or not attempt:
            return None
        histories = {}
        for kind, field in (("trainloss", "train"), ("valloss", "validation")):
            relative = attempt + f"/results/upstream/background/fit/{kind}_list.npy"
            try:
                path = verify_plot_input(root, report, relative)
            except (OSError, ValueError):
                return None
            values = np.asarray(np.load(path, allow_pickle=False), dtype=float)
            histories[field] = np.atleast_2d(values).mean(axis=0)
        return histories

    directory = root / "training"
    names = (f"{method}_model_train_losses.npy", f"{method}_model_val_losses.npy")
    if not all((directory / name).is_file() for name in names):
        names = ("my_ANODE_model_train_losses.npy", "my_ANODE_model_val_losses.npy")
    if not all((directory / name).is_file() for name in names):
        return None
    histories = {}
    for name, field in zip(names, ("train", "validation")):
        try:
            path = verify_plot_input(root, report, str((directory / name).relative_to(root)))
        except (OSError, ValueError):
            return None
        values = np.asarray(np.load(path, allow_pickle=False), dtype=float)
        histories[field] = np.atleast_2d(values).mean(axis=0)
    return histories


def _riddle_sr_model_nll_history(source, score_loader=None):
    root, report = source
    selection = read_metadata(root, report, "density/ensemble_selection.json")
    members = selection.get("members", [])
    audit = score_loader.fit_audit.get(str(root.resolve())) if score_loader is not None else None
    if audit is not None and audit.get("used_members"):
        members = audit["used_members"]
    histories = []
    for member in members:
        try:
            rows = _history_rows(root, report, "density/" + member["directory"] + "/residual_losses.json")
        except (OSError, ValueError, KeyError):
            return None
        if rows is None:
            return None
        histories.append(rows)
    if not histories or any(len(h) != len(histories[0]) for h in histories):
        return None
    return {
        "train": np.asarray(
            [np.mean([h[i]["train_nll"] for h in histories]) for i in range(len(histories[0]))], dtype=float
        ),
        "validation": np.asarray(
            [np.mean([h[i]["validation_nll"] for h in histories]) for i in range(len(histories[0]))], dtype=float
        ),
    }


def _ranode_sr_model_nll_history(source, score_loader=None):
    root, report = source
    protocol = read_metadata(root, report, "protocol.json")
    attempts = protocol.get("signal_attempts") or ([protocol["signal_attempt"]] if protocol.get("signal_attempt") else [])
    audit = score_loader.fit_audit.get(str(root.resolve())) if score_loader is not None else None
    if audit is not None and audit.get("used_members"):
        attempts = [member["attempt"] for member in audit["used_members"]]
    if not attempts:
        return None
    result = {}
    for kind, field in (("trainloss", "train"), ("valloss", "validation")):
        histories = []
        for attempt in attempts:
            relative = attempt + f"/results/upstream/signal/fit/{kind}.npy"
            try:
                path = verify_plot_input(root, report, relative)
            except (OSError, ValueError):
                return None
            histories.append(np.asarray(np.load(path, allow_pickle=False), dtype=float))
        if not histories or any(values.shape != histories[0].shape for values in histories):
            return None
        result[field] = np.asarray(histories).mean(axis=0)
    return result


def _lacathode_sr_classifier_history(source):
    """Return LaCATHODE classifier BCE histories for the SR-stage comparison."""
    root, report = source
    directory = root / "training"
    names = ("loss_matris.npy", "val_loss_matris.npy")
    if not all((directory / name).is_file() for name in names):
        return None
    histories = {}
    for name, field in zip(names, ("train", "validation")):
        try:
            path = verify_plot_input(root, report, str((directory / name).relative_to(root)))
        except (OSError, ValueError):
            return None
        values = np.asarray(np.load(path, allow_pickle=False), dtype=float)
        histories[field] = np.atleast_2d(values).mean(axis=0)
    return histories


def _sr_model_training_history(method, source, score_loader=None):
    """Return the native second-stage training objective for each method.

    RIDDLE and R-ANODE optimize signal-region density-model NLLs. LaCATHODE's
    second stage is instead a classifier, so its native objective is BCE. The
    cross-method plot therefore compares relative objective convergence, not
    absolute losses or like-for-like likelihood values.
    """
    if method == "riddle":
        return _riddle_sr_model_nll_history(source, score_loader)
    if method == "ranode":
        return _ranode_sr_model_nll_history(source, score_loader)
    if method == "lacathode":
        return _lacathode_sr_classifier_history(source)
    return None


def _relative_nll(values):
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not len(values) or not np.isfinite(values).all():
        return None
    scale = max(abs(float(values[0])), np.finfo(float).eps)
    return 100.0 * (values - values[0]) / scale


def _render_relative_history_comparison(
    histories, destination, stem, *, ylabel=r"Relative NLL change from epoch 1 [\%]", method_labels=None
):
    """Save one paper-ready standalone cross-method objective figure per split.

    Curves are normalized to their own epoch-1 objective, so the figure shows
    convergence trends only. This permits the SR-stage panel to include
    LaCATHODE's classifier BCE alongside RIDDLE/R-ANODE density NLLs without
    implying that their absolute objective values are numerically comparable.
    """
    if not histories:
        return
    method_labels = {} if method_labels is None else method_labels
    destination.mkdir(parents=True, exist_ok=True)
    for field, noun in (("train", "training"), ("validation", "validation")):
        fig, ax, _ = f.canvas(ylabel, "Training epoch")
        drawn_methods = []
        all_values = []
        max_epoch = 1
        for method, values in histories.items():
            relative = _relative_nll(values[field])
            if relative is None:
                continue
            default_label, color, _, _ = PHYSICAL_STYLES[method]
            label = method_labels.get(method, default_label)
            epochs = np.arange(1, len(relative) + 1)
            max_epoch = max(max_epoch, int(epochs[-1]))
            all_values.append(relative)



            if len(relative) >= 5:
                smooth = np.convolve(relative, np.ones(5) / 5, mode="valid")
                ax.plot(epochs[4:], smooth, color=color, ls="-", lw=2.0, zorder=2)
            else:
                ax.plot(epochs, relative, color=color, ls="-", lw=1.8, zorder=2)
            ax.plot(epochs, relative, color=color, ls=":", lw=1.15, alpha=.72, zorder=3)
            drawn_methods.append((label, color))

        if not drawn_methods:
            f.plt.close(fig)
            continue

        ax.axhline(0.0, color=".55", ls="--", lw=1.0, gid="publication-guide")
        ax.set_xlim(1, max_epoch)


        finite = np.concatenate([v[np.isfinite(v)] for v in all_values if np.isfinite(v).any()])
        if finite.size:
            low = min(float(finite.min()), 0.0)
            high = max(float(finite.max()), 0.0)
            span = high - low
            pad = 0.10 * span if span > 0 else max(0.1, 0.10 * max(abs(low), abs(high), 1.0))
            ax.set_ylim(low - pad, high + pad)
            ax._publication_fixed_ylim = True


        method_handles = [
            f.Line2D([], [], color=color, lw=2.0, ls="-", label=label)
            for label, color in drawn_methods
        ]
        style_handles = [
            f.Line2D([], [], color=".20", lw=2.0, ls="-", label="5-epoch average"),
            f.Line2D([], [], color=".20", lw=1.3, ls=":", label="Per epoch"),
        ]
        handles = [*method_handles, *style_handles]
        labels = [h.get_label() for h in handles]
        f.legend(fig, handles=handles, labels=labels, ncols=1)


        f.save(fig, destination / f"{stem}_{noun}")


def comparison_training_figures(group, output, score_loader=None):
    """Compare normalized training convergence while preserving objective semantics."""
    destination = output / "02_training"
    background = {}
    sr_model = {}
    for method, source in group.items():
        values = _background_nll_history(method, source)
        if values is not None:
            background[method] = values
        values = _sr_model_training_history(method, source, score_loader)
        if values is not None:
            sr_model[method] = values
    _render_relative_history_comparison(
        background, destination, "background_nll",
        ylabel=r"Relative NLL change from epoch 1 [\%]",
    )


    _render_relative_history_comparison(
        sr_model, destination, "sr_model_nll",
        ylabel=r"Relative NLL change from epoch 1 [\%]",
    )


def _plot_nll_history(history, destination, *, title, filename, ylabel="Negative log likelihood"):
    fig, ax, _ = f.canvas(ylabel, "Epoch")
    x = np.asarray([row.get("epoch", i) + 1 for i, row in enumerate(history)])
    plot_history_series(ax, x, [row["train_nll"] for row in history], "Train")
    plot_history_series(ax, x, [row["validation_nll"] for row in history], "Validation")
    f.legend(fig, title=title)
    f.save(fig, destination / filename)


def training_figures(group, output, score_loader=None):
    for method, (root, report) in group.items():
        if method == "ranode":
            audit = score_loader.fit_audit.get(str(root.resolve())) if score_loader is not None else None
            attempts = [m["attempt"] for m in audit["used_members"]] if audit else None
            render_ranode_training((root, report), output, signal_attempts=attempts)
            continue

        destination = output / STYLES[KEYS[method]][0] / "02_training"

        if method == "riddle":


            background_history = _history_rows(root, report, "background/history.json")
            if background_history is not None:
                _plot_nll_history(
                    background_history, destination,
                    title="RIDDLE | Background flow",
                    filename="background_nll",
                )



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
            _plot_nll_history(
                history, destination,
                title="RIDDLE | SR residual-mixture density",
                filename="sr_model_nll",
            )

            fig, ax, _ = f.canvas("Fitted mixture fraction", "Epoch")
            x = [r["epoch"] + 1 for r in history]
            plot_history_series(ax, x, [r["signal_fraction"] for r in history], "RIDDLE")
            f.legend(fig, title="RIDDLE | Residual-mixture fraction")
            f.save(fig, destination / "mixture_fraction")




            correction_name = "density/background_correction/history.json"
            if correction_name in report.get("artifacts_sha256", {}):
                correction_history = _history_rows(root, report, correction_name)
                if correction_history is not None:
                    _plot_nll_history(
                        correction_history, destination,
                        title=r"RIDDLE | Background correction $q_\phi$",
                        filename="background_correction_nll",
                    )

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
                for i, member, member_history in zip(positions, members, histories):
                    ax.plot(
                        [i],
                        [member_history[-1]["validation_nll"]],
                        "x",
                        color=".6",
                        label="Final epoch" if i == positions[0] else None,
                    )
                    ax.scatter(
                        np.full(len(member["epochs"]), i),
                        [member_history[e]["validation_nll"] for e in member["epochs"]],
                        s=12,
                        color=STYLES["residual"][1],
                        label="Selected" if i == positions[0] else None,
                    )
                ax.set(xticks=positions)
                f.legend(fig)
                f.save(fig, destination / "selected_vs_final_validation")
            continue



        for stage in ("background", "classifier"):
            directory = root / "training"
            names = (
                (f"{method}_model_train_losses.npy", f"{method}_model_val_losses.npy")
                if stage == "background"
                else ("loss_matris.npy", "val_loss_matris.npy")
            )
            if stage == "background" and not all((directory / name).is_file() for name in names):
                names = ("my_ANODE_model_train_losses.npy", "my_ANODE_model_val_losses.npy")
            if not all((directory / name).is_file() for name in names):
                continue
            fig, ax, _ = f.canvas(
                "Negative log likelihood" if stage == "background" else "Classification loss", "Epoch"
            )
            for name, label in zip(names, ("Train", "Validation")):
                verify_plot_input(root, report, str((directory / name).relative_to(root)))
                values = np.load(directory / name)
                fit_histories = np.atleast_2d(values)
                epochs = np.arange(fit_histories.shape[1])
                mean_history = fit_histories.mean(axis=0)
                line = plot_history_series(ax, epochs + 1, mean_history, label)
                if (len(fit_histories) > 1
                        and report.get("contract", {}).get("lacathode_run_layout") == "independent_background_classifier_v1"):
                    low, high = np.quantile(fit_histories, [.16, .84], axis=0)
                    ax.fill_between(epochs + 1, low, high, color=line.get_color(), alpha=.14, linewidth=0)
            f.legend(
                fig,
                title="LaCathode | " + ("Background flow" if stage == "background" else "Classifier"),
            )
            f.save(fig, destination / ("background_nll" if stage == "background" else "classifier_loss"))

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
    identified = [r for r in records.values() if "event_ids" in r]
    if identified and any(not np.array_equal(identified[0]["event_ids"], r["event_ids"])
                          for r in identified[1:]):
        raise ValueError("Comparison evaluation event identities differ")
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
    if scope not in ("signal_region", "full_region"):
        raise ValueError("Unknown comparison scope")
    return {method: source for method, source in group.items()
            if scope == "signal_region" or method_family(method) == "riddle"
            or METHOD_SPECS[method].score_scope == "full_region"}


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
    for key in ("event_ids", "preprocessing_mask", "score_domain_mask", "is_signal_region"):
        if key in record:
            result[key] = record[key][sr]
    if "score_scope" in record:
        result["score_scope"] = record["score_scope"]
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


def exact_background_selection(record, budget):
    """Return score-only fractional selections at an exact empirical BG efficiency.

    A finite unweighted sample cannot in general contain exactly ``budget * N``
    background events.  For publication working points we therefore use the
    standard randomized/interpolated ROC convention: events above the boundary
    score receive weight one, events below receive zero, and all events tied at
    the boundary receive the same fractional weight.  The fraction is chosen
    from background ranks only and is then applied identically to signal.

    This is a truth-assisted *evaluation* working point, never a training or
    model-selection threshold.  It makes the reported background efficiency
    exactly equal to ``budget`` while preserving a score-only decision rule.
    """
    labels = np.asarray(record["labels"])
    mask = np.asarray(record["mask"], dtype=bool)
    total_background = int(np.sum(labels == 0))
    total_signal = int(np.sum(labels == 1))
    target = float(budget) * total_background
    if total_background <= 0 or target <= 0:
        return None

    rows, cuts, fractions = [], [], []
    for scores in fit_scores(record):
        scores = np.asarray(scores, dtype=float)
        available = mask & (labels == 0) & np.isfinite(scores)
        bg_scores = scores[available]
        if len(bg_scores) < int(np.ceil(target)):
            return None
        ordered = np.sort(bg_scores)[::-1]
        boundary = float(ordered[int(np.ceil(target)) - 1])
        higher_background = int(np.sum(bg_scores > boundary))
        tied_background = int(np.sum(bg_scores == boundary))
        if tied_background < 1:
            raise ValueError("Exact background working point has no boundary tie")
        fraction = float((target - higher_background) / tied_background)

        fraction = float(np.clip(fraction, 0.0, 1.0))
        weights = np.zeros(len(scores), dtype=float)
        finite = mask & np.isfinite(scores)
        weights[finite & (scores > boundary)] = 1.0
        weights[finite & (scores == boundary)] = fraction
        rows.append(weights)
        cuts.append(boundary)
        fractions.append(fraction)

    weights = np.asarray(rows)
    bg_eff = float(np.mean(np.sum(weights[:, labels == 0], axis=1) / total_background))
    sig_eff = (
        float(np.mean(np.sum(weights[:, labels == 1], axis=1) / total_signal))
        if total_signal > 0 else None
    )


    if not np.isclose(bg_eff, budget, rtol=0, atol=5e-15):
        raise ValueError(f"Exact background selection failed: {bg_eff} != {budget}")
    return dict(
        weights=weights,
        cuts=np.asarray(cuts, dtype=float),
        boundary_fractions=np.asarray(fractions, dtype=float),
        background_efficiency=float(budget),
        signal_efficiency=sig_eff,
        fit_count=len(weights),
    )


def weighted_selection_efficiency(record, selection, truth):
    weights = np.atleast_2d(np.asarray(selection, dtype=float))
    population = np.asarray(record["labels"]) == truth
    total = int(population.sum())
    if total == 0:
        return None
    return float(np.mean(np.sum(weights[:, population], axis=1) / total))


def weighted_selection_histogram(values, edges, selection, population):
    """Mean selected histogram across fits, supporting fractional WP weights."""
    values = np.asarray(values)
    population = np.asarray(population, dtype=bool)
    weights = np.atleast_2d(np.asarray(selection, dtype=float))
    if weights.shape[1] != len(values):
        raise ValueError("Selection weights and plotted values have different lengths")
    histograms = [np.histogram(values, edges, weights=w * population)[0] for w in weights]
    return histograms[0] if len(histograms) == 1 else np.mean(histograms, axis=0)


def working_point_relation(budget):
    # Evaluate publication working points at exact physical background efficiency.

    return "="


def working_point_efficiency_label(value, budget, *, signal=False):
    if value is None:
        return "n/a"
    if signal:
        return f"{100 * value:.2f}%"
    return f"{100 * budget:.1f}%"


def render_physical_working_points(validation, test, output, *, scenario=None, scope="signal_region"):
    require_same_physical_population(test)
    # Select each method's validation region independently.


    validation = {m: physical_sr_record(r, scope_region({m: r}, scope)) for m, r in validation.items()}
    test = {m: physical_sr_record(r, scope_region(test, scope)) for m, r in test.items()}
    require_same_physical_population(test)
    base = next(iter(test.values()))
    edges = (np.linspace(3.3, 3.7, 21) if scope == "signal_region" else
             np.linspace(min(1., base["mass"].min()), max(9., base["mass"].max()), 81))
    centers = 0.5 * (edges[:-1] + edges[1:])

    def overlay_markers(axis, values, *, color, marker, ms=4.5, mew=0.9, zorder=4):
        values = np.asarray(values, dtype=float)
        keep = np.isfinite(values)
        if np.any(keep):
            axis.plot(
                centers[keep], values[keep], ls="none", marker=marker, ms=ms,
                mew=mew, color=color, zorder=zorder,
            )

    marker_cycle = {"lacathode": "o", "riddle": "s", "ranode": "^"}
    fallback_markers = ("D", "v", "P", "X", "<", ">")
    neutral_bg_marker = "o"
    neutral_signal_marker = "^"

    inclusive = np.histogram(base["mass"][base["labels"] == 0], edges)[0]
    audit = []
    budgets = (0.10, 0.05, 0.01, 0.004)
    for budget in budgets:
        selected, retention = {}, {}
        for method, val in validation.items():
            f.validate_fit_counts([val, test[method]])
            record = test[method]
            exact = exact_background_selection(record, budget)
            if exact is None:
                continue
            keep = exact["weights"]
            cut = exact["cuts"]
            selected[method] = keep
            retention[method] = [exact["background_efficiency"], exact["signal_efficiency"]]
            audit.append(
                dict(
                    method=method,
                    background_budget=budget,
                    evaluation_population=population_metadata(record, "test", scope),
                    exact_target=True,
                    cut=float(cut[0]) if len(cut) == 1 else cut.tolist(),
                    boundary_fraction=(
                        float(exact["boundary_fractions"][0])
                        if len(exact["boundary_fractions"]) == 1
                        else exact["boundary_fractions"].tolist()
                    ),
                    fit_count=exact["fit_count"],
                    threshold_source=(
                        "exact truth-assisted test-background ROC interpolation; "
                        "fractional boundary weight applied identically to signal"
                    ),
                    background_efficiency=float(budget),
                    signal_efficiency=exact["signal_efficiency"],
                )
            )
        if not selected:
            continue
        relation = working_point_relation(budget)
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
            overlay_markers(ax, norm, color=".65", marker=neutral_bg_marker, ms=4.0, mew=0.8, zorder=3)
            columns.append(("No cut", [(handle, "Background")]))
            if not shape:
                signal_hist = np.histogram(base["mass"][base["labels"] == 1], edges)[0]
                handle = ax.stairs(
                    signal_hist, edges, color=".45", ls="--", baseline=None
                )
                overlay_markers(ax, signal_hist, color=".45", marker=neutral_signal_marker, ms=4.0, mew=0.8, zorder=3)
                columns[0][1].append((
                    handle, "Signal" if scenario != "background_only" or signal_hist.sum() else None
                ))
            fallback_index = 0
            for method, keep in selected.items():
                label, color, _, signal_color = PHYSICAL_STYLES[method]
                record = test[method]
                bg_hist = weighted_selection_histogram(
                    record["mass"], edges, keep, record["labels"] == 0
                )
                hist = bg_hist / bg_hist.sum() if shape and bg_hist.sum() else bg_hist
                b, sig_eff = retention[method]
                background_text = working_point_efficiency_label(b, budget, signal=False)
                signal_text = working_point_efficiency_label(sig_eff, budget, signal=True)
                family = method_family(method)
                marker = marker_cycle.get(family)
                if marker is None:
                    marker = fallback_markers[fallback_index % len(fallback_markers)]
                    fallback_index += 1
                handle = ax.stairs(hist, edges, color=color, baseline=None)
                overlay_markers(ax, hist, color=color, marker=marker)
                entries = [
                    (
                        handle,
                        (f"B: {background_text}" if not shape else f"B: {background_text} | S: {signal_text}")
                        if b is not None
                        else (None if not shape else f"S: {signal_text}"),
                    )
                ]
                if scenario == "background_only":
                    entries = [(handle, f"B: {background_text}" if b is not None else "Background")]
                if not shape:
                    shist = weighted_selection_histogram(
                        record["mass"], edges, keep, record["labels"] == 1
                    )
                    handle = ax.stairs(
                        shist, edges, color=signal_color, ls="--", baseline=None
                    )
                    overlay_markers(ax, shist, color=signal_color, marker=marker)
                    signal_label = (
                        f.retention_label(1, sig_eff, int((record["labels"] == 1).sum()), scenario=scenario)
                        if scenario == "background_only" else f"S: {signal_text}"
                    )
                    entries.append((handle, signal_label))
                columns.append((label, entries))
                ratio = np.divide(
                    hist, norm, out=np.full(len(hist), np.nan), where=norm > 0
                )
                lower.stairs(ratio, edges, color=color, baseline=None)
                overlay_markers(lower, ratio, color=color, marker=marker, zorder=3)
            if shape:



                ax.set_ylim(0.0, 0.20)
                ax._publication_fixed_ylim = True
                lower.axhline(1.0, color=".5", ls=":", lw=1.0, gid="publication-guide")
                lower.set_ylim(0.0, 2.0)
                lower._publication_fixed_ylim = True
            else:




                lower.axhline(float(budget), color=".5", ls=":", lw=1.0,
                              gid="publication-guide")
                ax.set_yscale("symlog", linthresh=1)
            ax.set_xlim(edges[0], edges[-1])
            f.population_legend(
                fig, columns, title=f"{scope_label(scope)} | B {relation} {100 * budget:.1f}%"
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
                render_physical_features(
                    test, selected, retention, budget, output,
                    scenario=scenario, density=density, scope=scope
                )
    return audit


def render_physical_efficiency(records, selected, edges, budget, output, *, scope="signal_region"):
    fig, ax, _ = f.canvas("Background efficiency", r"$m_{jj}$ [TeV]")
    centers = 0.5 * (edges[:-1] + edges[1:])
    marker_cycle = {"lacathode": "o", "riddle": "s", "ranode": "^"}
    fallback_markers = ("D", "v", "P", "X", "<", ">")
    fallback_index = 0
    for method, keep in selected.items():
        record = records[method]
        bg = record["labels"] == 0
        full = np.histogram(record["mass"][bg], edges)[0]
        passed = weighted_selection_histogram(record["mass"], edges, keep, bg)
        values = np.divide(passed, full, out=np.full(len(full), np.nan), where=full > 0)
        label, color, ls, _ = PHYSICAL_STYLES[method]
        family = method_family(method)
        marker = marker_cycle.get(family)
        if marker is None:
            marker = fallback_markers[fallback_index % len(fallback_markers)]
            fallback_index += 1



        ax.stairs(values, edges, color=color, ls=ls, baseline=None)
        ax.plot(
            centers, values, ls="none", marker=marker, ms=4.5, mew=0.9,
            color=color, label=label,
        )
    ax.axhline(budget, color=".5", ls=":", label="Validation target")
    ax.set(xlim=(edges[0], edges[-1]), ylim=(0.0, 0.20))



    ax._publication_fixed_ylim = True
    relation = working_point_relation(budget)
    f.legend(fig, title=f"{scope_label(scope)} | B {relation} {100 * budget:.1f}%")
    tag = f"{100 * budget:g}".replace(".", "p") + "pct"
    f.save(fig, output / "03_mass_sculpting" / ("background_efficiency_" + tag))


def render_physical_no_cut_efficiency(records, output, *, scope="full_region"):
    """Plot the uncut background score/mapping acceptance versus mass.

    This is the no-selection companion to the fixed-B working-point plots.  The
    numerator is every scorable background event and the denominator is every
    physical background event in the same mass bin.  Thus a method with complete
    score coverage is exactly one across the full range; any dip reflects only
    score/mapping acceptance, not a background-selection threshold.
    """
    if scope != "full_region" or not records:
        return
    require_same_physical_population(records)
    base = next(iter(records.values()))
    edges = np.linspace(min(1., base["mass"].min()), max(9., base["mass"].max()), 81)
    centers = 0.5 * (edges[:-1] + edges[1:])
    fig, ax, _ = f.canvas("Background efficiency", r"$m_{jj}$ [TeV]")
    marker_cycle = {"lacathode": "o", "riddle": "s", "ranode": "^"}
    fallback_markers = ("D", "v", "P", "X", "<", ">")
    fallback_index = 0
    for method, record in records.items():
        bg = np.asarray(record["labels"]) == 0
        full = np.histogram(record["mass"][bg], edges)[0]
        scorable = bg & np.asarray(record["mask"], dtype=bool)
        passed = np.histogram(record["mass"][scorable], edges)[0]
        values = np.divide(passed, full, out=np.full(len(full), np.nan), where=full > 0)
        label, color, ls, _ = PHYSICAL_STYLES[method]
        family = method_family(method)
        marker = marker_cycle.get(family)
        if marker is None:
            marker = fallback_markers[fallback_index % len(fallback_markers)]
            fallback_index += 1
        ax.stairs(values, edges, color=color, ls=ls, baseline=None)
        ax.plot(
            centers, values, ls="none", marker=marker, ms=4.5, mew=0.9,
            color=color, label=label,
        )
    ax.axhline(1.0, color=".5", ls=":", label="No-cut target")
    ax.set(xlim=(edges[0], edges[-1]), ylim=(0.0, 1.05))
    ax._publication_fixed_ylim = True
    f.legend(fig, title=f"{scope_label(scope)} | No BG cut")
    f.save(fig, output / "03_mass_sculpting" / "background_efficiency_no_cut")


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
                hist = weighted_selection_histogram(
                    record["physical"][:, index], edges, keep, record["labels"] == truth
                )
                if density and hist.sum():
                    hist = hist / (hist.sum() * np.diff(edges))
                handle = ax.stairs(hist, edges, color=color, ls=ls, baseline=None)
                efficiency = retention[method][truth]
                percent = working_point_efficiency_label(
                    efficiency, budget, signal=(truth == 1)
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
            fig, columns, title=f"{scope_label(scope)} | B {working_point_relation(budget)} {100 * budget:.1f}%"
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
        for kind, color in (("trainloss", "#8B1A1A"), ("valloss", "#56B4E9")):
            values = []
            for attempt in attempts:
                path = attempt + f"/results/upstream/{stage}/fit/" + kind + suffix + ".npy"
                values.append(np.load(
                    verify_plot_input(root, report, path), allow_pickle=False,
                ))
            values = np.asarray(values)
            epochs = np.arange(1, values.shape[1] + 1)
            mean_history = values.mean(axis=0)
            label = "Train" if kind == "trainloss" else "Validation"
            plot_history_series(ax, epochs, mean_history, label, color=color)
        f.legend(
            fig,
            title="R-ANODE | " + (
                "Background flow" if stage == "background" else "SR data density (signal-mixture flow)"
            ),
        )
        f.save(
            fig,
            output / "R-ANODE" / "02_training" / (
                "background_nll" if stage == "background" else "sr_model_nll"
            ),
        )

def _ensemble_member_scores(method, source, score_loader, *, scope="signal_region"):
    """Return saved member scores for ensemble-convergence diagnostics.

    In SR-only scope, all three methods are supported.  In full-range scope,
    only methods with saved full-range score artifacts contribute.
    """
    root, report = source
    partition = "signal_region" if scope == "signal_region" else "test"
    record = score_loader(root, report, partition)
    identity = str(root.resolve())
    if method == "ranode":
        if scope != "signal_region":
            return record, None, None
        audit = score_loader.fit_audit.get(identity, {})
        members = audit.get("used_members", [])
        if not members:
            return record, None, None
        rows = [load_scores(root, report, "signal_region", attempt=m["attempt"]) for m in members]
        if any(not np.array_equal(record[k], row[k]) for row in rows for k in ("mass", "labels", "mask")):
            raise ValueError("R-ANODE ensemble-convergence members are misaligned")
        scores = np.stack([row["scores"] for row in rows]).astype(np.float64)
        return record, scores, "equal-weight log-mean-exp of likelihood-ratio fits"

    path = verify_plot_input(root, report, f"{partition}_scores.npz")
    with np.load(path, allow_pickle=False) as archive:
        scores = np.asarray(
            archive["fit_scores"] if "fit_scores" in archive else archive["scores"][None, :],
            dtype=np.float64,
        )
        if scores.ndim != 2 or scores.shape[1] != len(record["labels"]):
            raise ValueError(f"{METHOD_SPECS[method].label} ensemble-convergence score shape is invalid")
        if method == "riddle" and "fit_indices" in archive:
            audit = score_loader.fit_audit.get(identity, {})
            used = [m.get("fit_index") for m in audit.get("used_members", []) if m.get("fit_index") is not None]
            if used:
                indices = np.asarray(archive["fit_indices"], dtype=int)
                lookup = {int(value): i for i, value in enumerate(indices)}
                if any(int(value) not in lookup for value in used):
                    raise ValueError("RIDDLE ensemble-convergence fit identities disagree with accepted ensemble")
                scores = scores[[lookup[int(value)] for value in used]]
        score_kind = (
            str(archive["fit_score_kind"])
            if "fit_score_kind" in archive
            else str(archive["score_kind"])
            if "score_kind" in archive
            else ""
        )

    if not np.isfinite(scores[:, record["mask"]]).all() or not np.isnan(scores[:, ~record["mask"]]).all():
        raise ValueError(f"{METHOD_SPECS[method].label} ensemble-convergence member scores are invalid")
    if method == "riddle":
        aggregation = (
            "equal-weight log-mean-exp of residual likelihood-ratio fits"
            if "ratio" in score_kind or not score_kind
            else "arithmetic mean in saved member-score coordinate (diagnostic)"
        )
    else:
        # Use median member performance because pinned LaCATHODE does not average fits.


        aggregation = "median of per-fit performance; LaCATHODE production does not cross-fit-average scores"
    return record, scores, aggregation


def _exact_signal_efficiencies_for_score(record, scores, budgets):
    """Signal efficiencies at exact physical BG efficiencies for one score vector."""
    labels = np.asarray(record["labels"])
    mask = np.asarray(record["mask"], dtype=bool)
    scores = np.asarray(scores, dtype=float)
    bg_total = int(np.sum(labels == 0))
    sig_total = int(np.sum(labels == 1))
    available_bg = mask & (labels == 0) & np.isfinite(scores)
    bg_scores = scores[available_bg]
    sig_scores = scores[mask & (labels == 1) & np.isfinite(scores)]
    if bg_total <= 0 or sig_total <= 0 or not len(bg_scores):
        return None
    targets = np.asarray(budgets, dtype=float) * bg_total
    if np.any(np.ceil(targets).astype(int) > len(bg_scores)):
        return None
    kth = len(bg_scores) - np.ceil(targets).astype(int)
    partitioned = np.partition(bg_scores, np.unique(kth))
    result = []
    for target, index in zip(targets, kth):
        boundary = float(partitioned[index])
        higher_bg = int(np.sum(bg_scores > boundary))
        tied_bg = int(np.sum(bg_scores == boundary))
        if tied_bg < 1:
            raise ValueError("Exact ensemble-convergence working point has no boundary tie")
        fraction = float(np.clip((target - higher_bg) / tied_bg, 0.0, 1.0))
        passed_signal = float(np.sum(sig_scores > boundary) + fraction * np.sum(sig_scores == boundary))
        result.append(passed_signal / sig_total)
    return np.asarray(result, dtype=float)


def _fit_orderings(count, repetitions=32, seed=3407):
    if count < 1:
        return []
    if count == 1:
        return [np.array([0], dtype=int)]
    maximum = min(repetitions, math.factorial(count)) if count < 8 else repetitions
    rng = np.random.default_rng(seed + 7919 * count)
    orders = [np.arange(count, dtype=int)]
    seen = {tuple(orders[0])}
    while len(orders) < maximum:
        order = rng.permutation(count)
        key = tuple(int(v) for v in order)
        if key in seen:
            continue
        seen.add(key)
        orders.append(order)
    return orders


def _ensemble_progressive_scores(record, members, aggregation, order):
    """Yield progressive ensemble scores for one deterministic fit ordering."""
    valid = np.asarray(record["mask"], dtype=bool)
    use_log_mean = "log-mean-exp" in aggregation
    cumulative = None
    arithmetic = None
    for k, index in enumerate(order, start=1):
        score = np.asarray(members[index], dtype=np.float64)
        if not np.isfinite(score[valid]).all():
            raise ValueError(
                "Ensemble-convergence member contains nonfinite values on accepted events"
            )
        combined = np.full(score.shape, np.nan, dtype=np.float64)
        if use_log_mean:
            if cumulative is None:
                cumulative = score[valid].copy()
            else:
                cumulative = np.logaddexp(cumulative, score[valid])
            combined[valid] = cumulative - np.log(k)
        else:
            if arithmetic is None:
                arithmetic = score[valid].copy()
            else:
                arithmetic += score[valid]
            combined[valid] = arithmetic / k
        yield k, combined


def _convergence_ticks(maximum_members):
    step = 1 if maximum_members <= 10 else 2 if maximum_members <= 20 else max(1, maximum_members // 10)
    ticks = list(range(1, maximum_members + 1, step))
    if ticks[-1] != maximum_members:
        ticks.append(maximum_members)
    return ticks


def _overall_metrics_for_score(record, scores, *, min_background=10):
    metrics = oracle_metrics(record["labels"], scores, record["mask"], min_background=min_background)
    auc = metrics.get("conditional_auc")
    sic = metrics.get("full_pipeline_max_sic")
    if auc is None or sic is None:
        return None
    return np.asarray((float(auc), float(sic)), dtype=float)


def _set_convergence_ylim(ax, curves, *, floor=None):
    finite = [np.asarray(values, dtype=float)[np.isfinite(values)] for values in curves]
    finite = [values for values in finite if values.size]
    if not finite:
        return
    values = np.concatenate(finite)
    low = float(np.min(values))
    high = float(np.max(values))
    if floor is not None:
        low = min(low, floor)
    if np.isclose(low, high):
        pad = 0.05 * max(abs(low), 1.0)
    else:
        pad = 0.10 * (high - low)
    lower = low - pad
    upper = high + pad
    if floor is not None:
        lower = max(floor, lower)
    ax.set_ylim(lower, upper)


def ensemble_size_convergence(group, output, score_loader, *, repetitions=32):
    """Fit-count convergence at the four exact publication background working points.

    RIDDLE/R-ANODE use their native equal-weight likelihood-ratio ensemble rule.
    LaCATHODE has no cross-fit production ensemble, so its curve is the median
    exact-WP performance of the included classifier fits.  Random fit orderings
    provide a 16--84% subset band without using labels to choose members.

    Saves only publication-style standalone figures, one per exact
    background working point.
    """
    budgets = np.asarray((0.004, 0.01, 0.05, 0.10), dtype=float)
    results = {}
    scenario = next(iter(group.values()))[1].get("scenario") if group else None
    if scenario == "background_only":
        return {"status": "not_applicable_without_signal"}

    for method, source in group.items():
        record, members, aggregation = _ensemble_member_scores(method, source, score_loader, scope="signal_region")
        if members is None or len(members) < 1 or int(np.sum(record["labels"] == 1)) < 1:
            continue
        count = len(members)
        orderings = _fit_orderings(count, repetitions=repetitions)
        samples = np.full((len(orderings), count, len(budgets)), np.nan, dtype=float)

        if method == "lacathode":
            member_metrics = []
            for score in members:
                values = _exact_signal_efficiencies_for_score(record, score, budgets)
                if values is None:
                    member_metrics = []
                    break
                member_metrics.append(values)
            if not member_metrics:
                continue
            member_metrics = np.asarray(member_metrics)
            for r, order in enumerate(orderings):
                for k in range(1, count + 1):
                    samples[r, k - 1] = np.median(member_metrics[order[:k]], axis=0)
        else:
            for r, order in enumerate(orderings):
                for k, combined in _ensemble_progressive_scores(record, members, aggregation, order):
                    values = _exact_signal_efficiencies_for_score(record, combined, budgets)
                    if values is not None:
                        samples[r, k - 1] = values

        low, median, high = np.nanquantile(samples, [.16, .50, .84], axis=0)
        results[method] = dict(
            members=count,
            orderings=len(orderings),
            aggregation=aggregation,
            evaluation_population=population_metadata(record, "signal_region", "signal_region",
                threshold_source="truth-assisted interpolation on this evaluation's background; not a deployment threshold"),
            budgets=budgets.tolist(),
            low=low.tolist(), median=median.tolist(), high=high.tolist(),
        )

    if not results:
        return {"status": "unavailable"}

    destination = output / "07_ensemble_size_convergence"
    individual_destination = destination / "individual"
    individual_destination.mkdir(parents=True, exist_ok=True)
    maximum_members = max(result["members"] for result in results.values())

    for budget_index, budget in enumerate(budgets):
        fig_single, ax = f.plt.subplots(figsize=(5.8, 4.6))
        fig_single.subplots_adjust(left=.15, right=.97, bottom=.15, top=.97)
        for method, result in results.items():
            x = np.arange(1, result["members"] + 1)
            low = np.asarray(result["low"])[:, budget_index]
            median = np.asarray(result["median"])[:, budget_index]
            high = np.asarray(result["high"])[:, budget_index]
            label, color, ls, _ = PHYSICAL_STYLES[method]
            marker = f.method_marker(KEYS[method]) if KEYS[method] in f.METHOD_MARKERS else "o"
            ax.fill_between(x, low, high, color=color, alpha=.14, linewidth=0)
            ax.plot(x, median, color=color, ls=ls, marker=marker, ms=4, label=label)
        ax.set_xlim(left=.75)
        ax.set_ylim(0, 1)
        ax.set_xticks(_convergence_ticks(maximum_members))
        ax.set_xlabel("Number of fits")
        ax.set_ylabel("Signal efficiency")
        ax.minorticks_on()
        ax.legend(frameon=False, loc="best")
        slug = f.format_budget_percent(budget).replace('.', 'p').replace('%', 'pct')
        f.save(fig_single, individual_destination / f"ensemble_size_convergence_{slug}")

    write_json(destination / "ensemble_size_convergence.json", json_safe({
        "status": "available",
        "scope": "signal_region",
        "background_working_points": budgets.tolist(),
        "exact_background_efficiency": True,
        "repetitions": repetitions,
        "methods": results,
        "individual_files": [
            f"individual/ensemble_size_convergence_{f.format_budget_percent(b).replace('.', 'p').replace('%', 'pct')}"
            for b in budgets
        ],
    }))
    return {"status": "available", "scope": "signal_region", "methods": results}


def ensemble_size_convergence_overall(group, output, score_loader, *, scope="signal_region", repetitions=32, min_background=10):
    """Fit-count convergence for overall discrimination metrics.

    Plots how AUC and maximum SIC stabilize as more saved fit members are
    included in the ensemble.  In full-range scope, only methods with saved
    full-range scores contribute.

    Saves only standalone publication-style figures for each metric, each with
    its own legend and no extra text.
    """
    scoped = scope_group(group, scope)
    scenario = next(iter(group.values()))[1].get("scenario") if group else None
    if not scoped:
        return {"status": "no_methods_with_scores_in_scope", "scope": scope}
    if scenario == "background_only":
        return {"status": "not_applicable_without_signal", "scope": scope}

    results = {}
    for method, source in scoped.items():
        record, members, aggregation = _ensemble_member_scores(method, source, score_loader, scope=scope)
        if members is None or len(members) < 1 or int(np.sum(record["labels"] == 1)) < 1:
            continue
        count = len(members)
        orderings = _fit_orderings(count, repetitions=repetitions)
        samples = np.full((len(orderings), count, 2), np.nan, dtype=float)

        if method == "lacathode":
            member_metrics = []
            for score in members:
                values = _overall_metrics_for_score(record, score, min_background=min_background)
                if values is None:
                    member_metrics = []
                    break
                member_metrics.append(values)
            if not member_metrics:
                continue
            member_metrics = np.asarray(member_metrics)
            for r, order in enumerate(orderings):
                for k in range(1, count + 1):
                    samples[r, k - 1] = np.median(member_metrics[order[:k]], axis=0)
        else:
            for r, order in enumerate(orderings):
                for k, combined in _ensemble_progressive_scores(record, members, aggregation, order):
                    values = _overall_metrics_for_score(record, combined, min_background=min_background)
                    if values is not None:
                        samples[r, k - 1] = values

        low, median, high = np.nanquantile(samples, [.16, .50, .84], axis=0)
        results[method] = dict(
            members=count,
            orderings=len(orderings),
            aggregation=aggregation,
            evaluation_population=population_metadata(record, "signal_region" if scope == "signal_region" else "test", scope,
                threshold_source="oracle maximum on this evaluation; not a deployment threshold"),
            metrics=("conditional_auc", "full_pipeline_max_sic"),
            low=low.tolist(),
            median=median.tolist(),
            high=high.tolist(),
        )

    if not results:
        return {"status": "unavailable", "scope": scope}

    destination = output / "07_ensemble_size_convergence"
    individual_destination = destination / "individual"
    individual_destination.mkdir(parents=True, exist_ok=True)
    metric_specs = (
        (0, "AUC", 0.0, "auc"),
        (1, "Maximum significance improvement", 0.0, "maximum_significance_improvement"),
    )
    maximum_members = max(result["members"] for result in results.values())
    for metric_index, ylabel, floor, slug in metric_specs:
        fig_single, ax = f.plt.subplots(figsize=(5.8, 4.6))
        fig_single.subplots_adjust(left=.15, right=.97, bottom=.15, top=.97)
        curves = []
        for method, result in results.items():
            x = np.arange(1, result["members"] + 1)
            low = np.asarray(result["low"])[:, metric_index]
            median = np.asarray(result["median"])[:, metric_index]
            high = np.asarray(result["high"])[:, metric_index]
            label, color, ls, _ = PHYSICAL_STYLES[method]
            marker = f.method_marker(KEYS[method]) if KEYS[method] in f.METHOD_MARKERS else "o"
            ax.fill_between(x, low, high, color=color, alpha=.14, linewidth=0)
            ax.plot(x, median, color=color, ls=ls, marker=marker, ms=4, label=label)
            curves.extend((low, median, high))
        ax.set_xlim(left=.75)
        ax.set_xticks(_convergence_ticks(maximum_members))
        ax.set_xlabel("Number of fits")
        ax.set_ylabel(ylabel)
        ax.minorticks_on()
        _set_convergence_ylim(ax, curves, floor=floor)
        ax.legend(frameon=False, loc="best")
        suffix = "signal_region" if scope == "signal_region" else "full_range"
        f.save(fig_single, individual_destination / f"ensemble_size_convergence_overall_{slug}_{suffix}")

    write_json(destination / "ensemble_size_convergence_overall.json", json_safe({
        "status": "available",
        "scope": scope,
        "minimum_background_count": min_background,
        "repetitions": repetitions,
        "methods": results,
        "individual_files": [
            f"individual/ensemble_size_convergence_overall_{slug}_{'signal_region' if scope == 'signal_region' else 'full_range'}"
            for _, _, _, slug in metric_specs
        ],
    }))
    return {
        "status": "available",
        "scope": scope,
        "minimum_background_count": min_background,
        "methods": results,
    }


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
    common_auc = common_acceptance_auc(records, scope_region(records, scope))
    audit["common_acceptance_auc"] = common_auc
    for method, metrics in audit["methods"].items():
        metrics["evaluation_population"] = population_metadata(records[method], partition, scope,
            threshold_source="oracle test-sample ROC/SIC; not a deployable threshold")
        metrics["common_acceptance_auc"] = common_auc["methods"].get(method)
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
    if scope == "full_region":
        with ProgressStage("physical_no_cut_efficiency", "Plot full-range no-cut background efficiency"):
            render_physical_no_cut_efficiency(records, output, scope=scope)
    with ProgressStage("mass_flatness", f"Plot {label.lower()} background chi-squared"):
        seed = next(iter(group.values()))[1]["seed"]
        variant = result_variant(next(iter(group.values()))[1])
        audit["mass_flatness"] = physical_mass_summary(
            {(scenario, seed, variant): group}, output / "03_mass_sculpting", args, load_scores, scenario=scenario, scope=scope)
    write_json(output / "metrics.json", json_safe(audit))
    return audit


def physical_comparison_summary(groups, output, args, load_scores, *, scope="signal_region", view="comparison"):
    """Aggregate independent runs without crossing scenario, variant or protocol boundaries."""
    groups = {identity: scope_group(group, scope) for identity, group in groups.items() if scope_group(group, scope)}
    partition = "signal_region" if scope == "signal_region" else "test"
    audits = {}
    cohorts = sorted({(identity_parts(identity)[0], identity_parts(identity)[2]) for identity in groups})
    for scenario, variant in cohorts:
        destination = scoped_target(output, scope, scenario, variant=variant) / "summary" / view
        cohort_groups = {
            identity: group for identity, group in groups.items()
            if identity_parts(identity)[0] == scenario and identity_parts(identity)[2] == variant
        }
        partitions = {
            json.dumps(report["contract"]["inputs"].get("files"), sort_keys=True)
            for group in cohort_groups.values()
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
        for identity, group in cohort_groups.items():
            _, seed, _ = identity_parts(identity)
            records = {m: load_scores(root, report, partition) for m, (root, report) in group.items()}
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
                            b, sig, _ = efficiency_curve(record["labels"], score, record["mask"], full_pipeline=True)
                            use = b > 0 if metric == "roc" else (b >= 1e-4) & (
                                np.rint(b * (record["labels"] == 0).sum()) >= args.min_background)
                            native_curves[metric][method].append((b[use], sig[use]))
                            curve_groups[metric][method].append(len(identities[metric][method]))
                            unique, inverse = np.unique(b, return_inverse=True)
                            maxima = np.zeros(len(unique))
                            np.maximum.at(maxima, inverse, sig)
                            values = np.interp(grid, unique, maxima, left=np.nan, right=np.nan)
                            supported = grid >= (0 if metric == "roc" else args.min_background / (record["labels"] == 0).sum())
                            values[~supported] = np.nan
                            fixed_background_curves.append(values)
                        curves[metric][method].append(np.median(fixed_background_curves, axis=0))
                        identities[metric][method].append(dict(
                            seed=seed, fit=run, source=score_variation_source(method, record), member_id=run,
                            run_seed=int(record["run_seeds"][run]) if "run_seeds" in record else seed,
                        ))
        for metric in curves:
            fig, ax, _ = f.canvas(
                "Signal efficiency" if metric == "roc" else
                "Background rejection" if metric == "background_rejection" else "Significance improvement",
                "Signal efficiency" if metric in ("sic_signal", "background_rejection") else "Background efficiency",
            )
            audit = {}
            for method, metric_cohort in curves[metric].items():
                if not metric_cohort:
                    continue
                require_independent_run_ids(identities[metric][method])
                label, color, ls, _ = PHYSICAL_STYLES[method]
                values = np.asarray(metric_cohort) / (1 if metric == "roc" else np.sqrt(grid))
                audit[method] = (
                    f.draw_fit_curves(ax, native_curves[metric][method], metric, label, color, ls,
                                      run_groups=curve_groups[metric][method])
                    if (method == "lacathode" and len(native_curves[metric][method]) > 1)
                    or metric in ("sic_signal", "background_rejection")
                    else f.draw_band(ax, grid, values, label, color, ls)
                )
                audit[method].update(uncertainty_source=fit_uncertainty(identities[metric][method]),
                                     percentiles=[16, 50, 84], fit_count=len(metric_cohort), run_count=len(metric_cohort),
                                     members=identities[metric][method])
                audit[method].pop("independent_runs", None)
            if not audit:
                f.plt.close(fig)
                continue
            ax.plot(grid, grid if metric == "roc" else 1 / grid if metric == "background_rejection" else np.sqrt(grid),
                    color=".5", ls=":", label="Random")
            ax.set(xscale="linear" if metric in ("sic_signal", "background_rejection") else "log",
                   xlim=(0 if metric in ("sic_signal", "background_rejection") else 1e-4, 1),
                   yscale="log" if metric == "background_rejection" else "linear",
                   ylim=(1 if metric == "background_rejection" else 0, 1.02 if metric == "roc" else None))
            title = f.SCENARIO_LABELS[scenario] + " | " + scope_label(scope)
            if variant != "default":
                title += " | " + variant
            f.legend(fig, title=title, ncols=1)
            f.save(fig, destination / "01_performance" / (scope + "_" + metric))
            audits[f"{scenario}/{variant}/{metric}"] = audit
        audits[f"{scenario}/{variant}/mass_flatness"] = physical_mass_summary(
            cohort_groups, destination / "03_mass_sculpting", args, load_scores, scenario=scenario, scope=scope)
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
    for identity, group in groups.items():
        name, seed, _ = identity_parts(identity)
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
                # Aggregate classifier fits within complete runs before uncertainty bands.

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



def render_method_scope_assets(individual, destination, args, score_loader, *, scope):
    """Render scope-specific representation diagnostics and full-range mass scan."""
    method = next(iter(individual))
    key = KEYS[method]
    bundle = make_bundle(individual, args.confidence, score_loader)
    bundle["individual"] = method
    with methods([key], view="."), ProgressStage("representation", f"Plot {scope_label(scope).lower()} inputs and latents"):
        representation = dict(bundle)
        if key in bundle["latents_by_method"]:
            representation["latents"] = bundle["latents_by_method"][key]
        f.render_representation(representation, destination, regions=(("sr",) if scope == "signal_region" else ("full",)))
    if scope == "full_region":
        with methods([key], view="."):
            f.render_mass_scan(bundle, destination)
    return bundle

def refresh_plot_scopes(output, groups, scan_groups):
    """Remove only generated plot artifacts from requested old/new layout scopes."""
    generated_names = {"metrics.json", "injection_scan.json", "injection_scan.csv",
                       "significance_improvement_ratio.json"}

    def clean(scope):
        if not scope.is_dir():
            return
        for path in scope.rglob("*"):
            if path.is_file() and (path.suffix.lower() in (".pdf", ".png") or path.name in generated_names):
                path.unlink(missing_ok=True)
        for path in sorted((p for p in scope.rglob("*") if p.is_dir()), reverse=True):
            if not any(path.iterdir()):
                path.rmdir()
        if scope.is_dir() and not any(scope.iterdir()):
            scope.rmdir()

    scopes = set()
    for identity, group in groups.items():
        scenario, seed, variant = identity_parts(identity)
        for scope in ("signal_region", "full_region"):
            scopes.add(scoped_target(output, scope, scenario, seed, variant))
            scopes.add(scoped_target(output, scope, scenario, variant=variant) / "summary")

        scopes.add(output / scenario / f"seed_{seed:03d}")
        scopes.add(output / scenario / "summary")
    for identity in scan_groups:
        variant = identity[3] if len(identity) > 3 else "default"
        scopes.add(scoped_target(output, "signal_region", "signal_injection", variant=variant) / "injection_scan")
    if groups:
        scopes.add(scope_root(output, "signal_region") / "signal_injection" / "dataset_controls")
    scopes.update((output / "injection_scan", output / "comparison", output / "comparison_with_ranode"))
    for scope in sorted(scopes, key=lambda p: len(p.parts), reverse=True):
        clean(scope)
    (output / "ranode_comparison.json").unlink(missing_ok=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description="Publication figures from completed RIDDLE/LaCATHODE/R-ANODE results; no fitting")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--output", type=Path, default=Path("plots"))
    parser.add_argument("--data", type=Path, default=Path("data/lhco"),
                        help="Prepared LHCO root used only for exact event-count accounting in event_sizes.csv")
    parser.add_argument("--overwrite", action="store_true",
                        help="Replace generated figures in requested scopes, preserving unrelated non-plot files")
    parser.add_argument("--methods", nargs="+", default=None,
                        help="Optional method IDs to include; default: discover every completed method")
    parser.add_argument("--io-workers", type=int, default=2, help="CPU threads for numerical calculations and legacy RIDDLE checkpoint inference")
    parser.add_argument("--device", type=plot_device_argument, default="cpu",
                        help="RIDDLE safeguard/checkpoint inference device: cpu (default), cuda:<index>, or auto")
    parser.add_argument("--no-safeguard-filtering", action="store_true",
                        help="Disable plot-time RIDDLE/R-ANODE fit filtering and use saved ensemble scores")
    parser.add_argument("--plot-workers", type=int, default=min(8, max(1, (os.cpu_count() or 1) // 2)),
                        help="Parallel PDF/PNG export processes (default: half the CPUs, capped at 8)")
    parser.add_argument("--plot-cache-mb", type=int, default=1024,
                        help="Memory limit in MiB for repeated numerical calculations; 0 disables caching")
    parser.add_argument("--confidence", type=float, default=0.95)
    parser.add_argument("--min-background", type=int, default=10)
    parser.add_argument("--random-reference", choices=("bootstrap", "subset"), default="bootstrap")
    parser.add_argument("--allow-smoke", action="store_true", help="Permit synthetic QA fixtures")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1,
                        help="0: warnings/result; 1: compact progress; 2: detailed progress")
    args = parser.parse_args(argv)
    from threadpoolctl import threadpool_limits
    args.population_summary = True
    set_verbosity(args.verbose)
    try:
        if not 0 < args.confidence < 1 or args.min_background < 1 or args.io_workers < 1:
            raise ValueError("Invalid confidence level or background support")
        if args.plot_workers < 1 or args.plot_cache_mb < 0:
            raise ValueError("Plot workers must be positive and plot cache size nonnegative")
        if args.methods is not None:
            args.methods = list(dict.fromkeys(method_family(method) for method in args.methods))
        groups = discover(args.results, args.methods)
        scan_groups = discover(args.results, args.methods, scan=True)
        if not groups and not scan_groups:
            colored_status("No completed results for the requested methods", kind="WARNING")
            return 0
        args.results, args.output = args.results.resolve(), args.output.resolve()
        args.data = args.data.resolve()
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
        requested_groups = {key: dict(group) for key, group in groups.items()}
        requested_scan_groups = {key: dict(group) for key, group in scan_groups.items()}
        colored_status("Validate plot inputs | RIDDLE/R-ANODE safeguard filtering "
                       + ("disabled: use saved ensembles" if args.no_safeguard_filtering
                          else "active: check fits before plotting"), kind="INFO", level=1)
        with threadpool_limits(limits=args.io_workers):
            preflight_scores([*groups.values(), *scan_groups.values()], score_loader, allow_smoke=args.allow_smoke)
        groups = {key: group for key, group in groups.items() if group}
        scan_groups = {key: group for key, group in scan_groups.items() if group}
        args.output.mkdir(parents=True, exist_ok=True)
        scope_root(args.output, "signal_region").mkdir(parents=True, exist_ok=True)
        scope_root(args.output, "full_region").mkdir(parents=True, exist_ok=True)
        rows, settings = [], []
        comparisons = {}
        available = set().union(*(set(g) for g in [*groups.values(), *scan_groups.values()]))
        progress = PlotProgress(plot_task_count(groups, scan_groups))
        colored_status(f"Plot plan: {progress.total} task groups | {len(groups)} scenario/seed/variant groups | "
                       f"{len(scan_groups)} injection-scan points | {args.plot_workers} export workers | "
                       f"{args.plot_cache_mb} MiB calculation cache | saved results only, no training",
                       kind="INFO", level=1)
        with local_progress("Plots", display=progress if args.verbose < 2 else None), \
                f.plt.style.context(f.STYLE), threadpool_limits(limits=args.io_workers), locked(args.output / ".plot.lock"), \
                f.plot_resources(workers=args.plot_workers, cache_mb=args.plot_cache_mb) as exporter:
            if args.overwrite:
                refresh_plot_scopes(args.output, requested_groups, requested_scan_groups)
            scan_audit = {}
            if scan_groups:
                with progress.task("Injection scan | discrimination and significance"):
                    variants = sorted({identity[3] if len(identity) > 3 else "default" for identity in scan_groups})
                    for variant in variants:
                        variant_groups = {identity: group for identity, group in scan_groups.items()
                                          if (identity[3] if len(identity) > 3 else "default") == variant}
                        scan_target = scoped_target(args.output, "signal_region", "signal_injection", variant=variant) / "injection_scan"
                        scan_audit[variant] = render_injection_scan(variant_groups, scan_target / "comparison", args, score_loader)
                        for method in sorted(available):
                            cohort = {identity: {method: group[method]} for identity, group in variant_groups.items() if method in group}
                            if cohort:
                                render_injection_scan(cohort, scan_target / METHOD_SPECS[method].label, args, score_loader)
            for identity, group in groups.items():
                scenario, seed, variant = identity_parts(identity)
                if any(r["contract"].get("inputs", {}).get("synthetic_smoke_fixture") for _, r in group.values()) and not args.allow_smoke:
                    raise ValueError("Synthetic fixtures require --allow-smoke for QA")
                missing = set(args.methods or available) - set(group)
                label = plot_group_label(scenario, seed, group, variant)
                if missing:
                    names = ", ".join(METHOD_SPECS[m].label if m in METHOD_SPECS else m for m in sorted(missing))
                    colored_status(f"{label} | no completed results for: {names}; plotting available methods",
                                   kind="INFO", level=1)
                sr_target = scoped_target(args.output, "signal_region", scenario, seed, variant)
                full_target = scoped_target(args.output, "full_region", scenario, seed, variant)
                with progress.task(f"{label} | shared comparison plots"):
                    comparison = render_physical_comparison(group, sr_target / "comparison", args, score_loader, scope="signal_region")
                    comparison_training_figures(group, sr_target / "comparison", score_loader)
                    comparison["ensemble_size_convergence"] = ensemble_size_convergence(
                        group, sr_target / "comparison", score_loader
                    )
                    comparison["ensemble_size_convergence_overall"] = ensemble_size_convergence_overall(
                        group, sr_target / "comparison", score_loader,
                        scope="signal_region", min_background=args.min_background,
                    )
                    full_comparison = render_physical_comparison(group, full_target / "comparison", args, score_loader, scope="full_region")
                    full_comparison["ensemble_size_convergence_overall"] = ensemble_size_convergence_overall(
                        group, full_target / "comparison", score_loader,
                        scope="full_region", min_background=args.min_background,
                    )
                    comparison["full_region"] = full_comparison
                comparisons[f"{scenario}/{variant}/{seed}"] = comparison
                settings.extend(settings_rows(group, score_loader))
                rows.extend(comparison_rows(comparison, scenario, seed, variant=variant))
                rows.extend(comparison_rows(full_comparison, scenario, seed, scope="full_region", variant=variant))
                for method, source in group.items():
                    with progress.task(f"{label} | {METHOD_SPECS[method].label} individual plots"):
                        individual = {method: source}
                        sr_destination = sr_target / METHOD_SPECS[method].label
                        render_physical_comparison(individual, sr_destination, args, score_loader, scope="signal_region")
                        render_method_scope_assets(individual, sr_destination, args, score_loader, scope="signal_region")
                        training_figures(individual, sr_target, score_loader)
                        ensemble_size_convergence(individual, sr_destination, score_loader)
                        ensemble_size_convergence_overall(
                            individual, sr_destination, score_loader,
                            scope="signal_region", min_background=args.min_background,
                        )
                        if scope_group(individual, "full_region"):
                            full_destination = full_target / METHOD_SPECS[method].label
                            render_physical_comparison(individual, full_destination, args, score_loader, scope="full_region")
                            render_method_scope_assets(individual, full_destination, args, score_loader, scope="full_region")
                            ensemble_size_convergence_overall(
                                individual, full_destination, score_loader,
                                scope="full_region", min_background=args.min_background,
                            )
            variant_audit = render_variant_sic_ratio(groups, args.output, args, score_loader)
            audit = {"dataset_variant_ratio": variant_audit}
            cohorts = summary_groups(groups)
            if cohorts:
                with progress.task("Summary plots | SIC and mass-flatness uncertainty bands"):
                    audit["comparison"] = physical_comparison_summary(cohorts, args.output, args, score_loader)
                    full_cohorts = summary_groups(groups, scope="full_region")
                    if full_cohorts:
                        audit["comparison"]["full_region"] = physical_comparison_summary(
                            full_cohorts, args.output, args, score_loader, scope="full_region")
                    for method in sorted(available):
                        cohort = summary_groups(groups, method=method)
                        if not cohort:
                            continue
                        view = METHOD_SPECS[method].label
                        audit[method] = physical_comparison_summary(cohort, args.output, args, score_loader, view=view)
                        full_individual = summary_groups(groups, method=method, scope="full_region")
                        if full_individual:
                            audit[method]["full_region"] = physical_comparison_summary(
                                full_individual, args.output, args, score_loader, scope="full_region", view=view)
            with progress.task("Export comparison tables, event accounting and plot manifest"):
                exporter.flush()
                event_rows = event_size_rows(groups, score_loader, data_root=args.data)
                for filename, table in (("comparison.csv", rows), ("configuration.csv", settings),
                                        ("event_sizes.csv", event_rows)):
                    if table:
                        csv_write(args.output / filename, table)
                    else:
                        (args.output / filename).unlink(missing_ok=True)
                write_json(args.output / "comparison.json", json_safe(comparisons))
                write_json(args.output / "plot_manifest.json", json_safe({
                    "schema": 5,
                    "methods": {m: dict(label=METHOD_SPECS[m].label, score_transform=METHOD_SPECS[m].score_transform,
                                        score_scopes=sorted({
                                            riddle_plot_spec(g[m][1]).score_scope if m == "riddle" else METHOD_SPECS[m].score_scope
                                            for g in [*groups.values(), *scan_groups.values()] if m in g
                                        })) for m in sorted(available)},
                    "layout": "SR-Only/<scenario>[/variant_<name>]/{seed_<seed>,summary}/... and Full-Range/<scenario>[/variant_<name>]/{seed_<seed>,summary}/...; injection scans live below SR-Only/signal_injection/injection_scan; root CSV/JSON files are publication metadata, not plots.",
                    "publication": "Physical Review D-oriented vector PDF plus 600-dpi PNG; one plot per figure file except the multipage mass_cut_scan.pdf companion, which also retains every cut as an individual figure.",
                    "comparison": "All available methods in their supported scope; independent acceptance and uncut denominators. RIDDLE is included in Full-Range using its saved score mask, so intentionally unscored sideband events remain rejected; R-ANODE remains SR-only.",
                    "uncertainty": audit,
                    "cuts": "Publication working points at B=0.4%, 1.0%, 5.0% and 10.0% use exact truth-assisted test-background ROC interpolation with a fractional boundary tie applied identically to signal; no nearby empirical rank is reported as exact.",
                    "injection_scan": scan_audit,
                    "event_sizes": "event_sizes.csv uses checksum-protected result ledgers and exact prepared-array row counts when the matching --data manifest is available; unavailable counts are blank rather than inferred.",
                    "mass_cut_scan": {
                        "thresholds": list(f.SCORE_CUTS), "comparison": "strict score > threshold",
                        "scope": "full physical test mass range", "panels_per_page": 6,
                        "individual_directory": "04_mass_cuts/individual_cuts",
                        "histograms": "unweighted counts per fit, averaged over LaCathode fits; identical bins, no smoothing or pooled events"},
                    "mass_summary": "Common background in each plotted region: 300 equal-occupancy bins, test-derived cuts with uncut denominators; shape chi2 normalized by achieved efficiency.",
                    "uncertainty_scope": "16/50/84 percentiles across independent full method runs on fixed evaluation events; protocols and dataset variants are never combined into one band.",
                    "score_inputs": ("Legacy RIDDLE checkpoints evaluated on reserved validation; accepted scores reconstructed when needed. No training or modification of results."
                        if any(a.get("checkpoint_inference") for a in score_loader.fit_audit.values())
                        else "Saved predictions only; no checkpoint inference or modification of results"),
                    "safeguard_filtering": not args.no_safeguard_filtering,
                    "requested_inference_device": args.device,
                    "checkpoint_inference_devices": sorted({a["inference_device"] for a in score_loader.fit_audit.values() if a.get("checkpoint_inference")}),
                    "score_ensembles": list(score_loader.fit_audit.values()),
                    "ensemble_size_convergence": "SR-only diagnostic at exact B=0.4%, 1.0%, 5.0%, 10.0%; RIDDLE/R-ANODE use native likelihood-ratio aggregation, LaCATHODE uses median per-fit performance because its pinned production protocol does not cross-fit-average scores.",
                    "ensemble_size_convergence_overall": "Overall fit-count convergence for conditional AUC and full-pipeline maximum SIC. Produced in SR-Only for all methods with signal, and in Full-Range for methods with saved full-range scores.",
                    "summary_axes": f.SUMMARY_AXES, "publication_y_ranges": f.PUBLICATION_Y_RANGES,
                }))
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
