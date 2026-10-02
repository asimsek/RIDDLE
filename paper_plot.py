import argparse
import csv
import json
import math
import re
import shutil
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import yaml
from scipy.special import ndtri
from sklearn.metrics import roc_curve

from riddle.evaluation import common_acceptance_auc, riddle_score_scope
from riddle.metrics import efficiency_curve, oracle_metrics
from riddle.plotting import ScoreLoader, discover, event_size_rows, exact_background_selection, method_family, read_metadata, result_variant, scientific_protocol, settings_rows, verify_plot_input
from riddle.figures import equal_occupancy, shape_chi2
from riddle.stein_scoring import conditional_gaussianize
from riddle.storage import file_digest

METHOD_ORDER = ("riddle", "iad", "supervised", "lacathode", "ranode")
METHOD_ALIASES = {"riddle": "riddle", "riddlev2": "riddle", "riddlev3": "riddle", "iad": "iad", "idealized": "iad", "idealized_ad": "iad", "supervised": "supervised", "supervised_ad": "supervised", "lacathode": "lacathode", "ranode": "ranode"}
METHOD_LABELS = {"riddle": "RIDDLE", "iad": "Idealized AD", "supervised": "Supervised AD", "lacathode": "LaCATHODE", "ranode": "R-ANODE"}
METHOD_COLORS = {"riddle": "#D55E00", "iad": "#56B4E9", "supervised": "#009E73", "lacathode": "#0072B2", "ranode": "#8B1A1A"}
METHOD_DARK = {"riddle": "#9E4500", "iad": "#0072B2", "supervised": "#00543e", "lacathode": "#004C78", "ranode": "#5F1111"}
METHOD_LIGHT = {"riddle": "#F0A06A", "iad": "#78cdf5", "supervised": "#46eba2", "lacathode": "#6AB1D6", "ranode": "#C47777"}
METHOD_LINES = {"riddle": "-", "iad": "-", "supervised": "-", "lacathode": "-", "ranode": "-"}
METHOD_MARKERS = {"riddle": "o", "iad": "s", "supervised": "^", "lacathode": "D", "ranode": "v"}
VARIANT_ORDER = ("default", "deltaR", "shifted")
VARIANT_LABELS = {"default": "Default", "shifted": "Shifted", "deltaR": "DeltaR"}
SCENARIO_LABELS = {"signal_injection": "Signal-Injected", "background_only": "BG-Only"}
REGION_LABELS = {"signal_region": "Signal region", "full_region": "Full region"}
WORKING_POINTS = (0.004, 0.005, 0.01, 0.05, 0.10)
PLOT_WORKING_POINTS = (0.005, 0.01, 0.05, 0.10)
WORKING_POINT_LABELS = {0.004: "0.4%", 0.005: "0.5%", 0.01: "1%", 0.05: "5%", 0.10: "10%"}
FEATURE_LABELS = {"m1": r"$m_1$ (TeV)", "delta_m": r"$\Delta m$ (TeV)", "tau21_j1": r"$\tau_{21}^{J_1}$", "tau21_j2": r"$\tau_{21}^{J_2}$", "deltaR": r"$\Delta R_{jj}$"}
FEATURE_NAMES = {"default": ("m1", "delta_m", "tau21_j1", "tau21_j2"), "shifted": ("m1", "delta_m", "tau21_j1", "tau21_j2"), "deltaR": ("m1", "delta_m", "tau21_j1", "tau21_j2", "deltaR")}
SINGLE_COLUMN_IN = 8.6 / 2.54
DOUBLE_COLUMN_IN = 17.6 / 2.54
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans", "text.usetex": False, "font.size": 8.2, "axes.labelsize": 8.8, "axes.linewidth": 0.8, "xtick.labelsize": 7.7, "ytick.labelsize": 7.7, "legend.fontsize": 7.3, "legend.title_fontsize": 7.3, "lines.linewidth": 1.35, "lines.markersize": 3.2, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "savefig.facecolor": "white", "savefig.edgecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white"}


def say(message, verbose=1, level=1):
    if verbose >= level:
        print(message, flush=True)


def canonical_method(value):
    return METHOD_ALIASES.get(value, value)


def safe_component(value):
    text = str(value).strip().replace(" ", "-")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-.")
    return text or "item"


def folder_variant(variant):
    return VARIANT_LABELS.get(variant, safe_component(variant))


def folder_region(region):
    return "Signal-Region" if region == "signal_region" else "Full-Region"


def figure_path(base, scenario, variant, region):
    return base / "01_comparison" / SCENARIO_LABELS[scenario] / folder_variant(variant) / folder_region(region)


def configure_style():
    matplotlib.rcParams.update(STYLE)


def display_score_values(method, values):
    values = np.asarray(values, float)
    finite = values[np.isfinite(values)]
    if method == "ranode" and len(finite) and (np.min(finite) < 0.0 or np.max(finite) > 1.0):
        clipped = np.clip(values, -60.0, 60.0)
        return 1.0 / (1.0 + np.exp(-clipped))
    return values


def publication_axis_label(value):
    if not isinstance(value, str) or "\n" in value or "$" in value or len(value) <= 34:
        return value
    words = value.split()
    if len(words) < 2:
        return value
    best = None
    for index in range(1, len(words)):
        left = " ".join(words[:index])
        right = " ".join(words[index:])
        score = abs(len(left) - len(right))
        if best is None or score < best[0]:
            best = (score, left, right)
    return best[1] + "\n" + best[2]


def new_figure(xlabel, ylabel, wide=False):
    width = DOUBLE_COLUMN_IN if wide else SINGLE_COLUMN_IN
    height = 0.78 * width
    fig, ax = plt.subplots(figsize=(width, height))
    fig.subplots_adjust(left=0.19 if not wide else 0.12, right=0.97, bottom=0.18, top=0.96)
    ax.set_xlabel(publication_axis_label(xlabel))
    ax.set_ylabel(publication_axis_label(ylabel))
    ax.minorticks_on()
    return fig, ax


def legend_overlap_score(ax, legend):
    renderer = ax.figure.canvas.get_renderer()
    box = legend.get_window_extent(renderer=renderer)
    score = 0.0
    for line in ax.lines:
        if not line.get_visible() or line.get_label() == "_nolegend_":
            continue
        path = line.get_path().transformed(line.get_transform())
        vertices = path.vertices
        if len(vertices):
            inside = (vertices[:, 0] >= box.x0) & (vertices[:, 0] <= box.x1) & (vertices[:, 1] >= box.y0) & (vertices[:, 1] <= box.y1)
            score += 3.0 * float(np.sum(inside)) / max(1, len(vertices))
    for collection in ax.collections:
        if not collection.get_visible():
            continue
        try:
            bbox = collection.get_datalim(ax.transData).transformed(ax.transData)
            if box.overlaps(bbox):
                intersection = box.intersection(bbox)
                if intersection is not None:
                    score += intersection.width * intersection.height / max(1.0, box.width * box.height)
        except Exception:
            pass
    for patch in ax.patches:
        if isinstance(patch, Rectangle) and patch is ax.patch:
            continue
        try:
            bbox = patch.get_window_extent(renderer=renderer)
            if box.overlaps(bbox):
                intersection = box.intersection(bbox)
                if intersection is not None:
                    score += intersection.width * intersection.height / max(1.0, box.width * box.height)
        except Exception:
            pass
    return score


def inside_legend(ax, title=None, ncol=None, fontsize=None, borderaxespad=0.5, allow_headroom=True):
    handles, labels = ax.get_legend_handles_labels()
    pairs = []
    seen = set()
    for handle, label in zip(handles, labels):
        if not label or label.startswith("_") or label in seen:
            continue
        seen.add(label)
        pairs.append((handle, label))
    if not pairs:
        return None
    handles, labels = zip(*pairs)
    candidates = ("upper right", "upper left", "lower right", "lower left", "center right", "center left", "upper center", "lower center")
    columns = ncol or (2 if len(labels) >= 5 else 1)
    font = fontsize or STYLE["legend.fontsize"]
    best = None
    for location in candidates:
        legend = ax.legend(handles, labels, loc=location, frameon=True, framealpha=0.93, borderpad=0.42, labelspacing=0.32, handlelength=1.9, columnspacing=0.8, ncol=columns, title=title, fontsize=font, borderaxespad=borderaxespad)
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        box = legend.get_window_extent(renderer=renderer)
        axes_box = ax.get_window_extent(renderer=renderer)
        score = legend_overlap_score(ax, legend)
        if not (box.x0 >= axes_box.x0 and box.x1 <= axes_box.x1 and box.y0 >= axes_box.y0 and box.y1 <= axes_box.y1):
            score += 1000
        if best is None or score < best[0]:
            best = (score, location)
        legend.remove()
    score, location = best
    if score > 2.5 and allow_headroom:
        add_y_headroom(ax, 0.12)
    legend = ax.legend(handles, labels, loc=location, frameon=True, framealpha=0.93, borderpad=0.42, labelspacing=0.32, handlelength=1.9, columnspacing=0.8, ncol=columns, title=title, fontsize=max(6.7, font - 0.2 if score > 3.5 else font), borderaxespad=borderaxespad)
    ax.figure.canvas.draw()
    return legend


def add_y_headroom(ax, fraction=0.15):
    ymin, ymax = ax.get_ylim()
    if not np.isfinite(ymin) or not np.isfinite(ymax) or ymax <= ymin:
        return
    if ax.get_yscale() == "log":
        if ymin > 0:
            ax.set_ylim(ymin, ymax * (ymax / ymin) ** fraction)
    else:
        ax.set_ylim(ymin, ymax + fraction * (ymax - ymin))


def probability_contour_levels(histogram, fractions=(0.95, 0.68, 0.50)):
    values = np.asarray(histogram, float)
    positive = values[np.isfinite(values) & (values > 0)]
    if not len(positive):
        return np.asarray([], float)
    ordered = np.sort(positive)[::-1]
    cumulative = np.cumsum(ordered)
    total = cumulative[-1]
    thresholds = []
    for fraction in fractions:
        index = int(np.searchsorted(cumulative, float(fraction) * total, side="left"))
        index = min(index, len(ordered) - 1)
        thresholds.append(ordered[index])
    return np.unique(np.sort(np.asarray(thresholds, float)))


def white_log_density_mesh(ax, xedges, yedges, histogram, colorbar_label):
    values = np.asarray(histogram, float)
    positive = values[np.isfinite(values) & (values > 0)]
    if not len(positive):
        return None
    vmin = float(np.min(positive))
    vmax = float(np.max(positive))
    if vmax <= vmin:
        vmax = vmin * (1.0 + 1e-9)
    cmap = plt.get_cmap("viridis").copy()
    cmap.set_bad("white")
    cmap.set_under("white")
    masked = np.ma.masked_less_equal(values.T, 0)
    artist = ax.pcolormesh(xedges, yedges, masked, shading="auto", rasterized=True, cmap=cmap, norm=LogNorm(vmin=vmin, vmax=vmax))
    colorbar = ax.figure.colorbar(artist, ax=ax, pad=0.03)
    colorbar.set_label(colorbar_label)
    return artist


def cleanup_legacy_summary_folders(output):
    output = Path(output)
    root = output / "01_comparison" / "Signal-Injected"
    for name in ("Signal-region", "Full-region", "Dataset-Variant-Summary"):
        path = root / name
        if path.is_dir():
            shutil.rmtree(path)
    if output.is_dir():
        full_region_paths = sorted((path for path in output.rglob("Full-Region") if path.is_dir()), key=lambda path: len(path.parts), reverse=True)
        for path in full_region_paths:
            shutil.rmtree(path)
        mass_scan = output / "04_mass_scan"
        if mass_scan.is_dir():
            shutil.rmtree(mass_scan)
        for path in output.rglob("*0p4*"):
            if path.is_file() and path.suffix.lower() in (".png", ".pdf"):
                path.unlink()


def audit_layout(fig, ax, legend=None):
    if ax.get_title() or fig._suptitle is not None:
        raise RuntimeError("Publication figure contains a main title")
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_box = fig.bbox
    for artist in (ax.xaxis.label, ax.yaxis.label):
        box = artist.get_window_extent(renderer=renderer)
        if box.x0 < fig_box.x0 - 2 or box.x1 > fig_box.x1 + 2 or box.y0 < fig_box.y0 - 2 or box.y1 > fig_box.y1 + 2:
            raise RuntimeError("Axis label is clipped")
    if legend is not None:
        box = legend.get_window_extent(renderer=renderer)
        axes_box = ax.get_window_extent(renderer=renderer)
        if not (box.x0 >= axes_box.x0 - 1 and box.x1 <= axes_box.x1 + 1 and box.y0 >= axes_box.y0 - 1 and box.y1 <= axes_box.y1 + 1):
            raise RuntimeError("Legend is not fully inside plotting area")
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()
    if not np.isfinite(xlim).all() or not np.isfinite(ylim).all():
        raise RuntimeError("Figure has invalid axis limits")


def save_figure(fig, ax, stem, formats, overwrite, legend=None):
    stem.parent.mkdir(parents=True, exist_ok=True)
    try:
        fig.tight_layout(pad=0.45)
    except Exception:
        pass
    audit_layout(fig, ax, legend)
    for extension in formats:
        path = stem.with_suffix("." + extension)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {path}")
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if extension == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)
    plt.close(fig)


def verify_requested_artifact(root, report, relative):
    relative = str(relative)
    artifacts = report.get("artifacts_sha256", {})
    if relative not in artifacts:
        return None
    return verify_plot_input(root, report, relative)


def select_requested_methods(all_groups, requested):
    available = set()
    for group in all_groups.values():
        available.update(canonical_method(method) for method in group)
    if requested:
        normalized = []
        for item in requested:
            if item == "all":
                return [item for item in METHOD_ORDER if item in available] + sorted(available - set(METHOD_ORDER))
            method = canonical_method(item)
            if method not in normalized:
                normalized.append(method)
        missing = [method for method in normalized if method not in available]
        if missing:
            labels = ", ".join(METHOD_LABELS.get(method, method) for method in missing)
            raise ValueError(f"Explicitly requested publication methods were not discovered: {labels}")
        return normalized
    return [item for item in ("riddle", "iad", "supervised") if item in available]


def filter_groups(groups, methods, variants, scenarios, excluded_seeds):
    selected = {}
    for identity, group in groups.items():
        scenario, seed, variant = identity
        if seed in excluded_seeds or scenario not in scenarios or variant not in variants:
            continue
        normalized = {}
        for method, source in group.items():
            family = canonical_method(method)
            if family in methods:
                normalized[family] = source
        if normalized:
            selected[(scenario, seed, variant)] = normalized
    return selected


def score_scope(method, report):
    if method in ("riddle", "iad", "supervised"):
        return riddle_score_scope(report)
    if method == "ranode":
        return "signal_region"
    value = report.get("score_scope", report.get("plotting", {}).get("score_scope", "full_region"))
    return value if value in ("signal_region", "full_region") else "signal_region"


def group_protocol_key(method, report):
    inputs = report.get("contract", {}).get("inputs", {})
    return method, scientific_protocol(report), result_variant(report), report.get("scenario"), json.dumps(inputs.get("files", None), sort_keys=True)


def aggregation_protocol_signature(method, report):
    contract = report.get("contract", {})
    settings = contract.get("settings", {})
    if method in ("riddle", "iad", "supervised"):
        payload = {
            "scientific_version": contract.get("scientific_version"),
            "riddle": settings.get("riddle"),
            "background": settings.get("background"),
            "inputs": settings.get("inputs"),
            "oracle_benchmark": contract.get("oracle_benchmark") if method in ("iad", "supervised") else None,
            "score_scope": score_scope(method, report),
        }
    else:
        volatile = {"seed", "scenario", "device", "run_index", "campaign_seed"}
        payload = {
            "scientific_version": contract.get("scientific_version"),
            "settings": {key: value for key, value in settings.items() if key not in volatile},
            "score_scope": score_scope(method, report),
        }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)


def require_riddle_benchmark_alignment(group):
    native = {}
    for method in ("riddle", "iad", "supervised"):
        if method not in group:
            continue
        report = group[method][1]
        contract = report.get("contract", {})
        settings = contract.get("settings", {})
        riddle_settings = settings.get("riddle")
        background_settings = settings.get("background")
        inputs_settings = settings.get("inputs")
        if not isinstance(riddle_settings, dict) or not isinstance(background_settings, dict) or not isinstance(inputs_settings, dict):
            raise ValueError(f"{METHOD_LABELS[method]} is missing its RIDDLE configuration")
        if method in ("iad", "supervised"):
            oracle = contract.get("oracle_benchmark")
            if (report.get("benchmark_protocol") != "riddle_oracle" or not isinstance(oracle, dict)
                    or oracle.get("method") != method or oracle.get("core") != "riddle_stein_witness"):
                raise ValueError(f"{METHOD_LABELS[method]} is not a valid RIDDLE oracle benchmark result")
        native[method] = {"riddle": riddle_settings, "background": background_settings, "inputs": inputs_settings}
    if not native:
        return
    configurations = {json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) for value in native.values()}
    if len(configurations) > 1:
        raise ValueError("RIDDLE, Idealized AD, and Supervised AD do not share one RIDDLE configuration")


def load_record(loader, method, root, report, region):
    if region == "full_region" and score_scope(method, report) != "full_region":
        return None
    partition = "signal_region" if region == "signal_region" else "test"
    record = loader(root, report, partition)
    if method in ("riddle", "iad", "supervised"):
        record["paper_mass_conditioning"] = bool(report.get("contract", {}).get("settings", {}).get("riddle", {}).get("mass_conditioning", False))
    if region == "signal_region":
        if "is_signal_region" in record and not np.asarray(record["is_signal_region"]).all():
            region_mask = np.asarray(record["is_signal_region"], bool)
            record = slice_record(record, region_mask)
    return record


def slice_record(record, selection):
    selection = np.asarray(selection, bool)
    result = {}
    n = len(selection)
    for key, value in record.items():
        array = np.asarray(value) if isinstance(value, np.ndarray) else value
        if isinstance(array, np.ndarray) and array.ndim >= 1 and array.shape[-1] == n and key in ("fit_scores",):
            result[key] = array[..., selection]
        elif isinstance(array, np.ndarray) and array.ndim >= 1 and len(array) == n:
            result[key] = array[selection]
        else:
            result[key] = value
    return result


def require_population_compatibility(records):
    if len(records) < 2:
        return
    first = next(iter(records.values()))
    for other in list(records.values())[1:]:
        for key in ("mass", "labels", "physical"):
            if key in first and key in other and not np.array_equal(first[key], other[key]):
                raise ValueError("Methods do not share the same ordered physical evaluation population")
        if "event_ids" in first and "event_ids" in other and not np.array_equal(first["event_ids"], other["event_ids"]):
            raise ValueError("Methods do not share the same event identities")


def full_pipeline_curve(record):
    labels = np.asarray(record["labels"])
    scores = np.asarray(record["scores"])
    mask = np.asarray(record["mask"], bool)
    b, s, cuts = efficiency_curve(labels, scores, mask, full_pipeline=True)
    return np.asarray(b), np.asarray(s), np.asarray(cuts)


def central_metrics(record, min_background):
    labels = np.asarray(record["labels"])
    if len(np.unique(labels)) < 2:
        return None
    metrics = oracle_metrics(labels, record["scores"], record["mask"], min_background=min_background, min_efficiency=1e-4)
    b, s, cuts = full_pipeline_curve(record)
    supported = (b >= 1e-4) & (np.rint(b * np.sum(labels == 0)) >= min_background)
    wp = {}
    for budget in WORKING_POINTS:
        selection = exact_background_selection(record, budget)
        wp[budget] = None if selection is None else selection["signal_efficiency"]
    return {"auc": metrics["conditional_auc"], "max_sic": metrics["full_pipeline_max_sic"], "background_efficiency": b, "signal_efficiency": s, "cuts": cuts, "supported": supported, "working_points": wp, "acceptance": metrics["acceptance"]}


def interpolate_curve(x, y, grid):
    x = np.asarray(x, float)
    y = np.asarray(y, float)
    good = np.isfinite(x) & np.isfinite(y) & (x > 0)
    x, y = x[good], y[good]
    if len(x) < 2:
        return np.full_like(grid, np.nan, dtype=float)
    order = np.argsort(x)
    x, y = x[order], y[order]
    unique, indices = np.unique(x, return_index=True)
    values = y[indices]
    result = np.full_like(grid, np.nan, dtype=float)
    inside = (grid >= unique.min()) & (grid <= unique.max())
    result[inside] = np.interp(np.log10(grid[inside]), np.log10(unique), values)
    return result



def finite_column_percentiles(matrix, percentiles):
    matrix = np.asarray(matrix, float)
    outputs = [np.full(matrix.shape[1], np.nan) for _ in percentiles]
    supported = np.isfinite(matrix).any(axis=0)
    if np.any(supported):
        values = np.nanpercentile(matrix[:, supported], percentiles, axis=0)
        for output, row in zip(outputs, np.atleast_2d(values)):
            output[supported] = row
    return outputs

def aggregate_metric_runs(run_rows, curve_kind):
    if not run_rows:
        return None
    grid = np.geomspace(1e-4, 1, 350)
    curves = []
    for row in run_rows:
        b = row["metrics"]["background_efficiency"]
        s = row["metrics"]["signal_efficiency"]
        if curve_kind == "sic":
            y = np.divide(s, np.sqrt(b), out=np.full_like(s, np.nan), where=b > 0)
        elif curve_kind == "roc":
            y = s
        else:
            raise ValueError(curve_kind)
        curves.append(interpolate_curve(b, y, grid))
    matrix = np.asarray(curves, float)
    median = np.full(len(grid), np.nan)
    low = np.full(len(grid), np.nan)
    high = np.full(len(grid), np.nan)
    supported = np.isfinite(matrix).any(axis=0)
    if np.any(supported):
        median[supported] = np.nanpercentile(matrix[:, supported], 50, axis=0)
        low[supported] = np.nanpercentile(matrix[:, supported], 16, axis=0)
        high[supported] = np.nanpercentile(matrix[:, supported], 84, axis=0)
    return grid, median, low, high


def metric_cache(groups, loader, regions, min_background, verbose, require_compatible_populations=False):
    cache = {}
    records = {}
    protocol_groups = defaultdict(set)
    total = sum(len(group) * len(regions) for group in groups.values())
    index = 0
    for (scenario, seed, variant), group in sorted(groups.items()):
        require_riddle_benchmark_alignment(group)
        for method, (root, report) in group.items():
            for region in regions:
                index += 1
                say(f"[{index}/{total}] {METHOD_LABELS.get(method, method)} {scenario} {variant} seed {seed} {region}", verbose, 2)
                record = load_record(loader, method, root, report, region)
                if record is None:
                    if method == "riddle" and region == "full_region":
                        say("[SKIP] RIDDLE Full Region plots: saved scores are Signal Region only", verbose)
                    continue
                key = (method, scenario, variant, region, seed)
                records[key] = record
                protocol_groups[(method, scenario, variant, region)].add(aggregation_protocol_signature(method, report))
                if scenario == "signal_injection":
                    cache[key] = central_metrics(record, min_background)
                else:
                    labels = np.asarray(record["labels"])
                    cache[key] = {"auc": None, "max_sic": None, "working_points": {}, "acceptance": {"background": {"total": int(np.sum(labels == 0)), "mapped": int(np.sum((labels == 0) & np.asarray(record["mask"], bool))), "acceptance": float(np.mean(np.asarray(record["mask"], bool)[labels == 0])) if np.any(labels == 0) else None}}}
    for (method, scenario, variant, region), signatures in protocol_groups.items():
        if len(signatures) != 1:
            raise ValueError(f"Incompatible scientific protocols across independent runs for {METHOD_LABELS.get(method, method)}, {scenario}, {variant}, {region}")
    if require_compatible_populations:
        population_groups = defaultdict(dict)
        for (method, scenario, variant, region, seed), record in records.items():
            population_groups[(scenario, variant, region, seed)][method] = record
        for (scenario, variant, region, seed), population in population_groups.items():
            try:
                require_population_compatibility(population)
            except ValueError as error:
                raise ValueError(
                    f"Incompatible cross-method evaluation population for {scenario}, {variant}, {region}, seed {seed}"
                ) from error
    return cache, records


def compatible_run_rows(cache, method, scenario, variant, region, seeds=None):
    allowed = None if seeds is None else set(seeds)
    rows = []
    for key, metrics in cache.items():
        m, s, v, r, seed = key
        if ((m, s, v, r) == (method, scenario, variant, region) and metrics is not None
                and (allowed is None or seed in allowed)):
            rows.append({"seed": seed, "metrics": metrics})
    return sorted(rows, key=lambda row: row["seed"])


def comparison_seed_set(cache, methods, scenario, variant, region):
    populations = [
        {row["seed"] for row in compatible_run_rows(cache, method, scenario, variant, region)}
        for method in methods
    ]
    populations = [values for values in populations if values]
    if not populations:
        return []
    if len(methods) > 1 and len(populations) != len(methods):
        return []
    return sorted(set.intersection(*populations))


def comparison_seed_set_across_variants(cache, methods, scenario, variants, region):
    populations = []
    for method in methods:
        for variant in variants:
            values = {row["seed"] for row in compatible_run_rows(cache, method, scenario, variant, region)}
            if not values:
                return []
            populations.append(values)
    return sorted(set.intersection(*populations)) if populations else []


def comparison_methods(cache, scenario, variant, region):
    return [method for method in METHOD_ORDER if any(key[:4] == (method, scenario, variant, region) for key in cache)] + sorted({key[0] for key in cache if key[1:4] == (scenario, variant, region)} - set(METHOD_ORDER))


def plot_performance(cache, output, formats, overwrite, scenarios, variants, regions):
    for scenario in scenarios:
        if scenario != "signal_injection":
            continue
        for variant in variants:
            for region in regions:
                methods = comparison_methods(cache, scenario, variant, region)
                if not methods:
                    continue
                common_seeds = comparison_seed_set(cache, methods, scenario, variant, region)
                if len(methods) > 1 and not common_seeds:
                    say(f"[SKIP] {scenario} {variant} {region}: selected methods have no common independent seeds", 1)
                    continue
                destination = figure_path(output, scenario, variant, region)
                title = REGION_LABELS[region] + " · " + VARIANT_LABELS.get(variant, variant)
                for kind, ylabel, stem in (("sic", r"Significance improvement, $\epsilon_S/\sqrt{\epsilon_B}$", "sic_vs_background_efficiency"), ("roc", r"Signal efficiency, $\epsilon_S$", "roc_vs_background_efficiency")):
                    fig, ax = new_figure(r"Background efficiency, $\epsilon_B$", ylabel)
                    drawn = 0
                    for method in methods:
                        rows = compatible_run_rows(cache, method, scenario, variant, region, common_seeds)
                        aggregated = aggregate_metric_runs(rows, kind)
                        if aggregated is None:
                            continue
                        x, median, low, high = aggregated
                        valid = np.isfinite(median)
                        if not np.any(valid):
                            continue
                        ax.plot(x[valid], median[valid], label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method), ls="-")
                        if len(rows) >= 2:
                            band = valid & np.isfinite(low) & np.isfinite(high)
                            ax.fill_between(x[band], low[band], high[band], color=METHOD_COLORS.get(method), alpha=0.15, linewidth=0)
                        drawn += 1
                    if not drawn:
                        plt.close(fig)
                        continue
                    ax.set_xscale("log")
                    ax.set_xlim(1e-4, 1)
                    if kind == "roc":
                        ax.set_ylim(0, 1.02)
                    else:
                        ax.set_ylim(0, 20)
                    legend = inside_legend(ax, title=title, allow_headroom=False)
                    save_figure(fig, ax, destination / stem, formats, overwrite, legend)
                fig, ax = new_figure(r"Signal efficiency, $\epsilon_S$", r"Background rejection, $1/\epsilon_B$")
                random_x = np.geomspace(1e-5, 1, 500)
                ax.plot(random_x, 1.0 / random_x, color="0.45", ls="--", lw=0.55, label="Random", zorder=1)
                drawn = 0
                for method in methods:
                    rows = compatible_run_rows(cache, method, scenario, variant, region, common_seeds)
                    curves = []
                    grid = np.linspace(0, 1, 300)
                    for row in rows:
                        b = row["metrics"]["background_efficiency"]
                        s = row["metrics"]["signal_efficiency"]
                        good = np.isfinite(b) & np.isfinite(s) & (b > 0)
                        if np.sum(good) < 2:
                            continue
                        order = np.argsort(s[good])
                        sx = s[good][order]
                        rej = 1 / b[good][order]
                        unique, idx = np.unique(sx, return_index=True)
                        values = np.full_like(grid, np.nan, float)
                        inside = (grid >= unique.min()) & (grid <= unique.max())
                        values[inside] = np.interp(grid[inside], unique, rej[idx])
                        curves.append(values)
                    if not curves:
                        continue
                    matrix = np.asarray(curves)
                    median, = finite_column_percentiles(matrix, (50,))
                    valid = np.isfinite(median)
                    ax.plot(grid[valid], median[valid], label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method), ls="-")
                    if len(curves) >= 2:
                        low, high = finite_column_percentiles(matrix, (16, 84))
                        band = valid & np.isfinite(low) & np.isfinite(high)
                        ax.fill_between(grid[band], low[band], high[band], color=METHOD_COLORS.get(method), alpha=0.15, linewidth=0)
                    drawn += 1
                if drawn:
                    ax.set_xlim(0, 1)
                    ax.set_yscale("log")
                    ax.set_ylim(1, 1e5)
                    legend = inside_legend(ax, title=title, allow_headroom=False)
                    save_figure(fig, ax, destination / "background_rejection_vs_signal_efficiency", formats, overwrite, legend)
                else:
                    plt.close(fig)
                fig, ax = new_figure(r"Background working point, $\epsilon_B$", r"Signal efficiency, $\epsilon_S$")
                drawn = 0
                xs = np.asarray(PLOT_WORKING_POINTS)
                for method in methods:
                    rows = compatible_run_rows(cache, method, scenario, variant, region, common_seeds)
                    values = np.asarray([[row["metrics"]["working_points"].get(wp) for wp in PLOT_WORKING_POINTS] for row in rows], float)
                    if not values.size or not np.isfinite(values).any():
                        continue
                    median = np.nanmedian(values, axis=0)
                    ax.plot(xs, median, label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method), ls="-")
                    if len(values) >= 2:
                        low, high = np.nanpercentile(values, [16, 84], axis=0)
                        ax.fill_between(xs, low, high, color=METHOD_COLORS.get(method), alpha=0.15, linewidth=0)
                    drawn += 1
                if drawn:
                    ax.set_xscale("log")
                    ax.set_xlim(0.0045, 0.12)
                    ax.set_xticks(xs, [WORKING_POINT_LABELS[x] for x in xs])
                    ax.set_ylim(0, 1.2)
                    legend = inside_legend(ax, title=title, allow_headroom=False)
                    save_figure(fig, ax, destination / "signal_efficiency_vs_background_working_point", formats, overwrite, legend)
                else:
                    plt.close(fig)


def plot_variant_summaries(cache, output, formats, overwrite, regions):
    for region in regions:
        methods = [m for m in METHOD_ORDER if any(key[0] == m and key[1] == "signal_injection" and key[3] == region for key in cache)]
        if not methods:
            continue
        active_variants = [variant for variant in VARIANT_ORDER if any(compatible_run_rows(cache, method, "signal_injection", variant, region) for method in methods)]
        if len(methods) > 1:
            comparable_variants = [variant for variant in active_variants if all(compatible_run_rows(cache, method, "signal_injection", variant, region) for method in methods)]
        else:
            comparable_variants = active_variants
        if not comparable_variants:
            continue
        common_seeds = comparison_seed_set_across_variants(cache, methods, "signal_injection", comparable_variants, region)
        if not common_seeds:
            say(f"[SKIP] dataset-variant summary {region}: selected methods/variants have no common independent seeds", 1)
            continue
        destination = output / "01_comparison" / "Signal-Injected" / "Dataset-Variant-Summary" / folder_region(region)
        specs = [("auc", "AUC", "auc_vs_dataset_variant"), ("max_sic", "Maximum significance improvement", "max_sic_vs_dataset_variant")]
        for metric, ylabel, stem in specs:
            fig, ax = new_figure("Dataset variant", ylabel)
            drawn = 0
            for method in methods:
                medians = []
                lows = []
                highs = []
                positions = []
                spreads = []
                for i, variant in enumerate(VARIANT_ORDER):
                    if variant not in comparable_variants:
                        continue
                    rows = compatible_run_rows(cache, method, "signal_injection", variant, region, common_seeds)
                    values = [row["metrics"].get(metric) for row in rows if row["metrics"].get(metric) is not None]
                    if values:
                        positions.append(i)
                        medians.append(float(np.median(values)))
                        lows.append(float(np.percentile(values, 16)))
                        highs.append(float(np.percentile(values, 84)))
                        spreads.append(len(values) >= 2)
                if positions:
                    ax.plot(positions, medians, label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method), ls=METHOD_LINES.get(method, "-"), marker=METHOD_MARKERS.get(method, "o"))
                    if len(positions) > 1 and any(spreads):
                        ax.fill_between(positions, lows, highs, color=METHOD_COLORS.get(method), alpha=0.15, linewidth=0)
                    drawn += 1
            if drawn:
                ax.set_xticks(range(len(VARIANT_ORDER)), [VARIANT_LABELS[v] for v in VARIANT_ORDER])
                if metric == "auc":
                    ax.set_ylim(0.45, 1.0)
                legend = inside_legend(ax, title=REGION_LABELS[region])
                save_figure(fig, ax, destination / stem, formats, overwrite, legend)
            else:
                plt.close(fig)
        for wp in PLOT_WORKING_POINTS:
            fig, ax = new_figure("Dataset variant", r"Signal efficiency, $\epsilon_S$")
            drawn = 0
            for method in methods:
                positions, medians, lows, highs, spreads = [], [], [], [], []
                for i, variant in enumerate(VARIANT_ORDER):
                    if variant not in comparable_variants:
                        continue
                    rows = compatible_run_rows(cache, method, "signal_injection", variant, region, common_seeds)
                    values = [row["metrics"]["working_points"].get(wp) for row in rows if row["metrics"]["working_points"].get(wp) is not None]
                    if values:
                        positions.append(i)
                        medians.append(float(np.median(values)))
                        lows.append(float(np.percentile(values, 16)))
                        highs.append(float(np.percentile(values, 84)))
                        spreads.append(len(values) >= 2)
                if positions:
                    ax.plot(positions, medians, label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method), ls=METHOD_LINES.get(method, "-"), marker=METHOD_MARKERS.get(method, "o"))
                    if any(spreads):
                        ax.fill_between(positions, lows, highs, color=METHOD_COLORS.get(method), alpha=0.15, linewidth=0)
                    drawn += 1
            if drawn:
                ax.set_xticks(range(len(VARIANT_ORDER)), [VARIANT_LABELS[v] for v in VARIANT_ORDER])
                ax.set_ylim(0, 1.02)
                legend = inside_legend(ax, title=REGION_LABELS[region])
                token = WORKING_POINT_LABELS[wp].replace("%", "pct").replace(".", "p")
                save_figure(fig, ax, destination / f"epsS_{token}_vs_dataset_variant", formats, overwrite, legend)
            else:
                plt.close(fig)

def plot_score_distributions(groups, records, output, formats, overwrite):
    representative = representative_records(records)
    collections = defaultdict(dict)
    for (method, scenario, variant, region), (seed, record) in representative.items():
        if region != "signal_region":
            continue
        collections[(scenario, variant, region)][method] = record
    for (scenario, variant, region), method_records in collections.items():
        prepared = {}
        density_peak = 0.0
        count_peak = 0.0
        for method, selected in method_records.items():
            values = display_score_values(method, selected["scores"])
            labels = np.asarray(selected["labels"])
            mask = np.asarray(selected["mask"], bool) & np.isfinite(values)
            if not np.any(mask):
                continue
            finite_values = values[mask]
            if np.min(finite_values) >= 0.0 and np.max(finite_values) <= 1.0:
                edges = np.linspace(0.0, 1.0, 51)
            else:
                low, high = np.quantile(finite_values, [0.001, 0.999])
                if high <= low:
                    high = low + 1e-12
                edges = np.linspace(low, high, 51)
            population_specs = ((0, "BG", METHOD_LIGHT.get(method, METHOD_COLORS.get(method)), "-"),) if scenario == "background_only" else ((0, "BG", METHOD_LIGHT.get(method, METHOD_COLORS.get(method)), "-"), (1, "signal", METHOD_DARK.get(method, METHOD_COLORS.get(method)), "--"))
            populations = []
            for truth, noun, color, style in population_specs:
                population = mask & (labels == truth)
                if not np.any(population):
                    continue
                density_hist, _ = np.histogram(values[population], bins=edges, density=True)
                count_hist, _ = np.histogram(values[population], bins=edges)
                if np.isfinite(density_hist).any():
                    density_peak = max(density_peak, float(np.nanmax(density_hist)))
                if len(count_hist):
                    count_peak = max(count_peak, float(np.max(count_hist)))
                populations.append((noun, color, style, density_hist, count_hist))
            if populations:
                prepared[method] = (edges, populations)
        if not prepared:
            continue
        density_ymax = max(0.1, 1.20 * density_peak)
        count_ymax = max(10.0, 1.20 * count_peak)
        for method, (edges, populations) in prepared.items():
            destination = figure_path(output, scenario, variant, region) / METHOD_LABELS.get(method, safe_component(method))
            for density, stem, ylabel in ((True, "score_density", "Density"), (False, "score_counts", "Events / bin")):
                fig, ax = new_figure("Anomaly score", ylabel)
                for noun, color, style, density_hist, count_hist in populations:
                    histogram = density_hist if density else count_hist
                    ax.stairs(histogram, edges, baseline=None, label=f"{METHOD_LABELS.get(method, method)} {noun}", color=color, ls=style)
                if edges[0] >= 0.0 and edges[-1] <= 1.0:
                    ax.set_xlim(0.0, 1.0)
                if density:
                    ax.set_ylim(0.0, density_ymax)
                else:
                    ax.set_yscale("log")
                    ax.set_ylim(0.8, count_ymax)
                legend = inside_legend(ax, title=REGION_LABELS[region] + " · " + VARIANT_LABELS.get(variant, variant), borderaxespad=0.75, allow_headroom=False)
                save_figure(fig, ax, destination / stem, formats, overwrite, legend)


def physical_feature_values(record, variant):
    physical = np.asarray(record.get("physical"))
    names = FEATURE_NAMES.get(variant, tuple(f"feature_{i+1}" for i in range(physical.shape[1])))
    if physical.ndim != 2 or physical.shape[1] != len(names):
        names = tuple(f"feature_{i+1}" for i in range(physical.shape[1]))
    values = physical.astype(float).copy()
    for index, name in enumerate(names):
        if name in ("m1", "delta_m") and np.nanmedian(np.abs(values[:, index])) > 20:
            values[:, index] /= 1000.0
    return names, values


def representative_records(records):
    chosen = {}
    for key, record in records.items():
        method, scenario, variant, region, seed = key
        identity = (method, scenario, variant, region)
        if identity not in chosen or seed < chosen[identity][0]:
            chosen[identity] = (seed, record)
    return chosen


def plot_features(records, output, formats, overwrite, verbose):
    chosen = representative_records(records)
    by_population = {}
    for (method, scenario, variant, region), (seed, record) in chosen.items():
        if region != "signal_region":
            continue
        identity = (scenario, variant)
        priority = METHOD_ORDER.index(method) if method in METHOD_ORDER else len(METHOD_ORDER)
        current = by_population.get(identity)
        if current is None or priority < current[0]:
            by_population[identity] = (priority, method, seed, record)
    tasks = [(method, scenario, variant, seed, record) for (scenario, variant), (_, method, seed, record) in sorted(by_population.items())]
    total_pairs = sum(math.comb(len(FEATURE_NAMES.get(variant, ())), 2) for _, _, variant, _, _ in tasks)
    pair_index = 0
    for method, scenario, variant, seed, record in tasks:
        names, values = physical_feature_values(record, variant)
        labels = np.asarray(record["labels"])
        mask = np.ones(len(labels), bool)
        base = output / "03_features" / folder_variant(variant)
        for index, name in enumerate(names):
            finite = np.isfinite(values[:, index])
            low, high = np.quantile(values[finite, index], [0.005, 0.995])
            edges = np.linspace(low, high, 51)
            fig, ax = new_figure(FEATURE_LABELS.get(name, name), "Density")
            if scenario == "signal_injection":
                for truth, noun, color, style in ((0, "BG", "black", "-"), (1, "signal", "#FF0000", "--")):
                    pop = finite & mask & (labels == truth)
                    if np.any(pop):
                        hist, _ = np.histogram(values[pop, index], edges, density=True)
                        ax.stairs(hist, edges, baseline=None, color=color, ls=style, label=noun)
            else:
                pop = finite & mask & (labels == 0)
                if np.any(pop):
                    hist, _ = np.histogram(values[pop, index], edges, density=True)
                    ax.stairs(hist, edges, baseline=None, color="black", ls="-", label="BG")
            legend = inside_legend(ax, title="Signal region · " + VARIANT_LABELS.get(variant, variant))
            save_figure(fig, ax, base / "input_1d" / safe_component(name), formats, overwrite, legend)
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                pair_index += 1
                say(f"[{pair_index}/{max(1,total_pairs)}] input pair {variant}: {names[i]} vs {names[j]}", verbose, 2)
                finite_pair = mask & np.isfinite(values[:, i]) & np.isfinite(values[:, j])
                if not np.any(finite_pair):
                    continue
                xlow, xhigh = np.quantile(values[finite_pair, i], [0.005, 0.995])
                ylow, yhigh = np.quantile(values[finite_pair, j], [0.005, 0.995])
                if xhigh <= xlow or yhigh <= ylow:
                    continue
                fig, ax = new_figure(FEATURE_LABELS.get(names[i], names[i]), FEATURE_LABELS.get(names[j], names[j]), wide=True)
                for truth, noun, color, style in ((0, "BG", "black", "-"), (1, "signal", "#FF0000", "--")):
                    pop = finite_pair & (labels == truth)
                    if not np.any(pop) or (truth == 1 and scenario == "background_only"):
                        continue
                    hist, xe, ye = np.histogram2d(values[pop, i], values[pop, j], bins=60, range=((xlow, xhigh), (ylow, yhigh)))
                    levels = probability_contour_levels(hist)
                    if not len(levels):
                        continue
                    linewidths = np.linspace(0.9, 1.2, len(levels))
                    ax.contour(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), hist.T, levels=levels, colors=[color], linestyles=[style], linewidths=linewidths)
                    ax.plot([], [], color=color, ls=style, label=noun)
                ax.set_xlim(xlow, xhigh)
                ax.set_ylim(ylow, yhigh)
                legend = inside_legend(ax, title="Signal region · " + VARIANT_LABELS.get(variant, variant), fontsize=8.0, borderaxespad=0.9)
                save_figure(fig, ax, base / "input_2d" / f"{safe_component(names[i])}_vs_{safe_component(names[j])}", formats, overwrite, legend)
        plot_correlations(base, names, values, labels, record, variant, formats, overwrite)
        if method in ("riddle", "iad", "supervised"):
            plot_latents(base, method, scenario, variant, record, formats, overwrite, verbose)


def mapped_latents(record):
    if "latent" not in record:
        return None, None
    latent = np.asarray(record["latent"], float)
    mask = np.asarray(record["mask"], bool)
    if latent.ndim != 2 or len(latent) != int(mask.sum()):
        return None, None
    full = np.full((len(mask), latent.shape[1]), np.nan)
    full[mask] = latent
    width = latent.shape[1] - 1 if record.get("paper_mass_conditioning", False) and latent.shape[1] >= 2 else latent.shape[1]
    return full[:, :width], mask


def plot_latents(base, method, scenario, variant, record, formats, overwrite, verbose):
    latent, mapped = mapped_latents(record)
    if latent is None or latent.shape[1] == 0:
        return
    labels = np.asarray(record["labels"])
    names = [f"z{i+1}" for i in range(latent.shape[1])]
    one_dimensional = []
    global_peak = 1.0 / math.sqrt(2.0 * math.pi)
    for i, name in enumerate(names):
        finite = mapped & np.isfinite(latent[:, i])
        if not np.any(finite):
            continue
        low, high = np.quantile(latent[finite, i], [0.005, 0.995])
        if high <= low:
            high = low + 1e-12
        edges = np.linspace(low, high, 51)
        populations = []
        for truth, noun, color, style in ((0, "BG", "black", "-"), (1, "signal", "#FF0000", "--")):
            pop = finite & (labels == truth)
            if not np.any(pop) or (truth == 1 and scenario == "background_only"):
                continue
            hist, _ = np.histogram(latent[pop, i], edges, density=True)
            populations.append((hist, noun, color, style))
            if np.isfinite(hist).any():
                global_peak = max(global_peak, float(np.nanmax(hist)))
        normal_x = np.linspace(low, high, 400)
        normal_y = np.exp(-0.5 * normal_x ** 2) / math.sqrt(2.0 * math.pi)
        if len(normal_y):
            global_peak = max(global_peak, float(np.max(normal_y)))
        one_dimensional.append((i, name, edges, populations, normal_x, normal_y))
    shared_ymax = max(0.5, 1.24 * global_peak)
    for i, name, edges, populations, normal_x, normal_y in one_dimensional:
        fig, ax = new_figure(fr"$z_{{{i+1}}}$", "Density")
        for hist, noun, color, style in populations:
            ax.stairs(hist, edges, baseline=None, color=color, ls=style, label=f"RIDDLE {noun}")
        ax.plot(normal_x, normal_y, color="#00BFC4", ls="--", lw=0.575, label=r"$\mathcal{N}(0,1)$", zorder=5)
        ax.set_ylim(0.0, 2.5)
        legend = ax.legend(loc="upper left", frameon=True, framealpha=0.93, borderpad=0.42, labelspacing=0.32, handlelength=1.9, columnspacing=0.8, ncol=1, title="Signal region · " + VARIANT_LABELS.get(variant, variant), fontsize=STYLE["legend.fontsize"], borderaxespad=0.75)
        ax.figure.canvas.draw()
        save_figure(fig, ax, base / "latent_1d" / name, formats, overwrite, legend)
    pairs = [(i, j) for i in range(latent.shape[1]) for j in range(i + 1, latent.shape[1])]
    for index, (i, j) in enumerate(pairs, 1):
        say(f"[{index}/{len(pairs)}] latent pair {variant}: z{i+1} vs z{j+1}", verbose, 2)
        fig, ax = new_figure(fr"$z_{{{i+1}}}$", fr"$z_{{{j+1}}}$", wide=True)
        for truth, noun, color, style in ((0, "BG", "black", "-"), (1, "signal", "#FF0000", "--")):
            pop = mapped & (labels == truth) & np.isfinite(latent[:, i]) & np.isfinite(latent[:, j])
            if not np.any(pop) or (truth == 1 and scenario == "background_only"):
                continue
            hist, xe, ye = np.histogram2d(latent[pop, i], latent[pop, j], bins=60)
            levels = probability_contour_levels(hist)
            if not len(levels):
                continue
            ax.contour(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), hist.T, levels=levels, colors=[color], linestyles=[style], linewidths=np.linspace(0.9, 1.2, len(levels)))
            ax.plot([], [], color=color, ls=style, label=f"RIDDLE {noun}")
        ax.set_xlim(-10.0, 10.0)
        ax.set_ylim(-10.0, 10.0)
        legend = inside_legend(ax, title="Signal region · " + VARIANT_LABELS.get(variant, variant))
        save_figure(fig, ax, base / "latent_2d" / f"z{i+1}_vs_z{j+1}", formats, overwrite, legend)


def plot_correlations(base, names, physical, labels, record, variant, formats, overwrite):
    mass = np.asarray(record["mass"], float)
    mass_tev = mass / 1000.0 if np.nanmedian(np.abs(mass)) > 20 else mass
    input_matrix = np.column_stack((mass_tev, physical))
    input_names = [r"$m_{jj}$", *[FEATURE_LABELS.get(name, name).replace(" (TeV)", "") for name in names]]
    latent, mapped = mapped_latents(record)
    for truth, noun in ((0, "background"), (1, "signal")):
        if truth == 1 and not np.any(labels == 1):
            continue
        pop = labels == truth
        correlation_heatmap(input_matrix[pop], input_names, base / "correlations" / f"input_{noun}", formats, overwrite)
        if latent is not None:
            lpop = pop & mapped
            latent_matrix = np.column_stack((mass_tev[lpop], latent[lpop]))
            latent_names = [r"$m_{jj}$", *[fr"$z_{{{i+1}}}$" for i in range(latent.shape[1])]]
            correlation_heatmap(latent_matrix, latent_names, base / "correlations" / f"latent_{noun}", formats, overwrite)
    if latent is not None and physical.shape[1] == latent.shape[1]:
        bg = labels == 0
        before = np.corrcoef(physical[bg], rowvar=False)
        after = np.corrcoef(latent[bg & mapped], rowvar=False)
        before_values = np.abs(before[np.triu_indices_from(before, 1)])
        after_values = np.abs(after[np.triu_indices_from(after, 1)])
        fig, ax = new_figure("Representation", "Mean off-diagonal |Pearson correlation|")
        ax.plot([0, 1], [np.nanmean(before_values), np.nanmean(after_values)], marker="o", color=METHOD_COLORS["riddle"], ls="-")
        ax.set_xticks([0, 1], ["Input", "Latent"])
        ax.set_ylim(0, max(0.05, 1.12 * max(np.nanmean(before_values), np.nanmean(after_values))))
        save_figure(fig, ax, base / "correlations" / "background_offdiagonal_before_vs_after", formats, overwrite)


def correlation_heatmap(values, labels, stem, formats, overwrite):
    values = np.asarray(values, float)
    good = np.isfinite(values).all(axis=1)
    if np.sum(good) < 3 or values.shape[1] < 2:
        return
    matrix = np.corrcoef(values[good], rowvar=False)
    fig, ax = new_figure("", "", wide=True)
    image = ax.imshow(matrix, cmap="RdBu_r", norm=TwoSlopeNorm(vcenter=0, vmin=-1, vmax=1))
    ax.set_xticks(range(len(labels)), labels, rotation=35, ha="right")
    ax.set_yticks(range(len(labels)), labels)
    ax.minorticks_off()
    if len(labels) <= 6:
        for i in range(len(labels)):
            for j in range(len(labels)):
                ax.text(j, i, f"{matrix[i,j]:.2f}", ha="center", va="center", fontsize=6.3, color="white" if abs(matrix[i,j]) > 0.62 else "black")
    cbar = fig.colorbar(image, ax=ax, pad=0.03)
    cbar.set_label("Pearson correlation")
    save_figure(fig, ax, stem, formats, overwrite)


def plot_score_vs_mass(records, output, formats, overwrite):
    for (method, scenario, variant, region), (seed, record) in representative_records(records).items():
        if region != "signal_region":
            continue
        labels = np.asarray(record["labels"])
        mask = np.asarray(record["mask"], bool) & (labels == 0)
        if not np.any(mask):
            continue
        mass = np.asarray(record["mass"], float)
        if np.nanmedian(np.abs(mass)) > 20:
            mass = mass / 1000.0
        scores = display_score_values(method, record["scores"])
        finite = mask & np.isfinite(scores) & np.isfinite(mass)
        if not np.any(finite):
            continue
        fig, ax = new_figure(r"$m_{jj}$ (TeV)", "Anomaly score", wide=True)
        score_range = (0.0, 1.0) if method == "ranode" and np.nanmin(scores[finite]) >= 0.0 and np.nanmax(scores[finite]) <= 1.0 else (float(np.nanmin(scores[finite])), float(np.nanmax(scores[finite])))
        hist, xe, ye = np.histogram2d(mass[finite], scores[finite], bins=(70, 70), range=((float(np.nanmin(mass[finite])), float(np.nanmax(mass[finite]))), score_range))
        white_log_density_mesh(ax, xe, ye, hist, "Background events / bin")
        if method == "ranode" and score_range == (0.0, 1.0):
            ax.set_ylim(0.0, 1.0)
        save_figure(fig, ax, output / "03_features" / folder_variant(variant) / "score_vs_mass" / f"{safe_component(METHOD_LABELS.get(method, method))}_{safe_component(SCENARIO_LABELS[scenario])}", formats, overwrite)


def exact_mass_sculpt_curve(record, efficiencies):
    labels = np.asarray(record["labels"])
    bg = labels == 0
    mass = np.asarray(record["mass"], float)[bg]
    if len(mass) < 300:
        return None
    edges = equal_occupancy(mass, 300)
    full = np.histogram(mass, edges)[0]
    values = []
    for efficiency in efficiencies:
        selection = exact_background_selection(record, float(efficiency))
        if selection is None:
            values.append(np.nan)
            continue
        selected = np.mean(selection["weights"], axis=0)[bg]
        histogram = np.histogram(mass, edges, weights=selected)[0]
        values.append(shape_chi2(full, histogram, float(efficiency)))
    return np.asarray(values, float)


def random_reference_band(mass, efficiencies, trials=100):
    edges = equal_occupancy(mass, 300)
    full = np.histogram(mass, edges)[0]
    ids = np.minimum(np.searchsorted(edges, mass, side="right") - 1, len(full) - 1)
    rng = np.random.default_rng(42)
    rows = []
    for _ in range(trials):
        curve = []
        for efficiency in efficiencies:
            count = max(1, int(round(efficiency * len(mass))))
            chosen = rng.choice(len(mass), count, replace=False)
            curve.append(shape_chi2(full, np.bincount(ids[chosen], minlength=len(full)), count / len(mass)))
        rows.append(curve)
    return np.asarray(rows)


def plot_mass_sculpting(cache, records, output, formats, overwrite, scenarios, variants, regions, verbose):
    efficiencies = np.geomspace(0.005, 0.2, 28)
    for scenario in scenarios:
        for variant in variants:
            for region in regions:
                methods = comparison_methods(cache, scenario, variant, region)
                common_seeds = comparison_seed_set(cache, methods, scenario, variant, region)
                if len(methods) > 1 and not common_seeds:
                    say(f"[SKIP] mass sculpting {scenario} {variant} {region}: selected methods have no common independent seeds", verbose)
                    continue
                method_curves = {}
                for method in methods:
                    curves = []
                    for (m, sc, v, rg, seed), record in records.items():
                        if (m, sc, v, rg) != (method, scenario, variant, region) or (common_seeds and seed not in common_seeds):
                            continue
                        curve = exact_mass_sculpt_curve(record, efficiencies)
                        if curve is not None:
                            curves.append(curve)
                    if curves:
                        method_curves[method] = np.asarray(curves)
                if not method_curves:
                    continue
                fig, ax = new_figure(r"Target background efficiency, $\epsilon_B$", r"$\chi^2/n_{\mathrm{dof}}$")
                first_record = next(record for (m, sc, v, rg, seed), record in records.items() if (sc, v, rg) == (scenario, variant, region) and (not common_seeds or seed in common_seeds))
                mass = np.asarray(first_record["mass"])[np.asarray(first_record["labels"]) == 0]
                if len(mass) >= 300:
                    reference = random_reference_band(mass, efficiencies)
                    median = np.nanmedian(reference, axis=0)
                    low, high = np.nanpercentile(reference, [16, 84], axis=0)
                    ax.fill_between(efficiencies, low, high, color="0.5", alpha=0.12, linewidth=0, zorder=0)
                    ax.plot(efficiencies, median, color="0.45", ls="--", lw=0.675, label="Random subset", zorder=1)
                for method, matrix in method_curves.items():
                    median = np.nanmedian(matrix, axis=0)
                    ax.plot(efficiencies, median, color=METHOD_COLORS.get(method), ls="-", label=METHOD_LABELS.get(method, method))
                    if len(matrix) >= 2:
                        low, high = finite_column_percentiles(matrix, (16, 84))
                        ax.fill_between(efficiencies, low, high, color=METHOD_COLORS.get(method), alpha=0.15, linewidth=0)
                ax.set_xscale("linear")
                ax.set_xlim(0.20, 0.01)
                ax.set_yscale("log")
                ax.set_ylim(5e-1, 3.5e2)
                legend = inside_legend(ax, title=REGION_LABELS[region] + " · " + VARIANT_LABELS.get(variant, variant), borderaxespad=0.75, allow_headroom=False)
                save_figure(fig, ax, figure_path(output, scenario, variant, region) / "mass_sculpting_chi2_ndof_vs_background_efficiency", formats, overwrite, legend)
                working_point_plots = {}
                all_efficiencies = []
                for wp in PLOT_WORKING_POINTS[::-1]:
                    rows = []
                    for method in methods:
                        candidates = [(seed, record) for (m, sc, v, rg, seed), record in records.items() if (m, sc, v, rg) == (method, scenario, variant, region) and (not common_seeds or seed in common_seeds)]
                        if not candidates:
                            continue
                        seed, record = sorted(candidates)[0]
                        selection = exact_background_selection(record, wp)
                        if selection is None:
                            continue
                        labels = np.asarray(record["labels"])
                        mass = np.asarray(record["mass"], float)
                        if np.nanmedian(np.abs(mass)) > 20:
                            mass = mass / 1000.0
                        bg = labels == 0
                        edges = equal_occupancy(mass[bg], min(24, max(10, len(mass[bg]) // 50)))
                        total = np.histogram(mass[bg], edges)[0]
                        selected_counts = np.histogram(mass[bg], edges, weights=np.mean(selection["weights"], axis=0)[bg])[0]
                        efficiency = np.divide(selected_counts, total, out=np.full_like(selected_counts, np.nan, dtype=float), where=total > 0)
                        centers = 0.5 * (edges[:-1] + edges[1:])
                        rows.append((method, centers, efficiency))
                        all_efficiencies.extend(efficiency[np.isfinite(efficiency)].tolist())
                    if rows:
                        working_point_plots[wp] = rows
                if working_point_plots:
                    maximum = max(all_efficiencies) if all_efficiencies else max(PLOT_WORKING_POINTS)
                    shared_ymax = max(0.15, 1.18 * maximum, 1.18 * max(PLOT_WORKING_POINTS))
                    for wp, rows in working_point_plots.items():
                        fig, ax = new_figure(r"$m_{jj}$ (TeV)", r"Background efficiency, $\epsilon_B$")
                        ax.axhline(wp, color="0.55", ls="--", lw=0.5, zorder=1)
                        for method, centers, efficiency in rows:
                            ax.plot(centers, efficiency, color=METHOD_COLORS.get(method), ls="-", label=METHOD_LABELS.get(method, method))
                        ax.set_ylim(0, shared_ymax)
                        legend = inside_legend(ax, title=REGION_LABELS[region] + " · " + VARIANT_LABELS.get(variant, variant), borderaxespad=0.75, allow_headroom=False)
                        token = WORKING_POINT_LABELS[wp].replace("%", "pct").replace(".", "p")
                        save_figure(fig, ax, figure_path(output, scenario, variant, region) / f"background_efficiency_vs_mass_{token}", formats, overwrite, legend)


def plot_mass_scan(records, output, formats, overwrite, verbose):
    riddle_full = [(scenario, variant, seed, record) for (method, scenario, variant, region, seed), record in records.items() if method == "riddle" and region == "full_region"]
    if not riddle_full:
        say("[SKIP] RIDDLE Full Region mass scan: saved scores are Signal Region only", verbose)
        return
    for scenario, variant, seed, record in riddle_full:
        mass = np.asarray(record["mass"], float)
        if np.nanmedian(np.abs(mass)) > 20:
            mass = mass / 1000.0
        scores = np.asarray(record["scores"], float)
        mask = np.asarray(record["mask"], bool) & np.isfinite(scores)
        edges = np.linspace(mass.min(), mass.max(), 81)
        inclusive = np.histogram(mass, edges)[0]
        destination = output / "04_mass_scan" / "RIDDLE" / SCENARIO_LABELS[scenario] / folder_variant(variant) / f"seed_{seed:03d}"
        thresholds = np.round(np.arange(0.30, 1.00, 0.01), 2)
        for index, threshold in enumerate(thresholds, 1):
            say(f"[{index}/{len(thresholds)}] RIDDLE mass scan threshold {threshold:.2f}", verbose, 2)
            selected = mask & (scores > threshold)
            hist = np.histogram(mass[selected], edges)[0]
            fig, ax = new_figure(r"$m_{jj}$ (TeV)", "Events / bin")
            ax.stairs(inclusive, edges, baseline=None, color="0.72", ls=":", label="Inclusive")
            ax.stairs(hist, edges, baseline=None, color=METHOD_COLORS["riddle"], ls="-", label="RIDDLE")
            ax.set_yscale("log")
            legend = inside_legend(ax, title=REGION_LABELS["full_region"] + " · " + VARIANT_LABELS.get(variant, variant))
            save_figure(fig, ax, destination / f"score_gt_{threshold:.2f}".replace(".", "p"), formats, overwrite, legend)


def context_match_weights(reference_context, target_context, bins=30):
    reference_context = np.asarray(reference_context, float)
    target_context = np.asarray(target_context, float)
    if reference_context.ndim != 1 or target_context.ndim != 1 or len(reference_context) < 2 or len(target_context) < 2:
        return None
    if not np.isfinite(reference_context).all() or not np.isfinite(target_context).all():
        return None
    low = max(float(np.min(reference_context)), float(np.min(target_context)))
    high = min(float(np.max(reference_context)), float(np.max(target_context)))
    if not np.isfinite(low) or not np.isfinite(high) or high <= low:
        return None
    edges = np.linspace(low, high, int(bins) + 1)
    ref_hist = np.histogram(reference_context, edges)[0].astype(float)
    tar_hist = np.histogram(target_context, edges)[0].astype(float)
    ratio = np.divide(tar_hist, ref_hist, out=np.zeros_like(tar_hist), where=ref_hist > 0)
    ids = np.clip(np.searchsorted(edges, reference_context, side="right") - 1, 0, len(ratio) - 1)
    weights = ratio[ids]
    weights[(reference_context < low) | (reference_context > high)] = 0.0
    return weights


def plot_latent_closure(groups, records, output, formats, overwrite, verbose):
    for (scenario, seed, variant), group in groups.items():
        if "riddle" not in group:
            continue
        key = ("riddle", scenario, variant, "signal_region", seed)
        if key not in records:
            continue
        root, report = group["riddle"]
        record = records[key]
        latent, mapped = mapped_latents(record)
        if latent is None:
            continue
        relative_npz = "density/stein_scoring_reference.npz"
        relative_json = "density/stein_scoring_reference.json"
        npz_path = verify_requested_artifact(root, report, relative_npz)
        json_path = verify_requested_artifact(root, report, relative_json)
        if npz_path is None or json_path is None:
            continue
        with np.load(npz_path, allow_pickle=False) as archive:
            if "reference_B" not in archive:
                continue
            reference = np.asarray(archive["reference_B"], float)
        if reference.ndim != 2 or reference.shape[1] != latent.shape[1] + 1:
            continue
        labels = np.asarray(record["labels"])
        mapped_labels = labels[mapped]
        full_latent = np.asarray(record["latent"], float)
        actual = latent[mapped & (labels == 0)]
        actual_context = full_latent[mapped_labels == 0, -1]
        reference_z = reference[:, :-1]
        reference_context = reference[:, -1]
        weights = context_match_weights(reference_context, actual_context)
        if weights is None or not np.any(weights > 0):
            say(f"[SKIP] RIDDLE latent closure {variant} seed {seed}: incompatible mass-context ranges", verbose)
            continue
        base = output / "03_features" / folder_variant(variant) / "latent_closure"
        for i in range(actual.shape[1]):
            pooled = np.concatenate((actual[:, i], reference_z[:, i]))
            low, high = np.quantile(pooled[np.isfinite(pooled)], [0.005, 0.995])
            edges = np.linspace(low, high, 51)
            actual_hist, _ = np.histogram(actual[:, i], edges, density=True)
            ref_hist = np.histogram(reference_z[:, i], edges, weights=weights)[0].astype(float)
            if ref_hist.sum() > 0:
                ref_hist /= ref_hist.sum() * np.diff(edges)
            fig, ax = new_figure(fr"$z_{{{i+1}}}$", "Density")
            ax.stairs(actual_hist, edges, baseline=None, color="black", ls="-", label="Actual BG")
            ax.stairs(ref_hist, edges, baseline=None, color="#FF0000", ls="--", label="Synthetic BG")
            legend = inside_legend(ax, title="Signal region · " + VARIANT_LABELS.get(variant, variant))
            save_figure(fig, ax, base / f"z{i+1}", formats, overwrite, legend)
        pairs = [(i, j) for i in range(actual.shape[1]) for j in range(i + 1, actual.shape[1])]
        for index, (i, j) in enumerate(pairs, 1):
            say(f"[{index}/{len(pairs)}] latent closure {variant}: z{i+1} vs z{j+1}", verbose, 2)
            fig, ax = new_figure(fr"$z_{{{i+1}}}$", fr"$z_{{{j+1}}}$", wide=True)
            for values, noun, color, style, weight in ((actual, "Actual BG", "black", "-", None), (reference_z, "Synthetic BG", "#FF0000", "--", weights)):
                hist, xe, ye = np.histogram2d(values[:, i], values[:, j], bins=60, weights=weight)
                levels = probability_contour_levels(hist)
                if not len(levels):
                    continue
                ax.contour(0.5 * (xe[:-1] + xe[1:]), 0.5 * (ye[:-1] + ye[1:]), hist.T, levels=levels, colors=[color], linestyles=[style], linewidths=np.linspace(0.9, 1.2, len(levels)))
                ax.plot([], [], color=color, ls=style, label=noun)
            ax.set_xlim(-5.0, 5.0)
            ax.set_ylim(-5.0, 5.0)
            legend = inside_legend(ax, title="Signal region · " + VARIANT_LABELS.get(variant, variant), fontsize=8.0, borderaxespad=0.9)
            save_figure(fig, ax, base / f"z{i+1}_vs_z{j+1}", formats, overwrite, legend)


def active_stein_settings(report):
    contract = report.get("contract", {})
    settings = contract.get("settings", {}).get("riddle", {})
    return settings.get("stein", {}), settings


def support_quantities(root, report, record):
    latent, mapped = mapped_latents(record)
    if latent is None:
        return None
    full_latent = np.asarray(record["latent"], float)
    if full_latent.shape[1] != latent.shape[1] + 1:
        return None
    reference_path = verify_requested_artifact(root, report, "density/stein_scoring_reference.npz")
    if reference_path is None:
        return None
    with np.load(reference_path, allow_pickle=False) as archive:
        reference = np.asarray(archive["reference_B"], float)
    stein, riddle = active_stein_settings(report)
    scoring = stein.get("scoring", {})
    guard = scoring.get("support_guard", {})
    if not guard.get("enabled", False) or guard.get("statistic") != "radius":
        return None
    radius = np.linalg.norm(full_latent[:, :-1], axis=1)
    reference_radius = np.linalg.norm(reference[:, :-1], axis=1)
    zscore, metadata = conditional_gaussianize(reference_radius, reference[:, -1], radius, full_latent[:, -1], int(guard["mass_bins"]))
    gate = float(ndtri(float(guard["gate_quantile"])))
    penalty = float(guard["weight"]) * np.logaddexp(0.0, (zscore - gate) / float(guard["temperature"]))
    return radius, zscore, penalty, mapped


def plot_stein(groups, records, output, formats, overwrite):
    for (scenario, seed, variant), group in groups.items():
        if "riddle" not in group:
            continue
        key = ("riddle", scenario, variant, "signal_region", seed)
        if key not in records:
            continue
        root, report = group["riddle"]
        record = records[key]
        if "raw_scores" not in record:
            continue
        support = support_quantities(root, report, record)
        if support is None:
            continue
        radius, zscore, penalty, mapped = support
        raw = np.asarray(record["raw_scores"])[mapped]
        final = np.asarray(record["scores"])[mapped]
        labels = np.asarray(record["labels"])[mapped]
        fit_scores = np.asarray(record.get("fit_scores", []))
        base = output / "05_stein_witness" / "RIDDLE" / folder_variant(variant)
        quantities = [("final_score_distribution", final, "Final anomaly score"), ("guarded_raw_score_distribution", raw, "Guarded raw score"), ("support_radius_distribution", radius, "Support radius"), ("support_z_distribution", zscore, r"Support $Z_r$"), ("support_penalty_distribution", penalty, "Support penalty")]
        if fit_scores.ndim == 2 and fit_scores.shape[1] == len(record["labels"]):
            kind = str(np.asarray(record.get("fit_score_kind", "")).item()) if "fit_score_kind" in record else ""
            if "unguarded" in kind or "tail_focus" in kind:
                unguarded = np.nanmedian(fit_scores[:, mapped], axis=0)
                quantities.append(("unguarded_tail_focus_ensemble_distribution", unguarded, "Unguarded tail-focus ensemble score"))
        for stem, values, xlabel in quantities:
            finite = np.isfinite(values)
            if not np.any(finite):
                continue
            low, high = np.quantile(values[finite], [0.005, 0.995])
            if high <= low:
                high = low + 1e-12
            edges = np.linspace(low, high, 51)
            fig, ax = new_figure(xlabel, "Density")
            for truth, noun, color, style in ((0, "BG", "black", "-"), (1, "signal", "#FF0000", "--")):
                pop = finite & (labels == truth)
                if np.any(pop):
                    hist, _ = np.histogram(values[pop], edges, density=True)
                    ax.stairs(hist, edges, baseline=None, color=color, ls=style, label=f"RIDDLE {noun}")
            add_y_headroom(ax, 0.18)
            legend = inside_legend(ax, title="Signal region · " + VARIANT_LABELS.get(variant, variant), borderaxespad=0.75)
            save_figure(fig, ax, base / stem, formats, overwrite, legend)
        guarded = raw
        pairs = [("support_radius_vs_support_penalty", radius, penalty, "Support radius", "Support penalty"), ("support_z_vs_support_penalty", zscore, penalty, r"Support $Z_r$", "Support penalty"), ("support_z_vs_guarded_raw_score", zscore, guarded, r"Support $Z_r$", "Guarded raw score"), ("guarded_raw_vs_final_score", guarded, final, "Guarded raw score", "Final anomaly score")]
        if len(quantities) > 5:
            unguarded = quantities[-1][1]
            pairs.insert(0, ("unguarded_score_vs_guarded_score", unguarded, guarded, "Unguarded ensemble score", "Guarded raw score"))
        for stem, x, y, xlabel, ylabel in pairs:
            good = np.isfinite(x) & np.isfinite(y)
            if not np.any(good):
                continue
            fig, ax = new_figure(xlabel, ylabel, wide=True)
            hist, xe, ye = np.histogram2d(x[good], y[good], bins=70)
            white_log_density_mesh(ax, xe, ye, hist, "Events / bin")
            save_figure(fig, ax, base / stem, formats, overwrite)


def history_rows(root, report, relative):
    path = verify_requested_artifact(root, report, relative)
    if path is None:
        return None
    data = json.loads(path.read_text())
    return data.get("history", data) if isinstance(data, dict) else data


def objective_arrays(history, stein=False):
    if not isinstance(history, list) or not history:
        return None
    train_key = "train_objective" if stein and "train_objective" in history[0] else "train_nll"
    validation_key = "validation_objective" if stein and "validation_objective" in history[0] else "validation_nll"
    if train_key not in history[0] or validation_key not in history[0]:
        return None
    return np.asarray([row[train_key] for row in history], float), np.asarray([row[validation_key] for row in history], float)


def plot_training(groups, output, formats, overwrite):
    represented = set()
    for identity, group in groups.items():
        for method, (root, report) in group.items():
            key = (method, str(root))
            if key in represented:
                continue
            represented.add(key)
            destination = output / "02_training" / METHOD_LABELS.get(method, safe_component(method))
            if method in ("riddle", "iad", "supervised"):
                history = history_rows(root, report, "background/history.json")
                arrays = objective_arrays(history, False) if history is not None else None
                if arrays is not None:
                    plot_history(arrays, "Negative log likelihood", destination / "background_nll", formats, overwrite)
                if method in ("iad", "supervised"):
                    oracle_history = history_rows(root, report, "density/background_correction/history.json")
                    oracle_arrays = objective_arrays(oracle_history, False) if oracle_history is not None else None
                    if oracle_arrays is not None:
                        plot_history(oracle_arrays, "Negative log likelihood", destination / "oracle_background_nll", formats, overwrite)
                selection_path = verify_requested_artifact(root, report, "density/ensemble_selection.json")
                if selection_path is not None:
                    selection = json.loads(selection_path.read_text())
                    histories = []
                    for member in selection.get("members", []):
                        relative = "density/" + member["directory"] + "/residual_losses.json"
                        rows = history_rows(root, report, relative)
                        arrays = objective_arrays(rows, True) if rows is not None else None
                        if arrays is not None:
                            histories.append(arrays)
                    if histories:
                        min_len = min(len(item[0]) for item in histories)
                        train = np.asarray([item[0][:min_len] for item in histories])
                        validation = np.asarray([item[1][:min_len] for item in histories])
                        plot_history_band(train, validation, "Stein objective", destination / "stein_witness_objective", formats, overwrite)
            elif method == "lacathode":
                for stem, names, ylabel in (("background_nll", ("lacathode_model_train_losses.npy", "lacathode_model_val_losses.npy"), "Negative log likelihood"), ("classifier_bce", ("loss_matris.npy", "val_loss_matris.npy"), "Binary cross-entropy")):
                    paths = [verify_requested_artifact(root, report, "training/" + name) for name in names]
                    if all(path is not None for path in paths):
                        values = [np.atleast_2d(np.load(path, allow_pickle=False)) for path in paths]
                        plot_history_band(values[0], values[1], ylabel, destination / stem, formats, overwrite)


def plot_history(arrays, ylabel, stem, formats, overwrite):
    train, validation = arrays
    fig, ax = new_figure("Epoch", ylabel)
    ax.plot(np.arange(1, len(train) + 1), train, label="Train", color=METHOD_LIGHT["riddle"], ls="-")
    ax.plot(np.arange(1, len(validation) + 1), validation, label="Validation", color=METHOD_DARK["riddle"], ls="--")
    add_y_headroom(ax, 0.14)
    legend = inside_legend(ax, borderaxespad=0.75)
    save_figure(fig, ax, stem, formats, overwrite, legend)


def plot_history_band(train, validation, ylabel, stem, formats, overwrite, method="riddle", labels=("Train", "Validation")):
    train = np.atleast_2d(train)
    validation = np.atleast_2d(validation)
    length = min(train.shape[1], validation.shape[1])
    x = np.arange(1, length + 1)
    fig, ax = new_figure("Epoch", ylabel)
    for values, label, color, style in ((train[:, :length], labels[0], METHOD_LIGHT[method], "-"), (validation[:, :length], labels[1], METHOD_DARK[method], "--")):
        median = np.median(values, axis=0)
        ax.plot(x, median, label=label, color=color, ls=style)
        if len(values) >= 2:
            low, high = np.percentile(values, [16, 84], axis=0)
            ax.fill_between(x, low, high, color=color, alpha=0.15, linewidth=0)
    add_y_headroom(ax, 0.14)
    legend = inside_legend(ax, borderaxespad=0.75)
    save_figure(fig, ax, stem, formats, overwrite, legend)


def plot_stability(cache, output, formats, overwrite, regions):
    for variant in VARIANT_ORDER:
        for region in regions:
            methods = comparison_methods(cache, "signal_injection", variant, region)
            if not methods:
                continue
            common_seeds = comparison_seed_set(cache, methods, "signal_injection", variant, region)
            if len(methods) > 1 and not common_seeds:
                continue
            for field, ylabel, stem, wp in (("auc", "AUC", "auc_run_stability", None), ("max_sic", "Maximum significance improvement", "max_sic_run_stability", None), ("working", r"Signal efficiency at 0.5% BG", "exact_wp_signal_efficiency_run_stability_0p5pct", 0.005)):
                fig, ax = new_figure("Independent run / seed", ylabel)
                drawn = 0
                all_seeds = common_seeds or sorted({row["seed"] for method in methods for row in compatible_run_rows(cache, method, "signal_injection", variant, region)})
                seed_pos = {seed: i for i, seed in enumerate(all_seeds)}
                for method in methods:
                    rows = compatible_run_rows(cache, method, "signal_injection", variant, region, all_seeds)
                    if len(rows) < 2:
                        continue
                    xs, ys = [], []
                    for row in rows:
                        value = row["metrics"][field] if field != "working" else row["metrics"]["working_points"].get(wp)
                        if value is not None:
                            xs.append(seed_pos[row["seed"]])
                            ys.append(value)
                    if xs:
                        ax.plot(xs, ys, marker=METHOD_MARKERS.get(method, "o"), ls="none", color=METHOD_COLORS.get(method), label=METHOD_LABELS.get(method, method))
                        drawn += 1
                if drawn:
                    ax.set_xticks(range(len(all_seeds)), [str(seed) for seed in all_seeds])
                    legend = inside_legend(ax, title=REGION_LABELS[region] + " · " + VARIANT_LABELS.get(variant, variant))
                    save_figure(fig, ax, output / "01_comparison" / "Signal-Injected" / folder_variant(variant) / folder_region(region) / stem, formats, overwrite, legend)
                else:
                    plt.close(fig)


def plot_acceptance(cache, output, formats, overwrite):
    identities = sorted({(scenario, variant, region) for (_, scenario, variant, region, _) in cache})
    for scenario, variant, region in identities:
        methods = comparison_methods(cache, scenario, variant, region)
        if not methods:
            continue
        common_seeds = comparison_seed_set(cache, methods, scenario, variant, region)
        if len(methods) > 1 and not common_seeds:
            continue
        method_values = {}
        for method in methods:
            rows = compatible_run_rows(cache, method, scenario, variant, region, common_seeds)
            values = []
            for row in rows:
                acceptance = row["metrics"].get("acceptance", {})
                bg = acceptance.get("background", {}).get("acceptance")
                sig = acceptance.get("signal", {}).get("acceptance") if scenario == "signal_injection" else None
                values.append((bg, sig))
            if values:
                method_values[method] = values
        flattened = [value for values in method_values.values() for pair in values for value in pair if value is not None]
        if not flattened or max(flattened) - min(flattened) < 0.005:
            continue
        fig, ax = new_figure("Method", "Mapping acceptance")
        positions = np.arange(len(method_values))
        labels = []
        for i, method in enumerate(method_values):
            pairs = method_values[method]
            bg = np.median([pair[0] for pair in pairs if pair[0] is not None])
            ax.plot(i - 0.08, bg, marker="o", ls="none", color=METHOD_LIGHT.get(method, METHOD_COLORS.get(method)), label="BG" if i == 0 else "_nolegend_")
            if scenario == "signal_injection":
                sigs = [pair[1] for pair in pairs if pair[1] is not None]
                if sigs:
                    ax.plot(i + 0.08, np.median(sigs), marker="^", ls="none", color=METHOD_DARK.get(method, METHOD_COLORS.get(method)), label="Signal" if i == 0 else "_nolegend_")
            labels.append(METHOD_LABELS.get(method, method))
        ax.set_xticks(positions, labels, rotation=15)
        ax.set_ylim(0, 1.02)
        legend = inside_legend(ax, title=REGION_LABELS[region] + " · " + VARIANT_LABELS.get(variant, variant))
        save_figure(fig, ax, figure_path(output, scenario, variant, region) / "mapping_acceptance", formats, overwrite, legend)

def plot_variant_robustness(cache, output, formats, overwrite, regions):
    for region in regions:
        for target in ("shifted", "deltaR"):
            methods = [m for m in METHOD_ORDER if compatible_run_rows(cache, m, "signal_injection", "default", region) and compatible_run_rows(cache, m, "signal_injection", target, region)]
            if not methods:
                continue
            common_seeds = comparison_seed_set_across_variants(cache, methods, "signal_injection", ("default", target), region)
            if len(methods) > 1 and not common_seeds:
                continue
            fig, ax = new_figure(r"Signal efficiency, $\epsilon_S$", "SIC ratio")
            ax.axhline(1, color="0.5", ls="--", lw=0.5, zorder=1)
            drawn = 0
            signal_grid = np.linspace(0.05, 0.95, 250)
            for method in methods:
                default_rows = {row["seed"]: row for row in compatible_run_rows(cache, method, "signal_injection", "default", region, common_seeds)}
                target_rows = {row["seed"]: row for row in compatible_run_rows(cache, method, "signal_injection", target, region, common_seeds)}
                seeds = sorted(set(default_rows) & set(target_rows))
                ratios = []
                for seed in seeds:
                    curves = []
                    for row in (default_rows[seed], target_rows[seed]):
                        b = row["metrics"]["background_efficiency"]
                        s = row["metrics"]["signal_efficiency"]
                        good = np.isfinite(b) & np.isfinite(s) & (b > 0)
                        order = np.argsort(s[good])
                        sx = s[good][order]
                        sic = s[good][order] / np.sqrt(b[good][order])
                        unique, idx = np.unique(sx, return_index=True)
                        values = np.full_like(signal_grid, np.nan)
                        inside = (signal_grid >= unique.min()) & (signal_grid <= unique.max())
                        values[inside] = np.interp(signal_grid[inside], unique, sic[idx])
                        curves.append(values)
                    ratio = np.divide(curves[1], curves[0], out=np.full_like(signal_grid, np.nan), where=np.isfinite(curves[0]) & (curves[0] != 0))
                    ratios.append(ratio)
                if ratios:
                    matrix = np.asarray(ratios)
                    median = np.nanmedian(matrix, axis=0)
                    valid = np.isfinite(median)
                    ax.plot(signal_grid[valid], median[valid], color=METHOD_COLORS.get(method), ls="-", label=METHOD_LABELS.get(method, method))
                    if len(matrix) >= 2:
                        low, high = finite_column_percentiles(matrix, (16, 84))
                        ax.fill_between(signal_grid[valid], low[valid], high[valid], color=METHOD_COLORS.get(method), alpha=0.15, linewidth=0)
                    drawn += 1
            if drawn:
                legend = inside_legend(ax, title=REGION_LABELS[region])
                stem = f"sic_ratio_{target}_to_default_vs_signal_efficiency"
                save_figure(fig, ax, output / "01_comparison" / "Signal-Injected" / safe_component(REGION_LABELS[region]) / stem, formats, overwrite, legend)
            else:
                plt.close(fig)


def write_table_rows(base, rows, formats):
    if not rows:
        return
    base.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    for fmt in formats:
        path = base.with_suffix("." + fmt)
        if fmt == "csv":
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(rows)
        elif fmt == "json":
            path.write_text(json.dumps(rows, indent=2, allow_nan=False, ensure_ascii=False), encoding="utf-8")
        elif fmt == "yaml":
            path.write_text(yaml.safe_dump(rows, sort_keys=False, allow_unicode=True), encoding="utf-8")


def normalize_scalar(value):
    if value is None:
        return None
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return value if np.isfinite(value) else None
    return value


def concise_sample_name(value, validation=False):
    text = str(value or "").lower()
    if not text or "not used" in text:
        return ""
    if "sideband" in text or "outerdata" in text:
        return "SB data"
    if "signal-region" in text or "signal region" in text:
        return "SR data"
    if "internal residual validation" in text:
        return "SR validation data"
    if "latent" in text:
        return "latent-space data"
    if "background" in text and "sample" in text:
        return "background samples"
    if "synthetic" in text or "reference" in text:
        return "reference samples"
    return "validation data" if validation else "data"


def event_count_text(count, sample, validation=False):
    if count is None or count == "":
        return ""
    try:
        number = int(count)
    except Exception:
        return ""
    if number <= 0:
        return ""
    noun = concise_sample_name(sample, validation)
    return f"{number:,}" + (f" {noun}" if noun else "")


def compact_configuration_rows(extracted):
    values = {}
    for row in extracted:
        name = row.get("setting", row.get("Setting", row.get("name", "")))
        if not isinstance(name, str) or not name.startswith("settings."):
            continue
        values[name] = row.get("value", row.get("Value", ""))
    def get(name):
        return values.get(name)
    def text_value(value):
        if isinstance(value, bool):
            return "yes" if value else "no"
        text = str(value)
        aliases = {"True": "yes", "False": "no", "silu": "SiLU", "relu": "ReLU", "stein_witness": "Stein witness", "tail_focus": "tail focus", "background_cdf_power": "background CDF power"}
        return aliases.get(text, text)
    rows = []
    core = get("settings.riddle.core")
    mass_conditioning = get("settings.riddle.mass_conditioning")
    if core is not None:
        value = text_value(core)
        if mass_conditioning is not None:
            value += f"; mass conditioning: {text_value(mass_conditioning)}"
        rows.append(("Model", value))
    hidden = get("settings.riddle.stein.hidden_features")
    layers = get("settings.riddle.stein.hidden_layers")
    activation = get("settings.riddle.stein.activation")
    architecture = [part for part in (f"{layers} hidden layers" if layers is not None else None, f"{hidden} hidden features" if hidden is not None else None, text_value(activation) if activation is not None else None) if part]
    if architecture:
        rows.append(("Stein architecture", "; ".join(architecture)))
    fits = get("settings.riddle.ensemble_fit_count")
    epochs = get("settings.riddle.epochs")
    selected = get("settings.riddle.training.selected_checkpoints")
    fit_selection = get("settings.riddle.stein.ensemble_fit_selection")
    ensemble = [part for part in (f"{fits} fits" if fits is not None else None, f"{epochs} epochs" if epochs is not None else None, f"{selected} checkpoints/fit" if selected is not None else None, f"selection: {fit_selection}" if fit_selection is not None else None) if part]
    if ensemble:
        rows.append(("Stein ensemble", "; ".join(ensemble)))
    batch = get("settings.riddle.training.batch_size")
    lr = get("settings.riddle.training.learning_rate")
    wd = get("settings.riddle.training.weight_decay")
    clip = get("settings.riddle.training.gradient_clip_norm")
    optimizer = [part for part in (f"batch {batch}" if batch is not None else None, f"learning rate {lr}" if lr is not None else None, f"weight decay {wd}" if wd is not None else None, f"gradient clip {clip}" if clip is not None else None) if part]
    if optimizer:
        rows.append(("Stein optimization", "; ".join(optimizer)))
    mode = get("settings.riddle.stein.scoring.mode")
    energy = get("settings.riddle.stein.scoring.energy_weight")
    operator = get("settings.riddle.stein.scoring.operator_weight")
    gate = get("settings.riddle.stein.scoring.operator_gate_z")
    temperature = get("settings.riddle.stein.scoring.operator_temperature")
    scoring = [part for part in (text_value(mode) if mode is not None else None, f"energy weight {energy}" if energy is not None else None, f"operator weight {operator}" if operator is not None else None, f"gate z {gate}" if gate is not None else None, f"temperature {temperature}" if temperature is not None else None) if part]
    if scoring:
        rows.append(("Stein scoring", "; ".join(scoring)))
    statistic = get("settings.riddle.stein.scoring.support_guard.statistic")
    mass_bins = get("settings.riddle.stein.scoring.support_guard.mass_bins")
    quantile = get("settings.riddle.stein.scoring.support_guard.gate_quantile")
    guard_weight = get("settings.riddle.stein.scoring.support_guard.weight")
    guard_temperature = get("settings.riddle.stein.scoring.support_guard.temperature")
    enabled = get("settings.riddle.stein.scoring.support_guard.enabled")
    guard = [part for part in (f"enabled: {text_value(enabled)}" if enabled is not None else None, text_value(statistic) if statistic is not None else None, f"{mass_bins} mass bins" if mass_bins is not None else None, f"gate quantile {quantile}" if quantile is not None else None, f"weight {guard_weight}" if guard_weight is not None else None, f"temperature {guard_temperature}" if guard_temperature is not None else None) if part]
    if guard:
        rows.append(("Support guard", "; ".join(guard)))
    final_bins = get("settings.riddle.stein.scoring.final_mass_bins")
    transform = get("settings.riddle.stein.scoring.final_transform")
    power = get("settings.riddle.stein.scoring.final_power")
    calibration = [part for part in (text_value(transform) if transform is not None else None, f"{final_bins} mass bins" if final_bins is not None else None, f"power {power}" if power is not None else None) if part]
    if calibration:
        rows.append(("Final calibration", "; ".join(calibration)))
    reference_samples = get("settings.riddle.stein.scoring.reference_samples")
    reference_split = get("settings.riddle.stein.scoring.reference_split")
    reference = [part for part in (f"{reference_samples} samples" if reference_samples is not None else None, f"split {reference_split}" if reference_split is not None else None) if part]
    if reference:
        rows.append(("Stein reference", "; ".join(reference)))
    bg_model = get("settings.background.configuration.ModelType")
    bg_blocks = get("settings.background.configuration.num_blocks")
    bg_hidden = get("settings.background.configuration.num_hidden")
    bg_activation = get("settings.background.configuration.activation_function")
    background_architecture = [part for part in (text_value(bg_model) if bg_model is not None else None, f"{bg_blocks} blocks" if bg_blocks is not None else None, f"{bg_hidden} hidden units" if bg_hidden is not None else None, text_value(bg_activation) if bg_activation is not None else None) if part]
    if background_architecture:
        rows.append(("Background map", "; ".join(background_architecture)))
    bg_epochs = get("settings.background.epochs")
    bg_batch = get("settings.background.batch_size")
    bg_lr = get("settings.background.configuration.optimizer.lr")
    background_training = [part for part in (f"{bg_epochs} epochs" if bg_epochs is not None else None, f"batch {bg_batch}" if bg_batch is not None else None, f"learning rate {bg_lr}" if bg_lr is not None else None) if part]
    if background_training:
        rows.append(("Background-map training", "; ".join(background_training)))
    correction = get("settings.riddle.background_correction")
    if correction is not None:
        rows.append(("Background correction", text_value(correction)))
    return rows


def export_tables(groups, cache, loader, output, file_formats, data_root, verbose):
    tables = output / "07_tables"
    try:
        raw_usage = event_size_rows(groups, loader, data_root=data_root)
    except Exception as error:
        say(f"[WARN] Table 1 exact data usage unavailable: {error}", verbose)
        raw_usage = []
    table1 = []
    seen_usage = set()
    component_order = {"Background map": 0, "Oracle background q_phi": 1, "Background correction": 1, "Stein witness": 2, "Background density estimator": 3, "Classifier": 4}
    for row in raw_usage:
        component = str(row.get("component", row.get("Component", row.get("type", row.get("Type", "")))))
        model_type = str(row.get("model_type", row.get("Model type", "")))
        if "reserved" in component.lower() or "diagnostic" in model_type.lower():
            continue
        method = str(row.get("method", row.get("Method", "")))
        scenario = str(row.get("scenario", row.get("Scenario", "")))
        variant = str(row.get("variant", row.get("Dataset", "default")))
        training = event_count_text(row.get("train_events", row.get("Training events")), row.get("train_sample", row.get("Training sample", "")))
        validation = event_count_text(row.get("validation_events", row.get("Validation events")), row.get("validation_sample", row.get("Validation sample", "")), True)
        canonical = canonical_method(str(row.get("method_id", method)))
        reference_count = row.get("generated_reference_samples", row.get("Generated reference events"))
        reference = event_count_text(reference_count, "reference samples") if reference_count not in (None, "") else ""
        background = row.get("evaluation_background", row.get("Evaluation background"))
        signal = row.get("evaluation_signal", row.get("Evaluation signal"))
        evaluation = ""
        if background not in (None, ""):
            evaluation = f"{int(background):,} SR background"
            if scenario == "signal_injection" and signal not in (None, "") and int(signal) > 0:
                evaluation += f" + {int(signal):,} SR signal"
        output_row = {"Method": METHOD_LABELS.get(canonical_method(method), method), "Type": component or model_type, "Dataset": VARIANT_LABELS.get(variant, variant), "Scenario": SCENARIO_LABELS.get(scenario, scenario), "Training": training, "Validation": validation, "Reference pool": reference, "Evaluation": evaluation}
        identity = tuple(output_row.values())
        if identity not in seen_usage:
            seen_usage.add(identity)
            table1.append(output_row)
    table1.sort(key=lambda row: (VARIANT_ORDER.index(next((key for key, value in VARIANT_LABELS.items() if value == row["Dataset"]), "default")) if row["Dataset"] in VARIANT_LABELS.values() else 99, 0 if row["Scenario"] == "Signal-Injected" else 1, METHOD_ORDER.index(canonical_method(row["Method"].lower().replace("-", ""))) if canonical_method(row["Method"].lower().replace("-", "")) in METHOD_ORDER else 99, component_order.get(row["Type"], 99)))
    if table1 and not any(row.get("Reference pool") for row in table1):
        for row in table1:
            row.pop("Reference pool", None)
    write_table_rows(tables / "table_01_data_usage", table1, file_formats)
    configuration_values = defaultdict(lambda: defaultdict(set))
    seen_results = set()
    for identity, group in groups.items():
        if "riddle" not in group:
            continue
        root, report = group["riddle"]
        result_identity = (str(root), scientific_protocol(report))
        if result_identity in seen_results:
            continue
        seen_results.add(result_identity)
        try:
            extracted = settings_rows({"riddle": (root, report)}, loader)
        except Exception as error:
            say(f"[WARN] Table 2 configuration unavailable for {root}: {error}", verbose)
            continue
        variant = result_variant(report)
        scenario = report.get("scenario", "")
        protocol = scientific_protocol(report)
        for setting, value in compact_configuration_rows(extracted):
            configuration_values[(variant, protocol, setting)][scenario].add(str(value))
    setting_order = ["Model", "Stein architecture", "Stein ensemble", "Stein optimization", "Stein scoring", "Support guard", "Final calibration", "Stein reference", "Background map", "Background-map training", "Background correction"]
    table2 = []
    for (variant, protocol, setting), by_scenario in sorted(configuration_values.items(), key=lambda item: (VARIANT_ORDER.index(item[0][0]) if item[0][0] in VARIANT_ORDER else 99, setting_order.index(item[0][2]) if item[0][2] in setting_order else 99, item[0][2])):
        normalized = {scenario: sorted(values) for scenario, values in by_scenario.items()}
        all_single = all(len(values) == 1 for values in normalized.values())
        unique_values = {values[0] for values in normalized.values() if len(values) == 1} if all_single else set()
        if all_single and len(unique_values) == 1:
            table2.append({"Dataset": VARIANT_LABELS.get(variant, variant), "Scenario": "All selected", "Method": "RIDDLE", "Protocol": protocol, "Setting": setting, "Configuration": next(iter(unique_values))})
        else:
            for scenario, values in sorted(normalized.items()):
                if len(values) > 1:
                    say(f"[WARN] Table 2 {variant} {scenario} {setting} varies across runs", verbose)
                table2.append({"Dataset": VARIANT_LABELS.get(variant, variant), "Scenario": SCENARIO_LABELS.get(scenario, scenario), "Method": "RIDDLE", "Protocol": protocol, "Setting": setting, "Configuration": " | ".join(values)})
    write_table_rows(tables / "table_02_stein_training_configuration", table2, file_formats)
    performance = []
    for variant in VARIANT_ORDER:
        methods = [m for m in METHOD_ORDER if compatible_run_rows(cache, m, "signal_injection", variant, "signal_region")]
        common_seeds = comparison_seed_set(cache, methods, "signal_injection", variant, "signal_region")
        if len(methods) > 1 and not common_seeds:
            say(f"[SKIP] Table 3 {variant}: selected methods have no common independent seeds", verbose)
            continue
        for method in methods:
            rows = compatible_run_rows(cache, method, "signal_injection", variant, "signal_region", common_seeds)
            values = lambda getter: [getter(row["metrics"]) for row in rows if getter(row["metrics"]) is not None]
            auc = values(lambda m: m["auc"])
            sic = values(lambda m: m["max_sic"])
            wpvals = {wp: values(lambda m, wp=wp: m["working_points"].get(wp)) for wp in WORKING_POINTS}
            performance.append({"Dataset": VARIANT_LABELS.get(variant, variant), "Method": METHOD_LABELS.get(method, method), "AUC": normalize_scalar(np.median(auc) if auc else None), "Max. SIC": normalize_scalar(np.median(sic) if sic else None), "εS@10%": normalize_scalar(np.median(wpvals[0.10]) if wpvals[0.10] else None), "εS@5%": normalize_scalar(np.median(wpvals[0.05]) if wpvals[0.05] else None), "εS@1%": normalize_scalar(np.median(wpvals[0.01]) if wpvals[0.01] else None), "εS@0.5%": normalize_scalar(np.median(wpvals[0.005]) if wpvals[0.005] else None), "εS@0.4%": normalize_scalar(np.median(wpvals[0.004]) if wpvals[0.004] else None)})
    write_table_rows(tables / "table_03_performance_summary", performance, file_formats)


def scan_identity_data(scan_groups, methods):
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for identity, group in scan_groups.items():
        signal_events, replica, run_index, variant = identity
        if variant != "default":
            continue
        for method, source in group.items():
            family = canonical_method(method)
            if family not in methods:
                continue
            runs = result[family][signal_events][replica]
            if run_index in runs:
                raise ValueError(f"Duplicate injection-scan result for {METHOD_LABELS.get(family, family)}, N_inj={signal_events}, replica={replica}, run={run_index}")
            runs[run_index] = source
    return result


def scan_realized_counts(report):
    contract = report.get("contract", {})
    inputs = contract.get("inputs", {})
    candidates = [inputs.get("uncut_signal_region"), report.get("uncut_signal_region"), report.get("metrics", {}).get("uncut_signal_region"), contract.get("uncut_signal_region")]
    for candidate in candidates:
        if isinstance(candidate, dict) and "background" in candidate and "signal" in candidate:
            background = int(candidate["background"])
            signal = int(candidate["signal"])
            if background < 0 or signal < 0:
                raise ValueError("Invalid uncut signal-region population counts")
            return background, signal
    return None


def scan_method_run_indices(method_data, configured, replicas):
    indices = set()
    for level in configured:
        for replica in range(replicas):
            indices.update(method_data.get(level, {}).get(replica, {}))
    return sorted(indices)


def aggregate_scan_replica(run_rows):
    counts = {(row["background"], row["signal"]) for row in run_rows}
    if len(counts) != 1:
        raise ValueError("Independent runs of one injection replica disagree on the uncut physical population")
    background, signal = next(iter(counts))
    working_points = {}
    for wp in WORKING_POINTS:
        values = [row["metrics"]["working_points"].get(wp) for row in run_rows if row["metrics"]["working_points"].get(wp) is not None]
        working_points[wp] = None if not values else float(np.median(values))
    auc = [row["metrics"]["auc"] for row in run_rows if row["metrics"]["auc"] is not None]
    max_sic = [row["metrics"]["max_sic"] for row in run_rows if row["metrics"]["max_sic"] is not None]
    return {
        "metrics": {
            "auc": None if not auc else float(np.median(auc)),
            "max_sic": None if not max_sic else float(np.median(max_sic)),
            "working_points": working_points,
        },
        "background": background,
        "signal": signal,
        "s_over_b": signal / background if background else None,
        "nominal": signal / math.sqrt(background) if background else None,
        "runs": len(run_rows),
    }


def process_injection_scan(scan_groups, methods, loader, settings, output, formats, file_formats, overwrite, min_background, allow_partial, verbose, require_compatible_populations=False):
    configured = [int(value) for value in settings.get("injection_scan", {}).get("signal_events", [])]
    replicas = int(settings.get("injection_scan", {}).get("replicas", 0))
    if not configured or replicas < 1:
        say("[SKIP] Injection scan: configuration is missing signal_events or replicas", verbose)
        return
    if not scan_groups:
        say("[SKIP] Injection scan: no completed scan results were discovered", verbose)
        return
    data = scan_identity_data(scan_groups, methods)
    if allow_partial:
        requested_methods = [method for method in methods if method in data]
    else:
        missing_methods = [method for method in methods if method not in data]
        if missing_methods:
            labels = ", ".join(METHOD_LABELS.get(method, method) for method in missing_methods)
            raise ValueError(f"Strict injection-scan comparison is missing requested methods: {labels}")
        requested_methods = list(methods)
    expected_runs = {method: scan_method_run_indices(data[method], configured, replicas) for method in requested_methods}
    if not allow_partial and len(requested_methods) > 1:
        cohorts = {tuple(expected_runs[method]) for method in requested_methods}
        if len(cohorts) != 1:
            raise ValueError("Injection-scan methods do not share the same independent run-index cohort")
    method_reports = defaultdict(list)
    invalid = defaultdict(list)
    loaded = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    point_records = defaultdict(dict)
    point_counts = defaultdict(set)
    for method in requested_methods:
        runs_expected = expected_runs[method]
        if not runs_expected:
            invalid[method].append("no independent runs discovered")
            continue
        for level in configured:
            for replica in range(replicas):
                runs = data[method].get(level, {}).get(replica, {})
                if set(runs) != set(runs_expected):
                    missing = sorted(set(runs_expected) - set(runs))
                    extra = sorted(set(runs) - set(runs_expected))
                    invalid[method].append(f"N_inj={level} replica={replica} missing_runs={missing} extra_runs={extra}")
                for run_index, (root, report) in sorted(runs.items()):
                    method_reports[method].append(report)
                    try:
                        record = loader(root, report, "signal_region")
                        metrics = central_metrics(record, min_background)
                        if metrics is None:
                            raise ValueError("signal metrics are unavailable")
                        counts = scan_realized_counts(report)
                        if counts is None:
                            raise ValueError("uncut_signal_region provenance is missing")
                        background, signal = counts
                        loaded[method][level][replica][run_index] = {
                            "metrics": metrics,
                            "background": background,
                            "signal": signal,
                        }
                        point_records[(level, replica, run_index)][method] = record
                        point_counts[(level, replica)].add(counts)
                    except Exception as error:
                        invalid[method].append(f"N_inj={level} replica={replica} run={run_index}: {type(error).__name__}: {error}")
    for method, reports in method_reports.items():
        signatures = {aggregation_protocol_signature(method, report) for report in reports}
        if len(signatures) > 1:
            invalid[method].append("incompatible scientific protocols across scan points/runs")
    for identity, group in scan_groups.items():
        level, replica, run_index, variant = identity
        if variant != "default" or level not in configured or replica not in range(replicas):
            continue
        normalized = {canonical_method(method): source for method, source in group.items() if canonical_method(method) in requested_methods}
        if len(normalized) > 1:
            require_riddle_benchmark_alignment(normalized)
    if require_compatible_populations:
        for (level, replica, run_index), records in point_records.items():
            if len(records) > 1:
                try:
                    require_population_compatibility(records)
                except ValueError as error:
                    raise ValueError(f"Incompatible injection-scan evaluation population for N_inj={level}, replica={replica}, run={run_index}") from error
    for (level, replica), counts in point_counts.items():
        if len(counts) > 1:
            raise ValueError(f"Injection-scan methods disagree on uncut physical counts for N_inj={level}, replica={replica}")
    summaries = {}
    if not allow_partial:
        failures = [f"{METHOD_LABELS.get(method, method)}: " + "; ".join(invalid[method]) for method in requested_methods if invalid[method]]
        if failures:
            raise ValueError("Strict injection-scan comparison is incomplete: " + " | ".join(failures))
    for method in requested_methods:
        if invalid[method] and allow_partial:
            say("[WARN] Partial injection scan: " + METHOD_LABELS.get(method, method) + ": " + "; ".join(invalid[method]), verbose)
        levels = []
        for level in configured:
            replicas_data = []
            for replica in range(replicas):
                run_rows = list(loaded[method].get(level, {}).get(replica, {}).values())
                if not run_rows:
                    continue
                replicas_data.append(aggregate_scan_replica(run_rows))
            if replicas_data:
                levels.append((level, replicas_data))
        if levels and (allow_partial or (len(levels) == len(configured) and all(len(rows) == replicas for _, rows in levels))):
            summaries[method] = levels
    if not allow_partial and set(summaries) != set(requested_methods):
        missing = [method for method in requested_methods if method not in summaries]
        labels = ", ".join(METHOD_LABELS.get(method, method) for method in missing)
        raise ValueError(f"Strict injection-scan comparison could not build complete summaries for: {labels}")
    if not summaries:
        return
    destination = output / "06_injection_scan" / "Default"
    figure_specs = [("max_sic", "Maximum significance improvement", "maximum_sic_vs_s_over_b"), ("nominal_selected", "Maximum nominal significance after anomaly selection", "maximum_nominal_significance_vs_s_over_b"), ("auc", "AUC", "auc_vs_s_over_b")]
    for wp in PLOT_WORKING_POINTS[::-1]:
        figure_specs.append((f"wp_{wp}", r"Signal efficiency, $\epsilon_S$", f"epsS_{WORKING_POINT_LABELS[wp].replace('%','pct').replace('.','p')}_vs_s_over_b"))
    reference_axis = None
    for method, levels in summaries.items():
        axis = []
        for level, replicas_data in levels:
            sob = [row["s_over_b"] for row in replicas_data if row["s_over_b"] is not None]
            nominal = [row["nominal"] for row in replicas_data if row["nominal"] is not None]
            if sob and nominal:
                axis.append((level, 100 * float(np.median(sob)), float(np.median(nominal))))
        if reference_axis is None:
            reference_axis = axis
        elif axis != reference_axis:
            raise ValueError("Injection-scan methods do not share the same realized S/B and S/sqrt(B) axis")
    for field, ylabel, stem in figure_specs:
        fig, ax = new_figure("Injected SR S/B (%)", ylabel)
        drawn = 0
        for method, levels in summaries.items():
            x, y, low, high = [], [], [], []
            for level, replicas_data in levels:
                sob = [row["s_over_b"] for row in replicas_data if row["s_over_b"] is not None]
                if not sob:
                    continue
                if field == "max_sic":
                    values = [row["metrics"]["max_sic"] for row in replicas_data]
                elif field == "nominal_selected":
                    values = [row["metrics"]["max_sic"] * row["nominal"] for row in replicas_data if row["metrics"]["max_sic"] is not None and row["nominal"] is not None]
                elif field == "auc":
                    values = [row["metrics"]["auc"] for row in replicas_data]
                else:
                    wp = float(field.split("_", 1)[1])
                    values = [row["metrics"]["working_points"].get(wp) for row in replicas_data if row["metrics"]["working_points"].get(wp) is not None]
                values = [value for value in values if value is not None]
                if not values:
                    continue
                x.append(100 * float(np.median(sob)))
                y.append(float(np.median(values)))
                low.append(float(np.percentile(values, 16)))
                high.append(float(np.percentile(values, 84)))
            if x:
                order = np.argsort(x)
                x, y, low, high = [np.asarray(values)[order] for values in (x, y, low, high)]
                ax.plot(x, y, color=METHOD_COLORS.get(method), ls="-", label=METHOD_LABELS.get(method, method))
                ax.fill_between(x, low, high, color=METHOD_COLORS.get(method), alpha=0.15, linewidth=0)
                drawn += 1
        if drawn:
            top = ax.twiny()
            top.set_xlim(ax.get_xlim())
            if reference_axis:
                ordered_axis = sorted(reference_axis, key=lambda item: item[1])
                top.set_xticks([item[1] for item in ordered_axis])
                top.set_xticklabels([f"{item[2]:.2g}" for item in ordered_axis])
            top.set_xlabel(r"Uncut $S/\sqrt{B}$")
            legend = inside_legend(ax, title="Default")
            save_figure(fig, ax, destination / stem, formats, overwrite, legend)
        else:
            plt.close(fig)
    if "riddle" in summaries:
        rows = []
        for level, replicas_data in summaries["riddle"]:
            values = lambda getter: [getter(row["metrics"]) for row in replicas_data if getter(row["metrics"]) is not None]
            auc = values(lambda m: m["auc"])
            sic = values(lambda m: m["max_sic"])
            wpvals = {wp: values(lambda m, wp=wp: m["working_points"].get(wp)) for wp in WORKING_POINTS}
            rows.append({"N_inj": int(level), "AUC": normalize_scalar(np.median(auc) if auc else None), "Max. SIC": normalize_scalar(np.median(sic) if sic else None), "εS@10%": normalize_scalar(np.median(wpvals[0.10]) if wpvals[0.10] else None), "εS@5%": normalize_scalar(np.median(wpvals[0.05]) if wpvals[0.05] else None), "εS@1%": normalize_scalar(np.median(wpvals[0.01]) if wpvals[0.01] else None), "εS@0.5%": normalize_scalar(np.median(wpvals[0.005]) if wpvals[0.005] else None), "εS@0.4%": normalize_scalar(np.median(wpvals[0.004]) if wpvals[0.004] else None)})
        if rows and (allow_partial or (len(rows) == len(configured) and all(len(replicas_data) == replicas for _, replicas_data in summaries["riddle"]))):
            write_table_rows(output / "07_tables" / "table_04_default_signal_injection_dependence", rows, file_formats)


def load_settings_file(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Settings YAML must contain a mapping")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Physical Review D publication plotting for completed RIDDLE-compatible results")
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--scan-results", type=Path, default=Path("results/injection_scan"))
    parser.add_argument("--data", type=Path, default=Path("data/lhco"))
    parser.add_argument("--scan-data", type=Path, default=Path("data/injection_scan"))
    parser.add_argument("--config", type=Path, default=Path("config/settings.yaml"))
    parser.add_argument("--output", type=Path, default=Path("paper_plots"))
    parser.add_argument("--methods", nargs="+")
    parser.add_argument("--variants", nargs="+", choices=("default", "shifted", "deltaR"))
    parser.add_argument("--scenarios", nargs="+", choices=("signal_injection", "background_only"))
    parser.add_argument("--regions", nargs="+", choices=("signal_region",), default=("signal_region",))
    parser.add_argument("--plot-formats", nargs="+", choices=("pdf", "png"), default=("pdf", "png"))
    parser.add_argument("--file-formats", nargs="+", choices=("csv", "json", "yaml"), default=("csv",))
    parser.add_argument("--exclude-seeds", nargs="*", type=int, default=())
    parser.add_argument("--min-background", type=int, default=10)
    parser.add_argument("--allow-partial-injection-scan", action="store_true")
    parser.add_argument("--require-compatible-populations", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    configure_style()
    if args.overwrite:
        cleanup_legacy_summary_folders(args.output)
    settings = load_settings_file(args.config)
    say("[STAGE] Discover completed results", args.verbose)
    groups_all = discover(args.results, requested=None, scan=False) if args.results.exists() else {}
    scan_all = discover(args.scan_results, requested=None, scan=True) if args.scan_results.exists() else {}
    methods = select_requested_methods(groups_all, args.methods)
    if not methods:
        raise SystemExit("No requested publication methods were discovered")
    discovered_variants = sorted({identity[2] for identity in groups_all}, key=lambda value: VARIANT_ORDER.index(value) if value in VARIANT_ORDER else 99)
    variants = list(args.variants) if args.variants else discovered_variants
    scenarios = list(args.scenarios) if args.scenarios else sorted({identity[0] for identity in groups_all}, key=lambda value: ("signal_injection", "background_only").index(value))
    regions = ["signal_region"]
    groups = filter_groups(groups_all, methods, variants, scenarios, set(args.exclude_seeds))
    if not groups:
        raise SystemExit("No completed results remain after applying method, variant, scenario, and seed filters")
    say("[STAGE] Validate result provenance", args.verbose)
    loader = ScoreLoader(safeguard_filtering=True, ensemble_fit_selection=True, io_workers=2, device="cpu")
    say("[STAGE] Build publication metric cache", args.verbose)
    cache, records = metric_cache(groups, loader, regions, args.min_background, args.verbose, args.require_compatible_populations)
    say("[STAGE] Render comparison performance plots", args.verbose)
    plot_performance(cache, args.output, args.plot_formats, args.overwrite, scenarios, variants, regions)
    plot_score_distributions(groups, records, args.output, args.plot_formats, args.overwrite)
    plot_stability(cache, args.output, args.plot_formats, args.overwrite, regions)
    plot_acceptance(cache, args.output, args.plot_formats, args.overwrite)
    plot_variant_robustness(cache, args.output, args.plot_formats, args.overwrite, regions)
    say("[STAGE] Render mass-sculpting plots", args.verbose)
    plot_mass_sculpting(cache, records, args.output, args.plot_formats, args.overwrite, scenarios, variants, regions, args.verbose)
    say("[STAGE] Render feature diagnostics", args.verbose)
    plot_features(records, args.output, args.plot_formats, args.overwrite, args.verbose)
    plot_score_vs_mass(records, args.output, args.plot_formats, args.overwrite)
    say("[STAGE] Render latent diagnostics", args.verbose)
    plot_latent_closure(groups, records, args.output, args.plot_formats, args.overwrite, args.verbose)
    say("[STAGE] Render Stein-witness diagnostics", args.verbose)
    plot_stein(groups, records, args.output, args.plot_formats, args.overwrite)
    say("[STAGE] Render training histories", args.verbose)
    plot_training(groups, args.output, args.plot_formats, args.overwrite)
    say("[STAGE] Render signal-injection dependence", args.verbose)
    process_injection_scan(scan_all, methods, loader, settings, args.output, args.plot_formats, args.file_formats, args.overwrite, args.min_background, args.allow_partial_injection_scan, args.verbose, args.require_compatible_populations)
    say("[STAGE] Export publication tables", args.verbose)
    export_tables(groups, cache, loader, args.output, args.file_formats, args.data, args.verbose)
    say(f"[DONE] Publication outputs written to {args.output}", args.verbose)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
