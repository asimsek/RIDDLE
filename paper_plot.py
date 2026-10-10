import argparse
import csv
import json
import math
import multiprocessing
import os
import pickle
import re
import shutil
import sys
import time
from collections import defaultdict
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap, LogNorm, TwoSlopeNorm
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import LogLocator, NullFormatter
import numpy as np
import torch
import yaml
from scipy.special import ndtri
from sklearn.metrics import roc_curve
from threadpoolctl import threadpool_limits
from tqdm import tqdm

from riddle.evaluation import common_acceptance_auc, riddle_score_scope
from riddle.metrics import efficiency_curve, oracle_metrics
from riddle.plotting import ScoreLoader, discover, event_size_rows, exact_background_selection, method_family, read_discovery_result, read_metadata, result_paths, result_variant, scientific_protocol, settings_rows, verify_plot_input
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
STYLE = {"font.family": "sans-serif", "font.sans-serif": ["DejaVu Sans"], "mathtext.fontset": "dejavusans", "text.usetex": False, "font.size": 8.2, "axes.labelsize": 8.8, "axes.linewidth": 0.8, "xtick.labelsize": 7.7, "ytick.labelsize": 7.7, "legend.fontsize": 7.3, "legend.title_fontsize": 7.3, "lines.linewidth": 1.0, "lines.markersize": 3.2, "xtick.direction": "in", "ytick.direction": "in", "xtick.top": True, "ytick.right": True, "xtick.minor.visible": True, "ytick.minor.visible": True, "pdf.fonttype": 42, "ps.fonttype": 42, "savefig.dpi": 600, "savefig.facecolor": "white", "savefig.edgecolor": "white", "figure.facecolor": "white", "axes.facecolor": "white"}
_FIGURE_RENDERER = ContextVar("paper_figure_renderer", default=None)


def say(message, verbose=1, level=1):
    if verbose >= level:
        print(message, flush=True)


def incomplete_scan_error(message):
    hint = "For preliminary plots using available completed results, add --allow-partial-injection-scan."
    if sys.stderr.isatty() and "NO_COLOR" not in os.environ:
        hint = f"\x1b[1;91m{hint}\x1b[0m"
    return ValueError(f"{message}\n{hint}")


def canonical_method(value):
    return METHOD_ALIASES.get(value, value)


def safe_component(value):
    text = str(value).strip().replace(" ", "-")
    text = re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-.")
    return text or "item"


def folder_variant(variant):
    return VARIANT_LABELS.get(variant, safe_component(variant))


def variant_output(output, variant):
    return Path(output) / ("nominal" if variant == "default" else variant)


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
    options = dict(title=title, ncol=ncol, fontsize=fontsize, borderaxespad=borderaxespad,
                   allow_headroom=allow_headroom)
    renderer = _FIGURE_RENDERER.get()
    if renderer is not None and renderer.pool is not None:
        return {"deferred_legend": options}
    return _inside_legend(ax, **options)


def _inside_legend(ax, title=None, ncol=None, fontsize=None, borderaxespad=0.5, allow_headroom=True):
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
    if score >= 1000 and columns > 1:
        return _inside_legend(ax, title=title, ncol=columns - 1, fontsize=font,
                              borderaxespad=borderaxespad, allow_headroom=allow_headroom)
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


def _scan_legend_overlap(ax, legend):
    renderer = ax.figure.canvas.get_renderer()
    box = legend.get_window_extent(renderer=renderer).padded(2)
    intersections = sum(line.get_path().transformed(line.get_transform()).intersects_bbox(box, filled=False)
                        for line in ax.lines if line.get_visible())
    intersections += sum(any(path.transformed(collection.get_transform()).intersects_bbox(box, filled=True)
                             for path in collection.get_paths())
                         for collection in ax.collections if collection.get_visible())
    return intersections


def _scan_legend(ax, title=None):
    options = dict(title=title, frameon=True, framealpha=0.93, borderpad=0.42,
                   labelspacing=0.32, handlelength=1.5, fontsize=STYLE["legend.fontsize"] - 0.3)
    for location in ("upper right", "upper left", "lower right", "lower left", "center right",
                     "center left", "upper center", "lower center"):
        legend = ax.legend(loc=location, **options)
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        box = legend.get_window_extent(renderer=renderer)
        axes_box = ax.get_window_extent(renderer=renderer)
        if (box.x0 >= axes_box.x0 and box.x1 <= axes_box.x1
                and box.y0 >= axes_box.y0 and box.y1 <= axes_box.y1
                and _scan_legend_overlap(ax, legend) == 0):
            return legend
        legend.remove()
    legend = ax.legend(loc="upper right", **options)
    for _ in range(3):
        ax.figure.canvas.draw()
        renderer = ax.figure.canvas.get_renderer()
        box = legend.get_window_extent(renderer=renderer)
        axes_box = ax.get_window_extent(renderer=renderer)
        available = (box.y0 - 6 - axes_box.y0) / axes_box.height
        if available <= 0:
            raise RuntimeError("Injection-scan legend leaves no space for the curves")
        ymin, ymax = ax.get_ylim()
        peak = ax.dataLim.ymax
        required = ymin + (peak - ymin) / available
        if not np.isfinite(required) or required <= ymax:
            if _scan_legend_overlap(ax, legend):
                raise RuntimeError("Injection-scan legend overlaps the curves")
            return legend
        ax.set_ylim(ymin, required * (1 + 1e-12))
    raise RuntimeError("Injection-scan legend could not be separated from the curves")


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
    for variant in VARIANT_ORDER:
        for extension in ("csv", "json", "yaml"):
            (variant_output(output, variant) / "07_tables" /
             f"table_05_{variant}_signal_injection_score_stages.{extension}").unlink(missing_ok=True)
    root = output / "01_comparison" / "Signal-Injected"
    for name in ("Signal-region", "Full-region", "Dataset-Variant-Summary"):
        path = root / name
        if path.is_dir():
            shutil.rmtree(path)
    if output.is_dir():
        for method in METHOD_LABELS.values():
            for stem in ("background_nll", "oracle_background_nll", "stein_witness_objective", "classifier_bce"):
                for extension in ("png", "pdf"):
                    path = output / "02_training" / method / f"{stem}.{extension}"
                    if path.is_file():
                        path.unlink()
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
    try:
        renderer = _FIGURE_RENDERER.get()
        if renderer is not None:
            renderer.save(fig, ax, stem, formats, overwrite, legend)
        else:
            _save_figure(fig, ax, stem, formats, overwrite, legend)
    finally:
        plt.close(fig)


def _save_figure(fig, ax, stem, formats, overwrite, legend=None):
    spectrum_legend = legend if isinstance(legend, dict) and "spectrum_legend" in legend else None
    scan_legend = legend if isinstance(legend, dict) and "scan_legend" in legend else None
    if isinstance(legend, dict) and "deferred_legend" in legend:
        legend = _inside_legend(ax, **legend["deferred_legend"])
    stem.parent.mkdir(parents=True, exist_ok=True)
    if not fig.get_constrained_layout() and (spectrum_legend is None or len(fig.axes) == 1):
        try:
            fig.tight_layout(pad=0.45)
        except Exception:
            pass
    if scan_legend is not None:
        legend = _scan_legend(ax, **scan_legend["scan_legend"])
    if spectrum_legend is not None:
        from riddle.mass_spectrum import mass_legend

        legend = mass_legend(ax, **spectrum_legend["spectrum_legend"])
    audit_layout(fig, ax, legend)
    for extension in formats:
        path = stem.with_suffix("." + extension)
        if path.exists() and not overwrite:
            raise FileExistsError(f"Output exists: {path}")
        kwargs = {"bbox_inches": "tight", "facecolor": "white"}
        if extension == "png":
            kwargs["dpi"] = 600
        fig.savefig(path, **kwargs)


def _render_worker_init():
    torch.set_num_threads(1)
    threadpool_limits(limits=1)


def _render_figure(payload, stem, formats, overwrite, style):
    started = time.monotonic()
    with matplotlib.rc_context(style):
        fig, ax, legend = pickle.loads(payload)
        try:
            _save_figure(fig, ax, stem, formats, overwrite, legend)
        finally:
            plt.close(fig)
    return time.monotonic() - started, os.getpid()


class PaperFigureRenderer:
    def __init__(self, workers, verbose):
        if workers < 1:
            raise ValueError("Plot workers must be positive")
        self.pool = (ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"),
                                         initializer=_render_worker_init) if workers > 1 else None)
        self.pending = {}
        self.destinations = {}
        self.pending_bytes = 0
        self.max_pending = 2 * workers
        self.max_bytes = 128 * 1024**2
        self.verbose = verbose
        self.submitted = 0
        self.completed = 0
        self.saved_plots = set()
        self.saved_files = set()
        self.started = time.monotonic()
        self.last_update = self.started
        self.last_report = None
        say(f"[RENDER] workers={workers}", verbose)

    def record_saved(self, paths):
        self.saved_files.update(paths)
        self.saved_plots.update(str(Path(path).with_suffix("")) for path in paths)

    def report(self, force=False):
        now = time.monotonic()
        state = (len(self.saved_plots), len(self.saved_files), len(self.pending))
        if not self.saved_files and not self.pending:
            return
        if state == self.last_report and (force or not self.pending):
            return
        if force or now - self.last_update >= 10:
            say(f"[RENDER] {len(self.saved_plots)} unique plots saved; {len(self.pending)} pending; wall {now - self.started:.1f}s", self.verbose)
            say(f"[RENDER TASKS] {self.completed}/{self.submitted} operations completed", self.verbose, 2)
            self.last_update = now
            self.last_report = state

    def summary(self, output=None):
        if output is not None:
            for variant in VARIANT_ORDER:
                directory = variant_output(output, variant).resolve()
                files = {path for path in self.saved_files if Path(path).is_relative_to(directory)}
                if files:
                    plots = {str(Path(path).with_suffix("")) for path in files}
                    say(f"[PLOTS] {directory.name}: {len(plots)} plots; {len(files)} files", self.verbose)
        say(f"[PLOTS] Total saved: {len(self.saved_plots)} unique plots; {len(self.saved_files)} plot files", self.verbose)

    def finish(self, futures):
        for future in sorted(futures, key=lambda value: self.pending[value][3]):
            stem, paths, size, index = self.pending.pop(future)
            self.pending_bytes -= size
            for path in paths:
                if self.destinations.get(path) is future:
                    del self.destinations[path]
            try:
                elapsed, pid = future.result()
            except Exception as error:
                raise RuntimeError(f"Figure rendering failed for {stem}: {error}") from error
            self.completed += 1
            self.record_saved(paths)
            say(f"[RENDER {index}] {stem}; {elapsed:.1f}s; pid={pid}", self.verbose, 2)
        self.report()

    def wait_one(self):
        done, _ = wait(self.pending, timeout=5, return_when=FIRST_COMPLETED)
        self.finish(done)

    def save(self, fig, ax, stem, formats, overwrite, legend):
        paths = tuple(str(stem.with_suffix("." + extension).resolve()) for extension in formats)
        if self.pool is None:
            _save_figure(fig, ax, stem, formats, overwrite, legend)
            self.submitted += 1
            self.completed += 1
            self.record_saved(paths)
            self.report()
            return
        self.finish({future for future in self.pending if future.done()})
        dependencies = {self.destinations[path] for path in paths if path in self.destinations}
        while dependencies:
            done, _ = wait(dependencies, timeout=5, return_when=FIRST_COMPLETED)
            self.finish(done)
            dependencies -= done
        if not overwrite:
            for path in paths:
                if Path(path).exists():
                    raise FileExistsError(f"Output exists: {path}")
        try:
            payload = pickle.dumps((fig, ax, legend), protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PicklingError, AttributeError, TypeError):
            say(f"[RENDER] Serial fallback for {stem}", self.verbose, 2)
            _save_figure(fig, ax, stem, formats, overwrite, legend)
            self.submitted += 1
            self.completed += 1
            self.record_saved(paths)
            self.report()
            return
        while self.pending and (len(self.pending) >= self.max_pending
                                or self.pending_bytes + len(payload) > self.max_bytes):
            self.wait_one()
        future = self.pool.submit(_render_figure, payload, stem, tuple(formats), overwrite, dict(matplotlib.rcParams))
        self.submitted += 1
        self.pending[future] = (stem, paths, len(payload), self.submitted)
        self.pending_bytes += len(payload)
        self.destinations.update({path: future for path in paths})

    def flush(self):
        while self.pending:
            self.wait_one()
        self.report(force=True)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)


@contextmanager
def render_resources(workers, verbose, output=None):
    previous_threads = torch.get_num_threads()
    renderer = PaperFigureRenderer(workers, verbose)
    token = _FIGURE_RENDERER.set(renderer)
    try:
        if workers > 1:
            torch.set_num_threads(1)
        with threadpool_limits(limits=1 if workers > 1 else None):
            yield renderer
            renderer.flush()
            renderer.summary(output)
    finally:
        _FIGURE_RENDERER.reset(token)
        try:
            renderer.close()
        finally:
            torch.set_num_threads(previous_threads)


def verify_requested_artifact(root, report, relative):
    relative = str(relative)
    artifacts = report.get("artifacts_sha256", {})
    if relative not in artifacts:
        return None
    return verify_plot_input(root, report, relative)


def select_requested_methods(all_groups, requested, *, allow_missing=False, injection_scan=False):
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
        if missing and not allow_missing:
            labels = ", ".join(METHOD_LABELS.get(method, method) for method in missing)
            message = f"Explicitly requested publication methods were not discovered: {labels}"
            if injection_scan and available:
                raise incomplete_scan_error(message)
            raise ValueError(message)
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


def discover_paper_results(root, requested=None, *, scan=False, workers=1, verbose=0):
    root = Path(root)
    requested = None if requested is None else {canonical_method(method) for method in requested}
    last_update = 0.0

    def progress(directories, reports, elapsed, finished):
        nonlocal last_update
        if finished or elapsed - last_update >= 10:
            state = "Finished" if finished else "Searching"
            say(f"[DISCOVER] {state}: {directories} folders, {reports} reports; {elapsed:.1f}s", verbose)
            last_update = elapsed

    paths = list(result_paths(root, requested, workers=workers, progress=progress))

    def read(path):
        say(f"[REPORTS] {path}", verbose, 2)
        value = read_discovery_result(path, root, requested, scan=scan, independent_lacathode=True)
        if value is not None:
            value[2]["_full_mass_root"] = str(path.parent)
        return value

    reports = parallel_plot_tasks(read, paths, workers, "REPORTS", verbose)
    return discover(root, requested=requested, scan=scan, reports=reports)


def discover_variant_results(root, requested, variants, *, scan=False, workers=1, verbose=0):
    root = Path(root)
    roots = [root]
    if not any(root.name.endswith("_" + variant) for variant in ("deltaR", "shifted")):
        roots.extend(root.with_name(root.name + "_" + variant)
                     for variant in (variants or VARIANT_ORDER) if variant != "default")
    groups = {}
    for source in dict.fromkeys(roots):
        if not source.is_dir():
            continue
        found = discover_paper_results(source, requested, scan=scan, workers=workers, verbose=verbose)
        for identity, group in found.items():
            current = groups.setdefault(identity, {})
            for method, value in group.items():
                if method in current and current[method][0].resolve() != value[0].resolve():
                    raise ValueError(f"Duplicate {method} result for {identity} in the selected variant folders")
                current[method] = value
    return groups


def plot_dijet_spectra(groups, args, output, *, scan=False):
    from riddle.full_mass import load_full_mass
    from riddle.mass_spectrum import CUTS, mass_edges, match_ids, normalization_weights, residual_limits, spectrum_histograms, spectrum_summary, spectrum_figure, spectrum_cuts_figure

    tasks = [(identity, method, root, report) for identity, group in sorted(groups.items())
             for method, (root, report) in sorted(group.items()) if method in ("riddle", "iad", "supervised")]

    def load(task):
        identity, method, root, report = task
        derived_root = Path(report.get("_full_mass_root", root))
        record = load_full_mass(derived_root, report, workers=1)
        if record is None and derived_root != root:
            record = load_full_mass(root, report, workers=1)
            derived_root = root
        if record is not None and not scan and identity[0] == "background_only":
            with np.load(derived_root / "full_mass/reference_D.npz", allow_pickle=False) as archive:
                scores = np.asarray(archive["scores"], float)
                threshold = np.asarray(archive["threshold_indices"])
                assessment = np.asarray(archive["assessment_indices"])
            indices = np.r_[threshold, assessment]
            if (scores.ndim != 1 or not np.isfinite(scores).all()
                    or threshold.ndim != 1 or assessment.ndim != 1
                    or not np.issubdtype(indices.dtype, np.integer)
                    or not np.array_equal(np.sort(indices), np.arange(len(scores)))
                    or min(len(threshold), len(assessment)) < 1024):
                raise ValueError("Invalid independent background assessment split")
            record["background_count_reference"] = (scores[threshold], scores[assessment])
        return identity, method, record

    loaded = parallel_plot_tasks(load, tasks, args.io_workers, "SPECTRA LOAD", args.verbose)
    cohorts = defaultdict(list)
    populations = {}
    missing = 0
    for identity, method, record in loaded:
        if record is None:
            missing += 1
            continue
        scenario = "signal_injection" if scan else identity[0]
        level = identity[0] if scan else None
        population_key = (scenario, level)
        if population_key in populations:
            reference = populations[population_key]
            if len(record["event_ids"]) != len(reference["event_ids"]):
                raise ValueError("Full-mass method comparisons require identical primary held-out populations")
            indices = match_ids(record["event_ids"], reference["event_ids"])
            if (not np.array_equal(record["mass"][indices], reference["mass"])
                    or not np.array_equal(record["labels"][indices], reference["labels"])):
                raise ValueError("Full-mass method comparisons require identical primary held-out populations")
        else:
            populations[population_key] = record
        cohorts[(method, scenario, level)].append(record)
    if missing:
        say(f"[SPECTRA] {missing} results need --score-sidebands-only before full-mass plotting", args.verbose)
    for (method, scenario, level), records in cohorts.items():
        first = records[0]
        statuses = sorted({record["manifest"]["validation_status"] for record in records})
        if statuses != ["passed"]:
            say(f"[SPECTRA] {METHOD_LABELS[method]} {scenario}: closure {', '.join(statuses)}; diagnostic spectra", args.verbose)
        for record in records[1:]:
            indices = match_ids(record["event_ids"], first["event_ids"])
            if (not np.array_equal(record["mass"][indices], first["mass"])
                    or not np.array_equal(record["labels"][indices], first["labels"])):
                raise ValueError("Dijet seed bands require identical primary held-out event populations")
        edges = mass_edges(first["mass"])
        rows = []
        units = None
        for record in records:
            weights, record_units, _ = normalization_weights(record, getattr(args, "cross_section_weights", None))
            if units is not None and units != record_units:
                raise ValueError("Inconsistent mass-spectrum normalization")
            units = record_units
            rows.append(spectrum_histograms(record, weights, edges))
        destination = output / "04_dijet_spectra" / METHOD_LABELS[method] / SCENARIO_LABELS[scenario]
        if level is not None:
            destination = destination / f"N_{int(level):06d}"
        summaries = {cut: spectrum_summary(rows, cut) for cut in (None, *CUTS)}
        shared_limits = residual_limits(summaries)
        if args.overwrite:
            for extension in ("csv", "json", "yaml"):
                (destination / f"dijet_mass_spectra.{extension}").unlink(missing_ok=True)
        fig, ax = spectrum_cuts_figure(summaries, edges, units)
        save_figure(fig, ax, destination / "dijet_mass_score_cuts", args.plot_formats,
                    args.overwrite, {"spectrum_legend": {}})
        for cut in (None, *CUTS):
            fig, ax = spectrum_figure(summaries[cut], edges, cut, units,
                                      residual_ylim=shared_limits if cut is not None else None)
            stem = "dijet_mass_uncut" if cut is None else f"dijet_mass_{cut * 100:g}pct".replace("0.5pct", "0p5pct")
            save_figure(fig, ax, destination / stem, args.plot_formats, args.overwrite,
                        {"spectrum_legend": {}})
        if scenario == "background_only":
            plot_background_count_closure(method, records, args, output)


def background_count_curve(record):
    from riddle.full_mass import REFERENCE_SEEDS, tie_uniform
    from riddle.mass_spectrum import mixture_errors

    threshold_reference, sample = record["background_count_reference"]
    use = np.asarray(record["is_signal_region"], bool) & np.asarray(record["mask"], bool) & (record["labels"] == 0)
    data = np.asarray(record["scores"], float)[use]
    if len(data) < 2:
        return None
    if not np.isfinite(data).all():
        raise ValueError("Nonfinite held-out background scores")
    ordered = np.sort(threshold_reference, kind="stable")
    sample_ids = np.column_stack((np.full(len(sample), 2**64 - 1, dtype=np.uint64),
                                  np.arange(len(sample), dtype=np.uint64)))
    sample_ties = tie_uniform(sample_ids, REFERENCE_SEEDS["ties"])
    data_ties = tie_uniform(record["event_ids"][use], REFERENCE_SEEDS["ties"])
    counts = []
    for alpha in np.geomspace(1 / len(ordered), 1, 250):
        target = alpha * len(ordered)
        threshold = ordered[-int(np.ceil(target))]
        above = len(ordered) - np.searchsorted(ordered, threshold, side="right")
        tied = np.searchsorted(ordered, threshold, side="right") - np.searchsorted(ordered, threshold, side="left")
        probability = float(np.clip((target - above) / tied, 0, 1))
        if alpha == 1:
            threshold = -np.inf
        selected_data = (data > threshold) | ((data == threshold) & (data_ties < probability))
        selected_sample = (sample > threshold) | ((sample == threshold) & (sample_ties < probability))
        if np.any(selected_data):
            counts.append((int(selected_data.sum()), int(selected_sample.sum())))
    counts = np.asarray(counts, float)
    if not len(counts):
        return None
    data_counts, sample_counts = counts.T
    normalization = len(data) / len(sample)
    ratio = normalization * sample_counts / data_counts
    sample_low, sample_high = mixture_errors(sample_counts, sample_counts)
    data_low, data_high = mixture_errors(data_counts, data_counts)
    low_error = np.hypot(normalization * sample_low / data_counts, ratio * data_high / data_counts)
    high_error = np.hypot(normalization * sample_high / data_counts, ratio * data_low / data_counts)
    return dict(data_counts=data_counts, sample_counts=sample_counts, ratio=ratio,
                low=np.maximum(0, ratio - low_error), high=ratio + high_error,
                data_population=len(data), sample_population=len(sample),
                seed=int(record["manifest"]["identity"]["seed"]))


def background_count_axes(ax, maximum):
    ax.set(xscale="log", xlim=(1, max(2, maximum)), ylim=(0, 2),
           xlabel="Number of data background events",
           ylabel=r"$N_{\mathrm{bg}}^{\mathrm{Sample}} / N_{\mathrm{bg}}^{\mathrm{Data}}$")
    for value, style in ((1.0, "-"), (0.9, "--"), (1.1, "--"), (0.8, ":"), (1.2, ":")):
        ax.axhline(value, color="black", ls=style, lw=0.5, zorder=1)


def plot_background_count_closure(method, records, args, output):
    curves = parallel_plot_tasks(background_count_curve, records, args.io_workers, "BG COUNTS", args.verbose)
    curves = [curve for curve in curves if curve is not None]
    if not curves:
        return
    destination = output / "02_mass_sculpting" / METHOD_LABELS[method] / "BG-Only"
    stem = "background_sample_to_data_vs_background_events"
    color = METHOD_COLORS[method]
    for curve in curves:
        fig, ax = new_figure("", "")
        ax.plot(curve["data_counts"], curve["ratio"], color=color, label=METHOD_LABELS[method])
        ax.fill_between(curve["data_counts"], curve["low"], curve["high"], color=color, alpha=0.2, linewidth=0)
        background_count_axes(ax, curve["data_population"])
        legend = inside_legend(ax, allow_headroom=False)
        save_figure(fig, ax, destination / f"seed_{curve['seed']:03d}" / stem,
                    args.plot_formats, args.overwrite, legend)
    maximum = max(curve["data_population"] for curve in curves)
    grid = np.geomspace(1, maximum, 250)
    values = []
    for curve in curves:
        x, indices, counts = np.unique(curve["data_counts"], return_index=True, return_counts=True)
        ratio = np.add.reduceat(curve["ratio"], indices) / counts
        interpolated = np.full_like(grid, np.nan)
        inside = (grid >= x[0]) & (grid <= x[-1])
        interpolated[inside] = np.interp(np.log(grid[inside]), np.log(x), ratio)
        values.append(interpolated)
    fig, ax = new_figure("", "")
    draw_run_summary(ax, grid, values, label=METHOD_LABELS[method], color=color)
    background_count_axes(ax, maximum)
    legend = inside_legend(ax, allow_headroom=False)
    save_figure(fig, ax, destination / stem, args.plot_formats, args.overwrite, legend)
    say(f"[BG COUNTS] {METHOD_LABELS[method]}: independent assessment sample; "
        "seed plots show count uncertainty; summary shows the seed range", args.verbose)


def independent_records(record, seed):
    if not record.get("independent_runs", False) or "run_seeds" not in record:
        yield seed, record
        return
    scores = np.asarray(record["fit_scores"])
    seeds = np.asarray(record["run_seeds"])
    if scores.ndim != 2 or len(scores) != len(seeds) or len(np.unique(seeds)) != len(seeds):
        raise ValueError("Invalid independent run score inventory")
    for index, run_seed in enumerate(seeds):
        run = {key: value for key, value in record.items()
               if key not in ("fit_scores", "fit_latents", "run_seeds")}
        run.update(scores=scores[index], independent_runs=False, plot_saved_ensemble=True)
        if "fit_latents" in record:
            run["latent"] = record["fit_latents"][index]
        yield int(run_seed), run


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
    from riddle.scan_cache import aggregation_signature
    return aggregation_signature(method, report)


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
    order = np.argsort(x, kind="stable")
    x, y = x[order], y[order]
    unique, indices = np.unique(x, return_index=True)
    values = np.maximum.reduceat(y, indices)
    result = np.full_like(grid, np.nan, dtype=float)
    inside = (grid >= unique.min()) & (grid <= unique.max())
    result[inside] = np.interp(np.log10(grid[inside]), np.log10(unique), values)
    return result



def run_summary(matrix, *, common_support=False):
    matrix = np.asarray(matrix, float)
    if matrix.ndim != 2 or not len(matrix):
        raise ValueError("Run summary requires a nonempty run-by-point matrix")
    finite = np.isfinite(matrix)
    counts = finite.sum(axis=0)
    mean = np.divide(np.where(finite, matrix, 0).sum(axis=0), counts,
                     out=np.full(matrix.shape[1], np.nan), where=counts > 0)
    low = np.where(counts >= 2, np.min(np.where(finite, matrix, np.inf), axis=0), np.nan)
    high = np.where(counts >= 2, np.max(np.where(finite, matrix, -np.inf), axis=0), np.nan)
    if common_support:
        mean[counts != len(matrix)] = np.nan
        low[counts != len(matrix)] = np.nan
        high[counts != len(matrix)] = np.nan
    return mean, low, high


def draw_run_summary(ax, x, matrix, *, label, color, linestyle="-", common_support=False, **kwargs):
    mean, low, high = run_summary(matrix, common_support=common_support)
    ax.plot(x, mean, label=label, color=color, ls=linestyle, **kwargs)
    band = np.isfinite(low) & np.isfinite(high)
    if band.any():
        ax.fill_between(x, low, high, where=band, color=color, alpha=0.2, linewidth=0)
    return mean, low, high


def aggregate_metric_runs(run_rows, curve_kind):
    if not run_rows:
        return None
    grid = np.geomspace(1e-4, 1, 350)
    curves = []
    for row in run_rows:
        metrics = row["metrics"]
        supported = np.asarray(metrics["supported"], bool)
        b = np.asarray(metrics["background_efficiency"])[supported]
        s = np.asarray(metrics["signal_efficiency"])[supported]
        if curve_kind == "sic":
            y = np.divide(s, np.sqrt(b), out=np.full_like(s, np.nan), where=b > 0)
        elif curve_kind == "roc":
            y = s
        else:
            raise ValueError(curve_kind)
        curves.append(interpolate_curve(b, y, grid))
    return grid, *run_summary(curves, common_support=True)


def parallel_plot_tasks(function, tasks, workers, stage, verbose, describe=None):
    if not tasks:
        return []
    started = time.monotonic()
    completed = 0
    values = [None] * len(tasks)
    worker_count = min(workers, len(tasks))
    compact_progress = stage in ("LOAD", "SCAN LOAD", "METRICS", "SCAN METRICS", "WITNESS FIELDS") and verbose == 1
    live = compact_progress and sys.stdout.isatty()

    def progress_status():
        elapsed = time.monotonic() - started
        filled = 16 * completed // len(tasks)
        bar = "#" * filled + "-" * (16 - filled)
        remaining = tqdm.format_interval(elapsed * (len(tasks) - completed) / completed) if completed else "?"
        say(f"[{stage}] |{bar}| {completed}/{len(tasks)} "
            f"[{tqdm.format_interval(elapsed)}<{remaining}; workers={worker_count}]", verbose)

    if compact_progress:
        if not live:
            progress_status()
    else:
        say(f"[{stage}] 0/{len(tasks)}; workers={worker_count}", verbose)

    def timed(task):
        task_started = time.monotonic()
        return function(task), time.monotonic() - task_started

    with tqdm(total=len(tasks), desc=f"[{stage}]", file=sys.stdout, ascii=True,
              bar_format="{desc} |{bar:16}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}{postfix}]",
              postfix={"workers": worker_count}, mininterval=0.2, miniters=1, disable=not live) as progress, \
            threadpool_limits(limits=1), ThreadPoolExecutor(max_workers=worker_count) as executor:
        pending = {executor.submit(timed, task): i for i, task in enumerate(tasks)}
        last_update = started
        try:
            while pending:
                done, _ = wait(pending, timeout=5, return_when=FIRST_COMPLETED)
                for future in sorted(done, key=lambda item: pending[item]):
                    index = pending.pop(future)
                    values[index], elapsed = future.result()
                    completed += 1
                    now = time.monotonic()
                    detail = values[index][0] if stage == "LOAD" else None
                    if compact_progress:
                        if live:
                            progress.update(1)
                            last_update = now
                        elif now - last_update >= 10:
                            progress_status()
                            last_update = now
                    elif describe is not None:
                        say(f"[{stage} {completed}/{len(tasks)}] {describe(tasks[index], values[index])}; {elapsed:.1f}s", verbose)
                        last_update = now
                    elif detail is not None:
                        method, scenario, variant, region, seed = detail
                        name = f"{METHOD_LABELS.get(method, method)} s{seed} {variant}"
                        scenario_name = "signal" if scenario == "signal_injection" else "BG"
                        say(f"[{stage} {completed}/{len(tasks)}] {name} {scenario_name}; {elapsed:.1f}s", verbose)
                        last_update = now
                    elif now - last_update >= 5:
                        say(f"[{stage}] {completed}/{len(tasks)}; elapsed {now - started:.1f}s", verbose)
                        last_update = now
                now = time.monotonic()
                if pending and now - last_update >= 10:
                    if compact_progress:
                        if live:
                            progress.refresh()
                        else:
                            progress_status()
                    else:
                        say(f"[{stage}] {completed}/{len(tasks)}; elapsed {now - started:.1f}s; loading in progress" if stage.endswith("LOAD")
                            else f"[{stage}] {completed}/{len(tasks)}; elapsed {now - started:.1f}s", verbose)
                    last_update = now
        except BaseException:
            progress.close()
            for future in pending:
                future.cancel()
            if compact_progress:
                say(f"[{stage}] Failed at {completed}/{len(tasks)}; total {time.monotonic() - started:.1f}s", verbose)
            raise
    if compact_progress:
        if not live:
            progress_status()
    else:
        say(f"[{stage}] Finished {completed}/{len(tasks)}; total {time.monotonic() - started:.1f}s", verbose)
    return values


def metric_cache(groups, loader, regions, min_background, verbose, require_compatible_populations=False, io_workers=1):
    cache = {}
    records = {}
    protocol_groups = defaultdict(set)
    populations = {}
    tasks = []
    for (scenario, seed, variant), group in sorted(groups.items()):
        require_riddle_benchmark_alignment(group)
        for method, (root, report) in group.items():
            for region in regions:
                cohort = (method, scenario, variant, region)
                tasks.append(((*cohort, seed), root, report))

    def load(task):
        key, root, report = task
        method, scenario, variant, region, seed = key
        say(f"[LOAD] {METHOD_LABELS.get(method, method)} s{seed} {variant} {scenario}: {root}", verbose, 2)
        return key, load_record(loader, method, root, report, region), report

    for source, record, report in parallel_plot_tasks(load, tasks, io_workers, "LOAD", verbose):
        method, scenario, variant, region, seed = source
        if record is None:
            if method == "riddle" and region == "full_region":
                say("[SKIP] RIDDLE Full Region plots: saved scores are Signal Region only", verbose)
            continue
        cohort = (method, scenario, variant, region)
        protocol_groups[cohort].add(aggregation_protocol_signature(method, report))
        for run_seed, run in independent_records(record, seed):
            key = (*cohort, run_seed)
            if key in records:
                raise ValueError(f"Duplicate independent run: {key}")
            if cohort in populations:
                require_population_compatibility({"first": populations[cohort], "current": run})
            else:
                populations[cohort] = run
            records[key] = run
    for (method, scenario, variant, region), signatures in protocol_groups.items():
        if len(signatures) != 1:
            raise ValueError(f"Incompatible scientific protocols across independent runs for {METHOD_LABELS.get(method, method)}, {scenario}, {variant}, {region}")
    if require_compatible_populations:
        population_groups = defaultdict(dict)
        for (method, scenario, variant, region), record in populations.items():
            population_groups[(scenario, variant, region)][method] = record
        for (scenario, variant, region), population in population_groups.items():
            try:
                require_population_compatibility(population)
            except ValueError as error:
                raise ValueError(
                    f"Incompatible cross-method evaluation population for {scenario}, {variant}, {region}"
                ) from error
    def compute(item):
        key, record = item
        if key[1] == "signal_injection":
            return key, central_metrics(record, min_background)
        labels = np.asarray(record["labels"])
        mask = np.asarray(record["mask"], bool)
        bg = labels == 0
        return key, {"auc": None, "max_sic": None, "working_points": {}, "acceptance": {
            "background": {"total": int(bg.sum()), "mapped": int((bg & mask).sum()),
                           "acceptance": float(mask[bg].mean()) if bg.any() else None}}}
    cache.update(parallel_plot_tasks(compute, list(records.items()), io_workers, "METRICS", verbose))
    for method, scenario, variant, region in sorted(populations):
        count = len(compatible_run_rows(cache, method, scenario, variant, region))
        say(f"[SEEDS] {METHOD_LABELS.get(method, method)} {scenario} {variant} {region}: {count} completed seeds", verbose)
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
                destination = figure_path(output, scenario, variant, region)
                title = REGION_LABELS[region] + " · " + VARIANT_LABELS.get(variant, variant)
                for kind, ylabel, stem in (("sic", r"Significance improvement, $\epsilon_S/\sqrt{\epsilon_B}$", "sic_vs_background_efficiency"), ("roc", r"Signal efficiency, $\epsilon_S$", "roc_vs_background_efficiency")):
                    fig, ax = new_figure(r"Background efficiency, $\epsilon_B$", ylabel)
                    drawn = 0
                    for method in methods:
                        rows = compatible_run_rows(cache, method, scenario, variant, region)
                        aggregated = aggregate_metric_runs(rows, kind)
                        if aggregated is None:
                            continue
                        x, mean, low, high = aggregated
                        valid = np.isfinite(mean)
                        if not np.any(valid):
                            continue
                        ax.plot(x, mean, label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method), ls="-")
                        if len(rows) >= 2:
                            band = valid & np.isfinite(low) & np.isfinite(high)
                            ax.fill_between(x, low, high, where=band, color=METHOD_COLORS.get(method), alpha=0.2, linewidth=0)
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
                    rows = compatible_run_rows(cache, method, scenario, variant, region)
                    curves = []
                    grid = np.linspace(0, 1, 300)
                    for row in rows:
                        b = row["metrics"]["background_efficiency"]
                        s = row["metrics"]["signal_efficiency"]
                        good = np.isfinite(b) & np.isfinite(s) & (b > 0) & row["metrics"]["supported"]
                        if np.sum(good) < 2:
                            continue
                        order = np.argsort(s[good], kind="stable")
                        sx = s[good][order]
                        rej = 1 / b[good][order]
                        unique, idx = np.unique(sx, return_index=True)
                        values = np.full_like(grid, np.nan, float)
                        inside = (grid >= unique.min()) & (grid <= unique.max())
                        values[inside] = np.interp(grid[inside], unique, np.maximum.reduceat(rej, idx))
                        curves.append(values)
                    if not curves:
                        continue
                    draw_run_summary(ax, grid, curves, label=METHOD_LABELS.get(method, method),
                                     color=METHOD_COLORS.get(method), common_support=True)
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
                    rows = compatible_run_rows(cache, method, scenario, variant, region)
                    values = np.asarray([[row["metrics"]["working_points"].get(wp) for wp in PLOT_WORKING_POINTS] for row in rows], float)
                    if not values.size or not np.isfinite(values).any():
                        continue
                    draw_run_summary(ax, xs, values, label=METHOD_LABELS.get(method, method),
                                     color=METHOD_COLORS.get(method))
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
        destination = output / "01_comparison" / "Signal-Injected" / "Dataset-Variant-Summary" / folder_region(region)
        specs = [("auc", "AUC", "auc_vs_dataset_variant"), ("max_sic", "Maximum significance improvement", "max_sic_vs_dataset_variant")]
        for metric, ylabel, stem in specs:
            fig, ax = new_figure("Dataset variant", ylabel)
            drawn = 0
            for method in methods:
                means = []
                lows = []
                highs = []
                positions = []
                spreads = []
                for i, variant in enumerate(VARIANT_ORDER):
                    if variant not in comparable_variants:
                        continue
                    rows = compatible_run_rows(cache, method, "signal_injection", variant, region)
                    values = [row["metrics"].get(metric) for row in rows if row["metrics"].get(metric) is not None]
                    if values:
                        positions.append(i)
                        means.append(float(np.mean(values)))
                        lows.append(float(np.min(values)))
                        highs.append(float(np.max(values)))
                        spreads.append(len(values) >= 2)
                if positions:
                    ax.plot(positions, means, label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method), ls=METHOD_LINES.get(method, "-"), marker=METHOD_MARKERS.get(method, "o"))
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
                positions, means, lows, highs, spreads = [], [], [], [], []
                for i, variant in enumerate(VARIANT_ORDER):
                    if variant not in comparable_variants:
                        continue
                    rows = compatible_run_rows(cache, method, "signal_injection", variant, region)
                    values = [row["metrics"]["working_points"].get(wp) for row in rows if row["metrics"]["working_points"].get(wp) is not None]
                    if values:
                        positions.append(i)
                        means.append(float(np.mean(values)))
                        lows.append(float(np.min(values)))
                        highs.append(float(np.max(values)))
                        spreads.append(len(values) >= 2)
                if positions:
                    ax.plot(positions, means, label=METHOD_LABELS.get(method, method), color=METHOD_COLORS.get(method), ls=METHOD_LINES.get(method, "-"), marker=METHOD_MARKERS.get(method, "o"))
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
    collections = defaultdict(lambda: defaultdict(list))
    for (method, scenario, variant, region, seed), record in sorted(records.items()):
        if region != "signal_region":
            continue
        collections[(scenario, variant, region)][method].append(record)
    for (scenario, variant, region), method_records in collections.items():
        prepared = {}
        density_peak = 0.0
        count_peak = 0.0
        for method, runs in method_records.items():
            scores = [display_score_values(method, run["scores"]) for run in runs]
            masks = [np.asarray(run["mask"], bool) & np.isfinite(values) for run, values in zip(runs, scores)]
            finite_values = np.concatenate([values[mask] for values, mask in zip(scores, masks)])
            if not len(finite_values):
                continue
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
                densities, counts = [], []
                for run, values, mask in zip(runs, scores, masks):
                    population = mask & (np.asarray(run["labels"]) == truth)
                    if not np.any(population):
                        continue
                    count_hist = np.histogram(values[population], bins=edges)[0]
                    density_hist = np.divide(count_hist, count_hist.sum() * np.diff(edges),
                                             out=np.full(len(count_hist), np.nan), where=count_hist.sum() > 0)
                    densities.append(density_hist)
                    counts.append(count_hist)
                if not counts:
                    continue
                density_hist, count_hist = np.asarray(densities), np.asarray(counts)
                if np.isfinite(density_hist).any():
                    density_peak = max(density_peak, float(np.nanmax(density_hist)))
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
                    histograms = density_hist if density else count_hist
                    mean, low, high = run_summary(histograms)
                    ax.stairs(mean, edges, baseline=None, label=f"{METHOD_LABELS.get(method, method)} {noun}", color=color, ls=style)
                    if len(histograms) > 1:
                        ax.fill_between(edges, np.r_[low, low[-1]], np.r_[high, high[-1]], step="post", color=color, alpha=0.2, linewidth=0)
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
                method_curves = {}
                for method in methods:
                    curves = []
                    for (m, sc, v, rg, seed), record in records.items():
                        if (m, sc, v, rg) != (method, scenario, variant, region):
                            continue
                        curve = exact_mass_sculpt_curve(record, efficiencies)
                        if curve is not None:
                            curves.append(curve)
                    if curves:
                        method_curves[method] = np.asarray(curves)
                if not method_curves:
                    continue
                fig, ax = new_figure(r"Target background efficiency, $\epsilon_B$", r"$\chi^2/n_{\mathrm{dof}}$")
                first_record = next(record for (m, sc, v, rg, seed), record in records.items() if (sc, v, rg) == (scenario, variant, region))
                mass = np.asarray(first_record["mass"])[np.asarray(first_record["labels"]) == 0]
                if len(mass) >= 300:
                    reference = random_reference_band(mass, efficiencies)
                    median = np.nanmedian(reference, axis=0)
                    low, high = np.nanpercentile(reference, [16, 84], axis=0)
                    ax.fill_between(efficiencies, low, high, color="0.5", alpha=0.12, linewidth=0, zorder=0)
                    ax.plot(efficiencies, median, color="0.45", ls="--", lw=0.675, label="Random subset", zorder=1)
                for method, matrix in method_curves.items():
                    draw_run_summary(ax, efficiencies, matrix, label=METHOD_LABELS.get(method, method),
                                     color=METHOD_COLORS.get(method))
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
                        candidates = [(seed, record) for (m, sc, v, rg, seed), record in records.items() if (m, sc, v, rg) == (method, scenario, variant, region)]
                        if not candidates:
                            continue
                        seed, record = sorted(candidates)[0]
                        labels = np.asarray(record["labels"])
                        mass = np.asarray(record["mass"], float)
                        if np.nanmedian(np.abs(mass)) > 20:
                            mass = mass / 1000.0
                        bg = labels == 0
                        edges = equal_occupancy(mass[bg], min(24, max(10, len(mass[bg]) // 50)))
                        total = np.histogram(mass[bg], edges)[0]
                        efficiencies_by_run = []
                        for _, candidate in sorted(candidates):
                            selection = exact_background_selection(candidate, wp)
                            if selection is None:
                                continue
                            selected_counts = np.histogram(mass[bg], edges, weights=np.mean(selection["weights"], axis=0)[bg])[0]
                            efficiencies_by_run.append(np.divide(selected_counts, total, out=np.full_like(selected_counts, np.nan, dtype=float), where=total > 0))
                        if not efficiencies_by_run:
                            continue
                        centers = 0.5 * (edges[:-1] + edges[1:])
                        matrix = np.asarray(efficiencies_by_run)
                        rows.append((method, centers, matrix))
                        all_efficiencies.extend(matrix[np.isfinite(matrix)].tolist())
                    if rows:
                        working_point_plots[wp] = rows
                if working_point_plots:
                    maximum = max(all_efficiencies) if all_efficiencies else max(PLOT_WORKING_POINTS)
                    shared_ymax = max(0.15, 1.18 * maximum, 1.18 * max(PLOT_WORKING_POINTS))
                    for wp, rows in working_point_plots.items():
                        fig, ax = new_figure(r"$m_{jj}$ (TeV)", r"Background efficiency, $\epsilon_B$")
                        ax.axhline(wp, color="0.55", ls="--", lw=0.5, zorder=1)
                        for method, centers, matrix in rows:
                            draw_run_summary(ax, centers, matrix, label=METHOD_LABELS.get(method, method),
                                             color=METHOD_COLORS.get(method))
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
        if scenario == "signal_injection":
            continue
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


def plot_projected_witness(groups, records, args, output):
    from types import SimpleNamespace
    from riddle.full_mass import load_full_mass
    from riddle.mass_spectrum import match_ids
    from riddle.plotting import _prepared_data_directory
    from riddle.plotting import evaluate_saved_witness, parse_witness_pairs, witness_population_mask, project_witness_fields, witness_signal_selections, plot_witness_projections

    renderer = _FIGURE_RENDERER.get()
    if renderer is not None:
        renderer.flush()
    tasks = []
    for identity, group in sorted(groups.items()):
        scenario, seed, variant = identity
        for method, (root, report) in sorted(group.items()):
            key = (method, scenario, variant, "signal_region", seed)
            if method in ("riddle", "iad", "supervised") and key in records:
                tasks.append((identity, method, root, report, records[key]))

    def prepare(task):
        identity, method, root, report, record = task
        protocol = read_metadata(root, report, "protocol.json")
        if protocol.get("core") != "stein_witness":
            return None
        derived_root = Path(report.get("_full_mass_root", root))
        full = load_full_mass(derived_root, report, workers=1)
        if full is None and derived_root != root:
            full = load_full_mass(root, report, workers=1)
        if full is not None:
            indices = match_ids(record["event_ids"], full["event_ids"][full["is_signal_region"]])
            population = np.zeros(len(record["labels"]), bool)
            population[indices] = True
        else:
            prepared = _prepared_data_directory(args.data, report)
            if prepared is None:
                raise ValueError(f"Original prepared event IDs are required for {method}, seed {identity[1]}")
            population, _ = witness_population_mask(prepared, report, record, "primary")
        options = SimpleNamespace(field_events=getattr(args, "witness_field_events", 16384),
                                  grid_size=40, smoothing=1.2, min_cell_events=2.0,
                                  batch_size=1024, device="cpu", io_workers=1,
                                  sampling_seed=int(report["seed"]) + 10502,
                                  plot_formats=args.plot_formats, overwrite=args.overwrite, dpi=600,
                                  scenario=identity[0],
                                  verbose=args.verbose,
                                  label=f"{METHOD_LABELS[method]} s{identity[1]} {identity[0]} {identity[2]}")
        fields, metadata = evaluate_saved_witness(root, report, options)
        pairs = parse_witness_pairs(getattr(args, "witness_pairs", ["1:2"]), metadata["dimensions"])
        selections = witness_signal_selections(record, full)
        return identity, method, record, population, fields, selections, pairs, options

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1 if args.io_workers > 1 and len(tasks) > 1 else previous_threads)
        prepared = parallel_plot_tasks(prepare, tasks, args.io_workers, "WITNESS FIELDS", args.verbose)
    finally:
        torch.set_num_threads(previous_threads)
    for item in prepared:
        if item is None:
            continue
        identity, method, record, population, fields, selections, pairs, options = item
        scenario, seed, variant = identity
        destination = output / "05_stein_witness" / METHOD_LABELS[method] / SCENARIO_LABELS[scenario] / f"seed_{seed:03d}"
        for pair in pairs:
            grid = project_witness_fields(fields, pair, options)
            plot_witness_projections(grid, pair, record, population, destination, options,
                          selections=selections, save_figure=save_figure, legend_factory=inside_legend)


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
    histories_by_type = defaultdict(lambda: ([], []))
    def add(method, identity, stem, ylabel, train, validation, independent=False):
        key = (method, identity[0], identity[2], stem, ylabel)
        for target, values in zip(histories_by_type[key], (train, validation)):
            values = np.atleast_2d(values)
            target.extend(values if independent else [run_summary(values)[0]])
    for identity, group in groups.items():
        for method, (root, report) in group.items():
            key = (method, str(root))
            if key in represented:
                continue
            represented.add(key)
            if method in ("riddle", "iad", "supervised"):
                history = history_rows(root, report, "background/history.json")
                arrays = objective_arrays(history, False) if history is not None else None
                if arrays is not None:
                    add(method, identity, "background_nll", "Negative log likelihood", *arrays)
                if method in ("iad", "supervised"):
                    oracle_history = history_rows(root, report, "density/background_correction/history.json")
                    oracle_arrays = objective_arrays(oracle_history, False) if oracle_history is not None else None
                    if oracle_arrays is not None:
                        add(method, identity, "oracle_background_nll", "Negative log likelihood", *oracle_arrays)
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
                        train = history_matrix([item[0] for item in histories])
                        validation = history_matrix([item[1] for item in histories])
                        add(method, identity, "stein_witness_objective", "Stein objective", train, validation)
            elif method == "lacathode":
                for stem, names, ylabel in (("background_nll", ("lacathode_model_train_losses.npy", "lacathode_model_val_losses.npy"), "Negative log likelihood"), ("classifier_bce", ("loss_matris.npy", "val_loss_matris.npy"), "Binary cross-entropy")):
                    paths = [verify_requested_artifact(root, report, "training/" + name) for name in names]
                    if all(path is not None for path in paths):
                        values = [np.atleast_2d(np.load(path, allow_pickle=False)) for path in paths]
                        independent = report.get("contract", {}).get("lacathode_run_layout") == "independent_background_classifier_v1"
                        add(method, identity, stem, ylabel, *values, independent=independent)
    for (method, scenario, variant, stem, ylabel), (train, validation) in sorted(histories_by_type.items()):
        destination = output / "02_training" / METHOD_LABELS.get(method, safe_component(method)) / SCENARIO_LABELS[scenario] / folder_variant(variant)
        plot_history_band(history_matrix(train), history_matrix(validation), ylabel,
                          destination / stem, formats, overwrite, method=method)


def history_matrix(histories):
    matrix = np.full((len(histories), max(map(len, histories))), np.nan)
    for index, values in enumerate(histories):
        matrix[index, :len(values)] = values
    return matrix


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
        draw_run_summary(ax, x, values, label=label, color=color,
                         linestyle=style)
    add_y_headroom(ax, 0.14)
    legend = inside_legend(ax, borderaxespad=0.75)
    save_figure(fig, ax, stem, formats, overwrite, legend)


def plot_stability(cache, output, formats, overwrite, regions):
    for variant in VARIANT_ORDER:
        for region in regions:
            methods = comparison_methods(cache, "signal_injection", variant, region)
            if not methods:
                continue
            for field, ylabel, stem, wp in (("auc", "AUC", "auc_run_stability", None), ("max_sic", "Maximum significance improvement", "max_sic_run_stability", None), ("working", r"Signal efficiency at 0.5% BG", "exact_wp_signal_efficiency_run_stability_0p5pct", 0.005)):
                fig, ax = new_figure("Seed", ylabel)
                fig.set_figheight(fig.get_figheight() + 0.7)
                drawn = 0
                all_seeds = sorted({row["seed"] for method in methods for row in compatible_run_rows(cache, method, "signal_injection", variant, region)})
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
                    labels = [str(seed) for seed in all_seeds]
                    ax.set_xticks(range(len(all_seeds)), labels, rotation=90)
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
        method_values = {}
        for method in methods:
            rows = compatible_run_rows(cache, method, scenario, variant, region)
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
            backgrounds = [pair[0] for pair in pairs if pair[0] is not None]
            if backgrounds:
                bg = np.mean(backgrounds)
                ax.errorbar(i - 0.08, bg, yerr=[[bg - min(backgrounds)], [max(backgrounds) - bg]], marker="o", ls="none", color=METHOD_LIGHT.get(method, METHOD_COLORS.get(method)), label="BG" if i == 0 else "_nolegend_", capsize=2)
            if scenario == "signal_injection":
                sigs = [pair[1] for pair in pairs if pair[1] is not None]
                if sigs:
                    mean = np.mean(sigs)
                    ax.errorbar(i + 0.08, mean, yerr=[[mean - min(sigs)], [max(sigs) - mean]], marker="^", ls="none", color=METHOD_DARK.get(method, METHOD_COLORS.get(method)), label="Signal" if i == 0 else "_nolegend_", capsize=2)
            labels.append(METHOD_LABELS.get(method, method))
        ax.set_xticks(positions, labels, rotation=15)
        ax.set_ylim(0, 1.02)
        legend = inside_legend(ax, title=REGION_LABELS[region] + " · " + VARIANT_LABELS.get(variant, variant))
        save_figure(fig, ax, figure_path(output, scenario, variant, region) / "mapping_acceptance", formats, overwrite, legend)

def sic_at_signal_efficiency(metrics, signal_grid):
    b = np.asarray(metrics["background_efficiency"], float)
    s = np.asarray(metrics["signal_efficiency"], float)
    good = np.isfinite(b) & np.isfinite(s) & (b > 0) & np.asarray(metrics["supported"], bool)
    result = np.full_like(signal_grid, np.nan, dtype=float)
    if np.sum(good) < 2:
        return result
    order = np.argsort(s[good], kind="stable")
    sx = s[good][order]
    sic = sx / np.sqrt(b[good][order])
    unique, indices = np.unique(sx, return_index=True)
    if len(unique) < 2:
        return result
    values = np.maximum.reduceat(sic, indices)
    inside = (signal_grid >= unique[0]) & (signal_grid <= unique[-1])
    result[inside] = np.interp(signal_grid[inside], unique, values)
    return result


def plot_variant_robustness(cache, output, formats, overwrite, regions):
    for region in regions:
        for target in ("shifted", "deltaR"):
            methods = [m for m in METHOD_ORDER if compatible_run_rows(cache, m, "signal_injection", "default", region) and compatible_run_rows(cache, m, "signal_injection", target, region)]
            if not methods:
                continue
            fig, ax = new_figure(r"Signal efficiency, $\epsilon_S$", "SIC ratio")
            ax.axhline(1, color="0.5", ls="--", lw=0.5, zorder=1)
            drawn = 0
            signal_grid = np.linspace(0.05, 0.95, 250)
            for method in methods:
                default_rows = {row["seed"]: row for row in compatible_run_rows(cache, method, "signal_injection", "default", region)}
                target_rows = {row["seed"]: row for row in compatible_run_rows(cache, method, "signal_injection", target, region)}
                seeds = sorted(set(default_rows) & set(target_rows))
                ratios = []
                for seed in seeds:
                    curves = [sic_at_signal_efficiency(row["metrics"], signal_grid)
                              for row in (default_rows[seed], target_rows[seed])]
                    ratio = np.divide(curves[1], curves[0], out=np.full_like(signal_grid, np.nan),
                                      where=np.isfinite(curves[0]) & np.isfinite(curves[1]) & (curves[0] > 0))
                    ratios.append(ratio)
                if ratios:
                    draw_run_summary(ax, signal_grid, ratios, label=METHOD_LABELS.get(method, method),
                                     color=METHOD_COLORS.get(method), common_support=True)
                    drawn += 1
            if drawn:
                ax.set_xlim(signal_grid[-1], signal_grid[0])
                ax.set_ylim(-1.0, 3.0)
                legend = inside_legend(ax, title=REGION_LABELS[region], allow_headroom=False)
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


def performance_summary(metrics):
    row = {"Runs": len(metrics)}
    values = {"AUC": [m.get("auc") for m in metrics], "Max. SIC": [m.get("max_sic") for m in metrics]}
    for wp in reversed(WORKING_POINTS):
        values[f"εS@{WORKING_POINT_LABELS[wp]}"] = [m["working_points"].get(wp) for m in metrics]
    for name, samples in values.items():
        samples = np.asarray(samples, float)
        samples = samples[np.isfinite(samples)]
        row[name] = float(samples.mean()) if len(samples) else None
        row[name + " min"] = float(samples.min()) if len(samples) else None
        row[name + " max"] = float(samples.max()) if len(samples) else None
    return row


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
        for method in methods:
            rows = compatible_run_rows(cache, method, "signal_injection", variant, "signal_region")
            performance.append({"Dataset": VARIANT_LABELS.get(variant, variant),
                                "Method": METHOD_LABELS.get(method, method),
                                **performance_summary([row["metrics"] for row in rows])})
    write_table_rows(tables / "table_03_performance_summary", performance, file_formats)


def scan_identity_data(scan_groups, methods, variant="default"):
    result = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for identity, group in scan_groups.items():
        signal_events, replica, run_index, source_variant = identity
        if source_variant != variant:
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


def scan_method_run_indices(method_data, configured, cohort):
    indices = set()
    for level in configured:
        for replica in cohort:
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
        working_points[wp] = None if not values else float(np.mean(values))
    auc = [row["metrics"]["auc"] for row in run_rows if row["metrics"]["auc"] is not None]
    max_sic = [row["metrics"]["max_sic"] for row in run_rows if row["metrics"]["max_sic"] is not None]
    return {
        "metrics": {
            "auc": None if not auc else float(np.mean(auc)),
            "max_sic": None if not max_sic else float(np.mean(max_sic)),
            "working_points": working_points,
        },
        "background": background,
        "signal": signal,
        "s_over_b": signal / background if background else None,
        "nominal": signal / math.sqrt(background) if background else None,
        "runs": len(run_rows),
        "run_metrics": [row["metrics"] for row in run_rows],
    }


def process_injection_scan(scan_groups, methods, loader, settings, output, formats, file_formats, overwrite, min_background, allow_partial, verbose, require_compatible_populations=False, variant="default", io_workers=1, excluded_seeds=()):
    if overwrite:
        for extension in ("csv", "json", "yaml"):
            (output / "07_tables" / f"table_05_{variant}_signal_injection_score_stages.{extension}").unlink(missing_ok=True)
    configured = [int(value) for value in settings.get("injection_scan", {}).get("signal_events", [])]
    config = settings.get("injection_scan", {})
    cohort = config.get("seeds", list(range(int(config.get("replicas", 0)))))
    if not configured or not cohort:
        say("[SKIP] Injection scan: configuration is missing signal_events or seeds", verbose)
        return
    cohort = [seed for seed in cohort if seed not in excluded_seeds]
    if not cohort:
        say("[SKIP] Injection scan: all configured seeds were excluded", verbose)
        return
    if not scan_groups:
        say("[SKIP] Injection scan: no completed scan results were discovered", verbose)
        return
    scan_groups = {identity: group for identity, group in scan_groups.items()
                   if identity[3] == variant and identity[1] not in excluded_seeds}
    if not scan_groups:
        return
    schemas = {report["contract"]["inputs"]["injection_scan"].get("schema", 1)
               for group in scan_groups.values() for _, report in group.values()}
    if len(schemas) != 1 or (schemas == {2}) != ("seeds" in config):
        raise ValueError("Injection-scan results and configuration must use the same seed or legacy replica protocol")
    if schemas == {2}:
        prepared = defaultdict(set)
        for (level, seed, run_index, _), group in scan_groups.items():
            for _, report in group.values():
                if report["seed"] != seed or run_index != 0:
                    raise ValueError("Shared-population scans require one result per explicit seed")
                prepared[level].add(json.dumps(report["contract"]["inputs"], sort_keys=True))
        if any(len(values) != 1 for values in prepared.values()):
            raise ValueError("Scan methods and seeds must share identical prepared data at each injection strength")
    data = scan_identity_data(scan_groups, methods, variant)
    if allow_partial:
        requested_methods = [method for method in methods if method in data]
    else:
        missing_methods = [method for method in methods if method not in data]
        if missing_methods:
            labels = ", ".join(METHOD_LABELS.get(method, method) for method in missing_methods)
            raise incomplete_scan_error(f"Strict injection-scan comparison is missing requested methods: {labels}")
        requested_methods = list(methods)
    expected_runs = {method: scan_method_run_indices(data[method], configured, cohort) for method in requested_methods}
    from riddle.scan_cache import require_scan_compatibility
    for method in requested_methods:
        records = [record for level in configured for seed in cohort
                   for record in data[method].get(level, {}).get(seed, {}).values()]
        require_scan_compatibility(method, records)
    if not allow_partial and len(requested_methods) > 1:
        cohorts = {tuple(expected_runs[method]) for method in requested_methods}
        if len(cohorts) != 1:
            raise incomplete_scan_error("Injection-scan methods do not share the same independent run-index cohort")
    invalid = defaultdict(list)
    loaded = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    point_records = defaultdict(dict)
    point_counts = defaultdict(set)
    tasks = []
    for method in requested_methods:
        runs_expected = expected_runs[method]
        if not runs_expected:
            invalid[method].append("no independent runs discovered")
            continue
        for level in configured:
            for replica in cohort:
                runs = data[method].get(level, {}).get(replica, {})
                if set(runs) != set(runs_expected):
                    missing = sorted(set(runs_expected) - set(runs))
                    extra = sorted(set(runs) - set(runs_expected))
                    invalid[method].append(f"N_inj={level} seed={replica} missing_runs={missing} extra_runs={extra}")
                for run_index, (root, report) in sorted(runs.items()):
                    tasks.append(((method, level, replica, run_index), root, report))

    def load(task):
        key, root, report = task
        say(f"[SCAN LOAD] {key}: {root}", verbose, 2)
        try:
            return key, loader(root, report, "signal_region"), report, None
        except Exception as error:
            return key, None, report, f"{type(error).__name__}: {error}"

    def describe(task, result):
        method, level, replica, _ = task[0]
        status = "; failed" if result[3] else ""
        return f"{METHOD_LABELS.get(method, method)} s{replica} N={level} {variant}{status}"

    loaded_tasks = parallel_plot_tasks(load, tasks, io_workers, "SCAN LOAD", verbose, describe)

    def compute(task):
        key, record, report, error = task
        if error is not None:
            return key, record, None, None, error
        try:
            metrics = central_metrics(record, min_background)
            if metrics is None:
                raise ValueError("signal metrics are unavailable")
            counts = scan_realized_counts(report)
            if counts is None:
                raise ValueError("uncut_signal_region provenance is missing")
            return key, record, metrics, counts, None
        except Exception as error:
            return key, record, None, None, f"{type(error).__name__}: {error}"

    for key, record, metrics, counts, error in parallel_plot_tasks(compute, loaded_tasks, io_workers, "SCAN METRICS", verbose):
        method, level, replica, run_index = key
        if error is not None:
            invalid[method].append(f"N_inj={level} seed={replica} run={run_index}: {error}")
            continue
        background, signal = counts
        loaded[method][level][replica][run_index] = {
            "metrics": metrics,
            "background": background,
            "signal": signal,
        }
        point_records[(level, replica, run_index)][method] = record
        point_counts[(level, replica)].add(counts)
    for identity, group in scan_groups.items():
        level, replica, run_index, source_variant = identity
        if source_variant != variant or level not in configured or replica not in cohort:
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
                    raise ValueError(f"Incompatible injection-scan evaluation population for N_inj={level}, seed={replica}, run={run_index}") from error
    for (level, replica), counts in point_counts.items():
        if len(counts) > 1:
            raise ValueError(f"Injection-scan methods disagree on uncut physical counts for N_inj={level}, seed={replica}")
    summaries = {}
    if not allow_partial:
        failures = [f"{METHOD_LABELS.get(method, method)}: " + "; ".join(invalid[method]) for method in requested_methods if invalid[method]]
        if failures:
            raise incomplete_scan_error("Strict injection-scan comparison is incomplete: " + " | ".join(failures))
    for method in requested_methods:
        if invalid[method] and allow_partial:
            say("[WARN] Partial injection scan: " + METHOD_LABELS.get(method, method) + ": " + "; ".join(invalid[method]), verbose)
        levels = []
        for level in configured:
            replicas_data = []
            for replica in cohort:
                run_rows = list(loaded[method].get(level, {}).get(replica, {}).values())
                if not run_rows:
                    continue
                replicas_data.append(aggregate_scan_replica(run_rows))
            if replicas_data:
                levels.append((level, replicas_data))
        if levels and (allow_partial or (len(levels) == len(configured) and all(len(rows) == len(cohort) for _, rows in levels))):
            summaries[method] = levels
    if not allow_partial and set(summaries) != set(requested_methods):
        missing = [method for method in requested_methods if method not in summaries]
        labels = ", ".join(METHOD_LABELS.get(method, method) for method in missing)
        raise incomplete_scan_error(f"Strict injection-scan comparison could not build complete summaries for: {labels}")
    if not summaries:
        return
    destination = output / "06_injection_scan" / VARIANT_LABELS.get(variant, variant)
    figure_specs = [("max_sic", "Maximum significance improvement", "maximum_sic_vs_s_over_b"), ("nominal_selected", "Maximum achieved significance", "maximum_nominal_significance_vs_s_over_b"), ("auc", "AUC", "auc_vs_s_over_b")]
    for wp in PLOT_WORKING_POINTS[::-1]:
        figure_specs.append((f"wp_{wp}", r"Signal efficiency, $\epsilon_S$", f"epsS_{WORKING_POINT_LABELS[wp].replace('%','pct').replace('.','p')}_vs_s_over_b"))
    reference_axis = None
    reference_counts = None
    for method, levels in summaries.items():
        axis = []
        counts = []
        for level, replicas_data in levels:
            sob = [row["s_over_b"] for row in replicas_data if row["s_over_b"] is not None]
            nominal = [row["nominal"] for row in replicas_data if row["nominal"] is not None]
            if sob and nominal:
                axis.append((level, 100 * float(np.mean(sob)), float(np.mean(nominal))))
                counts.append({(row["background"], row["signal"]) for row in replicas_data})
        if reference_axis is None:
            reference_axis = axis
            reference_counts = counts
        elif ([item[0] for item in axis] != [item[0] for item in reference_axis]
              or counts != reference_counts
              or not np.allclose([item[1:] for item in axis], [item[1:] for item in reference_axis],
                                 rtol=1e-14, atol=0.0)):
            raise ValueError("Injection-scan methods do not share the same realized S/B and S/sqrt(B) axis")
    for field, ylabel, stem in figure_specs:
        fig, ax = new_figure("S/B (%)", ylabel)
        drawn = 0
        for method, levels in summaries.items():
            x, y, low, high = [], [], [], []
            for level, replicas_data in levels:
                sob = [row["s_over_b"] for row in replicas_data if row["s_over_b"] is not None]
                if not sob:
                    continue
                completed = [{"metrics": metrics, "nominal": row["nominal"]}
                             for row in replicas_data for metrics in row["run_metrics"]]
                if field == "max_sic":
                    values = [row["metrics"]["max_sic"] for row in completed]
                elif field == "nominal_selected":
                    values = [row["metrics"]["max_sic"] * row["nominal"] for row in completed if row["metrics"]["max_sic"] is not None and row["nominal"] is not None]
                elif field == "auc":
                    values = [row["metrics"]["auc"] for row in completed]
                else:
                    wp = float(field.split("_", 1)[1])
                    values = [row["metrics"]["working_points"].get(wp) for row in completed if row["metrics"]["working_points"].get(wp) is not None]
                values = [value for value in values if value is not None and np.isfinite(value)]
                if not values:
                    continue
                x.append(100 * float(np.mean(sob)))
                y.append(float(np.mean(values)))
                low.append(float(np.min(values)) if len(values) > 1 else np.nan)
                high.append(float(np.max(values)) if len(values) > 1 else np.nan)
            if x:
                order = np.argsort(x)
                x, y, low, high = [np.asarray(values)[order] for values in (x, y, low, high)]
                ax.plot(x, y, color=METHOD_COLORS.get(method), ls="-", marker="x",
                        label=METHOD_LABELS.get(method, method))
                ax.fill_between(x, low, high, where=np.isfinite(low) & np.isfinite(high), color=METHOD_COLORS.get(method), alpha=0.2, linewidth=0)
                drawn += 1
        if drawn:
            if field == "nominal_selected":
                for significance in (3, 5):
                    ax.axhline(significance, color="0.5", ls=":", lw=0.7, label="_nolegend_")
            ax.invert_xaxis()
            top = ax.twiny()
            top.set_xlim(ax.get_xlim())
            if reference_axis:
                ordered_axis = sorted(reference_axis, key=lambda item: item[1])
                top.set_xticks([item[1] for item in ordered_axis])
                top.set_xticklabels([f"{item[2]:.2g}" for item in ordered_axis])
            top.set_xlabel(r"$S/\sqrt{B}$")
            top.tick_params(axis="x", labelsize=6.7)
            legend = {"scan_legend": {"title": VARIANT_LABELS.get(variant, variant)}}
            save_figure(fig, ax, destination / stem, formats, overwrite, legend)
        else:
            plt.close(fig)
    rows = []
    for method, levels in summaries.items():
        for level, replicas_data in levels:
            metrics = [metrics for row in replicas_data for metrics in row["run_metrics"]]
            significance = [metric["max_sic"] * row["nominal"] for row in replicas_data
                            for metric in row["run_metrics"]
                            if metric["max_sic"] is not None and row["nominal"] is not None]
            significance = np.asarray(significance, float)
            significance = significance[np.isfinite(significance)]
            rows.append({"Method": METHOD_LABELS.get(method, method), "N_inj": int(level),
                         **performance_summary(metrics),
                         "Max. achieved significance": float(significance.mean()) if len(significance) else None,
                         "Max. achieved significance min": float(significance.min()) if len(significance) else None,
                         "Max. achieved significance max": float(significance.max()) if len(significance) else None})
    if rows:
        write_table_rows(output / "07_tables" / f"table_04_{variant}_signal_injection_dependence", rows, file_formats)


def load_settings_file(path):
    with Path(path).open("r", encoding="utf-8") as handle:
        value = yaml.safe_load(handle)
    if not isinstance(value, dict):
        raise ValueError("Settings YAML must contain a mapping")
    return value


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Physical Review D publication plotting for completed RIDDLE-compatible results")
    parser.add_argument("--workflow", choices=("normal", "scan"), help="Defaults to scan when --scan-data or --scan-results is supplied")
    parser.add_argument("--results", type=Path)
    parser.add_argument("--scan-results", type=Path)
    parser.add_argument("--data", type=Path)
    parser.add_argument("--scan-data", type=Path)
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
    parser.add_argument("--io-workers", type=int, default=2, help="Parallel discovery, result loading and metric workers")
    parser.add_argument("--plot-workers", type=int, default=min(8, max(1, (os.cpu_count() or 1) // 2)),
                        help="CPU processes for legend layout and image rendering; 1 runs serially")
    parser.add_argument("--witness-field-events", type=int, default=16384,
                        help="Maximum saved background-reference events for projected Stein diagnostics")
    parser.add_argument("--witness-pairs", nargs="+", default=["1:2"],
                        help="One-based latent-coordinate pairs for Stein projections, or all")
    parser.add_argument("--cross-section-weights", type=Path,
                        help="NPZ with event_ids and physical weights_pb; otherwise mass spectra show events/TeV")
    parser.add_argument("--allow-partial-injection-scan", action="store_true")
    parser.add_argument("--require-compatible-populations", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--verbose", type=int, choices=(0, 1, 2), default=1)
    args = parser.parse_args(argv)
    if args.io_workers < 1 or args.plot_workers < 1 or args.min_background < 1:
        parser.error("--io-workers, --plot-workers and --min-background must be positive")
    if args.witness_field_events < 16:
        parser.error("--witness-field-events must be at least 16")
    normal_requested = args.data is not None or args.results is not None
    scan_requested = args.scan_data is not None or args.scan_results is not None
    if normal_requested and scan_requested:
        parser.error("Use --data/--results for normal plots or --scan-data/--scan-results for injection plots in separate commands")
    if ((args.workflow == "normal" and scan_requested)
            or (args.workflow == "scan" and normal_requested)):
        parser.error("The input arguments must match --workflow")
    args.workflow = args.workflow or ("scan" if scan_requested else "normal")
    args.results = args.results if args.results is not None else Path("results")
    args.data = args.data if args.data is not None else Path("data/lhco")
    args.scan_results = args.scan_results if args.scan_results is not None else Path("results_injection_scan")
    args.scan_data = args.scan_data if args.scan_data is not None else Path("data/injection_scan")
    return args


def main(argv=None):
    args = parse_args(argv)
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(args.io_workers)
        with threadpool_limits(limits=args.io_workers):
            return run(args)
    finally:
        torch.set_num_threads(previous_threads)


def run(args):
    configure_style()
    if args.overwrite and args.workflow == "normal":
        cleanup_legacy_summary_folders(args.output)
    settings = load_settings_file(args.config)
    scan = args.workflow == "scan"
    source = args.scan_results if scan else args.results
    say(f"[STAGE] Discover completed {'injection-scan' if scan else 'normal'} results", args.verbose)
    requested = None if args.methods and "all" in args.methods else tuple(
        canonical_method(method) for method in (args.methods or ("riddle", "iad", "supervised"))
    )
    groups_all = discover_variant_results(source, requested, args.variants, scan=scan,
                                         workers=args.io_workers, verbose=args.verbose)
    methods = select_requested_methods(groups_all, args.methods,
                                       allow_missing=scan and args.allow_partial_injection_scan,
                                       injection_scan=scan)
    if not methods:
        raise SystemExit("No requested publication methods were discovered")
    discovered_variants = sorted({identity[3 if scan else 2] for identity in groups_all}, key=lambda value: VARIANT_ORDER.index(value) if value in VARIANT_ORDER else 99)
    variants = list(args.variants) if args.variants else discovered_variants
    for variant in variants:
        if variant not in discovered_variants:
            say(f"[WARN] No completed {variant} results found in the selected result folders", args.verbose)
    loader = ScoreLoader(safeguard_filtering=True, ensemble_fit_selection=True, io_workers=args.io_workers,
                         device="cpu", eager_partitions=False)
    if scan:
        groups = {identity: {canonical_method(method): value for method, value in group.items()
                             if canonical_method(method) in methods and (not args.scenarios or value[1]["scenario"] in args.scenarios)}
                  for identity, group in groups_all.items()
                  if identity[3] in variants and identity[1] not in args.exclude_seeds}
        groups = {identity: group for identity, group in groups.items() if group}
        if not groups:
            raise SystemExit("No completed injection-scan results remain after applying filters")
        say("[SUMMARY] Mean across completed seeds; bands show the min–max range across seeds", args.verbose)
        with render_resources(args.plot_workers, args.verbose, args.output):
            for variant in variants:
                if variant not in discovered_variants:
                    continue
                say(f"[STAGE] Build and render {variant} signal-injection dependence", args.verbose)
                destination = variant_output(args.output, variant)
                process_injection_scan(groups, methods, loader, settings, destination, args.plot_formats,
                                       args.file_formats, args.overwrite, args.min_background,
                                       args.allow_partial_injection_scan, args.verbose,
                                       args.require_compatible_populations, variant=variant,
                                       io_workers=args.io_workers, excluded_seeds=args.exclude_seeds)
                plot_dijet_spectra({identity: group for identity, group in groups.items() if identity[3] == variant},
                                   args, destination, scan=True)
        say(f"[DONE] Injection-scan outputs written to {args.output}", args.verbose)
        return 0
    scenarios = list(args.scenarios) if args.scenarios else sorted({identity[0] for identity in groups_all}, key=lambda value: ("signal_injection", "background_only").index(value))
    regions = ["signal_region"]
    groups = filter_groups(groups_all, methods, variants, scenarios, set(args.exclude_seeds))
    if not groups:
        raise SystemExit("No completed results remain after applying method, variant, scenario, and seed filters")
    say("[STAGE] Validate result provenance", args.verbose)
    say("[STAGE] Build publication metric cache", args.verbose)
    cache, records = metric_cache(groups, loader, regions, args.min_background, args.verbose, args.require_compatible_populations, args.io_workers)
    say("[SUMMARY] Mean across completed seeds; bands show the min–max range across seeds", args.verbose)
    with render_resources(args.plot_workers, args.verbose, args.output):
        for variant in variants:
            selected_groups = {identity: group for identity, group in groups.items() if identity[2] == variant}
            if not selected_groups:
                continue
            selected_cache = {key: value for key, value in cache.items() if key[2] == variant}
            selected_records = {key: value for key, value in records.items() if key[2] == variant}
            destination = variant_output(args.output, variant)
            say(f"[VARIANT] {destination.name}", args.verbose)
            render_publication_outputs(args, selected_groups, selected_cache, selected_records, loader,
                                       scenarios, [variant], regions, output=destination)
            say(f"[STAGE] Render {variant} full-mass dijet spectra", args.verbose)
            plot_dijet_spectra(selected_groups, args, destination)
        plot_variant_robustness(cache, variant_output(args.output, "default"), args.plot_formats, args.overwrite, regions)
    say(f"[DONE] Publication outputs written to {args.output}", args.verbose)
    return 0


def render_publication_outputs(args, groups, cache, records, loader, scenarios, variants, regions, output=None):
    from copy import copy
    args = copy(args)
    args.output = args.output if output is None else output
    say("[STAGE] Render comparison performance plots", args.verbose)
    plot_performance(cache, args.output, args.plot_formats, args.overwrite, scenarios, variants, regions)
    plot_score_distributions(groups, records, args.output, args.plot_formats, args.overwrite)
    plot_stability(cache, args.output, args.plot_formats, args.overwrite, regions)
    plot_acceptance(cache, args.output, args.plot_formats, args.overwrite)
    say("[STAGE] Render mass-sculpting plots", args.verbose)
    plot_mass_sculpting(cache, records, args.output, args.plot_formats, args.overwrite, scenarios, variants, regions, args.verbose)
    say("[STAGE] Render feature diagnostics", args.verbose)
    plot_features(records, args.output, args.plot_formats, args.overwrite, args.verbose)
    plot_score_vs_mass(records, args.output, args.plot_formats, args.overwrite)
    say("[STAGE] Render latent diagnostics", args.verbose)
    plot_latent_closure(groups, records, args.output, args.plot_formats, args.overwrite, args.verbose)
    say("[STAGE] Render Stein-witness diagnostics", args.verbose)
    plot_stein(groups, records, args.output, args.plot_formats, args.overwrite)
    say("[STAGE] Build projected Stein-witness diagnostics", args.verbose)
    plot_projected_witness(groups, records, args, args.output)
    say("[STAGE] Render training histories", args.verbose)
    plot_training(groups, args.output, args.plot_formats, args.overwrite)
    say("[STAGE] Export publication tables", args.verbose)
    export_tables(groups, cache, loader, args.output, args.file_formats, args.data, args.verbose)


if __name__ == "__main__":
    raise SystemExit(main())
