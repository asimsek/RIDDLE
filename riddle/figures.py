from collections import OrderedDict, deque
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from contextvars import ContextVar
from copy import deepcopy
from functools import wraps
import gc
import hashlib
import itertools
import math
import multiprocessing
import pickle
import re
import sys
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.ticker import LogFormatterSciNotation, LogLocator, MaxNLocator, NullFormatter
from matplotlib.transforms import Bbox
from mplhep import style as hep_style
import numpy as np
from scipy.interpolate import interp1d
from scipy.special import expit
from scipy.stats import beta, norm
from sklearn.metrics import roc_auc_score, roc_curve

METHODS = {"raw": ("LaCathode", "#0072B2", "-"), "residual": ("RIDDLE", "#D55E00", "--")}
POPULATIONS = {
    "raw": {0: ("#0072B2", "-"), 1: ("#CC79A7", "--")},
    "residual": {0: ("#D55E00", "-"), 1: ("#009E73", "--")},
}
VIEWS = {"comparison": tuple(METHODS), "LaCathode": ("raw",), "RIDDLE": ("residual",)}
BUDGETS = (("loose", 0.1), ("medium", 0.05), ("tight", 0.01), ("extra_tight", 0.004))
SCORE_CUTS = tuple(value / 100 for value in range(30, 100))
MASS_TARGETS = (0.20, 0.15, 0.10, 0.075, 0.05, 0.025, 0.01, 0.005, 0.004)
EFFICIENCIES = np.arange(0.01, 0.21, 0.01)[::-1]
GRID = np.logspace(-4, 0, 500)
SUMMARY_AXES = {
    "sic": dict(xscale="log", xlim=(1e-4, 1), yscale="linear"),
    "mass_flatness": dict(xscale="linear", xlim=(0.20, 0.01), yscale="log"),
}
# Shared by individual, comparison and summary panels, including new methods.
# These are minimum display ranges; data and drawn bands can always expand them.
PUBLICATION_Y_RANGES = {
    "mass_flatness": (0.5, 350.0),
    "sic": (0.0, 2.0),
    "ratio": (0.5, 1.5),
    "classification_loss_min_span": 0.05,
    "headroom": 0.15,
}
_Y_AXIS_KINDS = {
    r"$\chi^2/n_{\mathrm{dof}}$": "mass_flatness",
    "Significance improvement": "sic",
    "Classification loss": "classification_loss",
    "Fitted mixture fraction": "nonnegative",
    "Normalized background / bin": "nonnegative",
    "Density": "nonnegative",
    "Density (unit area per class)": "nonnegative",
    "Probability density": "nonnegative",
    r"Normalized density [TeV$^{-1}$]": "nonnegative",
    "Events / bin": "nonnegative",
    "Background events / bin": "nonnegative",
    "Background efficiency": "nonnegative",
    "Background efficiency [%]": "nonnegative",
    "Background kept / all": "nonnegative",
}
SCENARIOS = ("signal_injection", "background_only")
SCENARIO_LABELS = {"signal_injection": "Signal-Injected", "background_only": "BG-Only"}
STYLE = {
    **hep_style.CMS,
    "text.usetex": False,
    "mathtext.fontset": "dejavusans",
    "mathtext.fallback": "stixsans",
    "font.family": "sans-serif",
    "font.sans-serif": ["DejaVu Sans"],
    "font.size": 11,
    "figure.dpi": 120,
    "axes.labelsize": 13,
    "axes.titlesize": 11,
    "axes.labelpad": 6,
    "axes.titlepad": 7,
    "axes.axisbelow": True,
    "axes.linewidth": 1.15,
    "legend.fontsize": 10,
    "legend.handlelength": 2.0,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "xtick.major.width": 1.0,
    "ytick.major.width": 1.0,
    "xtick.minor.width": 0.8,
    "ytick.minor.width": 0.8,
    "lines.linewidth": 1.8,
    "lines.solid_capstyle": "round",
    "lines.dash_capstyle": "round",
    "xtick.direction": "in",
    "ytick.direction": "in",
    "xtick.top": True,
    "ytick.right": True,
    "xtick.minor.visible": True,
    "ytick.minor.visible": True,
    "xtick.major.size": 5,
    "ytick.major.size": 5,
    "xtick.minor.size": 2.5,
    "ytick.minor.size": 2.5,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "savefig.dpi": 600,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
}


_PLOT_CACHE = ContextVar("plot_calculation_cache", default=None)
_FIGURE_EXPORTER = ContextVar("figure_exporter", default=None)


def _calculation_key(value):
    if isinstance(value, np.ndarray):
        data = np.ascontiguousarray(value)
        return value.dtype.str, value.shape, hashlib.sha256(data).digest()
    if isinstance(value, (tuple, list)):
        return tuple(_calculation_key(item) for item in value)
    if isinstance(value, dict):
        return tuple((key, _calculation_key(item)) for key, item in sorted(value.items()))
    return value


def _calculation_bytes(value):
    if isinstance(value, np.ndarray):
        return value.nbytes + 128
    if isinstance(value, dict):
        return sys.getsizeof(value) + sum(_calculation_bytes(k) + _calculation_bytes(v) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return sys.getsizeof(value) + sum(map(_calculation_bytes, value))
    return sys.getsizeof(value)


def cached_plot_calculation(function):
    """Reuse exact numerical inputs within one plot run, with an LRU memory limit.

    Hash contents rather than array identities: slices and copied SR records can
    share results, while changed masks/scores never reuse a stale calculation.
    Copies isolate callers that normalize histograms or annotate metric dicts.
    """
    @wraps(function)
    def calculate(*args, **kwargs):
        cache = _PLOT_CACHE.get()
        if cache is None or cache["limit"] == 0:
            return function(*args, **kwargs)
        key = (function.__module__, function.__qualname__, _calculation_key(args), _calculation_key(kwargs))
        entries = cache["entries"]
        if key in entries:
            entries.move_to_end(key)
            return deepcopy(entries[key][0])
        result = function(*args, **kwargs)
        size = _calculation_bytes(key) + _calculation_bytes(result)
        if size <= cache["limit"]:
            while entries and (cache["bytes"] + size > cache["limit"] or len(entries) >= 4096):
                _, (_, old_size) = entries.popitem(last=False)
                cache["bytes"] -= old_size
            entries[key] = (deepcopy(result), size)
            cache["bytes"] += size
        return result
    return calculate


# The same saved fits appear in ROC, SIC, rejection and individual plots.
roc_curve = cached_plot_calculation(roc_curve)
roc_auc_score = cached_plot_calculation(roc_auc_score)


def _export_worker_init():
    # Export workers do no inference; avoid multiplying BLAS threads by workers.
    from threadpoolctl import threadpool_limits

    threadpool_limits(limits=1)


def _export_figure(payload, path, style):
    with matplotlib.rc_context(style):
        fig = pickle.loads(payload)
        try:
            _save_figure(fig, path)
        finally:
            plt.close(fig)
            gc.collect()


class FigureExporter:
    """Bound in-flight figures and propagate worker failures before completion."""

    def __init__(self, workers):
        self.pool = (ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"),
                                         initializer=_export_worker_init) if workers > 1 else None)
        self.pending = deque()
        self.max_pending = 2 * workers
        self.pending_bytes = 0

    def _finish_one(self):
        future, path, size = self.pending.popleft()
        self.pending_bytes -= size
        try:
            future.result()
        except Exception as error:
            raise RuntimeError(f"Figure export failed for {path}: {error}") from error

    def save(self, fig, path):
        if self.pool is None:
            _save_figure(fig, path)
            return
        # Snapshot before closing/reusing artists; workers receive no score data
        # or inference runtime and never access the parent pyplot globals.
        try:
            payload = pickle.dumps(fig, protocol=pickle.HIGHEST_PROTOCOL)
        except (pickle.PicklingError, AttributeError, TypeError):
            # Custom axes may contain local callbacks. Preserve their original
            # renderer instead of making serialization a plotting requirement.
            _save_figure(fig, path)
            return
        while self.pending and (len(self.pending) >= self.max_pending
                                or self.pending_bytes + len(payload) > 256 * 1024**2):
            self._finish_one()
        future = self.pool.submit(_export_figure, payload, path, dict(matplotlib.rcParams))
        self.pending.append((future, path, len(payload)))
        self.pending_bytes += len(payload)

    def flush(self):
        while self.pending:
            self._finish_one()

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)


@contextmanager
def plot_resources(*, workers=1, cache_mb=1024):
    """Caches and export processes live only for this plotting invocation."""
    if workers < 1 or cache_mb < 0:
        raise ValueError("Plot workers must be positive and cache size nonnegative")
    exporter = FigureExporter(workers)
    cache_token = _PLOT_CACHE.set(dict(entries=OrderedDict(), bytes=0, limit=cache_mb * 1024**2))
    export_token = _FIGURE_EXPORTER.set(exporter)
    try:
        yield exporter
        exporter.flush()
    finally:
        _PLOT_CACHE.reset(cache_token)
        _FIGURE_EXPORTER.reset(export_token)
        exporter.close()


def fit_scores(record):
    if (record.get("plot_saved_ensemble", False) or
            str(record.get("fit_score_kind", "")) == "member_ratio_mapped_by_frozen_ensemble_calibrator"):
        # Component densities are not independent calibrated methods. Averaging
        # their histograms/efficiencies does not evaluate the saved ensemble.
        return record["scores"][None, :]
    return record.get("fit_scores", record["scores"][None, :])


def fit_uncertainty(record):
    return ("independent_background_flow_and_classifier_runs" if record.get("independent_runs")
            else "classifier_fit_variation_with_shared_background_flow_per_seed")


def sample_fit_scores(sample, key):
    return sample.get(key + "_fit_scores", sample[key + "_scores"][None, :])


@cached_plot_calculation
def fit_histogram(values, edges, mask):
    """Mean counts per fit, retaining event denominators rather than pooling fits."""
    values, mask = np.broadcast_arrays(np.atleast_2d(values), np.atleast_2d(mask))
    counts = np.asarray([np.histogram(v[m], edges)[0] for v, m in zip(values, mask)])
    return counts[0] if len(counts) == 1 else counts.mean(axis=0)


def validate_fit_counts(records):
    counts = [len(fit_scores(record)) for record in records]
    if len(set(counts)) != 1:
        raise ValueError("LaCathode fit counts differ between evaluation partitions")
    return counts[0]


def fit_curve_summary(curves, run_groups=None):
    """Upstream interpolation of rejection and SIC at 1000 common signal efficiencies.

    The caller applies its statistical-support cut before interpolation. Every
    fit must contribute over the shared domain; unsupported fits are not dropped.
    """
    if not curves or any(len(b) < 2 or len(np.unique(s)) < 2 for b, s in curves):
        raise ValueError("Insufficient common LaCathode ROC support across all fits")
    low = max(float(np.min(s)) for _, s in curves)
    high = min(float(np.max(s)) for _, s in curves)
    if low >= high:
        raise ValueError("LaCathode fits have no shared signal-efficiency range")
    signal = np.linspace(low, high, 1000)
    rejection = np.asarray([interp1d(s, 1 / b)(signal) for b, s in curves])
    sic = np.asarray([interp1d(s, s / np.sqrt(b))(signal) for b, s in curves])
    if run_groups is not None:
        groups = np.asarray(run_groups)
        if groups.shape != (len(curves),):
            raise ValueError("Each curve needs a complete-run identity")
        # Shared-background LaCathode classifier fits retain their median
        # prediction, but only complete runs enter the uncertainty percentiles.
        rejection = np.asarray([np.median(rejection[groups == key], axis=0) for key in np.unique(groups)])
        sic = np.asarray([np.median(sic[groups == key], axis=0) for key in np.unique(groups)])
    return signal, np.percentile(rejection, [16, 50, 84], axis=0), np.percentile(sic, [16, 50, 84], axis=0)


def draw_fit_curves(ax, curves, metric, label, color, linestyle, *, uncertainty_source="independent_run_variation", band=True, run_groups=None):
    run_count = len(curves) if run_groups is None else len(set(run_groups))
    try:
        signal, rejection, sic = fit_curve_summary(curves, run_groups)
    except ValueError as error:
        drawn = False
        for b, s in curves:
            if not len(b):
                continue
            if metric == "sic_signal":
                x, y = s, s / np.sqrt(b)
            elif metric in ("rejection", "background_rejection"):
                x, y = s, 1 / b
            else:
                x, y = b, s if metric.startswith("roc") else s / np.sqrt(b)
            ax.plot(x, y, label=label if not drawn else None, color=color,
                    ls=linestyle, alpha=.4, marker="o" if len(b) == 1 else None)
            drawn = True
        return {"fit_count": len(curves), "band_drawn": False,
                "band_status": str(error), "aggregation": "unavailable; individual supported fit curves only"}
    if metric == "sic_signal":
        x, y = np.broadcast_to(signal, sic.shape), sic
    elif metric in ("rejection", "background_rejection"):
        x, y = np.broadcast_to(signal, rejection.shape), rejection
    elif metric.startswith("roc"):
        x, y = 1 / rejection, np.broadcast_to(signal, rejection.shape)
    else:
        x, y = 1 / rejection, sic
    if band and run_count > 1:
        # Parametric ribbon at fixed signal efficiency, retaining upstream axes.
        ax.fill(np.r_[x[0], x[2, ::-1]], np.r_[y[0], y[2, ::-1]],
                color=color, alpha=.18, linewidth=0)
    ax.plot(x[1], y[1], label=label, color=color, ls=linestyle)
    return {
        "fit_count": len(curves), "run_count": run_count, "band_drawn": bool(band and run_count > 1),
        "aggregation": "upstream common signal-efficiency interpolation; median rejection and SIC",
        "signal_efficiency": signal.tolist(), "background_rejection": rejection.tolist(),
        "sic": sic.tolist(), "display_x": x[1].tolist(), "display_y": y[1].tolist(),
        "percentiles": [16, 50, 84],
        "uncertainty_source": uncertainty_source,
        "display_band_x": x.tolist(), "display_band_y": y.tolist(),
    }


def aggregate_fit_metrics(records):
    """Median scalar diagnostics, with each original fit retained for auditing."""
    if len(records) == 1:
        return records[0]
    result = {}
    for key, value in records[0].items():
        values = [record[key] for record in records]
        if isinstance(value, dict):
            result[key] = value if all(v == value for v in values) else aggregate_fit_metrics(values)
        elif value is None or isinstance(value, (float, int, np.number)):
            result[key] = float(np.median(values)) if all(v is not None for v in values) else None
        elif all(v == value for v in values):
            result[key] = value
    result.update(fit_count=len(records), fits=records, aggregation="median of per-fit metrics")
    return result


def sr(mass):
    """Preserved LaCathode plotting convention; RIDDLE uses saved membership."""
    return (mass >= 3.3) & (mass <= 3.7)


def sample_sr(sample, key):
    if key in sample.get("sr_masks", {}):
        from .production import validate_region
        return validate_region(sample["sr_masks"][key], len(sample["mass"]))
    if key == "residual":
        raise ValueError("RIDDLE SR membership is missing; regenerate its score artifacts")
    return sr(sample["mass"])


def display(key, values):
    return expit(values) if key == "residual" else np.asarray(values)


def slug(value):
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def interval(k, n, confidence=0.95):
    k, n = np.broadcast_arrays(np.asarray(k), np.asarray(n))
    lo, hi = np.full(k.shape, np.nan), np.full(k.shape, np.nan)
    good = n > 0
    lo[good], hi[good] = 0, 1
    tail = (1 - confidence) / 2
    use = good & (k > 0)
    lo[use] = beta.ppf(tail, k[use], n[use] - k[use] + 1)
    use = good & (k < n)
    hi[use] = beta.ppf(1 - tail, k[use] + 1, n[use] - k[use])
    return lo, hi


def scalar(value):
    if value is None:
        return None
    value = float(value)
    return value if np.isfinite(value) else None


def safe_div(a, b):
    return scalar(a / b) if a is not None and b is not None and b > 0 else None


def choose_cut(sample, key, budget, strict=False):
    fits = sample_fit_scores(sample, key)
    if len(fits) > 1:
        rows = [choose_cut({**sample, key + "_scores": score, key + "_fit_scores": score[None, :]},
                           key, budget, strict) for score in fits]
        available = all(row["cut"] is not None for row in rows)
        return {"cut": [row["cut"] for row in rows] if available else None,
                "status": "available" if available else "unsupported_fit",
                "fit_count": len(rows), "fits": rows, "scope": "validation_sr"}
    mask = sample_sr(sample, key) & (sample["labels"] == 0)
    total = int(mask.sum())
    values = np.sort(sample[key + "_scores"][mask & sample["mask"]])
    allowed = math.floor(total * budget)
    if strict and total and allowed / total >= budget:
        allowed -= 1
    if not len(values) or allowed < 1 or values[0] == values[-1]:
        return {"cut": None, "status": "unsupported_sparse_or_constant", "background_total": total}
    cut = float(values[-allowed - 1]) if allowed < len(values) else float(np.nextafter(values[0], -np.inf))
    return {
        "cut": cut,
        "background_total": total,
        "background_passed": int((values > cut).sum()),
        "status": "available",
        "scope": "validation_sr",
        "strict": strict,
    }


def mass_edges(mass):
    values = np.asarray(mass, float)
    low, high = min(1.0, values.min()), max(9.0, values.max())
    candidates = np.unique([low, 2.0, 2.3, 2.6, 2.9, 3.3, 3.7, 4.1, 4.5, 5.5, high])
    minimum = max(1, math.ceil(0.03 * len(values)))
    edges = [low]
    for left, right in ((low, 3.3), (3.7, high)):
        region = candidates[(candidates >= left) & (candidates <= right)]
        merged, pending = [left], 0
        for edge, n in zip(region[1:], np.histogram(values, region)[0]):
            pending += n
            if pending >= minimum:
                merged.append(float(edge))
                pending = 0
        if merged[-1] != right:
            if len(merged) == 1:
                merged.append(right)
            else:
                merged[-1] = right
        if left == 3.7:
            edges.append(left)
        edges.extend(merged[1:])
    return np.asarray(edges)


def selected(sample, key, cut):
    fits = sample_fit_scores(sample, key)
    if len(fits) > 1:
        thresholds = np.asarray(cut)
        if thresholds.ndim:
            if thresholds.shape != (len(fits),):
                raise ValueError("Selection needs one validation threshold per LaCathode fit")
            thresholds = thresholds[:, None]
        return sample["mask"] & (fits > thresholds)
    return sample["mask"] & (sample[key + "_scores"] > cut)


def selection_stats(sample, key, cut, edges, confidence):
    fits = sample_fit_scores(sample, key)
    if len(fits) > 1:
        cuts = np.broadcast_to(cut, (len(fits),))
        rows = [selection_stats({**sample, key + "_scores": score, key + "_fit_scores": score[None, :]},
                                key, threshold, edges, confidence) for score, threshold in zip(fits, cuts)]
        out = {"cut": cuts.tolist(), "display_cut": display(key, cuts).tolist(),
               "status": "available", "fit_count": len(rows), "fits": rows,
               "aggregation": "mean per-fit counts/efficiencies; median per-fit shape diagnostics"}
        for name in ("signal", "background"):
            for suffix in ("", "_passed", "_total"):
                values = [r[name + suffix] for r in rows]
                out[name + suffix] = float(np.mean(values)) if all(v is not None for v in values) else None
            # Fit spread is not a binomial confidence interval on pooled events.
            out[name + "_ci_low"] = out[name + "_ci_high"] = None
        for name in ("total", "passed", "efficiency"):
            out[name] = np.mean([r[name] for r in rows], axis=0).tolist()
        out["overall_background_efficiency"] = np.mean([r["overall_background_efficiency"] for r in rows])
        for name in ("relative_mass_rms", "chi2_ndof", "chi2_bins"):
            values = [r[name] for r in rows]
            out[name] = float(np.median(values)) if all(v is not None for v in values) else None
        return out
    keep, region = selected(sample, key, cut), sample_sr(sample, key)
    out = {"cut": cut, "display_cut": float(display(key, cut)), "status": "available"}
    for name, label in (("signal", 1), ("background", 0)):
        population = region & (sample["labels"] == label)
        n, k = int(population.sum()), int((keep & population).sum())
        lo, hi = interval(k, n, confidence)
        out.update(
            {
                name + "_total": n,
                name + "_passed": k,
                name: safe_div(k, n),
                name + "_ci_low": scalar(lo),
                name + "_ci_high": scalar(hi),
            }
        )
    bmask = sample["labels"] == 0
    mass = sample["mass"].astype(float).copy()
    mass[region] = np.clip(mass[region], 3.3, np.nextafter(3.7, -np.inf))
    n, k = np.histogram(mass[bmask], edges)[0], np.histogram(mass[bmask & keep], edges)[0]
    rate = np.divide(k, n, out=np.full(len(n), np.nan), where=n > 0)
    overall = safe_div(int((bmask & keep).sum()), int(bmask.sum()))
    rms = np.sqrt(np.nanmean((rate / overall - 1) ** 2)) if overall else None
    valid = (n * (overall or 0) >= 10) & (n * (1 - (overall or 0)) >= 10)
    chi2 = (
        np.sum((k[valid] - n[valid] * overall) ** 2 / (n[valid] * overall * (1 - overall)))
        if overall and overall < 1
        else None
    )
    out.update(
        total=n.tolist(),
        passed=k.tolist(),
        efficiency=rate.tolist(),
        relative_mass_rms=scalar(rms),
        overall_background_efficiency=overall,
        chi2_ndof=safe_div(chi2, int(valid.sum()) - 1),
        chi2_bins=int(valid.sum()),
    )
    return out


def auc_components(labels, scores):
    s, b = np.asarray(scores)[labels == 1], np.asarray(scores)[labels == 0]
    if min(len(s), len(b)) < 2:
        return None
    bs, ss = np.sort(b), np.sort(s)
    vs = (np.searchsorted(bs, s, "left") + np.searchsorted(bs, s, "right")) / (2 * len(b))
    vb = 1 - (np.searchsorted(ss, b, "left") + np.searchsorted(ss, b, "right")) / (2 * len(s))
    return float(vs.mean()), vs, vb


def auc_interval(parts, confidence):
    if parts is None:
        return None, None
    mean, a, b = parts
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    half = norm.ppf((1 + confidence) / 2) * se
    return max(0.0, mean - half), min(1.0, mean + half)


def paired_auc_interval(parts, confidence):
    if any(parts[k] is None for k in METHODS):
        return None, None
    a, b = parts["raw"], parts["residual"]
    d1, d0 = b[1] - a[1], b[2] - a[2]
    half = norm.ppf((1 + confidence) / 2) * np.sqrt(d1.var(ddof=1) / len(d1) + d0.var(ddof=1) / len(d0))
    return max(-1.0, b[0] - a[0] - half), min(1.0, b[0] - a[0] + half)


def canvas(ylabel, xlabel, ratio=None):
    if ratio:
        fig, (ax, lower) = plt.subplots(
            2, 1, figsize=(7.2, 6.8), sharex=True, gridspec_kw={"height_ratios": [3.4, 1], "hspace": 0.07}
        )
        fig.subplots_adjust(left=0.16, right=0.97, top=0.96, bottom=0.12)
        lower.set(ylabel=ratio, xlabel=xlabel)
        lower.tick_params(labelsize=9)
        lower.yaxis.label.set_size(10)
        lower.minorticks_on()
        lower._publication_yaxis = "nonnegative" if ratio == "BG retention" else "ratio"
        ax.tick_params(labelbottom=False)
    else:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        fig.subplots_adjust(left=0.17, right=0.97, top=0.96, bottom=0.15)
        lower = None
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.minorticks_on()
    ax._publication_yaxis = _Y_AXIS_KINDS.get(ylabel)
    fig._publication_axis = ax
    fig._publication_ratio_axis = lower
    return fig, ax, lower


def legend(fig, handles=None, labels=None, *, ax=None, ncols=None, title=None, fontsize=10, headers=()):
    ax = ax or fig._publication_axis
    if handles is None:
        handles, labels = ax.get_legend_handles_labels()
    if handles:
        if not hasattr(fig, "_publication_legends"):
            fig._publication_legends = {}
        fig._publication_legends[ax] = {
            "handles": handles,
            "labels": labels,
            "ncols": ncols or (2 if len(handles) > 4 else 1),
            "title": title,
            "fontsize": fontsize,
            "headers": headers,
        }


def working_point_title(point, scope=None):
    relation = "<" if slug(point["name"]) == "extra_tight" else "≤"
    scope = "Signal region" if scope and scope.lower() in ("sr", "signal region") else "Full mass range"
    return f"{scope} | B {relation} {100 * point['validation_background_budget']:g}%"


def retention_label(truth, passed, total, *, scenario=None):
    if truth == 1 and scenario == "background_only":
        return "S" if total else None
    value = safe_div(passed, total)
    return ("B" if truth == 0 else "S") + (f": {100 * value:.3g}%" if value is not None else ": unavailable")


def signal_retention_label(key, passed, total, *, scenario=None):
    if scenario == "background_only":
        return METHODS[key][0]
    value = safe_div(passed, total)
    fraction = f"{100 * value:.3g}%" if value is not None else "n/a"
    return f"{METHODS[key][0]} (S: {fraction})"


def population_legend(fig, columns, title=None, *, ax=None):
    columns = [
        (heading, [(handle, label) for handle, label in entries if label is not None])
        for heading, entries in columns
    ]
    rows = 1 + max(len(entries) for _, entries in columns)
    handles, labels = [], []
    for heading, entries in columns:
        entries = [(None, heading), *entries]
        entries.extend([(None, "")] * (rows - len(entries)))
        for handle, label in entries:
            handles.append(handle if handle is not None else Line2D([], [], linestyle="none"))
            labels.append(label)
    if len(columns) >= 3 and ax is None:
        fig.set_figwidth(max(fig.get_figwidth(), 2.5 * len(columns)))
    legend(
        fig,
        handles,
        labels,
        ncols=len(columns),
        title=title,
        fontsize=9.5,
        headers=[name for name, _ in columns],
        ax=ax,
    )


def occupied_geometry(ax):
    paths, points = [], []
    for item in (*ax.lines, *ax.patches):
        if item.get_visible() and item.get_gid() != "publication-guide":
            paths.append(
                (
                    item.get_transform().transform_path(item.get_path()),
                    hasattr(item, "get_fill") and item.get_fill(),
                )
            )
    for item in ax.collections:
        if not item.get_visible():
            continue
        if isinstance(item, LineCollection):
            paths.extend((item.get_transform().transform_path(p), False) for p in item.get_paths())
        elif isinstance(item, PathCollection):
            points.extend(item.get_offset_transform().transform(item.get_offsets()))
        elif isinstance(item, PolyCollection):
            paths.extend((item.get_transform().transform_path(p), True) for p in item.get_paths())
    return paths, np.asarray(points).reshape(-1, 2)


def overlaps_data(ax, box, *, geometry=None):
    paths, points = occupied_geometry(ax) if geometry is None else geometry
    padded = box.padded(5)
    return any(p.intersects_bbox(padded, filled=filled) for p, filled in paths) or bool(
        len(points)
        and np.any(
            (points[:, 0] >= padded.x0)
            & (points[:, 0] <= padded.x1)
            & (points[:, 1] >= padded.y0)
            & (points[:, 1] <= padded.y1)
        )
    )


def place_legend(fig):
    for ax, spec in getattr(fig, "_publication_legends", {}).items():
        place_axis_legend(fig, ax, spec)


def place_axis_legend(fig, ax, spec):
    original_ylim = ax.get_ylim()
    base_size = spec["fontsize"]
    minimum = min(base_size, max(7.0, 0.8 * base_size))
    sizes = np.linspace(max(minimum, 0.95 * base_size), minimum, 4)
    positions = [(1, 1), (0, 1), (1, 0), (0, 0), (0.5, 1), (0.5, 0), (1, 0.5), (0, 0.5), (0.5, 0.5)]
    positions.extend(p for p in itertools.product(np.linspace(0, 1, 5), repeat=2) if p not in positions)
    for attempt in range(8):
        for fontsize in sizes:
            item = ax.legend(
                spec["handles"], spec["labels"], loc="lower left", ncol=spec["ncols"],
                title=spec["title"], fontsize=fontsize, title_fontsize=9.5 * fontsize / base_size,
                frameon=False, facecolor=ax.get_facecolor(), edgecolor="none", framealpha=0,
                handlelength=1.7, handletextpad=0.5, columnspacing=0.9,
                labelspacing=0.35, borderpad=0.25, borderaxespad=0,
            )
            for text in item.get_texts():
                if text.get_text() in spec["headers"]:
                    text.set_weight("bold")
            fig.canvas.draw()
            renderer = fig.canvas.get_renderer()
            area, size = ax.get_window_extent(renderer), item.get_window_extent(renderer)
            if size.width + 18 > area.width or size.height + 18 > area.height:
                continue
            geometry = occupied_geometry(ax)
            for x, y in positions:
                left = area.x0 + 9 + x * (area.width - size.width - 18)
                bottom = area.y0 + 9 + y * (area.height - size.height - 18)
                box = Bbox.from_bounds(left, bottom, size.width, size.height)
                if overlaps_data(ax, box, geometry=geometry):
                    continue
                anchor = ax.transAxes.inverted().transform((left, bottom))
                item.set_bbox_to_anchor(anchor, transform=ax.transAxes)
                actual = item.get_window_extent(renderer)
                if (
                    area.contains(actual.x0, actual.y0) and area.contains(actual.x1, actual.y1)
                    and not overlaps_data(ax, actual, geometry=geometry)
                ):
                    return
        if getattr(ax, "_publication_fixed_ylim", False) or attempt == 7:
            break
        transform = ax.yaxis.get_transform()
        low, high = transform.transform(ax.get_ylim())
        ax.set_ylim(*transform.inverted().transform([low, high + 0.3 * (high - low)]))
    # A crowded panel should never make the plotting campaign fail.  If an
    # overlap-free internal position does not exist, use a compact external
    # legend; bbox_inches="tight" keeps it in both vector and raster exports.
    ax.set_ylim(original_ylim)
    item = ax.legend(
        spec["handles"], spec["labels"], loc="upper left", bbox_to_anchor=(1.02, 1.0),
        ncol=spec["ncols"], title=spec["title"], fontsize=minimum,
        title_fontsize=9.5 * minimum / base_size, frameon=False,
        handlelength=1.8, handletextpad=0.5, columnspacing=0.9, labelspacing=0.35,
        borderaxespad=0,
    )
    for text in item.get_texts():
        if text.get_text() in spec["headers"]:
            text.set_weight("bold")


def publication_ylim(ax, kind):
    """Widen tight axes without altering, clipping or recomputing plotted data."""
    low, high = ax.dataLim.intervaly
    if not np.isfinite([low, high]).all():
        return
    margin = PUBLICATION_Y_RANGES["headroom"]
    if kind == "mass_flatness":
        positive = ax.dataLim.minposy
        if not np.isfinite(positive) or high <= 0:
            return
        low = min(PUBLICATION_Y_RANGES[kind][0], positive / (1 + margin))
        high = max(PUBLICATION_Y_RANGES[kind][1], high * (1 + margin))
        ax.set(yscale="log", ylim=(low, high))
        ax.yaxis.set_major_locator(LogLocator(base=10, subs=(1.,)))
        ax.yaxis.set_major_formatter(LogFormatterSciNotation(base=10, labelOnlyBase=True))
        ax.yaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(2., 10.)))
        ax.yaxis.set_minor_formatter(NullFormatter())
        return
    if kind in {"nonnegative", "sic"}:
        low, high = min(0., low), max(high * (1 + margin), np.finfo(float).tiny)
        if kind == "sic":
            high = max(PUBLICATION_Y_RANGES["sic"][1], high)
        elif high <= np.finfo(float).tiny:
            high = 1.
    elif kind == "ratio":
        minimum = PUBLICATION_Y_RANGES["ratio"]
        span = max(1 - minimum[0], minimum[1] - 1,
                   (1 + margin) * max(abs(low - 1), abs(high - 1)))
        low, high = min(low, max(0., 1 - span)), max(high, 1 + span)
    elif kind == "classification_loss":
        center = (low + high) / 2
        span = max(PUBLICATION_Y_RANGES[kind + "_min_span"], (high - low) * (1 + 2 * margin))
        low, high = center - span / 2, center + span / 2
    else:
        raise ValueError(f"Unknown publication y-axis kind: {kind}")
    locator = MaxNLocator(nbins=4 if kind == "ratio" else 5,
                         steps=[1, 2, 2.5, 5, 10], min_n_ticks=3)
    ticks = locator.tick_values(low, high)
    ax.set_ylim(ticks[0], ticks[-1])
    if ax.get_yscale() == "linear":
        ax.yaxis.set_major_locator(locator)
        ax.ticklabel_format(axis="y", style="plain", useOffset=False)


def _save_figure(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    for ax in fig.axes:
        kind = getattr(ax, "_publication_yaxis", None)
        if kind is not None:
            publication_ylim(ax, kind)
    place_legend(fig)
    metadata = {"Creator": "RIDDLE publication plotting framework"}
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06,
                facecolor="white", metadata=metadata)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.06,
                facecolor="white", dpi=600)


def save(fig, path):
    try:
        exporter = _FIGURE_EXPORTER.get()
        if exporter is None:
            _save_figure(fig, path)
        else:
            exporter.save(fig, path)
    finally:
        plt.close(fig)


def decorate_mass(ax, edges):
    ax.axvline(3.3, color=".65", lw=0.8, ls=":", gid="publication-guide")
    ax.axvline(3.7, color=".65", lw=0.8, ls=":", gid="publication-guide")
    ax.set_xlim(edges[0], edges[-1])


def render_roc(bundle, output, args):
    for dataset, (labels, scores) in bundle["curves"].items():
        if len(np.unique(labels)) != 2:
            bundle["warnings"].append(dataset + ": ROC unavailable without both classes")
            continue
        curves = {k: roc_curve(labels, v, drop_intermediate=False)[:2] for k, v in scores.items()}
        nb = int((labels == 0).sum())
        audit = bundle.setdefault("roc_audit", {}).setdefault(dataset, {})
        for key, (fpr, tpr) in curves.items():
            monotonic = bool(np.all(np.diff(fpr) >= 0) and np.all(np.diff(tpr) >= 0))
            if not monotonic:
                raise ValueError("ROC efficiencies are not monotonic")
            audit[METHODS[key][0]] = {
                "background_events": nb,
                "signal_events": int((labels == 1).sum()),
                "unique_scores": int(len(np.unique(scores[key]))),
                "auc": float(roc_auc_score(labels, scores[key])),
                "population": "conditional on successful mapping",
                "monotonic": monotonic,
                "interpolation_or_smoothing": False,
            }
        max_sic = 1.0
        for fpr, tpr in curves.values():
            keep = (fpr >= 1e-4) & (np.rint(fpr * nb) >= args.min_background)
            if keep.any():
                max_sic = max(max_sic, float(np.max(tpr[keep] / np.sqrt(fpr[keep]))))
        for view, keys in VIEWS.items():
            for kind in ("roc", "roc_log", "sic", "rejection"):
                if kind == "sic" and getattr(args, "population_summary", False):
                    continue
                ylabel = {
                    "roc": "Signal efficiency",
                    "roc_log": "Signal efficiency",
                    "sic": "Significance improvement",
                    "rejection": "Background rejection (1 / efficiency)",
                }[kind]
                fig, ax, _ = canvas(
                    ylabel, "Signal efficiency" if kind == "rejection" else "Background efficiency"
                )
                for key in keys:
                    label, color, ls = METHODS[key]
                    record = bundle.get("evaluation", {}).get(dataset, {}).get(key)
                    if record is not None and len(fit_scores(record)) > 1:
                        fit_curves = []
                        for score in fit_scores(record):
                            b, s, _ = roc_curve(labels, score[record["mask"]])
                            use = b > 0 if kind.startswith("roc") else (b >= 1e-4) & (np.rint(b * nb) >= args.min_background)
                            fit_curves.append((b[use], s[use]))
                        audit[label][kind] = draw_fit_curves(ax, fit_curves, kind, label, color, ls,
                                                           uncertainty_source=fit_uncertainty(record),
                                                           band=record.get("independent_runs", False),
                                                           run_groups=None if record.get("independent_runs", False) else [0] * len(fit_curves))
                        audit[label].update(fit_count=len(fit_curves),
                                            unique_scores=[int(len(np.unique(v[record["mask"]]))) for v in fit_scores(record)],
                                            interpolation_or_smoothing="linear interpolation on common signal efficiency; no smoothing",
                                            auc=float(np.median([roc_auc_score(labels, v[record["mask"]]) for v in fit_scores(record)])))
                        if kind == "sic":
                            max_sic = max([max_sic, *[float(np.max(s / np.sqrt(b))) for b, s in fit_curves if len(b)]])
                        continue
                    fpr, tpr = curves[key]
                    keep = (
                        np.ones(len(fpr), bool)
                        if kind == "roc"
                        else (
                            fpr > 0
                            if kind == "roc_log"
                            else (fpr >= 1e-4) & (np.rint(fpr * nb) >= args.min_background)
                        )
                    )
                    y = (
                        tpr[keep]
                        if kind.startswith("roc")
                        else (1 / fpr[keep] if kind == "rejection" else tpr[keep] / np.sqrt(fpr[keep]))
                    )
                    ax.plot(
                        tpr[keep] if kind == "rejection" else fpr[keep],
                        y,
                        color=color,
                        ls=ls,
                        marker="o" if keep.sum() == 1 else None,
                        label=label,
                    )
                grid = np.geomspace(1e-4, 1, 100)
                if kind == "roc":
                    grid = np.linspace(0, 1, 101)
                ax.plot(
                    grid,
                    grid if kind.startswith("roc") else (1 / grid if kind == "rejection" else np.sqrt(grid)),
                    color=".55",
                    ls=":",
                    lw=1,
                    label="Random",
                )
                if kind == "rejection":
                    ax.set(yscale="log", xlim=(0, 1), ylim=(0.9, min(1e4, nb / args.min_background) * 1.15))
                elif kind == "roc":
                    ax.set(xlim=(0, 1), ylim=(0, 1.02))
                else:
                    ax.set(
                        xscale="log", xlim=(1e-4, 1), ylim=(0, 1.02 if kind == "roc_log" else max_sic * 1.08)
                    )
                legend(fig, title=SCENARIO_LABELS.get(bundle.get("metrics", {}).get("scenario")))
                save(fig, output / view / "01_performance" / f"{dataset}_{kind}")


def render_scores(bundle, output):
    if not bundle["samples"]:
        return
    sample = bundle["samples"]["test"]
    edges = np.linspace(0, 1, 41)
    for region in ("sr", "full"):
        histograms = {}
        for key in METHODS:
            mask = sample_sr(sample, key) if region == "sr" else np.ones(len(sample["mass"]), bool)
            for truth in (0, 1):
                population = mask & sample["mask"] & (sample["labels"] == truth)
                counts = fit_histogram(display(key, sample_fit_scores(sample, key)), edges, population)
                histograms[key, truth] = (
                    counts / (population.sum() * np.diff(edges)) if population.any() else np.zeros(40)
                )
        ymax = max(max(v) for v in histograms.values()) * 1.1 or 1
        for view, keys in VIEWS.items():
            fig, ax, _ = canvas("Density (unit area per class)", "Score")
            for key in keys:
                label = METHODS[key][0]
                for truth, name in ((0, "B"), (1, "S")):
                    color, ls = POPULATIONS[key][truth]
                    ax.stairs(histograms[key, truth], edges, color=color, ls=ls, label=f"{label}: {name}")
            ax.set(xlim=(0, 1), ylim=(0, ymax))
            legend(fig)
            save(fig, output / view / "01_performance" / f"{region}_score_density")
            for truth, name in ((0, "background"), (1, "signal")):
                fig, ax, _ = canvas("Probability density", "Score")
                for key in keys:
                    label = METHODS[key][0]
                    color, ls = POPULATIONS[key][truth]
                    ax.stairs(histograms[key, truth], edges, color=color, ls=ls, label=label)
                ax.set(xlim=(0, 1), ylim=(0, ymax))
                legend(fig)
                save(fig, output / view / "01_performance/score_distributions" / f"{region}_{name}")


def retention_interval(row, confidence):
    if "fits" in row:
        return np.percentile([r["efficiency"] for r in row["fits"]], [16, 84], axis=0)
    return interval(row["passed"], row["total"], confidence)


def render_efficiency(bundle, output, args):
    metrics = bundle["metrics"]
    edges = np.asarray(metrics["mass_edges"])
    centers = (edges[1:] + edges[:-1]) / 2
    for point in metrics["working_points"]:
        name, target = slug(point["name"]), point["validation_background_budget"]
        rows = point["methods"]
        highs = [
            np.asarray(retention_interval(row, args.confidence)[1])
            for row in rows.values()
            if "passed" in row
        ]
        ymax = max([target * 1.2, *[float(np.nanmax(h)) for h in highs]]) * 100 * 1.08
        for view, keys in VIEWS.items():
            fig, ax, _ = canvas("Background efficiency [%]", r"$m_{jj}$ [TeV]")
            ax.axhline(100 * target, color=".6", lw=1, label="Target")
            for key in keys:
                row = rows[key]
                if "passed" not in row:
                    continue
                label, color, _ = METHODS[key]
                n, k = np.asarray(row["total"]), np.asarray(row["passed"])
                rate = np.divide(k, n, out=np.full(n.shape, np.nan), where=n > 0)
                low, high = retention_interval(row, args.confidence)
                ax.errorbar(
                    centers,
                    rate * 100,
                    xerr=np.diff(edges) / 2,
                    yerr=None if "fits" in row else np.array([rate - low, high - rate]) * 100,
                    fmt="o" if key == "raw" else "s",
                    ms=3,
                    color=color,
                    lw=1,
                    label=label,
                )
                if "fits" in row and bundle["evaluation"]["test"][key].get("independent_runs", False):
                    ax.vlines(centers, low * 100, high * 100, color=color, lw=1)
            decorate_mass(ax, edges)
            ax.set_ylim(0, ymax)
            legend(fig, title=working_point_title(point))
            save(fig, output / view / "03_mass_sculpting" / f"background_efficiency_{name}")
    for view, keys in VIEWS.items():
        fig, ax, _ = canvas("Relative mass-efficiency RMS [%]", "Background selection target [%]")
        ymax = max(
            [
                1.0,
                *[
                    100 * p["methods"][k].get("relative_mass_rms", 0)
                    for p in metrics["working_points"]
                    for k in METHODS
                    if p["methods"][k].get("relative_mass_rms") is not None
                ],
            ]
        )
        for key in keys:
            pairs = [
                (100 * p["validation_background_budget"], 100 * p["methods"][key]["relative_mass_rms"])
                for p in metrics["working_points"]
                if p["methods"][key].get("relative_mass_rms") is not None
            ]
            if pairs:
                pairs.sort()
                label, color, ls = METHODS[key]
                ax.plot(*np.asarray(pairs).T, marker="o", color=color, ls=ls, label=label)
        ticks = sorted({100 * p["validation_background_budget"] for p in metrics["working_points"]})
        ax.set(
            xscale="linear",
            xlim=(0, max(ticks) * 1.05),
            xticks=ticks,
            xticklabels=[f"{v:g}" for v in ticks],
            ylim=(0, ymax * 1.1),
        )
        legend(fig)
        save(fig, output / view / "03_mass_sculpting/mass_dependence_summary")
    scan = metrics.get("mass_comparisons", {}).get("scan", [])
    values = [r["methods"][k].get("chi2_ndof") for r in scan for k in METHODS]
    finite = [v for v in values if v is not None and np.isfinite(v)]
    if finite and not (
        getattr(args, "population_summary", False) and metrics.get("scenario") == "background_only"
    ):
        for view, keys in VIEWS.items():
            fig, ax, _ = canvas(r"$\chi^2/n_{\mathrm{dof}}$", "Background selection target")
            for key in keys:
                pairs = [
                    (p["validation_sr_budget"], p["methods"][key]["chi2_ndof"])
                    for p in scan
                    if p["methods"][key].get("chi2_ndof") is not None
                ]
                if pairs:
                    label, color, ls = METHODS[key]
                    ax.plot(*np.asarray(pairs).T, marker="o", ms=3, color=color, ls=ls, label=label)
            ax.axhline(1, color=".5", lw=1, ls=":")
            ax.set(xlim=(0.205, 0), ylim=(0, max(1.0, max(finite)) * 1.3))
            legend(fig, title=SCENARIO_LABELS.get(metrics.get("scenario")))
            save(fig, output / view / "03_mass_sculpting/mass_flatness_vs_selection")


def step(ax, counts, edges, **kwargs):
    return ax.stairs(np.asarray(counts), edges, baseline=None, **kwargs)


def count_scale(ax, ymax, linear_threshold=1.0):
    ax.set_yscale("symlog", linthresh=linear_threshold, linscale=0.5)
    ax.set_ylim(-0.15 * linear_threshold, ymax)


def normalized_counts(counts, edges):
    counts = np.asarray(counts)
    return counts / (counts.sum() * np.diff(edges)) if counts.sum() else np.full(len(counts), np.nan)


def mass_histograms(bundle, point):
    if bundle["samples"]:
        sample = bundle["samples"]["test"]
        mass, labels = sample["mass"], sample["labels"]
        edges = np.linspace(min(1.0, float(mass.min())), max(9.0, float(mass.max())), 81)
        nominal = {y: np.histogram(mass[labels == y], edges)[0] for y in (0, 1)}
        histograms = {}
        for key in METHODS:
            cut = point["methods"][key].get("cut")
            if cut is not None:
                mask = selected(sample, key, cut)
                histograms[key] = {y: fit_histogram(mass, edges, mask & (labels == y)) for y in (0, 1)}
        return edges, nominal, histograms
    return None


def bin_ratio(numerator, denominator):
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    return np.divide(
        numerator,
        denominator,
        out=np.full(numerator.shape, np.nan),
        where=np.isfinite(denominator) & (denominator > 0),
    )


def finish_ratio(fig, ax, edges, curves):
    values = np.concatenate([np.asarray(v)[np.isfinite(v)] for v in curves]) if curves else np.array([])
    span = max(0.2, float(np.max(np.abs(values - 1))) * 1.12) if len(values) else 0.2
    ax.axhline(1, color=".5", lw=0.8)
    decorate_mass(ax, edges)
    ax.set_ylim(max(0, 1 - span), 1 + span)
    legend(fig, ax=ax, fontsize=8, ncols=1)


def render_mass(bundle, output):
    scenario = bundle["metrics"].get("scenario")
    for point in bundle["metrics"]["working_points"]:
        records = mass_histograms(bundle, point)
        if records is None:
            continue
        edges, nominal, histograms = records
        name = slug(point["name"])
        ymax = max(1, max(nominal[0]), max(nominal[1])) * 1.3
        density_b = normalized_counts(nominal[0], edges)
        shapes = {k: normalized_counts(h[0], edges) for k, h in histograms.items()}
        positive = np.concatenate([v[v > 0] for v in (density_b, *shapes.values())])
        shape_max = max(positive) * 1.3 if len(positive) else 1.0
        shape_threshold = max(1e-7, min(positive) * 0.4) if len(positive) else 1e-4
        for view, keys in VIEWS.items():
            fig, ax, _ = canvas("Events / bin", r"$m_{jj}$ [TeV]")
            no_cut, columns = [], []
            for y in (0, 1):
                handle = step(
                    ax,
                    nominal[y],
                    edges,
                    color=".55" if y == 0 else ".75",
                    ls="-" if y == 0 else "--",
                    label="No cut: " + ("B" if y == 0 else "S"),
                )
                no_cut.append((handle, retention_label(
                    y, nominal[y].sum(), nominal[y].sum(), scenario=scenario
                )))
            columns.append(("No cut", no_cut))
            for key in keys:
                entries = []
                for y in (0, 1):
                    color, ls = POPULATIONS[key][y]
                    counts = histograms.get(key, {}).get(y)
                    handle = (
                        step(
                            ax,
                            counts,
                            edges,
                            color=color,
                            ls=ls,
                            label=METHODS[key][0] + (": B" if y == 0 else ": S"),
                        )
                        if counts is not None
                        else None
                    )
                    entries.append(
                        (
                            handle,
                            retention_label(
                                y, counts.sum() if counts is not None else None, nominal[y].sum(),
                                scenario=scenario,
                            ),
                        )
                    )
                columns.append((METHODS[key][0], entries))
            decorate_mass(ax, edges)
            count_scale(ax, ymax)
            population_legend(fig, columns, working_point_title(point, "full mass range"))
            save(fig, output / view / "04_mass_cuts" / f"mass_{name}")

            for kind, ylabel in (
                ("counts", "Background events / bin"),
                ("retention", "Background kept / all"),
                ("shape", r"Normalized density [TeV$^{-1}$]"),
            ):
                comparison = len(keys) == 2
                ratio_label = (
                    "Selected / inclusive"
                    if kind == "shape"
                    else (
                        "RIDDLE / LaCathode"
                        if comparison
                        else ("Selected / inclusive" if kind == "counts" else "Efficiency / overall")
                    )
                )
                fig, ax, lower = canvas(ylabel, r"$m_{jj}$ [TeV]", ratio=ratio_label)
                if kind == "counts":
                    step(ax, nominal[0], edges, color=".65", label="Inclusive background")
                elif kind == "shape":
                    ax.stairs(density_b, edges, fill=True, color=".85", label="Inclusive background")
                rates = {}
                for key in keys:
                    if key not in histograms:
                        continue
                    label = signal_retention_label(
                        key, histograms[key][1].sum(), nominal[1].sum(), scenario=scenario
                    )
                    color, ls = POPULATIONS[key][0]
                    counts = histograms[key][0]
                    rates[key] = bin_ratio(counts, nominal[0])
                    values = counts if kind == "counts" else shapes[key] if kind == "shape" else rates[key]
                    step(ax, values, edges, color=color, ls=ls, label=label)
                decorate_mass(ax, edges)
                if kind == "counts":
                    count_scale(ax, ymax)
                elif kind == "shape":
                    count_scale(ax, shape_max, shape_threshold)
                else:
                    finite = [v[np.isfinite(v)] for v in rates.values()]
                    top = max([0.01, *[float(v.max()) for v in finite if len(v)]]) * 1.2
                    ax.set_ylim(0, top)
                legend(fig, title=working_point_title(point))
                ratios = []
                if kind == "shape":
                    for key in keys:
                        if key in histograms:
                            values = bin_ratio(shapes[key], density_b)
                            step(
                                lower,
                                values,
                                edges,
                                color=POPULATIONS[key][0][0],
                                label=METHODS[key][0] + " / Inc. BG",
                            )
                            ratios.append(values)
                elif comparison and all(k in histograms for k in keys):
                    values = bin_ratio(histograms["residual"][0], histograms["raw"][0])
                    step(lower, values, edges, color="black", label="RIDDLE / LaCathode")
                    ratios.append(values)
                elif not comparison and keys[0] in histograms:
                    key = keys[0]
                    values = rates[key]
                    if kind == "retention":
                        overall = safe_div(histograms[key][0].sum(), nominal[0].sum())
                        values = values / overall if overall else np.full(len(values), np.nan)
                    step(
                        lower,
                        values,
                        edges,
                        color=POPULATIONS[key][0][0],
                        label=METHODS[key][0] + (" / Inc. BG" if kind == "counts" else " / overall"),
                    )
                    ratios.append(values)
                finish_ratio(fig, lower, edges, ratios)
                save(fig, output / view / "03_mass_sculpting" / f"background_mass_{kind}_{name}")


def strict_cut_histograms(values, edges, scores, mask, thresholds):
    """Exact unweighted histograms for many strict cuts, in one event pass.

    Bucket by the number of thresholds passed, then accumulate from high to
    low. Match numpy.histogram's closed final bin, and exclude NaN scores just
    as a direct score > cut comparison does. Preserve unsorted/repeated cuts.
    """
    thresholds = np.asarray(thresholds)
    bins = len(edges) - 1
    if not len(thresholds):
        return np.zeros((0, bins), dtype=np.int64)
    order = np.argsort(thresholds, kind="stable")
    bin_id = np.searchsorted(edges, values, side="right") - 1
    bin_id[values == edges[-1]] = bins - 1
    valid = mask & ~np.isnan(scores) & (bin_id >= 0) & (bin_id < bins)
    passed = np.searchsorted(thresholds[order], scores[valid], side="left")
    counts = np.bincount(passed * bins + bin_id[valid], minlength=(len(thresholds) + 1) * bins)
    cumulative = np.cumsum(counts.reshape(-1, bins)[::-1], axis=0)[::-1][1:]
    result = np.empty_like(cumulative)
    result[order] = cumulative
    # A NaN threshold never passes, irrespective of its sorted position.
    result[np.isnan(thresholds)] = 0
    return result


def mass_scan_histograms(sample, keys, cuts=SCORE_CUTS):
    """Histogram strict cuts in each method's displayed 0--1 score coordinate.

    RIDDLE stores log density ratios, so invert the display sigmoid before
    selecting events. Neither scores nor thresholds are fitted on test labels.
    Retentions use all physical test events, including rejected mapping rows.
    """
    from scipy.special import logit

    mass, labels = sample["mass"], sample["labels"]
    edges = np.linspace(min(1.0, float(mass.min())), max(9.0, float(mass.max())), 81)
    nominal = {y: np.histogram(mass[labels == y], edges)[0] for y in (0, 1)}
    cuts = tuple(cuts)
    for cut in cuts:
        if not np.isfinite(cut) or not 0 < cut < 1:
            raise ValueError("Mass-scan score thresholds must be strictly between zero and one")
    histograms = {}
    for key in keys:
        # Retain the scalar logit calculation and stored-array precision, so
        # events exactly at a cut fail identically to selected().
        thresholds = np.asarray([logit(cut) if key == "residual" else cut for cut in cuts],
                                dtype=sample[key + "_scores"].dtype)
        fits = sample_fit_scores(sample, key)
        if len(fits) == 1:
            fits = sample[key + "_scores"][None, :]
        histograms[key] = {}
        for y in (0, 1):
            counts = np.asarray([strict_cut_histograms(mass, edges, score, sample["mask"] & (labels == y),
                                                      thresholds) for score in fits])
            histograms[key][y] = counts[0] if len(counts) == 1 else counts.mean(axis=0)
    records = [(cut, {key: {y: histograms[key][y][i] for y in (0, 1)} for key in keys})
               for i, cut in enumerate(cuts)]
    return edges, nominal, records


def draw_mass_scan(fig, ax, edges, nominal, histograms, keys, cut, *, scenario=None):
    """Draw the same counts and legend in the individual and six-panel exports."""
    keys = tuple(key for key in POPULATIONS if key in keys)
    ax.set(xlabel=r"$m_{jj}$ [TeV]", ylabel="Events / bin")
    ax.minorticks_on()
    columns = []
    for key in (None, *keys):
        counts = nominal if key is None else histograms[key]
        heading = "No cut" if key is None else METHODS[key][0]
        total_style = (".65", "-") if key is None else (".22", "-" if key == "raw" else "-.")
        total = step(
            ax,
            counts[0] + counts[1],
            edges,
            color=total_style[0],
            ls=total_style[1],
            lw=2.1,
        )
        entries = [(total, "B + S")]
        for y in (0, 1):
            color, ls = (
                (("#9CBACB", "-") if y == 0 else ("#BDA8B4", "--"))
                if key is None
                else POPULATIONS[key][y]
            )
            handle = step(ax, counts[y], edges, color=color, ls=ls, lw=1.35)
            entries.append((handle, retention_label(
                y, counts[y].sum(), nominal[y].sum(), scenario=scenario
            )))
        columns.append((heading, entries))
    decorate_mass(ax, edges)
    count_scale(ax, max(1, float(np.max(nominal[0] + nominal[1]))) * 1.3)
    population_legend(fig, columns, f"Full mass range | Score > {cut:.2f}", ax=ax)


def render_mass_scan(bundle, output):
    """Save all 70 fixed-score cuts individually and six per multipage PDF."""
    from .storage import atomic_write
    from .worker_progress import ProgressStage

    sample = bundle["samples"].get("test")
    if sample is None or not len(sample["mass"]):
        return
    scenario = bundle["metrics"].get("scenario")
    edges, nominal, records = mass_scan_histograms(sample, tuple(METHODS), cuts=SCORE_CUTS)
    for view, keys in VIEWS.items():
        destination = output / view / "04_mass_cuts"
        individual = destination / "individual_cuts"

        def write_scan(temporary):
            metadata = {
                "Title": "Mass cut scan",
                "Subject": "Fixed displayed-score thresholds from 0.30 to 0.99",
            }
            with PdfPages(temporary, metadata=metadata) as pdf:
                with ProgressStage(
                    "mass_scan_" + view, "Plot mass cut scan: " + view, len(records), "cut"
                ) as progress:
                    for first in range(0, len(records), 6):
                        page, axes = plt.subplots(3, 2, figsize=(16, 17.2))
                        page.subplots_adjust(
                            left=0.075, right=0.985, bottom=0.055, top=0.985,
                            wspace=0.22, hspace=0.25,
                        )
                        try:
                            chunk = records[first : first + 6]
                            for ax, (cut, histograms) in zip(axes.flat, chunk):
                                draw_mass_scan(
                                    page, ax, edges, nominal, histograms, keys, cut, scenario=scenario
                                )
                                fig, single, _ = canvas("Events / bin", r"$m_{jj}$ [TeV]")
                                if len(keys) == 2:
                                    fig.set_figwidth(7.6)
                                try:
                                    draw_mass_scan(
                                        fig, single, edges, nominal, histograms, keys, cut, scenario=scenario
                                    )
                                    save(fig, individual / ("mass_score_" + f"{cut:.2f}".replace(".", "p")))
                                finally:
                                    plt.close(fig)
                            for ax in list(axes.flat)[len(chunk) :]:
                                ax.set_axis_off()
                            place_legend(page)
                            pdf.savefig(page)
                            progress.update(first + len(chunk))
                        finally:
                            plt.close(page)

        atomic_write(destination / "mass_cut_scan.pdf", write_scan)


def render_features(bundle, output):
    records = feature_records(bundle)
    if records is None:
        return
    scenario = bundle["metrics"].get("scenario")
    for view, keys in VIEWS.items():
        for page in records["pages"]:
            region = "sr" if page["region"].lower() in ("sr", "signal region") else "full"
            point = next(
                p
                for p in bundle["metrics"]["working_points"]
                if slug(p["name"]) == slug(page["working_point"])
            )
            target = output / view / "05_features" / region / slug(page["working_point"])
            for i, feature in enumerate(page["features"]):
                is_score = feature["name"] == "Score"
                edges = np.asarray(feature["edges"])
                ymax = max(
                    10, *[max(c["no_cut"]) * 1.3 for v in feature["methods"].values() for c in v["classes"]]
                )
                for kind, ylabel in (("counts", "Events / bin"), ("retention", "Kept / all")):
                    fig, ax, _ = canvas(ylabel, feature["name"])
                    no_cut, columns = [], []
                    for key in keys:
                        name = METHODS[key][0]
                        entries = []
                        for truth, cls in enumerate(feature["methods"][key]["classes"]):
                            color, ls = POPULATIONS[key][truth]
                            label = "B" if truth == 0 else "S"
                            base = np.asarray(cls["no_cut"])
                            if key == keys[0] or is_score:
                                values = base if kind == "counts" else bin_ratio(base, base)
                                handle = step(
                                    ax,
                                    values,
                                    edges,
                                    color=(".55" if truth == 0 else ".75") if not is_score else color,
                                    alpha=0.6 if not is_score else 0.45,
                                    ls=ls,
                                    lw=0.9,
                                    label=f"No cut: {name} {label}",
                                )
                                text = retention_label(truth, base.sum(), base.sum(), scenario=scenario)
                                if text is not None and is_score and len(keys) > 1:
                                    text = (
                                        text.replace(":", f" ({name}):", 1)
                                        if ":" in text else f"{text} ({name})"
                                    )
                                no_cut.append((handle, text))
                            handle, passed = None, None
                            if cls["selected"] is not None:
                                kept = np.asarray(cls["selected"])
                                if np.any(kept > base):
                                    raise ValueError("Selected feature counts exceed no-cut counts")
                                passed = kept.sum()
                                values = kept if kind == "counts" else bin_ratio(kept, base)
                                handle = step(ax, values, edges, color=color, ls=ls, label=f"{name}: {label}")
                            entries.append((handle, retention_label(
                                truth, passed, base.sum(), scenario=scenario
                            )))
                        columns.append((name, entries))
                    ax.set_xlim(edges[0], edges[-1])
                    if kind == "counts":
                        count_scale(ax, ymax)
                    else:
                        ax.set_ylim(0, 1.05)
                    population_legend(
                        fig,
                        [("No cut", no_cut), *columns],
                        working_point_title(point, "SR" if region == "sr" else "full mass range"),
                    )
                    save(
                        fig,
                        target
                        / (
                            feature.get("id", FEATURE_IDS[i] if i < len(FEATURE_IDS) else "deltaR")
                            + "_"
                            + kind
                        ),
                    )
            if bundle.get("verbose"):
                print(f"[WORK] {view} features: {region}, {slug(page['working_point'])}", flush=True)


def feature_density(values, edges):
    return (
        np.histogram(values, edges)[0] / (len(values) * np.diff(edges))
        if len(values)
        else np.zeros(len(edges) - 1)
    )


def render_representation(bundle, output, *, regions=None):
    """Render event-level input/latent diagnostics as one figure per quantity.

    ``regions`` can restrict rendering to ``("sr",)`` or ``("full",)``.  When
    exactly one region is requested the region name is omitted from the path;
    the caller is expected to have already placed the figure below the
    top-level ``SR-Only`` or ``Full-Range`` directory.  This keeps publication
    outputs flat enough to browse while preserving the legacy two-region mode.
    """
    sample = bundle["samples"].get("test")
    if sample is None:
        bundle["warnings"].append("Input/latent plots unavailable without event-level features")
        return
    requested = tuple(regions or ("sr", "full"))
    if not requested or any(region not in ("sr", "full") for region in requested):
        raise ValueError("Representation regions must be 'sr' and/or 'full'")
    spaces = {}
    if "feature_values" in bundle:
        labels = (r"$m_{J1}$ [TeV]", r"$\Delta m_J$ [TeV]", r"$\tau_{21,J1}$", r"$\tau_{21,J2}$")
        names = ("jet1_mass", "jet_mass_difference", "tau21_jet1", "tau21_jet2")
        if bundle.get("variant") == "shifted":
            labels = (r"$m_{J1}+0.1m_{jj}$ [TeV]", r"$\Delta m_J+0.1m_{jj}$ [TeV]", *labels[2:])
        if bundle["feature_values"].shape[1] == 6:
            labels, names = (*labels, r"$\Delta R_{jj}$"), (*names, "deltaR")
        spaces["inputs"] = (
            bundle["feature_values"][sample["mask"], 1:],
            labels,
            names,
        )
    if "latents" in bundle:
        spaces["latent"] = (
            bundle["latents"],
            tuple(r"$z_{%d}$" % i for i in range(1, bundle["latents"].shape[-1] + 1)),
            tuple("z%d" % i for i in range(1, bundle["latents"].shape[-1] + 1)),
        )
    y, m = sample["labels"][sample["mask"]], sample["mass"][sample["mask"]]
    bundle["representation"] = {
        "event_scope": "same scorable physical test events before and after mapping",
        "latent_reference": "standard normal; not a signal model",
        "histograms": "Mean per-run densities; no averaging of event latent coordinates",
        "regions": {},
    }
    region_mask = sample_sr(sample, next(iter(METHODS)))[sample["mask"]]
    region_masks = {"sr": region_mask, "full": np.ones(len(m), bool)}
    single_region = len(requested) == 1
    for region in requested:
        mask = region_masks[region]
        groups = {label: mask & (y == truth) for label, truth in (("background", 0), ("signal", 1))}
        record = {"events": {name: int(v.sum()) for name, v in groups.items()}, "spaces": {}}
        bundle["representation"]["regions"][region] = record
        for space, (values, labels, names) in spaces.items():
            if not mask.any():
                continue
            edges = []
            dimensions = values.shape[-1]
            cohort = values if values.ndim == 3 else values[None, :, :]
            for i in range(dimensions):
                low, high = (
                    (-5.0, 5.0)
                    if space == "latent"
                    else (float(cohort[:, mask, i].min()), float(cohort[:, mask, i].max()))
                )
                if high == low:
                    low, high = low - 0.5, high + 0.5
                edges.append(np.linspace(low, high, 61))
            record["spaces"][space] = {
                "ranges": [[e[0], e[-1]] for e in edges],
                "out_of_range": {
                    group: [
                        float(np.mean([np.count_nonzero(pop & ((v[:, i] < e[0]) | (v[:, i] > e[-1])))
                                       for v in cohort]))
                        for i, e in enumerate(edges)
                    ]
                    for group, pop in groups.items()
                },
            }
            for i in range(dimensions):
                densities = {
                    group: np.mean([feature_density(v[pop, i], edges[i]) for v in cohort], axis=0)
                    for group, pop in groups.items()
                }
                top = max(0.4 if space == "latent" else 0, *[max(v) for v in densities.values()]) * 1.25
                for view in VIEWS:
                    target = output / view / "06_input_latent"
                    if not single_region:
                        target /= region
                    target /= space
                    fig, ax, _ = canvas("Probability density", labels[i])
                    key = VIEWS[view][0]
                    for truth, group in ((0, "background"), (1, "signal")):
                        color, ls = POPULATIONS[key][truth]
                        if groups[group].any():
                            ax.stairs(densities[group], edges[i], color=color, ls=ls, label=group.title())
                    if space == "latent":
                        grid = np.linspace(-5, 5, 300)
                        ax.plot(grid, norm.pdf(grid), color=".5", ls=":", label=r"$\mathcal{N}(0,1)$")
                    ax.set(xlim=(edges[i][0], edges[i][-1]), ylim=(0, top))
                    legend(fig)
                    save(fig, target / ("density_" + names[i]))
            for i, j in itertools.combinations(range(dimensions), 2):
                histograms = {}
                for group, pop in groups.items():
                    if pop.any():
                        h = np.mean([np.histogram2d(v[pop, i], v[pop, j], bins=(edges[i], edges[j]))[0]
                                     for v in cohort], axis=0)
                        histograms[group] = h / (
                            pop.sum() * np.diff(edges[i])[:, None] * np.diff(edges[j])[None, :]
                        )
                maximum = max([1e-6, *[h.max() for h in histograms.values()]])
                for view in VIEWS:
                    base = output / view / "06_input_latent"
                    if not single_region:
                        base /= region
                    base /= space
                    for group, hist in histograms.items():
                        fig, ax, _ = canvas(labels[j], labels[i])
                        mesh = ax.pcolormesh(
                            edges[i], edges[j], np.ma.masked_less_equal(hist.T, 0),
                            norm=LogNorm(vmin=maximum * 1e-4, vmax=maximum),
                            cmap="viridis", rasterized=True,
                        )
                        fig.colorbar(mesh, ax=ax, label="Probability density", pad=0.025)
                        ax.set(title=group.title(), xlim=(edges[i][0], edges[i][-1]), ylim=(edges[j][0], edges[j][-1]))
                        save(fig, base / "pairs" / f"{group}_{names[i]}_{names[j]}")
            for group, pop in groups.items():
                if pop.sum() < 3 or any(np.any(np.std(v[pop], axis=0) == 0) for v in cohort):
                    continue
                matrix = np.mean([np.corrcoef(v[pop], rowvar=False) for v in cohort], axis=0)
                for view in VIEWS:
                    base = output / view / "06_input_latent"
                    if not single_region:
                        base /= region
                    base /= space
                    fig, ax, _ = canvas("", "")
                    art = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
                    ax.set(xticks=range(dimensions), yticks=range(dimensions),
                           xticklabels=labels, yticklabels=labels, title=group.title())
                    ax.tick_params(axis="x", labelsize=9, rotation=20)
                    ax.tick_params(axis="y", labelsize=9)
                    ax.minorticks_off()
                    for (a, b), value in np.ndenumerate(matrix):
                        ax.text(b, a, f"{value:.2f}", ha="center", va="center",
                                color="white" if abs(value) > 0.65 else "black")
                    fig.colorbar(art, ax=ax, label="Pearson correlation", pad=0.025)
                    save(fig, base / f"correlation_{group}")
            if bundle.get("verbose"):
                print(f"[WORK] Input/latent distributions: {region}, {space}", flush=True)

def equal_occupancy(mass, bins=300):
    mass = np.asarray(mass)
    if len(mass) < bins or not np.isfinite(mass).all():
        raise ValueError("Insufficient background events for 300-bin mass sculpting")
    edges = np.interp(np.linspace(0, len(mass), bins + 1), np.arange(len(mass)), np.sort(mass))
    if np.any(np.diff(edges) <= 0):
        raise ValueError("Equal-occupancy mass edges are not distinct")
    return edges


def shape_chi2(full, selected, efficiency):
    full, selected = np.asarray(full, float), np.asarray(selected, float)
    if selected.sum() <= 0:
        return None
    if full.shape != selected.shape or len(full) < 2 or np.any(full <= 0) or np.any(selected < 0):
        raise ValueError("Invalid mass histogram counts")
    expected = efficiency * full
    a, b = 1 / expected.sum(), 1 / selected.sum()
    return float(np.sum((a * expected - b * selected) ** 2 / (a * a * expected)) / (len(full) - 1))


def central68(values):
    values = np.asarray(values, float)
    if values.ndim != 2 or not len(values):
        raise ValueError("Expected one metric curve per fit/run")
    supported = np.isfinite(values).all(axis=0)
    summary = np.full((3, values.shape[1]), np.nan)
    summary[:, supported] = np.percentile(values[:, supported], [16, 50, 84], axis=0)
    return summary


def summary_axes(ax, metric):
    ax.set(**SUMMARY_AXES[metric])
    ax._publication_yaxis = metric


def draw_band(ax, x, values, label, color, linestyle, *, band=True):
    low, median, high = central68(values)
    drawn = bool(band and len(values) >= 2 and np.isfinite(median).any())
    if drawn:
        ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, median, color=color, ls=linestyle, label=label)
    return {
        "low": low.tolist(),
        "median": median.tolist(),
        "high": high.tolist(),

        "independent_runs": len(values),
        "band_drawn": drawn,
        "band_status": "drawn"
        if drawn
        else "requires_multiple_curves_with_common_support"
        if band
        else "disabled",
        "contributing_runs": np.isfinite(values).sum(axis=0).tolist(),
    }


def build_metrics(bundle, confidence):
    if "metrics" in bundle:
        return
    samples = bundle["samples"]
    edges = mass_edges(samples["test"]["mass"][samples["test"]["labels"] == 0])
    points = []
    for name, budget in BUDGETS:
        point = {"name": name, "validation_background_budget": budget, "methods": {}}
        for key in METHODS:
            chosen = choose_cut(samples["validation"], key, budget, strict=name == "extra_tight")
            row = {"validation": chosen, "status": chosen["status"]}
            if chosen["cut"] is not None:
                row.update(selection_stats(samples["test"], key, chosen["cut"], edges, confidence))
                row["sr_efficiency"] = {k: row[k] for k in ("signal", "background")}
            else:
                bundle["warnings"].append(
                    f"{METHODS[key][0]} {name}: insufficient validation tail statistics or constant scores"
                )
            point["methods"][key] = row
        points.append(point)
    report = bundle["report"]
    bundle["metrics"] = {
        "seed": report["seed"],
        "scenario": report["scenario"],
        "smoke": report.get("smoke"),
        "working_points": points,
        "working_point_protocol": "Truth-assisted MC benchmark: validation-SR background cuts, frozen before independent test evaluation; full physical class denominators",
        "oracle_maximum_protocol": "Test-truth-optimized SIC is an oracle benchmark, not a deployable threshold",
        "mass_edges": edges.tolist(),
        "protocol": report.get("protocol", {}),
        "ensemble": report.get("ensemble"),
        "score_numerics_version": report.get("score_numerics_version"),
    }
    scan = []
    for budget in MASS_TARGETS:
        row = {"validation_sr_budget": budget, "methods": {}}
        for key in METHODS:
            chosen = choose_cut(samples["validation"], key, budget, strict=budget == 0.004)
            row["methods"][key] = (
                selection_stats(samples["test"], key, chosen["cut"], edges, confidence)
                if chosen["cut"] is not None
                else {"status": chosen["status"]}
            )
        scan.append(row)
    bundle["metrics"]["mass_comparisons"] = {"scan": scan}


def feature_records(bundle):
    if bundle.get("features"):
        return bundle["features"]
    if "feature_values" not in bundle:
        return None
    values = bundle["feature_values"]
    sample = bundle["samples"]["test"]
    m, m1, dm, t1, t2 = values[:, :5].T
    physical = [
        m,
        m1,
        m1 + dm,
        dm,
        t1,
        t2,
        np.divide(dm, 2 * m1 + dm, out=np.zeros_like(dm), where=2 * m1 + dm != 0),
    ]
    names = [
        r"$m_{jj}$ [TeV]",
        r"$m_{J1}$ [TeV]",
        r"$m_{J2}$ [TeV]",
        r"$\Delta m_J$ [TeV]",
        r"$\tau_{21,J1}$",
        r"$\tau_{21,J2}$",
        r"$\Delta m_J/(m_{J1}+m_{J2})$",
        "Score",
    ]
    ids = list(FEATURE_IDS)
    if bundle.get("variant") == "shifted":
        names[1:4] = [
            r"$m_{J1}+0.1m_{jj}$ [TeV]",
            r"$m_{J2}+0.2m_{jj}$ [TeV]",
            r"$\Delta m_J+0.1m_{jj}$ [TeV]",
        ]
        names[6] = r"$(\Delta m_J+0.1m_{jj})/(m_{J1}+m_{J2}+0.3m_{jj})$"
    if values.shape[1] == 6:
        physical.append(values[:, 5])
        names.insert(-1, r"$\Delta R_{jj}$")
        ids.insert(-1, "deltaR")
    records = {"pages": []}
    for region in ("SR", "Full mass range"):
        for point in bundle["metrics"]["working_points"]:
            page = {"region": region, "working_point": point["name"], "features": []}
            for i, name in enumerate(names):
                if name == "Score":
                    edges = np.linspace(0, 1, 41)
                elif i == 0 and region == "SR":
                    edges = np.linspace(*np.asarray([3.3, 3.7], dtype=m.dtype).astype(float), 31)
                else:
                    low, high = float(physical[i].min()), float(physical[i].max())
                    width = max((high - low) * 0.001, 1e-5)
                    edges = np.linspace(low - width, high + width, 41)
                feature = {"id": ids[i], "name": name, "edges": edges.tolist(), "methods": {}}
                for key in METHODS:
                    scope = (
                        (sample_sr(sample, key) if key == "residual" else sr(m))
                        if region == "SR"
                        else np.ones(len(m), bool)
                    )
                    x = physical[i] if i < len(physical) else display(key, sample_fit_scores(sample, key))
                    classes = []
                    for label in (0, 1):
                        pop = scope & (sample["labels"] == label) & np.isfinite(x)
                        h = fit_histogram(x, edges, pop)
                        cut = point["methods"][key].get("cut")
                        kept = (
                            fit_histogram(x, edges, pop & selected(sample, key, cut))
                            if cut is not None
                            else None
                        )
                        classes.append(
                            {"no_cut": h.tolist(), "selected": kept.tolist() if kept is not None else None}
                        )
                    feature["methods"][key] = {"classes": classes}
                page["features"].append(feature)
            records["pages"].append(page)
    return records


FEATURE_IDS = (
    "mjj",
    "jet1_mass",
    "jet2_mass",
    "jet_mass_difference",
    "tau21_jet1",
    "tau21_jet2",
    "jet_mass_asymmetry",
    "score",
)
