import itertools
import math
import re
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.collections import LineCollection, PathCollection, PolyCollection
from matplotlib.colors import LogNorm
from matplotlib.lines import Line2D
from matplotlib.transforms import Bbox
from mplhep import style as hep_style
import numpy as np
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
    "sic": dict(xscale="log", xlim=(1e-4, 1), yscale="linear", ylim=(0, 17)),
    "mass_flatness": dict(xscale="linear", xlim=(0.20, 0.01), yscale="log", ylim=(0.5, 350)),
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
    "axes.labelsize": 13,
    "axes.titlesize": 11,
    "legend.fontsize": 10,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "axes.linewidth": 1.1,
    "lines.linewidth": 1.5,
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
    "savefig.dpi": 300,
}


def sr(mass):
    """Preserved LaCathode plotting convention; RIDDLE uses saved membership."""
    return (mass >= 3.3) & (mass <= 3.7)


def sample_sr(sample, key):
    if key == "residual":
        from .production import validate_region

        if key not in sample.get("sr_masks", {}):
            raise ValueError("RIDDLE SR membership is missing; regenerate its score artifacts")
        return validate_region(sample["sr_masks"][key], len(sample["mass"]))
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
    return sample["mask"] & (sample[key + "_scores"] > cut)


def selection_stats(sample, key, cut, edges, confidence):
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
        ax.tick_params(labelbottom=False)
    else:
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        fig.subplots_adjust(left=0.17, right=0.97, top=0.96, bottom=0.15)
        lower = None
        ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.minorticks_on()
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
    return f"{scope} | B {relation} {100 * point['validation_background_budget']:g}% (MC truth)"


def retention_label(truth, passed, total):
    value = safe_div(passed, total)
    return ("B" if truth == 0 else "S") + (f": {100 * value:.3g}%" if value is not None else ": unavailable")


def signal_retention_label(key, passed, total):
    value = safe_div(passed, total)
    fraction = f"{100 * value:.3g}%" if value is not None else "n/a"
    return f"{METHODS[key][0]} (S: {fraction})"


def population_legend(fig, columns, title=None, *, ax=None):
    rows = 1 + max(len(entries) for _, entries in columns)
    handles, labels = [], []
    for heading, entries in columns:
        entries = [(None, heading), *entries]
        entries.extend([(None, "")] * (rows - len(entries)))
        for handle, label in entries:
            handles.append(handle if handle is not None else Line2D([], [], linestyle="none"))
            labels.append(label)
    if len(columns) == 3 and ax is None:
        fig.set_figwidth(7.6)
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
        if item.get_visible():
            paths.append(
                (
                    item.get_transform().transform_path(item.get_path()),
                    hasattr(item, "get_fill") and item.get_fill(),
                )
            )
    for item in ax.collections:
        if isinstance(item, LineCollection):
            paths.extend((item.get_transform().transform_path(p), False) for p in item.get_paths())
        elif isinstance(item, PathCollection):
            points.extend(item.get_offset_transform().transform(item.get_offsets()))
        elif isinstance(item, PolyCollection):
            paths.extend((item.get_transform().transform_path(p), True) for p in item.get_paths())
    return paths, np.asarray(points).reshape(-1, 2)


def overlaps_data(ax, box):
    paths, points = occupied_geometry(ax)
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
    item = ax.legend(
        spec["handles"],
        spec["labels"],
        loc="lower left",
        ncol=spec["ncols"],
        title=spec["title"],
        fontsize=spec["fontsize"],
        title_fontsize=9.5,
        frameon=False,
        handlelength=1.9,
        columnspacing=1.4,
        borderaxespad=0,
    )
    for text in item.get_texts():
        if text.get_text() in spec["headers"]:
            text.set_weight("bold")
    for _ in range(8):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        area, size = ax.get_window_extent(renderer), item.get_window_extent(renderer)
        for x, y in ((1, 1), (0, 1), (1, 0), (0, 0), (0.5, 1), (0.5, 0), (1, 0.5), (0, 0.5), (0.5, 0.5)):
            left = area.x0 + 9 + x * (area.width - size.width - 18)
            bottom = area.y0 + 9 + y * (area.height - size.height - 18)
            box = Bbox.from_bounds(left, bottom, size.width, size.height)
            if (
                size.width + 18 <= area.width
                and size.height + 18 <= area.height
                and not overlaps_data(ax, box)
            ):
                anchor = ax.transAxes.inverted().transform((left, bottom))
                item.set_bbox_to_anchor(anchor, transform=ax.transAxes)
                fig.canvas.draw()
                if not overlaps_data(ax, item.get_window_extent(fig.canvas.get_renderer())):
                    return
        if getattr(ax, "_publication_fixed_ylim", False):
            break
        transform = ax.yaxis.get_transform()
        low, high = transform.transform(ax.get_ylim())
        ax.set_ylim(*transform.inverted().transform([low, high + 0.3 * (high - low)]))
    raise ValueError("Cannot place an internal legend without covering data")


def save(fig, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    place_legend(fig)
    fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.06)
    fig.savefig(path.with_suffix(".png"), bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)


def decorate_mass(ax, edges):
    ax.axvline(3.3, color=".65", lw=0.8, ls=":")
    ax.axvline(3.7, color=".65", lw=0.8, ls=":")
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
                legend(fig, title=SCENARIO_LABELS.get(bundle.get("metrics", {}).get("scenario"), "") + " | Mapped events")
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
                values = display(
                    key, sample[key + "_scores"][mask & sample["mask"] & (sample["labels"] == truth)]
                )
                counts = np.histogram(values, edges)[0]
                histograms[key, truth] = (
                    counts / (len(values) * np.diff(edges)) if len(values) else np.zeros(40)
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


def render_efficiency(bundle, output, args):
    metrics = bundle["metrics"]
    edges = np.asarray(metrics["mass_edges"])
    centers = (edges[1:] + edges[:-1]) / 2
    for point in metrics["working_points"]:
        name, target = slug(point["name"]), point["validation_background_budget"]
        rows = point["methods"]
        highs = [
            np.asarray(interval(row["passed"], row["total"], args.confidence)[1])
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
                low, high = interval(k, n, args.confidence)
                ax.errorbar(
                    centers,
                    rate * 100,
                    xerr=np.diff(edges) / 2,
                    yerr=np.array([rate - low, high - rate]) * 100,
                    fmt="o" if key == "raw" else "s",
                    ms=3,
                    color=color,
                    lw=1,
                    label=label,
                )
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
                histograms[key] = {y: np.histogram(mass[mask & (labels == y)], edges)[0] for y in (0, 1)}
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
                no_cut.append((handle, retention_label(y, nominal[y].sum(), nominal[y].sum())))
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
                                y, counts.sum() if counts is not None else None, nominal[y].sum()
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
                    label = signal_retention_label(key, histograms[key][1].sum(), nominal[1].sum())
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
    records = []
    for cut in cuts:
        if not np.isfinite(cut) or not 0 < cut < 1:
            raise ValueError("Mass-scan score thresholds must be strictly between zero and one")
        histograms = {}
        for key in keys:
            # Match the stored array precision so events exactly at a cut fail.
            threshold = logit(cut) if key == "residual" else cut
            threshold = np.asarray(threshold, dtype=sample[key + "_scores"].dtype)
            keep = selected(sample, key, threshold)
            histograms[key] = {
                y: np.histogram(mass[keep & (labels == y)], edges)[0] for y in (0, 1)
            }
        records.append((cut, histograms))
    return edges, nominal, records


def draw_mass_scan(fig, ax, edges, nominal, histograms, keys, cut):
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
            entries.append((handle, retention_label(y, counts[y].sum(), nominal[y].sum())))
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
    edges, nominal, records = mass_scan_histograms(sample, tuple(METHODS))
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
                                draw_mass_scan(page, ax, edges, nominal, histograms, keys, cut)
                                fig, single, _ = canvas("Events / bin", r"$m_{jj}$ [TeV]")
                                if len(keys) == 2:
                                    fig.set_figwidth(7.6)
                                try:
                                    draw_mass_scan(fig, single, edges, nominal, histograms, keys, cut)
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
                                text = retention_label(truth, base.sum(), base.sum())
                                if is_score and len(keys) > 1:
                                    text = text.replace(":", f" ({name}):", 1)
                                no_cut.append((handle, text))
                            handle, passed = None, None
                            if cls["selected"] is not None:
                                kept = np.asarray(cls["selected"])
                                if np.any(kept > base):
                                    raise ValueError("Selected feature counts exceed no-cut counts")
                                passed = kept.sum()
                                values = kept if kind == "counts" else bin_ratio(kept, base)
                                handle = step(ax, values, edges, color=color, ls=ls, label=f"{name}: {label}")
                            entries.append((handle, retention_label(truth, passed, base.sum())))
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


def render_representation(bundle, output):
    sample = bundle["samples"].get("test")
    if sample is None:
        bundle["warnings"].append("Input/latent plots unavailable without event-level features")
        return
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
            tuple(r"$z_{%d}$" % i for i in range(1, bundle["latents"].shape[1] + 1)),
            tuple("z%d" % i for i in range(1, bundle["latents"].shape[1] + 1)),
        )
    y, m = sample["labels"][sample["mask"]], sample["mass"][sample["mask"]]
    bundle["representation"] = {
        "event_scope": "same scorable physical test events before and after mapping",
        "latent_reference": "standard normal; not a signal model",
        "regions": {},
    }
    region_mask = sample_sr(sample, next(iter(METHODS)))[sample["mask"]]
    for region, mask in (("sr", region_mask), ("full", np.ones(len(m), bool))):
        groups = {label: mask & (y == truth) for label, truth in (("background", 0), ("signal", 1))}
        record = {"events": {name: int(v.sum()) for name, v in groups.items()}, "spaces": {}}
        bundle["representation"]["regions"][region] = record
        for space, (values, labels, names) in spaces.items():
            if not mask.any():
                continue
            edges = []
            dimensions = values.shape[1]
            for i in range(dimensions):
                low, high = (
                    (-5.0, 5.0)
                    if space == "latent"
                    else (float(values[mask, i].min()), float(values[mask, i].max()))
                )
                if high == low:
                    low, high = low - 0.5, high + 0.5
                edges.append(np.linspace(low, high, 61))
            record["spaces"][space] = {
                "ranges": [[e[0], e[-1]] for e in edges],
                "out_of_range": {
                    group: [
                        int(np.count_nonzero(pop & ((values[:, i] < e[0]) | (values[:, i] > e[-1]))))
                        for i, e in enumerate(edges)
                    ]
                    for group, pop in groups.items()
                },
            }
            for i in range(dimensions):
                densities = {
                    group: feature_density(values[pop, i], edges[i]) for group, pop in groups.items()
                }
                top = max(0.4 if space == "latent" else 0, *[max(v) for v in densities.values()]) * 1.25
                for view in VIEWS:
                    target = output / view / "06_input_latent" / region / space
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
                        h = np.histogram2d(values[pop, i], values[pop, j], bins=(edges[i], edges[j]))[0]
                        histograms[group] = h / (
                            pop.sum() * np.diff(edges[i])[:, None] * np.diff(edges[j])[None, :]
                        )
                maximum = max([1e-6, *[h.max() for h in histograms.values()]])
                for view in VIEWS:
                    for group, hist in histograms.items():
                        fig, ax, _ = canvas(labels[j], labels[i])
                        mesh = ax.pcolormesh(
                            edges[i],
                            edges[j],
                            np.ma.masked_less_equal(hist.T, 0),
                            norm=LogNorm(vmin=maximum * 1e-4, vmax=maximum),
                            cmap="viridis",
                            rasterized=True,
                        )
                        fig.colorbar(mesh, ax=ax, label="Probability density", pad=0.025)
                        ax.set(
                            title=group.title(),
                            xlim=(edges[i][0], edges[i][-1]),
                            ylim=(edges[j][0], edges[j][-1]),
                        )
                        save(
                            fig,
                            output
                            / view
                            / "06_input_latent"
                            / region
                            / space
                            / "pairs"
                            / f"{group}_{names[i]}_{names[j]}",
                        )
            for group, pop in groups.items():
                if pop.sum() < 3 or np.any(np.std(values[pop], axis=0) == 0):
                    continue
                matrix = np.corrcoef(values[pop], rowvar=False)
                for view in VIEWS:
                    fig, ax, _ = canvas("", "")
                    art = ax.imshow(matrix, vmin=-1, vmax=1, cmap="RdBu_r")
                    ax.set(
                        xticks=range(dimensions),
                        yticks=range(dimensions),
                        xticklabels=labels,
                        yticklabels=labels,
                        title=group.title(),
                    )
                    ax.tick_params(axis="x", labelsize=9, rotation=20)
                    ax.tick_params(axis="y", labelsize=9)
                    ax.minorticks_off()
                    for (a, b), value in np.ndenumerate(matrix):
                        ax.text(
                            b,
                            a,
                            f"{value:.2f}",
                            ha="center",
                            va="center",
                            color="white" if abs(value) > 0.65 else "black",
                        )
                    fig.colorbar(art, ax=ax, label="Pearson correlation", pad=0.025)
                    save(fig, output / view / "06_input_latent" / region / space / f"correlation_{group}")
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
        raise ValueError("Expected one metric curve per independent seed")
    supported = np.isfinite(values).all(axis=0)
    summary = np.full((3, values.shape[1]), np.nan)
    summary[:, supported] = np.percentile(values[:, supported], [16, 50, 84], axis=0)
    return summary


def summary_axes(ax, metric):
    ax.set(**SUMMARY_AXES[metric])
    ax._publication_fixed_ylim = True


def draw_band(ax, x, values, label, color, linestyle, *, band=True):
    low, median, high = central68(values)
    if band and len(values) >= 2:
        ax.fill_between(x, low, high, color=color, alpha=0.18, linewidth=0)
    ax.plot(x, median, color=color, ls=linestyle, label=label)
    return {
        "low": low.tolist(),
        "median": median.tolist(),
        "high": high.tolist(),
        "independent_runs": len(values),
        "band_drawn": bool(band and len(values) >= 2),
        "band_status": "drawn"
        if band and len(values) >= 2
        else "requires_multiple_independent_runs"
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
                    x = physical[i] if i < len(physical) else display(key, sample[key + "_scores"])
                    classes = []
                    for label in (0, 1):
                        pop = scope & (sample["labels"] == label) & np.isfinite(x)
                        h = np.histogram(x[pop], edges)[0]
                        cut = point["methods"][key].get("cut")
                        kept = (
                            np.histogram(x[pop & selected(sample, key, cut)], edges)[0]
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
