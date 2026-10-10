from pathlib import Path

import numpy as np


CUTS = (0.10, 0.05, 0.01, 0.005)
MASS_CLOSURE_PROTOCOL = "local_poisson_mass_closure_v2"
MASS_CLOSURE_RANGE = (2.8, 5.0)
CMS_MASS_BINS_TEV = np.array([
    1, 3, 6, 10, 16, 23, 31, 40, 50, 61, 74, 88, 103, 119, 137, 156, 176, 197, 220,
    244, 270, 296, 325, 354, 386, 419, 453, 489, 526, 565, 606, 649, 693, 740, 788,
    838, 890, 944, 1000, 1058, 1118, 1181, 1246, 1313, 1383, 1455, 1530, 1607, 1687,
    1770, 1856, 1945, 2037, 2132, 2231, 2332, 2438, 2546, 2659, 2775, 2895, 3019,
    3147, 3279, 3416, 3558, 3704, 3854, 4010, 4171, 4337, 4509, 4686, 4869, 5058,
    5253, 5455, 5663, 5877, 6099, 6328, 6564, 6808, 7060, 7320, 7589, 7866, 8152,
    8447, 8752, 9067, 9391, 9726, 10072, 10430, 10798, 11179, 11571, 11977, 12395,
    12827, 13272, 13732, 14000,
], dtype=float) / 1000


def event_keys(ids):
    ids = np.asarray(ids)
    if ids.ndim != 2 or ids.shape[1] != 2 or ids.dtype.kind not in "ui":
        raise ValueError("Expected two-column integer event IDs")
    values = np.ascontiguousarray(ids, dtype=np.uint64).view("V16").ravel()
    if len(np.unique(values)) != len(values):
        raise ValueError("Duplicate event IDs")
    return values


def match_ids(source_ids, target_ids):
    source, target = event_keys(source_ids), event_keys(target_ids)
    order = np.argsort(source, kind="stable")
    index = np.searchsorted(source[order], target)
    if (np.any(index >= len(source))
            or not np.array_equal(source[order[np.minimum(index, len(source) - 1)]], target)):
        raise ValueError("Requested event IDs are missing from the saved population")
    return order[index]


def mass_edges(mass, *, split_sr=False):
    mass = np.asarray(mass)
    if mass.ndim != 1 or not len(mass) or not np.isfinite(mass).all():
        raise ValueError("Invalid full-mass population")
    low = np.searchsorted(CMS_MASS_BINS_TEV, mass.min(), side="right") - 1
    high = np.searchsorted(CMS_MASS_BINS_TEV, mass.max(), side="left")
    if low < 0 or high >= len(CMS_MASS_BINS_TEV):
        raise ValueError("Mass population is outside the supplied dijet binning")
    edges = CMS_MASS_BINS_TEV[low:high + 1].copy()
    if split_sr:
        edges = np.unique(np.r_[edges, [b for b in (3.3, 3.7) if edges[0] < b < edges[-1]]])
    return edges


def background_closure_fit(edges, counts):
    from scipy.optimize import minimize
    from scipy.special import logsumexp, xlogy
    from scipy.stats import chi2

    edges, counts = np.asarray(edges, float), np.asarray(counts, float)
    if (edges.ndim != 1 or counts.shape != (len(edges) - 1,) or not np.isfinite(edges).all()
            or not np.isfinite(counts).all() or np.any(np.diff(edges) <= 0)
            or np.any(counts < 0) or np.any(counts != np.floor(counts))):
        raise ValueError("Invalid background counts for mass closure")
    keep = (edges >= MASS_CLOSURE_RANGE[0]) & (edges <= MASS_CLOSURE_RANGE[1])
    keep &= ~(((edges > 3.25) & (edges < 3.3)) | ((edges > 3.7) & (edges < 3.75)))
    positions = np.flatnonzero(keep)
    if len(positions) < 7 or not all(np.any(edges[positions] == b) for b in (3.3, 3.7)):
        return dict(status="inconclusive", reason="Insufficient mass coverage around the SR")
    local_edges = edges[positions]
    cumulative = np.r_[0, np.cumsum(counts)]
    local_counts = np.diff(cumulative[positions])
    widths = np.diff(local_edges)
    sr = (local_edges[:-1] >= 3.3) & (local_edges[1:] <= 3.7)
    fit_bins = ~sr
    result = dict(requested_mass_range_TeV=list(MASS_CLOSURE_RANGE),
                  fitted_mass_range_TeV=[float(local_edges[0]), float(local_edges[-1])],
                  fitted_bin_edges_TeV=local_edges.tolist(),
                  sideband_events=int(local_counts[fit_bins].sum()),
                  observed_SR_background=int(local_counts[sr].sum()),
                  covariance="Inverse observed Poisson likelihood Hessian; exploratory diagnostic")
    if (fit_bins.sum() < 5 or min(np.sum(fit_bins & (local_edges[1:] <= 3.3)),
                                np.sum(fit_bins & (local_edges[:-1] >= 3.7))) < 2
            or result["sideband_events"] < 50):
        return dict(result, status="inconclusive", reason="Insufficient sideband statistics")
    nodes, weights = np.polynomial.legendre.leggauss(8)
    log_mass = np.log((local_edges[:-1, None] + local_edges[1:, None]) / 2
                      + widths[:, None] * nodes / 2)
    raw = np.stack((log_mass, log_mass ** 2), axis=-1)
    selected = raw[fit_bins].reshape(-1, 2)
    center = selected.mean(axis=0)
    _, singular, vectors = np.linalg.svd(selected - center, full_matrices=False)
    if singular[-1] <= singular[0] * 1e-12:
        return dict(result, status="inconclusive", reason="Singular mass-fit design")
    transform = vectors.T * np.sqrt(len(selected)) / singular
    design = np.concatenate((np.ones((*raw.shape[:-1], 1)), (raw - center) @ transform), axis=-1)
    log_weights = np.log(widths[:, None] * weights / 2)
    observed = local_counts[fit_bins]

    def components(parameters, full=False):
        basis = design if full else design[fit_bins]
        terms = basis @ parameters + (log_weights if full else log_weights[fit_bins])
        log_mean = logsumexp(terms, axis=1)
        mean = np.exp(np.minimum(log_mean, 700))
        probability = np.exp(terms - log_mean[:, None])
        first = np.einsum("bn,bnk->bk", probability, basis)
        second = np.einsum("bn,bnk,bnl->bkl", probability, basis, basis)
        return log_mean, mean, first, second

    def objective(parameters):
        log_mean, mean, first, _ = components(parameters)
        value = np.sum(mean - observed + xlogy(observed, observed) - observed * log_mean)
        return float(value), np.einsum("b,bk->k", mean - observed, first)

    def hessian(parameters):
        _, mean, first, second = components(parameters)
        covariance = second - np.einsum("bk,bl->bkl", first, first)
        return np.einsum("b,bkl->kl", mean, second) - np.einsum("b,bkl->kl", observed, covariance)

    initial = np.array([np.log(result["sideband_events"] / widths[fit_bins].sum()), 0.0, 0.0])
    candidates = []
    for offset in (0.0, 0.25, -0.25):
        optimized = minimize(lambda x: objective(x)[0], initial + [0, offset, offset],
                             jac=lambda x: objective(x)[1], hess=hessian, method="trust-exact",
                             options=dict(gtol=1e-5, maxiter=100))
        if not np.isfinite(optimized.fun) or not np.isfinite(optimized.x).all():
            continue
        gradient = objective(optimized.x)[1]
        if np.isfinite(gradient).all() and np.linalg.norm(gradient) / result["sideband_events"] < 1e-7:
            candidates.append(optimized)
    if not candidates:
        return dict(result, status="inconclusive", reason="Mass fit did not converge")
    best = min(candidates, key=lambda fit: fit.fun)
    curvature = hessian(best.x)
    if not np.isfinite(curvature).all() or np.linalg.eigvalsh(curvature).min() <= 0:
        return dict(result, status="inconclusive", reason="Mass-fit covariance is not positive definite")
    _, expected, first, _ = components(best.x, full=True)
    if not np.isfinite(expected).all() or np.any(expected <= 0):
        return dict(result, status="inconclusive", reason="Invalid fitted background expectation")
    gradient = np.einsum("b,bk->k", expected[sr], first[sr])
    variance = float(gradient @ np.linalg.solve(curvature, gradient))
    prediction = float(expected[sr].sum())
    pull = (result["observed_SR_background"] - prediction) / np.sqrt(prediction + variance)
    deviance = max(0.0, float(2 * best.fun))
    degrees = int(fit_bins.sum() - 3)
    p_value = float(chi2.sf(deviance, degrees))
    failures = [name for name, failed in (("sideband_shape", p_value <= 0.001),
                                         ("SR_extrapolation", abs(pull) >= 5)) if failed]
    coefficients = np.r_[best.x[0] - center @ transform @ best.x[1:], transform @ best.x[1:]]
    return dict(result, status="failed" if failures else "passed", failed_checks=failures,
                sideband_deviance=deviance, degrees_of_freedom=degrees,
                approximate_p_value=p_value, predicted_SR_background=prediction,
                approximate_fit_variance=variance, approximate_SR_pull=float(pull),
                coefficients=coefficients.tolist(),
                minimum_fitted_sideband_count=float(expected[fit_bins].min()),
                sparse_sideband_bins=int(np.sum(expected[fit_bins] < 5)),
                scaled_gradient=float(np.linalg.norm(objective(best.x)[1]) / result["sideband_events"]))


def normalization_weights(record, path=None):
    if path is None:
        return np.ones(len(record["mass"])), "events", None
    from .storage import file_digest

    path = Path(path)
    with np.load(path, allow_pickle=False) as archive:
        weights = np.asarray(archive["weights_pb"], dtype=float)
        ids = archive["event_ids"]
    if weights.shape != (len(ids),) or not np.isfinite(weights).all() or np.any(weights < 0):
        raise ValueError("Invalid physical per-event cross-section weights in pb")
    return weights[match_ids(ids, record["event_ids"])], "pb", file_digest(path)


def spectrum_histograms(record, weights, edges, *, population=None, selections=None):
    mass, labels = np.asarray(record["mass"]), np.asarray(record["labels"])
    weights = np.asarray(weights, dtype=float)
    masks = np.asarray(record["cut_masks"] if selections is None else [selections[cut] for cut in CUTS], dtype=float)
    population = np.ones(len(mass), bool) if population is None else np.asarray(population, dtype=bool)
    if (masks.shape != (len(CUTS), len(mass)) or weights.shape != mass.shape
            or population.shape != mass.shape or labels.shape != mass.shape
            or not np.isfinite(weights).all() or np.any(weights < 0)
            or not np.isfinite(masks).all() or np.any((masks < 0) | (masks > 1))):
        raise ValueError("Misaligned full-mass selections or normalization weights")
    output = {}
    for cut, selected in [(None, np.ones(len(mass), bool)), *zip(CUTS, masks)]:
        row = {}
        for name, component in (("background", labels == 0), ("signal", labels == 1),
                                ("data", np.ones(len(mass), bool))):
            take = population & component & (selected > 0)
            event_weights = weights[take] * selected[take]
            row[name] = np.histogram(mass[take], edges, weights=event_weights)[0]
            row[name + "_variance"] = np.histogram(mass[take], edges, weights=event_weights ** 2)[0]
        output[cut] = row
    return output


def mass_ylabel(units):
    return (r"$d\sigma/dm_{jj}$ [pb/TeV]" if units == "pb"
            else r"$dN/dm_{jj}$ [events/TeV]")


def spectrum_summary(rows, cut):
    summary = {}
    for component in ("data", "background", "signal"):
        values = np.asarray([row[cut][component] for row in rows], dtype=float)
        variances = np.asarray([row[cut][component + "_variance"] for row in rows], dtype=float)
        summary[component] = values.mean(axis=0)
        summary[component + "_variance"] = variances.mean(axis=0)
        summary[component + "_min"] = values.min(axis=0)
        summary[component + "_max"] = values.max(axis=0)
    summary["seeds"] = len(rows)
    return summary


def mixture_errors(values, variance):
    from scipy.special import gammaincinv

    values, variance = np.asarray(values, float), np.asarray(variance, float)
    positive = (values > 0) & (variance > 0)
    effective = np.divide(values ** 2, variance, out=np.zeros_like(values), where=positive)
    scale = np.divide(variance, values, out=np.ones_like(values), where=positive)
    alpha = (1.0 - 0.6827) / 2
    lower = np.zeros_like(values)
    lower[positive] = scale[positive] * gammaincinv(effective[positive], alpha)
    upper = scale * gammaincinv(effective + 1, 1 - alpha)
    return np.maximum(values - lower, 0), np.maximum(upper - values, 0)


def spectrum_residual(summary):
    values = summary["data"]
    low, high = mixture_errors(values, summary["data_variance"])
    difference = values - summary["background"]
    error = np.where(difference < 0, high, low)
    error = np.where(error > 0, error, np.maximum(low, high))
    valid = (values > 0) & (error > 0)
    residual = np.divide(difference, error, out=np.zeros_like(values), where=valid)
    return residual, valid


def residual_limits(summaries):
    values = []
    for cut in CUTS:
        residual, valid = spectrum_residual(summaries[cut])
        values.extend(residual[valid].tolist())
    low, high = min([0.0, *values]), max([0.0, *values])
    padding = max(0.5, (high - low) * 0.12)
    return low - padding, high + padding


def spectrum_style(ax):
    import matplotlib as mpl

    scale = 0.8 * ax.figure.get_figwidth() / (8.6 / 2.54)
    ax.xaxis.label.set_fontsize(mpl.rcParams["axes.labelsize"] * scale)
    ax.yaxis.label.set_fontsize(mpl.rcParams["axes.labelsize"] * scale)
    ax.tick_params(axis="x", labelsize=mpl.rcParams["xtick.labelsize"] * scale)
    ax.tick_params(axis="y", labelsize=mpl.rcParams["ytick.labelsize"] * scale)


def selection_label(cut):
    return rf"$\epsilon_B={100 if cut is None else cut * 100:g}\%$"


def signal_region_lines(ax, bounds=(3.3, 3.7)):
    for boundary in bounds:
        line = ax.axvline(boundary, color="#555555", ls="--", lw=0.8, zorder=1)
        line.set_gid("signal_region_boundary")


def mass_legend_overlap(ax, legend):
    from matplotlib.collections import PathCollection

    box = legend.get_window_extent(ax.figure.canvas.get_renderer()).padded(2)
    for artist in [*ax.lines, *ax.patches, *ax.collections]:
        if not artist.get_visible() or artist.get_gid() == "signal_region_boundary":
            continue
        if isinstance(artist, PathCollection):
            offsets = artist.get_offset_transform().transform(artist.get_offsets())
            if any(box.contains(x, y) for x, y in offsets):
                return True
        else:
            paths = artist.get_paths() if hasattr(artist, "get_paths") else [artist.get_path()]
            for path in paths:
                path = path.transformed(artist.get_transform())
                if path.intersects_bbox(box, filled=hasattr(artist, "get_fill") and artist.get_fill()):
                    return True
    return False


def mass_legend(ax, title=None):
    import matplotlib as mpl

    handles, labels = ax.get_legend_handles_labels()
    priority = {"Background": 0, "Data": 1, "Signal": 2}
    order = sorted(range(len(labels)), key=lambda i: priority.get(labels[i].split()[0], 3))
    legend = ax.legend([handles[i] for i in order], [labels[i] for i in order],
                       loc="upper right", title=title, frameon=True, framealpha=0.93,
                       borderpad=0.42, labelspacing=0.32, handlelength=1.9, ncol=1,
                       fontsize=0.8 * mpl.rcParams["legend.fontsize"] * ax.figure.get_figwidth() / (8.6 / 2.54),
                       borderaxespad=0.5)
    for _ in range(20):
        ax.figure.canvas.draw()
        if not mass_legend_overlap(ax, legend):
            return legend
        low, high = ax.get_ylim()
        if ax.get_yscale() == "log":
            ax.set_ylim(low, high * (high / low) ** 0.15)
        else:
            ax.set_ylim(low, high + (high - low) * 0.15)
    raise RuntimeError("Dijet legend could not be separated from the distributions")


def spectrum_figure(summary, edges, cut, units, bounds=(3.3, 3.7), *, residual_ylim=None):
    import matplotlib.pyplot as plt

    width = 17.6 / 2.54
    fig, (ax, residual_ax) = plt.subplots(2, 1, figsize=(width, width * 0.8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.13, top=0.96)
    bin_width = np.diff(edges)
    centers = (edges[1:] + edges[:-1]) / 2
    suffix = " (" + selection_label(cut) + ")"
    for component, color, label, linestyle in (("background", "#71b6c1", "Background", "--"),
                                               ("signal", "#c64732", "Signal", "-")):
        values = summary[component] / bin_width
        if not np.any(values > 0):
            continue
        ax.stairs(values, edges, baseline=0, fill=False,
                  color=color, ls=linestyle, lw=1.0, label=label + suffix, zorder=2)
    values = summary["data"]
    positive = values > 0
    ax.errorbar(centers[positive], (values / bin_width)[positive], xerr=bin_width[positive] / 2,
                fmt="o", ls="none", color="black", markersize=3.2, elinewidth=0.8,
                label="Data" + suffix, zorder=4)
    signal_region_lines(ax, bounds)
    ax.set(yscale="log", ylabel=mass_ylabel(units), xlim=(edges[0], edges[-1]))
    ax.tick_params(labelbottom=False)
    residual, valid = spectrum_residual(summary)
    residual_ax.errorbar(centers[valid], residual[valid], xerr=bin_width[valid] / 2,
                         fmt="o", ls="none", color="black", markersize=3.2,
                         elinewidth=0.8, zorder=4)
    residual_ax.axhline(0, color="black", lw=0.8, zorder=5)
    residual_ax.set_axisbelow(True)
    residual_ax.grid(False, which="both")
    signal_region_lines(residual_ax, bounds)
    residual_ax.set(xlabel="Dijet Mass [TeV]",
                    ylabel=r"$\frac{\mathrm{Data}-\mathrm{Background}}{\sigma_{\mathrm{Data}}}$")
    if residual_ylim is None:
        low, high = min([0.0, *residual[valid]]), max([0.0, *residual[valid]])
        padding = max(0.5, (high - low) * 0.12)
        residual_ylim = low - padding, high + padding
    residual_ax.set_ylim(*residual_ylim)
    for axis in (ax, residual_ax):
        spectrum_style(axis)
    return fig, ax


def spectrum_cuts_figure(summaries, edges, units, bounds=(3.3, 3.7)):
    import matplotlib.pyplot as plt

    width = 17.6 / 2.54
    fig, (ax, residual_ax) = plt.subplots(2, 1, figsize=(width, width * 0.8), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.06})
    fig.subplots_adjust(left=0.14, right=0.97, bottom=0.13, top=0.96)
    bin_width = np.diff(edges)
    centers = (edges[1:] + edges[:-1]) / 2
    colors = ("#627887", "#276a87", "#71b6c1", "#ed9a56", "#c64732")
    for cut, color in zip((None, *CUTS), colors):
        summary = summaries[cut]
        background = summary["background"] / bin_width
        ax.stairs(background, edges, baseline=0, fill=False,
                  color=color, ls="--", lw=1.0, zorder=2)
        values = summary["data"]
        positive = values > 0
        label = "Data (" + selection_label(cut) + ")"
        ax.errorbar(centers[positive], (values / bin_width)[positive], xerr=bin_width[positive] / 2,
                    fmt="o", ls="none", color=color, markersize=3.2, elinewidth=0.7,
                    label=label, zorder=3)
        residual, valid = spectrum_residual(summary)
        residual_ax.errorbar(centers[valid], residual[valid], xerr=bin_width[valid] / 2,
                             fmt="o", ls="none", color=color, markersize=3.0,
                             elinewidth=0.7, zorder=3)
    ax.plot([], [], color="black", ls="--", lw=1.0, label="Background")
    signal_region_lines(ax, bounds)
    ax.set(ylabel=mass_ylabel(units), yscale="log",
           xlim=(edges[0], edges[-1]))
    signal_region_lines(residual_ax, bounds)
    residual_ax.axhline(0, color="black", lw=0.6, zorder=1)
    residual_ax.set(xlabel="Dijet Mass [TeV]",
                    ylabel=r"$\frac{\mathrm{Data}-\mathrm{Background}}{\sigma_{\mathrm{Data}}}$",
                    ylim=residual_limits(summaries))
    for axis in (ax, residual_ax):
        spectrum_style(axis)
    return fig, ax
