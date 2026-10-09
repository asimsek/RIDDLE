from pathlib import Path

import numpy as np


CUTS = (0.10, 0.05, 0.01, 0.005)
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


def spectrum_histograms(record, weights, edges):
    mass, labels = np.asarray(record["mass"]), np.asarray(record["labels"])
    masks = np.asarray(record["cut_masks"], dtype=bool)
    if masks.shape != (len(CUTS), len(mass)) or weights.shape != mass.shape:
        raise ValueError("Misaligned full-mass selections or normalization weights")
    output = {}
    for cut, selected in [(None, np.ones(len(mass), bool)), *zip(CUTS, masks)]:
        row = {}
        for name, population in (("background", labels == 0), ("signal", labels == 1),
                                 ("data", labels < 0)):
            take = selected & population
            row[name] = np.histogram(mass[take], edges, weights=weights[take])[0]
            row[name + "_variance"] = np.histogram(mass[take], edges, weights=weights[take] ** 2)[0]
        output[cut] = row
    return output


def mass_ylabel(units):
    return (r"$d\sigma/dm_{jj}$ [pb/TeV]" if units == "pb"
            else r"$dN/dm_{jj}$ [events/TeV]")
