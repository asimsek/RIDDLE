"""Independent sideband closure diagnostic; never a retraining criterion."""
import numpy as np
from scipy.stats import binomtest

from .integrity import require_finite


def assess_closure(logits, mass, *, targets=(.10, .05, .01, .004), bins_per_side=4, family_alpha=.01):
    logits, mass = np.asarray(logits), np.asarray(mass)
    require_finite(logits, "Closure scores"); require_finite(mass, "Closure masses")
    if logits.shape != mass.shape or logits.ndim != 1 or not len(logits) or ((mass > 3.3) & (mass < 3.7)).any():
        raise ValueError("Closure requires aligned independent sideband events")
    bins = []
    for side in (mass <= 3.3, mass >= 3.7):
        indices = np.flatnonzero(side)
        if len(indices):
            bins.extend(np.array_split(indices[np.argsort(mass[indices], kind="stable")], bins_per_side))
    bins = [i for i in bins if len(i)]
    tests = len(targets)*(len(bins)+1)
    cutoff = family_alpha/tests
    rows, eligible = [], 0
    for target in targets:
        selected = logits > np.log((1-target)/target)
        for i, ix in enumerate([np.arange(len(mass)), *bins]):
            k, n = int(selected[ix].sum()), len(ix)
            # Sparse tails cannot establish closure. Do not call them passed.
            sufficient = n*target >= 10 and n*(1-target) >= 10
            eligible += int(sufficient)
            p = float(binomtest(k, n, target).pvalue)
            rows.append(dict(target=target, bin="inclusive" if i == 0 else i-1,
                             mass_min=float(mass[ix].min()), mass_max=float(mass[ix].max()),
                             retained=k, events=n, efficiency=k/n, p_value=p,
                             sufficient_counts=sufficient, failed=p < cutoff))
    status = "failed" if any(r["failed"] for r in rows) else "passed" if eligible == len(rows) else "inconclusive"
    return dict(status=status, truth_labels_used=False, events=len(mass), tests=tests,
                family_alpha=family_alpha, bonferroni_cutoff=cutoff, working_points=rows,
                interpretation="sideband calibration diagnostic; not SR closure or search certification",
                sparse_tail_policy="Exact binomial tests can reject gross nonclosure at low counts; adequate expected counts are required to call closure passed",
                action="report only; no score clipping, evidence retry, or calibration retuning")
