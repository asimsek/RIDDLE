"""Smooth mass-dependent residual mixture fraction.

The gate f(m) is learned only from soft residual responsibilities in latent
feature space.  It is a training/mixture-model component and is deliberately
excluded from the final RIDDLE anomaly score p_S(z|m)/q_B(z|m).
"""

from __future__ import annotations

import math
import numpy as np


SCHEMA = 1
KIND = "natural_cubic_logistic_spline"


def _logit(value):
    value = np.asarray(value, dtype=np.float64)
    return np.log(value) - np.log1p(-value)


def _expit(value):
    value = np.asarray(value, dtype=np.float64)
    # Use a stable sigmoid without adding a SciPy inference dependency.
    out = np.empty_like(value)
    positive = value >= 0
    out[positive] = 1.0 / (1.0 + np.exp(-value[positive]))
    e = np.exp(value[~positive])
    out[~positive] = e / (1.0 + e)
    return out


def _config(settings):
    cfg = dict(settings.get("mass_fraction", {}))
    required = {
        "enabled", "control_points", "smoothness", "variation", "damping",
        "min_fraction", "max_fraction", "max_iterations",
    }
    if set(cfg) != required:
        raise ValueError("Invalid mass-fraction configuration")
    return cfg


def is_enabled(settings, fixed_fraction=None):
    """Return whether the learned f(m) gate is active for this fit."""
    return bool(_config(settings)["enabled"] and fixed_fraction is None)


def _contexts(context):
    x = np.asarray(context, dtype=np.float64).reshape(-1)
    if not len(x) or not np.isfinite(x).all():
        raise ValueError("Mass-fraction contexts must be finite and nonempty")
    # Validate normalized SR mass context before spline evaluation.
    if np.any(x < -1.00001) or np.any(x > 1.00001):
        raise ValueError("Mass-fraction context lies outside the signal region")
    return np.clip(x, -1.0, 1.0)


def _basis(context, control_points):
    """Natural-cubic interpolation basis evaluated at normalized mass context."""
    from scipy.interpolate import CubicSpline

    x = _contexts(context)
    nodes = np.linspace(-1.0, 1.0, int(control_points), dtype=np.float64)
    eye = np.eye(len(nodes), dtype=np.float64)
    basis = np.asarray(CubicSpline(nodes, eye, axis=0, bc_type="natural")(x), dtype=np.float64)
    if basis.shape != (len(x), len(nodes)) or not np.isfinite(basis).all():
        raise FloatingPointError("Invalid mass-fraction spline basis")
    # Partition of unity preserves common logit shifts across all masses.


    if not np.allclose(basis.sum(axis=1), 1.0, rtol=1e-8, atol=1e-10):
        raise FloatingPointError("Mass-fraction spline basis lost partition of unity")
    return nodes, basis


def initial_state(initial_fraction, settings):
    cfg = _config(settings)
    if not cfg["enabled"]:
        return None
    f = float(initial_fraction)
    if not math.isfinite(f) or not 0 < f < 1:
        raise ValueError("Initial mass-fraction value must lie in (0,1)")
    f = float(np.clip(f, cfg["min_fraction"], cfg["max_fraction"]))
    nodes = np.linspace(-1.0, 1.0, cfg["control_points"], dtype=np.float64)
    controls = np.full(len(nodes), float(_logit(f)), dtype=np.float64)
    return {
        "schema": SCHEMA,
        "kind": KIND,
        "control_contexts": nodes.tolist(),
        "control_logits": controls.tolist(),
        "updates": 0,
        "mean_fraction": f,
        "minimum_fraction": f,
        "maximum_fraction": f,
        "roughness": 0.0,
        "source": "flat_initialization",
    }


def _validate_state(state, settings):
    cfg = _config(settings)
    if state is None:
        raise ValueError("Missing mass-fraction state")
    if state.get("schema") != SCHEMA or state.get("kind") != KIND:
        raise ValueError("Unsupported mass-fraction state")
    nodes = np.asarray(state.get("control_contexts"), dtype=np.float64)
    controls = np.asarray(state.get("control_logits"), dtype=np.float64)
    expected = np.linspace(-1.0, 1.0, cfg["control_points"], dtype=np.float64)
    if (nodes.shape != expected.shape or controls.shape != expected.shape
            or not np.allclose(nodes, expected, rtol=0, atol=1e-12)
            or not np.isfinite(controls).all()):
        raise ValueError("Mass-fraction state disagrees with configured spline")
    return controls


def logits(state, context, settings):
    """Evaluate the smooth gate logit at normalized mjj context values."""
    controls = _validate_state(state, settings)
    _, basis = _basis(context, len(controls))
    values = basis @ controls
    cfg = _config(settings)
    lower, upper = float(_logit(cfg["min_fraction"])), float(_logit(cfg["max_fraction"]))
    values = np.clip(values, lower, upper)
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite mass-fraction logits")
    return values


def probabilities(state, context, settings):
    values = _expit(logits(state, context, settings))
    cfg = _config(settings)
    values = np.clip(values, cfg["min_fraction"], cfg["max_fraction"])
    if not np.isfinite(values).all():
        raise FloatingPointError("Nonfinite mass-fraction probabilities")
    return values


def _mean_shift(controls, basis, target, cfg):
    """Shift all spline logits so mean f(m) matches the EM responsibility mean."""
    from scipy.optimize import brentq

    lower_p, upper_p = cfg["min_fraction"], cfg["max_fraction"]
    target = float(np.clip(target, lower_p, upper_p))
    low, high = float(_logit(lower_p)), float(_logit(upper_p))

    def mean_at(delta):
        values = np.clip(basis @ (controls + delta), low, high)
        return float(_expit(values).mean())

    # Use a wide deterministic bracket so clipping reaches the configured limits.
    lo, hi = -40.0, 40.0
    flo, fhi = mean_at(lo) - target, mean_at(hi) - target
    if flo > 1e-12 or fhi < -1e-12:
        raise FloatingPointError("Unable to normalize smooth mass-fraction mean")
    if abs(flo) <= 1e-12:
        delta = lo
    elif abs(fhi) <= 1e-12:
        delta = hi
    else:
        delta = float(brentq(lambda d: mean_at(d) - target, lo, hi, xtol=1e-10, rtol=1e-10))
    return controls + delta


def fit(context, responsibilities, state, settings, *, source="residual_responsibilities"):
    """Penalized logistic-spline M-step for a smooth f(m).

    The soft labels are residual responsibilities computed from z-space density
    evidence.  Event counts in the mjj spectrum are never fit as a bump model.
    A five-control-point natural cubic spline (default) supplies the only mass
    dependence. Weak first- and second-difference penalties plus a damped EM
    update suppress noisy trends/wiggles while damping the global fraction mean
    toward the current responsibility mean.
    """
    from scipy.optimize import minimize

    cfg = _config(settings)
    old = _validate_state(state, settings)
    x = _contexts(context)
    r = np.asarray(responsibilities, dtype=np.float64).reshape(-1)
    if r.shape != x.shape or not np.isfinite(r).all() or np.any(r < 0) or np.any(r > 1):
        raise ValueError("Mass-fraction responsibilities must be aligned probabilities")
    nodes, basis = _basis(x, cfg["control_points"])
    target = float(np.clip(r.mean(), cfg["min_fraction"], cfg["max_fraction"]))

    d1 = np.zeros((max(0, len(old)-1), len(old)), dtype=np.float64)
    for i in range(len(d1)):
        d1[i, i:i+2] = (-1.0, 1.0)
    d2 = np.zeros((max(0, len(old)-2), len(old)), dtype=np.float64)
    for i in range(len(d2)):
        d2[i, i:i+3] = (1.0, -2.0, 1.0)
    smoothness = float(cfg["smoothness"])
    variation = float(cfg["variation"])

    def objective(c):
        eta = basis @ c
        # Compute BCE directly from logits for soft responsibilities.
        bce = float(np.mean(np.logaddexp(0.0, eta) - r * eta))
        slope = d1 @ c
        curvature = d2 @ c
        penalty = (variation * float(np.mean(slope * slope)) if len(slope) else 0.0)
        penalty += smoothness * float(np.mean(curvature * curvature)) if len(curvature) else 0.0
        return bce + penalty

    def gradient(c):
        eta = basis @ c
        p = _expit(eta)
        grad = basis.T @ (p-r) / len(r)
        if len(d1):
            slope = d1 @ c
            grad = grad + (2.0 * variation / len(slope)) * (d1.T @ slope)
        if len(d2):
            curvature = d2 @ c
            grad = grad + (2.0 * smoothness / len(curvature)) * (d2.T @ curvature)
        return grad

    result = minimize(objective, old, jac=gradient, method="L-BFGS-B",
                      options={"maxiter": int(cfg["max_iterations"]), "ftol": 1e-12, "gtol": 1e-8})
    if not result.success or not np.isfinite(result.x).all():
        raise FloatingPointError(f"Smooth mass-fraction fit failed: {result.message}")

    fitted = _mean_shift(np.asarray(result.x, dtype=np.float64), basis, target, cfg)
    damping = float(cfg["damping"])
    low, high = float(_logit(cfg["min_fraction"])), float(_logit(cfg["max_fraction"]))
    old_mean = float(_expit(np.clip(basis @ old, low, high)).mean())
    damped_target = (1.0-damping)*old_mean + damping*target
    controls = (1.0-damping)*old + damping*fitted
    controls = _mean_shift(controls, basis, damped_target, cfg)

    values = np.clip(basis @ controls, float(_logit(cfg["min_fraction"])),
                     float(_logit(cfg["max_fraction"])))
    probs = _expit(values)
    roughness = float(np.sqrt(np.mean(np.diff(controls, n=2)**2))) if len(controls) > 2 else 0.0
    if not np.isfinite([probs.mean(), probs.min(), probs.max(), roughness]).all():
        raise FloatingPointError("Invalid smooth mass-fraction state")
    return {
        "schema": SCHEMA,
        "kind": KIND,
        "control_contexts": nodes.tolist(),
        "control_logits": controls.tolist(),
        "updates": int(state.get("updates", 0)) + 1,
        "mean_fraction": float(probs.mean()),
        "minimum_fraction": float(probs.min()),
        "maximum_fraction": float(probs.max()),
        "roughness": roughness,
        "target_mean_responsibility": target,
        "objective": float(objective(controls)),
        "source": str(source),
    }


def state_summary(state, context, settings):
    p = probabilities(state, context, settings)
    return {
        "mean": float(p.mean()),
        "minimum": float(p.min()),
        "maximum": float(p.max()),
        "roughness": float(state.get("roughness", 0.0)),
        "updates": int(state.get("updates", 0)),
    }
