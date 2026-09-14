from __future__ import annotations
from typing import Any, Mapping
import numpy as np
from .data_spec import DatasetSpec

FEATURES = ("m1", "delta_m", "tau21_j1", "tau21_j2")
EVENT_COLUMNS = (
    "mjj",
    *FEATURES,
    "label",
    "event_weight",
    "source_index",
    "source_entry",
    "partition",
    "is_signal_region",
)
RAW_COLUMNS = (
    "pxj1",
    "pyj1",
    "pzj1",
    "mj1",
    "tau1j1",
    "tau2j1",
    "pxj2",
    "pyj2",
    "pzj2",
    "mj2",
    "tau1j2",
    "tau2j2",
)


def _column(columns: Mapping[str, Any], name: str, dtype=np.float64) -> np.ndarray:
    if name not in columns:
        raise ValueError(f"LHCO source is missing {name!r}")
    result = np.asarray(columns[name], dtype=dtype)
    if result.ndim != 1:
        raise ValueError(f"LHCO source column {name!r} must be one-dimensional")
    return result


def compute_lhco_features(
    columns: Mapping[str, Any],
    *,
    source_index: int = 0,
    require_label: bool = True,
    signal_minimum_gev: float = 3300.0,
    signal_maximum_gev: float = 3700.0,
) -> dict[str, np.ndarray]:
    raw = {name: _column(columns, name) for name in RAW_COLUMNS}
    lengths = {len(value) for value in raw.values()}
    if len(lengths) != 1:
        raise ValueError("LHCO source columns have inconsistent lengths")
    rows = lengths.pop()
    mass = np.column_stack((raw["mj1"], raw["mj2"]))
    tau21 = np.column_stack(
        (raw["tau2j1"] / (raw["tau1j1"] + 1e-05), raw["tau2j2"] / (raw["tau1j2"] + 1e-05))
    )
    lower = np.argmin(mass, axis=1)
    upper = np.argmax(mass, axis=1)
    row = np.arange(rows)
    energy1 = np.sqrt(raw["pxj1"] ** 2 + raw["pyj1"] ** 2 + raw["pzj1"] ** 2 + raw["mj1"] ** 2)
    energy2 = np.sqrt(raw["pxj2"] ** 2 + raw["pyj2"] ** 2 + raw["pzj2"] ** 2 + raw["mj2"] ** 2)
    mass_squared = (energy1 + energy2) ** 2 - (
        (raw["pxj1"] + raw["pxj2"]) ** 2 + (raw["pyj1"] + raw["pyj2"]) ** 2 + (raw["pzj1"] + raw["pzj2"]) ** 2
    )
    if np.min(mass_squared, initial=0.0) < -1e-06:
        raise ValueError("LHCO dijet invariant-mass calculation is negative")
    if require_label:
        labels = _column(columns, "label", np.int8)
        if len(labels) != rows or not np.all(np.isin(labels, (0, 1))):
            raise ValueError("LHCO labels must be exactly zero or one")
    else:
        labels = np.zeros(rows, dtype=np.int8)
    lower_mass = mass[row, lower]
    if "event_weight" in columns:
        event_weight = _column(columns, "event_weight", np.float64)
        if len(event_weight) != rows:
            raise ValueError("event_weight is not aligned with the event rows")
    else:
        event_weight = np.ones(rows, dtype=np.float64)
    if np.any(~np.isfinite(event_weight)) or np.any(event_weight < 0.0):
        raise ValueError("event_weight must be finite and nonnegative")
    result = {
        "mjj": np.sqrt(np.maximum(mass_squared, 0.0)).astype(np.float64),
        "m1": lower_mass.astype(np.float64),
        "delta_m": (mass[row, upper] - lower_mass).astype(np.float64),
        "tau21_j1": tau21[row, lower].astype(np.float64),
        "tau21_j2": tau21[row, upper].astype(np.float64),
        "label": labels.astype(np.int8),
        "event_weight": event_weight,
        "source_index": np.full(rows, source_index, dtype=np.uint16),
        "source_entry": np.arange(rows, dtype=np.uint64),
    }
    for name in ("mjj", *FEATURES):
        if np.any(~np.isfinite(result[name])):
            raise ValueError(f"Computed LHCO column {name!r} contains non-finite values")
    result["is_signal_region"] = (result["mjj"] > signal_minimum_gev) & (result["mjj"] < signal_maximum_gev)
    return result


def partition_labels(rows: int) -> np.ndarray:
    if rows < 0:
        raise ValueError("Partition size cannot be negative")
    result = np.empty(rows, dtype="U16")
    train_stop = rows // 2
    validation_stop = 2 * rows // 3
    result[:train_stop] = "training"
    result[train_stop:validation_stop] = "validation"
    result[validation_stop:] = "final_test"
    return result


def _take(arrays: Mapping[str, np.ndarray], indices: np.ndarray, partition: np.ndarray):
    if len(indices) != len(partition):
        raise ValueError("Selected indices and partitions are not aligned")
    result = {name: np.asarray(arrays[name])[indices] for name in EVENT_COLUMNS if name != "partition"}
    result["partition"] = np.asarray(partition).astype(str)
    return {name: result[name] for name in EVENT_COLUMNS}


def build_dataset_roles(
    arrays: Mapping[str, np.ndarray],
    spec: DatasetSpec,
    *,
    sic_background_arrays: Mapping[str, np.ndarray] | None = None,
    enforce_expected_counts: bool = True,
    independent_partition: bool = False,
) -> dict[str, dict[str, np.ndarray]]:
    labels = np.asarray(arrays["label"], dtype=np.int8)
    background = np.flatnonzero(labels == 0)
    signal = np.flatnonzero(labels == 1)
    if independent_partition:
        rng = np.random.RandomState(spec.preparation_seed)
        background, signal = rng.permutation(background), rng.permutation(signal)
    if enforce_expected_counts and (len(background), len(signal)) != (spec.background_rows, spec.signal_rows):
        raise RuntimeError(
            f"Expected dataset counts are {(spec.background_rows, spec.signal_rows)}, observed {(len(background), len(signal))}"
        )
    if not len(background):
        raise RuntimeError("Dataset does not contain any background events")
    if enforce_expected_counts and len(signal) < spec.injected_signal_rows:
        raise RuntimeError("Dataset has too few signal rows for the configured injection")
    background_partition = partition_labels(len(background))
    signal_region = np.asarray(arrays["is_signal_region"], dtype=bool)
    final_background = background[background_partition == "final_test"]
    sculpting_rows = (
        spec.sculpting_test_rows
        if enforce_expected_counts
        else min(spec.sculpting_test_rows, len(final_background))
    )
    if len(final_background) < sculpting_rows:
        raise RuntimeError("The final background partition is too small for the sculpting sample")
    roles = {
        "background_only": _take(arrays, background, background_partition),
        "background_sculpting_test": _take(
            arrays, final_background[:sculpting_rows], np.full(sculpting_rows, "sculpting_test", dtype="U16")
        ),
    }
    if sic_background_arrays is not None:
        extra_labels = np.asarray(sic_background_arrays["label"], dtype=np.int8)
        extra_signal_region = np.asarray(sic_background_arrays["is_signal_region"], dtype=bool)
        if np.any(extra_labels != 0):
            raise RuntimeError("The SIC background source must contain background events only")
        if enforce_expected_counts and len(extra_labels) != spec.sic_background_rows:
            raise RuntimeError(
                f"Expected {spec.sic_background_rows} dedicated SIC background rows, observed {len(extra_labels)}"
            )
        rng = np.random.RandomState(spec.preparation_seed)
        if spec.preparation_seed != 1:
            dummy_signal = np.arange(spec.signal_rows, dtype=np.uint32)
            rng.shuffle(dummy_signal)
        extra_order = np.arange(len(extra_labels), dtype=np.uint32)
        rng.shuffle(extra_order)
        extra_order = extra_order[extra_signal_region[extra_order]]
        if len(extra_order) <= spec.sic_background_validation_stop:
            raise RuntimeError("The dedicated SIC background source is too small after the signal-region cut")
        extra_test = extra_order[spec.sic_background_validation_stop :]
        primary_test_sr = final_background[signal_region[final_background]]
        primary_role = _take(
            arrays, primary_test_sr, np.full(len(primary_test_sr), "sic_evaluation", dtype="U16")
        )
        extra_role = _take(
            sic_background_arrays, extra_test, np.full(len(extra_test), "sic_evaluation", dtype="U16")
        )
        roles["sic_evaluation_background"] = {
            name: np.concatenate((primary_role[name], extra_role[name])) for name in EVENT_COLUMNS
        }
    evaluation_signal_background_only = signal[signal_region[signal]]
    if len(evaluation_signal_background_only):
        roles["sic_evaluation_signal_background_only"] = _take(
            arrays,
            evaluation_signal_background_only,
            np.full(len(evaluation_signal_background_only), "sic_evaluation", dtype="U16"),
        )
    if len(signal) >= spec.injected_signal_rows and spec.injected_signal_rows > 0:
        injected = signal[: spec.injected_signal_rows]
        injected_partition = partition_labels(len(injected))
        injection_indices = np.concatenate((background, injected))
        injection_partition = np.concatenate((background_partition, injected_partition))
        roles["signal_injection"] = _take(arrays, injection_indices, injection_partition)
        injected_final_start = 2 * spec.injected_signal_rows // 3
        evaluation_signal_injection = np.concatenate(
            (injected[injected_final_start:], signal[spec.injected_signal_rows :])
        )
        evaluation_signal_injection = evaluation_signal_injection[signal_region[evaluation_signal_injection]]
        if len(evaluation_signal_injection):
            roles["sic_evaluation_signal_signal_injection"] = _take(
                arrays,
                evaluation_signal_injection,
                np.full(len(evaluation_signal_injection), "sic_evaluation", dtype="U16"),
            )
    return roles


def partition_row_order(partition: np.ndarray, *, preparation_seed: int = 1) -> np.ndarray:
    partition = np.asarray(partition).astype(str)
    if set(partition.tolist()) != {"training", "validation", "final_test"}:
        raise ValueError("Scenario does not contain all three data partitions")
    rng = np.random.RandomState(preparation_seed)
    if preparation_seed != 1:
        dummy_signal = np.arange(100000, dtype=np.uint32)
        rng.shuffle(dummy_signal)
    dummy_extra_qcd = np.arange(612858, dtype=np.uint32)
    rng.shuffle(dummy_extra_qcd)
    ordered: list[np.ndarray] = []
    for name in ("training", "validation", "final_test"):
        indices = np.flatnonzero(partition == name)
        rng.shuffle(indices)
        ordered.append(indices)
    result = np.concatenate(ordered)
    if len(np.unique(result)) != len(partition):
        raise RuntimeError("Partition order is not a permutation")
    return result


def _remap_columns(columns: Mapping[str, Any], mapping: Mapping[str, str]) -> dict[str, Any]:
    names = list(columns.columns) if hasattr(columns, "columns") else list(columns)
    result = {str(name): columns[name] for name in names}
    for logical_name, input_name in mapping.items():
        if input_name not in result:
            raise ValueError(f"Configured input column {input_name!r} for {logical_name!r} is missing")
        result[str(logical_name)] = result[input_name]
    return result


def canonicalize_event_columns(
    columns: Mapping[str, Any],
    *,
    source_index: int,
    default_label: int = 0,
    default_weight: float = 1.0,
    column_mapping: Mapping[str, str] | None = None,
    signal_minimum_gev: float = 3300.0,
    signal_maximum_gev: float = 3700.0,
) -> dict[str, np.ndarray]:
    if source_index < 0 or source_index > np.iinfo(np.uint16).max:
        raise ValueError("source_index must fit in an unsigned 16-bit integer")
    if default_label not in (0, 1):
        raise ValueError("default_label must be zero or one")
    if not np.isfinite(default_weight) or default_weight < 0.0:
        raise ValueError("default_weight must be finite and nonnegative")
    remapped = _remap_columns(columns, column_mapping or {})
    canonical_names = ("mjj", *FEATURES)
    if all((name in remapped for name in canonical_names)):
        converted = {name: _column(remapped, name) for name in canonical_names}
        lengths = {len(value) for value in converted.values()}
        if len(lengths) != 1:
            raise ValueError("Canonical feature columns have inconsistent lengths")
        rows = lengths.pop()
        if rows == 0:
            raise ValueError("Input tables cannot be empty")
        converted["source_index"] = np.full(rows, source_index, dtype=np.uint16)
        converted["source_entry"] = np.arange(rows, dtype=np.uint64)
    elif all((name in remapped for name in RAW_COLUMNS)):
        converted = compute_lhco_features(
            remapped,
            source_index=source_index,
            require_label=False,
            signal_minimum_gev=signal_minimum_gev,
            signal_maximum_gev=signal_maximum_gev,
        )
        rows = len(converted["mjj"])
    else:
        missing_canonical = [name for name in canonical_names if name not in remapped]
        missing_raw = [name for name in RAW_COLUMNS if name not in remapped]
        raise ValueError(
            f"Input must provide either canonical columns {list(canonical_names)!r} (missing {missing_canonical!r}) or all raw jet columns (missing {missing_raw!r})"
        )
    if "label" in remapped:
        labels = _column(remapped, "label", np.int8)
        if len(labels) != rows or not np.all(np.isin(labels, (0, 1))):
            raise ValueError("label must be aligned and binary")
    else:
        labels = np.full(rows, default_label, dtype=np.int8)
    if "event_weight" in remapped:
        weights = _column(remapped, "event_weight", np.float64)
        if len(weights) != rows:
            raise ValueError("event_weight is not aligned with the event rows")
    else:
        weights = np.full(rows, default_weight, dtype=np.float64)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0.0):
        raise ValueError("event_weight must be finite and nonnegative")
    converted["label"] = labels.astype(np.int8)
    converted["event_weight"] = weights.astype(np.float64)
    for name in canonical_names:
        value = np.asarray(converted[name], dtype=np.float64)
        if np.any(~np.isfinite(value)):
            raise ValueError(f"Input column {name!r} contains non-finite values")
        converted[name] = value
    converted["is_signal_region"] = (converted["mjj"] > signal_minimum_gev) & (
        converted["mjj"] < signal_maximum_gev
    )
    return {name: np.asarray(converted[name]) for name in EVENT_COLUMNS if name != "partition"}
