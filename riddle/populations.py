"""Shared, source-identifiable benchmark populations; no model-dependent test set."""
from copy import deepcopy
from decimal import Decimal
import json
from pathlib import Path

import numpy as np
import yaml

from .storage import digest

EVALUATION_FILES = ("innerdata_test.npy", "innerdata_extrabkg_test.npy", "innerdata_extrasig.npy")
EVALUATION_KEY = "shared_evaluation_event_ids"
SPLIT_NAMES = ("training", "validation", "evaluation")


def percentage_counts(rows, percentages, names=SPLIT_NAMES):
    if type(rows) is not int or rows < 0:
        raise ValueError("Population size must be a nonnegative integer")
    if not isinstance(percentages, dict) or set(percentages) != set(names):
        raise ValueError(f"Expected percentage keys: {names}")
    values = []
    for name in names:
        value = percentages[name]
        if type(value) not in (int, float) or not np.isfinite(value) or not 0 < value < 100:
            raise ValueError(f"Invalid population percentage: {name}")
        values.append(Decimal(str(value)))
    if sum(values) != Decimal(100):
        raise ValueError("Population percentages must sum to exactly 100")
    # Floor boundaries; later slices receive the remainder.
    stops = [int(Decimal(rows) * sum(values[:i]) / 100) for i in range(1, len(values))] + [rows]
    return tuple(b - a for a, b in zip([0] + stops[:-1], stops))


def validate_config(value):
    value = deepcopy(value)
    if not isinstance(value, dict) or set(value) != {"schema", "injected_signal_events", "preparation_seed", "splits", "validation_roles"}:
        raise ValueError("Invalid population configuration keys")
    if type(value["schema"]) is not int or value["schema"] != 1:
        raise ValueError("Unsupported population configuration schema")
    for key, low, high in (("injected_signal_events", 3, 100000), ("preparation_seed", 0, 2**32)):
        if type(value[key]) is not int or not low <= value[key] < high:
            raise ValueError(f"Invalid population {key}")
    splits = value["splits"]
    if not isinstance(splits, dict) or set(splits) != {"background", "injected_signal", "additional_background", "supervised_signal"}:
        raise ValueError("Invalid population split pools")
    for percentages in splits.values():
        percentage_counts(100, percentages)
    roles = value["validation_roles"]
    if not isinstance(roles, dict) or set(roles) != {"oracle_background", "supervised_signal"}:
        raise ValueError("Invalid reserved validation pools")
    percentage_counts(100, roles["oracle_background"], ("fit", "closure", "selector"))
    percentage_counts(100, roles["supervised_signal"], ("assessment", "selector"))
    return value


def load_config(path=None):
    if path is None:
        from .settings import default_config_path
        path = default_config_path("populations.yaml")
    return validate_config(yaml.safe_load(Path(path).read_text()))


def role_indices(rows, percentages, names, seed):
    counts = percentage_counts(int(rows), percentages, names)
    if min(counts) < 1:
        raise ValueError("Too few validation events for independent selector reserves")
    order = np.random.RandomState(seed).permutation(rows)
    return dict(zip(names, np.split(order, np.cumsum(counts)[:-1])))


def contract(config, identities):
    evaluation = np.concatenate([identities[name] for name in EVALUATION_FILES])
    return {"configuration": validate_config(config), "evaluation_files": list(EVALUATION_FILES),
            "evaluation_events": len(evaluation), "evaluation_event_ids_sha256": digest(evaluation)}


def verify_evaluation(manifest, identities):
    saved = manifest["shared_population"]
    expected = contract(saved["configuration"], identities)
    if saved != expected or not np.array_equal(identities[EVALUATION_KEY],
                                               np.concatenate([identities[n] for n in EVALUATION_FILES])):
        raise ValueError("Shared evaluation population changed")
    return saved


def prepared_config(directory):
    manifest = json.loads((Path(directory) / "inputs.json").read_text())
    shared = manifest.get("shared_population")
    return None if shared is None else validate_config(shared["configuration"])
