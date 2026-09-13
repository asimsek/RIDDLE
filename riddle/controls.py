import numpy as np

from .features import FEATURES, _remap_columns

VARIANTS = ("default", "shifted", "deltaR")


def columns(variant):
    if variant not in VARIANTS:
        raise ValueError("Unknown LHCO dataset variant")
    return ["mjj", *FEATURES, *(["deltaR"] if variant == "deltaR" else []), "label"]


def description(variant):
    columns(variant)
    return {
        "default": {},
        "shifted": {
            "coefficient": 0.1,
            "shifted_features": ["m1", "delta_m"],
            "mass_and_membership_unchanged": True,
        },
        "deltaR": {
            "added_feature": "deltaR",
            "definition": "jet angular distance in pseudorapidity and wrapped azimuth; vector.deltaR",
        },
    }[variant]


def delta_r(table, mapping=None):
    import vector

    values = _remap_columns(table, mapping or {})
    jets = []
    for jet in (1, 2):
        coordinates = {}
        for axis in ("px", "py", "pz"):
            name = f"{axis}j{jet}"
            if name not in values:
                raise ValueError("DeltaR preparation requires both jets' px, py and pz columns")
            coordinates[axis] = np.asarray(values[name])[:, None]
        jets.append(vector.array(coordinates))
    result = jets[0].deltaR(jets[1]).flatten()
    if not np.isfinite(result).all() or (result < 0).any():
        raise ValueError("Invalid jet angular distance; no events were dropped")
    return result.astype(np.float64)


def attach_delta_r(roles, arrays):
    for role in roles.values():
        distances = np.empty(len(role["source_entry"]), dtype=np.float64)
        for source_index in np.unique(role["source_index"]):
            mask = role["source_index"] == source_index
            source = arrays[int(source_index)]
            distances[mask] = source["deltaR"][role["source_entry"][mask]]
        role["deltaR"] = distances


def transform(values, arrays, variant):
    columns(variant)
    if variant == "shifted":
        values[:, 1] += 0.1 * values[:, 0]
        values[:, 2] += 0.1 * values[:, 0]
    elif variant == "deltaR":
        values = np.column_stack((values[:, :-1], arrays["deltaR"], values[:, -1]))
    return values
