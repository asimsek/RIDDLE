import json
from pathlib import Path

import numpy as np

from .source import digest

FEATURES = ("m1", "delta_m", "tau21_j1", "tau21_j2")
COLUMNS = ("mjj", *FEATURES, "label")
BACKGROUND_PROTOCOL = "sideband_development_50_50_v1"
FILES = tuple(
    f"{region}data_{part}.npy"
    for region in ("inner", "outer")
    for part in ("train", "val", "test")
) + ("innerdata_extrabkg_test.npy", "innerdata_extrasig.npy")


def region(mass):
    return (mass > 3.3) & (mass < 3.7)


def input_features(variant):
    if variant not in ("default", "shifted", "deltaR"):
        raise ValueError("R-ANODE supports default, shifted and deltaR LHCO datasets")
    return FEATURES + (("deltaR",) if variant == "deltaR" else ())


def scientific_version(variant):
    input_features(variant)
    return "pinned_upstream_sidebands_deltaR_v1" if variant == "deltaR" else "pinned_upstream_sidebands_v1"


def validate_schema(meta):
    features = input_features(meta.get("variant", "default"))
    if (
        tuple(meta.get("columns", ())) != ("mjj", *features, "label")
        or meta.get("mass_unit") != "TeV"
        or meta.get("signal_region") != [3.3, 3.7]
        or meta.get("signal_region_boundary") != "strict"
    ):
        raise ValueError(
            "R-ANODE requires the ordered physical LHCO columns (including deltaR only for that variant), "
            "masses in TeV and strict 3.3 < mjj < 3.7"
        )
    return features


def validate(root):
    root = Path(root)
    meta = json.loads((root / "inputs.json").read_text())
    features = validate_schema(meta)
    arrays = {}
    for name in FILES:
        path = root / name
        if path.is_symlink() or digest(path) != meta.get("files", {}).get(name):
            raise ValueError("R-ANODE input hash mismatch: " + name)
        rows = np.load(path, allow_pickle=False)
        if (
            rows.ndim != 2
            or rows.shape[1] != len(features) + 2
            or rows.dtype != np.float64
            or not np.isfinite(rows).all()
            or not np.isin(rows[:, -1], [0, 1]).all()
        ):
            raise ValueError("Invalid physical R-ANODE input: " + name)
        if not np.all(region(rows[:, 0]) == name.startswith("inner")):
            raise ValueError("SR membership disagrees with input partition: " + name)
        arrays[name] = rows
    if np.any(arrays["innerdata_extrabkg_test.npy"][:, -1] != 0) or np.any(
        arrays["innerdata_extrasig.npy"][:, -1] != 1
    ):
        raise ValueError(
            "R-ANODE dedicated evaluation sources have inconsistent truth labels"
        )
    if meta.get("schema") == 2:
        identity_path = root / "event_ids.npz"
        if digest(identity_path) != meta.get("event_ids_sha256"):
            raise ValueError("R-ANODE event identity hash mismatch")
        with np.load(identity_path, allow_pickle=False) as archive:
            if set(archive.files) != set(FILES):
                raise ValueError("Incomplete R-ANODE event identity archive")
            identities = []
            for name in FILES:
                ids = archive[name]
                if ids.shape != (len(arrays[name]), 2) or ids.dtype != np.uint64:
                    raise ValueError("Misaligned R-ANODE event identities")
                identities.append(ids)
        ids = np.ascontiguousarray(np.concatenate(identities)).view("V16").ravel()
        if len(np.unique(ids)) != len(ids):
            raise ValueError("Repeated event identities across R-ANODE partitions")
    elif meta.get("schema") != 1:
        raise ValueError("Unknown R-ANODE prepared input schema")
    if meta.get("scenario") not in ("signal_injection", "background_only"):
        raise ValueError("Unknown R-ANODE input scenario")
    if meta["scenario"] == "background_only" and any(
        np.any(arrays[f"{r}data_{p}.npy"][:, -1] != 0)
        for r in ("inner", "outer")
        for p in ("train", "val", "test")
    ):
        raise ValueError("Signal found in background-only physical data")
    # Reject ambiguous memberships instead of assigning duplicate rows.
    train = row_keys(
        np.concatenate([arrays[f"{r}data_train.npy"] for r in ("inner", "outer")])
    )
    val = row_keys(
        np.concatenate([arrays[f"{r}data_val.npy"] for r in ("inner", "outer")])
    )
    test = row_keys(
        np.concatenate([arrays[n] for n in FILES if "test" in n or "extrasig" in n])
    )
    if set(train) & set(val) or (set(train) | set(val)) & set(test):
        raise ValueError(
            "Duplicate physical rows across R-ANODE development/test partitions"
        )
    return meta, arrays


def row_keys(rows):
    return [row.tobytes() for row in np.asarray(rows, dtype="<f8")]


def evaluation(arrays):
    return {
        "validation": np.concatenate(
            [arrays[f"{r}data_val.npy"] for r in ("inner", "outer")]
        ),
        "test": np.concatenate(
            [arrays[f"{r}data_test.npy"] for r in ("inner", "outer")]
        ),
        "signal_region": np.concatenate(
            [
                arrays[n]
                for n in (
                    "innerdata_test.npy",
                    "innerdata_extrabkg_test.npy",
                    "innerdata_extrasig.npy",
                )
            ]
        ),
    }


def latent_inputs(root, inputs, originals, *, expected_digest=None):
    """Read an explicitly identified pilot representation, never a fake physical dataset."""
    from riddle.data import diagnostic_profile

    root = Path(root).resolve()
    path = root / "manifest.json"
    if expected_digest is not None and digest(path) != expected_digest:
        raise ValueError("Latent-input manifest changed")
    meta = json.loads(path.read_text())
    if (diagnostic_profile(inputs) is None or meta.get("schema") != 1
            or meta.get("method") != "riddlev4" or meta.get("parent_inputs") != inputs
            or meta.get("representation") != "frozen_riddle_latents"):
        raise ValueError("Latent R-ANODE inputs require a matching verified CPU pilot")
    files = meta.get("files", {})
    if not set(FILES + ("mapping_masks.npz",)) <= set(files):
        raise ValueError("Incomplete latent-input receipt")
    for name, checksum in files.items():
        item = (root / name).resolve()
        if not item.is_relative_to(root) or (root / name).is_symlink() or digest(item) != checksum:
            raise ValueError("Latent-input artifact changed: " + name)
    arrays, masks = {}, {}
    with np.load(root / "mapping_masks.npz", allow_pickle=False) as saved:
        if set(saved.files) != set(FILES):
            raise ValueError("Incomplete latent event mapping")
        for name in FILES:
            mask = saved[name]
            rows = np.load(root / name, allow_pickle=False)
            original = originals[name]
            if (mask.dtype != bool or mask.shape != (len(original),)
                    or rows.dtype != np.float64 or rows.shape != (int(mask.sum()), original.shape[1])
                    or not len(rows) or not np.isfinite(rows).all()
                    or not np.array_equal(rows[:, [0, -1]], original[mask][:, [0, -1]])):
                raise ValueError("Latent inputs lost mass/label/event alignment: " + name)
            arrays[name], masks[name] = rows, mask
    return meta, arrays, masks


class MatchedPartitions:
    """Replace only input selection and train/validation membership, not preprocessing."""

    def __init__(self, arrays, stage):
        self.arrays, self.stage = arrays, stage
        self.membership = {}
        self.split_audit = None

    def resample_split(self, *args, **kwargs):
        inner = np.concatenate(
            [self.arrays[f"innerdata_{p}.npy"] for p in ("train", "val")]
        )
        outer = np.concatenate(
            [self.arrays[f"outerdata_{p}.npy"] for p in ("train", "val")]
        )
        s, b = (inner[:, -1] == 1).sum(), (inner[:, -1] == 0).sum()
        return inner, outer, s / len(inner), s / np.sqrt(b) if b else 0.0

    def register(self, original_transform, params):
        regions = ("outer",) if self.stage == "background" else ("inner",)
        self.membership = {}
        for part, is_train in (("train", True), ("val", False)):
            rows = np.concatenate([self.arrays[f"{r}data_{part}.npy"] for r in regions])
            values = (
                original_transform(rows, params) if self.stage == "background" else rows
            )
            for key in row_keys(values):
                previous = self.membership.setdefault(key, is_train)
                if previous != is_train:
                    raise ValueError(
                        "Ambiguous R-ANODE split after upstream preprocessing"
                    )

    def split(self, rows):
        try:
            train = np.array([self.membership[k] for k in row_keys(rows)], dtype=bool)
        except KeyError as error:
            raise ValueError(
                "Upstream split contains an unknown physical row"
            ) from error
        if not train.any() or train.all():
            raise ValueError(
                "Empty R-ANODE train/validation partition after upstream domain mask"
            )
        self.split_audit = {
            "training_rows": int(train.sum()),
            "validation_rows": int((~train).sum()),
        }
        return np.flatnonzero(train), np.flatnonzero(~train)


def export_scores(namespace, arrays, output, *, original_arrays=None, mapping_masks=None):
    records = evaluation(arrays)
    if (original_arrays is None) != (mapping_masks is None):
        raise ValueError("Original rows and mapping masks must be supplied together")
    original_records = evaluation(original_arrays) if original_arrays is not None else None
    original_masks = evaluation(mapping_masks) if mapping_masks is not None else None
    all_rows = np.concatenate([rows[region(rows[:, 0])] for rows in records.values()])
    from src.utils import logit_transform

    params = namespace["pre_parameters_CR"]
    _, accepted = logit_transform(all_rows[:, 1:-1], params["min"], params["max"])
    if not np.array_equal(namespace["x_test"], all_rows[accepted]):
        raise ValueError("R-ANODE score/event alignment failed")
    likelihood = np.asarray(namespace["likelihood"])
    if likelihood.shape != (int(accepted.sum()),) or not np.isfinite(likelihood).all():
        raise ValueError("Invalid upstream R-ANODE likelihood")
    all_scores = np.full(len(all_rows), np.nan)
    all_scores[accepted] = likelihood
    offset = 0
    for name, rows in records.items():
        sr = region(rows[:, 0])
        stop = offset + int(sr.sum())
        mask = np.zeros(len(rows), bool)
        mask[sr] = accepted[offset:stop]
        scores = np.full(len(rows), np.nan)
        scores[sr] = all_scores[offset:stop]
        if original_records is not None:
            mapped = original_masks[name]
            original = original_records[name]
            if (mapped.dtype != bool or mapped.shape != (len(original),) or int(mapped.sum()) != len(rows)
                    or not np.array_equal(rows[:, [0, -1]], original[mapped][:, [0, -1]])):
                raise ValueError("Latent score export lost original event alignment")
            full_mask, full_scores = np.zeros(len(original), bool), np.full(len(original), np.nan)
            full_mask[mapped], full_scores[mapped] = mask, scores
            rows, mask, scores, sr = original, full_mask, full_scores, region(original[:, 0])
        np.savez_compressed(
            Path(output) / (name + "_scores.npz"),
            mass=rows[:, 0].astype("float32"),
            physical=rows[:, 1:-1].astype("float32"),
            labels=rows[:, -1].astype("int8"),
            scores=scores,
            mask=mask,
            is_signal_region=sr,
        )
        offset = stop
