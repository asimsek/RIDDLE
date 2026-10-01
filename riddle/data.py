from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
from threading import Lock

import numpy as np

from .data_spec import DatasetSpec
from .features import FEATURES, canonicalize_event_columns, build_dataset_roles, partition_row_order
from .datasets import load_dataset_catalog, materialize_dataset_files
from .storage import locked, write_json, save_array, file_digest, verify_artifacts, digest, atomic_write, save_npz
from .progress import operation_progress, ProgressReporter
from . import controls


SCENARIOS = ("background_only", "signal_injection")
CORE_DATA_FILES = tuple(f"{r}data_{p}.npy" for r in ("inner", "outer") for p in ("train", "val", "test")) + (
    "innerdata_extrabkg_test.npy",
    "innerdata_extrasig.npy",
)
BASELINE_DATA_FILES = (
    "innerdata_extrabkg_train.npy",
    "innerdata_extrabkg_val.npy",
    "innerdata_extrasig_train.npy",
    "innerdata_extrasig_val.npy",
)
DATA_FILES = CORE_DATA_FILES + BASELINE_DATA_FILES
INJECTION_RESERVOIR_KEY = "injection_reservoir_event_ids"


def diagnostic_profile(manifest):
    """Explicit real-data pilot subsets; never label these as synthetic fixtures."""
    profile = manifest.get("diagnostic_subset")
    if profile is None:
        return None
    if (manifest.get("schema") != 2 or manifest.get("synthetic_smoke_fixture")
            or not isinstance(profile, dict) or profile.get("purpose") != "cpu_mass_dependence"
            or profile.get("schema") != 1):
        raise ValueError("Invalid diagnostic subset provenance")
    checksum = profile.get("parent_inputs_sha256", "")
    if (not isinstance(checksum, str) or len(checksum) != 64
            or any(c not in "0123456789abcdef" for c in checksum)):
        raise ValueError("Diagnostic subset requires its parent manifest checksum")
    for key, minimum in (("background_epochs", 11), ("reference_samples", 512)):
        if type(profile.get(key)) is not int or profile[key] < minimum:
            raise ValueError(f"Invalid diagnostic {key}")
    counts = profile.get("row_counts")
    if (not isinstance(counts, dict) or set(counts) != set(CORE_DATA_FILES)
            or any(type(n) is not int or n < 0 for n in counts.values())):
        raise ValueError("Invalid diagnostic partition counts")
    return profile


def rows(arrays, variant="default"):
    if not np.all(arrays["event_weight"] == 1):
        raise ValueError("Only unweighted LHCO inputs are supported")
    result = np.column_stack([arrays[k] for k in ("mjj", *FEATURES, "label")])
    result[:, :3] /= 1000.0
    return controls.transform(result, arrays, variant)


def export_roles(roles, output, spec, source, *, smoke=False, variant="default", scan=None, scenarios=SCENARIOS):
    output = Path(output)
    for scenario in scenarios:
        batch = roles[scenario]
        order = partition_row_order(batch["partition"], preparation_seed=spec.preparation_seed)
        batch = {k: v[order] for k, v in batch.items()}
        values = rows(batch, variant)
        directory = output / scenario
        directory.mkdir(parents=True, exist_ok=True)
        identities = {}
        def save_events(name, events, selected=None):
            if selected is None:
                selected = np.ones(len(events["source_entry"]), dtype=bool)
            save_array(directory / name, rows(events, variant)[selected])
            identities[name] = np.column_stack(
                (events["source_index"][selected], events["source_entry"][selected])
            ).astype(np.uint64)
        for part, suffix in (("training", "train"), ("validation", "val"), ("final_test", "test")):
            for region, prefix in ((True, "inner"), (False, "outer")):
                selected = (batch["partition"] == part) & (batch["is_signal_region"] == region)
                save_array(directory / f"{prefix}data_{suffix}.npy", values[selected])
                identities[f"{prefix}data_{suffix}.npy"] = np.column_stack(
                    (batch["source_index"][selected], batch["source_entry"][selected])
                ).astype(np.uint64)
        save_events("innerdata_extrabkg_train.npy", roles["baseline_background_train"])
        save_events("innerdata_extrabkg_val.npy", roles["baseline_background_val"])
        save_events("innerdata_extrasig_train.npy", roles["baseline_signal_train"])
        save_events("innerdata_extrasig_val.npy", roles["baseline_signal_val"])
        background = roles["sic_evaluation_background"]
        save_events("innerdata_extrabkg_test.npy", background, background["source_index"] != 0)
        signal = roles[f"sic_evaluation_signal_{scenario}"]
        save_events("innerdata_extrasig.npy", signal)
        reservoir = roles["injection_reservoir"]
        identities[INJECTION_RESERVOIR_KEY] = np.column_stack(
            (reservoir["source_index"], reservoir["source_entry"])
        ).astype(np.uint64)
        from .storage import save_npz
        save_npz(directory / "event_ids.npz", **identities)
        sr_labels = batch["label"][batch["is_signal_region"]]
        full_sr_b, full_sr_s = int((sr_labels == 0).sum()), int((sr_labels == 1).sum())
        files = {name: file_digest(directory / name) for name in DATA_FILES}
        baseline_counts = {name: int(len(identities[name])) for name in BASELINE_DATA_FILES + ("innerdata_extrabkg_test.npy", "innerdata_extrasig.npy")}
        write_json(
            directory / "inputs.json",
            {
                "schema": 3,
                "scenario": scenario,
                "files": files,
                "preparation": asdict(spec),
                "source": source,
                "mass_unit": "TeV",
                "columns": controls.columns(variant),
                "variant": variant,
                "control": controls.description(variant),
                "signal_region": [3.3, 3.7],
                "signal_region_boundary": "strict",
                "synthetic_smoke_fixture": smoke,
                "event_ids_sha256": file_digest(directory / "event_ids.npz"),
                "partition_policy": "independent_class_permutations" if scan else "fixed_upstream_partition",
                "injection_scan": scan,
                "injection_reservoir": {
                    "events": int(len(identities[INJECTION_RESERVOIR_KEY])),
                    "event_ids_sha256": digest(identities[INJECTION_RESERVOIR_KEY]),
                    "nested_prefix_selection": True,
                },
                "ad_baseline_pools": {
                    "counts": baseline_counts,
                    "background_reservation_events": int(spec.sic_background_validation_stop),
                    "signal_simulation_events": int(spec.signal_simulation_rows),
                    "signal_training_fraction": float(spec.signal_training_fraction),
                },
                "uncut_signal_region": {"background": full_sr_b, "signal": full_sr_s,
                    "signal_to_background": full_sr_s/full_sr_b if full_sr_b else None,
                    "nominal_significance": full_sr_s/np.sqrt(full_sr_b) if full_sr_b else None},
            },
        )
    write_json(
        output / "dataset.json",
        {
            "schema": 1,
            "scenarios": list(scenarios),
            "variant": variant,
            "manifests": {s: file_digest(output / s / "inputs.json") for s in scenarios},
        },
    )

def read_sources(args, root, variant):
    catalog = load_dataset_catalog(args.catalog)
    definition = catalog.select("lhco")
    with ProgressReporter(name="LHCO", installer_style=True) as progress:
        sources = materialize_dataset_files(
            definition, root.parent / "source", catalog_root=catalog.path.parent, progress=progress
        )

    hdf_reader = Lock()

    def convert(source):
        import pandas as pd

        with operation_progress("Read and convert LHCO source"):
            with hdf_reader:
                table = pd.read_hdf(source.path)
            converted = canonicalize_event_columns(
                table,
                source_index=source.source_index,
                default_label=source.default_label,
                default_weight=source.default_weight,
                column_mapping=source.columns,
            )
            if variant == "deltaR":
                converted["deltaR"] = controls.delta_r(table, source.columns)
            return converted

    with ThreadPoolExecutor(max_workers=min(args.io_workers, len(sources))) as pool:
        arrays = list(pool.map(convert, sources))
    primary = [a for a, s in zip(arrays, sources) if s.purpose == "primary"]
    extra = [a for a, s in zip(arrays, sources) if s.purpose == "sic_background"]
    if len(primary) != 1 or len(extra) != 1:
        raise ValueError("The LHCO contract requires one primary and one dedicated background source")
    provenance = [{"uri": s.uri, "md5": s.checksum, "sha256": s.sha256} for s in sources]
    return primary[0], extra[0], {s.source_index: a for s, a in zip(sources, arrays)}, provenance


def prepare(args):
    root = args.output.resolve()
    variant = getattr(args, "variant", "default")
    controls.columns(variant)
    if root.exists():
        if not args.resume:
            raise FileExistsError(
                "Prepared output exists; use --resume to verify it or choose a new directory"
            )
        for scenario in SCENARIOS:
            manifest = validate(root / scenario)
            if manifest.get("variant", "default") != variant:
                raise ValueError("Prepared variant differs; use a separate output directory")
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    with locked(root.parent / ("." + root.name + ".prepare.lock")):
        primary, extra, by_source, provenance = read_sources(args, root, variant)
        from .settings import load_settings
        settings = load_settings(getattr(args, "config", None) or load_settings.__defaults__[0])
        baseline_data = settings["ad_baselines"]["data"]
        reservoir_rows = max(settings["injection_scan"]["signal_events"])
        spec = replace(
            DatasetSpec(),
            injection_reservoir_rows=reservoir_rows,
            signal_simulation_rows=baseline_data["signal_simulation_events"],
            signal_training_fraction=baseline_data["signal_training_fraction"],
        )
        with operation_progress("Construct deterministic LHCO partitions"):
            roles = build_dataset_roles(
                primary, spec, sic_background_arrays=extra, enforce_expected_counts=True
            )
            if variant == "deltaR":
                controls.attach_delta_r(roles, by_source)
        stage = Path(tempfile.mkdtemp(prefix="." + root.name + "-", dir=root.parent))
        with operation_progress("Write and verify prepared arrays"):
            export_roles(roles, stage, spec, provenance, variant=variant)
            for scenario in SCENARIOS:
                validate(stage / scenario)
        os.rename(stage, root)


def validate(directory, *, require_event_ids=False, require_baseline=False):
    directory = Path(directory)
    manifest = json.loads((directory / "inputs.json").read_text())
    variant = manifest.get("variant", "default")
    expected_columns = controls.columns(variant)
    if manifest.get("columns", expected_columns) != expected_columns:
        raise ValueError("Prepared feature columns disagree with the variant")
    if manifest.get("control", controls.description(variant)) != controls.description(variant):
        raise ValueError("Prepared control definition changed")
    schema = manifest.get("schema")
    expected_files = DATA_FILES if schema == 3 else CORE_DATA_FILES
    if (schema not in (1, 2, 3) or manifest.get("scenario") not in SCENARIOS
            or set(manifest.get("files", {})) != set(expected_files)):
        raise ValueError("Invalid prepared LHCO manifest")
    if require_event_ids and schema not in (2, 3):
        raise ValueError("RIDDLE production training requires prepared data with event identities")
    if require_baseline and schema != 3:
        raise ValueError("Idealized AD and Supervised AD require schema-3 prepared data; re-run data preparation with the current framework")
    verify_artifacts(directory, manifest["files"], "Verify prepared LHCO arrays")
    counts = {part: np.zeros(2, dtype=np.int64) for part in ("train", "val", "test")}
    sr_counts = np.zeros(2, dtype=np.int64)
    lengths = {}
    for name in expected_files:
        array = np.load(directory / name, mmap_mode="r", allow_pickle=False)
        lengths[name] = len(array)
        if (array.ndim != 2 or array.shape[1] != len(expected_columns) or array.dtype != np.float64
                or not np.isfinite(array).all() or not np.isin(array[:, -1], (0, 1)).all()):
            raise ValueError("Invalid LHCO event array")
        region = (array[:, 0] > 3.3) & (array[:, 0] < 3.7)
        if (name.startswith("inner") and not region.all()) or (name.startswith("outer") and region.any()):
            raise ValueError("LHCO region membership changed")
        if name.startswith("innerdata_extrabkg") and np.any(array[:, -1] != 0):
            raise ValueError("Dedicated background source contains signal")
        if name.startswith("innerdata_extrasig") and np.any(array[:, -1] != 1):
            raise ValueError("Dedicated signal source contains background")
        if name in CORE_DATA_FILES[:6]:
            part = name.rsplit("_", 1)[1].removesuffix(".npy")
            counts[part] += np.bincount(array[:, -1].astype(int), minlength=2)
            sr_counts += np.bincount(array[region, -1].astype(int), minlength=2)
    if schema in (2, 3):
        id_path = directory / "event_ids.npz"
        if not id_path.is_file():
            raise ValueError("Prepared data is missing event identities")
        if file_digest(id_path) != manifest.get("event_ids_sha256"):
            raise ValueError("Prepared event identities changed")
        expected_id_keys = set(expected_files) | ({INJECTION_RESERVOIR_KEY} if schema == 3 else set())
        with np.load(id_path, allow_pickle=False) as ids:
            if set(ids.files) != expected_id_keys:
                raise ValueError("Missing event identity arrays")
            arrays = []
            saved_ids = {}
            for name in expected_files:
                values = ids[name]
                if values.shape != (lengths[name], 2) or values.dtype != np.uint64:
                    raise ValueError("Misaligned prepared event identities")
                arrays.append(values)
                saved_ids[name] = values
            reservoir_ids = None
            if schema == 3:
                reservoir_ids = ids[INJECTION_RESERVOIR_KEY]
                reservation = manifest.get("injection_reservoir", {})
                if (reservoir_ids.ndim != 2 or reservoir_ids.shape[1] != 2 or reservoir_ids.dtype != np.uint64
                        or len(reservoir_ids) != reservation.get("events")
                        or digest(reservoir_ids) != reservation.get("event_ids_sha256")
                        or len(np.unique(reservoir_ids, axis=0)) != len(reservoir_ids)):
                    raise ValueError("Invalid fixed injection-reservoir identities")
        all_ids = np.ascontiguousarray(np.concatenate(arrays)).view("V16").ravel()
        if len(np.unique(all_ids)) != len(all_ids):
            raise ValueError("Event overlap between preparation partitions/evaluation sources")
        if schema == 3:
            reservoir_view = np.ascontiguousarray(reservoir_ids).view("V16").ravel()
            baseline_signal_names = ("innerdata_extrasig_train.npy", "innerdata_extrasig_val.npy", "innerdata_extrasig.npy")
            baseline_signal_ids = np.ascontiguousarray(np.concatenate([saved_ids[name] for name in baseline_signal_names])).view("V16").ravel()
            if np.intersect1d(reservoir_view, baseline_signal_ids).size:
                raise ValueError("Injection reservoir overlaps classifier simulation or final signal evaluation")
            pools = manifest.get("ad_baseline_pools", {})
            expected_counts = {name: lengths[name] for name in BASELINE_DATA_FILES + ("innerdata_extrabkg_test.npy", "innerdata_extrasig.npy")}
            if pools.get("counts") != expected_counts:
                raise ValueError("AD baseline prepared event counts changed")
        stored = manifest.get("uncut_signal_region", {})
        if (stored.get("background"), stored.get("signal")) != tuple(sr_counts):
            raise ValueError("Uncut SR population counts changed")
    profile = diagnostic_profile(manifest)
    if profile is not None and {name: lengths[name] for name in CORE_DATA_FILES} != profile["row_counts"]:
        raise ValueError("Diagnostic subset partition counts changed")
    if not manifest.get("synthetic_smoke_fixture") and profile is None:
        if schema == 3:
            preparation = manifest.get("preparation")
            if not isinstance(preparation, dict):
                raise ValueError("Schema-3 prepared data is missing its preparation contract")
            try:
                spec = DatasetSpec(**preparation)
            except TypeError as error:
                raise ValueError("Invalid schema-3 preparation contract") from error
            if (spec.background_rows != DatasetSpec().background_rows or spec.signal_rows != DatasetSpec().signal_rows
                    or spec.sic_background_rows != DatasetSpec().sic_background_rows
                    or spec.sic_background_validation_stop != DatasetSpec().sic_background_validation_stop
                    or spec.sculpting_test_rows != DatasetSpec().sculpting_test_rows
                    or spec.injected_signal_rows > spec.injection_reservoir_rows
                    or spec.signal_simulation_rows <= 0 or not 0.0 < spec.signal_training_fraction < 1.0):
                raise ValueError("Prepared data does not match the LHCO protocol")
        else:
            spec = DatasetSpec()
            legacy = {k: v for k, v in asdict(spec).items() if k not in ("injection_reservoir_rows", "signal_simulation_rows", "signal_training_fraction")}
            if manifest.get("preparation") != legacy:
                raise ValueError("Prepared data does not match the legacy LHCO protocol")
        scan = manifest.get("injection_scan")
        if scan is not None:
            if manifest.get("partition_policy") != "independent_class_permutations" or schema not in (2, 3):
                raise ValueError("Scan requires traceable independently permuted partitions")
            if type(scan.get("signal_events")) is not int or not 3 <= scan["signal_events"] < spec.signal_rows:
                raise ValueError("Invalid scan signal count")
            if type(scan.get("preparation_seed")) is not int or not 0 <= scan["preparation_seed"] < 2**32:
                raise ValueError("Invalid scan preparation seed")
            if spec.injected_signal_rows != scan["signal_events"] or spec.preparation_seed != scan["preparation_seed"]:
                raise ValueError("Prepared scan split disagrees with its manifest")
        if manifest.get("mass_unit") != "TeV":
            raise ValueError("Prepared data does not match the LHCO protocol")
        for label, total in ((0, spec.background_rows), (1, spec.injected_signal_rows if manifest["scenario"] == "signal_injection" else 0)):
            expected = (total // 2, 2 * total // 3 - total // 2, total - 2 * total // 3)
            if tuple(counts[part][label] for part in ("train", "val", "test")) != expected:
                raise ValueError("LHCO partition event counts changed")
    return manifest


BASELINE_TRAINING_FILES = {
    "iad": {
        "train_class1": "innerdata_train.npy",
        "train_class0": "innerdata_extrabkg_train.npy",
        "val_class1": "innerdata_val.npy",
        "val_class0": "innerdata_extrabkg_val.npy",
    },
    "supervised": {
        "train_class1": "innerdata_extrasig_train.npy",
        "train_class0": "innerdata_extrabkg_train.npy",
        "val_class1": "innerdata_extrasig_val.npy",
        "val_class0": "innerdata_extrabkg_val.npy",
    },
}


def baseline_feature_order(manifest, include_mass=False):
    columns = list(manifest["columns"])
    features = columns[1:-1]
    return ([columns[0]] if include_mass else []) + features


def baseline_feature_matrix(event_rows, include_mass=False):
    event_rows = np.asarray(event_rows)
    return np.asarray(event_rows[:, :-1] if include_mass else event_rows[:, 1:-1], dtype=np.float64)


def load_prepared_event_ids(data_root):
    with np.load(Path(data_root) / "event_ids.npz", allow_pickle=False) as archive:
        return {name: archive[name] for name in archive.files}


def baseline_population_receipt(data_root, manifest, event_ids, filename):
    values = event_ids[filename]
    return {
        "file": filename,
        "events": int(len(values)),
        "data_sha256": manifest["files"][filename],
        "event_ids_sha256": digest(values),
    }


def baseline_preprocessing(data_root, output, classifier, manifest, event_ids):
    data_root = Path(data_root)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    source_name = "innerdata_extrabkg_train.npy"
    source_receipt = baseline_population_receipt(data_root, manifest, event_ids, source_name)
    order = baseline_feature_order(manifest, classifier["include_mass"])
    json_path = output / "preprocessing.json"
    npz_path = output / "preprocessing.npz"
    if json_path.is_file() and npz_path.is_file():
        try:
            saved = json.loads(json_path.read_text())
            if (
                saved.get("schema") == 1
                and saved.get("rule") == classifier["preprocessing"]
                and saved.get("feature_order") == order
                and saved.get("include_mass") == bool(classifier["include_mass"])
                and saved.get("source") == source_receipt
                and saved.get("artifact_sha256") == file_digest(npz_path)
            ):
                with np.load(npz_path, allow_pickle=False) as archive:
                    mean = np.asarray(archive["mean"], dtype=np.float64)
                    std = np.asarray(archive["std"], dtype=np.float64)
                    stored_order = archive["feature_order"].astype(str).tolist()
                    input_dimension = int(np.asarray(archive["input_dimension"]).item())
                    source_event_ids_sha256 = str(np.asarray(archive["source_event_ids_sha256"]).item())
                    source_data_sha256 = str(np.asarray(archive["source_data_sha256"]).item())
                    rule = str(np.asarray(archive["preprocessing"]).item())
                    include_mass = bool(np.asarray(archive["include_mass"]).item())
                valid = (
                    stored_order == order
                    and input_dimension == len(order)
                    and mean.shape == (len(order),)
                    and std.shape == (len(order),)
                    and source_event_ids_sha256 == source_receipt["event_ids_sha256"]
                    and source_data_sha256 == source_receipt["data_sha256"]
                    and rule == classifier["preprocessing"]
                    and include_mass == bool(classifier["include_mass"])
                    and np.isfinite(mean).all()
                    and np.isfinite(std).all()
                    and np.all(std > 0)
                    and saved.get("mean_sha256") == digest(mean)
                    and saved.get("std_sha256") == digest(std)
                    and int(saved.get("input_dimension", -1)) == len(order)
                )
                if valid:
                    return mean, std, saved
        except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
            pass
    source = np.load(data_root / source_name, mmap_mode="r", allow_pickle=False)
    values = baseline_feature_matrix(source, classifier["include_mass"])
    mean = values.mean(axis=0, dtype=np.float64)
    std = values.std(axis=0, dtype=np.float64)
    if len(order) != values.shape[1] or not np.isfinite(mean).all() or not np.isfinite(std).all() or np.any(std <= 0):
        raise ValueError("Invalid AD baseline background standardization statistics")
    receipt = {
        "schema": 1,
        "rule": classifier["preprocessing"],
        "feature_order": order,
        "input_dimension": int(values.shape[1]),
        "include_mass": bool(classifier["include_mass"]),
        "source": source_receipt,
        "mean_sha256": digest(mean),
        "std_sha256": digest(std),
    }
    arrays = dict(
        mean=mean,
        std=std,
        feature_order=np.asarray(order),
        input_dimension=np.asarray(values.shape[1], dtype=np.int64),
        source_event_ids_sha256=np.asarray(receipt["source"]["event_ids_sha256"]),
        source_data_sha256=np.asarray(receipt["source"]["data_sha256"]),
        preprocessing=np.asarray(classifier["preprocessing"]),
        include_mass=np.asarray(classifier["include_mass"]),
    )
    atomic_write(npz_path, lambda path: save_npz(path, **arrays))
    receipt["artifact_sha256"] = file_digest(npz_path)
    write_json(json_path, receipt)
    return mean, std, receipt


def standardize_baseline_rows(event_rows, mean, std, include_mass=False):
    values = baseline_feature_matrix(event_rows, include_mass)
    result = (values - mean) / std
    if not np.isfinite(result).all():
        raise ValueError("AD baseline preprocessing produced nonfinite features")
    return np.asarray(result, dtype=np.float32)


def balanced_classifier_weights(targets):
    targets = np.asarray(targets, dtype=np.float32)
    counts = np.bincount(targets.astype(np.int64), minlength=2)
    if np.any(counts <= 0):
        raise ValueError("AD baseline training requires both classifier classes")
    weights = np.where(targets == 1.0, len(targets) / (2.0 * counts[1]), len(targets) / (2.0 * counts[0]))
    return np.asarray(weights, dtype=np.float32)


def baseline_classifier_targets(class1_rows, class0_rows):
    return np.concatenate((np.ones(len(class1_rows), dtype=np.float32), np.zeros(len(class0_rows), dtype=np.float32)))


def build_baseline_training_arrays(method, data_root, classifier, mean, std):
    if method not in BASELINE_TRAINING_FILES:
        raise ValueError("Unknown AD baseline method")
    data_root = Path(data_root)
    files = BASELINE_TRAINING_FILES[method]
    loaded = {key: np.load(data_root / name, mmap_mode="r", allow_pickle=False) for key, name in files.items()}
    if np.any(loaded["train_class0"][:, -1] != 0) or np.any(loaded["val_class0"][:, -1] != 0):
        raise ValueError("AD baseline pure-background simulation contains signal")
    if method == "supervised" and (np.any(loaded["train_class1"][:, -1] != 1) or np.any(loaded["val_class1"][:, -1] != 1)):
        raise ValueError("Supervised AD signal simulation contains background")
    train_y = baseline_classifier_targets(loaded["train_class1"], loaded["train_class0"])
    val_y = baseline_classifier_targets(loaded["val_class1"], loaded["val_class0"])
    train_x = np.concatenate((
        standardize_baseline_rows(loaded["train_class1"], mean, std, classifier["include_mass"]),
        standardize_baseline_rows(loaded["train_class0"], mean, std, classifier["include_mass"]),
    ))
    val_x = np.concatenate((
        standardize_baseline_rows(loaded["val_class1"], mean, std, classifier["include_mass"]),
        standardize_baseline_rows(loaded["val_class0"], mean, std, classifier["include_mass"]),
    ))
    return train_x, train_y, balanced_classifier_weights(train_y), val_x, val_y, balanced_classifier_weights(val_y)


def prepare_baseline_training_cache(method, data_root, training_root, classifier, manifest):
    training_root = Path(training_root)
    cache = training_root / ".resume" / "cache"
    cache.mkdir(parents=True, exist_ok=True)
    event_ids = load_prepared_event_ids(data_root)
    mean, std, prep = baseline_preprocessing(data_root, training_root, classifier, manifest, event_ids)
    files = BASELINE_TRAINING_FILES[method]
    populations = {key: baseline_population_receipt(data_root, manifest, event_ids, name) for key, name in files.items()}
    cache_contract = {
        "schema": 1,
        "method": method,
        "classifier": classifier,
        "preprocessing": prep,
        "populations": populations,
    }
    contract_path = cache / "contract.json"
    names = ("train_x.npy", "train_y.npy", "train_weight.npy", "val_x.npy", "val_y.npy", "val_weight.npy")
    valid = False
    if contract_path.is_file() and all((cache / name).is_file() for name in names):
        previous = json.loads(contract_path.read_text())
        saved_contract = dict(previous)
        expected = saved_contract.pop("cache_sha256", {})
        valid = saved_contract == cache_contract
        if valid:
            valid = set(expected) == set(names) and all(file_digest(cache / name) == expected[name] for name in names)
    if not valid:
        arrays = build_baseline_training_arrays(method, data_root, classifier, mean, std)
        for name, array in zip(names, arrays):
            save_array(cache / name, array)
        cache_contract["cache_sha256"] = {name: file_digest(cache / name) for name in names}
        write_json(contract_path, cache_contract)
    population_path = training_root / "populations.json"
    write_json(population_path, {
        "schema": 1,
        "method": method,
        "truth_blind_training": method == "iad",
        "truth_supervised_training": method == "supervised",
        "class_balance": classifier["class_balance"],
        "training": {"class1": populations["train_class1"], "class0": populations["train_class0"]},
        "validation": {"class1": populations["val_class1"], "class0": populations["val_class0"]},
    })
    return cache, prep, populations


def load_baseline_training_cache(cache):
    cache = Path(cache)
    return tuple(np.load(cache / name, mmap_mode="r", allow_pickle=False) for name in (
        "train_x.npy", "train_y.npy", "train_weight.npy", "val_x.npy", "val_y.npy", "val_weight.npy"
    ))
