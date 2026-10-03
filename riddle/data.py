from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import tempfile
from threading import Lock

import numpy as np

from .data_spec import DatasetSpec
from .features import FEATURES, canonicalize_event_columns, build_dataset_roles, partition_row_order, partition_labels
from .datasets import load_dataset_catalog, materialize_dataset_files
from .storage import locked, write_json, save_array, file_digest, verify_artifacts, digest, atomic_write, save_npz
from .progress import operation_progress, ProgressReporter
from . import controls


SCENARIOS = ("background_only", "signal_injection")
CORE_DATA_FILES = tuple(f"{r}data_{p}.npy" for r in ("inner", "outer") for p in ("train", "val", "test")) + (
    "innerdata_extrabkg_test.npy",
    "innerdata_extrasig.npy",
)
ORACLE_DATA_FILES = (
    "innerdata_extrabkg_train.npy",
    "innerdata_extrabkg_val.npy",
    "innerdata_extrasig_train.npy",
    "innerdata_extrasig_val.npy",
)
DATA_FILES = CORE_DATA_FILES + ORACLE_DATA_FILES
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


def export_roles(roles, output, spec, source, *, smoke=False, variant="default", scan=None, scenarios=SCENARIOS, population=None):
    output = Path(output)
    for scenario in scenarios:
        batch = roles[scenario]
        order = partition_row_order(batch["partition"], preparation_seed=spec.preparation_seed, signal_rows=spec.signal_rows, sic_background_rows=spec.sic_background_rows)
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
        save_events("innerdata_extrabkg_train.npy", roles["oracle_background_train"])
        save_events("innerdata_extrabkg_val.npy", roles["oracle_background_val"])
        save_events("innerdata_extrasig_train.npy", roles["oracle_signal_train"])
        save_events("innerdata_extrasig_val.npy", roles["oracle_signal_val"])
        background = roles["sic_evaluation_background"]
        background_sources = [item for item in source if item.get("purpose") == "sic_background"]
        primary_sources = [item for item in source if item.get("purpose") == "primary"]
        if len(background_sources) != 1 or len(primary_sources) != 1:
            raise ValueError("Prepared RIDDLE oracle data require one primary and one dedicated background source")
        background_source_index = int(background_sources[0]["source_index"])
        primary_source_index = int(primary_sources[0]["source_index"])
        save_events("innerdata_extrabkg_test.npy", background, background["source_index"] == background_source_index)
        signal = roles[f"sic_evaluation_signal_{scenario}"]
        save_events("innerdata_extrasig.npy", signal)
        reservoir = roles["injection_reservoir"]
        identities[INJECTION_RESERVOIR_KEY] = np.column_stack(
            (reservoir["source_index"], reservoir["source_entry"])
        ).astype(np.uint64)
        from .storage import save_npz
        if population is not None:
            from .populations import EVALUATION_KEY, EVALUATION_FILES, contract
            identities[EVALUATION_KEY] = np.concatenate([identities[n] for n in EVALUATION_FILES])
        save_npz(directory / "event_ids.npz", **identities)
        sr_labels = batch["label"][batch["is_signal_region"]]
        full_sr_b, full_sr_s = int((sr_labels == 0).sum()), int((sr_labels == 1).sum())
        files = {name: file_digest(directory / name) for name in DATA_FILES}
        oracle_counts = {name: int(len(identities[name])) for name in ORACLE_DATA_FILES + ("innerdata_extrabkg_test.npy", "innerdata_extrasig.npy")}
        write_json(
            directory / "inputs.json",
            {
                "schema": 5 if population is not None else 4,
                **({"shared_population": contract(population, identities)} if population is not None else {}),
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
                "partition_policy": ("independent_class_permutations" if scan else
                                     "shared_percentages_fixed_order_v1" if population is not None else "fixed_upstream_partition"),
                "injection_scan": scan,
                "injection_reservoir": {
                    "events": int(len(identities[INJECTION_RESERVOIR_KEY])),
                    "event_ids_sha256": digest(identities[INJECTION_RESERVOIR_KEY]),
                    "nested_prefix_selection": True,
                },
                "riddle_oracle_pools": {
                    "counts": oracle_counts,
                    "partition_policy": "shared_percentages_v1" if population is not None else "riddle_partition_labels",
                    "injection_reservoir_removed_before_signal_partition": True,
                    "source_purpose_policy": "dataset_catalog_purpose_v1",
                    "primary_source_index": primary_source_index,
                    "background_source_index": background_source_index,
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
    provenance = [
        {
            "source_index": int(s.source_index),
            "purpose": s.purpose,
            "filename": s.path.name,
            "uri": s.uri,
            "md5": s.checksum,
            "sha256": s.sha256,
        }
        for s in sources
    ]
    return primary[0], extra[0], {s.source_index: a for s, a in zip(sources, arrays)}, provenance


def prepare(args):
    from .populations import load_config, percentage_counts
    from .settings import load_settings
    population = load_config(getattr(args, "population_config", None))
    settings = load_settings(getattr(args, "config", None) or load_settings.__defaults__[0])
    reservoir_rows = max(population["injected_signal_events"], max(settings["injection_scan"]["signal_events"]))
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
            if manifest.get("shared_population", {}).get("configuration") != population:
                raise ValueError("Prepared populations differ; use a new output directory")
            if manifest["preparation"]["injection_reservoir_rows"] != reservoir_rows:
                raise ValueError("Prepared injection reservoir differs; use a new output directory")
            if manifest.get("variant", "default") != variant:
                raise ValueError("Prepared variant differs; use a separate output directory")
        return
    root.parent.mkdir(parents=True, exist_ok=True)
    with locked(root.parent / ("." + root.name + ".prepare.lock")):
        primary, extra, by_source, provenance = read_sources(args, root, variant)
        spec = replace(DatasetSpec(), injection_reservoir_rows=reservoir_rows,
                       injected_signal_rows=population["injected_signal_events"],
                       preparation_seed=population["preparation_seed"],
                       sculpting_test_rows=percentage_counts(DatasetSpec().background_rows, population["splits"]["background"])[2])
        with operation_progress("Construct deterministic LHCO partitions"):
            roles = build_dataset_roles(
                primary, spec, sic_background_arrays=extra, enforce_expected_counts=True, population=population
            )
            if variant == "deltaR":
                controls.attach_delta_r(roles, by_source)
        stage = Path(tempfile.mkdtemp(prefix="." + root.name + "-", dir=root.parent))
        with operation_progress("Write and verify prepared arrays"):
            export_roles(roles, stage, spec, provenance, variant=variant, population=population)
            for scenario in SCENARIOS:
                validate(stage / scenario)
        os.rename(stage, root)


def benchmark_source_signature(manifest):
    source = manifest.get("source")
    if not isinstance(source, list):
        raise ValueError("RIDDLE benchmark source provenance is missing")
    signature = []
    for purpose in ("primary", "sic_background"):
        matches = [item for item in source if isinstance(item, dict) and item.get("purpose") == purpose]
        if len(matches) != 1:
            raise ValueError("RIDDLE benchmark requires one primary and one dedicated background source")
        item = matches[0]
        source_index = item.get("source_index")
        sha256 = item.get("sha256")
        if type(source_index) is not int or not isinstance(sha256, str) or len(sha256) != 64 or any(c not in "0123456789abcdef" for c in sha256):
            raise ValueError("RIDDLE benchmark source identity is invalid")
        signature.append({"purpose": purpose, "source_index": int(source_index), "sha256": sha256})
    return signature


def validate(directory, *, require_event_ids=False, require_oracle=False):
    directory = Path(directory)
    manifest = json.loads((directory / "inputs.json").read_text())
    variant = manifest.get("variant", "default")
    expected_columns = controls.columns(variant)
    if manifest.get("columns", expected_columns) != expected_columns:
        raise ValueError("Prepared feature columns disagree with the variant")
    if manifest.get("control", controls.description(variant)) != controls.description(variant):
        raise ValueError("Prepared control definition changed")
    schema = manifest.get("schema")
    from .populations import EVALUATION_KEY, verify_evaluation, validate_config, percentage_counts
    population = validate_config(manifest["shared_population"]["configuration"]) if schema == 5 else None
    if population is not None:
        preparation = manifest.get("preparation", {})
        if not manifest.get("injection_scan") and (
                preparation.get("injected_signal_rows") != population["injected_signal_events"]
                or preparation.get("preparation_seed") != population["preparation_seed"]):
            raise ValueError("Shared population configuration differs from its preparation contract")
    expected_files = DATA_FILES if schema in (3, 4, 5) else CORE_DATA_FILES
    if (schema not in (1, 2, 3, 4, 5) or manifest.get("scenario") not in SCENARIOS
            or set(manifest.get("files", {})) != set(expected_files)):
        raise ValueError("Invalid prepared LHCO manifest")
    if require_event_ids and schema not in (2, 3, 4, 5):
        raise ValueError("RIDDLE production training requires prepared data with event identities")
    if require_oracle and schema not in (4, 5):
        raise ValueError("Idealized RIDDLE and Supervised RIDDLE require schema-4/5 RIDDLE oracle data; re-run data preparation with the current framework")
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
    if schema in (2, 3, 4, 5):
        id_path = directory / "event_ids.npz"
        if not id_path.is_file():
            raise ValueError("Prepared data is missing event identities")
        if file_digest(id_path) != manifest.get("event_ids_sha256"):
            raise ValueError("Prepared event identities changed")
        expected_id_keys = set(expected_files) | ({INJECTION_RESERVOIR_KEY} if schema in (3, 4, 5) else set())
        if schema == 5:
            expected_id_keys.add(EVALUATION_KEY)
        with np.load(id_path, allow_pickle=False) as ids:
            if set(ids.files) != expected_id_keys:
                raise ValueError("Missing event identity arrays")
            if schema == 5:
                verify_evaluation(manifest, ids)
            arrays = []
            saved_ids = {}
            for name in expected_files:
                values = ids[name]
                if values.shape != (lengths[name], 2) or values.dtype != np.uint64:
                    raise ValueError("Misaligned prepared event identities")
                arrays.append(values)
                saved_ids[name] = values
            reservoir_ids = None
            if schema in (3, 4, 5):
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
        if schema in (3, 4, 5):
            reservoir_view = np.ascontiguousarray(reservoir_ids).view("V16").ravel()
            oracle_signal_names = ("innerdata_extrasig_train.npy", "innerdata_extrasig_val.npy", "innerdata_extrasig.npy")
            oracle_signal_ids = np.ascontiguousarray(np.concatenate([saved_ids[name] for name in oracle_signal_names])).view("V16").ravel()
            if np.intersect1d(reservoir_view, oracle_signal_ids).size:
                raise ValueError("Injection reservoir overlaps reserved simulation or final signal evaluation")
            expected_counts = {name: lengths[name] for name in ORACLE_DATA_FILES + ("innerdata_extrabkg_test.npy", "innerdata_extrasig.npy")}
            if schema in (4, 5):
                pools = manifest.get("riddle_oracle_pools", {})
                if (pools.get("counts") != expected_counts
                        or pools.get("partition_policy") != ("shared_percentages_v1" if schema == 5 else "riddle_partition_labels")
                        or pools.get("injection_reservoir_removed_before_signal_partition") is not True):
                    raise ValueError("RIDDLE oracle prepared event pools changed")
                background_names = ("innerdata_extrabkg_train.npy", "innerdata_extrabkg_val.npy", "innerdata_extrabkg_test.npy")
                signal_names = ("innerdata_extrasig_train.npy", "innerdata_extrasig_val.npy", "innerdata_extrasig.npy")
                for names, pool in ((background_names, "additional_background"), (signal_names, "supervised_signal")):
                    total = sum(lengths[name] for name in names)
                    labels = partition_labels(total, population["splits"][pool] if population else None)
                    expected_partition_counts = (
                        int(np.count_nonzero(labels == "training")),
                        int(np.count_nonzero(labels == "validation")),
                        int(np.count_nonzero(labels == "final_test")),
                    )
                    if tuple(lengths[name] for name in names) != expected_partition_counts:
                        raise ValueError("RIDDLE oracle simulation pools no longer follow RIDDLE partitioning")
                if require_oracle:
                    source_entries = manifest.get("source", [])
                    if (not isinstance(source_entries, list) or len(source_entries) < 2
                            or any(type(item.get("source_index")) is not int or item.get("purpose") not in ("primary", "sic_background") for item in source_entries)):
                        raise ValueError("RIDDLE oracle source-purpose provenance is missing; re-run data preparation with the current framework")
                    primary_entries = [item for item in source_entries if item["purpose"] == "primary"]
                    background_entries = [item for item in source_entries if item["purpose"] == "sic_background"]
                    if len(primary_entries) != 1 or len(background_entries) != 1:
                        raise ValueError("RIDDLE oracle data require one primary and one dedicated background source")
                    primary_index = int(primary_entries[0]["source_index"])
                    background_index = int(background_entries[0]["source_index"])
                    if (pools.get("source_purpose_policy") != "dataset_catalog_purpose_v1"
                            or pools.get("primary_source_index") != primary_index
                            or pools.get("background_source_index") != background_index):
                        raise ValueError("RIDDLE oracle source-purpose contract changed")
                    for name in background_names:
                        if np.any(saved_ids[name][:, 0] != background_index):
                            raise ValueError("RIDDLE oracle background simulation does not come from the dedicated background source")
                    for name in signal_names:
                        if np.any(saved_ids[name][:, 0] != primary_index):
                            raise ValueError("RIDDLE oracle signal simulation does not come from the primary LHCO source")
                    if np.any(reservoir_ids[:, 0] != primary_index):
                        raise ValueError("RIDDLE injection reservoir does not come from the primary LHCO source")
                core_names = CORE_DATA_FILES[:6]
                core_signal_ids = []
                for name in core_names:
                    rows = np.load(directory / name, mmap_mode="r", allow_pickle=False)
                    selected = rows[:, -1] == 1
                    if np.any(selected):
                        core_signal_ids.append(saved_ids[name][selected])
                if manifest["scenario"] == "signal_injection":
                    injected_ids = np.concatenate(core_signal_ids) if core_signal_ids else np.empty((0, 2), dtype=np.uint64)
                    injected_view = np.ascontiguousarray(injected_ids).view("V16").ravel()
                    if len(injected_ids) != manifest.get("preparation", {}).get("injected_signal_rows") or np.setdiff1d(injected_view, reservoir_view).size:
                        raise ValueError("Injected signal identities do not match the fixed RIDDLE injection reservoir")
                elif core_signal_ids:
                    raise ValueError("Background-only RIDDLE oracle data contains injected signal")
        stored = manifest.get("uncut_signal_region", {})
        if (stored.get("background"), stored.get("signal")) != tuple(sr_counts):
            raise ValueError("Uncut SR population counts changed")
    profile = diagnostic_profile(manifest)
    if population is not None:
        for label, pool in ((0, "background"), (1, "injected_signal")):
            observed = tuple(int(counts[part][label]) for part in ("train", "val", "test"))
            if observed != percentage_counts(sum(observed), population["splits"][pool]):
                raise ValueError("Shared main-population percentages changed")
    if profile is not None and {name: lengths[name] for name in CORE_DATA_FILES} != profile["row_counts"]:
        raise ValueError("Diagnostic subset partition counts changed")
    if not manifest.get("synthetic_smoke_fixture") and profile is None:
        if schema in (3, 4, 5):
            preparation = manifest.get("preparation")
            if not isinstance(preparation, dict):
                raise ValueError("Prepared data is missing its preparation contract")
            fields = set(DatasetSpec.__dataclass_fields__)
            try:
                spec = DatasetSpec(**{key: value for key, value in preparation.items() if key in fields})
            except TypeError as error:
                raise ValueError("Invalid prepared-data contract") from error
            if (spec.background_rows != DatasetSpec().background_rows or spec.signal_rows != DatasetSpec().signal_rows
                    or spec.sic_background_rows != DatasetSpec().sic_background_rows
                    or spec.sculpting_test_rows != (percentage_counts(spec.background_rows, population["splits"]["background"])[2] if population else DatasetSpec().sculpting_test_rows)
                    or spec.injected_signal_rows > spec.injection_reservoir_rows):
                raise ValueError("Prepared data does not match the LHCO protocol")
            if schema in (4, 5) and preparation != asdict(spec):
                raise ValueError("Schema-4 prepared data contains a non-RIDDLE benchmark partition contract")
        else:
            spec = DatasetSpec()
            preparation = manifest.get("preparation", {})
            legacy = {k: v for k, v in asdict(spec).items() if k != "injection_reservoir_rows"}
            if not isinstance(preparation, dict) or any(preparation.get(key) != value for key, value in legacy.items()):
                raise ValueError("Prepared data does not match the legacy LHCO protocol")
        scan = manifest.get("injection_scan")
        if scan is not None:
            if manifest.get("partition_policy") != "independent_class_permutations" or schema not in (2, 3, 4, 5):
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
            expected = (percentage_counts(total, population["splits"]["background" if label == 0 else "injected_signal"])
                        if population else (total // 2, 2 * total // 3 - total // 2, total - 2 * total // 3))
            if tuple(counts[part][label] for part in ("train", "val", "test")) != expected:
                raise ValueError("LHCO partition event counts changed")
    return manifest
