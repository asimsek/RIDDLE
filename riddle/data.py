from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
import json
import os
from pathlib import Path
import tempfile
from threading import Lock

import numpy as np

from .data_spec import DatasetSpec
from .features import FEATURES, canonicalize_event_columns, build_dataset_roles, partition_row_order
from .datasets import load_dataset_catalog, materialize_dataset_files
from .storage import locked, write_json, save_array, file_digest, verify_artifacts
from .progress import operation_progress, ProgressReporter
from . import controls


SCENARIOS = ("background_only", "signal_injection")
DATA_FILES = tuple(f"{r}data_{p}.npy" for r in ("inner", "outer") for p in ("train", "val", "test")) + (
    "innerdata_extrabkg_test.npy",
    "innerdata_extrasig.npy",
)


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
    if (not isinstance(counts, dict) or set(counts) != set(DATA_FILES)
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
        def save_events(name, events, selected):
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
        background = roles["sic_evaluation_background"]
        save_events("innerdata_extrabkg_test.npy", background, background["source_index"] != 0)
        signal = roles[f"sic_evaluation_signal_{scenario}"]
        heldout = (batch["partition"] == "final_test") & (batch["label"] == 1)
        extra_signal = ~np.isin(signal["source_entry"], batch["source_entry"][heldout])
        save_events("innerdata_extrasig.npy", signal, extra_signal)
        from .storage import save_npz
        save_npz(directory / "event_ids.npz", **identities)
        sr_labels = batch["label"][batch["is_signal_region"]]
        full_sr_b, full_sr_s = int((sr_labels == 0).sum()), int((sr_labels == 1).sum())
        files = {name: file_digest(directory / name) for name in DATA_FILES}
        write_json(
            directory / "inputs.json",
            {
                "schema": 2,
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
        spec = DatasetSpec()
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


def validate(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "inputs.json").read_text())
    variant = manifest.get("variant", "default")
    expected_columns = controls.columns(variant)
    if manifest.get("columns", expected_columns) != expected_columns:
        raise ValueError("Prepared feature columns disagree with the variant")
    if manifest.get("control", controls.description(variant)) != controls.description(variant):
        raise ValueError("Prepared control definition changed")
    if (
        manifest.get("schema") not in (1, 2)
        or manifest.get("scenario") not in SCENARIOS
        or set(manifest.get("files", {})) != set(DATA_FILES)
    ):
        raise ValueError("Invalid prepared LHCO manifest")
    verify_artifacts(directory, manifest["files"], "Verify prepared LHCO arrays")
    counts = {part: np.zeros(2, dtype=np.int64) for part in ("train", "val", "test")}
    sr_counts = np.zeros(2, dtype=np.int64)
    lengths = {}
    for name in DATA_FILES:
        array = np.load(directory / name, mmap_mode="r", allow_pickle=False)
        lengths[name] = len(array)
        if (
            array.ndim != 2
            or array.shape[1] != len(expected_columns)
            or array.dtype != np.float64
            or not np.isfinite(array).all()
            or not np.isin(array[:, -1], (0, 1)).all()
        ):
            raise ValueError("Invalid LHCO event array")
        region = (array[:, 0] > 3.3) & (array[:, 0] < 3.7)
        if (name.startswith("inner") and not region.all()) or (name.startswith("outer") and region.any()):
            raise ValueError("LHCO region membership changed")
        if name == "innerdata_extrabkg_test.npy" and np.any(array[:, -1] != 0):
            raise ValueError("Dedicated background source contains signal")
        if name == "innerdata_extrasig.npy" and np.any(array[:, -1] != 1):
            raise ValueError("Extra signal evaluation source contains background")
        if name in DATA_FILES[:6]:
            part = name.rsplit("_", 1)[1].removesuffix(".npy")
            counts[part] += np.bincount(array[:, -1].astype(int), minlength=2)
            sr_counts += np.bincount(array[region, -1].astype(int), minlength=2)
    if manifest["schema"] == 2:
        id_path = directory / "event_ids.npz"
        if file_digest(id_path) != manifest.get("event_ids_sha256"):
            raise ValueError("Prepared event identities changed")
        with np.load(id_path, allow_pickle=False) as ids:
            if set(ids.files) != set(DATA_FILES):
                raise ValueError("Missing event identity arrays")
            arrays = []
            for name in DATA_FILES:
                values = ids[name]
                if values.shape != (lengths[name], 2) or values.dtype != np.uint64:
                    raise ValueError("Misaligned prepared event identities")
                arrays.append(values)
        all_ids = np.ascontiguousarray(np.concatenate(arrays)).view("V16").ravel()
        if len(np.unique(all_ids)) != len(all_ids):
            raise ValueError("Event overlap between preparation partitions/evaluation sources")
        stored = manifest.get("uncut_signal_region", {})
        if (stored.get("background"), stored.get("signal")) != tuple(sr_counts):
            raise ValueError("Uncut SR population counts changed")
    profile = diagnostic_profile(manifest)
    if profile is not None and lengths != profile["row_counts"]:
        raise ValueError("Diagnostic subset partition counts changed")
    if not manifest.get("synthetic_smoke_fixture") and profile is None:
        spec = DatasetSpec()
        scan = manifest.get("injection_scan")
        if scan is not None:
            from dataclasses import replace
            if manifest.get("partition_policy") != "independent_class_permutations" or manifest["schema"] != 2:
                raise ValueError("Scan requires traceable independently permuted partitions")
            if type(scan.get("signal_events")) is not int or not 3 <= scan["signal_events"] < spec.signal_rows:
                raise ValueError("Invalid scan signal count")
            if type(scan.get("preparation_seed")) is not int or not 0 <= scan["preparation_seed"] < 2**32:
                raise ValueError("Invalid scan preparation seed")
            spec = replace(spec, injected_signal_rows=scan["signal_events"], preparation_seed=scan["preparation_seed"])
        if manifest.get("preparation") != asdict(spec) or manifest.get("mass_unit") != "TeV":
            raise ValueError("Prepared data does not match the LHCO protocol")
        for label, total in (
            (0, spec.background_rows),
            (1, spec.injected_signal_rows if manifest["scenario"] == "signal_injection" else 0),
        ):
            expected = (total // 2, 2 * total // 3 - total // 2, total - 2 * total // 3)
            if tuple(counts[part][label] for part in ("train", "val", "test")) != expected:
                raise ValueError("LHCO partition event counts changed")
    return manifest
