from __future__ import annotations
import hashlib
import math
import os
import re
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import yaml
from .progress import operation_progress

DATASET_CATALOG_SCHEMA = "riddle.datasets.v1"
SUPPORTED_INPUT_FORMATS = frozenset({"auto", "hdf5"})
SUPPORTED_CHECKSUMS = frozenset({"md5", "sha256"})
SUPPORTED_PURPOSES = frozenset({"primary", "sic_background"})


class DatasetCatalogError(ValueError):
    pass


@dataclass(frozen=True)
class DatasetFile:
    uri: str
    filename: str | None = None
    input_format: str = "auto"
    checksum_algorithm: str | None = None
    checksum: str | None = None
    default_label: int | None = None
    default_weight: float | None = None
    columns: Mapping[str, str] | None = None
    purpose: str = "primary"


@dataclass(frozen=True)
class DatasetDefinition:
    name: str
    files: tuple[DatasetFile, ...]
    input_format: str = "auto"
    enforce_expected_counts: bool = False
    default_label: int = 0
    default_weight: float = 1.0
    columns: Mapping[str, str] | None = None


@dataclass(frozen=True)
class DatasetCatalog:
    path: Path
    datasets: Mapping[str, DatasetDefinition]

    def select(self, name: str) -> DatasetDefinition:
        try:
            return self.datasets[name]
        except KeyError as exc:
            choices = ", ".join(sorted(self.datasets))
            raise DatasetCatalogError(f"Unknown dataset {name!r}; available datasets: {choices}") from exc


@dataclass(frozen=True)
class ResolvedDatasetFile:
    path: Path
    source_index: int
    uri: str
    input_format: str
    default_label: int
    default_weight: float
    columns: Mapping[str, str]
    checksum_algorithm: str | None = None
    checksum: str | None = None
    purpose: str = "primary"
    sha256: str | None = None


def _mapping(value: Any, path: str) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise DatasetCatalogError(f"{path} must be a mapping")
    return dict(value)


def _input_format(value: Any, path: str) -> str:
    result = str(value or "auto").lower()
    if result not in SUPPORTED_INPUT_FORMATS:
        raise DatasetCatalogError(f"{path} must be one of {sorted(SUPPORTED_INPUT_FORMATS)}")
    return result


def _default_label(value: Any, path: str) -> int:
    result = int(value)
    if result not in (0, 1):
        raise DatasetCatalogError(f"{path} must be zero or one")
    return result


def _default_weight(value: Any, path: str) -> float:
    result = float(value)
    if not result >= 0.0 or not result < float("inf"):
        raise DatasetCatalogError(f"{path} must be finite and nonnegative")
    return result


def _columns(value: Any, path: str) -> dict[str, str]:
    columns = _mapping(value, path)
    result = {str(target): str(source) for target, source in columns.items()}
    if any((not target or not source for target, source in result.items())):
        raise DatasetCatalogError(f"{path} cannot contain empty column names")
    return result


def _dataset_file(
    raw: Any,
    *,
    path: str,
    dataset_format: str,
    dataset_label: int,
    dataset_weight: float,
    dataset_columns: Mapping[str, str],
) -> DatasetFile:
    if isinstance(raw, str):
        raw = {"uri": raw}
    values = _mapping(raw, path)
    uri = str(values.get("uri", "")).strip()
    if not uri:
        raise DatasetCatalogError(f"{path}.uri cannot be empty")
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme and parsed.scheme not in {"file", "http", "https"}:
        raise DatasetCatalogError(f"{path}.uri uses unsupported scheme {parsed.scheme!r}")
    checksum_values = _mapping(values.get("checksum"), f"{path}.checksum")
    checksum_algorithm: str | None = None
    checksum: str | None = None
    if checksum_values:
        checksum_algorithm = str(checksum_values.get("algorithm", "")).lower()
        checksum = str(checksum_values.get("value", "")).lower()
        if checksum_algorithm not in SUPPORTED_CHECKSUMS:
            raise DatasetCatalogError(f"{path}.checksum.algorithm must be md5 or sha256")
        expected_length = hashlib.new(checksum_algorithm).digest_size * 2
        if len(checksum) != expected_length or any(
            (character not in "0123456789abcdef" for character in checksum)
        ):
            raise DatasetCatalogError(f"{path}.checksum.value is not a valid {checksum_algorithm} digest")
    file_columns = dict(dataset_columns)
    file_columns.update(_columns(values.get("columns"), f"{path}.columns"))
    purpose = str(values.get("purpose", "primary")).lower()
    if purpose not in SUPPORTED_PURPOSES:
        raise DatasetCatalogError(f"{path}.purpose must be one of {sorted(SUPPORTED_PURPOSES)}")
    return DatasetFile(
        uri=uri,
        filename=str(values["filename"]) if values.get("filename") else None,
        input_format=_input_format(values.get("format", dataset_format), f"{path}.format"),
        checksum_algorithm=checksum_algorithm,
        checksum=checksum,
        default_label=_default_label(values.get("default_label", dataset_label), f"{path}.default_label"),
        default_weight=_default_weight(
            values.get("default_weight", dataset_weight), f"{path}.default_weight"
        ),
        columns=file_columns,
        purpose=purpose,
    )


def load_dataset_catalog(path: str | Path | None = None) -> DatasetCatalog:
    if path is None:
        path = Path(__file__).resolve().parents[1] / "config" / "datasets.yaml"
    resolved = Path(path).expanduser().resolve()
    try:
        raw = yaml.safe_load(resolved.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise DatasetCatalogError(f"Invalid YAML in {resolved}: {exc}") from exc
    values = _mapping(raw, "catalog")
    if values.get("schema_version") != DATASET_CATALOG_SCHEMA:
        raise DatasetCatalogError("Unsupported dataset catalog schema_version")
    raw_datasets = _mapping(values.get("datasets"), "datasets")
    if not raw_datasets:
        raise DatasetCatalogError("datasets cannot be empty")
    datasets: dict[str, DatasetDefinition] = {}
    for name, raw_definition in raw_datasets.items():
        name = str(name)
        if re.fullmatch("[A-Za-z0-9][A-Za-z0-9_.-]*", name) is None:
            raise DatasetCatalogError(f"Invalid dataset name {name!r}")
        definition_path = f"datasets.{name}"
        definition = _mapping(raw_definition, definition_path)
        input_format = _input_format(definition.get("format", "auto"), f"{definition_path}.format")
        defaults = _mapping(definition.get("defaults"), f"{definition_path}.defaults")
        default_label = _default_label(defaults.get("label", 0), f"{definition_path}.defaults.label")
        default_weight = _default_weight(
            defaults.get("event_weight", 1.0), f"{definition_path}.defaults.event_weight"
        )
        columns = _columns(definition.get("columns"), f"{definition_path}.columns")
        raw_files = definition.get("files", ())
        if not isinstance(raw_files, list):
            raise DatasetCatalogError(f"{definition_path}.files must be a list")
        files = tuple(
            (
                _dataset_file(
                    item,
                    path=f"{definition_path}.files[{index}]",
                    dataset_format=input_format,
                    dataset_label=default_label,
                    dataset_weight=default_weight,
                    dataset_columns=columns,
                )
                for index, item in enumerate(raw_files)
            )
        )
        datasets[name] = DatasetDefinition(
            name=name,
            files=files,
            input_format=input_format,
            enforce_expected_counts=bool(definition.get("enforce_expected_counts", False)),
            default_label=default_label,
            default_weight=default_weight,
            columns=columns,
        )
    return DatasetCatalog(path=resolved, datasets=datasets)


_IO_BLOCK_BYTES = 8 * 1024 * 1024


def _file_hashes(
    path: Path, algorithms: tuple[str, ...], *, progress: Any | None = None, label: str | None = None
) -> dict[str, str]:
    unique_algorithms = tuple(dict.fromkeys(algorithms))
    digests = {algorithm: hashlib.new(algorithm) for algorithm in unique_algorithms}
    total = max(1, math.ceil(path.stat().st_size / _IO_BLOCK_BYTES))

    def consume(activity: Any | None = None) -> None:
        processed_bytes = 0
        with path.open("rb") as handle:
            for block in iter(lambda: handle.read(_IO_BLOCK_BYTES), b""):
                for digest in digests.values():
                    digest.update(block)
                processed_bytes += len(block)
                if activity is not None:
                    activity.update(read=f"{processed_bytes / 1024**3:.2f} GiB")

    if progress is None:
        consume()
    else:
        with progress.activity(label or f"Checksum {path.name}", total, unit="chunk") as activity:
            consume(activity)
    return {algorithm: digest.hexdigest() for algorithm, digest in digests.items()}


def _required_hashes(source: DatasetFile) -> tuple[str, ...]:
    return tuple(dict.fromkeys((source.checksum_algorithm or "sha256", "sha256")))


def _validated_sha256(source: DatasetFile, observed: Mapping[str, str]) -> str:
    if source.checksum_algorithm is not None and observed[source.checksum_algorithm] != source.checksum:
        raise RuntimeError(
            f"Checksum mismatch for {source.uri}: expected {source.checksum}, observed {observed[source.checksum_algorithm]}"
        )
    return observed["sha256"]


def _validate_checksum(path: Path, source: DatasetFile, *, progress: Any | None = None) -> str:
    observed = _file_hashes(
        path, _required_hashes(source), progress=progress, label=f"Verify and fingerprint {path.name}"
    )
    return _validated_sha256(source, observed)


def _safe_filename(source: DatasetFile, source_index: int) -> str:
    parsed = urllib.parse.urlparse(source.uri)
    candidate = source.filename or Path(urllib.parse.unquote(parsed.path)).name
    if not candidate or candidate in {".", ".."}:
        candidate = f"input_{source_index:05d}"
    candidate = Path(candidate).name
    return f"{source_index:05d}_{candidate}"


def _resolve_local_uri(uri: str, catalog_root: Path) -> Path:
    parsed = urllib.parse.urlparse(uri)
    if parsed.scheme == "file":
        return Path(urllib.request.url2pathname(parsed.path)).expanduser().resolve()
    path = Path(uri).expanduser()
    return (catalog_root / path).resolve() if not path.is_absolute() else path.resolve()


def materialize_dataset_files(
    definition: DatasetDefinition,
    directory: str | Path,
    *,
    catalog_root: str | Path | None = None,
    overwrite: bool = False,
    progress: Any | None = None,
) -> tuple[ResolvedDatasetFile, ...]:
    if not definition.files:
        raise DatasetCatalogError(f"Dataset {definition.name!r} has no files; add local paths or HTTPS URLs")
    directory = Path(directory).expanduser().resolve()
    directory.mkdir(parents=True, exist_ok=True)
    catalog_root = (
        Path(catalog_root).expanduser().resolve() if catalog_root is not None else Path.cwd().resolve()
    )
    resolved: list[ResolvedDatasetFile] = []
    if progress is not None:
        progress.start_stage("resolve and verify input files", len(definition.files))
    for source_index, source in enumerate(definition.files):
        if source_index > np_uint16_max():
            raise DatasetCatalogError("A dataset cannot contain more than 65,536 files")
        parsed = urllib.parse.urlparse(source.uri)
        display_name = (
            source.filename or Path(urllib.parse.unquote(parsed.path)).name or f"input_{source_index}"
        )
        if progress is not None:
            progress.start_task(f"input {source_index + 1}/{len(definition.files)} | {display_name}")
        if parsed.scheme in {"http", "https"}:
            target = directory / definition.name / _safe_filename(source, source_index)
            target.parent.mkdir(parents=True, exist_ok=True)
            if target.exists() and (not overwrite):
                source_sha256 = _validate_checksum(target, source, progress=progress)
            else:
                temporary = target.with_name(f".{target.name}.download-{uuid.uuid4().hex}")
                request = urllib.request.Request(source.uri, headers={"User-Agent": "RIDDLE/0.1"})
                try:
                    with (
                        urllib.request.urlopen(request, timeout=120) as response,
                        temporary.open("wb") as output,
                    ):
                        content_length = int(response.headers.get("Content-Length") or 0)
                        download_digests = {
                            algorithm: hashlib.new(algorithm) for algorithm in _required_hashes(source)
                        }

                        def download(activity: Any | None = None) -> None:
                            downloaded_bytes = 0
                            while block := response.read(_IO_BLOCK_BYTES):
                                output.write(block)
                                for digest in download_digests.values():
                                    digest.update(block)
                                downloaded_bytes += len(block)
                                if activity is not None:
                                    activity.update(downloaded=f"{downloaded_bytes / 1024**3:.2f} GiB")

                        if progress is not None and content_length > 0:
                            total = max(1, math.ceil(content_length / _IO_BLOCK_BYTES))
                            with progress.activity(
                                f"Download {target.name}", total, unit="chunk"
                            ) as activity:
                                download(activity)
                        elif progress is not None:
                            with operation_progress(f"Download {target.name}"):
                                download()
                        else:
                            download()
                        output.flush()
                        os.fsync(output.fileno())
                    source_sha256 = _validated_sha256(
                        source,
                        {algorithm: digest.hexdigest() for algorithm, digest in download_digests.items()},
                    )
                    os.replace(temporary, target)
                finally:
                    temporary.unlink(missing_ok=True)
            local_path = target
        else:
            local_path = _resolve_local_uri(source.uri, catalog_root)
            if not local_path.is_file():
                raise FileNotFoundError(f"Dataset input does not exist: {local_path}")
            source_sha256 = _validate_checksum(local_path, source, progress=progress)
        resolved.append(
            ResolvedDatasetFile(
                path=local_path,
                source_index=source_index,
                uri=source.uri,
                input_format=source.input_format,
                default_label=int(source.default_label),
                default_weight=float(source.default_weight),
                columns=dict(source.columns or {}),
                checksum_algorithm=source.checksum_algorithm,
                checksum=source.checksum,
                sha256=source_sha256,
                purpose=source.purpose,
            )
        )
        if progress is not None:
            size_gib = local_path.stat().st_size / 1024**3
            progress.advance(f"ready: {local_path.name} ({size_gib:.2f} GiB)")
    return tuple(resolved)


def np_uint16_max() -> int:
    return (1 << 16) - 1
