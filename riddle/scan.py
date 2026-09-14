"""Optional signal-strength study with paired methods and independent partition seeds."""

from copy import copy
from dataclasses import replace
import os
from pathlib import Path
import tempfile

from .data import read_sources, export_roles, validate
from .data_spec import DatasetSpec
from .features import build_dataset_roles
from . import controls
from .settings import load_settings
from .storage import locked, write_json, file_digest

SCAN_SCHEMA = 1


def points(args):
    config = load_settings(args.config)["injection_scan"]
    counts = getattr(args, "signal_events", None) or config["signal_events"]
    replicas = getattr(args, "replicas", None)
    replicas = list(range(config["replicas"])) if replicas is None else replicas
    if (
        not counts
        or not replicas
        or len(set(counts)) != len(counts)
        or len(set(replicas)) != len(replicas)
    ):
        raise ValueError("Empty or duplicate scan points")
    if any(n not in config["signal_events"] for n in counts):
        raise ValueError("Requested signal counts must be configured in settings.yaml")
    if any(type(r) is not int or not 0 <= r < config["replicas"] for r in replicas):
        raise ValueError("Replica index outside configured scan")
    return [
        dict(
            schema=SCAN_SCHEMA,
            signal_events=n,
            replica=r,
            preparation_seed=config["preparation_seed"] + r,
            training_seed=config["training_seed"] + r,
        )
        for n in counts
        for r in replicas
    ]


def point_name(point):
    return f"signal_{point['signal_events']:06d}/replica_{point['replica']:03d}"


def prepare_scan(args):
    root = args.output.resolve()
    plan = points(args)
    root.mkdir(parents=True, exist_ok=True)
    with locked(root / ".prepare.lock"):
        pending = []
        for point in plan:
            destination = root / point_name(point)
            if destination.exists():
                if not args.resume:
                    raise FileExistsError("Prepared scan point exists; use --resume or a new output")
                for scenario in ("signal_injection",):
                    manifest = validate(destination / scenario)
                    if (
                        manifest.get("injection_scan") != point
                        or manifest.get("variant") != args.variant
                    ):
                        raise ValueError("Prepared scan identity changed")
            else:
                pending.append(point)
        if not pending:
            return
        primary, extra, by_source, provenance = read_sources(args, root, args.variant)
        for point in pending:
            spec = replace(
                DatasetSpec(),
                injected_signal_rows=point["signal_events"],
                preparation_seed=point["preparation_seed"],
            )
            roles = build_dataset_roles(
                primary, spec, sic_background_arrays=extra, independent_partition=True
            )
            if args.variant == "deltaR":
                controls.attach_delta_r(roles, by_source)
            destination = root / point_name(point)
            destination.parent.mkdir(parents=True, exist_ok=True)
            stage = Path(tempfile.mkdtemp(prefix=".prepare-", dir=destination.parent))
            export_roles(
                roles,
                stage,
                spec,
                provenance,
                variant=args.variant,
                scan=point,
                scenarios=("signal_injection",),
            )
            for scenario in ("signal_injection",):
                validate(stage / scenario)
            os.rename(stage, destination)
        # Each point is atomic and self-describing; this inventory also includes resumed points.
        write_json(
            root / "scan_inputs.json",
            {
                "schema": SCAN_SCHEMA,
                "variant": args.variant,
                "points": {
                    str(p.parent.relative_to(root)): file_digest(p)
                    for p in sorted(root.glob("signal_*/replica_*/signal_injection/inputs.json"))
                },
            },
        )


def run_scan(args):
    from .cli import run_campaign

    plan = points(args)
    data_root, output_root = args.data.resolve(), args.output.resolve()
    # Validate the entire request before starting any expensive stage.
    for point in plan:
        manifest = validate(data_root / point_name(point) / "signal_injection")
        if manifest.get("injection_scan") != point:
            raise ValueError(
                "Prepared point differs from configured scan; prepare the matching scan first"
            )
    for point in plan:
        options = copy(args)
        options.data = data_root / point_name(point)
        options.output = output_root / point_name(point)
        options.scenarios = ["signal_injection"]
        options.seeds = [point["training_seed"]]
        run_campaign(options)
