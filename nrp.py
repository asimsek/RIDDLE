#!/usr/bin/env python3
import argparse
import os
from pathlib import Path
import re
import shlex
import sys

import yaml

from riddle.cli import METHODS, DEFAULT_METHODS, SCENARIOS, seeds, positive
from riddle.storage import atomic_write
from riddle.resume import add_resume_options

ROOT = Path(__file__).resolve().parent
PYTHON = "/opt/conda/bin/python"

# NRP resource key and, for generic GPU resources, the exact node product label.
GPU_TYPES = {
    "a100": ("nvidia.com/a100", None),
    "l40": ("nvidia.com/gpu", "NVIDIA-L40"),
    "l40s": ("nvidia.com/gpu", "NVIDIA-L40S"),
    "l4": ("nvidia.com/gpu", "NVIDIA-L4"),
    "a40": ("nvidia.com/a40", None),
    "rtxa6000": ("nvidia.com/rtxa6000", None),
    "rtx8000": ("nvidia.com/rtx8000", None),
    "rtx3090": ("nvidia.com/gpu", "NVIDIA-GeForce-RTX-3090"),
    "rtx4090": ("nvidia.com/gpu", "NVIDIA-GeForce-RTX-4090"),
    "h100": ("nvidia.com/h100", None),
    "h200": ("nvidia.com/h200", None),
}


def setup_runtime():
    spec = yaml.safe_load((ROOT / "config/nrp/jupyter.yaml").read_text())["spec"]
    image = next(c["image"] for c in spec["containers"] if c["name"] == "jupyter")
    return image, spec.get("imagePullSecrets", []), spec.get("nodeSelector", {}).get("topology.kubernetes.io/region")


def job(args):
    if "lacathode" in args.methods:
        from external.lacathode_utils.pipeline import classifier_settings

        classifier_settings(getattr(args, "runs", None), getattr(args, "epochs", None))
    if "ranode" in args.methods:
        runs, epochs = getattr(args, "runs", None), getattr(args, "epochs", None)
        if runs is not None and not 1 <= runs <= 20:
            raise ValueError("R-ANODE --runs must be between 1 and 20")
        if epochs is not None and epochs < 10:
            raise ValueError("R-ANODE --epochs must be at least 10 for checkpoint ensembling")
    default_image, default_secrets, default_region = setup_runtime()
    image = args.image or default_image
    region = getattr(args, "region", None) or default_region
    if not region or not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", region) or len(region) > 63:
        raise ValueError("Set a valid storage region in config/nrp/jupyter.yaml or pass --region")
    if not re.fullmatch(r"[^\s@]+/[^\s@]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("Provide an immutable registry/image@sha256:<64 hex digits> reference")
    if not re.fullmatch("[a-z0-9]([-a-z0-9]*[a-z0-9])?", args.name) or len(args.name) > 63:
        raise ValueError("Use a Kubernetes-compatible job name of at most 63 characters")
    command = [
        PYTHON,
        "run.py",
        getattr(args, "workflow", "run"),
        "--methods",
        *args.methods,
        "--data",
        str(getattr(args, "data", None) or ("data/injection_scan" if getattr(args, "workflow", "run") == "scan" else "data/lhco")),
        "--output",
        str(getattr(args, "results", None) or ("results/injection_scan" if getattr(args, "workflow", "run") == "scan" else "results")),
        "--device",
        "cuda:0",
        "--config",
        str(getattr(args, "config", "config/settings.yaml")),
        "--workers",
        str(args.workers),
        "--io-workers",
        str(args.io_workers),
        "--mps",
        "auto",
        "--resume",
        "--verbose",
        "1",
    ]
    if getattr(args, "workflow", "run") == "scan":
        for key in ("replicas", "signal_events"):
            if getattr(args, key, None) is not None:
                command.extend(["--" + key.replace("_", "-"), ",".join(map(str, getattr(args, key)))])
    else:
        command.extend(["--scenarios", *args.scenarios, "--seeds", ",".join(map(str, args.seeds))])
    for option in ("resume_across_code_change", "resume_across_device_change"):
        if getattr(args, option, False):
            command.append("--" + option.replace("_", "-"))
    for key in ("runs", "epochs"):
        value = getattr(args, key, None)
        if value is not None:
            command.extend(["--" + key, str(value)])
    if "ranode" in args.methods:
        command.extend(["--ranode-config", str(getattr(args, "ranode_config", "external/ranode_utils/ranode.yaml"))])
    script = "\n".join(
        [
            "set -Eeuo pipefail",
            "cd /shared/work/RIDDLE",
            f"{PYTHON} -m external.ranode_utils.runtime --require-cuda" if args.methods == ["ranode"] else f"{PYTHON} scripts/nrp_runtime.py --require-cuda",
            "nvidia-smi",
            "exec " + shlex.join(command),
        ]
    )
    gpu = getattr(args, "gpu", "a100").lower()
    if gpu not in GPU_TYPES:
        raise ValueError(f"Unknown GPU {gpu!r}; choose from {', '.join(GPU_TYPES)}")
    gpu_resource, gpu_product = GPU_TYPES[gpu]
    resource = {"cpu": "16", "memory": "64Gi", gpu_resource: 1}
    result = {
        "apiVersion": "batch/v1",
        "kind": "Job",
        "metadata": {"name": args.name},
        "spec": {
            "backoffLimit": 4,
            "template": {
                "spec": {
                    "priorityClassName": "owner-no-preempt",
                    "restartPolicy": "Never",
                    "terminationGracePeriodSeconds": 120,
                    "automountServiceAccountToken": False,
                    "nodeSelector": {
                        "kubernetes.io/arch": "amd64",
                        "topology.kubernetes.io/region": region,
                    },
                    "securityContext": {
                        "runAsUser": 1000,
                        "runAsGroup": 100,
                        "fsGroup": 100,
                        "fsGroupChangePolicy": "OnRootMismatch",
                    },
                    "containers": [
                        {
                            "name": "campaign",
                            "image": image,
                            "imagePullPolicy": "IfNotPresent",
                            "command": ["/bin/bash", "-lc"],
                            "args": [script],
                            "env": [
                                {"name": k, "value": v}
                                for k, v in {
                                    "PYTHONUNBUFFERED": "1",
                                    "PYTHONDONTWRITEBYTECODE": "1",
                                    "PYTHONNOUSERSITE": "1",
                                    "MPLCONFIGDIR": f"/shared/work/RIDDLE/.cache/{args.name}/matplotlib",
                                    "XDG_CACHE_HOME": f"/shared/work/RIDDLE/.cache/{args.name}",
                                }.items()
                            ],
                            "resources": {"requests": resource, "limits": resource},
                            "volumeMounts": [
                                {"name": "shared", "mountPath": "/shared"},
                            ],
                        }
                    ],
                    "volumes": [
                        {"name": "shared", "persistentVolumeClaim": {"claimName": "riddle-shared"}},
                    ],
                }
            },
        },
    }
    if gpu_product is not None:
        result["spec"]["template"]["spec"]["nodeSelector"]["nvidia.com/gpu.product"] = gpu_product
    secrets = [dict(item) for item in default_secrets]
    for name in getattr(args, "image_pull_secret", None) or []:
        if not re.fullmatch(r"[a-z0-9]([-a-z0-9.]*[a-z0-9])?", name) or len(name) > 253:
            raise ValueError("Use a Kubernetes-compatible image-pull secret name")
        if {"name": name} not in secrets:
            secrets.append({"name": name})
    if secrets:
        result["spec"]["template"]["spec"]["imagePullSecrets"] = secrets
    return result


def pin_image(image):
    if not re.fullmatch(r"[^\s@]+/[^\s@]+@sha256:[0-9a-f]{64}", image):
        raise ValueError("Provide an immutable registry/image@sha256:<64 hex digits> reference")
    path = ROOT / "config/nrp/jupyter.yaml"
    if path.is_symlink():
        raise ValueError("Refusing to replace a symbolic-link runtime configuration")
    metadata = path.stat()
    original = path.read_text()
    spec = yaml.safe_load(original)["spec"]
    containers = {c["name"]: c for key in ("containers", "initContainers") for c in spec.get(key, [])}
    names = ("jupyter", "initialize-shared-storage")
    if any(name not in containers for name in names):
        raise ValueError("Unrecognized Jupyter runtime containers; refusing to change the configuration")
    updated = original
    for previous in {containers[name]["image"] for name in names}:
        updated = updated.replace("image: " + previous, "image: " + image)
    actual = yaml.safe_load(updated)["spec"]
    expected = yaml.safe_load(original)["spec"]
    for key in ("containers", "initContainers"):
        for container in expected.get(key, []):
            if container["name"] in names:
                container["image"] = image
    if actual != expected:
        raise ValueError("Image update would change unrelated configuration")
    def write(temporary):
        temporary.write_text(updated)
        os.chmod(temporary, metadata.st_mode & 0o777)
        if os.geteuid() == 0:
            os.chown(temporary, metadata.st_uid, metadata.st_gid)
    atomic_write(path, write)
    print(f"[PASS] Pinned Jupyter and batch runtime to {image}")


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate an NRP GPU job or pin its runtime image; does not submit jobs")
    action = p.add_mutually_exclusive_group(required=True)
    action.add_argument("--name")
    action.add_argument("--pin-image", help="Save a built image digest in config/nrp/jupyter.yaml")
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(DEFAULT_METHODS))
    p.add_argument("--workflow", choices=("run", "scan"), default="run")
    p.add_argument("--replicas", type=seeds, help="Scan replica indices, e.g. 0-9")
    p.add_argument("--signal-events", type=seeds, help="Scan total signal counts, e.g. 1000,667")
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["signal_injection"])
    p.add_argument("--seeds", type=seeds, default=[42])
    p.add_argument("--config", default="config/settings.yaml", help="Settings path inside the job")
    p.add_argument("--ranode-config", default="external/ranode_utils/ranode.yaml", help="Independent upstream R-ANODE settings")
    p.add_argument("--data", help="Prepared dataset path; defaults to data/lhco or data/injection_scan for scans")
    p.add_argument("--results", help="Result directory; defaults to results or results/injection_scan for scans")
    p.add_argument("--runs", type=positive, help="Override RIDDLE/R-ANODE fit count and LaCathode classifier fit count")
    p.add_argument("--epochs", type=positive, help="Override RIDDLE/R-ANODE signal-fit and LaCathode classifier epochs; background stages are unchanged")
    p.add_argument("--workers", type=positive, default=2)
    p.add_argument("--io-workers", type=positive, default=4)
    p.add_argument(
        "--gpu", type=str.lower, choices=GPU_TYPES, default="a100",
        help="Request one GPU of this type (case-insensitive; default: a100); namespace access is still required",
    )
    add_resume_options(p, always_resume=True)
    p.add_argument("--region", help="Storage region; defaults to the Jupyter node selector")
    p.add_argument(
        "--image", help="Override the pinned runtime image from config/nrp/jupyter.yaml"
    )
    p.add_argument("--image-pull-secret", action="append", help="Existing namespace secret for a private image")
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    try:
        if args.pin_image is not None:
            pin_image(args.pin_image)
            return
        text = yaml.safe_dump(job(args), sort_keys=False)
        if args.output:
            if args.output.exists():
                raise FileExistsError("Job definition exists; choose a new filename")
            atomic_write(args.output, lambda path: path.write_text(text))
        else:
            sys.stdout.write(text)
    except (ValueError, OSError) as error:
        p.exit(1, f"[ERROR] {error}\n")


if __name__ == "__main__":
    main()
