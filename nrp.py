#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import shlex
import sys

import yaml

from riddle.cli import METHODS, SCENARIOS, seeds, positive
from riddle.storage import atomic_write

ROOT = Path(__file__).resolve().parent
PYTHON = "/opt/conda/bin/python"


def setup_runtime():
    spec = yaml.safe_load((ROOT / "config/nrp/jupyter.yaml").read_text())["spec"]
    image = next(c["image"] for c in spec["containers"] if c["name"] == "jupyter")
    return image, spec.get("imagePullSecrets", []), spec.get("nodeSelector", {}).get("topology.kubernetes.io/region")


def job(args):
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
        "run",
        "--methods",
        *args.methods,
        "--scenarios",
        *args.scenarios,
        "--seeds",
        ",".join(map(str, args.seeds)),
        "--data",
        str(getattr(args, "data", "data/lhco")),
        "--output",
        str(getattr(args, "results", "results")),
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
    for key in ("runs", "epochs"):
        value = getattr(args, key, None)
        if value is not None:
            command.extend(["--" + key, str(value)])
    script = "\n".join(
        [
            "set -Eeuo pipefail",
            "cd /shared/work/RIDDLE",
            f"{PYTHON} scripts/nrp_runtime.py --require-cuda",
            "nvidia-smi",
            "exec " + shlex.join(command),
        ]
    )
    resource = {"cpu": "16", "memory": "64Gi", "nvidia.com/a100": 1}
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
    secrets = [dict(item) for item in default_secrets]
    for name in getattr(args, "image_pull_secret", None) or []:
        if not re.fullmatch(r"[a-z0-9]([-a-z0-9.]*[a-z0-9])?", name) or len(name) > 253:
            raise ValueError("Use a Kubernetes-compatible image-pull secret name")
        if {"name": name} not in secrets:
            secrets.append({"name": name})
    if secrets:
        result["spec"]["template"]["spec"]["imagePullSecrets"] = secrets
    return result


def main(argv=None):
    p = argparse.ArgumentParser(description="Generate an NRP GPU job; does not submit it")
    p.add_argument("--name", required=True)
    p.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    p.add_argument("--scenarios", nargs="+", choices=SCENARIOS, default=["signal_injection"])
    p.add_argument("--seeds", type=seeds, default=[42])
    p.add_argument("--config", default="config/settings.yaml", help="Settings path inside the job")
    p.add_argument("--data", default="data/lhco", help="Prepared dataset path inside the job")
    p.add_argument("--results", default="results", help="Result directory inside the job")
    p.add_argument("--runs", type=positive, help="Override YAML fit count")
    p.add_argument("--epochs", type=positive, help="Override YAML epochs")
    p.add_argument("--workers", type=positive, default=2)
    p.add_argument("--io-workers", type=positive, default=4)
    p.add_argument("--region", help="Storage region; defaults to the Jupyter node selector")
    p.add_argument(
        "--image", help="Override the pinned runtime image from config/nrp/jupyter.yaml"
    )
    p.add_argument("--image-pull-secret", action="append", help="Existing namespace secret for a private image")
    p.add_argument("--output", type=Path)
    args = p.parse_args(argv)
    try:
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
