#!/usr/bin/env python3
import argparse
from pathlib import Path
import re
import shlex
import sys

import yaml

from riddle.cli import METHODS, SCENARIOS, seeds, positive
from riddle.storage import atomic_write


def job(args):
    if not args.image or not re.fullmatch(r"[^\s@]+/[^\s@]+@sha256:[0-9a-f]{64}", args.image):
        raise ValueError("Provide an immutable registry/image@sha256:<64 hex digits> reference")
    if not re.fullmatch("[a-z0-9]([-a-z0-9]*[a-z0-9])?", args.name) or len(args.name) > 63:
        raise ValueError("Use a Kubernetes-compatible job name of at most 63 characters")
    command = [
        "python",
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
            "source .venv/bin/activate",
            "python scripts/nrp_runtime.py",
            "nvidia-smi",
            "exec " + shlex.join(command),
        ]
    )
    resource = {"cpu": "16", "memory": "64Gi", "nvidia.com/a100": 1}
    return {
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
                    "nodeSelector": {"kubernetes.io/arch": "amd64"},
                    "securityContext": {
                        "runAsUser": 1000,
                        "runAsGroup": 100,
                        "fsGroup": 100,
                        "fsGroupChangePolicy": "OnRootMismatch",
                    },
                    "containers": [
                        {
                            "name": "campaign",
                            "image": args.image,
                            "command": ["/bin/bash", "-lc"],
                            "args": [script],
                            "env": [
                                {"name": k, "value": v}
                                for k, v in {
                                    "PYTHONUNBUFFERED": "1",
                                    "PYTHONDONTWRITEBYTECODE": "1",
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
    p.add_argument(
        "--image", required=True, help="Use the same immutable image digest as the environment setup pod"
    )
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
