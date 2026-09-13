import json
from pathlib import Path
import subprocess

from .storage import locked, verify_artifacts

COMMIT = "8ead8cd6671b93fc385d8f440c06b8fa870b0be5"
REPOSITORY = "https://github.com/HEPML-AnomalyDetection/CATHODE.git"


def verify(root):
    root = Path(root).resolve()
    if not (root / ".git").exists():
        raise ValueError("Pinned LaCathode checkout is missing; follow the README clone instructions")
    commit = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip()
    if commit != COMMIT:
        raise ValueError("LaCathode commit mismatch; refusing an unpinned checkout")
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain", "--untracked-files=all"], text=True
    )
    if status.strip():
        raise ValueError("LaCathode checkout is modified; use a clean pinned checkout")
    hashes = json.loads(Path(__file__).with_name("source_hashes.json").read_text())
    verify_artifacts(root, hashes, "Verify pinned LaCathode sources")
    return hashes


def setup(root):
    root = Path(root).resolve()
    with locked(root.parent / ".source-setup.lock"):
        return verify(root)
