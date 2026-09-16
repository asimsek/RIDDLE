import hashlib
import subprocess
from pathlib import Path

COMMIT = "d6deed7cb949eb4483c6f484b96e8e4ff2133e25"
REPOSITORY = "https://github.com/rd804/R-ANODE.git"


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def verify(root):
    root = Path(root).resolve()

    def git(*args):
        return subprocess.check_output(
            ["git", "-C", str(root), *args], text=True
        ).strip()

    if not (root / ".git").exists() or git("rev-parse", "HEAD") != COMMIT:
        raise ValueError(
            "Pinned R-ANODE checkout is missing or mismatched; follow the README checkout commands"
        )
    if git("status", "--porcelain", "--untracked-files=all"):
        raise ValueError("R-ANODE checkout is modified; use a clean pinned checkout")
    names = git("ls-tree", "-r", "--name-only", COMMIT).splitlines()
    for folder in ("src", "scripts"):
        if any(
            str(p.relative_to(root)) not in names for p in (root / folder).rglob("*.py")
        ):
            raise ValueError("Unexpected Python file in R-ANODE source tree")
    result = {}
    for name in names:
        path = root / name
        if path.is_symlink() or not path.is_file():
            raise ValueError("Invalid R-ANODE source file: " + name)
        result[name] = digest(path)
    return result
