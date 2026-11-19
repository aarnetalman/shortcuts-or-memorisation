"""Run manifests: what was run, with which code, packages, data and hardware."""

from __future__ import annotations

import json
import platform
import socket
import subprocess
import time
from importlib import metadata
from pathlib import Path

PACKAGES = ("torch", "transformers", "accelerate", "datasets", "nltk", "scikit-learn", "numpy", "nlicorrupt")


def git_sha() -> str | None:
    repo = Path(__file__).resolve().parents[2]
    try:
        sha = subprocess.run(["git", "-C", str(repo), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
        dirty = subprocess.run(["git", "-C", str(repo), "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True)
        return sha.stdout.strip() + ("-dirty" if dirty.stdout.strip() else "")
    except (OSError, subprocess.CalledProcessError):
        return None


def versions() -> dict[str, str | None]:
    out = {}
    for p in PACKAGES:
        try:
            out[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            out[p] = None
    return out


def hardware() -> dict:
    info = {"host": socket.gethostname(), "machine": platform.machine(), "python": platform.python_version()}
    try:
        import torch

        if torch.cuda.is_available():
            info["gpu"] = torch.cuda.get_device_name(0)
            info["gpu_count"] = torch.cuda.device_count()
    except ImportError:
        pass
    return info


def manifest(args: dict, extra: dict | None = None) -> dict:
    from .tasks import ANLI_REV, FEVER_NLI_REV, GLUE_REV, SNLI_REV

    m = {
        "time": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "args": args,
        "git": git_sha(),
        "versions": versions(),
        "hardware": hardware(),
        "dataset_revisions": {"glue": GLUE_REV, "anli": ANLI_REV, "snli": SNLI_REV, "fever_nli": FEVER_NLI_REV},
    }
    if extra:
        m.update(extra)
    return m


def write_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")
    tmp.replace(path)
