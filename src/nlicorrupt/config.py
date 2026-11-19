"""Loading YAML configs from the repo's configs/ directory."""

from __future__ import annotations

import os
from pathlib import Path

import yaml


def configs_dir() -> Path:
    env = os.environ.get("NLICORRUPT_CONFIGS")
    if env:
        return Path(env)
    return Path(__file__).resolve().parents[2] / "configs"


def load_yaml(name_or_path: str | Path) -> dict:
    p = Path(name_or_path)
    if not p.exists():
        p = configs_dir() / name_or_path
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def model_id(alias: str) -> str:
    """Resolve a short alias from configs/models.yaml to a hub id (ids pass through)."""
    reg = load_yaml("models.yaml")
    for group in reg.values():
        if isinstance(group, dict) and alias in group:
            return group[alias]
    return alias


def split_revision(model: str) -> tuple[str, str | None]:
    """'org/name@revision' -> ('org/name', 'revision'); plain ids and local paths pass through."""
    if "@" in model and not model.startswith(("/", ".", "~")):
        repo, rev = model.split("@", 1)
        return repo, rev or None
    return model, None


def model_group(group: str) -> dict[str, str]:
    return dict(load_yaml("models.yaml").get(group, {}))
