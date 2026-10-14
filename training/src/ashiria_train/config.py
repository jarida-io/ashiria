"""Config loading. Relative paths in the config resolve against the training/ folder."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

TRAINING_DIR = Path(__file__).resolve().parents[2]


class Config(dict):
    """A dict whose nested sections can also be read as attributes (cfg.train.lr)."""

    def __getattr__(self, name: str) -> Any:
        try:
            value = self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc
        return Config(value) if isinstance(value, dict) else value


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for key, value in override.items():
        out[key] = _merge(base[key], value) if isinstance(value, dict) and isinstance(base.get(key), dict) else value
    return out


def load_config(path: str | Path) -> Config:
    """Load a YAML config. `extends: other.yaml` loads that file first and overrides it."""
    file = resolve(path)
    with open(file) as f:
        raw = yaml.safe_load(f) or {}
    parent = raw.pop("extends", None)
    if parent:
        raw = _merge(load_config(file.parent / parent), raw)
    return Config(raw)


def resolve(p: str | Path) -> Path:
    p = Path(p).expanduser()
    if p.is_absolute():
        return p
    cwd = Path.cwd() / p
    return cwd if cwd.exists() else TRAINING_DIR / p


def training_relative(p: str | Path) -> Path:
    """Absolute paths stay as they are; relative ones are relative to the training/ folder."""
    p = Path(p).expanduser()
    return p if p.is_absolute() else TRAINING_DIR / p


def path(cfg: Config, key: str) -> Path:
    return training_relative(cfg.paths[key])
