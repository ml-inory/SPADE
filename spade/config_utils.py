"""Helpers for loading YAML configs into typed dataclasses."""

from __future__ import annotations

from dataclasses import fields, is_dataclass
from pathlib import Path
from typing import Any, Type, TypeVar

import yaml

T = TypeVar("T")


def load_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    if not isinstance(data, dict):
        raise ValueError(f"config {path} must contain a YAML mapping")
    return data


def dataclass_from_dict(cls: Type[T], data: dict[str, Any], name: str = "config") -> T:
    """Build a dataclass from a dict, rejecting unknown fields."""
    if not is_dataclass(cls):
        raise TypeError(f"{cls} is not a dataclass")
    known = {f.name for f in fields(cls)}
    unknown = set(data) - known
    if unknown:
        raise ValueError(f"unknown {name} fields: {sorted(unknown)}")
    return cls(**{k: v for k, v in data.items() if k in known})


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    try:
        import torch

        return "cuda" if torch.cuda.is_available() else "cpu"
    except ImportError:  # pragma: no cover
        return "cpu"

