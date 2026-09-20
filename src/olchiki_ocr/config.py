"""Lightweight, torch-free inference-time configuration for the core."""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["InferenceConfig"]


@dataclass(frozen=True)
class InferenceConfig:
    """Immutable inference-time configuration for :class:`ModelRecognizer`."""

    model_source: str | None = None
    cache_dir: str | None = None
    device: str | None = None
    decoder: str = "greedy"
