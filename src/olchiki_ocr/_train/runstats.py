"""Run-statistics accumulator for the ``[train]`` tier.

Defines the mutable :class:`RunStats` dataclass that accumulates counters,
metrics, and provenance for the Run_Report as a fine-tuning run executes.

This lived on the carried-over ``errors`` module in the old package. In the new
package the core :mod:`olchiki_ocr.errors` is kept lean (typed errors only, so
it stays a pure-stdlib eager import for the torch-free core), and this
training-only accumulator lives here in the ``[train]`` tier instead.
"""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["RunStats"]


@dataclass
class RunStats:
    """Mutable accumulator of counters, metrics, and provenance for a run.

    Populated as the workflow executes and consumed by the Run_Report writer.
    ``skipped_labels`` records non-fatal generation issues as ``(label,
    font_path)`` pairs. ``config_snapshot`` and ``charset_snapshot`` capture the
    applied Configuration and the ordered Charset for reproducibility.
    """

    dataset_sample_count: int = 0
    training_count: int = 0
    validation_count: int = 0
    skipped_labels: list[tuple[str, str]] = field(default_factory=list)
    compute_device: str = ""
    per_interval_metrics: list[tuple[int, float, float]] = field(default_factory=list)
    final_cer: float = 0.0
    final_word_accuracy: float = 0.0
    eval_dataset_cer: float | None = None
    eval_dataset_word_accuracy: float | None = None
    empty_validation_set: bool = False
    config_snapshot: dict = field(default_factory=dict)
    charset_snapshot: list[tuple[int, str]] = field(default_factory=list)
