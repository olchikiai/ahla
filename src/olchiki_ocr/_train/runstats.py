"""Mutable ``RunStats`` accumulator: counters, metrics, and provenance the
Run_Report consumes as a fine-tuning run executes."""

from __future__ import annotations

from dataclasses import dataclass, field

__all__ = ["RunStats"]


@dataclass
class RunStats:
    """Mutable run accumulator consumed by the Run_Report writer.

    ``skipped_labels`` holds ``(label, font_path)`` pairs; ``config_snapshot``
    and ``charset_snapshot`` capture provenance for reproducibility.
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
