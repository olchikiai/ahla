"""Run_Report writer for the ``[train]`` tier.

Writes a single UTF-8 provenance file recording the applied Config, dataset
counts, selected device, skipped labels, and per-interval/final metrics from
``RunStats``. Written at ``Config.run_report_path`` (else
``<output_dir>/run_report.txt``); a write failure raises ``OutputError``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..errors import OutputError

if TYPE_CHECKING:  # no runtime coupling
    from .config import Config
    from .runstats import RunStats

__all__ = ["DEFAULT_REPORT_FILENAME", "resolve_report_path", "write"]

# Default report filename under ``Config.output_dir``.
DEFAULT_REPORT_FILENAME = "run_report.txt"

# Applied Config fields recorded in the report, in a stable, explicit order.
_CONFIG_FIELDS: tuple[str, ...] = (
    # Required paths.
    "word_list_path",
    "font_paths",
    "pretrained_recognizer",
    "output_dir",
    "dtrb_repo_path",
    # Optional paths / provenance.
    "run_report_path",
    "eval_dataset_path",
    # Seed.
    "seed",
    # Charset / device.
    "extra_characters",
    "device",
    # Hyperparameters.
    "val_fraction",
    "num_iterations",
    "batch_size",
    "learning_rate",
    "val_interval",
    "best_checkpoint",
    # Generation / output options.
    "augmentations",
    "overwrite",
    "confidence_output",
    # DTRB network stages.
    "network_transformation",
    "network_feature",
    "network_sequence",
    "network_prediction",
    # Low-resource preset knob.
    "freeze_feature_extraction",
)


def resolve_report_path(cfg: "Config") -> str:
    """Return ``cfg.run_report_path`` if set, else
    ``<cfg.output_dir>/run_report.txt``."""
    if cfg.run_report_path:
        return cfg.run_report_path
    return os.path.join(cfg.output_dir, DEFAULT_REPORT_FILENAME)


def _format_value(value: object) -> str:
    """Render a Config value: sequences as comma-joined, None as ``<unset>``,
    empty string as ``<empty>``, else ``str``."""
    if value is None:
        return "<unset>"
    if isinstance(value, (tuple, list)):
        if not value:
            return "<none>"
        return ", ".join(str(item) for item in value)
    if isinstance(value, str) and value == "":
        return "<empty>"
    return str(value)


def _render(cfg: "Config", stats: "RunStats") -> str:
    """Build the full Run_Report text from the Config and RunStats."""
    lines: list[str] = []

    lines.append("OCR_Trainer Run Report")
    lines.append("=" * 70)
    lines.append("")

    # --- Configuration: every applied Config value. ---
    lines.append("Configuration")
    lines.append("-" * 70)
    for field_name in _CONFIG_FIELDS:
        value = getattr(cfg, field_name)
        lines.append(f"  {field_name}: {_format_value(value)}")

    # Append any extra config_snapshot provenance not covered by a Config field.
    extra_snapshot = {
        key: val
        for key, val in stats.config_snapshot.items()
        if key not in _CONFIG_FIELDS
    }
    if extra_snapshot:
        lines.append("")
        lines.append("  Additional recorded configuration:")
        for key in sorted(extra_snapshot):
            lines.append(f"    {key}: {_format_value(extra_snapshot[key])}")
    lines.append("")

    # --- Dataset / Training / Validation counts. ---
    lines.append("Dataset")
    lines.append("-" * 70)
    lines.append(f"  Dataset sample count: {stats.dataset_sample_count}")
    lines.append(f"  Training sample count: {stats.training_count}")
    lines.append(f"  Validation sample count: {stats.validation_count}")
    if stats.empty_validation_set:
        lines.append(
            "  Note: the Validation_Set is empty; validation metrics are not "
            "meaningful."
        )
    lines.append("")

    # --- Selected Compute_Device. ---
    lines.append("Compute device")
    lines.append("-" * 70)
    lines.append(f"  Selected device: {_format_value(stats.compute_device)}")
    lines.append("")

    # --- Skipped labels with their font paths. ---
    lines.append("Skipped labels")
    lines.append("-" * 70)
    if stats.skipped_labels:
        lines.append(
            f"  {len(stats.skipped_labels)} label(s) skipped (font could not "
            "render them):"
        )
        for label, font_path in stats.skipped_labels:
            lines.append(f"    {label!r}\t{font_path}")
    else:
        lines.append("  None")
    lines.append("")

    # --- Metrics: per-interval and final CER / Word_Accuracy. ---
    lines.append("Metrics")
    lines.append("-" * 70)
    if stats.per_interval_metrics:
        lines.append("  Per-interval CER / Word_Accuracy:")
        for iteration, cer, word_acc in stats.per_interval_metrics:
            lines.append(
                f"    iter {iteration}: CER={cer:.6f} Word_Accuracy={word_acc:.6f}"
            )
    else:
        lines.append("  Per-interval CER / Word_Accuracy: None recorded")
    lines.append(f"  Final CER: {stats.final_cer:.6f}")
    lines.append(f"  Final Word_Accuracy: {stats.final_word_accuracy:.6f}")

    # --- Optional separate eval-dataset metrics, only when present. ---
    if stats.eval_dataset_cer is not None or stats.eval_dataset_word_accuracy is not None:
        lines.append("")
        lines.append("  Separate evaluation dataset:")
        eval_cer = (
            f"{stats.eval_dataset_cer:.6f}"
            if stats.eval_dataset_cer is not None
            else "<unset>"
        )
        eval_word_acc = (
            f"{stats.eval_dataset_word_accuracy:.6f}"
            if stats.eval_dataset_word_accuracy is not None
            else "<unset>"
        )
        lines.append(f"    CER: {eval_cer}")
        lines.append(f"    Word_Accuracy: {eval_word_acc}")
    lines.append("")

    # Trailing newline so the file ends cleanly.
    return "\n".join(lines) + "\n"


def write(cfg: "Config", stats: "RunStats") -> str:
    """Write the UTF-8 Run_Report and return its path.

    Raises OutputError (naming the path) if the directory cannot be created or
    the file cannot be written.
    """
    report_path = resolve_report_path(cfg)

    # An unwritable location surfaces as OSError -> OutputError.
    parent = os.path.dirname(report_path)
    if parent:
        try:
            os.makedirs(parent, exist_ok=True)
        except OSError as exc:
            raise OutputError(
                parent, f"Cannot create run report directory ({exc.strerror or exc})"
            ) from exc

    content = _render(cfg, stats)

    try:
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(content)
    except OSError as exc:
        raise OutputError(
            report_path, f"Cannot write run report ({exc.strerror or exc})"
        ) from exc

    return report_path
