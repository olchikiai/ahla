"""Run_Report writer for the ``[train]`` tier.

Writes a single human-readable UTF-8 text file recording everything needed to
understand and reproduce a fine-tuning run. The report is the run's provenance
record; it captures the applied Configuration and the counters, metrics, and
non-fatal notes accumulated in :class:`~olchiki_ocr._train.runstats.RunStats` as
the workflow executed.

The report records, in order:

* **Configuration** -- every applied Config value: all paths, the seed, and all
  hyperparameters. Values are read from the validated
  :class:`~olchiki_ocr._train.config.Config` (the authoritative source of the
  applied settings); any additional provenance the workflow chose to stash in
  ``RunStats.config_snapshot`` is appended so nothing recorded there is lost.
* **Dataset counts** -- the Dataset / Training_Set / Validation_Set Sample
  counts, plus the empty-Validation_Set note when applicable.
* **Compute device** -- the selected Compute_Device.
* **Skipped labels** -- each Label skipped during generation together with the
  Font_File that could not render it, as ``(label, font_path)`` pairs.
* **Metrics** -- per-interval CER / Word_Accuracy recorded during training, the
  final CER / Word_Accuracy, and the optional separate-eval-dataset CER /
  Word_Accuracy when present.

Location: the report is written at ``Config.run_report_path`` when set, else at
``<Config.output_dir>/run_report.txt``. Any failure to write (unwritable
directory, write error) raises :class:`~olchiki_ocr.errors.OutputError` naming
the offending path, consistent with the rest of the package.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from ..errors import OutputError

if TYPE_CHECKING:  # annotations are strings under ``from __future__``; no runtime coupling
    from .config import Config
    from .runstats import RunStats

__all__ = ["DEFAULT_REPORT_FILENAME", "resolve_report_path", "write"]

# Default report filename, placed under ``Config.output_dir`` when the
# Configuration does not set an explicit ``run_report_path``.
DEFAULT_REPORT_FILENAME = "run_report.txt"

# The applied Config fields recorded in the report, in a stable, readable order:
# the required paths first, then the seed, then every hyperparameter and option.
# Kept explicit (rather than introspected) so the report layout is deterministic
# and every value is intentionally accounted for.
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
    # Low-resource preset knob (Req 10.3).
    "freeze_feature_extraction",
)


def resolve_report_path(cfg: "Config") -> str:
    """Return the path the Run_Report is written to.

    Uses ``cfg.run_report_path`` when it is set, otherwise defaults to
    ``<cfg.output_dir>/run_report.txt``.
    """
    if cfg.run_report_path:
        return cfg.run_report_path
    return os.path.join(cfg.output_dir, DEFAULT_REPORT_FILENAME)


def _format_value(value: object) -> str:
    """Render a Config value for the report.

    Sequences (``font_paths``, ``augmentations``) are rendered as a
    comma-separated list; ``None`` is rendered as ``<unset>``; empty strings are
    rendered as ``<empty>``; everything else uses ``str``.
    """
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

    # Append any extra provenance stashed in RunStats.config_snapshot that is not
    # already covered by a Config field, so nothing recorded there is lost.
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
        # Empty Validation_Set note: metrics on an empty set are 0.0.
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
    """Write the single UTF-8 Run_Report file and return its path.

    Records every applied Config value, the Dataset / Training_Set /
    Validation_Set counts, the skipped-label list with font paths, the selected
    Compute_Device, the per-interval and final CER / Word_Accuracy, the optional
    separate-eval-dataset metrics when present, and the empty-Validation_Set note
    when applicable.

    Args:
        cfg: The validated run Configuration; supplies every applied setting and
            (via :func:`resolve_report_path`) the report location.
        stats: The accumulated :class:`~olchiki_ocr._train.runstats.RunStats` for
            the run, supplying counts, metrics, the device, and skipped labels.

    Returns:
        The path the report was written to.

    Raises:
        OutputError: If the report directory cannot be created or the file
            cannot be written; the message names the offending path.
    """
    report_path = resolve_report_path(cfg)

    # Ensure the parent directory exists; an unwritable location surfaces as an
    # OSError -> OutputError naming the path.
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
