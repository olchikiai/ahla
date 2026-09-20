"""Pipeline orchestration for the ``[train]`` tier.

Exposes three ``(cfg) -> int`` entry points that catch the fatal
``OcrTrainerError`` family, print it to stderr, and return its ``exit_code``:
:func:`run` (full pipeline), :func:`run_data_prep`, and :func:`run_training`.

The data-prep and training phases can run as separate processes; they decouple
through a persisted ``val_manifest.txt`` (both must use the same
``cfg.output_dir`` and matching Config, data-prep first).

Finalization produces the Base_Weights ``.pth``. There is no EasyOCR artifact
and no inference-based validation-set eval here — that moved to the ONNX harness
(which reuses :mod:`.evaluator`).
"""

from __future__ import annotations

import os
import sys

from . import evaluator, lmdbbuilder, partition, runreport, synthgen
from ..charset import build_charset
from .config import Config, validate_config
from .device import select_device
from ..errors import OcrTrainerError, OutputError
from .runstats import RunStats
from .synthgen import Sample
from .trainer import train

__all__ = ["run", "run_data_prep", "run_training"]

# Per-partition LMDB subdirs under ``cfg.output_dir``.
_TRAIN_LMDB_SUBDIR = "lmdb/train"
_VAL_LMDB_SUBDIR = "lmdb/val"

# Persisted Validation_Set manifest that decouples the two phases.
_VAL_MANIFEST_SUBDIR = "val_manifest.txt"


def run(cfg: Config) -> int:
    """Run the full pipeline; return 0, or the error's exit_code on failure."""
    try:
        _run(cfg)
    except OcrTrainerError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    return 0


def run_data_prep(cfg: Config) -> int:
    """Run ONLY the data-prep phase (generate, partition, build LMDBs, persist
    the val manifest); return 0 or the error's exit_code."""
    try:
        _run_data_prep(cfg)
    except OcrTrainerError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    return 0


def run_training(cfg: Config) -> int:
    """Run ONLY the training/finalization phase (reads the persisted val
    manifest from a prior data-prep run); return 0 or the error's exit_code."""
    try:
        _run_training(cfg)
    except OcrTrainerError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    return 0


# --- pipeline -------------------------------------------------------------


def _init_run(cfg: Config) -> tuple[RunStats, "object", str]:
    """Shared preamble: validate, build the Charset, and select the device.

    Returns ``(stats, charset, device)``. Raises DeviceError when CUDA is
    requested but unavailable.
    """
    validate_config(cfg)

    stats = RunStats()
    stats.config_snapshot = _config_snapshot(cfg)

    charset = build_charset(cfg.extra_characters)
    stats.charset_snapshot = charset.snapshot()

    device = select_device(cfg.device)
    stats.compute_device = device

    return stats, charset, device


def _run(cfg: Config) -> None:
    """Execute the full pipeline in order; may raise ``OcrTrainerError``."""
    stats, charset, device = _init_run(cfg)
    train_lmdb, val_lmdb, validation = _prepare_data(cfg, charset, device, stats)
    _train_and_finalize(cfg, charset, device, train_lmdb, val_lmdb, validation, stats)


def _run_data_prep(cfg: Config) -> None:
    """Execute ONLY the data-prep phase; may raise ``OcrTrainerError``."""
    stats, charset, device = _init_run(cfg)

    _train_lmdb, _val_lmdb, validation = _prepare_data(cfg, charset, device, stats)

    # Persist the Validation_Set for a later run_training process.
    _write_val_manifest(cfg, validation)


def _run_training(cfg: Config) -> None:
    """Execute ONLY the training/finalization phase; may raise ``OcrTrainerError``."""
    stats, charset, device = _init_run(cfg)

    # Reconstruct the per-partition LMDB paths as _prepare_data derives them.
    train_lmdb = os.path.join(cfg.output_dir, _TRAIN_LMDB_SUBDIR)
    val_lmdb = os.path.join(cfg.output_dir, _VAL_LMDB_SUBDIR)

    # Read the Validation_Set back from the manifest written by data-prep.
    validation = _read_val_manifest(cfg)

    stats.validation_count = len(validation)
    stats.empty_validation_set = len(validation) == 0

    _train_and_finalize(cfg, charset, device, train_lmdb, val_lmdb, validation, stats)


# --- validation manifest (phase decoupling) -------------------------------


def _write_val_manifest(cfg: Config, validation: list[Sample]) -> None:
    """Write the Validation_Set to ``val_manifest.txt`` as UTF-8
    ``image_path<TAB>label`` lines. Counterpart of :func:`_read_val_manifest`."""
    manifest_path = os.path.join(cfg.output_dir, _VAL_MANIFEST_SUBDIR)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        for sample in validation:
            handle.write(f"{sample.image_path}\t{sample.label}\n")


def _read_val_manifest(cfg: Config) -> list[Sample]:
    """Read the Validation_Set back from ``val_manifest.txt`` (Samples get
    ``font_path=""``). Raises OutputError when the manifest is missing."""
    manifest_path = os.path.join(cfg.output_dir, _VAL_MANIFEST_SUBDIR)
    if not os.path.isfile(manifest_path):
        raise OutputError(
            manifest_path, "validation manifest not found; run data preparation first"
        )

    samples: list[Sample] = []
    with open(manifest_path, encoding="utf-8") as handle:
        for line in handle:
            line = line.rstrip("\n")
            if not line:
                continue
            image_path, _, label = line.partition("\t")
            samples.append(Sample(image_path=image_path, label=label, font_path=""))
    return samples


def _prepare_data(cfg: Config, charset, device, stats) -> tuple[str, str, list[Sample]]:
    """DATA-PREP phase: generate, partition, and build the per-partition LMDBs.

    Mutates ``stats`` with the partition counts and returns
    ``(train_lmdb, val_lmdb, validation)``.
    """
    samples = synthgen.generate(cfg, charset, stats)

    # Seeded, reproducible train/val split.
    seed = cfg.seed if cfg.seed is not None else 0
    training, validation = partition.partition(samples, cfg.val_fraction, seed)
    stats.training_count = len(training)
    stats.validation_count = len(validation)
    stats.empty_validation_set = len(validation) == 0

    # Build each partition's LMDB via DTRB (raises DtrbError on non-zero exit).
    synthetic_dir = os.path.join(cfg.output_dir, synthgen._WORK_SUBDIR)
    gt_file = os.path.join(synthetic_dir, synthgen._GT_FILENAME)
    train_lmdb = os.path.join(cfg.output_dir, _TRAIN_LMDB_SUBDIR)
    val_lmdb = os.path.join(cfg.output_dir, _VAL_LMDB_SUBDIR)

    lmdbbuilder.build_lmdb(synthetic_dir, gt_file, train_lmdb, cfg.dtrb_repo_path)
    lmdbbuilder.build_lmdb(synthetic_dir, gt_file, val_lmdb, cfg.dtrb_repo_path)

    return train_lmdb, val_lmdb, validation


def _train_and_finalize(
    cfg: Config,
    charset,
    device,
    train_lmdb: str,
    val_lmdb: str,
    validation: list[Sample],
    stats,
) -> None:
    """TRAIN/FINALIZE phase: fine-tune (produce the Base_Weights .pth), then
    write the Run_Report.

    No EasyOCR artifact and no inference-based validation-set eval here — that
    moved to the ONNX harness. The empty-set metrics are recorded so the report
    has a well-defined (0.0) baseline.
    """
    # Fine-tune via DTRB train.py; returns the selected Base_Weights .pth. Raises
    # PretrainedModelError (unloadable --saved_model) or DtrbError.
    checkpoint_path = train(cfg, charset, train_lmdb, val_lmdb, device, stats)

    # Record the .pth path for provenance / the export bridge.
    stats.config_snapshot = {
        **stats.config_snapshot,
        "base_weights_checkpoint": checkpoint_path,
    }

    # Record the empty-input baseline; real scoring is the ONNX harness's job.
    val_result = evaluator.evaluate([], [])
    stats.final_cer = val_result.cer
    stats.final_word_accuracy = val_result.word_accuracy

    # Write the Run_Report (raises OutputError when it cannot be written).
    runreport.write(cfg, stats)


def _config_snapshot(cfg: Config) -> dict:
    """Return a snapshot of every applied Config value (built explicitly)."""
    return {
        "word_list_path": cfg.word_list_path,
        "font_paths": cfg.font_paths,
        "pretrained_recognizer": cfg.pretrained_recognizer,
        "output_dir": cfg.output_dir,
        "dtrb_repo_path": cfg.dtrb_repo_path,
        "run_report_path": cfg.run_report_path,
        "extra_characters": cfg.extra_characters,
        "val_fraction": cfg.val_fraction,
        "num_iterations": cfg.num_iterations,
        "batch_size": cfg.batch_size,
        "learning_rate": cfg.learning_rate,
        "val_interval": cfg.val_interval,
        "best_checkpoint": cfg.best_checkpoint,
        "seed": cfg.seed,
        "device": cfg.device,
        "augmentations": cfg.augmentations,
        "eval_dataset_path": cfg.eval_dataset_path,
        "overwrite": cfg.overwrite,
        "confidence_output": cfg.confidence_output,
        "network_transformation": cfg.network_transformation,
        "network_feature": cfg.network_feature,
        "network_sequence": cfg.network_sequence,
        "network_prediction": cfg.network_prediction,
        "freeze_feature_extraction": cfg.freeze_feature_extraction,
    }
