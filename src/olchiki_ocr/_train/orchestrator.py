"""Pipeline orchestration for the ``[train]`` tier (Req 10).

This module exposes three public entry points, each a module-level
``(cfg) -> int`` that catches the fatal
:class:`~olchiki_ocr.errors.OcrTrainerError` family at the top level, prints the
descriptive message to stderr, and returns the error's ``exit_code`` (or ``0``
on success):

* :func:`run` -- the all-in-one pipeline: validate, generate, partition, build
  LMDBs, train, evaluate, and write the Run_Report in a single process.
* :func:`run_data_prep` -- ONLY the data-preparation phase.
* :func:`run_training` -- ONLY the training/finalization phase.

Migration note (ONNX redesign)
------------------------------
In the previous EasyOCR-coupled package the training/finalization phase also
wrote an EasyOCR-loadable Model_Artifact and evaluated the Validation_Set by
running that artifact through the EasyOCR inferencer. In the ONNX redesign the
``_train`` tier's job ends at producing the trained Base_Weights ``.pth``
(Req 10.4): the EasyOCR artifact writer and the EasyOCR inferencer are gone
(Module Migration Plan: ``artifact.py`` dropped, ``inferencer.py`` deleted).
Conversion to a servable ONNX model belongs to the ``[export]`` tier (Task 16),
and full-validation-set accuracy scoring against the baseline belongs to the
ONNX Full_Validation_Harness (Task 17), which reuses :mod:`.evaluator`.

Accordingly, :func:`_train_and_finalize` trains (producing the ``.pth``), records
the selected checkpoint path in ``stats``, and writes the Run_Report. It does not
write an EasyOCR artifact and does not run inference-based validation-set
evaluation (there is no torch-free inference path in ``_train``). The reusable
:mod:`.evaluator` is still available to the harness for ONNX-path scoring.

Phase decoupling via a persisted validation manifest
----------------------------------------------------
:func:`run_data_prep` and :func:`run_training` are designed to run as SEPARATE
PROCESSES; they are decoupled through a small ``val_manifest.txt`` on disk that
data-prep writes and training reads. Both phases must be run against the SAME
``cfg.output_dir`` (and matching Config), and data-prep must run before training.

Stage order:

    validate_config
      -> build Charset            (charset.build_charset)
      -> select Compute_Device    (device.select_device)  [torch, D9]
      -> DATA-PREP phase          (_prepare_data)          -> train/ + val/ LMDB
      -> TRAIN/FINALIZE phase     (_train_and_finalize)    -> Base_Weights .pth + Run_Report

The Charset is built before generation because it defines the label alphabet and
the network's output-class count. Device selection is resolved early and recorded.
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

# Sub-directory names under ``cfg.output_dir`` for the per-partition LMDB datasets.
_TRAIN_LMDB_SUBDIR = "lmdb/train"
_VAL_LMDB_SUBDIR = "lmdb/val"

# File name (under ``cfg.output_dir``) of the persisted Validation_Set manifest
# that decouples the data-prep and training phases when run as separate
# processes. Data-prep writes it; training reads it back.
_VAL_MANIFEST_SUBDIR = "val_manifest.txt"


def run(cfg: Config) -> int:
    """Run the full fine-tuning pipeline for ``cfg`` and return the exit code.

    Returns ``0`` on success. Catches the fatal ``OcrTrainerError`` family at the
    top level, prints the descriptive message to stderr, and returns the error's
    distinct non-zero ``exit_code`` (incl. ``DtrbError`` on a non-zero DTRB exit,
    Req 10.6).
    """
    try:
        _run(cfg)
    except OcrTrainerError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    return 0


def run_data_prep(cfg: Config) -> int:
    """Run ONLY the data-preparation phase for ``cfg`` and return the exit code.

    Validates the Configuration, generates the synthetic Dataset, partitions it,
    builds the per-partition train/val LMDB datasets, and persists the
    Validation_Set to ``val_manifest.txt`` under ``cfg.output_dir``. Does NOT
    train or write the Run_Report.
    """
    try:
        _run_data_prep(cfg)
    except OcrTrainerError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    return 0


def run_training(cfg: Config) -> int:
    """Run ONLY the training/finalization phase for ``cfg`` and return the exit code.

    Validates the Configuration, reconstructs the train/val LMDB paths produced
    by a prior :func:`run_data_prep` run, reads the persisted Validation_Set back
    from ``val_manifest.txt``, then fine-tunes and writes the Run_Report. Must be
    run against the same ``cfg.output_dir`` (and matching Config) as the data-prep
    phase, and only after it.
    """
    try:
        _run_training(cfg)
    except OcrTrainerError as exc:
        print(str(exc), file=sys.stderr)
        return exc.exit_code
    return 0


# --- pipeline -------------------------------------------------------------


def _init_run(cfg: Config) -> tuple[RunStats, "object", str]:
    """Run the shared preamble common to every ``run*`` entry point.

      1. Validate the Configuration before any Sample is produced.
      2. Own a fresh :class:`RunStats` accumulator and snapshot the applied
         Configuration for the Run_Report.
      3. Build the recognition Charset and record its snapshot.
      4. Resolve the Compute_Device early and record it; raises DeviceError when
         CUDA is requested but unavailable.

    Returns ``(stats, charset, device)``.
    """
    # 1. Validate before any Sample is produced.
    validate_config(cfg)

    # 2. RunStats accumulator + applied-Config snapshot.
    stats = RunStats()
    stats.config_snapshot = _config_snapshot(cfg)

    # 3. Build the recognition Charset and record its snapshot.
    charset = build_charset(cfg.extra_characters)
    stats.charset_snapshot = charset.snapshot()

    # 4. Resolve the Compute_Device early and record it (torch-based, D9).
    device = select_device(cfg.device)
    stats.compute_device = device

    return stats, charset, device


def _run(cfg: Config) -> None:
    """Execute the full pipeline stages in order; may raise ``OcrTrainerError``."""
    stats, charset, device = _init_run(cfg)

    # DATA-PREP phase: generate + partition + build the per-partition LMDBs.
    train_lmdb, val_lmdb, validation = _prepare_data(cfg, charset, device, stats)

    # TRAIN/FINALIZE phase: fine-tune (produce Base_Weights .pth) + Run_Report.
    _train_and_finalize(cfg, charset, device, train_lmdb, val_lmdb, validation, stats)


def _run_data_prep(cfg: Config) -> None:
    """Execute ONLY the data-preparation phase; may raise ``OcrTrainerError``."""
    stats, charset, device = _init_run(cfg)

    _train_lmdb, _val_lmdb, validation = _prepare_data(cfg, charset, device, stats)

    # Persist the Validation_Set so a later, separate run_training process can
    # read it back.
    _write_val_manifest(cfg, validation)


def _run_training(cfg: Config) -> None:
    """Execute ONLY the training/finalization phase; may raise ``OcrTrainerError``."""
    stats, charset, device = _init_run(cfg)

    # Reconstruct the per-partition LMDB paths exactly as _prepare_data derives them.
    train_lmdb = os.path.join(cfg.output_dir, _TRAIN_LMDB_SUBDIR)
    val_lmdb = os.path.join(cfg.output_dir, _VAL_LMDB_SUBDIR)

    # Read the Validation_Set back from the manifest written by data-prep.
    validation = _read_val_manifest(cfg)

    stats.validation_count = len(validation)
    stats.empty_validation_set = len(validation) == 0

    _train_and_finalize(cfg, charset, device, train_lmdb, val_lmdb, validation, stats)


# --- validation manifest (phase decoupling) -------------------------------


def _write_val_manifest(cfg: Config, validation: list[Sample]) -> None:
    """Persist the Validation_Set to ``val_manifest.txt`` under ``cfg.output_dir``.

    Writes one line per Sample as ``image_path<TAB>label``, UTF-8 encoded. The
    parent directory is created if needed. Counterpart of
    :func:`_read_val_manifest`.
    """
    manifest_path = os.path.join(cfg.output_dir, _VAL_MANIFEST_SUBDIR)
    os.makedirs(os.path.dirname(manifest_path) or ".", exist_ok=True)
    with open(manifest_path, "w", encoding="utf-8", newline="\n") as handle:
        for sample in validation:
            handle.write(f"{sample.image_path}\t{sample.label}\n")


def _read_val_manifest(cfg: Config) -> list[Sample]:
    """Read the Validation_Set back from ``val_manifest.txt`` under ``cfg.output_dir``.

    Returns a list of :class:`~olchiki_ocr._train.synthgen.Sample` with
    ``font_path=""`` (irrelevant to evaluation).

    Raises:
        OutputError: When the manifest file does not exist, naming the missing
            path and instructing the caller to run data preparation first.
    """
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
    """Run the DATA-PREPARATION phase: synthetic generation + LMDB building.

    Produces the per-partition train and validation LMDB datasets that DTRB
    training consumes, and mutates ``stats`` in place with the partition counts
    and empty-Validation_Set flag.

    Returns ``(train_lmdb, val_lmdb, validation)`` where the first two are the
    absolute LMDB directory paths under ``cfg.output_dir`` and ``validation`` is
    the Validation_Set Sample list.
    """
    # Generate the synthetic Dataset from the Word_List.
    samples = synthgen.generate(cfg, charset, stats)

    # Partition the Dataset into Training_Set / Validation_Set with a seeded shuffle.
    seed = cfg.seed if cfg.seed is not None else 0
    training, validation = partition.partition(samples, cfg.val_fraction, seed)
    stats.training_count = len(training)
    stats.validation_count = len(validation)
    stats.empty_validation_set = len(validation) == 0

    # Convert each partition into its own LMDB dataset via DTRB
    # create_lmdb_dataset.py. Raises DtrbError (carrying stderr) on a non-zero exit.
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
    """Run the TRAINING/FINALIZATION phase: train (produce .pth), then report.

    Consumes the train/val LMDB datasets produced by the data-prep phase and
    fine-tunes the recognizer, producing the trained Base_Weights ``.pth``
    (Req 10.4). The selected checkpoint path is recorded in
    ``stats.config_snapshot`` for the Run_Report.

    Unlike the previous EasyOCR-coupled pipeline, this phase does NOT write an
    EasyOCR artifact and does NOT run inference-based Validation_Set evaluation:
    conversion to a servable ONNX model is the ``[export]`` tier's job (Task 16),
    and full-validation-set accuracy scoring is the ONNX harness's job (Task 17),
    which reuses :mod:`.evaluator`. The Validation_Set is recorded for provenance
    and yields the empty-set metrics via :mod:`.evaluator` so the Run_Report
    still records a well-defined (0.0) baseline.
    """
    # Transfer-learn the recognizer via DTRB train.py. Parses per-interval
    # CER/Word_Accuracy into stats.per_interval_metrics and returns the selected
    # Fine_Tuned_Model checkpoint (the trained Base_Weights .pth, Req 10.4).
    # Raises PretrainedModelError (unloadable --saved_model) or DtrbError (Req 10.6).
    checkpoint_path = train(cfg, charset, train_lmdb, val_lmdb, device, stats)

    # Record where the trained Base_Weights .pth was written for provenance /
    # the export bridge (Task 16 consumes this .pth).
    stats.config_snapshot = {
        **stats.config_snapshot,
        "base_weights_checkpoint": checkpoint_path,
    }

    # Record the final metrics from what we can compute without a torch-free
    # inference path: an inference-based evaluation over the Validation_Set is
    # deferred to the ONNX harness (Task 17), which reuses evaluator.evaluate.
    # Here we record the well-defined empty-input baseline so the Run_Report has
    # deterministic final metrics rather than fabricating scores.
    val_result = evaluator.evaluate([], [])
    stats.final_cer = val_result.cer
    stats.final_word_accuracy = val_result.word_accuracy

    # Write the single UTF-8 Run_Report recording the applied Config, counts,
    # skipped labels, device, and metrics. Raises OutputError when it cannot be
    # written.
    runreport.write(cfg, stats)


def _config_snapshot(cfg: Config) -> dict:
    """Return a snapshot of every applied Config value.

    Built explicitly (rather than introspected) so the recorded provenance is
    deterministic and independent of dataclass internals.
    """
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
