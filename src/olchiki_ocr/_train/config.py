"""Training configuration value type and validation for the ``[train]`` tier.

This module defines the frozen ``Config`` dataclass that carries every runtime
parameter for a fine-tuning run, plus ``validate_config`` which enforces the
Configuration rules before any Sample is generated.

It is the training-oriented counterpart of the core, lightweight
:class:`olchiki_ocr.config.InferenceConfig`: per the design's config split
(Module Migration Plan), the training ``Config`` / ``validate_config`` live in
the ``_train`` tier and are NOT imported by the torch-free core.

Validation runs once, up front. On any invalid parameter ``validate_config``
raises :class:`~olchiki_ocr.errors.ConfigError` (exit code 2) naming the
offending setting -- or :class:`~olchiki_ocr.errors.CharsetError` (exit code 3)
for an invalid ``extra_characters`` value, or
:class:`~olchiki_ocr.errors.PretrainedModelError` (exit code 6) when
``pretrained_recognizer`` names a path that is not an existing file.

Low-resource preset (Req 10.3)
------------------------------
:func:`low_resource_config` returns a :class:`Config` wired for the low-resource
fine-tune preset: it freezes the VGG FeatureExtraction backbone and trains only
the BiLSTM SequenceModeling stage plus the CTC Prediction head, at a small batch
size. The freeze is expressed via :attr:`Config.freeze_feature_extraction`,
which the Trainer forwards to DTRB ``train.py`` as ``--freeze_FeatureExtraction``
(see :mod:`olchiki_ocr._train.trainer`).
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from ..errors import CharsetError, ConfigError, PretrainedModelError

__all__ = ["Config", "validate_config", "low_resource_config", "LOW_RESOURCE_BATCH_SIZE"]

# Compute devices accepted from the Configuration. Actual availability of a
# requested CUDA device is checked later by ``select_device``.
_VALID_DEVICES = ("cuda", "cpu")

# Small batch size used by the low-resource preset (Req 10.3). Kept modest so
# the frozen-backbone fine-tune fits on minimal compute.
LOW_RESOURCE_BATCH_SIZE = 16


@dataclass(frozen=True)
class Config:
    """Immutable set of runtime parameters controlling a fine-tuning run.

    Required parameters (``word_list_path``, ``font_paths``,
    ``pretrained_recognizer``, ``output_dir``, ``dtrb_repo_path``) have no
    default; the caller must supply them. All remaining parameters carry the
    defaults documented in the design's Config table and mandated by the
    requirements.
    """

    # Required parameters.
    word_list_path: str  # path to Word_List
    font_paths: tuple[str, ...]  # >=1 Font_File path
    pretrained_recognizer: str  # Pretrained_Recognizer (the Base_Weights .pth)
    output_dir: str  # output directory
    dtrb_repo_path: str  # local DTRB clone path (pinned commit)

    # Optional parameters with documented defaults.
    run_report_path: str | None = None  # default under output_dir
    extra_characters: str = ""  # extra Charset chars
    val_fraction: float = 0.1  # validation split fraction
    num_iterations: int = 20000  # training iterations
    batch_size: int = 192  # batch size
    learning_rate: float = 1.0  # Adadelta learning rate
    val_interval: int = 2000  # validation interval
    best_checkpoint: bool = True  # best-checkpoint selection
    seed: int | None = None  # random seed
    device: str | None = None  # Compute_Device override
    augmentations: tuple[str, ...] = ()  # image augmentations
    eval_dataset_path: str | None = None  # external eval dataset
    overwrite: bool = False  # overwrite outputs
    confidence_output: bool = False  # emit confidences
    network_transformation: str = "None"  # DTRB Trans stage
    network_feature: str = "VGG"  # DTRB Feat stage
    network_sequence: str = "BiLSTM"  # DTRB Seq stage
    network_prediction: str = "CTC"  # DTRB Pred stage
    # Low-resource preset knob (Req 10.3): when True, the Trainer freezes the
    # VGG FeatureExtraction backbone (via DTRB ``--freeze_FeatureExtraction``)
    # so only the BiLSTM SequenceModeling stage and the CTC Prediction head are
    # trained. Default False preserves full fine-tuning.
    freeze_feature_extraction: bool = False


def low_resource_config(base: Config) -> Config:
    """Return a copy of ``base`` wired for the low-resource fine-tune preset.

    The low-resource preset (Req 10.3) freezes the VGG FeatureExtraction
    backbone and trains only the BiLSTM SequenceModeling stage plus the CTC
    Prediction head, at a small batch size. Concretely this sets:

    * ``freeze_feature_extraction=True`` -- the Trainer forwards
      ``--freeze_FeatureExtraction`` to DTRB ``train.py`` so the VGG stage
      parameters are frozen and only Sequence + Prediction are updated.
    * ``batch_size=LOW_RESOURCE_BATCH_SIZE`` -- a small batch that fits on
      minimal compute.
    * ``network_feature="VGG"``, ``network_sequence="BiLSTM"``,
      ``network_prediction="CTC"`` -- the ``None-VGG-BiLSTM-CTC`` stack the
      Base_Weights are trained with (kept explicit so the preset is
      self-describing).

    All other fields are preserved from ``base`` so callers keep their paths,
    seed, and remaining hyperparameters. The returned Config is validated the
    same way as any other via :func:`validate_config`.
    """
    return replace(
        base,
        freeze_feature_extraction=True,
        batch_size=LOW_RESOURCE_BATCH_SIZE,
        network_feature="VGG",
        network_sequence="BiLSTM",
        network_prediction="CTC",
    )


def _is_int(value: object) -> bool:
    """Return True if ``value`` is an integer but not a bool.

    ``bool`` is a subclass of ``int`` in Python, but a boolean is not a valid
    hyperparameter value, so it is rejected here.
    """
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    """Return True if ``value`` is a real number (int or float) but not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_config(cfg: Config) -> None:
    """Validate every Configuration parameter before any Sample is produced.

    Raises :class:`ConfigError` (or :class:`CharsetError` for an invalid
    ``extra_characters`` value) naming the first invalid setting encountered.
    Because this runs before generation, an invalid Configuration results in
    zero Samples produced.

    Rules enforced, in order:
      * required params present and non-empty, with at least one ``font_path``.
      * ``pretrained_recognizer`` names an existing regular file, else
        :class:`PretrainedModelError` (exit code 6) -- fail-fast before any DTRB
        subprocess is launched.
      * each ``extra_characters`` char is a single Unicode code point, else
        :class:`CharsetError` naming the value.
      * ``0.0 <= val_fraction <= 1.0``.
      * ``num_iterations`` is a positive integer.
      * ``batch_size`` is a positive integer.
      * ``learning_rate`` is a positive real number.
      * ``device``, when set, is ``'cuda'`` or ``'cpu'`` (availability checked
        later by ``select_device``).
    """
    # Required, non-empty string parameters.
    if not cfg.word_list_path:
        raise ConfigError("word_list_path", "required parameter is missing or empty")
    if not cfg.font_paths:
        raise ConfigError("font_paths", "at least one font path is required")
    for font_path in cfg.font_paths:
        if not font_path:
            raise ConfigError("font_paths", "font path entries must be non-empty")
    if not cfg.pretrained_recognizer:
        raise ConfigError(
            "pretrained_recognizer", "required parameter is missing or empty"
        )
    # Fail fast before any DTRB subprocess if the checkpoint file is absent. The
    # empty-string ConfigError above runs first, so an empty value raises
    # ConfigError, not PretrainedModelError.
    if not os.path.isfile(cfg.pretrained_recognizer):
        raise PretrainedModelError(
            cfg.pretrained_recognizer,
            "pretrained recognizer file not found; set 'pretrained_recognizer' to "
            "a real checkpoint path",
        )
    if not cfg.output_dir:
        raise ConfigError("output_dir", "required parameter is missing or empty")
    if not cfg.dtrb_repo_path:
        raise ConfigError("dtrb_repo_path", "required parameter is missing or empty")

    # Each extra character must be a single Unicode code point. This is surfaced
    # here for early failure; ``build_charset`` also guards later.
    for char in cfg.extra_characters:
        if len(char) != 1:
            raise CharsetError(
                char, "extra character must be a single Unicode code point"
            )

    # Validation split fraction in [0.0, 1.0].
    if not _is_real(cfg.val_fraction):
        raise ConfigError("val_fraction", "value must be a number")
    if not (0.0 <= cfg.val_fraction <= 1.0):
        raise ConfigError(
            "val_fraction", "value must be in the range 0.0 to 1.0 inclusive"
        )

    # Number of training iterations: positive integer.
    if not _is_int(cfg.num_iterations):
        raise ConfigError("num_iterations", "value must be an integer")
    if cfg.num_iterations <= 0:
        raise ConfigError("num_iterations", "value must be a positive integer")

    # Batch size: positive integer.
    if not _is_int(cfg.batch_size):
        raise ConfigError("batch_size", "value must be an integer")
    if cfg.batch_size <= 0:
        raise ConfigError("batch_size", "value must be a positive integer")

    # Learning rate: positive real number.
    if not _is_real(cfg.learning_rate):
        raise ConfigError("learning_rate", "value must be a real number")
    if cfg.learning_rate <= 0:
        raise ConfigError("learning_rate", "value must be a positive real number")

    # Compute device, when set, must be 'cuda' or 'cpu'.
    if cfg.device is not None and cfg.device not in _VALID_DEVICES:
        raise ConfigError("device", "value must be 'cuda' or 'cpu' when specified")
