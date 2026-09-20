"""Training-side config and validation for the ``[train]`` tier.

Defines the frozen ``Config`` dataclass for a fine-tuning run and
``validate_config``. This is the training counterpart to the core
:class:`olchiki_ocr.config.InferenceConfig` and is not imported by the
torch-free core.

``validate_config`` raises ``ConfigError`` (invalid setting), ``CharsetError``
(bad ``extra_characters``), or ``PretrainedModelError`` (missing
``pretrained_recognizer`` file). :func:`low_resource_config` freezes the VGG
backbone and trains only BiLSTM + head at a small batch.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace

from ..errors import CharsetError, ConfigError, PretrainedModelError

__all__ = ["Config", "validate_config", "low_resource_config", "LOW_RESOURCE_BATCH_SIZE"]

# Accepted devices; CUDA availability is checked later by ``select_device``.
_VALID_DEVICES = ("cuda", "cpu")

# Small batch size for the low-resource preset.
LOW_RESOURCE_BATCH_SIZE = 16


@dataclass(frozen=True)
class Config:
    """Immutable runtime parameters for a fine-tuning run.

    The first five fields are required (no default); the rest carry defaults.
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
    # When True the Trainer forwards ``--freeze_FeatureExtraction`` so only the
    # BiLSTM stage and CTC head train. Default False = full fine-tuning.
    freeze_feature_extraction: bool = False


def low_resource_config(base: Config) -> Config:
    """Return a copy of ``base`` wired for the low-resource preset.

    Freezes the VGG backbone and trains only BiLSTM + CTC head at a small batch
    (``freeze_feature_extraction=True``, ``batch_size=LOW_RESOURCE_BATCH_SIZE``,
    ``None-VGG-BiLSTM-CTC`` stack). Other fields are preserved from ``base``.
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
    """Return True if ``value`` is an int but not a bool (bool is not valid)."""
    return isinstance(value, int) and not isinstance(value, bool)


def _is_real(value: object) -> bool:
    """Return True if ``value`` is a real number (int/float) but not a bool."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def validate_config(cfg: Config) -> None:
    """Validate every Config parameter before any Sample is produced.

    Raises ``ConfigError`` for an invalid setting, ``CharsetError`` for a bad
    ``extra_characters`` value, or ``PretrainedModelError`` when
    ``pretrained_recognizer`` is not an existing file (fail-fast before any DTRB
    subprocess).
    """
    # Required, non-empty strings.
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
    # Fail fast if the checkpoint file is absent (the empty-string check above
    # runs first, so an empty value is a ConfigError, not this).
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

    # Each extra character must be a single Unicode code point.
    for char in cfg.extra_characters:
        if len(char) != 1:
            raise CharsetError(
                char, "extra character must be a single Unicode code point"
            )

    # val_fraction in [0.0, 1.0].
    if not _is_real(cfg.val_fraction):
        raise ConfigError("val_fraction", "value must be a number")
    if not (0.0 <= cfg.val_fraction <= 1.0):
        raise ConfigError(
            "val_fraction", "value must be in the range 0.0 to 1.0 inclusive"
        )

    # num_iterations: positive integer.
    if not _is_int(cfg.num_iterations):
        raise ConfigError("num_iterations", "value must be an integer")
    if cfg.num_iterations <= 0:
        raise ConfigError("num_iterations", "value must be a positive integer")

    # batch_size: positive integer.
    if not _is_int(cfg.batch_size):
        raise ConfigError("batch_size", "value must be an integer")
    if cfg.batch_size <= 0:
        raise ConfigError("batch_size", "value must be a positive integer")

    # learning_rate: positive real number.
    if not _is_real(cfg.learning_rate):
        raise ConfigError("learning_rate", "value must be a real number")
    if cfg.learning_rate <= 0:
        raise ConfigError("learning_rate", "value must be a positive real number")

    # device, when set, must be 'cuda' or 'cpu'.
    if cfg.device is not None and cfg.device not in _VALID_DEVICES:
        raise ConfigError("device", "value must be 'cuda' or 'cpu' when specified")
