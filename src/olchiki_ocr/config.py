"""Lightweight inference-time configuration for the ``olchiki_ocr`` core.

This module defines :class:`InferenceConfig`, the small, torch-free config that
the core inference path uses (design: Components -> ``config.py``; Req 3, 4, 13).
It is intentionally distinct from the training-oriented ``Config`` /
``validate_config``, which live in the ``_train`` tier and are not imported by
the core.

Pure stdlib (``dataclasses`` only): importing this module never pulls in torch,
onnxruntime, numpy, Pillow, or easyocr.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["InferenceConfig"]


@dataclass(frozen=True)
class InferenceConfig:
    """Immutable inference-time configuration for :class:`ModelRecognizer`.

    Attributes:
        model_source: Optional ``Release_Host`` override used when resolving /
            downloading the Model_Artifact. ``None`` uses the packaged default
            (see :data:`olchiki_ocr.artifacts.DEFAULT_RELEASE_HOST`) (Req 4.2).
        cache_dir: Optional override for the Model_Cache directory. ``None``
            falls back to the ``OLCHIKI_OCR_CACHE`` env var and then the default
            ``~/.cache/olchiki-ocr/<model_version>/`` (Req 4.6).
        device: Compute device selector. ``None`` (or ``"cpu"``) selects CPU;
            ``"gpu"`` / ``"cuda"`` request GPU inference (Req 13).
        decoder: Default CTC decoder name; ``"greedy"`` is the package default
            and ``"beam"`` is opt-in (Req 7.1).
    """

    model_source: str | None = None
    cache_dir: str | None = None
    device: str | None = None
    decoder: str = "greedy"
