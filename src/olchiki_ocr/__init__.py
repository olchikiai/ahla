"""olchiki-ocr: Ol Chiki (Santali) OCR for pre-cropped word/line images.

``ModelRecognizer`` and ``Prediction`` are exported lazily via ``__getattr__``
(PEP 562) so a bare ``import olchiki_ocr`` stays free of onnxruntime/numpy/
Pillow; the engine loads only on first use. The typed errors are pure stdlib
and imported eagerly so they can be used in ``except`` clauses without the
engine.
"""

from .errors import (
    CharsetError,
    ConfigError,
    DeviceError,
    DtrbError,
    FontError,
    ImageError,
    IngestError,
    ModelArtifactError,
    ModelDownloadError,
    OcrTrainerError,
    OutputError,
    PretrainedModelError,
)

__version__ = "0.1.0"

__all__ = [
    # Lazily-resolved recognition surface (PEP 562 __getattr__).
    "ModelRecognizer",
    "Prediction",
    # Eagerly-imported typed error hierarchy (pure stdlib, engine-free).
    "OcrTrainerError",
    "ConfigError",
    "CharsetError",
    "IngestError",
    "FontError",
    "PretrainedModelError",
    "DeviceError",
    "DtrbError",
    "OutputError",
    "ImageError",
    "ModelArtifactError",
    "ModelDownloadError",
    "__version__",
]

# The recognition names are resolved lazily so ``import olchiki_ocr`` does not
# eagerly load onnxruntime (recognizer.py -> session.py imports it).
_LAZY_EXPORTS = frozenset({"ModelRecognizer", "Prediction"})


def __getattr__(name: str):
    """Lazily resolve ``ModelRecognizer``/``Prediction`` on first access (PEP 562)."""
    if name in _LAZY_EXPORTS:
        from . import recognizer

        return getattr(recognizer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily-exported names in ``dir(olchiki_ocr)``."""
    return sorted({*globals().keys(), *__all__})
