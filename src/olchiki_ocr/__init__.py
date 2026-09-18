"""olchiki-ocr: Ol Chiki (Santali) OCR for pre-cropped word/line images.

Recognition runs on **ONNX Runtime** (torch-free / easyocr-free core) with the
package's own preprocessing and CTC decoding. This module defines the top-level
public surface and is deliberately split into an *eager* tier and a *lazy* tier
so that a bare ``import olchiki_ocr`` stays cheap.

Public API surface (Req 1.1, 2.2, 2.3)
--------------------------------------
Installed callers reach the recognizer through this package's top level::

    from olchiki_ocr import ModelRecognizer, Prediction

plus the full typed error hierarchy (:class:`OcrTrainerError` and its
subclasses, including :class:`ImageError`, :class:`ModelArtifactError`,
:class:`ModelDownloadError`) so failures can be caught by type.

Keeping ``import olchiki_ocr`` lightweight (lazy engine load)
-------------------------------------------------------------
``import olchiki_ocr`` must NOT eagerly load the inference engine. The
recognition core reaches ``onnxruntime`` through
:mod:`olchiki_ocr.recognizer` -> :mod:`olchiki_ocr.session`; if
``ModelRecognizer``/``Prediction`` were imported eagerly here, a bare
``import olchiki_ocr`` would pull ``onnxruntime`` (and numpy/Pillow) in at
package-import time.

``onnxruntime`` is the core engine and a legitimate core dependency, so loading
it is not "wrong" -- but keeping package import free of it makes ``import
olchiki_ocr`` fast and, importantly, lets the typed errors be imported and used
in ``except`` clauses without loading the engine at all.

To preserve that property the two recognition names are exported **lazily** via
module ``__getattr__`` (PEP 562): they resolve ``from .recognizer import ...``
only on first attribute access (``olchiki_ocr.ModelRecognizer`` /
``from olchiki_ocr import ModelRecognizer``), so a bare ``import olchiki_ocr``
-- and importing the typed errors -- touches no heavy dependency. This mirrors
the old EasyOCR package's lazy pattern (which deferred ``torch``); here we defer
``onnxruntime``.

The typed errors come from :mod:`olchiki_ocr.errors`, which is pure-stdlib
(``__future__`` typing only), so they are imported **eagerly** and are always
available for ``except`` clauses without triggering the engine.
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
    """Lazily resolve ``ModelRecognizer``/``Prediction`` on first access (PEP 562).

    Deferring these imports to :mod:`olchiki_ocr.recognizer` keeps a bare
    ``import olchiki_ocr`` free of ``onnxruntime`` (Req 2.2, 2.3); the engine
    loads only when the recognition surface is actually used.
    """
    if name in _LAZY_EXPORTS:
        from . import recognizer

        return getattr(recognizer, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily-exported names in ``dir(olchiki_ocr)``."""
    return sorted({*globals().keys(), *__all__})
