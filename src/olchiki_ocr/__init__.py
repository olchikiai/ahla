"""olchiki-ocr: Ol Chiki (Santali) OCR for pre-cropped word/line images.

``ModelRecognizer`` and ``Prediction`` are exported lazily via ``__getattr__``
(PEP 562) so a bare ``import olchiki_ocr`` stays free of onnxruntime/numpy/
Pillow; the engine loads only on first use. The typed errors are pure stdlib
and imported eagerly so they can be used in ``except`` clauses without the
engine.

The document-segmenter surface (``PageSegmenter``, ``Segmentation_Result``,
``Region``, ``Page_Recognizer``, ``Page_Result``, ``RecognizedRegion``,
``Granularity``, ``CoordinateSpace``) is exported lazily the same way so a bare
``import olchiki_ocr`` also stays free of OpenCV: the segmentation modules load
only on first attribute access, and cv2 itself loads only when a segmentation
operation actually runs (Req 8.2, 8.5, 12.2, 12.3).
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

__version__ = "0.2.0"

__all__ = [
    # Lazily-resolved recognition surface (PEP 562 __getattr__).
    "ModelRecognizer",
    "Prediction",
    # Lazily-resolved document-segmenter surface (PEP 562 __getattr__).
    "PageSegmenter",
    "Segmentation_Result",
    "Region",
    "Page_Recognizer",
    "Page_Result",
    "RecognizedRegion",
    "Granularity",
    "CoordinateSpace",
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

# Each lazily-exported name maps to the submodule it resolves from. Resolving a
# name imports only that submodule, so ``import olchiki_ocr`` does not eagerly
# load onnxruntime (recognizer.py -> session.py imports it) or OpenCV. The
# segmentation submodules are themselves cv2-free at import time (cv2 loads only
# via their lazy ``_require_cv2()`` when a segmentation operation runs), so even
# accessing ``PageSegmenter``/``Page_Recognizer`` does not import cv2 (Req 8.2,
# 8.4, 12.2, 12.3).
_LAZY_EXPORTS: dict[str, str] = {
    # Recognition surface -> recognizer.py (imports onnxruntime on load).
    "ModelRecognizer": "recognizer",
    "Prediction": "recognizer",
    # Segmenter + enums -> segmentation.py (cv2-free at import).
    "PageSegmenter": "segmentation",
    "Granularity": "segmentation",
    "CoordinateSpace": "segmentation",
    # Segmentation data models -> regions.py (numpy + stdlib only).
    "Segmentation_Result": "regions",
    "Region": "regions",
    # End-to-end orchestrator + models -> page_recognizer.py (engine-free at import).
    "Page_Recognizer": "page_recognizer",
    "Page_Result": "page_recognizer",
    "RecognizedRegion": "page_recognizer",
}


def __getattr__(name: str):
    """Lazily resolve the recognition and segmenter surfaces on first access (PEP 562)."""
    module_name = _LAZY_EXPORTS.get(name)
    if module_name is not None:
        import importlib

        module = importlib.import_module(f".{module_name}", __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    """Include the lazily-exported names in ``dir(olchiki_ocr)``."""
    return sorted({*globals().keys(), *__all__})
