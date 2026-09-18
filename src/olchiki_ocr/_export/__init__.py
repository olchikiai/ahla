"""``olchiki_ocr._export`` -- the ``[export]`` tier (``.pth`` -> ONNX -> INT8).

This subpackage is the export bridge (Task 16; Req 6.2, 6.3, 6.4, 10.5, 11.2,
11.3, 11.4). It converts trained DTRB ``None-VGG-BiLSTM-CTC`` Base_Weights into
a servable ONNX model, verifies export fidelity, quantizes to INT8, and records
ONNX/quantization provenance.

Import isolation (LOAD-BEARING)
-------------------------------
* ``import olchiki_ocr`` (the core) MUST NOT import this subpackage, so the core
  stays torch-free. Nothing in the core imports ``olchiki_ocr._export``.
* Importing ``olchiki_ocr._export`` itself MUST NOT hard-fail when ``onnx`` /
  ``onnxruntime`` are absent. Only ``numpy`` (a core dependency) is imported at
  module top level by :mod:`._export.export`; ``torch`` (via
  :mod:`._export._model_arch`) and ``onnx`` / ``onnxruntime`` are imported
  **inside the functions that use them**, each guarded to raise a typed
  ``ConfigError`` naming the missing ``[export]`` extra. So the missing-dep
  error surfaces when a function is *called*, not at import time.

Dependency set (Design Decision D1): ``[export]`` = ``torch`` + ``onnx`` (plus
``onnxruntime`` for quantization/fidelity). No ``lmdb`` / DTRB clone; the export
architecture is vendored in :mod:`._export._model_arch`.
"""

from __future__ import annotations

from .export import (
    INPUT_NAME,
    INT8_DEGRADATION_THRESHOLD_PP,
    ONNX_OPSET,
    OUTPUT_NAME,
    ExportGeometry,
    RevalidationResult,
    export_to_onnx,
    quantize_int8,
    revalidate_and_record,
    verify_fidelity,
)
from .harness import (
    BASELINE_ACCURACY_PP,
    PARITY_MIN_ACCURACY_PP,
    PARITY_TOLERANCE_PP,
    HarnessResult,
    check_parity,
    load_manifest,
    run_full_validation,
)

__all__ = [
    "export_to_onnx",
    "verify_fidelity",
    "quantize_int8",
    "revalidate_and_record",
    "ExportGeometry",
    "RevalidationResult",
    "ONNX_OPSET",
    "INPUT_NAME",
    "OUTPUT_NAME",
    "INT8_DEGRADATION_THRESHOLD_PP",
    # Full_Validation_Harness (Task 17.1).
    "run_full_validation",
    "load_manifest",
    "check_parity",
    "HarnessResult",
    "BASELINE_ACCURACY_PP",
    "PARITY_TOLERANCE_PP",
    "PARITY_MIN_ACCURACY_PP",
]
