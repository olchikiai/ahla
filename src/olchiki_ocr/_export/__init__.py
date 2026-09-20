"""The ``[export]`` tier: ``.pth`` -> ONNX -> INT8 with fidelity + provenance.

Import isolation is load-bearing: importing the core never imports this
subpackage (core stays torch-free), and importing ``_export`` must not hard-fail
when ``onnx``/``onnxruntime`` are absent -- heavy deps are imported inside the
functions that use them, guarded to raise a ``ConfigError`` naming ``[export]``.
Dependency set is ``torch`` + ``onnx`` (plus ``onnxruntime``); the model
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
    "run_full_validation",
    "load_manifest",
    "check_parity",
    "HarnessResult",
    "BASELINE_ACCURACY_PP",
    "PARITY_TOLERANCE_PP",
    "PARITY_MIN_ACCURACY_PP",
]
