"""ONNX Runtime session wrapper and execution-provider selection.

This module is the **ONNX Runtime replacement** for the carried-over torch
``device.py`` (Design Decision D9). The torch ``select_device`` is NOT part of
the core; it survives only in ``_train``. Accordingly, this module MUST NOT
import torch, easyocr, or cv2.

Responsibilities (design: Components -> ``session.py`` (OnnxSession); Req 2, 13):

* :func:`select_providers` maps a requested device to an ordered list of ONNX
  Runtime execution providers:

  - ``None``            -> ``["CPUExecutionProvider"]``                 (Req 13.1)
  - ``"cpu"``           -> ``["CPUExecutionProvider"]``
  - ``"gpu"``/``"cuda"``-> ``["CUDAExecutionProvider"]`` iff CUDA is actually
    available in the installed onnxruntime, else :class:`DeviceError` (Req 13.2,
    13.3)
  - anything else       -> :class:`DeviceError` naming the value

  Availability is determined via ``onnxruntime.get_available_providers()`` so it
  is easy to mock/patch in tests.

* :class:`OnnxSession` loads an ONNX model with the selected providers and
  validates its geometry before use:

  - the ONNX file must exist and be loadable, else :class:`ModelArtifactError`
    naming the path (Req 2.6);
  - the input tensor rank must be 4 ``(batch, channel, H, W)`` (Req 2.7, D8);
  - the output's class dimension must equal the model's ``NUM_CLASSES``.

  Note on the "output class dimension is 48" wording in Req 2.7/8.4: that "48"
  is the count of **emit** classes. The actual ONNX output width is
  ``NUM_CLASSES = 49`` (48 emit classes + 1 CTC blank at index 0; Design
  Decision D8). We therefore validate the ONNX output class dimension against
  the total ``NUM_CLASSES`` (49) since that is the real tensor width. The
  authoritative source for these constants is :mod:`olchiki_ocr._ctc` (produced
  by Task 4.1); we import them from there and only fall back to module-level
  defaults if that module is not yet available.

onnxruntime and numpy are both **core** dependencies. numpy is imported at
module top. onnxruntime is also imported at module top (it is the core
inference engine, so importing it in the core is expected and correct -- unlike
torch). The import is guarded only so this module (and :func:`select_providers`,
which tests patch) stays importable in a development environment where
``onnxruntime`` may not yet be installed; when onnxruntime is genuinely required
(constructing an :class:`OnnxSession`) its absence surfaces as a clear error.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import numpy as np

from .errors import DeviceError, ModelArtifactError

# --- CTC class-layout constants (Design Decision D8) -------------------------
# Prefer importing the confirmed layout from ``olchiki_ocr._ctc`` (Task 4.1),
# which is the single source of truth once it lands. If that module is not yet
# available (parallel task not landed at import time), fall back to the design's
# documented defaults: NUM_CLASSES=49 (48 emit classes + 1 CTC blank),
# EMIT_CLASSES=48, BLANK_INDEX=0. The orchestrator should re-verify this module
# once _ctc lands to confirm the import path is taken.
try:  # pragma: no cover - trivial import guard
    from ._ctc import NUM_CLASSES, EMIT_CLASSES, BLANK_INDEX
except Exception:  # noqa: BLE001 - _ctc may not exist yet (Task 4.1 pending)
    # Fallback defaults; _ctc is the source of truth once Task 4.1 lands.
    NUM_CLASSES = 49
    EMIT_CLASSES = 48
    BLANK_INDEX = 0

# --- onnxruntime import (core engine) ----------------------------------------
# onnxruntime is a core dependency and importing it here is correct. The guard
# only keeps the module importable (and select_providers mockable) in a dev env
# where onnxruntime is not yet installed; OnnxSession construction still fails
# clearly if it is genuinely missing.
try:
    import onnxruntime  # type: ignore
except Exception:  # noqa: BLE001 - keep module importable without onnxruntime
    onnxruntime = None  # type: ignore[assignment]

__all__ = ["OnnxSession", "select_providers"]

_CPU_PROVIDER = "CPUExecutionProvider"
_CUDA_PROVIDER = "CUDAExecutionProvider"
_GPU_ALIASES = frozenset({"gpu", "cuda"})


def _available_providers() -> List[str]:
    """Return the execution providers available in the installed onnxruntime.

    Accessing ``onnxruntime.get_available_providers()`` indirectly (through the
    module reference) keeps this trivially mockable in tests via
    ``unittest.mock.patch("olchiki_ocr.session.onnxruntime.get_available_providers", ...)``.
    """
    if onnxruntime is None:  # onnxruntime not installed in this environment
        return []
    return list(onnxruntime.get_available_providers())


def select_providers(device: Optional[str]) -> List[str]:
    """Map a requested ``device`` to an ordered list of ONNX Runtime providers.

    Args:
        device: ``None`` or ``"cpu"`` selects CPU; ``"gpu"``/``"cuda"`` selects
            CUDA when available; any other value is invalid.

    Returns:
        The ordered list of execution-provider names to pass to
        ``onnxruntime.InferenceSession(..., providers=...)``.

    Raises:
        DeviceError: if a GPU device is requested but CUDA is not available in
            the installed onnxruntime, or if ``device`` is an unrecognized
            value. The error names the requested device (Req 13.3, 12.5).
    """
    if device is None:
        return [_CPU_PROVIDER]

    normalized = device.strip().lower() if isinstance(device, str) else device

    if normalized == "cpu":
        return [_CPU_PROVIDER]

    if normalized in _GPU_ALIASES:
        if _CUDA_PROVIDER in _available_providers():
            return [_CUDA_PROVIDER]
        raise DeviceError(
            device,
            message=(
                "GPU inference requested but CUDAExecutionProvider is not "
                "available in the installed onnxruntime (install the [gpu] "
                "extra / onnxruntime-gpu)"
            ),
        )

    raise DeviceError(device, message="Unrecognized compute device")


class OnnxSession:
    """Thin wrapper around ``onnxruntime.InferenceSession`` with geometry checks.

    On construction the ONNX model is loaded with the providers chosen by
    :func:`select_providers`, and its input/output geometry is validated:

    * the input tensor must have rank 4 ``(batch, channel, H, W)``;
    * the output's class dimension must equal :data:`NUM_CLASSES` (49).

    ONNX shapes may be symbolic (dynamic) for the batch and width/time
    dimensions -- such dims come back as strings or ``None``. Only the rank and
    the fixed class dimension are validated; dynamic ``W``/``T`` are accepted.
    """

    def __init__(self, onnx_path: str, device: Optional[str] = None) -> None:
        self.onnx_path = onnx_path
        self.device = device

        # Existence check first so a missing/unreadable artifact reports a clear
        # ModelArtifactError naming the path (Req 2.6) rather than an opaque
        # onnxruntime failure.
        if not os.path.isfile(onnx_path):
            raise ModelArtifactError(
                onnx_path, message="ONNX model file is absent or not a file"
            )
        if not os.access(onnx_path, os.R_OK):
            raise ModelArtifactError(
                onnx_path, message="ONNX model file is not readable"
            )

        if onnxruntime is None:  # engine genuinely unavailable
            raise ModelArtifactError(
                onnx_path,
                message=(
                    "onnxruntime is not installed; the core inference engine is "
                    "required to load the ONNX model"
                ),
            )

        providers = select_providers(device)

        # Catch onnxruntime load failures and re-raise as ModelArtifactError
        # naming the path (Req 2.6).
        try:
            self._session = onnxruntime.InferenceSession(
                onnx_path, providers=providers
            )
        except DeviceError:
            raise
        except Exception as exc:  # noqa: BLE001 - normalize to typed error
            raise ModelArtifactError(
                onnx_path, message=f"Failed to load ONNX model ({exc})"
            ) from exc

        inputs = self._session.get_inputs()
        outputs = self._session.get_outputs()
        if not inputs or not outputs:
            raise ModelArtifactError(
                onnx_path,
                message="ONNX model exposes no input or no output tensor",
            )

        self._input_name = inputs[0].name
        self._output_name = outputs[0].name
        input_shape = list(inputs[0].shape)
        output_shape = list(outputs[0].shape)

        # Retain the validated raw shapes as read-only attributes so other core
        # components (e.g. provenance.load_and_verify, Task 9.1) can cross-check
        # the ONNX geometry without re-opening the model. These are recorded
        # only after construction succeeds, so their presence implies the
        # geometry below was validated. This is purely additive and does not
        # change any existing behavior.
        self._input_shape = input_shape
        self._output_shape = output_shape

        self._validate_geometry(input_shape, output_shape)

    @property
    def input_shape(self) -> List[Any]:
        """The validated ONNX input tensor shape (rank-4; dims may be symbolic)."""
        return list(self._input_shape)

    @property
    def output_shape(self) -> List[Any]:
        """The validated ONNX output tensor shape (last dim == NUM_CLASSES)."""
        return list(self._output_shape)

    def input_rank(self) -> int:
        """Return the rank of the ONNX input tensor (validated to be 4)."""
        return len(self._input_shape)

    def output_class_dim(self) -> Any:
        """Return the ONNX output class dimension (the last output dim)."""
        return self._output_shape[-1]

    def _validate_geometry(self, input_shape: list, output_shape: list) -> None:
        """Validate ONNX input rank and output class dimension (Req 2.7, D8)."""
        # Input rank must be 4: (batch, channel, H, W).
        if len(input_shape) != 4:
            raise ModelArtifactError(
                self.onnx_path,
                message=(
                    f"ONNX input rank mismatch: expected rank 4 "
                    f"(batch, channel, H, W), got rank {len(input_shape)} "
                    f"with shape {input_shape!r}"
                ),
            )

        # Output class dimension must equal NUM_CLASSES (49). Output may be
        # (T, NUM_CLASSES) or (batch, T, NUM_CLASSES); the class dim is the last
        # dimension. Dynamic (symbolic/None) dims are ignored for T/batch/W.
        if not output_shape:
            raise ModelArtifactError(
                self.onnx_path,
                message="ONNX output has no dimensions",
            )
        class_dim = output_shape[-1]
        if _is_fixed_dim(class_dim) and int(class_dim) != NUM_CLASSES:
            raise ModelArtifactError(
                self.onnx_path,
                message=(
                    f"ONNX output class dimension mismatch: expected "
                    f"{NUM_CLASSES} (= {EMIT_CLASSES} emit classes + 1 CTC "
                    f"blank), got {class_dim!r} in output shape {output_shape!r}"
                ),
            )

    def run(self, tensor: "np.ndarray") -> "np.ndarray":
        """Run inference on a single preprocessed input tensor.

        Args:
            tensor: the preprocessed model input, shape ``(1, 1, 32, W)``,
                dtype float32, values in ``[-1, 1]`` (see preprocessing).

        Returns:
            The raw logits array produced by the model. The expected shape is
            ``(T, NUM_CLASSES)`` or ``(1, T, NUM_CLASSES)`` depending on how the
            model was exported; the array is returned unchanged for the caller
            (decoders) to interpret.
        """
        outputs = self._session.run([self._output_name], {self._input_name: tensor})
        return outputs[0]

    def get_providers(self) -> List[str]:
        """Report the execution providers actually active in the session."""
        return list(self._session.get_providers())


def _is_fixed_dim(dim: Any) -> bool:
    """Return True if ``dim`` is a concrete (non-symbolic) integer dimension.

    ONNX symbolic/dynamic dims come back as ``str`` (e.g. ``"batch"``) or
    ``None``; those are treated as not-fixed and skipped during validation.
    """
    if dim is None:
        return False
    if isinstance(dim, bool):  # guard: bool is an int subclass
        return False
    if isinstance(dim, int):
        return True
    # Some exporters yield numeric strings for otherwise-fixed dims.
    if isinstance(dim, str):
        try:
            int(dim)
        except (TypeError, ValueError):
            return False
        return True
    return False
