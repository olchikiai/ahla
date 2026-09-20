"""ONNX Runtime session wrapper and execution-provider selection.

onnxruntime is the inference engine here; this module must stay torch-free (no
torch/easyocr/cv2). :func:`select_providers` maps a device to providers: ``None``
or ``"cpu"`` -> CPU, ``"gpu"``/``"cuda"`` -> CUDA when available (else
:class:`DeviceError`), anything else -> :class:`DeviceError`. :class:`OnnxSession`
loads a model and validates its geometry (input rank 4, output class dim ==
``NUM_CLASSES``) before use.
"""

from __future__ import annotations

import os
from typing import Any, List, Optional

import numpy as np

from .errors import DeviceError, ModelArtifactError

# CTC class layout: NUM_CLASSES=49 (48 emit classes + 1 blank at index 0).
# ``_ctc`` is the source of truth; fall back to defaults if it isn't present.
try:  # pragma: no cover - trivial import guard
    from ._ctc import NUM_CLASSES, EMIT_CLASSES, BLANK_INDEX
except Exception:  # noqa: BLE001 - _ctc may not exist yet
    NUM_CLASSES = 49
    EMIT_CLASSES = 48
    BLANK_INDEX = 0

# onnxruntime is the core engine. The guard only keeps this module importable
# (and select_providers mockable) when it isn't installed; OnnxSession still
# fails clearly if it's genuinely missing.
try:
    import onnxruntime  # type: ignore
except Exception:  # noqa: BLE001 - keep module importable without onnxruntime
    onnxruntime = None  # type: ignore[assignment]

__all__ = ["OnnxSession", "select_providers"]

_CPU_PROVIDER = "CPUExecutionProvider"
_CUDA_PROVIDER = "CUDAExecutionProvider"
_GPU_ALIASES = frozenset({"gpu", "cuda"})


def _available_providers() -> List[str]:
    """Return the execution providers available in the installed onnxruntime."""
    if onnxruntime is None:  # onnxruntime not installed in this environment
        return []
    return list(onnxruntime.get_available_providers())


def select_providers(device: Optional[str]) -> List[str]:
    """Map a requested ``device`` to an ordered list of ONNX Runtime providers.

    ``None``/``"cpu"`` -> CPU; ``"gpu"``/``"cuda"`` -> CUDA when available.
    Raises ``DeviceError`` (naming the device) if a GPU is requested but CUDA is
    unavailable, or the value is unrecognized.
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

    On construction the model is loaded with the selected providers and its
    geometry validated: input rank 4 ``(batch, channel, H, W)`` and output class
    dim == :data:`NUM_CLASSES` (49). Symbolic (dynamic) batch/width/time dims are
    accepted; only rank and the fixed class dim are checked.
    """

    def __init__(self, onnx_path: str, device: Optional[str] = None) -> None:
        self.onnx_path = onnx_path
        self.device = device

        # Check existence first so a missing artifact gives a clear
        # ModelArtifactError instead of an opaque onnxruntime failure.
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

        # Normalize onnxruntime load failures to ModelArtifactError naming the path.
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

        # Retain the raw shapes so provenance.load_and_verify can cross-check
        # geometry without re-opening the model.
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
        """Validate ONNX input rank (4) and output class dimension (NUM_CLASSES)."""
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

        # Output class dim (the last dim) must equal NUM_CLASSES (49); output is
        # (T, NUM_CLASSES) or (batch, T, NUM_CLASSES). Dynamic dims are ignored.
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
        """Run inference on a single ``(1, 1, 32, W)`` tensor, returning raw logits.

        Output is ``(T, NUM_CLASSES)`` or ``(1, T, NUM_CLASSES)`` depending on the
        export, returned unchanged for the decoders to interpret.
        """
        outputs = self._session.run([self._output_name], {self._input_name: tensor})
        return outputs[0]

    def get_providers(self) -> List[str]:
        """Report the execution providers actually active in the session."""
        return list(self._session.get_providers())


def _is_fixed_dim(dim: Any) -> bool:
    """Return True if ``dim`` is a concrete (non-symbolic) integer dimension.

    Symbolic dims come back as ``str`` or ``None`` and are treated as not-fixed.
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
