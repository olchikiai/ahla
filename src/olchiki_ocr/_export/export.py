"""Export bridge: ``.pth`` -> ONNX -> INT8 with fidelity + provenance recording.

Converts the trained DTRB ``None-VGG-BiLSTM-CTC`` weights into a servable ONNX
model, verifies export fidelity, dynamically quantizes to INT8, and records
provenance -- using only ``torch`` + ``onnx`` (plus ``onnxruntime``). Heavy
imports are done inside the functions that need them, each guarded to raise a
``ConfigError`` naming the missing ``[export]`` extra, so the core stays
torch-free and importing this module never hard-fails.

ONNX I/O contract: input ``"input"`` shape ``(N, 1, 32, W)`` with dynamic
batch/width axes; output ``"logits"`` shape ``(N, T, 49)`` (48 emit classes + 1
CTC blank at index 0) with dynamic batch/time axes; opset 13 (LSTM support).

:func:`verify_fidelity` returns the max-abs per-logit torch-vs-ONNX diff; the
``< 1e-3`` threshold is asserted by callers where ``onnxruntime`` is present.
:func:`revalidate_and_record` applies the fallback rule: if INT8 degrades
accuracy by more than :data:`INT8_DEGRADATION_THRESHOLD_PP` pp, FP32 is recorded
as the default with INT8 opt-in. It takes the accuracy numbers (or callables) as
inputs so the tier stays decoupled from the validation harness.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from .._ctc import NUM_CLASSES

__all__ = [
    "ONNX_OPSET",
    "INPUT_NAME",
    "OUTPUT_NAME",
    "INT8_DEGRADATION_THRESHOLD_PP",
    "ExportGeometry",
    "RevalidationResult",
    "export_to_onnx",
    "verify_fidelity",
    "quantize_int8",
    "revalidate_and_record",
]

#: ONNX opset (13 supports the LSTM ops / dynamic axes the BiLSTM stack needs).
ONNX_OPSET: int = 13

#: ONNX graph input / output tensor names.
INPUT_NAME: str = "input"
OUTPUT_NAME: str = "logits"

#: INT8-vs-FP32 degradation threshold (pp); above this, ship FP32 default.
INT8_DEGRADATION_THRESHOLD_PP: float = 0.2

#: Message naming the missing ``[export]`` extra.
_EXPORT_EXTRA_MESSAGE_TMPL: str = (
    "the '[export]' extra is required for {what} "
    "(install: pip install olchiki-ocr[export])"
)


@dataclass(frozen=True)
class ExportGeometry:
    """DTRB ``None-VGG-BiLSTM-CTC`` geometry for building/exporting the model."""

    input_channel: int = 1
    output_channel: int = 512
    hidden_size: int = 256
    img_h: int = 32
    num_class: int = NUM_CLASSES

    def as_provenance_geometry(self) -> dict:
        """Return the geometry dict used by ``provenance.generate``."""
        return {
            "input_channel": self.input_channel,
            "output_channel": self.output_channel,
            "hidden_size": self.hidden_size,
            "imgH": self.img_h,
        }


@dataclass(frozen=True)
class RevalidationResult:
    """Outcome of INT8 re-validation and the FP32/INT8 default decision."""

    fp32_accuracy_pp: float
    int8_accuracy_pp: float
    accuracy_delta_pp: float
    default_format: str
    int8_opt_in: bool


# Dependency guards ([export] extra: onnx / onnxruntime).
def _export_extra_error(what: str, exc: Exception | None = None):
    """Build a ``ConfigError`` naming the missing ``[export]`` extra."""
    from ..errors import ConfigError

    err = ConfigError("[export]", _EXPORT_EXTRA_MESSAGE_TMPL.format(what=what))
    if exc is not None:
        err.__cause__ = exc
    return err


def _require_onnx():
    """Import and return the ``onnx`` module, or raise a ``[export]`` error."""
    try:
        import onnx  # noqa: PLC0415 - lazy import (keep core/import light)
    except ImportError as exc:
        raise _export_extra_error("ONNX export (the 'onnx' package)", exc) from exc
    return onnx


def _require_onnxruntime():
    """Import and return ``onnxruntime``, or raise a ``[export]`` error."""
    try:
        import onnxruntime  # noqa: PLC0415 - lazy import
    except ImportError as exc:
        raise _export_extra_error(
            "ONNX Runtime (the 'onnxruntime' package)", exc
        ) from exc
    return onnxruntime


def _require_ort_quantization():
    """Import ``onnxruntime.quantization.quantize_dynamic`` + ``QuantType``."""
    _require_onnxruntime()
    try:
        from onnxruntime.quantization import (  # noqa: PLC0415
            QuantType,
            quantize_dynamic,
        )
    except ImportError as exc:
        raise _export_extra_error(
            "INT8 quantization (onnxruntime.quantization)", exc
        ) from exc
    return quantize_dynamic, QuantType


# Export (.pth -> ONNX).
def export_to_onnx(
    pth_path: str,
    out_onnx: str,
    *,
    geometry: ExportGeometry | None = None,
    opset: int = ONNX_OPSET,
    sample_width: int = 100,
) -> str:
    """Export a trained ``.pth`` to ONNX with dynamic batch/width/time axes.

    Builds the vendored model, loads the checkpoint (stripping the DTRB
    ``module.`` prefix), and exports via ``torch.onnx.export``. Raises
    ``ConfigError`` if ``[export]`` is missing, ``PretrainedModelError`` on a
    bad/mismatched checkpoint. Returns the written path.
    """
    _require_onnx()  # fail fast if onnx is absent
    import torch  # noqa: PLC0415 - torch is a [export] dep, imported lazily

    from ._model_arch import build_model, load_dtrb_checkpoint

    geom = geometry or ExportGeometry()
    model = build_model(
        input_channel=geom.input_channel,
        output_channel=geom.output_channel,
        hidden_size=geom.hidden_size,
        num_class=geom.num_class,
    )
    load_dtrb_checkpoint(model, pth_path, map_location="cpu")
    model.eval()

    dummy = torch.randn(1, geom.input_channel, geom.img_h, sample_width)
    export_kwargs = dict(
        input_names=[INPUT_NAME],
        output_names=[OUTPUT_NAME],
        dynamic_axes={
            INPUT_NAME: {0: "batch", 3: "width"},
            OUTPUT_NAME: {0: "batch", 1: "time"},
        },
        opset_version=opset,
        do_constant_folding=True,
    )
    # Force the TorchScript exporter (dynamo=False): it needs only torch+onnx,
    # whereas torch 2.6+'s default dynamo exporter needs onnxscript. The dynamo
    # kwarg only exists on newer torch, so pass it defensively.
    with torch.no_grad():
        try:
            torch.onnx.export(model, dummy, out_onnx, dynamo=False, **export_kwargs)
        except TypeError:
            torch.onnx.export(model, dummy, out_onnx, **export_kwargs)
    return out_onnx


# Fidelity (torch vs ONNX per-logit diff).
def verify_fidelity(
    pth_path: str,
    onnx_path: str,
    *,
    geometry: ExportGeometry | None = None,
    sample_widths: tuple[int, ...] = (32, 100, 200),
    seed: int = 1234,
) -> float:
    """Return the max-abs per-logit difference between torch and ONNX.

    Runs both models on the same sampled inputs. The ``< 1e-3`` threshold is
    asserted by callers, not here, so the function stays reusable. Raises
    ``ConfigError`` if ``onnxruntime`` is unavailable.
    """
    ort = _require_onnxruntime()
    import torch  # noqa: PLC0415

    from ._model_arch import build_model, load_dtrb_checkpoint

    geom = geometry or ExportGeometry()
    model = build_model(
        input_channel=geom.input_channel,
        output_channel=geom.output_channel,
        hidden_size=geom.hidden_size,
        num_class=geom.num_class,
    )
    load_dtrb_checkpoint(model, pth_path, map_location="cpu")
    model.eval()

    session = ort.InferenceSession(
        onnx_path, providers=["CPUExecutionProvider"]
    )

    rng = np.random.default_rng(seed)
    max_abs_diff = 0.0
    for width in sample_widths:
        sample = rng.standard_normal(
            (1, geom.input_channel, geom.img_h, int(width))
        ).astype(np.float32)

        with torch.no_grad():
            torch_out = model(torch.from_numpy(sample)).cpu().numpy()

        onnx_out = session.run([OUTPUT_NAME], {INPUT_NAME: sample})[0]

        # Guard against a T-axis length mismatch before diffing.
        if torch_out.shape != onnx_out.shape:
            raise_shape = (
                f"torch output shape {torch_out.shape} != ONNX output shape "
                f"{onnx_out.shape} for input width {width}"
            )
            from ..errors import ModelArtifactError

            raise ModelArtifactError(onnx_path, raise_shape)

        diff = float(np.max(np.abs(torch_out - onnx_out)))
        max_abs_diff = max(max_abs_diff, diff)

    return max_abs_diff


# Quantization (ONNX FP32 -> INT8).
def quantize_int8(onnx_path: str, out_int8: str) -> str:
    """Dynamically quantize an FP32 ONNX model to INT8 (weight-only QInt8).

    Needs no calibration data. Returns the written path; raises ``ConfigError``
    if the ``onnxruntime`` quantization tooling is unavailable.
    """
    quantize_dynamic, QuantType = _require_ort_quantization()
    quantize_dynamic(
        model_input=onnx_path,
        model_output=out_int8,
        weight_type=QuantType.QInt8,
    )
    return out_int8


# Re-validation + provenance recording.
def _resolve_accuracy(value: "float | Callable[[], float]") -> float:
    """Return an accuracy value, calling it first if it is a callable."""
    if callable(value):
        return float(value())
    return float(value)


def revalidate_and_record(
    int8_path: str,
    *,
    fp32_accuracy_pp: "float | Callable[[], float]",
    int8_accuracy_pp: "float | Callable[[], float]",
    artifact_dir: str,
    model_version: str,
    source_checkpoint: str,
    charset,
    geometry: ExportGeometry | None = None,
    opset: int = ONNX_OPSET,
    quant_method: str = "dynamic",
    model_name: str = "ol_chiki_g2",
    write_provenance: bool = True,
):
    """Record the INT8-vs-FP32 accuracy delta in provenance and pick the default.

    Takes the FP32 and INT8 validation accuracies (pp) as floats or callables,
    computes the delta, and writes a provenance record. Fallback rule: if INT8
    degrades accuracy by more than :data:`INT8_DEGRADATION_THRESHOLD_PP` pp, the
    recorded ``format`` is ``"fp32"`` (INT8 opt-in), else ``"int8"``. Returns a
    ``(RevalidationResult, Provenance)`` tuple.
    """
    from .. import provenance as provenance_mod

    geom = geometry or ExportGeometry()

    fp32 = _resolve_accuracy(fp32_accuracy_pp)
    int8 = _resolve_accuracy(int8_accuracy_pp)
    delta = int8 - fp32  # negative => INT8 degraded

    # Degradation beyond the threshold => FP32 default, INT8 opt-in.
    degraded = delta < -INT8_DEGRADATION_THRESHOLD_PP
    default_format = "fp32" if degraded else "int8"

    result = RevalidationResult(
        fp32_accuracy_pp=fp32,
        int8_accuracy_pp=int8,
        accuracy_delta_pp=delta,
        default_format=default_format,
        int8_opt_in=degraded,
    )

    onnx_meta = {
        "opset": opset,
        "input_rank": 4,
        "output_classes": geom.num_class,
    }
    quantization_meta = {
        "format": default_format,
        "method": quant_method,
        "accuracy_delta_pp": delta,
    }

    prov = provenance_mod.generate(
        model_version=model_version,
        source_checkpoint=source_checkpoint,
        charset=charset,
        geometry=geom.as_provenance_geometry(),
        onnx=onnx_meta,
        quantization=quantization_meta,
        model_name=model_name,
    )

    if write_provenance:
        provenance_mod.write(artifact_dir, prov)

    return result, prov
