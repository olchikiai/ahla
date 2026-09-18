"""Export bridge: ``.pth`` -> ONNX -> INT8 with fidelity + provenance recording.

This is the ``[export]`` tier tooling (Task 16.1 / 16.2; Req 6.2, 6.3, 6.4,
10.5, 11.2, 11.3, 11.4). It converts the trained DTRB ``None-VGG-BiLSTM-CTC``
Base_Weights (``.pth``) into a servable ONNX model, verifies export fidelity,
dynamically quantizes to INT8, and records ONNX/quantization provenance --
depending only on ``torch`` + ``onnx`` (plus ``onnxruntime`` for the
quantization/fidelity steps), never on ``lmdb`` or the DTRB clone (Design
Decision D1). The bridge also serves fine-tuned checkpoints from ``_train``.

Dependency guards (Req 3.6, mirrors the ``[cv]``/``[train]`` guards)
--------------------------------------------------------------------
The core stays torch-free: ``import olchiki_ocr`` never imports this module, and
importing ``olchiki_ocr._export`` must not hard-fail when ``onnx`` /
``onnxruntime`` are absent. Heavy imports are therefore performed **inside the
functions that need them**, each guarded by a helper that raises a typed
:class:`~olchiki_ocr.errors.ConfigError` naming the missing ``[export]`` extra
(same approach as ``preprocessing._require_cv2`` for ``[cv]``). ``torch`` is
imported at the top of :mod:`._model_arch`, which is imported lazily here.

ONNX I/O contract (documented, LOAD-BEARING)
--------------------------------------------
* **Input**: name ``"input"``, shape ``(N, 1, 32, W)`` -- batch ``N``, 1
  grayscale channel, fixed height ``imgH=32``, variable width ``W``. Dynamic
  axes: axis 0 (``batch``) and axis 3 (``width``).
* **Output**: name ``"logits"``, shape ``(N, T, 49)`` -- batch ``N``, variable
  time steps ``T``, 49 classes (48 charset emit classes + 1 CTC blank at index
  0, per D8 / :mod:`olchiki_ocr._ctc`). Dynamic axes: axis 0 (``batch``) and
  axis 1 (``time``). This is batch-first, matching the vendored model's forward
  and DTRB's CTC branch.
* **Opset**: 13 (recent enough for LSTM/dynamic-axes support; see
  :data:`ONNX_OPSET`).

Fidelity note (Req 11.2, Task 16.3 coupling)
--------------------------------------------
:func:`verify_fidelity` returns the max-abs per-logit difference between the
torch model and the exported ONNX model on the **same** sampled input. The
``< 1e-3`` assertion is applied by the caller / the fixture test (Task 16.3),
which runs in the ``[export]``/CI environment where ``onnxruntime`` is present.
If ``onnxruntime`` is not importable in the current environment, the function
raises a clear ``ConfigError`` naming ``[export]`` rather than silently
skipping -- the actual threshold check runs where ORT is available.

Revalidation coupling (Req 11.3, 11.4, Task 17)
-----------------------------------------------
:func:`revalidate_and_record` records the INT8-vs-FP32 ``accuracy_delta_pp``
into provenance. The Full_Validation_Harness that produces those accuracy
numbers is Task 17; this function therefore accepts the accuracy numbers (or a
zero-argument callable returning them) as inputs, keeping the ``[export]`` tier
decoupled from the harness. It applies the >0.2 pp fallback decision (Req 11.4):
if INT8 degrades accuracy by more than :data:`INT8_DEGRADATION_THRESHOLD_PP`
percentage points, FP32 is recorded as the default with INT8 opt-in.
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

#: ONNX opset used for export. 13 supports the LSTM ops and dynamic axes the
#: BiLSTM stack needs; kept as a named default and recorded in provenance.
ONNX_OPSET: int = 13

#: ONNX graph input / output tensor names (see module docstring I/O contract).
INPUT_NAME: str = "input"
OUTPUT_NAME: str = "logits"

#: INT8-vs-FP32 accuracy degradation threshold in percentage points (Design
#: Decision D7 / Req 11.4). Above this, ship FP32 default with INT8 opt-in.
INT8_DEGRADATION_THRESHOLD_PP: float = 0.2

#: Shared message naming the missing ``[export]`` extra (mirrors the ``[cv]``
#: message shape in ``preprocessing._CV_EXTRA_MESSAGE``).
_EXPORT_EXTRA_MESSAGE_TMPL: str = (
    "the '[export]' extra is required for {what} "
    "(install: pip install olchiki-ocr[export])"
)


@dataclass(frozen=True)
class ExportGeometry:
    """DTRB ``None-VGG-BiLSTM-CTC`` geometry for building/exporting the model.

    Defaults match ``models/ol_chiki_g2/provenance.json`` plus the confirmed 49-class CTC
    head (48 emit + 1 blank). ``num_class`` defaults to
    :data:`olchiki_ocr._ctc.NUM_CLASSES`.
    """

    input_channel: int = 1
    output_channel: int = 512
    hidden_size: int = 256
    img_h: int = 32
    num_class: int = NUM_CLASSES

    def as_provenance_geometry(self) -> dict:
        """Return the geometry dict shape used by ``provenance.generate``."""
        return {
            "input_channel": self.input_channel,
            "output_channel": self.output_channel,
            "hidden_size": self.hidden_size,
            "imgH": self.img_h,
        }


@dataclass(frozen=True)
class RevalidationResult:
    """Outcome of INT8 re-validation and the FP32/INT8 default decision.

    Attributes:
        fp32_accuracy_pp: FP32 exact-match word accuracy in percentage points.
        int8_accuracy_pp: INT8 exact-match word accuracy in percentage points.
        accuracy_delta_pp: ``int8_accuracy_pp - fp32_accuracy_pp`` (negative =
            INT8 degraded). Recorded in provenance ``quantization``.
        default_format: ``"int8"`` if the INT8 model is shipped as default,
            else ``"fp32"`` (INT8 available as opt-in).
        int8_opt_in: True when FP32 is the default and INT8 is opt-in
            (i.e. degradation exceeded the threshold).
    """

    fp32_accuracy_pp: float
    int8_accuracy_pp: float
    accuracy_delta_pp: float
    default_format: str
    int8_opt_in: bool


# --------------------------------------------------------------------------- #
# Dependency guards ([export] extra: onnx / onnxruntime).                     #
# --------------------------------------------------------------------------- #
def _export_extra_error(what: str, exc: Exception | None = None):
    """Build a ``ConfigError`` naming the missing ``[export]`` extra.

    Uses :class:`~olchiki_ocr.errors.ConfigError` (a typed
    ``OcrTrainerError``) so the failure integrates with CLI exit-code handling,
    mirroring the ``[cv]`` guard in :mod:`olchiki_ocr.preprocessing`.
    """
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


# --------------------------------------------------------------------------- #
# Export (.pth -> ONNX).                                                       #
# --------------------------------------------------------------------------- #
def export_to_onnx(
    pth_path: str,
    out_onnx: str,
    *,
    geometry: ExportGeometry | None = None,
    opset: int = ONNX_OPSET,
    sample_width: int = 100,
) -> str:
    """Export a trained ``.pth`` to ONNX (Req 6.2, 10.5).

    Builds the vendored :class:`~olchiki_ocr._export._model_arch.NoneVGGBiLSTMCTC`
    with the given ``geometry``, loads the checkpoint (stripping the DTRB
    ``module.`` prefix), and exports via ``torch.onnx.export`` with dynamic
    ``batch``/``width`` (input) and ``batch``/``time`` (output) axes and the
    documented tensor names.

    Args:
        pth_path: Path to the trained Base_Weights ``.pth`` checkpoint.
        out_onnx: Destination path for the exported ``.onnx`` file.
        geometry: Model geometry; defaults to :class:`ExportGeometry` (the
            trained Ol Chiki geometry, 49-class CTC head).
        opset: ONNX opset version (default :data:`ONNX_OPSET` = 13).
        sample_width: Width ``W`` of the dummy export input (any valid width;
            dynamic axes make the exported graph width-agnostic).

    Returns:
        ``out_onnx`` (the path written).

    Raises:
        ConfigError: If the ``[export]`` extra (``onnx``) is not installed.
        PretrainedModelError: If the checkpoint cannot be read or does not match
            the geometry (raised by ``load_dtrb_checkpoint``).
    """
    _require_onnx()  # fail fast with a [export] error if onnx is absent
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
    # Use the stable TorchScript-based exporter (``dynamo=False``). torch 2.6+
    # defaults ``torch.onnx.export`` to the dynamo exporter, which requires the
    # extra ``onnxscript`` package; the TorchScript path needs only ``torch`` +
    # ``onnx`` (the declared ``[export]`` deps) and gives well-defined
    # ``dynamic_axes`` semantics for this VGG+BiLSTM+CTC graph. The ``dynamo``
    # kwarg only exists on newer torch, so pass it defensively.
    with torch.no_grad():
        try:
            torch.onnx.export(model, dummy, out_onnx, dynamo=False, **export_kwargs)
        except TypeError:
            # Older torch without the ``dynamo`` kwarg: the TorchScript exporter
            # is already the default.
            torch.onnx.export(model, dummy, out_onnx, **export_kwargs)
    return out_onnx


# --------------------------------------------------------------------------- #
# Fidelity (torch vs ONNX per-logit diff).                                     #
# --------------------------------------------------------------------------- #
def verify_fidelity(
    pth_path: str,
    onnx_path: str,
    *,
    geometry: ExportGeometry | None = None,
    sample_widths: tuple[int, ...] = (32, 100, 200),
    seed: int = 1234,
) -> float:
    """Return the max-abs per-logit difference between torch and ONNX (Req 11.2).

    Rebuilds the torch model from ``pth_path`` and runs the exported ONNX model
    (via ``onnxruntime``) on the **same** sampled input tensors, returning the
    maximum absolute difference across all logits and all samples. The caller /
    fixture test (Task 16.3) asserts this is ``< 1e-3``; the threshold is
    intentionally *not* asserted here so the function is reusable and the
    assertion lives with the test.

    Args:
        pth_path: Path to the trained ``.pth`` (the fidelity reference).
        onnx_path: Path to the exported ``.onnx`` to compare against.
        geometry: Model geometry; defaults to :class:`ExportGeometry`.
        sample_widths: Widths of the random sample inputs to compare over.
        seed: RNG seed for reproducible sample inputs.

    Returns:
        The maximum absolute per-logit difference (a non-negative ``float``).

    Raises:
        ConfigError: If ``onnxruntime`` (the ``[export]`` extra) is unavailable,
            so the fidelity check cannot run in this environment. The actual
            ``< 1e-3`` assertion runs where ORT is present (Task 16.3).
        PretrainedModelError: If the checkpoint cannot be loaded.
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

        # Guard against any T-axis length mismatch before diffing.
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


# --------------------------------------------------------------------------- #
# Quantization (ONNX FP32 -> INT8).                                            #
# --------------------------------------------------------------------------- #
def quantize_int8(onnx_path: str, out_int8: str) -> str:
    """Dynamically quantize an FP32 ONNX model to INT8 (Req 6.3).

    Uses ``onnxruntime.quantization.quantize_dynamic`` (weight-only dynamic
    quantization, ``QuantType.QInt8``), which needs no calibration data and is
    the standard path for LSTM/Linear-heavy recognition models.

    Args:
        onnx_path: Path to the FP32 ONNX model to quantize.
        out_int8: Destination path for the INT8 ONNX model.

    Returns:
        ``out_int8`` (the path written).

    Raises:
        ConfigError: If ``onnxruntime`` / its quantization tooling (the
            ``[export]`` extra) is unavailable.
    """
    quantize_dynamic, QuantType = _require_ort_quantization()
    quantize_dynamic(
        model_input=onnx_path,
        model_output=out_int8,
        weight_type=QuantType.QInt8,
    )
    return out_int8


# --------------------------------------------------------------------------- #
# Re-validation + provenance recording (Req 11.3, 11.4; couples to Task 17).   #
# --------------------------------------------------------------------------- #
def _resolve_accuracy(value: "float | Callable[[], float]") -> float:
    """Return an accuracy value, calling it first if it is a callable.

    Lets :func:`revalidate_and_record` accept either already-computed accuracy
    numbers or zero-argument callables (e.g. a bound Full_Validation_Harness run
    from Task 17), keeping the ``[export]`` tier decoupled from the harness.
    """
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
    """Record the INT8-vs-FP32 accuracy delta in provenance + decide the default.

    This is the export tool's provenance-recording step (Req 11.3, 11.4). It
    takes the FP32 and INT8 full-validation-set accuracies (in percentage
    points) -- either as plain floats or as zero-argument callables that compute
    them via the Full_Validation_Harness (Task 17) -- computes the delta, and
    writes a :class:`~olchiki_ocr.provenance.Provenance` record via
    :func:`olchiki_ocr.provenance.generate` / :func:`~olchiki_ocr.provenance.write`.

    Fallback decision (Req 11.4, Design Decision D7): if INT8 degrades accuracy
    by more than :data:`INT8_DEGRADATION_THRESHOLD_PP` (0.2) percentage points
    versus FP32, the recorded ``quantization.format`` is ``"fp32"`` (FP32 shipped
    as default, INT8 opt-in); otherwise it is ``"int8"``.

    Coupling to Task 17: the Full_Validation_Harness is implemented in Task 17;
    this function does not run the harness itself. Callers pass the accuracy
    numbers (or callables) so the ``[export]`` tier stays independent of the
    harness / validation dataset.

    Args:
        int8_path: Path to the quantized INT8 ONNX model (recorded / decided
            on). Named for traceability even though the decision is based on the
            accuracy numbers.
        fp32_accuracy_pp: FP32 exact-match accuracy (pp), or a callable → float.
        int8_accuracy_pp: INT8 exact-match accuracy (pp), or a callable → float.
        artifact_dir: Directory to write ``provenance.json`` into (when
            ``write_provenance`` is True).
        model_version: Model_Version to record (Req 14.3).
        source_checkpoint: The ``.pth`` the ONNX was exported from (Req 6.4).
        charset: The inference ``Charset`` (its ordering is recorded).
        geometry: Model geometry; defaults to :class:`ExportGeometry`.
        opset: ONNX opset recorded in provenance ``onnx``.
        quant_method: Quantization method string recorded in provenance
            (default ``"dynamic"``).
        model_name: Model name / filename stem (default ``"ol_chiki_g2"``).
        write_provenance: When True, write ``provenance.json`` to
            ``artifact_dir``; when False, only build the record + decision.

    Returns:
        A ``(RevalidationResult, Provenance)`` tuple. The ``Provenance`` carries
        the ``onnx`` and ``quantization`` dicts with the recorded
        ``accuracy_delta_pp`` and chosen ``format``.
    """
    from .. import provenance as provenance_mod

    geom = geometry or ExportGeometry()

    fp32 = _resolve_accuracy(fp32_accuracy_pp)
    int8 = _resolve_accuracy(int8_accuracy_pp)
    delta = int8 - fp32  # negative => INT8 degraded

    # Req 11.4: degradation beyond the threshold => FP32 default, INT8 opt-in.
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
