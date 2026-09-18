"""ONNX-based provenance record and integrity check for the bundled model.

A ``provenance.json`` file ships inside the Model_Artifact directory (alongside
``model.onnx`` and the charset resource) and records which checkpoint, charset
ordering, and network geometry the servable ONNX model was produced from, plus
the Model_Version (distinct from the Package_Version in ``pyproject.toml``,
Req 14.3).

This module is the **ONNX-adapted** replacement for the carried-over
EasyOCR-coupled ``provenance.py``. The old module read geometry/charset from an
EasyOCR ``user_network/<model>.yaml`` sidecar; that YAML coupling is dropped
entirely (design: Module Migration Plan -> ``provenance.py`` "Adapt"). The new
:class:`Provenance` schema describes the ONNX model directly (design: Data
Models -> "Provenance (new ONNX schema)"):

* ``model_version`` -- the Model_Version, distinct from the Package_Version.
* ``model_name`` -- e.g. ``"ol_chiki_g2"``.
* ``source_checkpoint`` -- the ``.pth`` the ONNX was exported from (Req 6.4).
* ``charset`` -- the charset ordering (the joined charset string; Req 8.4).
* ``geometry`` -- ``input_channel``/``output_channel``/``hidden_size``/``imgH``.
* ``onnx`` -- ``{"opset": int, "input_rank": 4, "output_classes": 49}``.
* ``quantization`` -- ``{"format", "method", "accuracy_delta_pp"}`` (Req 11.3).

Integrity check (:func:`load_and_verify`, Design Decision D8, Correctness
Property P11; Req 2.7, 8.4, 11.3) runs at model-resolution time (called by
``ModelRecognizer.from_pretrained`` in Task 11). It reconciles the charset size
with the ONNX output width and confirms the recorded charset ordering matches
the inference charset ordering, raising :class:`ModelArtifactError` (naming the
artifact dir and the offending field/dimension) on any mismatch. See the
:func:`load_and_verify` docstring for exactly which checks run and how the ONNX
geometry is obtained.

Import hygiene: this module is pure standard library plus the core typed-error
and charset/CTC helpers. It MUST NOT import torch, easyocr, or onnxruntime. In
particular it references :class:`~olchiki_ocr.session.OnnxSession` **by type
only** (a ``TYPE_CHECKING`` import), because importing ``olchiki_ocr.session``
pulls in onnxruntime; keeping the annotation behind ``TYPE_CHECKING`` lets
``import olchiki_ocr.provenance`` stay importable without onnxruntime installed.
The session parameter is duck-typed at runtime (it only needs to report the
validated ONNX input rank and output class dimension).
"""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any

from ._ctc import EMIT_CLASSES, NUM_CLASSES
from .errors import ModelArtifactError

if TYPE_CHECKING:  # pragma: no cover - typing only; avoids importing onnxruntime
    from .charset import Charset
    from .session import OnnxSession

__all__ = [
    "Provenance",
    "PROVENANCE_FILENAME",
    "load",
    "load_and_verify",
    "generate",
    "write",
]

#: The provenance sidecar filename, resolved relative to an artifact directory.
#: Matches ``artifacts.PROVENANCE_FILENAME``; defined here to avoid importing the
#: (heavier) artifacts module for a single constant.
PROVENANCE_FILENAME = "provenance.json"

# Fields required in a well-formed provenance record.
_REQUIRED_FIELDS = (
    "model_version",
    "model_name",
    "source_checkpoint",
    "charset",
    "geometry",
    "onnx",
    "quantization",
)


@dataclass(frozen=True)
class Provenance:
    """Recorded metadata describing the servable ONNX Model_Artifact's origin.

    Serialized as ``<artifact_dir>/provenance.json``. ``charset`` mirrors the
    inference charset ordering (``"".join(charset.characters)``), which is what
    yields correct CTC decoding (Req 8.4); ``geometry`` records the DTRB network
    geometry; ``onnx`` and ``quantization`` describe the exported/quantized
    model (Req 6.4, 11.3).
    """

    model_version: str  # distinct from Package_Version (Req 14.3)
    model_name: str  # e.g. "ol_chiki_g2"
    source_checkpoint: str  # .pth the ONNX was exported from (Req 6.4)
    charset: str  # charset ordering (Req 8.4)
    geometry: dict  # input_channel/output_channel/hidden_size/imgH
    onnx: dict  # {"opset": int, "input_rank": 4, "output_classes": 49}
    quantization: dict  # {"format", "method", "accuracy_delta_pp"} (Req 11.3)

    def to_dict(self) -> dict:
        """Return the record as a plain JSON-serializable dict (stable order)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict, *, artifact_dir: str) -> "Provenance":
        """Build a :class:`Provenance` from a parsed dict.

        Args:
            data: The parsed ``provenance.json`` mapping.
            artifact_dir: Directory the record came from, used only to name the
                error on a malformed record.

        Raises:
            ModelArtifactError: Naming ``artifact_dir`` when a required field is
                missing or the record is not a mapping.
        """
        if not isinstance(data, dict):
            raise ModelArtifactError(
                artifact_dir,
                f"malformed provenance record: expected a JSON object, got "
                f"{type(data).__name__}",
            )
        missing = [field for field in _REQUIRED_FIELDS if field not in data]
        if missing:
            raise ModelArtifactError(
                artifact_dir,
                f"malformed provenance record: missing field(s) "
                f"{', '.join(missing)}",
            )
        return cls(
            model_version=data["model_version"],
            model_name=data["model_name"],
            source_checkpoint=data["source_checkpoint"],
            charset=data["charset"],
            geometry=data["geometry"],
            onnx=data["onnx"],
            quantization=data["quantization"],
        )


def _provenance_path(artifact_dir: str) -> str:
    """Return the ``provenance.json`` path within ``artifact_dir``."""
    return os.path.join(artifact_dir, PROVENANCE_FILENAME)


def load(artifact_dir: str) -> Provenance:
    """Load and parse ``<artifact_dir>/provenance.json`` into a :class:`Provenance`.

    Args:
        artifact_dir: Directory holding ``provenance.json``.

    Returns:
        The parsed :class:`Provenance`.

    Raises:
        ModelArtifactError: Naming ``artifact_dir`` when the provenance file is
            missing, unreadable, not valid JSON, or missing required fields.
    """
    path = _provenance_path(artifact_dir)
    try:
        with open(path, "r", encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, ValueError) as exc:
        raise ModelArtifactError(
            artifact_dir, f"cannot read provenance ({exc})"
        ) from exc
    return Provenance.from_dict(data, artifact_dir=artifact_dir)


def _output_class_dim(session: "OnnxSession") -> Any:
    """Return the ONNX output class dimension reported by ``session``.

    Duck-typed so a lightweight stub (or the real :class:`OnnxSession`) works:
    prefer the ``output_class_dim()`` accessor, then an ``output_shape``
    sequence's last element. The real ``OnnxSession`` (Task 6.1) already
    validated this dimension against ``NUM_CLASSES`` at construction, so this is
    a cross-check, not the primary enforcement.
    """
    accessor = getattr(session, "output_class_dim", None)
    if callable(accessor):
        return accessor()
    output_shape = getattr(session, "output_shape", None)
    if output_shape:
        return list(output_shape)[-1]
    raise ModelArtifactError(
        getattr(session, "onnx_path", "<session>"),
        "session does not expose an ONNX output class dimension",
    )


def _input_rank(session: "OnnxSession") -> int:
    """Return the ONNX input rank reported by ``session`` (duck-typed).

    Prefers the ``input_rank()`` accessor, then ``len(input_shape)``. The real
    ``OnnxSession`` validated this to be 4 at construction.
    """
    accessor = getattr(session, "input_rank", None)
    if callable(accessor):
        return int(accessor())
    input_shape = getattr(session, "input_shape", None)
    if input_shape is not None:
        return len(list(input_shape))
    raise ModelArtifactError(
        getattr(session, "onnx_path", "<session>"),
        "session does not expose an ONNX input rank",
    )


def load_and_verify(
    artifact_dir: str,
    session: "OnnxSession",
    charset: "Charset",
) -> Provenance:
    """Load ``provenance.json`` and verify artifact/charset/ONNX agreement.

    Runs the model-resolution-time integrity check (Design Decision D8,
    Correctness Property P11; Req 2.7, 8.4, 11.3). It is cheap: it reads the
    small JSON record and cross-checks a few dimensions -- it never loads weight
    tensors.

    How the ONNX geometry is obtained: the :class:`~olchiki_ocr.session.OnnxSession`
    (Task 6.1) already validates, **at construction**, that the ONNX input rank
    is 4 and the output class dimension equals ``NUM_CLASSES`` (49), raising
    :class:`ModelArtifactError` otherwise. So by the time a live session exists,
    the ONNX-side geometry is enforced. This function additionally *cross-checks*
    those session-reported dimensions and enforces the provenance/charset
    agreement that the session cannot know about.

    Checks (each mismatch raises :class:`ModelArtifactError` naming
    ``artifact_dir`` and the offending field/dimension):

    1. ``provenance.json`` loads and parses with all required fields.
    2. ``charset.size() == EMIT_CLASSES`` (48) and ``charset.size() + 1 ==
       NUM_CLASSES`` (49) -- the inference charset must have exactly the emit-
       class count the model expects (reconciles the "48 emit / 49 total"
       wording of Req 2.7/8.4: 48 charset chars + 1 CTC blank == 49).
    3. The ONNX output class dimension reported by ``session`` equals
       ``NUM_CLASSES`` (49). (Enforced primarily at session construction; this
       is a defense-in-depth cross-check that also names the dimension.)
    4. The ONNX input rank reported by ``session`` equals 4.
    5. The provenance-recorded charset ordering equals the inference charset
       ordering (``provenance.charset == "".join(charset.characters)``); a
       reordered/incompatible charset fails here (Req 8.4).

    Args:
        artifact_dir: Directory holding the resolved artifact + ``provenance.json``.
        session: A constructed ``OnnxSession`` (or duck-typed stub) exposing the
            validated ONNX input rank and output class dimension.
        charset: The inference :class:`~olchiki_ocr.charset.Charset`.

    Returns:
        The verified :class:`Provenance` on success.

    Raises:
        ModelArtifactError: Naming ``artifact_dir`` and the mismatched
            field/dimension on any failure.
    """
    prov = load(artifact_dir)

    # --- 2. Charset size must be exactly the emit-class count (48). ----------
    inference_charset = charset.as_dtrb_character_arg()
    charset_size = charset.size()
    if charset_size != EMIT_CLASSES:
        raise ModelArtifactError(
            artifact_dir,
            f"charset size mismatch: inference charset has {charset_size} "
            f"characters but the model expects {EMIT_CLASSES} emit classes "
            f"(NUM_CLASSES={NUM_CLASSES} = {EMIT_CLASSES} emit + 1 CTC blank)",
        )

    # --- 3. ONNX output class dimension must equal NUM_CLASSES (49). ---------
    class_dim = _output_class_dim(session)
    if class_dim is not None and int(class_dim) != NUM_CLASSES:
        raise ModelArtifactError(
            artifact_dir,
            f"ONNX output class dimension mismatch: expected {NUM_CLASSES} "
            f"(= {EMIT_CLASSES} emit classes + 1 CTC blank), got {class_dim!r}",
        )

    # --- 4. ONNX input rank must be 4. ---------------------------------------
    rank = _input_rank(session)
    if rank != 4:
        raise ModelArtifactError(
            artifact_dir,
            f"ONNX input rank mismatch: expected rank 4 "
            f"(batch, channel, H, W), got rank {rank}",
        )

    # --- 5. Provenance charset ordering must match the inference charset. ----
    if prov.charset != inference_charset:
        raise ModelArtifactError(
            artifact_dir,
            "charset ordering mismatch between provenance and inference "
            "charset: the model was trained with a different charset ordering "
            "(field: charset)",
        )

    return prov


def generate(
    *,
    model_version: str,
    source_checkpoint: str,
    charset: "Charset",
    geometry: dict,
    onnx: dict,
    quantization: dict,
    model_name: str = "ol_chiki_g2",
) -> Provenance:
    """Build a :class:`Provenance` for the export tool to record (Task 16).

    Records the inference charset ordering (``"".join(charset.characters)``) so
    the generated record passes :func:`load_and_verify` against the same
    charset. The ``onnx``/``quantization`` dicts are supplied by the export tool
    (opset, output class count, INT8-vs-FP32 accuracy delta, etc.).

    Args:
        model_version: The Model_Version to record (Req 14.3).
        source_checkpoint: Identifier/path of the ``.pth`` the ONNX was exported
            from (Req 6.4).
        charset: The inference :class:`~olchiki_ocr.charset.Charset`; its joined
            characters are recorded as the ``charset`` ordering.
        geometry: DTRB network geometry
            (``input_channel``/``output_channel``/``hidden_size``/``imgH``).
        onnx: ONNX metadata, e.g.
            ``{"opset": 13, "input_rank": 4, "output_classes": 49}``.
        quantization: Quantization metadata, e.g.
            ``{"format": "int8", "method": "dynamic", "accuracy_delta_pp": 0.0}``.
        model_name: The model name / filename stem. Defaults to ``"ol_chiki_g2"``.

    Returns:
        The generated :class:`Provenance`.
    """
    return Provenance(
        model_version=model_version,
        model_name=model_name,
        source_checkpoint=source_checkpoint,
        charset=charset.as_dtrb_character_arg(),
        geometry=dict(geometry),
        onnx=dict(onnx),
        quantization=dict(quantization),
    )


def write(artifact_dir: str, provenance: Provenance) -> str:
    """Write ``provenance`` to ``<artifact_dir>/provenance.json`` and return the path.

    Serializes with UTF-8 and ``ensure_ascii=False`` (mirroring the old
    module's write behavior) so the Ol Chiki ``charset`` is stored readably, plus
    stable key order and a trailing newline.

    Args:
        artifact_dir: Directory to write ``provenance.json`` into.
        provenance: The record to serialize.

    Returns:
        The path to the written ``provenance.json``.

    Raises:
        ModelArtifactError: Naming ``artifact_dir`` when the file cannot be
            written.
    """
    path = _provenance_path(artifact_dir)
    try:
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(provenance.to_dict(), handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    except OSError as exc:
        raise ModelArtifactError(
            artifact_dir, f"cannot write provenance ({exc.strerror or exc})"
        ) from exc
    return path
