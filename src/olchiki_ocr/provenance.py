"""ONNX provenance record and integrity check for the bundled model.

A ``provenance.json`` ships in the artifact directory recording the source
checkpoint, charset ordering, network geometry, and model version. The session
is referenced by type only (``TYPE_CHECKING``) so importing this module does not
pull in onnxruntime.
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

#: Provenance sidecar filename; matches ``artifacts.PROVENANCE_FILENAME``,
#: duplicated here to avoid importing the heavier artifacts module.
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
    """Recorded metadata describing the servable ONNX model's origin.

    ``charset`` mirrors the inference charset ordering, which is what yields
    correct CTC decoding.
    """

    model_version: str  # distinct from the package version
    model_name: str  # e.g. "ol_chiki_g2"
    source_checkpoint: str  # .pth the ONNX was exported from
    charset: str  # charset ordering
    geometry: dict  # input_channel/output_channel/hidden_size/imgH
    onnx: dict  # {"opset": int, "input_rank": 4, "output_classes": 49}
    quantization: dict  # {"format", "method", "accuracy_delta_pp"}

    def to_dict(self) -> dict:
        """Return the record as a plain JSON-serializable dict (stable order)."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict, *, artifact_dir: str) -> "Provenance":
        """Build a :class:`Provenance` from a parsed dict.

        Raises ``ModelArtifactError`` (naming ``artifact_dir``) when the record
        is not a mapping or a required field is missing.
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
    """Load and parse ``<artifact_dir>/provenance.json``.

    Raises ``ModelArtifactError`` (naming ``artifact_dir``) when the file is
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
    """Return the ONNX output class dimension reported by ``session`` (duck-typed)."""
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
    """Return the ONNX input rank reported by ``session`` (duck-typed)."""
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

    Checks that the ONNX input rank is 4, the output class dim equals
    ``NUM_CLASSES`` (49), the charset size is ``EMIT_CLASSES`` (48), and the
    provenance-recorded charset ordering matches the inference charset. Any
    mismatch raises ``ModelArtifactError`` naming ``artifact_dir`` and the
    offending field/dimension. Returns the verified :class:`Provenance`.
    """
    prov = load(artifact_dir)

    # Charset size must equal the emit-class count (48).
    inference_charset = charset.as_dtrb_character_arg()
    charset_size = charset.size()
    if charset_size != EMIT_CLASSES:
        raise ModelArtifactError(
            artifact_dir,
            f"charset size mismatch: inference charset has {charset_size} "
            f"characters but the model expects {EMIT_CLASSES} emit classes "
            f"(NUM_CLASSES={NUM_CLASSES} = {EMIT_CLASSES} emit + 1 CTC blank)",
        )

    # ONNX output class dimension must equal NUM_CLASSES (49).
    class_dim = _output_class_dim(session)
    if class_dim is not None and int(class_dim) != NUM_CLASSES:
        raise ModelArtifactError(
            artifact_dir,
            f"ONNX output class dimension mismatch: expected {NUM_CLASSES} "
            f"(= {EMIT_CLASSES} emit classes + 1 CTC blank), got {class_dim!r}",
        )

    # ONNX input rank must be 4.
    rank = _input_rank(session)
    if rank != 4:
        raise ModelArtifactError(
            artifact_dir,
            f"ONNX input rank mismatch: expected rank 4 "
            f"(batch, channel, H, W), got rank {rank}",
        )

    # Provenance charset ordering must match the inference charset.
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
    """Build a :class:`Provenance` for the export tool to record.

    Records the inference charset ordering so the generated record passes
    :func:`load_and_verify` against the same charset.
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
    """Write ``provenance`` to ``<artifact_dir>/provenance.json``, return the path.

    Uses UTF-8 with ``ensure_ascii=False`` so the Ol Chiki charset stays
    readable. Raises ``ModelArtifactError`` (naming ``artifact_dir``) on write
    failure.
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
