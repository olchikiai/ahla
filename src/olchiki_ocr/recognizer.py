"""Primary public API: :class:`ModelRecognizer` and :class:`Prediction`.

``from_pretrained`` resolves an artifact dir, builds the ONNX session, runs the
provenance integrity check, and stores what ``predict`` needs. ``predict`` runs
the per-image pipeline: preprocess -> ONNX run -> CTC decode -> ``str`` (or a
:class:`Prediction` with confidence when ``confidence=True``).

The session is built eagerly in ``from_pretrained`` because the provenance
integrity check needs the geometry ``OnnxSession`` validates at construction;
the single session is reused in ``predict``.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import TYPE_CHECKING

from . import artifacts, provenance
from .charset import build_charset
from .decoders import BeamSearchDecoder, GreedyDecoder
from .errors import ConfigError
from .preprocessing import Preprocessing_Pipeline
from .session import OnnxSession

if TYPE_CHECKING:  # typing-only; avoids any runtime import cost/cycle
    from .charset import Charset
    from .decoders import ScoreFn
    from .lexicon import Lexicon

__all__ = ["ModelRecognizer", "Prediction"]

#: Default servable model version. Needed to name the cache dir
#: (``~/.cache/olchiki-ocr/<version>/``) before provenance is loaded, so it
#: can't come from provenance itself. Irrelevant for a ``path=`` load. Bump when
#: a new servable model ships.
DEFAULT_MODEL_VERSION = "0.1.0"

#: Accepted decoder names; "greedy" is the default, "beam" is opt-in.
_VALID_DECODERS = frozenset({"greedy", "beam"})


@dataclass(frozen=True)
class Prediction:
    """Structured recognition result: text plus optional confidence in ``[0, 1]``."""

    text: str
    confidence: float | None = None


def _validate_decoder(name: str) -> str:
    """Return ``name`` if a recognized decoder, else raise ``ConfigError``."""
    if name not in _VALID_DECODERS:
        raise ConfigError(
            "decoder",
            f"unknown decoder {name!r}; expected one of "
            f"{sorted(_VALID_DECODERS)}",
        )
    return name


class ModelRecognizer:
    """Load a Model_Artifact and recognize Ol Chiki text in cropped images.

    Construct via :meth:`from_pretrained`; the public initializer takes the
    already-resolved collaborators so the class stays testable and the
    resolution/verification policy lives in one place.
    """

    def __init__(
        self,
        session: OnnxSession,
        charset: "Charset",
        *,
        default_decoder: str = "greedy",
        provenance_record: "provenance.Provenance | None" = None,
    ) -> None:
        """Store the resolved collaborators (usually built by ``from_pretrained``).

        Raises ``ConfigError`` if ``default_decoder`` is not recognized.
        """
        self._session = session
        self._charset = charset
        self._default_decoder = _validate_decoder(default_decoder)
        self._provenance = provenance_record
        # A single stateless greedy decoder can be reused across calls.
        self._greedy = GreedyDecoder()

    @property
    def session(self) -> OnnxSession:
        """The underlying geometry-validated ONNX session."""
        return self._session

    @property
    def charset(self) -> "Charset":
        """The recognition charset used for decoding."""
        return self._charset

    @property
    def default_decoder(self) -> str:
        """The configured default decoder name (``"greedy"`` or ``"beam"``)."""
        return self._default_decoder

    @property
    def provenance(self) -> "provenance.Provenance | None":
        """The verified provenance record, if available."""
        return self._provenance

    @classmethod
    def from_pretrained(
        cls,
        path: str | None = None,
        *,
        model_source: str | None = None,
        cache_dir: str | None = None,
        device: str | None = None,
        decoder: str = "greedy",
    ) -> "ModelRecognizer":
        """Resolve, verify, and construct a ready-to-use recognizer.

        Flow: resolve the artifact dir (local ``path`` or cache/download) ->
        build the charset -> construct the ``OnnxSession`` (validates geometry
        and device) -> run the provenance integrity check -> store.

        ``device`` is ``None``/``"cpu"`` for CPU, ``"gpu"``/``"cuda"`` for GPU.

        Raises ``ConfigError`` (bad decoder), ``ModelArtifactError`` (bad
        artifact/geometry or provenance mismatch), ``DeviceError`` (GPU
        unavailable), or ``ModelDownloadError`` (download failure).
        """
        # Validate up front so a bad decoder fails before any I/O.
        default_decoder = _validate_decoder(decoder)

        # 1. Resolve the artifact directory (local path or cache/download).
        artifact_dir = artifacts.resolve(
            path=path,
            model_source=model_source,
            cache_dir=cache_dir,
            model_version=DEFAULT_MODEL_VERSION,
        )

        # 2. Build the recognition charset.
        charset = build_charset()

        # 3. Construct the ONNX session (validates geometry + device up front).
        onnx_path = os.path.join(artifact_dir, artifacts.ONNX_FILENAME)
        session = OnnxSession(onnx_path, device=device)

        # 4. Run the provenance integrity check (provenance/charset/geometry).
        provenance_record = provenance.load_and_verify(artifact_dir, session, charset)

        # 5. Store what predict needs.
        return cls(
            session,
            charset,
            default_decoder=default_decoder,
            provenance_record=provenance_record,
        )

    def predict(
        self,
        image: str,
        *,
        confidence: bool = False,
        decoder: str | None = None,
        beam_width: int = 10,
        lexicon: "Lexicon | None" = None,
        score_fn: "ScoreFn | None" = None,
    ) -> "str | Prediction":
        """Recognize the text in a single pre-cropped word/line image.

        Preprocess to a ``(1, 1, 32, W)`` tensor, run ONNX to get ``(T, 49)``
        logits, decode, and return the text (or a :class:`Prediction` when
        ``confidence=True``). ``decoder`` overrides the recognizer default for
        this call. Raises ``ImageError`` if ``image`` can't be read, or
        ``ConfigError`` for a bad decoder/beam width.
        """
        # Per-call override wins over the recognizer default.
        decoder_name = _validate_decoder(decoder if decoder is not None else self._default_decoder)

        # path -> (1, 1, 32, W) float32 tensor in [-1, 1]; ImageError on failure.
        tensor = Preprocessing_Pipeline().to_tensor(image)

        logits = self._session.run(tensor)

        # Greedy is stateless and reused; beam is per-call since its config
        # (width/lexicon/score_fn) is call-specific and validated at construction.
        if decoder_name == "greedy":
            text, conf = self._greedy.decode(logits, self._charset)
        else:  # "beam"
            beam = BeamSearchDecoder(
                beam_width=beam_width,
                lexicon=lexicon,
                score_fn=score_fn,
            )
            text, conf = beam.decode(logits, self._charset)

        if confidence:
            return Prediction(text=text, confidence=conf)
        return text
