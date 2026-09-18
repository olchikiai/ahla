"""Primary public API: :class:`ModelRecognizer` and :class:`Prediction`.

This module wires the core inference tier together (design: Components ->
``ModelRecognizer``; Req 1, 2, 13). It implements the two public entry points
described in the design's ``from_pretrained`` resolution flow and ``predict``
flow diagrams:

* :meth:`ModelRecognizer.from_pretrained` resolves a Model_Artifact directory
  (:func:`olchiki_ocr.artifacts.resolve`), constructs an
  :class:`~olchiki_ocr.session.OnnxSession` over the bundled ``model.onnx``
  (which validates ONNX geometry -- rank 4, class dim 49 -- and raises
  ``ModelArtifactError``/``DeviceError`` as appropriate; Req 2.6, 2.7, 13.3),
  runs the provenance integrity check
  (:func:`olchiki_ocr.provenance.load_and_verify`, enforcing
  provenance/charset/geometry agreement; Req 8.4), and stores what
  :meth:`predict` needs.

* :meth:`ModelRecognizer.predict` runs the per-image pipeline: preprocess
  (:class:`~olchiki_ocr.preprocessing.Preprocessing_Pipeline`, raising
  ``ImageError`` on an unreadable image; Req 1.7) -> ONNX ``run`` -> CTC decode
  (:class:`~olchiki_ocr.decoders.GreedyDecoder` by default, or
  :class:`~olchiki_ocr.decoders.BeamSearchDecoder` when ``decoder="beam"``) ->
  return a bare ``str`` (Req 1.3) or a :class:`Prediction` with a confidence in
  ``[0, 1]`` when ``confidence=True`` (Req 1.4).

Session eager-vs-lazy (design reconciliation): the design's ``from_pretrained``
diagram runs the provenance integrity check *at resolution time*, and that check
needs the validated ONNX geometry that ``OnnxSession`` produces at construction.
The design also has a "session lazily created" nice-to-have note, but making the
session lazy would either skip the up-front integrity check or force a second
ONNX load inside ``load_and_verify``. Correctness of the integrity check takes
precedence, so this implementation constructs the ``OnnxSession`` eagerly in
``from_pretrained`` (matching the resolution-flow diagram) and reuses that one
session in ``predict``. ``OnnxSession`` construction is a cheap metadata/geometry
load, so construction stays inexpensive.

MODEL_VERSION handling: ``artifacts.resolve`` needs a ``model_version`` *before*
provenance is loaded, because the cache directory is named
``~/.cache/olchiki-ocr/<model_version>/``. We therefore define a module-level
:data:`DEFAULT_MODEL_VERSION` constant -- the default servable Model_Version,
read from the bundled model's ``provenance.json`` (``"0.1.0"``, model
``ol_chiki_g2``) -- and pass it to ``resolve``. For a ``path=`` load the
``model_version`` is irrelevant to the result (the local directory is used
verbatim and no cache path is computed), so the constant only affects cache-dir
naming for the download/cache path. The value is confirmable and can be updated
when a new servable model version is published.

Import hygiene: this module imports ``onnxruntime`` only *indirectly* via
:mod:`olchiki_ocr.session` (the core inference engine -- expected and correct).
It never imports torch, easyocr, or cv2. numpy arrives transitively through the
session/preprocessing/decoder modules.
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

#: The default servable Model_Version, read from the bundled model's
#: ``provenance.json`` (model ``ol_chiki_g2``). ``artifacts.resolve`` needs a
#: model_version to name the cache directory
#: (``~/.cache/olchiki-ocr/<model_version>/``) *before* provenance is loaded, so
#: it cannot come from the provenance record itself. For a ``path=`` load this
#: value is irrelevant (the local dir is used verbatim, no cache path is
#: computed); for the download/cache path it names the versioned artifact +
#: cache subdir. Confirmable -- bump when a new servable model is published.
DEFAULT_MODEL_VERSION = "0.1.0"

#: Valid decoder names accepted by ``from_pretrained``/``predict`` (Req 1.5,
#: 7.1). "greedy" is the package default; "beam" is opt-in.
_VALID_DECODERS = frozenset({"greedy", "beam"})


@dataclass(frozen=True)
class Prediction:
    """Structured recognition result (design: Data Models -> ``Prediction``).

    Attributes:
        text: The recognized text (Req 1.3).
        confidence: The recognition confidence in ``[0, 1]``, or ``None`` when
            not requested. Greedy confidence is the mean per-timestep max
            softmax over emitted timesteps; beam confidence is the winning
            path's length-normalized probability (Design Decision D2, Req 1.4).
    """

    text: str
    confidence: float | None = None


def _validate_decoder(name: str) -> str:
    """Return ``name`` if it is a recognized decoder, else raise ``ConfigError``.

    Args:
        name: The requested decoder name.

    Returns:
        The validated decoder name.

    Raises:
        ConfigError: If ``name`` is not ``"greedy"`` or ``"beam"``; the message
            names the offending value (Req 1.5, 12.5).
    """
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

        Args:
            session: A constructed, geometry-validated ``OnnxSession``.
            charset: The recognition :class:`~olchiki_ocr.charset.Charset`.
            default_decoder: The default decoder name (``"greedy"`` or
                ``"beam"``); validated here.
            provenance_record: The verified provenance, retained for
                introspection (optional).

        Raises:
            ConfigError: If ``default_decoder`` is not a recognized decoder.
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

        Resolution/verification order (design ``from_pretrained`` flow):

        1. **resolve** the artifact directory via
           :func:`olchiki_ocr.artifacts.resolve` (``path=`` -> local, no
           network, Req 1.6/4.7; else cache-hit reuse or download, Req 1.2/4.x),
           passing :data:`DEFAULT_MODEL_VERSION` for cache-dir naming.
        2. build the recognition :class:`~olchiki_ocr.charset.Charset`.
        3. **session** -- construct an ``OnnxSession`` over
           ``<artifact_dir>/model.onnx``. This validates ONNX geometry (input
           rank 4, output class dim 49) and provider/device availability,
           raising ``ModelArtifactError`` (Req 2.6, 2.7) or ``DeviceError``
           (Req 13.3) as appropriate.
        4. **load_and_verify** -- run
           :func:`olchiki_ocr.provenance.load_and_verify` to enforce
           provenance/charset/geometry agreement, raising ``ModelArtifactError``
           on mismatch (Req 8.4).
        5. **store** the session, charset, default decoder, and provenance in a
           new recognizer.

        Args:
            path: Optional local artifact directory (offline load; Req 1.6).
            model_source: Optional ``Release_Host`` override (Req 4.2).
            cache_dir: Optional Model_Cache override (Req 4.6).
            device: ``None``/``"cpu"`` -> CPU; ``"gpu"``/``"cuda"`` -> GPU
                (Req 13).
            decoder: Default decoder for this recognizer (``"greedy"`` default,
                ``"beam"`` opt-in; Req 7.1). Validated up front.

        Returns:
            A ready-to-use :class:`ModelRecognizer`.

        Raises:
            ConfigError: If ``decoder`` is not a recognized decoder name.
            ModelArtifactError: On a missing/invalid artifact, bad ONNX
                geometry, or a provenance/charset mismatch.
            DeviceError: If a GPU device is requested but unavailable.
            ModelDownloadError: On a download/verify failure (non-``path=``).
        """
        # Validate the requested default decoder before doing any I/O so an
        # obvious misconfiguration fails fast (Req 1.5).
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

        Pipeline (design ``predict`` flow): preprocess the image to a
        ``(1, 1, 32, W)`` tensor, run the ONNX session to get ``(T, 49)`` logits,
        decode with the selected decoder, and return the text (or a
        :class:`Prediction`).

        Args:
            image: Filesystem path to a readable image (Req 1.3).
            confidence: When ``True``, return a :class:`Prediction` with a
                confidence in ``[0, 1]``; when ``False`` (default), return the
                bare recognized ``str`` (Req 1.3, 1.4).
            decoder: Per-call decoder override (``"greedy"``/``"beam"``); when
                ``None``, the recognizer's configured default is used, otherwise
                the per-call value wins (Req 1.5).
            beam_width: Beam width when the beam decoder is used (default 10;
                validated inside :class:`BeamSearchDecoder`, Req 7.2-7.4).
            lexicon: Optional lexicon constraining beam output (Req 7.5).
            score_fn: Optional prefix scorer for beam ranking (Req 7.6).

        Returns:
            The recognized text as a ``str`` (``confidence=False``) or a
            :class:`Prediction` carrying ``text`` and ``confidence``
            (``confidence=True``).

        Raises:
            ImageError: If ``image`` cannot be read or decoded (Req 1.7). Raised
                by the preprocessing pipeline, naming the offending path.
            ConfigError: If ``decoder`` names an unknown decoder, or the beam
                width is invalid.
            ModelArtifactError: Propagated from the session on a runtime model
                failure.
        """
        # Choose the decoder for this call: per-call override wins, else the
        # recognizer default (Req 1.5).
        decoder_name = _validate_decoder(decoder if decoder is not None else self._default_decoder)

        # Preprocess: path -> (1, 1, 32, W) float32 tensor in [-1, 1]. A plain
        # default pipeline reproduces the training-time preprocessing; unreadable
        # images raise ImageError(path) here (Req 1.7).
        tensor = Preprocessing_Pipeline().to_tensor(image)

        # Run the ONNX model -> logits (T, 49).
        logits = self._session.run(tensor)

        # Decode. Greedy is stateless and reused; beam is constructed per call
        # because its configuration (width/lexicon/score_fn) is call-specific
        # and its width is validated at construction (Req 7.4).
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
