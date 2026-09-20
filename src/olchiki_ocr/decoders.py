"""CTC decoders over per-timestep class scores.

Pure-numpy so the core stays torch-free (no torch/onnxruntime/easyocr/cv2).

CTC layout: the blank is at index 0, and a surviving emit index ``i`` maps to
``charset.characters[i - 1]``. Greedy decode is argmax per timestep -> collapse
consecutive dups -> drop the blank. Decoders take raw logits ``(T, NUM_CLASSES)``
(a leading batch dim of 1 is squeezed) and apply a stable softmax internally so
the confidence is a real probability.

Confidence: greedy = mean per-timestep max softmax over the emitted timesteps;
``0.0`` for empty output.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, runtime_checkable

import numpy as np

from ._ctc import BLANK_INDEX, NUM_CLASSES, emit_index_to_charset_index
from .errors import ConfigError

if TYPE_CHECKING:  # avoid any runtime import cost / cycle; charset.py is light
    from .charset import Charset
    from .lexicon import Lexicon

__all__ = ["GreedyDecoder", "BeamSearchDecoder", "ScoreFn"]


def _softmax(scores: np.ndarray) -> np.ndarray:
    """Numerically stable softmax over the last (class) axis."""
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class GreedyDecoder:
    """Greedy (best-path) CTC decoder — the default. Stateless and reusable."""

    def decode(self, probs: np.ndarray, charset: "Charset") -> tuple[str, float]:
        """Greedily decode one image's CTC logits to ``(text, confidence)``.

        ``probs`` are raw logits of shape ``(T, NUM_CLASSES)`` or
        ``(1, T, NUM_CLASSES)``. Raises ``ValueError`` if not 2-D (after
        squeezing a leading batch dim of 1) or the class dim isn't NUM_CLASSES.
        """
        scores = np.asarray(probs)

        # Handle an optional leading batch dimension of 1: (1, T, C) -> (T, C).
        if scores.ndim == 3 and scores.shape[0] == 1:
            scores = scores[0]

        if scores.ndim != 2:
            raise ValueError(
                f"expected logits of shape (T, {NUM_CLASSES}) or "
                f"(1, T, {NUM_CLASSES}); got array with shape {np.asarray(probs).shape}"
            )
        if scores.shape[1] != NUM_CLASSES:
            raise ValueError(
                f"expected class dimension {NUM_CLASSES}; got {scores.shape[1]}"
            )

        probabilities = _softmax(scores.astype(np.float64))

        best_indices = np.argmax(probabilities, axis=1)
        best_probs = probabilities[np.arange(probabilities.shape[0]), best_indices]

        # Greedy collapse: skip a class that repeats the previous one, then skip
        # the blank. Surviving positions are the "emitted timesteps".
        chars: list[str] = []
        emitted_probs: list[float] = []
        previous = -1  # sentinel: no previous class
        for t, cls in enumerate(best_indices):
            cls_int = int(cls)
            if cls_int != previous:
                if cls_int != BLANK_INDEX:
                    chars.append(charset.characters[emit_index_to_charset_index(cls_int)])
                    emitted_probs.append(float(best_probs[t]))
            previous = cls_int

        text = "".join(chars)

        # Confidence = mean max-softmax prob over emitted timesteps; 0.0 if none.
        if emitted_probs:
            confidence = float(np.mean(emitted_probs))
        else:
            confidence = 0.0

        return text, confidence


# Beam search decoder + pluggable ScoreFn hook.

#: Inclusive valid range for the beam width.
_MIN_BEAM_WIDTH = 1
_MAX_BEAM_WIDTH = 100


@runtime_checkable
class ScoreFn(Protocol):
    """Pluggable prefix-scoring hook for :class:`BeamSearchDecoder`.

    Any callable mapping a decoded ``prefix`` to a score where higher is better;
    the extension point for a user-supplied LM. The core ships none.

    Example::

        def unigram_score(prefix: str) -> float:
            return sum(my_log_probs.get(ch, -10.0) for ch in prefix)

    """

    def __call__(self, prefix: str) -> float:  # pragma: no cover - protocol
        """Return a score for ``prefix``; higher is better."""
        ...


def _validate_beam_width(beam_width: object) -> int:
    """Validate a beam width, raising :class:`ConfigError` naming the value.

    Must be an ``int`` (``bool`` rejected) in the inclusive range ``1..100``.
    """
    if isinstance(beam_width, bool) or not isinstance(beam_width, int):
        raise ConfigError(
            "beam_width",
            f"beam_width must be an integer in 1..{_MAX_BEAM_WIDTH}; "
            f"got {beam_width!r}",
        )
    if not (_MIN_BEAM_WIDTH <= beam_width <= _MAX_BEAM_WIDTH):
        raise ConfigError(
            "beam_width",
            f"beam_width must be in the inclusive range "
            f"{_MIN_BEAM_WIDTH}..{_MAX_BEAM_WIDTH}; got {beam_width!r}",
        )
    return beam_width


class BeamSearchDecoder:
    """CTC prefix beam-search decoder.

    Standard CTC prefix beam search: each beam tracks blank-ending and
    non-blank-ending prefix probabilities, equal prefixes merge, and only the
    top ``beam_width`` beams survive each timestep.

    An optional lexicon constrains the final output to its members; an optional
    :class:`ScoreFn` ranks candidates by ``(score_fn(prefix), ctc_prob)``. The
    trivial config (``beam_width == 1``, no lexicon/score_fn) reduces to greedy.
    """

    def __init__(
        self,
        beam_width: int = 10,
        lexicon: "Lexicon | None" = None,
        score_fn: "ScoreFn | None" = None,
    ) -> None:
        """Construct a beam-search decoder.

        ``beam_width`` (default 10) must be an int in ``1..100``, validated here
        (raises ``ConfigError``). ``lexicon`` and ``score_fn`` are optional.
        """
        self.beam_width = _validate_beam_width(beam_width)
        self.lexicon = lexicon
        self.score_fn = score_fn

    def decode(self, probs: np.ndarray, charset: "Charset") -> tuple[str, float]:
        """Beam-search decode one image's CTC logits to ``(text, confidence)``.

        Input handling matches :class:`GreedyDecoder` (logits, optional leading
        batch dim of 1, internal softmax).

        Lexicon: when a non-empty lexicon is set, the final beams are filtered to
        lexicon members (empty string always allowed) and the best is returned;
        if none match, the empty string is returned — so any non-empty output is
        always a lexicon member.

        Ranking: ``(score_fn(prefix), ctc_prob)`` when a score_fn is set (equal
        scores broken by higher CTC prob), else ``ctc_prob`` alone, where
        ``ctc_prob = p_blank + p_non_blank``.

        Confidence: the winning path's length-normalized probability
        ``ctc_prob ** (1 / len(text))`` (geometric mean, in ``[0, 1]``); ``0.0``
        for empty output.

        Raises ``ValueError`` if not 2-D (after squeezing) or the class dim isn't
        ``NUM_CLASSES``.
        """
        scores = np.asarray(probs)

        if scores.ndim == 3 and scores.shape[0] == 1:
            scores = scores[0]

        if scores.ndim != 2:
            raise ValueError(
                f"expected logits of shape (T, {NUM_CLASSES}) or "
                f"(1, T, {NUM_CLASSES}); got array with shape {np.asarray(probs).shape}"
            )
        if scores.shape[1] != NUM_CLASSES:
            raise ValueError(
                f"expected class dimension {NUM_CLASSES}; got {scores.shape[1]}"
            )

        probabilities = _softmax(scores.astype(np.float64))

        # Trivial config: with one beam and no lexicon/score_fn there's nothing
        # to rank, so delegate to greedy best-path. (Width-1 prefix beam search
        # sums mass over paths and need not agree with best-path.)
        if self.beam_width == 1 and self.lexicon is None and self.score_fn is None:
            return self._greedy_best_path(probabilities, charset)

        beams = self._run_prefix_beam_search(probabilities)

        # Convert each surviving prefix to (text, total CTC prob), then pick the
        # winner under the configured policy.
        candidates: list[tuple[str, float]] = []
        for prefix, (p_blank, p_non_blank) in beams.items():
            total = p_blank + p_non_blank
            text = "".join(
                charset.characters[emit_index_to_charset_index(idx)] for idx in prefix
            )
            candidates.append((text, total))

        text, total = self._select_winner(candidates)

        if not text:
            return "", 0.0

        # Length-normalized probability = geometric mean of per-char prob mass.
        confidence = float(total ** (1.0 / len(text)))
        # Guard against tiny floating error pushing just outside [0, 1].
        if confidence < 0.0:
            confidence = 0.0
        elif confidence > 1.0:
            confidence = 1.0
        return text, confidence

    def _greedy_best_path(
        self, probabilities: np.ndarray, charset: "Charset"
    ) -> tuple[str, float]:
        """Best-path (greedy) decode used for the trivial beam config.

        Produces the same text as :class:`GreedyDecoder`, but with the beam's
        length-normalized confidence (geometric mean of emitted max-softmax
        probs, in ``[0, 1]``; ``0.0`` for empty output).
        """
        best_indices = np.argmax(probabilities, axis=1)
        best_probs = probabilities[np.arange(probabilities.shape[0]), best_indices]

        chars: list[str] = []
        emitted_probs: list[float] = []
        previous = -1
        for t, cls in enumerate(best_indices):
            cls_int = int(cls)
            if cls_int != previous:
                if cls_int != BLANK_INDEX:
                    chars.append(charset.characters[emit_index_to_charset_index(cls_int)])
                    emitted_probs.append(float(best_probs[t]))
            previous = cls_int

        text = "".join(chars)
        if not text:
            return "", 0.0

        # Geometric mean of emitted max probs = length-normalized path prob.
        log_sum = float(np.sum(np.log(emitted_probs)))
        confidence = float(np.exp(log_sum / len(text)))
        if confidence < 0.0:
            confidence = 0.0
        elif confidence > 1.0:
            confidence = 1.0
        return text, confidence

    def _run_prefix_beam_search(
        self, probabilities: np.ndarray
    ) -> "dict[tuple[int, ...], tuple[float, float]]":
        """Run CTC prefix beam search, returning surviving beams.

        Beams are keyed by prefix (a tuple of emit-class indices; the blank is
        never in a prefix) with value ``(p_blank, p_non_blank)`` — the prob the
        prefix ends in blank vs. non-blank. Only the top ``beam_width`` by total
        prob survive each timestep.
        """
        # Empty prefix starts with all probability mass in the blank state.
        beams: dict[tuple[int, ...], tuple[float, float]] = {(): (1.0, 0.0)}

        num_timesteps = probabilities.shape[0]
        for t in range(num_timesteps):
            step = probabilities[t]
            next_beams: dict[tuple[int, ...], list[float]] = {}

            def _slot(prefix: tuple[int, ...]) -> list[float]:
                slot = next_beams.get(prefix)
                if slot is None:
                    slot = [0.0, 0.0]  # [p_blank, p_non_blank]
                    next_beams[prefix] = slot
                return slot

            blank_p = float(step[BLANK_INDEX])

            for prefix, (p_blank, p_non_blank) in beams.items():
                prefix_total = p_blank + p_non_blank

                # Case 1: emit a blank -> prefix unchanged, blank state.
                slot = _slot(prefix)
                slot[0] += prefix_total * blank_p

                # Case 2: repeat the last char -> prefix unchanged, non-blank
                # state. Only p_non_blank can extend a repeat; extending from
                # p_blank would be a NEW occurrence (case 3).
                if prefix:
                    last = prefix[-1]
                    slot[1] += p_non_blank * float(step[last])

                # Case 3: emit each real (non-blank) character.
                for cls in range(1, NUM_CLASSES):
                    char_p = float(step[cls])
                    if char_p == 0.0:
                        continue
                    if prefix and prefix[-1] == cls:
                        # Same char as last: only blank-ending mass starts a new
                        # occurrence (non-blank mass would collapse).
                        added = p_blank * char_p
                    else:
                        added = prefix_total * char_p
                    if added == 0.0:
                        continue
                    new_prefix = prefix + (cls,)
                    new_slot = _slot(new_prefix)
                    new_slot[1] += added

            # Prune to the top ``beam_width`` beams by total probability.
            ranked = sorted(
                next_beams.items(),
                key=lambda item: item[1][0] + item[1][1],
                reverse=True,
            )
            beams = {
                prefix: (slot[0], slot[1])
                for prefix, slot in ranked[: self.beam_width]
            }

        return beams

    def _select_winner(self, candidates: list[tuple[str, float]]) -> tuple[str, float]:
        """Pick the winning ``(text, ctc_prob)`` under the lexicon + score_fn policy.

        Applies the lexicon filter then ranks by ``(score_fn(text), ctc_prob)``.
        See :meth:`decode` for the exact policy.
        """
        if not candidates:
            return "", 0.0

        # Lexicon constraint: keep only candidates whose non-empty text is a
        # lexicon member. The empty string is always permitted (it is a valid
        # "no output" and never violates the invariant).
        if self.lexicon is not None:
            filtered = [
                (text, total)
                for text, total in candidates
                if text == "" or text in self.lexicon
            ]
            if not filtered:
                # No surviving beam is a lexicon member -> return empty so the
                # invariant "non-empty output is a lexicon member" always holds.
                return "", 0.0
            candidates = filtered

        if self.score_fn is not None:
            score_fn = self.score_fn

            def _key(item: tuple[str, float]) -> tuple[float, float]:
                text, total = item
                return (float(score_fn(text)), total)

        else:

            def _key(item: tuple[str, float]) -> tuple[float, float]:
                return (item[1], item[1])

        return max(candidates, key=_key)
