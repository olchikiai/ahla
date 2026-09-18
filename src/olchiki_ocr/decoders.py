"""CTC decoders over per-timestep class scores (design: ``decoders.py``; Req 7).

This module implements the package's own CTC decoding so the core inference tier
stays torch-free (Req 2.2, 2.3). It depends only on numpy plus the pure-stdlib
CTC-convention constants in :mod:`olchiki_ocr._ctc` and the :class:`~olchiki_ocr.charset.Charset`
type; it must NOT import torch, onnxruntime, easyocr, or cv2.

``GreedyDecoder`` (the default decoder, Req 7.1) performs the canonical DTRB
``CTCLabelConverter`` greedy decode: per-timestep argmax, collapse consecutive
duplicate class indices, then drop the CTC blank (index 0 per confirmed Design
Decision D8, see :mod:`olchiki_ocr._ctc`). Surviving emit indices map to charset
characters via ``charset.characters[i - 1]``.

Input / logits-vs-probabilities convention
-------------------------------------------
``GreedyDecoder.decode`` expects the **raw model output logits** for a single
image, matching the predict flow where ``OnnxSession.run`` returns logits of
shape ``(T, NUM_CLASSES)``. A leading batch dimension of 1 (shape
``(1, T, NUM_CLASSES)``) is accepted and squeezed. The decoder applies a
numerically stable softmax over the class dimension internally so the confidence
is a meaningful probability. Softmax is monotonic, so it does not change the
per-timestep argmax path; passing already-normalized probabilities instead of
logits therefore yields the same decoded text and an only-slightly-shifted
confidence (softmax is approximately idempotent for decode purposes), but the
documented and intended input is logits.

Confidence (Design Decision D2, Req 1.4)
----------------------------------------
Greedy confidence is the mean over the **emitted** timesteps of the per-timestep
maximum softmax probability, in ``[0, 1]``. "Emitted timesteps" are exactly the
argmax positions that survive collapse + blank-removal, i.e. the timesteps that
each contribute one character to the decoded text. If the decoded text is empty
(no emitted timesteps), the confidence is ``0.0``.
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
    """Numerically stable softmax over the last (class) axis.

    Subtracts the per-row max before exponentiating so large logits do not
    overflow. Returns an array of the same shape whose last axis sums to 1.
    """
    shifted = scores - np.max(scores, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)


class GreedyDecoder:
    """Greedy (best-path) CTC decoder — the default decoder (Req 7.1).

    Stateless: a single instance can decode any number of inputs. Construction
    takes no arguments so the recognizer can create one cheaply.
    """

    def decode(self, probs: np.ndarray, charset: "Charset") -> tuple[str, float]:
        """Greedily decode one image's CTC output to ``(text, confidence)``.

        Args:
            probs: The model output **logits** for a single image, of shape
                ``(T, NUM_CLASSES)`` or ``(1, T, NUM_CLASSES)`` (a leading batch
                dimension of 1 is squeezed). ``NUM_CLASSES`` is 49 (48 emit
                classes + 1 CTC blank at index 0). A stable softmax is applied
                over the class axis internally, so raw logits are expected.
            charset: The recognition :class:`~olchiki_ocr.charset.Charset`; emit
                class index ``i`` (1..48) maps to ``charset.characters[i - 1]``.

        Returns:
            A ``(text, confidence)`` tuple. ``text`` is the decoded string
            (consecutive duplicates collapsed and blanks removed). ``confidence``
            is the mean per-timestep max softmax probability over the emitted
            timesteps, in ``[0, 1]``; ``0.0`` when ``text`` is empty.

        Raises:
            ValueError: If the input does not have a class dimension of
                ``NUM_CLASSES`` after squeezing a leading batch dim of 1, or is
                not 2-dimensional.
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

        # Softmax over the class dimension so per-timestep max is a probability.
        probabilities = _softmax(scores.astype(np.float64))

        # Per-timestep best class (argmax path) and its probability.
        best_indices = np.argmax(probabilities, axis=1)
        best_probs = probabilities[np.arange(probabilities.shape[0]), best_indices]

        # CTC greedy collapse: walk the argmax path, skip a class that repeats
        # the immediately previous class (collapse consecutive duplicates), then
        # skip the blank. A surviving position is an "emitted timestep".
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


# ---------------------------------------------------------------------------
# Task 7.4: beam search decoder + pluggable ScoreFn hook (Req 7.2-7.8).
# ---------------------------------------------------------------------------

#: Inclusive valid range for the beam width (Req 7.3, 7.4).
_MIN_BEAM_WIDTH = 1
_MAX_BEAM_WIDTH = 100


@runtime_checkable
class ScoreFn(Protocol):
    """Pluggable prefix-scoring hook for :class:`BeamSearchDecoder` (Req 7.6, 7.8).

    A ``ScoreFn`` is any callable that maps a candidate decoded ``prefix`` (the
    string decoded so far) to a real-valued score, where a **higher** score is
    better. This is the extension point for a user-supplied language model: a
    KenLM or neural LM can be wrapped as a ``ScoreFn``, but the core ships none
    and requires none (Req 7.7) — the protocol is the only contract.

    Example::

        def unigram_score(prefix: str) -> float:
            return sum(my_log_probs.get(ch, -10.0) for ch in prefix)

    """

    def __call__(self, prefix: str) -> float:  # pragma: no cover - protocol
        """Return a score for ``prefix``; higher is better."""
        ...


def _validate_beam_width(beam_width: object) -> int:
    """Validate a beam width, raising :class:`ConfigError` naming the value.

    A valid beam width is an ``int`` (note: ``bool`` is rejected even though it
    subclasses ``int``) in the inclusive range ``1..100`` (Req 7.3). Anything
    else — a non-integer such as a float or string, or an out-of-range integer —
    raises ``ConfigError(parameter="beam_width", ...)`` whose message names the
    offending value (Req 7.4). ``ConfigError`` is chosen over a bare
    ``ValueError`` because it names the offending parameter/value and integrates
    with the CLI exit-code mapping (exit code 2).

    Args:
        beam_width: The candidate beam width to validate.

    Returns:
        The validated ``int`` beam width.

    Raises:
        ConfigError: If ``beam_width`` is not an integer, is a bool, or lies
            outside the inclusive range ``1..100``; the message names the value.
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
    """CTC prefix beam-search decoder (Req 7.2-7.8).

    Implements standard CTC prefix beam search over the per-timestep class
    scores: each beam tracks its accumulated blank-ending and non-blank-ending
    prefix probabilities, equal prefixes are merged, and only the top
    ``beam_width`` beams (by total probability) are kept at each timestep.

    Optional constraints:

    - **Lexicon** (Req 7.5): when a non-empty :class:`~olchiki_ocr.lexicon.Lexicon`
      is supplied, the final decoded output is constrained to lexicon members —
      see :meth:`decode` for the exact policy (Property 7).
    - **ScoreFn** (Req 7.6): when a :class:`ScoreFn` is supplied, final
      candidates are ranked by ``(score_fn(prefix), ctc_prob)`` compared
      lexicographically, so equal scores fall back to the higher accumulated CTC
      probability (Property 9).

    Configuration reduces to greedy in the trivial case: ``beam_width == 1`` with
    no lexicon and no score_fn produces the same text as :class:`GreedyDecoder`
    for any input (Property 6).
    """

    def __init__(
        self,
        beam_width: int = 10,
        lexicon: "Lexicon | None" = None,
        score_fn: "ScoreFn | None" = None,
    ) -> None:
        """Construct a beam-search decoder.

        Args:
            beam_width: Number of beams retained per timestep. Defaults to 10
                (Req 7.2). Must be an integer in the inclusive range ``1..100``
                (Req 7.3); validated at construction (Req 7.4).
            lexicon: Optional lexicon constraining the final output (Req 7.5).
            score_fn: Optional prefix scorer used to rank candidates (Req 7.6).

        Raises:
            ConfigError: If ``beam_width`` is not an integer in ``1..100``; the
                message names the offending value.
        """
        self.beam_width = _validate_beam_width(beam_width)
        self.lexicon = lexicon
        self.score_fn = score_fn

    def decode(self, probs: np.ndarray, charset: "Charset") -> tuple[str, float]:
        """Beam-search decode one image's CTC output to ``(text, confidence)``.

        Input handling matches :class:`GreedyDecoder`: ``probs`` are the model
        output **logits** of shape ``(T, NUM_CLASSES)`` or ``(1, T, NUM_CLASSES)``
        (a leading batch dim of 1 is squeezed); a stable softmax is applied over
        the class axis to obtain per-timestep probabilities.

        Lexicon-constraint policy (Req 7.5, Property 7): if a non-empty lexicon
        is configured, the final beams are filtered to those whose decoded
        string is a lexicon member (the empty string is always allowed) and the
        best surviving candidate is returned. If no beam's string is in the
        lexicon, the empty string is returned. This guarantees the invariant
        "any non-empty beam-search output is a member of the lexicon."

        Ranking / tie-break (Req 7.6, Property 9): the final candidate ranking
        key is, in descending priority, ``(score_fn(prefix), ctc_prob)`` compared
        lexicographically when a ``score_fn`` is set, else just ``ctc_prob``.
        Because ``ctc_prob`` is the secondary key, equal ``score_fn`` scores are
        broken by preferring the higher accumulated CTC probability. ``ctc_prob``
        is a beam's total probability ``p_blank + p_non_blank``.

        Confidence (Design Decision D2, Req 1.4): the winning path's
        length-normalized probability ``ctc_prob ** (1 / len(text))`` for
        non-empty ``text`` — the geometric mean of per-character probability
        mass, which lies in ``[0, 1]`` since ``ctc_prob`` does. Empty output ->
        ``0.0``.

        Args:
            probs: Model output logits, shape ``(T, NUM_CLASSES)`` or
                ``(1, T, NUM_CLASSES)``.
            charset: The recognition :class:`~olchiki_ocr.charset.Charset`.

        Returns:
            A ``(text, confidence)`` tuple.

        Raises:
            ValueError: If the input is not 2-D after squeezing a leading batch
                dim of 1, or its class dimension is not ``NUM_CLASSES``.
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

        # Trivial configuration reduces to greedy best-path decoding
        # (Property 6): with a single beam and no lexicon/score_fn there is no
        # candidate ranking to perform, so the correct and cheapest behavior is
        # the canonical CTC best-path decode (argmax -> collapse dups -> drop
        # blank), which is exactly what GreedyDecoder produces. Prefix beam
        # search with width 1 sums probability mass over paths and is a
        # genuinely different algorithm that need not agree with best-path, so
        # we delegate rather than approximate.
        if self.beam_width == 1 and self.lexicon is None and self.score_fn is None:
            return self._greedy_best_path(probabilities, charset)

        beams = self._run_prefix_beam_search(probabilities)

        # Convert each surviving prefix (tuple of emit-class indices) to text and
        # its total CTC probability, then select the winner under the configured
        # policy.
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
        """Best-path (greedy) decode used for the trivial beam configuration.

        Produces exactly the :class:`GreedyDecoder` text (argmax per timestep,
        collapse consecutive duplicates, drop the blank) so Property 6 holds.
        The confidence follows the beam's length-normalized-probability
        convention (Design Decision D2): the geometric mean over the emitted
        timesteps of the per-timestep max softmax probability. This equals the
        winning best-path probability ``prod(emitted max-probs)`` raised to
        ``1/len(text)``, which lies in ``[0, 1]``. Empty output -> ``0.0``.

        Args:
            probabilities: Per-timestep class probabilities, shape ``(T, C)``.
            charset: The recognition Charset.

        Returns:
            The ``(text, confidence)`` best-path result.
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

        # Geometric mean of emitted per-timestep max probs = length-normalized
        # path probability in [0, 1].
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

        Each beam is keyed by its prefix as a tuple of **emit-class** indices
        (the model class index, 1..48; the blank is never part of a prefix). The
        value is ``(p_blank, p_non_blank)``: the probability that the prefix ends
        in a blank vs. a non-blank at the current timestep. Only the top
        ``beam_width`` beams by total probability survive each timestep.

        Args:
            probabilities: Per-timestep class probabilities, shape ``(T, C)``.

        Returns:
            A mapping ``prefix -> (p_blank, p_non_blank)`` for surviving beams.
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

                # Case 1: emit a blank -> prefix unchanged, lands in blank state.
                slot = _slot(prefix)
                slot[0] += prefix_total * blank_p

                # Case 2: repeat the last char -> prefix unchanged, non-blank
                # state. A repeat can only extend from the previous non-blank
                # mass (p_non_blank); extending from p_blank would create a NEW
                # occurrence (handled in case 3).
                if prefix:
                    last = prefix[-1]
                    slot[1] += p_non_blank * float(step[last])

                # Case 3: emit each real (non-blank) character.
                for cls in range(1, NUM_CLASSES):
                    char_p = float(step[cls])
                    if char_p == 0.0:
                        continue
                    if prefix and prefix[-1] == cls:
                        # Same char as last: only the blank-ending mass can start
                        # a new occurrence (non-blank-ending mass would collapse).
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
        """Pick the winning ``(text, ctc_prob)`` under lexicon + score_fn policy.

        Applies the lexicon filter (Req 7.5) then ranks by
        ``(score_fn(text), ctc_prob)`` (Req 7.6). See :meth:`decode` for the
        exact documented policy.

        Args:
            candidates: ``(text, ctc_prob)`` pairs for the surviving beams.

        Returns:
            The winning ``(text, ctc_prob)``; ``("", 0.0)`` if no candidate
            survives the lexicon filter with a non-empty string.
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
