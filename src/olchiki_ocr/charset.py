"""Recognition character-set construction for the ``olchiki_ocr`` package.

This module builds the ordered, deduplicated ``Charset`` the recognition model
can predict. The Charset covers the entire Ol Chiki Unicode block plus any
extra characters supplied by the caller.

Construction order (a pure function of the inputs) is:

1. Every Ol Chiki block code point U+1C50..U+1C7F in ascending order
   (Requirement 8.1, 8.2). The block range is the vendored
   :data:`OL_CHIKI_BLOCK` constant defined below.
2. Each configured extra character in first-appearance order, skipping any
   character already present so each character appears at most once.

Because the order depends only on the inputs, rebuilding the Charset from the
same inputs yields identical characters and identical per-character indices,
giving stable indices across runs for identical inputs. This ordering is
**load-bearing for CTC decode correctness**: the char<->index mapping produced
here must exactly match the ordering the ONNX model was trained with, or
decoding maps emit indices to the wrong characters (Requirement 8.3, 8.4).

The resulting Charset and each character index are recorded in the model
artifact/provenance via :meth:`Charset.snapshot`.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import CharsetError

# Vendored from ``ol_chiki_corpus.charclassifier.OL_CHIKI_BLOCK`` (Design
# Decision D5) so the ``olchiki_ocr`` core is self-contained and does NOT depend
# on the ``ol_chiki_corpus`` package at runtime. The upper bound is exclusive in
# Python ``range()`` and inclusive in prose: this is the Ol Chiki Unicode block
# U+1C50..U+1C7F inclusive (48 code points).
#
# The ascending ordering of this range is LOAD-BEARING for CTC decode
# correctness: the charset char<->index mapping derived from it must match the
# ordering the trained ONNX model uses, or emit indices decode to the wrong
# characters. Do not reorder.
OL_CHIKI_BLOCK = range(0x1C50, 0x1C80)  # U+1C50..U+1C7F inclusive

__all__ = ["Charset", "build_charset"]


@dataclass(frozen=True)
class Charset:
    """An ordered, deduplicated set of recognition characters.

    ``characters`` holds the Charset in construction order: every Ol Chiki block
    code point in ascending order first, then the deduplicated extra characters
    in first-appearance order. The tuple is ordered and every character is
    unique, so ``index_of`` returns a stable index that is identical across runs
    for identical inputs.
    """

    characters: tuple[str, ...]

    def index_of(self, ch: str) -> int:
        """Return the stable index of ``ch`` within the Charset.

        Args:
            ch: A single character known to be in the Charset.

        Returns:
            The zero-based position of ``ch`` in :attr:`characters`.

        Raises:
            ValueError: If ``ch`` is not in the Charset.
        """
        return self.characters.index(ch)

    def as_dtrb_character_arg(self) -> str:
        """Return the ``--character`` string DTRB uses to size the output layer.

        The returned string is the Charset characters joined in order, which
        sets the label alphabet and therefore the number of output classes when
        passed to DTRB's ``train.py``.
        """
        return "".join(self.characters)

    def size(self) -> int:
        """Return the number of characters (the output-class count)."""
        return len(self.characters)

    def snapshot(self) -> list[tuple[int, str]]:
        """Return ``(index, character)`` pairs for the artifact/provenance.

        For every character, its stable index and the character itself, in
        Charset order.
        """
        return [(index, ch) for index, ch in enumerate(self.characters)]


def build_charset(extra_characters: str = "") -> Charset:
    """Build the ordered, deduplicated recognition character set.

    Order: every Ol Chiki block code point U+1C50..U+1C7F in ascending order,
    followed by each configured extra character in first-appearance order,
    skipping any already present so each character appears at most once. The
    order is a pure function of the inputs, giving stable indices across runs
    for identical inputs.

    Args:
        extra_characters: A string of additional characters to append to the
            Charset. Each character must be a single Unicode code point.

    Returns:
        The constructed :class:`Charset`.

    Raises:
        CharsetError: If an extra character is not encodable as a single Unicode
            code point (``len(ch) != 1``); the error names the offending value.
    """
    # Ascending Ol Chiki block code points as the base of the Charset.
    characters: list[str] = [chr(code_point) for code_point in OL_CHIKI_BLOCK]
    seen: set[str] = set(characters)

    # Append extra characters in first-appearance order, deduplicating so each
    # character appears at most once.
    for ch in extra_characters:
        # A Python iteration over a str yields one code point at a time, so a
        # length other than 1 signals a value that is not a single code point
        # (empty or multi-code-point input). Guard here so build_charset is
        # safe even when called directly.
        if len(ch) != 1:
            raise CharsetError(ch, "Extra character is not a single Unicode code point")
        if ch in seen:
            continue
        characters.append(ch)
        seen.add(ch)

    return Charset(characters=tuple(characters))
