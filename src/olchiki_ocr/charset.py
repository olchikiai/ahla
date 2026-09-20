"""Recognition character-set construction for the ``olchiki_ocr`` package.

Builds the ordered, deduplicated ``Charset`` the model can predict: every Ol
Chiki block code point (U+1C50..U+1C7F) in ascending order, then any extra
characters in first-appearance order (deduplicated). Order is a pure function of
the inputs, so indices are stable across runs. This ordering is load-bearing for
CTC decode correctness — the char<->index mapping must match what the ONNX model
was trained with.
"""

from __future__ import annotations

from dataclasses import dataclass

from .errors import CharsetError

# Vendored so the core doesn't depend on ol_chiki_corpus at runtime: the Ol Chiki
# Unicode block U+1C50..U+1C7F inclusive (48 code points). The ascending order is
# LOAD-BEARING for CTC decode correctness (must match the trained model); do not
# reorder.
OL_CHIKI_BLOCK = range(0x1C50, 0x1C80)  # U+1C50..U+1C7F inclusive

__all__ = ["Charset", "build_charset"]


@dataclass(frozen=True)
class Charset:
    """An ordered, deduplicated tuple of recognition characters.

    ``characters`` is in construction order (block code points, then extras), so
    ``index_of`` returns a stable index for identical inputs.
    """

    characters: tuple[str, ...]

    def index_of(self, ch: str) -> int:
        """Return the zero-based index of ``ch``, or raise ``ValueError`` if absent."""
        return self.characters.index(ch)

    def as_dtrb_character_arg(self) -> str:
        """Return the characters joined in order (DTRB's ``--character`` arg)."""
        return "".join(self.characters)

    def size(self) -> int:
        """Return the number of characters (the output-class count)."""
        return len(self.characters)

    def snapshot(self) -> list[tuple[int, str]]:
        """Return ``(index, character)`` pairs in Charset order for provenance."""
        return [(index, ch) for index, ch in enumerate(self.characters)]


def build_charset(extra_characters: str = "") -> Charset:
    """Build the ordered, deduplicated recognition character set.

    Ol Chiki block code points (ascending) followed by ``extra_characters`` in
    first-appearance order, deduplicated. Raises ``CharsetError`` (naming the
    value) if an extra character isn't a single Unicode code point.
    """
    characters: list[str] = [chr(code_point) for code_point in OL_CHIKI_BLOCK]
    seen: set[str] = set(characters)

    # Append extras in first-appearance order, deduplicating.
    for ch in extra_characters:
        # len != 1 means not a single code point (empty/multi-char input).
        if len(ch) != 1:
            raise CharsetError(ch, "Extra character is not a single Unicode code point")
        if ch in seen:
            continue
        characters.append(ch)
        seen.add(ch)

    return Charset(characters=tuple(characters))
