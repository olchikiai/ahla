"""Lexicon type and packaged default Ol Chiki word list (Req 8.5-8.7).

A :class:`Lexicon` is a validated, ordered, deduplicated word list used to
constrain the ``BeamSearchDecoder`` (design: Components -> ``lexicon.py``;
Design Decision D4). Every word in a Lexicon is guaranteed to contain only
characters present in the recognition :class:`~olchiki_ocr.charset.Charset`, so
beam candidates constrained to the lexicon are always decodable.

Loading behavior:

- :meth:`Lexicon.from_wordlist` reads a UTF-8 word-list file (one word per
  line), stripping surrounding whitespace and skipping blank lines. Duplicate
  words are dropped while the first-appearance order is preserved, so the stored
  ``words`` tuple is ordered and unique.
- :meth:`Lexicon.default` loads the project's Ol Chiki word list shipped as
  packaged data under ``olchiki_ocr/data/`` (via :mod:`importlib.resources`, so
  it works from an installed wheel) and validates it against the supplied
  Charset.

Validation (Req 8.7): if any word contains a character outside the Charset, a
:class:`~olchiki_ocr.errors.CharsetError` is raised naming the first offending
word. ``CharsetError`` is the natural choice because it already carries the
offending value in its message and lives in the package error hierarchy.

This module is intentionally torch/onnxruntime/easyocr/cv2-free: it uses only
the standard library (:mod:`importlib.resources`) plus the package's own
``charset``/``errors`` modules, so importing it never pulls in a heavy runtime
dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Iterator

from .charset import Charset
from .errors import CharsetError

__all__ = ["Lexicon"]

# Packaged default word list location (Design Decision D4). Shipped under
# ``olchiki_ocr/data/`` so it is available from an installed wheel via
# ``importlib.resources``. We anchor on the top-level ``olchiki_ocr`` package
# (which is a regular package) and join the ``data`` subdirectory, so the
# lookup does not depend on ``data`` itself being an importable subpackage.
_ANCHOR_PACKAGE = "olchiki_ocr"
_DATA_DIR = "data"
_DEFAULT_WORDLIST = "lexicon.txt"


@dataclass(frozen=True)
class Lexicon:
    """An ordered, deduplicated word list validated against a Charset.

    ``words`` holds each accepted word in first-appearance order with duplicates
    removed, so the tuple is ordered and every word is unique. Every word is
    guaranteed to consist solely of characters present in the Charset it was
    validated against.
    """

    words: tuple[str, ...]

    @classmethod
    def from_wordlist(cls, path: str, charset: Charset) -> "Lexicon":
        """Load and validate a word list from a UTF-8 file.

        The file is read one word per line; surrounding whitespace is stripped
        and blank lines are skipped. Duplicate words are dropped while the
        first-appearance order is preserved.

        Args:
            path: Filesystem path to a UTF-8 word-list file (one word per line).
            charset: The recognition Charset every word must validate against.

        Returns:
            A :class:`Lexicon` whose ``words`` are the deduplicated, in-order
            valid words from the file.

        Raises:
            CharsetError: If a word contains a character outside ``charset``;
                the error names the first offending word (Req 8.7).
        """
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
        return cls._build(lines, charset)

    @classmethod
    def default(cls, charset: Charset) -> "Lexicon":
        """Load the packaged default Ol Chiki word list (Req 8.6).

        Reads the project's Ol Chiki word list shipped as packaged data under
        ``olchiki_ocr/data/`` using :mod:`importlib.resources` (so it works from
        an installed wheel), then validates it against ``charset`` via the same
        path as :meth:`from_wordlist`.

        Args:
            charset: The recognition Charset every word must validate against.

        Returns:
            The default :class:`Lexicon`.

        Raises:
            CharsetError: If a word contains a character outside ``charset``;
                the error names the first offending word (Req 8.7).
        """
        resource = resources.files(_ANCHOR_PACKAGE).joinpath(_DATA_DIR, _DEFAULT_WORDLIST)
        text = resource.read_text(encoding="utf-8")
        return cls._build(text.splitlines(), charset)

    @classmethod
    def _build(cls, lines: "list[str]", charset: Charset) -> "Lexicon":
        """Strip, filter, dedup, and validate raw word-list lines.

        Args:
            lines: Raw lines (as read from a file or resource).
            charset: The Charset every word must validate against.

        Returns:
            The constructed :class:`Lexicon`.

        Raises:
            CharsetError: Naming the first word containing an out-of-charset
                character (Req 8.7).
        """
        allowed = frozenset(charset.characters)
        seen: set[str] = set()
        words: list[str] = []
        for line in lines:
            word = line.strip()
            if not word:
                continue
            for ch in word:
                if ch not in allowed:
                    raise CharsetError(
                        word,
                        "Lexicon word contains a character outside the charset",
                    )
            if word in seen:
                continue
            seen.add(word)
            words.append(word)
        return cls(words=tuple(words))

    def __contains__(self, word: object) -> bool:
        """Return whether ``word`` is a member of the Lexicon."""
        return word in self.words

    def __len__(self) -> int:
        """Return the number of unique words in the Lexicon."""
        return len(self.words)

    def __iter__(self) -> Iterator[str]:
        """Iterate the Lexicon words in first-appearance order."""
        return iter(self.words)
