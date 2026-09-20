"""Lexicon type and packaged default Ol Chiki word list.

A :class:`Lexicon` is a validated, ordered, deduplicated word list used to
constrain beam search. ``from_wordlist`` reads a UTF-8 file (one word per line)
and validates each word against the charset, raising ``CharsetError`` naming the
first offending word; ``default`` loads the packaged list. Stdlib-only, so
importing it pulls in no heavy runtime dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Iterator

from .charset import Charset
from .errors import CharsetError

__all__ = ["Lexicon"]

# Packaged default word list, anchored on the top-level package so the lookup
# works from an installed wheel and does not require ``data`` to be a subpackage.
_ANCHOR_PACKAGE = "olchiki_ocr"
_DATA_DIR = "data"
_DEFAULT_WORDLIST = "lexicon.txt"


@dataclass(frozen=True)
class Lexicon:
    """An ordered, deduplicated word list validated against a Charset."""

    words: tuple[str, ...]

    @classmethod
    def from_wordlist(cls, path: str, charset: Charset) -> "Lexicon":
        """Load and validate a word list from a UTF-8 file (one word per line).

        Blank lines are skipped and duplicates dropped in first-appearance
        order. Raises ``CharsetError`` naming the first word with an
        out-of-charset character.
        """
        with open(path, encoding="utf-8") as handle:
            lines = handle.readlines()
        return cls._build(lines, charset)

    @classmethod
    def default(cls, charset: Charset) -> "Lexicon":
        """Load and validate the packaged default Ol Chiki word list.

        Raises ``CharsetError`` naming the first word with an out-of-charset
        character.
        """
        resource = resources.files(_ANCHOR_PACKAGE).joinpath(_DATA_DIR, _DEFAULT_WORDLIST)
        text = resource.read_text(encoding="utf-8")
        return cls._build(text.splitlines(), charset)

    @classmethod
    def _build(cls, lines: "list[str]", charset: Charset) -> "Lexicon":
        """Strip, filter, dedup, and validate raw word-list lines.

        Raises ``CharsetError`` naming the first word with an out-of-charset
        character.
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
