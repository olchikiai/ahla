# olchiki_ocr packaged data

Packaged resources for `olchiki_ocr` (design: Package layout -> `data/`).

This directory holds resources shipped inside the wheel.

## `lexicon.txt` — default Ol Chiki word list

The project's Ol Chiki word list, loaded on demand via `Lexicon.default(...)`
(Design Decision D4). One word per line, UTF-8.

Source: the training corpus `output/ol_chiki_corpus.txt` produced by the
project's corpus pipeline (the same word list referenced by
`finetune_config.py` as `word_list_path`). When copied here it was **filtered to
Ol-Chiki-only words**: any line containing a character outside the Ol Chiki
Unicode block U+1C50..U+1C7F (e.g. zero-width joiners U+200C/U+200D, the Bengali
danda U+09F7, soft hyphens, Latin letters, or digits) was dropped so the file
loads via `Lexicon.default(build_charset())` without raising. Words are
deduplicated with first-appearance order preserved.

Every remaining word validates against `build_charset()` (all characters in
U+1C50..U+1C7F).
