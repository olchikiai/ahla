"""Confirmed CTC output-layer convention for the trained Ol Chiki model.

CORRECTNESS-CRITICAL. Confirmed convention for the DTRB None-VGG-BiLSTM-CTC
model: BLANK_INDEX=0, EMIT_CLASSES=48, NUM_CLASSES=49. Model output index 0 is
the CTC blank; indices 1..48 map to ``charset.characters[i - 1]`` (charset
order). Equivalently charset index ``j`` occupies output index ``j + 1``.

This ordering is load-bearing: getting it wrong decodes every emit index to the
wrong character. Do not change the constants or mapping without re-confirming
against the trained model. Pure stdlib, so the core inference tier stays
torch-free.
"""

from __future__ import annotations

__all__ = [
    "BLANK_INDEX",
    "EMIT_CLASSES",
    "NUM_CLASSES",
    "emit_index_to_charset_index",
    "charset_index_to_emit_index",
]

#: Index of the CTC blank token in the model output. Confirmed: 0.
BLANK_INDEX: int = 0

#: Number of real charset characters the model can emit (the charset size).
#: Confirmed 48 (the Ol Chiki block U+1C50..U+1C7F).
EMIT_CLASSES: int = 48

#: Total number of model output classes = EMIT_CLASSES + 1 CTC blank.
#: Confirmed 49 (matches the trained ``Prediction`` layer ``out_features``).
NUM_CLASSES: int = EMIT_CLASSES + 1


def emit_index_to_charset_index(class_index: int) -> int:
    """Map a model output class index (1..48) to its 0-based charset index.

    Raises ValueError for the blank index or anything outside 1..EMIT_CLASSES.
    """
    if class_index == BLANK_INDEX:
        raise ValueError(
            f"class index {class_index} is the CTC blank; it maps to no character"
        )
    if not (1 <= class_index <= EMIT_CLASSES):
        raise ValueError(
            f"emit class index {class_index} is out of range 1..{EMIT_CLASSES}"
        )
    return class_index - 1


def charset_index_to_emit_index(charset_index: int) -> int:
    """Map a 0-based charset index (0..47) to its model output class index (j+1).

    Raises ValueError if ``charset_index`` is outside 0..EMIT_CLASSES - 1.
    """
    if not (0 <= charset_index < EMIT_CLASSES):
        raise ValueError(
            f"charset index {charset_index} is out of range 0..{EMIT_CLASSES - 1}"
        )
    return charset_index + 1
