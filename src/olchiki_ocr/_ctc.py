"""Confirmed CTC output-layer convention for the trained Ol Chiki model.

This module records the **empirically confirmed** CTC blank / emit-class layout
of the trained DTRB ``None-VGG-BiLSTM-CTC`` model (Design Decision D8). The
decoders (Task 7.x) and the provenance integrity check (Task 9.1) import these
constants so the emit-index -> character mapping is defined in exactly one
place. Getting this wrong maps every emit index to the wrong character, so the
convention below was verified against ground truth rather than assumed.

Pure standard library only: importing this module must NOT pull in torch,
onnxruntime, numpy, or Pillow, so the core inference tier stays torch-free.

Confirmed convention
--------------------
The model output is a per-timestep class-score sequence of shape
``(T, NUM_CLASSES)`` where ``NUM_CLASSES == 49``:

- Index ``0`` is the CTC blank token (``[CTCblank]``).
- Indices ``1..48`` are the 48 real charset characters, in charset order.

Therefore the mapping from a model output class index ``i`` to a character is::

    i == 0            -> CTC blank (emits nothing)
    i in 1..48        -> charset.characters[i - 1]

Equivalently, a charset character at ``charset`` index ``j`` (0-based, 0..47)
occupies model output index ``j + 1``.

Empirical evidence (verified 2024, this task)
---------------------------------------------
1. **DTRB label-converter source** — the model is trained by the pinned DTRB
   clone (``external/deep-text-recognition-benchmark``, commit
   ``e2117f2fb882b3c6085030500a260c113be27a63``). Its ``CTCLabelConverter``
   (``utils.py``) fixes the convention::

       for i, char in enumerate(dict_character):
           # NOTE: 0 is reserved for 'CTCblank' token required by CTCLoss
           self.dict[char] = i + 1
       self.character = ['[CTCblank]'] + dict_character  # index 0 == blank

   and its ``decode`` drops index 0 and collapses repeats::

       if t[i] != 0 and (not (i > 0 and t[i - 1] == t[i])):  # remove blank + repeats
           char_list.append(self.character[t[i]])

   So the blank is at index 0 and real characters occupy indices 1..N.

2. **Output-layer size** — DTRB ``train.py`` sets
   ``opt.num_class = len(converter.character)`` where ``converter.character``
   is ``['[CTCblank]'] + dict_character``; i.e. ``num_class = charset_size + 1``.
   For the 48-character Ol Chiki alphabet this is 49 (a code comment in the same
   file explicitly refers to the "Ol Chiki 49-class head").

3. **Trained weights** — loading the trained checkpoint
   (``models/ol_chiki_g2/model/ol_chiki_g2.pth``) with torch on CPU, the
   final CTC linear layer ``module.Prediction.weight`` has shape ``(49, 256)``,
   i.e. ``out_features == 49``. This weight-shape check was RUN and confirms 49
   output classes.

4. **Charset size & ordering** — the artifact ``character_list`` (in
   ``models/ol_chiki_g2/user_network/ol_chiki_g2.yaml``) and the recorded ``charset`` (in
   ``models/ol_chiki_g2/provenance.json``) are each 48 characters, all within
   U+1C50..U+1C7F, and are **byte-for-byte identical** to
   ``"".join(build_charset().characters)``. The charset ordering the model was
   trained with therefore matches the core's ``build_charset()`` ordering; no
   contradiction was found.

WARNING: the charset ordering is load-bearing. These constants and
``build_charset()`` must agree with the ordering the ONNX/torch model was
trained with, or emit indices decode to the wrong characters. Do not change
``BLANK_INDEX`` or the mapping rule without re-confirming against the trained
model.
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
    """Map a model output class index to its 0-based charset index.

    Model output index ``i`` (in ``1..48``) maps to ``charset.characters[i - 1]``.

    Args:
        class_index: A model output class index in the inclusive range
            ``1..EMIT_CLASSES`` (i.e. a non-blank emit class).

    Returns:
        The 0-based index into ``Charset.characters`` (``0..EMIT_CLASSES - 1``).

    Raises:
        ValueError: If ``class_index`` is the blank index or otherwise outside
            the emit range ``1..EMIT_CLASSES``.
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
    """Map a 0-based charset index to its model output class index.

    Charset index ``j`` (in ``0..47``) occupies model output index ``j + 1``.

    Args:
        charset_index: A 0-based index into ``Charset.characters``
            (``0..EMIT_CLASSES - 1``).

    Returns:
        The model output class index (``1..EMIT_CLASSES``).

    Raises:
        ValueError: If ``charset_index`` is outside ``0..EMIT_CLASSES - 1``.
    """
    if not (0 <= charset_index < EMIT_CLASSES):
        raise ValueError(
            f"charset index {charset_index} is out of range 0..{EMIT_CLASSES - 1}"
        )
    return charset_index + 1
