"""Recognition-quality metrics for the ``[train]`` tier and the harness.

This module is *pure*: it depends on no external framework (no DTRB, no torch,
no onnxruntime) and computes its metrics solely from the provided prediction and
label strings. Isolating the metric math here makes it exhaustively testable
independent of training.

It is reused by the Full_Validation_Harness (Task 17): its ``evaluate(...)``
API (exact-match ``word_accuracy`` plus CER) is kept intact so the ONNX greedy
path can be scored against the baseline.

Two metrics are computed over an evaluation set of parallel ``predictions`` and
``labels`` lists:

* **Character_Error_Rate (CER)** - the total Levenshtein edit distance between
  each prediction and its Label, divided by the total number of Label
  characters.
* **Word_Accuracy** - the count of Samples whose prediction exactly equals its
  Label, divided by the number of Samples.

Degenerate cases are handled without raising: an empty evaluation set yields CER
0.0 and Word_Accuracy 0.0 with the empty flag conveyed on the result; a
non-empty set whose Labels contain zero characters in total (all Labels empty
strings) also yields CER 0.0 rather than dividing by zero.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["EvalResult", "levenshtein", "evaluate"]


@dataclass(frozen=True)
class EvalResult:
    """Aggregate evaluation metrics over an evaluation set.

    Attributes:
        cer: Character_Error_Rate - total Levenshtein distance divided by the
            total Label character count. A non-negative real number; defined as
            0.0 when there are no Label characters.
        word_accuracy: Proportion of Samples whose prediction exactly equals its
            Label, in the range 0.0 to 1.0 inclusive; 0.0 for an empty set.
        sample_count: The number of (prediction, label) pairs evaluated.
        empty: True when the evaluation set contained zero Samples, so the
            Run_Report can note the empty Validation_Set.
    """

    cer: float
    word_accuracy: float
    sample_count: int
    empty: bool = False


def levenshtein(a: str, b: str) -> int:
    """Return the Levenshtein edit distance between two strings.

    The edit distance is the minimum number of single-character insertions,
    deletions, or substitutions required to transform ``a`` into ``b``. The
    function is symmetric (``levenshtein(a, b) == levenshtein(b, a)``), returns
    0 for identical strings, and equals ``len(a)`` when ``b`` is empty (and
    vice versa).

    Implemented with the standard two-row dynamic-programming recurrence, which
    uses O(min(len(a), len(b))) additional space.
    """
    if a == b:
        return 0
    # Keep the inner (per-row) dimension the smaller of the two for less memory.
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

    # previous_row[j] = edit distance between a[:i] (i == 0 initially) and b[:j].
    previous_row = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current_row = [i]
        for j, cb in enumerate(b, start=1):
            insertion = current_row[j - 1] + 1
            deletion = previous_row[j] + 1
            substitution = previous_row[j - 1] + (0 if ca == cb else 1)
            current_row.append(min(insertion, deletion, substitution))
        previous_row = current_row
    return previous_row[-1]


def evaluate(predictions: list[str], labels: list[str]) -> EvalResult:
    """Compute aggregate CER and Word_Accuracy over an evaluation set.

    ``predictions`` and ``labels`` are parallel lists: ``predictions[i]`` is the
    model's predicted sequence for the Sample whose ground-truth Label is
    ``labels[i]``. They MUST be of equal length.

    Returns an :class:`EvalResult` where:

    * ``cer`` is the sum over all pairs of ``levenshtein(prediction, label)``
      divided by the sum over all pairs of ``len(label)``. When the total Label
      character count is zero - either because the set is empty or because every
      Label is the empty string - ``cer`` is defined as 0.0 to avoid division by
      zero.
    * ``word_accuracy`` is the number of pairs where ``prediction == label``
      divided by the number of pairs; 0.0 for an empty set.
    * ``sample_count`` is the number of pairs, and ``empty`` is True exactly when
      that count is zero.

    Raises:
        ValueError: if ``predictions`` and ``labels`` differ in length.
    """
    if len(predictions) != len(labels):
        raise ValueError(
            "predictions and labels must be parallel lists of equal length "
            f"(got {len(predictions)} predictions and {len(labels)} labels)"
        )

    sample_count = len(labels)
    if sample_count == 0:
        # Empty evaluation set: both metrics 0.0, flagged for the Run_Report.
        return EvalResult(cer=0.0, word_accuracy=0.0, sample_count=0, empty=True)

    total_distance = 0
    total_label_chars = 0
    exact_matches = 0
    for prediction, label in zip(predictions, labels):
        total_distance += levenshtein(prediction, label)
        total_label_chars += len(label)
        if prediction == label:
            exact_matches += 1

    # Guard division-by-zero when every Label is empty (total label chars == 0).
    cer = (total_distance / total_label_chars) if total_label_chars > 0 else 0.0
    word_accuracy = exact_matches / sample_count

    return EvalResult(
        cer=cer,
        word_accuracy=word_accuracy,
        sample_count=sample_count,
        empty=False,
    )
