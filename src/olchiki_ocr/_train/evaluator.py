"""Recognition-quality metrics (pure; no framework dependency).

``evaluate()`` computes exact-match word_accuracy and CER over parallel
prediction/label lists, handling the empty set gracefully. Reused by the ONNX
Full_Validation_Harness.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["EvalResult", "levenshtein", "evaluate"]


@dataclass(frozen=True)
class EvalResult:
    """Aggregate metrics: CER, word_accuracy, sample_count, and empty flag."""

    cer: float
    word_accuracy: float
    sample_count: int
    empty: bool = False


def levenshtein(a: str, b: str) -> int:
    """Return the Levenshtein edit distance between two strings.

    Two-row DP using O(min(len(a), len(b))) space.
    """
    if a == b:
        return 0
    # Keep the inner (per-row) dimension the smaller of the two for less memory.
    if len(a) < len(b):
        a, b = b, a
    if not b:
        return len(a)

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
    """Compute CER and exact-match word_accuracy over parallel lists.

    ``cer`` is total edit distance over total label chars (0.0 when there are no
    label chars); ``word_accuracy`` is the exact-match fraction (0.0 when empty).

    Raises ValueError if the two lists differ in length.
    """
    if len(predictions) != len(labels):
        raise ValueError(
            "predictions and labels must be parallel lists of equal length "
            f"(got {len(predictions)} predictions and {len(labels)} labels)"
        )

    sample_count = len(labels)
    if sample_count == 0:
        # Empty set: both metrics 0.0, flagged.
        return EvalResult(cer=0.0, word_accuracy=0.0, sample_count=0, empty=True)

    total_distance = 0
    total_label_chars = 0
    exact_matches = 0
    for prediction, label in zip(predictions, labels):
        total_distance += levenshtein(prediction, label)
        total_label_chars += len(label)
        if prediction == label:
            exact_matches += 1

    # Guard divide-by-zero when every Label is empty.
    cer = (total_distance / total_label_chars) if total_label_chars > 0 else 0.0
    word_accuracy = exact_matches / sample_count

    return EvalResult(
        cer=cer,
        word_accuracy=word_accuracy,
        sample_count=sample_count,
        empty=False,
    )
