"""Full validation harness: ONNX greedy parity vs the trained baseline.

Runs the package's own ONNX greedy inference path (the same core components a
bare install uses at runtime) over the full validation set and scores
exact-match word accuracy with the pure ``_train.evaluator.evaluate``. The
measured accuracy is compared against the trained baseline of 99.844% within a
0.2 pp tolerance, so the parity bar is >= 99.644%.

It lives under ``_export`` because it is a validation tool that uses
``onnxruntime`` (via the core ``OnnxSession``) and is the counterpart to the
INT8 re-validation step -- ``import olchiki_ocr`` never imports it, keeping the
core torch-free. Only ``numpy`` and the core (lazily) are needed to run it.

Validation data is a DTRB-style ``val_manifest.txt`` (UTF-8, tab-separated
``<image_path>\t<label>`` lines) read by :func:`load_manifest`, or any iterable
of ``(image, label)`` pairs where ``image`` is a path or a preprocessed
``(1,1,32,W)`` ndarray, so tests can drive it with real or synthetic data.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Iterator, Optional, Sequence, Tuple, Union

import numpy as np

from .._train.evaluator import EvalResult, evaluate

__all__ = [
    "BASELINE_ACCURACY_PP",
    "PARITY_TOLERANCE_PP",
    "PARITY_MIN_ACCURACY_PP",
    "HarnessResult",
    "load_manifest",
    "run_full_validation",
    "check_parity",
]

#: Trained model's exact-match word accuracy on the full validation set (pp).
BASELINE_ACCURACY_PP: float = 99.844

#: Parity tolerance (pp).
PARITY_TOLERANCE_PP: float = 0.2

#: Minimum acceptable ONNX greedy accuracy (pp): baseline minus tolerance.
PARITY_MIN_ACCURACY_PP: float = BASELINE_ACCURACY_PP - PARITY_TOLERANCE_PP

#: A validation sample: an image (path or ``(1,1,32,W)`` ndarray) plus its label.
ImageOrArray = Union[str, "Path", np.ndarray]
Sample = Tuple[ImageOrArray, str]


@dataclass(frozen=True)
class HarnessResult:
    """Result of a validation harness run (accuracy in pp + parity bookkeeping)."""

    eval_result: EvalResult
    word_accuracy_pp: float
    baseline_accuracy_pp: float
    min_accuracy_pp: float
    meets_parity: bool

    @property
    def word_accuracy(self) -> float:
        """Exact-match word accuracy in ``[0, 1]``."""
        return self.eval_result.word_accuracy

    @property
    def sample_count(self) -> int:
        """Number of validation samples scored."""
        return self.eval_result.sample_count

    @property
    def accuracy_delta_pp(self) -> float:
        """``word_accuracy_pp - baseline_accuracy_pp`` (negative = below baseline)."""
        return self.word_accuracy_pp - self.baseline_accuracy_pp


def load_manifest(manifest_path: Union[str, "Path"]) -> list[Sample]:
    """Load a DTRB-style ``val_manifest.txt`` into ``(image_path, label)`` pairs.

    UTF-8, tab-separated ``<image_path>\\t<label>`` lines; blank lines skipped.
    Relative paths are tried as-written (relative to CWD, as the manifest stores
    them) and only fall back to manifest-dir-relative if that misses, so
    relocated manifest+images copies still resolve. Raises ``FileNotFoundError``
    if the manifest is missing and ``ValueError`` on a line without a tab.
    """
    manifest = Path(manifest_path)
    if not manifest.is_file():
        raise FileNotFoundError(f"validation manifest not found: {manifest_path}")

    base_dir = manifest.parent
    samples: list[Sample] = []
    with manifest.open("r", encoding="utf-8") as handle:
        for lineno, raw in enumerate(handle, start=1):
            line = raw.rstrip("\n").rstrip("\r")
            if not line.strip():
                continue
            if "\t" not in line:
                raise ValueError(
                    f"malformed manifest line {lineno} (expected "
                    f"'<image_path>\\t<label>'): {line!r}"
                )
            image_field, label = line.split("\t", 1)
            # Normalize Windows-written backslashes before resolving.
            image_field = image_field.replace("\\", "/")
            image_path = Path(image_field)
            if not image_path.is_absolute():
                # Prefer the as-written path; fall back to manifest-dir-relative.
                if not image_path.is_file():
                    candidate = base_dir / image_path
                    if candidate.is_file():
                        image_path = candidate
            samples.append((str(image_path), label))
    return samples


def _iter_samples(
    validation_data: Union[str, "Path", Iterable[Sample]],
) -> Iterator[Sample]:
    """Yield ``(image, label)`` samples from a manifest path or an iterable."""
    if isinstance(validation_data, (str, Path)):
        yield from load_manifest(validation_data)
    else:
        for sample in validation_data:
            yield sample


def run_full_validation(
    onnx_path: str,
    validation_data: Union[str, "Path", Iterable[Sample]],
    *,
    device: Optional[str] = None,
    extra_characters: str = "",
    session: Optional[object] = None,
    progress: Optional[Callable[[int, int], None]] = None,
) -> HarnessResult:
    """Run the ONNX greedy path over the validation set and score it.

    For each sample: preprocess (or use a supplied ndarray tensor), run the ONNX
    session, and greedy-decode against ``build_charset(extra_characters)``.
    Predictions and labels are scored with ``evaluate`` and compared to the
    baseline within the parity tolerance. Pass ``session`` to inject a
    duck-typed ``run(tensor) -> logits`` object (bypassing ``onnx_path``/
    ``device``), e.g. for tests. Returns a :class:`HarnessResult`.
    """
    # Lazy core imports: keep import-light and defer loading onnxruntime.
    from ..charset import build_charset
    from ..decoders import GreedyDecoder
    from ..preprocessing import Preprocessing_Pipeline

    charset = build_charset(extra_characters)
    decoder = GreedyDecoder()
    pipeline = Preprocessing_Pipeline()

    run_session = session
    if run_session is None:
        from ..session import OnnxSession

        run_session = OnnxSession(onnx_path, device=device)

    # Materialize only a concrete sequence (to report a total); else stream
    # with total == -1.
    samples_source: Union[Sequence[Sample], Iterator[Sample]]
    total = -1
    if isinstance(validation_data, (str, Path)):
        materialized = load_manifest(validation_data)
        samples_source = materialized
        total = len(materialized)
    elif isinstance(validation_data, Sequence):
        samples_source = validation_data
        total = len(validation_data)
    else:
        samples_source = _iter_samples(validation_data)

    predictions: list[str] = []
    labels: list[str] = []

    for done, (image, label) in enumerate(samples_source, start=1):
        if isinstance(image, np.ndarray):
            tensor = image
        else:
            tensor = pipeline.to_tensor(str(image))

        logits = run_session.run(tensor)
        text, _confidence = decoder.decode(np.asarray(logits), charset)

        predictions.append(text)
        labels.append(label)

        if progress is not None:
            progress(done, total)

    eval_result = evaluate(predictions, labels)
    word_accuracy_pp = eval_result.word_accuracy * 100.0

    return HarnessResult(
        eval_result=eval_result,
        word_accuracy_pp=word_accuracy_pp,
        baseline_accuracy_pp=BASELINE_ACCURACY_PP,
        min_accuracy_pp=PARITY_MIN_ACCURACY_PP,
        meets_parity=word_accuracy_pp >= PARITY_MIN_ACCURACY_PP,
    )


def check_parity(
    result: Union[HarnessResult, EvalResult, float],
    *,
    min_accuracy_pp: float = PARITY_MIN_ACCURACY_PP,
) -> bool:
    """Return whether a result meets the parity bar (default 99.644 pp).

    Accepts a :class:`HarnessResult`, an ``EvalResult`` (``word_accuracy`` a
    fraction), or a plain value read as a fraction when ``<= 1.0`` and otherwise
    as pp, so both ``0.99844`` and ``99.844`` work.
    """
    if isinstance(result, HarnessResult):
        accuracy_pp = result.word_accuracy_pp
    elif isinstance(result, EvalResult):
        accuracy_pp = result.word_accuracy * 100.0
    else:
        value = float(result)
        accuracy_pp = value * 100.0 if value <= 1.0 else value
    return accuracy_pp >= min_accuracy_pp
