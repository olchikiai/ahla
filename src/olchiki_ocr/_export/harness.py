"""Full_Validation_Harness: ONNX greedy parity vs the trained baseline.

This is the validation/CI tool behind Task 17 (Req 11.1, Correctness Property
P16). It runs the package's **own ONNX greedy inference path** -- the exact same
core components a bare ``pip install olchiki-ocr`` uses at runtime
(:class:`~olchiki_ocr.session.OnnxSession`,
:class:`~olchiki_ocr.preprocessing.Preprocessing_Pipeline`,
:class:`~olchiki_ocr.decoders.GreedyDecoder`, and
:func:`~olchiki_ocr.charset.build_charset`) -- over the full validation set and
scores exact-match word accuracy with the carried-over
:func:`olchiki_ocr._train.evaluator.evaluate`. The measured accuracy is compared
against the trained model's 99.844% baseline within the 0.2 pp tolerance
(Design Decision D7): the parity bar is **>= 99.644%**.

Why it lives under ``_export`` (Task 17.1 location decision)
------------------------------------------------------------
The harness is a parity/fidelity *validation* tool, not part of the core
inference surface. It legitimately uses ``onnxruntime`` (through the core
``OnnxSession``) and runs in the ``[export]`` / CI environment alongside the
export tooling (Task 16 installed ``onnx`` / ``onnxruntime`` there), and the
INT8 re-validation step (:func:`olchiki_ocr._export.export.revalidate_and_record`,
Req 11.3) is exactly the caller that needs FP32-vs-INT8 accuracy numbers. Placing
it in ``olchiki_ocr._export.harness`` keeps it in the export/CI story and out of
the torch-free core: ``import olchiki_ocr`` never imports this module, so the
core import stays engine-light (verified by Task 14.2's smoke test and re-checked
in Task 17.1's verification).

Import isolation (LOAD-BEARING)
-------------------------------
Only core dependencies (``numpy``; and, lazily, the core ``OnnxSession`` which
imports ``onnxruntime``) are needed to *run* the harness. It reuses
:mod:`olchiki_ocr._train.evaluator`, which is a **pure** module (no torch / no
lmdb / no onnxruntime). No torch import happens here -- producing a real ONNX to
validate is the caller's concern (they use :mod:`olchiki_ocr._export.export`).

Validation dataset format
--------------------------
The baseline (DTRB fine-tune) writes a validation manifest ``val_manifest.txt``
under the shared ``output_dir`` -- a UTF-8, tab-separated file of
``<image_path>\t<label>`` lines (one validation sample per line). That manifest
IS the full validation set the 99.844% baseline was measured over (the run
report records 12,836 validation samples, matching the manifest line count), so
reading it reproduces the baseline's sample set faithfully on the ONNX path.
:func:`load_manifest` reads that format; the DTRB clone also stores the same
samples in an LMDB (``output_dir/lmdb/val``) that requires the ``[train]`` extra
(``lmdb``) to read -- the manifest reader is preferred because it needs no extra
dependency and directly names the on-disk images.

The primary entry point :func:`run_full_validation` accepts EITHER a path to the
manifest OR any iterable of ``(image, label)`` pairs (where ``image`` is a
filesystem path or a preprocessed ``(1,1,32,W)`` ndarray), so the parity test
(Task 17.2) can drive it with real or synthetic data.
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

#: The trained model's exact-match word accuracy on the full validation set, in
#: percentage points (design Overview / Req 11.1).
BASELINE_ACCURACY_PP: float = 99.844

#: Parity tolerance in percentage points (Design Decision D7 / Req 11.1).
PARITY_TOLERANCE_PP: float = 0.2

#: The minimum acceptable ONNX greedy accuracy (pp): baseline minus tolerance =
#: 99.644%. Task 17.2 asserts the measured accuracy is >= this value (P16).
PARITY_MIN_ACCURACY_PP: float = BASELINE_ACCURACY_PP - PARITY_TOLERANCE_PP

#: A validation sample: an image (filesystem path or preprocessed ndarray input
#: tensor of shape ``(1,1,32,W)``) paired with its ground-truth label.
ImageOrArray = Union[str, "Path", np.ndarray]
Sample = Tuple[ImageOrArray, str]


@dataclass(frozen=True)
class HarnessResult:
    """Result of a Full_Validation_Harness run.

    Wraps the pure :class:`~olchiki_ocr._train.evaluator.EvalResult` (exact-match
    ``word_accuracy`` in ``[0, 1]`` plus CER) and exposes the accuracy in
    percentage points alongside the baseline/parity bookkeeping so callers (the
    parity test P16, and :func:`revalidate_and_record`) can read the numbers
    directly.

    Attributes:
        eval_result: The underlying evaluator result (``word_accuracy`` in
            ``[0, 1]``, ``cer``, ``sample_count``, ``empty``).
        word_accuracy_pp: ``eval_result.word_accuracy * 100`` -- exact-match
            accuracy in percentage points, directly comparable to
            :data:`BASELINE_ACCURACY_PP`.
        baseline_accuracy_pp: The baseline accuracy compared against
            (:data:`BASELINE_ACCURACY_PP`).
        min_accuracy_pp: The parity bar (:data:`PARITY_MIN_ACCURACY_PP`).
        meets_parity: True iff ``word_accuracy_pp >= min_accuracy_pp`` (P16).
    """

    eval_result: EvalResult
    word_accuracy_pp: float
    baseline_accuracy_pp: float
    min_accuracy_pp: float
    meets_parity: bool

    @property
    def word_accuracy(self) -> float:
        """Exact-match word accuracy in ``[0, 1]`` (from the evaluator)."""
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
    """Load a DTRB-style validation manifest into ``(image_path, label)`` pairs.

    The manifest is the ``val_manifest.txt`` written by the fine-tune data-prep
    phase under the shared ``output_dir``: a UTF-8 text file with one validation
    sample per line, formatted ``<image_path>\\t<label>``. Blank lines are
    skipped.

    Image-path resolution: absolute paths are used as-is. A relative path is
    tried first **as written** (relative to the current working directory, which
    is how the DTRB-written manifest stores paths -- e.g.
    ``output/finetune/synthetic/images/..``, relative to the project root), and
    only if that file does not exist is it resolved **relative to the manifest
    file's directory** (so a manifest copied alongside its images still works).
    This dual strategy handles both the repo's real manifest and relocated
    copies without double-prefixing.

    Args:
        manifest_path: Path to the ``val_manifest.txt`` file.

    Returns:
        A list of ``(image_path, label)`` tuples in manifest order, where
        ``image_path`` is a resolved filesystem path string.

    Raises:
        FileNotFoundError: If ``manifest_path`` does not exist.
        ValueError: If a non-blank line does not contain a tab separator.
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
            # Manifest paths may use backslashes (Windows-written). Normalize the
            # separators, then resolve.
            image_field = image_field.replace("\\", "/")
            image_path = Path(image_field)
            if not image_path.is_absolute():
                # Prefer the path as written (relative to CWD): the DTRB-written
                # manifest stores project-root-relative paths. Fall back to
                # manifest-dir-relative only if the as-written path is missing,
                # so a relocated manifest+images copy still resolves.
                if not image_path.is_file():
                    candidate = base_dir / image_path
                    if candidate.is_file():
                        image_path = candidate
            samples.append((str(image_path), label))
    return samples


def _iter_samples(
    validation_data: Union[str, "Path", Iterable[Sample]],
) -> Iterator[Sample]:
    """Yield ``(image, label)`` samples from a manifest path or an iterable.

    A ``str``/``Path`` is treated as a manifest file and loaded via
    :func:`load_manifest`; anything else is treated as an already-prepared
    iterable of ``(image, label)`` pairs and yielded as-is. This is what lets
    :func:`run_full_validation` accept either the real dataset or test-supplied
    synthetic data.
    """
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
    """Run the ONNX greedy path over the validation set and score it (P16).

    For each ``(image, label)`` sample this runs the package's **own** core
    inference path -- identical to what a runtime ``predict`` does with the
    default greedy decoder:

    1. preprocess the image with the default
       :class:`~olchiki_ocr.preprocessing.Preprocessing_Pipeline`
       (DTRB ``ResizeNormalize`` -> ``(1,1,32,W)`` in ``[-1,1]``). If a sample's
       ``image`` is already an ndarray, it is used directly as the input tensor
       (so callers can pre-batch / cache preprocessing).
    2. run :meth:`~olchiki_ocr.session.OnnxSession.run` to get ``(T, 49)`` (or
       ``(1, T, 49)``) logits;
    3. greedy-decode with :class:`~olchiki_ocr.decoders.GreedyDecoder` against
       ``build_charset(extra_characters)`` to get the predicted text.

    Predictions and labels are then scored with the pure
    :func:`~olchiki_ocr._train.evaluator.evaluate`, and the exact-match
    ``word_accuracy`` is compared against :data:`BASELINE_ACCURACY_PP` within
    :data:`PARITY_TOLERANCE_PP` (parity bar :data:`PARITY_MIN_ACCURACY_PP`).

    Args:
        onnx_path: Path to the ONNX model to validate (FP32 or INT8). Ignored
            when ``session`` is supplied.
        validation_data: Either a path to a ``val_manifest.txt`` (loaded via
            :func:`load_manifest`) or an iterable of ``(image, label)`` pairs,
            where ``image`` is a filesystem path or a preprocessed ndarray.
        device: Compute device for the ONNX session (``None`` -> CPU;
            ``"gpu"``/``"cuda"`` -> CUDA when available). Ignored when
            ``session`` is supplied.
        extra_characters: Extra characters passed to
            :func:`~olchiki_ocr.charset.build_charset` (default ``""`` -> the
            48-character Ol Chiki charset the baseline used).
        session: An optional pre-built session object exposing a
            ``run(tensor) -> logits`` method (duck-typed). When provided,
            ``onnx_path``/``device`` are not used to construct a session -- this
            is the injection point the verification/tests use to drive the
            harness with a fake session and no real ONNX file.
        progress: Optional ``callback(done, total_or_-1)`` invoked after each
            sample for long runs (``total`` is ``-1`` when the sample source is
            a lazy iterable of unknown length).

    Returns:
        A :class:`HarnessResult` with the exact-match accuracy (fraction and
        pp), the baseline/parity bookkeeping, and ``meets_parity``.
    """
    # Lazy core imports: keep this module import-light and avoid constructing a
    # session (which loads onnxruntime) until a run actually happens.
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

    # Materialize a list only if we were given a concrete sequence, so we can
    # report a meaningful total; otherwise stream and report total == -1.
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
    """Return whether a harness result meets the parity bar (P16, Req 11.1).

    Accepts a :class:`HarnessResult`, a raw
    :class:`~olchiki_ocr._train.evaluator.EvalResult` (whose ``word_accuracy`` is
    a fraction in ``[0, 1]``), or a plain accuracy value. A plain value is
    interpreted as a fraction in ``[0, 1]`` when ``<= 1.0`` and otherwise as
    already being in percentage points, so both ``0.99844`` and ``99.844`` work.

    Args:
        result: The harness/eval result or accuracy value to check.
        min_accuracy_pp: The parity bar in pp (default
            :data:`PARITY_MIN_ACCURACY_PP` = 99.644).

    Returns:
        True iff the accuracy is ``>= min_accuracy_pp``.
    """
    if isinstance(result, HarnessResult):
        accuracy_pp = result.word_accuracy_pp
    elif isinstance(result, EvalResult):
        accuracy_pp = result.word_accuracy * 100.0
    else:
        value = float(result)
        accuracy_pp = value * 100.0 if value <= 1.0 else value
    return accuracy_pp >= min_accuracy_pp
