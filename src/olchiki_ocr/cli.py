"""``olchiki-ocr`` console-script entry point (design: Components -> ``cli.py``;
Req 1.8, 12.3, 12.4).

This module implements the ``olchiki-ocr`` console script: it recognizes a
pre-cropped word/line image and prints the recognized text to stdout. The
single public callable :func:`main` is what the ``[project.scripts]`` entry
point ``olchiki-ocr = "olchiki_ocr.cli:main"`` (declared in
``packaging/olchiki-ocr/pyproject.toml``) invokes; setuptools console scripts
use ``main``'s integer return value as the process exit code.

Top-level error handling (design: Error Handling -> "CLI mapping"; Req 12.3,
12.4): the whole recognition operation is wrapped so that a raised
:class:`~olchiki_ocr.errors.OcrTrainerError` (any subclass -- ``ImageError``,
``ModelArtifactError``, ``ModelDownloadError``, ``DeviceError``, ``ConfigError``,
...) is caught at the top level, its ``str(err)`` message is printed to stderr,
and its distinct ``err.exit_code`` is returned. These expected typed errors do
not leak a traceback. Any other (unexpected, non-``OcrTrainerError``) exception
maps to exit code 1 with a concise stderr message. A successful recognition
prints the text and returns 0.

Import hygiene (Req 2.2, 2.3): importing this module is cheap. The typed error
base is imported eagerly (pure stdlib, needed by the handler), but
:class:`~olchiki_ocr.recognizer.ModelRecognizer` is imported *inside*
:func:`_run` so that ``import olchiki_ocr.cli`` never pulls in
``onnxruntime``/``numpy``/``Pillow`` (they arrive only when a recognition is
actually performed). This module never imports torch, easyocr, or cv2.
"""

from __future__ import annotations

import argparse
import sys
from typing import Sequence

from .errors import OcrTrainerError

__all__ = ["main"]

#: Program name used in ``--help`` / usage output (matches the console script).
_PROG = "olchiki-ocr"

#: Exit code returned for an unexpected (non-typed) failure (Req 12.4).
_UNEXPECTED_EXIT_CODE = 1


def _build_parser() -> argparse.ArgumentParser:
    """Construct the argument parser for the ``olchiki-ocr`` console script.

    The surface mirrors the :class:`~olchiki_ocr.recognizer.ModelRecognizer`
    API: a positional image path plus options that map onto
    ``from_pretrained(...)`` and ``predict(...)``.

    Returns:
        The configured :class:`argparse.ArgumentParser`.
    """
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description=(
            "Recognize Ol Chiki (Santali) text in a pre-cropped word/line image "
            "and print it to stdout."
        ),
    )
    parser.add_argument(
        "image",
        help="Filesystem path to the (pre-cropped) image to recognize.",
    )
    # --model / --path: a local artifact directory -> from_pretrained(path=...),
    # i.e. an offline load with no network access (Req 1.6, 4.7).
    parser.add_argument(
        "--model",
        "--path",
        dest="model",
        metavar="PATH",
        default=None,
        help=(
            "Path to a local Model_Artifact directory to load offline (no "
            "network). Omit to resolve/download the default model."
        ),
    )
    parser.add_argument(
        "--device",
        choices=["cpu", "gpu", "cuda"],
        default=None,
        help="Compute device: 'cpu' (default), or 'gpu'/'cuda' (requires the "
        "[gpu] extra).",
    )
    parser.add_argument(
        "--decoder",
        choices=["greedy", "beam"],
        default="greedy",
        help="CTC decoder to use (default: greedy).",
    )
    parser.add_argument(
        "--beam-width",
        type=int,
        default=10,
        metavar="INT",
        help="Beam width when --decoder beam is used (default: 10).",
    )
    parser.add_argument(
        "--confidence",
        action="store_true",
        help="Also print the recognition confidence as 'text\\tconfidence'.",
    )
    parser.add_argument(
        "--cache-dir",
        metavar="PATH",
        default=None,
        help="Override the Model_Cache directory used for download/cache.",
    )
    parser.add_argument(
        "--model-source",
        metavar="URL",
        default=None,
        help="Override the Release_Host used to download the Model_Artifact.",
    )
    return parser


def _run(args: argparse.Namespace, out) -> None:
    """Perform the recognition described by parsed ``args`` and print the result.

    Any :class:`~olchiki_ocr.errors.OcrTrainerError` (or unexpected exception)
    raised here propagates to :func:`main`, which maps it to an exit code. On
    success the recognized text (optionally with confidence) is written to
    ``out``.

    Args:
        args: Parsed command-line arguments.
        out: The text stream to write the result to (usually ``sys.stdout``).
    """
    # Imported lazily so that `import olchiki_ocr.cli` stays cheap and
    # torch/onnxruntime-free until a recognition is actually requested.
    from .recognizer import ModelRecognizer, Prediction

    recognizer = ModelRecognizer.from_pretrained(
        path=args.model,
        model_source=args.model_source,
        cache_dir=args.cache_dir,
        device=args.device,
        decoder=args.decoder,
    )

    result = recognizer.predict(
        args.image,
        confidence=args.confidence,
        decoder=args.decoder,
        beam_width=args.beam_width,
    )

    if args.confidence:
        # predict(confidence=True) returns a Prediction; print "text\tconfidence"
        # (tab-separated so the text -- which may contain spaces -- stays intact).
        assert isinstance(result, Prediction)
        conf = result.confidence if result.confidence is not None else 0.0
        print(f"{result.text}\t{conf:.4f}", file=out)
    else:
        # predict() returns a bare str (Req 1.3).
        print(result, file=out)


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point: recognize an image and print its text.

    Parses ``argv`` (defaulting to ``sys.argv[1:]``), runs recognition, and
    returns a process exit code. Typed errors are mapped to their
    ``exit_code``; unexpected errors map to ``1`` (Req 12.3, 12.4).

    Args:
        argv: Command-line arguments (without the program name). ``None`` uses
            ``sys.argv[1:]``.

    Returns:
        ``0`` on success; a typed error's distinct ``exit_code`` when an
        :class:`~olchiki_ocr.errors.OcrTrainerError` is raised; ``1`` for any
        unexpected error.
    """
    parser = _build_parser()
    # argparse raises SystemExit on --help (code 0) or a parse error (code 2);
    # that is the standard, desired CLI behavior, so it is intentionally not
    # caught here.
    args = parser.parse_args(argv)

    try:
        _run(args, sys.stdout)
    except OcrTrainerError as err:
        # Expected, typed failure: print the message (which names the offending
        # identifier, Req 12.5) to stderr and return the distinct exit code.
        # No traceback escapes (Req 12.3, 12.4).
        print(str(err), file=sys.stderr)
        return err.exit_code
    except Exception as err:  # noqa: BLE001 -- top-level CLI safety net
        # Unexpected failure: keep it concise and map to exit code 1 (Req 12.4).
        print(f"{_PROG}: unexpected error: {err}", file=sys.stderr)
        return _UNEXPECTED_EXIT_CODE

    return 0


if __name__ == "__main__":  # pragma: no cover - thin shim
    sys.exit(main())
