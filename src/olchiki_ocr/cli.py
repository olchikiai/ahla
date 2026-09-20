"""``olchiki-ocr`` console-script entry point.

Recognizes a pre-cropped word/line image and prints the text to stdout. Typed
errors are printed to stderr and mapped to their ``exit_code``; unexpected
errors map to 1. ``ModelRecognizer`` is imported inside :func:`_run` so
``import olchiki_ocr.cli`` stays free of onnxruntime/numpy/Pillow until a
recognition actually runs.
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
    """Construct the argument parser for the ``olchiki-ocr`` console script."""
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
    # --model / --path: a local artifact directory loaded offline (no network).
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
    """Run recognition for parsed ``args`` and write the result to ``out``.

    Errors propagate to :func:`main`, which maps them to exit codes.
    """
    # Imported lazily to keep `import olchiki_ocr.cli` engine-free.
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
        # Tab-separated so text containing spaces stays intact.
        assert isinstance(result, Prediction)
        conf = result.confidence if result.confidence is not None else 0.0
        print(f"{result.text}\t{conf:.4f}", file=out)
    else:
        print(result, file=out)


def main(argv: Sequence[str] | None = None) -> int:
    """Console-script entry point: recognize an image and print its text.

    Returns 0 on success, a typed error's ``exit_code`` on ``OcrTrainerError``,
    or 1 for any unexpected error.
    """
    parser = _build_parser()
    # argparse raises SystemExit on --help / parse errors; that is intended.
    args = parser.parse_args(argv)

    try:
        _run(args, sys.stdout)
    except OcrTrainerError as err:
        # Expected typed failure: print the message, return its exit code.
        print(str(err), file=sys.stderr)
        return err.exit_code
    except Exception as err:  # noqa: BLE001 -- top-level CLI safety net
        print(f"{_PROG}: unexpected error: {err}", file=sys.stderr)
        return _UNEXPECTED_EXIT_CODE

    return 0


if __name__ == "__main__":  # pragma: no cover - thin shim
    sys.exit(main())
