"""``olchiki-ocr`` console-script entry point.

Recognizes a pre-cropped word/line image and prints the text to stdout. Typed
errors are printed to stderr and mapped to their ``exit_code``; unexpected
errors map to 1. ``ModelRecognizer`` is imported inside :func:`_run` so
``import olchiki_ocr.cli`` stays free of onnxruntime/numpy/Pillow until a
recognition actually runs.

In addition to the legacy single-image recognition invocation
(``olchiki-ocr IMAGE ...``, preserved unchanged for backward compatibility --
Req 9.7), two document-segmenter subcommands are provided:

* ``segment PAGE --out-dir DIR`` -- segment a full page into region crops and
  write each crop as an indexed image to ``DIR`` (Req 9.1, 9.2).
* ``recognize-page PAGE`` -- segment then recognize a full page, printing one
  region per line in reading order (Req 9.3, 9.4).

All engine/segmentation imports live inside the per-subcommand run functions so
``import olchiki_ocr.cli`` stays free of onnxruntime and OpenCV. Every path
funnels through the same typed-error handling in :func:`main` (Req 9.5, 9.6).
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

#: Subcommand verbs handled by the segmenter subparser. When ``argv``'s first
#: token is one of these, :func:`main` dispatches to the subparser; otherwise it
#: falls back to the legacy single-image parser unchanged (Req 9.7).
_SUBCOMMANDS = frozenset({"segment", "recognize-page"})


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


def _add_recognizer_options(parser: argparse.ArgumentParser) -> None:
    """Add the shared model/device/decoder options used by recognition commands.

    These mirror the legacy single-image parser's option names exactly so the
    ``recognize-page`` subcommand accepts the same model resolution flags as the
    legacy invocation (Req 9.3, 9.7).
    """
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


def _add_segmenter_options(parser: argparse.ArgumentParser) -> None:
    """Add the shared segmentation options used by ``segment``/``recognize-page``.

    Granularity defaults to ``word`` (Req 9.2); deskew and denoise default to
    enabled and are turned off with ``--no-deskew`` / ``--no-denoise``.
    """
    parser.add_argument(
        "--granularity",
        choices=["line", "word"],
        default="word",
        help="Segmentation granularity: 'line' or 'word' (default: word).",
    )
    parser.add_argument(
        "--no-deskew",
        dest="deskew",
        action="store_false",
        help="Disable deskew correction (enabled by default).",
    )
    parser.add_argument(
        "--no-denoise",
        dest="denoise",
        action="store_false",
        help="Disable noise reduction (enabled by default).",
    )
    parser.add_argument(
        "--min-area",
        dest="min_area",
        type=int,
        default=20,
        metavar="N",
        help="Minimum region area in px^2; smaller components are dropped "
        "(default: 20).",
    )
    parser.set_defaults(deskew=True, denoise=True)


def _build_subparser() -> argparse.ArgumentParser:
    """Construct the parser for the ``segment`` / ``recognize-page`` subcommands."""
    parser = argparse.ArgumentParser(
        prog=_PROG,
        description=(
            "Segment or recognize a full Ol Chiki (Santali) document page."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # `segment PAGE --out-dir DIR [...]` (Req 9.1, 9.2)
    seg = subparsers.add_parser(
        "segment",
        help="Segment a page into region crops written to an output directory.",
    )
    seg.add_argument("page", metavar="PAGE", help="Filesystem path to the page image.")
    seg.add_argument(
        "--out-dir",
        dest="out_dir",
        metavar="DIR",
        required=True,
        help="Directory to write the region crops into (created if needed).",
    )
    _add_segmenter_options(seg)
    seg.set_defaults(func=_run_segment)

    # `recognize-page PAGE [--confidence] [...]` (Req 9.3, 9.4)
    rec = subparsers.add_parser(
        "recognize-page",
        help="Segment then recognize a page, printing one region per line.",
    )
    rec.add_argument("page", metavar="PAGE", help="Filesystem path to the page image.")
    rec.add_argument(
        "--confidence",
        action="store_true",
        help="Also print each region's confidence as 'text\\tconfidence'.",
    )
    _add_segmenter_options(rec)
    _add_recognizer_options(rec)
    rec.set_defaults(func=_run_recognize_page)

    return parser


def _run_segment(args: argparse.Namespace, out) -> None:
    """Segment a page and write region crops, printing the count written.

    Builds a ``PageSegmenter`` from the parsed options, segments ``args.page``,
    writes the crops to ``args.out_dir``, and prints the number of crops
    written. Segmentation/OpenCV imports are performed here (not at module top)
    so ``import olchiki_ocr.cli`` stays engine-free and OpenCV-free (Req 9.1,
    9.2). Typed errors propagate to :func:`main`.
    """
    # Imported lazily to keep `import olchiki_ocr.cli` engine-free and cv2-free.
    from .segmentation import PageSegmenter

    segmenter = PageSegmenter(
        granularity=args.granularity,
        deskew=args.deskew,
        denoise=args.denoise,
        min_region_area=args.min_area,
    )
    result = segmenter.segment(args.page)
    written = result.write_crops(args.out_dir)
    print(len(written), file=out)


def _run_recognize_page(args: argparse.Namespace, out) -> None:
    """Segment then recognize a page, printing one region per line in reading order.

    Builds a ``Page_Recognizer`` from the parsed model/segmentation options,
    recognizes ``args.page``, and prints each region's text on its own line in
    reading order. With ``--confidence`` each line is ``text\\tconfidence``
    (tab-separated). Text is written as UTF-8. Engine/segmentation imports are
    performed here so ``import olchiki_ocr.cli`` stays engine-free (Req 9.3,
    9.4). Typed errors propagate to :func:`main`.
    """
    # Imported lazily to keep `import olchiki_ocr.cli` engine-free and cv2-free.
    from .page_recognizer import Page_Recognizer

    recognizer = Page_Recognizer.from_pretrained(
        args.model,
        granularity=args.granularity,
        deskew=args.deskew,
        denoise=args.denoise,
        min_region_area=args.min_area,
        model_source=args.model_source,
        cache_dir=args.cache_dir,
        device=args.device,
        decoder=args.decoder,
    )

    result = recognizer.recognize_page(
        args.page,
        confidence=args.confidence,
        decoder=args.decoder,
        beam_width=args.beam_width,
    )

    for region in result.regions:
        if args.confidence:
            conf = region.confidence if region.confidence is not None else 0.0
            line = f"{region.text}\t{conf:.4f}"
        else:
            line = region.text
        print(line, file=out)


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

    Backward compatibility (Req 9.7): the first token of ``argv`` is inspected.
    When it is a known subcommand (``segment`` / ``recognize-page``) the
    invocation is dispatched to the subparser; otherwise it falls back to the
    legacy single-image parser unchanged, so ``olchiki-ocr IMAGE ...`` and all
    its flags behave exactly as before.
    """
    tokens = list(sys.argv[1:] if argv is None else argv)

    if tokens and tokens[0] in _SUBCOMMANDS:
        # Segmenter subcommands. argparse raises SystemExit on --help / parse
        # errors; that is intended.
        parser = _build_subparser()
        args = parser.parse_args(tokens)
        run = args.func
    else:
        # Legacy single-image recognition parser, unchanged (Req 9.7).
        parser = _build_parser()
        args = parser.parse_args(tokens)
        run = _run

    try:
        run(args, sys.stdout)
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
