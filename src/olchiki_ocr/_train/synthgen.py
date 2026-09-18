"""Synthetic training-sample generator for the ``[train]`` tier (Req 10).

The ``Synthetic_Generator`` renders each non-empty line of the ``Word_List``
into a raster ``Text_Image`` paired with its ground-truth ``Label``, producing
the labeled Dataset used to transfer-learn the recognizer. Because labeled Ol
Chiki image data is scarce, this synthetic pipeline is the sole source of
training Samples.

Behavior:

* Read the Word_List, one non-empty line per Label, preserving original order.
  A missing Word_List path is fatal: :class:`IngestError` naming the path.
* Render each Label with a configured Font_File using Pillow
  (``ImageFont.truetype`` + ``ImageDraw``), cycling across the configured fonts
  round-robin when more than one is given. A font that cannot be loaded is
  fatal: :class:`FontError` naming the path.
* Missing-glyph detection: a Label is renderable with a font iff every
  ``ord(ch)`` is a key in that font's best cmap
  (``fontTools.ttLib.TTFont(font_path).getBestCmap()``). If any character is
  unmapped, the Label is skipped and ``(label, font_path)`` is recorded in
  ``stats.skipped_labels``. TTFont/cmap objects are cached per font path.
* Apply the configured augmentations (rotation jitter, gaussian blur, additive
  noise, resize scaling) drawn from a seeded RNG so runs are reproducible.
  Unknown augmentation names are fatal (raise ``ValueError``) so a typo does not
  silently produce a different Dataset.
* Encode each image as PNG.
* Emit ``<work_dir>/images/*.png`` plus a ``gt.txt`` with
  ``images/<name>.png<TAB><label>`` lines (DTRB raw layout), then return the
  ordered ``list[Sample]``.

Reproducibility: every random decision is driven by a single ``random.Random``
seeded from ``cfg.seed``; PNG encoding is deterministic (fixed image mode, no
timestamp chunk), so identical Config + inputs yield an identical Dataset - the
same Sample sequence *and* identical rendered image bytes. When ``cfg.seed`` is
``None`` this function falls back to a fixed default seed (``_DEFAULT_SEED``) so
determinism still holds; the Orchestrator is responsible for drawing and
recording a run-level seed when one is not supplied.

Work directory layout (a pure function of ``cfg.output_dir``)::

    <cfg.output_dir>/synthetic/
        images/<index>.png      # zero-padded per-Sample PNG
        gt.txt                  # "images/<index>.png\\t<label>" lines

fontTools and Pillow are imported at module load, so importing this module
requires the ``[train]`` extra (see :mod:`olchiki_ocr._train`).
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass

from fontTools.ttLib import TTFont
from PIL import Image, ImageDraw, ImageFont

from ..charset import Charset
from .config import Config
from .runstats import RunStats
from ..errors import FontError, IngestError

__all__ = ["Sample", "generate"]

# Fixed fallback seed used only when ``cfg.seed`` is None. The Orchestrator is
# expected to draw and record a run-level seed in that case; this default merely
# guarantees determinism for a fixed (here, absent) seed within this function.
_DEFAULT_SEED = 0

# Sub-directory of ``cfg.output_dir`` holding the emitted Dataset.
_WORK_SUBDIR = "synthetic"
_IMAGES_SUBDIR = "images"
_GT_FILENAME = "gt.txt"

# Rendering geometry. Fixed values keep rendering deterministic; augmentations
# perturb the base render, not these constants.
_FONT_SIZE = 32
_PADDING = 8
_BACKGROUND = 255  # white (grayscale "L" mode)
_FOREGROUND = 0  # black text

# Augmentation names supported by ``generate``. Any name in ``cfg.augmentations``
# outside this set is rejected with a clear error.
_ROTATION = "rotation"
_BLUR = "blur"
_NOISE = "noise"
_RESIZE = "resize"
_SUPPORTED_AUGMENTATIONS = frozenset({_ROTATION, _BLUR, _NOISE, _RESIZE})


@dataclass(frozen=True)
class Sample:
    """A single training/evaluation unit: a rendered image and its Label.

    Attributes:
        image_path: Filesystem path to the rendered PNG Text_Image.
        label: The ground-truth character sequence rendered into the image.
        font_path: The Font_File used to render the image (for reporting and
            multi-font provenance).
    """

    image_path: str
    label: str
    font_path: str


def _read_labels(word_list_path: str) -> list[str]:
    """Return the non-empty lines of the Word_List as Labels, in order.

    A line is considered empty (and therefore skipped) when it contains no
    characters after stripping the trailing newline and surrounding whitespace.
    The relative order of the non-empty lines is preserved.

    Raises:
        IngestError: If ``word_list_path`` does not exist.
    """
    if not os.path.isfile(word_list_path):
        raise IngestError(word_list_path, "Word list path does not exist")

    labels: list[str] = []
    with open(word_list_path, encoding="utf-8") as handle:
        for line in handle:
            label = line.strip()
            if label:
                labels.append(label)
    return labels


def _load_font(font_path: str) -> ImageFont.FreeTypeFont:
    """Load a TrueType/OpenType font for rendering.

    Raises:
        FontError: If the font cannot be loaded by Pillow, naming the offending
            path.
    """
    try:
        return ImageFont.truetype(font_path, _FONT_SIZE)
    except OSError as exc:
        raise FontError(font_path, "Cannot load font file") from exc


def _load_cmap(font_path: str) -> set[int]:
    """Return the set of Unicode code points the font provides glyphs for.

    Uses ``fontTools.ttLib.TTFont(font_path).getBestCmap()``: the best cmap is a
    dict keyed by Unicode code point, so a character is renderable iff
    ``ord(ch)`` is one of its keys.

    Raises:
        FontError: If the font cannot be parsed by fontTools.
    """
    try:
        ttfont = TTFont(font_path)
        best_cmap = ttfont.getBestCmap()
    except Exception as exc:  # fontTools raises a variety of parse errors
        raise FontError(font_path, "Cannot read font glyph table") from exc
    return set(best_cmap.keys())


def _is_renderable(label: str, cmap: set[int]) -> bool:
    """Return True iff every character of ``label`` has a glyph in ``cmap``."""
    return all(ord(ch) in cmap for ch in label)


def _validate_augmentations(augmentations: tuple[str, ...]) -> None:
    """Reject any unknown augmentation name with a clear error.

    Unknown names are treated as fatal configuration mistakes rather than
    silently ignored, so that a typo cannot quietly change the produced Dataset.

    Raises:
        ValueError: If an augmentation name is not one of the supported names.
    """
    for name in augmentations:
        if name not in _SUPPORTED_AUGMENTATIONS:
            raise ValueError(
                f"Unknown augmentation {name!r}; supported augmentations are "
                f"{sorted(_SUPPORTED_AUGMENTATIONS)}"
            )


def _render_base(label: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """Render ``label`` onto a tight white grayscale canvas.

    Sizing uses ``font.getbbox`` / ``ImageDraw.textbbox`` because Pillow 12
    removed ``font.getsize``. The image mode is fixed to ``"L"`` (8-bit
    grayscale) so encoding is deterministic and readable.
    """
    # Measure the text box. textbbox on a scratch draw accounts for bearings.
    scratch = Image.new("L", (1, 1), _BACKGROUND)
    draw = ImageDraw.Draw(scratch)
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_width = max(right - left, 1)
    text_height = max(bottom - top, 1)

    width = text_width + 2 * _PADDING
    height = text_height + 2 * _PADDING

    image = Image.new("L", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    # Offset by (-left, -top) so glyphs with negative bearings sit inside the pad.
    draw.text((_PADDING - left, _PADDING - top), label, fill=_FOREGROUND, font=font)
    return image


def _apply_augmentations(
    image: Image.Image, augmentations: tuple[str, ...], rng: random.Random
) -> Image.Image:
    """Apply the configured augmentations, drawing all randomness from ``rng``.

    Each augmentation is gated by presence in ``augmentations`` and applied in a
    fixed order (rotation -> blur -> noise -> resize) so the transformation is a
    deterministic function of the RNG state. ``rng`` values are consumed only for
    augmentations that are enabled, keeping the RNG stream identical across runs
    for identical Config.
    """
    from PIL import ImageFilter

    result = image

    if _ROTATION in augmentations:
        # Small rotation jitter in degrees; expand so corners are not clipped.
        angle = rng.uniform(-5.0, 5.0)
        result = result.rotate(
            angle, resample=Image.BICUBIC, expand=True, fillcolor=_BACKGROUND
        )

    if _BLUR in augmentations:
        radius = rng.uniform(0.0, 1.2)
        result = result.filter(ImageFilter.GaussianBlur(radius=radius))

    if _NOISE in augmentations:
        # Additive per-pixel noise. Read pixels as a flat row-major byte
        # sequence (one byte per pixel for the fixed 8-bit "L" mode), which
        # matches the order the RNG is consumed in, so output stays a
        # deterministic function of the seed. Avoids the deprecated
        # Image.getdata().
        strength = rng.uniform(5.0, 25.0)
        pixels = list(result.tobytes())
        noisy = []
        for value in pixels:
            delta = int(round(rng.uniform(-strength, strength)))
            noisy.append(min(255, max(0, value + delta)))
        result = result.copy()
        result.putdata(noisy)

    if _RESIZE in augmentations:
        scale = rng.uniform(0.8, 1.2)
        new_width = max(1, int(round(result.width * scale)))
        new_height = max(1, int(round(result.height * scale)))
        result = result.resize((new_width, new_height), resample=Image.BICUBIC)

    return result


def _save_png(image: Image.Image, path: str) -> None:
    """Write ``image`` to ``path`` as a deterministic PNG.

    ``optimize`` is left off and no ancillary time chunk is written, so the
    encoded bytes depend only on the pixel data and mode - giving identical
    bytes across runs for identical input.
    """
    image.save(path, format="PNG")


def generate(cfg: Config, charset: Charset, stats: RunStats) -> list[Sample]:
    """Render the Word_List into Samples and emit the DTRB raw dataset layout.

    See the module docstring for the full behavior contract. Returns the ordered
    list of produced :class:`Sample` objects (skipped Labels are omitted) and
    records ``stats.dataset_sample_count`` plus any ``stats.skipped_labels``.

    Args:
        cfg: The validated run Configuration. Uses ``word_list_path``,
            ``font_paths``, ``augmentations``, ``seed``, and ``output_dir``.
        charset: The recognition Charset (accepted for interface symmetry with
            the other stages; rendering does not restrict to it - missing-glyph
            handling is per-font).
        stats: The mutable accumulator; ``skipped_labels`` and
            ``dataset_sample_count`` are updated in place.

    Returns:
        The ordered list of produced Samples.

    Raises:
        IngestError: If ``cfg.word_list_path`` does not exist.
        FontError: If a configured Font_File cannot be loaded.
        ValueError: If ``cfg.augmentations`` names an unsupported augmentation.
    """
    _validate_augmentations(cfg.augmentations)

    labels = _read_labels(cfg.word_list_path)

    # Seed a single RNG from the Config seed (fixed default when None) so the
    # whole generation - font cycling perturbations and augmentations - is a
    # deterministic function of the seed.
    seed = cfg.seed if cfg.seed is not None else _DEFAULT_SEED
    rng = random.Random(seed)

    # Prepare the work directory layout under output_dir.
    work_dir = os.path.join(cfg.output_dir, _WORK_SUBDIR)
    images_dir = os.path.join(work_dir, _IMAGES_SUBDIR)
    os.makedirs(images_dir, exist_ok=True)

    # Cache loaded fonts and their cmaps per path for efficiency.
    font_cache: dict[str, ImageFont.FreeTypeFont] = {}
    cmap_cache: dict[str, set[int]] = {}

    def font_for(path: str) -> ImageFont.FreeTypeFont:
        if path not in font_cache:
            font_cache[path] = _load_font(path)
        return font_cache[path]

    def cmap_for(path: str) -> set[int]:
        if path not in cmap_cache:
            cmap_cache[path] = _load_cmap(path)
        return cmap_cache[path]

    # Eagerly load every configured font so an unloadable font fails fast with
    # FontError before any Sample is emitted.
    for font_path in cfg.font_paths:
        font_for(font_path)
        cmap_for(font_path)

    samples: list[Sample] = []
    gt_lines: list[str] = []
    # Zero-pad the image index so filenames sort naturally and stay stable.
    index_width = max(1, len(str(max(len(labels) - 1, 0))))

    for label_index, label in enumerate(labels):
        # Cycle across fonts round-robin so Samples span every configured font.
        # Single-font configs always pick font_paths[0].
        font_path = cfg.font_paths[label_index % len(cfg.font_paths)]
        cmap = cmap_for(font_path)

        # Missing-glyph detection: skip and record if any char is unmapped.
        if not _is_renderable(label, cmap):
            stats.skipped_labels.append((label, font_path))
            continue

        font = font_for(font_path)
        image = _render_base(label, font)
        image = _apply_augmentations(image, cfg.augmentations, rng)

        name = f"{len(samples):0{index_width}d}.png"
        image_rel_path = f"{_IMAGES_SUBDIR}/{name}"
        image_abs_path = os.path.join(images_dir, name)
        _save_png(image, image_abs_path)

        samples.append(
            Sample(image_path=image_abs_path, label=label, font_path=font_path)
        )
        # DTRB raw layout: "images/<name>.png<TAB><label>".
        gt_lines.append(f"{image_rel_path}\t{label}")

    # Write the ground-truth manifest as UTF-8 with a trailing newline per line.
    gt_path = os.path.join(work_dir, _GT_FILENAME)
    with open(gt_path, "w", encoding="utf-8", newline="\n") as handle:
        for gt_line in gt_lines:
            handle.write(gt_line + "\n")

    stats.dataset_sample_count = len(samples)
    return samples
