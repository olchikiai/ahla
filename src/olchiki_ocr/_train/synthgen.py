"""Synthetic training-sample generator for the ``[train]`` tier.

``generate()`` renders each word from the Word_List into a labeled image across
the configured fonts (round-robin), skips labels with missing glyphs (recording
them in ``stats.skipped_labels``), and emits the DTRB raw layout
(``images/*.png`` + ``gt.txt``). It raises ``IngestError`` for a missing word
list and ``FontError`` for a bad font.

Generation is fully seeded, so identical Config + inputs yield identical Sample
sequence and image bytes. fontTools and Pillow are imported at load, so this
module needs the ``[train]`` extra.
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

# Fallback seed when ``cfg.seed`` is None (keeps generation deterministic).
_DEFAULT_SEED = 0

# Emitted Dataset layout under ``cfg.output_dir``.
_WORK_SUBDIR = "synthetic"
_IMAGES_SUBDIR = "images"
_GT_FILENAME = "gt.txt"

# Fixed rendering geometry (deterministic; augmentations perturb the render).
_FONT_SIZE = 32
_PADDING = 8
_BACKGROUND = 255  # white (grayscale "L" mode)
_FOREGROUND = 0  # black text

# Supported augmentation names; anything else is rejected.
_ROTATION = "rotation"
_BLUR = "blur"
_NOISE = "noise"
_RESIZE = "resize"
_SUPPORTED_AUGMENTATIONS = frozenset({_ROTATION, _BLUR, _NOISE, _RESIZE})


@dataclass(frozen=True)
class Sample:
    """A rendered image plus its Label and the font used to render it."""

    image_path: str
    label: str
    font_path: str


def _read_labels(word_list_path: str) -> list[str]:
    """Return the non-empty, order-preserving lines of the Word_List as Labels.

    One word per line. Raises IngestError if the path does not exist.
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
    """Load a font for rendering; raises FontError if Pillow cannot load it."""
    try:
        return ImageFont.truetype(font_path, _FONT_SIZE)
    except OSError as exc:
        raise FontError(font_path, "Cannot load font file") from exc


def _load_cmap(font_path: str) -> set[int]:
    """Return the Unicode code points the font has glyphs for (its best cmap).

    Raises FontError if fontTools cannot parse the font.
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
    """Raise ValueError on any unknown augmentation name (a typo must not
    silently change the Dataset)."""
    for name in augmentations:
        if name not in _SUPPORTED_AUGMENTATIONS:
            raise ValueError(
                f"Unknown augmentation {name!r}; supported augmentations are "
                f"{sorted(_SUPPORTED_AUGMENTATIONS)}"
            )


def _render_base(label: str, font: ImageFont.FreeTypeFont) -> Image.Image:
    """Render ``label`` onto a tight white grayscale ("L") canvas.

    Uses ``textbbox`` for sizing (Pillow 12 removed ``font.getsize``).
    """
    # textbbox on a scratch draw accounts for bearings.
    scratch = Image.new("L", (1, 1), _BACKGROUND)
    draw = ImageDraw.Draw(scratch)
    left, top, right, bottom = draw.textbbox((0, 0), label, font=font)
    text_width = max(right - left, 1)
    text_height = max(bottom - top, 1)

    width = text_width + 2 * _PADDING
    height = text_height + 2 * _PADDING

    image = Image.new("L", (width, height), _BACKGROUND)
    draw = ImageDraw.Draw(image)
    # Offset by (-left, -top) so negative bearings stay inside the pad.
    draw.text((_PADDING - left, _PADDING - top), label, fill=_FOREGROUND, font=font)
    return image


def _apply_augmentations(
    image: Image.Image, augmentations: tuple[str, ...], rng: random.Random
) -> Image.Image:
    """Apply enabled augmentations in a fixed order (rotation, blur, noise,
    resize), drawing all randomness from ``rng`` for reproducibility."""
    from PIL import ImageFilter

    result = image

    if _ROTATION in augmentations:
        # Small rotation jitter; expand so corners are not clipped.
        angle = rng.uniform(-5.0, 5.0)
        result = result.rotate(
            angle, resample=Image.BICUBIC, expand=True, fillcolor=_BACKGROUND
        )

    if _BLUR in augmentations:
        radius = rng.uniform(0.0, 1.2)
        result = result.filter(ImageFilter.GaussianBlur(radius=radius))

    if _NOISE in augmentations:
        # Additive per-pixel noise over the flat "L"-mode byte sequence, so the
        # result stays a deterministic function of the seed.
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
    """Write ``image`` as a deterministic PNG (no optimize/time chunk)."""
    image.save(path, format="PNG")


def generate(cfg: Config, charset: Charset, stats: RunStats) -> list[Sample]:
    """Render the Word_List into Samples and emit the DTRB raw dataset layout.

    Returns the ordered Samples (skipped labels omitted) and updates
    ``stats.dataset_sample_count`` / ``stats.skipped_labels``. Raises IngestError
    (missing word list), FontError (bad font), or ValueError (unknown
    augmentation).
    """
    _validate_augmentations(cfg.augmentations)

    labels = _read_labels(cfg.word_list_path)

    # Single seeded RNG so the whole generation is deterministic.
    seed = cfg.seed if cfg.seed is not None else _DEFAULT_SEED
    rng = random.Random(seed)

    work_dir = os.path.join(cfg.output_dir, _WORK_SUBDIR)
    images_dir = os.path.join(work_dir, _IMAGES_SUBDIR)
    os.makedirs(images_dir, exist_ok=True)

    # Cache fonts and cmaps per path.
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

    # Eagerly load every font so a bad font fails fast before any Sample.
    for font_path in cfg.font_paths:
        font_for(font_path)
        cmap_for(font_path)

    samples: list[Sample] = []
    gt_lines: list[str] = []
    # Zero-pad the index so filenames sort naturally.
    index_width = max(1, len(str(max(len(labels) - 1, 0))))

    for label_index, label in enumerate(labels):
        # Round-robin across the configured fonts.
        font_path = cfg.font_paths[label_index % len(cfg.font_paths)]
        cmap = cmap_for(font_path)

        # Skip and record labels with any unmapped glyph.
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

    # Write the gt.txt manifest as UTF-8.
    gt_path = os.path.join(work_dir, _GT_FILENAME)
    with open(gt_path, "w", encoding="utf-8", newline="\n") as handle:
        for gt_line in gt_lines:
            handle.write(gt_line + "\n")

    stats.dataset_sample_count = len(samples)
    return samples
