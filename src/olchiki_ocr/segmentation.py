"""Classical-CV document segmentation for the ``olchiki_ocr`` package.

:class:`PageSegmenter` turns a full scanned Ol Chiki page into the ordered
word/line crops the existing recognizer consumes. Segmentation is a classical
computer-vision problem (connected components, projection profiles, contour
analysis), so it depends on OpenCV.

To keep the torch-free core also OpenCV-free at import time, only numpy and the
stdlib are imported at module top; cv2 is imported lazily via
:func:`_require_cv2`, which raises a typed :class:`ConfigError` naming the
``[segmentation]`` extra rather than a bare ``ImportError`` when the extra is
absent.
"""

from __future__ import annotations

from enum import Enum

import numpy as np

__all__ = ["PageSegmenter", "Granularity", "CoordinateSpace"]


class Granularity(str, Enum):
    """The unit of segmentation: one Region per line or per word."""

    LINE = "line"
    WORD = "word"


class CoordinateSpace(str, Enum):
    """The pixel space in which a Segmentation_Result's boxes are reported."""

    ORIGINAL = "original"  # boxes in the input page's pixel space
    DESKEWED = "deskewed"  # boxes in the rotated (leveled) page's pixel space


#: Message naming the missing ``[segmentation]`` extra, shared by
#: :func:`_require_cv2`.
_SEG_EXTRA_MESSAGE: str = (
    "the '[segmentation]' extra is required for document segmentation "
    "(install: pip install olchiki-ocr[segmentation])"
)


def _require_cv2():
    """Import and return the OpenCV (``cv2``) module lazily.

    Keeps the core OpenCV-free at import time; every segmentation CV step calls
    this first. Raises ``ConfigError`` naming the ``[segmentation]`` extra if
    cv2 is missing (ConfigError so it maps to the CLI's config exit code, not a
    bare ImportError).
    """
    try:
        import cv2  # noqa: PLC0415 - lazy import is intentional (keep core cv2-free)
    except ImportError as exc:
        # Local import keeps the top-level import surface numpy + stdlib only.
        from .errors import ConfigError

        raise ConfigError("[segmentation]", _SEG_EXTRA_MESSAGE) from exc
    return cv2


# ---------------------------------------------------------------------------
# Region-detection tuning constants and pure helpers.
#
# The constants are expressed as integer numerator/denominator pairs so the
# derived thresholds are pure integer arithmetic (no floats, no RNG) and the
# detection stays bit-for-bit deterministic across runs and processes
# (Req 11.1, 11.2). The helpers are pure numpy/stdlib and touch no cv2, so they
# stay importable without the [segmentation] extra.
# ---------------------------------------------------------------------------

#: A row is "text" when its foreground count is >= this fraction (num/den) of
#: the busiest row's count. Small so faint line tops/bottoms still join a band.
_LINE_ROW_THRESHOLD_NUM: int = 1
_LINE_ROW_THRESHOLD_DEN: int = 10

#: Two components in a line band belong to the same word when the horizontal
#: gap between them is smaller than this fraction (num/den) of the band height.
_WORD_MERGE_GAP_NUM: int = 6
_WORD_MERGE_GAP_DEN: int = 10

#: A column is "text" when its foreground count is >= this fraction (num/den)
#: of the busiest column's count. Columns below this are treated as (near-)empty
#: whitespace so wide low-density gutters read as column separators.
_COLUMN_COL_THRESHOLD_NUM: int = 1
_COLUMN_COL_THRESHOLD_DEN: int = 20

#: A run of below-threshold columns splits the page into separate columns only
#: when it is at least this fraction (num/den) of the page width wide. This is
#: what makes a *wide* gutter (not the ordinary inter-word gap) a column break,
#: so a single-column page stays one column (Req 3.4). Set to 1/8 of the page
#: width so an ordinary inter-word space never reads as a column separator while
#: a real inter-column gutter (typically a large fraction of the page) does.
_COLUMN_GUTTER_MIN_NUM: int = 1
_COLUMN_GUTTER_MIN_DEN: int = 8


def _box_area(box: "tuple[int, int, int, int]") -> int:
    """Return ``width * height`` for a ``(x, y, width, height)`` box."""
    return box[2] * box[3]


def _contiguous_runs(mask: np.ndarray) -> "list[tuple[int, int]]":
    """Return ``(start, stop)`` half-open index ranges of ``True`` runs.

    ``mask`` is a 1-D boolean array; each returned ``(start, stop)`` marks a
    maximal contiguous run of ``True`` values with ``stop`` exclusive. Scanning
    is strictly left-to-right so the output order is deterministic.
    """
    runs: "list[tuple[int, int]]" = []
    start: "int | None" = None
    for i, value in enumerate(mask.tolist()):
        if value and start is None:
            start = i
        elif not value and start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, int(mask.shape[0])))
    return runs


def _merge_horizontally(
    comp_boxes: "list[tuple[int, int, int, int]]", merge_gap: int
) -> "list[tuple[int, int, int, int]]":
    """Merge x-sorted component boxes separated by < ``merge_gap`` into words.

    ``comp_boxes`` must already be sorted left-to-right. Two adjacent boxes are
    merged when the horizontal gap between the running word's right edge and the
    next box's left edge is smaller than ``merge_gap``; the merged box is the
    union rectangle. Emitted left-to-right, so the output is deterministic.
    """
    merged: "list[tuple[int, int, int, int]]" = []
    for box in comp_boxes:
        if not merged:
            merged.append(box)
            continue
        px, py, pw, ph = merged[-1]
        bx, by, bw, bh = box
        gap = bx - (px + pw)
        if gap < merge_gap:
            # Union of the two rectangles becomes the running word box.
            nx = min(px, bx)
            ny = min(py, by)
            nx1 = max(px + pw, bx + bw)
            ny1 = max(py + ph, by + bh)
            merged[-1] = (nx, ny, nx1 - nx, ny1 - ny)
        else:
            merged.append(box)
    return merged


class PageSegmenter:
    """Classical-CV segmenter: full page image -> ordered word/line crops.

    ``__init__`` validates configuration up front so misconfiguration fails
    fast with a typed :class:`ConfigError`. The actual segmentation pipeline
    (loading, deskew, denoise, binarize, region detection, reading order) is
    implemented in later tasks; cv2 is touched only through
    :func:`_require_cv2` so a bare import stays OpenCV-free.
    """

    def __init__(
        self,
        *,
        granularity: "str | Granularity" = Granularity.WORD,
        deskew: bool = True,
        denoise: bool = True,
        min_region_area: int = 20,
    ) -> None:
        # Local import keeps the top-level import surface numpy + stdlib only.
        from .errors import ConfigError

        # Normalize/validate granularity: accept the enum or its string value.
        if isinstance(granularity, Granularity):
            normalized = granularity
        else:
            try:
                normalized = Granularity(granularity)
            except ValueError as exc:
                raise ConfigError(
                    "granularity",
                    f"granularity must be 'line' or 'word', got {granularity!r}",
                ) from exc

        # Validate min_region_area is non-negative.
        if min_region_area < 0:
            raise ConfigError(
                "min_region_area",
                f"min_region_area must be >= 0, got {min_region_area!r}",
            )

        self.granularity: Granularity = normalized
        self.deskew: bool = deskew
        self.denoise: bool = denoise
        self.min_region_area: int = min_region_area

    def _load_page(self, page_image: str) -> np.ndarray:
        """Open ``page_image`` and return it as a grayscale ``uint8`` ndarray.

        Opens the page via Pillow, forces decode with ``img.load()`` while the
        handle is open (so a corrupt payload raises here), converts to
        single-channel grayscale (``L``) so a color/multi-channel input is
        reduced to one channel while a single-channel input stays single-channel
        (Req 1.5), and returns a contiguous ``uint8`` array of shape ``(H, W)``.

        A missing path or an undecodable / unsupported-format payload is
        re-raised as :class:`ImageError` naming ``page_image`` (Req 1.3, 1.4,
        10.1), mirroring :meth:`Preprocessing_Pipeline.to_tensor`.
        """
        # Local imports keep the top-level import surface numpy + stdlib only.
        from PIL import Image, UnidentifiedImageError

        from .errors import ImageError

        try:
            with Image.open(page_image) as img:
                # Force decode while the handle is open so a corrupt payload
                # raises here (re-raised as ImageError below).
                img.load()
                # Reduce any color/multi-channel input to single-channel gray.
                gray = img.convert("L")
        except (FileNotFoundError, OSError, ValueError, UnidentifiedImageError) as exc:
            raise ImageError(page_image) from exc

        return np.ascontiguousarray(np.asarray(gray, dtype=np.uint8))

    def _deskew(self, page: np.ndarray) -> "tuple[np.ndarray, CoordinateSpace]":
        """Level page skew and report the crop/box coordinate space.

        When ``self.deskew`` is enabled, build an inverted-Otsu foreground mask,
        estimate the skew angle via ``cv2.minAreaRect`` on the foreground pixels
        (normalized into ``[-45, 45]`` degrees for the smallest correction), and
        rotate the page with ``cv2.warpAffine`` using a white border fill so the
        leveled text sits on a paper-white background (mirrors
        :func:`preprocessing.deskew`). Returns the deskewed page together with
        :attr:`CoordinateSpace.DESKEWED` (Req 4.1, 4.4).

        When deskew is disabled the page is returned unchanged with
        :attr:`CoordinateSpace.ORIGINAL` (Req 4.2's counterpart). A blank page
        (no foreground) or a skew estimate that falls outside ``[-45, 45]`` after
        normalization is left unrotated but still reported in the space that
        actually ran, so the round-trip crop property holds either way.

        ``_require_cv2()`` is called first so the ``[segmentation]`` extra is
        validated at operation time (Req 8.3, 8.4).
        """
        if not self.deskew:
            return page, CoordinateSpace.ORIGINAL

        cv2 = _require_cv2()

        # Invert so dark text -> non-zero foreground for angle estimation.
        _, binary = cv2.threshold(
            page, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        coords = cv2.findNonZero(binary)
        if coords is None:
            # Blank page: nothing to level, but deskew still "ran".
            return page, CoordinateSpace.DESKEWED

        angle = cv2.minAreaRect(coords)[-1]
        # cv2 returns an angle in (-90, 0]; normalize into [-45, 45] so we apply
        # the smallest rotation that levels the text.
        if angle < -45:
            angle = 90 + angle

        # Out-of-range fallback: if the estimate is not a sensible small skew
        # correction, leave the page unrotated rather than over-rotating it.
        if angle < -45 or angle > 45:
            return page, CoordinateSpace.DESKEWED

        h, w = page.shape[:2]
        center = (w / 2.0, h / 2.0)
        matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(
            page,
            matrix,
            (w, h),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=255,  # white fill matches typical paper background
        )
        return np.ascontiguousarray(rotated, dtype=np.uint8), CoordinateSpace.DESKEWED

    def _denoise(self, page: np.ndarray) -> np.ndarray:
        """Reduce page noise before region detection (Req 4.3).

        When ``self.denoise`` is enabled, apply ``cv2.fastNlMeansDenoising``,
        falling back to a median blur if fast NL-means is absent from the
        installed OpenCV build (mirrors :func:`preprocessing.denoise`). When
        disabled, the page is returned unchanged.

        ``_require_cv2()`` is called first so the ``[segmentation]`` extra is
        validated at operation time (Req 8.3, 8.4).
        """
        if not self.denoise:
            return page

        cv2 = _require_cv2()
        fast_nl_means = getattr(cv2, "fastNlMeansDenoising", None)
        if fast_nl_means is not None:
            result = fast_nl_means(page, None, 10, 7, 21)
        else:  # pragma: no cover - depends on the installed OpenCV build
            result = cv2.medianBlur(page, 3)
        return np.ascontiguousarray(result, dtype=np.uint8)

    def _binarize(self, page: np.ndarray) -> np.ndarray:
        """Binarize the page with inverted Otsu so text is foreground.

        Uses ``cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU`` so dark text becomes
        the non-zero (255) foreground and the paper background becomes 0, which
        is what the connected-component / projection-profile detection expects.
        Returns a ``uint8`` array whose only values are ``0`` and ``255``.

        ``_require_cv2()`` is called first so the ``[segmentation]`` extra is
        validated at operation time (Req 8.3, 8.4).
        """
        cv2 = _require_cv2()
        _, binary = cv2.threshold(
            page, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        return np.ascontiguousarray(binary, dtype=np.uint8)
    def _detect_regions(self, binary: np.ndarray) -> "list[tuple[int, int, int, int]]":
        """Detect line/word bounding boxes from a binarized page.

        Input contract
        --------------
        ``binary`` is the binarized page produced by :meth:`_binarize`: a
        ``uint8`` ``(H, W)`` array whose foreground (text) pixels are ``255``
        and background is ``0``.

        Return contract
        ---------------
        Returns a plain ``list`` of integer bounding boxes ``(x, y, width,
        height)`` in the pixel space of ``binary``. The boxes are **not yet
        ordered into reading order and carry no index** -- reading-order
        assignment is :meth:`_order_reading` (task 3.7) and the final
        ``Region``/``Segmentation_Result`` assembly with contiguous indices is
        :meth:`segment` (task 3.8). Every returned box satisfies ``width > 0``,
        ``height > 0`` and lies fully within ``binary`` so downstream slicing
        ``binary[y:y+height, x:x+width]`` is always valid. Boxes whose area
        ``width * height`` is below ``self.min_region_area`` are excluded
        (speckle filter, Req 4.5). An empty page yields ``[]`` (Req 5.6).

        Algorithm (deterministic; no RNG)
        ---------------------------------
        1. Horizontal projection profile: sum foreground pixels per row. A
           data-derived threshold (a fraction of the profile's own maximum)
           separates text rows from inter-line whitespace; contiguous runs of
           above-threshold rows delimit line bands (Req 3.1).
        2. ``line`` granularity: one box per band spanning the horizontal
           extent of foreground within that band (Req 2.3).
        3. ``word`` granularity: within each band run
           ``cv2.connectedComponentsWithStats`` and merge components that are
           horizontally close (gap analysis on a vertical projection within the
           band) into word boxes (Req 2.4).
        4. Speckle filter drops any box with area below ``min_region_area``
           (Req 4.5).

        Determinism is guaranteed: thresholds are derived from the image data
        (no sampling), bands are scanned top-to-bottom, and word boxes are
        emitted left-to-right by column, so the same input always yields an
        identical list. ``_require_cv2()`` is called first so the
        ``[segmentation]`` extra is validated at operation time (Req 8.3, 8.4).
        """
        cv2 = _require_cv2()

        height, width = binary.shape[:2]
        if height == 0 or width == 0:
            return []

        # Foreground mask as 0/1 so projection profiles count pixels, not 255s.
        foreground = (binary > 0).astype(np.int32)

        # --- Step 1: horizontal projection profile -> line bands. ----------
        # Row-sum of foreground pixels; a data-derived threshold (fraction of
        # the profile's own max) separates text rows from whitespace so the
        # threshold adapts to the page without any random value.
        row_sums = foreground.sum(axis=1)
        max_row = int(row_sums.max())
        if max_row == 0:
            return []  # blank page: no foreground anywhere (Req 5.6)

        # A row counts as "text" when its foreground pixel count is at least a
        # small fraction of the busiest row. The fraction is fixed (not random)
        # and the actual cutoff scales with the page's own ink density.
        row_threshold = max(1, (max_row * _LINE_ROW_THRESHOLD_NUM) // _LINE_ROW_THRESHOLD_DEN)
        line_bands = _contiguous_runs(row_sums >= row_threshold)

        boxes: "list[tuple[int, int, int, int]]" = []
        for band_top, band_bottom in line_bands:  # band_bottom exclusive
            band = foreground[band_top:band_bottom, :]

            if self.granularity == Granularity.LINE:
                # One box per band spanning the horizontal foreground extent.
                col_sums = band.sum(axis=0)
                cols = np.nonzero(col_sums > 0)[0]
                if cols.size == 0:
                    continue
                x0 = int(cols[0])
                x1 = int(cols[-1]) + 1  # exclusive
                # Tighten the vertical extent to rows that actually have ink so
                # the band box hugs the text (keeps the round-trip crop tight).
                band_row_sums = band.sum(axis=1)
                rows = np.nonzero(band_row_sums > 0)[0]
                y0 = band_top + int(rows[0])
                y1 = band_top + int(rows[-1]) + 1  # exclusive
                box = (x0, y0, x1 - x0, y1 - y0)
                if _box_area(box) >= self.min_region_area:
                    boxes.append(box)
                continue

            # --- Step 3: word granularity -- connected components in band. --
            band_u8 = np.ascontiguousarray(
                (band > 0).astype(np.uint8) * 255
            )
            num_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(
                band_u8, connectivity=8
            )

            # Collect component boxes (skip label 0 == background), converting
            # band-local y back to full-page y. Sort left-to-right by x (then y)
            # so the merge pass and the emitted order are deterministic.
            comp_boxes: "list[tuple[int, int, int, int]]" = []
            for label in range(1, num_labels):
                cx = int(stats[label, cv2.CC_STAT_LEFT])
                cy = int(stats[label, cv2.CC_STAT_TOP]) + band_top
                cw = int(stats[label, cv2.CC_STAT_WIDTH])
                ch = int(stats[label, cv2.CC_STAT_HEIGHT])
                comp_boxes.append((cx, cy, cw, ch))
            comp_boxes.sort(key=lambda b: (b[0], b[1]))

            # --- Merge horizontally close components into words. ------------
            # The merge gap scales with the band height so it adapts to text
            # size; components separated by less than the gap belong to the
            # same word, larger gaps mark a word boundary.
            band_height = band_bottom - band_top
            merge_gap = max(1, (band_height * _WORD_MERGE_GAP_NUM) // _WORD_MERGE_GAP_DEN)
            word_boxes = _merge_horizontally(comp_boxes, merge_gap)

            for box in word_boxes:
                if _box_area(box) >= self.min_region_area:
                    boxes.append(box)

        return boxes

    def _detect_columns(
        self, binary: np.ndarray, boxes: "list[tuple[int, int, int, int]]"
    ) -> "list[tuple[int, int]]":
        """Detect column bands from a binarized page's vertical profile.

        Input contract
        --------------
        ``binary`` is the binarized page produced by :meth:`_binarize` (a
        ``uint8`` ``(H, W)`` array, foreground ``255`` / background ``0``).
        ``boxes`` is the unordered, unindexed list of ``(x, y, width, height)``
        region boxes from :meth:`_detect_regions`; it is used only to keep the
        returned bands consistent with the detected content (a page with no
        boxes has no columns).

        Return contract
        ---------------
        Returns a ``list`` of ``(x_start, x_end)`` half-open column bands in
        pixel-x coordinates, **sorted left-to-right** with ``x_start <
        x_end``. A single-column page (no wide gutter) yields exactly one band
        spanning the page's foreground horizontal extent (Req 3.4). A page with
        no foreground / no boxes yields ``[]``. The bands are non-overlapping
        and cover every detected box (each box centre falls inside exactly one
        band), so :meth:`_order_reading` can map every region to a column.

        Algorithm (deterministic; no RNG)
        ---------------------------------
        1. Vertical projection profile: sum foreground pixels per column.
        2. A data-derived threshold (a small fraction of the busiest column's
           count) marks near-empty columns; contiguous runs of above-threshold
           columns are the candidate text columns.
        3. Only gutters (runs of below-threshold columns *between* text) that
           are at least a fraction of the page width wide split columns; narrow
           low-density valleys (ordinary inter-word gaps) do not, so a single
           column stays one band (Req 3.4).

        Every step is a pure function of ``binary`` (thresholds derived from the
        data, columns scanned left-to-right), so the result is bit-for-bit
        identical across runs and processes (Req 11.1, 11.2).
        """
        if not boxes:
            return []

        height, width = binary.shape[:2]
        if height == 0 or width == 0:
            return []

        foreground = (binary > 0).astype(np.int32)
        col_sums = foreground.sum(axis=0)
        max_col = int(col_sums.max())
        if max_col == 0:
            return []  # no foreground anywhere

        # Columns with at least this many foreground pixels count as "text".
        # The fraction is fixed; the cutoff scales with the page's ink density.
        col_threshold = max(
            1, (max_col * _COLUMN_COL_THRESHOLD_NUM) // _COLUMN_COL_THRESHOLD_DEN
        )
        text_runs = _contiguous_runs(col_sums >= col_threshold)
        if not text_runs:
            return []

        # A gutter only splits columns when it is a *wide* low-density valley,
        # not an ordinary inter-word gap. Merge adjacent text runs whose
        # separating gap is narrower than the minimum gutter width.
        min_gutter = max(
            1, (width * _COLUMN_GUTTER_MIN_NUM) // _COLUMN_GUTTER_MIN_DEN
        )
        bands: "list[tuple[int, int]]" = [text_runs[0]]
        for start, stop in text_runs[1:]:
            prev_start, prev_stop = bands[-1]
            gap = start - prev_stop
            if gap < min_gutter:
                # Narrow valley: same column, extend the current band.
                bands[-1] = (prev_start, stop)
            else:
                # Wide gutter: a new column begins.
                bands.append((start, stop))

        return bands

    def _order_reading(
        self,
        boxes: "list[tuple[int, int, int, int]]",
        columns: "list[tuple[int, int]]",
    ) -> "list[tuple[int, int, int, int]]":
        """Return ``boxes`` in reading order (columns, then lines, then x).

        Input contract
        --------------
        ``boxes`` is the unordered, unindexed list of ``(x, y, width, height)``
        region boxes from :meth:`_detect_regions`. ``columns`` is the
        left-to-right list of ``(x_start, x_end)`` bands from
        :meth:`_detect_columns` (``[]`` when there is no content).

        Return contract
        ---------------
        Returns the **same boxes** (still ``(x, y, width, height)`` tuples, no
        index assigned -- indexing is :meth:`segment`, task 3.8) reordered into
        Reading_Order: columns left-to-right, then lines top-to-bottom within a
        column, then left-to-right within a line (Req 3.3). For a single column
        this reduces to a ``(y, x)`` ordering -- top-to-bottom then
        left-to-right (Req 3.4). An empty ``boxes`` yields ``[]``.

        Column assignment (Req 3.2)
        ---------------------------
        Each box is assigned to the column band with which it horizontally
        overlaps most; ties (equal overlap, e.g. a box straddling a boundary)
        break to the left-most column. A box that overlaps no band (possible
        only if columns are degenerate) falls back to the band containing its
        centre x, then to the nearest band, so every box gets a column index.

        Determinism
        -----------
        Ordering uses Python's stable :func:`sorted` on the integer key
        ``(column_index, y, x)`` only -- no set/dict iteration, no wall-clock,
        no RNG -- so identical input always yields an identical order (Req
        11.1, 11.2).
        """
        if not boxes:
            return []

        # With no detected columns, treat the whole page as a single column so
        # ordering degenerates to (y, x): top-to-bottom then left-to-right.
        if not columns:
            return sorted(boxes, key=lambda b: (b[1], b[0]))

        def _column_index(box: "tuple[int, int, int, int]") -> int:
            x, _y, w, _h = box
            box_start = x
            box_end = x + w  # exclusive
            best_index = 0
            best_overlap = -1
            for idx, (c_start, c_end) in enumerate(columns):
                # Half-open overlap width between [box_start, box_end) and the
                # column band [c_start, c_end).
                overlap = min(box_end, c_end) - max(box_start, c_start)
                # Strict ``>`` keeps the left-most column on an overlap tie
                # (columns are iterated left-to-right), per Req 3.2's tie-break.
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_index = idx

            if best_overlap > 0:
                return best_index

            # No positive overlap with any band: fall back to the band whose
            # span contains the box centre x, else the nearest band by centre.
            center = x + w // 2
            for idx, (c_start, c_end) in enumerate(columns):
                if c_start <= center < c_end:
                    return idx

            nearest_index = 0
            nearest_dist: "int | None" = None
            for idx, (c_start, c_end) in enumerate(columns):
                band_center = (c_start + c_end) // 2
                dist = abs(center - band_center)
                if nearest_dist is None or dist < nearest_dist:
                    nearest_dist = dist
                    nearest_index = idx
            return nearest_index

        # Stable sort on integer key (column_index, y, x): columns
        # left-to-right, lines top-to-bottom, then left-to-right within a line.
        return sorted(boxes, key=lambda b: (_column_index(b), b[1], b[0]))

    def segment(self, page_image: str) -> "Segmentation_Result":
        """Segment ``page_image`` into ordered word/line crops end-to-end.

        Wires the full classical-CV pipeline in order:
        :meth:`_load_page` -> :func:`_require_cv2` (validate the extra before
        any CV work) -> :meth:`_deskew` -> :meth:`_denoise` -> :meth:`_binarize`
        -> :meth:`_detect_regions` -> :meth:`_detect_columns` ->
        :meth:`_order_reading`, then builds the immutable
        :class:`Segmentation_Result`.

        The ``working`` page -- the grayscale array after deskew and denoise --
        is both the array whose coordinate space is reported *and* the array
        every crop is sliced from (``working[y : y + h, x : x + w]``), which is
        what guarantees the round-trip crop property (Req 6.1): the boxes index
        the same array the crops come from. Binarization is a detection-only
        view; crops are never taken from the binary image.

        :param page_image: Filesystem path to a full-page image.
        :returns: A :class:`Segmentation_Result` whose ``regions`` are in
            Reading_Order with contiguous zero-based indices (Req 5.1, 5.4,
            6.3), whose ``page_width`` / ``page_height`` are the working page's
            dimensions in the reported coordinate space (Req 5.5), and whose
            ``coordinate_space`` is ``"deskewed"`` when deskew ran else
            ``"original"`` (Req 4.4). An empty page yields ``regions == ()``
            (Req 5.6).
        :raises ImageError: If ``page_image`` is missing or undecodable.
        :raises ConfigError: If the ``[segmentation]`` extra (OpenCV) is absent.
        """
        # Local import keeps the top-level import surface numpy + stdlib only
        # (Req 12.1): the data models live in the cv2-free regions module.
        from .regions import Region, Segmentation_Result

        # 1. Load + grayscale (raises ImageError on missing/undecodable input).
        page = self._load_page(page_image)

        # 2. Validate the [segmentation] extra before any CV work (Req 8.3, 8.4).
        _require_cv2()

        # 3-4. Deskew (sets the reported coordinate space) then denoise. The
        # result is the ``working`` page: the array whose coordinate space is
        # reported and the array crops are sliced from (guarantees round-trip).
        working, space = self._deskew(page)
        working = self._denoise(working)

        # 5. Binarize -- a detection-only view; crops come from ``working``.
        binary = self._binarize(working)

        # 6-8. Detect region boxes, detect columns, order into reading order.
        boxes = self._detect_regions(binary)
        columns = self._detect_columns(binary, boxes)
        ordered = self._order_reading(boxes, columns)

        # Build a Region per box, slicing the crop from the SAME working array
        # whose coordinate space is reported, and assign contiguous indices
        # 0..n-1 in final reading order (Req 5.4, 6.3).
        regions: "list[Region]" = []
        for index, (x, y, width, height) in enumerate(ordered):
            crop = np.ascontiguousarray(working[y : y + height, x : x + width])
            regions.append(
                Region(
                    index=index,
                    x=int(x),
                    y=int(y),
                    width=int(width),
                    height=int(height),
                    crop=crop,
                )
            )

        return Segmentation_Result(
            regions=tuple(regions),
            page_width=int(working.shape[1]),
            page_height=int(working.shape[0]),
            coordinate_space=space.value,
        )
