"""Segmentation output data models for the ``olchiki_ocr`` document segmenter.

These are the immutable output contract for :class:`PageSegmenter`: an ordered
sequence of :class:`Region` objects (each carrying an integer bounding box, a
grayscale crop, and a zero-based contiguous index) wrapped in a
:class:`Segmentation_Result` with page dimensions and the coordinate space the
boxes live in.

The module is intentionally dependency-light -- only numpy, ``dataclasses``,
and stdlib are imported at module top (no cv2, no torch) -- consistent with the
package's torch-free lazy-import convention (see ``preprocessing.py``), so the
data model can be imported and constructed in tests without OpenCV.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

__all__ = ["Region", "Segmentation_Result"]


@dataclass(frozen=True)
class Region:
    """A single detected text unit (word or line) on a segmented page.

    Frozen so it is hashable, immutable, and safe to reuse. The ``crop`` is a
    ``uint8`` grayscale ``(height, width)`` ndarray equal to the working page
    sliced to this region's bounding box (``page[y : y + height, x : x + width]``),
    which is what makes the round-trip crop property hold.
    """

    #: Zero-based position in reading order (equals the reading-order index).
    index: int
    #: Left edge of the bounding box in page pixel coordinates.
    x: int
    #: Top edge of the bounding box in page pixel coordinates.
    y: int
    #: Bounding-box width in pixels.
    width: int
    #: Bounding-box height in pixels.
    height: int
    #: Grayscale ``uint8`` crop, shape ``(height, width)``.
    crop: np.ndarray

    @property
    def bounding_box(self) -> tuple[int, int, int, int]:
        """Return the axis-aligned bounding box as ``(x, y, width, height)``."""
        return (self.x, self.y, self.width, self.height)


@dataclass(frozen=True)
class Segmentation_Result:
    """The ordered, structured result of segmenting one page.

    Carries the :class:`Region` sequence (in reading order), the page
    dimensions in the reported coordinate space, and the ``coordinate_space``
    label (``"original"`` or ``"deskewed"``). An empty page yields
    ``regions == ()``.
    """

    #: Ordered regions in reading order (top-to-bottom, then left-to-right).
    regions: tuple[Region, ...]
    #: Page width in pixels, in the reported coordinate space.
    page_width: int
    #: Page height in pixels, in the reported coordinate space.
    page_height: int
    #: Coordinate space the boxes live in: ``"original"`` or ``"deskewed"``.
    coordinate_space: str

    def __len__(self) -> int:
        """Return the number of regions in the result."""
        return len(self.regions)

    def __iter__(self) -> Iterator[Region]:
        """Iterate over the regions in reading order."""
        return iter(self.regions)

    def write_crops(self, out_dir: str) -> list[str]:
        """Write each region crop to ``out_dir`` as an indexed PNG file.

        Creates ``out_dir`` if needed, then writes one PNG per region named by
        its zero-based index, zero-padded so lexical filename ordering matches
        region (reading) order (e.g. ``00.png``, ``01.png``, ...). The pad
        width is chosen from the number of regions.

        Pillow is imported locally so this module stays importable with only
        numpy + stdlib at the top level (matching the package's lazy-import
        convention).

        :param out_dir: Destination directory for the crop image files.
        :returns: The list of written file paths, in region order.
        :raises OutputError: If the directory cannot be created or a crop file
            cannot be written (naming ``out_dir``).
        """
        # Local imports keep the module's top-level import surface to numpy +
        # dataclasses + stdlib.
        import os

        from PIL import Image

        from .errors import OutputError

        # Zero-pad width so filenames sort lexically in index order. At least
        # width 1 (for the empty/single-region case).
        pad = max(1, len(str(len(self.regions) - 1))) if self.regions else 1

        try:
            os.makedirs(out_dir, exist_ok=True)
        except (OSError, PermissionError) as exc:
            raise OutputError(out_dir) from exc

        written: list[str] = []
        try:
            for region in self.regions:
                filename = f"{region.index:0{pad}d}.png"
                path = os.path.join(out_dir, filename)
                Image.fromarray(region.crop, mode="L").save(path)
                written.append(path)
        except (OSError, PermissionError) as exc:
            raise OutputError(out_dir) from exc

        return written
