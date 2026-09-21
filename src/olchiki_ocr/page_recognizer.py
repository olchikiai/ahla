"""End-to-end page-to-text orchestration for the ``olchiki_ocr`` document segmenter.

This module hosts :class:`Page_Recognizer`, the orchestrator that runs the
:class:`~olchiki_ocr.segmentation.PageSegmenter` and then feeds each detected
region crop to the existing ``ModelRecognizer`` through its published
path-based interface, plus the immutable output data models
:class:`RecognizedRegion` and :class:`Page_Result`.

Only ``dataclasses`` and stdlib are imported at module top (no cv2, no torch,
no onnxruntime) -- consistent with the package's torch-free lazy-import
convention (see ``preprocessing.py`` and ``regions.py``). Heavy collaborators
(the recognizer, the segmenter, Pillow) are imported lazily inside the methods
that need them so a bare ``import olchiki_ocr`` stays engine-free.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator

if TYPE_CHECKING:  # typing-only; no runtime import so the module stays engine-free
    from .recognizer import ModelRecognizer
    from .segmentation import PageSegmenter

__all__ = ["Page_Recognizer", "Page_Result", "RecognizedRegion"]


@dataclass(frozen=True)
class RecognizedRegion:
    """A single recognized text unit on a page: geometry plus recognized text.

    Frozen so it is hashable, immutable, and safe to reuse. Mirrors the
    geometry of :class:`~olchiki_ocr.regions.Region` (its zero-based reading
    order index and integer bounding box) and adds the recognized ``text`` and
    an optional ``confidence`` in ``[0, 1]`` (``None`` when confidence was not
    requested).
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
    #: Recognized text for this region.
    text: str
    #: Optional recognition confidence in ``[0, 1]``; ``None`` when not requested.
    confidence: float | None = None


@dataclass(frozen=True)
class Page_Result:
    """The ordered, structured result of recognizing one page.

    Carries the :class:`RecognizedRegion` sequence (in reading order) and the
    page dimensions. An empty page yields ``regions == ()``.
    """

    #: Ordered recognized regions in reading order (top-to-bottom, then left-to-right).
    regions: tuple[RecognizedRegion, ...]
    #: Page width in pixels, in the reported coordinate space.
    page_width: int
    #: Page height in pixels, in the reported coordinate space.
    page_height: int

    def __len__(self) -> int:
        """Return the number of recognized regions in the result."""
        return len(self.regions)

    def __iter__(self) -> Iterator[RecognizedRegion]:
        """Iterate over the recognized regions in reading order."""
        return iter(self.regions)


class Page_Recognizer:
    """End-to-end page-to-text orchestrator: segment a page, then recognize it.

    Wraps a :class:`~olchiki_ocr.segmentation.PageSegmenter` (which turns a full
    page image into ordered word/line crops) and a ``ModelRecognizer`` (which
    recognizes a single pre-cropped image). :meth:`recognize_page` (task 5.3)
    wires them together; the public initializer takes the already-built
    collaborators so the class stays testable and the ``from_pretrained``
    resolution policy lives in one place -- mirroring ``ModelRecognizer``.

    The heavy collaborators are imported lazily inside :meth:`from_pretrained`
    so a bare ``import olchiki_ocr.page_recognizer`` imports neither the
    recognition engine (onnxruntime) nor OpenCV (Req 12.2, 12.3).
    """

    def __init__(
        self, recognizer: "ModelRecognizer", segmenter: "PageSegmenter"
    ) -> None:
        """Store the resolved collaborators (usually built by ``from_pretrained``).

        :param recognizer: A ready-to-use ``ModelRecognizer`` whose published
            ``predict`` interface recognizes a single cropped image.
        :param segmenter: A configured ``PageSegmenter`` that turns a full page
            image into ordered ``Region`` crops.
        """
        self._recognizer = recognizer
        self._segmenter = segmenter

    @property
    def recognizer(self) -> "ModelRecognizer":
        """The underlying recognizer used to recognize each region crop."""
        return self._recognizer

    @property
    def segmenter(self) -> "PageSegmenter":
        """The underlying segmenter used to split a page into region crops."""
        return self._segmenter

    @classmethod
    def from_pretrained(
        cls,
        path: str | None = None,
        *,
        granularity: str = "word",
        deskew: bool = True,
        denoise: bool = True,
        min_region_area: int = 20,
        model_source: str | None = None,
        cache_dir: str | None = None,
        device: str | None = None,
        decoder: str = "greedy",
    ) -> "Page_Recognizer":
        """Build a ready-to-use orchestrator with one-line setup.

        Constructs a ``ModelRecognizer`` via its existing
        :meth:`ModelRecognizer.from_pretrained` classmethod (forwarding the
        model/device/decoder/cache/source parameters unchanged) and a
        :class:`~olchiki_ocr.segmentation.PageSegmenter` from the segmentation
        parameters, then returns the orchestrator wrapping both. Mirrors
        ``ModelRecognizer.from_pretrained`` so callers get a single entry point.

        The recognizer and segmenter modules are imported lazily here (not at
        module top) so ``import olchiki_ocr.page_recognizer`` stays engine-free
        and OpenCV-free (Req 12.2, 12.3).

        :param path: Local artifact directory for the recognizer, or ``None`` to
            resolve from cache/download.
        :param granularity: Segmentation granularity (``"word"`` or ``"line"``).
        :param deskew: Whether the segmenter levels page skew before detection.
        :param denoise: Whether the segmenter denoises the page before detection.
        :param min_region_area: Minimum region area (px^2) below which detected
            components are dropped as speckle.
        :param model_source: Optional override for the recognizer's model source.
        :param cache_dir: Optional recognizer artifact cache directory.
        :param device: Recognizer device (``None``/``"cpu"`` for CPU,
            ``"gpu"``/``"cuda"`` for GPU).
        :param decoder: Recognizer default decoder (``"greedy"`` or ``"beam"``).
        :returns: A configured :class:`Page_Recognizer`.
        :raises ConfigError: For an invalid ``granularity``/``min_region_area``
            (from the segmenter) or an unknown ``decoder`` (from the recognizer).
        """
        # Lazy imports keep the top-level import surface dataclasses + stdlib
        # only: building the recognizer pulls in onnxruntime and building the
        # segmenter can pull in OpenCV, neither of which should load at
        # ``import olchiki_ocr.page_recognizer`` time (Req 12.2, 12.3).
        from .recognizer import ModelRecognizer
        from .segmentation import PageSegmenter

        recognizer = ModelRecognizer.from_pretrained(
            path,
            model_source=model_source,
            cache_dir=cache_dir,
            device=device,
            decoder=decoder,
        )
        segmenter = PageSegmenter(
            granularity=granularity,
            deskew=deskew,
            denoise=denoise,
            min_region_area=min_region_area,
        )
        return cls(recognizer, segmenter)

    def recognize_page(
        self,
        page_image: str,
        *,
        confidence: bool = False,
        decoder: str | None = None,
        beam_width: int = 10,
    ) -> "Page_Result":
        """Segment a full page image, then recognize each region in reading order.

        Runs the segmenter on ``page_image`` to obtain an ordered
        ``Segmentation_Result``, then for each region (in reading order)
        materializes the region crop to a temporary PNG file, recognizes it via
        the recognizer's published path-based ``predict`` interface, and
        assembles a :class:`RecognizedRegion` preserving the source region's
        zero-based index and bounding box. The temporary file is deleted
        deterministically after each region (even on failure).

        The recognizer's ``predict`` signature is used unchanged: it consumes a
        filesystem path, returning a ``str`` when ``confidence=False`` and a
        ``Prediction`` (with ``.text`` and ``.confidence``) when
        ``confidence=True`` (Req 7.6). An empty ``Segmentation_Result`` yields a
        :class:`Page_Result` with empty ``regions`` and makes no ``predict``
        calls (Req 7.5).

        Pillow, ``tempfile`` and ``os`` are imported lazily here so the module
        top-level import surface stays dataclasses + stdlib only (Req 12.2, 12.3).

        :param page_image: Filesystem path to the full page image to recognize.
        :param confidence: When ``True``, request a confidence value in
            ``[0, 1]`` for each recognized region (Req 7.4).
        :param decoder: Optional per-call decoder override forwarded to the
            recognizer (``None`` uses the recognizer default).
        :param beam_width: Beam width forwarded to the recognizer when the beam
            decoder is used.
        :returns: A :class:`Page_Result` with one :class:`RecognizedRegion` per
            detected region, in reading order (Req 7.1, 7.2, 7.3).
        :raises ImageError: If the page image cannot be read/decoded (from the
            segmenter).
        :raises ConfigError: For an invalid configuration value (e.g. an
            unknown decoder) raised by the collaborators.
        """
        # Lazy imports keep the module top-level surface to dataclasses + stdlib
        # only; Pillow is only needed when we actually materialize a crop.
        import os
        import tempfile

        from PIL import Image

        # Segment the page into ordered region crops (Req 7.1). ImageError /
        # ConfigError from the segmenter propagate unchanged.
        result = self._segmenter.segment(page_image)

        # Empty page -> empty result, and make no predict calls (Req 7.5).
        recognized: list[RecognizedRegion] = []
        for region in result.regions:
            # Materialize the crop to a temp PNG the recognizer's path-based
            # predict can consume, then delete it deterministically. Create the
            # temp file with delete=False so we control the (Windows-safe)
            # close-then-reopen-then-remove lifecycle via try/finally.
            handle = tempfile.NamedTemporaryFile(suffix=".png", delete=False)
            tmp_path = handle.name
            handle.close()
            try:
                Image.fromarray(region.crop, mode="L").save(tmp_path)
                prediction = self._recognizer.predict(
                    tmp_path,
                    confidence=confidence,
                    decoder=decoder,
                    beam_width=beam_width,
                )
            finally:
                try:
                    os.remove(tmp_path)
                except OSError:
                    # Best-effort cleanup; never mask the real result/error.
                    pass

            # predict returns a bare str when confidence is not requested, and a
            # Prediction (with .text/.confidence) when it is (Req 7.4).
            if confidence:
                text = prediction.text
                conf = prediction.confidence
            else:
                text = prediction
                conf = None

            recognized.append(
                RecognizedRegion(
                    index=region.index,
                    x=region.x,
                    y=region.y,
                    width=region.width,
                    height=region.height,
                    text=text,
                    confidence=conf,
                )
            )

        return Page_Result(
            regions=tuple(recognized),
            page_width=result.page_width,
            page_height=result.page_height,
        )
