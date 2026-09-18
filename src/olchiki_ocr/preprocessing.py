"""Image preprocessing for the ``olchiki_ocr`` core inference tier.

This module implements the composable :class:`Preprocessing_Pipeline` whose
default path exactly matches the model's training-time preprocessing, the DTRB
``None-VGG-BiLSTM-CTC`` ``ResizeNormalize`` transform (design: Components ->
``preprocessing.py``; Req 9, 2.4, 1.7).

Default path (:func:`resize_normalize`, Req 9.1-9.3, 2.4):

1. grayscale (PIL mode ``"L"``),
2. resize to ``imgH = 32`` preserving aspect ratio
   (``W = max(1, round(W0 * 32 / H0))``) using bicubic resampling,
3. convert to a ``float32`` array and normalize ``(x / 255 - 0.5) / 0.5`` so
   ``[0, 1]`` pixel values map to ``[-1, 1]``,
4. return a tensor of shape ``(1, 1, 32, W)`` (batch, grayscale channel,
   ``H = 32``, width).

DTRB parity note: the reference ``ResizeNormalize`` (confirmed against the
carried-over DTRB clone at ``external/deep-text-recognition-benchmark/dataset.py``)
resizes with ``Image.BICUBIC``, then applies ``torchvision.transforms.ToTensor``
(a divide-by-255 to ``[0, 1]``) followed by ``img.sub_(0.5).div_(0.5)``. This
module reproduces that numerically with numpy + Pillow only (no torch), using
``Image.BICUBIC`` and ``(x / 255 - 0.5) / 0.5``.

The core stays torch-free and OpenCV-free at import time: only numpy and Pillow
are imported at module top level. The ``[cv]`` advanced steps (deskew, threshold,
denoise) live behind lazy OpenCV imports and are added by Task 5.2 (see the
marked section near the end of this module); they are intentionally NOT
implemented here.
"""

from __future__ import annotations

from typing import Protocol, Sequence, runtime_checkable

import numpy as np
from PIL import Image, UnidentifiedImageError

__all__ = [
    "PreprocessStep",
    "resize_normalize",
    "Preprocessing_Pipeline",
]

#: Fixed model input height for the DTRB ``None-VGG-BiLSTM-CTC`` geometry
#: (``imgH = 32``; see design Research Notes and ``models/ol_chiki_g2/provenance.json``).
DEFAULT_IMG_H: int = 32


@runtime_checkable
class PreprocessStep(Protocol):
    """A composable preprocessing step.

    A ``PreprocessStep`` is any callable that maps a :class:`PIL.Image.Image`
    to a :class:`PIL.Image.Image`. Steps run on the PIL image *before* the
    numeric ``resize_normalize`` default, so they operate in image space (they
    may change size, mode, or pixel content) while the default guarantees the
    final tensor shape/range invariants (Req 9.4).
    """

    def __call__(self, img: "Image.Image") -> "Image.Image": ...


def resize_normalize(image: "Image.Image", img_h: int = DEFAULT_IMG_H) -> np.ndarray:
    """Apply the DTRB ``ResizeNormalize`` default and return a model tensor.

    Steps (Req 9.1-9.3, 2.4):

    * convert ``image`` to grayscale (PIL mode ``"L"``),
    * resize to height ``img_h`` (default 32) preserving aspect ratio, with
      width ``W = max(1, round(W0 * img_h / H0))`` for the original width
      ``W0`` and height ``H0``, using ``Image.BICUBIC`` (DTRB parity),
    * convert to ``float32`` and normalize ``(x / 255 - 0.5) / 0.5`` mapping
      ``[0, 1]`` to ``[-1, 1]``.

    Args:
        image: A :class:`PIL.Image.Image` of any size or mode.
        img_h: Target height (``imgH``); defaults to 32.

    Returns:
        A ``float32`` :class:`numpy.ndarray` of shape ``(1, 1, img_h, W)`` with
        every value in ``[-1, 1]``.
    """
    # 1. grayscale (Req 9.1)
    gray = image.convert("L")

    # 2. resize to img_h preserving aspect ratio (Req 9.2, DTRB ResizeNormalize)
    w0, h0 = gray.size
    if h0 <= 0:
        # Degenerate/empty image guard; keep at least a 1px width.
        target_w = 1
    else:
        target_w = max(1, int(round(w0 * img_h / h0)))
    resized = gray.resize((target_w, img_h), Image.BICUBIC)

    # 3. to float32 array in [0, 1] then normalize to [-1, 1] (Req 9.3)
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5

    # Clip to guard against BICUBIC overshoot pushing values slightly outside
    # [-1, 1]; this preserves the "every value in [-1, 1]" invariant (Property 2).
    np.clip(arr, -1.0, 1.0, out=arr)

    # 4. shape (1, 1, H, W): batch, grayscale channel, H, W (Req 2.4)
    return arr.reshape(1, 1, img_h, target_w)


class Preprocessing_Pipeline:
    """Composable image preprocessing around the DTRB ``ResizeNormalize`` default.

    Composition contract (Req 9.4):

    Both ``pre`` and ``post`` steps are :class:`PreprocessStep` callables
    (``PIL.Image -> PIL.Image``) that run *before* the numeric default. The
    application order in :meth:`to_tensor` is::

        pre[0] -> ... -> pre[n] -> post[0] -> ... -> post[m] -> resize_normalize

    Rationale: the default (:func:`resize_normalize`) returns a numeric
    ``float32`` ndarray, not an image, so a step that ran *after* it could not be
    a ``PIL.Image -> PIL.Image`` callable and could trivially break the tensor
    shape/range invariants. Keeping every user step in image space and applying
    the default last guarantees Property 2 (the output is always ``float32`` of
    shape ``(1, 1, 32, W)`` with values in ``[-1, 1]``) regardless of the steps
    supplied. ``pre`` and ``post`` are two ordered groups so callers can express
    "always-first" versus "just-before-the-default" intent; both are honored in
    the order listed, with all of ``pre`` running ahead of all of ``post``.
    """

    def __init__(
        self,
        pre: Sequence[PreprocessStep] = (),
        post: Sequence[PreprocessStep] = (),
        *,
        img_h: int = DEFAULT_IMG_H,
    ) -> None:
        self.pre: tuple[PreprocessStep, ...] = tuple(pre)
        self.post: tuple[PreprocessStep, ...] = tuple(post)
        self.img_h = img_h

    def to_tensor(self, image_path: str) -> np.ndarray:
        """Open ``image_path`` and produce the model input tensor.

        Opens the image with Pillow, applies the ``pre`` steps then the ``post``
        steps (both in order, in image space), then the default
        :func:`resize_normalize`, and returns a ``float32`` tensor of shape
        ``(1, 1, 32, W)`` in ``[-1, 1]`` (Req 2.4, 9.1-9.4).

        Raises:
            ImageError: If ``image_path`` cannot be opened or decoded (Req 1.7).
                The raised error names the offending path.
        """
        # Import here to avoid a hard module-level dependency edge; errors is
        # pure-stdlib and cheap, but this keeps the top-level import surface
        # focused on numpy + Pillow.
        from .errors import ImageError

        try:
            with Image.open(image_path) as img:
                # Force decode while the file handle is open so a truncated or
                # corrupt payload raises here (and is re-raised as ImageError).
                img.load()
                pil = img.copy()
        except (FileNotFoundError, OSError, ValueError, UnidentifiedImageError) as exc:
            raise ImageError(image_path) from exc

        for step in self.pre:
            pil = step(pil)
        for step in self.post:
            pil = step(pil)

        return resize_normalize(pil, img_h=self.img_h)


# ---------------------------------------------------------------------------
# [cv] advanced steps (Task 5.2) -- deskew / threshold / denoise.
#
# These are implemented behind LAZY OpenCV imports so the core stays
# OpenCV-free at import time; invoking one without the ``[cv]`` extra installed
# raises an error naming the missing ``[cv]`` extra (Req 3.7, 9.5). Task 5.2
# appends the implementation (and a helper that raises the missing-extra error)
# below this marker. Intentionally not implemented in Task 5.1.
# ---------------------------------------------------------------------------

# Extend the public surface with the [cv]-gated advanced steps (Req 9.5).
__all__ += ["deskew", "threshold", "denoise"]

#: Message naming the missing ``[cv]`` extra, shared by :func:`_require_cv2` so
#: the identifier ``[cv]`` and the install hint are consistent everywhere
#: (Req 3.7).
_CV_EXTRA_MESSAGE: str = (
    "the '[cv]' extra is required for advanced preprocessing "
    "(install: pip install olchiki-ocr[cv])"
)


def _require_cv2():
    """Import and return the OpenCV (``cv2``) module lazily.

    The core stays OpenCV-free at import time (Task 5.1 invariant): ``cv2`` is
    never imported at module top level and is only imported on first use inside
    this helper. Every ``[cv]`` advanced step calls this first.

    Design choice for the missing-extra error (Req 3.4, 3.7): the design's
    error hierarchy has no dedicated missing-dependency type. We raise
    :class:`~olchiki_ocr.errors.ConfigError` — a typed
    :class:`~olchiki_ocr.errors.OcrTrainerError` subclass (``exit_code = 2``) —
    so the failure integrates with the CLI exit-code handling like every other
    fatal package error, rather than surfacing as a bare ``ImportError`` that
    the CLI would map to the generic exit code 1. The ``ConfigError`` parameter
    is ``"[cv]"`` and its message names the ``[cv]`` extra plus the install
    hint, satisfying "raise an error naming the missing ``[cv]`` extra".

    Returns:
        The imported ``cv2`` module.

    Raises:
        ConfigError: If ``cv2`` (the ``[cv]`` extra / ``opencv-python``) is not
            installed. The error names the ``[cv]`` extra (Req 3.7, 9.5).
    """
    try:
        import cv2  # noqa: PLC0415 - lazy import is intentional (keep core cv2-free)
    except ImportError as exc:
        # Imported here to keep the top-level import surface numpy + Pillow only.
        from .errors import ConfigError

        raise ConfigError("[cv]", _CV_EXTRA_MESSAGE) from exc
    return cv2


def _to_gray_array(img: "Image.Image") -> np.ndarray:
    """Convert a PIL image to a contiguous ``uint8`` grayscale ndarray for cv2."""
    gray = img.convert("L")
    return np.ascontiguousarray(np.asarray(gray, dtype=np.uint8))


def deskew(img: "Image.Image") -> "Image.Image":
    """Estimate and correct the skew angle of ``img`` (``[cv]`` extra, Req 9.5).

    Algorithm: binarize the grayscale image with an inverted Otsu threshold so
    foreground (text) pixels are non-zero, collect their coordinates, estimate
    the dominant orientation via :func:`cv2.minAreaRect`, normalize the returned
    angle into ``[-45, 45]`` degrees, and rotate the original image by the
    negative of that angle (with a white border fill) to level the text.

    A :class:`PreprocessStep` (``PIL.Image -> PIL.Image``) usable in a
    :class:`Preprocessing_Pipeline` ``pre``/``post`` list. Requires the ``[cv]``
    extra; without it, :func:`_require_cv2` raises a ``ConfigError`` naming
    ``[cv]``.
    """
    cv2 = _require_cv2()
    gray = _to_gray_array(img)

    # Foreground as non-zero pixels (invert so dark text -> white on black).
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None:
        # No foreground detected (blank image); nothing to deskew.
        return img

    angle = cv2.minAreaRect(coords)[-1]
    # cv2 returns the angle in (-90, 0]; normalize into [-45, 45] so we rotate
    # by the smallest correction rather than flipping the page.
    if angle < -45:
        angle = 90 + angle
    # Rotate by -angle to counteract the detected skew.
    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,  # white fill so padding matches typical paper background
    )
    return Image.fromarray(rotated, mode="L")


def threshold(img: "Image.Image") -> "Image.Image":
    """Binarize ``img`` with Otsu's method (``[cv]`` extra, Req 9.5).

    Converts to grayscale and applies a global Otsu threshold, producing a
    two-level (black/white) image. This is a common, reasonable binarization
    that adapts its cut point to the image's intensity histogram.

    A :class:`PreprocessStep` (``PIL.Image -> PIL.Image``). Requires the
    ``[cv]`` extra; without it, :func:`_require_cv2` raises a ``ConfigError``
    naming ``[cv]``.
    """
    cv2 = _require_cv2()
    gray = _to_gray_array(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary, mode="L")


def denoise(img: "Image.Image") -> "Image.Image":
    """Reduce noise in ``img`` (``[cv]`` extra, Req 9.5).

    Converts to grayscale and applies :func:`cv2.fastNlMeansDenoising`
    (non-local means), a common, reasonable denoiser that smooths noise while
    preserving edges. Falls back to a median blur only if the fast NL-means
    variant is unavailable in the installed OpenCV build.

    A :class:`PreprocessStep` (``PIL.Image -> PIL.Image``). Requires the
    ``[cv]`` extra; without it, :func:`_require_cv2` raises a ``ConfigError``
    naming ``[cv]``.
    """
    cv2 = _require_cv2()
    gray = _to_gray_array(img)
    fast_nl_means = getattr(cv2, "fastNlMeansDenoising", None)
    if fast_nl_means is not None:
        result = fast_nl_means(gray, None, 10, 7, 21)
    else:  # pragma: no cover - depends on the installed OpenCV build
        result = cv2.medianBlur(gray, 3)
    return Image.fromarray(result, mode="L")
