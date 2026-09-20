"""Image preprocessing for the ``olchiki_ocr`` core inference tier.

The default path (:func:`resize_normalize`) reproduces the DTRB
``ResizeNormalize`` contract with numpy + Pillow (no torch): grayscale, resize
to ``imgH = 32`` preserving aspect ratio (bicubic), normalize
``(x / 255 - 0.5) / 0.5`` to ``[-1, 1]``, and return a ``(1, 1, 32, W)`` tensor.

Only numpy and Pillow are imported at module top; the ``[cv]`` advanced steps
(deskew/threshold/denoise) import cv2 lazily so the core stays OpenCV-free.
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

#: Fixed model input height (``imgH = 32``) for the DTRB geometry.
DEFAULT_IMG_H: int = 32


@runtime_checkable
class PreprocessStep(Protocol):
    """A composable ``PIL.Image -> PIL.Image`` preprocessing step.

    Steps run in image space before the numeric ``resize_normalize`` default,
    which guarantees the final tensor shape/range invariants.
    """

    def __call__(self, img: "Image.Image") -> "Image.Image": ...


def resize_normalize(image: "Image.Image", img_h: int = DEFAULT_IMG_H) -> np.ndarray:
    """Apply the DTRB ``ResizeNormalize`` default and return a model tensor.

    Grayscale, resize to height ``img_h`` (default 32) preserving aspect ratio
    (bicubic), normalize ``(x / 255 - 0.5) / 0.5``. Returns a ``float32``
    ``(1, 1, img_h, W)`` array with every value in ``[-1, 1]``.
    """
    gray = image.convert("L")

    # Resize to img_h preserving aspect ratio (DTRB uses BICUBIC).
    w0, h0 = gray.size
    if h0 <= 0:
        target_w = 1  # degenerate/empty image guard
    else:
        target_w = max(1, int(round(w0 * img_h / h0)))
    resized = gray.resize((target_w, img_h), Image.BICUBIC)

    # To [0, 1] then normalize to [-1, 1].
    arr = np.asarray(resized, dtype=np.float32) / 255.0
    arr = (arr - 0.5) / 0.5

    # Clip against BICUBIC overshoot so every value stays in [-1, 1].
    np.clip(arr, -1.0, 1.0, out=arr)

    # (1, 1, H, W): batch, grayscale channel, H, W.
    return arr.reshape(1, 1, img_h, target_w)


class Preprocessing_Pipeline:
    """Composable image preprocessing around the DTRB ``ResizeNormalize`` default.

    All ``pre`` then all ``post`` steps run (in image space) before the numeric
    default::

        pre[0] -> ... -> post[0] -> ... -> resize_normalize

    The default runs last and returns the ndarray, so keeping user steps in
    image space guarantees the output is always ``float32`` ``(1, 1, 32, W)`` in
    ``[-1, 1]`` regardless of the steps supplied. ``pre``/``post`` are two
    ordered groups for "always-first" vs "just-before-default" intent.
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

        Applies ``pre`` then ``post`` steps (in order) then
        :func:`resize_normalize`, returning a ``float32`` ``(1, 1, 32, W)``
        tensor in ``[-1, 1]``. Raises ``ImageError`` (naming the path) if the
        image can't be opened or decoded.
        """
        # Local import keeps the top-level import surface to numpy + Pillow.
        from .errors import ImageError

        try:
            with Image.open(image_path) as img:
                # Force decode while the handle is open so a corrupt payload
                # raises here (re-raised as ImageError).
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
# [cv] advanced steps -- deskew / threshold / denoise. Implemented behind lazy
# OpenCV imports so the core stays OpenCV-free at import time; invoking one
# without the [cv] extra raises an error naming the missing [cv] extra.
# ---------------------------------------------------------------------------

__all__ += ["deskew", "threshold", "denoise"]

#: Message naming the missing ``[cv]`` extra, shared by :func:`_require_cv2`.
_CV_EXTRA_MESSAGE: str = (
    "the '[cv]' extra is required for advanced preprocessing "
    "(install: pip install olchiki-ocr[cv])"
)


def _require_cv2():
    """Import and return the OpenCV (``cv2``) module lazily.

    Keeps the core OpenCV-free at import time; every ``[cv]`` step calls this
    first. Raises ``ConfigError`` naming the ``[cv]`` extra if cv2 is missing
    (ConfigError so it maps to the CLI's config exit code, not a bare
    ImportError).
    """
    try:
        import cv2  # noqa: PLC0415 - lazy import is intentional (keep core cv2-free)
    except ImportError as exc:
        # Local import keeps the top-level import surface numpy + Pillow only.
        from .errors import ConfigError

        raise ConfigError("[cv]", _CV_EXTRA_MESSAGE) from exc
    return cv2


def _to_gray_array(img: "Image.Image") -> np.ndarray:
    """Convert a PIL image to a contiguous ``uint8`` grayscale ndarray for cv2."""
    gray = img.convert("L")
    return np.ascontiguousarray(np.asarray(gray, dtype=np.uint8))


def deskew(img: "Image.Image") -> "Image.Image":
    """Estimate and correct the skew angle of ``img`` (``[cv]`` extra).

    Binarize (inverted Otsu), estimate orientation via ``cv2.minAreaRect``,
    normalize into ``[-45, 45]`` degrees, and rotate to level the text.
    A :class:`PreprocessStep`; requires the ``[cv]`` extra.
    """
    cv2 = _require_cv2()
    gray = _to_gray_array(img)

    # Invert so dark text -> non-zero foreground.
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = cv2.findNonZero(binary)
    if coords is None:
        return img  # blank image, nothing to deskew

    angle = cv2.minAreaRect(coords)[-1]
    # cv2 returns (-90, 0]; normalize into [-45, 45] for the smallest correction.
    if angle < -45:
        angle = 90 + angle
    h, w = gray.shape[:2]
    center = (w / 2.0, h / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        gray,
        matrix,
        (w, h),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=255,  # white fill matches typical paper background
    )
    return Image.fromarray(rotated, mode="L")


def threshold(img: "Image.Image") -> "Image.Image":
    """Binarize ``img`` with a global Otsu threshold (``[cv]`` extra).

    A :class:`PreprocessStep`; requires the ``[cv]`` extra.
    """
    cv2 = _require_cv2()
    gray = _to_gray_array(img)
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return Image.fromarray(binary, mode="L")


def denoise(img: "Image.Image") -> "Image.Image":
    """Reduce noise in ``img`` via non-local means (``[cv]`` extra).

    Falls back to a median blur if fast NL-means isn't in the OpenCV build.
    A :class:`PreprocessStep`; requires the ``[cv]`` extra.
    """
    cv2 = _require_cv2()
    gray = _to_gray_array(img)
    fast_nl_means = getattr(cv2, "fastNlMeansDenoising", None)
    if fast_nl_means is not None:
        result = fast_nl_means(gray, None, 10, 7, 21)
    else:  # pragma: no cover - depends on the installed OpenCV build
        result = cv2.medianBlur(gray, 3)
    return Image.fromarray(result, mode="L")
