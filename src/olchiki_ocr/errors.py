"""Typed exception hierarchy for the ``olchiki_ocr`` package.

This module defines the fatal-error exception hierarchy used across the
package. It provides the typed error classes plus
:class:`ModelDownloadError`.

The base class :class:`OcrTrainerError` retains its name for exit-code
stability and by Design Decision D6.
Although the name mentions "trainer", it is now the **general package error
base** for the whole ``olchiki_ocr`` package (core inference, download/cache,
export, and training tiers alike) — not a trainer-only base.

Each custom exception carries the offending identifier (a parameter name, a
character, a filesystem path, a device string, a URL + cache path, or captured
subprocess output) so error messages are actionable. Each also exposes a
distinct ``exit_code`` so a top-level handler can catch these, print the
message to stderr, and return the corresponding non-zero process exit code.

This module is intentionally pure-stdlib (only ``__future__`` typing): importing
it never pulls in torch, onnxruntime, numpy, Pillow, or easyocr, which is what
lets ``__init__.py`` import the typed errors eagerly.
"""

from __future__ import annotations

__all__ = [
    "OcrTrainerError",
    "ConfigError",
    "CharsetError",
    "IngestError",
    "FontError",
    "PretrainedModelError",
    "DeviceError",
    "DtrbError",
    "OutputError",
    "ImageError",
    "ModelArtifactError",
    "ModelDownloadError",
]


class OcrTrainerError(Exception):
    """Base class for all fatal ``olchiki_ocr`` errors.

    The name is retained for exit-code/compatibility parity with the
    carried-over hierarchy (Design Decision D6); it is now the general package
    error base, not trainer-only. Subclasses set a distinct ``exit_code`` that a
    top-level handler returns as the process exit code when the error propagates
    to the top level.
    """

    #: Default non-zero exit code; overridden by subclasses.
    exit_code: int = 1


class ConfigError(OcrTrainerError):
    """Raised when a Configuration parameter is invalid.

    Carries the name of the offending setting so the message names the invalid
    parameter.
    """

    exit_code: int = 2

    def __init__(self, parameter: str, message: str | None = None) -> None:
        self.parameter = parameter
        if message is None:
            message = f"Invalid configuration parameter: {parameter!r}"
        else:
            message = f"{message} (parameter: {parameter!r})"
        super().__init__(message)


class CharsetError(OcrTrainerError):
    """Raised when a Charset input is invalid.

    Carries the offending value (e.g. an extra character that is not a single
    Unicode code point) so the message identifies it.
    """

    exit_code: int = 3

    def __init__(self, value: str, message: str | None = None) -> None:
        self.value = value
        detail = message if message is not None else "Invalid charset value"
        super().__init__(f"{detail}: {value!r}")


class IngestError(OcrTrainerError):
    """Raised when an input path (such as the Word_List) cannot be read.

    Carries the missing/offending path so the message identifies it.
    """

    exit_code: int = 4

    def __init__(self, path: str, message: str | None = None) -> None:
        self.path = path
        detail = message if message is not None else "Input path does not exist"
        super().__init__(f"{detail}: {path!r}")


class FontError(OcrTrainerError):
    """Raised when a Font_File cannot be loaded.

    Carries the offending font path so the message identifies it.
    """

    exit_code: int = 5

    def __init__(self, path: str, message: str | None = None) -> None:
        self.path = path
        detail = message if message is not None else "Cannot load font file"
        super().__init__(f"{detail}: {path!r}")


class PretrainedModelError(OcrTrainerError):
    """Raised when the pretrained model to fine-tune cannot be loaded.

    Carries the offending model path so the message identifies it.
    """

    exit_code: int = 6

    def __init__(self, path: str, message: str | None = None) -> None:
        self.path = path
        detail = message if message is not None else "Cannot load pretrained model"
        super().__init__(f"{detail}: {path!r}")


class DeviceError(OcrTrainerError):
    """Raised when the requested Compute_Device is unavailable.

    Carries the offending device string so the message identifies it.
    """

    exit_code: int = 7

    def __init__(self, device: str, message: str | None = None) -> None:
        self.device = device
        detail = message if message is not None else "Requested compute device unavailable"
        super().__init__(f"{detail}: {device!r}")


class DtrbError(OcrTrainerError):
    """Raised when a shelled-out DTRB subprocess exits non-zero.

    Carries the captured subprocess ``stderr`` so the failure is diagnosable.
    """

    exit_code: int = 8

    def __init__(self, stderr: str, message: str | None = None) -> None:
        self.stderr = stderr
        detail = message if message is not None else "DTRB subprocess failed"
        super().__init__(f"{detail}: stderr={stderr!r}")


class OutputError(OcrTrainerError):
    """Raised when an output artifact cannot be written.

    Carries the offending path so the message identifies it.
    """

    exit_code: int = 9

    def __init__(self, path: str, message: str | None = None) -> None:
        self.path = path
        detail = message if message is not None else "Cannot write output"
        super().__init__(f"{detail}: {path!r}")


class ImageError(OcrTrainerError):
    """Raised when an input image for inference cannot be read/decoded.

    Carries the offending image path so the message identifies it (Req 1.7).
    """

    exit_code: int = 10

    def __init__(self, path: str, message: str | None = None) -> None:
        self.path = path
        detail = message if message is not None else "Cannot read image"
        super().__init__(f"{detail}: {path!r}")


class ModelArtifactError(OcrTrainerError):
    """Raised when the Model_Artifact cannot be produced or loaded.

    Carries the offending artifact path (or mismatched dimension) so the
    message identifies it (Req 2.6, 2.7, 8.4).
    """

    exit_code: int = 11

    def __init__(self, path: str, message: str | None = None) -> None:
        self.path = path
        detail = message if message is not None else "Invalid model artifact"
        super().__init__(f"{detail}: {path!r}")


class ModelDownloadError(OcrTrainerError):
    """Raised when a Model_Artifact download fails or fails verification.

    Raised on a network error, HTTP error, timeout, missing artifact, or a
    payload whose SHA-256 does not match the published checksum. Carries both
    the ``url`` that was fetched and the ``cache_path`` it was destined for so
    the message names both (Req 4.4, 4.8, 12.2, 12.5).
    """

    exit_code: int = 12

    def __init__(self, url: str, cache_path: str, message: str | None = None) -> None:
        self.url = url
        self.cache_path = cache_path
        detail = message if message is not None else "Model download failed"
        super().__init__(f"{detail}: url={url!r}, cache_path={cache_path!r}")
