"""Typed exception hierarchy for the ``olchiki_ocr`` package.

Pure stdlib, so importing it pulls in no heavy deps and ``__init__.py`` can
import the typed errors eagerly. The base :class:`OcrTrainerError` carries an
``exit_code``; each subclass has a distinct fixed code and carries the
offending identifier for actionable messages.
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

    Subclasses set a distinct ``exit_code`` that a top-level handler returns as
    the process exit code.
    """

    #: Default non-zero exit code; overridden by subclasses.
    exit_code: int = 1


class ConfigError(OcrTrainerError):
    """Raised when a configuration parameter is invalid."""

    exit_code: int = 2

    def __init__(self, parameter: str, message: str | None = None) -> None:
        self.parameter = parameter
        if message is None:
            message = f"Invalid configuration parameter: {parameter!r}"
        else:
            message = f"{message} (parameter: {parameter!r})"
        super().__init__(message)


class CharsetError(OcrTrainerError):
    """Raised when a charset input is invalid."""

    exit_code: int = 3

    def __init__(self, value: str, message: str | None = None) -> None:
        self.value = value
        detail = message if message is not None else "Invalid charset value"
        super().__init__(f"{detail}: {value!r}")


class IngestError(OcrTrainerError):
    """Raised when an input path cannot be read."""

    exit_code: int = 4

    def __init__(self, path: str, message: str | None = None) -> None:
        self.path = path
        detail = message if message is not None else "Input path does not exist"
        super().__init__(f"{detail}: {path!r}")


class FontError(OcrTrainerError):
    """Raised when a font file cannot be loaded."""

    exit_code: int = 5

    def __init__(self, path: str, message: str | None = None) -> None:
        self.path = path
        detail = message if message is not None else "Cannot load font file"
        super().__init__(f"{detail}: {path!r}")


class PretrainedModelError(OcrTrainerError):
    """Raised when the pretrained model to fine-tune cannot be loaded."""

    exit_code: int = 6

    def __init__(self, path: str, message: str | None = None) -> None:
        self.path = path
        detail = message if message is not None else "Cannot load pretrained model"
        super().__init__(f"{detail}: {path!r}")


class DeviceError(OcrTrainerError):
    """Raised when the requested compute device is unavailable."""

    exit_code: int = 7

    def __init__(self, device: str, message: str | None = None) -> None:
        self.device = device
        detail = message if message is not None else "Requested compute device unavailable"
        super().__init__(f"{detail}: {device!r}")


class DtrbError(OcrTrainerError):
    """Raised when a shelled-out DTRB subprocess exits non-zero."""

    exit_code: int = 8

    def __init__(self, stderr: str, message: str | None = None) -> None:
        self.stderr = stderr
        detail = message if message is not None else "DTRB subprocess failed"
        super().__init__(f"{detail}: stderr={stderr!r}")


class OutputError(OcrTrainerError):
    """Raised when an output artifact cannot be written."""

    exit_code: int = 9

    def __init__(self, path: str, message: str | None = None) -> None:
        self.path = path
        detail = message if message is not None else "Cannot write output"
        super().__init__(f"{detail}: {path!r}")


class ImageError(OcrTrainerError):
    """Raised when an input image for inference cannot be read/decoded."""

    exit_code: int = 10

    def __init__(self, path: str, message: str | None = None) -> None:
        self.path = path
        detail = message if message is not None else "Cannot read image"
        super().__init__(f"{detail}: {path!r}")


class ModelArtifactError(OcrTrainerError):
    """Raised when the ONNX artifact is missing or malformed."""

    exit_code: int = 11

    def __init__(self, path: str, message: str | None = None) -> None:
        self.path = path
        detail = message if message is not None else "Invalid model artifact"
        super().__init__(f"{detail}: {path!r}")


class ModelDownloadError(OcrTrainerError):
    """Raised when a model download fails or fails SHA-256 verification.

    Carries both the fetched ``url`` and the destination ``cache_path``.
    """

    exit_code: int = 12

    def __init__(self, url: str, cache_path: str, message: str | None = None) -> None:
        self.url = url
        self.cache_path = cache_path
        detail = message if message is not None else "Model download failed"
        super().__init__(f"{detail}: url={url!r}, cache_path={cache_path!r}")
