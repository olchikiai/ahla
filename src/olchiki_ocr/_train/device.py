"""Compute device selection for the ``[train]`` tier (torch-based).

Resolves the hardware target (CUDA GPU or CPU) once, early in a training run,
so the selected device can be recorded in the Run_Report and reused by the
Trainer.

Per Design Decision D9, the torch-based ``select_device`` lives ONLY in the
``_train`` tier. The torch-free core uses an ONNX Runtime provider selector
(``olchiki_ocr.session.select_providers``) instead; importing this module pulls
in torch and is therefore gated behind the ``[train]`` extra.

Selection rules:

* When no device is requested, prefer a CUDA GPU if one is available and fall
  back to the CPU otherwise.
* When a device is explicitly requested and it is available, use it.
* When a CUDA GPU is explicitly requested but none is available, raise
  ``DeviceError`` naming the requested device.

The caller records the returned device string in ``RunStats.compute_device``;
this function only resolves and returns the string.
"""

from __future__ import annotations

import torch

from ..errors import DeviceError

__all__ = ["select_device"]


def select_device(requested: str | None) -> str:
    """Return the Compute_Device string, either ``'cuda'`` or ``'cpu'``.

    Args:
        requested: The Configuration-specified device (``'cuda'`` or ``'cpu'``),
            or ``None`` to auto-select.

    Returns:
        ``'cuda'`` when a CUDA GPU is used, otherwise ``'cpu'``.

    Raises:
        DeviceError: When ``requested == 'cuda'`` but no CUDA GPU is available.
            The error names the requested device.
    """
    cuda_available = torch.cuda.is_available()

    if requested is None:
        # Auto-select: prefer CUDA when present, else fall back to CPU.
        return "cuda" if cuda_available else "cpu"

    if requested == "cuda":
        if not cuda_available:
            # CUDA explicitly requested but unavailable -> fatal.
            raise DeviceError(requested)
        return "cuda"

    # Any other explicitly requested device (e.g. 'cpu') is used as-is.
    return requested
