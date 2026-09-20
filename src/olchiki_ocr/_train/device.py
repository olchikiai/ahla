"""Torch-based compute-device selection for the ``[train]`` tier.

This ``select_device`` lives only in ``_train`` (the torch-free core uses the
ONNX Runtime provider selector instead); importing it pulls in torch.
"""

from __future__ import annotations

import torch

from ..errors import DeviceError

__all__ = ["select_device"]


def select_device(requested: str | None) -> str:
    """Return ``'cuda'`` or ``'cpu'``, auto-selecting when ``requested`` is None.

    Raises DeviceError when ``'cuda'`` is requested but no CUDA GPU is available.
    """
    cuda_available = torch.cuda.is_available()

    if requested is None:
        # Auto-select: prefer CUDA, else CPU.
        return "cuda" if cuda_available else "cpu"

    if requested == "cuda":
        if not cuda_available:
            raise DeviceError(requested)
        return "cuda"

    # Any other explicitly requested device (e.g. 'cpu') is used as-is.
    return requested
