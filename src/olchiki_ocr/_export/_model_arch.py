"""Self-contained ``None-VGG-BiLSTM-CTC`` architecture for the ``[export]`` tier.

This module vendors a **minimal, self-contained** PyTorch definition of the DTRB
``None-VGG-BiLSTM-CTC`` recognition stack that the trained Ol Chiki weights
(``models/ol_chiki_g2/model/ol_chiki_g2.pth``) were produced with. It exists so the
``[export]`` tier can load the ``.pth`` and export it to ONNX using only
``torch`` (plus ``onnx`` at export time) -- **without** importing ``easyocr`` or
depending on the external DTRB clone / the ``[train]`` tier (Design Decision D1:
``[export]`` = ``torch`` + ``onnx`` only).

Why vendor instead of import
----------------------------
The design (Task 16.1) explicitly weighs three options:

* import ``easyocr.model.vgg_model.Model`` -- rejected: ``easyocr`` is **not** a
  declared ``[export]`` dependency (``[export]`` = ``torch`` + ``onnx``), and it
  pulls a large runtime stack.
* import the DTRB clone's ``model.Model`` -- rejected as the primary path: the
  clone uses top-level ``from modules.x import y`` imports that only resolve
  with the clone directory on ``sys.path``, and requires an ``opt`` argparse
  namespace; wiring that in couples ``[export]`` to the ``[train]`` clone.
* **vendor a minimal ``Model``** -- chosen: the ``None-VGG-BiLSTM-CTC`` stack is
  small and stable. Vendoring it keeps ``[export]`` self-contained and
  independent of ``[train]`` while still producing byte-compatible state-dict
  keys.

State-dict key compatibility (LOAD-BEARING)
-------------------------------------------
The submodule **attribute names and internal structure here mirror the DTRB
``model.Model`` exactly** so the trained checkpoint's ``state_dict`` keys line
up 1:1:

* ``FeatureExtraction.ConvNet.*`` -- the VGG feature extractor
  (``VGG_FeatureExtractor``; keys ``FeatureExtraction.ConvNet.0.weight`` ...).
* ``SequenceModeling.0`` / ``SequenceModeling.1`` -- two stacked
  ``BidirectionalLSTM`` layers (``SequenceModeling.0.rnn.*``,
  ``SequenceModeling.0.linear.*``, ...).
* ``Prediction`` -- the CTC ``nn.Linear`` head
  (``Prediction.weight`` shape ``(num_class, hidden_size)`` = ``(49, 256)``).

The ``AdaptiveAvgPool`` layer has no parameters (so no state-dict keys) but is
kept to reproduce the exact forward geometry.

DTRB checkpoints are saved from a ``DataParallel``-wrapped model, so every key
carries a ``module.`` prefix (e.g. ``module.FeatureExtraction.ConvNet.0.weight``).
:func:`load_dtrb_checkpoint` strips that prefix; see its docstring.

The forward output is ``(N, T, num_class)`` (batch-first), matching DTRB's CTC
branch (``prediction = self.Prediction(contextual_feature.contiguous())``).

Everything in this module imports ``torch`` at module top level, which is why
this module lives under ``_export`` and is imported lazily by the tooling only
after the ``[export]`` extra has been confirmed present.
"""

from __future__ import annotations

import torch
import torch.nn as nn

__all__ = [
    "VGG_FeatureExtractor",
    "BidirectionalLSTM",
    "NoneVGGBiLSTMCTC",
    "build_model",
    "load_dtrb_checkpoint",
]


class VGG_FeatureExtractor(nn.Module):
    """VGG feature extractor of the DTRB CRNN stack (vendored verbatim).

    Structure and layer order copied exactly from DTRB
    ``modules/feature_extraction.py::VGG_FeatureExtractor`` so the ``ConvNet.*``
    state-dict keys match the trained checkpoint.
    """

    def __init__(self, input_channel: int, output_channel: int = 512) -> None:
        super().__init__()
        self.output_channel = [
            int(output_channel / 8),
            int(output_channel / 4),
            int(output_channel / 2),
            output_channel,
        ]  # [64, 128, 256, 512]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(
                self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False
            ),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
            nn.Conv2d(
                self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False
            ),
            nn.BatchNorm2d(self.output_channel[3]),
            nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0),
            nn.ReLU(True),
        )

    def forward(self, input: "torch.Tensor") -> "torch.Tensor":
        return self.ConvNet(input)


class BidirectionalLSTM(nn.Module):
    """Bidirectional LSTM + linear projection (vendored verbatim from DTRB).

    Copied from DTRB ``modules/sequence_modeling.py::BidirectionalLSTM`` so the
    ``rnn.*`` / ``linear.*`` state-dict keys match the trained checkpoint.
    """

    def __init__(self, input_size: int, hidden_size: int, output_size: int) -> None:
        super().__init__()
        self.rnn = nn.LSTM(
            input_size, hidden_size, bidirectional=True, batch_first=True
        )
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input: "torch.Tensor") -> "torch.Tensor":
        self.rnn.flatten_parameters()
        recurrent, _ = self.rnn(input)
        output = self.linear(recurrent)
        return output


class NoneVGGBiLSTMCTC(nn.Module):
    """The ``None-VGG-BiLSTM-CTC`` recognition model (vendored, CTC head only).

    Reproduces the ``Transformation="None"`` / ``FeatureExtraction="VGG"`` /
    ``SequenceModeling="BiLSTM"`` / ``Prediction="CTC"`` configuration of the
    DTRB ``model.Model`` with its exact submodule attribute names
    (``FeatureExtraction``, ``AdaptiveAvgPool``, ``SequenceModeling``,
    ``Prediction``) and forward geometry, so a DTRB checkpoint for this
    configuration loads with strict key matching.

    Forward output shape: ``(N, T, num_class)`` (batch-first) -- the raw CTC
    logits over ``num_class`` classes (index 0 = CTC blank).
    """

    def __init__(
        self,
        input_channel: int = 1,
        output_channel: int = 512,
        hidden_size: int = 256,
        num_class: int = 49,
    ) -> None:
        super().__init__()
        # --- FeatureExtraction (VGG) ---
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        # --- SequenceModeling (2x BiLSTM) ---
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(
                self.FeatureExtraction_output, hidden_size, hidden_size
            ),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size),
        )
        self.SequenceModeling_output = hidden_size

        # --- Prediction (CTC linear head) ---
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)

    def forward(self, input: "torch.Tensor") -> "torch.Tensor":
        """Run the CTC forward pass; returns logits ``(N, T, num_class)``.

        Mirrors DTRB ``model.Model.forward`` for the ``None-VGG-BiLSTM-CTC``
        configuration (no transformation, VGG features, BiLSTM sequence, CTC
        linear head). ``input`` is ``(N, 1, 32, W)``.

        ONNX-export note (numerically identical to DTRB): DTRB does
        ``AdaptiveAvgPool2d((None, 1))`` on the ``(b, w, c, h)``-permuted feature
        then ``squeeze(3)``. For ``imgH=32`` the VGG stack always collapses the
        feature height to exactly 1 (``(b, c, 1, w)``), so that adaptive pool
        over a size-1 axis is an identity. Pooling with a dynamic (``None``)
        output size is not ONNX-exportable via the TorchScript exporter
        ("adaptive pooling, since output_size is not constant"), so this forward
        instead squeezes the constant height axis directly and permutes to
        ``(b, w, c)``. This yields the **same** values as the DTRB path while
        exporting cleanly with dynamic batch/width.
        """
        # Transformation stage is "None" -> input passes through unchanged.
        visual_feature = self.FeatureExtraction(input)  # (b, c, 1, w)
        # Drop the height axis (always 1 for imgH=32) and reorder to (b, w, c).
        visual_feature = visual_feature.squeeze(2)  # (b, c, w)
        visual_feature = visual_feature.permute(0, 2, 1)  # (b, w, c)
        contextual_feature = self.SequenceModeling(visual_feature)
        prediction = self.Prediction(contextual_feature.contiguous())
        return prediction


def build_model(
    *,
    input_channel: int = 1,
    output_channel: int = 512,
    hidden_size: int = 256,
    num_class: int = 49,
) -> NoneVGGBiLSTMCTC:
    """Construct the vendored model with the trained Ol Chiki geometry.

    Defaults match the recorded geometry in ``models/ol_chiki_g2/provenance.json``
    (``input_channel=1``, ``output_channel=512``, ``hidden_size=256``) plus the
    confirmed ``num_class=49`` (48 charset emit classes + 1 CTC blank, D8).
    """
    return NoneVGGBiLSTMCTC(
        input_channel=input_channel,
        output_channel=output_channel,
        hidden_size=hidden_size,
        num_class=num_class,
    )


def _strip_module_prefix(state_dict: "dict") -> "dict":
    """Strip the ``module.`` DataParallel prefix from every state-dict key.

    DTRB saves checkpoints from a ``DataParallel``-wrapped model, so every key
    is prefixed with ``module.`` (documented in the old ``artifact.py``: "DTRB
    saves weights with the ``module.`` prefix already present"). The vendored
    :class:`NoneVGGBiLSTMCTC` is a bare (un-wrapped) module, so the prefix is
    stripped before loading. Keys without the prefix are passed through
    unchanged so an already-stripped checkpoint also loads.
    """
    prefix = "module."
    new_state = {}
    for key, value in state_dict.items():
        if key.startswith(prefix):
            new_state[key[len(prefix):]] = value
        else:
            new_state[key] = value
    return new_state


def load_dtrb_checkpoint(
    model: NoneVGGBiLSTMCTC, pth_path: str, *, map_location: str = "cpu"
) -> NoneVGGBiLSTMCTC:
    """Load a DTRB ``.pth`` checkpoint into ``model`` (strips ``module.`` prefix).

    Handles the DTRB DataParallel ``module.`` key prefix (see
    :func:`_strip_module_prefix`). Some checkpoints save the raw ``state_dict``
    directly; others nest it under a ``"state_dict"`` / ``"model"`` key -- both
    layouts are handled. The load is **strict** (every key must match) so a
    geometry mismatch surfaces immediately rather than silently loading a
    partial model.

    Args:
        model: A constructed :class:`NoneVGGBiLSTMCTC`.
        pth_path: Filesystem path to the trained ``.pth`` checkpoint.
        map_location: Torch ``map_location`` (default ``"cpu"`` so export runs
            without a GPU).

    Returns:
        ``model`` with the checkpoint weights loaded, set to ``eval()`` mode.

    Raises:
        PretrainedModelError: If the checkpoint cannot be read or its keys do
            not match the model geometry. The error names ``pth_path``.
    """
    from ..errors import PretrainedModelError

    try:
        checkpoint = torch.load(pth_path, map_location=map_location)
    except (OSError, RuntimeError, EOFError) as exc:
        raise PretrainedModelError(
            pth_path, f"cannot read checkpoint ({exc})"
        ) from exc

    # Unwrap common nested layouts.
    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(
        checkpoint["model"], dict
    ):
        state_dict = checkpoint["model"]
    else:
        state_dict = checkpoint

    state_dict = _strip_module_prefix(state_dict)

    try:
        model.load_state_dict(state_dict, strict=True)
    except (RuntimeError, KeyError) as exc:
        raise PretrainedModelError(
            pth_path,
            f"checkpoint keys/shapes do not match the "
            f"None-VGG-BiLSTM-CTC geometry ({exc})",
        ) from exc

    model.eval()
    return model
