"""Vendored ``None-VGG-BiLSTM-CTC`` architecture for the ``[export]`` tier.

Vendors a minimal PyTorch definition of the DTRB ``None-VGG-BiLSTM-CTC`` stack
that the trained weights (``models/ol_chiki_g2/model/ol_chiki_g2.pth``) came
from, so ``[export]`` can load the ``.pth`` and export to ONNX using only
``torch`` + ``onnx`` -- without ``easyocr`` or the DTRB clone.

Submodule attribute names and structure mirror DTRB ``model.Model`` exactly so
the checkpoint's ``state_dict`` keys line up 1:1 (``FeatureExtraction.ConvNet.*``,
two ``SequenceModeling`` BiLSTM layers, the ``Prediction`` CTC linear head).
Checkpoints are saved from a ``DataParallel`` model, so keys carry a ``module.``
prefix that :func:`load_dtrb_checkpoint` strips. Forward output is
``(N, T, num_class)`` (batch-first). ``torch`` is imported at module top level,
hence this module is imported lazily only after ``[export]`` is confirmed.
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
    """VGG feature extractor, copied verbatim from DTRB so ConvNet.* keys match."""

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
    """BiLSTM + linear projection, copied from DTRB so rnn.*/linear.* keys match."""

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

    Mirrors the DTRB ``model.Model`` submodule names and forward geometry so a
    DTRB checkpoint loads with strict key matching. Forward output is
    ``(N, T, num_class)`` (batch-first); class index 0 is the CTC blank.
    """

    def __init__(
        self,
        input_channel: int = 1,
        output_channel: int = 512,
        hidden_size: int = 256,
        num_class: int = 49,
    ) -> None:
        super().__init__()
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(
                self.FeatureExtraction_output, hidden_size, hidden_size
            ),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size),
        )
        self.SequenceModeling_output = hidden_size

        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)

    def forward(self, input: "torch.Tensor") -> "torch.Tensor":
        """Run the CTC forward pass; returns logits ``(N, T, num_class)``.

        ``input`` is ``(N, 1, 32, W)``. DTRB does
        ``AdaptiveAvgPool2d((None, 1))`` then ``squeeze``, but for ``imgH=32``
        the VGG stack always collapses the feature height to 1, so that pool is
        an identity. Since a dynamic (``None``) pool size is not exportable via
        the TorchScript exporter, we squeeze the constant height axis directly --
        numerically identical to DTRB but ONNX-exportable with dynamic axes.
        """
        visual_feature = self.FeatureExtraction(input)  # (b, c, 1, w)
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

    Defaults match ``models/ol_chiki_g2/provenance.json`` plus ``num_class=49``
    (48 charset emit classes + 1 CTC blank).
    """
    return NoneVGGBiLSTMCTC(
        input_channel=input_channel,
        output_channel=output_channel,
        hidden_size=hidden_size,
        num_class=num_class,
    )


def _strip_module_prefix(state_dict: "dict") -> "dict":
    """Strip the ``module.`` DataParallel prefix from every state-dict key.

    The vendored model is un-wrapped; keys without the prefix pass through so an
    already-stripped checkpoint also loads.
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
    """Load a DTRB ``.pth`` into ``model`` (strips ``module.``, strict, eval mode).

    Handles both raw and ``"state_dict"``/``"model"``-nested layouts. The load
    is strict so a geometry mismatch surfaces immediately. Raises
    ``PretrainedModelError`` (naming ``pth_path``) on a bad or mismatched
    checkpoint.
    """
    from ..errors import PretrainedModelError

    try:
        checkpoint = torch.load(pth_path, map_location=map_location)
    except (OSError, RuntimeError, EOFError) as exc:
        raise PretrainedModelError(
            pth_path, f"cannot read checkpoint ({exc})"
        ) from exc

    # Unwrap nested layouts.
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
