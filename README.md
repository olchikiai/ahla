# olchiki-ocr

Ol Chiki (Santali) OCR: recognize pre-cropped word/line images with an ONNX
Runtime core (torch-free), plus optional GPU, OpenCV, training, and export
tiers.

The core recognizes pre-cropped word/line images using a validated DTRB
`None-VGG-BiLSTM-CTC` model executed with ONNX Runtime. A bare install is
torch-free and EasyOCR-free: only ONNX Runtime (CPU), numpy, and Pillow.

## Install

> Published via **GitHub Releases** (PyPI publication planned for a future
> release). Dependencies (`onnxruntime`, `numpy`, `Pillow`) are resolved from
> PyPI as usual — only the `olchiki-ocr` wheel itself is hosted on GitHub.

Install the prebuilt wheel from the latest release:

```bash
pip install https://github.com/olchikiai/ahla/releases/download/v0.1.0/olchiki_ocr-0.1.0-py3-none-any.whl
```

Or let pip discover the right wheel from the Releases page:

```bash
pip install --find-links https://github.com/olchikiai/ahla/releases olchiki-ocr
```

Install from source (builds locally; requires git):

```bash
pip install git+https://github.com/olchikiai/ahla.git@v0.1.0
```

Once published to PyPI, the standard form will also work:

```bash
pip install olchiki-ocr   # planned
```

### Optional extras

| Extra       | Adds                                   | For                                  |
|-------------|----------------------------------------|--------------------------------------|
| `[gpu]`     | `onnxruntime-gpu`                      | GPU inference                        |
| `[cv]`      | `opencv-python`                        | Advanced preprocessing (deskew, threshold, denoise) |
| `[train]`   | `torch`, `lmdb`, `fonttools`, `numpy`  | Fine-tuning (DTRB via pinned git clone) |
| `[export]`  | `torch`, `onnx`                        | `.pth` -> ONNX -> INT8 export        |

Append the extra to the install target, e.g. from a release wheel:

```bash
pip install "olchiki-ocr[gpu] @ https://github.com/olchikiai/ahla/releases/download/v0.1.0/olchiki_ocr-0.1.0-py3-none-any.whl"
```

Notes:

- `[export]` is intentionally independent of `[train]` (no lmdb, no DTRB). The
  INT8 quantization tooling is provided by the core `onnxruntime`
  (`onnxruntime.quantization`).
- `[train]` obtains DTRB (`clovaai/deep-text-recognition-benchmark`) via a
  pinned external git clone at commit
  `e2117f2fb882b3c6085030500a260c113be27a63` that the trainer shells out to.
  `fonttools` is required by the synthetic-data generator.

## Usage

```python
from olchiki_ocr import ModelRecognizer

recognizer = ModelRecognizer.from_pretrained()
text = recognizer.predict("word.png")
```

CLI:

```bash
olchiki-ocr word.png
```

## Versions

The **Package_Version** (this distribution's `pyproject.toml` `version`) is
distinct from the **Model_Version**, which is recorded in the downloaded model
artifact's `provenance.json` -- not in `pyproject.toml`.

## Licensing

This project uses a deliberate license split:

- **Packaging code** (the `olchiki_ocr` import package and build files):
  **MIT** -- see [LICENSE](LICENSE).
- **Published model artifacts** (the trainable Base_Weights `.pth` and the
  servable ONNX_Model, including its INT8 form): **Apache-2.0** -- see
  [LICENSE-MODEL](LICENSE-MODEL). Apache-2.0 is chosen to match the DTRB /
  EasyOCR upstream lineage and to carry Apache-2.0's patent grant.

See [NOTICE](NOTICE) for full attribution (EasyOCR / JaidedAI, DTRB / clovaai,
Noto Sans Ol Chiki fonts, and the AI4Bharat IndicCorp corpus).
