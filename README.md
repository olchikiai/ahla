# olchiki-ocr

Ol Chiki (Santali) text recognition. Give it a cropped word or line image and it
returns the text, using a DTRB `None-VGG-BiLSTM-CTC` model run through ONNX
Runtime. The core install is torch-free — just ONNX Runtime, numpy, and Pillow.

It recognizes cropped word/line images, not whole pages. If you're starting from
a full scan, segment it into lines first, then pass each crop to the recognizer.

## Install

Not on PyPI yet — install the wheel from GitHub Releases:

```bash
pip install https://github.com/olchikiai/ahla/releases/download/v0.1.0/olchiki_ocr-0.1.0-py3-none-any.whl
```

Or from source:

```bash
pip install git+https://github.com/olchikiai/ahla.git@v0.1.0
```

### Extras

| Extra      | For                                              |
|------------|--------------------------------------------------|
| `[gpu]`    | GPU inference (`onnxruntime-gpu`)                |
| `[cv]`     | Deskew / threshold / denoise (`opencv-python`)   |
| `[train]`  | Fine-tuning (`torch`, `lmdb`, `fonttools`)       |
| `[export]` | `.pth` → ONNX → INT8 export (`torch`, `onnx`)    |

```bash
pip install "olchiki-ocr[cv]"
```

## Usage

```python
from olchiki_ocr import ModelRecognizer

recognizer = ModelRecognizer.from_pretrained()
print(recognizer.predict("word.png"))
```

Or from the command line:

```bash
olchiki-ocr word.png
```

`from_pretrained()` downloads the model on first use, verifies its checksum, and
caches it under `~/.cache/olchiki-ocr/` (set `OLCHIKI_OCR_CACHE` to change the
location, or pass `path=` to load a local copy offline).

## Releases

The package and the model ship under separate tags on `olchikiai/ahla`:

- `v<version>` — the wheel + sdist (what `pip install` fetches)
- `model-v<version>` — the model artifact, downloaded by `from_pretrained()`

Cut a package release with `./scripts/release.sh <version>` (or push a `v*` tag
to let CI build it). Upload the model separately under its `model-v<version>`
tag. The package and model are versioned independently.

## License

Code is MIT ([LICENSE](LICENSE)); the model weights and ONNX artifact are
Apache-2.0 ([LICENSE-MODEL](LICENSE-MODEL)). The training corpus (AI4Bharat
IndicCorp) is CC0, and the training fonts (Noto Sans Ol Chiki) are under the SIL
Open Font License. See [NOTICE](NOTICE) for attribution.
