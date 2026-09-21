# olchiki-ocr

Ol Chiki (Santali) text recognition. It reads a whole scanned document — segment
a page into its words/lines and recognize them in reading order — or a single
pre-cropped word/line image, using a DTRB `None-VGG-BiLSTM-CTC` model run through
ONNX Runtime. The core install is torch-free — just ONNX Runtime, numpy, and
Pillow.

Two entry points: `ModelRecognizer` recognizes a pre-cropped word/line image,
and `Page_Recognizer` reads a full document — it segments the page into ordered
regions and recognizes each one. Document segmentation is a classical-CV step
that needs OpenCV, so it lives behind the optional `[segmentation]` extra; the
bare install stays torch-free and OpenCV-free.

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

| Extra            | For                                                        |
|------------------|------------------------------------------------------------|
| `[segmentation]` | Read whole documents — page segmentation (`opencv-python`) |
| `[gpu]`          | GPU inference (`onnxruntime-gpu`)                          |
| `[cv]`           | Deskew / threshold / denoise (`opencv-python`)             |
| `[train]`        | Fine-tuning (`torch`, `lmdb`, `fonttools`)                 |
| `[export]`       | `.pth` → ONNX → INT8 export (`torch`, `onnx`)              |

```bash
pip install "olchiki-ocr[segmentation]"
```

## Usage

### Read a whole document

Give it a full scanned page and it segments the page into regions and recognizes
each one, in reading order (needs the `[segmentation]` extra):

```python
from olchiki_ocr import Page_Recognizer

recognizer = Page_Recognizer.from_pretrained()
result = recognizer.recognize_page("document.png")

for region in result.regions:
    print(region.text)          # region.index, region.x/y/width/height also available
```

Or from the command line:

```bash
olchiki-ocr recognize-page document.png              # one region's text per line
olchiki-ocr recognize-page document.png --confidence # text<TAB>confidence per line
```

`recognize_page` accepts `granularity="word"` (default) or `"line"`, and
`deskew` / `denoise` (both on by default). Pass `confidence=True` to get a
per-region confidence in `[0, 1]`.

To just split a document into crop images without recognizing them:

```bash
olchiki-ocr segment document.png --out-dir crops
```

### Recognize a single pre-cropped image

```python
from olchiki_ocr import ModelRecognizer

recognizer = ModelRecognizer.from_pretrained()
print(recognizer.predict("word.png"))
```

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
