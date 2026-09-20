# olchiki-ocr

Ol Chiki (Santali) OCR: recognize Ol Chiki text from word or line images —
for example, crops from a scanned page or photo — with an ONNX Runtime core
(torch-free), plus optional GPU, OpenCV, training, and export tiers.

The recognizer reads a single word or line of Ol Chiki text from an image
using a validated DTRB `None-VGG-BiLSTM-CTC` model executed with ONNX
Runtime. It is a text recognizer, not a page detector: feed it a cropped
word/line region (segment the page first if you are starting from a full
scan).

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

## Versions and release tags

The **Package_Version** (this distribution's `pyproject.toml` `version`) is
distinct from the **Model_Version**, which is recorded in the downloaded
model artifact's `provenance.json` — not in `pyproject.toml`. They are
versioned and released independently.

The package and the model are published under **separate GitHub release
tags** on `olchikiai/ahla`:

| Tag              | Contains                                              | Consumed by                     |
|------------------|-------------------------------------------------------|---------------------------------|
| `v<version>`     | the `olchiki-ocr` wheel + sdist                       | `pip install`                   |
| `model-v<version>` | the model artifact (`olchiki-ocr-model-<version>.tar.gz` + `.sha256`) | `ModelRecognizer.from_pretrained()` at first use |

So `v0.1.0` holds the installable package, and `model-v0.1.0` holds the
model weights. On first use, `from_pretrained()` downloads the model archive
from the `model-v<version>` release, verifies its SHA-256, and caches it
under `~/.cache/olchiki-ocr/<model_version>/` (override with the
`OLCHIKI_OCR_CACHE` environment variable, or pass `model_source=` to load
from a mirror, or `path=` to load a local artifact directory offline).

## Releasing (maintainers)

Publishing a release is two independent steps — the package and the model
go to different tags (see above).

**1. Package release** (wheel + sdist under `v<version>`):

```bash
# from the olchiki-ocr project root (bash: Git Bash / Linux / macOS)
./scripts/release.sh 0.1.0
```

This cleans `dist/`, builds the wheel + sdist, runs `twine check`, and
creates the `v<version>` GitHub release with both artifacts attached. It can
also run automatically in CI: pushing a `v*` tag triggers
`.github/workflows/release.yml`, which builds and attaches the artifacts.

On Windows, an equivalent PowerShell script (`scripts/release.ps1 -Version 0.1.0`)
is kept as a local convenience but is not tracked in the repository (it is
git-ignored); `release.sh` is the canonical release script.

**2. Model release** (model artifact under `model-v<version>`):

```bash
gh release create model-v0.1.0 \
  output/release/olchiki-ocr-model-0.1.0.tar.gz \
  output/release/olchiki-ocr-model-0.1.0.tar.gz.sha256 \
  --repo olchikiai/ahla \
  --title "olchiki-ocr model 0.1.0" \
  --notes "Model artifact for olchiki-ocr 0.1.0"
```

The model archive + its `.sha256` are produced by the export tooling and are
**not** committed to the repository (they are large binaries distributed via
the release). `from_pretrained()` will not work until the `model-v<version>`
release exists with these two assets.

**Verify the full round-trip** in a clean environment (dependencies resolved
from public PyPI):

```bash
pip install --index-url https://pypi.org/simple/ \
  https://github.com/olchikiai/ahla/releases/download/v0.1.0/olchiki_ocr-0.1.0-py3-none-any.whl
python -c "from olchiki_ocr import ModelRecognizer; ModelRecognizer.from_pretrained(); print('ok')"
```

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

### Data provenance

The model was trained on synthetic images rendered from the **AI4Bharat
IndicCorp** Santali corpus, and the bundled default lexicon
(`olchiki_ocr/data/lexicon.txt`) is a filtered word list derived from that
corpus. IndicCorpV2 is released under **CC0** (public-domain dedication), so
the corpus and the derived word list may be used and redistributed without
restriction, including commercially — no attribution is required (AI4Bharat
is credited as a courtesy). The training fonts (Noto Sans Ol Chiki) are under
the SIL Open Font License. See [NOTICE](NOTICE) for attribution details.
