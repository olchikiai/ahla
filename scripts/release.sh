#!/usr/bin/env bash
# release.sh — build olchiki-ocr and publish a GitHub Release with the wheel+sdist.
#
# Bash equivalent of release.ps1 (for Git Bash / Linux / macOS).
#
# Prerequisites:
#   - GitHub CLI installed and authenticated over HTTPS:  gh auth login
#   - build + twine in the active environment:            python -m pip install --upgrade build twine
#
# Usage (from anywhere; the script cd's to the olchiki-ocr project root):
#   ./scripts/release.sh 0.1.0
#   ./scripts/release.sh --version 0.1.0
#
# This does NOT publish to PyPI. It builds the wheel+sdist, validates them with
# `twine check`, and creates a GitHub Release (tag v<version>) on olchikiai/ahla
# with the wheel+sdist attached.
#
# NOTE: the MODEL artifact is published under a SEPARATE tag `model-v<version>`
# (from_pretrained() downloads the model from that tag, decoupled from the
# package version). See the note printed at the end.

set -euo pipefail

REPO="olchikiai/ahla"

usage() {
  echo "Usage: $0 [--version] <VERSION>   (e.g. $0 0.1.0)" >&2
  exit 2
}

# --- Parse the version argument (positional or --version/-v). ---------------
VERSION=""
case "${1:-}" in
  -h|--help) usage ;;
  -v|--version) VERSION="${2:-}" ;;
  "") usage ;;
  *) VERSION="$1" ;;
esac
if [ -z "$VERSION" ]; then
  usage
fi

TAG="v$VERSION"
MODEL_TAG="model-v$VERSION"

# --- cd to the project root (the olchiki-ocr/ dir = parent of scripts/). -----
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

echo "==> Cleaning old build artifacts"
rm -rf dist build

echo "==> Building wheel + sdist"
python -m build .

echo "==> Validating artifacts with twine check"
python -m twine check dist/*

echo "==> Creating GitHub release $TAG and uploading wheel + sdist"
# Requires an existing git tag on the remote OR gh creates the tag from the current commit.
gh release create "$TAG" dist/* \
  --repo "$REPO" \
  --title "olchiki-ocr $VERSION" \
  --notes "olchiki-ocr $VERSION. Install: pip install https://github.com/$REPO/releases/download/$TAG/olchiki_ocr-$VERSION-py3-none-any.whl"

echo "==> Done. Package release $TAG published with wheel + sdist."
echo
echo "    NEXT: publish the MODEL artifact under its OWN tag '$MODEL_TAG' so"
echo "          ModelRecognizer.from_pretrained() can download it, e.g.:"
echo
echo "      gh release create $MODEL_TAG \\"
echo "        <path>/olchiki-ocr-model-$VERSION.tar.gz \\"
echo "        <path>/olchiki-ocr-model-$VERSION.tar.gz.sha256 \\"
echo "        --repo $REPO \\"
echo "        --title \"olchiki-ocr model $VERSION\" \\"
echo "        --notes \"Model artifact for olchiki-ocr $VERSION\""
echo
echo "    (The model .tar.gz + .sha256 are produced under output/release/ by the"
echo "     export/build tooling; they are not committed to this repo.)"
