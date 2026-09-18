# release.ps1 — build olchiki-ocr and publish a GitHub Release with the wheel+sdist.
#
# Prerequisites:
#   - GitHub CLI installed and authenticated over HTTPS:  gh auth login
#   - build + twine in the active environment:            python -m pip install --upgrade build twine
#
# Usage (from the olchiki-ocr project root):
#   .\scripts\release.ps1 -Version 0.1.0
#
# This does NOT publish to PyPI. It builds the wheel+sdist, validates them
# with `twine check`, tags the release, and uploads the artifacts (plus the
# model artifact if present) to a GitHub Release on olchikiai/ahla.

param(
    [Parameter(Mandatory = $true)][string]$Version
)
$ErrorActionPreference = "Stop"

$tag = "v$Version"
$projectRoot = Split-Path -Parent $PSScriptRoot   # the olchiki-ocr/ dir
Push-Location $projectRoot
try {
    Write-Host "==> Cleaning old build artifacts"
    if (Test-Path dist) { Remove-Item -Recurse -Force dist }
    if (Test-Path build) { Remove-Item -Recurse -Force build }

    Write-Host "==> Building wheel + sdist"
    python -m build .

    Write-Host "==> Validating artifacts with twine check"
    python -m twine check dist/*

    Write-Host "==> Creating GitHub release $tag and uploading artifacts"
    # Requires an existing git tag on the remote OR gh will create the tag from the current commit.
    gh release create $tag dist/* `
        --repo olchikiai/ahla `
        --title "olchiki-ocr $Version" `
        --notes "olchiki-ocr $Version. Install: pip install https://github.com/olchikiai/ahla/releases/download/$tag/olchiki_ocr-$Version-py3-none-any.whl"

    Write-Host "==> Done. Release $tag published with dist artifacts."
    Write-Host "    NOTE: upload the model artifact (.tar.gz + .sha256) to this release separately if not already present,"
    Write-Host "          e.g.: gh release upload $tag <path-to-model>.tar.gz <path-to-model>.tar.gz.sha256 --repo olchikiai/ahla"
}
finally {
    Pop-Location
}
