"""Model artifact resolution and caching for the core.

Resolution: a given ``path=`` loads a local artifact directory (never touches
the network); otherwise resolve the cache dir and reuse a cache hit or download.
Cache-dir precedence: explicit arg -> ``OLCHIKI_OCR_CACHE`` env -> default
``~/.cache/olchiki-ocr/<model_version>/``. Downloads go to a temp dir, are
SHA-256 verified, safely extracted (path-traversal guarded), then atomically
moved into place, so a failed or mismatched download never becomes a valid cache
entry. Pure stdlib.
"""

from __future__ import annotations

import hashlib
import os
import shutil
import tarfile
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

from .errors import ModelArtifactError, ModelDownloadError

__all__ = [
    "DEFAULT_RELEASE_HOST",
    "MODEL_TAG_PREFIX",
    "DOWNLOAD_TIMEOUT_SECONDS",
    "CACHE_ENV_VAR",
    "ONNX_FILENAME",
    "PROVENANCE_FILENAME",
    "CHARSET_FILENAME",
    "CHECKSUM_MARKER_FILENAME",
    "REQUIRED_ARTIFACT_MEMBERS",
    "resolve_cache_dir",
    "resolve",
]

#: Default Release_Host for downloads (GitHub Releases). Override via
#: ``model_source`` / ``InferenceConfig.model_source``.
DEFAULT_RELEASE_HOST = "https://github.com/olchikiai/ahla/releases/download"

#: Release-tag scheme: the model is published under its own tag
#: ``model-v<model_version>`` (decoupled from the package version), so the URL
#: is ``<host>/releases/download/model-v<model_version>/<asset>``.
MODEL_TAG_PREFIX = "model-v"

#: Maximum seconds a download may run before it is treated as a failure.
DOWNLOAD_TIMEOUT_SECONDS = 300

#: Environment variable that overrides the Model_Cache location.
CACHE_ENV_VAR = "OLCHIKI_OCR_CACHE"

# --- Expected artifact member filenames --------------------------------------

ONNX_FILENAME = "model.onnx"

PROVENANCE_FILENAME = "provenance.json"

CHARSET_FILENAME = "charset.txt"

#: Files that must be present for an artifact directory to be valid. The charset
#: file is excluded; its ordering is validated later by the provenance check.
REQUIRED_ARTIFACT_MEMBERS = (ONNX_FILENAME, PROVENANCE_FILENAME)

#: Cache-validity marker written last, after a successful download+verify+move.
#: The published checksum covers the archive, not the extracted dir, so this
#: marker (the recorded verified digest) is what distinguishes a fully verified
#: cache entry from a stray directory that merely has the required files.
CHECKSUM_MARKER_FILENAME = ".sha256"

#: Chunk size (bytes) for streamed SHA-256 hashing of the downloaded archive.
_HASH_CHUNK_BYTES = 1024 * 1024


def _archive_name(model_version: str) -> str:
    """Return the versioned archive filename."""
    return f"olchiki-ocr-model-{model_version}.tar.gz"


def _model_release_tag(model_version: str) -> str:
    """Return the model's own release tag ``model-v<model_version>``."""
    return f"{MODEL_TAG_PREFIX}{model_version}"


def _artifact_url(release_host: str, model_version: str) -> str:
    """Return the archive URL ``<host>/releases/download/<tag>/<asset>``."""
    tag = _model_release_tag(model_version)
    return f"{release_host.rstrip('/')}/{tag}/{_archive_name(model_version)}"


def resolve_cache_dir(model_version: str, cache_dir: str | None) -> str:
    """Resolve the Model_Cache directory for ``model_version``.

    Precedence: explicit ``cache_dir`` -> ``OLCHIKI_OCR_CACHE`` env (non-empty)
    -> default ``~/.cache/olchiki-ocr/<model_version>/``. The directory is not
    created here.
    """
    if cache_dir is not None and cache_dir != "":
        return cache_dir

    env_value = os.environ.get(CACHE_ENV_VAR)
    if env_value:
        return env_value

    default = Path("~/.cache/olchiki-ocr") / model_version
    return str(default.expanduser())


def _artifact_files_present(artifact_dir: str) -> bool:
    """Return ``True`` when every :data:`REQUIRED_ARTIFACT_MEMBERS` file exists."""
    base = Path(artifact_dir)
    return all((base / member).is_file() for member in REQUIRED_ARTIFACT_MEMBERS)


def _cache_entry_valid(artifact_dir: str) -> bool:
    """Return ``True`` for a valid cache entry: required members present AND the
    verified-download checksum marker exists.
    """
    base = Path(artifact_dir)
    if not _artifact_files_present(artifact_dir):
        return False
    return (base / CHECKSUM_MARKER_FILENAME).is_file()


def _fetch_bytes(url: str, timeout: int) -> bytes:
    """Fetch ``url`` over HTTP(S) and return the raw body.

    The single network-touching primitive, so tests monkeypatch just this seam.
    """
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        return response.read()


def _sha256_of_file(file_path: Path) -> str:
    """Return the lowercase hex SHA-256 of ``file_path``, streamed in chunks."""
    digest = hashlib.sha256()
    with open(file_path, "rb") as handle:
        for chunk in iter(lambda: handle.read(_HASH_CHUNK_BYTES), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_published_sha256(raw: bytes) -> str:
    """Extract the lowercased hex digest from a companion ``.sha256`` payload.

    Accepts a bare digest or the ``sha256sum`` ``"<hex>  <name>"`` layout.
    """
    text = raw.decode("utf-8", errors="strict").strip()
    token = text.split()[0] if text else ""
    return token.lower()


def _is_within_directory(directory: Path, target: Path) -> bool:
    """Return ``True`` when ``target`` resolves to a path inside ``directory``."""
    directory = directory.resolve()
    target = target.resolve()
    try:
        target.relative_to(directory)
    except ValueError:
        return False
    return True


def _safe_extract_tar(tar: tarfile.TarFile, dest_dir: Path) -> None:
    """Extract ``tar`` into ``dest_dir``, refusing unsafe (path-traversal) members.

    Guards against tar-slip: members with absolute or ``..`` names, or links
    escaping ``dest_dir``, raise ``ValueError`` (which the caller maps to a
    download failure) before anything is written. Manual check because
    ``requires-python >=3.9`` predates the 3.12 ``filter="data"`` extractor.
    """
    for member in tar.getmembers():
        member_path = dest_dir / member.name
        if os.path.isabs(member.name) or ".." in Path(member.name).parts:
            raise ValueError(
                f"Refusing to extract unsafe archive member: {member.name!r}"
            )
        if not _is_within_directory(dest_dir, member_path):
            raise ValueError(
                f"Refusing to extract member outside staging dir: {member.name!r}"
            )
        if member.issym() or member.islnk():
            link_target = dest_dir / member.name
            resolved_target = (link_target.parent / member.linkname)
            if os.path.isabs(member.linkname) or not _is_within_directory(
                dest_dir, resolved_target
            ):
                raise ValueError(
                    f"Refusing to extract unsafe link member: {member.name!r}"
                )
    tar.extractall(dest_dir)  # noqa: S202  (members validated above)


def _download_and_verify(
    model_source: str | None,
    cache_dir: str,
    model_version: str,
) -> str:
    """Download to temp -> SHA-256 verify -> safe extract -> atomic move into cache.

    Downloads the versioned archive (and its companion ``.sha256``) from the
    tagged release URL, verifies the digest, extracts safely, then atomically
    moves the staged contents into ``cache_dir`` and records the verified digest
    marker last. Any failure raises :class:`ModelDownloadError` naming the URL
    and cache path, leaving no partial or valid-looking cache entry. Returns
    ``cache_dir`` on success.
    """
    release_host = model_source if model_source else DEFAULT_RELEASE_HOST
    url = _artifact_url(release_host, model_version)
    checksum_url = f"{url}.sha256"

    cache_path = Path(cache_dir)
    parent = cache_path.parent

    # Keep temp staging and the rename target on the same filesystem so
    # os.replace is atomic.
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ModelDownloadError(
            url, cache_dir, f"Cannot prepare cache parent directory: {exc}"
        ) from exc

    # Work inside a temp dir next to the cache dir: a failure cleans up with no
    # partial cache, a success renames atomically.
    tmp_root = tempfile.mkdtemp(prefix=".olchiki-dl-", dir=str(parent))
    tmp_root_path = Path(tmp_root)
    staging_final = tmp_root_path / "artifact"
    try:
        # Download archive + published checksum.
        try:
            archive_bytes = _fetch_bytes(url, DOWNLOAD_TIMEOUT_SECONDS)
            checksum_bytes = _fetch_bytes(checksum_url, DOWNLOAD_TIMEOUT_SECONDS)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            # URLError covers HTTP/connection errors, OSError timeouts/IO,
            # ValueError bad URLs.
            raise ModelDownloadError(
                url, cache_dir, f"Download failed: {exc}"
            ) from exc

        archive_path = tmp_root_path / _archive_name(model_version)
        archive_path.write_bytes(archive_bytes)

        # Verify SHA-256 against the published checksum.
        published = _parse_published_sha256(checksum_bytes)
        actual = _sha256_of_file(archive_path)
        if not published or actual != published:
            raise ModelDownloadError(
                url,
                cache_dir,
                "Checksum mismatch: "
                f"published={published!r} actual={actual!r}",
            )

        # Safely extract into the temp staging dir.
        staging_final.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(archive_path, mode="r:gz") as tar:
                _safe_extract_tar(tar, staging_final)
        except (tarfile.TarError, ValueError, OSError) as exc:
            raise ModelDownloadError(
                url, cache_dir, f"Archive extraction failed: {exc}"
            ) from exc

        # Descend into a single nested top-level directory if needed.
        artifact_root = _locate_artifact_root(staging_final)
        if artifact_root is None:
            raise ModelDownloadError(
                url,
                cache_dir,
                "Extracted archive is missing required member(s): "
                f"{', '.join(REQUIRED_ARTIFACT_MEMBERS)}",
            )

        # Atomic move into the cache dir: drop any stale dir, then rename.
        try:
            if cache_path.exists():
                shutil.rmtree(cache_path)
            os.replace(str(artifact_root), str(cache_path))
        except OSError as exc:
            raise ModelDownloadError(
                url, cache_dir, f"Atomic cache move failed: {exc}"
            ) from exc

        # Record the verified digest marker last, so its presence marks the
        # entry valid; clean up if this final write fails.
        try:
            (cache_path / CHECKSUM_MARKER_FILENAME).write_text(
                actual, encoding="utf-8"
            )
        except OSError as exc:
            shutil.rmtree(cache_path, ignore_errors=True)
            raise ModelDownloadError(
                url, cache_dir, f"Failed to record cache marker: {exc}"
            ) from exc

        return cache_dir
    finally:
        # Always clean up temp scaffolding.
        shutil.rmtree(tmp_root_path, ignore_errors=True)


def _locate_artifact_root(staging_dir: Path) -> Path | None:
    """Return the directory holding the required members, or ``None``.

    Handles a flat archive or one nested under a single top-level directory.
    """
    if all(
        (staging_dir / member).is_file() for member in REQUIRED_ARTIFACT_MEMBERS
    ):
        return staging_dir
    entries = [p for p in staging_dir.iterdir()]
    subdirs = [p for p in entries if p.is_dir()]
    if len(subdirs) == 1 and all(
        (subdirs[0] / member).is_file() for member in REQUIRED_ARTIFACT_MEMBERS
    ):
        return subdirs[0]
    return None


def resolve(
    path: str | None,
    model_source: str | None,
    cache_dir: str | None,
    model_version: str,
) -> str:
    """Resolve and return the Model_Artifact directory.

    A given ``path`` is loaded as a local artifact directory (never touches the
    network), raising :class:`ModelArtifactError` if it is missing or
    incomplete. Otherwise the cache dir is resolved and a valid cache entry
    reused, or :func:`_download_and_verify` is invoked. No ONNX/provenance
    integrity checking happens here (that is ``provenance.load_and_verify``).
    """
    # Local load: path= given -> never touch the network.
    if path is not None:
        base = Path(path)
        if not base.is_dir():
            raise ModelArtifactError(
                path, "Local artifact directory does not exist"
            )
        if not _artifact_files_present(path):
            missing = [
                member
                for member in REQUIRED_ARTIFACT_MEMBERS
                if not (base / member).is_file()
            ]
            raise ModelArtifactError(
                path,
                "Local artifact directory is missing required file(s): "
                f"{', '.join(missing)}",
            )
        return path

    # No path= given: resolve cache dir, reuse a valid entry, else download.
    resolved_cache_dir = resolve_cache_dir(model_version, cache_dir)

    if _cache_entry_valid(resolved_cache_dir):
        return resolved_cache_dir

    return _download_and_verify(model_source, resolved_cache_dir, model_version)
