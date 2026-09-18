"""Model_Artifact resolution and caching for the ``olchiki_ocr`` core.

This module resolves a Model_Artifact *directory* (containing ``model.onnx``,
the charset resource, and ``provenance.json``) for :class:`ModelRecognizer`,
following the ``from_pretrained`` resolution flow in the design:

    path= given?  -- yes --> load the local artifact directory (NEVER network)
                  -- no  --> resolve cache dir -> cache hit? reuse : download

Cache-directory precedence (Req 4.6):

    explicit ``cache_dir`` arg  ->  ``OLCHIKI_OCR_CACHE`` env var (non-empty)
                                ->  ``~/.cache/olchiki-ocr/<model_version>/``

This module implements the resolution and cache logic -- ``path=`` local load
(Req 4.7), cache-dir precedence (Req 4.6), the tightened cache-hit reuse
(Req 4.5), and the network download + SHA-256 verification + safe extraction +
atomic move into the cache (:func:`_download_and_verify`, Req 4.1, 4.3, 4.4,
4.8). A failed, partial, or checksum-mismatched download never becomes a valid
cache entry.

Pure stdlib (``os``, ``pathlib``, ``urllib``, ``hashlib``, ``tarfile``,
``tempfile``, ``shutil``): importing this module never pulls in torch,
onnxruntime, numpy, Pillow, or easyocr.
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

# --- Shared configuration constants (also consumed by Task 10.2) -------------

#: Default ``Release_Host`` serving Model_Artifact downloads (GitHub Releases,
#: Design Decision D3). This is the GitHub Releases host for the
#: ``olchikiai/ahla`` repository. It can be overridden via ``model_source`` /
#: ``InferenceConfig.model_source`` when serving from a mirror.
DEFAULT_RELEASE_HOST = "https://github.com/olchikiai/ahla/releases/download"

#: Maximum seconds a download may run before it is treated as a failure
#: (Req 4.8; proposed default, confirmable).
DOWNLOAD_TIMEOUT_SECONDS = 300

#: Environment variable that overrides the Model_Cache location
#: (Req 4.6; proposed default, confirmable).
CACHE_ENV_VAR = "OLCHIKI_OCR_CACHE"

# --- Expected artifact member filenames --------------------------------------

#: The servable ONNX model file inside a resolved artifact directory.
ONNX_FILENAME = "model.onnx"

#: The provenance metadata file inside a resolved artifact directory.
PROVENANCE_FILENAME = "provenance.json"

#: The charset resource file inside a resolved artifact directory.
CHARSET_FILENAME = "charset.txt"

#: Files that MUST be present for an artifact directory to be considered valid.
#: The charset file is intentionally excluded from the *minimum* set: presence
#: of ``model.onnx`` + ``provenance.json`` is what the resolution flow checks
#: (charset ordering is validated later by the provenance integrity check).
REQUIRED_ARTIFACT_MEMBERS = (ONNX_FILENAME, PROVENANCE_FILENAME)

#: Cache-validity marker written after a successful download+verify+move.
#:
#: The published checksum is for the *archive*, not the extracted directory, so
#: a bare "files present" test cannot re-verify a cache entry against the
#: published digest. Instead, :func:`_download_and_verify` records the verified
#: archive SHA-256 into ``<cache>/.sha256`` after the atomic move succeeds. A
#: cache entry is then treated as valid only when the required members are
#: present AND this marker exists (see :func:`_cache_entry_valid`). Because the
#: marker is the *last* thing written on a successful download, its presence is
#: proof that a verified archive was fully extracted and moved into place -- a
#: partial or mismatched download never gets this far, so it can never look like
#: a valid cache entry.
CHECKSUM_MARKER_FILENAME = ".sha256"

#: Chunk size (bytes) for streamed SHA-256 hashing of the downloaded archive.
_HASH_CHUNK_BYTES = 1024 * 1024


def _archive_name(model_version: str) -> str:
    """Return the versioned archive filename (Design Decision D3)."""
    return f"olchiki-ocr-model-{model_version}.tar.gz"


def _artifact_url(release_host: str, model_version: str) -> str:
    """Return the archive download URL under ``release_host`` (Design D3)."""
    return f"{release_host.rstrip('/')}/{_archive_name(model_version)}"


def resolve_cache_dir(model_version: str, cache_dir: str | None) -> str:
    """Resolve the Model_Cache directory for ``model_version``.

    Precedence (highest first, Req 4.6):

    1. an explicit, non-empty ``cache_dir`` argument;
    2. the ``OLCHIKI_OCR_CACHE`` environment variable, when set and non-empty;
    3. the default ``~/.cache/olchiki-ocr/<model_version>/`` (``~`` expanded).

    The directory is *not* created here -- callers create it lazily only when
    they actually need to write a downloaded artifact into it (the ``path=`` and
    cache-hit paths never need to create it).

    Args:
        model_version: The Model_Version whose cache subdirectory is resolved;
            only used for the default (case 3) path.
        cache_dir: An optional explicit cache directory override.

    Returns:
        The resolved cache directory as a string path.
    """
    # 1. Explicit argument wins.
    if cache_dir is not None and cache_dir != "":
        return cache_dir

    # 2. Environment variable, only when set to a non-empty value.
    env_value = os.environ.get(CACHE_ENV_VAR)
    if env_value:
        return env_value

    # 3. Default per-version cache under the user home cache dir.
    default = Path("~/.cache/olchiki-ocr") / model_version
    return str(default.expanduser())


def _artifact_files_present(artifact_dir: str) -> bool:
    """Return ``True`` when every :data:`REQUIRED_ARTIFACT_MEMBERS` file exists.

    This is the "files present" half of the cache-hit test. The checksum-match
    half is added by :func:`_cache_entry_valid`.
    """
    base = Path(artifact_dir)
    return all((base / member).is_file() for member in REQUIRED_ARTIFACT_MEMBERS)


def _cache_entry_valid(artifact_dir: str) -> bool:
    """Return ``True`` for a *valid* cache entry (tightened cache-hit test).

    A directory is a valid cache entry only when it contains all
    :data:`REQUIRED_ARTIFACT_MEMBERS` **and** the checksum marker
    (:data:`CHECKSUM_MARKER_FILENAME`) recorded by a successful
    download+verify+move. The published checksum covers the *archive*, so this
    marker (written last, only after a verified extract has been atomically
    moved into place) is what distinguishes a fully verified cache entry from a
    stray directory that merely happens to contain the required files
    (Req 4.5).
    """
    base = Path(artifact_dir)
    if not _artifact_files_present(artifact_dir):
        return False
    return (base / CHECKSUM_MARKER_FILENAME).is_file()


def _fetch_bytes(url: str, timeout: int) -> bytes:
    """Fetch ``url`` over HTTP(S) and return the raw response body.

    Factored out as the single network-touching primitive so it is the one
    seam tests monkeypatch (return local bytes / raise ``URLError`` etc.)
    without hitting a real host. Uses ``urllib.request`` (stdlib) with the
    supplied ``timeout``.
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
    """Extract the hex digest from a companion ``.sha256`` payload.

    Accepts both a bare hex digest and the common ``"<hex>  <filename>"``
    ``sha256sum`` layout; returns the lowercased hex digest.
    """
    text = raw.decode("utf-8", errors="strict").strip()
    # ``sha256sum`` format is "<hex><space><space><name>"; take the first token.
    token = text.split()[0] if text else ""
    return token.lower()


def _is_within_directory(directory: Path, target: Path) -> bool:
    """Return ``True`` when ``target`` resolves to a path inside ``directory``.

    Defense-in-depth companion to :func:`_safe_extract_tar` member validation.
    """
    directory = directory.resolve()
    target = target.resolve()
    try:
        target.relative_to(directory)
    except ValueError:
        return False
    return True


def _safe_extract_tar(tar: tarfile.TarFile, dest_dir: Path) -> None:
    """Extract ``tar`` into ``dest_dir``, refusing unsafe members.

    The archive is downloaded from a remote host, so extraction must guard
    against path-traversal ("tar-slip"): a member whose name is absolute or
    contains a ``..`` component, or a link whose target escapes ``dest_dir``,
    could otherwise write outside the staging directory. Because
    ``requires-python`` is ``>=3.9`` (predating the 3.12 ``filter="data"``
    extraction filter), this implements a manual safe-extraction check that
    rejects unsafe members *before* anything is written.

    Raises:
        ModelDownloadError-worthy failure is surfaced by the caller; here a
        rejected member raises ``ValueError`` which the caller maps to a
        download failure.
    """
    for member in tar.getmembers():
        member_path = dest_dir / member.name
        # Reject absolute paths and ``..`` traversal in the member name.
        if os.path.isabs(member.name) or ".." in Path(member.name).parts:
            raise ValueError(
                f"Refusing to extract unsafe archive member: {member.name!r}"
            )
        if not _is_within_directory(dest_dir, member_path):
            raise ValueError(
                f"Refusing to extract member outside staging dir: {member.name!r}"
            )
        # Also validate link targets for link members.
        if member.issym() or member.islnk():
            link_target = dest_dir / member.name
            resolved_target = (link_target.parent / member.linkname)
            if os.path.isabs(member.linkname) or not _is_within_directory(
                dest_dir, resolved_target
            ):
                raise ValueError(
                    f"Refusing to extract unsafe link member: {member.name!r}"
                )
    # All members validated -> safe to extract.
    tar.extractall(dest_dir)  # noqa: S202  (members validated above)


def _download_and_verify(
    model_source: str | None,
    cache_dir: str,
    model_version: str,
) -> str:
    """Download + SHA-256 verify + atomic move of the Model_Artifact archive.

    Downloads the versioned archive
    ``olchiki-ocr-model-<model_version>.tar.gz`` from the ``Release_Host``
    (``model_source`` or :data:`DEFAULT_RELEASE_HOST`) into a temp location with
    a :data:`DOWNLOAD_TIMEOUT_SECONDS` timeout, fetches the companion
    ``.sha256``, verifies the archive's streamed SHA-256 against the published
    digest, safely extracts it into a temp staging dir (rejecting path-traversal
    members), verifies the expected :data:`REQUIRED_ARTIFACT_MEMBERS` are
    present, then **atomically** moves the staged contents into ``cache_dir``
    (temp-then-rename via :func:`os.replace`). Finally it records the verified
    archive digest into ``<cache_dir>/.sha256`` (the cache-validity marker).

    On any failure -- network error, HTTP error, timeout, missing artifact
    (404), checksum mismatch, or an unsafe/incomplete archive -- it raises
    :class:`ModelDownloadError` naming the URL and the cache path, and ensures
    no partial/temp/mismatched files remain as a valid cache entry (all temp
    state is cleaned up, and the marker that makes a cache entry "valid" is
    never written) (Req 4.1, 4.3, 4.4, 4.8).

    Returns the resolved artifact directory (``cache_dir``) on success.
    """
    release_host = model_source if model_source else DEFAULT_RELEASE_HOST
    url = _artifact_url(release_host, model_version)
    checksum_url = f"{url}.sha256"

    cache_path = Path(cache_dir)
    parent = cache_path.parent

    # Ensure the parent exists so both the temp staging dir and the atomic
    # rename target live on the same filesystem (so os.replace is atomic).
    try:
        parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        raise ModelDownloadError(
            url, cache_dir, f"Cannot prepare cache parent directory: {exc}"
        ) from exc

    # Work entirely inside a temp dir next to the final cache dir, so a failure
    # cleans up with no partial cache and a success can atomically rename.
    tmp_root = tempfile.mkdtemp(prefix=".olchiki-dl-", dir=str(parent))
    tmp_root_path = Path(tmp_root)
    staging_final = tmp_root_path / "artifact"
    try:
        # --- 1. Download archive + published checksum. --------------------
        try:
            archive_bytes = _fetch_bytes(url, DOWNLOAD_TIMEOUT_SECONDS)
            checksum_bytes = _fetch_bytes(checksum_url, DOWNLOAD_TIMEOUT_SECONDS)
        except (urllib.error.URLError, OSError, ValueError) as exc:
            # URLError covers HTTPError (404) + name/connection errors; OSError
            # covers socket timeouts and other IO; ValueError covers bad URLs.
            raise ModelDownloadError(
                url, cache_dir, f"Download failed: {exc}"
            ) from exc

        archive_path = tmp_root_path / _archive_name(model_version)
        archive_path.write_bytes(archive_bytes)

        # --- 2. Verify SHA-256 against the published checksum. ------------
        published = _parse_published_sha256(checksum_bytes)
        actual = _sha256_of_file(archive_path)
        if not published or actual != published:
            raise ModelDownloadError(
                url,
                cache_dir,
                "Checksum mismatch: "
                f"published={published!r} actual={actual!r}",
            )

        # --- 3. Safely extract into the temp staging dir. ----------------
        staging_final.mkdir(parents=True, exist_ok=True)
        try:
            with tarfile.open(archive_path, mode="r:gz") as tar:
                _safe_extract_tar(tar, staging_final)
        except (tarfile.TarError, ValueError, OSError) as exc:
            raise ModelDownloadError(
                url, cache_dir, f"Archive extraction failed: {exc}"
            ) from exc

        # Some archives nest the members under a top-level directory; if the
        # required members are not directly present but there is a single sub-
        # directory that contains them, descend into it.
        artifact_root = _locate_artifact_root(staging_final)
        if artifact_root is None:
            raise ModelDownloadError(
                url,
                cache_dir,
                "Extracted archive is missing required member(s): "
                f"{', '.join(REQUIRED_ARTIFACT_MEMBERS)}",
            )

        # --- 4. Atomic move into the final cache dir. --------------------
        # Remove any stale (invalid) dir at the destination first, then rename
        # the staged, verified directory into place in one atomic step.
        try:
            if cache_path.exists():
                shutil.rmtree(cache_path)
            os.replace(str(artifact_root), str(cache_path))
        except OSError as exc:
            raise ModelDownloadError(
                url, cache_dir, f"Atomic cache move failed: {exc}"
            ) from exc

        # --- 5. Record the verified digest marker (last write). ----------
        # Written only after a verified archive has been fully moved into
        # place, so its presence marks the entry as valid (see
        # _cache_entry_valid). If this final write fails, clean up so we do
        # not leave a half-valid entry.
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
        # Always clean up temp state (archive + any leftover staging). On the
        # success path the artifact root was moved out via os.replace, so only
        # the temp scaffolding remains.
        shutil.rmtree(tmp_root_path, ignore_errors=True)


def _locate_artifact_root(staging_dir: Path) -> Path | None:
    """Return the directory holding the required members, or ``None``.

    Handles both a flat archive (members at the staging root) and an archive
    that nests everything under a single top-level directory.
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

    The returned directory contains the expected artifact members
    (:data:`ONNX_FILENAME`, :data:`PROVENANCE_FILENAME`, and normally
    :data:`CHARSET_FILENAME`). This function performs no ONNX/provenance
    integrity checking -- that happens later in ``provenance.load_and_verify``.

    Resolution flow (design ``from_pretrained`` diagram):

    * ``path`` is provided -> treat it as a local artifact directory. Verify it
      exists and contains :data:`REQUIRED_ARTIFACT_MEMBERS`; otherwise raise
      :class:`ModelArtifactError` naming ``path``. NEVER touches the network
      (Req 4.7).
    * ``path`` is ``None`` -> resolve the cache dir via :func:`resolve_cache_dir`.
      If a cached artifact is already present there, reuse it (cache hit,
      Req 4.5). Otherwise delegate to the download seam
      :func:`_download_and_verify` (implemented in Task 10.2).

    Cache-hit test: a cache entry is reused only when it is *valid* -- the
    required members are present AND the checksum marker recorded by a
    successful download+verify+move exists (see :func:`_cache_entry_valid`).
    A stray directory that merely contains the required files (but no verified
    marker) is treated as a cache miss and re-downloaded.

    Args:
        path: Optional local artifact directory (offline load).
        model_source: Optional ``Release_Host`` override (used by the download
            seam only).
        cache_dir: Optional cache directory override.
        model_version: The Model_Version to resolve.

    Returns:
        The resolved artifact directory as a string path.

    Raises:
        ModelArtifactError: When ``path`` is given but is missing or incomplete.
        ModelDownloadError: (via the Task 10.2 seam) on download/verify failure.
    """
    # --- Local load: path= given -> NEVER touch the network (Req 4.7). --------
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

    # --- Resolve/download: no path= given. ------------------------------------
    resolved_cache_dir = resolve_cache_dir(model_version, cache_dir)

    # Cache hit: required files present AND the verified-download marker exists.
    if _cache_entry_valid(resolved_cache_dir):
        return resolved_cache_dir

    # Cache miss: download + SHA-256 verify + atomic move into the cache.
    return _download_and_verify(model_source, resolved_cache_dir, model_version)
