"""LMDB dataset builder for the ``[train]`` tier.

Shells out to DTRB's vendored ``create_lmdb_dataset.py`` to convert a raw
image-folder + ``gt.txt`` layout into an LMDB dataset for ``train.py``, and
raises ``DtrbError`` on a non-zero exit. Using a subprocess keeps DTRB and its
deps out of this package's imports.
"""

from __future__ import annotations

import os
import subprocess
import sys

from ..errors import DtrbError

__all__ = ["build_lmdb"]


def build_lmdb(
    input_folder: str,
    gt_file: str,
    output_lmdb: str,
    dtrb_repo_path: str,
    python_executable: str | None = None,
) -> None:
    """Invoke DTRB ``create_lmdb_dataset.py`` to produce an LMDB dataset.

    ``python_executable`` defaults to ``sys.executable`` (whose environment
    carries DTRB's deps). Raises ``DtrbError`` carrying stderr on a non-zero exit.
    """
    interpreter = python_executable if python_executable is not None else sys.executable
    script_path = os.path.join(dtrb_repo_path, "create_lmdb_dataset.py")

    # Argument list (no shell string) to avoid command injection.
    command = [
        interpreter,
        script_path,
        "--inputPath",
        input_folder,
        "--gtFile",
        gt_file,
        "--outputPath",
        output_lmdb,
    ]

    # Force UTF-8 in the child so DTRB can emit the non-Latin Ol Chiki charset on
    # Windows (cp1252) without crashing.
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

    # check=False so we can wrap a non-zero exit in DtrbError; decode pipes as
    # UTF-8 (errors="replace") to tolerate odd bytes.
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        env=env,
    )

    if result.returncode != 0:
        raise DtrbError(result.stderr or "")
