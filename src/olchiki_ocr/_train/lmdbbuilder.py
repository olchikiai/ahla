"""LMDB dataset builder for the ``[train]`` tier.

Converts a raw DTRB dataset layout (a folder of rendered images plus a
``gt.txt`` of ``imagepath<TAB>label`` lines, as emitted by the
Synthetic_Generator) into an LMDB dataset consumed by DTRB ``train.py``.

The conversion itself is performed by DTRB's vendored
``create_lmdb_dataset.py`` script, which this module shells out to via
``subprocess``. Keeping DTRB behind a subprocess boundary avoids import-time
coupling to DTRB's internal module layout and its own dependency set
(``fire``, ``lmdb``, ``cv2``, ``numpy`` live in the DTRB environment, not in
this package's core dependencies).

Verified against the pinned DTRB clone (commit
``e2117f2fb882b3c6085030500a260c113be27a63``): ``create_lmdb_dataset.py``
exposes ``fire.Fire(createDataset)`` with the signature
``createDataset(inputPath, gtFile, outputPath, checkValid=True)``, so ``fire``
maps the parameters to the CLI flags ``--inputPath``, ``--gtFile``,
``--outputPath`` (and optional ``--checkValid``).
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

    Shells out to::

        <python> <dtrb_repo_path>/create_lmdb_dataset.py \\
            --inputPath <input_folder> \\
            --gtFile    <gt_file> \\
            --outputPath <output_lmdb>

    The command is assembled as an argument list (never a shell string), so
    values containing spaces or shell metacharacters are passed verbatim to the
    script and cannot be interpreted by a shell (no injection surface).

    Args:
        input_folder: Folder the ``gt.txt`` image paths are relative to
            (DTRB ``--inputPath``).
        gt_file: Path to the ``gt.txt`` of ``imagepath<TAB>label`` lines
            (DTRB ``--gtFile``).
        output_lmdb: Destination directory for the LMDB dataset
            (DTRB ``--outputPath``).
        dtrb_repo_path: Path to the local DTRB clone containing
            ``create_lmdb_dataset.py``.
        python_executable: Interpreter used to run the DTRB script. Defaults to
            ``sys.executable`` (the same interpreter running the Trainer), which
            is the interpreter whose environment carries DTRB's dependencies.
            Overridable so a separate DTRB environment can be targeted.

    Raises:
        DtrbError: When the DTRB subprocess exits with a non-zero status. The
            error carries the captured subprocess ``stderr`` so the failure is
            diagnosable.
    """
    interpreter = python_executable if python_executable is not None else sys.executable
    script_path = os.path.join(dtrb_repo_path, "create_lmdb_dataset.py")

    # Build the argument list explicitly (no shell string interpolation) so no
    # value is ever interpreted by a shell (avoids command injection).
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

    # Run the child Python in UTF-8 mode. On Windows the child otherwise defaults
    # to the cp1252 locale encoding, so when DTRB emits the Ol Chiki (non-Latin)
    # charset it would raise a Unicode error; PYTHONUTF8=1 makes the child's
    # std streams and open() default to UTF-8 (PYTHONIOENCODING is a belt-and-
    # suspenders for the std streams).
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

    # Capture stdout/stderr as text; do not raise automatically so we can wrap a
    # non-zero exit in DtrbError carrying the captured stderr. Decode the
    # captured pipes explicitly as UTF-8 (errors="replace") so the parent never
    # crashes on non-Latin/odd bytes in DTRB's diagnostic output.
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
