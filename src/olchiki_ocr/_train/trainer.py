"""Transfer-learning Trainer for the ``[train]`` tier.

Fine-tunes the recognition model by shelling out to the vendored clovaai
deep-text-recognition-benchmark (DTRB) ``train.py`` script. Keeping DTRB behind
a subprocess boundary avoids import-time coupling to its internal module layout
and its own dependency set, and mirrors how ``lmdbbuilder.py`` invokes
``create_lmdb_dataset.py``.

This module exposes three units:

* :func:`build_train_args` -- a *pure* function assembling the ``train.py``
  flag list (``--character`` equals the full Charset so the output-class count
  equals the Charset size).
* :func:`train` -- invokes DTRB ``train.py`` via subprocess, parses its log for
  per-interval validation metrics into ``RunStats``, maps load failures to
  :class:`PretrainedModelError` and other non-zero exits to :class:`DtrbError`,
  and returns the selected Fine_Tuned_Model (Base_Weights ``.pth``) checkpoint
  path.
* :func:`select_best_checkpoint` -- a *pure* function selecting the lowest-CER
  checkpoint when best-checkpoint selection is enabled, else the final-iteration
  checkpoint.

Low-resource preset (Req 10.3)
------------------------------
When ``cfg.freeze_feature_extraction`` is set (as :func:`config.low_resource_config`
does), :func:`build_train_args` forwards ``--freeze_FeatureExtraction`` to DTRB
``train.py``. The vendored DTRB clone (pinned commit
``e2117f2fb882b3c6085030500a260c113be27a63``) honors that flag by setting
``requires_grad=False`` on every FeatureExtraction (VGG) parameter before the
optimizer collects trainable parameters (DTRB only optimizes params where
``requires_grad`` is True), so the VGG backbone is frozen and only the BiLSTM
SequenceModeling stage and the CTC Prediction head are trained. The preset also
uses a small ``batch_size`` (set by the Config factory). Together with the
frozen backbone this is the documented low-resource fine-tune (Req 10.3), and it
still produces a trained Base_Weights ``.pth`` on completion (Req 10.4).

Verified DTRB facts (pinned commit; read directly from ``train.py``):

* ``train.py`` uses ``argparse``. The flags this module forwards all exist with
  these exact names: ``--train_data`` (required), ``--valid_data`` (required),
  ``--saved_model`` (default ``''``), ``--character``, ``--num_iter``,
  ``--batch_size``, ``--lr``, ``--manualSeed`` (default 1111), ``--valInterval``,
  the four *required* stage flags, ``--FT``, ``--workers``, ``--select_data``,
  ``--batch_ratio``, and (added for the preset) ``--freeze_FeatureExtraction``.
* ``--FT``: required for transfer learning from a pretrained recognizer whose
  ``--character`` set differs; DTRB filters mismatched-shape checkpoint entries
  and loads the rest with ``strict=False`` so the resized Prediction layer is
  tolerated. The Trainer always passes ``--FT`` when a ``--saved_model`` is
  supplied.
* Checkpoints/logs are written to ``./saved_models/<exp_name>/`` relative to the
  process cwd; the Trainer passes an explicit ``--exp_name`` and runs with a
  controlled ``cwd`` so checkpoints land in a known location.
* On reaching ``--num_iter``, ``train.py`` calls ``sys.exit()`` (exit code 0);
  any non-zero exit is a failure.
* The subprocess runs in Python UTF-8 mode so DTRB can print/write the non-Latin
  Ol Chiki ``--character`` charset on Windows; the parent decodes as UTF-8.
* Log format (validation blocks)::

      [<iter>/<num_iter>] Train loss: <..>, Valid loss: <..>, Elapsed_time: <..>
      Current_accuracy   : <acc>, Current_norm_ED   : <norm_ed>
      Best_accuracy      : <..>, Best_norm_ED       : <..>

  ``word_accuracy = Current_accuracy / 100`` and ``cer = 1 - Current_norm_ED``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import TYPE_CHECKING

from ..errors import DtrbError, PretrainedModelError

if TYPE_CHECKING:  # avoid runtime coupling; annotations are strings under __future__
    from ..charset import Charset
    from .config import Config
    from .runstats import RunStats

__all__ = ["build_train_args", "train", "select_best_checkpoint"]

# ``train.py``'s own ``--manualSeed`` default. Used when ``cfg.seed is None`` so
# the run is still deterministic and reproducible from the recorded seed.
_DEFAULT_MANUAL_SEED = 1111

# DTRB writes checkpoints/logs under this directory (relative to the process cwd).
_SAVED_MODELS_DIRNAME = "saved_models"

# Matches a DTRB validation log block. The iteration count is on the
# ``[<iter>/<num_iter>]`` line; the accuracy/norm_ED are on the following
# ``Current_accuracy ... Current_norm_ED ...`` line.
_ITER_RE = re.compile(r"\[(\d+)/\d+\]\s+Train loss:")
_METRIC_RE = re.compile(
    r"Current_accuracy\s*:\s*([-\d.]+),\s*Current_norm_ED\s*:\s*([-\d.]+)"
)


def build_train_args(
    cfg: "Config",
    charset: "Charset",
    train_lmdb: str,
    val_lmdb: str,
    device: str,
    exp_name: str | None = None,
    saved_model: str | None = None,
) -> list[str]:
    """Assemble the DTRB ``train.py`` argument list (flags only).

    The returned list contains only the flags forwarded to ``train.py`` -- not
    the interpreter or the script path. Values are stringified explicitly and the
    list is built without any shell string interpolation, so no value is ever
    interpreted by a shell (no injection surface).

    The ``--character`` value is exactly ``charset.as_dtrb_character_arg()`` so
    DTRB's label alphabet -- and therefore the model output-class count -- equals
    the Charset size.

    ``--FT`` is included whenever a pretrained model is supplied so DTRB loads
    the checkpoint tolerantly (dropping mismatched-shape entries, ``strict=False``)
    and tolerates the resized Prediction layer. When ``cfg.seed`` is ``None`` the
    DTRB default manual seed (:data:`_DEFAULT_MANUAL_SEED`) is used.

    When ``cfg.freeze_feature_extraction`` is set (low-resource preset, Req 10.3)
    the function also forwards ``--freeze_FeatureExtraction`` so DTRB freezes the
    VGG backbone and trains only SequenceModeling + Prediction.

    ``--select_data "/"`` and ``--batch_ratio "1"`` treat our single flat LMDB as
    one dataset; ``--workers 0`` uses single-process data loading (DTRB's
    ``LmdbDataset`` holds an open, unpicklable LMDB ``Environment``).

    Args:
        cfg: The validated run Configuration.
        charset: The built recognition Charset.
        train_lmdb: Path to the training LMDB dataset (``--train_data``).
        val_lmdb: Path to the validation LMDB dataset (``--valid_data``).
        device: The selected Compute_Device (recorded by the caller; DTRB itself
            auto-detects CUDA).
        exp_name: Optional explicit ``--exp_name``.
        saved_model: Optional override for ``--saved_model``; defaults to
            ``cfg.pretrained_recognizer``.

    Returns:
        The list of ``train.py`` flags.
    """
    seed = cfg.seed if cfg.seed is not None else _DEFAULT_MANUAL_SEED
    saved_model_path = saved_model if saved_model is not None else cfg.pretrained_recognizer

    args: list[str] = [
        "--train_data",
        train_lmdb,
        "--valid_data",
        val_lmdb,
        # Our Synthetic_Generator emits ONE flat LMDB per partition (no MJ/ST
        # sub-datasets), so select the dataset root itself with full batch share.
        "--select_data",
        "/",
        "--batch_ratio",
        "1",
        # Single-process data loading (num_workers=0): DTRB's LmdbDataset holds
        # an open LMDB Environment that cannot be pickled under spawn/forkserver.
        "--workers",
        "0",
        "--saved_model",
        saved_model_path,
        # --FT: load the pretrained checkpoint tolerantly so the resized
        # Prediction layer (new Charset size) is accepted.
        "--FT",
        # --character sets the label alphabet and therefore the output-class
        # count == Charset size.
        "--character",
        charset.as_dtrb_character_arg(),
        "--num_iter",
        str(cfg.num_iterations),
        "--batch_size",
        str(cfg.batch_size),
        "--lr",
        str(cfg.learning_rate),
        "--manualSeed",
        str(seed),
        "--valInterval",
        str(cfg.val_interval),
        # Network-stage flags (all four are required by train.py's argparse).
        "--Transformation",
        cfg.network_transformation,
        "--FeatureExtraction",
        cfg.network_feature,
        "--SequenceModeling",
        cfg.network_sequence,
        "--Prediction",
        cfg.network_prediction,
    ]

    # Low-resource preset (Req 10.3): freeze the VGG FeatureExtraction backbone
    # so only SequenceModeling (BiLSTM) + Prediction (head) are trained. DTRB
    # sets requires_grad=False on FeatureExtraction params for this flag; its
    # optimizer only collects requires_grad=True params, so the backbone is
    # frozen.
    if cfg.freeze_feature_extraction:
        args.append("--freeze_FeatureExtraction")

    if exp_name is not None:
        args += ["--exp_name", exp_name]

    return args


def _default_exp_name(cfg: "Config", seed: int) -> str:
    """Reproduce DTRB's default ``exp_name`` so checkpoints land predictably.

    DTRB builds
    ``{Transformation}-{FeatureExtraction}-{SequenceModeling}-{Prediction}-Seed{manualSeed}``
    when ``--exp_name`` is not supplied. We pass this explicitly so we always
    know the checkpoint directory.
    """
    return (
        f"{cfg.network_transformation}-{cfg.network_feature}-"
        f"{cfg.network_sequence}-{cfg.network_prediction}-Seed{seed}"
    )


def _parse_metrics(stdout: str) -> list[tuple[int, float, float]]:
    """Parse DTRB stdout into ``(iteration, cer, word_accuracy)`` tuples.

    DTRB emits, per validation interval, an ``[<iter>/<num_iter>] Train loss:``
    line immediately followed by a ``Current_accuracy : <acc>, Current_norm_ED :
    <norm_ed>`` line. This scans the log line by line, pairing each iteration
    line with the next metric line encountered.

    Derivation: DTRB does not log CER directly, so
    ``word_accuracy = accuracy / 100`` and ``cer = 1 - norm_ED``.

    Args:
        stdout: The captured ``train.py`` standard output.

    Returns:
        A list of ``(iteration, cer, word_accuracy)`` tuples in log order. Lines
        that do not match the expected format are skipped.
    """
    metrics: list[tuple[int, float, float]] = []
    pending_iteration: int | None = None

    for line in stdout.splitlines():
        iter_match = _ITER_RE.search(line)
        if iter_match is not None:
            pending_iteration = int(iter_match.group(1))
            continue

        metric_match = _METRIC_RE.search(line)
        if metric_match is not None and pending_iteration is not None:
            accuracy = float(metric_match.group(1))
            norm_ed = float(metric_match.group(2))
            word_accuracy = accuracy / 100.0
            cer = 1.0 - norm_ed
            metrics.append((pending_iteration, cer, word_accuracy))
            pending_iteration = None

    return metrics


# State-dict-specific substrings that unambiguously indicate the --saved_model
# checkpoint could not be loaded. Matched case-insensitively.
_LOAD_FAILURE_MARKERS = (
    "load_state_dict",
    "error(s) in loading state_dict",
    "unexpected key",
    "missing key",
    "size mismatch",
)

# Load/checkpoint verbs that, when appearing near the saved_model path, indicate
# the failure is about loading that checkpoint specifically.
_LOAD_CONTEXT_MARKERS = (
    "no such file or directory",
    "loading",
    "load",
    "checkpoint",
)


def _looks_like_load_failure(output: str, saved_model: str) -> bool:
    """Heuristically decide whether a failure is a pretrained-model load error.

    Scoped to the ``--saved_model`` checkpoint so unrelated failures (a missing
    ``train.py`` script, a missing LMDB dataset) are NOT misclassified.

    Returns True when the failure is genuinely about loading the checkpoint, so
    the Trainer can raise the more specific :class:`PretrainedModelError` instead
    of a generic :class:`DtrbError`.
    """
    haystack = output.lower()
    if any(marker in haystack for marker in _LOAD_FAILURE_MARKERS):
        return True
    if saved_model:
        saved_model_name = os.path.basename(saved_model).lower()
        if saved_model_name and saved_model_name in haystack:
            if any(marker in haystack for marker in _LOAD_CONTEXT_MARKERS):
                return True
    return False


def select_best_checkpoint(
    candidates: list[tuple[str, float]],
    best_checkpoint: bool,
) -> str:
    """Return the Fine_Tuned_Model checkpoint path from scored candidates.

    Args:
        candidates: ``(checkpoint_path, cer)`` pairs. Ordering is assumed to be
            *training order*, so the **last** element is the final-iteration
            checkpoint.
        best_checkpoint: When True, return a checkpoint whose CER is the minimum
            over all candidates. When False, return the final-iteration
            checkpoint -- the last element of ``candidates``.

    Returns:
        The selected checkpoint path.

    Raises:
        ValueError: If ``candidates`` is empty.
    """
    if not candidates:
        raise ValueError("select_best_checkpoint requires at least one candidate")

    if best_checkpoint:
        # min() returns the first candidate achieving the minimum CER.
        return min(candidates, key=lambda pair: pair[1])[0]

    # Final-iteration checkpoint == last candidate in training order.
    return candidates[-1][0]


def train(
    cfg: "Config",
    charset: "Charset",
    train_lmdb: str,
    val_lmdb: str,
    device: str,
    stats: "RunStats",
    python_executable: str | None = None,
    work_dir: str | None = None,
) -> str:
    """Fine-tune the Recognition_Model via DTRB ``train.py`` and return the
    selected Fine_Tuned_Model (Base_Weights ``.pth``) checkpoint path.

    Shells out to ``<python> <dtrb_repo_path>/train.py <build_train_args(...)>``
    with ``cwd`` set to ``work_dir`` (default ``cfg.output_dir``) so DTRB's
    hard-coded ``saved_models/<exp_name>/`` output directory is under our
    control. The command is assembled as an argument list (never a shell string).

    Per-interval validation metrics parsed from DTRB's stdout are appended to
    ``stats.per_interval_metrics`` as ``(iteration, cer, word_accuracy)`` tuples.

    Args:
        cfg: The validated run Configuration.
        charset: The built recognition Charset.
        train_lmdb: Path to the training LMDB dataset.
        val_lmdb: Path to the validation LMDB dataset.
        device: The selected Compute_Device (recorded provenance).
        stats: The mutable RunStats accumulator; ``per_interval_metrics`` is
            populated here.
        python_executable: Interpreter used to run ``train.py``. Defaults to
            ``sys.executable``.
        work_dir: Working directory for the subprocess; DTRB writes
            ``saved_models/<exp_name>/`` beneath it. Defaults to
            ``cfg.output_dir``.

    Returns:
        The filesystem path to the selected Fine_Tuned_Model checkpoint (the
        trained Base_Weights ``.pth``, Req 10.4).

    Raises:
        PretrainedModelError: When ``--saved_model`` cannot be loaded.
        DtrbError: On any other non-zero subprocess exit, carrying the captured
            stderr (Req 10.6).
    """
    interpreter = python_executable if python_executable is not None else sys.executable
    # Resolve every path DTRB receives to an absolute path so the command is
    # independent of the subprocess cwd.
    script_path = os.path.abspath(os.path.join(cfg.dtrb_repo_path, "train.py"))
    run_cwd = os.path.abspath(work_dir if work_dir is not None else cfg.output_dir)

    abs_train_lmdb = os.path.abspath(train_lmdb)
    abs_val_lmdb = os.path.abspath(val_lmdb)

    seed = cfg.seed if cfg.seed is not None else _DEFAULT_MANUAL_SEED
    exp_name = _default_exp_name(cfg, seed)

    command = [
        interpreter,
        script_path,
        *build_train_args(
            cfg,
            charset,
            abs_train_lmdb,
            abs_val_lmdb,
            device,
            exp_name=exp_name,
            saved_model=os.path.abspath(cfg.pretrained_recognizer),
        ),
    ]

    # Run the child Python in UTF-8 mode so DTRB can print/write the Ol Chiki
    # (non-Latin) --character string on Windows (cp1252) without a Unicode crash.
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

    # Decode the captured pipes explicitly as UTF-8 (errors="replace").
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
        cwd=run_cwd,
        env=env,
    )

    stdout = result.stdout or ""
    stderr = result.stderr or ""

    # Record per-interval validation metrics regardless of exit status.
    stats.per_interval_metrics.extend(_parse_metrics(stdout))

    if result.returncode != 0:
        combined = f"{stdout}\n{stderr}"
        if _looks_like_load_failure(combined, cfg.pretrained_recognizer):
            raise PretrainedModelError(
                cfg.pretrained_recognizer,
                "DTRB train.py could not load the pretrained model; DTRB stderr:\n"
                + (stderr.strip() or "<empty>"),
            )
        raise DtrbError(stderr)

    # Assemble the candidate checkpoint set from DTRB's known outputs.
    checkpoint_dir = os.path.join(run_cwd, _SAVED_MODELS_DIRNAME, exp_name)
    candidate_files = _discover_checkpoints(checkpoint_dir)

    if not candidate_files:
        # No checkpoint on disk (e.g. run too short to write one). Fall back to
        # the conventional best_accuracy.pth path so the caller has a concrete
        # path to load; the artifact/export step surfaces a load failure if absent.
        return os.path.join(checkpoint_dir, "best_accuracy.pth")

    # Score each checkpoint with the lowest CER observed during training.
    best_cer = min((cer for _, cer, _ in stats.per_interval_metrics), default=0.0)
    candidates = [(path, best_cer) for path in candidate_files]

    return select_best_checkpoint(candidates, cfg.best_checkpoint)


def _discover_checkpoints(checkpoint_dir: str) -> list[str]:
    """Return existing DTRB checkpoint paths in discovery/selection order.

    Orders ``iter_<n>.pth`` by ascending iteration number, then appends the
    best_* checkpoints in the order ``best_accuracy.pth`` then
    ``best_norm_ED.pth`` when present.

    Because :func:`train` pairs every discovered checkpoint with the same
    ``best_cer`` score, :func:`select_best_checkpoint` (with
    ``best_checkpoint=True``) resolves the tie by returning the FIRST candidate
    via ``min()``; putting ``best_accuracy.pth`` first ships DTRB's
    accuracy-selected checkpoint, the conventional "best" artifact.

    Returns an empty list when the directory does not exist or holds no known
    checkpoint files.
    """
    if not os.path.isdir(checkpoint_dir):
        return []

    iter_re = re.compile(r"^iter_(\d+)\.pth$")
    iter_checkpoints: list[tuple[int, str]] = []
    best_checkpoints: list[str] = []

    for name in os.listdir(checkpoint_dir):
        match = iter_re.match(name)
        if match is not None:
            iter_checkpoints.append((int(match.group(1)), name))
        elif name in ("best_norm_ED.pth", "best_accuracy.pth"):
            best_checkpoints.append(name)

    iter_checkpoints.sort(key=lambda pair: pair[0])
    ordered = [name for _, name in iter_checkpoints]
    for name in ("best_accuracy.pth", "best_norm_ED.pth"):
        if name in best_checkpoints:
            ordered.append(name)

    return [os.path.join(checkpoint_dir, name) for name in ordered]
