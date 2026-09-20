"""Transfer-learning Trainer for the ``[train]`` tier.

Fine-tunes the recognizer by shelling out to the pinned DTRB clone's
``train.py`` (subprocess boundary keeps DTRB's deps out of our imports). On a
non-zero exit it raises ``DtrbError(stderr)``, or ``PretrainedModelError`` when
the failure is about loading ``--saved_model``.

Three units: :func:`build_train_args` (pure flag builder; ``--character`` equals
the full Charset, and ``--freeze_FeatureExtraction`` is forwarded only when the
low-resource preset sets it), :func:`train` (runs the subprocess, parses
per-interval CER/Word_Accuracy into ``RunStats``, returns the selected
Base_Weights ``.pth``), and :func:`select_best_checkpoint` (pure selector).

DTRB log format parsed for metrics::

    [<iter>/<num_iter>] Train loss: ...
    Current_accuracy : <acc>, Current_norm_ED : <norm_ed>

with ``word_accuracy = accuracy / 100`` and ``cer = 1 - norm_ED``.
"""

from __future__ import annotations

import os
import re
import subprocess
import sys
from typing import TYPE_CHECKING

from ..errors import DtrbError, PretrainedModelError

if TYPE_CHECKING:  # no runtime coupling
    from ..charset import Charset
    from .config import Config
    from .runstats import RunStats

__all__ = ["build_train_args", "train", "select_best_checkpoint"]

# DTRB's own ``--manualSeed`` default, used when ``cfg.seed`` is None.
_DEFAULT_MANUAL_SEED = 1111

# DTRB writes checkpoints/logs under this dir (relative to the process cwd).
_SAVED_MODELS_DIRNAME = "saved_models"

# Match the DTRB validation log block (iteration line, then metric line).
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
    """Assemble the DTRB ``train.py`` flag list (no interpreter/script path).

    ``--character`` equals ``charset.as_dtrb_character_arg()`` (so the output
    class count matches the Charset). ``--FT`` is always passed with a saved
    model so DTRB loads it tolerantly (``strict=False``). When
    ``cfg.freeze_feature_extraction`` is set, ``--freeze_FeatureExtraction`` is
    forwarded so only SequenceModeling + Prediction train. ``saved_model``
    defaults to ``cfg.pretrained_recognizer``.
    """
    seed = cfg.seed if cfg.seed is not None else _DEFAULT_MANUAL_SEED
    saved_model_path = saved_model if saved_model is not None else cfg.pretrained_recognizer

    args: list[str] = [
        "--train_data",
        train_lmdb,
        "--valid_data",
        val_lmdb,
        # One flat LMDB per partition: select the root with full batch share.
        "--select_data",
        "/",
        "--batch_ratio",
        "1",
        # workers=0: DTRB's LmdbDataset holds an unpicklable open Environment.
        "--workers",
        "0",
        "--saved_model",
        saved_model_path,
        # --FT: load the checkpoint tolerantly (resized Prediction layer).
        "--FT",
        # --character sets the label alphabet == output-class count.
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
        # Network-stage flags (all four required by train.py).
        "--Transformation",
        cfg.network_transformation,
        "--FeatureExtraction",
        cfg.network_feature,
        "--SequenceModeling",
        cfg.network_sequence,
        "--Prediction",
        cfg.network_prediction,
    ]

    # Low-resource preset: freeze the VGG backbone (train only BiLSTM + head).
    if cfg.freeze_feature_extraction:
        args.append("--freeze_FeatureExtraction")

    if exp_name is not None:
        args += ["--exp_name", exp_name]

    return args


def _default_exp_name(cfg: "Config", seed: int) -> str:
    """Reproduce DTRB's default ``exp_name`` so we know the checkpoint dir."""
    return (
        f"{cfg.network_transformation}-{cfg.network_feature}-"
        f"{cfg.network_sequence}-{cfg.network_prediction}-Seed{seed}"
    )


def _parse_metrics(stdout: str) -> list[tuple[int, float, float]]:
    """Parse DTRB stdout into ``(iteration, cer, word_accuracy)`` tuples.

    Pairs each iteration line with the following metric line;
    ``word_accuracy = accuracy / 100`` and ``cer = 1 - norm_ED``. Unmatched
    lines are skipped.
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


# State-dict substrings that indicate the --saved_model could not be loaded.
_LOAD_FAILURE_MARKERS = (
    "load_state_dict",
    "error(s) in loading state_dict",
    "unexpected key",
    "missing key",
    "size mismatch",
)

# Load verbs that, near the saved_model path, scope a failure to that checkpoint.
_LOAD_CONTEXT_MARKERS = (
    "no such file or directory",
    "loading",
    "load",
    "checkpoint",
)


def _looks_like_load_failure(output: str, saved_model: str) -> bool:
    """Heuristically decide whether a failure is a pretrained-model load error.

    Scoped to ``--saved_model`` so unrelated failures are not misclassified,
    letting the Trainer raise PretrainedModelError instead of DtrbError.
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
    """Return the selected checkpoint from ``(path, cer)`` pairs in training order.

    When ``best_checkpoint`` is True, pick the lowest-CER candidate, else the
    last (final-iteration) one. Raises ValueError if ``candidates`` is empty.
    """
    if not candidates:
        raise ValueError("select_best_checkpoint requires at least one candidate")

    if best_checkpoint:
        # min() returns the first candidate achieving the minimum CER.
        return min(candidates, key=lambda pair: pair[1])[0]

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
    """Fine-tune via DTRB ``train.py`` and return the selected Base_Weights
    ``.pth`` checkpoint path.

    Runs with ``cwd=work_dir`` (default ``cfg.output_dir``) so DTRB's
    ``saved_models/<exp_name>/`` lands where we expect, appends per-interval
    metrics to ``stats.per_interval_metrics``, and raises PretrainedModelError
    (unloadable ``--saved_model``) or DtrbError (any other non-zero exit).
    """
    interpreter = python_executable if python_executable is not None else sys.executable
    # Absolute paths so the command is independent of the subprocess cwd.
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

    # Force UTF-8 in the child so DTRB can emit the non-Latin --character on
    # Windows (cp1252) without crashing.
    env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}

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

    # Record per-interval metrics regardless of exit status.
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

    checkpoint_dir = os.path.join(run_cwd, _SAVED_MODELS_DIRNAME, exp_name)
    candidate_files = _discover_checkpoints(checkpoint_dir)

    if not candidate_files:
        # No checkpoint on disk: fall back to the conventional best_accuracy.pth.
        return os.path.join(checkpoint_dir, "best_accuracy.pth")

    # Score each checkpoint with the lowest CER seen during training.
    best_cer = min((cer for _, cer, _ in stats.per_interval_metrics), default=0.0)
    candidates = [(path, best_cer) for path in candidate_files]

    return select_best_checkpoint(candidates, cfg.best_checkpoint)


def _discover_checkpoints(checkpoint_dir: str) -> list[str]:
    """Return existing DTRB checkpoints in selection order.

    ``iter_<n>.pth`` by ascending iteration, then ``best_accuracy.pth`` then
    ``best_norm_ED.pth``. Ordering best_accuracy first makes it win the equal-CER
    tie in :func:`select_best_checkpoint`. Empty when the dir is absent.
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
