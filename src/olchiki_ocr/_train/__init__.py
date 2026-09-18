"""``[train]`` tier for ``olchiki_ocr``: the DTRB fine-tune pipeline.

This subpackage contains the synthetic-data-generation and fine-tuning code
(``synthgen``, ``partition``, ``lmdbbuilder``, ``trainer``, ``evaluator``,
``runreport``, ``config``, ``device``, and the ``orchestrator`` that wires them
together). It reuses the carried-over ``_training/`` pipeline and shells out to a
pinned external DTRB clone (commit
``e2117f2fb882b3c6085030500a260c113be27a63``) to run training. It produces the
trained Base_Weights ``.pth`` (Req 10.4).

``[train]`` extra gating (Req 3.8)
----------------------------------
The training pipeline depends on ``torch``, ``lmdb`` (via the shelled-out DTRB
scripts / the LMDB datasets), and ``fontTools`` + ``Pillow`` (synthetic-image
rendering). These are provided ONLY by the ``[train]`` extra and are NOT core
dependencies. Two invariants keep the torch-free core clean:

1. **The core never imports this subpackage.** ``import olchiki_ocr`` must not
   import ``olchiki_ocr._train`` (its ``__init__`` from Task 14.1 does not, and
   this module is only reached by an explicit ``import olchiki_ocr._train`` or an
   attribute access on it). So a bare core import stays torch-free.

2. **This ``__init__`` imports nothing heavy at module load.** Importing
   ``olchiki_ocr._train`` itself does NOT import torch/lmdb/fontTools — the heavy
   submodules (``orchestrator`` -> ``device``/``synthgen``/``trainer``) are
   imported *lazily* on first attribute access via PEP 562 ``__getattr__``. This
   keeps ``import olchiki_ocr._train`` cheap and lets :func:`require_train` run
   its dependency check first.

When a training entry point is invoked without the ``[train]`` extra installed,
the lazy import of a heavy submodule raises ``ModuleNotFoundError`` for the
missing dependency. :func:`require_train` catches that and re-raises a clear
:class:`~olchiki_ocr.errors.ConfigError` naming the missing ``[train]`` extra
(mirroring the ``[cv]`` guard used by preprocessing). The public ``run*`` entry
points call :func:`require_train` first, so invoking training without ``[train]``
always fails with an actionable, extra-naming error rather than a raw import
error.
"""

from __future__ import annotations

from ..errors import ConfigError

__all__ = [
    "run",
    "run_data_prep",
    "run_training",
    "require_train",
    "low_resource_config",
    "Config",
    "validate_config",
]

# The heavy third-party modules the ``[train]`` extra provides. Presence of all
# of these is what distinguishes a ``[train]``-installed environment.
_TRAIN_DEPENDENCIES = ("torch", "lmdb", "fontTools", "PIL")


def require_train() -> None:
    """Ensure the ``[train]`` extra's heavy dependencies are importable (Req 3.8).

    Attempts to import each dependency the training pipeline needs (torch, lmdb,
    fontTools, Pillow). If any is missing, raises a
    :class:`~olchiki_ocr.errors.ConfigError` whose message names the ``[train]``
    extra (and the specific missing module) so the user knows exactly what to
    install::

        pip install "olchiki-ocr[train]"

    This mirrors the ``[cv]`` extra guard used by the preprocessing pipeline: a
    clear, extra-naming error rather than a raw ``ModuleNotFoundError``.

    Raises:
        ConfigError: naming the ``[train]`` extra when a required dependency is
            not importable.
    """
    import importlib

    for module_name in _TRAIN_DEPENDENCIES:
        try:
            importlib.import_module(module_name)
        except ImportError as exc:
            raise ConfigError(
                "[train]",
                "training requires the '[train]' extra, but its dependency "
                f"{module_name!r} is not installed; install it with "
                "'pip install \"olchiki-ocr[train]\"'",
            ) from exc


def run(cfg):
    """Run the full fine-tuning pipeline (guarded by :func:`require_train`)."""
    require_train()
    from .orchestrator import run as _run

    return _run(cfg)


def run_data_prep(cfg):
    """Run ONLY the data-preparation phase (guarded by :func:`require_train`)."""
    require_train()
    from .orchestrator import run_data_prep as _run_data_prep

    return _run_data_prep(cfg)


def run_training(cfg):
    """Run ONLY the training/finalization phase (guarded by :func:`require_train`)."""
    require_train()
    from .orchestrator import run_training as _run_training

    return _run_training(cfg)


# Names resolved lazily via PEP 562 so importing this subpackage stays cheap and
# torch-free until the heavy pipeline is actually used. ``Config``/
# ``validate_config``/``low_resource_config`` come from the pure-stdlib
# ``.config`` module (safe to import), but are exposed here lazily too for a
# uniform surface.
_LAZY_CONFIG_EXPORTS = frozenset({"Config", "validate_config", "low_resource_config"})


def __getattr__(name: str):
    """Lazily resolve config exports on first access (PEP 562).

    ``.config`` is pure stdlib, so this never triggers a heavy import; the heavy
    pipeline submodules are only imported inside the guarded ``run*`` wrappers.
    """
    if name in _LAZY_CONFIG_EXPORTS:
        from . import config

        return getattr(config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
