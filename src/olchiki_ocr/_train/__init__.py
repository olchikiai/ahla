"""``[train]`` tier: the DTRB fine-tune pipeline.

Bundles the synthetic-data and fine-tuning code and shells out to a pinned DTRB
clone to produce the trained Base_Weights ``.pth``.

Importing this subpackage stays light: the heavy submodules (torch/lmdb/
fontTools/PIL) are imported lazily on first access, so the torch-free core never
pulls them in. The public ``run*`` entry points call :func:`require_train`
first, which checks the ``[train]`` deps and raises a ``ConfigError`` naming the
``[train]`` extra when one is missing.
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

# The heavy deps the ``[train]`` extra provides.
_TRAIN_DEPENDENCIES = ("torch", "lmdb", "fontTools", "PIL")


def require_train() -> None:
    """Check the ``[train]`` deps are importable, else raise ConfigError.

    Tries to import each of torch/lmdb/fontTools/PIL; a missing one raises a
    :class:`~olchiki_ocr.errors.ConfigError` naming the ``[train]`` extra.
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


# Config exports resolved lazily (PEP 562) for a uniform surface. ``.config`` is
# pure stdlib, so this stays torch-free.
_LAZY_CONFIG_EXPORTS = frozenset({"Config", "validate_config", "low_resource_config"})


def __getattr__(name: str):
    """Lazily resolve config exports on first access (PEP 562)."""
    if name in _LAZY_CONFIG_EXPORTS:
        from . import config

        return getattr(config, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted({*globals().keys(), *__all__})
