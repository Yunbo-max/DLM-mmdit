# LSME — Latent-Steered Masked Editing
# Delta build on existing codebase: mmdit_latent/

import importlib
import sys

# Register bundled mmdit_latent as a top-level module alias so that
#   from mmdit_latent.utils import sample_categorical
# resolves to lsme.mmdit_latent.utils even when the top-level
# mmdit_latent/ directory is not on sys.path.
#
# Only register the alias if mmdit_latent is not already importable
# (i.e. when lsme/ is used as a standalone package).
try:
    importlib.import_module("mmdit_latent")
except ImportError:
    from lsme import mmdit_latent as _bundled

    sys.modules["mmdit_latent"] = _bundled

    # Also register submodules that are directly imported elsewhere
    _submodules = [
        "utils",
        "sampling",
        "checkpoints",
        "diffusion_process",
        "loss",
        "optimizer",
        "data_simple",
        "modeling_latent",
        "trainer_latent",
        "train_latent_dit",
        "models",
        "models.dit",
        "models.mmdit_block",
        "models.mmdit_latent",
    ]
    for _sub in _submodules:
        _full = f"mmdit_latent.{_sub}"
        if _full not in sys.modules:
            try:
                _mod = importlib.import_module(f"lsme.mmdit_latent.{_sub}")
                sys.modules[_full] = _mod
            except ImportError:
                pass  # optional submodule, skip
