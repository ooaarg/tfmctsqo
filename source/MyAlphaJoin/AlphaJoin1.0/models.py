"""
Proxy module: the original code keeps the network in `3.models.py`,
while the rest of AlphaJoin does `import models`.

This module simply imports `ValueNet` from `3.models.py`.
"""

import importlib.util
import pathlib

_path = pathlib.Path(__file__).with_name("3.models.py")
_spec = importlib.util.spec_from_file_location("alpha_models_impl", _path)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_mod)  # type: ignore[attr-defined]

ValueNet = _mod.ValueNet  # type: ignore[attr-defined]
