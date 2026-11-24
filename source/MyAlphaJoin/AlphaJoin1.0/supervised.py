"""
Compatibility proxy module: the original AlphaJoin code
uses `4.supervised.py` for the implementation, while imports expect
`import supervised`.

We simply re-export the `supervised` class from `4.supervised.py`,
loading it by file path.
"""

import importlib.util
import pathlib

_path = pathlib.Path(__file__).with_name("4.supervised.py")
_spec = importlib.util.spec_from_file_location("alpha_supervised_impl", _path)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_mod)  # type: ignore[attr-defined]

# Ensure correct serialization of the data class via pickle:
# 1) declare that it belongs to the 'supervised' module
# 2) export the name data from this module
if hasattr(_mod, "data"):
    _mod.data.__module__ = __name__  # type: ignore[attr-defined]
    data = _mod.data  # type: ignore[attr-defined]

supervised = _mod.supervised  # type: ignore[attr-defined]
