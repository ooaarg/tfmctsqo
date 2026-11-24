"""
Proxy module: the original MCTS implementation lives in `7.mcts.py`,
while `8.findBestPlan.py` expects `from mcts import mcts`.

This module loads `7.mcts.py` by file path and re-exports the `mcts` class.
"""

import importlib.util
import pathlib

_path = pathlib.Path(__file__).with_name("7.mcts.py")
_spec = importlib.util.spec_from_file_location("alpha_mcts_impl", _path)
_mod = importlib.util.module_from_spec(_spec)  # type: ignore[arg-type]
assert _spec is not None and _spec.loader is not None
_spec.loader.exec_module(_mod)  # type: ignore[attr-defined]

mcts = _mod.mcts  # type: ignore[attr-defined]
