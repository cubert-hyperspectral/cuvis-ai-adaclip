"""Import smoke test: every example script must import cleanly.

Guards against silent bit-rot of the ``examples/adaclip/`` scripts. They import the
cuvis.ai stack (nodes, trainers, DataModules) but nothing else imported them, so a
removed symbol (e.g. the former ``cuvis_ai.data.MultiFileCu3sDataModule`` or
``cuvis_ai_core.data.datasets.SingleCu3sDataModule``) went unnoticed until someone ran
the script. Importing each module here executes its top-level imports and the Hydra
decorator without running ``main()``, so a dead import fails CI immediately.

Requires the ``examples`` extra (cuvis-ai + cuvis-ai-dataloader); skipped otherwise so a
dev who installed only the ``dev`` extra does not see a spurious failure. CI installs
``--all-extras``, so the smoke always runs there.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import pytest

pytest.importorskip("cuvis_ai", reason="examples extra (cuvis-ai) not installed")
pytest.importorskip(
    "cuvis_ai_dataloader", reason="examples extra (cuvis-ai-dataloader) not installed"
)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

_EXAMPLES_DIR = _REPO_ROOT / "examples" / "adaclip"
_MODULES = sorted(p.stem for p in _EXAMPLES_DIR.glob("*.py") if not p.name.startswith("_"))


@pytest.mark.parametrize("module_name", _MODULES)
def test_example_module_imports(module_name: str) -> None:
    """Importing ``examples.adaclip.<module>`` must not raise (no dead imports)."""
    importlib.import_module(f"examples.adaclip.{module_name}")
