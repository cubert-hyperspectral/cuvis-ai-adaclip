"""Nothing shipped here may name the pre-0.8 re-export packages of cuvis-ai.

cuvis-ai kept top-level ``anomaly`` and ``deciders`` packages only as deprecated re-exports of
``cuvis_ai.node.anomaly`` and ``cuvis_ai.node.deciders``. Those re-exports are being removed, so
every pipeline ``class_name``, every example import and every docstring in this repository must
use the real module path; a released tag that still names a re-export breaks on first import
against a cuvis-ai without it.
"""

from __future__ import annotations

import re
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parent.parent
_SHIM_PACKAGE = re.compile(r"\bcuvis_ai\.(?:anomaly|deciders)\b")
_SCANNED = ("configs", "examples", "cuvis_ai_adaclip", "tests", "README.md")
_SUFFIXES = {".py", ".yaml", ".yml", ".md"}


def _scanned_files() -> list[Path]:
    """Every shipped source, config and doc file, this test excluded."""
    files: list[Path] = []
    for entry in _SCANNED:
        path = _REPO_ROOT / entry
        if path.is_file():
            files.append(path)
            continue
        files.extend(p for p in path.rglob("*") if p.is_file() and p.suffix in _SUFFIXES)
    return sorted(p for p in files if p != Path(__file__).resolve())


@pytest.mark.parametrize(
    "path", _scanned_files(), ids=lambda p: p.relative_to(_REPO_ROOT).as_posix()
)
def test_file_names_no_shim_package(path: Path) -> None:
    """A file that names a re-export package fails with the offending line numbers."""
    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    hits = [number for number, line in enumerate(lines, start=1) if _SHIM_PACKAGE.search(line)]
    assert not hits, (
        f"{path.relative_to(_REPO_ROOT).as_posix()} names a removed cuvis_ai re-export package "
        f"on lines {hits}; use the cuvis_ai.node.* path"
    )
