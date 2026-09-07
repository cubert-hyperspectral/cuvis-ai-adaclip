"""The plugin's weight declarations (``WEIGHTS``) and the tables derived from them.

No network and no checkpoint: the declarations are data and the registry is in-process.
"""

from __future__ import annotations

import inspect
import subprocess
import sys
from pathlib import Path

import pytest
from cuvis_ai_core.data.model_weights import ModelWeights
from cuvis_ai_schemas.plugin import PluginWeightEntry

import cuvis_ai_adaclip
import cuvis_ai_adaclip.weights as weights_mod
from cuvis_ai_adaclip import AdaCLIPDetector
from cuvis_ai_adaclip.adaclip_upstream import _MIRRORED_CLIP_BACKBONES
from cuvis_ai_adaclip.weights import (
    ADACLIP_WEIGHTS,
    CLIP_BACKBONE,
    CLIP_BACKBONE_NAME,
    PLUGIN_NAME,
    WEIGHTS,
)

pytestmark = pytest.mark.unit

HEADS = ["adaclip_mvtec_colondb", "adaclip_visa_clinicdb", "adaclip_all"]


def _heads() -> list[PluginWeightEntry]:
    return [entry for entry in WEIGHTS if entry.selected_by == "weight_name"]


def test_declares_three_heads_and_the_backbone_with_full_pins() -> None:
    assert [entry.name for entry in WEIGHTS] == [*HEADS, "clip_vit_l_14_336"]
    for entry in WEIGHTS:
        assert isinstance(entry, PluginWeightEntry)
        assert len(entry.revision) == 40 and len(entry.sha256) == 64
        assert entry.size_bytes > 0
        assert entry.license == "unspecified (code: MIT)" and entry.license_file is None
        assert entry.used_for and entry.summary and entry.description
    for head in _heads():
        assert head.repo_id == "cubert-gmbh/adaclip"
        assert head.aliases == [head.filename.removesuffix(".pth")]
        assert head.explicit_path_hparams == ["checkpoint_path"]
        assert head.size_bytes == 42_673_907
    assert CLIP_BACKBONE is WEIGHTS[-1]
    assert CLIP_BACKBONE.repo_id == "cubert-gmbh/clip"
    assert CLIP_BACKBONE.selected_by is None and CLIP_BACKBONE.default is False
    assert "Backbone" in CLIP_BACKBONE.used_for


def test_register_called_at_import() -> None:
    assert PLUGIN_NAME == "adaclip"
    for entry in WEIGHTS:
        row = ModelWeights.get(entry.name)
        assert row.plugin == PLUGIN_NAME
        assert row.source == "plugin"
        assert row.entry == entry
    assert cuvis_ai_adaclip.WEIGHTS is WEIGHTS


def test_weight_name_table_is_derived_from_the_declarations() -> None:
    assert set(ADACLIP_WEIGHTS) == {head.aliases[0] for head in _heads()}
    for head in _heads():
        cfg = ADACLIP_WEIGHTS[head.aliases[0]]
        assert cfg["registry_name"] == head.name
        assert cfg["filename"] == head.filename
        assert cfg["description"] == head.description
        assert ModelWeights.get(head.aliases[0]).entry == head, "alias resolves in the registry"


def test_backbone_table_is_derived_from_the_declaration() -> None:
    assert _MIRRORED_CLIP_BACKBONES == {CLIP_BACKBONE_NAME: CLIP_BACKBONE.name}
    assert _MIRRORED_CLIP_BACKBONES == {"ViT-L-14-336": "clip_vit_l_14_336"}


def test_default_head_matches_the_node_default() -> None:
    (default,) = [entry for entry in WEIGHTS if entry.default]
    params = inspect.signature(AdaCLIPDetector.__init__).parameters
    assert params["weight_name"].default == default.aliases[0] == "pretrained_all"
    assert "checkpoint_path" in params


def test_weights_module_is_side_effect_free() -> None:
    """The module declares only: loading it alone must not import torch, core or the plugin."""
    path = Path(weights_mod.__file__)
    code = (
        "import importlib.util, sys\n"
        f"spec = importlib.util.spec_from_file_location('weights_probe', {str(path)!r})\n"
        "mod = importlib.util.module_from_spec(spec)\n"
        "spec.loader.exec_module(mod)\n"
        "assert len(mod.WEIGHTS) == 4 and len(mod.ADACLIP_WEIGHTS) == 3\n"
        "for heavy in ('torch', 'cuvis_ai_core', 'open_clip', 'method', 'cuvis_ai_adaclip'):\n"
        "    assert heavy not in sys.modules, heavy\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=180, check=False
    )
    assert result.returncode == 0, result.stderr
