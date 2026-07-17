"""Coverage for the shared-model-cache redirection and the ``checkpoint_path`` override.

Exercises the CLIP-backbone cache-dir resolution in ``AdaCLIPModel._init_model`` and the
``checkpoint_path`` short-circuit in ``AdaCLIPDetector._ensure_model_loaded`` with the heavy
model construction / weight download stubbed out (no network, no real CLIP backbone).
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from torch import nn

from cuvis_ai_adaclip import adaclip_upstream
from cuvis_ai_adaclip.adaclip_upstream import AdaCLIPModel
from cuvis_ai_adaclip.node import adaclip_node
from cuvis_ai_adaclip.node.adaclip_node import AdaCLIPDetector

pytestmark = pytest.mark.unit


def _patch_clip_builders(monkeypatch, captured: dict) -> None:
    """Stub the upstream OpenCLIP builders so ``_init_model`` runs offline."""

    def fake_create_model_and_transforms(model_name, img_size, pretrained, device, cache_dir):
        captured["cache_dir"] = cache_dir
        preprocess = MagicMock()
        preprocess.transforms = [MagicMock(), MagicMock()]
        return MagicMock(name="clip_model"), MagicMock(), preprocess

    monkeypatch.setattr(
        adaclip_upstream, "create_model_and_transforms", fake_create_model_and_transforms
    )
    monkeypatch.setattr(
        adaclip_upstream,
        "get_model_config",
        lambda backbone: {"embed_dim": 8, "vision_cfg": {"width": 16}},
    )
    # AdaCLIP is wrapped as a submodule; a real nn.Module keeps ._clip_model.to() valid.
    monkeypatch.setattr(adaclip_upstream, "AdaCLIP", lambda **kwargs: nn.Identity())


class TestClipBackboneCacheDir:
    """``AdaCLIPModel`` routes the OpenCLIP backbone through the shared model cache."""

    def test_uses_shared_cache_when_env_set(self, tmp_path: Path, monkeypatch) -> None:
        """$CUVIS_MODEL_CACHE_DIR redirects the CLIP backbone under ``<cache>/clip``."""
        captured: dict = {}
        monkeypatch.setenv("CUVIS_MODEL_CACHE_DIR", str(tmp_path / "mc"))
        _patch_clip_builders(monkeypatch, captured)

        model = AdaCLIPModel(backbone="ViT-L-14-336", image_size=32, device="cpu")
        model._init_model()

        assert captured["cache_dir"] == str(tmp_path / "mc" / "clip")

    def test_cache_dir_none_when_env_unset(self, monkeypatch) -> None:
        """Without the env var the CLIP backbone falls back to OpenCLIP's default cache."""
        captured: dict = {}
        monkeypatch.delenv("CUVIS_MODEL_CACHE_DIR", raising=False)
        _patch_clip_builders(monkeypatch, captured)

        model = AdaCLIPModel(backbone="ViT-L-14-336", image_size=32, device="cpu")
        model._init_model()

        assert captured["cache_dir"] is None


class _StubAdaCLIPModel(nn.Module):
    """Minimal stand-in for ``AdaCLIPModel`` to exercise the node's load path."""

    def load_weights(self, weight_path) -> None:
        self.loaded_path = str(weight_path)

    def get_preprocess(self):
        return MagicMock()


class TestCheckpointPathOverride:
    """``_ensure_model_loaded`` prefers an explicit ``checkpoint_path`` over a download."""

    def test_checkpoint_path_skips_download(self, monkeypatch) -> None:
        """A provided ``checkpoint_path`` loads the local file and never downloads."""
        download = MagicMock()
        monkeypatch.setattr(adaclip_node, "download_weights", download)
        stub = _StubAdaCLIPModel()
        monkeypatch.setattr(adaclip_node, "AdaCLIPModel", lambda **kwargs: stub)

        detector = AdaCLIPDetector(checkpoint_path="/provisioned/adaclip.pth")
        detector._ensure_model_loaded()

        download.assert_not_called()
        assert stub.loaded_path == "/provisioned/adaclip.pth"

    def test_downloads_when_no_checkpoint_path(self, tmp_path: Path, monkeypatch) -> None:
        """Without ``checkpoint_path`` the node falls back to ``download_weights``."""
        weight_file = tmp_path / "pretrained_all.pth"
        download = MagicMock(return_value=weight_file)
        monkeypatch.setattr(adaclip_node, "download_weights", download)
        stub = _StubAdaCLIPModel()
        monkeypatch.setattr(adaclip_node, "AdaCLIPModel", lambda **kwargs: stub)

        detector = AdaCLIPDetector()
        detector._ensure_model_loaded()

        download.assert_called_once_with("pretrained_all")
        assert stub.loaded_path == str(weight_file)
