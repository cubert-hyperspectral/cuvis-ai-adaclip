"""Tests for the weight manager (cuvis_ai_adaclip/weights.py).

The heads are resolved through cuvis-ai-core's weight registry; core is stubbed, so
nothing touches the network or the Hugging Face cache.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from cuvis_ai_core.data.model_weights import ModelWeights, ModelWeightsMissingError

from cuvis_ai_adaclip.weights import (
    ADACLIP_WEIGHTS,
    download_weights,
    get_local_weight_path,
    get_weights_dir,
    list_available_weights,
)

pytestmark = pytest.mark.unit


def _stub_resolve(monkeypatch, result: Path | None = None, error: Exception | None = None):
    calls: list[tuple[str, dict]] = []

    def fake(cls, name, **kwargs):
        calls.append((name, kwargs))
        if error is not None:
            raise error
        return result

    monkeypatch.setattr(ModelWeights, "resolve", classmethod(fake))
    return calls


class TestRegistry:
    """The weight table maps the upstream Drive filenames to core registry names."""

    def test_keys_are_the_upstream_filenames(self) -> None:
        for name, cfg in ADACLIP_WEIGHTS.items():
            assert cfg["filename"] == f"{name}.pth"
            assert cfg["registry_name"].startswith("adaclip_")
            assert cfg["description"]

    def test_list_available_weights(self) -> None:
        assert set(list_available_weights()) == {
            "pretrained_all",
            "pretrained_mvtec_colondb",
            "pretrained_visa_clinicdb",
        }


class TestDownloadWeights:
    """``download_weights`` goes through ``ModelWeights.resolve``."""

    def test_resolves_through_core(self, monkeypatch, tmp_path: Path) -> None:
        calls = _stub_resolve(monkeypatch, result=tmp_path / "pretrained_all.pth")

        assert download_weights("pretrained_all") == tmp_path / "pretrained_all.pth"
        assert calls == [("adaclip_all", {})]

    def test_dataset_heads_map_to_their_registry_names(self, monkeypatch, tmp_path: Path) -> None:
        calls = _stub_resolve(monkeypatch, result=tmp_path / "w.pth")

        download_weights("pretrained_mvtec_colondb")
        download_weights("pretrained_visa_clinicdb")
        assert [c[0] for c in calls] == ["adaclip_mvtec_colondb", "adaclip_visa_clinicdb"]

    def test_force_redownloads_through_core(self, monkeypatch, tmp_path: Path) -> None:
        calls: list[tuple[str, dict]] = []

        def fake_download(cls, name, **kwargs):
            calls.append((name, kwargs))
            return tmp_path / "pretrained_all.pth"

        monkeypatch.setattr(ModelWeights, "download_model", classmethod(fake_download))
        _stub_resolve(monkeypatch, error=AssertionError("resolve must not be used with force"))

        assert download_weights("pretrained_all", force=True) == tmp_path / "pretrained_all.pth"
        assert calls == [("adaclip_all", {"force": True})]

    def test_invalid_name(self, monkeypatch) -> None:
        _stub_resolve(monkeypatch, error=AssertionError("resolve must not be called"))
        with pytest.raises(ValueError, match="Unknown weight"):
            download_weights("invalid_weight_name")

    def test_offline_miss_surfaces_core_error(self, monkeypatch) -> None:
        _stub_resolve(
            monkeypatch,
            error=ModelWeightsMissingError(
                "'adaclip_all' is not in the model cache. Provision it with: "
                "uv run download-model download adaclip_all"
            ),
        )
        with pytest.raises(ModelWeightsMissingError, match="download-model download adaclip_all"):
            download_weights("pretrained_all")


class TestGetLocalWeightPath:
    """``get_local_weight_path`` is a pure cache lookup."""

    def test_returns_cached_path(self, monkeypatch, tmp_path: Path) -> None:
        calls = _stub_resolve(monkeypatch, result=tmp_path / "pretrained_all.pth")

        assert get_local_weight_path("pretrained_all") == tmp_path / "pretrained_all.pth"
        assert calls == [("adaclip_all", {"download": False})]

    def test_returns_none_when_not_cached(self, monkeypatch) -> None:
        _stub_resolve(monkeypatch, error=ModelWeightsMissingError("not cached"))
        assert get_local_weight_path("pretrained_all") is None

    def test_invalid_name_returns_none(self, monkeypatch) -> None:
        calls = _stub_resolve(monkeypatch, result=Path("unused"))
        assert get_local_weight_path("nonexistent_weights") is None
        assert calls == []


class TestSharedModelCacheRedirect:
    """The legacy ``get_weights_dir`` honors the orchestrator's shared model cache env."""

    def test_honors_cuvis_model_cache_dir(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.setenv("CUVIS_MODEL_CACHE_DIR", str(tmp_path / "mc"))
        result = get_weights_dir()
        assert result == tmp_path / "mc" / "adaclip"
        assert result.is_dir()

    def test_falls_back_to_user_cache_when_unset(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("CUVIS_MODEL_CACHE_DIR", raising=False)
        monkeypatch.setattr("cuvis_ai_adaclip.weights.Path.home", lambda: tmp_path / "home")
        result = get_weights_dir()
        assert result == tmp_path / "home" / ".cache" / "cuvis_ai" / "adaclip"
