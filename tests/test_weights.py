"""Tests for weight download manager (cuvis_ai_adaclip/weights.py).

Tests pure path/file logic without network calls or weight downloads.
"""

from __future__ import annotations

import os
from pathlib import Path
from unittest.mock import patch

import pytest

from cuvis_ai_adaclip.weights import (
    _looks_like_html,
    download_weights,
    get_local_weight_path,
    get_weights_dir,
)

pytestmark = pytest.mark.unit


# ============================================================================
# _looks_like_html Tests
# ============================================================================


class TestLooksLikeHtml:
    """Tests for the HTML detection heuristic."""

    def test_detects_doctype(self, tmp_path: Path) -> None:
        """File starting with <!DOCTYPE html should be detected as HTML."""
        f = tmp_path / "bad.pth"
        f.write_bytes(b"<!DOCTYPE html><html><body>Error</body></html>")
        assert _looks_like_html(f) is True

    def test_detects_html_tag(self, tmp_path: Path) -> None:
        """File containing <html tag should be detected as HTML."""
        f = tmp_path / "bad.pth"
        f.write_bytes(b"<html><head><title>Drive</title></head></html>")
        assert _looks_like_html(f) is True

    def test_detects_case_insensitive(self, tmp_path: Path) -> None:
        """Detection should be case-insensitive."""
        f = tmp_path / "bad.pth"
        f.write_bytes(b"<!DOCTYPE HTML><HTML><BODY>Error</BODY></HTML>")
        assert _looks_like_html(f) is True

    def test_rejects_binary(self, tmp_path: Path) -> None:
        """Binary file should not be detected as HTML."""
        f = tmp_path / "good.pth"
        f.write_bytes(os.urandom(256))
        assert _looks_like_html(f) is False

    def test_empty_file(self, tmp_path: Path) -> None:
        """Empty file (size 0) should be flagged as invalid."""
        f = tmp_path / "empty.pth"
        f.write_bytes(b"")
        assert _looks_like_html(f) is True

    def test_nonexistent_file(self, tmp_path: Path) -> None:
        """Non-existent file should be flagged as invalid."""
        f = tmp_path / "missing.pth"
        assert _looks_like_html(f) is True


# ============================================================================
# get_local_weight_path Tests
# ============================================================================


class TestGetLocalWeightPath:
    """Tests for get_local_weight_path."""

    def test_returns_none_when_missing(self, tmp_path: Path, monkeypatch) -> None:
        """Valid weight name but no cached file should return None."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        result = get_local_weight_path("pretrained_all")
        assert result is None

    def test_returns_path_when_cached(self, tmp_path: Path, monkeypatch) -> None:
        """When a cached weight file exists, return its Path."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        cache_dir = get_weights_dir()
        fake_weight = cache_dir / "pretrained_all.pth"
        fake_weight.write_bytes(os.urandom(64))

        result = get_local_weight_path("pretrained_all")
        assert result is not None
        assert result == fake_weight

    def test_invalid_name_returns_none(self) -> None:
        """Invalid weight name should return None."""
        result = get_local_weight_path("nonexistent_weights")
        assert result is None


# ============================================================================
# download_weights Cache Behavior Tests
# ============================================================================


class TestDownloadWeightsCache:
    """Tests for download_weights cache hit/miss behavior."""

    def test_uses_cache_when_file_exists(self, tmp_path: Path, monkeypatch) -> None:
        """Should return cached path without downloading when file exists."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        cache_dir = get_weights_dir()
        fake_weight = cache_dir / "pretrained_all.pth"
        # Write binary content (not HTML) so it passes validation
        fake_weight.write_bytes(os.urandom(256))

        with patch("cuvis_ai_adaclip.weights._download_from_gdrive") as mock_dl:
            result = download_weights("pretrained_all")
            mock_dl.assert_not_called()
            assert result == fake_weight

    def test_force_redownloads(self, tmp_path: Path, monkeypatch) -> None:
        """With force=True, should delete cached file and re-download."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        cache_dir = get_weights_dir()
        fake_weight = cache_dir / "pretrained_all.pth"
        fake_weight.write_bytes(os.urandom(256))

        def mock_download(file_id, target_path):
            target_path.write_bytes(os.urandom(256))

        with patch(
            "cuvis_ai_adaclip.weights._download_from_gdrive", side_effect=mock_download
        ) as mock_dl:
            result = download_weights("pretrained_all", force=True)
            mock_dl.assert_called_once()
            assert result == fake_weight

    def test_redownloads_html_file(self, tmp_path: Path, monkeypatch) -> None:
        """Should re-download if cached file looks like HTML."""
        monkeypatch.setattr(Path, "home", lambda: tmp_path)
        cache_dir = get_weights_dir()
        fake_weight = cache_dir / "pretrained_all.pth"
        fake_weight.write_bytes(b"<!DOCTYPE html><html>Error page</html>")

        def mock_download(file_id, target_path):
            target_path.write_bytes(os.urandom(256))

        with patch(
            "cuvis_ai_adaclip.weights._download_from_gdrive", side_effect=mock_download
        ) as mock_dl:
            result = download_weights("pretrained_all")
            mock_dl.assert_called_once()
            assert result == fake_weight
