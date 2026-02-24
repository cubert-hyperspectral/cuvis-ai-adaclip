"""Tests for Gaussian smoothing and torch-based preprocessing.

Tests pure tensor operations on CPU — no GPU or weight downloads needed.
"""

from __future__ import annotations

import pytest
import torch

from cuvis_ai_adaclip import AdaCLIPDetector, AdaCLIPModel

pytestmark = pytest.mark.unit


# ============================================================================
# Gaussian Smoothing Tests (AdaCLIPModel._gaussian_smooth_2d)
# ============================================================================


class TestGaussianSmooth:
    """Tests for AdaCLIPModel._gaussian_smooth_2d (static method)."""

    def test_preserves_shape(self) -> None:
        """Output shape should match input shape [B, H, W]."""
        x = torch.rand(2, 32, 32)
        result = AdaCLIPModel._gaussian_smooth_2d(x, sigma=4.0)
        assert result.shape == x.shape

    def test_reduces_peaks(self) -> None:
        """Smoothing should reduce the value of a single bright pixel."""
        x = torch.zeros(1, 32, 32)
        x[0, 16, 16] = 1.0  # Single bright pixel
        result = AdaCLIPModel._gaussian_smooth_2d(x, sigma=4.0)
        # Peak should be significantly reduced
        assert result[0, 16, 16].item() < 0.5

    def test_energy_preservation(self) -> None:
        """Total energy (sum) should be approximately preserved on large images."""
        x = torch.rand(1, 128, 128)
        result = AdaCLIPModel._gaussian_smooth_2d(x, sigma=2.0)
        # Boundary effects are small on large images
        assert abs(result.sum().item() - x.sum().item()) / x.sum().item() < 0.05

    def test_small_sigma_near_identity(self) -> None:
        """Very small sigma should produce output close to input."""
        x = torch.rand(1, 16, 16)
        result = AdaCLIPModel._gaussian_smooth_2d(x, sigma=0.1)
        assert torch.allclose(result, x, atol=1e-3)

    def test_larger_sigma_smoother(self) -> None:
        """Larger sigma should produce smoother (lower variance) output."""
        x = torch.rand(2, 32, 32)
        result_small = AdaCLIPModel._gaussian_smooth_2d(x, sigma=1.0)
        result_large = AdaCLIPModel._gaussian_smooth_2d(x, sigma=4.0)
        # Larger sigma → lower pixel variance (smoother)
        assert result_large.var().item() < result_small.var().item()

    def test_uniform_input_unchanged(self) -> None:
        """A uniform tensor should be unchanged by smoothing (center pixels)."""
        x = torch.full((1, 128, 128), 0.5)
        result = AdaCLIPModel._gaussian_smooth_2d(x, sigma=4.0)
        # Center region should be unchanged (boundary pixels may differ due to padding)
        center = result[0, 32:96, 32:96]
        assert torch.allclose(center, torch.full_like(center, 0.5), atol=1e-5)


# ============================================================================
# Torch Preprocessing Tests (AdaCLIPDetector._preprocess_rgb_torch_pil_match)
# ============================================================================


class TestPreprocessRgbTorch:
    """Tests for AdaCLIPDetector._preprocess_rgb_torch_pil_match.

    The method only uses self.image_size, self.enable_gradients,
    self._clip_mean, and self._clip_std — all set in __init__.
    No model loading is needed.
    """

    @pytest.fixture
    def detector(self) -> AdaCLIPDetector:
        """Create a detector without loading the model."""
        return AdaCLIPDetector(image_size=518, use_torch_preprocess=True)

    def test_output_shape(self, detector: AdaCLIPDetector) -> None:
        """Input [B, H, W, 3] should produce [B, 3, S, S] output."""
        rgb = torch.rand(1, 64, 64, 3)
        result = detector._preprocess_rgb_torch_pil_match(rgb)
        assert result.shape == (1, 3, 518, 518)

    def test_output_is_normalized(self, detector: AdaCLIPDetector) -> None:
        """Output should be CLIP-normalized (not in 0-255 range)."""
        rgb = torch.rand(1, 64, 64, 3)
        result = detector._preprocess_rgb_torch_pil_match(rgb)
        # CLIP normalization subtracts mean ~0.48 and divides by std ~0.27
        # so values should span roughly [-2, 3], not [0, 255]
        assert result.max().item() < 10.0
        assert result.min().item() > -10.0

    def test_handles_01_range(self, detector: AdaCLIPDetector) -> None:
        """Input in [0, 1] range should work without error."""
        rgb = torch.rand(1, 32, 32, 3)  # Already in [0, 1]
        result = detector._preprocess_rgb_torch_pil_match(rgb)
        assert result.shape == (1, 3, 518, 518)
        assert result.dtype == torch.float32

    def test_handles_0255_range(self, detector: AdaCLIPDetector) -> None:
        """Input in [0, 255] range should work without error."""
        rgb = torch.randint(0, 256, (1, 32, 32, 3), dtype=torch.float32)
        result = detector._preprocess_rgb_torch_pil_match(rgb)
        assert result.shape == (1, 3, 518, 518)
        assert result.dtype == torch.float32

    def test_batch_dimension(self, detector: AdaCLIPDetector) -> None:
        """Batch size > 1 should work correctly."""
        rgb = torch.rand(4, 32, 32, 3)
        result = detector._preprocess_rgb_torch_pil_match(rgb)
        assert result.shape == (4, 3, 518, 518)

    def test_bhwc_to_bchw_permutation(self, detector: AdaCLIPDetector) -> None:
        """Output should have channels in dim 1 (BCHW), not dim 3 (BHWC)."""
        rgb = torch.rand(1, 64, 64, 3)
        result = detector._preprocess_rgb_torch_pil_match(rgb)
        # dim 1 should be 3 (channels), dim 2 and 3 should be image_size
        assert result.shape[1] == 3
        assert result.shape[2] == 518
        assert result.shape[3] == 518

    def test_different_image_sizes(self) -> None:
        """Different image_size parameter should resize accordingly."""
        detector = AdaCLIPDetector(image_size=224, use_torch_preprocess=True)
        rgb = torch.rand(1, 64, 64, 3)
        result = detector._preprocess_rgb_torch_pil_match(rgb)
        assert result.shape == (1, 3, 224, 224)

    def test_gradients_mode_preserves_grad(self) -> None:
        """With enable_gradients=True, gradients should flow through preprocessing."""
        detector = AdaCLIPDetector(image_size=64, use_torch_preprocess=True, enable_gradients=True)
        rgb = torch.rand(1, 32, 32, 3, requires_grad=True)
        result = detector._preprocess_rgb_torch_pil_match(rgb)
        # Should be able to compute gradients
        loss = result.sum()
        loss.backward()
        assert rgb.grad is not None
