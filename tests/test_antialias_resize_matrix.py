"""Tests for AntialiasResizeMatrix, the traceable stand-in for AdaCLIP's
antialiased bicubic preprocessing resize.

ONNX Runtime's antialiased cubic filter does not match torchvision's kernel
closely enough at real image sizes: the acceptance export on real cu3s frames
(1080x1000 RGB from the channel selector, downscaled to 518x518) failed the
exporter's default parity gate on the detector's scores port (max abs diff
about 3e-3 against atol=1e-5). AntialiasResizeMatrix replaces the resize with
two fixed matrices built from torchvision's own kernel, so ONNX Runtime and
torch run the identical matrix multiplication instead of two different
resize implementations.
"""

from __future__ import annotations

import pytest
import torch
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2.functional import resize

from cuvis_ai_adaclip.node._antialias_resize_matrix import AntialiasResizeMatrix


@pytest.mark.parametrize(
    ("h_in", "w_in", "h_out", "w_out"),
    [
        (1080, 1000, 518, 518),  # the acceptance pipeline's real downscale
        (64, 64, 32, 32),  # small downscale, square
        (32, 32, 64, 64),  # upscale
        (600, 500, 448, 448),  # downscale, non-square
    ],
)
def test_antialias_resize_matrix_matches_torchvision(h_in, w_in, h_out, w_out) -> None:
    """The matrix operator matches torchvision's antialiased bicubic resize."""
    torch.manual_seed(0)
    x = torch.rand(2, 3, h_in, w_in)
    expected = resize(x, [h_out, w_out], interpolation=InterpolationMode.BICUBIC, antialias=True)

    op = AntialiasResizeMatrix((h_out, w_out))
    got = op(x)

    assert torch.allclose(got, expected, atol=5e-5, rtol=0)


def test_antialias_resize_matrix_quantized_matches_uint8_resize_closely() -> None:
    """``quantize_range`` brings a uint8-sourced resize within a few levels.

    torchvision's antialiased bicubic resize on an integer dtype is not the
    same linear operator as on a float dtype (verified directly against
    ``_upsample_bicubic2d_aa``, on ordinary random data, independent of
    saturation: the two kernels disagree by multiple intensity levels on
    about 15 percent of pixels even with no clamping involved). No linear
    operator built from float-domain probing can close that gap to less than
    about one level, the floor this test checks for. On a real saturated
    recorded frame, unquantized matrix application drifted by up to 2.3
    levels; with ``quantize_range=(0, 255)`` it drops to about 1.
    """
    torch.manual_seed(0)
    x = (torch.rand(1, 3, 1080, 1000) * 255).round().clamp(0, 255).to(torch.uint8)
    expected = resize(x, [518, 518], interpolation=InterpolationMode.BICUBIC, antialias=True)
    expected = expected.to(torch.float64)

    op = AntialiasResizeMatrix((518, 518), quantize_range=(0.0, 255.0))
    got = op(x.double()).round().clamp(0, 255)

    diff = (expected - got).abs()
    assert diff.max() <= 2.0, f"max diff {diff.max().item()} exceeds the one-to-two-level floor"
    assert diff.mean() < 0.05, f"mean diff {diff.mean().item()} is too far from the floor"


def test_antialias_resize_matrix_rejects_untraced_unbuilt_call() -> None:
    """Calling the operator under tracing before an eager warm-up raises."""
    import unittest.mock as mock

    op = AntialiasResizeMatrix((392, 392))
    with mock.patch("torch.compiler.is_compiling", return_value=True):
        with pytest.raises(RuntimeError, match="not built yet"):
            op(torch.rand(1, 3, 56, 56))


def test_antialias_resize_matrix_rejects_shape_change_after_build() -> None:
    """The operator is fixed to the input size it first saw."""
    op = AntialiasResizeMatrix((56, 56))
    op(torch.rand(1, 3, 56, 56))
    with pytest.raises(RuntimeError, match="fixed to one input size"):
        op(torch.rand(1, 3, 64, 64))
