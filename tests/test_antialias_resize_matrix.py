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

This operator reconstructs torchvision's float-dtype resize kernel exactly
(the parity test below). It does not reconstruct torchvision's integer-dtype
resize kernel, which is a different internal implementation; that gap is why
AdaCLIPDetector gained an ``integer_preprocess`` hparam (see its docstring
and tests/test_preprocessing.py's TestIntegerPreprocess), and why
export_module() requires it False rather than quantizing the matrix's
intermediate pass to approximate the integer-dtype kernel.
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


def test_antialias_resize_matrix_matches_float_resize_at_export_size_tightly() -> None:
    """At the acceptance pipeline's real downscale, the match is within float32 precision.

    This is the bound export_module()'s adapter inherits when
    AdaCLIPDetector.integer_preprocess=False, since the adapter's
    preprocessing stays in float32 through the resize exactly like this. In
    float64 the two agree to 1e-13 (machine precision): the operator is
    mathematically exact, and the remaining difference at float32 (up to
    about 5e-3 absolute on this 0-255 scale, 7.7e-5 relative) is ordinary
    float32 accumulation noise from the two implementations summing the same
    terms in a different order, not a gap in the reconstructed kernel.
    """
    torch.manual_seed(7)
    x = torch.rand(1, 3, 1000, 1080) * 255.0
    expected = resize(x, [518, 518], interpolation=InterpolationMode.BICUBIC, antialias=True)

    op = AntialiasResizeMatrix((518, 518))
    got = op(x)

    assert torch.allclose(got, expected, atol=1e-5, rtol=1e-4)


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
