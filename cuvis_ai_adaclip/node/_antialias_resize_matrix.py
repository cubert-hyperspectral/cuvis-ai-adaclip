"""Antialiased bicubic resize as a fixed linear operator, for ONNX export.

Torchvision's antialiased bicubic resize (``torchvision.transforms.v2.functional.resize``,
what ``AdaCLIPDetector``'s torch preprocessing path uses) lowers to an ONNX
``Resize`` node with antialiasing enabled. ONNX Runtime's antialiased cubic
filter and coefficients do not match torchvision's exactly, so on a real-sized
downscale (for example 1080x1000 to 518x518) the two diverge by a few
thousandths, well above this exporter's parity tolerance. At a fixed input and
output size the resize is linear and separable, so it is exactly
``A_h @ X @ A_w^T`` per channel, with ``A_h`` (H_out x H_in) and ``A_w``
(W_out x W_in) built once from torchvision's own resize. Applying the two
matrices with ``einsum`` lowers to ``MatMul``/``Gemm`` nodes instead of
``Resize``, so ONNX Runtime and torch necessarily agree on the result (it is
the same matrix multiplication in both).

This mirrors ``cuvis_ai_dinomaly.node._antialias_resize_matrix.AntialiasResizeMatrix``
(bilinear there; AdaCLIP's preprocessing uses bicubic, so the interpolation
mode differs). A shared helper in cuvis-ai-core is a natural follow-up once a
second plugin needs the same pattern; for now this is a plugin-local copy.

This operator reconstructs torchvision's **float-dtype** resize kernel
exactly (see the module's own parity test: within 2.2e-5 on a real-sized
downscale). It is not a reconstruction of torchvision's **integer-dtype**
resize: resizing a uint8 tensor is not the same linear operator in
torchvision/ATen as resizing the same values in float32 (confirmed directly
against ``_upsample_bicubic2d_aa`` on ordinary random data, independent of
saturation or rounding convention; see ``AdaCLIPDetector``'s
``integer_preprocess`` hparam, which must be ``False`` for ``export_module()``
so the node's own preprocessing stays in float32 through the resize and this
operator is an exact stand-in for it).
"""

from __future__ import annotations

import torch
from torch import Tensor, nn
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2.functional import resize


def _axis_matrix(size_in: int, size_out: int, other_size: int, *, resize_h: bool) -> Tensor:
    """The ``size_out x size_in`` matrix of the antialiased bicubic resize along one axis.

    Built by feeding the identity (one unit vector per input position) through
    torchvision's own resize and reading the result off as the matrix's columns.
    The probe image's other axis is set to ``other_size`` (a real image size, not
    1): torchvision's antialiased kernel treats a size-1 axis as a different
    boundary case and gives a resize matrix that does not match a real image's
    two-axis resize. The probe is built and resized once in float64 for a
    stable matrix, then cast to the working dtype.
    """
    eye = torch.eye(size_in, dtype=torch.float64)
    if resize_h:
        probe = eye.reshape(size_in, 1, size_in, 1).expand(size_in, 1, size_in, other_size)
        out = resize(
            probe.contiguous(),
            [size_out, other_size],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        matrix = out[:, 0, :, 0].T
    else:
        probe = eye.reshape(size_in, 1, 1, size_in).expand(size_in, 1, other_size, size_in)
        out = resize(
            probe.contiguous(),
            [other_size, size_out],
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        )
        matrix = out[:, 0, 0, :].T
    return matrix.contiguous()


class AntialiasResizeMatrix(nn.Module):
    """Antialiased bicubic resize from a fixed ``(h_in, w_in)`` to a fixed
    ``(h_out, w_out)``, as two buffers applied with matrix multiplication.

    The matrices are built lazily on the first eager call (outside tracing) from
    the actual input size, so the module adapts to whatever size it first sees;
    every later call must carry that same size, matching the exporter's
    fixed-shape contract. Calling it under ``torch.compiler.is_compiling()``
    before the matrices exist raises, since a traced graph cannot run the
    (control-flow-heavy) matrix construction itself.

    The two matrices are applied as two separate passes (width then height,
    torchvision's own pass order for a landscape image), which is exactly
    equivalent to one combined two-axis matrix multiply for a float-dtype
    input: both are the same linear operator, just associated differently.
    This matches torchvision's own float-dtype antialiased resize to float
    precision (see the module's parity test), with no quantization step
    needed between the two passes.
    """

    def __init__(self, size_out: tuple[int, int]) -> None:
        super().__init__()
        self.size_out = (int(size_out[0]), int(size_out[1]))
        self._built = False
        self._size_in: tuple[int, int] | None = None

    def _build(self, size_in: tuple[int, int], dtype: torch.dtype, device: torch.device) -> None:
        h_in, w_in = size_in
        h_out, w_out = self.size_out
        a_h = _axis_matrix(h_in, h_out, w_in, resize_h=True).to(dtype=dtype, device=device)
        a_w = _axis_matrix(w_in, w_out, h_in, resize_h=False).to(dtype=dtype, device=device)
        self.register_buffer("a_h", a_h, persistent=False)
        self.register_buffer("a_w", a_w, persistent=False)
        self._size_in = size_in
        self._built = True

    def forward(self, x: Tensor) -> Tensor:
        """Resize ``x`` (``[B, C, H, W]``) to ``size_out``."""
        size_in = (int(x.shape[-2]), int(x.shape[-1]))
        if not self._built:
            if torch.compiler.is_compiling():
                raise RuntimeError(
                    "AntialiasResizeMatrix: the resize matrices are not built yet. Run one "
                    "eager forward (outside tracing) on a real input before exporting, so "
                    "the matrices exist when the module is traced."
                )
            self._build(size_in, x.dtype, x.device)
        elif size_in != self._size_in:
            raise RuntimeError(
                f"AntialiasResizeMatrix: built for input size {self._size_in}, called with "
                f"{size_in}. The module is fixed to one input size once built."
            )
        width_resized = torch.einsum("bchw,pw->bchp", x, self.a_w)
        return torch.einsum("oh,bchw->bcow", self.a_h, width_resized)
