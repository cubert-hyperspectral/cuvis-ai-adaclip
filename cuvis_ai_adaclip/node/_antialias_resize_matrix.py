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

AdaCLIP's preprocessing resizes a uint8 tensor (to match PIL's byte-level
behavior), and torchvision's antialiased bicubic resize on an integer dtype
is not the same linear operator as on a float dtype (confirmed directly
against ``_upsample_bicubic2d_aa``, on ordinary random data, independent of
saturation). ``quantize_range`` on :class:`AntialiasResizeMatrix` narrows that
gap from several intensity levels to about one, the practical floor for
reconstructing an integer-dtype kernel from float-domain probing; see its
docstring. A real recorded reflectance frame (0 to about 18700, saturating at
the 255 clamp before resize) needs ``quantize_range`` to bring the detector's
``scores`` port within low 1e-3 of the node's own forward(), down from the
unmitigated 2.998e-3; it does not reach the exporter's default 1e-5 on that
frame. On the tiny random fixtures this repository tests against by default,
the gap stays below the export tests' tolerance even without
``quantize_range``, which is why the default export tests pass while the
acceptance run on real cu3s frames needs a wider tolerance (documented on the
acceptance test) until a native ONNX antialiased-resize operator gives true
bit parity.
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

    ``quantize_range`` reproduces an integer dtype's rounding between the two
    separable passes (width then height, torchvision's own pass order for a
    landscape image: the axis with the larger downscale factor runs first).
    Resizing an integer-dtype tensor is not one linear operator in
    torchvision/ATen: measured directly (``_upsample_bicubic2d_aa``), its
    integer-dtype kernel and its float-dtype kernel disagree by a few
    intensity levels on ordinary random data, independent of saturation, and
    the gap is not a rounding-convention mismatch (round-half-up and
    round-half-to-even were both tried; neither closes it) or a probing
    artifact (the probed matrix's rows sum to 1 at float64 machine precision).
    It is a different internal implementation per dtype. Quantizing between
    the two passes the way an integer dtype would reduces a multi-level
    drift (observed up to 2.3 on a 0-255 scale, from one combined matrix
    multiply, on a real recorded reflectance frame) to the one-level
    rounding quantum that is the practical floor for any float-domain
    reconstruction of that kernel; it does not eliminate the floor. Leave it
    ``None`` (the default) for a resize on continuous, unclamped data, where
    the single linear operator already matches torchvision's own float-dtype
    resize closely (see the module's own parity test).
    """

    def __init__(
        self, size_out: tuple[int, int], quantize_range: tuple[float, float] | None = None
    ) -> None:
        super().__init__()
        self.size_out = (int(size_out[0]), int(size_out[1]))
        self.quantize_range = quantize_range
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
        """Resize ``x`` (``[B, C, H, W]``) to ``size_out``.

        Applied as two separate passes (width then height, matching
        torchvision's own pass order for this orientation), with the
        intermediate result rounded and clamped to ``quantize_range`` when
        set; see the class docstring for why.
        """
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
        if self.quantize_range is not None:
            low, high = self.quantize_range
            width_resized = width_resized.round().clamp(low, high)
        return torch.einsum("oh,bchw->bcow", self.a_h, width_resized)
