"""Tests for AdaCLIPDetector.export_module() and the ONNX export it enables.

Covers the adapter-against-forward parity (the exporter's own gate), the
semantic_fusion=True refusal, and an end-to-end dynamo export plus ONNX
Runtime CPU parity check on the tiny fixture model. onnx, onnxscript and
onnxruntime are optional dependencies (the ``onnx`` extra); every test here
skips cleanly when they are absent.
"""

from __future__ import annotations

import pytest
import torch
from conftest import install_tiny_model, rgb_batch

from cuvis_ai_adaclip import AdaCLIPDetector

onnx = pytest.importorskip("onnx")
onnxscript = pytest.importorskip("onnxscript")
onnxruntime = pytest.importorskip("onnxruntime")


def _build_export_ready_detector(prompt_text: str = "", gaussian_sigma: float = 4.0):
    torch.manual_seed(0)
    detector = AdaCLIPDetector(
        use_half_precision=False,
        enable_warmup=False,
        image_size=64,
        semantic_fusion=False,
        prompt_text=prompt_text,
        gaussian_sigma=gaussian_sigma,
    )
    install_tiny_model(detector, use_hsf=False, k_clusters=2)
    detector.eval()
    return detector


class TestExportModuleRefusal:
    """export_module() only makes sense with semantic_fusion=False."""

    def test_semantic_fusion_true_raises(self) -> None:
        """export_module() refuses when HSF would need tracing sklearn's KMeans."""
        detector = AdaCLIPDetector()
        with pytest.raises(ValueError, match="semantic_fusion=False"):
            detector.export_module()


class TestExportModuleParity:
    """The adapter export_module() returns must match forward() on real samples."""

    def test_adapter_matches_forward_single_image(self) -> None:
        """A single-image batch: every output port matches within tight tolerance."""
        detector = _build_export_ready_detector()
        adapter = detector.export_module()
        adapter.eval()
        rgb = rgb_batch(1)

        out_adapter = adapter(rgb_image=rgb)
        out_forward = detector.forward(rgb_image=rgb)

        for key, forward_value in out_forward.items():
            assert torch.allclose(out_adapter[key], forward_value, atol=1e-5, rtol=1e-4), key

    def test_adapter_matches_forward_batched_with_prompt_and_sigma(self) -> None:
        """A batch of three, a non-empty prompt and a non-default sigma all still match."""
        detector = _build_export_ready_detector(prompt_text="bottle", gaussian_sigma=2.0)
        adapter = detector.export_module()
        adapter.eval()
        rgb = rgb_batch(3)

        out_adapter = adapter(rgb_image=rgb)
        out_forward = detector.forward(rgb_image=rgb)

        for key, forward_value in out_forward.items():
            assert torch.allclose(out_adapter[key], forward_value, atol=1e-5, rtol=1e-4), key


class TestOnnxExportAndRuntimeParity:
    """End-to-end: dynamo export at opset 18, then ONNX Runtime CPU parity."""

    def test_export_and_onnxruntime_match_forward(self, tmp_path) -> None:
        """The exported graph, run on ONNX Runtime CPU, matches the node's forward()."""
        detector = _build_export_ready_detector()
        adapter = detector.export_module()
        adapter.eval()
        rgb = rgb_batch(2)

        out_forward = detector.forward(rgb_image=rgb)
        output_names = list(out_forward.keys())

        onnx_program = torch.onnx.export(
            adapter,
            (rgb,),
            dynamo=True,
            opset_version=18,
            input_names=["rgb_image"],
            output_names=output_names,
        )
        onnx_path = tmp_path / "adaclip_detector.onnx"
        onnx_program.save(str(onnx_path))

        session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        outputs = session.run(None, {"rgb_image": rgb.numpy()})
        onnx_outputs = dict(zip([o.name for o in session.get_outputs()], outputs, strict=True))

        for key, forward_value in out_forward.items():
            onnx_value = torch.from_numpy(onnx_outputs[key])
            assert torch.allclose(onnx_value, forward_value, atol=1e-5, rtol=1e-4), key


class TestOnnxExportSlow:
    """Export with the registry weights, when they are already in the local HF cache."""

    @pytest.mark.slow
    def test_export_with_registry_weights(self, tmp_path) -> None:
        """Skips unless the default AdaCLIP head and CLIP backbone are cached locally."""
        from cuvis_ai_core.data.model_weights import ModelDownloadError, ModelWeights

        from cuvis_ai_adaclip.weights import CLIP_BACKBONE, get_local_weight_path

        try:
            backbone_cached = ModelWeights.resolve(CLIP_BACKBONE.name, download=False)
        except ModelDownloadError:
            backbone_cached = None
        head_cached = get_local_weight_path("pretrained_all")
        if not head_cached or not backbone_cached:
            pytest.skip("AdaCLIP registry weights are not in the local Hugging Face cache")

        detector = AdaCLIPDetector(
            use_half_precision=False,
            enable_warmup=False,
            semantic_fusion=False,
        )
        detector.eval()
        adapter = detector.export_module()
        adapter.eval()
        rgb = rgb_batch(1, size=detector.image_size)

        out_forward = detector.forward(rgb_image=rgb)
        onnx_program = torch.onnx.export(
            adapter,
            (rgb,),
            dynamo=True,
            opset_version=18,
            input_names=["rgb_image"],
            output_names=list(out_forward.keys()),
        )
        onnx_path = tmp_path / "adaclip_detector_full.onnx"
        onnx_program.save(str(onnx_path))

        session = onnxruntime.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        outputs = session.run(None, {"rgb_image": rgb.numpy()})
        onnx_outputs = dict(zip([o.name for o in session.get_outputs()], outputs, strict=True))
        for key, forward_value in out_forward.items():
            onnx_value = torch.from_numpy(onnx_outputs[key])
            # patch_tokens is the one large tensor here (1369 patches x 768 dims x
            # 4 layers): ONNX Runtime's kernels accumulate the backbone's matmuls
            # in a different order than PyTorch, so its float32 error is a little
            # above the tolerance that holds for every other, smaller port.
            atol = 5e-5 if key == "patch_tokens" else 1e-5
            assert torch.allclose(onnx_value, forward_value, atol=atol, rtol=1e-4), key
