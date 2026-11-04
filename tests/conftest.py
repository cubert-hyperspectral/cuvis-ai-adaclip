"""Shared fixtures for the cuvis_ai_adaclip test suite.

The ``tiny_adaclip_model`` fixture builds a real (but tiny) AdaCLIP model with
randomly initialized weights, bypassing the upstream OpenAI download: the spike
for the ONNX export work found that ``method.custom_clip.create_model`` always
fetches pretrained weights, so a CLIP backbone for tests is built directly from
a tiny config instead.
"""

from __future__ import annotations

import torch

from method.adaclip import AdaCLIP
from method.clip_model import CLIP

# A tiny CLIP config: few layers, small width, small image, so a forward pass
# (and an export) takes seconds rather than minutes and needs no checkpoint.
TINY_IMAGE_SIZE = 64
TINY_PATCH_SIZE = 16
TINY_EMBED_DIM = 32
TINY_TEXT_WIDTH = 32
TINY_VISUAL_WIDTH = 32
TINY_OUTPUT_LAYERS = [1, 2]


def build_tiny_clip() -> CLIP:
    """Build a randomly initialized CLIP model at test scale."""
    vision_cfg = {
        "image_size": TINY_IMAGE_SIZE,
        "layers": max(TINY_OUTPUT_LAYERS),
        "width": TINY_VISUAL_WIDTH,
        "patch_size": TINY_PATCH_SIZE,
        "head_width": TINY_VISUAL_WIDTH,
    }
    text_cfg = {
        "context_length": 77,
        "vocab_size": 49408,
        "width": TINY_TEXT_WIDTH,
        "heads": 4,
        "layers": 2,
    }
    return CLIP(embed_dim=TINY_EMBED_DIM, vision_cfg=vision_cfg, text_cfg=text_cfg)


def build_tiny_adaclip(
    use_hsf: bool = True,
    prompting_depth: int = 2,
    prompting_length: int = 3,
    k_clusters: int = 2,
) -> AdaCLIP:
    """Build a randomly initialized AdaCLIP model (the upstream nn.Module) at test scale."""
    clip_model = build_tiny_clip()
    return AdaCLIP(
        freeze_clip=clip_model,
        text_channel=TINY_EMBED_DIM,
        visual_channel=TINY_VISUAL_WIDTH,
        prompting_length=prompting_length,
        prompting_depth=prompting_depth,
        prompting_branch="VL",
        prompting_type="SD",
        use_hsf=use_hsf,
        k_clusters=k_clusters,
        output_layers=TINY_OUTPUT_LAYERS,
        device="cpu",
        image_size=TINY_IMAGE_SIZE,
    )


def install_tiny_model(detector, use_hsf: bool, k_clusters: int = 2) -> None:
    """Attach a tiny, randomly initialized AdaCLIP model to a detector node.

    Bypasses ``AdaCLIPDetector._ensure_model_loaded`` (which downloads the real
    backbone and prompt head) and ``AdaCLIPModel._init_model`` (which always
    fetches OpenAI weights), so the detector runs end to end on random weights.
    """
    from cuvis_ai_adaclip.adaclip_upstream import AdaCLIPModel

    model = AdaCLIPModel(
        backbone=detector.backbone,
        image_size=TINY_IMAGE_SIZE,
        prompting_depth=detector.prompting_depth,
        prompting_length=detector.prompting_length,
        use_hsf=use_hsf,
        k_clusters=k_clusters,
    )
    model._clip_model = build_tiny_adaclip(
        use_hsf=use_hsf,
        prompting_depth=detector.prompting_depth,
        prompting_length=detector.prompting_length,
        k_clusters=k_clusters,
    )
    model._initialized = True
    model.eval()
    for param in model.parameters():
        param.requires_grad_(False)

    detector.add_module("adaclip_model", model)
    detector._adaclip_model = model
    # use_torch_preprocess=True (the node default) never reads self._preprocess;
    # it only needs to be non-None to pass the "model initialized" guard.
    detector._preprocess = model.get_preprocess
    detector._initialized_flag.fill_(True)
    detector.image_size = TINY_IMAGE_SIZE
    detector.eval()


def rgb_batch(batch_size: int = 1, size: int = TINY_IMAGE_SIZE) -> torch.Tensor:
    """A random float32 RGB batch in [0, 1], BHWC, matching the detector's input port."""
    return torch.rand(batch_size, size, size, 3, dtype=torch.float32)
