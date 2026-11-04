"""AdaCLIP detector node for the cuvis_ai_adaclip plugin.

This node is a **self-contained** AdaCLIP integration that:

- Uses the upstream AdaCLIP implementation from this repository
  (via :mod:`cuvis_ai_adaclip.adaclip_upstream`).
- Does **not** import any adaclip-related code from the main ``cuvis_ai``
  package (only generic node/port/typing utilities).
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from typing import Any

import numpy as np
import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.execution import Context
from cuvis_ai_schemas.pipeline import PortSpec
from PIL import Image
from torchvision.transforms import Compose

try:
    from torchvision.transforms import InterpolationMode
    from torchvision.transforms.v2 import functional as F_v2

    HAS_TORCHVISION_V2 = True
except ImportError:
    HAS_TORCHVISION_V2 = False
    InterpolationMode = None
    F_v2 = None

from loguru import logger

from cuvis_ai_adaclip.adaclip_upstream import (
    OPENAI_DATASET_MEAN,
    OPENAI_DATASET_STD,
    AdaCLIPModel,
    download_weights,
)
from cuvis_ai_adaclip.node._antialias_resize_matrix import AntialiasResizeMatrix


class _AdaCLIPExportModule(torch.nn.Module):
    """Traceable adapter for :meth:`AdaCLIPDetector.export_module`.

    Replaces the parts of the eager forward pass a traced graph cannot carry:
    the PIL/branching preprocessing becomes tensor-only resize and normalize,
    the antialiased bicubic resize becomes :class:`AntialiasResizeMatrix` (two
    buffers applied with ``einsum`` instead of an ONNX ``Resize`` node with
    antialiasing, which ONNX Runtime does not evaluate identically to
    torchvision at real image sizes, and which TensorRT does not accept at
    all), the per-call BPE tokenization becomes two registered buffers of
    token ids (precomputed once, outside tracing, for the node's
    ``prompt_text``), and the AdaCLIP model itself runs with ``use_hsf``
    forced off so the k-means fusion in
    :class:`cuvis_ai_adaclip.node.semantic_fusion.AdaCLIPSemanticFusion` stays
    outside the graph. The text encoder still runs inside the graph (the
    prompt depth and prompter layers are dynamic), only the tokenization is
    baked.

    The resize runs with ``quantize_range=(0, 255)`` (see
    :class:`AntialiasResizeMatrix`), matching the uint8 rounding the node's
    own preprocessing applies between the two resize passes; this narrows the
    gap between the matrix and the node's own forward() from several
    intensity levels to about one on real, saturated image content. A
    real recorded frame still disagrees with forward() by a few parts in a
    thousand on the ``scores`` port after the full backbone: see
    ``tests/test_antialias_resize_matrix.py`` for the measured floor.

    The resize matrices are not known until the first real input is seen, so
    :meth:`AdaCLIPDetector.export_module` does not need to warm them up itself:
    the exporter's own eager parity check runs this adapter on a real example
    before tracing starts, which builds them.
    """

    def __init__(self, node: AdaCLIPDetector) -> None:
        super().__init__()
        assert node._adaclip_model is not None
        self.clip_model = node._adaclip_model._clip_model
        self.image_size = node.image_size
        self.gaussian_sigma = node.gaussian_sigma
        self.resize = AntialiasResizeMatrix(
            (node.image_size, node.image_size), quantize_range=(0.0, 255.0)
        )

        text_layer = self.clip_model.text_embedding_layer
        prompt_text = node.prompt_text.replace("-", " ")
        token_id_tensors = []
        for state in text_layer.prompt_state:
            prompted_state = [phrase.format(prompt_text) for phrase in state]
            prompted_sentences = [
                template.format(s)
                for s in prompted_state
                for template in text_layer.prompt_templates
            ]
            token_id_tensors.append(text_layer.tokenize(prompted_sentences, context_length=77))

        # One buffer per prompt state (normal, abnormal); their token counts differ
        # (len(prompt_normal) != len(prompt_abnormal) times the template count), so
        # they are kept separate rather than stacked into one regular tensor.
        self.register_buffer("_normal_token_ids", token_id_tensors[0], persistent=False)
        self.register_buffer("_abnormal_token_ids", token_id_tensors[1], persistent=False)

        mean = torch.tensor(OPENAI_DATASET_MEAN, dtype=torch.float32).view(1, 3, 1, 1)
        std = torch.tensor(OPENAI_DATASET_STD, dtype=torch.float32).view(1, 3, 1, 1)
        self.register_buffer("_clip_mean", mean, persistent=False)
        self.register_buffer("_clip_std", std, persistent=False)

    def _preprocess(self, rgb_bhwc: torch.Tensor) -> torch.Tensor:
        """Tensor-only resize and normalize, matching the node's inference preprocessing.

        Mirrors ``AdaCLIPDetector._preprocess_rgb_torch_pil_match`` at
        ``enable_gradients=False``: antialiased bicubic resize on a uint8
        tensor, then normalize. The resize itself runs through
        :class:`AntialiasResizeMatrix` rather than
        ``torchvision.transforms.v2.functional.resize`` directly, so the
        exported graph carries matrix multiplications instead of an
        antialiased ``Resize`` node; its output is already exactly
        ``(image_size, image_size)``, so no center crop is needed. The export
        graph never needs gradients through preprocessing, so the traceable
        uint8 path is the only one implemented.
        """
        x = rgb_bhwc
        x = torch.where(x.amax(dim=(1, 2, 3), keepdim=True) <= 1.0, x * 255.0, x)
        x = x.round().clamp(0, 255).to(torch.uint8)
        x = x.permute(0, 3, 1, 2)  # BHWC -> BCHW
        # torchvision resizes this uint8 tensor directly, so the bicubic kernel's
        # ringing (negative lobes near a sharp edge or, as here, a saturated
        # region) saturates to the dtype's [0, 255] range as part of the resize.
        # AntialiasResizeMatrix is a plain float linear operator with no such
        # saturation, so its raw output can overshoot past 255 on real, highly
        # saturated data (observed up to about 257 on a recorded reflectance
        # frame); round and clamp its output the same way the uint8 dtype would.
        x = self.resize(x.to(torch.float32)).round().clamp(0, 255)
        x = x / 255.0
        return (x - self._clip_mean) / self._clip_std

    def _encode_text_state(self, token_ids: torch.Tensor) -> torch.Tensor:
        """Reproduce ``TextEmbebddingLayer.encode_text`` for one prompt state."""
        class_embeddings = self.clip_model.encode_text(token_ids)
        class_embeddings = class_embeddings / class_embeddings.norm(dim=-1, keepdim=True)
        class_embedding = class_embeddings.mean(dim=0)
        return class_embedding / class_embedding.norm()

    def forward(self, rgb_image: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run the exportable span: preprocessing, CLIP encode, plain similarity.

        Parameters
        ----------
        rgb_image :
            RGB image tensor ``[B, H, W, 3]``, matching the node's input port.
        """
        _, h, w, _ = rgb_image.shape
        img_tensor = self._preprocess(rgb_image)

        image_feature, patch_tokens, text_feature = self.clip_model.extract_feat(img_tensor, [""])

        normal_feature = self._encode_text_state(self._normal_token_ids)
        abnormal_feature = self._encode_text_state(self._abnormal_token_ids)
        text_feature = torch.stack([normal_feature, abnormal_feature], dim=1).unsqueeze(0)
        text_feature = torch.nn.functional.normalize(text_feature, dim=1)

        anomaly_map, anomaly_score = self.clip_model.visual_text_similarity(
            image_feature, patch_tokens, text_feature, aggregation=True
        )
        anomaly_map = anomaly_map.squeeze(1)  # [B, H', W'], matching AdaCLIP.forward
        if self.gaussian_sigma > 0:
            anomaly_map = AdaCLIPModel._gaussian_smooth_2d(anomaly_map, self.gaussian_sigma)
        anomaly_map = torch.nn.functional.interpolate(
            anomaly_map.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False
        )
        scores = anomaly_map.squeeze(1).unsqueeze(-1)  # [B, H, W, 1]

        return {
            "scores": scores,
            "anomaly_score": anomaly_score,
            "per_layer_scores": torch.zeros(1, 1, 1, 1, device=scores.device),
            "image_score_2ch": torch.stack([1 - anomaly_score, anomaly_score], dim=1),
            "patch_tokens": torch.stack(patch_tokens, dim=1),
            "image_feature": image_feature,
            "text_feature": text_feature,
        }


class AdaCLIPDetector(Node):
    """AdaCLIP zero-shot anomaly detector node (plugin version).

    This node applies AdaCLIP for anomaly detection on RGB images.
    It takes RGB images (either uint8 or float32) and outputs pixel-level
    anomaly scores and image-level anomaly scores.

    The node uses lazy loading to avoid initializing the model until
    it's actually needed (first forward pass). The underlying AdaCLIP model
    is registered as a submodule so that ``state_dict()`` captures its weights.

    Fine-tuning support
    -------------------
    When ``pipeline.unfreeze_nodes_by_name(["adaclip"])`` is called, only the
    lightweight adapter layers (~10 M params) become trainable while the CLIP
    backbone (~890 MB) stays frozen.  Set ``enable_gradients=True`` and
    ``use_half_precision=False`` for any gradient-based training.

    When ``training_aggregation`` is False the upstream model returns per-layer
    anomaly maps (a list) which matches the paper's per-layer loss computation.
    At inference time ``aggregation=True`` is always used regardless of the flag.
    """

    _category = NodeCategory.MODEL
    _tags = frozenset(
        {
            NodeTag.RGB,
            NodeTag.IMAGE,
            NodeTag.ANOMALY,
            NodeTag.MASK,
            NodeTag.INFERENCE,
            NodeTag.LEARNABLE,
            NodeTag.TORCH,
            # Exports with semantic_fusion=False: the k-means fusion runs outside the
            # graph in AdaCLIPSemanticFusion, since sklearn's KMeans is not traceable.
            NodeTag.ONNX,
        }
    )
    ADAPTER_MODULE_NAMES: tuple[str, ...] = (
        "text_prompter",
        "visual_prompter",
        "patch_token_layer",
        "cls_token_layer",
        "dynamic_visual_prompt_generator",
        "dynamic_text_prompt_generator",
    )

    INPUT_SPECS = {
        "rgb_image": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 3),
            description="RGB image [B, H, W, 3] in float32 (0-1 or 0-255 range)",
        ),
    }

    OUTPUT_SPECS = {
        "scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, 1),
            description="Pixel-level anomaly scores [B, H, W, 1]",
        ),
        "anomaly_score": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Image-level anomaly score [B]",
        ),
        "per_layer_scores": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description=(
                "Per-layer softmaxed anomaly maps stacked as [B, num_layers*2, H, W]. "
                "Only populated during training when training_aggregation=False. "
                "Empty (1×1) tensor otherwise."
            ),
            optional=True,
        ),
        "image_score_2ch": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            description=(
                "Image-level 2-channel score [B, 2] (softmaxed [normal, anomaly]). "
                "Only populated during training when training_aggregation=False."
            ),
            optional=True,
        ),
        "patch_tokens": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description=(
                "Projected patch tokens for every output layer, stacked as "
                "[B, num_layers, N, D] (N patches, D the text embedding width). "
                "At image_size=518 with the default backbone this is about 17 MB "
                "per frame in float32. Only populated when semantic_fusion=False; "
                "empty (1x1x1x1) tensor otherwise."
            ),
            optional=True,
        ),
        "image_feature": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            description=(
                "Pooled and projected image feature [B, D], before the k-means "
                "semantic fusion. Only populated when semantic_fusion=False; "
                "empty (1x1) tensor otherwise."
            ),
            optional=True,
        ),
        "text_feature": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1),
            description=(
                "Text embedding [1, D, 2] (normal, anomaly) shared across the image "
                "batch, in the shape AdaCLIP's visual_text_similarity expects. Only "
                "populated when semantic_fusion=False; empty (1x1x1) tensor otherwise."
            ),
            optional=True,
        ),
    }

    def __init__(
        self,
        weight_name: str = "pretrained_all",
        checkpoint_path: str | None = None,
        backbone: str = "ViT-L-14-336",
        prompt_text: str = "",
        image_size: int = 518,
        prompting_depth: int = 4,
        prompting_length: int = 5,
        gaussian_sigma: float = 4.0,
        use_half_precision: bool = True,
        enable_warmup: bool = True,
        enable_gradients: bool = False,
        use_torch_preprocess: bool = True,
        training_aggregation: bool = True,
        semantic_fusion: bool = True,
        k_clusters: int = 20,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            weight_name=weight_name,
            checkpoint_path=checkpoint_path,
            backbone=backbone,
            prompt_text=prompt_text,
            image_size=image_size,
            prompting_depth=prompting_depth,
            prompting_length=prompting_length,
            gaussian_sigma=gaussian_sigma,
            use_half_precision=use_half_precision,
            enable_warmup=enable_warmup,
            enable_gradients=enable_gradients,
            use_torch_preprocess=use_torch_preprocess,
            training_aggregation=training_aggregation,
            semantic_fusion=semantic_fusion,
            k_clusters=k_clusters,
            **kwargs,
        )

        self.weight_name = weight_name
        self.checkpoint_path = checkpoint_path
        self.backbone = backbone
        self.prompt_text = prompt_text
        self.image_size = image_size
        self.prompting_depth = prompting_depth
        self.prompting_length = prompting_length
        self.gaussian_sigma = gaussian_sigma
        self.use_half_precision = use_half_precision
        self.enable_warmup = enable_warmup
        self.enable_gradients = enable_gradients
        self.use_torch_preprocess = use_torch_preprocess
        self.training_aggregation = training_aggregation
        # True (default) keeps today's behavior exactly: the k-means hybrid semantic
        # fusion runs inside the model and folds into anomaly_score. False drops HSF
        # from the image score and exposes patch_tokens/image_feature/text_feature so
        # an AdaCLIPSemanticFusion node downstream (or the ONNX export) can reproduce
        # it outside the traced graph.
        self.semantic_fusion = semantic_fusion
        self.k_clusters = k_clusters
        self._warmup_done = False

        # Log initialization parameters at DEBUG level (only shown if debug logging enabled)
        logger.debug(
            f"[AdaCLIPDetector] Initialized: "
            f"prompt_text='{self.prompt_text}', "
            f"use_half_precision={self.use_half_precision}, "
            f"backbone={self.backbone}, "
            f"image_size={self.image_size}"
        )

        # Lazy initialization - will be registered as submodule when loaded
        self._adaclip_model: AdaCLIPModel | None = None
        self._preprocess: Compose | None = None

        # Track initialization state as a buffer (survives state_dict)
        self.register_buffer(
            "_initialized_flag", torch.tensor(False, dtype=torch.bool), persistent=True
        )

        # Cache mean/std as buffers for efficient preprocessing (don't recreate every call)
        self.register_buffer(
            "_clip_mean",
            torch.tensor(OPENAI_DATASET_MEAN, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_clip_std",
            torch.tensor(OPENAI_DATASET_STD, dtype=torch.float32).view(1, 3, 1, 1),
            persistent=False,
        )

    @property
    def current_device(self) -> torch.device:
        """Discover current device from module parameters/buffers."""
        for param in self.parameters():
            return param.device
        for buf in self.buffers():
            return buf.device
        return torch.device("cpu")

    def _ensure_model_loaded(self) -> None:
        """Lazy load model on first forward pass.

        The model is created and then moved to the current device of this node.
        Subsequent .to() calls on the pipeline will move the registered submodule.
        """
        if self._adaclip_model is not None:
            return

        # A local checkpoint_path (shipped / provisioned weights) skips the
        # download entirely; otherwise fetch into the shared model cache.
        weight_path = self.checkpoint_path or download_weights(self.weight_name)

        # Get current device from this node's tensors
        device = self.current_device

        # Create model with device hint for initial creation
        model = AdaCLIPModel(
            backbone=self.backbone,
            image_size=self.image_size,
            prompting_depth=self.prompting_depth,
            prompting_length=self.prompting_length,
            use_hsf=self.semantic_fusion,
            k_clusters=self.k_clusters,
            device=str(device),  # Initial device hint
        )
        model.load_weights(weight_path)
        model.eval()

        # Freeze AdaCLIP weights by default. Even when enable_gradients is True,
        # we want gradients to flow THROUGH AdaCLIP to upstream nodes, but we
        # don't want to update AdaCLIP parameters themselves.
        for param in model.parameters():
            param.requires_grad_(False)

        # Register as a submodule so it moves with .to() calls on the pipeline
        # Using add_module ensures it's part of the module tree
        self.add_module("adaclip_model", model)
        self._adaclip_model = model
        self._preprocess = model.get_preprocess()
        self._initialized_flag.fill_(True)

        # Debug: Log device information (DEBUG level - only shown if debug logging enabled)
        model_device = (
            next(self._adaclip_model.parameters()).device
            if list(self._adaclip_model.parameters())
            else torch.device("cpu")
        )
        logger.debug(f"[AdaCLIPDetector] Model initialized on device: {model_device}")
        logger.debug(f"[AdaCLIPDetector] CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            logger.debug(f"[AdaCLIPDetector] CUDA device: {torch.cuda.get_device_name(0)}")
        logger.debug(
            f"[AdaCLIPDetector] Preprocessing method: {'tensor (fast)' if self.use_torch_preprocess else 'PIL (exact match)'}"
        )

        # Apply optimizations after model is loaded
        if self.use_half_precision and torch.cuda.is_available():
            try:
                logger.debug(
                    "[AdaCLIPDetector] Converting model to half precision (FP16) for faster inference..."
                )
                self._adaclip_model = self._adaclip_model.half()
                if (
                    hasattr(self._adaclip_model, "_clip_model")
                    and self._adaclip_model._clip_model is not None
                ):
                    self._adaclip_model._clip_model = self._adaclip_model._clip_model.half()
                logger.debug("[AdaCLIPDetector] ✅ Model converted to FP16")
            except Exception as e:
                logger.warning(
                    f"[AdaCLIPDetector] ⚠️  FP16 conversion failed: {e}, continuing with FP32"
                )
                self.use_half_precision = False

        # Enable cuDNN benchmarking for consistent input sizes
        if torch.cuda.is_available():
            torch.backends.cudnn.benchmark = True

    # ------------------------------------------------------------------
    # Freeze / unfreeze — selective adapter-layer support
    # ------------------------------------------------------------------

    def unfreeze(self) -> None:
        """Unfreeze only the lightweight adapter layers for fine-tuning.

        The CLIP backbone remains frozen.  Only the six adapter module groups
        defined in ``ADAPTER_MODULE_NAMES`` get ``requires_grad_(True)``.

        This is called automatically by
        ``pipeline.unfreeze_nodes_by_name(["adaclip"])``.
        """
        self._ensure_model_loaded()
        assert self._adaclip_model is not None

        n_unfrozen = 0
        for name, param in self._adaclip_model.named_parameters():
            if any(
                name == adapter or name.startswith(adapter + ".")
                for adapter in self.ADAPTER_MODULE_NAMES
            ):
                param.requires_grad_(True)
                n_unfrozen += 1

        self._frozen = False
        logger.info(
            f"[AdaCLIPDetector] Unfroze {n_unfrozen} adapter parameters (backbone stays frozen)"
        )

    def freeze(self) -> None:
        """Freeze all parameters (adapter layers + backbone)."""
        if self._adaclip_model is not None:
            for param in self._adaclip_model.parameters():
                param.requires_grad_(False)
        self._frozen = True

    def _preprocess_rgb_torch_pil_match(self, rgb_bhwc: torch.Tensor) -> torch.Tensor:
        """Torch implementation that closely matches the PIL pipeline.

        Pipeline: Resize((S,S), BICUBIC) -> CenterCrop(S) -> ToTensor() -> Normalize(mean,std)

        This matches upstream PIL preprocessing exactly:
        - Fixed-size resize (no aspect ratio preservation)
        - Bicubic interpolation with antialias
        - uint8 quantization before resizing (like PIL sees it) - SKIPPED when enable_gradients=True
        - ToTensor() equivalent: uint8 -> float in [0,1]
        - Normalize with cached mean/std

        NOTE: When enable_gradients=True, we skip uint8 conversion to preserve gradient flow.
        uint8 is not a floating-point type and breaks the computation graph.
        """
        x = rgb_bhwc  # [B, H, W, 3]

        # Bring to [0,255] like image pixels
        if x.max() <= 1.0:
            x = x * 255.0

        # When enable_gradients=True, we MUST stay in float to preserve gradient flow.
        # torch.uint8 is not a floating-point type and breaks the computation graph.
        if self.enable_gradients:
            # Differentiable path: clamp to [0,255] but stay in float32
            x = x.clamp(0, 255).to(torch.float32)  # [B, H, W, 3] float32
        else:
            # Non-differentiable path: exact PIL match with uint8 quantization
            x = x.round().clamp(0, 255).to(torch.uint8)  # [B, H, W, 3] uint8

        # BHWC -> BCHW
        x = x.permute(0, 3, 1, 2)  # [B, 3, H, W]

        S = self.image_size

        # Resize to fixed size (S,S) with bicubic, antialias if available
        if HAS_TORCHVISION_V2 and not self.enable_gradients:
            # torchvision.transforms.v2 can handle uint8 directly
            try:
                x = F_v2.resize(
                    x,
                    size=[S, S],
                    interpolation=InterpolationMode.BICUBIC,
                    antialias=True,
                )
                # CenterCrop (mostly no-op because it's already SxS)
                x = F_v2.center_crop(x, output_size=[S, S])
            except Exception:
                # Fallback: use functional interpolate
                x = x.to(torch.float32)
                x = torch.nn.functional.interpolate(
                    x, size=(S, S), mode="bicubic", align_corners=False
                )
        else:
            # Differentiable path or fallback: use interpolate (needs float)
            if x.dtype != torch.float32:
                x = x.to(torch.float32)
            try:
                x = torch.nn.functional.interpolate(
                    x,
                    size=(S, S),
                    mode="bicubic",
                    align_corners=False,
                    antialias=True,
                )
            except TypeError:
                # Older PyTorch versions don't support antialias
                x = torch.nn.functional.interpolate(
                    x, size=(S, S), mode="bicubic", align_corners=False
                )

        # ToTensor(): uint8 -> float in [0,1]
        # IMPORTANT: Ensure float32 conversion happens before normalization
        if x.dtype != torch.float32:
            x = x.to(torch.float32)
        if x.max() > 1.0:  # Still in [0, 255] range
            x = x / 255.0

        # Normalize using cached buffers
        # Extra safety: ensure we're in float32 before normalization
        # (mean/std operations require float dtype)
        if x.dtype not in (torch.float32, torch.float16, torch.bfloat16):
            x = x.to(torch.float32)
        mean = self._clip_mean.to(device=x.device, dtype=x.dtype)
        std = self._clip_std.to(device=x.device, dtype=x.dtype)
        x = (x - mean) / std

        return x

    def _preprocess_rgb(self, rgb_bhwc: torch.Tensor) -> torch.Tensor:
        """Preprocess RGB tensor for model input.

        When ``use_torch_preprocess`` is True (default), uses fast tensor-based
        preprocessing that closely matches PIL preprocessing with bicubic interpolation.
        This is faster than PIL preprocessing and works well for inference.

        When ``use_torch_preprocess`` is False, uses exact PIL-based preprocessing
        that matches the original CLIP preprocessing pipeline.

        The ``enable_gradients`` flag controls whether gradients flow through the
        preprocessing, but the preprocessing method choice is independent.
        """
        if self._preprocess is None:
            raise RuntimeError("AdaCLIPDetector model not initialized")

        # Fast tensor preprocessing path (default) - PIL-matching implementation
        if self.use_torch_preprocess:
            return self._preprocess_rgb_torch_pil_match(rgb_bhwc)

        # PIL preprocessing path (exact match with original CLIP preprocessing)
        else:
            b = rgb_bhwc.shape[0]

            # Convert to uint8 numpy for PIL processing (happens on CPU)
            rgb_np = rgb_bhwc.detach().cpu().numpy()

            # Handle different input ranges
            if rgb_np.max() <= 1.0:
                rgb_np = (rgb_np * 255).astype(np.uint8)
            else:
                rgb_np = rgb_np.astype(np.uint8)

            # Process each image through CLIP preprocessing (creates CPU tensors)
            preprocessed = []
            for i in range(b):
                pil_img = Image.fromarray(rgb_np[i], mode="RGB")
                img_tensor = self._preprocess(pil_img)
                preprocessed.append(img_tensor)

            batch_tensor = torch.stack(preprocessed, dim=0)

            target_device = self.current_device
            return batch_tensor.to(target_device)

    def forward(
        self,
        rgb_image: torch.Tensor,
        context: Context | None = None,  # noqa: ARG002
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Run AdaCLIP inference on RGB images.

        Parameters
        ----------
        rgb_image :
            RGB image tensor in BHWC format with 3 channels.
        """
        self._ensure_model_loaded()
        assert self._adaclip_model is not None

        b, h, w, _ = rgb_image.shape

        # DEBUG: Log input tensor stats for debugging score differences (TRACE level - not shown by default)
        # Convert to float for mean calculation if needed (uint8 doesn't support mean())
        if rgb_image.dtype in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
            mean_val = rgb_image.float().mean().item()
        else:
            mean_val = rgb_image.mean().item()
        logger.trace(
            f"[AdaCLIPDetector] Input: shape={rgb_image.shape}, "
            f"device={rgb_image.device}, dtype={rgb_image.dtype}, "
            f"min={rgb_image.min().item():.6f}, max={rgb_image.max().item():.6f}, "
            f"mean={mean_val:.6f}"
        )

        # Preprocess images
        preprocess_start = time.perf_counter()
        img_tensor = self._preprocess_rgb(rgb_image)
        preprocess_end = time.perf_counter()
        preprocess_time_ms = (preprocess_end - preprocess_start) * 1000.0
        logger.debug(
            f"[AdaCLIPDetector] Preprocessing time: {preprocess_time_ms:.2f}ms "
            f"(method={'tensor' if self.use_torch_preprocess else 'PIL'})"
        )

        # DEBUG: Log preprocessed tensor stats (TRACE level - not shown by default)
        logger.trace(
            f"[AdaCLIPDetector] Preprocessed: shape={img_tensor.shape}, "
            f"device={img_tensor.device}, dtype={img_tensor.dtype}, "
            f"min={img_tensor.min().item():.6f}, max={img_tensor.max().item():.6f}, "
            f"mean={img_tensor.mean().item():.6f}"
        )

        # Convert to half precision if enabled
        if self.use_half_precision and torch.cuda.is_available():
            img_tensor = img_tensor.half()

        # Warmup runs (only once, on first forward pass)
        if self.enable_warmup and not self._warmup_done and torch.cuda.is_available():
            logger.debug("[AdaCLIPDetector]  Running warmup inference to optimize CUDA kernels...")
            try:
                with torch.no_grad():
                    if self.use_half_precision:
                        with torch.amp.autocast("cuda", dtype=torch.float16):
                            _ = self._adaclip_model.predict(
                                img_tensor[:1] if img_tensor.shape[0] > 1 else img_tensor,
                                prompt=self.prompt_text,
                                sigma=self.gaussian_sigma,
                            )
                    else:
                        _ = self._adaclip_model.predict(
                            img_tensor[:1] if img_tensor.shape[0] > 1 else img_tensor,
                            prompt=self.prompt_text,
                            sigma=self.gaussian_sigma,
                        )
                    torch.cuda.synchronize()
                self._warmup_done = True
                logger.debug("[AdaCLIPDetector]  Warmup complete")
            except Exception as e:
                logger.warning(f"[AdaCLIPDetector]   Warmup failed: {e}, continuing without warmup")

        # Decide aggregation: use per-layer outputs during training when
        # training_aggregation is False (matches the paper's per-layer loss).
        # Always aggregate at inference time.
        use_aggregation = True
        if self.training and not self.training_aggregation:
            use_aggregation = False

        # Run inference. With semantic_fusion=False, the model already dropped HSF
        # from the image score (use_hsf=False on the underlying AdaCLIP model), and
        # we additionally ask for the raw features an AdaCLIPSemanticFusion node
        # downstream would need to reproduce it.
        want_components = not self.semantic_fusion
        components: dict[str, torch.Tensor] = {}
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_start = time.perf_counter()
        grad_ctx = nullcontext if self.enable_gradients else torch.no_grad
        with grad_ctx():
            if self.use_half_precision and torch.cuda.is_available():
                with torch.amp.autocast("cuda", dtype=torch.float16):
                    predict_out = self._adaclip_model.predict(
                        img_tensor,
                        prompt=self.prompt_text,
                        sigma=self.gaussian_sigma,
                        enable_gradients=self.enable_gradients,
                        aggregation=use_aggregation,
                        return_components=want_components,
                    )
            else:
                predict_out = self._adaclip_model.predict(
                    img_tensor,
                    prompt=self.prompt_text,
                    sigma=self.gaussian_sigma,
                    enable_gradients=self.enable_gradients,
                    aggregation=use_aggregation,
                    return_components=want_components,
                )
        if want_components:
            anomaly_map, anomaly_score, components = predict_out
        else:
            anomaly_map, anomaly_score = predict_out
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        inference_end = time.perf_counter()
        inference_time_ms = (inference_end - inference_start) * 1000.0

        logger.debug(
            f"[AdaCLIPDetector] Inference time: {inference_time_ms:.1f}ms "
            f"(batch_size={b}, per_image={inference_time_ms / b:.1f}ms, "
            f"aggregation={use_aggregation})"
        )

        # Placeholder for loss-port compatibility when per-layer outputs are absent.
        # Consumer contract: shape[1] == 1 means placeholder; real stacked per-layer
        # scores have shape [B, num_layers*2, H, W] with shape[1] > 1.
        per_layer_scores = torch.zeros(1, 1, 1, 1, device=img_tensor.device)
        image_score_2ch = anomaly_score  # fallback: may be [B] or [B, 2]

        if isinstance(anomaly_map, list):
            # anomaly_map is a list of per-layer softmaxed maps [B, 2, Hm, Wm].
            # Resize each to original (h, w) and stack for the loss port.
            resized_layers = []
            for am in anomaly_map:
                am_up = torch.nn.functional.interpolate(
                    am,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=True,
                )
                resized_layers.append(am_up)  # [B, 2, h, w]
            per_layer_scores = torch.cat(resized_layers, dim=1)  # [B, num_layers*2, h, w]

            # image_score_2ch: keep the raw [B, 2] softmaxed score
            image_score_2ch = anomaly_score  # already [B, 2] from upstream

            # Build the aggregated scores for metrics/viz (same formula as
            # upstream aggregation=True path).
            stacked = torch.stack(resized_layers, dim=0)  # [L, B, 2, h, w]
            agg = torch.mean(stacked, dim=0)  # [B, 2, h, w]
            agg = torch.softmax(agg, dim=1)
            anomaly_map = (agg[:, 1:, :, :] + 1 - agg[:, 0:1, :, :]) / 2.0  # [B, 1, h, w]
            anomaly_map = anomaly_map.squeeze(1)  # [B, h, w]
            anomaly_score_1d = anomaly_score[:, 1]  # scalar per image
        else:
            anomaly_score_1d = anomaly_score
            # Ensure image_score_2ch is [B, 2] even in aggregated mode
            # This [1-p, p] is a fallback representation, not identical to the
            # upstream training-mode 2-channel softmax distribution.
            if anomaly_score.dim() == 1:
                image_score_2ch = torch.stack([1 - anomaly_score, anomaly_score], dim=1)

        logger.trace(
            f"[AdaCLIPDetector] Raw anomaly_map: shape={anomaly_map.shape}, "
            f"min={anomaly_map.min().item():.6f}, max={anomaly_map.max().item():.6f}, "
            f"mean={anomaly_map.mean().item():.6f}"
        )

        # Resize aggregated anomaly map if needed (aggregated path already resized above)
        if anomaly_map.shape[-2] != h or anomaly_map.shape[-1] != w:
            input_dtype = anomaly_map.dtype
            if anomaly_map.dim() == 3:
                anomaly_map = torch.nn.functional.interpolate(
                    anomaly_map.unsqueeze(1),
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                ).squeeze(1)
            else:
                anomaly_map = torch.nn.functional.interpolate(
                    anomaly_map,
                    size=(h, w),
                    mode="bilinear",
                    align_corners=False,
                )
            anomaly_map = anomaly_map.to(input_dtype)

        if anomaly_map.dim() == 4 and anomaly_map.shape[1] == 1:
            anomaly_map = anomaly_map.squeeze(1)

        scores = anomaly_map.unsqueeze(-1)  # [B, H, W, 1]

        # Convert outputs back to FP32 for compatibility with downstream nodes
        # This allows FP16 computation internally for speed while maintaining FP32 interface
        # NOTE: This conversion is necessary because OUTPUT_SPECS specifies float32, and
        # downstream nodes (decider, visualizers) expect float32 inputs.
        if scores.dtype in (torch.float16, torch.bfloat16):
            scores = scores.float()
        if anomaly_score.dtype in (torch.float16, torch.bfloat16):
            anomaly_score = anomaly_score.float()

        # DEBUG: Save intermediate tensors
        # if hasattr(self, "_debug_save_dir") and self._debug_save_dir:
        #     if context is not None:
        #         for frame_idx in range(b):
        #             self._save_debug_tensor(rgb_image[frame_idx], "input_rgb", context, frame_idx)
        #             self._save_debug_tensor(scores[frame_idx], "output_scores", context, frame_idx)
        #     else:
        #         if hasattr(self, "_debug") and self._debug:
        #             print(f"[AdaCLIPDetector] DEBUG: context is None, skipping debug save")
        # else:
        #     if hasattr(self, "_debug") and self._debug:
        #         print(f"[AdaCLIPDetector] DEBUG: _debug_save_dir not set or empty. "
        #               f"hasattr={hasattr(self, '_debug_save_dir')}, "
        #               f"value={getattr(self, '_debug_save_dir', 'NOT_SET')}")

        # # DEBUG: Print output info
        # if hasattr(self, "_debug") and self._debug:
        #     print(f"[AdaCLIPDetector] Input RGB: shape={rgb_image.shape}, "
        #           f"min={rgb_image.min().item():.4f}, max={rgb_image.max().item():.4f}, "
        #           f"mean={rgb_image.mean().item():.4f}, requires_grad={rgb_image.requires_grad}")
        #     print(f"[AdaCLIPDetector] Output scores: shape={scores.shape}, "
        #           f"min={scores.min().item():.4f}, max={scores.max().item():.4f}, "
        #           f"mean={scores.mean().item():.4f}, requires_grad={scores.requires_grad}")

        if components:
            patch_tokens = components["patch_tokens"]
            image_feature = components["image_feature"]
            text_feature = components["text_feature"]
        else:
            patch_tokens = torch.zeros(1, 1, 1, 1, device=scores.device)
            image_feature = torch.zeros(1, 1, device=scores.device)
            text_feature = torch.zeros(1, 1, 1, device=scores.device)

        return {
            "scores": scores,
            "anomaly_score": anomaly_score_1d
            if anomaly_score_1d.dim() == 1
            else anomaly_score_1d.squeeze(),
            "per_layer_scores": per_layer_scores,
            "image_score_2ch": image_score_2ch,
            "patch_tokens": patch_tokens,
            "image_feature": image_feature,
            "text_feature": text_feature,
        }

    def export_module(self) -> torch.nn.Module:
        """Return the traceable adapter the span exporter runs in place of forward.

        Valid only with ``semantic_fusion=False``: the k-means hybrid semantic
        fusion runs on sklearn's ``KMeans``, which the dynamo exporter cannot
        trace. Add :class:`cuvis_ai_adaclip.node.semantic_fusion.AdaCLIPSemanticFusion`
        after this node to reproduce it outside the graph.
        """
        if self.semantic_fusion:
            raise ValueError(
                "AdaCLIPDetector.export_module() needs semantic_fusion=False: the "
                "k-means hybrid semantic fusion cannot be traced. Set "
                "semantic_fusion=False and add AdaCLIPSemanticFusion after this node."
            )
        self._ensure_model_loaded()
        return _AdaCLIPExportModule(self)

    # def _save_debug_tensor(
    #     self, tensor: torch.Tensor, name: str, context: Context | None, frame_idx: int
    # ) -> None:
    #     """Save tensor for debugging if debug mode is enabled."""
    #     if not (hasattr(self, "_debug_save_dir") and self._debug_save_dir):
    #         if hasattr(self, "_debug") and self._debug:
    #             print(f"[AdaCLIPDetector._save_debug_tensor] Skipping: _debug_save_dir not set")
    #         return

    #     if context is None:
    #         if hasattr(self, "_debug") and self._debug:
    #             print(f"[AdaCLIPDetector._save_debug_tensor] Skipping: context is None")
    #         return

    #     # Create directory structure: {stage}/epoch_{epoch}/batch_{batch_idx}/frame_{frame_idx}/
    #     # Convert ExecutionStage enum to string (e.g., ExecutionStage.TRAIN -> "train")
    #     stage_str = context.stage.value if hasattr(context.stage, "value") else str(context.stage)
    #     save_dir = (
    #         Path(self._debug_save_dir)
    #         / stage_str
    #         / f"epoch_{context.epoch:03d}"
    #         / f"batch_{context.batch_idx:03d}"
    #         / f"frame_{frame_idx:03d}"
    #     )
    #     save_dir.mkdir(parents=True, exist_ok=True)

    # Convert tensor to numpy and save
    # try:
    #     tensor_np = tensor.detach().cpu().numpy()
    #     save_path = save_dir / f"{self.name}_{name}.npy"
    #     np.save(save_path, tensor_np)
    #     if hasattr(self, "_debug") and self._debug:
    #         print(f"[AdaCLIPDetector._save_debug_tensor] Saved: {save_path}")
    # except Exception as e:
    #     if hasattr(self, "_debug") and self._debug:
    #         print(f"[AdaCLIPDetector._save_debug_tensor] ERROR saving {name}: {e}")

    def load_state_dict(self, state_dict, strict: bool = True) -> Any:
        """Load state dict with key remapping for _clip_model/_model compatibility.

        Handles backward compatibility when saved weights use different attribute names.
        The AdaCLIPModel wrapper now only uses _clip_model (not _model), but older
        saved weights might have _model keys that need to be remapped.
        """
        # Create a remapped state_dict that handles both _clip_model and _model keys
        remapped_state_dict = {}
        for key, value in state_dict.items():
            # If key has adaclip_model._model, remap to adaclip_model._clip_model
            if "adaclip_model._model." in key:
                new_key = key.replace("adaclip_model._model.", "adaclip_model._clip_model.")
                remapped_state_dict[new_key] = value
                # Also keep original in case both exist
                remapped_state_dict[key] = value
            else:
                remapped_state_dict[key] = value

        # Call parent load_state_dict with remapped keys
        # Use strict=False to allow partial loading if some keys don't match
        result = super().load_state_dict(remapped_state_dict, strict=False)

        # Log any remaining mismatches
        if hasattr(result, "missing_keys") and result.missing_keys:
            logger.warning(
                f"[AdaCLIPDetector] Missing keys after remapping (first 5): "
                f"{list(result.missing_keys)[:5]}..."
            )
        if hasattr(result, "unexpected_keys") and result.unexpected_keys:
            logger.debug(
                f"[AdaCLIPDetector] Unexpected keys (first 5): "
                f"{list(result.unexpected_keys)[:5]}..."
            )

        return result


__all__ = ["AdaCLIPDetector"]
