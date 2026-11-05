"""Standalone hybrid semantic fusion node for the cuvis_ai_adaclip plugin.

Runs the same k-means image-score fusion AdaCLIPDetector applies internally
when ``semantic_fusion=True``, but as a separate node over the raw features
AdaCLIPDetector exposes when ``semantic_fusion=False``. This is the piece that
stays in Python after an export: sklearn's KMeans is not traceable, so the
exported graph stops at the plain (non-fused) score and this node fuses it
back in afterward.
"""

from __future__ import annotations

from typing import Any

import torch
from cuvis_ai_core.node.node import Node
from cuvis_ai_schemas.enums import NodeCategory, NodeTag
from cuvis_ai_schemas.pipeline import PortSpec

from method.adaclip import HybridSemanticFusion


def compute_semantic_fusion_score(
    patch_tokens: torch.Tensor,
    image_feature: torch.Tensor,
    text_feature: torch.Tensor,
    hsf: HybridSemanticFusion,
    alpha: float = 0.2,
) -> torch.Tensor:
    """Fuse the k-means clustered feature into the image-level anomaly score.

    This is the exact computation :meth:`method.adaclip.AdaCLIP.visual_text_similarity`
    runs when ``use_hsf`` is True, factored out so both the in-node path (the
    detector with ``semantic_fusion=True``) and :class:`AdaCLIPSemanticFusion`
    call the same ``HybridSemanticFusion.forward``.

    Parameters
    ----------
    patch_tokens :
        Projected patch tokens for every output layer, stacked ``[B, L, N, D]``
        (the detector's ``patch_tokens`` output port).
    image_feature :
        Pooled and projected image feature ``[B, D]``.
    text_feature :
        Text embedding ``[1, D, 2]`` (normal, anomaly).
    hsf :
        The ``HybridSemanticFusion`` module performing the k-means clustering.
    alpha :
        Blend weight between the clustered feature and the plain image feature,
        matching the upstream constant.

    Returns
    -------
    torch.Tensor
        Image-level anomaly score ``[B]``, softmaxed and reduced to the
        anomaly channel exactly as the in-node fused path does.
    """
    patch_token_list = [patch_tokens[:, layer] for layer in range(patch_tokens.shape[1])]
    anomaly_maps = [100.0 * token @ text_feature for token in patch_token_list]

    clustered_feature = hsf.forward(patch_token_list, anomaly_maps)
    cur_image_feature = alpha * clustered_feature + (1 - alpha) * image_feature
    cur_image_feature = torch.nn.functional.normalize(cur_image_feature, dim=1)

    anomaly_score = 100.0 * cur_image_feature.unsqueeze(1) @ text_feature
    anomaly_score = anomaly_score.squeeze(1)
    anomaly_score = torch.softmax(anomaly_score, dim=1)
    return anomaly_score[:, 1]


class AdaCLIPSemanticFusion(Node):
    """Fuse AdaCLIP's k-means hybrid semantic feature into the image score.

    Pairs with :class:`cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector` run
    with ``semantic_fusion=False``: the detector exports its CLIP graph to ONNX
    and outputs the plain image score plus the raw features this node needs;
    this node reproduces the fused score the detector would otherwise compute
    internally, so the two together are output-identical to the in-node path.
    """

    _category = NodeCategory.MODEL
    _tags = frozenset({NodeTag.ANOMALY, NodeTag.INFERENCE, NodeTag.TORCH})

    INPUT_SPECS = {
        "patch_tokens": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1, -1),
            description="Projected patch tokens [B, num_layers, N, D], from AdaCLIPDetector.",
        ),
        "image_feature": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1),
            description="Pooled and projected image feature [B, D], from AdaCLIPDetector.",
        ),
        "text_feature": PortSpec(
            dtype=torch.float32,
            shape=(-1, -1, -1),
            description="Text embedding [1, D, 2] (normal, anomaly), from AdaCLIPDetector.",
        ),
    }

    OUTPUT_SPECS = {
        "anomaly_score": PortSpec(
            dtype=torch.float32,
            shape=(-1,),
            description="Image-level anomaly score [B], with the k-means fusion applied.",
        ),
    }

    def __init__(self, k_clusters: int = 20, **kwargs: Any) -> None:
        super().__init__(k_clusters=k_clusters, **kwargs)
        self.k_clusters = k_clusters
        self._hsf = HybridSemanticFusion(k_clusters)

    def forward(
        self,
        patch_tokens: torch.Tensor,
        image_feature: torch.Tensor,
        text_feature: torch.Tensor,
        **_: Any,
    ) -> dict[str, torch.Tensor]:
        """Fuse the k-means clustered feature into the image-level anomaly score."""
        anomaly_score = compute_semantic_fusion_score(
            patch_tokens, image_feature, text_feature, self._hsf
        )
        return {"anomaly_score": anomaly_score}


__all__ = ["AdaCLIPSemanticFusion", "compute_semantic_fusion_score"]
