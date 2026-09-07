"""cuvis_ai_adaclip: AdaCLIP wrapper and cuvis.ai plugin package.

This package lives inside the forked AdaCLIP repository and provides:

- A high-level AdaCLIP model wrapper backed by the upstream implementation
  (see :mod:`cuvis_ai_adaclip.adaclip_upstream`).
- A cuvis.ai-compatible Node, :class:`cuvis_ai_adaclip.node.AdaCLIPDetector`,
  which plugs into the cuvis.ai canvas/Node system.
- The plugin's weight declarations (see :mod:`cuvis_ai_adaclip.weights`), registered
  with cuvis-ai-core's model-weight registry when the package is imported.
"""

from cuvis_ai_core.data.model_weights import ModelWeights

from .adaclip_upstream import (  # noqa: F401
    OPENAI_DATASET_MEAN,
    OPENAI_DATASET_STD,
    AdaCLIPModel,
    create_adaclip_model,
    download_weights,
    list_available_weights,
)
from .node import AdaCLIPDetector, AdaCLIPFocalDiceLoss, LossNode  # noqa: F401
from .weights import ADACLIP_WEIGHTS, PLUGIN_NAME, WEIGHTS, get_weights_dir  # noqa: F401

# Declaring the weights here (idempotent) is what lets ``download_weights`` resolve the
# heads, and ``download-model`` in the same environment list them, without a manifest.
ModelWeights.register(PLUGIN_NAME, WEIGHTS)


def register_all_nodes() -> int:
    """Register all cuvis_ai_adaclip nodes in the cuvis.ai NodeRegistry.

    Returns
    -------
    int
        The number of node classes that were registered.
    """
    # Local import to avoid importing cuvis_ai at plugin import time
    from cuvis_ai_core.utils.node_registry import NodeRegistry

    return NodeRegistry.auto_register_package("cuvis_ai_adaclip")


__all__ = [
    "ADACLIP_WEIGHTS",
    "AdaCLIPDetector",
    "AdaCLIPFocalDiceLoss",
    "AdaCLIPModel",
    "PLUGIN_NAME",
    "WEIGHTS",
    "create_adaclip_model",
    "download_weights",
    "get_weights_dir",
    "list_available_weights",
    "LossNode",
    "OPENAI_DATASET_MEAN",
    "OPENAI_DATASET_STD",
    "register_all_nodes",
]
