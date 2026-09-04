"""Pretrained AdaCLIP heads, resolved through cuvis-ai-core's weight registry.

The three checkpoints the AdaCLIP authors released on Google Drive are mirrored
unchanged under the ``cubert-gmbh`` Hugging Face organisation
(``cubert-gmbh/adaclip``) and registered in ``cuvis_ai_core.data.model_weights``.
``download_weights`` returns the cached file, downloading it when online, so no
Google Drive access is involved any more; in the sandboxed runtime (offline) a
missing weight raises core's ``ModelWeightsMissingError`` naming the provisioning
command (``download-model download adaclip_all``).

The weight keys are the upstream Drive filenames. The upstream README's weights
table labels ``pretrained_mvtec_colondb.pth`` as "MVTec AD & ClinicDB" and
``pretrained_visa_clinicdb.pth`` as "VisA & ColonDB", while its Train section
pairs MVTec AD with ColonDB and VisA with ClinicDB, matching the file names.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from loguru import logger

ADACLIP_WEIGHTS: dict[str, dict[str, Any]] = {
    "pretrained_mvtec_colondb": {
        "registry_name": "adaclip_mvtec_colondb",
        "description": "Trained on MVTec AD & ColonDB (upstream table label: MVTec AD & ClinicDB)",
        "filename": "pretrained_mvtec_colondb.pth",
    },
    "pretrained_visa_clinicdb": {
        "registry_name": "adaclip_visa_clinicdb",
        "description": "Trained on VisA & ClinicDB (upstream table label: VisA & ColonDB)",
        "filename": "pretrained_visa_clinicdb.pth",
    },
    "pretrained_all": {
        "registry_name": "adaclip_all",
        "description": "Trained on all datasets (MVTec, VisA, ColonDB, ClinicDB, etc.)",
        "filename": "pretrained_all.pth",
    },
}


def get_weights_dir() -> Path:
    """Legacy AdaCLIP cache directory, kept for callers that place their own files.

    Honors ``$CUVIS_MODEL_CACHE_DIR`` (the shared model cache the cuvis-ai
    orchestrator injects into the child runtime), else ``~/.cache/cuvis_ai``.
    The registry weights themselves live in the Hugging Face cache managed by
    cuvis-ai-core; this directory is not consulted by :func:`download_weights`.
    """
    root = os.environ.get("CUVIS_MODEL_CACHE_DIR")
    base = Path(root) if root else Path.home() / ".cache" / "cuvis_ai"
    cache_dir = base / "adaclip"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def list_available_weights() -> list[str]:
    """List available pretrained weight names."""
    return list(ADACLIP_WEIGHTS.keys())


def _config(weight_name: str) -> dict[str, Any]:
    if weight_name not in ADACLIP_WEIGHTS:
        available = list_available_weights()
        raise ValueError(f"Unknown weight: {weight_name}. Available: {available}")
    return ADACLIP_WEIGHTS[weight_name]


def download_weights(weight_name: str, force: bool = False) -> Path:
    """Return the local path of a pretrained AdaCLIP head, fetching it if allowed.

    Parameters
    ----------
    weight_name :
        Name of the pretrained weights. One of:
        - ``"pretrained_mvtec_colondb"``: MVTec AD & ColonDB.
        - ``"pretrained_visa_clinicdb"``: VisA & ClinicDB.
        - ``"pretrained_all"``: All datasets combined.
    force :
        If ``True``, re-download the file into the shared cache even if cached.

    Raises
    ------
    ValueError
        Unknown ``weight_name``.
    cuvis_ai_core.data.model_weights.ModelWeightsMissingError
        The weight is not cached and downloading is not allowed (offline child).
    """
    cfg = _config(weight_name)
    from cuvis_ai_core.data.model_weights import ModelWeights

    if force:
        logger.info(f"Re-downloading AdaCLIP weights: {weight_name} ({cfg['description']})")
        return ModelWeights.download_model(cfg["registry_name"], force=True)
    return ModelWeights.resolve(cfg["registry_name"])


def get_local_weight_path(weight_name: str) -> Path | None:
    """Return the cached path of ``weight_name`` without downloading, or ``None``.

    ``None`` for an unknown name or a weight that is not in the shared cache yet.
    """
    if weight_name not in ADACLIP_WEIGHTS:
        return None
    from cuvis_ai_core.data.model_weights import ModelDownloadError, ModelWeights

    try:
        return ModelWeights.resolve(ADACLIP_WEIGHTS[weight_name]["registry_name"], download=False)
    except ModelDownloadError:
        return None
