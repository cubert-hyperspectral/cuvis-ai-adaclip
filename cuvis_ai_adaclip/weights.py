"""Pretrained AdaCLIP heads and the CLIP backbone, declared for cuvis-ai-core's registry.

``WEIGHTS`` is the plugin's weight declaration: the three prompt heads the AdaCLIP
authors released (mirrored unchanged under ``cubert-gmbh/adaclip``) and the OpenAI
CLIP ViT-L/14 at 336 px backbone every head was trained on (``cubert-gmbh/clip``).
``cuvis_ai_adaclip/__init__`` registers the tuple with ``ModelWeights`` at import, and
cuvis-ai's ``emit_metadata`` projects it into the plugin manifest's ``weights:`` block,
so CuvisNEXT and the installer know what to provision without importing the plugin.
The pins come from ``tools/mirror_weights.py`` in cuvis-ai-core.

``ADACLIP_WEIGHTS`` (the node's ``weight_name`` vocabulary, keyed by the upstream Drive
filenames) is derived from ``WEIGHTS``: every head row's alias is its ``weight_name``.
``download_weights`` returns the cached file, downloading it when online; in the
sandboxed runtime (offline) a missing weight raises core's ``ModelWeightsMissingError``
naming the provisioning command (``download-model download adaclip_all``).

The upstream README's weights table labels ``pretrained_mvtec_colondb.pth`` as
"MVTec AD & ClinicDB" and ``pretrained_visa_clinicdb.pth`` as "VisA & ColonDB", while
its Train section pairs MVTec AD with ColonDB and VisA with ClinicDB, matching the
file names.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

from cuvis_ai_schemas.plugin import PluginWeightEntry
from loguru import logger

PLUGIN_NAME = "adaclip"
"""The manifest name of this plugin (what pipelines list under ``plugins:``)."""

CLIP_BACKBONE_NAME = "ViT-L-14-336"
"""The OpenCLIP model name of the backbone every shipped AdaCLIP head was trained on."""

_ADACLIP_REPO = "cubert-gmbh/adaclip"
_ADACLIP_REVISION = "16153b4ba74c2fe54a99679fc2e1b1e29993dc3f"
_HEAD_SIZE_BYTES = 42_673_907
# The upstream repos publish their code under MIT and state no licence for the weights.
_WEIGHTS_LICENSE = "unspecified (code: MIT)"


def _head(
    name: str,
    weight_name: str,
    display_name: str,
    summary: str,
    *,
    sha256: str,
    default: bool = False,
    description: str,
) -> PluginWeightEntry:
    return PluginWeightEntry(
        name=name,
        display_name=display_name,
        summary=summary,
        used_for=["Anomaly detection", "Zero-shot"],
        repo_id=_ADACLIP_REPO,
        filename=f"{weight_name}.pth",
        revision=_ADACLIP_REVISION,
        sha256=sha256,
        size_bytes=_HEAD_SIZE_BYTES,
        license=_WEIGHTS_LICENSE,
        license_file=None,
        aliases=[weight_name],
        selected_by="weight_name",
        default=default,
        explicit_path_hparams=["checkpoint_path"],
        description=description,
    )


CLIP_BACKBONE = PluginWeightEntry(
    name="clip_vit_l_14_336",
    display_name="CLIP ViT-L/14 at 336 px",
    summary="Backbone every AdaCLIP head needs",
    used_for=["Backbone", "Anomaly detection"],
    repo_id="cubert-gmbh/clip",
    filename="ViT-L-14-336px.pt",
    revision="a223b3db0b7bd1b55cf8f6421629b30b46c995de",
    sha256="3035c92b350959924f9f00213499208652fc7ea050643e8b385c2dac08641f02",
    size_bytes=934_088_680,
    license=_WEIGHTS_LICENSE,
    license_file=None,
    description=(
        "OpenAI CLIP ViT-L/14 at 336 px, the frozen backbone every AdaCLIP head runs on; "
        "materialized into the OpenCLIP cache the vendored loader reads (mirror of the "
        "OpenAI release, unmodified)."
    ),
)
"""The one CLIP backbone the plugin can provision; other backbones download from OpenAI."""

WEIGHTS: tuple[PluginWeightEntry, ...] = (
    _head(
        "adaclip_mvtec_colondb",
        "pretrained_mvtec_colondb",
        "AdaCLIP head (MVTec AD, ColonDB)",
        "Zero-shot anomaly detection, MVTec AD and ColonDB head",
        sha256="be51a42c052bd4cf060e54f503a1f5d0b2a3b899bc8dc2e243042f18b215427e",
        description=(
            "AdaCLIP prompt head trained on MVTec AD and ColonDB (the upstream weights "
            "table labels it MVTec AD & ClinicDB); weight_name: pretrained_mvtec_colondb."
        ),
    ),
    _head(
        "adaclip_visa_clinicdb",
        "pretrained_visa_clinicdb",
        "AdaCLIP head (VisA, ClinicDB)",
        "Zero-shot anomaly detection, VisA and ClinicDB head",
        sha256="3deabbbaf1e412cfdfcb42923a500b986f4b9ee96ccbc7a735d89dbc87df44c8",
        description=(
            "AdaCLIP prompt head trained on VisA and ClinicDB (the upstream weights table "
            "labels it VisA & ColonDB); weight_name: pretrained_visa_clinicdb."
        ),
    ),
    _head(
        "adaclip_all",
        "pretrained_all",
        "AdaCLIP head (all datasets)",
        "Zero-shot anomaly detection; the default AdaCLIP head",
        sha256="33e8d3db1cb4aab030866b8b70a46e10aa27ebf2c23b5463cb07f2574addd98c",
        default=True,
        description=(
            "AdaCLIP prompt head trained on all upstream datasets (MVTec AD, VisA, ColonDB, "
            "ClinicDB and more); the head an AdaCLIPDetector loads when weight_name is not set."
        ),
    ),
    CLIP_BACKBONE,
)
"""Every weight the adaclip nodes load: the three heads (picked by ``weight_name``) and the backbone."""

ADACLIP_WEIGHTS: dict[str, dict[str, Any]] = {
    entry.aliases[0]: {
        "registry_name": entry.name,
        "description": entry.description,
        "filename": entry.filename,
    }
    for entry in WEIGHTS
    if entry.selected_by == "weight_name"
}
"""``weight_name`` -> registry name, description and upstream filename (derived from ``WEIGHTS``)."""


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
