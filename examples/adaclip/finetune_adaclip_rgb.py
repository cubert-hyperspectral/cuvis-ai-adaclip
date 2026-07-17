"""Fine-tune AdaCLIP adapter layers with standard RGB band selection.

Matches the AdaCLIP paper (Cao et al., ECCV 2024):
  - Only adapter layers (~10 M params) are updated; CLIP backbone is frozen.
  - Focal + Dice loss (pixel-level) + Focal loss (image-level).
  - lr=0.01, AdamW with betas=(0.5, 0.999), 5 epochs, batch_size=1.

Pipeline:
  HSI → LentilsAnomalyDataNode → MinMaxNormalizer → FixedWavelengthSelector(RGB)
      → AdaCLIPDetector (adapter layers unfrozen) → AdaCLIPFocalDiceLoss

Usage:
    cd /home/dev/anish/cuvis-ai-adaclip
    uv run python examples/adaclip/finetune_adaclip_rgb.py
    uv run python examples/adaclip/finetune_adaclip_rgb.py \\
        data.splits_csv=lentils_splits.csv
"""

from __future__ import annotations

import hydra
from omegaconf import DictConfig

from .finetune_adaclip_common import run_finetune


@hydra.main(
    config_path="../../configs/",
    config_name="trainrun/finetune_adaclip",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    run_finetune(cfg, mode="rgb")


if __name__ == "__main__":
    main()
