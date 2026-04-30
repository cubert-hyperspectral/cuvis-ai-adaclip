"""Fine-tune AdaCLIP adapter layers with CIR (Color Infrared) band selection.

Same as finetune_adaclip_rgb.py but uses CIRSelector (NIR→R, Red→G, Green→B).

Usage:
    cd /home/dev/anish/cuvis-ai-adaclip
    uv run python examples/adaclip/finetune_adaclip_cir.py
    uv run python examples/adaclip/finetune_adaclip_cir.py \\
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
    run_finetune(cfg, mode="cir")


if __name__ == "__main__":
    main()
