"""Analyze quantile thresholds for frames with and without stones.

This script:
  * Runs inference on all frames in the dataset
  * Calculates the 99.5% quantile threshold for each frame
  * Separates frames with stones vs without stones (using ground truth)
  * Plots and saves threshold distributions for comparison
  * Helps identify if thresholds differ between frames with/without stones
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from loguru import logger

try:
    import seaborn as sns

    sns.set_style("whitegrid")
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

from cuvis_ai.node.channel_selector import CIRSelector
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import ExecutionStage

from cuvis_ai_adaclip import (
    AdaCLIPDetector,
    download_weights,
    list_available_weights,
)
from cuvis_ai_adaclip.cli_utils import AdaCLIPCLI

# Create reusable CLI instance
cli = AdaCLIPCLI("AdaCLIP Threshold Analysis")


@cli.add_common_options
@cli.add_data_options
@cli.add_cir_options
@click.command()
@click.option("--quantile", type=float, default=0.995, help="Quantile for threshold calculation")
@click.option(
    "--output-dir",
    type=str,
    default="outputs/threshold_analysis",
    help="Output directory for analysis results",
)
def main(**kwargs) -> None:
    """Analyze quantile thresholds for frames with and without stones."""
    logger.info("=== AdaCLIP Threshold Analysis ===")
    run_start = time.perf_counter()

    # Parse configuration
    output_dir = Path(kwargs["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = cli.parse_data_config(**kwargs)

    # ----------------------------
    # Data & weights
    # ----------------------------
    datamodule = SingleCu3sDataModule(**data_config)
    datamodule.setup(stage=None)

    wavelengths = datamodule.train_ds.wavelengths
    logger.info("Wavelength range: {:.1f}-{:.1f} nm", wavelengths.min(), wavelengths.max())

    model_name = kwargs["backbone_name"]
    weight_name = kwargs["pretrained_adaclip"]
    prompt_text = kwargs["prompt_text"]
    quantile = kwargs["quantile"]

    logger.info("Available AdaCLIP weights: {}", list_available_weights())
    download_weights(weight_name)

    # CIR false-color wavelengths from CLI options
    nir_nm = kwargs["nir_nm"]
    red_nm = kwargs["red_nm"]
    green_nm = kwargs["green_nm"]

    logger.info("Model: {} | Weights: {}", model_name, weight_name)
    logger.info("Prompt: {}", prompt_text)
    logger.info("Quantile: {}", quantile)
    logger.info(
        "CIR wavelengths: NIR={:.1f} nm, Red={:.1f} nm, Green={:.1f} nm", nir_nm, red_nm, green_nm
    )

    # ----------------------------
    # Build simplified pipeline (no decider/metrics)
    # ----------------------------
    pipeline = CuvisPipeline("AdaCLIP_Threshold_Analysis")

    data_node = LentilsAnomalyDataNode(
        normal_class_ids=[0, 1],
    )
    band_selector = CIRSelector(nir_nm=nir_nm, red_nm=red_nm, green_nm=green_nm)

    use_half_precision = kwargs.get("use_half_precision", True)
    enable_warmup = kwargs.get("enable_warmup", True)
    use_torch_preprocess = kwargs.get("use_torch_preprocess", True)
    image_size = 518
    gaussian_sigma = kwargs["gaussian_sigma"]

    logger.info(
        f"AdaCLIP optimizations: FP16={use_half_precision}, Warmup={enable_warmup}, TorchPreprocess={use_torch_preprocess}"
    )
    logger.info(f"AdaCLIP image_size: {image_size}")

    adaclip = AdaCLIPDetector(
        weight_name=weight_name,
        backbone=model_name,
        prompt_text=prompt_text,
        image_size=image_size,
        gaussian_sigma=gaussian_sigma,
        use_half_precision=use_half_precision,
        enable_warmup=enable_warmup,
        use_torch_preprocess=use_torch_preprocess,
    )

    # Wiring: cube → band selector → AdaCLIP
    pipeline.connect(
        (data_node.outputs.cube, band_selector.inputs.cube),
        (data_node.outputs.wavelengths, band_selector.inputs.wavelengths),
        (band_selector.outputs.rgb_image, adaclip.inputs.rgb_image),
    )

    # Move pipeline to GPU if available
    device = cli.get_device()
    logger.info(f"Moving pipeline to device: {device}")
    pipeline.to(device)

    # ----------------------------
    # Collect all datasets (train + val + test)
    # ----------------------------
    all_datasets = []
    if datamodule.train_ds is not None:
        all_datasets.append(("train", datamodule.train_ds))
    if datamodule.val_ds is not None:
        all_datasets.append(("val", datamodule.val_ds))
    if datamodule.test_ds is not None:
        all_datasets.append(("test", datamodule.test_ds))

    logger.info(
        f"Processing {sum(len(ds) for _, ds in all_datasets)} frames across {len(all_datasets)} splits"
    )

    # ----------------------------
    # Process all frames
    # ----------------------------
    results = []
    total_frames = 0

    for split_name, dataset in all_datasets:
        logger.info(f"\n=== Processing {split_name} split ({len(dataset)} frames) ===")

        for idx in range(len(dataset)):
            sample = dataset[idx]
            cube_np = np.asarray(sample["cube"])  # [H, W, C]
            wl_np = np.asarray(sample["wavelengths"])  # [C]
            mask_np = sample.get("mask", None)  # [H, W] with class IDs
            mesu_index = int(sample.get("mesu_index", idx))

            # Check if frame has stones (class 3 = anomaly/stone)
            has_stones = False
            if mask_np is not None:
                has_stones = (mask_np == 3).any()

            # Build batch dict
            cube_t = torch.from_numpy(cube_np).unsqueeze(0).to(device)  # [1, H, W, C]
            wl_t = torch.from_numpy(wl_np.astype(np.int32)).unsqueeze(0).to(device)  # [1, C]

            batch = {
                "cube": cube_t,
                "wavelengths": wl_t,
                "mask": torch.from_numpy(mask_np).unsqueeze(0).to(device)
                if mask_np is not None
                else None,
            }

            # Run inference
            with torch.no_grad():
                outputs = pipeline.forward(
                    batch=batch,
                    stage=ExecutionStage.INFERENCE,
                )

            # Get scores from AdaCLIP
            scores = outputs[(adaclip.name, "scores")]  # [1, H, W, 1]
            scores_np = scores.squeeze().cpu().numpy()  # [H, W]

            # Calculate threshold using quantile (same as QuantileBinaryDecider)
            scores_flat = scores_np.flatten()
            threshold = float(np.quantile(scores_flat, quantile))

            # Calculate additional statistics
            score_min = float(scores_flat.min())
            score_max = float(scores_flat.max())
            score_mean = float(scores_flat.mean())
            score_std = float(scores_flat.std())
            score_median = float(np.median(scores_flat))

            # Count pixels above threshold
            pixels_above_threshold = int((scores_flat >= threshold).sum())
            total_pixels = len(scores_flat)
            fraction_above_threshold = pixels_above_threshold / total_pixels

            results.append(
                {
                    "split": split_name,
                    "frame_idx": idx,
                    "mesu_index": mesu_index,
                    "has_stones": has_stones,
                    "threshold": threshold,
                    "score_min": score_min,
                    "score_max": score_max,
                    "score_mean": score_mean,
                    "score_std": score_std,
                    "score_median": score_median,
                    "pixels_above_threshold": pixels_above_threshold,
                    "total_pixels": total_pixels,
                    "fraction_above_threshold": fraction_above_threshold,
                }
            )

            total_frames += 1
            if (idx + 1) % 10 == 0:
                logger.info(f"  Processed {idx + 1}/{len(dataset)} frames...")

    logger.info(f"\n=== Processed {total_frames} frames total ===")

    # ----------------------------
    # Analyze results
    # ----------------------------
    thresholds_with_stones = [r["threshold"] for r in results if r["has_stones"]]
    thresholds_without_stones = [r["threshold"] for r in results if not r["has_stones"]]

    logger.info("\n=== Threshold Statistics ===")
    logger.info(f"Frames with stones: {len(thresholds_with_stones)}")
    if thresholds_with_stones:
        logger.info(f"  Threshold mean: {np.mean(thresholds_with_stones):.6f}")
        logger.info(f"  Threshold std: {np.std(thresholds_with_stones):.6f}")
        logger.info(f"  Threshold min: {np.min(thresholds_with_stones):.6f}")
        logger.info(f"  Threshold max: {np.max(thresholds_with_stones):.6f}")
        logger.info(f"  Threshold median: {np.median(thresholds_with_stones):.6f}")

    logger.info(f"\nFrames without stones: {len(thresholds_without_stones)}")
    if thresholds_without_stones:
        logger.info(f"  Threshold mean: {np.mean(thresholds_without_stones):.6f}")
        logger.info(f"  Threshold std: {np.std(thresholds_without_stones):.6f}")
        logger.info(f"  Threshold min: {np.min(thresholds_without_stones):.6f}")
        logger.info(f"  Threshold max: {np.max(thresholds_without_stones):.6f}")
        logger.info(f"  Threshold median: {np.median(thresholds_without_stones):.6f}")

    # ----------------------------
    # Save results to JSON
    # ----------------------------
    results_file = output_dir / "threshold_analysis_results.json"
    logger.info(f"\nSaving results to: {results_file}")
    with open(results_file, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"  Saved {len(results)} frame results")

    # ----------------------------
    # Create visualizations
    # ----------------------------
    logger.info("\n=== Creating visualizations ===")

    # Set style
    plt.rcParams["figure.figsize"] = (12, 8)
    if HAS_SEABORN:
        sns.set_style("whitegrid")

    # 1. Threshold distribution comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Histogram comparison
    ax = axes[0, 0]
    if thresholds_with_stones:
        ax.hist(
            thresholds_with_stones,
            bins=30,
            alpha=0.7,
            label="With stones",
            color="red",
            edgecolor="black",
        )
    if thresholds_without_stones:
        ax.hist(
            thresholds_without_stones,
            bins=30,
            alpha=0.7,
            label="Without stones",
            color="blue",
            edgecolor="black",
        )
    ax.set_xlabel(f"Threshold (quantile={quantile})")
    ax.set_ylabel("Frequency")
    ax.set_title("Threshold Distribution Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Box plot comparison
    ax = axes[0, 1]
    data_to_plot = []
    labels = []
    if thresholds_with_stones:
        data_to_plot.append(thresholds_with_stones)
        labels.append("With stones")
    if thresholds_without_stones:
        data_to_plot.append(thresholds_without_stones)
        labels.append("Without stones")
    if data_to_plot:
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        bp["boxes"][0].set_facecolor("red") if len(bp["boxes"]) > 0 else None
        if len(bp["boxes"]) > 1:
            bp["boxes"][1].set_facecolor("blue")
        ax.set_ylabel(f"Threshold (quantile={quantile})")
        ax.set_title("Threshold Box Plot Comparison")
        ax.grid(True, alpha=0.3)

    # Violin plot
    ax = axes[1, 0]
    if thresholds_with_stones and thresholds_without_stones:
        data_dict = {
            "With stones": thresholds_with_stones,
            "Without stones": thresholds_without_stones,
        }
        parts = ax.violinplot(
            [data_dict["With stones"], data_dict["Without stones"]],
            positions=[0, 1],
            showmeans=True,
            showmedians=True,
        )
        ax.set_xticks([0, 1])
        ax.set_xticklabels(["With stones", "Without stones"])
        ax.set_ylabel(f"Threshold (quantile={quantile})")
        ax.set_title("Threshold Violin Plot")
        ax.grid(True, alpha=0.3)
        # Color the violins
        for pc, color in zip(parts["bodies"], ["red", "blue"], strict=True):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)

    # Scatter plot: threshold vs frame index
    ax = axes[1, 1]
    frame_indices_with = [i for i, r in enumerate(results) if r["has_stones"]]
    frame_indices_without = [i for i, r in enumerate(results) if not r["has_stones"]]
    if thresholds_with_stones:
        ax.scatter(
            frame_indices_with,
            thresholds_with_stones,
            alpha=0.6,
            label="With stones",
            color="red",
            s=30,
            edgecolors="black",
            linewidths=0.5,
        )
    if thresholds_without_stones:
        ax.scatter(
            frame_indices_without,
            thresholds_without_stones,
            alpha=0.6,
            label="Without stones",
            color="blue",
            s=30,
            edgecolors="black",
            linewidths=0.5,
        )
    ax.set_xlabel("Frame Index")
    ax.set_ylabel(f"Threshold (quantile={quantile})")
    ax.set_title("Threshold vs Frame Index")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_file = output_dir / "threshold_comparison.png"
    plt.savefig(plot_file, dpi=150, bbox_inches="tight")
    logger.info(f"  Saved: {plot_file}")
    plt.close()

    # 2. Score statistics comparison
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    score_stats = ["score_mean", "score_median", "score_max", "score_std"]
    for idx, stat in enumerate(score_stats):
        ax = axes[idx // 2, idx % 2]
        values_with = [r[stat] for r in results if r["has_stones"]]
        values_without = [r[stat] for r in results if not r["has_stones"]]

        data_to_plot = []
        labels = []
        if values_with:
            data_to_plot.append(values_with)
            labels.append("With stones")
        if values_without:
            data_to_plot.append(values_without)
            labels.append("Without stones")

        if data_to_plot:
            bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
            if len(bp["boxes"]) > 0:
                bp["boxes"][0].set_facecolor("red")
            if len(bp["boxes"]) > 1:
                bp["boxes"][1].set_facecolor("blue")
            ax.set_ylabel(stat.replace("_", " ").title())
            ax.set_title(f"{stat.replace('_', ' ').title()} Comparison")
            ax.grid(True, alpha=0.3)

    plt.tight_layout()
    score_stats_file = output_dir / "score_statistics_comparison.png"
    plt.savefig(score_stats_file, dpi=150, bbox_inches="tight")
    logger.info(f"  Saved: {score_stats_file}")
    plt.close()

    # 3. Fraction above threshold comparison
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    fractions_with = [r["fraction_above_threshold"] for r in results if r["has_stones"]]
    fractions_without = [r["fraction_above_threshold"] for r in results if not r["has_stones"]]

    if fractions_with:
        ax.hist(
            fractions_with, bins=30, alpha=0.7, label="With stones", color="red", edgecolor="black"
        )
    if fractions_without:
        ax.hist(
            fractions_without,
            bins=30,
            alpha=0.7,
            label="Without stones",
            color="blue",
            edgecolor="black",
        )
    ax.set_xlabel(f"Fraction of pixels above threshold (quantile={quantile})")
    ax.set_ylabel("Frequency")
    ax.set_title("Fraction of Pixels Above Threshold")
    ax.legend()
    ax.grid(True, alpha=0.3)
    # Add vertical line at expected fraction (1 - quantile)
    expected_fraction = 1.0 - quantile
    ax.axvline(
        expected_fraction,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Expected ({expected_fraction:.3f})",
    )
    ax.legend()

    plt.tight_layout()
    fraction_file = output_dir / "fraction_above_threshold.png"
    plt.savefig(fraction_file, dpi=150, bbox_inches="tight")
    logger.info(f"  Saved: {fraction_file}")
    plt.close()

    # ----------------------------
    # Summary statistics table
    # ----------------------------
    summary = {
        "quantile": quantile,
        "total_frames": total_frames,
        "frames_with_stones": len(thresholds_with_stones),
        "frames_without_stones": len(thresholds_without_stones),
        "thresholds_with_stones": {
            "mean": float(np.mean(thresholds_with_stones)) if thresholds_with_stones else None,
            "std": float(np.std(thresholds_with_stones)) if thresholds_with_stones else None,
            "min": float(np.min(thresholds_with_stones)) if thresholds_with_stones else None,
            "max": float(np.max(thresholds_with_stones)) if thresholds_with_stones else None,
            "median": float(np.median(thresholds_with_stones)) if thresholds_with_stones else None,
        },
        "thresholds_without_stones": {
            "mean": float(np.mean(thresholds_without_stones))
            if thresholds_without_stones
            else None,
            "std": float(np.std(thresholds_without_stones)) if thresholds_without_stones else None,
            "min": float(np.min(thresholds_without_stones)) if thresholds_without_stones else None,
            "max": float(np.max(thresholds_without_stones)) if thresholds_without_stones else None,
            "median": float(np.median(thresholds_without_stones))
            if thresholds_without_stones
            else None,
        },
    }

    summary_file = output_dir / "summary_statistics.json"
    logger.info(f"\nSaving summary to: {summary_file}")
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    total_duration = time.perf_counter() - run_start
    logger.info("\n=== Analysis Complete ===")
    logger.info(f"Results saved to: {output_dir}")
    logger.info(f"Total duration: {total_duration:.2f} seconds")
    logger.info(f"Average time per frame: {total_duration / total_frames:.3f} seconds")


if __name__ == "__main__":
    main()
