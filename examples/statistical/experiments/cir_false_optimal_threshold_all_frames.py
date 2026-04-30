"""AdaCLIP CIR False Color with optimal threshold selection using all frames.

This script:
1. Processes all available frames to compute image-level scores
2. Finds optimal threshold based on F1 or recall using ALL frames
3. Applies the threshold to all frames
4. Always generates masks using quantile decider (regardless of gate)
5. Saves visualizations for all frames with CIR image, GT, and predictions
6. Saves comprehensive metrics and results
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from cuvis_ai.deciders.binary_decider import QuantileBinaryDecider
from cuvis_ai.node.channel_selector import CIRSelector
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_schemas.enums import ExecutionStage
from loguru import logger
from sklearn.metrics import (
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)

from cuvis_ai_adaclip import (
    AdaCLIPDetector,
    download_weights,
    list_available_weights,
)
from cuvis_ai_adaclip.cli_utils import AdaCLIPCLI

cli = AdaCLIPCLI("AdaCLIP CIR False Color (Optimal Threshold - All Frames)")


def compute_image_score(anomaly_map: np.ndarray, method: str = "top_k_mean", k: int = 100) -> float:
    """Compute image-level anomaly score from anomaly map."""
    if method == "max":
        return float(anomaly_map.max())
    elif method == "top_k_mean":
        flat = anomaly_map.flatten()
        top_k = np.partition(flat, -k)[-k:]
        return float(top_k.mean())
    else:
        raise ValueError(f"Unknown method: {method}")


def resolve_top_k(num_pixels: int, top_k: int, fraction: float) -> int:
    """Resolve how many pixels to use for top_k_mean."""
    if top_k > 0:
        return top_k
    computed = int(np.ceil(fraction * num_pixels))
    return max(1, computed)


def compute_iou(pred_mask: np.ndarray, gt_mask: np.ndarray) -> float:
    """Compute Intersection over Union (IoU) between binary masks."""
    pred_bool = pred_mask.astype(bool)
    gt_bool = gt_mask.astype(bool)

    intersection = np.logical_and(pred_bool, gt_bool).sum()
    union = np.logical_or(pred_bool, gt_bool).sum()

    if union == 0:
        return 1.0 if intersection == 0 else 0.0

    return float(intersection / union)


def find_optimal_threshold(
    y_true: np.ndarray, y_scores: np.ndarray, metric: str = "f1"
) -> tuple[float, float, dict]:
    """Find optimal threshold using all frames.

    Returns:
        (optimal_threshold, optimal_metric_value, metrics_dict)
    """
    if metric == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = (
            thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
        )
        optimal_f1 = float(f1_scores[optimal_idx])
        optimal_precision = float(precision[optimal_idx])
        optimal_recall = float(recall[optimal_idx])

        return (
            float(optimal_threshold),
            optimal_f1,
            {
                "f1": optimal_f1,
                "precision": optimal_precision,
                "recall": optimal_recall,
            },
        )
    elif metric == "recall":
        # Find threshold that maximizes recall while keeping precision reasonable
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        # Find threshold with recall >= 0.9 and highest precision
        valid_idx = recall >= 0.9
        if valid_idx.any():
            optimal_idx = np.argmax(precision[valid_idx])
            optimal_threshold = (
                thresholds[valid_idx][optimal_idx]
                if optimal_idx < len(thresholds[valid_idx])
                else thresholds[-1]
            )
            optimal_recall = float(recall[valid_idx][optimal_idx])
            optimal_precision = float(precision[valid_idx][optimal_idx])
        else:
            # Fallback to max recall
            optimal_idx = np.argmax(recall)
            optimal_threshold = (
                thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
            )
            optimal_recall = float(recall[optimal_idx])
            optimal_precision = float(precision[optimal_idx])

        optimal_f1 = (
            2 * (optimal_precision * optimal_recall) / (optimal_precision + optimal_recall + 1e-10)
        )

        return (
            float(optimal_threshold),
            optimal_recall,
            {
                "f1": float(optimal_f1),
                "precision": optimal_precision,
                "recall": optimal_recall,
            },
        )
    else:
        raise ValueError(f"Unknown metric: {metric}")


def save_frame_visualization(
    output_dir: Path,
    frame_idx: int,
    mesu_index: int,
    rgb_image: np.ndarray,
    gt_mask: np.ndarray | None,
    pred_mask: np.ndarray,
    image_score: float,
    threshold: float,
    passed_gate: bool,
    iou: float | None = None,
) -> None:
    """Save comprehensive visualization for a single frame."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Normalize RGB image
    rgb_uint8 = (np.clip(rgb_image, 0, 1) * 255).astype(np.uint8)

    # Create figure with 4 subplots
    fig, axes = plt.subplots(2, 2, figsize=(16, 16))

    # Main title with score and threshold info
    gate_status = "PASS" if passed_gate else "FAIL"
    gate_color = "green" if passed_gate else "red"
    main_title = f"Frame {frame_idx} (mesu {mesu_index}) | Image Score: {image_score:.6f} | Threshold: {threshold:.6f} | Gate: {gate_status}"
    fig.suptitle(main_title, fontsize=14, fontweight="bold", color=gate_color)

    # Subplot 1: CIR RGB image
    axes[0, 0].imshow(rgb_uint8)
    axes[0, 0].set_title(
        f"CIR False-Color RGB\nImage Score: {image_score:.6f}\nThreshold: {threshold:.6f}\nGate: {gate_status}",
        fontweight="bold" if passed_gate else "normal",
        color=gate_color,
    )
    axes[0, 0].axis("off")

    # Subplot 2: Ground truth mask
    if gt_mask is not None:
        gt_bool = (gt_mask == 3).astype(bool)
        axes[0, 1].imshow(gt_bool, cmap="Reds", vmin=0, vmax=1)
        axes[0, 1].set_title(f"Ground Truth Mask\n({gt_bool.sum()} pixels)")
        axes[0, 1].axis("off")
    else:
        axes[0, 1].text(0.5, 0.5, "No GT available", ha="center", va="center")
        axes[0, 1].axis("off")

    # Subplot 3: Predicted mask
    pred_bool = pred_mask.astype(bool)
    axes[1, 0].imshow(pred_bool, cmap="Reds", vmin=0, vmax=1)
    title = f"Predicted Mask\n({pred_bool.sum()} pixels)"
    if passed_gate and iou is not None:
        title += f"\nIoU: {iou:.4f}"
    elif not passed_gate:
        title += "\n(Gate FAILED - Blank Mask)"
    axes[1, 0].set_title(
        title,
        fontweight="bold" if passed_gate else "normal",
        color=gate_color if not passed_gate else "black",
    )
    axes[1, 0].axis("off")

    # Subplot 4: Overlay comparison
    overlay = rgb_uint8.copy().astype(float) / 255.0
    if gt_mask is not None and passed_gate:
        gt_bool = (gt_mask == 3).astype(bool)
        # Green: TP, Red: FP, Yellow: FN
        tp = pred_bool & gt_bool
        fp = pred_bool & ~gt_bool
        fn = ~pred_bool & gt_bool

        overlay[tp] = [0, 1, 0]  # Green: True Positives
        overlay[fp] = [1, 0, 0]  # Red: False Positives
        overlay[fn] = [1, 1, 0]  # Yellow: False Negatives

        overlay_title = "Overlay (Green=TP, Red=FP, Yellow=FN)\n"
        overlay_title += (
            f"Image Score: {image_score:.6f}\nThreshold: {threshold:.6f}\nGate: {gate_status}"
        )
        if iou is not None:
            overlay_title += f"\nIoU: {iou:.4f}"
    elif gt_mask is not None and not passed_gate:
        # Gate failed, no overlay
        overlay_title = "Overlay (Gate FAILED - No Mask)\n"
        overlay_title += (
            f"Image Score: {image_score:.6f}\nThreshold: {threshold:.6f}\nGate: {gate_status}"
        )
    else:
        # No GT available
        if passed_gate:
            overlay[pred_bool] = [1, 0, 0]  # Red for predictions
            overlay_title = f"Prediction Overlay\nImage Score: {image_score:.6f}\nThreshold: {threshold:.6f}\nGate: {gate_status}"
        else:
            overlay_title = f"Prediction Overlay (Gate FAILED)\nImage Score: {image_score:.6f}\nThreshold: {threshold:.6f}\nGate: {gate_status}"

    axes[1, 1].imshow(overlay)
    axes[1, 1].set_title(
        overlay_title,
        fontweight="bold" if passed_gate else "normal",
        color=gate_color if not passed_gate else "black",
    )
    axes[1, 1].axis("off")

    plt.tight_layout()
    output_path = output_dir / f"frame_{frame_idx:03d}_mesu{mesu_index:03d}.png"
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.debug(f"Saved visualization: {output_path}")


@cli.add_common_options
@cli.add_data_options
@cli.add_cir_options
@click.command()
@click.option(
    "--top-k-fraction",
    type=float,
    default=0.001,
    help="Fraction of pixels used for top-k mean (default 0.1%)",
)
@click.option(
    "--test-only/--no-test-only",
    default=True,
    help="Skip training/validation splits and put frames 0-13 into the test split",
)
@click.option(
    "--threshold-metric",
    type=click.Choice(["f1", "recall"]),
    default="f1",
    help="Metric for finding optimal threshold: f1 or recall",
)
@click.option(
    "--output-dir",
    type=str,
    default="outputs/cir_false_optimal_threshold_all_frames",
    help="Output directory for results",
)
def main(**kwargs) -> None:
    logger.info("=== AdaCLIP CIR false-color (optimal threshold - all frames) ===")
    run_start = time.perf_counter()

    output_dir = Path(kwargs["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)

    data_config = cli.parse_data_config(**kwargs)

    # Force test-only mode: all frames 0-13 in test
    if kwargs.get("test_only", True):
        forced_test_ids = list(range(0, 14))
        data_config["train_ids"] = []
        data_config["val_ids"] = []
        data_config["test_ids"] = forced_test_ids
        logger.info("Forcing only test split. Frames 0-13 will be treated as test data.")

    datamodule = SingleCu3sDataModule(**data_config)
    datamodule.setup(stage=None)

    # Get wavelengths from any available dataset
    if datamodule.train_ds is not None:
        wavelengths = datamodule.train_ds.wavelengths
    elif datamodule.val_ds is not None:
        wavelengths = datamodule.val_ds.wavelengths
    elif datamodule.test_ds is not None:
        wavelengths = datamodule.test_ds.wavelengths
    else:
        raise ValueError("No dataset available!")

    logger.info("Wavelength range: {:.1f}-{:.1f} nm", wavelengths.min(), wavelengths.max())

    model_name = kwargs["backbone_name"]
    weight_name = kwargs["pretrained_adaclip"]
    # Use empty prompt
    prompt_text = ""
    logger.info(f"Using empty prompt: '{prompt_text}'")

    logger.info("Available AdaCLIP weights: {}", list_available_weights())
    download_weights(weight_name)

    quantile = kwargs["quantile"]
    gaussian_sigma = kwargs["gaussian_sigma"]
    top_k_fraction = kwargs["top_k_fraction"]
    threshold_metric = kwargs["threshold_metric"]

    nir_nm = kwargs["nir_nm"]
    red_nm = kwargs["red_nm"]
    green_nm = kwargs["green_nm"]

    logger.info("Quantile: {}", quantile)
    logger.info("Top-k fraction: {}", top_k_fraction)
    logger.info("Threshold metric: {}", threshold_metric)

    # Build pipeline
    pipeline = CuvisPipeline("AdaCLIP_CIR_FalseColor_OptimalThreshold_AllFrames")

    data_node = LentilsAnomalyDataNode(normal_class_ids=[0, 1])
    band_selector = CIRSelector(nir_nm=nir_nm, red_nm=red_nm, green_nm=green_nm)

    image_size = 518
    adaclip = AdaCLIPDetector(
        weight_name=weight_name,
        backbone=model_name,
        prompt_text=prompt_text,
        image_size=image_size,
        gaussian_sigma=gaussian_sigma,
        use_half_precision=kwargs.get("use_half_precision", True),
        enable_warmup=kwargs.get("enable_warmup", True),
        use_torch_preprocess=kwargs.get("use_torch_preprocess", True),
    )

    # Quantile decider (always generates masks)
    quantile_decider = QuantileBinaryDecider(quantile=quantile)

    pipeline.connect(
        (data_node.outputs.cube, band_selector.inputs.cube),
        (data_node.outputs.wavelengths, band_selector.inputs.wavelengths),
        (band_selector.outputs.rgb_image, adaclip.inputs.rgb_image),
        (adaclip.outputs.scores, quantile_decider.inputs.logits),
    )

    device = cli.get_device()
    logger.info(f"Moving pipeline to device: {device}")
    pipeline.to(device)

    # Collect all frames
    all_frames = []
    if datamodule.test_ds is not None:
        for idx in range(len(datamodule.test_ds)):
            sample = datamodule.test_ds[idx]
            mask_np = sample.get("mask", None)
            has_stones = False
            if mask_np is not None:
                has_stones = (mask_np == 3).any()
            all_frames.append(
                {
                    "split": "test",
                    "frame_idx": idx,
                    "mesu_index": int(sample.get("mesu_index", idx)),
                    "has_stones": bool(has_stones),
                    "sample": sample,
                }
            )

    normal_frames = [f for f in all_frames if not f["has_stones"]]
    anomaly_frames = [f for f in all_frames if f["has_stones"]]

    logger.info(
        f"Total frames: {len(all_frames)} (normal: {len(normal_frames)}, anomaly: {len(anomaly_frames)})"
    )

    # Step 1: Process ALL frames to get image scores
    logger.info("\n=== Processing all frames to compute image scores ===")
    all_frame_results = []

    for frame_info in all_frames:
        sample = frame_info["sample"]
        cube_np = np.asarray(sample["cube"])
        wl_np = np.asarray(sample["wavelengths"])
        mask_np = sample.get("mask", None)

        cube_t = torch.from_numpy(cube_np).unsqueeze(0).to(device)
        wl_t = torch.from_numpy(wl_np.astype(np.int32)).unsqueeze(0).to(device)

        batch = {
            "cube": cube_t,
            "wavelengths": wl_t,
        }

        with torch.no_grad():
            outputs = pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)

        scores = outputs[(adaclip.name, "scores")]
        scores_np = scores.squeeze().cpu().numpy()

        rgb_image = outputs[(band_selector.name, "rgb_image")]
        rgb_np = rgb_image[0].cpu().numpy()

        # Compute image score
        resolved_top_k = resolve_top_k(scores_np.size, 0, top_k_fraction)
        image_score = compute_image_score(scores_np, method="top_k_mean", k=resolved_top_k)

        all_frame_results.append(
            {
                "frame_info": frame_info,
                "anomaly_map": scores_np,
                "rgb_image": rgb_np,
                "mask": mask_np,
                "image_score": image_score,
            }
        )

    # Step 2: Find optimal threshold using ALL frames
    logger.info(
        f"\n=== Finding optimal threshold (metric: {threshold_metric}) using ALL frames ==="
    )
    all_image_scores = np.array([r["image_score"] for r in all_frame_results])
    all_labels = np.array([1 if r["frame_info"]["has_stones"] else 0 for r in all_frame_results])

    optimal_threshold, optimal_metric_value, optimal_metrics = find_optimal_threshold(
        all_labels, all_image_scores, metric=threshold_metric
    )

    logger.info(f"Optimal threshold: {optimal_threshold:.6f}")
    logger.info(f"Optimal {threshold_metric}: {optimal_metric_value:.4f}")
    logger.info(f"  Precision: {optimal_metrics['precision']:.4f}")
    logger.info(f"  Recall: {optimal_metrics['recall']:.4f}")
    logger.info(f"  F1: {optimal_metrics['f1']:.4f}")

    # Evaluate with optimal threshold
    all_predictions = (all_image_scores >= optimal_threshold).astype(int)
    all_accuracy = (all_predictions == all_labels).mean()
    all_f1 = f1_score(all_labels, all_predictions)
    all_auc = roc_auc_score(all_labels, all_image_scores)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    tn, fp, fn, tp = cm.ravel()

    logger.info("\n=== Results with optimal threshold ===")
    logger.info(f"Accuracy: {all_accuracy:.4f}")
    logger.info(f"F1: {all_f1:.4f}")
    logger.info(f"AUC: {all_auc:.4f}")
    logger.info("Confusion Matrix:")
    logger.info(f"  True Negatives: {tn}")
    logger.info(f"  False Positives: {fp}")
    logger.info(f"  False Negatives: {fn}")
    logger.info(f"  True Positives: {tp}")

    # Step 3: Process all frames again with optimal threshold and generate masks
    logger.info("\n=== Processing all frames with optimal threshold and generating masks ===")
    final_results = []

    for frame_result in all_frame_results:
        frame_info = frame_result["frame_info"]
        scores_np = frame_result["anomaly_map"]
        rgb_np = frame_result["rgb_image"]
        mask_np = frame_result["mask"]
        image_score = frame_result["image_score"]

        # Check if gate passes first
        passed_gate = image_score >= optimal_threshold

        # Only generate mask if gate passes, otherwise use blank mask
        if passed_gate:
            # Re-run pipeline to get mask via quantile decider
            sample = frame_info["sample"]
            cube_np = np.asarray(sample["cube"])
            wl_np = np.asarray(sample["wavelengths"])

            cube_t = torch.from_numpy(cube_np).unsqueeze(0).to(device)
            wl_t = torch.from_numpy(wl_np.astype(np.int32)).unsqueeze(0).to(device)

            batch = {
                "cube": cube_t,
                "wavelengths": wl_t,
            }

            with torch.no_grad():
                outputs = pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)

            # Generate mask using quantile decider
            decisions = outputs[(quantile_decider.name, "decisions")]
            pred_mask_np = decisions.squeeze().cpu().numpy().astype(bool)

            # Compute IoU if GT available (only when gate passes)
            iou = None
            if mask_np is not None:
                gt_mask = (mask_np == 3).astype(bool)
                iou = compute_iou(pred_mask_np, gt_mask)
        else:
            # Gate failed - use blank mask
            pred_mask_np = np.zeros_like(scores_np, dtype=bool)
            iou = None  # Don't compute IoU when gate fails

        # Save visualization
        save_frame_visualization(
            output_dir=output_dir / "visualizations",
            frame_idx=frame_info["frame_idx"],
            mesu_index=frame_info["mesu_index"],
            rgb_image=rgb_np,
            gt_mask=mask_np,
            pred_mask=pred_mask_np,
            image_score=image_score,
            threshold=optimal_threshold,
            passed_gate=passed_gate,
            iou=iou,
        )

        final_results.append(
            {
                "frame_idx": frame_info["frame_idx"],
                "mesu_index": frame_info["mesu_index"],
                "has_stones": frame_info["has_stones"],
                "image_score": float(image_score),
                "threshold": float(optimal_threshold),
                "passed_gate": bool(passed_gate),
                "pixels_detected": int(pred_mask_np.sum()),
                "iou": float(iou) if iou is not None else None,
            }
        )

    # Save comprehensive results JSON
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(
            {
                "config": {
                    "quantile": quantile,
                    "top_k_fraction": top_k_fraction,
                    "threshold_metric": threshold_metric,
                    "optimal_threshold": float(optimal_threshold),
                    "optimal_metric_value": float(optimal_metric_value),
                    "optimal_metrics": optimal_metrics,
                    "prompt_text": prompt_text,
                },
                "overall_metrics": {
                    "accuracy": float(all_accuracy),
                    "f1": float(all_f1),
                    "auc": float(all_auc),
                    "confusion_matrix": {
                        "tn": int(tn),
                        "fp": int(fp),
                        "fn": int(fn),
                        "tp": int(tp),
                    },
                },
                "results": final_results,
            },
            f,
            indent=2,
        )

    logger.info(f"✅ Results saved to: {results_file}")
    logger.info(f"✅ Visualizations saved to: {output_dir / 'visualizations'}")

    # Create summary plot
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Plot 1: Score distribution
    normal_scores = [r["image_score"] for r in final_results if not r["has_stones"]]
    anomaly_scores = [r["image_score"] for r in final_results if r["has_stones"]]

    axes[0].hist(normal_scores, bins=20, alpha=0.6, label="Normal", color="blue", edgecolor="black")
    axes[0].hist(
        anomaly_scores,
        bins=20,
        alpha=0.6,
        label="Anomaly",
        color="red",
        edgecolor="black",
        hatch="///",
    )
    axes[0].axvline(
        optimal_threshold,
        color="green",
        linestyle="--",
        linewidth=2,
        label=f"Threshold ({optimal_threshold:.4f})",
    )
    axes[0].set_xlabel("Image-Level Anomaly Score")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("Score Distribution")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot 2: ROC curve
    fpr, tpr, _ = roc_curve(all_labels, all_image_scores)
    axes[1].plot(fpr, tpr, label=f"ROC (AUC={all_auc:.3f})", linewidth=2)
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title("ROC Curve")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    summary_plot_path = output_dir / "summary_plots.png"
    plt.savefig(summary_plot_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Summary plots saved to: {summary_plot_path}")

    total_duration = time.perf_counter() - run_start
    logger.info("\n=== Experiment Complete ===")
    logger.info(f"Total duration: {total_duration:.2f} seconds")
    logger.info(f"Optimal threshold: {optimal_threshold:.6f}")
    logger.info(
        f"Final metrics - Accuracy: {all_accuracy:.4f}, F1: {all_f1:.4f}, AUC: {all_auc:.4f}"
    )


if __name__ == "__main__":
    main()
