"""Two-stage threshold learning: image-level gate + pixel-level thresholding.

This script implements the two-stage decision strategy:
1. Image-level gate: decide if frame is anomalous at all
2. Pixel-level mask: only if image-level passes, apply pixel thresholding

Strategy:
- Calibration set: 5 normal + 5 stone frames (to find optimal thresholds)
- Validation set: 2 normal + 2 stone frames (to evaluate)
- Finds optimal image-level threshold using calibration set
- Applies two-stage decision and evaluates on validation set
- Saves all results, images, and thresholds

Supports multiple band selectors:
- CIR false RGB: Fixed CIR wavelengths
- Supervised CIR: Supervised band selection with CIR windows
- Supervised full spectrum: Supervised band selection across full spectrum
"""

from __future__ import annotations

import json
import time
from collections.abc import Generator
from pathlib import Path
from typing import Any

import click
import matplotlib.pyplot as plt
import numpy as np
import torch
from cuvis_ai.node.channel_selector import (
    CIRSelector,
    SupervisedCIRSelector,
    SupervisedFullSpectrumSelector,
)
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai_dataloader.data import Cu3sDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import StatisticalTrainer
from cuvis_ai_schemas.enums import ExecutionStage
from loguru import logger
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score, roc_curve

from cuvis_ai_adaclip import (
    AdaCLIPDetector,
    download_weights,
    list_available_weights,
)
from cuvis_ai_adaclip.cli_utils import AdaCLIPCLI

# Create reusable CLI instance
cli = AdaCLIPCLI("Two-Stage Threshold Learning")


def compute_image_score(anomaly_map: np.ndarray, method: str = "max", k: int = 100) -> float:
    """Compute image-level anomaly score from anomaly map.

    Args:
        anomaly_map: 2D array [H, W] with anomaly scores
        method: "max" or "top_k_mean"
        k: number of top pixels to average (for top_k_mean)

    Returns:
        Image-level anomaly score
    """
    if method == "max":
        return float(anomaly_map.max())
    elif method == "top_k_mean":
        flat = anomaly_map.flatten()
        top_k = np.partition(flat, -k)[-k:]
        return float(top_k.mean())
    else:
        raise ValueError(f"Unknown method: {method}")


def compute_relative_score_tail_minus_median(
    anomaly_map: np.ndarray, q_tail: float = 0.999, q_median: float = 0.5
) -> float:
    """Compute relative score: q_tail - q_median (tail minus median).

    This captures the contrast between high-anomaly regions and typical regions.

    Args:
        anomaly_map: 2D array [H, W] with anomaly scores
        q_tail: Quantile for tail (default: 0.999 = 99.9th percentile)
        q_median: Quantile for median (default: 0.5 = 50th percentile)

    Returns:
        Relative score (tail - median)
    """
    flat = anomaly_map.flatten()
    q_tail_val = float(np.quantile(flat, q_tail))
    q_median_val = float(np.quantile(flat, q_median))
    return q_tail_val - q_median_val


def compute_relative_score_tail_over_median(
    anomaly_map: np.ndarray, q_tail: float = 0.999, q_median: float = 0.5, eps: float = 1e-6
) -> float:
    """Compute relative score: q_tail / (q_median + eps) (tail divided by median).

    This captures the relative contrast between high-anomaly regions and typical regions.

    Args:
        anomaly_map: 2D array [H, W] with anomaly scores
        q_tail: Quantile for tail (default: 0.999 = 99.9th percentile)
        q_median: Quantile for median (default: 0.5 = 50th percentile)
        eps: Small epsilon to avoid division by zero (default: 1e-6)

    Returns:
        Relative score (tail / (median + eps))
    """
    flat = anomaly_map.flatten()
    q_tail_val = float(np.quantile(flat, q_tail))
    q_median_val = float(np.quantile(flat, q_median))
    return q_tail_val / (q_median_val + eps)


def resolve_top_k(num_pixels: int, top_k: int, fraction: float) -> int:
    """Resolve how many pixels to use for top_k_mean (either explicit or fraction)."""
    if top_k > 0:
        return top_k
    computed = int(np.ceil(fraction * num_pixels))
    return max(1, computed)


def find_optimal_threshold(
    y_true: np.ndarray, y_scores: np.ndarray, metric: str = "f1"
) -> tuple[float, float]:
    """Find optimal threshold using calibration set.

    Args:
        y_true: Binary labels (0=normal, 1=anomaly)
        y_scores: Anomaly scores
        metric: "f1", "youden" (max TPR - FPR), or "roc_optimal"

    Returns:
        (optimal_threshold, metric_value)
    """
    if metric == "f1":
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = (
            thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
        )
        return float(optimal_threshold), float(f1_scores[optimal_idx])

    elif metric == "youden":
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        youden_scores = tpr - fpr
        optimal_idx = np.argmax(youden_scores)
        optimal_threshold = (
            thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
        )
        return float(optimal_threshold), float(youden_scores[optimal_idx])

    elif metric == "roc_optimal":
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        # Find point closest to (0, 1) - perfect classifier
        distances = np.sqrt((fpr - 0) ** 2 + (tpr - 1) ** 2)
        optimal_idx = np.argmin(distances)
        optimal_threshold = (
            thresholds[optimal_idx] if optimal_idx < len(thresholds) else thresholds[-1]
        )
        return float(optimal_threshold), float(1.0 - distances[optimal_idx])

    else:
        raise ValueError(f"Unknown metric: {metric}")


def create_band_selector(
    selector_type: str,
    num_spectral_bands: int,
    nir_nm: float | None = None,
    red_nm: float | None = None,
    green_nm: float | None = None,
) -> Any:
    """Create band selector based on type.

    Args:
        selector_type: "cir_false", "supervised_cir", or "supervised_full"
        num_spectral_bands: Number of spectral bands in the dataset
        nir_nm: NIR wavelength for CIR false (optional)
        red_nm: Red wavelength for CIR false (optional)
        green_nm: Green wavelength for CIR false (optional)

    Returns:
        Band selector instance
    """
    if selector_type == "cir_false":
        if nir_nm is None or red_nm is None or green_nm is None:
            raise ValueError("nir_nm, red_nm, and green_nm required for cir_false selector")
        return CIRSelector(nir_nm=nir_nm, red_nm=red_nm, green_nm=green_nm)
    elif selector_type == "supervised_cir":
        return SupervisedCIRSelector(num_spectral_bands=num_spectral_bands)
    elif selector_type == "supervised_full":
        return SupervisedFullSpectrumSelector(num_spectral_bands=num_spectral_bands)
    else:
        raise ValueError(f"Unknown selector type: {selector_type}")


def run_analysis_for_band_selector(
    selector_type: str,
    datamodule: Cu3sDataModule,
    calibration_frames: list[dict],
    validation_frames: list[dict],
    output_dir: Path,
    kwargs: dict[str, Any],
    num_spectral_bands: int,
) -> dict[str, Any]:
    """Run two-stage threshold learning analysis for a specific band selector.

    Args:
        selector_type: "cir_false", "supervised_cir", or "supervised_full"
        datamodule: Data module with datasets
        calibration_frames: List of calibration frame info dicts
        validation_frames: List of validation frame info dicts
        output_dir: Base output directory
        kwargs: CLI arguments

    Returns:
        Dictionary with results summary
    """
    logger.info(f"\n{'=' * 60}")
    logger.info(f"Running analysis for: {selector_type}")
    logger.info(f"{'=' * 60}")

    # Create subdirectory for this selector
    selector_output_dir = output_dir / selector_type
    selector_output_dir.mkdir(parents=True, exist_ok=True)
    (selector_output_dir / "calibration").mkdir(exist_ok=True)
    (selector_output_dir / "validation").mkdir(exist_ok=True)

    # Extract parameters
    model_name = kwargs["backbone_name"]
    weight_name = kwargs["pretrained_adaclip"]
    prompt_text = kwargs["prompt_text"]
    quantile = kwargs["quantile"]
    image_score_method = kwargs["image_score_method"]
    top_k = kwargs["top_k"]
    top_k_fraction = kwargs["top_k_fraction"]
    threshold_metric = kwargs["threshold_metric"]
    use_calibrated_threshold = kwargs["use_calibrated_threshold"]
    image_threshold = kwargs["image_threshold"]
    nir_nm = kwargs.get("nir_nm")
    red_nm = kwargs.get("red_nm")
    green_nm = kwargs.get("green_nm")

    use_half_precision = kwargs.get("use_half_precision", True)
    enable_warmup = kwargs.get("enable_warmup", True)
    use_torch_preprocess = kwargs.get("use_torch_preprocess", True)
    image_size = 518
    gaussian_sigma = kwargs["gaussian_sigma"]

    device = cli.get_device()

    # Build pipeline
    pipeline = CuvisPipeline(f"TwoStage_Threshold_Learning_{selector_type}")
    data_node = LentilsAnomalyDataNode(normal_class_ids=[0, 1])

    # Create band selector
    band_selector = create_band_selector(
        selector_type, num_spectral_bands, nir_nm, red_nm, green_nm
    )

    # For supervised selectors, need to fit them first
    if selector_type in ["supervised_cir", "supervised_full"]:
        logger.info(f"Fitting {selector_type} band selector...")
        # Create a temporary pipeline just for fitting
        temp_pipeline = CuvisPipeline("temp_fit")
        temp_pipeline.connect(
            (data_node.outputs.cube, band_selector.inputs.cube),
            (data_node.outputs.wavelengths, band_selector.inputs.wavelengths),
            (data_node.outputs.mask, band_selector.inputs.mask),  # Supervised selectors need mask
        )
        temp_pipeline.to(device)

        # Use StatisticalTrainer to fit the band selector
        # If train_ds is not available, we need to create a temporary datamodule with calibration frames
        if datamodule.train_ds is not None:
            trainer = StatisticalTrainer(pipeline=temp_pipeline, datamodule=datamodule)
            trainer.fit()
            logger.info(f"✅ {selector_type} band selector fitted using train_ds")
        else:
            # When train_ds is None (test_only mode), create input stream from calibration frames
            # The input stream is a generator that yields dicts matching band_selector.INPUT_SPECS
            logger.info("No train_ds available, using calibration frames for fitting...")

            def create_input_stream_from_frames() -> Generator[dict[str, Any], None, None]:
                """Create input stream generator from calibration frames (similar to StatisticalTrainer._create_input_stream)."""
                for frame_info in calibration_frames + validation_frames:
                    sample = frame_info["sample"]
                    cube_np = np.asarray(sample["cube"])
                    wl_np = np.asarray(sample["wavelengths"])
                    mask_np = sample.get("mask", None)

                    # Create batch dict
                    batch = {
                        "cube": torch.from_numpy(cube_np).unsqueeze(0).to(device),
                        "wavelengths": torch.from_numpy(wl_np.astype(np.int32))
                        .unsqueeze(0)
                        .to(device),
                    }
                    if mask_np is not None:
                        batch["mask"] = torch.from_numpy(mask_np).unsqueeze(0).to(device)

                    # Execute pipeline up to band_selector to get transformed inputs
                    outputs = temp_pipeline.forward(
                        batch=batch,
                        stage=ExecutionStage.INFERENCE,
                        upto_node=band_selector,
                    )

                    # Gather inputs for band_selector from predecessor outputs
                    node_inputs = {}
                    predecessors = list(temp_pipeline._graph.predecessors(band_selector))

                    if not predecessors:
                        # Entry node - get directly from batch
                        for port_name in band_selector.INPUT_SPECS:
                            if port_name in batch:
                                node_inputs[port_name] = batch[port_name]
                    else:
                        # Get from parent outputs via graph edges
                        for parent_node in predecessors:
                            for edge_data in temp_pipeline._graph[parent_node][
                                band_selector
                            ].values():
                                from_port = edge_data["from_port"]
                                to_port = edge_data["to_port"]
                                node_inputs[to_port] = outputs[(parent_node.name, from_port)]

                    yield node_inputs

            # Create input stream and fit
            input_stream = create_input_stream_from_frames()
            band_selector.statistical_initialization(input_stream)
            logger.info(f"✅ {selector_type} band selector fitted using calibration frames")

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

    # Wire pipeline
    if selector_type in ["supervised_cir", "supervised_full"]:
        # Supervised selectors need mask input
        pipeline.connect(
            (data_node.outputs.cube, band_selector.inputs.cube),
            (data_node.outputs.wavelengths, band_selector.inputs.wavelengths),
            (data_node.outputs.mask, band_selector.inputs.mask),  # Supervised selectors need mask
            (band_selector.outputs.rgb_image, adaclip.inputs.rgb_image),
        )
    else:
        # CIR false selector doesn't need mask
        pipeline.connect(
            (data_node.outputs.cube, band_selector.inputs.cube),
            (data_node.outputs.wavelengths, band_selector.inputs.wavelengths),
            (band_selector.outputs.rgb_image, adaclip.inputs.rgb_image),
        )

    pipeline.to(device)

    # Process calibration set
    logger.info(f"\n=== Processing calibration set ({selector_type}) ===")
    calibration_results = []

    for frame_info in calibration_frames:
        sample = frame_info["sample"]
        cube_np = np.asarray(sample["cube"])
        wl_np = np.asarray(sample["wavelengths"])
        mask_np = sample.get("mask", None)

        cube_t = torch.from_numpy(cube_np).unsqueeze(0).to(device)
        wl_t = torch.from_numpy(wl_np.astype(np.int32)).unsqueeze(0).to(device)

        batch = {
            "cube": cube_t,
            "wavelengths": wl_t,
            "mask": torch.from_numpy(mask_np).unsqueeze(0).to(device)
            if mask_np is not None
            else None,
        }

        with torch.no_grad():
            outputs = pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)

        scores = outputs[(adaclip.name, "scores")]
        scores_np = scores.squeeze().cpu().numpy()

        anomaly_score = outputs[(adaclip.name, "anomaly_score")]
        anomaly_score_np = anomaly_score.squeeze().cpu().item()

        rgb_image = outputs[(band_selector.name, "rgb_image")]
        rgb_np = rgb_image[0].cpu().numpy()

        resolved_top_k = resolve_top_k(scores_np.size, top_k, top_k_fraction)
        image_score = compute_image_score(scores_np, method=image_score_method, k=resolved_top_k)

        # Compute relative scores
        relative_score_diff = compute_relative_score_tail_minus_median(scores_np)
        relative_score_ratio = compute_relative_score_tail_over_median(scores_np)

        scores_flat = scores_np.flatten()
        pixel_threshold = float(np.quantile(scores_flat, quantile))

        calibration_results.append(
            {
                "frame_info": frame_info,
                "anomaly_map": scores_np,
                "rgb_image": rgb_np,
                "mask": mask_np,
                "image_score": image_score,
                "anomaly_score": anomaly_score_np,
                "relative_score_tail_minus_median": relative_score_diff,
                "relative_score_tail_over_median": relative_score_ratio,
                "pixel_threshold": pixel_threshold,
            }
        )

    # Find optimal threshold
    logger.info(f"\n=== Finding optimal image-level threshold ({selector_type}) ===")
    cal_image_scores = np.array([r["image_score"] for r in calibration_results])
    cal_labels = np.array([1 if r["frame_info"]["has_stones"] else 0 for r in calibration_results])

    if use_calibrated_threshold:
        optimal_img_threshold, optimal_metric_value = find_optimal_threshold(
            cal_labels, cal_image_scores, metric=threshold_metric
        )
        logger.info(f"Optimal image-level threshold: {optimal_img_threshold:.6f}")
    else:
        optimal_img_threshold = image_threshold
        optimal_metric_value = None
        logger.info(f"Using fixed image-level threshold: {optimal_img_threshold:.6f}")

    cal_predictions = (cal_image_scores >= optimal_img_threshold).astype(int)
    cal_accuracy = (cal_predictions == cal_labels).mean()
    cal_f1 = f1_score(cal_labels, cal_predictions)
    cal_auc = roc_auc_score(cal_labels, cal_image_scores)

    logger.info(f"Calibration - Accuracy: {cal_accuracy:.4f}, F1: {cal_f1:.4f}, AUC: {cal_auc:.4f}")

    # Process validation set
    logger.info(f"\n=== Processing validation set ({selector_type}) ===")
    validation_results = []

    for frame_info in validation_frames:
        sample = frame_info["sample"]
        cube_np = np.asarray(sample["cube"])
        wl_np = np.asarray(sample["wavelengths"])
        mask_np = sample.get("mask", None)

        cube_t = torch.from_numpy(cube_np).unsqueeze(0).to(device)
        wl_t = torch.from_numpy(wl_np.astype(np.int32)).unsqueeze(0).to(device)

        batch = {
            "cube": cube_t,
            "wavelengths": wl_t,
            "mask": torch.from_numpy(mask_np).unsqueeze(0).to(device)
            if mask_np is not None
            else None,
        }

        with torch.no_grad():
            outputs = pipeline.forward(batch=batch, stage=ExecutionStage.INFERENCE)

        scores = outputs[(adaclip.name, "scores")]
        scores_np = scores.squeeze().cpu().numpy()

        anomaly_score = outputs[(adaclip.name, "anomaly_score")]
        anomaly_score_np = anomaly_score.squeeze().cpu().item()

        rgb_image = outputs[(band_selector.name, "rgb_image")]
        rgb_np = rgb_image[0].cpu().numpy()

        resolved_top_k = resolve_top_k(scores_np.size, top_k, top_k_fraction)
        image_score = compute_image_score(scores_np, method=image_score_method, k=resolved_top_k)

        # Compute relative scores
        relative_score_diff = compute_relative_score_tail_minus_median(scores_np)
        relative_score_ratio = compute_relative_score_tail_over_median(scores_np)

        if image_score < optimal_img_threshold:
            pixel_mask = np.zeros_like(scores_np, dtype=bool)
            pixel_threshold = None
            passed_image_gate = False
        else:
            scores_flat = scores_np.flatten()
            pixel_threshold = float(np.quantile(scores_flat, quantile))
            pixel_mask = scores_np >= pixel_threshold
            passed_image_gate = True

        validation_results.append(
            {
                "frame_info": frame_info,
                "anomaly_map": scores_np,
                "rgb_image": rgb_np,
                "mask": mask_np,
                "image_score": image_score,
                "anomaly_score": anomaly_score_np,
                "relative_score_tail_minus_median": relative_score_diff,
                "relative_score_tail_over_median": relative_score_ratio,
                "pixel_threshold": pixel_threshold,
                "pixel_mask": pixel_mask,
                "passed_image_gate": passed_image_gate,
            }
        )

    # Evaluate validation
    val_image_scores = np.array([r["image_score"] for r in validation_results])
    val_labels = np.array([1 if r["frame_info"]["has_stones"] else 0 for r in validation_results])

    val_predictions = (val_image_scores >= optimal_img_threshold).astype(int)
    val_accuracy = (val_predictions == val_labels).mean()
    val_f1 = f1_score(val_labels, val_predictions)
    val_auc = roc_auc_score(val_labels, val_image_scores)

    logger.info(f"Validation - Accuracy: {val_accuracy:.4f}, F1: {val_f1:.4f}, AUC: {val_auc:.4f}")

    # Pixel-level metrics
    val_pixel_metrics = []
    for r in validation_results:
        if r["passed_image_gate"] and r["mask"] is not None:
            gt_mask = (r["mask"] == 3).astype(bool)
            pred_mask = r["pixel_mask"]
            if gt_mask.any():
                intersection = (gt_mask & pred_mask).sum()
                union = (gt_mask | pred_mask).sum()
                iou = intersection / union if union > 0 else 0.0
                val_pixel_metrics.append(
                    {
                        "frame_idx": int(r["frame_info"]["frame_idx"]),
                        "iou": float(iou),
                        "pixels_detected": int(pred_mask.sum()),
                        "pixels_gt": int(gt_mask.sum()),
                    }
                )

    # Save results
    config = {
        "selector_type": selector_type,
        "quantile": quantile,
        "image_score_method": image_score_method,
        "top_k": top_k,
        "top_k_fraction": top_k_fraction,
        "threshold_metric": threshold_metric,
        "optimal_image_threshold": optimal_img_threshold,
        "optimal_metric_value": optimal_metric_value,
        "use_calibrated_threshold": use_calibrated_threshold,
        "image_threshold": image_threshold,
        "calibration_set": {
            "accuracy": float(cal_accuracy),
            "f1": float(cal_f1),
            "auc": float(cal_auc),
        },
        "validation_set": {
            "accuracy": float(val_accuracy),
            "f1": float(val_f1),
            "auc": float(val_auc),
            "pixel_metrics": val_pixel_metrics,
        },
    }

    config_file = selector_output_dir / "two_stage_config.json"
    with open(config_file, "w") as f:
        json.dump(config, f, indent=2)

    # Save calibration results
    cal_results_data = []
    for r in calibration_results:
        cal_results_data.append(
            {
                "frame_idx": r["frame_info"]["frame_idx"],
                "mesu_index": r["frame_info"]["mesu_index"],
                "has_stones": bool(r["frame_info"]["has_stones"]),
                "image_score": float(r["image_score"]),  # Computed (max or top_k_mean)
                "anomaly_score": float(r["anomaly_score"]),  # AdaCLIP's built-in image-level score
                "relative_score_tail_minus_median": float(
                    r["relative_score_tail_minus_median"]
                ),  # q99.9 - q50
                "relative_score_tail_over_median": float(
                    r["relative_score_tail_over_median"]
                ),  # q99.9 / (q50 + eps)
                "pixel_threshold": float(r["pixel_threshold"]),
            }
        )
    with open(selector_output_dir / "calibration_results.json", "w") as f:
        json.dump(cal_results_data, f, indent=2)

    # Save validation results
    val_results_data = []
    for r in validation_results:
        val_results_data.append(
            {
                "frame_idx": int(r["frame_info"]["frame_idx"]),
                "mesu_index": int(r["frame_info"]["mesu_index"]),
                "has_stones": bool(r["frame_info"]["has_stones"]),
                "image_score": float(r["image_score"]),  # Computed (max or top_k_mean)
                "anomaly_score": float(r["anomaly_score"]),  # AdaCLIP's built-in image-level score
                "relative_score_tail_minus_median": float(
                    r["relative_score_tail_minus_median"]
                ),  # q99.9 - q50
                "relative_score_tail_over_median": float(
                    r["relative_score_tail_over_median"]
                ),  # q99.9 / (q50 + eps)
                "pixel_threshold": float(r["pixel_threshold"])
                if r["pixel_threshold"] is not None
                else None,
                "passed_image_gate": bool(r["passed_image_gate"]),
                "pixels_detected": int(r["pixel_mask"].sum()),
            }
        )
    with open(selector_output_dir / "validation_results.json", "w") as f:
        json.dump(val_results_data, f, indent=2)

    # Create visualizations
    # ROC curve
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    cal_fpr, cal_tpr, _ = roc_curve(cal_labels, cal_image_scores)
    ax.plot(cal_fpr, cal_tpr, label=f"Calibration (AUC={cal_auc:.3f})", linewidth=2)
    val_fpr, val_tpr, _ = roc_curve(val_labels, val_image_scores)
    ax.plot(val_fpr, val_tpr, label=f"Validation (AUC={val_auc:.3f})", linewidth=2, linestyle="--")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(f"ROC Curve - {selector_type}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(selector_output_dir / "roc_curve.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Score distribution - main image score
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cal_scores_normal = [
        r["image_score"] for r in calibration_results if not r["frame_info"]["has_stones"]
    ]
    cal_scores_anomaly = [
        r["image_score"] for r in calibration_results if r["frame_info"]["has_stones"]
    ]
    val_scores_normal = [
        r["image_score"] for r in validation_results if not r["frame_info"]["has_stones"]
    ]
    val_scores_anomaly = [
        r["image_score"] for r in validation_results if r["frame_info"]["has_stones"]
    ]

    # Use distinct colors and styles for all 4 groups
    ax.hist(
        cal_scores_normal,
        bins=20,
        alpha=0.6,
        label="Calibration Normal",
        color="blue",
        edgecolor="black",
        linewidth=1.5,
    )
    ax.hist(
        cal_scores_anomaly,
        bins=20,
        alpha=0.6,
        label="Calibration Anomaly",
        color="red",
        edgecolor="black",
        hatch="///",
        linewidth=1.5,
    )
    ax.hist(
        val_scores_normal,
        bins=20,
        label="Validation Normal",
        color="green",
        edgecolor="darkgreen",
        histtype="step",
        linewidth=3,
        linestyle="-",
    )
    ax.hist(
        val_scores_anomaly,
        bins=20,
        label="Validation Anomaly",
        color="purple",
        edgecolor="darkviolet",
        histtype="step",
        linewidth=3,
        linestyle="--",
    )
    ax.axvline(
        optimal_img_threshold,
        color="orange",
        linestyle=":",
        linewidth=2.5,
        label=f"Threshold ({optimal_img_threshold:.4f})",
    )
    ax.set_xlabel("Image-Level Anomaly Score")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Score Distribution - {selector_type} (Main Score)")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(selector_output_dir / "image_score_distribution.png", dpi=150, bbox_inches="tight")
    plt.close()

    # Relative score: tail minus median (q99.9 - q50)
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cal_rel_diff_normal = [
        r["relative_score_tail_minus_median"]
        for r in calibration_results
        if not r["frame_info"]["has_stones"]
    ]
    cal_rel_diff_anomaly = [
        r["relative_score_tail_minus_median"]
        for r in calibration_results
        if r["frame_info"]["has_stones"]
    ]
    val_rel_diff_normal = [
        r["relative_score_tail_minus_median"]
        for r in validation_results
        if not r["frame_info"]["has_stones"]
    ]
    val_rel_diff_anomaly = [
        r["relative_score_tail_minus_median"]
        for r in validation_results
        if r["frame_info"]["has_stones"]
    ]

    # Use distinct colors and styles for all 4 groups
    ax.hist(
        cal_rel_diff_normal,
        bins=20,
        alpha=0.6,
        label="Calibration Normal",
        color="blue",
        edgecolor="black",
        linewidth=1.5,
    )
    ax.hist(
        cal_rel_diff_anomaly,
        bins=20,
        alpha=0.6,
        label="Calibration Anomaly",
        color="red",
        edgecolor="black",
        hatch="///",
        linewidth=1.5,
    )
    ax.hist(
        val_rel_diff_normal,
        bins=20,
        label="Validation Normal",
        color="green",
        edgecolor="darkgreen",
        histtype="step",
        linewidth=3,
        linestyle="-",
    )
    ax.hist(
        val_rel_diff_anomaly,
        bins=20,
        label="Validation Anomaly",
        color="purple",
        edgecolor="darkviolet",
        histtype="step",
        linewidth=3,
        linestyle="--",
    )
    ax.set_xlabel("Relative Score (q99.9 - q50)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Relative Score Distribution - {selector_type}\n(Tail minus Median)")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        selector_output_dir / "relative_score_tail_minus_median_distribution.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # Relative score: tail over median (q99.9 / (q50 + eps))
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    cal_rel_ratio_normal = [
        r["relative_score_tail_over_median"]
        for r in calibration_results
        if not r["frame_info"]["has_stones"]
    ]
    cal_rel_ratio_anomaly = [
        r["relative_score_tail_over_median"]
        for r in calibration_results
        if r["frame_info"]["has_stones"]
    ]
    val_rel_ratio_normal = [
        r["relative_score_tail_over_median"]
        for r in validation_results
        if not r["frame_info"]["has_stones"]
    ]
    val_rel_ratio_anomaly = [
        r["relative_score_tail_over_median"]
        for r in validation_results
        if r["frame_info"]["has_stones"]
    ]

    # Use distinct colors and styles for all 4 groups
    ax.hist(
        cal_rel_ratio_normal,
        bins=20,
        alpha=0.6,
        label="Calibration Normal",
        color="blue",
        edgecolor="black",
        linewidth=1.5,
    )
    ax.hist(
        cal_rel_ratio_anomaly,
        bins=20,
        alpha=0.6,
        label="Calibration Anomaly",
        color="red",
        edgecolor="black",
        hatch="///",
        linewidth=1.5,
    )
    ax.hist(
        val_rel_ratio_normal,
        bins=20,
        label="Validation Normal",
        color="green",
        edgecolor="darkgreen",
        histtype="step",
        linewidth=3,
        linestyle="-",
    )
    ax.hist(
        val_rel_ratio_anomaly,
        bins=20,
        label="Validation Anomaly",
        color="purple",
        edgecolor="darkviolet",
        histtype="step",
        linewidth=3,
        linestyle="--",
    )
    ax.set_xlabel("Relative Score (q99.9 / (q50 + eps))")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Relative Score Distribution - {selector_type}\n(Tail over Median)")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(
        selector_output_dir / "relative_score_tail_over_median_distribution.png",
        dpi=150,
        bbox_inches="tight",
    )
    plt.close()

    # ROC curves for all score types
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))

    # Main image score ROC
    cal_fpr_main, cal_tpr_main, _ = roc_curve(cal_labels, cal_image_scores)
    cal_auc_main = roc_auc_score(cal_labels, cal_image_scores)
    val_fpr_main, val_tpr_main, _ = roc_curve(val_labels, val_image_scores)
    val_auc_main = roc_auc_score(val_labels, val_image_scores)
    axes[0].plot(
        cal_fpr_main, cal_tpr_main, label=f"Calibration (AUC={cal_auc_main:.3f})", linewidth=2
    )
    axes[0].plot(
        val_fpr_main,
        val_tpr_main,
        label=f"Validation (AUC={val_auc_main:.3f})",
        linewidth=2,
        linestyle="--",
    )
    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title(f"ROC - Main Image Score\n({selector_type})")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Relative score diff ROC
    cal_rel_diff_scores = np.array(
        [r["relative_score_tail_minus_median"] for r in calibration_results]
    )
    val_rel_diff_scores = np.array(
        [r["relative_score_tail_minus_median"] for r in validation_results]
    )
    cal_fpr_diff, cal_tpr_diff, _ = roc_curve(cal_labels, cal_rel_diff_scores)
    cal_auc_diff = roc_auc_score(cal_labels, cal_rel_diff_scores)
    val_fpr_diff, val_tpr_diff, _ = roc_curve(val_labels, val_rel_diff_scores)
    val_auc_diff = roc_auc_score(val_labels, val_rel_diff_scores)
    axes[1].plot(
        cal_fpr_diff, cal_tpr_diff, label=f"Calibration (AUC={cal_auc_diff:.3f})", linewidth=2
    )
    axes[1].plot(
        val_fpr_diff,
        val_tpr_diff,
        label=f"Validation (AUC={val_auc_diff:.3f})",
        linewidth=2,
        linestyle="--",
    )
    axes[1].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    axes[1].set_xlabel("False Positive Rate")
    axes[1].set_ylabel("True Positive Rate")
    axes[1].set_title(f"ROC - Tail minus Median\n({selector_type})")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    # Relative score ratio ROC
    cal_rel_ratio_scores = np.array(
        [r["relative_score_tail_over_median"] for r in calibration_results]
    )
    val_rel_ratio_scores = np.array(
        [r["relative_score_tail_over_median"] for r in validation_results]
    )
    cal_fpr_ratio, cal_tpr_ratio, _ = roc_curve(cal_labels, cal_rel_ratio_scores)
    cal_auc_ratio = roc_auc_score(cal_labels, cal_rel_ratio_scores)
    val_fpr_ratio, val_tpr_ratio, _ = roc_curve(val_labels, val_rel_ratio_scores)
    val_auc_ratio = roc_auc_score(val_labels, val_rel_ratio_scores)
    axes[2].plot(
        cal_fpr_ratio, cal_tpr_ratio, label=f"Calibration (AUC={cal_auc_ratio:.3f})", linewidth=2
    )
    axes[2].plot(
        val_fpr_ratio,
        val_tpr_ratio,
        label=f"Validation (AUC={val_auc_ratio:.3f})",
        linewidth=2,
        linestyle="--",
    )
    axes[2].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Random")
    axes[2].set_xlabel("False Positive Rate")
    axes[2].set_ylabel("True Positive Rate")
    axes[2].set_title(f"ROC - Tail over Median\n({selector_type})")
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(selector_output_dir / "roc_curves_all_scores.png", dpi=150, bbox_inches="tight")
    plt.close()

    logger.info(f"✅ Results saved to: {selector_output_dir}")

    return {
        "selector_type": selector_type,
        "cal_accuracy": float(cal_accuracy),
        "cal_f1": float(cal_f1),
        "cal_auc": float(cal_auc),
        "val_accuracy": float(val_accuracy),
        "val_f1": float(val_f1),
        "val_auc": float(val_auc),
        "optimal_threshold": float(optimal_img_threshold),
    }


@cli.add_common_options
@cli.add_data_options
@cli.add_cir_options
@click.command()
@click.option(
    "--quantile", type=float, default=0.995, help="Quantile for pixel-level threshold calculation"
)
@click.option(
    "--output-dir",
    type=str,
    default="outputs/two_stage_threshold",
    help="Output directory for results",
)
@click.option(
    "--test-only/--no-test-only",
    default=True,
    help="Skip training/validation splits and put frames 0-13 into the test split",
)
@click.option(
    "--image-score-method",
    type=click.Choice(["max", "top_k_mean"]),
    default="top_k_mean",
    help="Method for computing image-level score",
)
@click.option(
    "--top-k", type=int, default=0, help="Top-k pixels (set 0 to use --top-k-fraction of pixels)"
)
@click.option(
    "--top-k-fraction",
    type=float,
    default=0.001,
    help="Fraction of pixels to average when using top_k_mean (0.1% -> 0.001)",
)
@click.option(
    "--threshold-metric",
    type=click.Choice(["f1", "youden", "roc_optimal"]),
    default="f1",
    help="Metric for finding optimal image-level threshold",
)
@click.option(
    "--use-calibrated-threshold/--no-use-calibrated-threshold",
    default=False,
    help="Use calibration data to learn the image threshold instead of fixed --image-threshold",
)
@click.option(
    "--image-threshold",
    type=float,
    default=0.5,
    help="Fixed image-level threshold when not calibrating",
)
@click.option(
    "--calibration-normal", type=int, default=5, help="Number of normal frames for calibration"
)
@click.option(
    "--calibration-anomaly", type=int, default=5, help="Number of anomaly frames for calibration"
)
@click.option(
    "--band-selector",
    type=click.Choice(["cir_false", "supervised_cir", "supervised_full", "all"]),
    default="all",
    help="Band selector to use: cir_false, supervised_cir, supervised_full, or all",
)
def main(**kwargs) -> None:
    """Learn two-stage thresholds and evaluate on validation set."""
    logger.info("=== Two-Stage Threshold Learning ===")
    run_start = time.perf_counter()

    output_dir = Path(kwargs["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = cli.parse_data_config(**kwargs)

    if kwargs.get("test_only", False):
        forced_test_ids = list(range(0, 14))
        data_config["train_ids"] = []
        data_config["val_ids"] = []
        data_config["test_ids"] = forced_test_ids
        logger.info("Forcing only test split. Frames 0-13 will be treated as test data.")

    # Setup data
    datamodule = Cu3sDataModule(**data_config)
    datamodule.setup(stage=None)

    if datamodule.train_ds is not None:
        wavelengths = datamodule.train_ds.wavelengths
    elif datamodule.val_ds is not None:
        wavelengths = datamodule.val_ds.wavelengths
    elif datamodule.test_ds is not None:
        wavelengths = datamodule.test_ds.wavelengths
    else:
        raise ValueError("No dataset available!")

    logger.info("Wavelength range: {:.1f}-{:.1f} nm", wavelengths.min(), wavelengths.max())

    # Get number of spectral bands
    num_spectral_bands = len(wavelengths)
    logger.info("Number of spectral bands: {}", num_spectral_bands)

    logger.info("Available AdaCLIP weights: {}", list_available_weights())
    download_weights(kwargs["pretrained_adaclip"])

    # Collect frames
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

    np.random.seed(42)
    np.random.shuffle(normal_frames)
    np.random.shuffle(anomaly_frames)

    calibration_normal = kwargs["calibration_normal"]
    calibration_anomaly = kwargs["calibration_anomaly"]
    calibration_frames = normal_frames[:calibration_normal] + anomaly_frames[:calibration_anomaly]
    validation_frames = (
        normal_frames[calibration_normal : calibration_normal + 2]
        + anomaly_frames[calibration_anomaly : calibration_anomaly + 2]
    )

    logger.info(
        f"Calibration: {len(calibration_frames)} frames ({calibration_normal} normal, {calibration_anomaly} anomaly)"
    )
    logger.info(f"Validation: {len(validation_frames)} frames (2 normal, 2 anomaly)")

    # Determine which selectors to run
    band_selector_option = kwargs["band_selector"]
    if band_selector_option == "all":
        selectors_to_run = ["cir_false", "supervised_cir", "supervised_full"]
    else:
        selectors_to_run = [band_selector_option]

    # Run analysis for each selector
    all_results = []
    for selector_type in selectors_to_run:
        try:
            result = run_analysis_for_band_selector(
                selector_type=selector_type,
                datamodule=datamodule,
                calibration_frames=calibration_frames,
                validation_frames=validation_frames,
                output_dir=output_dir,
                kwargs=kwargs,
                num_spectral_bands=num_spectral_bands,
            )
            all_results.append(result)
        except Exception as e:
            logger.error(f"❌ Failed to run analysis for {selector_type}: {e}")
            import traceback

            traceback.print_exc()

    # Create comparison summary
    if len(all_results) > 1:
        logger.info("\n" + "=" * 60)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 60)

        comparison = {
            "results": all_results,
            "summary": {
                "best_cal_auc": max(all_results, key=lambda x: x["cal_auc"]),
                "best_val_auc": max(all_results, key=lambda x: x["val_auc"]),
                "best_cal_f1": max(all_results, key=lambda x: x["cal_f1"]),
                "best_val_f1": max(all_results, key=lambda x: x["val_f1"]),
            },
        }

        logger.info("\nCalibration Set:")
        for r in all_results:
            logger.info(
                f"  {r['selector_type']:20s} - AUC: {r['cal_auc']:.4f}, F1: {r['cal_f1']:.4f}, Acc: {r['cal_accuracy']:.4f}"
            )

        logger.info("\nValidation Set:")
        for r in all_results:
            logger.info(
                f"  {r['selector_type']:20s} - AUC: {r['val_auc']:.4f}, F1: {r['val_f1']:.4f}, Acc: {r['val_accuracy']:.4f}"
            )

        with open(output_dir / "comparison_summary.json", "w") as f:
            json.dump(comparison, f, indent=2)
        logger.info(f"\n✅ Comparison summary saved to: {output_dir / 'comparison_summary.json'}")

    total_duration = time.perf_counter() - run_start
    logger.info("\n" + "=" * 60)
    logger.info("=== Two-Stage Threshold Learning Complete ===")
    logger.info(f"Total duration: {total_duration:.2f} seconds")
    logger.info(f"Results saved to: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
