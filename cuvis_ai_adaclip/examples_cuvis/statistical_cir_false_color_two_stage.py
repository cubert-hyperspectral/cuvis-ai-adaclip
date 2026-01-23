from __future__ import annotations

import time
from pathlib import Path

import click
from cuvis_ai.deciders.two_stage_decider import TwoStageBinaryDecider
from cuvis_ai.node.band_selection import CIRFalseColorSelector
from cuvis_ai.node.data import LentilsAnomalyDataNode
from cuvis_ai.node.metrics import AnomalyDetectionMetrics
from cuvis_ai.node.monitor import TensorBoardMonitorNode
from cuvis_ai.node.visualizations import RGBAnomalyMask, ScoreHeatmapVisualizer
from cuvis_ai_core.data.datasets import SingleCu3sDataModule
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline
from cuvis_ai_core.training import StatisticalTrainer
from cuvis_ai_core.training.config import (
    PipelineMetadata,
    TrainingConfig,
    TrainRunConfig,
)
from loguru import logger

from cuvis_ai_adaclip import (
    AdaCLIPDetector,
    download_weights,
    list_available_weights,
)
from cuvis_ai_adaclip.cli_utils import AdaCLIPCLI

cli = AdaCLIPCLI("AdaCLIP CIR False Color (Two-Stage)")


@cli.add_common_options
@cli.add_data_options
@cli.add_cir_options
@cli.add_visualization_options
@click.command()
@click.option("--image-threshold", type=float, default=0.5, help="Image-level top-k gate threshold")
@click.option(
    "--top-k-fraction",
    type=float,
    default=0.001,
    help="Fraction of pixels used for top-k mean (default 0.1%)",
)
def main(**kwargs) -> None:
    logger.info("=== AdaCLIP CIR false-color (two-stage) ===")
    run_start = time.perf_counter()

    output_dir = Path(kwargs["output_dir"])
    data_config = cli.parse_data_config(**kwargs)

    datamodule = SingleCu3sDataModule(**data_config)
    datamodule.setup(stage=None)

    wavelengths = datamodule.train_ds.wavelengths
    logger.info("Wavelength range: {:.1f}-{:.1f} nm", wavelengths.min(), wavelengths.max())

    model_name = kwargs["backbone_name"]
    weight_name = kwargs["pretrained_adaclip"]
    prompt_text = kwargs["prompt_text"]

    logger.info("Available AdaCLIP weights: {}", list_available_weights())
    download_weights(weight_name)

    quantile = kwargs["quantile"]
    visualize_upto = kwargs["visualize_upto"]
    gaussian_sigma = kwargs["gaussian_sigma"]
    image_threshold = kwargs["image_threshold"]
    top_k_fraction = kwargs["top_k_fraction"]

    nir_nm = kwargs["nir_nm"]
    red_nm = kwargs["red_nm"]
    green_nm = kwargs["green_nm"]

    logger.info("Quantile: {}", quantile)
    logger.info("Top-k fraction: {}", top_k_fraction)

    pipeline = CuvisPipeline("AdaCLIP_CIR_FalseColor_TwoStage")

    data_node = LentilsAnomalyDataNode(normal_class_ids=[0, 1])
    band_selector = CIRFalseColorSelector(nir_nm=nir_nm, red_nm=red_nm, green_nm=green_nm)

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

    decider = TwoStageBinaryDecider(
        image_threshold=image_threshold,
        top_k_fraction=top_k_fraction,
        quantile=quantile,
    )

    standard_metrics = AnomalyDetectionMetrics(name="detection_metrics")
    score_viz = ScoreHeatmapVisualizer(normalize_scores=True, up_to=visualize_upto)
    mask_viz = RGBAnomalyMask(up_to=visualize_upto)
    monitor = TensorBoardMonitorNode(
        run_name=pipeline.name,
        output_dir=str(output_dir / ".." / "tensorboard"),
    )

    pipeline.connect(
        (data_node.outputs.cube, band_selector.inputs.cube),
        (data_node.outputs.wavelengths, band_selector.inputs.wavelengths),
        (band_selector.outputs.rgb_image, adaclip.inputs.rgb_image),
        (adaclip.outputs.scores, decider.inputs.logits),
        (adaclip.outputs.scores, score_viz.inputs.scores),
        (adaclip.outputs.scores, mask_viz.inputs.scores),
        (decider.outputs.decisions, standard_metrics.inputs.decisions),
        (data_node.outputs.mask, standard_metrics.inputs.targets),
        (decider.outputs.decisions, mask_viz.inputs.decisions),
        (data_node.outputs.mask, mask_viz.inputs.mask),
        (band_selector.outputs.rgb_image, mask_viz.inputs.rgb_image),
        (standard_metrics.outputs.metrics, monitor.inputs.metrics),
        (score_viz.outputs.artifacts, monitor.inputs.artifacts),
        (mask_viz.outputs.artifacts, monitor.inputs.artifacts),
    )

    device = cli.get_device()
    logger.info(f"Moving pipeline to device: {device}")
    pipeline.to(device)

    trainer = StatisticalTrainer(pipeline=pipeline, datamodule=datamodule)

    val_duration = 0.0
    if data_config["val_ids"]:
        val_start = time.perf_counter()
        logger.info("Running validation...")
        trainer.validate()
        val_duration = time.perf_counter() - val_start
        logger.info(f"Validation duration: {val_duration:.2f} seconds")
    else:
        logger.info("Skipping validation (no val_ids provided)")

    test_start = time.perf_counter()
    logger.info("Running test...")
    trainer.test()
    test_duration = time.perf_counter() - test_start
    logger.info(f"Test duration: {test_duration:.2f} seconds")

    results_dir = output_dir / "trained_models"
    pipeline_metadata = PipelineMetadata(
        name=pipeline.name,
        description=("AdaCLIP CIR false-color pipeline with image-level gate + quantile mask"),
        tags=["statistical", "adaclip", "two_stage"],
        author="cuvis.ai",
    )

    pipeline_output_path = results_dir / f"{pipeline.name}.yaml"
    pipeline.save_to_file(str(pipeline_output_path), metadata=pipeline_metadata)

    pipeline_config_dir = Path("configs/pipeline")
    pipeline_config_dir.mkdir(parents=True, exist_ok=True)
    pipeline_config_path = pipeline_config_dir / "adaclip_cir_false_color_two_stage.yaml"
    pipeline.save_to_file(str(pipeline_config_path), metadata=pipeline_metadata)

    training_cfg = TrainingConfig()
    pipeline_config = pipeline.serialize()
    trainrun_config = TrainRunConfig(
        name="adaclip_cir_false_color_two_stage_cli",
        pipeline=pipeline_config,
        data=data_config,
        training=training_cfg,
        output_dir=str(output_dir),
        loss_nodes=[],
        metric_nodes=["detection_metrics"],
        freeze_nodes=[],
        unfreeze_nodes=[],
    )

    trainrun_output_path = results_dir / "adaclip_cir_false_color_two_stage_cli_trainrun.yaml"
    trainrun_config.save_to_file(str(trainrun_output_path))

    total_duration = time.perf_counter() - run_start
    logger.info("=== Experiment Complete ===")
    logger.info(
        f"Timing: validation {val_duration:.2f}s, test {test_duration:.2f}s, total {total_duration:.2f}s"
    )


if __name__ == "__main__":
    main()
