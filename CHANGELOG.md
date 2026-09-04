# Changelog

## 0.4.0 - 2026-09-04

- **Weights come from the `cubert-gmbh` Hugging Face mirrors through cuvis-ai-core's weight registry.** `download_weights` resolves the AdaCLIP heads with `ModelWeights.resolve` (`cubert-gmbh/adaclip`, the upstream Drive files unchanged) and the CLIP ViT-L/14@336px backbone is materialized from `cubert-gmbh/clip` into the directory the vendored OpenCLIP loader reads (`$CUVIS_MODEL_CACHE_DIR/clip`, else `~/.cache/clip`), so no Google Drive, `gdown` or OpenAI CDN access is needed and the offline child loads from the provisioned cache (`download-model download adaclip_all` and `download-model download clip_vit_l_14_336`). `gdown` is no longer a dependency, `get_local_weight_path` reports the cached mirror file, and the weight keys are unchanged (`pretrained_all`, `pretrained_mvtec_colondb`, `pretrained_visa_clinicdb`; the upstream README table labels the two dataset heads the other way round, while the file names and the README's Train section agree with the keys). Requires cuvis-ai-core 0.16.0. CLIP backbones other than ViT-L-14-336 still download from OpenAI.

## 0.3.2 - 2026-09-04

- `LossNode` declares its execution stages on the class (`EXECUTION_STAGES = {TRAIN, VAL, TEST}`, cuvis-ai-core 0.14.1) instead of passing `execution_stages=` to the constructor; the `execution_stages` guard assert is gone. Floors `cuvis-ai-core>=0.14.1`, which moves the locked `cuvis-ai-schemas` to 0.10.0.
- Dropped `leakage_check="off"` from the CLI split config (`parse_data_config`): `cuvis-ai-schemas` 0.9 replaced the field with a `constraints` list whose empty default declares no checks, so the intentional split overlap of the examples is unchanged and the config loads again. Floors `cuvis-ai-schemas>=0.10.0` to match.
- Resurrected the `examples/adaclip/` scripts against the current stack and added an import smoke test so they cannot rot silently again. They imported several removed symbols: `cuvis_ai.data.MultiFileCu3sDataModule` (now `cuvis_ai_dataloader.data.MultiCu3sDataModule`), `cuvis_ai_core.data.datasets.SingleCu3sDataModule` (now `cuvis_ai_dataloader.data.Cu3sDataModule`), and `cuvis_ai.training.MultiFileTrainRunConfig` (removed; the scripts already persist the pipeline via `save_to_file`, so the extra trainrun-config save was dropped). The multi-file cu3s scripts now pass `universe_csv` (was `splits_csv`) and no longer pass `pin_memory` / `persistent_workers` / `worker_multiprocessing_context`, which `MultiCu3sDataModule` rejects.
- Renamed the `splits_csv` trainrun key to `universe_csv` in the lentils cu3s configs (`finetune_adaclip`, `lentils_concrete_adaclip`, `lentils_drcnn_adaclip`), matching the dataloader's unified `universe.csv` vocabulary.
- Added `tests/test_examples_importable.py`, which imports every `examples/adaclip/*.py` module in the existing test job so a dead import fails CI. Added `cuvis-ai-dataloader>=0.4.0` to the `dev` extra for it; runtime still provisions the plugin via `configs/plugins/cuvis_ai_dataloader.yaml` rather than a package dependency. Bump the floor to `>=0.5.0` once the universe.csv unification releases, since the cu3s examples pass `universe_csv`.

## 0.3.1 - 2026-08-20

- Removed the dead `[tool.uv.sources]` / `[[tool.uv.index]]` torch cu128 configuration: torch is not a direct dependency of this package and the committed lock resolves it from PyPI, so the tables had no effect anywhere (uv honours them only at the resolution root). Composed child environments receive the host-mirrored torch build from `cuvis-ai-core>=0.12.1`.

## 0.3.0 - 2026-07-22

- Declared node-catalog metadata on the loss base: `LossNode` (and thus `AdaCLIPFocalDiceLoss`) now sets `category = loss` and tags `[differentiable, torch, training]`, so the loss node self-describes in the cuvis-ai node catalog instead of relying on hand-written manifest metadata.

## 0.2.0 - 2026-07-17

- Raised the framework floors to `cuvis-ai-core>=0.11.2` / `cuvis-ai-schemas>=0.8.0`, which fold `TrainerConfig` into a flat `TrainingConfig`. The `examples_cuvis/` scripts construct `TrainingConfig()` with defaults and never reached into the old nested `trainer`, so no example code changed.
- Added a `no-local-sources` CI workflow that fails if `pyproject.toml` declares a local `[tool.uv.sources]` path entry (a machine-specific path must not ship in a release).
- Added training-path support in `AdaCLIPDetector` for non-aggregated outputs (`per_layer_scores`, `image_score_2ch`) while preserving aggregated inference behavior.
- Added a `training_aggregation` constructor parameter (default `True`) to control train-time per-layer vs aggregated behavior.
- Added selective adapter-layer freeze/unfreeze so `pipeline.unfreeze_nodes_by_name(["adaclip"])` updates adapter module groups while keeping the CLIP backbone frozen.
- Added prompt template coverage for `clean / foreign object / contamination` patterns and fixed abnormal prompt duplication (`clean {}` -> `dirty {}`) to preserve normal/anomaly contrast.
- Fixed upstream `predict()` smoothing to skip Gaussian smoothing when per-layer list outputs are returned (`aggregation=False`).
- Added a `predict(**_kwargs)` forward-compatibility path in the upstream wrapper.
- CI: mark the workspace as `git safe.directory` for editable installs.
- **Honor a shared model cache and add a `checkpoint_path` override.** When `CUVIS_MODEL_CACHE_DIR` is set (injected by the cuvis-ai-core run spawner), the fine-tuned weights (`get_weights_dir()`) and the OpenCLIP backbone (`create_model_and_transforms(cache_dir=…)`) resolve under that shared cache instead of `~/.cache`, so a sandboxed child loads them offline after the first download. The AdaCLIP node also accepts an explicit `checkpoint_path` to load fine-tuned weights from a local file.

## 0.1.5 - 2026-06-23

- Require `cuvis-ai-core>=0.10.0` and `cuvis-ai-schemas>=0.7.0`, adopting the released framework versions.
- Dropped the direct `cuvis` SDK dependency (`cuvis==3.5.0`); the plugin node code never used it. cu3s loading for the examples now comes from the `cuvis-ai-dataloader` plugin.
- Migrated the `examples_cuvis/` scripts from the removed `cuvis_ai_core.data.datasets.SingleCu3sDataModule` to `cuvis_ai_dataloader.data.Cu3sDataModule`, building train/val/test as `file_indices` selectors in `DataSplitConfig`.
- Declared `cuvis-ai-dataloader` as a provisioned plugin in `configs/plugins/cuvis_ai_dataloader.yaml` (repo + tag, `[cu3s, coco]` extras) instead of a package dependency, so this plugin's pyproject no longer hard-depends on a sibling plugin. Provision it before running a cu3s example. The `examples` extra keeps only `cuvis-ai>=0.9.0` (which dropped the cuvis SDK) and `loguru`, so neither the plugin nor its examples pull the cuvis SDK.

## 0.1.4 - 2026-06-10

- Require `cuvis-ai-core>=0.7.1` and `cuvis-ai-schemas>=0.5.2` (inherits the upstream security floors transitively).
- Declared the bare-name `plugins:` block (`adaclip`, `cuvis_ai_builtin`) in all eight pipeline configs.
- Added the `cuvis_ai_compat.yml` dependency-compatibility workflow (audits the plugin's deps against the cuvis-ai-core lock).
- Removed the redundant `tests/test_adaclip.py` (covered by `test_cuvis_ai_adaclip.py` and the unit-test files); the integration tests now skip cleanly when the cuvis-ai node catalog is absent.
- Stripped `torch` / `torchvision` wheel hashes from `uv.lock`.

## 0.1.3 - 2026-04-29

- Annotated `AdaCLIPDetector` with `_category = NodeCategory.MODEL` and `_tags = {RGB, IMAGE, ANOMALY, MASK, INFERENCE, LEARNABLE, TORCH}` ClassVars so the node surfaces under the correct category and tag filters in the cuvis-ai palette.
- Pinned `cuvis-ai-schemas>=0.4.0` directly in dependencies (`NodeCategory` / `NodeTag` enums were added there in v0.4.0).
- Dropped `cuvis-ai` and `cuvis-ai-core` git branch overrides from `[tool.uv.sources]`; the whole cuvis stack now resolves from PyPI.
- Stripped `hash` fields from `torch` / `torchvision` wheel entries in `uv.lock`.

## 0.1.2 - 2026-02-25

- Updated cuvis-ai node module paths: `band_selection` → `channel_selector`, `selector` → `channel_selector`, `visualizations` → `anomaly_visualization` (ALL-5300 Steps 1-9)
- Renamed 7 selector classes: `BaselineFalseRGBSelector` → `FixedWavelengthSelector`, `CIRFalseColorSelector` → `CIRSelector`, `HighContrastBandSelector` → `HighContrastSelector`, `SupervisedCIRBandSelector` → `SupervisedCIRSelector`, `SupervisedWindowedFalseRGBSelector` → `SupervisedWindowedSelector`, `SupervisedFullSpectrumBandSelector` → `SupervisedFullSpectrumSelector`, `BandSelectorBase` → `ChannelSelectorBase`
- Updated cuvis-ai-core imports: `CuvisCanvas` → `CuvisPipeline`, `auto_register_package` → `NodeRegistry.auto_register_package`
- Updated cuvis-ai-schemas imports: `PortSpec`, `Context`, `ExecutionStage` moved from `cuvis_ai_core` to `cuvis_ai_schemas`
- Updated 8 pipeline YAML configs with new `class` paths and node names
- Updated 2 test files with new import paths and class names
- Updated README.md code examples with new import paths
- Pinned cuvis-ai and cuvis-ai-core dependencies to `nima/features/consolidation` branch
- Removed duplicate file `statistical_cir_false_color copy.py`
- Removed `statistical_adaclip_channel_selector.py` example (used legacy CuvisCanvas API)

## 0.1.0 - 2026-01-23

- Initial plugin release for cuvis-ai framework with standalone package structure
- `AdaCLIPDetector` node implementing zero-shot anomaly detection (ECCV 2024 AdaCLIP)
- Lazy model initialization with automatic weight download and caching
- Performance optimizations: FP16 inference, CUDA kernel warmup, tensor-based preprocessing
- Gradient flow support for training upstream nodes (channel selectors, preprocessors) while AdaCLIP weights remain frozen
- Dual preprocessing modes: fast tensor-based (default) and exact PIL match for reproducibility
- Input/output ports: `rgb_image` [B,H,W,3] → `scores` [B,H,W,1], `anomaly_score` [B]
- Dependencies: `cuvis-ai-core` v0.1.0, `cuvis-ai` v0.2.3, PyTorch with CUDA 12.8 support
