# Changelog

## [Unreleased]

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
