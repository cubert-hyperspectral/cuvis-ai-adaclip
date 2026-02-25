# Changelog

## [Unreleased]

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
