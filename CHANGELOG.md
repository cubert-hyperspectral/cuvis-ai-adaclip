# Changelog

## V0.1.0
- Initial plugin release for cuvis-ai framework with standalone package structure
- `AdaCLIPDetector` node implementing zero-shot anomaly detection (ECCV 2024 AdaCLIP)
- Lazy model initialization with automatic weight download and caching
- Performance optimizations: FP16 inference, CUDA kernel warmup, tensor-based preprocessing
- Gradient flow support for training upstream nodes (channel selectors, preprocessors) while AdaCLIP weights remain frozen
- Dual preprocessing modes: fast tensor-based (default) and exact PIL match for reproducibility
- Input/output ports: `rgb_image` [B,H,W,3] → `scores` [B,H,W,1], `anomaly_score` [B]
- Dependencies: `cuvis-ai-core` v0.1.0, `cuvis-ai` v0.2.3, PyTorch with CUDA 12.8 support
