# AdaCLIP Plugin for cuvis.ai

[![CI Status](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip/actions/workflows/ci.yml/badge.svg)](https://github.com/cubert-hyperspectral/cuvis-ai-adaclip/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/cubert-hyperspectral/cuvis-ai-adaclip/branch/main/graph/badge.svg)](https://codecov.io/gh/cubert-hyperspectral/cuvis-ai-adaclip)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/downloads/)

A **[cuvis.ai](https://github.com/cubert-hyperspectral/cuvis-ai) plugin** for [AdaCLIP](https://arxiv.org/abs/2407.15795), a zero-shot anomaly detection method that adapts CLIP with hybrid learnable prompts for hyperspectral imaging.

> **Note**: For the original AdaCLIP repository and training code, see [README_UPSTREAM.md](README_UPSTREAM.md).

## Installation

### Prerequisites

- [uv](https://docs.astral.sh/uv/) (Python dependency manager)
- [cuvis.ai framework](https://github.com/cubert-hyperspectral/cuvis-ai) (automatically installed as dependency)

The plugin itself does not depend on the cuvis SDK, and it does not hard-depend on the
sibling `cuvis-ai-dataloader` plugin. The cu3s example scripts under
`cuvis_ai_adaclip/examples_cuvis/` load `.cu3s` session files through the
`cuvis-ai-dataloader` plugin, which is declared (not pip-pinned) in
`configs/plugins/cuvis_ai_dataloader.yaml` and provisioned into the run environment. First
install the example helpers:

```bash
uv pip install -e ".[examples]"
```

Then provision the data plugin from its manifest (this installs `cuvis-ai-dataloader` with
its `[cu3s, coco]` extras; the cuvis SDK comes transitively through the `cu3s` extra):

```bash
uv run provision --plugins-dir configs/plugins
```

### Setup

```bash
# Clone and install
git clone <repository-url>
cd cuvis-ai-adaclip
uv sync --all-extras --dev
```

> **Note**: The `cuvis.ai` framework is automatically installed as a dependency. For local development with editable `cuvis.ai`, clone it at the same level as this repository (see `pyproject.toml` path dependencies).

### Enable Git Hooks

```bash
git config core.hooksPath .githooks
```

## Usage

### Direct Python Dependency (Local Development)

For developers working directly with the code:

```python
from cuvis_ai_adaclip import AdaCLIPDetector, download_weights
from cuvis_ai.node.channel_selector import CIRSelector
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

# Download weights
download_weights("pretrained_all")

# Create pipeline
pipeline = CuvisPipeline("adaclip_pipeline")
band_selector = CIRSelector(nir_nm=860.0, red_nm=670.0, green_nm=560.0)
adaclip = AdaCLIPDetector(
    weight_name="pretrained_all",
    backbone="ViT-L-14-336",
    prompt_text="normal: lentils, anomaly: stones"
)

# Wire and run
pipeline.connect(
    (band_selector.outputs.rgb_image, adaclip.inputs.rgb_image)
)
```

See [examples/statistical/README.md](examples/statistical/README.md) for complete examples.

### Plugin Usage (Production)

Load AdaCLIP as a plugin via NodeRegistry:

**YAML Manifest (`plugins.yaml`):**
```yaml
plugins:
  adaclip:
    repo: "git@github.com:cubert-hyperspectral/cuvis-ai-adaclip.git"
    ref: "v1.0.0"
    provides:
      - cuvis_ai_adaclip.node.adaclip_node.AdaCLIPDetector
```

**Python:**
```python
from cuvis_ai_core.utils.node_registry import NodeRegistry

NodeRegistry.load_plugins("plugins.yaml")
# Now AdaCLIPDetector is available in pipelines
```

See [cuvis.ai/examples/plugin/](https://github.com/cubert-hyperspectral/cuvis.ai/tree/main/examples/plugin) for more examples.

### gRPC Usage (Remote/Production)

Use AdaCLIP via gRPC for remote deployments:

```python
import grpc
from cuvis_ai_core.grpc.v1 import cuvis_ai_pb2, cuvis_ai_pb2_grpc

channel = grpc.insecure_channel("localhost:50051")
client = cuvis_ai_pb2_grpc.CuvisAIServiceStub(channel)

# Create session
session = client.CreateSession(cuvis_ai_pb2.CreateSessionRequest())

# Load plugins (optional, if using as plugin)
# ... load plugin manifest via LoadPlugins RPC

# Resolve and apply pipeline config
resolved = client.ResolveConfig(
    cuvis_ai_pb2.ResolveConfigRequest(
        session_id=session.session_id,
        config_name="adaclip_baseline"
    )
)
client.SetTrainRunConfig(
    cuvis_ai_pb2.SetTrainRunConfigRequest(
        session_id=session.session_id,
        config_bytes=resolved.config_bytes
    )
)

# Train (statistical evaluation for AdaCLIP)
for progress in client.Train(
    cuvis_ai_pb2.TrainRequest(
        session_id=session.session_id,
        trainer_type=cuvis_ai_pb2.TRAINER_TYPE_STATISTICAL
    )
):
    print(f"Progress: {progress.current_step}/{progress.total_steps}")

# Inference
inference = client.Inference(
    cuvis_ai_pb2.InferenceRequest(
        session_id=session.session_id,
        inputs=cuvis_ai_pb2.InputBatch(...)
    )
)

# Cleanup
client.CloseSession(cuvis_ai_pb2.CloseSessionRequest(session_id=session.session_id))
```

See [cuvis.ai/examples/grpc/](https://github.com/cubert-hyperspectral/cuvis.ai/tree/main/examples/grpc) for complete gRPC examples.

## Examples

This repository includes comprehensive examples demonstrating different band selection strategies and usage patterns.

### Python Examples (Direct Import)

See **[examples/statistical/README.md](examples/statistical/README.md)** for complete documentation.

Available examples:
- **Baseline**: Fixed false-RGB (650/550/450 nm)
- **CIR False-Color**: NIR-Red-Green mapping
- **High-Contrast**: Variance + Laplacian selection
- **Supervised**: mRMR-based band selection
- **Learnable**: Gradient-trained channel selector

Quick start:
```bash
uv run python examples/statistical/statistical_baseline.py \
    --backbone-name ViT-L-14-336 \
    --pretrained-adaclip pretrained_all \
    --visualize-upto 5
```

### Plugin & gRPC Examples

See the [cuvis.ai repository examples](https://github.com/cubert-hyperspectral/cuvis.ai/tree/main/examples):
- `plugin/plugin_example.py` - NodeRegistry plugin loading
- `grpc/adaclip_client.py` - Basic gRPC workflow
- `grpc/adaclip_cir_false_color_client.py` - CIR via gRPC

## Node API

### `AdaCLIPDetector`

A cuvis.ai `Node` for zero-shot anomaly detection on RGB images.

**Inputs:**
- `rgb_image`: `torch.Tensor` of shape `[B, H, W, 3]` (float32, 0-1 or 0-255)

**Outputs:**
- `scores`: `torch.Tensor` of shape `[B, H, W, 1]` - Pixel-level anomaly scores
- `anomaly_score`: `torch.Tensor` of shape `[B]` - Image-level anomaly scores

**Key Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `weight_name` | str | `"pretrained_all"` | Pre-trained weight identifier |
| `backbone` | str | `"ViT-L-14-336"` | CLIP backbone model |
| `prompt_text` | str | `""` | Text prompt for anomaly classes |
| `image_size` | int | `518` | Input image size |
| `gaussian_sigma` | float | `4.0` | Gaussian smoothing sigma |
| `use_half_precision` | bool | `False` | Enable FP16 optimization |

**Backbone Options:** `"ViT-L-14-336"`, `"ViT-L-14"`, `"ViT-B-16"`, `"ViT-B-32"`, `"ViT-H-14"`

## Pre-trained Weights

The three checkpoints released by the AdaCLIP authors are mirrored unchanged at
[`cubert-gmbh/adaclip`](https://huggingface.co/cubert-gmbh/adaclip) and served through
cuvis-ai-core's weight registry. `download_weights` returns the cached file and downloads it
when online; the sandboxed runtime is offline, so provision once with `download-model`:

```bash
uv run download-model download adaclip_all
uv run download-model download clip_vit_l_14_336   # the CLIP ViT-L/14@336px backbone
```

```python
from cuvis_ai_adaclip import list_available_weights, download_weights

print(list_available_weights())  # ['pretrained_mvtec_colondb', 'pretrained_visa_clinicdb', 'pretrained_all']
download_weights("pretrained_all")
```

| Weight Name | Registry name | Description |
|------------|---------------|-------------|
| `pretrained_all` | `adaclip_all` | Trained on all auxiliary datasets |
| `pretrained_mvtec_colondb` | `adaclip_mvtec_colondb` | Trained on MVTec AD & ColonDB (the upstream README's weights table labels this file "MVTec AD & ClinicDB"; the file name and the README's Train section say ColonDB) |
| `pretrained_visa_clinicdb` | `adaclip_visa_clinicdb` | Trained on VisA & ClinicDB (upstream table label: "VisA & ColonDB") |

The CLIP ViT-L/14@336px backbone comes from the [`cubert-gmbh/clip`](https://huggingface.co/cubert-gmbh/clip)
mirror and is placed where the vendored OpenCLIP loader expects it (`$CUVIS_MODEL_CACHE_DIR/clip`,
else `~/.cache/clip`). Other backbones are not mirrored and still download from OpenAI.

## Development

### Testing

```bash
uv run pytest
uv run pytest --cov=cuvis_ai_adaclip --cov-report=term-missing
```

### Code Quality

```bash
uv run ruff format .
uv run ruff check .
```

### Building

```bash
uv build
```

## Release Notes

See [CHANGELOG.md](CHANGELOG.md) for version history and upgrade guidance.

## Compatibility

- **Python**: 3.11
- **PyTorch**: Provided by cuvis.ai dependency
- **CUDA**: GPU recommended for optimal performance

## Citation

If you use AdaCLIP in your research, please cite:

```bibtex
@inproceedings{AdaCLIP,
  title={AdaCLIP: Adapting CLIP with Hybrid Learnable Prompts for Zero-Shot Anomaly Detection},
  author={Cao, Yunkang and Zhang, Jiangning and Frittoli, Luca and Cheng, Yuqi and Shen, Weiming and Boracchi, Giacomo},
  booktitle={European Conference on Computer Vision},
  year={2024}
}
```

## License

MIT License (see [LICENSE](LICENSE) file)

## Acknowledgments

- Original AdaCLIP: [caoyunkang/AdaCLIP](https://github.com/caoyunkang/AdaCLIP)
- cuvis.ai framework: [cubert-hyperspectral/cuvis.ai](https://github.com/cubert-hyperspectral/cuvis.ai)
