# AdaCLIP Plugin for cuvis.ai

A **[cuvis.ai](https://github.com/cubert-hyperspectral/cuvis-ai) plugin** for [AdaCLIP](https://arxiv.org/abs/2407.15795), a zero-shot anomaly detection method that adapts CLIP with hybrid learnable prompts for hyperspectral imaging.

> **Note**: For the original AdaCLIP repository and training code, see [README_UPSTREAM.md](README_UPSTREAM.md).

## Installation

### Prerequisites

- [cuvis C SDK](https://cloud.cubert-gmbh.de/s/qpxkyWkycrmBK9m) (for .cu3s session files)
- [uv](https://docs.astral.sh/uv/) (Python dependency manager)
- [cuvis.ai framework](https://github.com/cubert-hyperspectral/cuvis-ai) (automatically installed as dependency)

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
from cuvis_ai.node.band_selection import CIRFalseColorSelector
from cuvis_ai_core.pipeline.pipeline import CuvisPipeline

# Download weights
download_weights("pretrained_all")

# Create pipeline
pipeline = CuvisPipeline("adaclip_pipeline")
band_selector = CIRFalseColorSelector(nir_nm=860.0, red_nm=670.0, green_nm=560.0)
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

See [cuvis_ai_adaclip/examples_cuvis/README.md](cuvis_ai_adaclip/examples_cuvis/README.md) for complete examples.

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

See **[cuvis_ai_adaclip/examples_cuvis/README.md](cuvis_ai_adaclip/examples_cuvis/README.md)** for complete documentation.

Available examples:
- **Baseline**: Fixed false-RGB (650/550/450 nm)
- **CIR False-Color**: NIR-Red-Green mapping
- **High-Contrast**: Variance + Laplacian selection
- **Supervised**: mRMR-based band selection
- **Learnable**: Gradient-trained channel selector

Quick start:
```bash
uv run python cuvis_ai_adaclip/examples_cuvis/statistical_baseline.py \
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

Weights are automatically downloaded and cached on first use:

```python
from cuvis_ai_adaclip import list_available_weights, download_weights

print(list_available_weights())  # ['pretrained_all', 'pretrained_mvtec_clinicdb', ...]
download_weights("pretrained_all")
```

| Weight Name | Description | Google Drive |
|------------|-------------|--------------|
| `pretrained_all` | Trained on all datasets | [Link](https://drive.google.com/file/d/1Cgkfx3GAaSYnXPLolx-P7pFqYV0IVzZF/view) |
| `pretrained_mvtec_clinicdb` | MVTec AD & ClinicDB | [Link](https://drive.google.com/file/d/1xVXANHGuJBRx59rqPRir7iqbkYzq45W0/view) |
| `pretrained_visa_colondb` | VisA & ColonDB | [Link](https://drive.google.com/file/d/1QGmPB0ByPZQ7FucvGODMSz7r5Ke5wx9W/view) |

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

- **Python**: 3.10-3.13
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
