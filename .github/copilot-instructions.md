# Copilot Coding Agent Instructions for cuvis-ai-adaclip

## Project Overview
- **cuvis-ai-adaclip** is a plugin package for the cuvis.ai framework, providing AdaCLIP-based zero-shot anomaly detection for hyperspectral imaging.
- Major components: `cuvis_ai_adaclip/` (plugin code), `method/` (upstream AdaCLIP), `configs/` (YAML configs), `tests/`, and `model_configs/`.
- The upstream AdaCLIP code (`method/`, `dataset/`, `adaclip_tools/`, `loss.py`, `train.py`, `test.py`, `app.py`) is vendored and excluded from linting.

## Development Workflow
- **Dependency & environment management:** Use [`uv`](https://docs.astral.sh/uv/) exclusively. Never use bare `python` or `pip`.
  - Sync environment: `uv sync` (add `--all-extras` for full toolchain)
  - Run scripts/tests: `uv run python ...` or `uv run pytest`
- **Testing:**
  - All tests use `pytest` (run with `uv run pytest`).
  - Test files in `tests/`.
  - Use built-in `tmp_path` for temp dirs.
- **Linting/Formatting:**
  - Use Ruff: `uv run ruff check cuvis_ai_adaclip/` and `uv run ruff format cuvis_ai_adaclip/`
  - Configured via `pyproject.toml`.
- **Builds:**
  - Build package: `uv build`

## Project Conventions
- **Configuration:** Store pipeline/training configs in `configs/`.
- **Directory structure:**
  - `cuvis_ai_adaclip/` — plugin package (nodes, CLI, weights, examples)
  - `method/` — upstream AdaCLIP code (do not modify)
  - `dataset/` — upstream dataset utilities (do not modify)
  - `adaclip_tools/` — upstream tools (do not modify)
  - `model_configs/` — model configuration JSONs
  - `configs/` — pipeline YAML configurations
  - `tests/` — all tests

## Patterns & Guardrails
- Use only open-source, permissive dependencies (MIT/BSD/Apache-2.0).
- Keep docstrings concise; use Google/NumPy style for public APIs.
- Use `uv run` for all commands to ensure correct environment.
- Do not modify upstream code in `method/`, `dataset/`, `adaclip_tools/`.

## Examples
- Run tests: `uv run pytest tests/ -m "not gpu"`
- Coverage: `uv run pytest --cov=cuvis_ai_adaclip --cov-report=term-missing`
- Build: `uv build`

## References
- [README.md](../README.md) — project intro and setup
- [README_UPSTREAM.md](../README_UPSTREAM.md) — original AdaCLIP documentation

---
For unclear or missing conventions, prefer patterns from the above files and existing code.
