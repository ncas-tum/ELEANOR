# Changelog

All notable changes to ELEANOR are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and entries are grouped chronologically by quarter.

## [Unreleased]

### 2026 Q2
- Migrated build configuration to `uv` via `pyproject.toml`; dev dependencies
  moved to a `[dependency-groups]` table; added `tox.ini` and `.python-version`;
  bumped minimum Python to 3.11.
- Removed the `snnax` dependency entirely from the project.
- Removed `chex` from the JAX modules (`_bruno`, `_felif`, `_heracles`); type
  annotations now use `jaxtyping` (`Array`, `PRNGKeyArray`).
- Restructured the JAX models API: split each model into `*Cell` and `*Params`,
  introduced a `NeuronModel` abstract base class, an `RNN` scan wrapper in a new `_base.py`.
- Documentation: restructured `docs/source/` into `about/`,
  `getting_started/`, `tutorials/`, and `api/` sections; added `quickstart`,
  `concepts`, `contributing`, and `changelog` pages; added an API reference
  covering `jax`, `torch`, `datasets`, `learner`, and `utils`; renamed
  `examples/` to `tutorials/`; removed the legacy `introduction/`,
  `getting-start.rst`, and `usage.rst` pages; refreshed `conf.py`,
  `install.rst`, and the root `index.rst`.
- Added this `CHANGELOG.md`.
- `eleanor/__init__.py` now re-exports the `datasets`, `utils`, `torch`, and
  `jax` submodules at the top level.
- `eleanor/utils.py`: removed the legacy `snnax`-based `forward_fn` helper as
  part of the `snnax` cleanup.
- `eleanor/datasets.py`: `loadBraille` now accepts a `full=` flag to return the
  unsplit dataset.

### 2026 Q1
- Updated README.

### 2025 Q4
- JAX-based federated learning implementation.
- PyTorch federated learning script; updated Braille FL task.
- Federated learning scripts for the Braille dataset.
- Analysis script for the Bruno paper with variability updates.
- Analysis script comparing Bruno with checkpoints.
- JSB experiment and dataset.
- Pure Equinox module of Bruno; `variability` now uses `eqx.partition`.
- Checkpoint method using the FeLIF neuron; defaults applied when checkpoint is `None`.
- `StaticWrapper` for D2DVar parameters.
- Restructured model example notebooks (`bruno`, `felif`, `heracles`).
- Variability added to the PyTorch C++ implementation of Bruno.
- Heracles (PyTorch): configurable number of inner steps; JAX gradient scaling
  aligned with PyTorch.

### 2025 Q3
- Variability in FeLIF and Heracles (PyTorch).
- Heracles fixes: `dp` calculation; negative-polarization bug.
- Negative polarization implemented in PyTorch.
- MPI support added to the Docker workspace.
- Bipolar switching on PyTorch.
- Two-phase plasticity on PyTorch.
- Heracles implemented with Bruno on both PyTorch and JAX.

### 2025 Q2
- PyTorch implementation of ELEANOR; package reorganized.
- Docker container updated.
- Simplified FeLIF model.
- Memory and time analysis utilities; time-measurement test.
- `NoBruno` class; adjusted `initial_P`.

### 2025 Q1
- Variability added to the models.
- Generalized weight quantization class supporting non-stochastic rounding.
- Import fix in `models.py`.

## [Pre-v0.0.dev0] — 2024

### 2024 Q4
- Optuna HPO experiments for Heracles and LIF; analysis scripts.
- Weight quantization with STE scaling.
- FeLIFv2; reset polarization based on SPICE simulation.
- Yin Yang dataset benchmark.
- Scaler model; Heracles firing based on voltage.
- Heracles parameter documentation.
- ROCm JAX support in `pyproject`; CUDA Dockerfile.

### 2024 Q3
- Renamed project to **ELEANOR**; refactor + documentation pass.
- Docker container created.
- Quantization work started; refactor toward SNNAX.
- New model based on the IEDM submission.

### 2024 Q2
- FeLIF on Equinox.
- SHD split into LIF and FeLIF paths; LI output layer.
- Inner polarization loop script.
- Neuron optimization example; NNI exploration with regularizer.

### 2024 Q1
- Initial commit; FeLIF neuron in JAX; Braille training example.

[Unreleased]: https://github.com/ncas-tum/ELEANOR/compare/v0.0.dev0...HEAD
[v0.0.dev0]: https://github.com/ncas-tum/ELEANOR/releases/tag/v0.0.dev0
