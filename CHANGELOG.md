# Changelog

All notable changes to ELEANOR are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and entries are grouped chronologically by quarter.

## [Unreleased]

### 2026 Q4
- Migrated the build backend from `setup.py` to `scikit-build-core` with a
  `CMakeLists.txt`; the PyTorch C++/CUDA extension is now built as
  `_C_cpu` or `_C_cu<version>` depending on whether a CUDA compiler is found.
- Version is now derived from git tags via `setuptools-scm` (written to
  `eleanor/_version.py`).
- Added mutually exclusive `cpu`, `cuda12`, `cuda13`, and `mps` extras; each
  pulls `torch`/`torchvision` from the matching PyTorch index and the matching
  `jax` variant.
- `uv sync --extra <name>` now compiles the extension against the torch of
  the selected extra instead of the default PyPI (CUDA) wheel; `torch` was
  removed from `build-system.requires`.
- Development: the hardware variant is always chosen explicitly with
  `uv sync --extra cpu` (or `cuda12` / `cuda13` / `mps`); a sync without one
  fails with a message saying so.
- uv now rebuilds the package on new commits/tags and on changes to
  `CMakeLists.txt` or the C++/CUDA sources.
- Refactored the C++/CUDA implementations of Bruno and Heracles: shared code
  moved to `common.h` and per-model headers (`bruno.h`, `heracles.h`).
- Renamed the distribution to `eleanor-ncas` (`eleanor` is taken on PyPI); the
  import name is still `eleanor`.
- Release wheels: one `cp311-abi3` manylinux wheel per hardware variant
  (`+cpu`, `+cu126`, `+cu130`) that works on Python 3.11+. The extension now
  uses only the limited Python API, and each wheel pins the exact
  `torch`/`torchvision` build it was compiled against plus the matching `jax`.
- Per-variant package index on GitHub Pages
  (`https://ncas-tum.github.io/ELEANOR/whl/<variant>`):
  `pip install eleanor-ncas --extra-index-url .../whl/cpu` or
  `uv add eleanor-ncas --index .../whl/cpu` installs ELEANOR with the matching
  PyTorch and JAX.
- `eleanor.torch.models` picks the extension that matches the installed
  torch, falling back to another build of the same CUDA major version, and
  lists the available builds when none match.
- Fixed: installed wheels reported version `0.0.dev0`, because the generated
  `eleanor/_version.py` was left out of the wheel.
- Added a `Dockerfile` that builds each variant (with a CUDA architecture check
  and `auditwheel repair` that keeps torch and CUDA libraries, including
  NVRTC, external), scripts to turn the builds into variant wheels and an
  index (`scripts/make_variant_wheel.py`, `scripts/build_index.py`), and a
  `Dockerfile.dev` + devcontainer for development.
- CI: the wheels workflow builds every variant, checks the stable ABI, installs
  each wheel from a locally served index with pip and uv on Python 3.11 and
  3.14, and on `v*` tags attaches the wheels to a GitHub release and rebuilds
  the docs + index. Lint and docs workflows install the `cpu` extra.

### 2026 Q3
- Bruno (JAX): stopping the gradient through `tau` and through the reset is
  now configurable with `stop_tau_grad` and `stop_reset_grad` (both default to
  `True`).
- `eleanor.torch.models` now loads the compiled extension that matches the
  environment (CPU or CUDA build).
- Braille dataset is now stored under `/tmp/data/braille` by default.

### 2026 Q2
- Bruno (JAX): gradient through `tau` stopped; added an overflow-safe
  exponential (`limexp`) and a second-derivative `tau` gradient.
- Braille benchmark.
- `eleanor.jax.surrogate` is now a public module (was `_surrogate`).
- Documentation is built and published to GitHub Pages.
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
