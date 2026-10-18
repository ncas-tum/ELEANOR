[![Lint](https://github.com/ncas-tum/ELEANOR/actions/workflows/lint.yml/badge.svg?event=push)](https://github.com/ncas-tum/ELEANOR/actions/workflows/lint.yml)
[![Docs](https://github.com/ncas-tum/ELEANOR/actions/workflows/pages.yml/badge.svg)](https://ncas-tum.github.io/ELEANOR/)
[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.txt)

# ELEANOR (f**E**rroelectric **L**eaky-int**E**grate **A**nd fire **N**eural m**O**del simulato**R**)

ELEANOR is a simulator for Ferroelectric Leaky-Integrate-and-Fire (FeLIF)
neurons. It provides the FeLIF, Bruno, and Heracles neuron models with two
backends:

- **JAX** (`eleanor.jax`), built on Equinox.
- **PyTorch** (`eleanor.torch`), with C++/CUDA kernels compiled as a PyTorch
  extension.

Documentation: <https://ncas-tum.github.io/ELEANOR/>

## Installation

ELEANOR needs Python 3.11–3.14 and is installed as `eleanor-ncas` (the import
name stays `eleanor`). Pick **one** hardware variant; it selects the matching
PyTorch and JAX builds.

### Prebuilt wheels (Linux x86_64)

Each variant has its own package index. Add it next to PyPI, and pip or uv
installs ELEANOR with the matching PyTorch and JAX:

| Variant | Hardware              | Index                                         |
|---------|-----------------------|-----------------------------------------------|
| `cpu`   | CPU only              | `https://ncas-tum.github.io/ELEANOR/whl/cpu`   |
| `cu126` | NVIDIA GPU, CUDA 12.6 | `https://ncas-tum.github.io/ELEANOR/whl/cu126` |
| `cu130` | NVIDIA GPU, CUDA 13.0 | `https://ncas-tum.github.io/ELEANOR/whl/cu130` |

```bash
pip install eleanor-ncas --extra-index-url https://ncas-tum.github.io/ELEANOR/whl/cpu
```

or, in a uv project:

```bash
uv add eleanor-ncas --index https://ncas-tum.github.io/ELEANOR/whl/cpu
```

Use `--extra-index-url` (pip) or `--index` (uv), not `--index-url`: the index
only holds ELEANOR and PyTorch, and everything else comes from PyPI. On Python
3.12 and newer, one dependency (`expelliarmus`, via `tonic`) is compiled during
installation, so a C compiler is needed.

The wheels are also attached to every
[GitHub release](https://github.com/ncas-tum/ELEANOR/releases).

### From source (uv)

Building from source compiles the PyTorch extension, so you need a C++20
compiler with OpenMP, and `nvcc` for the CUDA variants. The variant is
selected with an extra (`cpu`, `cuda12`, `cuda13` or `mps`):

```bash
git clone https://github.com/ncas-tum/ELEANOR.git
cd ELEANOR
uv sync --extra cpu        # or cuda12 / cuda13 / mps
```

The extension is compiled against the PyTorch of the selected extra. Once
synced, `uv run pytest` and other `uv run` commands keep that PyTorch. Always
pass the extra to `uv sync`: a plain `uv sync` removes PyTorch.

### Development container

`Dockerfile.dev` and `.devcontainer/` provide a ready-made environment with
the toolchain installed. Inside the container, run `dev-sync` for an editable
install.

## Quick check

```python
import eleanor

print(eleanor.__version__)
```

## Contributing

`uv sync --extra cpu` also installs the development tools. Code is linted and
formatted with Ruff (`uvx ruff check eleanor`, `uvx ruff format eleanor`), and
the docs are built with `uv run sphinx-build -b html docs/source docs/_build`.
See [CHANGELOG.md](CHANGELOG.md) for recent changes.

## License

MIT; see [LICENSE.txt](LICENSE.txt). Contributors are listed in
[AUTHORS.rst](AUTHORS.rst).
