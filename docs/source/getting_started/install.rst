.. _installation:

Installation
============

Requirements
------------

- Python ≥ 3.11
- One of the supported backends:

  - **JAX** ≥ 0.10.1 (CPU, CUDA, or ROCm)
  - **PyTorch** ≥ 2.0 (CPU or CUDA)

ELEANOR is tested on Linux and macOS. Windows is not officially supported but
the JAX backend may work via WSL2.

Install from GitHub
-------------------

ELEANOR is not yet on PyPI. We recommend using `uv`_ for fast and
reproducible environments. Install the latest version directly from GitHub:

.. code-block:: bash

    uv add git+https://github.com/ncas-tum/ELEANOR.git

For GPU support with JAX:

.. code-block:: bash

    uv add "eleanor[gpu] @ git+https://github.com/ncas-tum/ELEANOR.git"

Development install
-------------------

Clone and install in editable mode.

.. code-block:: bash

    git clone https://github.com/ncas-tum/ELEANOR.git
    cd ELEANOR
    uv sync                       # production deps
    uv sync --group dev           # add development tools

.. _uv: https://docs.astral.sh/uv/

With plain pip:

.. code-block:: bash

    git clone https://github.com/ncas-tum/ELEANOR.git
    pip install -e ELEANOR

Verify the install
------------------

.. code-block:: python

    import eleanor
    print(eleanor.__version__)

If you see a version string, you are ready to continue with the
:doc:`quickstart`.
