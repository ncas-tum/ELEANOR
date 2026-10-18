.. _installation:

Installation
============

Requirements
------------

- Python 3.11–3.14
- Linux or macOS. Windows is not officially supported, but ELEANOR may work
  via WSL2.

ELEANOR ships with both a JAX and a PyTorch backend, and is installed as
``eleanor-ncas`` (the import name stays ``eleanor``). You choose **one**
hardware variant at install time, and it selects the matching PyTorch and JAX
builds. The variants are mutually exclusive: an environment can only hold one
of them.

Prebuilt wheels
---------------

Prebuilt wheels are available for Linux (x86_64). Each variant has its own
package index; add it next to PyPI, and pip or uv installs ELEANOR with the
matching PyTorch and JAX:

.. list-table::
   :header-rows: 1

   * - Variant
     - Hardware
     - Index
   * - ``cpu``
     - CPU only
     - ``https://ncas-tum.github.io/ELEANOR/whl/cpu``
   * - ``cu126``
     - NVIDIA GPU, CUDA 12.6
     - ``https://ncas-tum.github.io/ELEANOR/whl/cu126``
   * - ``cu130``
     - NVIDIA GPU, CUDA 13.0
     - ``https://ncas-tum.github.io/ELEANOR/whl/cu130``

With pip:

.. code-block:: bash

    pip install eleanor-ncas --extra-index-url https://ncas-tum.github.io/ELEANOR/whl/cpu

In a uv project:

.. code-block:: bash

    uv add eleanor-ncas --index https://ncas-tum.github.io/ELEANOR/whl/cpu

Use ``--extra-index-url`` (pip) or ``--index`` (uv), not ``--index-url``: the
index only holds ELEANOR and PyTorch, and everything else comes from PyPI.

.. note::

   On Python 3.12 and newer, one dependency (``expelliarmus``, required by
   ``tonic``) has no prebuilt wheel and is compiled during installation, so a
   C compiler such as ``gcc`` is needed.

The wheels are also attached to every
`GitHub release <https://github.com/ncas-tum/ELEANOR/releases>`_.

Install from source
-------------------

Installing from source compiles the PyTorch C++/CUDA extension. The variant is
selected with an extra: ``cpu``, ``cuda12``, ``cuda13`` or ``mps``. You need:

- a C++20 compiler with OpenMP support;
- ``nvcc`` matching the CUDA version, for the ``cuda12`` and ``cuda13``
  variants;
- `uv`_, which installs the right PyTorch build for the chosen variant and
  compiles the extension against it.

.. code-block:: bash

    git clone https://github.com/ncas-tum/ELEANOR.git
    cd ELEANOR
    uv sync --extra cpu        # or cuda12 / cuda13 / mps

This creates ``.venv`` with ELEANOR installed in editable mode, together with
the development tools.

Once synced, ``uv run pytest`` and other ``uv run`` commands keep the PyTorch
of the selected extra. Always pass the extra to ``uv sync``: a plain
``uv sync`` removes PyTorch.

.. note::

   ELEANOR's version comes from git tags, so uv builds the package inside the
   project environment. If a uv command fails with a message asking you to
   install ``scikit_build_core``, run
   ``uv pip install scikit-build-core setuptools-scm cmake ninja`` once and
   retry. This is only needed for environments created with an older ELEANOR
   checkout.

.. _uv: https://docs.astral.sh/uv/

Development container
---------------------

The repository includes a ``Dockerfile.dev`` and a ``.devcontainer/``
configuration with the full toolchain (compiler, CUDA, uv) already installed.
Inside the container, run ``dev-sync`` for an editable install.

Verify the install
------------------

.. code-block:: python

    import eleanor
    print(eleanor.__version__)

If you see a version string, you are ready to continue with the
:doc:`quickstart`.
