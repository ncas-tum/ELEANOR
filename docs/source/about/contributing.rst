Contributing
============

Contributions are welcome — bug reports, feature requests, documentation
fixes, and pull requests alike.

Reporting issues
----------------

Open an issue on `GitHub <https://github.com/ncas-tum/ELEANOR/issues>`_. For
bugs, include:

- ELEANOR version (``python -c "import eleanor; print(eleanor.__version__)"``)
- Backend (JAX, PyTorch) and version
- Python version and OS
- A minimal reproducer

Development setup
-----------------

.. code-block:: bash

    git clone https://github.com/ncas-tum/ELEANOR.git
    cd ELEANOR
    uv sync --group dev
    pre-commit install

Style and checks
----------------

- Formatting and Linting: ``ruff`` (run via ``tox -e format``).
- Tests: ``pytest`` from the repo root.
- Type checks (when applicable): align with the existing usage of
  ``jaxtyping`` in the JAX modules.

Pull requests
-------------

- Open the PR against ``main``.
- Keep the PR focused — one feature or fix per PR.
- Add a one-line entry under ``[Unreleased]`` in :file:`CHANGELOG.md`.

.. todo::

   Add a CONTRIBUTING.md at the repo root and reference it from here.
