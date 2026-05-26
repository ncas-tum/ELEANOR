ELEANOR
=======

.. note::

   This project is under active development.

**ELEANOR** (f\ **E**\ rroelectric **L**\ eaky-int\ **E**\ grate **A**\ nd fire
**N**\ eural m\ **O**\ del simulato\ **R**) is a simulator for spiking neural
networks based on ferroelectric leaky-integrate-and-fire (FeLIF) neurons,
shipped with two backends — **JAX** (via `Equinox`_) and **PyTorch** (via
`snnTorch`_) — sharing the same model definitions.

.. _Equinox: https://docs.kidger.site/equinox/
.. _snnTorch: https://snntorch.readthedocs.io/

The package provides

- the **FeLIF**, **Bruno**, and **Heracles** neuron models,
- **device-to-device** and **cycle-to-cycle** variability primitives,
- **weight quantization** utilities for training-aware quantization,

.. - **federated learning** examples on the Braille tactile dataset.

Getting started
---------------

If you are new here, the recommended path is:

1. :doc:`Install <getting_started/install>` ELEANOR.
2. Run the :doc:`quickstart <getting_started/quickstart>` to verify the install
   and see a neuron simulate.
3. Read the :doc:`concepts <getting_started/concepts>` page if you want the
   ferroelectric background.
4. Browse the :doc:`user_guide/index` for task-oriented guides
   (training, variability, quantization, federated learning).
5. Dig into the :doc:`tutorials <tutorials/index>` and
   :doc:`API reference <api/index>` when you need depth.

.. toctree::
   :hidden:
   :caption: Getting started

   getting_started/install
   getting_started/quickstart
   getting_started/concepts

.. toctree::
   :hidden:
   :caption: User guide

   user_guide/index

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials/index

.. toctree::
   :hidden:
   :caption: API reference

   api/index

.. toctree::
   :hidden:
   :caption: About

   about/index
