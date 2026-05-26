Neuron models
=============

ELEANOR ships three ferroelectric neuron models. Each model has matching
implementations in the JAX and PyTorch backends.

.. list-table::
   :header-rows: 1
   :widths: 14 18 18 50

   * - Model
     - JAX class
     - PyTorch class
     - Use when
   * - FeLIF
     - :class:`~eleanor.jax.models.FeLIFCell`
     - :class:`~eleanor.torch.models.FeLIF`
     - You want the fast, normalized-unit FeLIF for ML training.
   * - Heracles
     - :class:`~eleanor.jax.models.HeraclesCell`
     - :class:`~eleanor.torch.models.Heracles`
     - You need physics-grounded parameters (device dimensions, permittivity, switching current).
   * - Bruno
     - :class:`~eleanor.jax.models.BrunoCell`
     - :class:`~eleanor.torch.models.Bruno`
     - You want a tradeoff between physics-grounded parameters and ML training. 
       Use for reproducing or building on the Bruno paper's undersampled backpropagation method.

FeLIF
-----

The compact ferroelectric LIF. Tracks three states (i.e. spike, membrane voltage,
polarization) and a small set of normalized parameters:

.. todo::

   Auto-generate this parameter table from
   :class:`eleanor.jax.models.FeLIFParams`.

See :class:`eleanor.jax.models.FeLIFParams` for parameter names and defaults.

Heracles
--------

Heracles parameters correspond to physical device quantities (ferroelectric
layer thickness, depletion width, dielectric permittivities, switching current,
temperature, ...). Use Heracles when you need to relate training results back to
device design choices, or when you want to sweep physical parameters during
optimization.

.. todo::

   Auto-generate this parameter table from
   :class:`eleanor.jax.models.HeraclesParams`.

See :class:`eleanor.jax.models.HeraclesParams` for parameter names and defaults.

Bruno
-----

Bruno implements the **B**\ ackpropagation **r**\ unning **u**\ ndersampled
for **n**\ ovel device **o**\ ptimisation method from Fehlings et al., NCE
1.    See :doc:`../about/citation`.

.. todo::

   Auto-generate this parameter table from
   :class:`eleanor.jax.models.BrunoParams`.

See :class:`eleanor.jax.models.BrunoParams` for parameter names and defaults.

Examples
=========

Executable Jupyter notebooks covering the three different neuron models using 
PyTorch backend.

.. toctree::
   :maxdepth: 1

   examples/felif
   examples/bruno
   examples/heracles