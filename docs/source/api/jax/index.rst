JAX backend
===========

The JAX implementation of ELEANOR lives under :mod:`eleanor.jax`. Neuron
models are Equinox modules; time evolution is handled by an explicit
``jax.lax.scan`` inside the :class:`~eleanor.jax.models.RNN` wrapper.

.. toctree::
   :maxdepth: 1

   models
   variability
   weight_quantization
   surrogate
