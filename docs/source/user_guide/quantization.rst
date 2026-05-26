Weight quantization
===================

Training-aware weight quantization for ferroelectric inference hardware. The
current implementation lives in the JAX backend
(:mod:`eleanor.jax.weight_quantization`); the PyTorch equivalent is planned.

Use case
--------

Ferroelectric synapses store weights at limited bit-precision. ELEANOR's
quantizer rounds weights to a target bit-width during the forward pass while
keeping a straight-through estimator (STE) on the backward pass, so gradients
flow as if the quantization were the identity.

You can pick stochastic or non-stochastic rounding.

Basic usage
-----------

.. todo::

   Concrete code example once the public API of
   :mod:`eleanor.jax.weight_quantization` stabilizes. See
   :doc:`../tutorials/precision` for a worked notebook.

See also
--------

- :doc:`../tutorials/precision` — interactive precision-sweep notebook.
- :doc:`../api/jax/weight_quantization` — full API reference.
