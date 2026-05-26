Choosing a backend
==================

ELEANOR provides two parallel implementations of every neuron model: JAX
(``eleanor.jax``) and PyTorch (``eleanor.torch``). They share parameter names
and parameter defaults, so a model expressed in one backend can usually be
ported to the other line-for-line.

Quick comparison
----------------

.. list-table::
   :header-rows: 1

   * - Dimension
     - JAX (``eleanor.jax``)
     - PyTorch (``eleanor.torch``)
   * - Underlying framework
     - `Equinox`_ modules, ``jax.lax.scan`` for time
     - `snnTorch`_ ``SpikingNeuron`` subclasses
   * - API style
     - Split ``Cell`` + ``Params`` + ``RNN`` wrapper
     - Single ``nn.Module`` per neuron with stateful ``forward``
   * - Compilation
     - ``jax.jit`` / ``eqx.filter_jit``
     - ``torch.compile`` (optional)
   * - C++/CUDA kernels
     - Pure-Python
     - Native kernels for Bruno and Heracles

.. _Equinox: https://docs.kidger.site/equinox/
.. _snnTorch: https://snntorch.readthedocs.io/

When to pick JAX
----------------

- Functional, pure-functions training loop with explicit PRNG state.
- Easy ``jax.vmap`` over devices/parameter sweeps.
- Variability handling is built around immutable pytrees
  (``StaticWrapper``, ``D2DVar``), which composes naturally with
  ``equinox.tree_at`` patterns.

When to pick PyTorch
--------------------

- Existing PyTorch training infrastructure.
- You want the native C++/CUDA kernels for Bruno or Heracles.
- PyTorch/snnTorch is already part of your stack.

Same model, two backends
------------------------

A FeLIF in each backend with parity defaults:

.. code-block:: python

    # JAX
    from eleanor.jax.models import FeLIFCell, FeLIFParams
    cell = FeLIFCell(shape=(1,), params=FeLIFParams(), key=key)

    # PyTorch
    from eleanor.torch.models import FeLIF
    neuron = FeLIF(tau_p=0.6, tau_m=0.95, threshold=1.0, dt=1e-3)

The parameter names match (``tau_p``, ``tau_m``, ``threshold``, ``dt``, ...);
the JAX side groups them into a ``FeLIFParams`` dataclass, while the PyTorch
side passes them directly to the constructor.
