Variability
===========

Real ferroelectric devices are stochastic. ELEANOR models two regimes:

- **Device-to-device (D2D)** — fixed per device, varies across the population.
  Captured by :class:`~eleanor.jax.variability.D2DVar`
  / :class:`~eleanor.torch.variability.D2DVar`.
- **Cycle-to-cycle (C2C)** — fresh noise per call. Captured by
  :class:`~eleanor.jax.variability.C2CVar`
  / :class:`~eleanor.torch.variability.C2CVar`.

All ELEANOR neuron cells accept a ``variability`` constructor argument that
applies D2D variability to the relevant device parameters.

JAX backend
-----------

Pass ``variability=σ`` (relative standard deviation) when building a cell:

.. code-block:: python

    from eleanor.jax.models import FeLIFCell, FeLIFParams

    cell = FeLIFCell(
        shape=(128,),
        params=FeLIFParams(),
        variability=0.1,        # 10% D2D variability
        key=key,
    )

Each D2D parameter (``P_s``, ``tau_p``, ``tau_m``, ``threshold``) is wrapped
in a :class:`~eleanor.jax.variability.D2DVar` and applied multiplicatively:

.. math::

    \theta_i = \mu \, (1 + \sigma \, \epsilon_i),
    \quad \epsilon_i \sim \mathcal{N}(0, 1).

Helpers in :mod:`eleanor.jax.variability` let you re-sample or set the
variability post-hoc:

.. autosummary::

   eleanor.jax.variability.update_d2d_variability
   eleanor.jax.variability.update_d2d_variability_name
   eleanor.jax.variability.set_d2d_variability
   eleanor.jax.variability.set_d2d_variability_name

A common pattern — sweeping variability levels at evaluation:

.. code-block:: python

    import jax
    from eleanor.jax.variability import (
        set_d2d_variability,
        update_d2d_variability,
    )

    for sigma in [0.0, 0.05, 0.1, 0.2]:
        model_sigma = set_d2d_variability(trained_model, sigma)
        model_sigma = update_d2d_variability(model_sigma, jax.random.key(42))
        evaluate(model_sigma, test_data)

PyTorch backend
---------------

Same idea, exposed as ``nn.Module``\ s with mutable buffers:

.. code-block:: python

    from eleanor.torch.models import FeLIF

    neuron = FeLIF(tau_p=0.6, tau_m=0.95, variability=0.1)

The :func:`~eleanor.torch.variability.set_d2d_variability` and
:func:`~eleanor.torch.variability.update_d2d_variability` helpers operate on
the whole module tree:

.. code-block:: python

    from eleanor.torch.variability import (
        set_d2d_variability,
        update_d2d_variability,
    )

    set_d2d_variability(model, 0.1)
    update_d2d_variability(model, shape=(128,))

Cycle-to-cycle noise
--------------------

``C2CVar`` resamples on every call, so its noise is gradient-friendly in the
same way standard dropout is. Use it when the parameter is genuinely
re-randomized per timestep rather than fixed per device.
