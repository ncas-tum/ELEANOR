Concepts
========

This page explains the device-physics intuition behind ELEANOR's neuron models
for readers coming from the ML side. If you already work on ferroelectric
devices, you can skip and jump straight to :doc:`../user_guide/models`.

The ferroelectric LIF neuron
----------------------------

A conventional **leaky integrate-and-fire (LIF)** neuron tracks a single state,
the membrane potential :math:`V`, which decays exponentially toward zero and
integrates synaptic input until it crosses a threshold and emits a spike.

A **ferroelectric LIF (FeLIF)** neuron extends this with a second state
variable, the **polarization** :math:`P` of a ferroelectric capacitor. The
polarization is a non-volatile state that stores charge and feeds back into 
the membrane dynamics through a polarization current :math:`I_P`. 
The effect is that the neuron has internal memory beyond a single time constant.

.. todo::

   Add the FeLIF equations and a state diagram figure here.

Neuron types in ELEANOR
--------------------------

ELEANOR has three increasingly device-grounded neuron variants:

**FeLIF** (:class:`eleanor.jax.models.FeLIFCell`,
:class:`eleanor.torch.models.FeLIF`)

   The compact form. A handful of dimensionless parameters
   (:math:`\tau_p`, :math:`\tau_m`, :math:`P_s`, :math:`\alpha`,
   :math:`\beta`, ...) describe a ferroelectric LIF in normalized units. This is
   the right starting point for ML training.

**Heracles** (:class:`eleanor.jax.models.HeraclesCell`,
:class:`eleanor.torch.models.Heracles`)

   A physics-grounded model whose parameters correspond to actual device
   quantities, i.e. depletion width, ferroelectric thickness, dielectric
   permittivity, switching current. Due to the complexity of the model,
   is slower than FeLIF on training and simulation, but lets you sweep
   physical device parameters during training.

**Bruno** (:class:`eleanor.jax.models.BrunoCell`,
:class:`eleanor.torch.models.Bruno`)

   Is a tradeoff between physical-grounded model like Heracles and ML 
   training oriented. Used in the *Bruno: backpropagation running 
   undersampled for novel device optimisation* paper. See
   :doc:`../about/citation`.

.. todo::

   Add per-model parameter tables sourced from the dataclasses.

Variability
-----------

Analogue devices have two kinds of stochasticity:

- **Device-to-device (D2D)** — fabrication variations that are fixed for a
  given device. In ELEANOR this is :class:`~eleanor.jax.variability.D2DVar`,
  resampled when you change devices, not every time step.
- **Cycle-to-cycle (C2C)** — fresh noise on every operation. Modeled by
  :class:`~eleanor.jax.variability.C2CVar`.

The :doc:`../user_guide/variability` guide shows how to attach these to a
model and how to scan over variability levels during training.

Surrogate gradients
-------------------

The spike function (Heaviside) is non-differentiable, so ELEANOR uses
surrogate gradients during backpropagation. Both backends default to a
``tanh``-based surrogate; the JAX backend also supports the SuperSpike
surrogate. See ``eleanor.jax._surrogate`` and ``eleanor.torch._surrogate``.
