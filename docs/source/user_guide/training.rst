Training
========

This guide collects the minimal training patterns. For interactive walkthroughs, see :doc:`../tutorials/index`.

JAX: a minimal training loop
----------------------------

Equinox makes the model a pytree; ``eqx.filter_jit`` and
``eqx.filter_value_and_grad`` handle which leaves are differentiable.
In the case of D2D variability, the ``D2DVar`` parameters are non-differentiable, so they
are automatically filtered out with StaticWrapper with ``eqx.partition``.

.. code-block:: python

    import equinox as eqx
    import jax
    import jax.numpy as jnp
    import optax

    from eleanor.jax.models import FeLIFCell, FeLIFParams, RNN
    from eleanor.jax.variability import StaticWrapper

    key = jax.random.key(0)
    model = RNN(FeLIFCell(shape=(64,), params=FeLIFParams(), key=key))
    params, static = eqx.partition(model, eqx.is_array, is_leaf=lambda x: isinstance(x, StaticWrapper))

    opt = optax.adam(1e-3)
    opt_state = opt.init(eqx.filter(model, eqx.is_array))

    @eqx.filter_jit
    def loss_fn(params, inputs, targets):
        model = eqx.combine(params, static)
        outputs = model(inputs)
        return jnp.mean((outputs.sum(axis=0) - targets) ** 2)

    @eqx.filter_jit
    def step(model, opt_state, inputs, targets):
        loss, grads = eqx.filter_value_and_grad(loss_fn)(model, inputs, targets)
        updates, opt_state = opt.update(grads, opt_state, model)
        model = eqx.apply_updates(model, updates)
        return model, opt_state, loss

PyTorch: a minimal training loop
--------------------------------

ELEANOR's PyTorch neurons are ``snntorch.SpikingNeuron`` subclasses, so they
drop into a standard PyTorch loop:

.. code-block:: python

    import torch
    from eleanor.torch.models import FeLIF

    neuron = FeLIF(tau_p=0.6, tau_m=0.95)
    optim = torch.optim.Adam(neuron.parameters(), lr=1e-3)

    for inputs, targets in dataloader:
        spk, pol, mem = None, None, None
        outputs = []
        for t in range(inputs.shape[0]):
            spk, pol, mem = neuron(inputs[t], pol, mem)
            outputs.append(spk)
        outputs = torch.stack(outputs)

        loss = (outputs.sum(0) - targets).pow(2).mean()
        optim.zero_grad()
        loss.backward()
        optim.step()

Patterns that come up
---------------------

**Mixing layers with neurons.** Both backends compose freely with their host
framework so it can use their native models like ``equinox.nn.Linear`` in JAX, 
and ``torch.nn.Linear`` in PyTorch.

**Surrogate gradients.** The default surrogate is a smoothed ``tanh``. Pass a
different ``spike_fn`` (JAX) or ``spike_grad`` (PyTorch) to swap it.

**Training under variability.** See :doc:`variability` — the typical pattern
is to instantiate the model with ``variability=0.0``, train, then evaluate
across a range of D2D variability levels.

**Checkpointing.** The JAX backend ships an
:class:`eleanor.utils.EquinoxCheckpointHandler` that plugs into Orbax. PyTorch
users can rely on ``torch.save`` / ``torch.load`` as usual.

See also
--------

- :doc:`../tutorials/train` — end-to-end neuron optimization notebook.
- :doc:`../tutorials/shd` — Spiking Heidelberg Digits training.
- :doc:`../tutorials/precision` — training under weight quantization.
