Quickstart
==========

This page runs a FeLIF neuron in each backend so you can verify your install
and see the shape of the API. The same neuron model is available in both
backends with parameter parity; pick whichever fits the rest of your stack.

JAX backend
-----------
    
.. code-block:: python

    import jax
    import jax.numpy as jnp

    from eleanor.jax.models import FeLIFCell, FeLIFParams, RNN

    # Create an RNN with a single FeLIF cell.
    model = RNN(
        FeLIFCell,
        shape=(1,),
        params=FeLIFParams(threshold=1.0, dt=1e-3),
        key=jax.random.key(0),
    )

    # 100 timesteps of constant input current
    inputs = jnp.ones((100, 1)) * 0.5
    spikes = model(inputs)

    print("Output shape:", spikes.shape)
    print("Total spikes:", int(jnp.sum(spikes)))

PyTorch backend
---------------

.. code-block:: python

    import torch

    from eleanor.torch.models import FeLIF

    neuron = FeLIF(tau_p=0.6, tau_m=0.95, threshold=1.0, dt=1e-3)

    # 100 timesteps, batch size 1, 1 neuron
    inputs = torch.ones(100, 1) * 0.5
    spikes = []
    pol = mem = None
    for t in range(inputs.shape[0]):
        spk, pol, mem = neuron(inputs[t], pol, mem)
        spikes.append(spk)
    spikes = torch.stack(spikes)

    print("Output shape:", spikes.shape)
    print("Total spikes:", int(spikes.sum()))

Where to go next
----------------

- :doc:`concepts` — the ferroelectric LIF model and how to read the parameters.
- :doc:`../user_guide/models` — overview of all neuron models
  (FeLIF, Bruno, Heracles).
- :doc:`../user_guide/backends` — when to choose JAX or PyTorch.
- :doc:`../tutorials/index` — full training notebooks.
