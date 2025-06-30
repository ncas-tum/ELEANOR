from typing import Tuple, Union, Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import jax.random as jrand
from chex import Array, PRNGKey
from snnax.snn.layers.stateful import StateShape, StatefulLayer, default_init_fn
from snnax.functional.surrogate import SpikeFn, superspike_surrogate

from ._surrogate import tau_surr

_spike_fn = superspike_surrogate(10.0)


class FeLIF(StatefulLayer):
    tau_p: float
    tau_m: float
    gamma: float
    P_s: float
    alpha: float
    beta: float
    threshold: float
    dt: float
    spike_fn: SpikeFn
    _tau_fn: Callable

    def __init__(
        self,
        tau_p: float,
        tau_m: float,
        P_s: float = 0.27,
        alpha: float = 1.0,
        beta: float = 1.0,
        dt: float = 1e-3,
        threshold: float = 1.0,
        tau_alpha: float = 1.3,
        E_a: float = 1.0,
        soft_E: float = 1e-18,
        spike_fn: SpikeFn = _spike_fn,
        init_fn: Optional[Callable] = default_init_fn,
        shape: Optional[StateShape] = None,
        key: PRNGKey = None,
        **kwargs,
    ) -> None:
        super().__init__(init_fn, shape)
        self.spike_fn = spike_fn

        self.P_s = P_s
        self.gamma = jnp.exp(-dt / tau_m)
        self.tau_m = tau_m
        self.tau_p = tau_p
        self.alpha = alpha
        self.beta = beta
        self.dt = dt
        self.threshold = threshold
        self._tau_fn = tau_surr(tau_alpha, E_a, soft_E)

    def init_state(
        self, shape: Union[Sequence[int], int], key: PRNGKey, *args, **kwargs
    ) -> Sequence[Array]:
        """
        Initialize the state of the FeLIF model.

        Parameters
        ==========
        shape: Union[Sequence[int], int]
            Input shape of the layer.
        key: PRNGKey
            JAX random key

        Returns
        =======
        Initial state of the FeLIF neuron.

        """
        k1, k2 = jrand.split(key, 2)
        init_state_vol = self.init_fn(shape, k1, *args, **kwargs)
        init_state_pol = self.init_fn(shape, k2, *args, **kwargs) - 1
        init_state_spk = jnp.zeros(shape)
        return [init_state_vol, init_state_pol, init_state_spk]

    @jax.named_scope("eleanor.models.FeLIF")
    def __call__(
        self, state: Array, synaptic_input: Array, *, key: Optional[PRNGKey] = None
    ) -> Tuple[Sequence[Array], Sequence[Array]]:
        v, p, s = state

        E = v * self.alpha - p * self.beta
        tau = self._tau_fn(E, self.tau_p)
        gamma_p = jnp.exp(-self.dt * tau)

        Ip = self.P_s * (jnp.sign(E) - p) * self.dt * tau
        p = gamma_p * p + (1 - gamma_p) * jnp.sign(E)
        v = self.gamma * v - (1 - self.gamma) * Ip + synaptic_input

        spikes_ref = jax.lax.stop_gradient(s)
        p = p - (p + 1) * spikes_ref
        v = v - v * spikes_ref
        v = jnp.clip(v, -5, 5)
        s = self.spike_fn(v - self.threshold)

        # s = synaptic_input + jax.lax.stop_gradient(s - synaptic_input)

        state = [v, p, s]
        return [state, [s, v, p]]
