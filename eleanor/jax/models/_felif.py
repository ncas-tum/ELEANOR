from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray

from ..surrogate import tanh_surrogate, tau_surr
from ..variability import D2DVar, StaticWrapper
from ._base import NeuronModel, default_floating_dtype


@dataclass
class FeLIFParams:
    """Parameters of the FeLIF neuron model."""

    tau_p: float = 0.6
    tau_m: float = 0.95
    P_s: float = 0.27
    alpha: float = 1.0
    beta: float = 1.0
    tau_alpha: float = 1.3
    E_a: float = 1.0
    soft_E: float = 1e-18
    threshold: float = 1.0
    dt: float = 1e-3


class FeLIFCell(NeuronModel):
    shape: Sequence[int] = eqx.field(static=True)
    params: FeLIFParams = eqx.field(static=True)
    spike_fn: Callable[[Array], Array] = eqx.field(static=True)
    _tau_fn: Callable = eqx.field(static=True)
    return_states: bool = eqx.field(static=True)

    P_s_var: StaticWrapper[D2DVar]
    tau_p_var: StaticWrapper[D2DVar]
    tau_m_var: StaticWrapper[D2DVar]
    threshold_var: StaticWrapper[D2DVar]

    def __init__(
        self,
        shape: Sequence[int],
        params: Optional[FeLIFParams] = None,
        variability: float = 0.0,
        spike_fn: Callable[[Array], Array] = tanh_surrogate,
        dtype=None,
        return_states: bool = False,
        *,
        key: PRNGKeyArray,
    ) -> None:
        dtype = default_floating_dtype() if dtype is None else dtype

        self.shape = shape
        self.spike_fn = spike_fn
        self.return_states = return_states

        if params is None:
            params = FeLIFParams()
        self.params = params

        self._tau_fn = tau_surr(params.tau_alpha, params.E_a, params.soft_E)

        keys = jax.random.split(key, 4)
        self.P_s_var = StaticWrapper(D2DVar("P_s", variability, self.shape, keys[0]))
        self.tau_p_var = StaticWrapper(
            D2DVar("tau_p", variability, self.shape, keys[1])
        )
        self.tau_m_var = StaticWrapper(
            D2DVar("tau_m", variability, self.shape, keys[2])
        )
        self.threshold_var = StaticWrapper(
            D2DVar("threshold", variability, self.shape, keys[3])
        )

    def init_state(self):
        """
        Initialize the state of the FeLIF model.

        Returns
        =======
        Initial state of the FeLIF neuron.

        """

        init_state_vol = jnp.zeros(self.shape)
        init_state_pol = -self.P_s_var(self.params.P_s)
        init_state_spk = jnp.zeros(self.shape)
        return (init_state_spk, init_state_vol, init_state_pol)

    @jax.named_scope("eleanor.models.FeLIF")
    def __call__(
        self,
        state: Sequence[Array],
        synaptic_input: Array,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        s, v, p = state

        P_s = self.P_s_var(self.params.P_s)
        tau_p = self.tau_p_var(self.params.tau_p)
        tau_m = self.tau_m_var(self.params.tau_m)
        threshold = self.threshold_var(self.params.threshold)

        E = v * self.params.alpha - p * self.params.beta
        tau = self._tau_fn(E, tau_p)
        gamma_p = jnp.exp(-self.params.dt * tau)
        gamma = jnp.exp(-self.params.dt / tau_m)

        Ip = P_s * (jnp.sign(E) - p) * self.params.dt * tau
        p = gamma_p * p + (1 - gamma_p) * jnp.sign(E)
        v = gamma * v - (1 - gamma) * Ip + synaptic_input

        spikes_ref = jax.lax.stop_gradient(s)
        p = p - (p + 1) * spikes_ref
        v = v - v * spikes_ref
        s = self.spike_fn(v - threshold)

        if self.return_states:
            return (s, v, p), (s, v, p)
        else:
            return (s, v, p), s
