from dataclasses import dataclass
from functools import partial
from typing import Callable, Optional, Sequence

import equinox as eqx
import equinox.internal as eqxi
import jax
import jax.numpy as jnp
import jax.random as jrand
from jaxtyping import Array, PRNGKeyArray

from eleanor.jax.variability import D2DVar, StaticWrapper

from ..surrogate import tanh_surrogate
from ._base import NeuronModel, default_floating_dtype, limexp


def safe_pow(base, alpha):
    safe_base = jnp.where(base == 0, 1.0, base)
    result = safe_base**alpha
    return jnp.where(base == 0, 0.0, result)


@dataclass
class BrunoParams:
    """
    Parameters of the Bruno neural model :cite:`fehlings2026bruno`
    """

    A: float = 25e-12
    t_hzo: float = 10e-9
    t_int: float = 1.375e-9
    eps_hzo: float = 25.2
    eps_int: float = 33
    E_a: float = 12.7e8
    P_s: float = 22e-2
    tau_0: float = 1e-13
    I_0: float = 1e-4
    V_t: float = 0.32
    C_par: float = 15e-15
    alpha: float = 1.3
    soft_E: float = 5e-6
    I_dsc: float = 10e-12
    threshold: float = 2.5
    dt: float = 1e-3


class BrunoCell(NeuronModel):
    """
    Implementation of a Ferroelectric Leaky Integrate and Fire (FeLIF) neuron model using Bruno for training :cite:`fehlings2026bruno`
    """

    shape: Sequence[int] = eqx.field(static=True)
    params: BrunoParams = eqx.field(static=True)
    variability: float = eqx.field(static=True)
    _eps0: float = eqx.field(static=True)
    spikefn: Callable[[Array], Array] = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    return_states: bool = eqx.field(static=True)

    A_var: StaticWrapper
    E_a_var: StaticWrapper
    P_s_var: StaticWrapper
    I_0_var: StaticWrapper
    Iin_var: StaticWrapper
    t_hzo_var: StaticWrapper
    t_int_var: StaticWrapper

    def __init__(
        self,
        shape: Sequence[int],
        params: Optional[BrunoParams] = None,
        param_scale: float = 1e12,
        variability: float = 0.0,
        spikefn: Callable[[Array], Array] = tanh_surrogate,
        dtype=None,
        n_steps=1000,
        return_states: bool = False,
        *,
        key: PRNGKeyArray,
    ):
        self.n_steps = n_steps
        dtype = default_floating_dtype() if dtype is None else dtype

        self.shape = shape
        self.spikefn = spikefn
        self.return_states = return_states

        self._eps0 = 8.85418792394420013968e-12 * param_scale

        if params is None:
            params = BrunoParams()

        self.variability = variability
        params.A = params.A * param_scale
        params.t_hzo = params.t_hzo * param_scale
        params.t_int = params.t_int * param_scale
        params.E_a = params.E_a / param_scale
        params.C_par = params.C_par * param_scale
        params.soft_E = params.soft_E / param_scale
        params.I_dsc = params.I_dsc * param_scale
        self.params = params

        k1, k2, k3, k4, k5, k6, k7 = jrand.split(key, 7)
        self.A_var = StaticWrapper(D2DVar("A", variability, self.shape, k1))
        self.E_a_var = StaticWrapper(D2DVar("E_a", variability, self.shape, k2))
        self.P_s_var = StaticWrapper(D2DVar("P_s", variability, self.shape, k3))
        self.I_0_var = StaticWrapper(D2DVar("I_0", variability, self.shape, k4))
        self.Iin_var = StaticWrapper(D2DVar("Iin", variability, self.shape, k5))
        self.t_hzo_var = StaticWrapper(D2DVar("t_hzo", variability, self.shape, k6))
        self.t_int_var = StaticWrapper(D2DVar("t_int", variability, self.shape, k7))

    @property
    def C_tot(self):
        """Total capacitance of the neuron"""
        A = self.A_var(self.params.A, self.shape)
        t_hzo = self.t_hzo_var(self.params.t_hzo, self.shape)

        C_0 = self._eps0 * self.params.eps_hzo / t_hzo * A
        return C_0 + self.params.C_par

    def init_state(self) -> Sequence[Array]:
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
        P_s = self.P_s_var(self.params.P_s)

        init_state_vol = jnp.zeros(self.shape)
        init_state_pol = -P_s
        init_state_spk = jnp.zeros(self.shape)
        return (init_state_spk, init_state_vol, init_state_pol)

    def __call__(
        self,
        state: Sequence[Array],
        isyn: Array,
        *,
        key: Optional[PRNGKeyArray] = None,
    ):
        s, v, p = state

        A = self.A_var(self.params.A)
        E_a = self.E_a_var(self.params.E_a)
        P_s = self.P_s_var(self.params.P_s)
        I_0 = self.I_0_var(self.params.I_0)
        t_hzo = self.t_hzo_var(self.params.t_hzo)
        t_int = self.t_int_var(self.params.t_int)

        C_0 = self._eps0 * self.params.eps_hzo / t_hzo * A
        C_tot = C_0 + self.params.C_par

        cap_divider = self.params.eps_int / (
            t_hzo * self.params.eps_int + t_int * self.params.eps_hzo
        )
        depol_divider = (
            1
            / self._eps0
            * t_int
            / (t_hzo * self.params.eps_int + t_int * self.params.eps_hzo)
        )

        @jax.custom_jvp
        def tau_fn(E, E_a):
            return 1 / (
                self.params.tau_0
                * limexp((E_a / (jnp.abs(E) + self.params.soft_E)) ** self.params.alpha)
            )

        @tau_fn.defjvp
        def tau_fn_jvp(primals, tangents):
            E, E_a = primals
            dE, dE_a = tangents
            tau = tau_fn(E, E_a)

            exponential = safe_pow(
                E_a / (jnp.abs(E) + self.params.soft_E), self.params.alpha
            )
            numerator = self.params.alpha * limexp(-exponential) * exponential

            safe_E = jax.lax.stop_gradient(E)
            denumerator = (
                self.params.tau_0 * self.params.soft_E * jnp.abs(safe_E)
                + self.params.tau_0 * safe_E**2
            )

            safe_denom = jnp.where(E == 0, 1.0, denumerator)
            dtau_dE = jnp.where(E == 0, 0.0, (E * numerator) / safe_denom)

            safe_E_a = jnp.where(E_a == 0, 1.0, E_a)
            dtau_dE_a = jnp.where(
                E_a == 0, 0.0, -numerator / (self.params.tau_0 * safe_E_a)
            )

            tangent_out = dtau_dE * dE + dtau_dE_a * dE_a
            return tau, tangent_out

        def micro_step(carry, isyn, step_dt=1e-3):
            s, v, p = carry
            E = v * cap_divider - p * depol_divider

            I_p_new = (jnp.sign(E) * P_s - p) * A * tau_fn(E, E_a)
            dp = I_p_new / A
            p_new = jnp.clip(p + step_dt * self.params.dt * dp, -P_s, P_s)

            I_leak = (
                I_0 * A * jnp.expm1(v / self.params.V_t) + self.params.I_dsc
            ) * jnp.sign(v)
            dv = (isyn - I_leak - I_p_new) / C_tot
            v_new = jnp.clip(v + step_dt * self.params.dt * dv, -5, 5)

            spikes_ref = jax.lax.stop_gradient(s)
            v = (1 - spikes_ref) * v_new + spikes_ref * v
            p = (1 - spikes_ref) * p_new + spikes_ref * p
            s = self.spikefn(v - self.params.threshold)

            return (s, v, p), None

        (_, v_inner, p_inner), _ = jax.lax.scan(
            partial(micro_step, step_dt=1.0 / self.n_steps),
            (jnp.zeros_like(v), v, p),
            jnp.repeat(isyn[None, ...], self.n_steps, axis=0),
            length=self.n_steps,
        )
        E = v * cap_divider - p * depol_divider

        I_p_new = (jnp.sign(E) * P_s - p) * A * tau_fn(E, E_a)
        dp = I_p_new / A
        p_outer = jnp.clip(p + self.params.dt * dp, -P_s, P_s)

        I_leak = jax.lax.stop_gradient(
            I_0 * A * jnp.expm1(v / self.params.V_t) + self.params.I_dsc
        ) * jnp.sign(v)
        dv = (isyn - I_leak - I_p_new) / C_tot
        v_outer = jnp.clip(v + self.params.dt * dv, -5, 5)

        v = v_outer + jax.lax.stop_gradient(v_inner - v_outer)
        p = p_outer + jax.lax.stop_gradient(p_inner - p_outer)

        spikes_ref = jax.lax.stop_gradient(s)
        v = (1 - spikes_ref) * v - 1.5 * spikes_ref
        p = (1 - spikes_ref) * p - (spikes_ref * P_s)
        s = self.spikefn(v - self.params.threshold)

        if self.return_states:
            return (s, v, p), (s, v, p)
        else:
            return (s, v, p), s


class CheckpointCell(BrunoCell):
    checkpoints: Optional[int] = eqx.field(static=True)

    def __init__(self, *args, **kwargs):
        self.checkpoints = kwargs.pop("checkpoints", None)
        super(CheckpointCell, self).__init__(*args, **kwargs)

    def __call__(
        self, state: Sequence[Array], isyn: Array, *, key: PRNGKeyArray | None = None
    ):
        s, v, p = state

        A = self.A_var(self.params.A)
        E_a = self.E_a_var(self.params.E_a)
        P_s = self.P_s_var(self.params.P_s)
        I_0 = self.I_0_var(self.params.I_0)
        t_hzo = self.t_hzo_var(self.params.t_hzo)
        t_int = self.t_int_var(self.params.t_int)

        C_0 = self._eps0 * self.params.eps_hzo / t_hzo * A
        C_tot = C_0 + self.params.C_par

        cap_divider = self.params.eps_int / (
            t_hzo * self.params.eps_int + t_int * self.params.eps_hzo
        )
        depol_divider = (
            1
            / self._eps0
            * t_int
            / (t_hzo * self.params.eps_int + t_int * self.params.eps_hzo)
        )

        @jax.custom_jvp
        def tau_fn(E, E_a):
            return 1 / (
                self.params.tau_0
                * limexp((E_a / (jnp.abs(E) + self.params.soft_E)) ** self.params.alpha)
            )

        @tau_fn.defjvp
        def tau_fn_jvp(primals, tangents):
            E, E_a = primals
            dE, dE_a = tangents
            tau = tau_fn(E, E_a)

            exponential = safe_pow(
                E_a / (jnp.abs(E) + self.params.soft_E), self.params.alpha
            )
            numerator = self.params.alpha * limexp(-exponential) * exponential

            safe_E = jax.lax.stop_gradient(E)
            denumerator = (
                self.params.tau_0 * self.params.soft_E * jnp.abs(safe_E)
                + self.params.tau_0 * safe_E**2
            )

            safe_denom = jnp.where(E == 0, 1.0, denumerator)
            dtau_dE = jnp.where(E == 0, 0.0, (E * numerator) / safe_denom)

            safe_E_a = jnp.where(E_a == 0, 1.0, E_a)
            dtau_dE_a = jnp.where(
                E_a == 0, 0.0, -numerator / (self.params.tau_0 * safe_E_a)
            )

            tangent_out = dtau_dE * dE + dtau_dE_a * dE_a
            return tau, tangent_out

        def micro_step(carry, isyn, step_dt=1e-3):
            s, v, p = carry
            E = v * cap_divider - p * depol_divider

            I_p_new = (jnp.sign(E) * P_s - p) * A * tau_fn(E, E_a)
            dp = I_p_new / A
            p_new = jnp.clip(p + step_dt * self.params.dt * dp, -P_s, P_s)

            I_leak = jax.lax.stop_gradient(
                I_0 * A * jnp.expm1(v / self.params.V_t) + self.params.I_dsc
            ) * jnp.sign(v)
            dv = (isyn - I_leak - I_p_new) / C_tot
            v_new = jnp.clip(v + step_dt * self.params.dt * dv, -5, 5)

            spikes_ref = jax.lax.stop_gradient(s)
            v = (1 - spikes_ref) * v_new + spikes_ref * v
            p = (1 - spikes_ref) * p_new + spikes_ref * p
            s = self.spikefn(v - self.params.threshold)

            return (s, v, p), None

        if self.checkpoints is not None and self.checkpoints < 0:
            (_, v, p), _ = jax.lax.scan(
                partial(micro_step, step_dt=1.0 / self.n_steps),
                (v, p, jnp.zeros_like(v)),
                jnp.repeat(isyn[None, ...], self.n_steps, axis=0),
                self.n_steps,
            )
        else:
            (_, v, p), _ = eqxi.scan(
                partial(micro_step, step_dt=1.0 / self.n_steps),
                (v, p, jnp.zeros_like(v)),
                jnp.repeat(isyn[None, ...], self.n_steps, axis=0),
                self.n_steps,
                kind="checkpointed",
                checkpoints=self.checkpoints,
            )

        spikes_ref = jax.lax.stop_gradient(s)
        v = (1 - spikes_ref) * v - 1.5 * spikes_ref
        p = (1 - spikes_ref) * p - (spikes_ref * P_s)  # 0.05308533)
        s = self.spikefn(v - self.params.threshold)

        if self.return_states:
            return (s, v, p), (s, v, p)
        else:
            return (s, v, p), s
