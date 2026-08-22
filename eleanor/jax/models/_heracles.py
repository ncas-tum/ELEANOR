from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrand
from jaxtyping import Array, PRNGKeyArray

from eleanor.jax.variability import D2DVar, StaticWrapper

from ..surrogate import tanh_surrogate
from ._base import NeuronModel, default_floating_dtype, limexp


@jax.custom_gradient
def _scale_grad(x):
    def gradient(g):
        return 1e-3 * g

    return x, gradient


@dataclass
class HeraclesParams:
    """
    Parameters of the Heracles neural model :cite:`fehlings2025heracles`
    """

    A: float = 25e-12
    t_fe: float = 9.8e-9
    eps_fe: float = 70
    eps_depl: float = 3.6
    q_fix_depl: float = 945e-4
    n_depl: float = 1.4e28
    e_off: float = 2e7
    temp: float = 294
    w_b: float = 1.05
    d_e: float = 7.5e-9
    P_s: float = 27e-2
    I_0: float = 1e-4
    V_t: float = 0.32
    C_par: float = 15e-15
    C_fe: float = 1.5811049864186072e-12
    I_dsc: float = 10e-12
    threshold: float = 3.5
    dt: float = 1e-3


class HeraclesCell(NeuronModel):
    """
    Implementation of Heracles neural model :cite:`fehlings2025heracles`
    """

    shape: Sequence[int] = eqx.field(static=True)
    params: HeraclesParams = eqx.field(static=True)
    variability: float = eqx.field(static=True)
    _eps0: float = eqx.field(static=True)
    _q: float = eqx.field(static=True)
    _k: float = eqx.field(static=True)
    _h: float = eqx.field(static=True)
    spike_fn: Callable[[Array], Array] = eqx.field(static=True)
    param_scale: float = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)
    return_states: bool = eqx.field(static=True)

    # Parameters with variability
    A_var: StaticWrapper[D2DVar]
    n_depl_var: StaticWrapper[D2DVar]
    P_s_var: StaticWrapper[D2DVar]
    t_fe_var: StaticWrapper[D2DVar]

    def __init__(
        self,
        shape: Sequence[int],
        params: HeraclesParams | None = None,
        param_scale: float = 1e12,
        variability: float = 0.0,
        spike_fn: Callable[[Array], Array] = tanh_surrogate,
        dtype=None,
        n_steps=1000,
        return_states: bool = False,
        *,
        key: PRNGKeyArray,
    ) -> None:
        """
        Parameters
        ==========
        shape : StateShape
            if given, the parameters will be expanded into vectors and
            initialized accordingly
        params : HeraclesParams
            Heracles physical parameters
        param_scale : float
            Scale parameters to avoid underflow
        variability : float
            Percentage of device to device variability
        spike_fn : SpikeFn
            Spike threshold function with custom surrogate gradient.
        key : PRNGKey
            used to initialize the parameters when shape is not None
        """
        self.n_steps = n_steps
        dtype = default_floating_dtype() if dtype is None else dtype

        self.shape = shape
        self.spike_fn = spike_fn
        self.param_scale = param_scale
        self.return_states = return_states

        self._eps0 = 8.85418792394420013968e-12 * param_scale  # Vacuum permittivity
        self._q = 1.60217663e-19 * param_scale
        self._k = 1.380649e-23 * param_scale  # Boltzmann constant
        self._h = 6.62607015e-34 * param_scale  # Planck constant

        if params is None:
            params = HeraclesParams()

        self.variability = variability
        params.A = params.A * param_scale
        params.t_fe = params.t_fe * param_scale
        params.e_off = params.e_off / param_scale
        params.d_e = params.d_e * param_scale
        params.C_par = params.C_par * param_scale
        params.I_dsc = params.I_dsc * param_scale
        params.n_depl = params.n_depl / param_scale
        params.C_fe = self._eps0 * params.eps_fe / params.t_fe * params.A
        self.params = params

        kA, kdpl, kPs, kfe = jrand.split(key, 4)
        self.A_var = StaticWrapper(D2DVar("A", variability, shape, kA))
        self.n_depl_var = StaticWrapper(D2DVar("n_depl", variability, shape, kdpl))
        self.P_s_var = StaticWrapper(D2DVar("P_s", variability, shape, kPs))
        self.t_fe_var = StaticWrapper(D2DVar("t_fe", variability, shape, kfe))

    def init_state(self) -> Sequence[Array]:
        """
        Initialize the state of the Heracles model.

        Returns
        =======
        Initial state of the Heracles neuron.

        """
        P_s = self.P_s_var(self.params.P_s)

        init_state_vol = jnp.zeros(self.shape)
        init_state_pol = -P_s
        init_state_spk = jnp.zeros(self.shape)
        return (init_state_spk, init_state_vol, init_state_pol)

    def calculate_params(self, v, p, A, n_depl, P_s, t_fe):
        prob = p / 2 / P_s + 0.5
        e_dummy = v / t_fe
        w_depl_d = (
            (self._eps0 * self.params.eps_fe * e_dummy + self.params.q_fix_depl)
            * self.param_scale
            / self._q
            / n_depl
        )
        w_depl_u = jnp.abs(
            (self._eps0 * self.params.eps_fe * e_dummy - self.params.q_fix_depl)
            * self.param_scale
            / self._q
            / n_depl
        )
        w_depl = w_depl_d * w_depl_u / (prob * w_depl_u + (1 - prob) * w_depl_d)
        C_tot = 1 / (
            1 / (self.params.C_fe + self.params.C_par)
            + 1 / (self._eps0 * self.params.eps_depl / w_depl * A)
        )
        cap_divider = self.params.eps_depl / (
            t_fe * self.params.eps_depl + w_depl * self.params.eps_fe
        )
        depol_divider = (
            1
            / self._eps0
            * w_depl
            / (t_fe * self.params.eps_depl + w_depl * self.params.eps_fe)
        )

        # C_tot = jax.lax.stop_gradient(C_tot)
        # cap_divider = jax.lax.stop_gradient(cap_divider)
        # depol_divider = jax.lax.stop_gradient(depol_divider)

        return prob, C_tot, cap_divider, depol_divider

    @jax.named_scope("eleanor.models.jax.Heracles")
    def __call__(
        self,
        state: Sequence[Array],
        synaptic_input: Array,
        *,
        key: PRNGKeyArray | None = None,
    ):
        spikes, v, p = state

        A = self.A_var(self.params.A)
        n_depl = self.n_depl_var(self.params.n_depl)
        P_s = self.P_s_var(self.params.P_s)
        t_fe = self.t_fe_var(self.params.t_fe)

        def step(state, _, int_div=1):
            v, p, s = state

            prob, C_tot, cap_divider, depol_divider = self.calculate_params(
                v, p, A, n_depl, P_s, t_fe
            )

            E = v * cap_divider - p * depol_divider
            w_e = (E - self.params.e_off) * self.params.d_e
            w_exp_down = limexp(
                -jax.nn.relu(self.params.w_b - w_e)
                * self._q
                / self._k
                / self.params.temp
            )
            k_down = self._k * self.params.temp / self._h * w_exp_down
            w_exp_up = limexp(
                -jax.nn.relu(self.params.w_b + w_e)
                * self._q
                / self._k
                / self.params.temp
            )
            k_up = self._k * self.params.temp / self._h * w_exp_up

            dp = 2 * P_s * (k_down * (1 - prob) - k_up * prob)
            I_p = dp * A

            # FeLIF
            I_leak = jax.lax.stop_gradient(
                self.params.I_0 * A * jnp.expm1(v / self.params.V_t) + self.params.I_dsc
            ) * jnp.sign(v)
            dv = (synaptic_input - I_leak - I_p) / C_tot

            v_new = jnp.clip(v + int_div * self.params.dt * dv, -5, 5)
            p_new = jnp.clip(p + int_div * self.params.dt * dp, -P_s, P_s)

            spikes_ref = jax.lax.stop_gradient(s)
            v = (1 - spikes_ref) * v_new + spikes_ref * v
            p = (1 - spikes_ref) * p_new + spikes_ref * p
            s = self.spike_fn(v - self.params.threshold)

            return (
                v,
                p,
                s,
            ), None

        step_state = (
            v,
            p,
            jnp.zeros_like(p),
        )
        (v_inner, p_inner, _), _ = jax.lax.scan(
            partial(step, int_div=1.0 / self.n_steps),
            step_state,
            None,
            length=self.n_steps,
        )
        (v_outer, p_outer, _), _ = step(step_state, None, int_div=1)

        v = v_outer + jax.lax.stop_gradient(v_inner - v_outer)
        p = p_outer + jax.lax.stop_gradient(p_inner - p_outer)

        spikes_ref = jax.lax.stop_gradient(spikes)
        v_new = (1 - spikes_ref) * v
        p_new = (1 - spikes_ref) * p - (spikes_ref * P_s)

        # Calculate spike
        spikes = self.spike_fn(v_new - self.params.threshold)

        if self.return_states:
            return (spikes, v_new, p_new), (spikes, v_new, p_new)
        else:
            return (spikes, v_new, p_new), spikes
