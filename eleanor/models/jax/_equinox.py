import math
from typing import Tuple, Callable, Sequence, cast
from functools import partial

import jax
import chex
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrand
from chex import Array, PRNGKey
from equinox.nn._misc import default_init

from eleanor.models.jax.variability import D2DVar, StaticWrapper


def default_floating_dtype():
    if jax.config.jax_enable_x64:
        return jnp.float64
    else:
        return jnp.float32


def tanh_surrogate() -> Callable[[Array], Array]:
    """
    Implementation of the sigmoidal surrogate gradient function as described in
    'The remarkable robustness of surrogate gradient learning
    for instilling complex function in spiking neural networks' by Zenke and
    Vogels: (https://www.biorxiv.org/content/10.1101/2020.06.29.176925v1)

    Arguments:
        `beta` (float): Parameter to control the steepness of the surrogate
            gradient. Default is .5.

    Returns:
        A function that returns the surrogate gradient of the heaviside function.
    """

    @jax.custom_jvp
    def heaviside_with_tanh_surrogate(x):
        return jnp.heaviside(x, 1.0)

    @heaviside_with_tanh_surrogate.defjvp
    def f_jvp(primals, tangents):
        (x,) = primals
        (x_dot,) = tangents
        primal_out = heaviside_with_tanh_surrogate(x)
        tangent_out = x_dot * (1.0 - jnp.tanh(x) ** 2)
        return primal_out, tangent_out

    return heaviside_with_tanh_surrogate


_spikefn = tanh_surrogate()


@chex.dataclass
class BrunoParams:
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


class BrunoCell(eqx.Module):
    weight_ih: Array
    # weight_hh: Array
    bias: Array | None
    shape: Sequence[int] = eqx.field(static=True)
    in_features: int = eqx.field(static=True)
    out_features: int = eqx.field(static=True)
    use_bias: bool = eqx.field(static=True)
    params: BrunoParams = eqx.field(static=True)
    variability: float = eqx.field(static=True)
    _eps0: float = eqx.field(static=True)
    spikefn: Callable[[Array], Array] = eqx.field(static=True)
    n_steps: int = eqx.field(static=True)

    A_var: StaticWrapper
    E_a_var: StaticWrapper
    P_s_var: StaticWrapper
    I_0_var: StaticWrapper
    Iin_var: StaticWrapper
    t_hzo_var: StaticWrapper
    t_int_var: StaticWrapper

    def __init__(
        self,
        # shape: Sequence[int],
        in_features: int,
        out_features: int,
        use_bias: bool = True,
        params: BrunoParams = None,
        param_scale: float = 1e12,
        variability: float = 0.0,
        spikefn: Callable[[Array], Array] = _spikefn,
        dtype=None,
        # init_fn=default_init,
        n_steps=1000,
        *,
        key: PRNGKey,
    ):
        self.n_steps = n_steps
        dtype = default_floating_dtype() if dtype is None else dtype
        ihkey, bkey, hhkey = jrand.split(key, 3)
        lim = math.sqrt(1 / out_features)

        inshape = (out_features, in_features)
        # self.weight_ih = default_init(ihkey, inshape, dtype, lim)
        self.weight_ih = jrand.uniform(ihkey, inshape, dtype, minval=0, maxval=lim)
        # self.weight_ih = jnp.zeros(inshape, dtype)

        # connection_matrix = jnp.zeros(inshape, dtype)
        # keys = jrand.split(ihkey, out_features)
        # for neuron_idx in range(out_features):
        #     connected_inputs = jrand.choice(keys[neuron_idx], out_features, shape=(4,), replace=False)
        #     connection_matrix = connection_matrix.at[neuron_idx, connected_inputs].set(1)
        # self.weight_ih = connection_matrix

        # self.weight_hh = default_init(hhkey, (out_features,out_features), dtype, lim)
        if use_bias:
            # self.bias = default_init(bkey, (out_features,), dtype, lim)
            self.bias = jrand.uniform(
                bkey, (out_features,), dtype, minval=0, maxval=lim
            )
        else:
            self.bias = None

        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = use_bias
        self.shape = (out_features,)
        self.spikefn = spikefn

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
        self, input: Array, hidden: Sequence[Array], *, key: PRNGKey | None = None
    ):
        s, v, p = hidden

        A = self.A_var(self.params.A)
        E_a = self.E_a_var(self.params.E_a)
        P_s = self.P_s_var(self.params.P_s)
        I_0 = self.I_0_var(self.params.I_0)
        t_hzo = self.t_hzo_var(self.params.t_hzo)
        t_int = self.t_int_var(self.params.t_int)

        isyn = self.weight_ih @ input  # + self.weight_hh@s
        if self.use_bias:
            isyn = isyn + self.bias
        # isyn = self.Iin_var(isyn)

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

        @jax.custom_gradient
        def tau_fn(E, E_a):
            tau = 1 / (
                self.params.tau_0
                * jnp.exp(
                    (E_a / (jnp.abs(E) + self.params.soft_E)) ** self.params.alpha
                )
            )

            exponential = (E_a / (jnp.abs(E) + self.params.soft_E)) ** self.params.alpha
            numerator = self.params.alpha * jnp.exp(-exponential) * exponential
            denumerator = (
                self.params.tau_0 * self.params.soft_E * jnp.abs(E)
                + self.params.tau_0 * E**2
            )
            denumerator = jnp.where(
                denumerator, denumerator, 1.0
            )  # If E is 0 then tangent_E is 0

            tangent_E = (E * numerator) / denumerator
            tangent_E_a = -numerator / (self.params.tau_0 * E_a)

            return tau, lambda g: (g * tangent_E, g * tangent_E_a)

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
            self.n_steps,
        )
        E = v * cap_divider - p * depol_divider

        I_p_new = (jnp.sign(E) * P_s - p) * A * jax.lax.stop_gradient(tau_fn(E, E_a))
        dp = I_p_new / A
        p_outer = jnp.clip(p + self.params.dt * dp, -P_s, P_s)

        I_leak = (
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

        return (s, v, p)


class CheckpointCell(BrunoCell):
    def __call__(
        self, input: Array, hidden: Sequence[Array], *, key: PRNGKey | None = None
    ):
        s, v, p = hidden

        A = self.A_var(self.params.A, self.shape)
        E_a = self.E_a_var(self.params.E_a, self.shape)
        P_s = self.P_s_var(self.params.P_s, self.shape)
        I_0 = self.I_0_var(self.params.I_0, self.shape)
        t_hzo = self.t_hzo_var(self.params.t_hzo, self.shape)
        t_int = self.t_int_var(self.params.t_int, self.shape)

        isyn = self.Iin_var(input, self.shape)

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

        @jax.custom_gradient
        def tau_fn(E, E_a):
            tau = 1 / (
                self.params.tau_0
                * jnp.exp(
                    (E_a / (jnp.abs(E) + self.params.soft_E)) ** self.params.alpha
                )
            )

            exponential = (E_a / (jnp.abs(E) + self.params.soft_E)) ** self.params.alpha
            numerator = self.params.alpha * jnp.exp(-exponential) * exponential
            denumerator = (
                self.params.tau_0 * self.params.soft_E * jnp.abs(E)
                + self.params.tau_0 * E**2
            )
            denumerator = jnp.where(
                denumerator, denumerator, 1.0
            )  # If E is 0 then tangent_E is 0

            tangent_E = (E * numerator) / denumerator
            tangent_E_a = -numerator / (self.params.tau_0 * E_a)

            return tau, lambda g: (g * tangent_E, g * tangent_E_a)

        E = v * cap_divider - p * depol_divider

        I_p_new = (jnp.sign(E) * P_s - p) * A * jax.lax.stop_gradient(tau_fn(E, E_a))
        dp = I_p_new / A
        p_new = jnp.clip(p + self.params.dt * dp, -P_s, P_s)

        I_leak = (
            I_0 * A * jnp.expm1(v / self.params.V_t) + self.params.I_dsc
        ) * jnp.sign(v)
        dv = (isyn - I_leak - I_p_new) / C_tot
        v_new = jnp.clip(v + self.params.dt * dv, -5, 5)

        spikes_ref = jax.lax.stop_gradient(s)
        v = (1 - spikes_ref) * v_new - 1.5 * spikes_ref
        p = (1 - spikes_ref) * p_new - (spikes_ref * P_s)  # 0.05308533)
        s = self.spikefn(v_new - self.params.threshold)

        return (s, v, p)
