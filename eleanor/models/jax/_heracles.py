from typing import Tuple, Union, Callable, Optional, Sequence
from functools import partial

import jax
import jax.numpy as jnp
import jax.random as jrand
from chex import Array, PRNGKey
from snnax.snn.layers.stateful import StateShape, StatefulLayer, default_init_fn
from snnax.functional.surrogate import SpikeFn, superspike_surrogate

from eleanor.models.jax.variability import D2DVar

_spike_fn = superspike_surrogate(10.0)


@jax.custom_gradient
def _scale_grad(x):
    def gradient(g):
        return 1e-3 * g

    return x, gradient


class Heracles(StatefulLayer):
    """
    Implementation of Heracles neural model [2]_

    .. [2] (https://github.com/bics-rug/heracles)
    """

    A: float
    t_fe: float
    eps_fe: float
    eps_depl: float
    q_fix_depl: float
    n_depl: float
    e_off: float
    temp: float
    w_b: float
    d_e: float
    P_s: float
    I_0: float
    V_t: float
    C_par: float
    C_fe: float
    C_tot_init: float
    I_dsc: float
    threshold: float
    dt: float
    _eps0: float
    _q: float
    _k: float
    _h: float
    spike_fn: SpikeFn

    # Parameters with variability
    A_var: D2DVar
    n_depl_var: D2DVar
    P_s_var: D2DVar
    t_fe_var: D2DVar

    paramsScale: float

    def __init__(
        self,
        shape: Sequence[int],
        A: float = 25e-12,
        t_fe: float = 9.8e-9,
        eps_fe: float = 70,
        eps_depl: float = 3.6,
        q_fix_depl: float = 945e-4,
        n_depl: float = 1.4e28,
        e_off: float = 2e7,
        temp: float = 294,
        w_b: float = 1.05,
        d_e: float = 7.5e-9,
        P_s: float = 27e-2,
        I_0: float = 1e-4,
        V_t: float = 0.32,
        C_par: float = 15e-15,
        I_dsc: float = 10e-12,
        threshold: float = 3.5,
        dt: float = 1e-3,
        paramsScale: float = 1e12,
        variability: float = 0.0,
        spike_fn: SpikeFn = _spike_fn,
        init_fn: Optional[Callable] = default_init_fn,
        *,
        key: PRNGKey,
    ) -> None:
        """
        Parameters
        ==========
        A : float
            Device area [m²]
        t_fe : float
            Ferroelectric layer thickness [m]
        eps_fe : float
            Relative permittivity of HZO [1]
        eps_depl : float
            Relative permittivity of the electrode [1]
        q_fix_depl : float
            Fixed charge at depletion/ferrolectric interface [C/m²]
        n_depl : float
            Carrier density in interface depletion region [1/m³]
        e_off : float
            Offset electric field [V/m]
        temp : float
            Device temperature [K]
        w_b : float
            Switching barrier height [eV]
        d_e : float
            Electric field action distance [m]
        P_s : float
            Saturation polarization [C/m²]
        I_0 : float
            Leakage current scalar [A/m²]
        V_t : float
            Scaling factor for voltage in leakage current [V]
        C_par : float
            Parasitic capacitance from the circuit [F]
        I_dsc : float
            Discharge current, set the "dendritic time constant" [A]
        threshold : float
            Spiking threshold [V]
        dt : float
            Time resolution [s]
        paramsScale : float
            Scale parameters to avoid underflow
        variability : float
            Percentage of device to device variability
        spike_fn : SpikeFn
            Spike threshold function with custom surrogate gradient.
        init_fn : Callable
            Function to initialize the initial state of the spiking neurons.
            Defaults to initialization with zeros if nothing else if provided.
        shape : StateShape
            if given, the parameters will be expanded into vectors and
            initialized accordingly
        key : PRNGKey
            used to initialize the parameters when shape is not None
        """
        super().__init__(init_fn, shape)
        self.spike_fn = spike_fn

        A = A * paramsScale
        t_fe = t_fe * paramsScale
        # eps_depl = eps_depl * paramsScale
        # eps_fe = eps_fe * paramsScale
        q_fix_depl = q_fix_depl  # * paramsScale
        e_off = e_off / paramsScale
        d_e = d_e * paramsScale
        C_par = C_par * paramsScale
        I_dsc = I_dsc * paramsScale
        n_depl = n_depl / paramsScale

        _eps0 = 8.85418792394420013968e-12 * paramsScale  # Vacuum permittivity
        _q = 1.60217663e-19 * paramsScale
        _k = 1.380649e-23 * paramsScale  # Boltzmann constant
        _h = 6.62607015e-34 * paramsScale  # Planck constant

        # Initial values, to be checked whether sensible
        e_dummy = 0
        prob = 0

        C_fe = _eps0 * eps_fe / t_fe * A
        w_depl_d = (_eps0 * eps_fe * e_dummy + q_fix_depl) * paramsScale / _q / n_depl
        w_depl_u = jnp.abs(
            (_eps0 * eps_fe * e_dummy - q_fix_depl) * paramsScale / _q / n_depl
        )
        w_depl = w_depl_d * w_depl_u / (prob * w_depl_u + (1 - prob) * w_depl_d)
        C_tot_init = 1 / (1 / (C_fe + C_par) + 1 / (_eps0 * eps_depl / w_depl * A))

        # Save parameters in object
        self.A = A
        self.t_fe = t_fe
        self.eps_fe = eps_fe
        self.eps_depl = eps_depl
        self.q_fix_depl = q_fix_depl
        self.n_depl = n_depl
        self.e_off = e_off
        self.temp = temp
        self.w_b = w_b
        self.d_e = d_e
        self.P_s = P_s
        self.I_0 = I_0
        self.V_t = V_t
        self.C_par = C_par
        self.C_fe = C_fe
        self.C_tot_init = C_tot_init
        self.I_dsc = I_dsc
        self.threshold = threshold
        self.dt = dt
        self._eps0 = _eps0
        self._q = _q
        self._k = _k
        self._h = _h
        self.paramsScale = paramsScale

        kA, kdpl, kPs, kfe = jrand.split(key, 4)
        self.A_var = D2DVar("A", variability, shape, kA)
        self.n_depl_var = D2DVar("n_depl", variability, shape, kdpl)
        self.P_s_var = D2DVar("P_s", variability, shape, kPs)
        self.t_fe_var = D2DVar("t_fe", variability, shape, kfe)

    def init_state(
        self, shape: Union[Sequence[int], int], key: PRNGKey, *args, **kwargs
    ) -> Sequence[Array]:
        """
        Initialize the state of the Heracles model.

        Parameters
        ==========
        shape: Union[Sequence[int], int]
            Input shape of the layer.
        key: PRNGKey
            JAX random key

        Returns
        =======
        Initial state of the Heracles neuron.

        """
        init_state_vol = self.init_fn(shape, key, *args, **kwargs)
        init_state_pol = jnp.zeros(shape) - self.P_s
        init_state_spk = jnp.zeros(shape)
        return [init_state_vol, init_state_pol, init_state_spk]

    def calculate_params(self, v, p, A, n_depl, P_s, t_fe):
        prob = p / 2 / P_s + 0.5
        e_dummy = v / t_fe
        w_depl_d = (
            (self._eps0 * self.eps_fe * e_dummy + self.q_fix_depl)
            * self.paramsScale
            / self._q
            / n_depl
        )
        w_depl_u = jnp.abs(
            (self._eps0 * self.eps_fe * e_dummy - self.q_fix_depl)
            * self.paramsScale
            / self._q
            / n_depl
        )
        w_depl = w_depl_d * w_depl_u / (prob * w_depl_u + (1 - prob) * w_depl_d)
        C_tot = 1 / (
            1 / (self.C_fe + self.C_par) + 1 / (self._eps0 * self.eps_depl / w_depl * A)
        )
        cap_divider = self.eps_depl / (t_fe * self.eps_depl + w_depl * self.eps_fe)
        depol_divider = (
            1 / self._eps0 * w_depl / (t_fe * self.eps_depl + w_depl * self.eps_fe)
        )

        C_tot = jax.lax.stop_gradient(C_tot)
        cap_divider = jax.lax.stop_gradient(cap_divider)
        depol_divider = jax.lax.stop_gradient(depol_divider)

        return prob, C_tot, cap_divider, depol_divider

    @jax.named_scope("eleanor.models.jax.Heracles")
    def __call__(
        self, state: Array, synaptic_input: Array, *, key: Optional[PRNGKey] = None
    ) -> Tuple[Sequence[Array], Sequence[Array]]:
        v, p, spikes = state

        A = self.A_var(self.A)
        n_depl = self.n_depl_var(self.n_depl)
        P_s = self.P_s_var(self.P_s)
        t_fe = self.t_fe_var(self.t_fe)

        def step(state, synaptic_input, int_div=1):
            v, p, s, A, n_depl, P_s, t_fe, _, _ = state

            prob, C_tot, cap_divider, depol_divider = self.calculate_params(
                v, p, A, n_depl, P_s, t_fe
            )

            E = v * cap_divider - p * depol_divider
            w_e = (E - self.e_off) * self.d_e
            w_exp_down = jnp.exp(
                -jax.nn.relu(self.w_b - w_e) * self._q / self._k / self.temp
            )
            k_down = self._k * self.temp / self._h * w_exp_down
            w_exp_up = jnp.exp(
                -jax.nn.relu(self.w_b + w_e) * self._q / self._k / self.temp
            )
            k_up = self._k * self.temp / self._h * w_exp_up

            dp = _scale_grad(2 * P_s * (k_down * (1 - prob) - k_up * prob))
            I_p = dp * A

            # FeLIF
            I_leak = jax.lax.stop_gradient(
                self.I_0 * A * jnp.expm1(v / self.V_t) + self.I_dsc
            ) * jnp.sign(v)
            dv = (synaptic_input - I_leak - I_p) / C_tot

            v_new = jnp.clip(v + int_div * self.dt * dv, -5, 5)
            p_new = jnp.clip(p + int_div * self.dt * dp, -P_s, P_s)

            spikes_ref = jax.lax.stop_gradient(s)
            v = (1 - spikes_ref) * v_new + spikes_ref * v
            p = (1 - spikes_ref) * p_new + spikes_ref * p
            s = self.spike_fn(v - self.threshold)

            return (
                v,
                p,
                s,
                A,
                n_depl,
                P_s,
                t_fe,
                I_leak,
                k_down * (1 - prob) - k_up * prob,
            ), None

        step_state = (
            v,
            p,
            jnp.zeros_like(p),
            A,
            n_depl,
            P_s,
            t_fe,
            jnp.zeros_like(t_fe),
            jnp.zeros_like(t_fe),
        )
        (v_inner, p_inner, _, _, _, _, _, k_down, k_up), _ = jax.lax.scan(
            partial(step, int_div=1e-3),
            step_state,
            jnp.repeat(synaptic_input[None, ...], 1000, axis=0),
        )
        (v_outer, p_outer, _, _, _, _, _, _, _), _ = step(
            step_state, synaptic_input, int_div=1
        )

        v = v_outer + jax.lax.stop_gradient(v_inner - v_outer)
        p = p_outer + jax.lax.stop_gradient(p_inner - p_outer)

        spikes_ref = jax.lax.stop_gradient(spikes)
        v_new = (1 - spikes_ref) * v
        p_new = (1 - spikes_ref) * p - (spikes_ref * P_s)

        # Calculate spike
        spikes = self.spike_fn(v_new - self.threshold)
        state = [v_new, p_new, spikes]
        return [state, [spikes, v_new, p_new, k_down, k_up]]
