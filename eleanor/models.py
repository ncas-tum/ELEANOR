from typing import Tuple, Union, Callable, Optional, Sequence

import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrand
from jax import custom_jvp
from chex import Array, PRNGKey

from eleanor.variability import D2DVar
from snnax.snn.layers.stateful import StateShape, StatefulLayer, default_init_fn
from snnax.functional.surrogate import SpikeFn, superspike_surrogate

_spike_fn = superspike_surrogate(10.0)


class Scaler(eqx.Module):
    """
    Simple module to scale the output and gradient of a layer.

    Attributes
    ----------
    scale: float
        Multiplicative factor of the input.
    grad_scale: float
        Multiplicative factor of the gradient (default the same as scale).
    """

    scale: float
    grad_scale: float

    def __init__(self, scale: float = 1.0, grad_scale: float = None) -> None:
        self.scale = scale
        if grad_scale is None:
            self.grad_scale = scale
        else:
            self.grad_scale = grad_scale

    @jax.named_scope("eleanor.models.Scaler")
    def __call__(self, x: Array, *, key: Optional[PRNGKey] = None) -> Array:
        """ """

        @custom_jvp
        def ste(x):
            return x * self.scale

        @ste.defjvp
        def f_jvp(primals, tangents):
            (x,) = primals
            (x_dot,) = tangents
            primal_out = ste(x)
            return primal_out, x_dot * self.grad_scale

        return ste(x)


class FeLIF(StatefulLayer):
    """
    Implementation of FeLIF neuron model [1]_ using Bruno for gradient updates.

    .. math::

        \\frac{dP}{dt} = \\frac{sign(E) P_s - P}{\\tau(E)} \\\\
        \\frac{dV}{dt} = \\frac{I_{in} - \\frac{dP}{dt}}{C_0} \\\\
        \\tau(E) = \\tau_0 exp(\\frac{E_a}{|E|})

    .. [1] P. Gibertini, L. Fehlings, T. Mikolajick, E. Chicca, D. Kappel and E. Covi, "Coincidence Detection with an Analog Spiking Neuron Exploiting Ferroelectric Polarization," 2024 IEEE International Symposium on Circuits and Systems (ISCAS), Singapore, Singapore, 2024, pp. 1-5, doi: 10.1109/ISCAS58744.2024.10558196. # noqa B950

    Attributes
    ----------
    A: float
        Device area
    t_hzo : float
            Thikness ferroelectric
    t_int : float
        Thikness interlayer
    eps_hzo : float
        Ferroelectric dielectric constant
    eps_int : float
        Interlayer dielectric constant
    E_a : float
            Coercitive field
    P_s : float
        Max polarisation
    tau_0 : float
        Multiplicative factor for switching time constant
    I_0 : float
        Multiplicative factor for leakage current
    V_t : float
        Normalization factor for voltage in leakage current
    C_par : float
        Parasitic capacitance from the circuit
    alpha : float
        To fit tau exponential
    soft_E : float
        Soft boudary for the electric field, avoid tau to diverge
    I_dsc : float
        Discharge current, set the "dendritic time constant"
    V_thr : float
        Spiking threshold
    dt : float
        Time resolution
    spike_fn : SpikeFn
        Spike threshold function with custom surrogate gradient.
    """

    A: float
    t_hzo: float
    t_int: float
    eps_hzo: float
    eps_int: float
    E_a: float
    P_s: float
    tau_0: float
    I_0: float
    V_t: float
    C_par: float
    alpha: float
    soft_E: float
    I_dsc: float
    V_thr: float
    dt: float
    initial_P: float
    spike_fn: SpikeFn

    _eps0: float

    A_var: D2DVar
    E_a_var: D2DVar
    P_s_var: D2DVar
    I_0_var: D2DVar
    Iin_var: D2DVar
    t_hzo_var: D2DVar
    t_int_var: D2DVar

    def __init__(
        self,
        A: float = 25e-12,
        t_hzo: float = 10e-9,
        t_int: float = 1.375e-9,
        eps_hzo: float = 25.2,
        eps_int: float = 33,
        E_a: float = 12.7e8,
        P_s: float = 22e-2,
        tau_0: float = 1e-13,
        I_0: float = 1e-4,
        V_t: float = 0.32,
        C_par: float = 15e-15,
        alpha: float = 1.3,
        soft_E: float = 5e-6,
        I_dsc: float = 10e-12,
        V_thr: float = 2.5,
        initial_P: float = 0.1478602,
        variability: float = 0.0,
        dt: float = 1e-3,
        paramsScale: float = 1e12,
        spike_fn: SpikeFn = _spike_fn,
        init_fn: Optional[Callable] = default_init_fn,
        shape: Optional[StateShape] = None,
        key: PRNGKey = None,
        **kwargs,
    ) -> None:
        """
        Parameters
        ---------
        A : float
            Device area
        t_hzo : float
            Thikness ferroelectric
        t_int : float
            Thikness interlayer
        eps_hzo : float
            Ferroelectric dielectric constant
        eps_int : float
            Interlayer dielectric constant
        E_a : float
            Coercitive field
        P_s : float
            Max polarisation
        tau_0 : float
            Multiplicative factor for switching time constant
        I_0 : float
            Multiplicative factor for leakage current
        V_t : float
            Normalization factor for voltage in leakage current
        C_par : float
            Parasitic capacitance from the circuit
        alpha : float
            To fit tau exponential
        soft_E : float
            Soft boudary for the electric field, avoid tau to diverge
        I_dsc : float
            Discharge current, set the "dendritic time constant"
        V_thr : float
            Spiking threshold
        initial_P : float
            Initial polarization value
        variability: float
            Device to device variability of the device and input current.
        dt : float
            Time resolution
        paramsScale : float
            Scale parameters to avoid underflow
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

        self._eps0 = 8.85418792394420013968e-12 * paramsScale

        A = A * paramsScale
        self.A = A

        self.t_hzo = t_hzo * paramsScale
        self.t_int = t_int * paramsScale
        self.eps_hzo = eps_hzo
        self.eps_int = eps_int

        E_a = E_a / paramsScale
        self.E_a = E_a

        P_s = P_s
        self.P_s = P_s

        self.tau_0 = tau_0
        self.I_0 = I_0
        self.V_t = V_t
        self.C_par = C_par * paramsScale

        self.alpha = alpha
        soft_E = soft_E / paramsScale
        self.soft_E = soft_E

        I_dsc = I_dsc * paramsScale
        self.I_dsc = I_dsc

        self.V_thr = V_thr

        k1, k2, k3, k4, k5, k6, k7 = jrand.split(key, 7)
        self.A_var = D2DVar("A", variability, k1)
        self.E_a_var = D2DVar("E_a", variability, k2)
        self.P_s_var = D2DVar("P_s", variability, k3)
        self.I_0_var = D2DVar("I_0", variability, k4)
        self.Iin_var = D2DVar("Iin", variability, k5)
        self.t_hzo_var = D2DVar("t_hzo", variability, k6)
        self.t_int_var = D2DVar("t_int", variability, k7)

        self.initial_P = initial_P
        self.dt = dt

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
        init_state_pol = self.initial_P * (
            1 + self.P_s_var.variability * jrand.normal(k2, shape)
        )
        init_state_spk = jnp.zeros(shape)
        return [init_state_vol, init_state_pol, init_state_spk]

    @property
    def C_tot(self):
        C_0 = self._eps0 * self.eps_hzo / self.t_hzo * self.A
        return C_0 + self.C_par

    @jax.named_scope("eleanor.models.FeLIF")
    def __call__(
        self, state: Array, synaptic_input: Array, *, key: Optional[PRNGKey] = None
    ) -> Tuple[Sequence[Array], Sequence[Array]]:
        @jax.custom_vjp
        def tau_fn(E, E_a):
            tau = self.tau_0 * jnp.exp((E_a / (jnp.abs(E) + self.soft_E)) ** self.alpha)

            return tau

        def tau_fn_fwd(E, E_a):
            return tau_fn(E, E_a), (E, E_a)

        def tau_fn_bw(res, g):
            (E, E_a) = res
            exp_x = jnp.clip((E_a / (jnp.abs(E) + self.soft_E)) ** self.alpha, 0, 1)
            tau_prime = -(
                self.tau_0
                * E_a
                * self.alpha
                * E
                * jnp.exp(exp_x)
                * (E_a / (jnp.abs(E) + self.soft_E)) ** (self.alpha - 1)
            ) / (jnp.abs(E) * (E + self.soft_E) ** 2)

            tangents_out = (g * tau_prime, None)
            return tangents_out

        tau_fn.defvjp(tau_fn_fwd, tau_fn_bw)

        def step(state, synaptic_input):
            v, p, s, cap_divider, depol_divider, E_a, P_s, A, I_0, C_tot = state
            E = v * cap_divider - p * depol_divider

            tau = tau_fn(E, E_a)

            I_p_new = (jnp.sign(E) * P_s - p) * A / tau
            dp = I_p_new / A
            p_new = jnp.clip(p + 1e-3 * self.dt * dp, -P_s, P_s)

            I_leak = (I_0 * A * jnp.expm1(v / self.V_t) + self.I_dsc) * jnp.sign(v)
            dv = (synaptic_input - I_leak - I_p_new) / C_tot
            v_new = jnp.clip(v + 1e-3 * self.dt * dv, -5, 5)

            spikes_ref = jax.lax.stop_gradient(s)
            v = (1 - spikes_ref) * v_new + spikes_ref * v
            p = (1 - spikes_ref) * p_new + spikes_ref * p
            s = self.spike_fn(v - self.V_thr)

            return (v, p, s, cap_divider, depol_divider, E_a, P_s, A, I_0, C_tot), None

        def step2(
            v, p, synaptic_input, cap_divider, depol_divider, E_a, P_s, A, I_0, C_tot
        ):
            E = v * cap_divider - p * depol_divider

            tau = tau_fn(E, E_a)

            I_p_new = (jnp.sign(E) * P_s - p) * A / jax.lax.stop_gradient(tau)
            dp = I_p_new / A

            I_leak = (I_0 * A * jnp.expm1(v / self.V_t) + self.I_dsc) * jnp.sign(v)
            dv = (synaptic_input - I_leak - I_p_new) / C_tot

            p = jnp.clip(p + self.dt * dp, -P_s, P_s)
            v = jnp.clip(v + self.dt * dv, -5, 5)

            return v, p

        v, p, s = state

        A = self.A_var(self.A, v.shape)
        E_a = self.E_a_var(self.E_a, v.shape)
        P_s = self.P_s_var(self.P_s, v.shape)
        I_0 = self.I_0_var(self.I_0, v.shape)
        t_hzo = self.t_hzo_var(self.t_hzo, v.shape)
        t_int = self.t_int_var(self.t_int, v.shape)

        synaptic_input = self.Iin_var(synaptic_input, v.shape)

        C_0 = self._eps0 * self.eps_hzo / t_hzo * A
        C_tot = C_0 + self.C_par

        cap_divider = self.eps_int / (t_hzo * self.eps_int + t_int * self.eps_hzo)
        depol_divider = (
            1 / self._eps0 * t_int / (t_hzo * self.eps_int + t_int * self.eps_hzo)
        )

        (v_inner, p_inner, _, _, _, _, _, _, _, _), _ = jax.lax.scan(
            step,
            (
                v,
                p,
                jnp.zeros_like(p),
                cap_divider,
                depol_divider,
                E_a,
                P_s,
                A,
                I_0,
                C_tot,
            ),
            jnp.repeat(synaptic_input[None, ...], 1000, axis=0),
        )
        v_outer, p_outer = step2(
            v, p, synaptic_input, cap_divider, depol_divider, E_a, P_s, A, I_0, C_tot
        )

        v = v_outer + jax.lax.stop_gradient(v_inner - v_outer)
        p = p_outer + jax.lax.stop_gradient(p_inner - p_outer)

        spikes_ref = jax.lax.stop_gradient(s)
        v = (1 - spikes_ref) * v - 1.5 * spikes_ref
        p = (1 - spikes_ref) * p - (spikes_ref * P_s)  # 0.05308533)
        s = self.spike_fn(v - self.V_thr)

        state = [v, p, s]
        return [state, [s, v, p]]


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
    V_thr: float
    dt: float
    _eps0: float
    _q: float
    _k: float
    _h: float
    spike_fn: SpikeFn

    A_var: D2DVar
    n_depl_var: D2DVar
    P_s_var: D2DVar
    # t_int_var dist=gauss std=2.2e-10

    paramsScale: float

    def __init__(
        self,
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
        V_thr: float = 3.5,
        dt: float = 1e-3,
        paramsScale: float = 1e12,
        variability: float = 0.0,
        spike_fn: SpikeFn = _spike_fn,
        init_fn: Optional[Callable] = default_init_fn,
        shape: Optional[StateShape] = None,
        key: PRNGKey = None,
        **kwargs,
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
        V_thr : float
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
        self.V_thr = V_thr
        self.dt = dt
        self._eps0 = _eps0
        self._q = _q
        self._k = _k
        self._h = _h
        self.paramsScale = paramsScale

        kA, kdpl, kPs, kfe = jrand.split(key, 4)
        self.A_var = D2DVar("A", variability, kA)
        self.n_depl_var = D2DVar("n_depl", variability, kdpl)
        self.P_s_var = D2DVar("P_s", variability, kPs)
        self.t_fe_var = D2DVar("t_fe", variability, kfe)

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

    @jax.named_scope("eleanor.models.Heracles")
    def __call__(
        self, state: Array, synaptic_input: Array, *, key: Optional[PRNGKey] = None
    ) -> Tuple[Sequence[Array], Sequence[Array]]:
        v, p, spikes = state

        A = self.A_var(self.A, v.shape)
        n_depl = self.n_depl_var(self.n_depl, v.shape)
        P_s = self.P_s_var(self.P_s, v.shape)
        t_fe = self.t_fe_var(self.t_fe, v.shape)

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
            1 / (self.C_fe + self.C_par)
            + 1 / (self._eps0 * self.eps_depl / w_depl * self.A)
        )
        cap_divider = self.eps_depl / (t_fe * self.eps_depl + w_depl * self.eps_fe)
        depol_divider = (
            1 / self._eps0 * w_depl / (t_fe * self.eps_depl + w_depl * self.eps_fe)
        )

        C_tot = jax.lax.stop_gradient(C_tot)
        cap_divider = jax.lax.stop_gradient(cap_divider)
        depol_divider = jax.lax.stop_gradient(depol_divider)

        E = v * cap_divider - p * depol_divider
        w_e = (E - self.e_off) * self.d_e
        w_exp = jnp.exp(-(self.w_b - w_e) * self._q / self._k / self.temp)
        k_plus = self._k * self.temp / self._h * w_exp

        # Gradient clipping of the probability due to exponential of k_plus
        @jax.custom_gradient
        def calcDp(k_plus, prob):
            dp = k_plus * (1 - prob)

            def gradient(g):
                # return (g*2*self.P_s*(1 - prob), -g*2*self.P_s*jnp.log(k_plus))
                # return (
                #     g * 2 * self.P_s * (1 - prob),
                #     jnp.clip(-g * 2 * self.P_s * k_plus, -1e4, 1e4),
                # )
                return (g * (1 - prob), -g * k_plus * 1e-5)

            return dp, gradient

        dp = 2 * P_s * calcDp(k_plus, prob)
        I_p = dp * A

        # FeLIF
        I_leak = (self.I_0 * A * jnp.expm1(v / self.V_t) + self.I_dsc) * jnp.sign(v)
        dv = (synaptic_input - I_leak - I_p) / C_tot

        v_upper = jnp.clip(v + self.dt * dv, 0.0, 3.5)
        p_upper = jnp.clip(p + self.dt * dp, -P_s, P_s)

        spikes_ref = jax.lax.stop_gradient(spikes)
        v_new = (1 - spikes_ref) * v_upper
        p_new = (1 - spikes_ref) * p_upper - (spikes_ref * P_s)

        # Calculate spike
        spikes = self.spike_fn(v_new - self.V_thr)
        state = [v_new, p_new, spikes]
        return [state, [spikes, v_new, p_new, C_tot, k_plus, E]]
