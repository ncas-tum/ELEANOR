from typing import Tuple, Union, Callable, Optional, Sequence

import jax
import jax.numpy as jnp
import jax.random as jrand
from chex import Array, PRNGKey
from snnax.snn.layers.stateful import StateShape, StatefulLayer, default_init_fn
from snnax.functional.surrogate import SpikeFn, superspike_surrogate

from eleanor.models.jax.variability import D2DVar

_spike_fn = superspike_surrogate(10.0)


def bruno_step(
    synaptic_input: Array,
    v: Array,
    p: Array,
    cap_divider: float,
    depol_divider: float,
    P_s: float,
    A: float,
    I_0: float,
    E_a: float,
    V_t: float,
    I_dsc: float,
    tau_0: float,
    C_tot: float,
    soft_E: float,
    alpha: float,
    threshold: float,
    dt: float,
):
    @jax.custom_gradient
    def tau_fn(E, E_a):
        tau = 1 / (tau_0 * jnp.exp((E_a / (jnp.abs(E) + soft_E)) ** alpha))

        exponential = (E_a / (jnp.abs(E) + soft_E)) ** alpha
        numerator = alpha * jnp.exp(-exponential) * exponential
        denumerator = tau_0 * soft_E * jnp.abs(E) + tau_0 * E**2
        denumerator = jnp.where(
            denumerator, denumerator, 1.0
        )  # If E is 0 then tangent_E is 0

        tangent_E = (E * numerator) / denumerator
        tangent_E_a = -numerator / (tau_0 * E_a)

        return tau, lambda g: (g * tangent_E, g * tangent_E_a)

    def step(state, synaptic_input):
        v, p, s, cap_divider, depol_divider, E_a, P_s, A, I_0, C_tot = state
        E = v * cap_divider - p * depol_divider

        # # tau = tau_fn(E, E_a)
        tau = 1 / (tau_0 * jnp.exp((E_a / (jnp.abs(E) + soft_E)) ** alpha))

        I_p_new = (jnp.sign(E) * P_s - p) * A * tau
        dp = I_p_new / A
        p_new = jnp.clip(p + 1e-3 * dt * dp, -P_s, P_s)

        I_leak = (I_0 * A * jnp.expm1(v / V_t) + I_dsc) * jnp.sign(v)
        dv = (synaptic_input - I_leak - I_p_new) / C_tot
        v_new = jnp.clip(v + 1e-3 * dt * dv, -5, 5)

        spikes_ref = jax.lax.stop_gradient(s)
        v = (1 - spikes_ref) * v_new + spikes_ref * v
        p = (1 - spikes_ref) * p_new + spikes_ref * p
        s = (v > threshold).astype(jnp.float32)

        p = p_new
        v = v_new

        return (v, p, s, cap_divider, depol_divider, E_a, P_s, A, I_0, C_tot), None

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

    # (v_inner, p_inner, _, _, _, _, _, _, _, _), _ = step((
    #         v,
    #         p,
    #         jnp.zeros_like(p),
    #         cap_divider,
    #         depol_divider,
    #         E_a,
    #         P_s,
    #         A,
    #         I_0,
    #         C_tot,
    #     ), synaptic_input)

    return v_inner, p_inner


class Bruno(StatefulLayer):
    """
    Implementation of FeLIF neuron model [1]_ using Bruno for gradient updates.

    .. math::

        \\frac{dP}{dt} = \\frac{sign(E) P_s - P}{\\tau(E)} \\\\
        \\frac{dV}{dt} = \\frac{I_{in} - \\frac{dP}{dt}}{C_0} \\\\
        \\tau(E) = \\tau_0 exp(\\frac{E_a}tau_0{|E|})

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
    spike_fn: SpikeFn
    tau_fn: Callable[[Array, Array], Array]

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
        ---------
        A : float
            Device area
        t_hzo : float
            Thikness ferroelectrictau_0
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

        self.dt = dt

        @jax.custom_gradient
        def tau_fn(E, E_a):
            tau_0 = self.tau_0
            tau = 1 / (
                self.tau_0 * jnp.exp((E_a / (jnp.abs(E) + self.soft_E)) ** self.alpha)
            )

            exponential = (E_a / (jnp.abs(E) + self.soft_E)) ** self.alpha
            numerator = self.alpha * jnp.exp(-exponential) * exponential
            denumerator = tau_0 * self.soft_E * jnp.abs(E) + tau_0 * E**2
            denumerator = jnp.where(
                denumerator, denumerator, 1.0
            )  # If E is 0 then tangent_E is 0

            tangent_E = (E * numerator) / denumerator
            tangent_E_a = -numerator / (tau_0 * E_a)

            return tau, lambda g: (g * tangent_E, g * tangent_E_a)

        # Legacy
        # @jax.custom_gradient
        # def tau_fn(E, E_a):
        #     tau = self.tau_0 * jnp.exp((E_a / (jnp.abs(E) + self.soft_E)) ** self.alpha)

        #     exp_x = jnp.clip(
        #         (E_a / (jnp.abs(E) + self.soft_E)) ** self.alpha, 0, 1
        #     )
        #     tangent_E = -(
        #         self.tau_0
        #         * E_a
        #         * self.alpha
        #         * E
        #         * jnp.exp(exp_x)
        #         * (self.E_a / (jnp.abs(E) + self.soft_E)) ** (self.alpha + 1)
        #     ) / (jnp.abs(E) * (E + self.soft_E) ** 2)

        #     tangent_E_a = -(
        #         self.tau_0
        #         * self.alpha
        #         * jnp.exp(exp_x)
        #         * exp_x
        #     ) / E_a

        #     return tau, lambda g: (g * tangent_E, g * tangent_E_a)

        self.tau_fn = tau_fn

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
        init_state_pol = -self.P_s * (
            1 + self.P_s_var.variability * jrand.normal(k2, shape)
        )
        # init_state_pol = 0.1478602 * (
        #     1 + self.P_s_var.variability * jrand.normal(k2, shape)
        # )
        init_state_spk = jnp.zeros(shape)
        return [init_state_vol, init_state_pol, init_state_spk]

    @property
    def C_tot(self):
        C_0 = self._eps0 * self.eps_hzo / self.t_hzo * self.A
        return C_0 + self.C_par

    @jax.named_scope("eleanor.models.jax.Bruno")
    def __call__(
        self, state: Array, synaptic_input: Array, *, key: Optional[PRNGKey] = None
    ) -> Tuple[Sequence[Array], Sequence[Array]]:
        def step(state, synaptic_input):
            v, p, s, cap_divider, depol_divider, E_a, P_s, A, I_0, C_tot = state
            E = v * cap_divider - p * depol_divider

            tau = self.tau_fn(E, E_a)

            I_p_new = (jnp.sign(E) * P_s - p) * A * tau
            # I_p_new = (jnp.sign(E) * P_s - p) * A / tau  # Legacy
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

            tau = self.tau_fn(E, E_a)

            I_p_new = (jnp.sign(E) * P_s - p) * A * jax.lax.stop_gradient(tau)
            # I_p_new = (jnp.sign(E) * P_s - p) * A / jax.lax.stop_gradient(tau) # Legacy
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
        # p = (1 - spikes_ref) * p - (spikes_ref * 0.1478602)  # Legacy
        s = self.spike_fn(v - self.V_thr)

        state = [v, p, s]
        return [state, [s, v, p]]


class NoBruno(Bruno):
    @jax.named_scope("eleanor.models.NoBruno")
    def __call__(
        self, state: Array, synaptic_input: Array, *, key: Optional[PRNGKey] = None
    ) -> Tuple[Sequence[Array], Sequence[Array]]:
        def step(
            v, p, synaptic_input, cap_divider, depol_divider, E_a, P_s, A, I_0, C_tot
        ):
            E = v * cap_divider - p * depol_divider

            tau = self.tau_fn(E, E_a)

            I_p_new = (jnp.sign(E) * P_s - p) * A * jax.lax.stop_gradient(tau)
            # I_p_new = (jnp.sign(E) * P_s - p) * A / jax.lax.stop_gradient(tau)  # Legacy
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

        v, p = step(
            v, p, synaptic_input, cap_divider, depol_divider, E_a, P_s, A, I_0, C_tot
        )

        spikes_ref = jax.lax.stop_gradient(s)
        v = (1 - spikes_ref) * v - 1.5 * spikes_ref
        # p = (1 - spikes_ref) * p - (spikes_ref * P_s)  # 0.05308533)
        p = (1 - spikes_ref) * p - (spikes_ref * 0.1478602)  # Replicate results
        s = self.spike_fn(v - self.V_thr)

        state = [v, p, s]
        return [state, [s, v, p]]
