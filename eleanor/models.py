from typing import Tuple, Union, Callable, Optional, Sequence

import jax
import jax.numpy as jnp
from chex import Array, PRNGKey
from snnax.snn.layers.stateful import StateShape, StatefulLayer, default_init_fn
from snnax.functional.surrogate import SpikeFn, superspike_surrogate

_spike_fn = superspike_surrogate(10.0)


class FeLIF(StatefulLayer):
    """
    Implementation of FeLIF neuron model [1]_

    .. math::

        \\frac{dP}{dt} = \\frac{sign(E) P_s - P}{\\tau(E)} \\\\
        \\frac{dV}{dt} = \\frac{I_{in} - \\frac{dP}{dt}}{C_0} \\\\
        \\tau(E) = \\tau_0 exp(\\frac{E_a}{|E|})

    .. [1] P. Gibertini, L. Fehlings, T. Mikolajick, E. Chicca, D. Kappel and E. Covi, "Coincidence Detection with an Analog Spiking Neuron Exploiting Ferroelectric Polarization," 2024 IEEE International Symposium on Circuits and Systems (ISCAS), Singapore, Singapore, 2024, pp. 1-5, doi: 10.1109/ISCAS58744.2024.10558196.

    Attributes
    ----------
    spike_fn
    A: float
        Device area
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
    C_tot : float
        Total capacitance from the circuit
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
    cap_divider : float
        Capacitor constant divider
    depol_divider : float
        Polarization constant divider
    """

    A: float
    E_a: float
    P_s: float
    tau_0: float
    I_0: float
    V_t: float
    C_tot: float
    alpha: float
    soft_E: float
    I_dsc: float
    V_thr: float
    dt: float
    cap_divider: float
    depol_divider: float
    spike_fn: SpikeFn

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
        soft_E: float = 5e6,
        I_dsc: float = 10e-12,
        V_thr: float = 2.5,
        dt: float = 1e-3,
        paramsScale: float = 1e12,
        spike_fn: SpikeFn = _spike_fn,
        init_fn: Optional[Callable] = default_init_fn,
        shape: Optional[StateShape] = None,
        key: Optional[PRNGKey] = None,
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
        dt : float
            Time resolution
        paramsScale : float
            Scale parameters to avoid underflow
        spike_fn : SpikeFn
            Spike threshold function with custom surrogate gradient.
        init_fn : Callable
            FUnction to initialize the initial state of the spiking neurons. Defaults to initialization with zeros if nothing else if provided.
        shape : StateShape
            if given, the parameters will be expanded into vectors and initialized accordingly
        key : PRNGKey
            used to initialize the parameters when shape is not None
        """
        super().__init__(init_fn, shape)
        self.spike_fn = spike_fn

        _eps0 = 8.85418792394420013968e-12 * paramsScale

        A = A * paramsScale
        self.A = A

        t_hzo = t_hzo * paramsScale
        t_int = t_int * paramsScale

        E_a = E_a / paramsScale
        self.E_a = E_a

        P_s = P_s
        self.P_s = P_s

        self.tau_0 = tau_0
        self.I_0 = I_0
        self.V_t = V_t
        C_par = C_par * paramsScale

        self.alpha = alpha
        soft_E = soft_E / paramsScale
        self.soft_E = soft_E

        I_dsc = I_dsc * paramsScale
        self.I_dsc = I_dsc

        self.V_thr = V_thr

        C_0 = _eps0 * eps_hzo / t_hzo * A
        C_tot = C_0 + C_par
        self.C_tot = C_tot

        cap_divider = eps_int / (t_hzo * eps_hzo + t_int * eps_int)
        self.cap_divider = cap_divider
        depol_divider = 1 / _eps0 * t_int / (t_hzo * eps_hzo + t_int * eps_int)
        self.depol_divider = depol_divider
        self.dt = dt

    def init_state(
        self, shape: Union[Sequence[int], int], key: PRNGKey, *args, **kwargs
    ) -> Sequence[Array]:
        init_state_vol = self.init_fn(shape, key, *args, **kwargs)
        init_state_pol = jnp.zeros(shape) - self.P_s
        init_state_spk = jnp.zeros(shape)
        return [init_state_vol, init_state_pol, init_state_spk]

    def __call__(
        self, state: Array, synaptic_input: Array, *, key: Optional[PRNGKey] = None
    ) -> Tuple[Sequence[Array], Sequence[Array]]:

        def updatePol(v, p):

            @jax.custom_vjp
            def tau_fn(E):
                tau = self.tau_0 * jnp.exp(
                    (self.E_a / (jnp.abs(E) + 5e-6)) ** self.alpha
                )

                return tau

            def tau_fn_fwd(E):
                return tau_fn(E), (E,)

            def tau_fn_bw(res, g):
                (E,) = res
                exp_x = jnp.clip(
                    (self.E_a / (jnp.abs(E) + self.soft_E)) ** self.alpha, 0, 1
                )
                tau_prime = -(
                    self.tau_0
                    * self.E_a
                    * self.alpha
                    * E
                    * jnp.exp(exp_x)
                    * (self.E_a / (jnp.abs(E) + self.soft_E)) ** (self.alpha - 1)
                ) / (jnp.abs(E) * (E + self.soft_E) ** 2)

                tangents_out = (g * tau_prime,)
                return tangents_out

            tau_fn.defvjp(tau_fn_fwd, tau_fn_bw)

            def pol_step(state, _):
                p, _ = state
                E = v * self.cap_divider - p * self.depol_divider

                tau = self.tau_0 * jnp.exp(
                    (self.E_a / (jnp.abs(E) + 5e-6)) ** self.alpha
                )

                I_p_new = (jnp.sign(E) * self.P_s - p) * self.A / tau
                dp = I_p_new / self.A
                p = jnp.clip(p + 1e-3 * self.dt * dp, -self.P_s, self.P_s)
                return (p, I_p_new), None

            def pol_step2(p):
                E = v * self.cap_divider - p * self.depol_divider

                tau = tau_fn(E)

                I_p_new = (
                    (jnp.sign(E) * self.P_s - p) * self.A / jax.lax.stop_gradient(tau)
                )
                dp = I_p_new / self.A
                p = jnp.clip(p + self.dt * dp, -self.P_s, self.P_s)
                return p, I_p_new

            (p_inner, I_p_inner), _ = jax.lax.scan(
                pol_step, (p, jnp.zeros_like(p)), jnp.arange(1000)
            )

            p_outer, I_p_outer = pol_step2(p)

            p = p_outer + jax.lax.stop_gradient(p_inner - p_outer)
            I_p = I_p_outer + jax.lax.stop_gradient(I_p_inner - I_p_outer)

            return p, I_p

        v, p, spikes = state

        p_upper, I_p = updatePol(v, p)

        I_leak = (self.I_0 * self.A * jnp.expm1(v / self.V_t) + self.I_dsc) * jnp.sign(
            v
        )
        # Multiply by 1000 to convert to current.
        dv = (synaptic_input * 1000 - I_leak - I_p) / self.C_tot

        v_upper = jnp.clip(v + self.dt * dv, -5, 5)

        spikes_ref = jax.lax.stop_gradient(spikes)
        v_new = (1 - spikes_ref) * v_upper
        p_new = (1 - spikes_ref) * p_upper - (spikes_ref * self.P_s)

        spikes = self.spike_fn(v - self.V_thr)
        state = [v_new, p_new, spikes]
        return [state, [spikes, v_new, p_new]]
