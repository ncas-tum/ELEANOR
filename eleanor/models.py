from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from spyx.axn import arctan

_spike_fn = arctan(k=2)


class FeLIF(eqx.Module):
    out_size: int = eqx.field(static=True)
    P_s: float = eqx.field(static=True)  # max polarisation
    A: float = eqx.field(static=True)
    C_tot: float = eqx.field(static=True)
    threshold: float = eqx.field(static=True)
    step: Callable[[Any, Any], Any] = eqx.field(static=True)
    updatePol: Callable[[Any, Any], Any] = eqx.field(static=True)
    cap_divider: float = eqx.field(static=True)
    depol_divider: float = eqx.field(static=True)

    def __init__(
        self,
        out_size,
        A=25e-12,  # device area
        t_hzo=10e-9,  # thikness ferroelectric
        t_int=1.375e-9,  # thikness interlayer
        eps_hzo=25.2,  # ferroelectric dielectric constant
        eps_int=33,  # interlayer dielectric constant
        E_a=12.7e8,  # coercitive field
        P_s=22e-2,  # max polarisation
        tau_0=1e-13,  # multiplicative factor for switching time constant
        I_0=1e-4,  # mult factor for leakage current
        V_t=0.32,  # normalization factor for voltage in leakage current
        C_par=15e-15,  # parasitic capacitance form the circuit
        alpha=1.3,  # to fit tau exponential
        soft_E=5e6,  # soft boudary for the electric field, avoid tau to diverge
        I_dsc=10e-12,  # discharge current, set the "dendritic time constant"
        V_thr=2.5,
        dt=1e-3,  # 1us timestep resolution
        innerStep=1000,
        paramsScale=1e12,  # Scale parameters to avoid underflow
        spike_fn=_spike_fn,
        stepFull=False,
        *,
        key=None,
    ):
        _eps0 = 8.85418792394420013968e-12 * paramsScale
        self.out_size = out_size

        A = A * paramsScale
        self.A = A
        t_hzo = t_hzo * paramsScale
        t_int = t_int * paramsScale
        eps_hzo = eps_hzo
        eps_int = eps_int
        E_a = E_a / paramsScale
        P_s = P_s
        tau_0 = tau_0
        I_0 = I_0
        V_t = V_t
        C_par = C_par * paramsScale
        alpha = alpha
        soft_E = soft_E / paramsScale
        I_dsc = I_dsc * paramsScale
        V_thr = V_thr
        dt = dt
        innerStep = innerStep
        paramsScale = paramsScale

        C_0 = _eps0 * eps_hzo / t_hzo * A
        C_tot = C_0 + C_par
        self.C_tot = C_tot

        cap_divider = eps_int / (t_hzo * eps_hzo + t_int * eps_int)
        self.cap_divider = cap_divider
        depol_divider = 1 / _eps0 * t_int / (t_hzo * eps_hzo + t_int * eps_int)
        self.depol_divider = depol_divider
        threshold = V_thr * C_tot + P_s * A
        self.threshold = threshold
        self.P_s = P_s

        @jax.custom_vjp
        def tau_fn(E):
            tau = tau_0 * jnp.exp((E_a / (jnp.abs(E) + 5e-6)) ** alpha)

            return tau

        def tau_fn_fwd(E):
            return tau_fn(E), (E,)

        def tau_fn_bw(res, g):
            (E,) = res
            exp_x = jnp.clip((E_a / (jnp.abs(E) + soft_E)) ** alpha, 0, 1)
            tau_prime = -(
                tau_0
                * E_a
                * alpha
                * E
                * jnp.exp(exp_x)
                * (E_a / (jnp.abs(E) + soft_E)) ** (alpha - 1)
            ) / (jnp.abs(E) * (E + soft_E) ** 2)

            tangents_out = (g * tau_prime,)
            return tangents_out

        tau_fn.defvjp(tau_fn_fwd, tau_fn_bw)

        def updatePol(v, p):

            def pol_step(state, input_):
                p, _ = state
                E = v * cap_divider - p * depol_divider

                tau = tau_0 * jnp.exp((E_a / (jnp.abs(E) + 5e-6)) ** alpha)

                I_p_new = (jnp.sign(E) * P_s - p) * A / tau
                dp = I_p_new / A
                p = jnp.clip(p + 1e-3 * dt * dp, -P_s, P_s)
                return (p, I_p_new), None

            def pol_step2(p):
                E = v * cap_divider - p * depol_divider

                tau = tau_fn(E)

                I_p_new = (jnp.sign(E) * P_s - p) * A / jax.lax.stop_gradient(tau)
                dp = I_p_new / A
                p = jnp.clip(p + dt * dp, -P_s, P_s)
                return p, I_p_new

            (p_inner, I_p_inner), _ = jax.lax.scan(
                pol_step, (p, jnp.zeros_like(p)), jnp.arange(1000)
            )
            # I_p_inner = I_p_inner / 1000

            p_outer, I_p_outer = pol_step2(p)

            p = p_outer + jax.lax.stop_gradient(p_inner - p_outer)
            I_p = I_p_outer + jax.lax.stop_gradient(I_p_inner - I_p_outer)

            return p, I_p

        self.updatePol = updatePol

        def step(state, input_):
            v, p = state

            # charge = v * C_tot + p * A
            spikes = spike_fn(v - V_thr)

            p_upper, I_p = updatePol(v, p)

            I_leak = (I_0 * A * jnp.expm1(v / V_t) + I_dsc) * jnp.sign(v)
            dv = (input_ - I_leak - I_p) / C_tot

            v_upper = jnp.clip(v + dt * dv, -5, 5)

            spikes_ref = jax.lax.stop_gradient(spikes)
            v_new = (1 - spikes_ref) * v_upper
            p_new = (1 - spikes_ref) * p_upper - (spikes_ref * P_s)

            return (v_new, p_new), (v_new, p_new, spikes)

        def step_full(state, input_):
            v, p = state

            # charge = v * C_tot + p * A
            spikes = spike_fn(v - V_thr)

            E = v * self.cap_divider - p * self.depol_divider

            tau = tau_0 * jnp.exp((E_a / (jnp.abs(E) + 5e-6)) ** alpha)

            I_p = (jnp.sign(E) * self.P_s - p) * A / tau
            dp = I_p / A

            I_leak = (I_0 * A * jnp.expm1(v / V_t) + I_dsc) * jnp.sign(v)
            dv = (input_ - I_leak - I_p) / C_tot

            v_upper = jnp.clip(v + dt * dv, -5, 5)
            p_upper = jnp.clip(p + dt * dp, -self.P_s, self.P_s)

            spikes_ref = jax.lax.stop_gradient(spikes)
            v_new = (1 - spikes_ref) * v_upper
            p_new = (1 - spikes_ref) * p_upper - (spikes_ref * P_s)

            return (v_new, p_new), (v_new, p_new, spikes)

        if stepFull:
            self.step = jax.jit(step_full)
        else:
            self.step = jax.jit(step)

    @eqx.filter_jit
    def getCharge(self, v, p):
        return v * self.C_tot + p * self.A

    @jax.named_scope("nn.FeLIF")
    def __call__(self, input_):
        v0 = jnp.zeros((self.out_size,))
        p0 = jnp.zeros((self.out_size,)) - self.P_s

        state = (v0, p0)
        _, out = jax.lax.scan(self.step, state, input_)
        v, p, s = out

        return s, self.getCharge(v, p), v, p


class CUBALIF(eqx.Module):
    hidden_size: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)

    def __init__(self, hidden_size, alpha, beta, *, key=None):
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.beta = beta

    def __call__(self, input_):
        hidden = (jnp.zeros((self.hidden_size,)), jnp.zeros((self.hidden_size,)))

        def f(carry, inp):
            syn, mem = carry

            mtr = mem - 1.0
            out = _spike_fn(mtr)
            rst = jax.lax.stop_gradient(out)

            new_syn = self.alpha * syn + inp
            new_mem = (self.beta * mem) * (1.0 - rst) + new_syn

            return (new_syn, new_mem), out

        _, out = jax.lax.scan(f, hidden, input_)

        return out


class RCUBALIF(eqx.Module):
    hidden_size: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)
    beta: float = eqx.field(static=True)
    recurrent: eqx.nn.Linear

    def __init__(self, hidden_size, alpha, beta, *, key=None):
        self.hidden_size = hidden_size
        self.alpha = alpha
        self.beta = beta
        self.recurrent = eqx.nn.Linear(hidden_size, hidden_size, key=key)

    def __call__(self, input_):
        hidden = (jnp.zeros((self.hidden_size,)), jnp.zeros((self.hidden_size,)))

        def f(carry, inp):
            syn, mem = carry

            mtr = mem - 1.0
            out = _spike_fn(mtr)
            rst = jax.lax.stop_gradient(out)

            new_syn = self.alpha * syn + inp + self.recurrent(out)
            new_mem = (self.beta * mem) * (1.0 - rst) + new_syn

            return (new_syn, new_mem), out

        _, out = jax.lax.scan(f, hidden, input_)

        return out
