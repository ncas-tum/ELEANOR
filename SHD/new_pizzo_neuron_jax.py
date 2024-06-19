from typing import Any, Callable

import equinox as eqx
import jax
import jax.numpy as jnp
from spyx.axn import custom

# _spike_fn = arctan(k=2)
_spike_fn = custom(lambda x: 1)


class FeLIF(eqx.Module):
    out_size: int = eqx.field(static=True)
    P_s: float = eqx.field(static=True)  # max polarisation
    A: float = eqx.field(static=True)
    C_tot: float = eqx.field(static=True)
    threshold: float = eqx.field(static=True)
    step: Callable[[Any, Any], Any] = eqx.field(static=True)

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
        *,
        key=None
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
        depol_divider = 1 / _eps0 * t_int / (t_hzo * eps_hzo + t_int * eps_int)
        threshold = V_thr * C_tot + P_s * A
        self.threshold = threshold
        self.P_s = P_s

        def _derivative(v, p):
            E = v * cap_divider - p * depol_divider
            tau = tau_0 * jnp.exp((E_a / (jnp.abs(E) + soft_E)) ** alpha)
            tau = jnp.clip(tau, 1.1e-7, 1.1e7)

            I_leak = (I_0 * A * jnp.expm1(v / V_t) + I_dsc) * jnp.sign(v)
            I_p = (jnp.sign(E) * P_s - p) * A / tau

            # Limit the currents to 100uA
            I_p = jnp.clip(I_p, -100e-6 * paramsScale, 100e-6 * paramsScale)
            I_leak = jnp.clip(I_leak, -100e-6 * paramsScale, 100e-6 * paramsScale)

            dv = (1 / C_tot) * (-I_leak - I_p)
            dp = I_p / A

            return (dv, dp)

        def inner_loop(state, input_):
            (v, p, s) = state

            (dv, dp) = _derivative(v, p)

            v_new = v + (1 - s) * dt / innerStep * dv
            p_new = p + (1 - s) * dt / innerStep * dp

            v_new = jnp.clip(v_new, 0, 5)
            p_new = jnp.clip(p_new, -P_s, P_s)
            charge = v_new * C_tot + p_new * self.A

            s_new = spike_fn(charge - threshold)

            return (v_new, p_new, s_new), None

        def f(v, p):
            last_state, _ = jax.lax.scan(
                inner_loop,
                (v, p, jnp.zeros_like(v)),
                None,
                innerStep,
                unroll=1,
            )
            (v_inner, p_inner, _) = last_state

            return v_inner, p_inner

        fvjp = jax.custom_vjp(f)

        def fvjp_fwd(v, p):
            return fvjp(v, p), (v, p)

        def fvjp_bwd(res, g):
            v, p = res

            E = v * cap_divider - p * depol_divider

            exp_x = jnp.clip((E_a / (jnp.abs(E) + soft_E)) ** alpha, 0, 60)
            tau_prime = -(
                tau_0
                * E_a
                * alpha
                * E
                * jnp.exp(exp_x)
                * (E_a / (jnp.abs(E) + soft_E)) ** (alpha - 1)
            ) / (jnp.abs(E) * (E + soft_E) ** 2)

            Ipa = (jnp.sign(E) * P_s - p) * A
            Ipb = tau_0 * jnp.exp(exp_x)

            dIp_dtau = -Ipa / Ipb**2
            dIp_dipa = 1 / Ipb

            dE_dv = cap_divider
            dE_dp = -depol_divider
            dtau_dE = tau_prime
            dIpa_dp = -A

            dIp_dv = dIp_dtau * dtau_dE * dE_dv
            dIp_dp = dIp_dipa * dIpa_dp + dIp_dtau * dtau_dE * dE_dp

            dIleak_dv = I_0 * A * jnp.exp(v / V_t) / V_t

            ddv_dIleak = -1 / C_tot
            ddv_dIp = -1 / C_tot

            ddv_dv = ddv_dIleak * dIleak_dv + ddv_dIp * dIp_dv
            ddv_dp = ddv_dIp * dIp_dp

            ddp_dIp = 1 / A
            ddp_dv = ddp_dIp * dIp_dv
            ddp_dp = ddp_dIp * dIp_dp

            dv1_ddv = dt
            dp1_ddp = dt

            dv1_dv = 1 + dv1_ddv * ddv_dv
            dv1_dp = dv1_ddv * ddv_dp

            dp1_dp = 1 + dp1_ddp * ddp_dp
            dp1_dv = dp1_ddp * ddp_dv

            tangents_out = (
                g[0] * dv1_dv * 1e-2 + g[1] * dp1_dv * 1e-2,
                g[0] * dv1_dp * 1e-2 + g[1] * dp1_dp * 1e-2,
            )
            return tangents_out

        fvjp.defvjp(fvjp_fwd, fvjp_bwd)

        def step(state, input_):
            v, p = state

            charge = v * C_tot + p * A
            spikes = spike_fn(charge - threshold)

            v_upper, p_upper = fvjp(jnp.clip(v + input_, -5, 5), p)

            spikes_ref = jax.lax.stop_gradient(spikes)
            v_new = (1 - spikes_ref) * v_upper
            p_new = (1 - spikes_ref) * p_upper - (spikes_ref * P_s)

            return (v_new, p_new), (v_new, p_new, spikes)

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

        return s, self.getCharge(v, p)  # (s, self.getCharge(v,p), v, p)


class LI(eqx.Module):
    hidden_size: int = eqx.field(static=True)
    alpha: float = eqx.field(static=True)

    def __init__(self, hidden_size, alpha, *, key):
        self.hidden_size = hidden_size
        self.alpha = alpha

    def __call__(self, input_):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            v = carry * self.alpha + inp
            return v, v

        _, out = jax.lax.scan(f, hidden, input_)

        return out


class IF(eqx.Module):
    hidden_size: int = eqx.field(static=True)

    def __init__(self, hidden_size, *, key):
        self.hidden_size = hidden_size

    def __call__(self, input_):
        hidden = jnp.zeros((self.hidden_size,))

        def f(carry, inp):
            s = _spike_fn(carry - 1.0)
            v = carry + inp - jax.lax.stop_gradient(s)
            return v, s

        _, out = jax.lax.scan(f, hidden, input_)

        return out


if __name__ == "__main__":

    class Network(eqx.Module):
        felif: FeLIF
        li: LI
        If: IF
        linear1: eqx.nn.Linear
        linear2: eqx.nn.Linear

        def __init__(self, in_size, hid_size, out_size, *, key):
            key1, key2 = jax.random.split(key, 2)
            self.linear1 = eqx.nn.Linear(in_size, hid_size, use_bias=False, key=key1)
            self.felif = FeLIF(hid_size, spike_fn=_spike_fn, key=None)
            self.If = IF(hid_size, key=None)

            self.linear2 = eqx.nn.Linear(hid_size, out_size, use_bias=False, key=key2)
            self.li = LI(out_size, 0.9, key=None)

        def __call__(self, input_, *, sigma=0.5):
            x = eqx.filter_vmap(self.linear1)(input_)
            s, _, _, _ = self.felif(x)
            # s = self.If(x)

            x = eqx.filter_vmap(self.linear2)(s)
            ouput = self.li(x)
            return ouput

    @jax.jit
    @jax.grad
    def loss_fn(model, x, y):
        pred_y = jax.vmap(model)(x)
        return jax.numpy.mean((y - pred_y) ** 2)

    batch_size, in_size, out_size = 32, 2, 3
    model = Network(in_size, 4, out_size, key=jax.random.key(0))
    x = jax.numpy.ones((batch_size, 128, in_size)) * 0.1
    y = jax.numpy.ones((batch_size, 128, out_size)) * 0.5
    grads = loss_fn(model, x, y)
    print(grads.linear1.weight)
