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
        paramsScale=1e12,  # Scale parameters to avoid underflow
        spike_fn=_spike_fn,
        *,
        key=None
    ):
        _eps0 = 8.85418792394420013968e-12 * paramsScale
        self.out_size = out_size

        # Parameters
        A = A * paramsScale
        t_hzo = t_hzo * paramsScale
        t_int = t_int * paramsScale
        E_a = E_a / paramsScale
        C_par = C_par * paramsScale
        soft_E = soft_E / paramsScale
        I_dsc = I_dsc * paramsScale

        C_0 = _eps0 * eps_hzo / t_hzo * A
        C_tot = C_0 + C_par

        cap_divider = eps_int / (t_hzo * eps_int + t_int * eps_hzo)
        depol_divider = 1 / _eps0 * t_int / (t_hzo * eps_int + t_int * eps_hzo)

        # Save parameters on object
        self.A = A
        self.P_s = P_s
        self.C_tot = C_tot

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

            def pol_step(state, xs):
                p, _ = state
                E = v * cap_divider - p * depol_divider

                tau = tau_fn(E)

                I_p_new = (jnp.sign(E) * P_s - p) * A / jax.lax.stop_gradient(tau)
                dp = I_p_new / A
                p = jnp.clip(p + 1e-3 * dt * dp, -P_s, P_s)
                return (p, I_p_new), None

            init_state = (p, jnp.zeros_like(p))
            (p_inner, I_p_inner), _ = jax.lax.scan(
                pol_step, init_state, xs=None, length=1000
            )

            (p_outer, I_p_outer), _ = pol_step(init_state, None)

            p = p_outer + jax.lax.stop_gradient(p_inner - p_outer)
            I_p = I_p_outer + jax.lax.stop_gradient(I_p_inner - I_p_outer)

            return p, I_p

        def step(state, input_):
            v, p = state

            spikes = spike_fn(v - V_thr)

            # Update polarization
            p_upper, I_p = updatePol(v, p)

            # Update voltage
            I_leak = (I_0 * A * jnp.expm1(v / V_t) + I_dsc) * jnp.sign(v)
            dv = (input_ - I_leak - jax.lax.stop_gradient(I_p)) / C_tot
            v_upper = jnp.clip(v + dt * dv, 0, 5)

            # Reset when spike
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

        return s, self.getCharge(v, p), v, p


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import optax
    from tqdm import trange

    plt.style.use("dark_background")

    # Input every 5ms
    input_ = 500 * jnp.array([[1.0], [0], [0], [0], [0], [0], [0], [0], [0], [0]] * 10)
    target = jnp.array([2.0])

    class Network(eqx.Module):
        linear: eqx.nn.Linear
        felif: FeLIF

        def __init__(self, *, key):
            linear = eqx.nn.Linear(1, 1, use_bias=False, key=key)
            self.felif = FeLIF(
                1,
                dt=1e-3,
                A=25e-12,
                I_dsc=3.2532312693054174e-11,
                V_thr=1.698682296181096,
                P_s=0.13321217250476625,
                spike_fn=arctan(),
                paramsScale=1e12,
            )
            new_weight = jnp.asarray([[1.0]])
            self.linear = eqx.tree_at(lambda x: x.weight, linear, new_weight)

        @jax.named_scope("nn.Network")
        def __call__(self, input_):
            x1 = jax.vmap(self.linear)(input_)
            s, c, v, p = self.felif(x1)

            return s, c, v, p

    @eqx.filter_jit
    def predict(model, input_):
        spikes, charge, V, P = model(input_)

        return spikes, charge, V, P

    @eqx.filter_jit
    def loss(model, input_, target):
        preds, _, _, _ = predict(model, input_)
        preds = jnp.sum(preds, axis=0)  # Sum over time
        return jnp.mean((preds - target) ** 2)

    model = Network(key=jax.random.key(0))
    opt = optax.sgd(learning_rate=1e-3)
    opt_state = opt.init(model)

    nb_epochs = 10

    print("Optimizing")
    loss_rec = []
    param_rec = []
    state = [model, opt_state]
    for _ in trange(nb_epochs):
        grad_params, opt_state = state

        loss_val, grads = eqx.filter_value_and_grad(loss)(grad_params, input_, target)
        updates, opt_state = opt.update(grads, opt_state, grad_params)
        loss_rec.append(loss_val)

        new_param = eqx.apply_updates(grad_params, updates)
        state = [new_param, opt_state]
        param_rec.append(new_param.linear.weight[0, 0])
    loss_rec = jnp.stack(loss_rec)
    param_rec = jnp.stack(param_rec)

    print("Predict")
    _, preds_before, _, _ = predict(model, input_)
    _, preds, V, P = predict(state[0], input_)
    print("Finish")

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title("Neuron charge")
    plt.plot(preds_before, label="Before optimizing")
    plt.plot(preds, label="After optimizing")
    plt.legend()

    ax1 = plt.subplot(3, 1, 2)
    plt.title("Loss over epochs")

    color = "#8dd3c7"
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss", color=color)
    ax1.plot(loss_rec, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()

    color = "#feffb3"
    ax2.set_ylabel("Weight", color=color)
    ax2.plot(param_rec, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    ax1 = plt.subplot(3, 1, 3)

    color = "#8dd3c7"
    ax1.set_xlabel("time (s)")
    ax1.set_ylabel("voltage", color=color)
    ax1.plot(V, color=color)
    ax1.tick_params(axis="y", labelcolor=color)

    ax2 = ax1.twinx()

    color = "#feffb3"
    ax2.set_ylabel("Polarization", color=color)
    ax2.plot(P, color=color)
    ax2.tick_params(axis="y", labelcolor=color)

    plt.show()
