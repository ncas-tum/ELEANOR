import jax
import jax.numpy as jnp
from spyx.axn import superspike


def FeLIF(
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
    spike_fn=None,
):
    A = A * paramsScale  # device area
    t_hzo = t_hzo * paramsScale  # thikness ferroelectric
    t_int = t_int * paramsScale  # thikness interlayer
    E_a = E_a / paramsScale  # coercitive field
    C_par = C_par * paramsScale  # parasitic capacitance form the circuit
    soft_E = (
        soft_E / paramsScale
    )  # soft boudary for the electric field, avoid tau to diverge
    I_dsc = I_dsc * paramsScale  # discharge current, set the "dendritic time constant"

    _eps0 = 8.85418792394420013968e-12 * paramsScale

    C_0 = _eps0 * eps_hzo / t_hzo * A
    C_tot = C_0 + C_par
    cap_divider = eps_int / (t_hzo * eps_hzo + t_int * eps_int)
    depol_divider = 1 / _eps0 * t_int / (t_hzo * eps_hzo + t_int * eps_int)
    threshold = V_thr * C_tot + P_s * A

    if spike_fn is None:
        spike_fn = superspike()

    def _derivative(V, P, input_):
        E = V * cap_divider - P * depol_divider
        tau = jax.lax.stop_gradient(
            tau_0 * jnp.exp((E_a / (jnp.abs(E) + soft_E)) ** alpha)
        )
        tau = jnp.clip(tau, 1.1e-7, 1.1e7)

        I_leak = jax.lax.stop_gradient(
            (I_0 * A * jnp.expm1(V / V_t) + I_dsc) * jnp.sign(V)
        )
        I_p = (jnp.sign(E) * P_s - P) * A / tau

        # Limit the currents to 100uA
        I_p = jnp.clip(I_p, -100e-6 * paramsScale, 100e-6 * paramsScale)
        I_leak = jnp.clip(I_leak, -100e-6 * paramsScale, 100e-6 * paramsScale)

        dv = (1 / C_tot) * (input_ - I_leak - I_p)
        dP = I_p / A

        return dv, dP

    def _inner_loop(state, input_):
        [V, P, spikes] = state
        dv, dP = _derivative(V, P, input_)

        V_new = V + (1 - spikes) * dt / innerStep * dv
        P_new = P + (1 - spikes) * dt / innerStep * dP

        # Limit voltage and polarization
        V_new = jnp.clip(V_new, 0, 5)
        P_new = jnp.clip(P_new, -P_s, P_s)

        charge = V_new * C_tot + P_new * A
        spikes = spike_fn(charge - threshold)

        return [V_new, P_new, spikes], None

    def step(state, input_):
        V, P, charge = state

        # Inner loop with 10us pulse input current
        last_state, _ = jax.lax.scan(
            _inner_loop,
            [V, P, jnp.zeros_like(charge)],
            jnp.stack(
                [input_ * 1e5 * C_tot] * 10  # Convert the input into current
                + [jnp.zeros_like(input_)] * (innerStep - 10)  # 10 us current
            ),
            innerStep,
            unroll=20,
        )
        [V_inner, P_inner, _] = last_state
        dv, dP = _derivative(V, P, jnp.zeros_like(input_))
        V_upper = V + dt * dv + input_
        P_upper = P + dt * dP

        # Limit voltage and polarization
        V_upper = jnp.clip(V_upper, 0, 5)
        P_upper = jnp.clip(P_upper, -P_s, P_s)

        V_new = (
            jax.lax.stop_gradient(V_inner) + V_upper - jax.lax.stop_gradient(V_upper)
        )
        P_new = (
            jax.lax.stop_gradient(P_inner) + P_upper - jax.lax.stop_gradient(P_upper)
        )

        charge_new = V_new * C_tot + P_new * A

        spikes = spike_fn(charge_new - threshold)
        V = (1 - spikes) * V_new
        P = (1 - spikes) * P_new - (spikes * P_s)

        return (V, P, charge_new), (spikes, charge_new, V, P)

    def initial_state(nb_neurons):
        V0 = jnp.zeros((nb_neurons,))
        P0 = jnp.zeros((nb_neurons,)) - P_s
        C0 = V0 * C_tot + P0 * A

        return (V0, P0, C0)

    return jax.jit(step), initial_state


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import optax
    from tqdm import trange

    plt.style.use("dark_background")

    # Input every 5ms
    input_ = jnp.array([1.0, 0, 0, 0, 0] * 20)
    target = jnp.array([5.0])
    felif_step, felif_reset = FeLIF(
        dt=1e-3,
        innerStep=1000,
        A=25e-12,
        I_dsc=3.2532312693054174e-11,
        V_thr=1.698682296181096,
        P_s=0.13321217250476625,
        spike_fn=superspike(),
        paramsScale=1e12,
    )

    @jax.jit
    def predict(params, input_):
        _, (spikes, charge, V, P) = jax.lax.scan(
            felif_step, felif_reset(1), input_ * params[0], input_.shape[0], unroll=20
        )
        return spikes, charge, V, P

    @jax.jit
    def loss(params, input_, target):
        preds, _, _, _ = predict(params, input_)
        preds = jnp.sum(preds, axis=0)  # Sum over time
        return jnp.mean((preds - target) ** 2)

    params = [jnp.ones((1,)) * 0.3]
    opt = optax.sgd(learning_rate=1e-2)
    opt_state = opt.init(params)

    nb_epochs = 500

    print("Optimizing")
    loss_rec = []
    param_rec = []
    state = [params, opt_state]
    for _ in trange(nb_epochs):
        grad_params, opt_state = state

        loss_val, grads = jax.value_and_grad(loss)(grad_params, input_, target)
        updates, opt_state = opt.update(grads, opt_state, grad_params)
        loss_rec.append(loss_val)

        new_param = optax.apply_updates(grad_params, updates)
        state = [new_param, opt_state]
        param_rec.append(new_param[0])
    loss_rec = jnp.stack(loss_rec)
    param_rec = jnp.stack(param_rec)
    print("Predict")
    _, preds_before, _, _ = predict(params, input_)
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
