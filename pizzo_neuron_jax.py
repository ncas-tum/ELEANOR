from spyx.axn import superspike
import jax
import jax.numpy as jnp


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
    print(threshold)

    if spike_fn is None:
        spike_fn = superspike()

    def _derivative(V, P):
        E = V * cap_divider - P * depol_divider
        tau = jax.lax.stop_gradient(
            tau_0 * jnp.exp((E_a / (jnp.abs(E) + soft_E)) ** alpha)
        )
        tau = jnp.clip(tau, 1.1e-7, 1.1e7)

        I_leak = jax.lax.stop_gradient(
            (I_0 * A * jnp.expm1(V / V_t) + I_dsc) * jnp.sign(V)
        )
        I_p = (jnp.sign(E) * P_s - P) * A / tau

        I_p = jnp.clip(I_p, -100e6, 100e6)
        I_leak = jnp.clip(I_leak, -100, 100)

        dv = (1 / C_tot) * (-I_leak - I_p)
        dP = I_p / A
        return dv, dP

    def _inner_loop(state, input_):
        [V, P] = state
        dv, dP = _derivative(V, P)

        V_new = jnp.clip(V + dt / innerStep * dv, 0, 5)
        P_new = jnp.clip(P + dt / innerStep * dP, -0.22, 0.22)

        return [V_new, P_new], V_new

    def step(state, input_):
        V, P, charge = state

        last_state, _ = jax.lax.scan(_inner_loop, [V, P], None, innerStep, unroll=20)
        [V_inner, P_inner] = last_state

        V_inner = jnp.clip(V_inner + input_, 0, 5)
        P_inner = jnp.clip(P_inner, -P_s, P_s)

        charge_real = V_inner * C_tot + P_inner * A
        charge_surr = charge + C_tot * input_
        charge_new = jax.lax.stop_gradient(charge_real) + (
            charge_surr - jax.lax.stop_gradient(charge_surr)
        )
        # Charge equal to the real charge in forward pass
        # but surrogated as IF in backward pass

        spikes = spike_fn(charge_new - threshold)
        V = (1 - spikes) * jax.lax.stop_gradient(V_inner)
        P = (1 - spikes) * jax.lax.stop_gradient(P_inner) - spikes * P_s

        return (V, P, charge_new), (spikes, charge_new, V, P)

    def initial_state(nb_neurons):
        V0 = jnp.zeros((nb_neurons,))
        P0 = jnp.zeros((nb_neurons,)) - P_s
        C0 = V0 * C_tot + P0 * A

        return (V0, P0, C0)

    return jax.jit(step), initial_state


if __name__ == "__main__":
    import optax
    from tqdm import trange
    import matplotlib.pyplot as plt
    from spyx.axn import tanh

    plt.style.use("dark_background")

    input_ = jnp.array([[0.2], [0.0]] * 75 + [[0.0], [0.0]] * 75)
    target = jnp.array([2.0])
    felif_step, reset = FeLIF(P_s=0.22)

    @jax.jit
    def predict(params, input_):
        _, (spikes, charge, V, P) = jax.lax.scan(
            felif_step, reset(1), input_ * params[0], input_.shape[0], unroll=20
        )
        return spikes, charge, V, P

    @jax.jit
    def loss(params, input_, target):
        preds, _, _, _ = predict(params, input_)
        preds = jnp.sum(preds, axis=0)  # Sum over time
        return jnp.mean((preds - target) ** 2)

    params = [jnp.ones((1,))]
    opt = optax.adamax(learning_rate=0.01, b1=0.9, b2=0.995)
    opt_state = opt.init(params)

    nb_epochs = 100

    print("Optimizing")
    loss_rec = []
    state = [params, opt_state]
    for _ in trange(nb_epochs):
        grad_params, opt_state = state

        loss_val, grads = jax.value_and_grad(loss)(grad_params, input_, target)
        updates, opt_state = opt.update(grads, opt_state, grad_params)
        loss_rec.append(loss_val)

        state = [optax.apply_updates(grad_params, updates), opt_state]
    loss_rec = jnp.stack(loss_rec)
    print("Predict")
    _, preds_before, _, _ = predict(params, input_)
    _, preds, _, _ = predict(state[0], input_)
    print("Finish")

    plt.figure()
    plt.subplot(3, 1, 1)
    plt.title("Neuron charge")
    plt.plot(preds_before, label="Before optimizing")
    plt.plot(preds, label="After optimizing")
    plt.legend()
    plt.subplot(3, 1, 2)
    plt.title("Loss over epochs")
    plt.plot(loss_rec)
    plt.subplot(3, 1, 3)
    plt.plot(jax.vmap(jax.grad(tanh(k=0.1)), (0,))(preds[:, 0] - 6.932034598021211))
    plt.show()
