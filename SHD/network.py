import equinox as eqx
import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import matplotlib.pyplot as plt
import optax
from spyx.axn import superspike
from spyx.loaders import SHD_loader

from new_pizzo_neuron_jax import FeLIF

_spike_fn = superspike(k=100)


class CUBAIF(eqx.Module):
    hidden_size: int = eqx.field(static=True)

    def __init__(self, hidden_size, alpha, beta, *, key=None):
        self.hidden_size = hidden_size

    def __call__(self, input_):
        hidden = (
            jnp.zeros((self.hidden_size,)),
            jnp.zeros((self.hidden_size,)),
            jnp.zeros((self.hidden_size,)),
        )

        def f(carry, inp):
            v, current, rest = carry

            s = _spike_fn(v - 1.0)

            rest = (rest + current) * (1 - jax.lax.stop_gradient(s))
            v = (v + 0.2 * (rest - v) + current) * (1 - jax.lax.stop_gradient(s))
            current = 0.1 * current + 0.9 * inp
            return (v, current, rest), (s, v)

        _, (out, v) = jax.lax.scan(f, hidden, input_)

        return out, v


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


class CUBALI(eqx.Module):
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
            flt, out = carry

            new_flt = self.alpha * flt + inp
            new_out = self.beta * out + new_flt

            return (new_flt, new_out), new_out

        _, out = jax.lax.scan(f, hidden, input_)

        return out


class Network(eqx.Module):
    layer1: eqx.Module
    linear1: eqx.nn.Linear

    layer2: CUBALI
    linear2: eqx.nn.Linear

    def __init__(self, in_size, hidden_size, out_size, alpha, beta, *, key):
        key1, key2 = jax.random.split(key, 2)
        # self.layer1 = CUBAIF(hidden_size, alpha, beta)
        self.layer1 = FeLIF(hidden_size, I_dsc=100e-12)
        self.linear1 = eqx.nn.Linear(in_size, hidden_size, use_bias=False, key=key1)

        self.layer2 = CUBALI(out_size, alpha, beta)
        self.linear2 = eqx.nn.Linear(hidden_size, out_size, key=key2)

    def __call__(self, input_):
        x1 = jax.vmap(self.linear1)(input_)
        s1, v1 = self.layer1(x1)

        x2 = jax.vmap(self.linear2)(s1)
        s2 = self.layer2(x2)

        return s2, s1, v1


batch_size = 128
nb_epochs = 200
nb_steps = 72
nb_inputs = 700
nb_hidden = 200
nb_outputs = 20
learning_rate = 2e-4
time_step = 14e-3

tau_mem = 1680e-3
tau_syn = 5e-3

alpha = 0  # jnp.exp(-time_step/tau_syn)
beta = jnp.exp(-time_step / tau_mem)


@jax.jit
def accuracy_fn(output, y):
    predicted_class = jnp.argmax(jnp.max(output, axis=1), axis=1)
    return jnp.mean(predicted_class == y)


@eqx.filter_value_and_grad
def compute_loss(diff_model, static_model, x, y):
    model = eqx.combine(diff_model, static_model)
    pred_y, spks, _ = jax.vmap(model)(x)

    # reg_loss1 = 2e-6 * jnp.sum(spks)  # L1 loss on total number of spikes
    # reg_loss2 = 2e-6 * jnp.mean(
    #     jnp.sum(jnp.sum(spks, axis=0), axis=0) ** 2
    # )  # L2 loss on spikes per neuron

    m = jnp.max(pred_y, axis=1)  # Sum over time
    # Here we combine supervised loss and the regularizer
    loss_val = optax.softmax_cross_entropy(
        m, jax.nn.one_hot(y, nb_outputs)
    ).mean()  # + reg_loss1 + reg_loss2

    return loss_val


@eqx.filter_value_and_grad
def compute_loss_combined(model, x, y):
    pred_y, spks, _ = jax.vmap(model)(x)

    reg_loss1 = 2e-6 * jnp.sum(spks)  # L1 loss on total number of spikes
    reg_loss2 = 2e-6 * jnp.mean(
        jnp.sum(spks, axis=1) ** 2
    )  # L2 loss on spikes per neuron

    # reg_loss1 = 2e-6*jnp.mean(jnp.abs(jnp.sum(spks,axis=1).ravel()-4.0))
    # reg_loss2 = 2e-6*jnp.mean((jnp.sum(spks,axis=1).ravel()-4.0)**2)

    m = jnp.max(pred_y, axis=1)  # Sum over time
    # Here we combine supervised loss and the regularizer
    loss_val = (
        optax.softmax_cross_entropy(m, jax.nn.one_hot(y, nb_outputs)).mean()
        + reg_loss1
        + reg_loss2
    )

    return loss_val


@eqx.filter_jit
def make_step(model, x, y, opt_state):
    # diff_model, static_model = eqx.partition(model, filter_spec)
    # loss, grads = compute_loss(diff_model, static_model, x, y)
    loss, grads = compute_loss_combined(model, x, y)

    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
    # model = eqx.combine(diff_model, static_model)
    return loss, model, opt_state


@eqx.filter_jit
def epoch_step(state, data):
    model, opt_state = state
    x, y = data

    x = jnp.unpackbits(x, axis=1)
    loss, model, opt_state = make_step(model, x, y, opt_state)

    return (model, opt_state), loss


@eqx.filter_jit
def test_step(model, data):
    x, y = data

    x = jnp.unpackbits(x, axis=1)
    pred_y, _, _ = jax.vmap(model)(x)

    return model, accuracy_fn(pred_y, y)


rngkey = jax.random.key(0)
shd_dl = SHD_loader(batch_size, nb_steps, nb_inputs, 0.2)
model = Network(nb_inputs, nb_hidden, nb_outputs, alpha, beta, key=rngkey)
# model = eqx.tree_deserialise_leaves("shd2.eqx", model)


def init_linear_weight(model, init_fn, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_weights = lambda m: [
        x.weight
        for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear)
        if is_linear(x)
    ]
    weights = get_weights(model)
    new_weights = [
        init_fn(subkey, weight.shape)
        for weight, subkey in zip(weights, jax.random.split(key, len(weights)))
    ]
    new_model = eqx.tree_at(get_weights, model, new_weights)
    return new_model


def init_linear_bias(model, init_fn, key):
    is_linear = lambda x: isinstance(x, eqx.nn.Linear)
    get_biases = lambda m: [
        x.bias for x in jax.tree_util.tree_leaves(m, is_leaf=is_linear) if is_linear(x)
    ]
    biases = get_biases(model)
    new_bias = [
        init_fn(subkey, bias.shape) if bias is not None else None
        for bias, subkey in zip(biases, jax.random.split(key, len(biases)))
    ]
    new_model = eqx.tree_at(get_biases, model, new_bias)
    return new_model


def init_LSUV(key, model, data_batch, tgt_mu=0.0, tgt_var=1.0):
    wkey, bkey = jax.random.split(key, 2)
    model = init_linear_bias(model, jax.nn.initializers.constant(0), bkey)
    model = init_linear_weight(model, jax.nn.initializers.orthogonal(), wkey)

    alldone = False
    while not alldone:
        _, spks, v1 = jax.vmap(model)(data_batch)
        v = jnp.var(v1[:, -1, :].ravel())
        m = jnp.mean(v1[:, -1, :].ravel())
        mus = jnp.mean(jnp.sum(spks, axis=1).ravel())
        print(
            "Layer: {0}, Variance: {1:.3}, Mean U: {2:.3}, Mean S: {3:.3}".format(
                0, v, m, mus
            )
        )

        alldone = True
        if jnp.abs(mus - tgt_var) > 0.1:
            model = eqx.tree_at(
                lambda m: m.linear1.weight,
                model,
                model.linear1.weight / jnp.sqrt(jnp.maximum(mus, 1e-3)),
            )
            model = eqx.tree_at(
                lambda m: m.linear1.weight,
                model,
                model.linear1.weight * jnp.sqrt(tgt_var),
            )
            done = False
        else:
            done = True
        alldone *= done

        if alldone:
            print("Initialization finalized:")
            print(
                "Layer: {0}, Variance: {1:.3}, Mean U: {2:.3}, Mean S: {3:.3}".format(
                    0, v, m, mus
                )
            )


x_train, y_train = shd_dl.train_epoch(jax.random.key(0))
init_LSUV(
    jax.random.key(0),
    model,
    jnp.unpackbits(x_train[0], axis=1),
    model.layer1.threshold / 2,
    4,
)

# Step 2
filter_spec = jtu.tree_map(lambda _: False, model)
filter_spec = eqx.tree_at(
    lambda tree: (tree.linear2.weight, tree.linear2.bias),
    filter_spec,
    replace=(True, True),
)
# diff_model, static_model = eqx.partition(model, filter_spec)

optim = optax.adamax(learning_rate, b1=0.9, b2=0.999)
# scheduler_fn = optax.sgdr_schedule([dict(init_value=learning_rate, decay_steps=50,
# peak_value=learning_rate, warmup_steps=1, end_value=1e-4,
# exponent=1.0) for _ in range(jnp.ceil(nb_epochs/50).astype(int))])
# optim = optax.inject_hyperparams(optax.adamax)(
#     learning_rate=scheduler_fn, b1=0.9, b2=0.999
# )
opt_state = optim.init(model)
# scheduler_state = scheduler_fn.init(init_params)
# scheduler_state

total_loss = []
total_accuracy = []

for epoch in range(nb_epochs):
    epoch_key = jax.random.fold_in(rngkey, nb_epochs)
    x_train, y_train = shd_dl.train_epoch(epoch_key)
    (model, opt_state), loss = jax.lax.scan(
        epoch_step, (model, opt_state), (x_train, y_train)
    )

    x_test, y_test = shd_dl.test_epoch()
    _, accuracy = jax.lax.scan(test_step, model, (x_test, y_test))

    total_loss.append(jnp.mean(loss).item())
    total_accuracy.append(jnp.mean(accuracy).item())
    print(
        "Epoch %i: loss=%.5f accuracy=%.5f"
        % (epoch + 1, jnp.mean(loss), jnp.mean(accuracy))
    )

    if epoch % 10 == 0:
        # eqx.tree_serialise_leaves("shd.eqx", model)
        plt.figure()
        plt.plot(total_loss)
        plt.savefig("loss2.png")

        plt.figure()
        plt.plot(total_accuracy)
        plt.savefig("accuracy2.png")

eqx.tree_serialise_leaves("shd_final.eqx", model)
plt.figure()
plt.plot(total_loss)
plt.savefig("loss2.png")

plt.figure()
plt.plot(total_accuracy)
plt.savefig("accuracy2.png")
