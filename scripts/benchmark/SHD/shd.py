from typing import Union, Literal, Callable, Optional

import jax
import optax
import equinox as eqx
import seaborn as sns
import jax.numpy as jnp
import matplotlib.pyplot as plt
from tqdm import trange
from spyx.axn import arctan
from jaxtyping import Array, PRNGKeyArray
from spyx.loaders import SHD_loader
from aqt.jax.v2.aqt_quantizer import Quantizer, quantizer_make

from scripts.utils import make_fake_quant
from scripts.pizzo_neuron_jax import FeLIF

sns.set()


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


class QLinear(eqx.nn.Linear):
    """Performs a quantized linear transformation."""

    quantizer: Quantizer = eqx.field(static=True)
    fake_quant: Callable = eqx.field(static=True)

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        dtype=None,
        n_bits=3,
        *,
        key: PRNGKeyArray,
    ):
        super(QLinear, self).__init__(
            in_features, out_features, use_bias, dtype, key=key
        )

        self.quantizer = quantizer_make(n_bits=n_bits)
        self.fake_quant = make_fake_quant(self.quantizer, calibration_axes=(0, 1))

    @jax.named_scope("QLinear")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        qweights = self.fake_quant(self.weight)

        if self.in_features == "scalar":
            if jnp.shape(x) != ():
                raise ValueError("x must have scalar shape")
            x = jnp.broadcast_to(x, (1,))
        x = qweights @ x
        if self.bias is not None:
            x = x + self.bias
        if self.out_features == "scalar":
            assert jnp.shape(x) == (1,)
            x = jnp.squeeze(x)
        return x


class Network(eqx.Module):
    layer1: RCUBALIF
    linear1: eqx.nn.Linear

    layer2: FeLIF
    linear2: eqx.nn.Linear

    def __init__(self, in_size, hidden_size, out_size, alpha, beta, *, key):
        key1, key2, key3 = jax.random.split(key, 3)
        self.layer1 = RCUBALIF(hidden_size, alpha, beta, key=key3)
        self.linear1 = eqx.nn.Linear(in_size, hidden_size, use_bias=False, key=key1)

        self.layer2 = FeLIF(out_size, spike_fn=_spike_fn, dt=1e-3, stepFull=False)
        self.linear2 = QLinear(hidden_size, out_size, key=key2)

        # get_weights = lambda m: m.weight
        # new_weights = linear2.weight*1000
        # self.linear2 = eqx.tree_at(get_weights, linear2, new_weights)

    def __call__(self, input_):
        x1 = jax.vmap(self.linear1)(input_)
        s1 = self.layer1(x1)

        x2 = jax.vmap(self.linear2)(s1 * 1000)
        s2, charge, v2, _ = self.layer2(x2)

        return s2, (s1, charge)


@jax.jit
def accuracy_fn(output, y):
    predicted_class = jnp.argmax(jnp.sum(output, axis=1), axis=1)
    return jnp.mean(predicted_class == y)


@eqx.filter_value_and_grad
def compute_loss(model, x, y):
    pred_y, (spks, _) = jax.vmap(model)(x)

    reg_loss1 = 2e-6 * jnp.sum(spks)  # L1 loss on total number of spikes
    reg_loss2 = 2e-6 * jnp.mean(
        jnp.sum(spks, axis=1) ** 2
    )  # L2 loss on spikes per neuron

    reg_loss3 = 2e-4 * jnp.mean(jax.nn.relu(20 - jnp.sum(pred_y, axis=(1, 2))) ** 2)

    m = jnp.sum(pred_y, axis=1)  # Sum over time
    # Here we combine supervised loss and the regularizer
    loss_val = (
        optax.softmax_cross_entropy(m, jax.nn.one_hot(y, nb_outputs)).mean()
        + reg_loss1
        + reg_loss2
        + reg_loss3
    )

    return loss_val


@eqx.filter_jit
def make_step(model, x, y, opt_state):
    loss, grads = compute_loss(model, x, y)

    updates, opt_state = optim.update(grads, opt_state)
    model = eqx.apply_updates(model, updates)
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
    pred_y, (_, _) = jax.vmap(model)(x)

    return model, accuracy_fn(pred_y, y)


# @title Parameters
batch_size = 128  # @param {type:"number"}
nb_epochs = 200  # @param {type:"number"}
nb_steps = 72  # @param {type:"number"}
nb_inputs = 700  # @param {type:"number"}
nb_hidden = 200  # @param {type:"number"}
nb_outputs = 20  # @param {type:"number"}
save_every = 10  # @param {type:"number"}
learning_rate = 2e-4  # @param {type:"number"}
time_step = 14e-3  # @param {type:"number"}

tau_mem = 1680e-3  # @param {type:"number"}
tau_syn = 5e-3  # @param {type:"number"}

alpha = jnp.exp(-time_step / tau_syn)
beta = jnp.exp(-time_step / tau_mem)
shd_dl = SHD_loader(batch_size, nb_steps, nb_inputs, 0.2)

_spike_fn = arctan(k=2)

rngkey = jax.random.key(0)
model = Network(nb_inputs, nb_hidden, nb_outputs, alpha, beta, key=rngkey)
optim = optax.adamax(learning_rate, b1=0.9, b2=0.999)
opt_state = optim.init(model)

total_loss = []
total_accuracy = []
pbar = trange(nb_epochs)
for epoch in pbar:
    epoch_key = jax.random.fold_in(rngkey, nb_epochs)
    x_train, y_train = shd_dl.train_epoch(epoch_key)
    (model, opt_state), loss = jax.lax.scan(
        epoch_step, (model, opt_state), (x_train, y_train)
    )

    x_test, y_test = shd_dl.test_epoch()
    _, accuracy = jax.lax.scan(test_step, model, (x_test, y_test))

    total_loss.append(jnp.mean(loss).item())
    total_accuracy.append(jnp.mean(accuracy).item())

    metrics = {"loss": jnp.mean(loss).item(), "accuracy": jnp.mean(accuracy).item()}
    pbar.set_postfix(metrics)

    if epoch % save_every == 0:
        jnp.save("accuracy.npy", total_accuracy)
        jnp.save("loss.npy", total_loss)

        plt.figure()
        plt.plot(total_loss)
        plt.savefig("loss_polloop.png")

        plt.figure()
        plt.plot(total_accuracy)
        plt.savefig("accuracy_polloop.png")

        plt.close("all")

plt.figure(figsize=(12, 5))
plt.plot(total_loss)
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.savefig("loss_polloop.png")

plt.figure(figsize=(12, 5))
plt.plot(total_accuracy)
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.savefig("accuracy_polloop.png")
