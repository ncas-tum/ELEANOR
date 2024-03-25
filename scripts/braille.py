# %%
import argparse
import os
import pickle
import sys
from collections import namedtuple

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nni
import numpy as np
import optax
from jax import vmap
from jax.lib import xla_bridge
from jax_tqdm import scan_tqdm
from matplotlib.gridspec import GridSpec
from sklearn import metrics as skmetrics
from spyx.axn import superspike

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pizzo_neuron_jax import FeLIF  # noqa: E402

plt.style.use("dark_background")
print(xla_bridge.get_backend().platform)

parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--nb_epochs", type=int, default=200)
parser.add_argument("--nb_hidden", type=int, default=200)
parser.add_argument("--nb_repetitions", type=int, default=50)
parser.add_argument("--nb_upsample", type=int, default=2)
parser.add_argument("--enc_fan_out", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--tau_mem", type=float, default=20e-3)
parser.add_argument("--tau_syn", type=float, default=10e-3)
parser.add_argument("--k", type=float, default=1.0)
parser.add_argument("--A", type=float, default=25e-12)
parser.add_argument("--I_dsc", type=float, default=10e-12)
parser.add_argument("--V_thr", type=float, default=2.5)
parser.add_argument("--P_s", type=float, default=22e-2)
parser.add_argument("--nni", action="store_true")

args = parser.parse_args()

# if args.nni:
nni_params = nni.get_next_parameter()
d = vars(args)  # copy by reference (checked below)
for key, val in nni_params.items():
    d[key] = val
    assert args.__dict__[key] == d[key]

print(args)

file_name = "/home/p306945/data/braille/braille_spiking_data.pkl"
with open(file_name, "rb") as infile:
    data_dict = pickle.load(infile)

letter_written = [
    "Space",
    "A",
    "B",
    "C",
    "D",
    "E",
    "F",
    "G",
    "H",
    "I",
    "J",
    "K",
    "L",
    "M",
    "N",
    "O",
    "P",
    "Q",
    "R",
    "S",
    "T",
    "U",
    "V",
    "W",
    "X",
    "Y",
    "Z",
]

# Extract data
data = []
labels = []
for i, _ in enumerate(letter_written):
    for repetition in np.arange(args.nb_repetitions):
        idx = i * args.nb_repetitions + repetition
        dat = 1.0 - data_dict[idx]["taxel_data"][:] / 255
        data.append(dat)
        labels.append(i)

# Crop to same length
data_steps = l = np.min([len(d) for d in data])
# data = torch.tensor(np.array([ d[:l] for d in data ]), dtype=torch.float)
# labels = torch.tensor(labels,dtype=torch.long)
data = jnp.array([d[:l] for d in data])
labels = jnp.array(labels)

# Select nonzero inputs
nzid = [1, 2, 6, 10]
data = data[:, :, nzid]

# Standardize data
rshp = data.reshape((-1, data.shape[2]))
data = (data - rshp.mean(0)) / (rshp.std(0) + 1e-3)


# Upsample
def upsample(data, n=2):
    shp = data.shape
    tmp = jnp.tile(data, (1, 1, 1, n))
    return tmp.reshape((shp[0], n * shp[1], shp[2]))


data = upsample(data, n=args.nb_upsample)

data = jax.random.permutation(jax.random.PRNGKey(0), data, axis=0)
labels = jax.random.permutation(jax.random.PRNGKey(0), labels, axis=0)

a = int(0.8 * len(labels))
x_train, x_test = data[:a], data[a:]
y_train, y_test = labels[:a], labels[a:]

# y_test = jnp.concatenate([y_test[1].reshape((1,)),
#                           y_test[0].reshape((1,)),
#                           y_test[2:]])

nb_channels = len(nzid)

# Network parameters
nb_inputs = nb_channels * args.enc_fan_out
nb_hidden = args.nb_hidden
nb_outputs = len(np.unique(labels)) + 1
time_step = (
    2e-3 / args.nb_upsample
)  # TODO needs to be updated to reflect the correct time scale
nb_steps = (
    args.nb_upsample * data_steps
)  # TODO We should change this and upsample the input data

print("Number of training data %i" % len(x_train))
print("Number of testing data %i" % len(x_test))
print("Number of outputs %i" % nb_outputs)
print("Number of timesteps %i" % nb_steps)

State = namedtuple("State", "obs labels")


def shuffle(dataset, shuffle_rng):
    x, y = dataset

    cutoff = y.shape[0] % args.batch_size

    obs = jax.random.permutation(shuffle_rng, x, axis=0)[:-cutoff]
    labels = jax.random.permutation(shuffle_rng, y, axis=0)[:-cutoff]

    obs = jnp.reshape(obs, (-1, args.batch_size) + obs.shape[1:])
    labels = jnp.reshape(
        labels, (-1, args.batch_size)
    )  # should make batch size a global

    return State(obs=obs, labels=labels)


alpha = float(np.exp(-time_step / args.tau_syn))
beta = float(np.exp(-time_step / args.tau_mem))

encoder_weight_scale = 1.0
fwd_weight_scale = 3.0
rec_weight_scale = 1e-2 * fwd_weight_scale

# Parameters

# Encoder
key = jax.random.key(0)
key, subkey1, subkey2, subkey3, subkey4, subkey5 = jax.random.split(key, 6)
enc_gain = jax.random.normal(subkey1, shape=(nb_inputs,)) * encoder_weight_scale
enc_bias = jax.random.normal(subkey1, shape=(nb_inputs,))

# Spiking network
w1 = (
    jax.random.normal(subkey3, shape=(nb_inputs, nb_hidden))
    * fwd_weight_scale
    / np.sqrt(nb_inputs)
)
w2 = (
    jax.random.normal(subkey4, shape=(nb_hidden, nb_outputs))
    * fwd_weight_scale
    / np.sqrt(nb_hidden)
)
# v1 = (
#     jax.random.normal(subkey5, shape=(nb_hidden, nb_hidden))
#     * rec_weight_scale
#     / np.sqrt(nb_hidden)
# )

print("init done")


def plot_voltage_traces(mem, spk=None, dim=(3, 5), spike_height=5, **kwargs):
    gs = GridSpec(*dim)
    if spk is not None:
        dat = 1.0 * mem
        dat[spk > 0.0] = spike_height
    else:
        dat = mem
    for i in range(np.prod(dim)):
        if i == 0:
            a0 = ax = plt.subplot(gs[i])
        else:
            ax = plt.subplot(gs[i], sharey=a0)
        ax.plot(dat[i], **kwargs)
        ax.axis("off")


def plot_spk_charge(spk_in, charge, V, P, spk_out, title):
    # Generate Plots
    fig, ax = plt.subplots(
        5,
        1,
        figsize=(12, 10),
        sharex=True,
        gridspec_kw={"height_ratios": [0.4, 1, 1, 1, 0.4]},
    )

    # Plot input current
    ax[0].scatter(*jnp.where(spk_in[0]), s=4, c="white", marker="|")
    ax[0].set_ylabel("Input Spikes")
    ax[0].set_yticks([])

    # Plot membrane potential
    ax[1].plot(jnp.cumsum(charge[0], axis=0))
    ax[1].set_ylabel("Internal charge")
    ax[1].set_xlabel("Time step")

    # Plot membrane potential
    ax[2].plot(V[0])
    ax[2].set_ylabel("Voltage")
    ax[2].set_xlabel("Time step")

    # Plot membrane potential
    ax[3].plot(P[0])
    ax[3].set_ylabel("Polarization")
    ax[3].set_xlabel("Time step")

    ax[4].scatter(*jnp.where(spk_out[0]), s=4, c="white", marker="|")
    ax[4].set_ylabel("Output Spikes")
    ax[4].set_yticks([])

    plt.suptitle(title)

    plt.savefig("braille.png")


spike_fn = superspike(k=25)


@jax.jit
def encoder_step(state, input_):
    enc, input_spk = state

    # Compute encoder activity
    new_enc = (beta * enc + (1.0 - beta) * input_) * (
        1.0 - jax.lax.stop_gradient(input_spk)
    )
    new_input_spk = spike_fn(enc - 1.0)

    return (new_enc, new_input_spk), new_input_spk


@jax.jit
def hidden_step(state, input_):
    syn, mem, out = state

    # Compute hidden layer activity
    # h1 = jnp.dot(input_, w1) #+ jnp.dot(out, v1)
    mthr = mem - 1.0
    new_out = spike_fn(mthr)
    rst = jax.lax.stop_gradient(new_out)  # We do not want to backprop through the reset

    new_syn = alpha * syn + input_
    new_mem = (beta * mem + (1.0 - beta) * syn) * (1.0 - rst)

    return (new_syn, new_mem, new_out), new_out


@jax.jit
def output_step(state, input_):
    syn, mem = state

    new_syn = alpha * syn + input_
    new_mem = beta * mem + (1.0 - beta) * syn

    return (new_syn, new_mem), new_mem


@jax.jit
def predict(params, input_):
    encoder_currents = params[0]["gain"] * (
        jnp.tile(input_, (1, args.enc_fan_out)) + params[0]["bias"]
    )

    enc = jnp.zeros((nb_inputs,))
    input_spk = jnp.zeros((nb_inputs,))
    _, enc_spk = jax.lax.scan(
        encoder_step, (enc, input_spk), encoder_currents, nb_steps, unroll=1
    )

    h1 = jnp.dot(enc_spk, params[1])
    syn = jnp.zeros((nb_hidden,))
    mem = jnp.zeros((nb_hidden,))
    out = jnp.zeros((nb_hidden,))
    _, spk_rec = jax.lax.scan(hidden_step, (syn, mem, out), h1, nb_steps, unroll=1)

    h2 = jnp.dot(spk_rec, params[2])
    felif_step, felif_reset = FeLIF(
        dt=1e-3,
        A=args.A,
        I_dsc=args.I_dsc,
        V_thr=args.V_thr,
        P_s=args.P_s,
        spike_fn=superspike(),  # tanh(k=args.k),
    )
    # _, out_rec = jax.lax.scan(output_step, (syn2, mem2), h2, nb_steps, unroll=1)
    _, (out_rec, charge, V_rec, P_Rec) = jax.lax.scan(
        felif_step,
        felif_reset(nb_outputs),
        h2,
        nb_steps,
        unroll=1,
    )
    return out_rec, charge, V_rec, P_Rec, h2, spk_rec, h1, enc_spk, encoder_currents


batched_predict = vmap(predict, in_axes=(None, 0))


# %%


def one_hot(x, k, dtype=jnp.float32):
    """Create a one-hot encoding of x of size k."""
    return jnp.array(x[:, None] == jnp.arange(k), jnp.float32)


@jax.jit
def NLLloss(logits, labels):
    return jnp.mean(-jnp.sum(labels * logits, axis=-1))


def gd(dataset, params, lr=1e-3, nb_epochs=10):
    opt = optax.adamax(learning_rate=lr, b1=0.9, b2=0.995)

    # create and initialize the optimizer
    opt_state = opt.init(params)
    grad_params = params

    @jax.jit
    def net_eval(params, x, y):
        preds = batched_predict(params, x)
        output, charge, V, P, h2, spks, h1, enc_spk, encoder_currents = preds

        m = jnp.sum(output, axis=1)  # Sum over time
        # log_p_y = jax.nn.log_softmax(m, axis=1)

        # reg_loss = 1e-3 * jnp.mean(jnp.sum(spks, axis=1))
        # reg_loss = 1e-3*jnp.mean(jnp.mean(jnp.sum(output,axis=1), axis=1)-100)
        loss_val = optax.softmax_cross_entropy_with_integer_labels(m, y).mean()
        # loss_val = NLLloss(log_p_y, one_hot(y, nb_outputs))  # + reg_loss
        # pred = jnp.argmax(m, axis=1)
        # acc = jnp.array(pred == targets, jnp.float32)
        return loss_val

    # Use JAX to create a function that calculates the loss and the gradient!
    surrogate_grad = jax.value_and_grad(net_eval)

    rng = jax.random.PRNGKey(0)

    # compile the meat of our training loop for speed
    @jax.jit
    def train_step(state, data):
        grad_params, opt_state, rng = state
        events, targets = data  # fix this
        # compute loss and gradient                    # need better augment rng
        loss, grads = surrogate_grad(grad_params, events, targets)
        # generate updates based on the gradients and optimizer
        updates, opt_state = opt.update(grads, opt_state, grad_params)
        # return the updated parameters
        new_state = [optax.apply_updates(grad_params, updates), opt_state, rng]
        return new_state, loss

    # Here's the start of our training loop!
    @scan_tqdm(nb_epochs, print_rate=1)
    def epoch(epoch_state, epoch_num):
        curr_params, curr_opt_state, rng = epoch_state

        shuffle_rng = jax.random.fold_in(rng, epoch_num)

        train_data = shuffle(dataset, shuffle_rng)

        # train epoch
        end_state, train_metrics = jax.lax.scan(
            train_step,  # func
            [curr_params, curr_opt_state, shuffle_rng],  # init
            train_data,  # xs
            train_data.obs.shape[0],  # len
        )

        # acc = jnp.mean(train_metrics[0])
        # loss = jnp.mean(train_metrics[1])
        return end_state, jnp.mean(train_metrics)

    # end epoch

    # epoch loop
    final_state, metrics = jax.lax.scan(
        epoch,
        [grad_params, opt_state, rng],  # metric arrays
        jnp.arange(nb_epochs),  #
        nb_epochs,  # len of loop
    )

    final_params, _, _ = final_state

    # return our final, optimized network.
    return final_params, metrics


def test_gd(dataset, params):
    @jax.jit
    def test_step(test_state, data):
        params = test_state
        events, targets = data

        preds = batched_predict(params, events)
        output, charge, V, P, h2, spks, h1, enc_spk, encoder_currents = preds

        m = jnp.sum(output, axis=1)

        # log_p_y = jax.nn.log_softmax(m, axis=1)
        # reg_loss = 1e-3 * jnp.mean(jnp.sum(spks, axis=1))
        # reg_loss = 1e-3*jnp.mean(jnp.mean(jnp.sum(output,axis=1), axis=1)-100)
        # loss_val = NLLloss(log_p_y, one_hot(targets, nb_outputs))  # + reg_loss
        loss_val = optax.softmax_cross_entropy_with_integer_labels(m, targets).mean()
        pred = jnp.argmax(m, axis=1)
        acc = jnp.array(pred == targets, jnp.float32)
        # acc, pred = spyx.fn.integral_accuracy(time_axis=1)(output, targets)
        return params, [acc, loss_val, pred, targets]

    test_data = shuffle(dataset, jax.random.PRNGKey(0))

    _, test_metrics = jax.lax.scan(
        test_step,  # func
        params,  # init
        test_data,  # xs
        test_data.obs.shape[0],  # len
    )

    acc = jnp.mean(test_metrics[0])
    loss = jnp.mean(test_metrics[1])
    preds = jnp.array(test_metrics[2]).flatten()
    tgts = jnp.array(test_metrics[3]).flatten()
    return acc, loss, preds, tgts


params = [{"gain": enc_gain, "bias": enc_bias}, w1, w2]
grad_params, metrics = gd(
    (x_train, y_train), params, nb_epochs=args.nb_epochs, lr=args.lr
)
acc, loss, preds, tgts = test_gd((x_test, y_test), grad_params)
print("Accuracy:", acc, "Loss:", loss)

if args.nni:
    for m in metrics:
        nni.report_intermediate_result({"Loss": float(m)})

    nni.report_final_result(
        {"default": float(acc), "Accuracy": float(acc), "Loss": float(loss)}
    )
else:

    confusion_matrix = skmetrics.confusion_matrix(tgts, preds)
    cm_display = skmetrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    plt.savefig("cm.png")
    plt.figure()
    plt.plot(metrics)
    plt.savefig("loss.png")

    preds = batched_predict(grad_params, x_test)
    output, charge, V, P, h2, spks, h1, enc_spk, encoder_currents = preds
    plot_spk_charge(
        enc_spk, charge, V, P, output, "Braille dataset with FeLIF Neuron outputs"
    )
