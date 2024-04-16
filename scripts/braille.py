import argparse

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import nni
import numpy as np
import optax
from dataset import loadBraille, shuffle
from network import network_builder
from sklearn import metrics as skmetrics
from spyx.axn import tanh
from tqdm import trange

plt.style.use("dark_background")


def plot_spk_charge(spk_in, charge, V, P, spk_out, title):
    idx = jnp.argmax(jnp.sum(spk_out[0], axis=0))
    # Generate Plots
    _, ax = plt.subplots(
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
    ax[1].plot(charge[0][:, idx])
    ax[1].set_ylabel("Internal charge")
    ax[1].set_xlabel("Time step")

    # Plot membrane potential
    ax[2].plot(V[0][:, idx])
    ax[2].set_ylabel("Voltage")
    ax[2].set_xlabel("Time step")

    # Plot membrane potential
    ax[3].plot(P[0][:, idx])
    ax[3].set_ylabel("Polarization")
    ax[3].set_xlabel("Time step")

    ax[4].scatter(*jnp.where(spk_out[0]), s=4, c="white", marker="|")
    ax[4].set_ylabel("Output Spikes")
    ax[4].set_yticks([])

    plt.suptitle(title)
    # plt.show()
    plt.savefig("braille.png")


parser = argparse.ArgumentParser()
parser.add_argument("--lr", type=float, default=0.008654453540431036)
parser.add_argument("--reg", type=float, default=1e-3)
parser.add_argument("--nb_epochs", type=int, default=300)
parser.add_argument("--nb_hidden", type=int, default=256)
parser.add_argument("--nb_repetitions", type=int, default=200)
parser.add_argument("--nb_upsample", type=int, default=2)
parser.add_argument("--enc_fan_out", type=int, default=32)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--tau_mem", type=float, default=0.014844073641062557)
parser.add_argument("--tau_syn", type=float, default=0.029928455308049037)
parser.add_argument("--k", type=float, default=8.33030241666825)
parser.add_argument("--A", type=float, default=25e-12)
parser.add_argument("--I_dsc", type=float, default=8.599065698995122e-11)
parser.add_argument("--V_thr", type=float, default=1.6585232527273086)
parser.add_argument("--P_s", type=float, default=0.22)
parser.add_argument("--felif_weight_scale", type=float, default=27.90474745349244)
parser.add_argument("--enc_weight_scale", type=float, default=0.18436009935019085)
parser.add_argument(
    "--optimizer", type=str, choices=["adamax", "sgd"], default="adamax"
)
parser.add_argument("--nni", action="store_true")

args = parser.parse_args()

if args.nni:
    nni_params = nni.get_next_parameter()
    d = vars(args)  # copy by reference (checked below)
    for key, val in nni_params.items():
        d[key] = val
        assert args.__dict__[key] == d[key]

print(args)

trainset, testset, nb_outputs, nb_channels, nb_steps, time_step = loadBraille(
    args.nb_upsample, args.nb_repetitions
)

batch_size = args.batch_size
nb_inputs = nb_channels * args.enc_fan_out
nb_hidden = args.nb_hidden
alpha = float(np.exp(-time_step / args.tau_syn))
beta = float(np.exp(-time_step / args.tau_mem))

# Network builder
predict = network_builder(
    # Network params
    nb_inputs,
    nb_hidden,
    nb_outputs,
    nb_steps,
    # Encoding
    args.enc_fan_out,
    alpha,
    beta,
    # FeLIF
    args.A,
    args.I_dsc,
    args.V_thr,
    args.P_s,
    spike_fn=tanh(k=args.k),
)

# Parameters creation
# Encoder
key = jax.random.key(0)
key, subkey1, subkey2, subkey3, subkey4, subkey5 = jax.random.split(key, 6)
enc_gain = jax.random.normal(subkey1, shape=(nb_inputs,)) * args.enc_weight_scale
enc_bias = jax.random.normal(subkey1, shape=(nb_inputs,))

# FeLIF
fwd_weight_scale = 3.0
felif_weight_scale = args.felif_weight_scale
w1 = (
    jax.random.normal(subkey3, shape=(nb_inputs, nb_hidden))
    * fwd_weight_scale
    / np.sqrt(nb_inputs)
)
w2 = (
    jax.random.normal(subkey4, shape=(nb_hidden, nb_outputs))
    * felif_weight_scale
    / np.sqrt(nb_hidden)
)
params = [{"gain": enc_gain, "bias": enc_bias}, w1, w2]


@jax.jit
def loss_fn(output, y, firing_rate=10.0):
    # m = jnp.sum(output, axis=1)  # Sum over time
    # loss_val = jnp.mean((m - y * firing_rate) ** 2)
    # loss_val = optax.softmax_cross_entropy(m, y).mean()

    fs = (
        output
        * (
            jnp.tile(
                jnp.arange(output.shape[1])[None, :, None], (batch_size, 1, nb_outputs)
            )
            - 300
        )
        + 300
    )
    t_fs = jnp.min(fs, axis=1) / 300
    loss_val = optax.softmax_cross_entropy(-t_fs, y).mean()
    loss_reg = jnp.sum(y * jnp.exp(t_fs), axis=1).mean()
    return loss_val + args.reg * loss_reg


@jax.jit
def accuracy_fn(output, y):
    fs = (
        output
        * (
            jnp.tile(
                jnp.arange(output.shape[1])[None, :, None], (batch_size, 1, nb_outputs)
            )
            - 300
        )
        + 300
    )
    t_fs = jnp.min(fs, axis=1)
    predicted_class = jnp.argmin(t_fs, axis=1)
    # predicted_class = jnp.argmax(jnp.sum(output, axis=1), axis=1)
    return jnp.mean(predicted_class == y)


# @jax.jit
def loss_eval(params, x, y):
    preds = predict(params, x)
    output, charge, _, _, _, _, _, _, _ = preds
    loss_val = loss_fn(output, y)

    # loss_cha = loss_fn(charge, y)
    # loss_spk = loss_fn(output, y)
    # loss_val = (
    #     loss_cha + loss_spk + args.reg * jnp.mean((jnp.sum(output, axis=1) - 10) ** 2)
    # )
    return loss_val


surrogate_grad = jax.value_and_grad(loss_eval)

if args.optimizer == "adamax":
    opt = optax.adamax(learning_rate=args.lr, b1=0.9, b2=0.995)
elif args.optimizer == "sgd":
    opt = optax.sgd(learning_rate=args.lr)
else:
    raise ValueError("Unknown optimizer")

opt_state = opt.init(params)

pbar = trange(args.nb_epochs)
for _ in pbar:
    key, epoch_key = jax.random.split(key)
    x_train, y_train = shuffle(trainset, epoch_key, batch_size)
    loss_train = []
    for x, y in zip(x_train, y_train):
        loss, grads = surrogate_grad(params, x, jax.nn.one_hot(y, nb_outputs))
        updates, opt_state = opt.update(grads, opt_state, params)
        params = optax.apply_updates(params, updates)

        loss_train.append(loss)
    loss_train = jnp.mean(jnp.asarray(loss_train))

    x_test, y_test = shuffle(testset, jax.random.key(0), batch_size)
    loss_test = []
    accuracy_test = []
    # accuracy_charge_test = []
    for x, y in zip(x_test, y_test):
        preds = predict(params, x)
        output, charge, V, P, h2, spks, h1, enc_spk, encoder_currents = preds

        # loss_cha = loss_fn(charge, jax.nn.one_hot(y, nb_outputs))
        # loss_spk = loss_fn(output, jax.nn.one_hot(y, nb_outputs))
        # loss_val = (
        #     loss_cha
        #     + loss_spk
        #     + args.reg * jnp.mean((jnp.sum(output, axis=1) - 10) ** 2)
        # )
        loss_val = loss_fn(output, jax.nn.one_hot(y, nb_outputs))
        accuracy = accuracy_fn(output, y)
        # accuracy_charge = accuracy_fn(charge, y)

        loss_test.append(loss_val)
        accuracy_test.append(accuracy)
        # accuracy_charge_test.append(accuracy_charge)
    loss_test = jnp.mean(jnp.asarray(loss_test))
    accuracy_test = jnp.mean(jnp.asarray(accuracy_test))
    # accuracy_charge_test = jnp.mean(jnp.asarray(accuracy_charge_test))

    pbar.set_postfix(
        {
            "Loss": loss_train,
            "Accuracy": accuracy_test,
        }
    )

    if args.nni:
        nni.report_intermediate_result(
            {
                "default": loss_test.item(),
                "Accuracy": accuracy_test.item(),
                # "Accuracy (charge)": accuracy_charge_test.item(),
                "Loss train": loss_train.item(),
                "Loss test": loss_test.item(),
            }
        )

if args.nni:
    nni.report_final_result(
        {
            "default": accuracy_test.item(),
            "Accuracy": accuracy_test.item(),
            # "Accuracy (charge)": accuracy_charge_test.item(),
            "Loss train": loss_train.item(),
            "Loss test": loss_test.item(),
        }
    )
else:
    x_test, y_test = shuffle(testset, key, batch_size)

    tgts = []
    predicted_class = []
    for x, y in zip(x_test, y_test):
        preds = predict(params, x)
        output, charge, V, P, h2, spks, h1, enc_spk, encoder_currents = preds

        # loss_cha = loss_fn(charge, jax.nn.one_hot(y, nb_outputs))
        # loss_spk = loss_fn(output, jax.nn.one_hot(y, nb_outputs))
        # loss_val = (
        #     loss_cha
        #     + loss_spk
        #     + args.reg * jnp.mean((jnp.sum(output, axis=1) - 10) ** 2)
        # )
        loss_val = loss_fn(output, jax.nn.one_hot(y, nb_outputs))
        # accuracy = accuracy_fn(output, y)
        # accuracy_charge = accuracy_fn(charge, y)

        tgts.append(y)
        predicted_class.append(jnp.argmax(jnp.sum(output, axis=1), axis=1))
    tgts = jnp.concatenate(tgts)
    predicted_class = jnp.concatenate(predicted_class)

    confusion_matrix = skmetrics.confusion_matrix(tgts, predicted_class)
    cm_display = skmetrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    plt.savefig("cm.png")
    plt.figure()
    plot_spk_charge(
        enc_spk, charge, V, P, output, "Braille dataset with FeLIF Neuron outputs"
    )
