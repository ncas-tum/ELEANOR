import os
import pickle
from typing import Optional
from functools import partial

import jax
import optax
import optuna
import equinox as eqx
import jax.numpy as jnp
import snnax.snn as snn
import jax.random as jrandom
import optuna.storages.journal
from chex import Array, PRNGKey
from tqdm import trange

from eleanor.models import Heracles
from eleanor.datasets import shuffle, loadBraille
from eleanor.weight_quantization import QuantizedLinear

NBEPOCHS = 150
BATCHSIZE = 128


# Model definition
class EncodingLayer(eqx.Module):
    gain: Array
    bias: Array
    expansion: float

    def __init__(self, gain: Array, bias: Array, expansion: float) -> None:
        self.gain = gain
        self.bias = bias
        self.expansion = expansion

    def __call__(self, synaptic_input: Array, *, key: Optional[PRNGKey] = None):
        output = self.gain * (jnp.tile(synaptic_input, self.expansion) + self.bias)
        return output


def define_Heracles(key, trial):
    alpha = trial.params["alpha"]
    beta = trial.params["beta"]
    V_thr = trial.params["V_thr"]
    paramScale = 10 ** trial.params["paramScale"]

    key1, key2, key3, key4, key5, key6 = jrandom.split(key, 6)
    enc_gain = jax.random.normal(key1, shape=(128,)) * 0.18436009935019085
    enc_bias = jax.random.normal(key2, shape=(128,))
    model = snn.Sequential(
        EncodingLayer(enc_gain, enc_bias, 32),
        QuantizedLinear(128, 256, quant_bits=3, key=key3),
        snn.LIF([alpha, beta], key=key4),
        QuantizedLinear(256, 27, quant_bits=3, key=key5),
        Heracles(dt=1e-3, V_thr=V_thr, paramsScale=paramScale, key=key6),
    )
    return model


def define_LIF(key, trial):
    alpha = trial.params["alpha"]
    beta = trial.params["beta"]
    alpha_o = trial.params["alpha_o"]
    beta_o = trial.params["beta_o"]

    key1, key2, key3, key4, key5, key6 = jrandom.split(key, 6)
    enc_gain = jax.random.normal(key1, shape=(128,)) * 0.18436009935019085
    enc_bias = jax.random.normal(key2, shape=(128,))
    model = snn.Sequential(
        EncodingLayer(enc_gain, enc_bias, 32),
        QuantizedLinear(128, 256, quant_bits=3, key=key3),
        snn.LIF([alpha, beta], key=key4),
        QuantizedLinear(256, 27, quant_bits=3, key=key5),
        snn.LIF([alpha_o, beta_o], key=key6),
    )
    return model


# Simple batched loss function
@partial(jax.vmap, in_axes=(None, None, 0, 0, 0))
def loss_fn(model, in_states, in_spikes, tgt_class, key):
    out_state, out_spikes = model(in_states, in_spikes, key=key)

    # Get the output of last layer
    # final_layer_out = out_spikes[-1][0]
    final_layer_out = out_spikes[-1]
    pred = final_layer_out.sum(axis=0)

    target = jax.nn.one_hot(tgt_class, 27)
    loss = optax.softmax_cross_entropy(pred, target)
    return loss


# Calculating the gradient with Equinox PyTree filters and
# subsequently jitting the resulting function
@eqx.filter_value_and_grad
def loss_and_grad(model, in_states, in_spikes, tgt_class, key):
    keys = jax.random.split(key, BATCHSIZE)
    return jnp.mean(loss_fn(model, in_states, in_spikes, tgt_class, keys))


@partial(jax.vmap, in_axes=(None, None, 0, 0, 0))
def accuracy_fn(model, in_states, in_spikes, tgt_class, key):
    out_state, out_spikes = model(in_states, in_spikes, key=key)
    # final_layer_out = out_spikes[-1][0]
    final_layer_out = out_spikes[-1]
    pred = final_layer_out.sum(axis=0)
    predicted_class = jnp.argmax(pred)
    return predicted_class == tgt_class


@eqx.filter_jit
def calc_accuracy(model, in_states, in_spikes, tgt_class, key):
    keys = jax.random.split(key, BATCHSIZE)
    accuracy = accuracy_fn(model, in_states, in_spikes, tgt_class, keys)
    return jnp.mean(accuracy)


# Finally, we update the parameters using a simple optimizer
@eqx.filter_jit
def update(model, optim, in_states, opt_state, in_spikes, tgt_class, key):
    # Get gradients
    loss, grads = loss_and_grad(model, in_states, in_spikes, tgt_class, key)

    # Calculate parameter updates using the optimizer
    updates, opt_state = optim.update(grads, opt_state)

    # Update parameter PyTree with Equinox and optax
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def run_trial(trial, SEED):
    trainset, testset, nb_outputs, nb_channels, nb_steps, time_step = loadBraille(
        2, 200
    )

    key = jrandom.key(SEED)
    key, kmodel, kstate = jrandom.split(key, 3)

    model = define_LIF(kmodel, trial)
    optim = optax.adamax(learning_rate=trial.params["lr"], b1=0.9, b2=0.995)
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    initial_state = model.init_state(in_shape=(4,), key=kstate)
    total_accuracy = []

    pbar = trange(0, NBEPOCHS)
    for epoch in pbar:
        key, epoch_key, train_key, test_key = jax.random.split(key, 4)
        x_train, y_train = shuffle(trainset, epoch_key, BATCHSIZE)
        for i, (in_spikes, tgt_class) in enumerate(zip(x_train, y_train)):
            # Initializing the membrane potentials of LIF neurons
            model, opt_state, _ = update(
                model,
                optim,
                initial_state,
                opt_state,
                in_spikes,
                tgt_class,
                jrandom.fold_in(train_key, i),
            )

        x_test, y_test = shuffle(testset, jax.random.key(0), BATCHSIZE)
        accuracy_test = []
        for i, (in_spikes, tgt_class) in enumerate(zip(x_test, y_test)):
            # Initializing the membrane potentials of LIF neurons
            accuracy = calc_accuracy(
                model, initial_state, in_spikes, tgt_class, jrandom.fold_in(test_key, i)
            )
            accuracy_test.append(accuracy)
        accuracy_test = jnp.mean(jnp.asarray(accuracy_test))
        total_accuracy.append(accuracy_test.item())

    return total_accuracy


model_name = "LIF"
quantization = "3"
# storage = optuna.storages.JournalStorage(
#     optuna.storages.journal.JournalFileBackend("./bruno.log")
# )
storage = optuna.storages.RDBStorage("sqlite:///bruno.db")

try:
    restored_sampler = pickle.load(
        open(f"sampler_{quantization}bit_{model_name}.pkl", "rb")
    )
except FileNotFoundError:
    restored_sampler = optuna.samplers.TPESampler()

try:
    restored_pruner = pickle.load(
        open(f"pruner_{quantization}bit_{model_name}.pkl", "rb")
    )
except FileNotFoundError:
    restored_pruner = optuna.pruners.NopPruner()

study = optuna.create_study(
    storage=storage,
    study_name=f"{quantization}bit {model_name}",
    direction="maximize",
    sampler=restored_sampler,
    pruner=restored_pruner,
    load_if_exists=True,
)

logdir = f"results/{quantization}bit_{model_name}"
if not os.path.exists(logdir):
    os.makedirs(logdir)

for seed in range(0, 100):
    accuracy = run_trial(study.best_trial, seed)
    jnp.save(os.path.join(logdir, f"{seed}"), accuracy)
