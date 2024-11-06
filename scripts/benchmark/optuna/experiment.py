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
from chex import Array, PRNGKey
from tqdm import trange

from eleanor.models import Scaler, Heracles
from eleanor.datasets import shuffle, loadBraille
from eleanor.quantization import QLinear

SEED = 13
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


def define_model(key, trial):
    # alpha = 0.5581817206696561#trial.suggest_float("alpha", 0.1, 1.0, log=False)
    # beta = 0.3920435184318871#trial.suggest_float("beta", 0.1, 1.0, log=False)
    alpha = trial.suggest_float("alpha", 0.1, 1.0, log=False)
    beta = trial.suggest_float("beta", 0.1, 1.0, log=False)
    scaler = trial.suggest_float("scaler", 1.0, 1000.0, log=False)
    V_thr = trial.suggest_float("V_thr", 0.2, 3.5, log=False)

    key1, key2, key3, key4, key5 = jrandom.split(key, 5)
    enc_gain = jax.random.normal(key1, shape=(128,)) * 0.18436009935019085
    enc_bias = jax.random.normal(key2, shape=(128,))
    model = snn.Sequential(
        EncodingLayer(enc_gain, enc_bias, 32),
        QLinear(128, 256, n_bits=2, key=key3),
        snn.LIF([alpha, beta], key=key4),
        QLinear(256, 27, n_bits=2, key=key5),
        # snn.LIF([alpha, beta]),
        Scaler(scaler),
        Heracles(dt=1e-3, V_thr=V_thr, paramsScale=1e12),
    )
    return model


# Simple batched loss function
@partial(jax.vmap, in_axes=(None, None, 0, 0, 0))
def loss_fn(model, in_states, in_spikes, tgt_class, key):
    out_state, out_spikes = model(in_states, in_spikes, key=key)

    # Get the output of last layer
    final_layer_out = out_spikes[-1][0]
    # final_layer_out = out_spikes[-1]
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
    final_layer_out = out_spikes[-1][0]
    # final_layer_out = out_spikes[-1]
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


def objective(trial):
    trainset, testset, nb_outputs, nb_channels, nb_steps, time_step = loadBraille(
        2, 200
    )

    key = jrandom.key(SEED)
    key, kmodel = jrandom.split(key, 2)

    model = define_model(kmodel, trial)
    optim = optax.adamax(
        learning_rate=trial.suggest_float("lr", 1e-5, 1e-1, log=True), b1=0.9, b2=0.995
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    initial_state = model.init_state(in_shape=(4,), key=jax.random.key(0))

    pbar = trange(0, NBEPOCHS)
    for epoch in pbar:
        key, epoch_key, train_key, test_key = jax.random.split(key, 4)
        x_train, y_train = shuffle(trainset, epoch_key, BATCHSIZE)
        for in_spikes, tgt_class in zip(x_train, y_train):
            # Initializing the membrane potentials of LIF neurons
            model, opt_state, _ = update(
                model, optim, initial_state, opt_state, in_spikes, tgt_class, train_key
            )

        x_test, y_test = shuffle(testset, jax.random.key(0), BATCHSIZE)
        accuracy_test = []
        for in_spikes, tgt_class in zip(x_test, y_test):
            # Initializing the membrane potentials of LIF neurons
            accuracy = calc_accuracy(
                model, initial_state, in_spikes, tgt_class, test_key
            )
            accuracy_test.append(accuracy)
        accuracy_test = jnp.mean(jnp.asarray(accuracy_test))
        trial.report(accuracy_test, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy_test


model = "Heracles"
quantization = "1.5"
storage = optuna.storages.JournalStorage(
    optuna.storages.journal.JournalFileBackend("./experiments_lif.log")
)
try:
    restored_sampler = pickle.load(open(f"sampler_{quantization}bit_{model}.pkl", "rb"))
except FileNotFoundError:
    restored_sampler = optuna.samplers.TPESampler(seed=SEED)

try:
    restored_pruner = pickle.load(open(f"pruner_{quantization}bit_{model}.pkl", "rb"))
except FileNotFoundError:
    restored_pruner = optuna.pruners.NopPruner()

study = optuna.create_study(
    storage=storage,
    study_name=f"{quantization}bit {model}",
    direction="maximize",
    sampler=restored_sampler,
    pruner=restored_pruner,
    load_if_exists=True,
)
study.optimize(objective, n_trials=300)
with open(f"sampler_{quantization}bit_{model}.pkl", "wb") as fout:
    pickle.dump(study.sampler, fout)
with open(f"pruner_{quantization}bit_{model}.pkl", "wb") as fout:
    pickle.dump(study.pruner, fout)
