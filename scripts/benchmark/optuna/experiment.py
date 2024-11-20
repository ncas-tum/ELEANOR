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

from eleanor.models import FeLIFV2
from eleanor.datasets import shuffle, loadBraille
from eleanor.weight_quantization import QuantizedLinear

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
    # encoding_gain = trial.suggest_float("encoding_gain", 0.01, 1.0, log=False)
    alpha = trial.suggest_float("alpha", 0.1, 1.0, log=False)
    beta = trial.suggest_float("beta", 0.1, 1.0, log=False)
    # alpha_o = trial.suggest_float("alpha_o", 0.1, 1.0, log=False)
    # beta_o = trial.suggest_float("beta_o", 0.1, 1.0, log=False)
    # scaler = trial.suggest_float("scaler", 1.0, 1000.0, log=False)
    V_thr = trial.suggest_float("V_thr", 2.5, 3.5, log=False)
    paramScale = 10 ** trial.suggest_int("paramScale", 5, 12)

    key1, key2, key3, key4, key5, key6 = jrandom.split(key, 6)
    enc_gain = jax.random.normal(key1, shape=(128,)) * 0.18436009935019085
    enc_bias = jax.random.normal(key2, shape=(128,))
    model = snn.Sequential(
        EncodingLayer(enc_gain, enc_bias, 32),
        QuantizedLinear(128, 256, quant_bits=3, key=key3),
        snn.LIF([alpha, beta], key=key4),
        QuantizedLinear(256, 27, quant_bits=3, key=key5),
        # snn.LIF([alpha_o, beta_o], key=key6),
        FeLIFV2(dt=1e-3, V_thr=V_thr, paramsScale=paramScale, key=key6),
        # Heracles(dt=1e-3, V_thr=V_thr, paramsScale=paramScale, key=key6),
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
    key, kmodel, kstate = jrandom.split(key, 3)

    model = define_model(kmodel, trial)
    optim = optax.adamax(
        learning_rate=trial.suggest_float("lr", 1e-5, 1e-1, log=True), b1=0.9, b2=0.995
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    initial_state = model.init_state(in_shape=(4,), key=kstate)

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
        trial.report(accuracy_test, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return accuracy_test


class SaveStateCallback:

    def __call__(
        self, study: optuna.study.Study, trial: optuna.trial.FrozenTrial
    ) -> None:
        with open(f"sampler_{quantization}bit_{model}.pkl", "wb") as fout:
            pickle.dump(study.sampler, fout)
        with open(f"pruner_{quantization}bit_{model}.pkl", "wb") as fout:
            pickle.dump(study.pruner, fout)


model = "FeLIF"
quantization = "3"
storage = optuna.storages.RDBStorage("sqlite:///bruno.db")

# try:
#     restored_sampler = pickle.load(open(f"sampler_{quantization}bit_{model}.pkl", "rb"))
# except FileNotFoundError:
#     restored_sampler = optuna.samplers.TPESampler(seed=SEED)

# try:
#     restored_pruner = pickle.load(open(f"pruner_{quantization}bit_{model}.pkl", "rb"))
# except FileNotFoundError:
#     restored_pruner = optuna.pruners.NopPruner()

# study = optuna.create_study(
#     storage=storage,
#     study_name=f"{quantization}bit {model}",
#     direction="maximize",
#     sampler=restored_sampler,
#     pruner=restored_pruner,
#     load_if_exists=True,
# )

study = optuna.load_study(
    storage=storage,
    study_name=f"{quantization}bit {model}",
)
# study.optimize(objective, n_trials=60, callbacks=[SaveStateCallback()])
study.optimize(objective, n_trials=60)
