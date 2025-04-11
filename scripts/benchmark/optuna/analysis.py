import os
import argparse
from typing import Union, Callable, Optional, Sequence
from functools import partial

import jax
import optax
import optuna
import pandas as pd
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrandom
import optuna.storages.journal
from chex import Array, PRNGKey
from tqdm import trange

import snnax.snn as snn
from eleanor.models import FeLIF, Heracles
from eleanor.datasets import shuffle, loadBraille
from snnax.snn.layers.stateful import StateShape, StatefulOutput, default_init_fn
from snnax.functional.surrogate import SpikeFn, superspike_surrogate
from eleanor.weight_quantization import QuantizedLinear

NBEPOCHS = 150
BATCHSIZE_TRAIN = 128  # 4320
BATCHSIZE_TEST = 128  # 1080


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


_spikefn = superspike_surrogate(10.0)


class RLIF(snn.LIF):
    recurrent: eqx.Module

    def __init__(
        self,
        n_features,
        quant_bits,
        decay_constants: Union[Sequence[float], Array],
        spike_fn: SpikeFn = _spikefn,
        threshold: Array = 1.0,
        stop_reset_grad: bool = True,
        reset_val: Optional[Array] = None,
        init_fn: Optional[Callable] = default_init_fn,
        shape: Optional[StateShape] = None,
        use_bias: Optional[bool] = False,
        key: Optional[PRNGKey] = None,
    ) -> None:

        key, krec = jrandom.split(key, 2)
        super().__init__(
            decay_constants,
            spike_fn,
            threshold,
            stop_reset_grad,
            reset_val,
            init_fn,
            shape,
            key,
        )
        if quant_bits == "FP":
            self.recurrent = eqx.nn.Linear(
                n_features, n_features, use_bias=use_bias, key=krec
            )
        else:
            quant_bits = int(quant_bits)
            self.recurrent = QuantizedLinear(
                n_features,
                n_features,
                quant_bits=quant_bits,
                use_bias=use_bias,
                key=krec,
            )

    def __call__(
        self,
        state: Sequence[Array],
        synaptic_input: Array,
        *,
        key: Optional[PRNGKey] = None,
    ) -> StatefulOutput:
        mem_pot, syn_curr, spike_output = state

        if self.reset_val is None:
            reset_pot = mem_pot * spike_output
        else:
            reset_pot = (mem_pot - self.reset_val) * spike_output

        # Optionally stop gradient propagation through refectory potential
        refectory_potential = (
            jax.lax.stop_gradient(reset_pot) if self.stop_reset_grad else reset_pot
        )
        mem_pot = mem_pot - refectory_potential

        alpha = jax.lax.clamp(0.5, self.decay_constants[0], 1.0)
        beta = jax.lax.clamp(0.5, self.decay_constants[1], 1.0)

        mem_pot = alpha * mem_pot + (1.0 - alpha) * (syn_curr)
        syn_curr = beta * syn_curr + (1.0 - beta) * (
            synaptic_input + self.recurrent(spike_output, key=key)
        )

        spike_output = self.spike_fn(mem_pot - self.threshold)

        state = [mem_pot, syn_curr, spike_output]
        return [state, spike_output]


def define_model(key, model_name, quant_bits, use_bias, trial):
    # encoding_gain = trial.suggest_float("encoding_gain", 0.01, 1.0, log=False)
    alpha = trial.suggest_float("alpha", 0.1, 1.0, log=False)
    beta = trial.suggest_float("beta", 0.1, 1.0, log=False)

    key1, key2, key3, key4 = jrandom.split(key, 4)
    if model_name == "LIF" or model_name == "RLIF":
        alpha_o = trial.suggest_float("alpha_o", 0.1, 1.0, log=False)
        beta_o = trial.suggest_float("beta_o", 0.1, 1.0, log=False)
        ouputLayer = snn.LIF([alpha_o, beta_o], key=key4)
    elif model_name == "Heracles" or model_name == "FeLIF":
        V_thr = trial.suggest_float("V_thr", 2.5, 3.5, log=False)
        paramScale = 10 ** trial.suggest_int("paramScale", 5, 12)
        if model_name == "FeLIF":
            ouputLayer = FeLIF(dt=1e-3, V_thr=V_thr, paramsScale=paramScale, key=key4)
        else:
            ouputLayer = Heracles(
                dt=1e-3, V_thr=V_thr, paramsScale=paramScale, key=key4
            )
    else:
        raise Exception(f"Model {model_name} not found")

    if model_name == "RLIF":
        hiddenLayer = RLIF(256, quant_bits, [alpha, beta], use_bias=use_bias, key=key2)
    else:
        hiddenLayer = snn.LIF([alpha, beta], key=key2)

    enc_gain = jax.random.normal(key1, shape=(128,)) * 0.18436009935019085
    enc_bias = jax.random.normal(key2, shape=(128,))

    if quant_bits == "FP":
        model = snn.Sequential(
            EncodingLayer(enc_gain, enc_bias, 32),
            eqx.nn.Linear(128, 256, use_bias=use_bias, key=key1),
            hiddenLayer,
            eqx.nn.Linear(256, 27, use_bias=use_bias, key=key3),
            ouputLayer,
        )
    else:
        quant_bits = int(quant_bits)
        model = snn.Sequential(
            EncodingLayer(enc_gain, enc_bias, 32),
            QuantizedLinear(
                128, 256, use_bias=use_bias, quant_bits=quant_bits, key=key1
            ),
            hiddenLayer,
            QuantizedLinear(
                256, 27, use_bias=use_bias, quant_bits=quant_bits, key=key3
            ),
            ouputLayer,
        )
    return model


def _outputLIF(model, in_states, in_spikes, key):
    out_state, out_spikes = model(in_states, in_spikes, key=key)
    final_layer_out = out_spikes[-1]
    return final_layer_out


def _outputFeLIF(model, in_states, in_spikes, key):
    out_state, out_spikes = model(in_states, in_spikes, key=key)
    final_layer_out = out_spikes[-1][0]
    return final_layer_out


# Simple batched loss function
@partial(jax.vmap, in_axes=(None, None, 0, 0, 0))
def loss_fn(model, in_states, in_spikes, tgt_class, key):
    # Get the output of last layer
    final_layer_out = getOutput(model, in_states, in_spikes, key)
    # final_layer_out = out_spikes[-1]
    pred = final_layer_out.sum(axis=0)

    target = jax.nn.one_hot(tgt_class, 27)
    loss = optax.softmax_cross_entropy(pred, target)
    return loss


# Calculating the gradient with Equinox PyTree filters and
# subsequently jitting the resulting function
@eqx.filter_value_and_grad
def loss_and_grad(model, in_states, in_spikes, tgt_class, key):
    keys = jax.random.split(key, in_spikes.shape[0])
    return jnp.mean(loss_fn(model, in_states, in_spikes, tgt_class, keys))


@partial(jax.vmap, in_axes=(None, None, 0, 0, 0))
def accuracy_fn(model, in_states, in_spikes, tgt_class, key):
    final_layer_out = getOutput(model, in_states, in_spikes, key)
    pred = final_layer_out.sum(axis=0)
    predicted_class = jnp.argmax(pred)
    return predicted_class == tgt_class


@eqx.filter_jit
def calc_accuracy(model, in_states, in_spikes, tgt_class, key):
    keys = jax.random.split(key, in_spikes.shape[0])
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


def _objective(model_name, quant_bits, use_bias, trial, seed):
    trainset, testset, nb_outputs, nb_channels, nb_steps, time_step = loadBraille(
        2, 200
    )

    key = jrandom.key(seed)
    key, kmodel, kstate = jrandom.split(key, 3)

    model = define_model(kmodel, model_name, quant_bits, use_bias, trial)
    optim = optax.adamax(
        learning_rate=trial.suggest_float("lr", 1e-5, 1e-1, log=True), b1=0.9, b2=0.995
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    initial_state = model.init_state(in_shape=(4,), key=kstate)
    total_accuracy = []
    total_loss = []

    pbar = trange(0, NBEPOCHS, leave=False)
    for _ in pbar:
        key, epoch_key, train_key, test_key = jax.random.split(key, 4)
        x_train, y_train = shuffle(trainset, epoch_key, BATCHSIZE_TRAIN)

        loss_train = []
        for i, (in_spikes, tgt_class) in enumerate(zip(x_train, y_train)):
            # Initializing the membrane potentials of LIF neurons
            model, opt_state, loss = update(
                model,
                optim,
                initial_state,
                opt_state,
                in_spikes,
                tgt_class,
                jrandom.fold_in(train_key, i),
            )
            loss_train.append(loss)
        loss_train = jnp.mean(jnp.asarray(loss_train))
        total_loss.append(loss_train.item())

        x_test, y_test = shuffle(testset, jax.random.key(0), BATCHSIZE_TEST)
        accuracy_test = []
        for i, (in_spikes, tgt_class) in enumerate(zip(x_test, y_test)):
            # Initializing the membrane potentials of LIF neurons
            accuracy = calc_accuracy(
                model, initial_state, in_spikes, tgt_class, jrandom.fold_in(test_key, i)
            )
            accuracy_test.append(accuracy)
        accuracy_test = jnp.mean(jnp.asarray(accuracy_test))
        total_accuracy.append(accuracy_test.item())
        pbar.set_postfix({"Acc": accuracy_test.item()})

    return total_accuracy, total_loss


parser = argparse.ArgumentParser()
parser.add_argument("-m", "--model", type=str, default="FeLIF")
parser.add_argument("-q", "--quantization", type=str, default="FP")
parser.add_argument("--seed", type=int, default=0)

args = parser.parse_args()
print(args.model, args.quantization, args.seed)

if args.model == "LIF" or args.model == "RLIF":
    getOutput = _outputLIF
elif args.model == "Heracles" or args.model == "FeLIF":
    getOutput = _outputFeLIF
else:
    raise Exception(f"Model {args.model} not found")

objective = partial(_objective, args.model, args.quantization, False)
storage = optuna.storages.RDBStorage("sqlite:///bruno.db")

study = optuna.load_study(
    storage=storage, study_name=f"{args.quantization}bit {args.model}"
)

logdir = f"results/{args.quantization}bit_{args.model}"
if not os.path.exists(logdir):
    os.makedirs(logdir)

try:
    df = pd.read_pickle(os.path.join(logdir, f"model_{args.seed}.pkl"))
except Exception:
    df = pd.DataFrame()

# for seed in trange(0, 50):
# trial = study.trials[np.argsort([-t.value for t in study.trials])[1]]
accuracy, loss = objective(study.best_trial, args.seed)

nepochs = len(accuracy)
newdf = pd.DataFrame(
    {
        "Accuracy": accuracy,
        "Loss": loss,
        "Epoch": jnp.arange(nepochs),
        "Seed": [args.seed] * nepochs,
        "Quantization": [args.quantization] * nepochs,
        "Model": [args.model] * nepochs,
    }
)

df = pd.concat(
    [df, newdf],
    ignore_index=True,
)
df.to_pickle(os.path.join(logdir, f"model_{args.seed}.pkl"))
