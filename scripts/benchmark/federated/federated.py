import argparse
import os
import time
from functools import partial
from typing import Callable, Optional, Sequence, Union

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
import optax
import optuna
import pandas as pd
import snnax.snn as snn
from chex import Array, PRNGKey
from eleanor.models.jax import Bruno, Heracles
from eleanor.models.jax.variability import D2DVar, StaticWrapper, update_d2d_variability
from eleanor.models.jax.weight_quantization import QuantizedLinear
from flwr_datasets.partitioner import DirichletPartitioner
from snnax.functional.surrogate import SpikeFn, superspike_surrogate
from snnax.snn.layers.stateful import StatefulOutput, StateShape, default_init_fn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm, trange

from dataset import create_dataset

# ROUNDS = 150
# NBEPOCHS = 10
BATCHSIZE_TRAIN = 128  # 4320
BATCHSIZE_TEST = 128  # 1080

fds = None
partitioner = None  # Cache FederatedDataset


def is_trainable(x):
    return (
        eqx.is_array(x)
        and not isinstance(x, StaticWrapper)
        and not isinstance(x, D2DVar)
    )


def load_data(
    partition_id: int,
    num_partitions: int,
    split: bool = True,
    batch_size: int = 128,
    alpha: int = 100,
    upsample: int = 2,
):
    global partitioner
    global fds

    if fds is None:
        fds, _, _, _, _ = create_dataset(upsample, 200)

    if partitioner is None:
        # partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions, partition_by="label", alpha=alpha
        )
        partitioner.dataset = fds["train"]

    partition = partitioner.load_partition(partition_id)
    fds.set_format("numpy")
    partition.set_format("numpy")

    if split:
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        trainloader = DataLoader(
            partition_train_test["train"], batch_size=batch_size, shuffle=True
        )
        valloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
        testloader = DataLoader(fds["test"], batch_size=batch_size)

        return trainloader, valloader, testloader
        # return partition_train_test["train"], partition_train_test["test"], fds["test"]
    else:
        trainloader = DataLoader(partition, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(fds["test"], batch_size=batch_size)

        return trainloader, testloader
        # return partition, fds["test"]


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


def define_model(key, model_name, quant_bits, use_bias, variability, trial):
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
            ouputLayer = Bruno(
                (27,),
                key=key4,
                dt=1e-3,
                threshold=V_thr,
                paramsScale=paramScale,
                variability=variability,
            )
        else:
            ouputLayer = Heracles(
                (27,),
                key=key4,
                dt=1e-3,
                threshold=V_thr,
                paramsScale=paramScale,
                variability=variability,
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


def get_param(model):
    flat = jnp.concatenate(
        [
            p.flatten()
            for p in jax.tree_util.tree_leaves(
                eqx.filter(
                    model, is_trainable, is_leaf=lambda x: isinstance(x, StaticWrapper)
                )
            )
        ]
    )
    return flat


def unflatten_to_pytree(flat_array, reference_pytree):
    """Reconstruct PyTree from flattened array."""
    leaves = jax.tree_util.tree_leaves(
        eqx.filter(
            reference_pytree,
            is_trainable,
            is_leaf=lambda x: isinstance(x, StaticWrapper),
        )
    )

    # Split flat array according to shapes
    new_leaves = []
    idx = 0
    for leaf in leaves:
        size = leaf.size
        new_leaf = flat_array[idx : idx + size].reshape(leaf.shape)
        new_leaves.append(new_leaf)
        idx += size

    # Reconstruct PyTree
    tree_def = jax.tree_util.tree_structure(
        eqx.filter(
            reference_pytree,
            is_trainable,
            is_leaf=lambda x: isinstance(x, StaticWrapper),
        )
    )
    model_unflatten = jax.tree_util.tree_unflatten(tree_def, new_leaves)
    return eqx.combine(model_unflatten, reference_pytree)


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
def loss_and_grad(trainable, static, in_states, in_spikes, tgt_class, key):
    model = eqx.combine(trainable, static)
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
    trainable, static = eqx.partition(
        model, eqx.is_array, is_leaf=lambda x: isinstance(x, StaticWrapper)
    )
    loss, grads = loss_and_grad(trainable, static, in_states, in_spikes, tgt_class, key)

    # Calculate parameter updates using the optimizer
    updates, opt_state = optim.update(grads, opt_state)

    # Update parameter PyTree with Equinox and optax
    model = eqx.apply_updates(model, updates)
    return model, opt_state, loss


def _train_round(
    model, opt_state, trainset, valset, key, NBEPOCHS, optim, shared_params
):
    model = eqx.combine(model, shared_params)
    key, kstate = jrandom.split(key)
    initial_state = model.init_state(in_shape=(4,), key=kstate)

    start_time = time.process_time()
    for _ in trange(NBEPOCHS, leave=False, desc="Epochs"):
        key, epoch_key, train_key, test_key = jax.random.split(key, 4)
        # x_train, y_train = shuffle(trainset, epoch_key, len(trainset[1]))

        loss_train = []
        # for i, (in_spikes, tgt_class) in enumerate(
        #     tqdm(zip(x_train, y_train), total=len(y_train), leave=False)
        # ):
        for i, sample in enumerate(tqdm(trainset, leave=False, desc="Batch")):
            in_spikes = jnp.asarray(sample["data"])
            tgt_class = jnp.asarray(sample["label"])
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

        accuracy_test = []
        # x_test, y_test = shuffle(valset, jax.random.key(0), len(valset[1]))
        # for i, (in_spikes, tgt_class) in enumerate(zip(x_test, y_test)):
        for i, sample in enumerate(valset):
            in_spikes = jnp.asarray(sample["data"])
            tgt_class = jnp.asarray(sample["label"])
            accuracy = calc_accuracy(
                model, initial_state, in_spikes, tgt_class, jrandom.fold_in(test_key, i)
            )
            accuracy_test.append(accuracy)
        accuracy_test = jnp.mean(jnp.asarray(accuracy_test))
    end_time = time.process_time()

    model, _ = eqx.partition(model, eqx.is_array)

    return (
        accuracy_test,
        loss_train,
        model,
        opt_state,
        end_time - start_time,
        len(trainset.dataset),
        len(valset.dataset),
    )


def _objective(
    model_name,
    quant_bits,
    use_bias,
    variability,
    num_partitions,
    ROUNDS,
    NBEPOCHS,
    trial,
    key,
    writer=None,
):
    key, kmodel = jrandom.split(key)

    model_list = [
        define_model(k, model_name, quant_bits, use_bias, variability, trial)
        for k in jax.random.split(kmodel, num_partitions)
    ]
    key, subkey = jax.random.split(key)
    model_test = define_model(
        subkey, model_name, quant_bits, use_bias, variability, trial
    )

    model_list, shared_params = eqx.partition(model_list, eqx.is_array)
    shared_params = shared_params[0]
    # models_stacked = jax.tree.map(lambda *args: jnp.stack(args), eqx.filter(*model_list, eqx.is_array))

    optim = optax.adamax(
        learning_rate=trial.suggest_float("lr", 1e-5, 1e-1, log=True), b1=0.9, b2=0.995
    )
    train_round = partial(
        _train_round, NBEPOCHS=NBEPOCHS, optim=optim, shared_params=shared_params
    )

    opt_states = [
        optim.init(
            eqx.filter(
                model, eqx.is_array, is_leaf=lambda x: isinstance(x, StaticWrapper)
            )
        )
        for model in model_list
    ]
    # opt_states_stacked = jax.tree.map(lambda *args: jnp.stack(args), *opt_states)

    # X_train = []
    # y_train = []
    # X_val = []
    # y_val = []
    trainsets = []
    valsets = []
    for i in range(num_partitions):
        trainset, valset, _ = load_data(
            i, num_partitions, True, BATCHSIZE_TRAIN, 100, 2
        )
        # X_train.append(trainset["data"])
        # y_train.append(trainset["label"])

        # X_val.append(valset["data"])
        # y_val.append(valset["label"])
        trainsets.append(trainset)
        valsets.append(valset)
    _, testset = load_data(0, 1, False, BATCHSIZE_TRAIN, 100, 2)

    # X_train = jnp.stack(X_train)
    # y_train = jnp.stack(y_train)
    # X_val = jnp.stack(X_val)
    # y_val = jnp.stack(y_val)

    total_loss = []
    total_accuracy = []
    total_accuracy_test = []
    total_time = []
    print("Training sizes ", [len(y.dataset) for y in trainsets])
    print("Testing sizes ", [len(y.dataset) for y in valsets])

    pbar = trange(0, ROUNDS, leave=False, desc="Rounds")
    for current_epoch in pbar:
        key, train_key, test_key = jax.random.split(key, 3)

        loss_train = []
        accuracy_val = []
        new_opt_states = []
        round_times = []

        total_len_train = 0
        total_len_val = 0
        total_params = 0.0

        keys = jrandom.split(train_key, num_partitions)
        # acc, loss, models_stacked, opt_states_stacked, round_time, train_size, test_size = jax.pmap(train_round, axis_name='partitions')(models_stacked, opt_states_stacked, (X_train, y_train), (X_val, y_val), keys)

        for i, (model, opt_state, k) in enumerate(
            tqdm(
                zip(model_list, opt_states, keys),
                leave=False,
                total=len(model_list),
                desc="Partitions",
            )
        ):
            acc, loss, model, opt_state, round_time, train_size, val_size = train_round(
                model, opt_state, trainsets[i], valsets[i], k
            )

            total_len_val += val_size
            total_len_train += train_size
            total_params = total_params + get_param(model) * train_size

            new_opt_states.append(opt_state)
            loss_train.append(loss * train_size)
            accuracy_val.append(acc * val_size)
            round_times.append(round_time * train_size)

        total_params = total_params / total_len_train
        model_list = [unflatten_to_pytree(total_params, m) for m in model_list]
        opt_states = new_opt_states

        loss_train = jnp.sum(jnp.asarray(loss_train)).item() / total_len_train
        accuracy_val = jnp.sum(jnp.asarray(accuracy_val)).item() / total_len_val
        round_times = jnp.sum(jnp.asarray(round_times)).item() / total_len_train

        total_loss.append(loss_train)
        total_accuracy.append(accuracy_val)
        total_time.append(round_times)

        accuracy_test = []
        kstate, ktest = jrandom.split(test_key)
        model_test = unflatten_to_pytree(total_params, model_test)
        initial_state = model_test.init_state(in_shape=(4,), key=kstate)
        for i, sample in enumerate(tqdm(testset, leave=False)):
            in_spikes = jnp.asarray(sample["data"])
            tgt_class = jnp.asarray(sample["label"])
            accuracy = calc_accuracy(
                model_test,
                initial_state,
                in_spikes,
                tgt_class,
                jrandom.fold_in(ktest, i),
            )
            accuracy_test.append(accuracy)
        accuracy_test = jnp.mean(jnp.asarray(accuracy_test)).item()
        total_accuracy_test.append(accuracy_test)

        pbar.set_postfix(
            {
                "Acc (val)": accuracy_val,
                "Acc (test)": accuracy_test,
                "Loss": loss_train,
                "Time": round_time,
            }
        )

        if writer is not None:
            writer.add_scalar("Accuracy (val)", accuracy_val, current_epoch)
            writer.add_scalar("Accuracy (test)", accuracy_test, current_epoch)
            writer.add_scalar("Loss", loss_train, current_epoch)
            writer.add_scalar("Time", round_time, current_epoch)

    final_model = unflatten_to_pytree(total_params, model_list[0])
    return (
        total_accuracy,
        total_accuracy_test,
        total_loss,
        total_time,
        eqx.combine(final_model, shared_params),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="FeLIF")
    parser.add_argument("-q", "--quantization", type=str, default="FP")
    parser.add_argument("-v", "--variability", type=float, default=0.1)
    parser.add_argument("-p", "--num_partitions", type=int, default=10)
    parser.add_argument("-r", "--rounds", type=int, default=150)
    parser.add_argument("-e", "--epochs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    print(args.model, args.quantization, args.seed)

    if args.model == "LIF" or args.model == "RLIF":
        getOutput = _outputLIF
    elif args.model == "Heracles" or args.model == "FeLIF":
        getOutput = _outputFeLIF
    else:
        raise Exception(f"Model {args.model} not found")

    if args.log and not args.test:
        writer = SummaryWriter(
            comment=f" - {args.num_partitions} Partitions {args.variability}"
        )
    else:
        writer = None
    objective = partial(
        _objective,
        args.model,
        args.quantization,
        False,
        args.variability,
        args.num_partitions,
        args.rounds,
        args.epochs,
        writer=writer,
    )
    storage = optuna.storages.RDBStorage(
        "sqlite:////home/p306945/Projects/Bruno/scripts/benchmark/federated/bruno.db"
    )

    study = optuna.load_study(
        storage=storage, study_name=f"{args.quantization}bit {args.model}"
    )

    logdir = f"results_federated/{args.num_partitions}/{args.epochs}/{args.model}/{args.quantization}/{args.variability}"
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    data_dir = os.path.join(logdir, f"data_{args.seed}.pkl")
    model_dir = os.path.join(logdir, f"model_{args.seed}.eqx")

    # for seed in trange(0, 50):
    # trial = study.trials[np.argsort([-t.value for t in study.trials])[1]]
    key = jrandom.key(args.seed)
    key, kopt = jrandom.split(key)

    if args.test:
        final_model = eqx.tree_deserialise_leaves(
            model_dir,
            define_model(
                kopt,
                args.model,
                args.quantization,
                False,
                args.variability,
                study.best_trial,
            ),
        )
        total_acc = []
        for i in trange(10):
            key, kvar = jrandom.split(key)
            final_model = update_d2d_variability(final_model, kvar)

            key, kstate = jrandom.split(key)
            initial_state = final_model.init_state(in_shape=(4,), key=kstate)

            kshuffle, ktest = jrandom.split(key)
            _, testset = load_data(
                0, args.num_partitions, False, BATCHSIZE_TEST, 100, 2
            )

            accuracy_test = []
            for i, sample in enumerate(tqdm(testset, leave=False)):
                in_spikes = jnp.asarray(sample["data"])
                tgt_class = jnp.asarray(sample["label"])
                accuracy = calc_accuracy(
                    final_model,
                    initial_state,
                    in_spikes,
                    tgt_class,
                    jrandom.fold_in(ktest, i),
                )
                accuracy_test.append(accuracy)
            accuracy_test = jnp.mean(jnp.asarray(accuracy_test)).item()
            total_acc.append(accuracy_test)
        jnp.save(os.path.join(logdir, "variability.npy"), total_acc)
    else:
        accuracy_train, accuracy_var, loss_train, times, final_model = objective(
            study.best_trial, kopt
        )

        key, kvar = jrandom.split(key)
        final_model = update_d2d_variability(final_model, kvar)

        key, kstate = jrandom.split(key)
        initial_state = final_model.init_state(in_shape=(4,), key=kstate)

        kshuffle, ktest = jrandom.split(key)
        _, testset = load_data(0, args.num_partitions, False, BATCHSIZE_TEST, 100, 2)

        accuracy_test = []
        for i, sample in enumerate(tqdm(testset, leave=False)):
            in_spikes = jnp.asarray(sample["data"])
            tgt_class = jnp.asarray(sample["label"])
            accuracy = calc_accuracy(
                final_model,
                initial_state,
                in_spikes,
                tgt_class,
                jrandom.fold_in(ktest, i),
            )
            accuracy_test.append(accuracy)
        accuracy_test = jnp.mean(jnp.asarray(accuracy_test)).item()

        if writer is not None:
            writer.add_hparams(
                {
                    "Variability": args.variability,
                    "Partitions": args.num_partitions,
                    "Epochs": args.epochs,
                },
                {"Test accuracy": accuracy_test},
                run_name=".",
            )
        eqx.tree_serialise_leaves(model_dir, final_model)
        nepochs = len(accuracy_train)
        df = pd.DataFrame(
            {
                "Accuracy": accuracy_train,
                "Accuracy (test)": accuracy_var,
                "Loss": loss_train,
                "Epoch": jnp.arange(nepochs),
                "Seed": [args.seed] * nepochs,
                "Variability": [args.variability] * nepochs,
                "Quantization": [args.quantization] * nepochs,
                "Model": [args.model] * nepochs,
                "Time": times,
                "Partitions": [args.num_partitions] * nepochs,
                "Final Accuracy": [accuracy_test] * nepochs,
            }
        )

        # try:
        #     df = pd.read_pickle(data_dir)
        # except Exception:
        #     df = pd.DataFrame()

        # df = pd.concat(
        #     [df, newdf],
        #     ignore_index=True,
        # )
        df.to_pickle(data_dir)
