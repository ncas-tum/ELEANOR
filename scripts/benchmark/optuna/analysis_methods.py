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
import snnax.snn as snn
import jax.random as jrandom
from chex import Array, PRNGKey
from tqdm import trange
from snnax.snn.layers.stateful import StateShape, StatefulOutput, default_init_fn
from snnax.functional.surrogate import SpikeFn, superspike_surrogate

from eleanor.datasets import shuffle, loadBraille
from eleanor.models.jax import Bruno, Heracles, Checkpoint
from eleanor.models.jax.weight_quantization import QuantizedLinear
from eleanor.models.jax.variability import StaticWrapper

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


def define_model(key, model_name, quant_bits, use_bias, trial, method):
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
            if method == "Bruno":
                ouputLayer = Bruno(
                    dt=1e-3, V_thr=V_thr, paramsScale=paramScale, key=key4
                )
            elif method == "Checkpoints":
                ouputLayer = Checkpoint(
                    dt=1e-3,
                    V_thr=V_thr,
                    paramsScale=paramScale,
                    checkpoints=None,
                    key=key4,
                )
            else:
                ouputLayer = Checkpoint(
                    dt=1e-3,
                    V_thr=V_thr,
                    paramsScale=paramScale,
                    checkpoints=-1,
                    key=key4,
                )
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
def _loss_fn(model, in_states, in_spikes, tgt_class, key):
    # Get the output of last layer
    final_layer_out = getOutput(model, in_states, in_spikes, key)
    # final_layer_out = out_spikes[-1]
    pred = final_layer_out.sum(axis=0)

    target = jax.nn.one_hot(tgt_class, 27)
    loss = optax.softmax_cross_entropy(pred, target)
    return loss


def loss_fn(trainable, static, in_states, in_spikes, tgt_class, key):
    model = eqx.combine(trainable, static)
    keys = jax.random.split(key, in_spikes.shape[0])
    return jnp.mean(_loss_fn(model, in_states, in_spikes, tgt_class, keys))


loss_and_grad = eqx.filter_value_and_grad(loss_fn)


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


@eqx.filter_jit
def update_scan(carry, inputs, static_model=None, in_states=None, optim=None):
    diff_model, opt_state = carry

    model = eqx.combine(diff_model, static_model)

    in_spikes, tgt_class, key = inputs

    # Get gradients
    trainable, static = eqx.partition(
        model, eqx.is_array, is_leaf=lambda x: isinstance(x, StaticWrapper)
    )
    loss, grads = loss_and_grad(trainable, static, in_states, in_spikes, tgt_class, key)

    # Calculate parameter updates using the optimizer
    updates, opt_state = optim.update(grads, opt_state)

    # Update parameter PyTree with Equinox and optax
    model = eqx.apply_updates(model, updates)

    diff_model, static_model = eqx.partition(model, eqx.is_array)
    return (diff_model, opt_state), loss


def _objective(model_name, quant_bits, use_bias, trial, seed, method="Bruno"):
    trainset, testset, nb_outputs, nb_channels, nb_steps, time_step = loadBraille(
        2, 200
    )

    key = jrandom.key(seed)
    key, kmodel, kstate = jrandom.split(key, 3)

    model = define_model(kmodel, model_name, quant_bits, use_bias, trial, method)
    optim = optax.adamax(
        learning_rate=trial.suggest_float("lr", 1e-5, 1e-1, log=True), b1=0.9, b2=0.995
    )
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    initial_state = model.init_state(in_shape=(4,), key=kstate)
    total_accuracy = []
    total_loss = []

    # tracker = OptimizationTrajectoryTracker(lambda x, y, z: loss_fn(x, initial_state, y, z, jax.random.key(0)))

    pbar = trange(0, NBEPOCHS, leave=False)
    for e in pbar:
        key, epoch_key, train_key, test_key = jax.random.split(key, 4)
        x_train, y_train = shuffle(trainset, epoch_key, BATCHSIZE_TRAIN)

        eqx.tree_serialise_leaves(f"models/{method}/{e}.eqx", model)

        # tracker.record_step(model, testset[0], testset[1])

        # keys = jax.random.split(train_key, x_train.shape[0])

        # diff_model, static_model = eqx.partition(model, eqx.is_array)
        # (model, opt_state), loss_train = jax.lax.scan(partial(update_scan, static_model=static_model, in_states=initial_state, optim=optim), (diff_model, opt_state), (x_train, y_train, keys))
        # model = eqx.combine(diff_model, static_model)
        # loss_train = jnp.mean(loss_train)
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

    eqx.tree_serialise_leaves(f"models/{method}/{e}.eqx", model)
    # tracker.record_step(model, testset[0], testset[1])
    return total_accuracy


def get_param(model):
    flat = jnp.concatenate(
        [
            p.flatten()
            for p in jax.tree_util.tree_leaves(eqx.filter(model, eqx.is_array))
        ]
    )
    return flat


def unflatten_to_pytree(flat_array, reference_pytree):
    """Reconstruct PyTree from flattened array."""
    leaves = jax.tree_util.tree_leaves(eqx.filter(reference_pytree, eqx.is_array))

    # Split flat array according to shapes
    new_leaves = []
    idx = 0
    for leaf in leaves:
        size = leaf.size
        new_leaf = flat_array[idx : idx + size].reshape(leaf.shape)
        new_leaves.append(new_leaf)
        idx += size

    # Reconstruct PyTree
    tree_def = jax.tree_util.tree_structure(eqx.filter(reference_pytree, eqx.is_array))
    return jax.tree_util.tree_unflatten(tree_def, new_leaves)


def project_trajectory(direction1, direction2, parameters, reference_model):
    alpha_coords = []
    beta_coords = []

    for param in parameters:
        model = unflatten_to_pytree(param, reference_model)

        displacement = jax.tree_util.tree_map(
            lambda m, ref: m - ref if eqx.is_array(m) else 0, model, reference_model
        )

        alpha = sum(
            [
                jnp.sum(d * d1)
                for d, d1 in zip(
                    jax.tree_util.tree_leaves(eqx.filter(displacement, eqx.is_array)),
                    jax.tree_util.tree_leaves(eqx.filter(direction1, eqx.is_array)),
                )
            ]
        )

        beta = sum(
            [
                jnp.sum(d * d2)
                for d, d2 in zip(
                    jax.tree_util.tree_leaves(eqx.filter(displacement, eqx.is_array)),
                    jax.tree_util.tree_leaves(eqx.filter(direction2, eqx.is_array)),
                )
            ]
        )

        alpha_coords.append(alpha)
        beta_coords.append(beta)

    return jnp.array(alpha_coords), jnp.array(beta_coords)


def filter_normalized_directions(
    model,
    loss_fn: Callable,
    data,
    center_model=None,  # Optional: use a specific model as center (e.g., final model)
    n_points: int = 50,
    alpha_range: float = 1.0,
    beta_range: float = 1.0,
    seed: int = 42,
):
    import numpy as np

    key = jax.random.PRNGKey(seed)
    key1, key2 = jax.random.split(key)

    if center_model is None:
        center_model = model

    def generate_random_direction(key, reference_pytree):
        """Generate random direction matching the structure of the model."""

        def make_random(leaf, key):
            if eqx.is_array(leaf):
                return jax.random.normal(key, leaf.shape)
            return leaf

        keys = jax.random.split(
            key,
            len(jax.tree_util.tree_leaves(eqx.filter(reference_pytree, eqx.is_array))),
        )

        return jax.tree_util.tree_map(
            lambda leaf, k: make_random(leaf, k) if eqx.is_array(leaf) else leaf,
            reference_pytree,
            jax.tree_util.tree_unflatten(
                jax.tree_util.tree_structure(
                    eqx.filter(reference_pytree, eqx.is_array)
                ),
                keys,
            ),
            is_leaf=lambda x: eqx.is_array(x),
        )

    direction_alpha = generate_random_direction(key1, center_model)
    direction_beta = generate_random_direction(key2, center_model)

    def filter_normalize(direction_pytree, reference_pytree):
        """Normalize each layer/filter by the norm of the corresponding model parameters."""

        def normalize_leaf(d_leaf, r_leaf):
            if eqx.is_array(d_leaf) and eqx.is_array(r_leaf):
                # Get norm of reference parameter
                ref_norm = jnp.linalg.norm(r_leaf)
                # Get norm of direction
                dir_norm = jnp.linalg.norm(d_leaf)
                # Normalize direction and scale by reference norm
                if dir_norm > 0:
                    return (d_leaf / dir_norm) * ref_norm
                return d_leaf
            return d_leaf

        return jax.tree_util.tree_map(
            normalize_leaf,
            direction_pytree,
            reference_pytree,
            is_leaf=lambda x: eqx.is_array(x),
        )

    direction_alpha = filter_normalize(direction_alpha, center_model)
    direction_beta = filter_normalize(direction_beta, center_model)

    alpha_coords = jnp.linspace(-alpha_range, alpha_range, n_points)
    beta_coords = jnp.linspace(-beta_range, beta_range, n_points)
    alpha_grid, beta_grid = jnp.meshgrid(alpha_coords, beta_coords)

    def perturb_model(model, dir_alpha, dir_beta, alpha, beta):
        """Perturb model: model + alpha*dir_alpha + beta*dir_beta"""

        def add_perturbation(m_leaf, da_leaf, db_leaf):
            if eqx.is_array(m_leaf):
                return m_leaf + alpha * da_leaf + beta * db_leaf
            return m_leaf

        return jax.tree_util.tree_map(
            add_perturbation,
            model,
            dir_alpha,
            dir_beta,
            is_leaf=lambda x: eqx.is_array(x),
        )

    loss_grid = np.zeros((n_points, n_points))

    print("Computing loss landscape with filter normalization...")
    for i in range(n_points):
        for j in range(n_points):
            alpha = alpha_grid[i, j]
            beta = beta_grid[i, j]

            # Create perturbed model
            perturbed = perturb_model(
                center_model, direction_alpha, direction_beta, alpha, beta
            )

            # Evaluate loss
            loss = loss_fn(perturbed, data)
            loss_grid[i, j] = float(loss)

        if (i + 1) % 10 == 0:
            print(f"Progress: {i+1}/{n_points} rows completed")

    return alpha_grid, beta_grid, loss_grid


def unflatten_params(flat_params, reference_model):
    leaves, treedef = jax.tree_util.tree_flatten(
        eqx.filter(reference_model, eqx.is_array)
    )
    shapes = [leaf.shape for leaf in leaves]
    sizes = [leaf.size for leaf in leaves]

    new_leaves = []
    idx = 0
    for shape, size in zip(shapes, sizes):
        new_leaves.append(flat_params[idx : idx + size].reshape(shape))
        idx += size

    new_params = jax.tree_util.tree_unflatten(treedef, new_leaves)
    return eqx.combine(new_params, reference_model)


if __name__ == "__main__":
    import numpy as np
    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from tqdm import trange
    from matplotlib.lines import Line2D

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, default="FeLIF")
    parser.add_argument("-q", "--quantization", type=str, default="FP")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--plot", action="store_true")
    parser.add_argument("--compute", action="store_true")
    parser.add_argument("--bruno", action="store_true")

    args = parser.parse_args()
    print(args)    
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

    _, (x_test, y_test), _, _, _, _ = loadBraille(2, 200)

    if args.plot:
        print('Plotting')
        key = jax.random.key(0)
        key, kmodel = jax.random.split(key)
        bruno = define_model(kmodel, "FeLIF", "FP", False, study.best_trial, "Bruno")
        check = define_model(
            kmodel, "FeLIF", "FP", False, study.best_trial, "Checkpoint"
        )

        bruno_ref = bruno
        check_ref = check

        in_states = bruno.init_state((4,), key)

        loss_bruno = []
        loss_check = []
        params_bruno = []
        params_check = []
        for i in trange(NBEPOCHS):
            bruno = eqx.tree_deserialise_leaves(
                f"models/Bruno/{i}.eqx",
                bruno,
            )
            check = eqx.tree_deserialise_leaves(
                f"models2/Checkpoint/{i}.eqx",
                check,
            )

            params_bruno.append(get_param(bruno))
            params_check.append(get_param(check))

            loss_bruno.append(loss_fn(bruno, in_states, x_test, y_test, key))
            loss_check.append(loss_fn(check, in_states, x_test, y_test, key))

        params_bruno = jnp.stack(params_bruno)
        params_check = jnp.stack(params_check)
        params = jnp.concatenate([params_bruno, params_check])

        param_mean = jnp.mean(params, axis=0)
        centered = params - param_mean
        U, S, Vt = jnp.linalg.svd(centered, full_matrices=False)

        num_directions = 2
        U = U[:, :num_directions]
        S = S[:num_directions]
        Vt = Vt[:num_directions, :]

        directions = []
        for i in range(num_directions):
            pc = Vt[i]
            directions.append(pc)
        explained_variance = (S**2) / (params.shape[0] - 1)
        print(f"Direction variance: {explained_variance}")

        traj_A_2d = (jnp.stack([m for m in params_bruno]) - param_mean) @ jnp.stack(
            [directions[0], directions[1]]
        ).T
        traj_B_2d = (jnp.stack([m for m in params_check]) - param_mean) @ jnp.stack(
            [directions[0], directions[1]]
        ).T

        all_2d = jnp.concatenate([traj_A_2d, traj_B_2d], axis=0)
        alpha_min, alpha_max = all_2d[:, 0].min(), all_2d[:, 0].max()
        beta_min, beta_max = all_2d[:, 1].min(), all_2d[:, 1].max()

        margin = 0.2
        alpha_range = (alpha_max - alpha_min) * margin
        beta_range = (beta_max - beta_min) * margin

        n_points = 2
        alpha_coords = jnp.linspace(
            alpha_min - alpha_range, alpha_max + alpha_range, n_points
        )
        beta_coords = jnp.linspace(
            beta_min - beta_range, beta_max + beta_range, n_points
        )
        alpha_grid, beta_grid = jnp.meshgrid(alpha_coords, beta_coords)

        loss_grid = np.zeros((n_points, n_points))

        print("Computing loss landscape...")
        for i in range(n_points):
            for j in range(n_points):
                flat_params = (
                    param_mean
                    + alpha_grid[i, j] * directions[0]
                    + beta_grid[i, j] * directions[1]
                )
                model = unflatten_params(flat_params, bruno_ref)
                loss_grid[i, j] = float(
                    loss_fn(model, in_states, x_test, y_test, jax.random.key(0))
                )

                if (j + 1) % 10 == 0:
                    print(f"Progress: {i+1}/{n_points}-{j+1}/{n_points}")

        losses_A = np.array(
            [
                float(
                    loss_fn(
                        unflatten_params(m, bruno_ref),
                        in_states,
                        x_test,
                        y_test,
                        jax.random.key(0),
                    )
                )
                for m in params_bruno
            ]
        )
        losses_B = np.array(
            [
                float(
                    loss_fn(
                        unflatten_params(m, check_ref),
                        in_states,
                        x_test,
                        y_test,
                        jax.random.key(0),
                    )
                )
                for m in params_check
            ]
        )

        with mpl.style.context("boilerplot.ieeetran"):
            fig, axes = plt.subplots(1, 2, figsize=(6.9, 2.3), dpi=200)
            ax = axes[0]
            levels = np.linspace(loss_grid.min(), loss_grid.max(), 20)
            contour = ax.contourf(
                alpha_grid, beta_grid, loss_grid, levels=levels, cmap="viridis", alpha=0.7
            )
            ax.contour(
                alpha_grid,
                beta_grid,
                loss_grid,
                levels=levels,
                colors="black",
                alpha=0.2,
                linewidths=0.5,
            )

            # Plot trajectory A
            ax.plot(
                traj_A_2d[:, 0],
                traj_A_2d[:, 1],
                "-",
                color="#1b9e77",
                marker="o",
                markersize=3,
                linewidth=2,
                alpha=0.8,
                # label="Bruno",
            )
            ax.plot(traj_A_2d[0, 0], traj_A_2d[0, 1], "o", color="#1b9e77", markersize=6)
            ax.plot(traj_A_2d[-1, 0], traj_A_2d[-1, 1], "*", color="#1b9e77", markersize=9)

            # Plot trajectory B
            ax.plot(
                traj_B_2d[:, 0],
                traj_B_2d[:, 1],
                "--",
                color="#d95f02",
                linewidth=2,
                alpha=0.8,
                # label="Checkpoints",
            )
            ax.plot(traj_B_2d[0, 0], traj_B_2d[0, 1], "o", color="#d95f02", markersize=6, label='Start')
            ax.plot(traj_B_2d[-1, 0], traj_B_2d[-1, 1], "*", color="#d95f02", markersize=9, label='End')

            plt.colorbar(contour, ax=ax, label="Loss")
            ax.set_xlabel(f"PC1 ({explained_variance[0]:.1%} var)")
            ax.set_ylabel(f"PC2 ({explained_variance[1]:.1%} var)")
            ax.set_title("Loss Landscape with Bruno and Checkpoints")
            ax.legend(loc="best")
            # ax.grid(True, alpha=0.3)

            # Right: Loss over time comparison
            ax = axes[1]
            ax.plot(
                losses_A, "-", linewidth=2, color="#1b9e77", marker="o", markersize=3, label="Bruno"
            )
            ax.plot(
                losses_B, "--", linewidth=2, color="#d95f02", markersize=3, label="Checkpoints"
            )
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Loss")
            ax.set_title("Loss Comparison Over Training")
            ax.legend()
            ax.grid(True, alpha=0.3)

            # plt.tight_layout()
            plt.savefig("bruno_checkpoint_comparison.pdf")

    elif args.compute:
        if args.bruno:
            print('Running bruno')
            accuracy_bruno = objective(study.best_trial, args.seed, method="Bruno")
        else:
            print('Running checkpoint')
            accuracy_checkpoint = objective(
                study.best_trial, args.seed, method="Checkpoint"
            )
