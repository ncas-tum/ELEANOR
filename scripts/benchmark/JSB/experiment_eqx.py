import jax
import numpy as np
import optax
import optuna
import equinox as eqx
import sklearn
import jax.numpy as jnp
import jax.random as jrandom
import matplotlib as mpl
import matplotlib.pyplot as plt
from tqdm import tqdm, trange
from models import RNN, SNUCell
from datasets import JSBChorales
from torch.utils.tensorboard import SummaryWriter

from eleanor.models.jax.variability import D2DVar, StaticWrapper

SEED = 42
NBEPOCHS = 100
WARMUP = 5


def is_trainable(x):
    return (
        eqx.is_array(x)
        and not isinstance(x, StaticWrapper)
        and not isinstance(x, D2DVar)
    )


@eqx.filter_jit
def loss_fn(trainable, static, x, y, key):
    model = eqx.combine(trainable, static)
    pred_y = model(x, key=key)

    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_y[WARMUP:], y[WARMUP:]))
    # loss = jnp.mean(jnp.sum(optax.sigmoid_binary_cross_entropy(pred_y, y), axis=1))
    # pred_y = jax.nn.sigmoid(pred_y)
    # loss = -(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))
    # loss = jnp.sum(loss)
    # loss = jnp.mean(jnp.sum(optax.losses.sigmoid_focal_loss(pred_y, y), axis=1))

    return loss


@eqx.filter_jit
def loss_pred(model, x, y, key):
    pred_y = model(x, key=key)

    loss = jnp.mean(optax.sigmoid_binary_cross_entropy(pred_y[WARMUP:], y[WARMUP:]))
    # loss = jnp.mean(jnp.sum(optax.sigmoid_binary_cross_entropy(pred_y, y), axis=1))
    # loss = -(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))
    # loss = jnp.mean(jnp.sum(optax.losses.sigmoid_focal_loss(pred_y, y), axis=1))
    # pred_y = jax.nn.sigmoid(pred_y)
    # loss = -(y * jnp.log(pred_y) + (1 - y) * jnp.log(1 - pred_y))
    # loss = jnp.sum(loss)

    return loss, pred_y


loss_and_grad = eqx.filter_value_and_grad(loss_fn)


@eqx.filter_jit
def update(model, optim, opt_state, transform_state, in_spikes, tgt_class, key):
    # Get gradients
    trainable, static = eqx.partition(
        model, eqx.is_array, is_leaf=lambda x: isinstance(x, StaticWrapper)
    )
    loss, grads = loss_and_grad(trainable, static, in_spikes, tgt_class, key)

    # Calculate parameter updates using the optimizer
    updates, opt_state = optim.update(grads, opt_state)
    updates = optax.tree.scale(transform_state.scale, updates)

    # Update parameter PyTree with Equinox and optax
    model = eqx.apply_updates(model, updates)
    return model, opt_state, transform_state, loss


def test(model_name, variability, fix, params):
    key = jrandom.key(SEED)

    print("Loading data...")
    key, subkey = jrandom.split(key)
    testset = JSBChorales("data/JSB_test", shuffle=False)

    print("Creating model...")
    key, subkey = jrandom.split(key)
    model = RNN(model_name, 54, 54, 150, variability, params, key=subkey)

    inference_model = eqx.nn.inference_mode(model)
    for i, (x, y) in enumerate(testset):
        key, subkey = jrandom.split(key)
        (pred_y, spk, v, p) = inference_model.record(x, key=subkey)
        pred_y = jax.nn.sigmoid(pred_y)
        print(v)
        with mpl.style.context("boilerplot.ieeetran"):
            fig_spikes, ax = plt.subplots(1, 3, figsize=(6.9, 2.3), dpi=200)
            ax[0].imshow(spk.T, aspect="auto")
            ax[1].imshow(pred_y.T)
            ax[2].plot(v[:, 0])
            ax[2].plot(p[:, 0])
            fig_spikes.savefig(f"plots/{i}.png")
        break


def train(model_name, variability, fix, params, trial=None, writer=None):
    key = jrandom.key(SEED)

    print("Loading data...")
    key, subkey = jrandom.split(key)
    trainset = JSBChorales("data/JSB_train", shuffle=True, key=subkey)
    testset = JSBChorales("data/JSB_test", shuffle=False)

    print("Creating model...")
    key, subkey = jrandom.split(key)
    model = RNN(model_name, 54, 54, 150, variability, params, key=subkey)

    print("Optimizer...")
    if params["optimizer"] == "adamax":
        optim = optax.adamax(learning_rate=params["lr"], b1=0.9, b2=0.995)
    elif params["optimizer"] == "sgd":
        optim = optax.sgd(learning_rate=params["lr"])
    opt_state = optim.init(eqx.filter(model, eqx.is_inexact_array))

    transform = optax.contrib.reduce_on_plateau(
        patience=10,
        cooldown=0,
        factor=0.9,
        rtol=1e-4,
        accumulation_size=1,
    )
    transform_state = transform.init(eqx.filter(model, eqx.is_inexact_array))
    # transform_state = None

    if fix:
        update_lr = lambda updates, state, value: state
    else:
        update_lr = lambda updates, state, value: transform.update(
            updates=updates, state=state, value=value
        )[1]

    key, subkey = jrandom.split(key)
    print("Initialize training loop")
    pbar = trange(0, NBEPOCHS)
    for epoch in pbar:
        train_loss = []

        for i, (x, y) in enumerate(trainset):
            key, subkey = jrandom.split(key)
            model, opt_state, transform_state, loss = update(
                model, optim, opt_state, transform_state, x, y, subkey
            )
            train_loss.append(loss)
        train_loss = jnp.stack(train_loss).mean().item()

        test_loss = []
        labels = []
        predictions = []
        inference_model = eqx.nn.inference_mode(model)
        for i, (x, y) in enumerate(testset):
            key, subkey = jrandom.split(key)
            loss, pred_y = loss_pred(inference_model, x, y, subkey)
            pred_y = jax.nn.sigmoid(pred_y)

            test_loss.append(loss)
            labels.append(y)
            predictions.append(pred_y)
        test_loss = jnp.stack(test_loss).mean()

        transform_state = update_lr(
            eqx.filter(model, eqx.is_inexact_array), transform_state, test_loss
        )
        test_loss = test_loss.item()

        labels = np.concatenate(labels)
        predictions = np.concatenate(predictions)

        # active_labels = np.where(labels.sum(axis=0) > 0)[0]
        # f1_micro = sklearn.metrics.f1_score(
        #     labels, predictions > 0.5, average="micro"
        # )
        # f1_macro = sklearn.metrics.f1_score(
        #     labels[:,active_labels], predictions[:,active_labels] > 0.5, average="macro"
        # )
        # f1_weighted = sklearn.metrics.f1_score(
        #     labels, predictions > 0.5, average="weighted", zero_division=0
        # )
        # f1_samples = sklearn.metrics.f1_score(
        #     labels, predictions > 0.5, average="samples"
        # )

        pbar.set_postfix({"Train loss": train_loss, "Test loss": test_loss})

        if writer is not None:
            writer.add_scalar("lr", transform_state.scale.item(), epoch)
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", test_loss, epoch)
            # writer.add_scalar("F1 (micro)", f1_micro, epoch)
            # writer.add_scalar("F1 (macro)", f1_macro, epoch)
            # writer.add_scalar("F1 (weighted)", f1_weighted, epoch)
            # writer.add_scalar("F1 (samples)", f1_samples, epoch)
            writer.add_pr_curve("pr_curve", labels, predictions, epoch)

            pred_y, spk = model.record(x, key=key)
            pred_y = jax.nn.sigmoid(pred_y)
            with mpl.style.context("boilerplot.ieeetran"):
                fig_spikes, ax = plt.subplots(1, 2, figsize=(3.45, 2.3), dpi=200)
                ax[0].imshow(spk.T, aspect="auto")
                ax[1].imshow(pred_y.T)
            writer.add_figure("Raster", fig_spikes, epoch)

            # if epoch % 10 == 0:
            writer.add_image("Target", np.asarray(y), epoch, dataformats="WH")
            writer.add_image(
                "Prediction", np.asarray(pred_y) > 0.5, epoch, dataformats="WH"
            )

        if trial is not None:
            trial.report(loss, epoch)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return loss


def _objective(model_name, variability, fix, trial):
    params = {
        "lr": trial.suggest_float("lr", 1e-5, 1e-1, log=True),
        "optimizer": trial.suggest_categorical("optimizer", ["sgd", "adamax"]),
    }

    if model_name == "FeLIF":
        params["I_dsc"] = trial.suggest_float("I_dsc", 10e-12, 100e-12, log=False)
        params["threshold"] = trial.suggest_float("threshold", 2.5, 3.5, log=False)
        params["paramScale"] = trial.suggest_int("paramScale", 5, 12)
    elif model_name == "LIF" or model_name == "SNU":
        params["decay"] = trial.suggest_float("decay", 0.1, 1.0, log=False)
        params["threshold"] = trial.suggest_float("threshold", 0.1, 1.0, log=False)

    print(params)
    return train(model_name, variability, fix, params, trial)


if __name__ == "__main__":
    import argparse
    from functools import partial

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", type=str, required=True)
    parser.add_argument("-v", "--variability", type=float, default=0.0)

    parser.add_argument("--optimizer", choices=["adamax", "sgd"], default="adamax")
    parser.add_argument("--lr", type=float, default=0.005)

    parser.add_argument("--I_dsc", type=float, default=90e-12)
    parser.add_argument("--threshold", type=float, default=2.0)
    parser.add_argument("--paramScale", type=int, default=10)

    parser.add_argument("--decay", type=float, default=0.4)
    parser.add_argument("--thr", type=float, default=1.0)

    parser.add_argument("--hpo", action="store_true")
    parser.add_argument("--best", action="store_true")
    parser.add_argument("--fix", action="store_true")
    parser.add_argument("--test", action="store_true")

    args = parser.parse_args()
    if args.fix:
        study_name = (
            f"{args.model} {args.variability}"
            if args.model == "FeLIF"
            else f"{args.model}"
        )
    else:
        study_name = (
            f"{args.model} {args.variability} - Reduce on plateau ({args.lr})"
            if args.model == "FeLIF"
            else f"{args.model} - Reduce on plateau ({args.lr})"
        )
    storage = optuna.storages.JournalStorage(
        optuna.storages.journal.JournalFileBackend("./jsb-bruno.log")
    )
    if args.hpo:
        objective = partial(_objective, args.model, args.variability, args.fix)
        study = optuna.create_study(
            storage=storage,
            study_name=study_name,
            direction=optuna.study.StudyDirection.MINIMIZE,
            pruner=optuna.pruners.HyperbandPruner(
                min_resource=1, max_resource=NBEPOCHS, reduction_factor=3
            ),
            load_if_exists=True,
        )
        study.optimize(objective, n_trials=100)  # 50 trials for warmup
    elif args.best:
        writer = SummaryWriter(log_dir=f"bruno/{study_name}")
        study = optuna.load_study(
            storage=storage,
            study_name=study_name,
        )
        train(args.model, args.variability, args.fix, study.best_params, writer=writer)
    elif args.test:
        test(args.model, args.variability, args.fix, vars(args))
    else:
        writer = SummaryWriter(log_dir=f"bruno/{study_name}")
        train(args.model, args.variability, args.fix, vars(args), writer=writer)
