import json
import time

import optuna
from mpi4py import MPI
from optuna.storages import JournalStorage
from optuna.storages.journal import JournalFileBackend


def objective(
    trial,
    num_partitions=10,
    alpha_dirichlet=100,
    model="lif",
    rounds=1,
    epochs=1,
    variability=0.1,
):
    num_hidden = trial.suggest_int("num_hidden", 3, 10)
    num_hidden = int(2**num_hidden)
    gain = trial.suggest_float("gain", 0, 1)  # 0.331672689117444
    bias = trial.suggest_float("bias", 0, 1)  # 0.4475318813798862

    if model == "lif":
        beta = trial.suggest_float("beta", 0, 1)
        threshold = trial.suggest_float("threshold", 0, 5)

        config = json.dumps(
            {
                "rounds": rounds,
                "epochs": epochs,
                "model": model,
                "num_hidden": num_hidden,
                "alpha_dirichlet": alpha_dirichlet,
                "gain": gain,
                "bias": bias,
                "beta": beta,
                "threshold": threshold,
            }
        )

    elif model == "felif":
        P_s = trial.suggest_float("P_s", 0, 100)  # 76.07685578175987
        alpha = trial.suggest_float("alpha", 0, 1)  # 0.29208951288980767
        beta = trial.suggest_float("beta", 0, 1)  # 0.08717596748142253
        tau_p = trial.suggest_float("tau_p", 1, 100, log=True)  # 70.07106568069094
        tau_m = trial.suggest_float("tau_m", 1, 100, log=True)  # 17.08904771900422
        threshold = trial.suggest_float("threshold", 0, 5)  # 1.2255335187426544

        config = json.dumps(
            {
                "rounds": rounds,
                "epochs": epochs,
                "model": model,
                "num_hidden": num_hidden,
                "alpha_dirichlet": alpha_dirichlet,
                "gain": gain,
                "bias": bias,
                "P_s": P_s,
                "alpha": alpha,
                "beta": beta,
                "tau_p": tau_p,
                "tau_m": tau_m,
                "threshold": threshold,
                "variability": variability,
            }
        )
    elif model == "heracles":
        idsc = trial.suggest_float("idsc", 0, 100e-12)  # 76.07685578175987
        threshold = trial.suggest_float("threshold", 0, 4)  # 1.2255335187426544

        config = json.dumps(
            {
                "rounds": rounds,
                "epochs": epochs,
                "model": model,
                "num_hidden": num_hidden,
                "alpha_dirichlet": alpha_dirichlet,
                "gain": gain,
                "bias": bias,
                "idsc": idsc,
                "threshold": threshold,
                "variability": variability,
            }
        )
    else:
        raise Exception(f"Model {model} not implemented")

    # Spawn 4 MPI workers running mpi_worker.py
    print("Spawning")
    comm_child = MPI.COMM_SELF.Spawn(
        command="python3", args=["FL_worker.py"], maxprocs=num_partitions
    )

    # Send the same config to all workers
    for i in range(num_partitions):
        comm_child.send(config, dest=i, tag=0)

    # Collect intermediate reports
    final_metrics = 0
    pruned = False
    while True:
        status = MPI.Status()
        msg = comm_child.recv(source=MPI.ANY_SOURCE, tag=MPI.ANY_TAG, status=status)

        if status.tag == 1:  # intermediate report
            data = json.loads(msg)
            trial.report(data["val_acc"], step=data["round"])
            print(
                f"[Trial {trial.number}] Round {data['round']} → {data['val_acc']:.4f}"
            )

            if trial.should_prune():
                print("Parent: pruning trial")
                # send stop signal to child
                comm_child.send("STOP", dest=0, tag=99)
                pruned = True
        elif status.tag == 2:  # final result
            final_metrics = float(msg)
            break
    comm_child.Barrier()
    comm_child.Disconnect()
    # gc.collect()
    time.sleep(2)  # Allow MPI runtime to cleanup

    if pruned:
        raise optuna.TrialPruned()
    return final_metrics


if __name__ == "__main__":
    import argparse
    from functools import partial

    parser = argparse.ArgumentParser()
    parser.add_argument("--partitions", type=int, default=10)
    parser.add_argument("--alpha", type=float, default=100)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--variability", type=float, default=0.1)
    parser.add_argument("--model", choices=["lif", "felif", "heracles"], default="lif")
    args = parser.parse_args()

    storage = JournalStorage(
        JournalFileBackend(
            f"federated_{args.model}-{args.rounds}-{args.epochs}-var{args.variability}.log"
        )
    )
    study = optuna.create_study(
        study_name=f"{args.partitions} partitions - alpha {args.alpha}",
        direction="maximize",
        sampler=optuna.samplers.CmaEsSampler(),
        storage=storage,
        load_if_exists=True,
    )
    study.optimize(
        partial(
            objective,
            num_partitions=args.partitions,
            alpha_dirichlet=args.alpha,
            model=args.model,
            rounds=args.rounds,
            epochs=args.epochs,
        ),
        n_trials=30,
    )
    print("Best trial:", study.best_trial)
