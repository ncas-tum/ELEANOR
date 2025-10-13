import json
import time

import numpy as np
import torch
from mpi4py import MPI
from tasks.braille import (
    test,
    train,
    load_lif,
    load_data,
    load_felif,
    get_weights,
    set_weights,
    load_heracles,
)

comm_parent = MPI.Comm.Get_parent()
comm = MPI.COMM_WORLD
size = MPI.COMM_WORLD.Get_size()
rank = MPI.COMM_WORLD.Get_rank()
print(f"[{rank}] Connected to parent communicator.")

np.random.seed(123 + rank)  # for reproducibility
torch.manual_seed(123 + rank)

SERVER = 0  # rank 0 is the server
device = "cpu"  # 'cuda' if torch.cuda.is_available() else 'cpu'

# Receive hyperparameter config

config_str = comm_parent.recv(source=0, tag=0)
config = json.loads(config_str)

trainloader, valloader, testloader = load_data(
    partition_id=rank, num_partitions=size, split=True, alpha=config["alpha_dirichlet"]
)

if config["model"] == "lif":
    model = load_lif(
        use_bias=False,
        num_hidden=config["num_hidden"],
        threshold=config["threshold"],
        gain=config["gain"],
        bias=config["bias"],
        beta=config["beta"],
    )
elif config["model"] == "felif":
    model = load_felif(
        use_bias=False,
        num_hidden=config["num_hidden"],
        P_s=config["P_s"],
        alpha=config["alpha"],
        beta=config["beta"],
        tau_p=config["tau_p"],
        tau_m=config["tau_m"],
        threshold=config["threshold"],
        gain=config["gain"],
        bias=config["bias"],
        variability=config["variability"],
    )
elif config["model"] == "heracles":
    model = load_heracles(
        use_bias=False,
        num_hidden=config["num_hidden"],
        I_dsc=config["idsc"],
        threshold=config["threshold"],
        gain=config["gain"],
        bias=config["bias"],
        variability=config["variability"],
    )
else:
    raise Exception(f"Model {config['model']} not implemented")


# Simulate federated training
nrounds = config["rounds"]
for round in range(nrounds):
    # Check if parent sent a prune signal
    stop = False
    if rank == 0:
        status = MPI.Status()
        if comm_parent.Iprobe(source=0, tag=99, status=status):
            _ = comm_parent.recv(source=0, tag=99)
            stop = True
    # Broadcast stop signal to all child ranks
    stop = comm.bcast(stop, root=0)
    if stop:
        print(f"[Rank {rank}] stopping at round {round}")
        break

    # broadcast server's model
    start_time = time.process_time()
    weights = get_weights(model)
    weights = comm.bcast(weights, root=0)
    set_weights(model, weights)

    loss_train = train(model, trainloader, config["epochs"], device)
    loss_val, acc_val = test(model, valloader, device)
    end_time = time.process_time()
    training_time = end_time - start_time

    # federate step
    data = [
        len(trainloader.dataset),
        get_weights(model),
        loss_train,
        loss_val,
        acc_val,
        training_time,
    ]

    data = comm.gather(data, root=0)
    if rank == SERVER:
        weights = [np.zeros(layer.shape) for layer in data[0][1]]

        for i in range(len(weights)):
            N = 0.0
            for n, w, _, _, _, _ in data:
                weights[i] += n * w[i]
                N += n

            weights[i] /= N

        N = 0.0
        loss_train = 0
        loss_val = 0
        val_acc = 0
        time_train = 0
        for n, _, lt, lv, av, t in data:
            time_train += n * t
            loss_train += n * lt
            loss_val += n * lv
            val_acc += n * av
            N += n

        time_train /= N
        loss_train /= N
        loss_val /= N
        val_acc /= N

        set_weights(model, weights)

        # Report intermediate result
        msg = json.dumps({"round": round, "val_acc": val_acc})
        comm_parent.send(msg, dest=0, tag=1)

if rank == SERVER:
    # Final result (simulate test accuracy)
    comm_parent.send(str(val_acc), dest=0, tag=2)

comm_parent.Barrier()
comm_parent.Disconnect()
MPI.Finalize()
