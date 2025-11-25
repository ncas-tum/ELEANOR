import sys
import time
import argparse

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
    update_variability,
)
from torch.utils.tensorboard import SummaryWriter

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()

np.random.seed(123 + rank)  # for reproducibility
torch.manual_seed(123 + rank)

SERVER = 0  # rank 0 is the server


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--num_hidden", type=int, default=128)
    parser.add_argument("--gain", type=float, default=0.331672689117444)
    parser.add_argument("--bias", type=float, default=0.4475318813798862)
    parser.add_argument("--alpha-dirichlet", type=float, default=100)
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--cuda", action="store_true")

    subparsers = parser.add_subparsers(
        dest="model", required=True, help="Choose the model to run"
    )

    # ----- LIF -----
    lif_parser = subparsers.add_parser("lif", help="Run LIF neurons")
    lif_parser.add_argument("--threshold", type=float, default=1.0)
    lif_parser.add_argument("--beta", type=float, default=0.9)

    # ----- FeLIF -----
    felif_parser = subparsers.add_parser("felif", help="Run FeLIF neurons")
    felif_parser.add_argument("--P_s", type=float, default=76.07685578175987)
    felif_parser.add_argument("--alpha", type=float, default=1.29208951288980767)
    felif_parser.add_argument("--beta", type=float, default=0.08717596748142253)
    felif_parser.add_argument("--tau_p", type=float, default=1.07106568069094e-3)
    felif_parser.add_argument("--tau_m", type=float, default=17.08904771900422e-3)
    felif_parser.add_argument("--threshold", type=float, default=1.0)
    felif_parser.add_argument("--variability", type=float, default=0.01)

    # ----- Heracles -----
    heracles_parser = subparsers.add_parser("heracles", help="Run Heracles neurons")
    heracles_parser.add_argument("--idsc", type=float, default=100e-12)
    heracles_parser.add_argument("--threshold", type=float, default=2.5)
    heracles_parser.add_argument("--variability", type=float, default=0.1)

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() and args.cuda else "cpu"

    trainloader, valloader, testloader = load_data(
        partition_id=rank,
        num_partitions=size,
        split=True,
        alpha=args.alpha_dirichlet,
        upsample=2,
    )

    if args.model == "lif":
        model = load_lif(
            use_bias=False,
            num_hidden=args.num_hidden,
            threshold=args.threshold,
            gain=args.gain,
            bias=args.bias,
            beta=args.beta,
        )
    elif args.model == "felif":
        model = load_felif(
            use_bias=False,
            num_hidden=args.num_hidden,
            P_s=args.P_s,
            alpha=args.alpha,
            beta=args.beta,
            tau_p=args.tau_p,
            tau_m=args.tau_m,
            threshold=args.threshold,
            gain=args.gain,
            bias=args.bias,
            variability=args.variability,
        )
    elif args.model == "heracles":
        model = load_heracles(
            use_bias=False,
            num_hidden=args.num_hidden,
            I_dsc=args.idsc,
            threshold=args.threshold,
            gain=args.gain,
            bias=args.bias,
            variability=args.variability,
        )
    else:
        raise Exception(f"Model {args.model} not implemented")

    if rank == SERVER and args.log:
        writer = SummaryWriter()

    nepoch = args.rounds
    for e in range(nepoch):
        if rank == SERVER and args.log:
            print("Server's Epoch: " + str(e + 1))
            sys.stdout.flush()

        # broadcast server's model
        weights = get_weights(model)

        start_time = time.process_time()
        weights = comm.bcast(weights, root=0)
        set_weights(model, weights)

        loss_train = train(model, trainloader, args.epochs, device)
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
        if rank == SERVER:
            start_time = time.process_time()
        data = comm.gather(data, root=0)
        if rank == SERVER:
            weights = [np.zeros(l.shape) for l in data[0][1]]

            for i in range(len(weights)):
                N = 0.0
                for n, w, _, _, _, _ in data:
                    weights[i] += n * w[i]
                    N += n

                weights[i] /= N

            end_time = time.process_time()

            N = 0.0
            loss_train = 0
            loss_val = 0
            acc_val = 0
            time_train = 0
            for n, _, lt, lv, av, t in data:
                time_train += n * t
                loss_train += n * lt
                loss_val += n * lv
                acc_val += n * av
                N += n

            time_train /= N
            loss_train /= N
            loss_val /= N
            acc_val /= N
            if args.log:
                print(f"Acuracy on epoch {e + 1}: {acc_val} in {time_train}")
                writer.add_scalar("Time/train", loss_train, e)
                writer.add_scalar("Loss/train", loss_train, e)
                writer.add_scalar("Loss/val", loss_val, e)
                writer.add_scalar("Accuracy/val", acc_val, e)

            set_weights(model, weights)

            loss, accuracy = test(model, testloader, device)
            if args.log:
                print("Test loss:", loss)
                print("Test accuracy:", accuracy)
                writer.add_scalar("Loss/test", loss, e)
                writer.add_scalar("Accuracy/test", accuracy, e)
                writer.flush()

    if rank == SERVER:
        print(f"Accuracy {accuracy}")
        if args.model != "lif":
            update_variability(model, (1, 27))  # Batch size 1
            loss, accuracy = test(model, testloader, device)
            print(f"Accuracy different system {accuracy}")
        args_dict = vars(args)
        args_dict.pop("log")
        writer.add_hparams(
            args_dict, {"hparam/accuracy": accuracy, "hparam/loss": loss}
        )
        writer.close()


if __name__ == "__main__":
    main()
