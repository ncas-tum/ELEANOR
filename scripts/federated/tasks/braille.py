from collections import OrderedDict

import numpy as np
import torch
import snntorch as snn
import torch.nn as nn
import lightning as L
from models import EncodingLayer
from dataset import create_dataset
from torch.utils.data import DataLoader
from flwr_datasets.partitioner import DirichletPartitioner

from eleanor.models.torch import FeLIF, Heracles

# fds = None
# partitioner = None  # Cache FederatedDataset


class LitBraille(L.LightningModule):
    def __init__(self, model, *args, **kwargs):
        super().__init__()
        if model == "lif":
            self.model = load_lif(*args, **kwargs)
        elif model == "felif":
            self.model = load_felif(*args, **kwargs)
        elif model == "heracles":
            self.model = load_heracles(*args, **kwargs)
        self.criterion = torch.nn.CrossEntropyLoss()

    def configure_optimizers(self):
        optimizer = torch.optim.Adamax(self.parameters(), lr=1e-3, betas=(0.9, 0.995))
        return optimizer

    def training_step(self, batch, batch_idx):
        in_spk = batch["data"]
        labels = batch["label"]
        reset(self.model)

        spk_rec = []
        for t in range(in_spk.shape[1]):
            spk_out = self.model(in_spk[:, t])
            spk_rec.append(spk_out)
        spk_rec = torch.stack(spk_rec)

        pred = spk_rec.sum(axis=0)
        loss = self.criterion(pred, labels)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        in_spk = batch["data"]
        labels = batch["label"]
        reset(self.model)

        spk_rec = []
        for t in range(in_spk.shape[1]):
            spk_out = self.model(in_spk[:, t])
            spk_rec.append(spk_out)
        spk_rec = torch.stack(spk_rec)

        pred = spk_rec.sum(axis=0)
        loss = self.criterion(pred, labels)  # + 1e-2 * torch.mean(spk_rec)
        correct = (torch.max(pred, 1)[1] == labels).sum().item()
        accuracy = correct / len(batch["label"])
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", accuracy, prog_bar=True)

    def test_step(self, batch, batch_idx):
        in_spk = batch["data"]
        labels = batch["label"]
        reset(self.model)

        spk_rec = []
        for t in range(in_spk.shape[1]):
            spk_out = self.model(in_spk[:, t])
            spk_rec.append(spk_out)
        spk_rec = torch.stack(spk_rec)

        pred = spk_rec.sum(axis=0)
        loss = self.criterion(pred, labels)
        correct = (torch.max(pred, 1)[1] == labels).sum().item()
        accuracy = correct / len(batch["label"])
        self.log("test_loss", loss)
        self.log("test_acc", accuracy)


def load_data(
    partition_id: int,
    num_partitions: int,
    split: bool = True,
    batch_size: int = 128,
    alpha: int = 100,
    upsample: int = 2,
):
    # global partitioner
    # global fds
    fds = None
    partitioner = None  # Cache FederatedDataset

    if fds is None:
        fds, _, _, _, _ = create_dataset(upsample, 200)

    if partitioner is None:
        # partitioner = IidPartitioner(num_partitions=num_partitions)
        partitioner = DirichletPartitioner(
            num_partitions=num_partitions, partition_by="label", alpha=alpha
        )
        partitioner.dataset = fds["train"]

    partition = partitioner.load_partition(partition_id)
    fds.set_format("torch")
    partition.set_format("torch")

    if split:
        partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
        trainloader = DataLoader(
            partition_train_test["train"], batch_size=batch_size, shuffle=True
        )
        valloader = DataLoader(partition_train_test["test"], batch_size=batch_size)
        testloader = DataLoader(fds["test"], batch_size=batch_size)

        return trainloader, valloader, testloader
    else:
        trainloader = DataLoader(partition, batch_size=batch_size, shuffle=True)
        testloader = DataLoader(fds["test"], batch_size=batch_size)

        return trainloader, testloader


def load_felif(
    use_bias: bool = False,
    num_hidden: int = 256,
    P_s: float = 100,
    alpha: float = 1.0,
    beta: float = 0.0,
    tau_p: float = 2,
    tau_m: float = 20,
    threshold: float = 1.0,
    gain: float = 0.18436009935019085,
    bias: float = 1.0,
    dt=1,
    variability: float = 0.0,
):
    enc_gain = torch.randn(num_hidden) * gain
    enc_bias = torch.randn(num_hidden) * bias

    model = nn.Sequential(
        EncodingLayer(enc_gain, enc_bias, num_hidden // 4),
        snn.Leaky(0.95, threshold=1, learn_beta=False, init_hidden=True),
        nn.Linear(num_hidden, 27, bias=use_bias),
        nn.BatchNorm1d(27),
        FeLIF(
            tau_p,
            tau_m,
            P_s,
            alpha,
            beta,
            dt,
            threshold=threshold,
            variability=variability,
            init_hidden=True,
        ),
    )

    return model


def load_lif(
    use_bias: bool = False,
    num_hidden: int = 256,
    threshold: float = 1,
    gain: float = 0.18436009935019085,
    bias: float = 1.0,
    beta: float = 0.9,
):
    enc_gain = torch.randn(num_hidden) * gain
    enc_bias = torch.randn(num_hidden) * bias

    model = nn.Sequential(
        EncodingLayer(enc_gain, enc_bias, num_hidden // 4),
        snn.Leaky(0.95, threshold=1, learn_beta=False, init_hidden=True),
        nn.Linear(num_hidden, 27, bias=use_bias),
        snn.Leaky(beta, threshold=threshold, learn_beta=True, init_hidden=True),
    )

    return model


def load_heracles(
    use_bias: bool = False,
    num_hidden: int = 256,
    threshold: float = 2.5,
    gain: float = 0.18436009935019085,
    bias: float = 1.0,
    I_dsc: float = 10e-12,
    dt=1e-3,
    variability: float = 0.0,
):
    enc_gain = torch.randn(num_hidden) * gain
    enc_bias = torch.randn(num_hidden) * bias

    model = nn.Sequential(
        EncodingLayer(enc_gain, enc_bias, num_hidden // 4),
        nn.BatchNorm1d(num_hidden),
        snn.Leaky(np.exp(-dt / 20e-3), threshold=1, learn_beta=False, init_hidden=True),
        nn.Linear(num_hidden, 27, bias=use_bias),
        # Gain(2e-9),
        nn.BatchNorm1d(27),
        Heracles(
            I_dsc=I_dsc,
            threshold=threshold,
            paramsScale=1e9,
            variability=variability,
            dt=dt,
            init_hidden=True,
            nsteps=1000,
        ),
    )

    return model


def update_variability(model, shape):
    model[-1].update_variability(shape)


def reset(model):
    for m in model.modules():
        if isinstance(m, snn.SpikingNeuron):
            m.reset_mem()


def train(net, trainloader, epochs, device):
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adamax(net.parameters(), lr=1e-3, betas=(0.9, 0.995))
    net.train()
    for _ in range(epochs):
        running_loss = 0.0
        for batch in trainloader:
            in_spk = batch["data"].to(device)
            labels = batch["label"].to(device)
            optimizer.zero_grad()
            reset(net)

            spk_rec = []
            for t in range(in_spk.shape[1]):
                spk_out = net(in_spk[:, t])
                spk_rec.append(spk_out)
            spk_rec = torch.stack(spk_rec)

            pred = spk_rec.sum(axis=0)
            loss = criterion(pred, labels)  # + 1e-2 * torch.mean(spk_rec)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        avg_trainloss = running_loss / len(trainloader)

    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for batch in testloader:
            in_spk = batch["data"].to(device)
            labels = batch["label"].to(device)

            reset(net)

            spk_rec = []
            for t in range(in_spk.shape[1]):
                spk_out = net(in_spk[:, t])
                spk_rec.append(spk_out)
            spk_rec = torch.stack(spk_rec)

            pred = spk_rec.sum(axis=0)
            loss += criterion(
                pred, labels
            ).item()  # + 1e-2 * torch.mean(spk_rec)).item()
            correct += (torch.max(pred, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    return loss, accuracy


# def get_weights(net):
#     # return [val.cpu().numpy() for _, val in net.state_dict().items()]
#     return {name: param.detach().clone() for name, param in net.named_parameters()}


# def set_weights(net, parameters):
#     params_dict = zip(net.state_dict().keys(), parameters)
#     state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
#     net.load_state_dict(state_dict, strict=True)


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
