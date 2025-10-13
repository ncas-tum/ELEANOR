import torch
import torch.nn as nn


class EncodingLayer(nn.Module):
    gain: torch.Tensor
    bias: torch.Tensor
    expansion: float

    def __init__(self, gain: torch.Tensor, bias: torch.Tensor, expansion: int) -> None:
        super().__init__()

        self.register_buffer("gain", gain)
        self.register_buffer("bias", bias)

        self.expansion = expansion

    def __call__(self, synaptic_input: torch.Tensor):
        output = self.gain * (
            torch.tile(synaptic_input, (1, self.expansion)) + self.bias
        )
        return output


class Gain(nn.Module):
    gain: torch.Tensor

    def __init__(self, gain: float = 2000e-12) -> None:
        super().__init__()
        self.register_buffer("gain", torch.tensor(gain))

    def forward(self, x):
        return x * self.gain
