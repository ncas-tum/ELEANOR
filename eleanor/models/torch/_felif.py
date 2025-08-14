from typing import Sequence

import numpy as np
import torch
from snntorch import SpikingNeuron

from ._surrogate import tau_surr
from .variability import D2DVar


class FeLIF(SpikingNeuron):
    def __init__(
        self,
        tau_p: float,
        tau_m: float,
        P_s: float = 0.27,
        alpha: float = 1.0,
        beta: float = 1.0,
        dt: float = 1e-3,
        threshold: float = 1.0,
        tau_alpha: float = 1.3,
        E_a: float = 1.0,
        soft_E: float = 1e-18,
        variability: float = 0.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
        learn_P_s=False,
        learn_alpha=False,
        learn_beta=False,
        learn_gamma=False,
        learn_threshold=False,
        reset_mechanism="zero",
        state_quant=False,
        output=False,
        reset_delay=True,
        graded_spikes_factor=1.0,
        learn_graded_spikes_factor=False,
    ):

        super().__init__(
            threshold,
            spike_grad,
            surrogate_disable,
            init_hidden,
            inhibition,
            learn_threshold,
            reset_mechanism,
            state_quant,
            output,
            graded_spikes_factor,
            learn_graded_spikes_factor,
        )

        self._reset_mechanism = reset_mechanism
        self.tau_surr_fn = tau_surr(tau_alpha, E_a, soft_E)
        self._register_buffer("P_s", P_s, learn_P_s)
        self._register_buffer("alpha", alpha, learn_alpha)
        self._register_buffer("beta", beta, learn_beta)
        self._register_buffer("gamma", np.exp(-dt / tau_m), learn_gamma)
        self._register_buffer("tau_p", tau_p, False)
        self._register_buffer("dt", dt, False)

        self.P_s_var = D2DVar("P_s", variability)
        self.alpha_var = D2DVar("alpha", variability)
        self.beta_var = D2DVar("beta", variability)
        self.gamma_var = D2DVar("gamma", variability)
        self.tau_p_var = D2DVar("tau_p", variability)

        self._init_mem()

        if self.reset_mechanism_val == 0:  # reset by subtraction
            self.state_function = self._base_sub
        elif self.reset_mechanism_val == 1:  # reset to zero
            self.state_function = self._base_zero
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            self.state_function = self._base_int

        self.reset_delay = reset_delay

    def update_variability(self, shape: Sequence[int]) -> None:
        self.P_s_var.update_variability(shape)
        self.alpha_var.update_variability(shape)
        self.beta_var.update_variability(shape)
        self.gamma_var.update_variability(shape)
        self.tau_p_var.update_variability(shape)

    def _register_buffer(self, name: str, param: torch.Tensor, learn: bool):
        if not isinstance(param, torch.Tensor):
            param = torch.as_tensor(param)
        self.register_buffer(name, param)

    def _init_mem(self):
        mem = torch.zeros(0)
        pol = torch.zeros(0) - 1

        self.register_buffer("mem", mem, False)
        self.register_buffer("pol", pol, False)

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        self.pol = -torch.ones_like(self.pol, device=self.pol.device)
        return self.pol, self.mem

    def forward(self, input_, pol=None, mem=None):
        if pol is not None:
            self.pol = pol

        if mem is not None:
            self.mem = mem

        if self.init_hidden and (mem is not None or pol is not None):
            raise TypeError(
                "`mem` or `syn` should not be passed as an argument "
                "while `init_hidden=True`"
            )

        if not self.pol.shape == input_.shape:
            self.pol = -torch.ones_like(input_, device=self.pol.device)

        if not self.mem.shape == input_.shape:
            self.mem = torch.zeros_like(input_, device=self.mem.device)

        self.reset = self.mem_reset(self.mem)
        self.pol, self.mem = self.state_function(input_)

        if self.state_quant:
            self.mem = self.state_quant(self.mem)
            self.pol = self.state_quant(self.pol)

        if self.inhibition:
            spk = self.fire_inhibition(self.mem.size(0), self.mem)  # batch_size
        else:
            spk = self.fire(self.mem)

        if not self.reset_delay:
            # reset membrane potential _right_ after spike
            do_reset = (
                spk / self.graded_spikes_factor - self.reset
            )  # avoid double reset
            if self.reset_mechanism_val == 0:  # reset by subtraction
                self.mem = self.mem - do_reset * self.threshold
                self.pol = self.pol - do_reset * 2
            elif self.reset_mechanism_val == 1:  # reset to zero
                self.mem = self.mem - do_reset * self.mem
                self.pol = self.pol - do_reset * (1 + self.pol)

        if self.output:
            return spk, self.pol, self.mem
        elif self.init_hidden:
            return spk
        else:
            return spk, self.pol, self.mem

    def _base_state_function(self, input_):
        v, p = self.mem, self.pol

        if len(v.shape) > 1:  # In case does not have batch
            shape = (1,) + v.shape[1:]
        else:
            shape = v.shape
        P_s = self.P_s_var(self.P_s, shape)
        alpha = self.alpha_var(self.alpha, shape)
        beta = self.beta_var(self.beta, shape)
        gamma = self.gamma_var(self.gamma, shape)
        tau_p = self.tau_p_var(self.tau_p, shape)

        gamma = gamma.clamp(0, 1)

        E = v * alpha - p * beta
        tau = self.tau_surr_fn(E, tau_p)
        gamma_p = torch.exp(-self.dt * tau)

        Ip = P_s * (torch.sign(E) - p) * self.dt * tau

        base_fn_pol = gamma_p * p + (1 - gamma_p) * torch.sign(E)
        base_fn_mem = gamma * v - (1 - gamma) * Ip + input_

        return base_fn_pol, base_fn_mem

    def _base_sub(self, input_):
        pol, mem = self._base_state_function(input_)
        mem = mem - self.reset * self.threshold
        pol = pol - self.reset * 2
        return pol, mem

    def _base_zero(self, input_):
        pol, mem = self._base_state_function(input_)
        pol2, mem2 = self._base_state_function(input_)
        pol -= (pol2 + 1) * self.reset
        mem -= mem2 * self.reset
        return pol, mem

    def _base_int(self, input_):
        return self._base_state_function(input_)

    @classmethod
    def detach_hidden(cls):
        """Used to detach hidden states from the current graph.
        Intended for use in truncated backpropagation through
        time where hidden state variables are instance variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], FeLIF):
                cls.instances[layer].pol.detach_()
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance
        variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], FeLIF):
                cls.instances[layer].pol = (
                    torch.zeros_like(
                        cls.instances[layer].pol,
                        device=cls.instances[layer].pol.device,
                    )
                    - cls.instances[layer].P_s
                )
                cls.instances[layer].mem = torch.zeros_like(
                    cls.instances[layer].mem,
                    device=cls.instances[layer].mem.device,
                )
