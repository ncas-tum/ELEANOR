import numpy as np
import torch
from torch import Tensor
from snntorch import SpikingNeuron

from .variability import D2DVar

__all__ = ["Heracles"]


@torch.library.register_fake("eleanor::heracles")
def _(
    synaptic_input: Tensor,
    v: Tensor,
    p: Tensor,
    A: float,
    t_fe: float,
    eps_fe: float,
    eps_depl: float,
    q_fix_depl: float,
    n_depl: float,
    e_off: float,
    temp: float,
    w_b: float,
    d_e: float,
    P_s: float,
    I_0: float,
    V_t: float,
    C_par: float,
    C_fe: float,
    C_tot_init: float,
    I_dsc: float,
    _eps0: float,
    _q: float,
    _k: float,
    _h: float,
    threshold: float,
    dt: float,
    paramsScale: float,
) -> None:
    torch._check(synaptic_input.shape == v.shape)
    torch._check(p.shape == v.shape)
    torch._check(synaptic_input.dtype == torch.float)
    torch._check(v.dtype == torch.float)
    torch._check(p.dtype == torch.float)
    torch._check(synaptic_input.device == v.device)
    torch._check(p.device == v.device)

    return [torch.empty_like(v), torch.empty_like(p)]


def _backward(ctx, grads):
    grad_v_out, grad_p_out = grads
    I, v, p = ctx.saved_tensors

    return (
        grad_v_out,
        grad_v_out,
        grad_p_out,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
        None,
    )


def _setup_context(ctx, inputs, output):
    (
        synaptic_input,
        v,
        p,
        A,
        t_fe,
        eps_fe,
        eps_depl,
        q_fix_depl,
        n_depl,
        e_off,
        temp,
        w_b,
        d_e,
        P_s,
        I_0,
        V_t,
        C_par,
        C_fe,
        C_tot_init,
        I_dsc,
        _eps0,
        _q,
        _k,
        _h,
        threshold,
        dt,
        paramsScale,
    ) = inputs

    ctx.save_for_backward(synaptic_input, v, p)


torch.library.register_autograd(
    "eleanor::heracles", _backward, setup_context=_setup_context
)


class Heracles(SpikingNeuron):
    def __init__(
        self,
        A: float = 25e-12,
        t_fe: float = 9.8e-9,
        eps_fe: float = 70,
        eps_depl: float = 3.6,
        q_fix_depl: float = 945e-4,
        n_depl: float = 1.4e28,
        e_off: float = 2e7,
        temp: float = 294,
        w_b: float = 1.05,
        d_e: float = 7.5e-9,
        P_s: float = 27e-2,
        I_0: float = 1e-4,
        V_t: float = 0.32,
        C_par: float = 15e-15,
        I_dsc: float = 10e-12,
        threshold: float = 3.5,
        dt: float = 1e-3,
        paramsScale: float = 1e12,
        variability: float = 0.0,
        spike_grad=None,
        surrogate_disable=False,
        init_hidden=False,
        inhibition=False,
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

        A = A * paramsScale
        t_fe = t_fe * paramsScale
        # eps_depl = eps_depl * paramsScale
        # eps_fe = eps_fe * paramsScale
        q_fix_depl = q_fix_depl  # * paramsScale
        e_off = e_off / paramsScale
        d_e = d_e * paramsScale
        C_par = C_par * paramsScale
        I_dsc = I_dsc * paramsScale
        n_depl = n_depl / paramsScale

        _eps0 = 8.85418792394420013968e-12 * paramsScale  # Vacuum permittivity
        _q = 1.60217663e-19 * paramsScale
        _k = 1.380649e-23 * paramsScale  # Boltzmann constant
        _h = 6.62607015e-34 * paramsScale  # Planck constant

        # Initial values, to be checked whether sensible
        e_dummy = 0
        prob = 0

        C_fe = _eps0 * eps_fe / t_fe * A
        w_depl_d = (_eps0 * eps_fe * e_dummy + q_fix_depl) * paramsScale / _q / n_depl
        w_depl_u = np.abs(
            (_eps0 * eps_fe * e_dummy - q_fix_depl) * paramsScale / _q / n_depl
        )
        w_depl = w_depl_d * w_depl_u / (prob * w_depl_u + (1 - prob) * w_depl_d)
        C_tot_init = 1 / (1 / (C_fe + C_par) + 1 / (_eps0 * eps_depl / w_depl * A))

        self._register_buffer("A", A, False)
        self._register_buffer("t_fe", t_fe, False)
        self._register_buffer("eps_fe", eps_fe, False)
        self._register_buffer("eps_depl", eps_depl, False)
        self._register_buffer("q_fix_depl", q_fix_depl, False)
        self._register_buffer("n_depl", n_depl, False)
        self._register_buffer("e_off", e_off, False)
        self._register_buffer("temp", temp, False)
        self._register_buffer("w_b", w_b, False)
        self._register_buffer("d_e", d_e, False)
        self._register_buffer("P_s", P_s, False)
        self._register_buffer("I_0", I_0, False)
        self._register_buffer("V_t", V_t, False)
        self._register_buffer("C_par", C_par, False)
        self._register_buffer("C_fe", C_fe, False)
        self._register_buffer("C_tot_init", C_tot_init, False)
        self._register_buffer("I_dsc", I_dsc, False)
        self._register_buffer("paramsScale", paramsScale, False)
        self._register_buffer("dt", dt, False)
        self._register_buffer("_eps0", _eps0, False)
        self._register_buffer("_q", _q, False)
        self._register_buffer("_k", _k, False)
        self._register_buffer("_h", _h, False)

        self.A_var = D2DVar("A", variability)
        self.n_depl_var = D2DVar("n_depl", variability)
        self.P_s_var = D2DVar("P_s", variability)
        self.t_fe_var = D2DVar("t_fe", variability)

        self._init_mem()

        if self.reset_mechanism_val == 0:  # reset by subtraction
            self.state_function = self._base_sub
        elif self.reset_mechanism_val == 1:  # reset to zero
            self.state_function = self._base_zero
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            self.state_function = self._base_int

        self.reset_delay = reset_delay

    def _register_buffer(self, name: str, param: torch.Tensor, learn: bool):
        if not isinstance(param, torch.Tensor):
            param = torch.as_tensor(param)
        self.register_buffer(name, param)

    def _init_mem(self):
        mem = torch.zeros(0)
        pol = torch.zeros(0) - self.P_s

        self.register_buffer("mem", mem, False)
        self.register_buffer("pol", pol, False)

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        self.pol = torch.zeros_like(self.pol, device=self.pol.device) - self.P_s_var(
            self.P_s, self.pol.shape
        )
        return self.pol, self.mem

    def heracles_step(
        self,
        input_: Tensor,
        v: Tensor,
        p: Tensor,
        A: float,
        t_fe: float,
        n_depl: float,
        P_s: float,
    ) -> None:
        return torch.ops.eleanor.heracles.default(
            input_,
            v,
            p,
            A,
            t_fe,
            self.eps_fe,
            self.eps_depl,
            self.q_fix_depl,
            n_depl,
            self.e_off,
            self.temp,
            self.w_b,
            self.d_e,
            P_s,
            self.I_0,
            self.V_t,
            self.C_par,
            self.C_fe,
            self.C_tot_init,
            self.I_dsc,
            self._eps0,
            self._q,
            self._k,
            self._h,
            self.threshold,
            self.dt,
            self.paramsScale,
        )

    def forward(self, input_, pol=None, mem=None):
        if pol is not None:
            self.pol = pol

        if mem is not None:
            self.mem = mem

        if self.init_hidden and (mem is not None or pol is not None):
            raise TypeError(
                "`mem` or `pol` should not be passed as an argument "
                "while `init_hidden=True`"
            )

        P_s = self.P_s_var(self.P_s, input_.shape)
        if not self.pol.shape == input_.shape:
            self.pol = torch.zeros_like(input_, device=self.pol.device) - P_s

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
                self.pol = self.pol - do_reset * (P_s + self.pol)

        if self.output:
            return spk, self.pol, self.mem
        elif self.init_hidden:
            return spk
        else:
            return spk, self.pol, self.mem

    def _base_state_function(self, input_):
        v, p = self.mem, self.pol

        A = self.A_var(self.A, v.shape)
        n_depl = self.n_depl_var(self.n_depl, v.shape)
        P_s = self.P_s_var(self.P_s, v.shape)
        t_fe = self.t_fe_var(self.t_fe, v.shape)

        mem, pol = self.heracles_step(
            input_,
            v,
            p,
            A,
            t_fe,
            n_depl,
            P_s,
        )

        mem = torch.clip(mem, 0, 5)
        pol = torch.clip(pol, -self.P_s, self.P_s)

        return pol, mem

    def _base_sub(self, input_):
        pol, mem = self._base_state_function(input_)
        P_s = self.P_s_var(self.P_s, pol.shape)

        mem = mem - self.reset * self.threshold
        pol = pol - self.reset * 2 * P_s
        return pol, mem

    def _base_zero(self, input_):
        pol, mem = self._base_state_function(input_)
        pol2, mem2 = self._base_state_function(input_)
        P_s = self.P_s_var(self.P_s, pol.shape)

        pol -= (pol2 + P_s) * self.reset
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
            if isinstance(cls.instances[layer], Heracles):
                cls.instances[layer].pol.detach_()
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance
        variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Heracles):
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
