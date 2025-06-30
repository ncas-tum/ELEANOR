import torch
from torch import Tensor
from snntorch import SpikingNeuron

__all__ = ["Bruno"]


@torch.library.register_fake("eleanor::bruno")
def _(
    synaptic_input: Tensor,
    v: Tensor,
    p: Tensor,
    cap_divider: float,
    depol_divider: float,
    P_s: float,
    A: float,
    I_0: float,
    E_a: float,
    V_t: float,
    I_dsc: float,
    tau_0: float,
    C_tot: float,
    soft_E: float,
    alpha: float,
    threshold: float,
    dt: float,
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
    grad_I, grad_v, grad_p = None, None, None

    E = ctx.cap_divider * v - ctx.depol_divider * p
    I_pa = ctx.A * (torch.sign(E) * ctx.P_s - p)
    I_pb = 1 / (
        ctx.tau_0
        * torch.exp(torch.pow(ctx.E_a / (torch.abs(E) + ctx.soft_E), ctx.alpha))
    )
    # I_p = I_pa * I_pb

    # tau = 1 / (ctx.tau_0*torch.exp((ctx.E_a / (torch.abs(E) + ctx.soft_E)) ** ctx.alpha))
    exponential = (ctx.E_a / (torch.abs(E) + ctx.soft_E)) ** ctx.alpha
    numerator = ctx.alpha * torch.exp(-exponential) * exponential
    denumerator = ctx.tau_0 * ctx.soft_E * torch.abs(E) + ctx.tau_0 * E**2
    denumerator = torch.where(torch.abs(denumerator) > 0, denumerator, 1.0)
    tangent_E = (E * numerator) / denumerator
    tangent_E = tangent_E

    dI_pdp = -ctx.A * I_pb - I_pa * tangent_E * ctx.depol_divider
    dIldv = ctx.I_0 * ctx.A * torch.exp(v / ctx.V_t) * torch.sign(v) / ctx.V_t
    dI_pdv = I_pa * tangent_E * ctx.cap_divider

    dvdv = 1 + ctx.dt / ctx.C_tot * (-dIldv - dI_pdv)
    dvdp = -ctx.dt / ctx.C_tot * dI_pdp
    dpdv = ctx.dt / ctx.A * ctx.cap_divider * tangent_E * I_pa
    dpdp = 1 + ctx.dt * dI_pdp / ctx.A

    if ctx.needs_input_grad[0]:
        grad_I = grad_v_out * ctx.dt / ctx.C_tot
    if ctx.needs_input_grad[1]:
        grad_v = grad_v_out * dvdv + grad_p_out * dpdv
    if ctx.needs_input_grad[2]:
        grad_p = grad_v_out * dvdp + grad_p_out * dpdp
        # grad_p = grad_p_out * dpdp
    return (
        grad_I,
        grad_v,
        grad_p,
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
        I,
        v,
        p,
        cap_divider,
        depol_divider,
        P_s,
        A,
        I_0,
        E_a,
        V_t,
        I_dsc,
        tau_0,
        C_tot,
        soft_E,
        alpha,
        threshold,
        dt,
    ) = inputs

    ctx.save_for_backward(I, v, p)
    ctx.cap_divider = cap_divider
    ctx.depol_divider = depol_divider
    ctx.P_s = P_s
    ctx.A = A
    ctx.I_0 = I_0
    ctx.E_a = E_a
    ctx.V_t = V_t
    ctx.I_dsc = I_dsc
    ctx.tau_0 = tau_0
    ctx.C_tot = C_tot
    ctx.soft_E = soft_E
    ctx.alpha = alpha
    ctx.threshold = threshold
    ctx.dt = dt


torch.library.register_autograd(
    "eleanor::bruno", _backward, setup_context=_setup_context
)


def bruno_step(
    synaptic_input: Tensor,
    v: Tensor,
    p: Tensor,
    cap_divider: float,
    depol_divider: float,
    P_s: float,
    A: float,
    I_0: float,
    E_a: float,
    V_t: float,
    I_dsc: float,
    tau_0: float,
    C_tot: float,
    soft_E: float,
    alpha: float,
    threshold: float,
    dt: float,
) -> None:
    return torch.ops.eleanor.bruno.default(
        synaptic_input,
        v,
        p,
        cap_divider,
        depol_divider,
        P_s,
        A,
        I_0,
        E_a,
        V_t,
        I_dsc,
        tau_0,
        C_tot,
        soft_E,
        alpha,
        threshold,
        dt,
    )


class Bruno(SpikingNeuron):

    def __init__(
        self,
        A: float = 25e-12,
        t_hzo: float = 10e-9,
        t_int: float = 1.375e-9,
        eps_hzo: float = 25.2,
        eps_int: float = 33,
        E_a: float = 12.7e8,
        P_s: float = 22e-2,
        tau_0: float = 1e-13,
        I_0: float = 1e-4,
        V_t: float = 0.32,
        C_par: float = 15e-15,
        alpha: float = 1.3,
        soft_E: float = 5e-6,
        I_dsc: float = 10e-12,
        threshold: float = 2.5,
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

        A *= paramsScale
        t_hzo *= paramsScale
        t_int *= paramsScale
        E_a /= paramsScale
        C_par *= paramsScale
        soft_E /= paramsScale
        I_dsc *= paramsScale

        _eps0 = 8.85418792394420013968e-12 * paramsScale

        C_0 = _eps0 * eps_hzo / t_hzo * A
        C_tot = C_par + C_0

        cap_divider = eps_int / (t_hzo * eps_int + t_int * eps_hzo)
        depol_divider = 1 / _eps0 * t_int / (t_hzo * eps_int + t_int * eps_hzo)

        params = {
            "A": A,
            "cap_divider": cap_divider,
            "depol_divider": depol_divider,
            "P_s": P_s,
            "I_0": I_0,
            "E_a": E_a,
            "V_t": V_t,
            "I_dsc": I_dsc,
            "tau_0": tau_0,
            "C_tot": C_tot,
            "soft_E": soft_E,
            "alpha": alpha,
            "dt": dt,
        }

        self._register_buffer(params)
        self._init_mem()

        if self.reset_mechanism_val == 0:  # reset by subtraction
            self.state_function = self._base_sub
        elif self.reset_mechanism_val == 1:  # reset to zero
            self.state_function = self._base_zero
        elif self.reset_mechanism_val == 2:  # no reset, pure integration
            self.state_function = self._base_int

        self.reset_delay = reset_delay

    def _register_buffer(self, params):
        for param_name, val in params.items():
            val = torch.as_tensor(val)
            self.register_buffer(param_name, val)

    def _init_mem(self):
        mem = torch.zeros(0)
        pol = torch.zeros(0) - self.P_s

        self.register_buffer("mem", mem, False)
        self.register_buffer("pol", pol, False)

    def reset_mem(self):
        self.mem = torch.zeros_like(self.mem, device=self.mem.device)
        self.pol = torch.zeros_like(self.pol, device=self.pol.device) - self.P_s

        return self.pol, self.mem

    def forward(self, input_, pol=None, mem=None):
        if pol is not None:
            self.pol = pol

        if mem is not None:
            self.mem = mem

        if self.init_hidden and (mem is not None or pol is not None):
            raise TypeError(
                "`mem` or `pol` should not be passed as an argument while `init_hidden=True`"
            )
        if not self.pol.shape == input_.shape:
            self.pol = torch.zeros_like(input_, device=self.pol.device) - self.P_s

        if not self.mem.shape == input_.shape:
            self.mem = torch.zeros_like(input_, device=self.mem.device)

        self.reset = self.mem_reset(self.mem)
        self.pol, self.mem = self.state_function(input_)

        if self.state_quant:
            self.pol = self.state_quant(self.pol)
            self.mem = self.state_quant(self.mem)

        if self.inhibition:
            spk = self.fire_inhibition(self.mem.size(0), self.mem)
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
                self.pol = self.pol - do_reset * (self.P_s + self.pol)

        if self.output:
            return spk, self.pol, self.mem
        elif self.init_hidden:
            return spk
        else:
            return self.reset, self.pol, self.mem

    def _base_state_function(self, input_):
        mem, pol = bruno_step(
            input_,
            self.mem,
            self.pol,
            self.cap_divider,
            self.depol_divider,
            self.P_s,
            self.A,
            self.I_0,
            self.E_a,
            self.V_t,
            self.I_dsc,
            self.tau_0,
            self.C_tot,
            self.soft_E,
            self.alpha,
            self.threshold,
            self.dt,
        )
        mem = torch.clip(mem, -5, 5)
        pol = torch.clip(pol, -self.P_s, self.P_s)

        return pol, mem

    def _base_sub(self, input_):
        pol, mem = self._base_state_function(input_)

        mem = mem - self.reset * self.threshold
        pol = pol - self.reset * 2 * self.P_s
        return pol, mem

    def _base_zero(self, input_):
        pol, mem = self._base_state_function(input_)
        pol2, mem2 = self._base_state_function(input_)

        pol -= (pol2 + self.P_s) * self.reset
        mem -= (mem2 + 1.5) * self.reset
        return pol, mem

    def _base_int(self, input_):
        return self._base_state_function(input_)

    @classmethod
    def detach_hidden(cls):
        """Used to detach hidden states from the current graph.
        Intended for use in truncated backpropagation through
        time where hidden state variables are instance variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Bruno):
                cls.instances[layer].pol.detach_()
                cls.instances[layer].mem.detach_()

    @classmethod
    def reset_hidden(cls):
        """Used to clear hidden state variables to zero.
        Intended for use where hidden state variables are instance
        variables."""
        for layer in range(len(cls.instances)):
            if isinstance(cls.instances[layer], Bruno):
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
