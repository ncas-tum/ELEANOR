import torch


class _TauSurr(torch.autograd.Function):

    @staticmethod
    def forward(ctx, E, tau_p, E_a, alpha, soft_E):
        ctx.save_for_backward(E, tau_p)
        ctx.E_a = E_a
        ctx.alpha = alpha
        ctx.soft_E = soft_E
        return 1 / (tau_p * torch.exp((E_a / (torch.abs(E) + soft_E)) ** alpha))

    @staticmethod
    def backward(ctx, grad_output):

        E, tau_p = ctx.saved_tensors
        E_a = ctx.E_a
        alpha = ctx.alpha
        soft_E = ctx.soft_E
        exponential = (E_a / (torch.abs(E) + soft_E)) ** alpha

        # Tau_p gradient
        grad_tau_p = -torch.exp(-exponential) / (tau_p**2)

        # E gradient
        numerator = alpha * E * torch.exp(-exponential) * exponential

        denumerator = soft_E * tau_p * torch.abs(E) + E**2 * tau_p
        denumerator = torch.where(torch.abs(E) > 0.0, denumerator, 1.0)

        grad_E = numerator / denumerator

        return grad_output * grad_E, grad_output * grad_tau_p, None, None, None


def tau_surr(alpha: float = 1.3, E_a: float = 1.0, soft_E: float = 1e-18):

    def inner(E, tau_p):
        return _TauSurr.apply(E, tau_p, E_a, alpha, soft_E)

    return inner
