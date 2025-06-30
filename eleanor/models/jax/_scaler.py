from typing import Optional

import jax
import equinox as eqx
from jax import custom_jvp
from chex import Array, PRNGKey


class Scaler(eqx.Module):
    """
    Simple module to scale the output and gradient of a layer.

    Attributes
    ----------
    scale: float
        Multiplicative factor of the input.
    grad_scale: float
        Multiplicative factor of the gradient (default the same as scale).
    """

    scale: float
    grad_scale: float

    def __init__(self, scale: float = 1.0, grad_scale: float = None) -> None:
        self.scale = scale
        if grad_scale is None:
            self.grad_scale = scale
        else:
            self.grad_scale = grad_scale

    @jax.named_scope("eleanor.models.Scaler")
    def __call__(self, x: Array, *, key: Optional[PRNGKey] = None) -> Array:
        """ """

        @custom_jvp
        def ste(x):
            return x * self.scale

        @ste.defjvp
        def f_jvp(primals, tangents):
            (x,) = primals
            (x_dot,) = tangents
            primal_out = ste(x)
            return primal_out, x_dot * self.grad_scale

        return ste(x)
