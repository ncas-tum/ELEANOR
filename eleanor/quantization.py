from typing import Union, Literal, Callable, Optional

import jax
import equinox as eqx
import jax.numpy as jnp
from jax import custom_jvp
from jaxtyping import Array, PRNGKeyArray
from aqt.jax.v2.aqt_quantizer import Quantizer, quantizer_make


def make_fake_quant(quantizer: Quantizer, calibration_axes=None):
    @custom_jvp
    def fake_quant(x):
        x_q, _ = quantizer.quant(x, calibration_axes=calibration_axes)
        return x_q.dequant()

    @fake_quant.defjvp
    def fake_quant_jvp(primals, tangents):
        (x,) = primals
        (x_dot,) = tangents
        x_q, grad_fn = quantizer.quant(x, calibration_axes=calibration_axes)
        primal_out = x_q.dequant()
        tangent_out = grad_fn(x_dot)[0]
        return primal_out, tangent_out

    return fake_quant


class QLinear(eqx.nn.Linear):
    """Performs a quantized linear transformation."""

    quantizer: Quantizer = eqx.field(static=True)
    fake_quant: Callable = eqx.field(static=True)

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        dtype=None,
        n_bits=3,
        *,
        key: PRNGKeyArray,
    ):
        super(QLinear, self).__init__(
            in_features, out_features, use_bias, dtype, key=key
        )

        self.quantizer = quantizer_make(n_bits=n_bits)
        self.fake_quant = make_fake_quant(self.quantizer, calibration_axes=(0, 1))

    @jax.named_scope("QLinear")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        qweights = self.fake_quant(self.weight)

        if self.in_features == "scalar":
            if jnp.shape(x) != ():
                raise ValueError("x must have scalar shape")
            x = jnp.broadcast_to(x, (1,))
        x = qweights @ x
        if self.bias is not None:
            x = x + self.bias
        if self.out_features == "scalar":
            assert jnp.shape(x) == (1,)
            x = jnp.squeeze(x)
        return x
