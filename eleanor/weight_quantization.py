from typing import Union, Literal, Optional

import jax
import equinox as eqx
import jax.numpy as jnp
from jaxtyping import Array, PRNGKeyArray


def quantize_weights(
    weights: Array, random_key: PRNGKeyArray, num_bits: int = 8
) -> Array:
    """
    Quantize weights to a specified bit precision.

    Args:
        weights: The weights to quantize.
        num_bits: Number of bits for quantization (e.g., 8 for int8).
        random_key: A random key is used to implement stochastic rounding.

    Returns:
        Quantized weights.
    """

    # Obtaining the maximum value for the specified number of bits
    max_val = 2 ** (num_bits - 1) - 1  # e.g., 127 for int8

    # Calculate absolute max across all elements
    abs_max = jnp.max(jnp.abs(weights), keepdims=True)

    # Prevent division by zero
    scale = jnp.where(abs_max == 0.0, jnp.ones_like(abs_max), abs_max)

    # Scale weights to the range and quantize
    scaled_weights = weights / scale * max_val
    random_uniform = jax.random.uniform(random_key, shape=weights.shape)
    quantized_weights = jnp.clip(
        jnp.round(scaled_weights + random_uniform - 0.5), -max_val, max_val
    ).astype(jnp.int8)

    return quantized_weights, max_val / scale


class QuantizedLinear(eqx.nn.Linear):
    """Adapted from nn.Linear. Performs a linear transformation,
    and produces quantized weights when called."""

    quant_bits: int = 8  # Specify number of bits for quantization

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        dtype=None,
        quant_bits: int = 8,
        *,
        key: PRNGKeyArray,
    ):
        """**Arguments:**

        - `in_features`: The input size. The input to the layer should be a vector of
            shape `(in_features,)`
        - `out_features`: The output size. The output from the layer will be a vector
            of shape `(out_features,)`.
        - `use_bias`: Whether to add on a bias as well.
        - `dtype`: The dtype to use for the weight and the bias in this layer.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        - `key`: A `jax.random.PRNGKey` used to provide randomness for parameter
            initialisation. (Keyword only argument.)

        Note that `in_features` also supports the string `"scalar"` as a special value.
        In this case the input to the layer should be of shape `()`.

        Likewise `out_features` can also be a string `"scalar"`, in which case the
        output from the layer will have shape `()`.
        """
        super(QuantizedLinear, self).__init__(
            in_features, out_features, use_bias, dtype, key=key
        )
        self.quant_bits = quant_bits

    @jax.named_scope("eleanor.weight_quantization.QuantizedLinear")
    def __call__(self, x: Array, *, key: Optional[PRNGKeyArray] = None) -> Array:
        """**Arguments:**

        - `x`: The input. Should be a JAX array of shape `(in_features,)`. (Or shape
            `()` if `in_features="scalar"`.)
        - `key`: Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        !!! info

            If you want to use higher order tensors as inputs (for example featuring "
            "batch dimensions) then use `jax.vmap`. For example, for an input `x` of "
            "shape `(batch, in_features)`, using
            ```python
            linear = eleanor.weight_quantization.QuantizedLinear(...)
            jax.vmap(linear)(x)
            ```
            will produce the appropriate output of shape `(batch, out_features)`.

        **Returns:**

        A JAX array of shape `(out_features,)`. (Or shape `()` if
        `out_features="scalar"`.)
        """

        if self.in_features == "scalar":
            if jnp.shape(x) != ():
                raise ValueError("x must have scalar shape")
            x = jnp.broadcast_to(x, (1,))

        # Quantize weights straight-through estimator
        quantized_weights, scale = quantize_weights(
            self.weight, key, num_bits=self.quant_bits
        )

        # Use the straight-through estimator approach to ensure gradients
        # are calculated using floating point weights.
        qweights = self.weight + jax.lax.stop_gradient(
            quantized_weights / scale - self.weight
        )

        x = qweights @ x
        if self.bias is not None:
            x = x + self.bias
        if self.out_features == "scalar":
            assert jnp.shape(x) == (1,)
            x = jnp.squeeze(x)
        return x
