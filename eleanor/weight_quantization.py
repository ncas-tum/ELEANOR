from typing import Union, Literal, Optional
from functools import partial

import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrand
from chex import Array, PRNGKey


@partial(jax.jit, static_argnames=("preserve_zero", "preserve_max_val"))
def get_quant_bound(bits, preserve_zero=False, preserve_max_val=False):
    def get_edge_of_last_int_bucket():
        ret = 2.0 ** (bits - 1)
        if preserve_zero:
            # Lose one bucket.
            ret -= 0.5
        return ret

    def get_center_of_last_int_bucket():
        return get_edge_of_last_int_bucket() - 0.5

    if preserve_max_val:
        return get_center_of_last_int_bucket()
    else:
        return get_edge_of_last_int_bucket()


@partial(
    jax.jit,
    static_argnames=("num_bits", "preserve_zero", "preserve_max_val", "stochastic"),
)
def quantize_weights(
    weights: Array,
    num_bits: int = 8,
    preserve_zero: bool = True,
    preserve_max_val: bool = False,
    stochastic: bool = True,
    key: Optional[PRNGKey] = None,
) -> Array:
    """
    Quantize weights to a specified bit precision. It can apply
    stochastic rounding using the weight value as the probability
    for rounding.

    Parameters
    ----------
    weights: Array
        The weights to quantize.
    num_bits: int
        Number of bits for quantization (e.g., 8 for int8).
    preserve_zero: bool
        Preserve zero value in the quantization.
    preserve_max_val: bool
        preserve maximum weight value in the quantization.
    stochastic: bool
        Apply stochastic rounding based on the weight value.
    key: PRNGKey
        A random key is used to implement stochastic rounding.
        Optional in case not using stochastic rounding.

    Returns
    -------
    Quantized weights.
    """

    # Obtaining the maximum value for the specified number of bits
    max_val = get_quant_bound(num_bits, preserve_zero, preserve_max_val)

    # Calculate absolute max across all elements
    abs_max = jnp.max(jnp.abs(weights), keepdims=True)

    # Prevent division by zero
    scale = jnp.where(abs_max == 0.0, jnp.ones_like(abs_max), abs_max)

    # Scale weights to the range and quantize
    scaled_weights = weights / scale * max_val

    if stochastic:
        w_floor = jnp.floor(scaled_weights)
        w_ceil = jnp.ceil(scaled_weights)

        prob = scaled_weights - w_floor
        index = jrand.uniform(key, scaled_weights.shape) < prob
        qval = jnp.where(index, w_ceil, w_floor)
    else:
        qval = jnp.round(scaled_weights)

    quantized_weights = jnp.clip(qval, -max_val, max_val).astype(jnp.int8)

    return quantized_weights, max_val / scale


class QuantizedLinear(eqx.nn.Linear):
    """Adapted from nn.Linear. Performs a linear transformation,
    and produces quantized weights when called."""

    quant_bits: int  # Specify number of bits for quantization
    stochastic: bool

    def __init__(
        self,
        in_features: Union[int, Literal["scalar"]],
        out_features: Union[int, Literal["scalar"]],
        use_bias: bool = True,
        dtype=None,
        quant_bits: int = 8,
        stochastic: bool = True,
        *,
        key: PRNGKey,
    ):
        """
        Parameters
        ---------
        in_features: int
            The input size.
            The input to the layer should be a vector of shape `(in_features,)`
        out_features: int
            The output size.
            The output from the layer will be a vector of shape `(out_features,)`.
        use_bias: bool
            Whether to add on a bias as well.
        dtype: object
            The dtype to use for the weight and the bias in this layer.
            Defaults to either `jax.numpy.float32` or `jax.numpy.float64` depending
            on whether JAX is in 64-bit mode.
        quant_bits: int
            The number of bits to quantize the weights.
        stochastic: bool
            Whether to use stochastic rounding during the quantization.
        key: PRNGKey
            A `jax.random.PRNGKey` used to provide randomness for parameter
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
        self.stochastic = stochastic

    @jax.named_scope("eleanor.weight_quantization.QuantizedLinear")
    def __call__(self, x: Array, *, key: Optional[PRNGKey] = None) -> Array:
        """
        Parameters
        ---------
        x: Array
            The input. Should be a JAX array of shape `(in_features,)`. (Or shape
            `()` if `in_features="scalar"`.)
        key: PRNGKey
            Ignored; provided for compatibility with the rest of the Equinox API.
            (Keyword only argument.)

        Notes
        -----
        If you want to use higher order tensors as inputs (for example featuring "
        "batch dimensions) then use `jax.vmap`. For example, for an input `x` of "
        "shape `(batch, in_features)`, using
        ```python
        linear = eleanor.weight_quantization.QuantizedLinear(...)
        jax.vmap(linear)(x)
        ```
        will produce the appropriate output of shape `(batch, out_features)`.

        Returns
        -------
        A JAX array of shape `(out_features,)`. (Or shape `()` if
        `out_features="scalar"`.)
        """

        if self.in_features == "scalar":
            if jnp.shape(x) != ():
                raise ValueError("x must have scalar shape")
            x = jnp.broadcast_to(x, (1,))

        # Quantize weights straight-through estimator
        quantized_weights, scale = quantize_weights(
            self.weight,
            num_bits=self.quant_bits,
            preserve_zero=True,
            preserve_max_val=False,
            stochastic=self.stochastic,
            key=key,
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
