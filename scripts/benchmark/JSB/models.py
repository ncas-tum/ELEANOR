import math

import jax
import equinox as eqx
import jax.numpy as jnp
import jax.random as jrand
from jax import custom_jvp
from chex import Array, PRNGKey
from equinox.nn._misc import default_init
from snnax.functional.surrogate import SpikeFn

from eleanor.models.jax._equinox import BrunoCell, BrunoParams, default_floating_dtype
from eleanor.models.jax.variability import StaticWrapper


def tanh_surrogate() -> SpikeFn:
    """
    Implementation of the sigmoidal surrogate gradient function as described in
    'The remarkable robustness of surrogate gradient learning
    for instilling complex function in spiking neural networks' by Zenke and
    Vogels: (https://www.biorxiv.org/content/10.1101/2020.06.29.176925v1)

    Arguments:
        `beta` (float): Parameter to control the steepness of the surrogate
            gradient. Default is .5.

    Returns:
        A function that returns the surrogate gradient of the heaviside function.
    """

    @custom_jvp
    def heaviside_with_tanh_surrogate(x):
        return jnp.heaviside(x, 1.0)

    @heaviside_with_tanh_surrogate.defjvp
    def f_jvp(primals, tangents):
        (x,) = primals
        (x_dot,) = tangents
        primal_out = heaviside_with_tanh_surrogate(x)
        # TODO multiplication by beta correct here?
        tangent_out = x_dot * (1.0 - jnp.tanh(x) ** 2)
        return primal_out, tangent_out

    return heaviside_with_tanh_surrogate


class SNUCell(eqx.Module):

    weight_ih: Array
    weight_hh: Array
    bias_i: Array | None
    bias_h: Array | None
    threshold: Array | None
    decay: StaticWrapper | None
    input_size: int = eqx.field(static=True)
    hidden_size: int = eqx.field(static=True)
    recurrent: bool = eqx.field(static=True)
    spike_fn: SpikeFn

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        decay: float = 0.4,
        threshold: float = 1.0,
        recurrent: bool = False,
        spike_fn=None,
        dtype=None,
        *,
        key: PRNGKey,
    ):
        dtype = default_floating_dtype() if dtype is None else dtype
        ihkey, hhkey, bikey, bhkey = jrand.split(key, 4)
        lim = math.sqrt(1 / hidden_size)

        ihshape = (hidden_size, input_size)
        self.weight_ih = default_init(ihkey, ihshape, dtype, lim)
        hhshape = (hidden_size, hidden_size)
        self.weight_hh = default_init(hhkey, hhshape, dtype, lim)

        bshape = (hidden_size,)
        self.bias_i = default_init(bikey, bshape, dtype, lim)
        self.bias_h = default_init(bhkey, bshape, dtype, lim)

        self.threshold = jnp.full((hidden_size,), threshold)
        self.decay = StaticWrapper(jnp.full((hidden_size,), decay))

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.recurrent = recurrent
        if spike_fn is None:
            spike_fn = jax.nn.sigmoid

        self.spike_fn = spike_fn

    @jax.named_scope("SNUCell")
    def __call__(self, input, hidden, *, key=None):
        h, s = hidden

        h = h * (1 - s) * self.decay.content

        # bias_i = cast(Array, self.bias_i)
        igates = self.weight_ih @ input  # + bias_i
        h = h + igates

        if self.recurrent:
            # bias_h = cast(Array, self.bias_h)
            hgates = self.weight_hh @ s  # + bias_h
            h = h + hgates

        s = self.spike_fn(h - self.threshold)
        return (h, s)


class RNN(eqx.Module):
    hidden_size: int
    cell: eqx.Module
    linear: eqx.nn.Linear

    def __init__(
        self, model_name, in_size, out_size, hidden_size, variability, params, *, key
    ):
        ckey, lkey = jrand.split(key, 2)
        # ckey, lkey = jrand.split(key)
        self.hidden_size = hidden_size

        if model_name == "GRU":
            self.cell = eqx.nn.GRUCell(in_size, hidden_size, key=ckey)
        elif model_name == "LSTM":
            self.cell = eqx.nn.LSTMCell(in_size, hidden_size, key=ckey)
        elif model_name == "SNU":
            self.cell = SNUCell(
                in_size,
                hidden_size,
                params["decay"],
                params["threshold"],
                recurrent=True,
                spike_fn=jax.nn.sigmoid,
                key=ckey,
            )
        elif model_name == "LIF":
            self.cell = SNUCell(
                in_size,
                hidden_size,
                params["decay"],
                params["threshold"],
                recurrent=True,
                spike_fn=tanh_surrogate(),
                key=ckey,
            )
        elif model_name == "FeLIF":
            bparams = BrunoParams(I_dsc=params["I_dsc"], threshold=params["threshold"])
            self.cell = BrunoCell(
                in_size,
                hidden_size,
                params=bparams,
                use_bias=False,
                param_scale=10 ** params["paramScale"],
                spikefn=tanh_surrogate(),
                variability=variability,
                key=ckey,
            )
        self.linear = eqx.nn.Linear(hidden_size, out_size, use_bias=True, key=lkey)

    def init_state(self):
        if isinstance(self.cell, BrunoCell):
            return self.cell.init_state()
        elif isinstance(self.cell, eqx.nn.GRUCell):
            return jnp.zeros((self.hidden_size,))
        elif isinstance(self.cell, SNUCell):
            return (jnp.zeros((self.hidden_size,)), jnp.zeros((self.hidden_size,)))
        elif isinstance(self.cell, eqx.nn.LSTMCell):
            return (jnp.zeros((self.hidden_size,)), jnp.zeros((self.hidden_size,)))

    def __call__(self, input, *, key):
        hidden = self.init_state()

        def f(carry, inp):

            if isinstance(self.cell, BrunoCell):
                carry = self.cell(inp, carry, key=key)
                spk = carry[0]
            elif isinstance(self.cell, eqx.nn.GRUCell):
                carry = self.cell(inp, carry)
                spk = carry
            elif isinstance(self.cell, eqx.nn.LSTMCell):
                carry = self.cell(inp, carry)
                spk = carry[0]
            elif isinstance(self.cell, SNUCell):
                carry = self.cell(inp, carry)
                spk = carry[1]

            x = self.linear(spk)

            return carry, x

        _, out = jax.lax.scan(f, hidden, input)
        return out

    def record(self, input, *, key):
        hidden = self.init_state()

        def f(carry, inp):

            if isinstance(self.cell, BrunoCell):
                carry = self.cell(inp, carry, key=key)
                spk = carry[0]
            elif isinstance(self.cell, eqx.nn.GRUCell):
                carry = self.cell(inp, carry)
                spk = carry
            elif isinstance(self.cell, eqx.nn.LSTMCell):
                carry = self.cell(inp, carry)
                spk = carry[0]
            elif isinstance(self.cell, SNUCell):
                carry = self.cell(inp, carry)
                spk = carry[1]

            x = self.linear(spk)

            return carry, (x, spk)

        _, out = jax.lax.scan(f, hidden, input)
        return out
