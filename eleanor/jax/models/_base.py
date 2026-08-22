from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import overload

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jrandom
from jaxtyping import Array, PRNGKeyArray


def default_floating_dtype():
    if jax.config.jax_enable_x64:
        return jnp.float64
    else:
        return jnp.float32


def limexp(x):
    safe_x = jnp.minimum(x, 80)
    exp_branch = jnp.exp(safe_x)
    linear_branch = jnp.exp(80) * (1 + (x - safe_x))
    return jnp.where(x > 80, linear_branch, exp_branch)


class NeuronModel(eqx.Module, ABC):
    """
    Base class for neuron models. All neuron models should inherit from this class and implement the `init_state` and `__call__` methods.
    """

    @abstractmethod
    def init_state(self): ...

    @abstractmethod
    def __call__(self, state: Sequence[Array], x: Array, *, key: PRNGKeyArray): ...


class RNN(eqx.Module):
    """
    Wrapper to use a NeuronModel as a recurrent layer in a scan.

    Attributes
    ----------
    base_model: NeuronModel
        Neuron model to be used as a recurrent layer.
    """

    base_model: NeuronModel

    @overload
    def __init__(self, base_model: NeuronModel): ...

    @overload
    def __init__(
        self, base_model_cls: type, *NeuronModel_args, **NeuronModel_kwargs
    ): ...

    def __init__(self, base_model_or_cls, *args, **kwargs):
        if isinstance(base_model_or_cls, NeuronModel):
            self.base_model = base_model_or_cls
        else:
            self.base_model = base_model_or_cls(*args, **kwargs)

    def __call__(self, x: Array, *, key: PRNGKeyArray | None = None):

        state = self.base_model.init_state()

        def scan_fn(state, input):
            x, kmodel = input
            state, output = self.base_model(state, x, key=kmodel)

            return state, output

        keys: PRNGKeyArray | list[None]
        if key is None:
            keys = [None] * x.shape[0]
        else:
            keys = jrandom.split(key, x.shape[0])

        _, output = jax.lax.scan(scan_fn, state, (x, keys))

        return output
