from typing import Generic, TypeVar, Sequence
from dataclasses import dataclass

import jax
import equinox as eqx
import jax.random as jrand
import jax.tree_util as jtu
from chex import Array, PRNGKey
from snnax.snn.composed import StateShape

T = TypeVar("T")


@dataclass(frozen=True)
class StaticWrapper:
    content: T

    def __call__(self, *args, **kwargs):
        return self.content(*args, **kwargs)

    def tree_flatten(self):
        return ((self.content,), {"static": True})

    @classmethod
    def tree_unflatten(cls, metadata, children):
        return cls(children[0])


jtu.register_pytree_node(
    StaticWrapper, StaticWrapper.tree_flatten, StaticWrapper.tree_unflatten
)

# # Register wrapper as a leaf
# jtu.register_pytree_node(
#     StaticWrapper,
#     lambda x: ((), x.content),  # No children, everything in aux_data
#     lambda content, _: StaticWrapper(content),
# )


def find_all_D2D_wrappers_name(model, name):
    wrappers = []
    leaves = jtu.tree_leaves(model, is_leaf=lambda x: isinstance(x, StaticWrapper))
    for leaf in leaves:
        if (
            isinstance(leaf, StaticWrapper)
            and isinstance(leaf.content, D2DVar)
            and leaf.content.name == name
        ):
            wrappers.append(leaf)
    return wrappers


def find_all_D2D_wrappers(model):
    wrappers = []
    leaves = jtu.tree_leaves(model, is_leaf=lambda x: isinstance(x, StaticWrapper))
    for leaf in leaves:
        if isinstance(leaf, StaticWrapper) and isinstance(leaf.content, D2DVar):
            wrappers.append(leaf)
    return wrappers


def update_d2d_variability(model: Array, key: PRNGKey) -> Array:
    """
    Update the device to device variability of all
    the D2DVar parameters with a new key.

    Parameters
    ==========
    model: Array
        PyTree of the model with the parameters that want to update the D2D random key.
    key: PRNGKey
        New random key for the D2D variables.

    Returns
    =======
    Model with the update parameters.
    """

    old_wrappers = find_all_D2D_wrappers(model)
    keys = jrand.split(key, len(old_wrappers))

    # Create mapping using object identity
    wrapper_map = {}
    for old_wrapper, new_key in zip(old_wrappers, keys):
        new_noise = jrand.normal(new_key, old_wrapper.content.shape)
        new_x = eqx.tree_at(lambda x: x.noise, old_wrapper.content, new_noise)
        wrapper_map[id(old_wrapper)] = StaticWrapper(new_x)

    # Replace in the entire tree
    def replace_fn(node):
        if isinstance(node, StaticWrapper) and id(node) in wrapper_map:
            return wrapper_map[id(node)]
        return node

    return jtu.tree_map(
        replace_fn, model, is_leaf=lambda x: isinstance(x, StaticWrapper)
    )


def update_d2d_variability_name(model: Array, name: str, key: PRNGKey) -> Array:
    """
    Update the device to device variability of all
    the D2DVar parameters with the same name with a new key.

    Parameters
    ==========
    model: Array
        PyTree of the model with the parameters that want to update the D2D random key.
    name: Array
        Name of the parameter to update.
    key: PRNGKey
        New random key for the D2D variables.

    Returns
    =======
    Model with the update parameters.

    """

    old_wrappers = find_all_D2D_wrappers_name(model, name)
    keys = jrand.split(key, len(old_wrappers))

    # Create mapping using object identity
    wrapper_map = {}
    for old_wrapper, new_key in zip(old_wrappers, keys):
        new_noise = jrand.normal(new_key, old_wrapper.content.shape)
        new_x = eqx.tree_at(lambda x: x.noise, old_wrapper.content, new_noise)
        wrapper_map[id(old_wrapper)] = StaticWrapper(new_x)

    # Replace in the entire tree
    def replace_fn(node):
        if isinstance(node, StaticWrapper) and id(node) in wrapper_map:
            return wrapper_map[id(node)]
        return node

    return jtu.tree_map(
        replace_fn, model, is_leaf=lambda x: isinstance(x, StaticWrapper)
    )


def set_d2d_variability(model, variability: float | Sequence[float]):
    old_wrappers = find_all_D2D_wrappers(model)

    if len(old_wrappers) != len(variability):
        raise ValueError(
            f"Found {len(old_wrappers)} D2DVar instances but got {len(variability)} variability values"
        )

    # Create mapping using object identity
    wrapper_map = {}
    for old_wrapper, new_var in zip(old_wrappers, variability):
        new_x = eqx.tree_at(lambda x: x.variability, old_wrapper.content, new_var)
        wrapper_map[id(old_wrapper)] = StaticWrapper(new_x)

    # Replace in the entire tree
    def replace_fn(node):
        if isinstance(node, StaticWrapper) and id(node) in wrapper_map:
            return wrapper_map[id(node)]
        return node

    return jtu.tree_map(
        replace_fn, model, is_leaf=lambda x: isinstance(x, StaticWrapper)
    )


def set_d2d_variability_name(
    model: Array, name: str, variability: float | Sequence[float]
) -> Array:
    old_wrappers = find_all_D2D_wrappers_name(model, name)

    if len(old_wrappers) != len(variability):
        raise ValueError(
            f"Found {len(old_wrappers)} D2DVar instances but got {len(variability)} variability values"
        )

    # Create mapping using object identity
    wrapper_map = {}
    for old_wrapper, new_var in zip(old_wrappers, variability):
        new_x = eqx.tree_at(lambda x: x.variability, old_wrapper.content, new_var)
        wrapper_map[id(old_wrapper)] = StaticWrapper(new_x)

    # Replace in the entire tree
    def replace_fn(node):
        if isinstance(node, StaticWrapper) and id(node) in wrapper_map:
            return wrapper_map[id(node)]
        return node

    return jtu.tree_map(
        replace_fn, model, is_leaf=lambda x: isinstance(x, StaticWrapper)
    )


class D2DVar(eqx.Module):
    """Device to device variability for eleanor models
    Apply a percentage of variability to a parameter of the models

    Attributes
    ==========
    name: str
        Name of the variability parameter.
    variability: float
        Percentage of variability.
    key: PRNGKey
        Random key to generate the variability.

    Example
    -------
    >>> param_var = D2DVar("param", 0.1, key)
    >>> param_with_variability = param_var(param)
    """

    name: str | None = eqx.field(static=True)
    shape: Sequence[int] = eqx.field(static=True)
    variability: float
    noise: Array

    def __init__(self, name, variability, shape, key):
        self.name = name
        self.shape = shape
        self.variability = variability
        self.noise = jrand.normal(key, shape)

    @jax.named_scope("eleanor.models.D2DVar")
    def __call__(self, mu: Array, *, key: PRNGKey = None) -> Array:
        """
        Apply D2D variability into the input parameter.
        key parameter mantained for compatibility.

        Parameters
        ----------
        mu: Array
            Mean value of the parameter to apply the variability.
        shape: StateShape
            Output shape of the array.

        Returns
        -------
        Array with coefficient of variation :math:`\\text{variability} = \\sigma/\\mu`
        """
        return mu * (1 + self.variability * self.noise)


class C2CVar(eqx.Module):
    """Cycle to cycle variability for eleanor models
    Apply a percentage of variability to a parameter of the models on evey call

    Attributes
    ==========
    name: str
        Name of the variability parameter.
    variability: float
        Percentage of variability.

    Example
    -------
    >>> param_var = C2CVar("param", 0.1)
    >>> param_with_variability = param_var(param, key=key)

    """

    name: str | None
    variability: float

    @jax.named_scope("eleanor.models.D2DVarParam")
    def __call__(self, mu: Array, shape: StateShape, *, key: PRNGKey) -> Array:
        """
        Apply C2C variability into the input parameter.

        Parameters
        ----------
        mu: Array
            Mean value of the parameter to apply the variability.
        shape: StateShape
            Output shape of the array.
        key: PRNGKey
            Key to generate random variability

        Returns
        -------
        Array with coefficient of variation :math:`\\text{variability} = \\sigma/\\mu`
        """
        return mu * (1 + self.variability * jrand.normal(key, shape))
