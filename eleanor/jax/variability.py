from collections.abc import Sequence
from dataclasses import dataclass
from typing import Generic, TypeVar

import equinox as eqx
import jax
import jax.random as jrand
import jax.tree_util as jtu
from jaxtyping import Array, PRNGKeyArray, PyTree

T = TypeVar("T")


@dataclass(frozen=True)
class StaticWrapper(Generic[T]):
    """
    Wrapper for JAX traceable pytrees that are expected to be fixed.

    Attributes
    ----------
    content: T
        PyTree to be fixed.
    """

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


def find_all_D2D_wrappers_name(model: PyTree, name: str) -> list[PyTree]:
    """
    Find all the StaticWrappers with Device to Device variability
    based on the name `name`

    Parameters
    ==========
    model: PyTree
        PyTree of the model with the D2D variability.
    name: str
        Name of the D2D parameter.

    Returns
    =======
    List of all the StaticWrappers with D2D objects with a given name.

    """
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


def find_all_D2D_wrappers(model: PyTree) -> list[PyTree]:
    """
    Find all the StaticWrappers with Device to Device variability.

    Parameters
    ==========
    model: PyTree
        PyTree of the model with the D2D variability.

    Returns
    =======
    List of all the StaticWrappers with D2D objects.
    """
    wrappers = []
    leaves = jtu.tree_leaves(model, is_leaf=lambda x: isinstance(x, StaticWrapper))
    for leaf in leaves:
        if isinstance(leaf, StaticWrapper) and isinstance(leaf.content, D2DVar):
            wrappers.append(leaf)
    return wrappers


def update_d2d_variability(model: PyTree, key: PRNGKeyArray) -> PyTree:
    """
    Update the device to device variability of all
    the D2DVar parameters with a new key.

    Parameters
    ==========
    model: PyTree
        PyTree of the model with the parameters that want to update the D2D random key.
    key: PRNGKeyArray
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


def update_d2d_variability_name(model: PyTree, name: str, key: PRNGKeyArray) -> PyTree:
    """
    Update the device to device variability of all
    the D2DVar parameters with the same name with a new key.

    Parameters
    ==========
    model: Array
        PyTree of the model with the parameters that want to update the D2D random key.
    name: str
        Name of the parameter to update.
    key: PRNGKeyArray
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


def set_d2d_variability(model: PyTree, variability: float | Sequence[float]) -> PyTree:
    """
    Set a new variability value to all the D2D objects of a model

    Parameters
    ==========
    model: PyTree
        Model that contains the D2D objects.
    variability: float | Sequence[float]
        New variability value/s.
    """
    old_wrappers = find_all_D2D_wrappers(model)

    if isinstance(variability, Sequence):
        if len(old_wrappers) != len(variability):
            raise ValueError(
                f"Found {len(old_wrappers)} D2DVar instances but got {len(variability)} variability values"
            )
    else:
        variability = [variability] * len(old_wrappers)

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
    model: PyTree, name: str, variability: float | Sequence[float]
) -> PyTree:
    """
    Set a new variability value to all the D2D objects of a model with
    a given `name`

    Parameters
    ==========
    model: PyTree
        Model that contains the D2D objects.
    name: str
        Name of the D2D parameter.
    variability: float | Sequence[float]
        New variability value/s.
    """
    old_wrappers = find_all_D2D_wrappers_name(model, name)

    if isinstance(variability, Sequence):
        if len(old_wrappers) != len(variability):
            raise ValueError(
                f"Found {len(old_wrappers)} D2DVar instances but got {len(variability)} variability values"
            )
    else:
        variability = [variability] * len(old_wrappers)

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
    key: PRNGKeyArray
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
    def __call__(self, mu: Array, *, key: PRNGKeyArray | None = None) -> Array:
        """
        Apply D2D variability into the input parameter.
        key parameter mantained for compatibility.

        Parameters
        ----------
        mu: Array
            Mean value of the parameter to apply the variability.

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

    @jax.named_scope("eleanor.models.C2CVarParam")
    def __call__(self, mu: Array, shape: Sequence[int], *, key: PRNGKeyArray) -> Array:
        """
        Apply C2C variability into the input parameter.

        Parameters
        ----------
        mu: Array
            Mean value of the parameter to apply the variability.
        shape: Sequence[int]
            Output shape of the array.
        key: PRNGKey
            Key to generate random variability

        Returns
        -------
        Array with coefficient of variation :math:`\\text{variability} = \\sigma/\\mu`
        """
        return mu * (1 + self.variability * jrand.normal(key, shape))
