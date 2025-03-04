from typing import Sequence

import jax
import equinox as eqx
import jax.random as jrand
from chex import Array, PRNGKey

from snnax.snn.composed import StateShape


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

    def is_d2dvar(x):
        return isinstance(x, D2DVar)

    def get_keys(m):
        return [
            x.key
            for x in jax.tree_util.tree_leaves(m, is_leaf=is_d2dvar)
            if is_d2dvar(x)
        ]

    keys = get_keys(model)
    new_keys = jrand.split(key, len(keys))
    new_model = eqx.tree_at(get_keys, model, new_keys)
    return new_model


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

    def is_d2dvar(x):
        return isinstance(x, D2DVar) and x.name == name

    def get_keys(m):
        return [
            x.key
            for x in jax.tree_util.tree_leaves(m, is_leaf=is_d2dvar)
            if is_d2dvar(x)
        ]

    keys = get_keys(model)
    new_keys = jrand.split(key, len(keys))
    new_model = eqx.tree_at(get_keys, model, new_keys)
    return new_model


def set_d2d_variability(model: Array, variability: float | Sequence[float]) -> Array:
    def is_d2dvar(x):
        return isinstance(x, D2DVar)

    def get_var(m):
        return [
            x.variability
            for x in jax.tree_util.tree_leaves(m, is_leaf=is_d2dvar)
            if is_d2dvar(x)
        ]

    if not hasattr(variability, "__len__"):
        variability = [variability] * len(get_var(model))

    new_model = eqx.tree_at(get_var, model, variability)
    return new_model


def set_d2d_variability_name(
    model: Array, name: str, variability: float | Sequence[float]
) -> Array:
    def is_d2dvar_name(x):
        return isinstance(x, D2DVar) and x.name == name

    def is_d2dvar(x):
        return isinstance(x, D2DVar)

    # Filter only the model with the attribute A
    model_filtered = eqx.filter(model, is_d2dvar_name, is_leaf=is_d2dvar)

    def get_var(m):
        return [
            x.variability
            for x in jax.tree_util.tree_leaves(m, is_leaf=is_d2dvar)
            if is_d2dvar(x)
        ]

    # Replace the model with the attribute A with the new one
    if not hasattr(variability, "__len__"):
        variability = [variability] * len(get_var(model_filtered))

    new_model = eqx.tree_at(get_var, model_filtered, variability)

    # Put back the rest of the model
    new_model = eqx.combine(new_model, model, is_leaf=is_d2dvar)
    return new_model


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

    name: str | None
    variability: float
    key: PRNGKey

    @jax.named_scope("eleanor.models.D2DVarParam")
    def __call__(self, mu: Array, shape: StateShape, *, key: PRNGKey = None) -> Array:
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
        return mu * (1 + self.variability * jrand.normal(self.key, shape))


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
