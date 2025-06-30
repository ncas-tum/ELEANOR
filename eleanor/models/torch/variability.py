from typing import Sequence

import torch


def update_d2d_variability(model: torch.nn.Module, shape: Sequence[int]) -> None:
    for module in model.modules():
        if isinstance(module, D2DVar):
            module.update_variability(shape)


def update_d2d_variability_name(
    model: torch.nn.Module, name: str, shape: Sequence[int]
) -> None:
    for module in model.modules():
        if isinstance(module, D2DVar) and module.name == name:
            module.update_variability(shape)


def set_d2d_variability(
    model: torch.nn.Module, variability: float | Sequence[float]
) -> None:
    modules = []
    for module in model.modules():
        if isinstance(module, D2DVar):
            modules.append(module)

    if not hasattr(variability, "__len__"):
        variability = [variability] * len(modules)

    assert len(variability) == len(
        modules
    ), "Number of modules does not match the number of variabilities"

    for var, module in zip(variability, modules):
        module.set_variability(var)


def set_d2d_variability_name(
    model: torch.nn.Module, name: str, variability: float | Sequence[float]
) -> None:
    modules = []
    for module in model.modules():
        if isinstance(module, D2DVar) and module.name == name:
            modules.append(module)

    if not hasattr(variability, "__len__"):
        variability = [variability] * len(modules)

    assert len(variability) == len(
        modules
    ), "Number of modules does not match the number of variabilities"

    for var, module in zip(variability, modules):
        module.set_variability(var)


class D2DVar(torch.nn.Module):

    def __init__(self, name: str, variability: float):
        super(D2DVar, self).__init__()

        self.name = name
        self.variability = variability
        self.register_buffer("_rval", torch.zeros(0), False)

    def update_variability(self, shape: Sequence[int]) -> None:
        """
        Update the D2D variability array.

        Parameters
        ----------
        mu: torch.Tensor
            Mean value of the parameter to apply the variability.
        shape: Sequence[int]
            Output shape of the array.

        """
        self._rval = torch.randn(shape, device=self._rval.device)

    def set_variability(self, variability: float) -> None:
        """
        Set the D2D variability percentage.

        Parameters
        ----------
        variability: float
            Variability percentage.

        """
        self.variability = variability

    def __call__(self, mu: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        """
        Apply D2D variability into the input parameter.

        Parameters
        ----------
        mu: torch.Tensor
            Mean value of the parameter to apply the variability.
        shape: Sequence[int]
            Output shape of the array.

        Returns
        -------
        torch.Tensor with coefficient of variation :math:`\\text{variability} = \\sigma/\\mu`
        """
        if not self._rval.shape == shape:
            self.update_variability(shape)

        return mu * (1 + self.variability * self._rval)


class C2CVar(torch.nn.Module):
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
    >>> param_with_variability = param_var(param)

    """

    def __init__(self, name: str, variability: float):
        super(C2CVar, self).__init__()

        self.name = name
        self.variability = variability

    def set_variability(self, variability: float) -> None:
        """
        Set the D2D variability percentage.

        Parameters
        ----------
        variability: float
            Variability percentage.

        """
        self.variability = variability

    def __call__(self, mu: torch.Tensor, shape: Sequence[int]) -> torch.Tensor:
        """
        Apply C2C variability into the input parameter.

        Parameters
        ----------
        mu: torch.Tensor
            Mean value of the parameter to apply the variability.
        shape: StateShape
            Output shape of the array.
        key: PRNGKey
            Key to generate random variability

        Returns
        -------
        Array with coefficient of variation :math:`\\text{variability} = \\sigma/\\mu`
        """
        return mu * (1 + self.variability * torch.randn(shape))
