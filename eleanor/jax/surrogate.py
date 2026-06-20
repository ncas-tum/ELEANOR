import jax
import jax.numpy as jnp
from jaxtyping import Array, Float

from eleanor.jax.models._base import limexp


def _unbroadcast(grad, target_shape):
    """Sum-reduce ``grad`` so its shape matches ``target_shape``.

    Used inside ``custom_gradient`` bwd rules to undo broadcasting when one
    primal input was broadcast against another (e.g. scalar ``tau_p``
    broadcasting against array ``E``). Works for any combination of scalar
    and array inputs.
    """
    target_shape = tuple(target_shape)
    grad_shape = jnp.shape(grad)
    extra = len(grad_shape) - len(target_shape)
    if extra > 0:
        grad = jnp.sum(grad, axis=tuple(range(extra)))
        grad_shape = jnp.shape(grad)
    sum_axes = tuple(
        i for i, (g, t) in enumerate(zip(grad_shape, target_shape)) if t == 1 and g != 1
    )
    if sum_axes:
        grad = jnp.sum(grad, axis=sum_axes, keepdims=True)
    return grad


def tau_surr(alpha: float = 1.3, E_a: float = 1.0, soft_E: float = 1e-18):
    @jax.custom_gradient
    def surrogate(E, tau_p):
        E_shape = jnp.shape(E)
        tau_p_shape = jnp.shape(tau_p)

        exponential = (E_a / (jnp.abs(E) + soft_E)) ** alpha

        tau = 1 / (tau_p * limexp(exponential))

        # Tau_p gradient
        grad_tau_p = -limexp(-exponential) / (tau_p**2)

        # E gradient
        numerator = alpha * E * limexp(-exponential) * exponential
        denumerator = soft_E * tau_p * jnp.abs(E) + E**2 * tau_p
        denumerator = jnp.where(
            jnp.abs(E) > 0.0,
            denumerator,
            1.0,  # If E is 0 then the numerator is also 0
        )
        grad_E = numerator / denumerator

        def bwd(g):
            return (
                _unbroadcast(g * grad_E, E_shape),
                _unbroadcast(g * grad_tau_p, tau_p_shape),
            )

        return tau, bwd

    return surrogate


@jax.custom_gradient
def atan_surrogate(x: Float[Array, "..."]):
    """Surrogate gradient function based on the arctangent."""

    y = jnp.heaviside(x, 1.0)

    def grad(dy):
        alpha = 2.0
        dx = alpha / 2 / (1 + (jnp.pi / 2 * alpha * x) ** 2) * dy
        return (dx,)

    return y, grad


@jax.custom_gradient
def tanh_surrogate(x: Float[Array, "..."]):
    """Surrogate gradient function based on the hyperbolic tangent."""

    y = jnp.heaviside(x, 1.0)

    def grad(dy):
        dx = dy * (1.0 - jnp.tanh(x) ** 2)
        return (dx,)

    return y, grad
