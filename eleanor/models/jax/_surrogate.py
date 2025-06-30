import jax
import jax.numpy as jnp


def tau_surr(alpha: float = 1.3, E_a: float = 1.0, soft_E: float = 1e-18):
    @jax.custom_gradient
    def surrogate(E, tau_p):
        exponential = (E_a / (jnp.abs(E) + soft_E)) ** alpha

        tau = 1 / (tau_p * jnp.exp(exponential))

        # Tau_p gradient
        grad_tau_p = -jnp.exp(-exponential) / (tau_p**2)

        # E gradient
        numerator = alpha * E * jnp.exp(-exponential) * exponential
        denumerator = soft_E * tau_p * jnp.abs(E) + E**2 * tau_p
        denumerator = jnp.where(
            jnp.abs(E) > 0.0, denumerator, 1.0  # If E is 0 then the numerator is also 0
        )
        grad_E = numerator / denumerator

        return tau, lambda g: (g * grad_E, g * grad_tau_p)

    return surrogate
