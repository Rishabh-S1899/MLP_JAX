"""Dense-layer parameter initialization and forward propagation."""

import jax
import jax.numpy as jnp

from .random import sample_normal_matrix


def init_linear_layer(
    key: jax.Array, in_dim: int, out_dim: int, scale: float = 0.1
) -> dict[str, jax.Array]:
    """Initialize parameters for a dense layer."""
    return {
        "W": scale * sample_normal_matrix(key, (in_dim, out_dim)),
        "b": jnp.zeros((out_dim,)),
    }


def linear_forward(x: jax.Array, layer_params: dict[str, jax.Array]) -> jax.Array:
    """Apply one dense layer to a batch of examples."""
    return x @ layer_params["W"] + layer_params["b"]


def relu_activation(x: jax.Array) -> jax.Array:
    """Apply ReLU elementwise."""
    return jnp.maximum(x, 0)
