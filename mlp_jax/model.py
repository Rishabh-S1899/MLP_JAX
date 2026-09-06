"""MLP parameter initialization and inference."""

import jax
import jax.numpy as jnp

from .layers import init_linear_layer, linear_forward, relu_activation


def init_mlp_params(
    key: jax.Array, layer_sizes: list[int], scale: float = 0.1
) -> list[dict[str, jax.Array]]:
    """Initialize one dense layer for each adjacent pair in ``layer_sizes``."""
    if len(layer_sizes) < 2:
        raise ValueError("layer_sizes must contain an input and output size")
    keys = jax.random.split(key, len(layer_sizes) - 1)
    return [
        init_linear_layer(layer_key, in_dim, out_dim, scale)
        for layer_key, in_dim, out_dim in zip(keys, layer_sizes[:-1], layer_sizes[1:])
    ]


def softmax_probabilities(logits: jax.Array) -> jax.Array:
    """Convert logits to probabilities along the class axis."""
    return jax.nn.softmax(logits, axis=-1)


def mlp_forward(params: list[dict[str, jax.Array]], x: jax.Array) -> jax.Array:
    """Return output logits from a stack of dense ReLU layers."""
    activations = x
    for index, layer_params in enumerate(params):
        activations = linear_forward(activations, layer_params)
        if index < len(params) - 1:
            activations = relu_activation(activations)
    return activations


def predict_classes(params: list[dict[str, jax.Array]], x: jax.Array) -> jax.Array:
    """Predict the index of the most likely class for each example."""
    return jnp.argmax(mlp_forward(params, x), axis=-1)
