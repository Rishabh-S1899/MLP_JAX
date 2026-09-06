"""Loss, evaluation, gradient, and SGD training operations."""

import jax
import jax.numpy as jnp

from .model import mlp_forward


def log_softmax_logits(logits: jax.Array) -> jax.Array:
    """Compute a numerically stable log-softmax over classes."""
    return jax.nn.log_softmax(logits, axis=-1)


def cross_entropy_loss(logits: jax.Array, one_hot_targets: jax.Array) -> jax.Array:
    """Return the mean categorical cross-entropy loss."""
    return -jnp.mean(jnp.sum(log_softmax_logits(logits) * one_hot_targets, axis=-1))


def classification_accuracy(logits: jax.Array, labels: jax.Array) -> jax.Array:
    """Return the fraction of correctly classified examples."""
    return jnp.mean(jnp.argmax(logits, axis=-1) == labels)


def loss_fn_of_params(
    params: list[dict[str, jax.Array]], x: jax.Array, one_hot_targets: jax.Array
) -> jax.Array:
    """Evaluate cross-entropy for model parameters and a data batch."""
    return cross_entropy_loss(mlp_forward(params, x), one_hot_targets)


def compute_param_grads(
    params: list[dict[str, jax.Array]], x: jax.Array, one_hot_targets: jax.Array
) -> list[dict[str, jax.Array]]:
    """Differentiate the loss with respect to every layer parameter."""
    return jax.grad(loss_fn_of_params)(params, x, one_hot_targets)


def sgd_update_params(
    params: list[dict[str, jax.Array]],
    grads: list[dict[str, jax.Array]],
    learning_rate: float,
) -> list[dict[str, jax.Array]]:
    """Apply a single stochastic-gradient-descent parameter update."""
    return [
        {
            "W": layer_params["W"] - learning_rate * layer_grads["W"],
            "b": layer_params["b"] - learning_rate * layer_grads["b"],
        }
        for layer_params, layer_grads in zip(params, grads)
    ]


def training_step(
    params: list[dict[str, jax.Array]],
    x: jax.Array,
    one_hot_targets: jax.Array,
    learning_rate: float,
) -> tuple[list[dict[str, jax.Array]], jax.Array]:
    """Calculate a batch loss and return parameters after one SGD update."""
    loss_value = loss_fn_of_params(params, x, one_hot_targets)
    grads = compute_param_grads(params, x, one_hot_targets)
    return sgd_update_params(params, grads, learning_rate), loss_value


def train_mlp(
    params: list[dict[str, jax.Array]],
    x: jax.Array,
    one_hot_targets: jax.Array,
    learning_rate: float,
    num_epochs: int,
) -> list[dict[str, jax.Array]]:
    """Run full-batch SGD for ``num_epochs`` and return final parameters."""
    for _ in range(num_epochs):
        params, _ = training_step(params, x, one_hot_targets, learning_rate)
    return params
