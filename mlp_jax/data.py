"""Synthetic-data helpers used by the MLP example."""

import jax
import jax.numpy as jnp

from .random import sample_normal_matrix


def sample_input_features(
    key: jax.Array, batch_size: int, num_features: int
) -> jax.Array:
    """Sample a standard-normal feature batch."""
    return sample_normal_matrix(key, (batch_size, num_features))


def assign_class_labels(inputs: jax.Array, num_classes: int) -> jax.Array:
    """Assign each example to the largest of its first class features."""
    if inputs.ndim != 2:
        raise ValueError("inputs must be a two-dimensional batch")
    if not 1 <= num_classes <= inputs.shape[1]:
        raise ValueError("num_classes must be between 1 and the feature count")
    return jnp.argmax(inputs[:, :num_classes], axis=1).astype(jnp.int32)


def one_hot_encode_labels(labels: jax.Array, num_classes: int) -> jax.Array:
    """Convert integer labels to one-hot encoded target vectors."""
    return jax.nn.one_hot(labels, num_classes)
