"""Utilities for JAX pseudo-random number generation."""

import jax
import jax.numpy as jnp


def make_prng_key(seed: int) -> jax.Array:
    """Create a reproducible JAX pseudo-random key."""
    return jax.random.PRNGKey(seed)


def split_prng_key(key: jax.Array, num: int) -> jax.Array:
    """Split ``key`` into ``num`` independent keys."""
    if num < 1:
        raise ValueError("num must be at least 1")
    return jax.random.split(key, num)


def sample_normal_matrix(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Draw an array of independent standard-normal samples."""
    return jax.random.normal(key, shape)
