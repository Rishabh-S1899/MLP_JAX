import jax
import jax.numpy as jnp

def sample_normal_matrix(key, shape):

    arr=jax.random.normal(key,shape)
    return arr
    # TODO: return a jnp array of the given shape with i.i.d. N(0,1) samples drawn from key

# ── Step 004  sample_input_features ──
import jax
import jax.numpy as jnp

def sample_input_features(key, batch_size, num_features):
    """Sample a (batch_size, num_features) standard-normal feature batch."""
    arr= sample_normal_matrix(key,(batch_size,num_features))
    return arr
