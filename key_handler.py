import numpy as np

# ── Step 001  make_prng_key ──
import jax
import jax.numpy as jnp

def make_prng_key(seed):
    key_val= jax.random.PRNGKey(seed)
    return key_val

# ── Step 002  split_prng_key ──
import jax
import jax.numpy as jnp
def split_prng_key(key, num):
    if(num ==1 ): return jnp.array([key])
    if(num==2):
        key1, key2 = jax.random.split(key)
        return jnp.array([key1,key2])
    ans=[]    
    for i in range(num-1):
        var_key = ans[-1] if len(ans)!=0 else key
        key1,key2= jax.random.split(var_key)
        ans.append(key1)
        ans.append(key2)
    if(num%2): ans.pop()
    return jnp.array(ans)