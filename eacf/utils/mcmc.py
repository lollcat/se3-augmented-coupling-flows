from typing import Sequence

import jax
import jax.numpy as jnp
import blackjax



def get_samples_simple(log_prob_fn, key, n_vertices: int = 2, dim: int = 2, n_steps: int = 64, batch_size: int = 32,
                       step_sizes: Sequence[float] = (2.0,), n_warmup_steps: int = 1000, init_scale = 1.0):
    # Build the kernel
    kernels = [blackjax.rmh(log_prob_fn, sigma=step_size) for step_size in step_sizes]

    # Initialize the state
    initial_position = jax.random.normal(key, shape=(batch_size, n_vertices, dim)) * init_scale
    state = jax.vmap(kernels[0].init)(initial_position)

    # Iterate
    rng_key = jax.random.PRNGKey(0)

    def one_step(carry, xs):
        state = carry
        key = xs
        rng_key_batch = jax.random.split(key, batch_size)
        for kernel in kernels:
            state, _ = jax.vmap(kernel.step)(rng_key_batch, state)
        return state, state.position

    _, positions = jax.lax.scan(one_step, init=state, xs=jax.random.split(rng_key, n_steps + n_warmup_steps))
    positions = positions[n_warmup_steps:]  # discard warmup positions
    return jax.tree_map(lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])), positions)

