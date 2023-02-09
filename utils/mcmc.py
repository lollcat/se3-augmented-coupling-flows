from typing import Sequence

import jax
import jax.numpy as jnp
import blackjax



def get_samples_with_tuning(log_prob_fn, key, n_vertices: int = 2, dim: int = 2, n_steps: int = 64,
                            batch_size: int = 32,
                            burn_in: int = 1000, algorithm_type: str = "hmc"):
    """Get samples tuning the mcmc params. Mostly copied from
    https://blackjax-devs.github.io/blackjax/examples/change_of_variable_hmc.html"""

    if algorithm_type == "nuts":
        algorithm = blackjax.nuts
        kwargs = {}
    else:
        algorithm = blackjax.hmc
        assert algorithm_type == "hmc"
        kwargs = {"num_integration_steps": jnp.array(5)}

    # Initialize the state
    initial_position = jax.random.normal(key, shape=(batch_size, n_vertices, dim))

    warmup = blackjax.window_adaptation(algorithm, log_prob_fn, initial_step_size=1e-3, progress_bar=True,
                                        target_acceptance_rate=0.8, is_mass_matrix_diagonal=True, **kwargs)

    @jax.vmap
    def call_warmup(seed, param):
        initial_states, _, tuned_params = warmup.run(seed, param, burn_in)
        return initial_states, tuned_params

    key, subkey = jax.random.split(key)
    rng_key_batch = jax.random.split(subkey, batch_size)
    initial_states, tuned_params = jax.jit(call_warmup)(rng_key_batch, initial_position)

    def inference_loop_multiple_chains(
            rng_key, initial_states, tuned_params, log_prob_fn, num_samples, num_chains
    ):
        step_fn = algorithm.kernel()

        def kernel(key, state, **params):
            return step_fn(key, state, log_prob_fn, **params, **kwargs)

        def one_step(states, rng_key):
            keys = jax.random.split(rng_key, num_chains)
            [tuned_params.pop(key) for key in kwargs.keys()]
            states, infos = jax.vmap(kernel)(keys, states, **tuned_params)
            return states, (states, infos)

        keys = jax.random.split(rng_key, num_samples)
        _, (states, infos) = jax.lax.scan(one_step, initial_states, keys)

        return (states, infos)


    states, infos = inference_loop_multiple_chains(
        key, initial_states, tuned_params, log_prob_fn, n_steps, batch_size
    )
    return jax.tree_map(lambda x: jnp.reshape(x, (x.shape[0] * x.shape[1], *x.shape[2:])), states.position)



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


