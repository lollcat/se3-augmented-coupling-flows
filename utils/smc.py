"""Follows https://blackjax-devs.github.io/sampling-book/algorithms/TemperedSMC.html#id2. """
from typing import Callable

from functools import partial

import chex
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.stats import multivariate_normal

import blackjax
import blackjax.smc.resampling as resampling

def smc_inference_loop(rng_key, smc_kernel, initial_state):
    """Run the temepered SMC algorithm.

    We run the adaptive algorithm until the tempering parameter lambda reaches the value
    lambda=1.

    """

    def cond(carry):
        i, state, _k = carry
        return state.lmbda < 1

    def one_step(carry):
        i, state, k = carry
        k, subk = jax.random.split(k, 2)
        state, _ = smc_kernel(subk, state)
        return i + 1, state, k

    n_iter, final_state, _ = jax.lax.while_loop(
        cond, one_step, (0, initial_state, rng_key)
    )

    return n_iter, final_state

def prior_log_prob(x: chex.Array, scale: float = 1.):
    d = x.shape[0]
    return multivariate_normal.logpdf(x, jnp.zeros((d,)), jnp.eye(d)*scale)


def run_smc_molecule(
        target_log_prob: Callable,
        key: chex.PRNGKey,
        dim: int,
        n_nodes: int,
        n_samples: int,
        num_mcmc_steps: int,
        base_scale: float = 1.,
        target_ess: float = 0.5,
        hmc_step_size: float = 1e-4):
    def flatten(x: chex.Array) -> chex.Array:
        return jnp.reshape(x, (*x.shape[:-2], dim*n_nodes))

    def unflatten(x: chex.Array) -> chex.Array:
        return jnp.reshape(x, (*x.shape[:-1], n_nodes, dim))

    def flat_log_prob_fn(x: chex.Array) -> chex.Array:
        """Takes in flat x."""
        return target_log_prob(unflatten(x))

    flat_dim = dim * n_nodes
    smc_samples = run_smc(
        flat_log_prob_fn,
        key,
        flat_dim,
        n_samples,
        num_mcmc_steps,
        target_ess=target_ess,
        base_scale=base_scale,
        hmc_step_size=hmc_step_size)

    samples = unflatten(smc_samples.particles)
    return samples, smc_samples.weights, smc_samples.lmbda


def run_smc(
        target_log_prob: Callable,
        key: chex.PRNGKey,
        dim: int,
        n_samples: int,
        num_mcmc_steps: int,
        base_scale: float = 1.,
        target_ess: float = 0.5,
        hmc_step_size: float = 1e-4):

    inv_mass_matrix = jnp.eye(dim)

    hmc_parameters = dict(
        step_size=hmc_step_size, inverse_mass_matrix=inv_mass_matrix, num_integration_steps=1
    )

    base_log_prob = partial(prior_log_prob, scale=base_scale)
    tempered = blackjax.adaptive_tempered_smc(
        base_log_prob,
        target_log_prob,
        blackjax.hmc.kernel(),
        blackjax.hmc.init,
        hmc_parameters,
        resampling_fn=resampling.systematic,
        target_ess=target_ess,
        num_mcmc_steps=num_mcmc_steps,
    )

    initial_smc_state = jax.random.multivariate_normal(
        jax.random.PRNGKey(0), jnp.zeros([dim,]), jnp.eye(dim), (n_samples,)
    ) * base_scale
    initial_smc_state = tempered.init(initial_smc_state)

    n_iter, smc_samples = smc_inference_loop(key, tempered.step, initial_smc_state)
    return smc_samples



if __name__ == '__main__':
    key = jax.random.PRNGKey(0)


    n_samples = 10_000


    def V(x):
        return 5 * jnp.square(jnp.sum(x ** 2) - 1)

    loglikelihood = lambda x: -V(x)

    smc_samples = run_smc(target_log_prob=loglikelihood, dim=1, n_samples=1000, key=key, n_nodes=1)

    samples = np.array(smc_samples.particles[:, 0])
    _ = plt.hist(samples, bins=100, density=True)


    linspace = jnp.linspace(-2, 2, 5000).reshape(-1, 1)
    lambdas = jnp.linspace(0.0, 1.0, 5)
    prior_logvals = jnp.vectorize(prior_log_prob, signature="(d)->()")(linspace)
    potential_vals = jnp.vectorize(V, signature="(d)->()")(linspace)
    log_res = prior_logvals.reshape(1, -1) - jnp.expand_dims(
        lambdas, 1
    ) * potential_vals.reshape(1, -1)

    density = jnp.exp(log_res)
    normalizing_factor = jnp.sum(density, axis=1, keepdims=True) * (
            linspace[1] - linspace[0]
    )
    density /= normalizing_factor

    _ = plt.plot(linspace.squeeze(), density[-1])
    plt.show()


