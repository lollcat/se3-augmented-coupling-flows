from typing import Union

import chex
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt

from utils.numerical import get_pairwise_distances, set_diagonal_to_zero
from utils.smc import run_smc_molecule

def energy(x: chex.Array, epsilon: float = 1.0, tau: float = 1.0, r: Union[float, chex.Array] = 1.0,
           harmonic_potential_coef: float = 0.3) -> chex.Array:
    chex.assert_rank(x, 2)
    if isinstance(r, float):
        r = jnp.ones(x.shape[:-1])
    d = get_pairwise_distances(x)
    term_inside_sum = (r[:, None] / d)**12 - 2*(r[:, None] / d)**6
    term_inside_sum = set_diagonal_to_zero(term_inside_sum)
    energy = epsilon / (2 * tau) * jnp.sum(term_inside_sum)
    harmonic_potential = jnp.sum(d) * harmonic_potential_coef
    return energy + harmonic_potential


def log_prob_fn(x: chex.Array):
    if len(x.shape) == 2:
        return - energy(x)
    elif len(x.shape) == 3:
        return - jax.vmap(energy)(x)
    else:
        raise Exception


def make_dataset(seed: int = 0, n_vertices=2, dim=2, n_samples: int = 8192):
    key = jax.random.PRNGKey(seed)
    samples = run_smc_molecule(target_log_prob=log_prob_fn,
                        dim=dim,
                        n_samples = n_samples,
                        base_scale = 5.,
                        hmc_step_size= 1e-4,
                        key=key)
    np.save(f"data/lj_data_vertices{n_vertices}_dim{dim}.npy", np.asarray(samples))



def plot_sample_hist(samples, ax = None, dim=(0,1)):
    if ax == None:
        fig, ax = plt.subplots()
    d = jnp.linalg.norm(samples[:, 0, dim] - samples[:, 1, dim], axis=-1)
    ax.hist(d, bins=50, density=True, alpha=0.4)


if __name__ == '__main__':
    USE_64_BIT = False
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)


    # Visualise 2D energy fn as a function of distance
    key = jax.random.PRNGKey(0)

    dim = 2
    batch_size = 512
    x0 = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 2))   #  jnp.zeros((batch_size, 2)) + 0.1
    d = jnp.linspace(0.8, 10, batch_size)
    x1 = x0 + jnp.sqrt(d**2/2)[:, None]

    x = jnp.stack([x0, x1], axis=1)
    log_probs = log_prob_fn(x)

    print(jax.grad(log_prob_fn)(x[0]))

    plt.plot(d, log_probs)
    plt.show()

    fig, ax = plt.subplots()  # 2D
    ax.plot(d, jnp.exp(log_probs - 1))  # approx normalise

    n_nodes = 13
    dim = 3
    samples, weights, lmbda = run_smc_molecule(target_log_prob=log_prob_fn,
                        dim=dim,
                        n_nodes=n_nodes,
                        key=key,
                        n_samples =1000,
                        num_mcmc_steps=30,
                        target_ess=0.0,
                        base_scale=2.,
                        hmc_step_size= 1e-3)

    plot_sample_hist(samples, ax=ax)
    plt.show()

    # make_dataset(dim=3, n_vertices=13)
