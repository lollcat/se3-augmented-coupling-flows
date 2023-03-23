from typing import Union

import chex
import jax.numpy as jnp
import jax
import numpy as np
import matplotlib.pyplot as plt

from utils.numerical import get_pairwise_distances, set_diagonal_to_zero
from utils.mcmc import get_samples_simple, get_samples_with_tuning

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
    batch_size = 128
    key = jax.random.PRNGKey(seed)
    samples = get_samples_simple(log_prob_fn=log_prob_fn,
                                 key=key, n_vertices=n_vertices, dim=dim, n_steps=n_samples // batch_size,
                                 batch_size=batch_size,
                                 n_warmup_steps=20000,
                                 step_sizes=(0.025,))
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

    fig, ax = plt.subplots()  # 13 nodes, 3D.
    samples = get_samples_with_tuning(log_prob_fn, key, n_vertices= 13, dim = 3, n_steps = 64,
                            batch_size=32, burn_in=1000, algorithm_type = "hmc")
    plot_sample_hist(samples, ax=ax)
    plt.show()


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

    samples = get_samples_simple(log_prob_fn, key, n_steps=2, batch_size=1028, n_warmup_steps=20000, step_sizes=(0.025,),
                                 init_scale=0.1)
    plot_sample_hist(samples, ax=ax)
    plt.show()

    # make_dataset(dim=3, n_vertices=13)
