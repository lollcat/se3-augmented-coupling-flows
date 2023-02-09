import jax.numpy as jnp
import numpy as np
import jax
import chex
from functools import partial

from utils.numerical import get_pairwise_distances
from utils.mcmc import get_samples_simple

def energy(x, a = 0.0, b = -4., c = 0.9, d0 = 4.0, tau = 1.0):
    """Compute energy. Default hyper-parameters from https://arxiv.org/pdf/2006.02425.pdf.
    If we want to add conditioning info we could condition on the parameters a,b,c,d,tau. """
    differences = get_pairwise_distances(x)
    diff_minus_d0 = differences - d0
    return jnp.sum(a*diff_minus_d0 + b*diff_minus_d0**2 + c*diff_minus_d0**4, axis=(-1, -2)) / tau / 2


def log_prob_fn(x: chex.Array, temperature=1.0):
    if len(x.shape) == 2:
        return - energy(x, tau=temperature)
    elif len(x.shape) == 3:
        return - jax.vmap(partial(energy, tau=temperature))(x)
    else:
        raise Exception


def make_dataset(seed: int = 0, n_vertices=4, dim=2, n_samples: int = 10000, temperature: float = -1.,
                 n_warmup_steps=10000, step_sizes=(5.0, 1.0, 0.2, 0.1, 0.1, 0.1, 0.05), save=False):
    batch_size = 64
    key = jax.random.PRNGKey(seed)
    samples = get_samples_simple(partial(log_prob_fn, temperature=temperature),
                                 key, n_vertices, dim, n_samples // batch_size, batch_size,
                                 step_sizes=step_sizes, n_warmup_steps=n_warmup_steps,
                                 init_scale=10)
    if save:
        np.save(f"data/dw_data_vertices{n_vertices}_dim{dim}_temperature{temperature}.npy", np.asarray(samples))
    return samples



def plot_sample_hist(samples, ax = None, dims=(0,1)):
    if ax == None:
        fig, ax = plt.subplots()
    d = jnp.linalg.norm(samples[:, 0, dims] - samples[:, 1, dims], axis=-1)
    ax.hist(d, bins=50, density=True, alpha=0.4)



if __name__ == '__main__':
    USE_64_BIT = False
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)

    # Visualise 2D energy fn as a function of distance
    import matplotlib.pyplot as plt

    dim = 2
    batch_size = 512
    x0 = jax.random.normal(jax.random.PRNGKey(0), (batch_size, 2))  #  jnp.zeros((batch_size, 2)) + 0.1
    d = jnp.linspace(0.7, 7.5, batch_size)
    x1 = x0 + jnp.sqrt(d**2/2)[:, None]

    x = jnp.stack([x0, x1], axis=1)
    log_probs = log_prob_fn(x)

    print(jax.grad(log_prob_fn)(x[0]))

    plt.plot(d, log_probs)
    plt.show()

    fig, ax = plt.subplots()
    ax.plot(d, jnp.exp(log_probs + 162))  # approx normalise
    # plt.show()

    key = jax.random.PRNGKey(0)
    samples = get_samples_simple(log_prob_fn, key, n_steps=100, batch_size=128, step_sizes=(0.5,), n_warmup_steps=1000,
                                 init_scale=0.1)
    plot_sample_hist(samples, ax=ax)
    plt.show()


    fig, ax = plt.subplots()
    samples = make_dataset(n_vertices=4, temperature=0.1)
    plot_sample_hist(samples, ax=ax)
    plt.title("dataset samples")
    plt.show()




