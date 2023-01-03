import chex
import jax.numpy as jnp
import jax
import blackjax

def get_pairwise_distances(x):
    chex.assert_rank(x, 2)
    return jnp.linalg.norm(x - x[:, None] + 1e-10, ord=2, axis=-1)  # Add 1e-10 to prevent nans

def energy(x, a = 0.0, b = -4., c = 0.9, d0 = 4.0, tau = 1.0):
    """Compute energy. Default hyper-parameters from https://arxiv.org/pdf/2006.02425.pdf"""
    differences = get_pairwise_distances(x)
    diff_minus_d0 = differences - d0
    return jnp.sum(a*diff_minus_d0 + b*diff_minus_d0**2 + c*diff_minus_d0**4, axis=(-1, -2)) / tau / 2


def log_prob_fn(x):
    if len(x.shape) == 2:
        return - energy(x)
    elif len(x.shape) == 3:
        return - jax.vmap(energy)(x)
    else:
        raise Exception


def get_samples(key, n_vertices=2, dim=2, n_steps: int = 64, batch_size=32, burn_in=10):
    # Build the kernel
    step_size = 1e-3
    inverse_mass_matrix = jnp.ones(n_vertices*dim)
    nuts = blackjax.nuts(log_prob_fn, step_size, inverse_mass_matrix)

    # Initialize the state
    initial_position = jax.random.normal(key, shape=(batch_size, n_vertices, dim))
    state = jax.vmap(nuts.init)(initial_position)

    # Iterate
    rng_key = jax.random.PRNGKey(0)
    samples = []
    for i in range(int(n_steps + burn_in)):
        print(i)
        _, rng_key = jax.random.split(rng_key)
        rng_key_batch = jax.random.split(rng_key, batch_size)
        state, _ = jax.jit(jax.vmap(nuts.step))(rng_key_batch, state)
        if i >= burn_in:
            samples.append(state.position)
    return jnp.concatenate(samples, axis=0)


def make_dataset(n_vertices=2, dim=2, n_steps: int = int(1e3)):
    pass


if __name__ == '__main__':
    # Visualise 2D energy fn as a function of distance
    import matplotlib.pyplot as plt

    dim = 2
    batch_size = 50
    x0 = jnp.zeros((batch_size, 2))
    d = jnp.linspace(0.7, 5, batch_size)
    x1 = x0 + d[:, None]

    x = jnp.stack([x0, x1], axis=1)
    log_probs = log_prob_fn(x)

    grad_log_prob = jax.jacfwd(log_prob_fn)(x)
    print(grad_log_prob)

    plt.plot(d, log_probs)
    plt.show()

    key = jax.random.PRNGKey(0)
    samples = get_samples(key)
    d = jnp.linalg.norm(samples[:, 0, :] - samples[:, 1, :], axis=-1)
    plt.hist(d, bins=50, density=True)
    plt.show()


