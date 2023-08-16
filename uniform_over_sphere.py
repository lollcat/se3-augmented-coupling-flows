from typing import Tuple
import chex
import jax.numpy as jnp
import jax
import distrax
import matplotlib.pyplot as plt

# def sample_and_log_prob_z1_t1_t2(key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
#     """
#     Represent uniform distribution on sphere by [z, t1].
#     We sample z \propto circumpherence(z) \propto sqrt(1 - z^2)
#     \int_{-1}_{1} sqrt(1 - z^2) dz = pi / 2
#     This gives p(z) = sqrt(1 - z^2) / (pi / 2)
#     And also sample uniformly on [-pi, pi] for the atom 2."""
#     key1, key2, key3 = jax.random.split(key, 3)
#
#     xyz = jax.random.normal(key1, shape=(3,))
#     z1 = xyz[-1] / jnp.linalg.norm(xyz)
#
#     p_z1 = jnp.sqrt(1 - z1**2) / (jnp.pi / 2)
#     log_p_z1 = jnp.log(p_z1)
#
#     t1, log_p_t1 = distrax.Uniform(low=-jnp.pi, high=jnp.pi).sample_and_log_prob(seed=key2)
#     t2, log_p_t2 = distrax.Uniform(low=-jnp.pi, high=jnp.pi).sample_and_log_prob(seed=key2)
#     return jnp.stack([z1, t1, t2]), log_p_z1 + log_p_t1 + log_p_t2


def sample_on_sphere(key: chex.PRNGKey, version1: bool = True) -> chex.Array:
    if version1:
        key1, key2, key3 = jax.random.split(key, 3)
        z1, log_p_z1 = distrax.Uniform(low=-1, high=1.).sample_and_log_prob(seed=key1)
        t1, log_p_t1 = distrax.Uniform(low=-jnp.pi, high=jnp.pi).sample_and_log_prob(seed=key2)

        remainder_norm = jnp.sqrt(1 - z1**2)
        x = jnp.array([z1, remainder_norm*jnp.cos(t1), remainder_norm*jnp.sin(t1)])
    else:
        x = jax.random.normal(key=key, shape=(3,))
        x = x / jnp.linalg.norm(x)
    return x


if __name__ == '__main__':
    version_1 = False
    n_samples = 10000
    key = jax.random.PRNGKey(0)
    samples = jax.vmap(sample_on_sphere, in_axes=(0, None))(jax.random.split(key, n_samples), version_1)

    plt.plot(samples[:, 0], samples[:, 1], "o", alpha=0.1)
    plt.show()

    plt.plot(samples[:, 1], samples[:, 2], "o", alpha=0.1)
    plt.show()

    plt.plot(samples[:, 0], samples[:, 2], "o", alpha=0.1)
    plt.show()

    plt.hist(samples[:, 0])
    plt.show()

    plt.hist(samples[:, 1])
    plt.show()

    plt.hist(samples[:, 2])
    plt.show()



