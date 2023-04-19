import chex
import jax
import jax.numpy as jnp

from molboil.utils.numerical import rotate_translate_permute_general

from flow.x_base_dist import HarmoticPotential, assert_mean_zero


def test_harmonic_potential():
    """Test that the distribution does not smoke. And that it's log prob is invariant to
    rotation and translation."""
    key = jax.random.PRNGKey(0)
    dim = 3
    n_nodes = 7
    batch_size = 5
    shape = (batch_size, n_nodes, dim)

    edges = list(zip(range(n_nodes - 1), range(1, n_nodes)))

    dist = HarmoticPotential(dim=dim, n_nodes=n_nodes, edges=edges)

    # Sample: Test that it does not smoke.
    sample = dist.sample(seed=key, sample_shape=batch_size)
    chex.assert_shape(sample, shape)
    assert_mean_zero(sample, node_axis=1)

    # Log prob: Test shape.
    log_prob = dist.log_prob(sample)
    chex.assert_shape(log_prob, (batch_size,))

    # Log prob: Test that it is invariant to rotation and translation.
    key1, key2, key3 = jax.random.split(key, 3)
    x = jax.random.normal(key1, shape) * 0.1
    theta = jax.random.uniform(key2, shape=shape[:1]) * 2 * jnp.pi
    translation = jnp.zeros((batch_size, dim))
    phi = jax.random.uniform(key3, shape=shape[:1]) * 2 * jnp.pi
    x_rot = jax.vmap(rotate_translate_permute_general)(x, translation, theta, phi)

    log_prob = dist.log_prob(x)
    log_prob_rot = dist.log_prob(x_rot)

    rtol = 1e-5 if x.dtype == jnp.float64 else 1e-3
    chex.assert_trees_all_close(log_prob, log_prob_rot, rtol=rtol)


    # Single sample and log prob: Test that it does not smoke.
    sample = dist.sample(seed=key)
    log_prob = dist.log_prob(sample)


if __name__ == '__main__':
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)

    test_harmonic_potential()
