import chex
import jax

from flow.base_dist import JointBaseDistribution
from flow.centre_of_mass_gaussian import assert_mean_zero
from utils.test import test_fn_is_invariant


def tesst_base_distribution():
    """Test that the base distribution does not smoke. And that it's log prob is invariant to
    rotation and translation."""
    key = jax.random.PRNGKey(0)
    dim = 2
    n_nodes = 5
    n_aux = 3
    batch_size = 7
    shape = (batch_size, n_nodes,  n_aux + 1, dim)

    dist = JointBaseDistribution(dim=dim, n_nodes=n_nodes, n_aux=n_aux)

    # Sample: Test that it does not smoke.
    sample = dist.sample(seed=key, sample_shape=batch_size)
    chex.assert_shape(sample, shape)
    assert_mean_zero(sample, node_axis=-3)

    # Log prob: Test that it is invariant to translation and rotation.
    log_prob = dist.log_prob(sample)
    chex.assert_shape(log_prob, (batch_size,))
    test_fn_is_invariant(invariant_fn=dist.log_prob, key=key, event_shape=shape[1:])


    # Single sample and log prob: Test that it does not smoke.
    sample = dist.sample(seed=key)
    log_prob = dist.log_prob(sample)


if __name__ == '__main__':
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)

    tesst_base_distribution()
