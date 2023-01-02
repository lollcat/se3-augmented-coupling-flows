import chex
import jax

from base import CentreGravityGaussian, assert_mean_zero
from test_utils import test_fn_is_invariant


def test_base_distribution():
    """Test that the base distribution does not smoke. And that it's log prob is invariant to
    rotation and translation."""
    key = jax.random.PRNGKey(0)
    dim = 2
    n_nodes = 3
    batch_size = 2
    shape = (batch_size, n_nodes, dim)
    dist = CentreGravityGaussian(dim, n_nodes)

    # Sample: Test that it does not smoke.
    sample = dist.sample(seed=key, sample_shape=batch_size)
    chex.assert_shape(sample, shape)
    assert_mean_zero(sample)

    # Log prob: Test that it is invariant to translation and rotation.
    log_prob = dist.log_prob(sample)
    chex.assert_shape(log_prob, (batch_size,))
    test_fn_is_invariant(dist.log_prob, key)


    # Single sample and log prob: Test that it does not smoke.
    sample = dist.sample(seed=key)
    log_prob = dist.log_prob(sample)


if __name__ == '__main__':
    test_base_distribution()
