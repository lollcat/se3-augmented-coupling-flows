import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import haiku as hk

from test_utils import test_fn_is_invariant, test_fn_is_equivariant
from distribution import make_equivariant_augmented_flow_dist


def test_distribution():
    """Visualise samples from the distribution, and check that it's log prob is invariant to
    translation and rotation."""

    dim = 2
    n_nodes = 20
    n_layers = 2
    batch_size = 5
    key = jax.random.PRNGKey(0)


    @hk.transform
    def sample_and_log_prob_fn(sample_shape=()):
        distribution = make_equivariant_augmented_flow_dist(dim=dim, nodes=n_nodes, n_layers=n_layers)
        return distribution.sample_and_log_prob(seed=hk.next_rng_key(), sample_shape=sample_shape)


    @hk.without_apply_rng
    @hk.transform
    def log_prob_fn(x):
        distribution = make_equivariant_augmented_flow_dist(dim=dim, nodes=n_nodes, n_layers=n_layers)
        return distribution.log_prob(x)


    # Init params.
    key, subkey = jax.random.split(key)
    params = sample_and_log_prob_fn.init(subkey)

    key, subkey = jax.random.split(key)
    sample, log_prob = sample_and_log_prob_fn.apply(params, subkey, (batch_size,))

    plt.plot(sample[0, :, 0], sample[0, :, 1], 'o')
    plt.show()

    # Test log prob function is invariant.
    key, subkey = jax.random.split(key)
    test_fn_is_invariant(lambda x: log_prob_fn.apply(params, x), subkey, n_nodes=n_nodes)


def test_flow():
    dim = 2
    n_nodes = 20
    n_layers = 2
    batch_size = 5
    key = jax.random.PRNGKey(0)

    @hk.without_apply_rng
    @hk.transform
    def forward_and_log_det(x):
        distribution = make_equivariant_augmented_flow_dist(dim=dim, nodes=n_nodes, n_layers=n_layers)
        return distribution.bijector.forward_and_log_det(x)


    key, subkey = jax.random.split(key)
    x_and_a = jnp.zeros((n_nodes, dim*2))
    x_and_a = x_and_a + jax.random.normal(subkey, shape=x_and_a.shape)*0.1
    params = forward_and_log_det.init(subkey, x_and_a)

    # Run a forward pass
    x_and_a_new, log_det = forward_and_log_det.apply(params, x_and_a)

    # Check equivariance of forward, and invariance of log_det
    key, subkey = jax.random.split(key)
    test_fn_is_equivariant(lambda x: forward_and_log_det.apply(params, x)[0], subkey, n_nodes=n_nodes)
    key, subkey = jax.random.split(key)
    test_fn_is_invariant(lambda x: forward_and_log_det.apply(params, x)[1], subkey, n_nodes=n_nodes)



if __name__ == '__main__':
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)
    test_flow()
    test_distribution()
