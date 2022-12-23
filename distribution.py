import distrax
import haiku as hk
import jax.random

from base import CentreGravityGaussian
from bijector_with_proj import make_se_equivariant_split_coupling


def make_equivariant_augmented_flow_dist(dim, nodes, n_layers):
    base = CentreGravityGaussian(dim=int(dim*2), n_nodes=nodes)

    bijectors = []
    for i in range(n_layers):
        bijector = make_se_equivariant_split_coupling(dim, swap=i % 2 == 0)
        bijectors.append(bijector)

    flow = distrax.Chain(bijectors)
    distribution = distrax.Transformed(base, flow)
    return distribution



if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from test_utils import test_fn_is_invariant, test_fn_is_equivariant
    dim = 2
    n_nodes = 20
    n_layers = 1
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


