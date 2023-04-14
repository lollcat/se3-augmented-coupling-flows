import chex
import jax.numpy as jnp
import jax

from flow.aug_flow_dist import FullGraphSample
from flow.conditional_dist import build_aux_dist

_N_NODES = 7
_FEATURE_DIM = 1


def test_conditional_dist(dim: int = 3, n_aug: int = 3):
    n_nodes = _N_NODES
    batch_size = 5
    key = jax.random.PRNGKey(0)

    make_aux_target = build_aux_dist(name='target', n_aug=n_aug, augmented_scale_init=1.,
                                     trainable_scale=False)

    # Init params.
    key, subkey = jax.random.split(key)
    dummy_samples = FullGraphSample(positions=jax.random.normal(subkey, (batch_size, n_nodes, dim)),
                                    features=jnp.zeros((batch_size, n_nodes, _FEATURE_DIM)))
    dist_v1 = make_aux_target(dummy_samples)

    # Test aux-target distribution does not smoke.
    key, subkey = jax.random.split(key)
    aug_samples, aux_log_probs = dist_v1.sample_and_log_prob(seed=subkey)



if __name__ == '__main__':
    test_conditional_dist(dim=2)
