import jax
import jax.numpy as jnp
import haiku as hk
import chex

from utils.test import get_minimal_nets_config
from nets.base import EGNN

def test_net_does_not_smoke(type="egnn"):
    nets_config = get_minimal_nets_config(type=type)
    zero_init_invariant_feat = False
    n_invariant_feat_out = 5
    n_equivariant_vectors_in = 2
    n_equivariant_vectors_out = n_equivariant_vectors_in * 2
    dim = 2 if type == 'egnn' else 3
    n_nodes = 5
    multiplicity = 2

    @hk.without_apply_rng
    @hk.transform
    def forward(positions, features):
        vectors, scalars = EGNN('dogfish',
                                nets_config=nets_config,
                                zero_init_invariant_feat=zero_init_invariant_feat,
                                n_invariant_feat_out = n_invariant_feat_out,
                                n_equivariant_vectors_out = n_equivariant_vectors_out,
                                )(positions, features)
        return vectors, scalars


    key = jax.random.PRNGKey(0)
    positions = jax.random.normal(key, (n_nodes, multiplicity, dim))
    features = jnp.zeros((n_nodes, 1, 2))

    params = forward.init(key, positions, features)

    vectors, scalars = forward.apply(params, positions, features)
    chex.assert_shape(vectors, (n_nodes, n_equivariant_vectors_out, dim))
    chex.assert_shape(scalars, (n_nodes, n_invariant_feat_out))


if __name__ == '__main__':
    test_net_does_not_smoke()
