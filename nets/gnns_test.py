import jax
import jax.numpy as jnp
import haiku as hk
import chex

from flow.test_utils import get_minimal_nets_config
from nets.base import EGNN
from flow.test_utils import test_fn_is_invariant, test_fn_is_equivariant

def test_net_does_not_smoke(type="egnn_v0"):
    nets_config = get_minimal_nets_config(type=type)
    zero_init_invariant_feat = False
    n_invariant_feat_out = 5
    n_equivariant_vectors_out = 2
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
    chex.assert_shape(vectors, (n_nodes, multiplicity, n_equivariant_vectors_out, dim))
    chex.assert_shape(scalars, (n_nodes, multiplicity, n_invariant_feat_out))

    def invariant_fn(positions):
        vectors, scalars = forward.apply(params, positions, features)
        return scalars

    def equivariant_fn(positions):
        vectors, scalars = forward.apply(params, positions, features)
        return vectors[:, 1, :, :]  # take the first vector

    test_fn_is_invariant(invariante_fn=invariant_fn, key=key, event_shape=positions.shape)
    test_fn_is_equivariant(equivariant_fn=equivariant_fn, key=key, event_shape=positions.shape, translate=False)




if __name__ == '__main__':
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)

    test_net_does_not_smoke()
