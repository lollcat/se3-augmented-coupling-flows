import chex
import haiku as hk
import jax
import jax.numpy as jnp
from functools import partial

from utils.numerical import rotate_translate_permute_2d
from nets.en_gnn import en_gnn_net, MultiEgnnConfig, EgnnTorsoConfig


def test_equivariant_fn(dim: int = 2, n_nodes: int = 8, batch_size: int = 3,
                        n_heads: int = 2):
    """Run the EGNN forward pass, and check that it is equivariant."""
    eggn_config = EgnnTorsoConfig(mlp_units=(16,)
                                  , n_layers=2,
                                  zero_init_h=False,
                                  h_embedding_dim=3,
                                  h_linear_softmax=True)
    multi_x_config = MultiEgnnConfig(
        name="multi_x_egnn",
        n_equivariant_vectors_out=n_heads,
        n_invariant_feat_out=3*n_heads, torso_config=eggn_config, invariant_feat_zero_init=False)

    equivariant_fn = hk.without_apply_rng(hk.transform(lambda x: en_gnn_net(multi_x_config)(x)))

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    # Create dummy x.
    x = jax.random.normal(subkey, shape=(batch_size, n_nodes, dim))
    key, subkey = jax.random.split(key)
    theta = jax.random.normal(subkey, shape=(batch_size, ))
    shift = jnp.zeros((batch_size, dim))


    x_transformed = jax.vmap(partial(rotate_translate_permute_2d, permute=False))(x, theta, shift)
    x_untransformed = jax.vmap(partial(rotate_translate_permute_2d, rotate_first=False, permute=False))(x_transformed, -theta, -shift)
    chex.assert_trees_all_close(x_untransformed, x)

    if x.dtype == jnp.float64:
        rtol = 1e-5
    else:
        rtol = 1e-3

    # Initialise bijector parameters.
    params = equivariant_fn.init(key, x)

    # Perform a forward pass.
    x_out, h_out = equivariant_fn.apply(params, x)
    x_out_transformed = jax.vmap(jax.vmap(rotate_translate_permute_2d), in_axes=(2, None, None), out_axes=2)(
        x_out, theta, shift)
    x_transformed_out, h_transformed_out = equivariant_fn.apply(params, x_transformed)

    # Check that rotate-translate THEN transform, is the same as transform THEN rotate-translate.
    chex.assert_trees_all_close(x_transformed_out, x_out_transformed, rtol=rtol)
    chex.assert_trees_all_close(h_transformed_out, h_out, rtol=rtol)



if __name__ == '__main__':
    test_equivariant_fn()
