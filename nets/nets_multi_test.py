import chex
import haiku as hk
import jax
import jax.numpy as jnp
from functools import partial

from utils.numerical import rotate_translate_permute_2d
from nets.nets_multi_x import multi_se_equivariant_net, HConfig, MultiEgnnConfig, EgnnConfig


def test_equivariant_fn(dim: int = 2, n_nodes: int = 8, batch_size: int = 3,
                        n_heads: int = 2):
    """Run the EGNN forward pass, and check that it is equivariant."""
    h_config = HConfig(h_embedding_dim=3, h_out=True, h_out_dim=2, share_h=True, linear_softmax=True)
    eggn_config = EgnnConfig(name='egnn', mlp_units=(16,), identity_init_x=False, n_layers=2,
                                  h_config=h_config, zero_init_h=False)
    multi_x_config = MultiEgnnConfig(n_heads=n_heads, egnn_config=eggn_config)

    equivariant_fn = hk.without_apply_rng(hk.transform(lambda x: multi_se_equivariant_net(multi_x_config)(x)))

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    # Create dummy x.
    x = jax.random.normal(subkey, shape=(batch_size, n_nodes, dim))
    key, subkey = jax.random.split(key)
    theta = jax.random.normal(subkey, shape=(batch_size, ))
    key, subkey = jax.random.split(key)
    shift = jax.random.normal(subkey, shape=(batch_size, dim))


    x_transformed = jax.vmap(rotate_translate_permute_2d)(x, theta, shift)
    x_untransformed = jax.vmap(partial(rotate_translate_permute_2d, rotate_first=False))(x_transformed, -theta, -shift)
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
