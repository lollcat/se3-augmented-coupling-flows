import chex
import jax.random
import haiku as hk
import jax.numpy as jnp

from utils.numerical import rotate_translate_permute_2d, rotate_translate_permute_3d
from nets.mace import MaceNet, MACEConfig


def test_mace(dim: int = 3, n_nodes: int = 4):
    if jnp.ones(()).dtype == jnp.float64:
        rtol = 1e-5
    else:
        rtol = 1e-4

    config = MACEConfig(
        name='mace',
        n_invariant_feat_readout=2,
        n_vectors_readout=5,
        n_vectors_hidden=5,
        n_invariant_feat_hidden=5,
        avg_num_neighbors=n_nodes,
        r_max=5.0,
        num_species=1,
        n_interactions=2)

    @hk.without_apply_rng
    @hk.transform
    def mace_forward_fn(x: chex.Array):
        return MaceNet(config)(x)


    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    x = jax.random.normal(key=subkey, shape=(n_nodes, dim))

    key, subkey = jax.random.split(key)
    params = mace_forward_fn.init(subkey, x)
    h, x_out = mace_forward_fn.apply(params, x)

    # Test equivariance.
    key, subkey = jax.random.split(key)
    theta = jax.random.uniform(subkey) * 2 * jnp.pi
    key, subkey = jax.random.split(key)
    translation = jax.random.normal(subkey, shape=(dim,))
    key, subkey = jax.random.split(key)
    phi = jax.random.uniform(subkey) * 2 * jnp.pi
    def group_action(x, theta=theta, translation=translation):
        if dim == 2:
            x_rot = rotate_translate_permute_2d(x, theta, translation, permute=False)
        else:  #  dim == 3:
            x_rot = rotate_translate_permute_3d(x, theta, phi, translation, permute=False)
        return x_rot

    x_g = group_action(x)
    h_g, x_g_out = mace_forward_fn.apply(params, x_g)

    x_out_g = jax.vmap(group_action)(x_out)

    chex.assert_trees_all_close(h, h_g, rtol=rtol)
    chex.assert_trees_all_close(x_out_g, x_g_out, rtol=rtol)





if __name__ == '__main__':
    test_mace()
