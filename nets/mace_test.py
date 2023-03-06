import chex
import jax.random
import haiku as hk
import jax.numpy as jnp
import matplotlib.pyplot as plt

from utils.numerical import rotate_translate_permute_3d
from nets.mace import MaceNet, MACEConfig, MACETorsoConfig
from utils.plotting import plot_points_and_vectors


def test_mace(dim: int = 3, n_nodes: int = 5):
    if jnp.ones(()).dtype == jnp.float64:
        rtol = 1e-5
    else:
        rtol = 1e-4

    torso_config = MACETorsoConfig(
        bessel_number=5,
        n_vectors_hidden=5,
        n_invariant_feat_hidden=5,
        avg_num_neighbors=n_nodes,
        r_max=5.0,
        num_species=1,
        n_interactions=2,
    )

    config = MACEConfig(
        name='mace',
        n_invariant_feat_readout=2,
        n_vectors_readout=4,
        torso_config=torso_config,
        zero_init_invariant_feat=False
    )

    @hk.without_apply_rng
    @hk.transform
    def mace_forward_fn(x: chex.Array):
        mace_net = MaceNet(config)
        x = mace_net(x)
        return x


    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    positions = jax.random.normal(key=subkey, shape=(n_nodes, dim))*0.5

    key, subkey = jax.random.split(key)
    params = mace_forward_fn.init(subkey, positions)
    vectors_out, h = jax.jit(mace_forward_fn.apply)(params, positions)
    chex.assert_shape(vectors_out, (n_nodes, config.n_vectors_readout, dim))
    chex.assert_shape(h, (n_nodes, config.n_invariant_feat_readout))

    fig, ax = plot_points_and_vectors(positions)
    ax.set_title("postions in")
    plt.show()

    # Visualise vectors.
    fig, ax = plot_points_and_vectors(vectors_out / jnp.linalg.norm(vectors_out, axis=-1, keepdims=True))
    ax.set_title('normalized vectors out')
    plt.show()


    # Test equivariance.
    key, subkey = jax.random.split(key)
    theta = jax.random.uniform(subkey) * 2 * jnp.pi
    key, subkey = jax.random.split(key)
    phi = jax.random.uniform(subkey) * 2 * jnp.pi
    def group_action(x, theta=theta, translation=jnp.zeros(dim)):
        x_rot = rotate_translate_permute_3d(x, theta, phi, translation, permute=False)
        return x_rot

    x_g = group_action(positions)
    x_g_out, h_g = mace_forward_fn.apply(params, x_g)

    x_out_g = jax.vmap(group_action)(vectors_out)

    chex.assert_trees_all_close(h, h_g, rtol=rtol)
    chex.assert_trees_all_close(x_out_g, x_g_out, rtol=rtol)

    print(jnp.linalg.norm(x_out_g, axis=-1))





if __name__ == '__main__':
    test_mace()
