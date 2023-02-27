import chex
import jax
import haiku as hk
import e3nn_jax as e3nn
import jax.numpy as jnp
import matplotlib.pyplot as plt

from nets.e3_gnn import EGCL, E3GNNConfig, E3GNNTorsoConfig, E3Gnn
from utils.numerical import rotate_translate_permute_3d

def plot_points_and_vectors(positions, max_radius = 1000.):
    senders, receivers = e3nn.radius_graph(positions, r_max=max_radius)
    vectors = positions[senders] - positions[receivers]

    ax = plt.figure().add_subplot(projection='3d')
    if len(positions.shape) == 2:
        ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=20)
        ax.quiver(positions[receivers, 0], positions[receivers, 1], positions[receivers, 2],
                  vectors[:, 0], vectors[:, 1], vectors[:, 2], alpha=0.4)
    else:
        assert len(positions.shape) == 3
        c = ['red', 'green', 'blue', 'cyan']
        for i in range(positions.shape[1]):
            ax.scatter(positions[:, i, 0], positions[:, i, 1], positions[:, i, 2], s=20, color=c[i])
            ax.quiver(positions[receivers, i, 0], positions[receivers, i, 1], positions[receivers, i, 2],
                      vectors[:, i, 0], vectors[:, i, 1], vectors[:, i, 2], alpha=0.4, color=c[i])
    plt.show()


def layer_test():
    dim = 3
    n_nodes = 4
    n_vectors_hidden = 3
    n_invariant_feat_hidden = 2

    layer_config = {}
    layer_config.update(
        n_vectors_hidden=n_vectors_hidden, n_invariant_feat_hidden=n_invariant_feat_hidden,
        mlp_units=(4,))

    @hk.without_apply_rng
    @hk.transform
    def layer_fn(positions, features):
        return EGCL(**layer_config)(positions, features)



    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    vectors = jax.random.normal(subkey, (n_nodes, n_vectors_hidden, dim))
    vectors = vectors - jnp.mean(vectors, axis=(0, 1), keepdims=True)
    features = jax.random.normal(subkey, (n_nodes, n_invariant_feat_hidden))

    params = layer_fn.init(subkey, vectors, features)
    positions_out, vectors_out = layer_fn.apply(params, vectors, features)

    plot_points_and_vectors(vectors)
    plot_points_and_vectors(positions_out)


    # Rotation then forward pass
    key, subkey = jax.random.split(key)
    theta = jax.random.uniform(subkey) * 2 * jnp.pi * 0.1
    key, subkey = jax.random.split(key)
    phi = jax.random.uniform(subkey) * 2 * jnp.pi * 0.1

    def group_action(x, theta=theta, translation=jnp.zeros(dim)):
        x_rot = rotate_translate_permute_3d(x, theta, phi, translation, permute=False)
        return x_rot

    positions_g = jax.vmap(group_action)(vectors)
    positions_out_g, featuers_out_g = layer_fn.apply(params, positions_g, features)

    plot_points_and_vectors(positions_g)
    plot_points_and_vectors(positions_out_g)

    if vectors.dtype == jnp.float64:
        rtol = 1e-5
    else:
        rtol = 1e-3
    chex.assert_tree_all_close(featuers_out_g, vectors_out)
    chex.assert_trees_all_close(positions_out_g, jax.vmap(group_action)(positions_out), rtol=rtol)


def e3gnn_test():
    dim = 3
    n_nodes = 4
    n_vectors_hidden = 5
    n_invariant_feat_hidden = 5
    n_vectors_readout = 4
    n_invariant_feat_readout = 4

    torso_config = E3GNNTorsoConfig(n_blocks=2, mlp_units=(4,), n_invariant_feat_hidden=n_invariant_feat_hidden,
                                    n_vectors_hidden=n_vectors_hidden)
    e3gnn_config = E3GNNConfig(name="dogfish",
                               n_vectors_readout=n_vectors_readout,
                               n_invariant_feat_readout=n_invariant_feat_readout,
                               zero_init_invariant_feat=False,
                               torso_config=torso_config)

    @hk.without_apply_rng
    @hk.transform
    def e3gnn_forward(positions):
        return E3Gnn(e3gnn_config)(positions)



    key = jax.random.PRNGKey(1)
    key, subkey = jax.random.split(key)
    positions = jax.random.normal(subkey, (n_nodes, dim))*2

    params = e3gnn_forward.init(subkey, positions)
    positions_out, featuers_out = e3gnn_forward.apply(params, positions)

    plot_points_and_vectors(positions)
    plot_points_and_vectors(positions_out)


    # Rotation then forward pass
    key, subkey = jax.random.split(key)
    theta = jax.random.uniform(subkey) * 2 * jnp.pi * 0.1
    key, subkey = jax.random.split(key)
    translation = jax.random.normal(subkey, shape=(dim,)) * 0.1
    key, subkey = jax.random.split(key)
    phi = jax.random.uniform(subkey) * 2 * jnp.pi * 0.1
    def group_action(x, theta=theta, translation=translation):
        x_rot = rotate_translate_permute_3d(x, theta, phi, translation, permute=False)
        return x_rot

    positions_g = group_action(positions)
    positions_out_g, featuers_out_g = e3gnn_forward.apply(params, positions_g)

    plot_points_and_vectors(positions_g)
    plot_points_and_vectors(positions_out_g)

    if positions.dtype == jnp.float64:
        rtol = 1e-5
    else:
        rtol = 1e-3
    chex.assert_tree_all_close(featuers_out_g, featuers_out, rtol=rtol)
    chex.assert_trees_all_close(positions_out_g, group_action(positions_out), rtol=rtol)





if __name__ == '__main__':
    # layer_test()
    e3gnn_test()
