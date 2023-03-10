import chex
import jax
import haiku as hk
import jax.numpy as jnp
import matplotlib.pyplot as plt

from nets.e3_gnn import EGCL, E3GNNConfig, E3GNNTorsoConfig, E3Gnn
from utils.numerical import rotate_translate_permute_3d
from utils.plotting import plot_points_and_vectors


def layer_test():

    dim = 3
    n_nodes = 4
    n_vectors_hidden = 3
    n_invariant_feat_hidden = 3

    torso_config = E3GNNTorsoConfig(n_blocks=2, mlp_units=(4,), n_invariant_feat_hidden=n_invariant_feat_hidden,
                                    n_vectors_hidden=n_vectors_hidden,
                                    get_shifts_via_tensor_product=True)

    @hk.without_apply_rng
    @hk.transform
    def layer_fn(positions, features):
        return EGCL(**torso_config.get_EGCL_kwargs())(positions, features)



    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    vectors = jax.random.normal(subkey, (n_nodes, n_vectors_hidden, dim))
    vectors = vectors - jnp.mean(vectors, axis=(0, 1), keepdims=True)
    features = jax.random.normal(subkey, (n_nodes, n_invariant_feat_hidden))

    params = layer_fn.init(subkey, vectors, features)
    vectors_out, features_out = layer_fn.apply(params, vectors, features)

    fig, ax = plot_points_and_vectors(vectors)
    ax.set_title("vectors in")
    fig, ax = plot_points_and_vectors(vectors_out)
    ax.set_title("vectors out")
    plt.show()


    # Rotation then forward pass
    key, subkey = jax.random.split(key)
    theta = jax.random.uniform(subkey) * 2 * jnp.pi * 0.1
    key, subkey = jax.random.split(key)
    phi = jax.random.uniform(subkey) * 2 * jnp.pi * 0.1

    def group_action(x, theta=theta, translation=jnp.zeros(dim)):
        x_rot = rotate_translate_permute_3d(x, theta, phi, translation, permute=False)
        return x_rot

    vectors_g = jax.vmap(group_action)(vectors)
    vectors_out_g, featuers_out_g = layer_fn.apply(params, vectors_g, features)


    # plot_points_and_vectors(vectors_g / jnp.linalg.norm(vectors, axis=-1, keepdims=True))
    # plot_points_and_vectors(vectors_out_g / jnp.linalg.norm(vectors, axis=-1, keepdims=True))

    if vectors.dtype == jnp.float64:
        rtol = 1e-5
    else:
        rtol = 1e-3
    chex.assert_tree_all_close(featuers_out_g, features_out, rtol=rtol)
    chex.assert_trees_all_close(vectors_out_g, jax.vmap(group_action)(vectors_out), rtol=rtol)


def e3gnn_test():
    dim = 3
    n_nodes = 20
    n_vectors_hidden = 5
    n_invariant_feat_hidden = 5
    n_vectors_readout = 4
    n_invariant_feat_readout = 4

    torso_config = E3GNNTorsoConfig(n_blocks=3, mlp_units=(4,), n_invariant_feat_hidden=n_invariant_feat_hidden,
                                    n_vectors_hidden=n_vectors_hidden,
                                    layer_stack=False,
                                    get_shifts_via_tensor_product=True)
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
    vectors_out, featuers_out = e3gnn_forward.apply(params, positions)

    grads = jax.grad(lambda param, pos: jnp.sum(e3gnn_forward.apply(param, pos)[0]))(params, positions)
    chex.assert_tree_all_finite(grads)

    # Visualise.
    fig, ax = plot_points_and_vectors(positions)
    ax.set_title("postions in")
    plt.show()


    fig, ax = plot_points_and_vectors(vectors_out / jnp.linalg.norm(vectors_out, axis=-1, keepdims=True))
    ax.set_title('normalized vectors out')
    plt.show()


    # Rotation then forward pass
    key, subkey = jax.random.split(key)
    theta = jax.random.uniform(subkey) * 2 * jnp.pi * 0.1
    key, subkey = jax.random.split(key)
    phi = jax.random.uniform(subkey) * 2 * jnp.pi * 0.1

    def group_action(x, theta=theta, translation=jnp.zeros((dim,))):
        x_rot = rotate_translate_permute_3d(x, theta, phi, translation, permute=False)
        return x_rot

    positions_g = group_action(positions)
    vectors_out_g, featuers_out_g = e3gnn_forward.apply(params, positions_g)

    if positions.dtype == jnp.float64:
        rtol = 1e-5
    else:
        rtol = 1e-3
    chex.assert_tree_all_close(featuers_out_g, featuers_out, rtol=rtol)
    chex.assert_trees_all_close(vectors_out_g, jax.vmap(group_action)(vectors_out), rtol=rtol)





if __name__ == '__main__':
    layer_test()
    # e3gnn_test()
