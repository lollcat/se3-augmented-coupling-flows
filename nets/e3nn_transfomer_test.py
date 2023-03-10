import jax
import haiku as hk
import e3nn_jax as e3nn
import jax.numpy as jnp
import matplotlib.pyplot as plt

from nets.e3nn_transformer import EnTransformerConfig, EnTransformerBlock, \
    EnTransformer, EnTransformerTorsoConfig
from utils.plotting import plot_points_and_vectors

def layer_test():
    dim = 3
    n_nodes = 3
    max_radius = 10.
    n_vectors_hidden = 3
    n_invariant_feat_hidden = 2

    layer_config = {}
    layer_config.update(
        num_heads=1, n_vectors_hidden=n_vectors_hidden, n_invariant_feat_hidden=n_invariant_feat_hidden,
        bessel_number=3, r_max=max_radius, mlp_units=(4,))

    @hk.without_apply_rng
    @hk.transform
    def layer_fn(positions, features):
        return EnTransformerBlock(**layer_config)(positions, features)



    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    positions = jax.random.normal(subkey, (n_nodes, n_vectors_hidden, dim))
    features = jax.random.normal(subkey, (n_nodes, n_invariant_feat_hidden))

    params = layer_fn.init(subkey, positions, features)
    out = layer_fn.apply(params, positions, features)

def transformer_test():
    num_heads = 1
    torso_config = EnTransformerTorsoConfig(num_heads=num_heads, n_vectors_hidden=num_heads * 2,
                                            n_invariant_feat_hidden=num_heads*3, bessel_number=8, r_max=10.0,
                                            mlp_units=(4,), n_blocks=3, layer_stack=False)
    transformer_config = EnTransformerConfig(name="dogfish", n_invariant_feat_readout=7, n_vectors_readout=5,
                                             zero_init_invariant_feat=False, torso_config=torso_config)
    @hk.without_apply_rng
    @hk.transform
    def forward_fn(positions):
        x, h = EnTransformer(transformer_config)(positions)
        return x, h

    dim = 3
    feat_dim = 2
    n_nodes = 3
    max_radius = 10.

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    positions = jax.random.normal(subkey, (n_nodes, dim))
    plot_points_and_vectors(positions)

    params = forward_fn.init(subkey, positions)
    x_out, h_out = forward_fn.apply(params, positions)
    vectors = x_out - jnp.mean(x_out, axis=(0, 1), keepdims=True)
    fig, ax = plot_points_and_vectors(vectors)
    ax.set_title("vectors out")
    plt.show()
    fig, ax = plot_points_and_vectors(vectors / jnp.linalg.norm(vectors, axis=-1, keepdims=True))
    ax.set_title("vectors out normalised")
    plt.show()
    print((x_out[:, 0] - x_out[:, 2]) == 0)
    print((x_out[:, 1] - x_out[:, 2]) == 0)
    pass



if __name__ == '__main__':
    # layer_test()
    transformer_test()
