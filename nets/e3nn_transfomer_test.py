import jax
import haiku as hk
import e3nn_jax as e3nn
import matplotlib.pyplot as plt

from nets.e3nn_transformer import EnTransformerConfig, EnTransformerBlock, \
    EnTransformer, EnTransformerTorsoConfig


def plot_points_and_vectors(positions, max_radius = 1000.):
    senders, receivers = e3nn.radius_graph(positions, r_max=max_radius)
    vectors = positions[senders] - positions[receivers]

    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(positions[:, 0], positions[:, 1], positions[:, 2], s=20)
    ax.quiver(positions[receivers, 0], positions[receivers, 1], positions[receivers, 2],
              vectors[:, 0], vectors[:, 1], vectors[:, 2], alpha=0.4)
              # headwidth=2, headlength=2)
    plt.show()


def layer_test():
    dim = 3
    n_nodes = 3
    max_radius = 10.
    n_vectors_hidden = 3
    n_invariant_feat_hidden = 2

    layer_config = {}
    layer_config.update(
        num_heads=2, n_vectors_hidden=n_vectors_hidden, n_invariant_feat_hidden=n_invariant_feat_hidden,
        bessel_number=3, r_max=max_radius, mlp_units=(4,))

    @hk.without_apply_rng
    @hk.transform
    def layer_fn(positions, features):
        return EnTransformerBlock(*layer_config)(positions, features)



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
    plot_points_and_vectors(x_out[:, 0])
    plot_points_and_vectors(x_out[:, 1])
    plot_points_and_vectors(x_out[:, 2])
    print(positions - x_out[:, 0])
    print((x_out[:, 0] - x_out[:, 2]) == 0)
    print((x_out[:, 1] - x_out[:, 2]) == 0)
    pass



if __name__ == '__main__':
    # layer_test()
    transformer_test()
