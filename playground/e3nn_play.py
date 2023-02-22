import chex
import jax.numpy as jnp
import jax
import haiku as hk
import e3nn_jax as e3nn
import matplotlib.pyplot as plt

def plot_points_and_vectors(points, vectors, receivers):
    ax = plt.figure().add_subplot(projection='3d')
    ax.scatter(points[:, 0], points[:, 1], points[:, 2], s=20)
    ax.quiver(points[receivers, 0], points[receivers, 1], points[receivers, 2],
              vectors[:, 0], vectors[:, 1], vectors[:, 2], alpha=0.4)
              # headwidth=2, headlength=2)
    plt.show()




if __name__ == '__main__':
    dim = 3
    n_nodes = 3
    n_invariant_node_featuers = 5
    max_radius = 10.

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)


    positions = jax.random.normal(subkey, (n_nodes, dim))
    positions = positions - jnp.mean(positions, axis=-2)  # Centre.
    invariant_node_feat = jax.random.normal(subkey, (n_nodes, n_invariant_node_featuers))

    senders, receivers = e3nn.radius_graph(positions, r_max=max_radius)

    edge_vec = positions[senders] - positions[receivers]
    plot_points_and_vectors(positions, edge_vec, receivers)

    # Spherical Harmonics
    irreps_sh = e3nn.Irreps.spherical_harmonics(lmax=2)

    # Shape [E, 1 + 3 + 5]
    edge_sh = e3nn.spherical_harmonics(irreps_sh, edge_vec, normalize=False, normalization='component')
    # If we set normalize=False, then edge_sh_vec == edge_vec
    edge_sh_vec = edge_sh.filter("1x1o")
    edge_sh_scalar = edge_sh.filter("1x0e")
    plot_points_and_vectors(positions, edge_vec, receivers)


    # Tensor product
    # e3nn.tensor_product()
    # First setup irreps.
    # Node irreps.
    node_features = e3nn.concatenate([
        e3nn.IrrepsArray(irreps=f"{n_invariant_node_featuers}x0e", array=invariant_node_feat),
        e3nn.IrrepsArray(irreps=f"1x0e", array=jnp.linalg.norm(positions, axis=-1, keepdims=True)),
        e3nn.IrrepsArray(irreps=f"1x1e", array=positions / jnp.linalg.norm(positions, axis=-1))
    ]
    )  # 5x0e+1x0e+1x1e
    node_features = node_features.simplify()  # 6x0e+1x1e
    # Edge irreps.
    edge_attributes = e3nn.concatenate([
        e3nn.IrrepsArray(irreps="1x0e", array=jnp.linalg.norm(edge_vec, axis=-1, keepdims=True)),
        edge_sh])




