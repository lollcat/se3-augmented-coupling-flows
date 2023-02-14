import chex
import jax.numpy as jnp
import jax

from utils.numerical import rotate_translate_2d, rotate_translate_3d
from flow.bijector_proj_real_nvp_v2 import matmul_in_invariant_space, inverse_matmul_in_invariant_space,\
    perform_low_rank_matmul, perform_low_rank_matmul_inverse, reshape_things_for_low_rank_matmul



def test_low_rank_mat_mul_and_inverse(n_nodes: int = 5, dim: int = 2, n_vectors: int = 3):
    """Check that low rank mat mul is invertible."""
    if jnp.ones(()).dtype == jnp.float64:
        rtol = 1e-6
    else:
        rtol = 1e-4


    key = jax.random.PRNGKey(0)

    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (n_nodes, dim))

    key, subkey = jax.random.split(key)
    scale = jnp.exp(jax.random.normal(subkey, (n_nodes, dim)))
    key, subkey = jax.random.split(key)
    vectors = jax.random.normal(subkey, (n_vectors, n_nodes, dim))


    x, scale, vectors = reshape_things_for_low_rank_matmul(x, scale, vectors)

    # Forward transform.
    x_out = perform_low_rank_matmul(x, scale, vectors)

    # Invert.
    x_original = perform_low_rank_matmul_inverse(x_out, scale, vectors)

    chex.assert_trees_all_close(x_original, x, rtol=rtol)
    assert jnp.sum(jnp.abs(x - x_out)) > 1e-3


def test_matmul_transform_in_new_space(n_nodes: int = 5, dim: int = 2, n_vectors: int = 3):
    """Test equivariance and invertibility."""
    if jnp.ones(()).dtype == jnp.float64:
        rtol = 1e-6
    else:
        rtol = 1e-4


    key = jax.random.PRNGKey(0)


    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (n_nodes, dim))
    origin = jnp.mean(x, axis=-2)
    change_of_basis_matrix = jnp.eye(dim)
    key, subkey = jax.random.split(key)
    shift = jax.random.normal(subkey, (n_nodes, dim))

    key, subkey = jax.random.split(key)
    scale = jnp.exp(jax.random.normal(subkey, (n_nodes, dim)))
    key, subkey = jax.random.split(key)
    vectors = jax.random.normal(subkey, (n_vectors, n_nodes, dim))

    x_out = matmul_in_invariant_space(x, change_of_basis_matrix, origin, scale, shift, vectors)
    x_original = inverse_matmul_in_invariant_space(x_out, change_of_basis_matrix, origin, scale, shift, vectors)

    # Test invertibility.
    chex.assert_trees_all_close(x_original, x, rtol=rtol)
    assert jnp.sum(jnp.abs(x - x_out)) > 1e-3

    # Test equivariance.
    # Get rotated version of x_and_a.
    key, subkey = jax.random.split(key)
    theta = jax.random.uniform(subkey) * 2 * jnp.pi
    key, subkey = jax.random.split(key)
    translation = jax.random.normal(subkey, shape=(dim,))
    key, subkey = jax.random.split(key)
    phi = jax.random.uniform(subkey) * 2 * jnp.pi

    def group_action(x, theta=theta, translation=translation):
        if dim == 2:
            x_rot = rotate_translate_2d(x, theta, translation)
        else:  #  dim == 3:
            x_rot = rotate_translate_3d(x, theta, phi, translation)
        return x_rot

    # Apply group action to relevant things.
    x_g = group_action(x)
    origin = jnp.mean(x_g, axis=-2)
    change_of_basis_matrix = group_action(change_of_basis_matrix.T, translation=jnp.zeros_like(translation)).T


    x_g_out = matmul_in_invariant_space(x_g, change_of_basis_matrix, origin, scale, shift, vectors)

    x_out_g = group_action(x_out)

    chex.assert_trees_all_close(x_g_out, x_out_g)






def test_zero_vectors_equivalent_to_scaling():
    # If vectors are zero, we retain hammard product.
    pass





if __name__ == '__main__':
    test_matmul_transform_in_new_space()
    test_low_rank_mat_mul_and_inverse()
