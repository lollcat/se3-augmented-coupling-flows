import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import chex


from flow.test_utils import bijector_test
from utils.numerical import rotate_translate_permute_2d, rotate_translate_permute_3d
from flow.bijectors.bijector_proj_real_nvp import make_se_equivariant_split_coupling_with_projection, \
    affine_transform_in_new_space, inverse_affine_transform_in_new_space
from flow.test_utils import get_minimal_nets_config


def test_matmul_transform_in_new_space(n_nodes: int = 5, dim: int = 3):
    """Test equivariance and invertibility."""
    if jnp.ones(()).dtype == jnp.float64:
        rtol = 1e-6
    else:
        rtol = 1e-4


    key = jax.random.PRNGKey(0)


    key, subkey = jax.random.split(key)
    x = jax.random.normal(subkey, (n_nodes, dim))
    origin = x + jax.random.normal(subkey, (n_nodes, dim))*0.1
    change_of_basis_matrix = jnp.repeat(jnp.eye(dim)[None, ...], n_nodes, axis=0)
    key, subkey = jax.random.split(key)
    shift = jax.random.normal(subkey, (n_nodes, dim))

    key, subkey = jax.random.split(key)
    scale = jnp.exp(jax.random.normal(subkey, (n_nodes, dim)))

    x_out = jax.vmap(affine_transform_in_new_space)(x, change_of_basis_matrix, origin, scale, shift)
    x_original = jax.vmap(inverse_affine_transform_in_new_space)(x_out, change_of_basis_matrix, origin, scale, shift)

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
            x_rot = rotate_translate_permute_2d(x, theta, translation, permute=False)
        else:  #  dim == 3:
            x_rot = rotate_translate_permute_3d(x, theta, phi, translation, permute=False)
        return x_rot

    def group_action_to_column_matrix_of_vecs(change_of_basis_matrix, theta=theta):
        if dim == 2:
            change_of_basis_matrix_rot = rotate_translate_permute_2d(
                change_of_basis_matrix.T, theta, jnp.zeros_like(translation),
                permute=False
            ).T
        else:  #  dim == 3:
            change_of_basis_matrix_rot = rotate_translate_permute_3d(
                change_of_basis_matrix.T, theta, phi, jnp.zeros_like(translation),
                permute=False).T
        return change_of_basis_matrix_rot

    # Apply group action to relevant things.
    x_g = group_action(x)
    origin = group_action(origin)

    change_of_basis_matrix = jax.vmap(group_action_to_column_matrix_of_vecs, in_axes=0)(change_of_basis_matrix)

    x_g_out = jax.vmap(affine_transform_in_new_space)(x_g, change_of_basis_matrix, origin, scale, shift)

    x_out_g = group_action(x_out)

    chex.assert_trees_all_close(x_g_out, x_out_g)


def test_bijector_with_proj(dim: int = 3, n_layers: int = 4, type='egnn',
                            n_nodes: int = 4, n_aux: int = 3):
    nets_config = get_minimal_nets_config(type=type)

    graph_features = jnp.zeros((n_nodes, 1, 1))

    def make_flow():
        bijectors = []
        for i in range(n_layers):
            swap = i % 2 == 0
            bijector = make_se_equivariant_split_coupling_with_projection(
                graph_features=graph_features,
                layer_number=i,
                dim=dim,
                n_aux=n_aux,
                swap=swap,
                identity_init=False,
                nets_config=nets_config)
            bijectors.append(bijector)
        flow = distrax.Chain(bijectors)
        return flow

    @hk.without_apply_rng
    @hk.transform
    def bijector_forward(x):
        flow = make_flow()
        return flow.forward_and_log_det(x)

    @hk.without_apply_rng
    @hk.transform
    def bijector_backward(x):
        flow = make_flow()
        return flow.inverse_and_log_det(x)

    bijector_test(bijector_forward, bijector_backward, dim=dim, n_nodes=n_nodes, n_aux=n_aux)


if __name__ == '__main__':
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)

    test_bijector_with_proj(dim=3)
    print('passed test in 3D')
    test_bijector_with_proj(dim=2)
    print('passed test in 2D')


