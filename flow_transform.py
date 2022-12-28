import jax.numpy as jnp
import jax
import haiku as hk
import chex
import matplotlib.pyplot as plt


def get_pairwise_distances(x):
    return jnp.linalg.norm(x - x[:, None], ord=2, axis=-1)


def affine_transform_in_new_space(point, change_of_basis_matrix, origin, scale, shift):
    """Perform affine transformation in the space define by the `origin` and `change_of_basis_matrix`, and then
    go back into the original space."""
    point_in_new_space = jnp.linalg.inv(change_of_basis_matrix) @ (point - origin)
    transformed_point_in_new_space = point_in_new_space * scale + shift
    new_point_original_space = change_of_basis_matrix @ transformed_point_in_new_space + origin
    return new_point_original_space


def inverse_affine_transform_in_new_space(point, change_of_basis_matrix, origin, scale, shift):
    """Inverse of `affine_transform_in_new_space`."""
    point_in_new_space = jnp.linalg.inv(change_of_basis_matrix)  @ (point - origin)
    transformed_point_in_new_space = (point_in_new_space - shift) / scale
    new_point_original_space = change_of_basis_matrix @ transformed_point_in_new_space + origin
    return new_point_original_space


def make_se2_real_nvp():

    def equivariant_fn(x):
        diff_combos = x - x[:, None]  # [n_nodes, n_nodes, dim]
        norms = jnp.linalg.norm(diff_combos, ord=2, axis=-1)
        m = jnp.squeeze(hk.nets.MLP((5, 1), activation=jax.nn.elu)(norms[..., None]), axis=-1) * 3
        return x + jnp.einsum('ijd,ij->id', diff_combos, m)

    def invariant_fn(x, n_vals):
        equivariant_x = jnp.stack([equivariant_fn(x) for _ in range(n_vals)], axis=-1)
        return jnp.linalg.norm(x[..., None] - equivariant_x, ord=2, axis=-2)


    def forward_fn(x, a):
        """Takes in x and a which are nodes in a graph, and performs a transform that is SE(n) invariant with respect
        to each node in the graph. We condition a on x (coupling)."""
        assert x.shape == a.shape
        dim = x.shape[-1]

        # Calculate new basis for the affine transform
        origin = equivariant_fn(x)
        y_basis_point = equivariant_fn(x)
        x_basis_point = equivariant_fn(x)

        y_basis_vector = y_basis_point - origin
        x_basis_vector = x_basis_point - origin
        change_of_basis_matrix = jnp.stack([x_basis_vector, y_basis_vector], axis=-1)

        # Get scale and shift
        log_scale = invariant_fn(x, dim)
        scale = jnp.exp(log_scale)
        shift = invariant_fn(x, dim)


        # Perform transform and calculate log determinant.
        new_a = jax.vmap(affine_transform_in_new_space)(a, change_of_basis_matrix, origin, scale, shift)
        log_det = jnp.sum(log_scale, axis=-1)

        return x, new_a, log_det


    def inverse_fn(x, a):
        """Inverse of `forward_fn`."""
        assert x.shape[-1] == a.shape[-1]
        dim = x.shape[-1]

        # Calculate new basis for the affine transform
        origin = equivariant_fn(x)
        y_basis_point = equivariant_fn(x)
        x_basis_point = equivariant_fn(x)

        y_basis_vector = y_basis_point - origin
        x_basis_vector = x_basis_point - origin
        change_of_basis_matrix = jnp.stack([x_basis_vector, y_basis_vector], axis=-1)

        # Get scale and shift
        log_scale = invariant_fn(x, dim)
        scale = jnp.exp(log_scale)
        shift = invariant_fn(x, dim)

        # Perform transform and calculate log determinant.
        prev_a = jax.vmap(inverse_affine_transform_in_new_space)(a, change_of_basis_matrix, origin, scale, shift)
        log_det = - jnp.sum(log_scale, axis=-1)

        return x, prev_a, log_det


    return hk.without_apply_rng(hk.transform(forward_fn)), hk.without_apply_rng(hk.transform(inverse_fn))



if __name__ == '__main__':
    from test_utils import test_fn_is_equivariant, test_fn_is_invariant
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config

        config.update("jax_enable_x64", True)


    if USE_64_BIT:
        r_tol = 1e-6
    else:
        # Need quite high r_tol for checks to pass with 32-bit. I guess we need to do numerical stability tricks.
        r_tol = 1e-3

    forward_fn, inverse_fn = make_se2_real_nvp()

    dim = 2
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    # Create dummy x.
    x = jnp.zeros((3, 2))
    x = x + jax.random.normal(subkey, shape=x.shape)*0.1

    # Create dummy a.
    key, subkey = jax.random.split(key)
    a = x + jax.random.normal(subkey, shape=x.shape)*0.1

    print(f"*********** initialised params************ \n\n")
    # Initialise GNN params.
    key, subkey = jax.random.split(key)
    params = forward_fn.init(subkey, x, a)


    print(f"*********** do a forward pass ************ \n\n")
    # Perform a forward pass and inverse.
    _, a_new, log_det = forward_fn.apply(params, x, a)

    _, a_original, log_det_inverse = inverse_fn.apply(params, x, a_new)
    chex.assert_trees_all_close(a, a_original, rtol=r_tol)  # check that inverse works
    chex.assert_trees_all_close(log_det, -log_det_inverse)

    print(f"*********** rotation ************ \n\n")
    theta = jnp.pi * 0.5
    rotation_matrix = jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)],
         [jnp.sin(theta), jnp.cos(theta)]]
    )

    x_rot = jax.vmap(jnp.matmul, in_axes=(None, 0))(rotation_matrix, x)
    a_rot = jax.vmap(jnp.matmul, in_axes=(None, 0))(rotation_matrix, a)

    # Check the norms have been preserved.
    chex.assert_trees_all_close(get_pairwise_distances(a), get_pairwise_distances(a_rot))
    chex.assert_trees_all_close(get_pairwise_distances(x),
                                get_pairwise_distances(x_rot))


    # Perform forward pass.
    _, a_rot_new, log_det_rot = forward_fn.apply(params, x_rot, a_rot)

    chex.assert_trees_all_close(log_det, log_det_rot)
    # Check the norms have been preserved.
    norms_original = jnp.linalg.norm(a_new - a_new[:, None], ord=2, axis=-1)
    norms_after_rot = jnp.linalg.norm(a_rot_new - a_rot_new[:, None], ord=2, axis=-1)
    chex.assert_trees_all_close(norms_original, norms_after_rot, rtol=r_tol)

    # Check that if we rotate `a_new` then this is equivalent to doing the original transform (no rotation).
    a_new_rot = jax.vmap(jnp.matmul, in_axes=(None, 0))(rotation_matrix, a_new)
    chex.assert_trees_all_close(a_rot_new, a_new_rot)


