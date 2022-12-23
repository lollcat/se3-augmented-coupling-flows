import chex
import jax.numpy as jnp
import jax

def get_pairwise_distances(x):
    return jnp.linalg.norm(x - x[:, None], ord=2, axis=-1)


def rotate_translate_2d(x_and_a, theta, translation):
    rotation_matrix = jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)],
         [jnp.sin(theta), jnp.cos(theta)]]
    )
    x, a = jnp.split(x_and_a, axis=-1, indices_or_sections=2)
    x_rot = jax.vmap(jnp.matmul, in_axes=(None, 0))(rotation_matrix, x) + translation
    a_rot = jax.vmap(jnp.matmul, in_axes=(None, 0))(rotation_matrix, a) + translation
    return jnp.concatenate([x_rot, a_rot], axis=-1)


def test_fn_is_equivariant(equivariant_fn, key, n_nodes=7):

    dim = 2
    # Setup
    key1, key2, key3 = jax.random.split(key, 3)
    x_and_a = jnp.zeros((n_nodes, dim * 2))
    x_and_a = x_and_a + jax.random.normal(key1, shape=x_and_a.shape) * 0.1

    # Get rotated version of x_and_a.
    theta = jax.random.uniform(key2) * 2*jnp.pi
    translation = jax.random.normal(key3, shape=(1, dim))
    x_and_a_rot = rotate_translate_2d(x_and_a, theta, translation)

    # Compute equivariant_fn of both the original and rotated matrices.
    x_and_a_new = equivariant_fn(x_and_a)
    x_and_a_new_rot = equivariant_fn(x_and_a_rot)

    # Check that rotating x_and_a_new_rot gives x_and_a_new
    if x_and_a.dtype == jnp.float64:
        rtol = 1e-6
    else:
        rtol = 1e-3
    chex.assert_trees_all_close(x_and_a_new_rot, rotate_translate_2d(x_and_a_new, theta, translation), rtol=rtol)

    chex.assert_trees_all_close(get_pairwise_distances(x_and_a_new_rot),
                                get_pairwise_distances(x_and_a_new), rtol=rtol)


def test_fn_is_invariant(invariante_fn, key, n_nodes=7):

    dim = 2
    # Setup
    key1, key2, key3 = jax.random.split(key, 3)
    x_and_a = jnp.zeros((n_nodes, dim * 2))
    x_and_a = x_and_a + jax.random.normal(key1, shape=x_and_a.shape) * 0.1

    # Get rotated version of x_and_a.
    theta = jax.random.uniform(key2) * 2 * jnp.pi
    translation = jax.random.normal(key3, shape=(1, dim,))
    x_and_a_rot = rotate_translate_2d(x_and_a, theta, translation)

    # Compute invariante_fn of both the original and rotated matrices.
    out = invariante_fn(x_and_a)
    out_rot = invariante_fn(x_and_a_rot)

    # Check that rotating x_and_a_new_rot gives x_and_a_new
    if x_and_a.dtype == jnp.float64:
        rtol = 1e-6
    else:
        rtol = 1e-3
    chex.assert_trees_all_close(out, out_rot, rtol=rtol)



