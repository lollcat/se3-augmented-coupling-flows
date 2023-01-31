import chex
import jax
import jax.numpy as jnp

def get_pairwise_distances(x):
    chex.assert_rank(x, 2)
    diff_combos = x - x[:, None]  # [n_nodes, n_nodes, dim]
    diff_combos = diff_combos.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # prevents nan grads
    norms = jnp.linalg.norm(diff_combos, ord=2, axis=-1)
    return norms


def set_diagonal_to_zero(x):
    chex.assert_rank(x, 2)
    return jnp.where(jnp.eye(x.shape[0]), jnp.zeros_like(x), x)


def rotate_3d(x, theta, phi):
    rotation_matrix_1 = jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta), 0],
         [jnp.sin(theta), jnp.cos(theta), 0],
         [0,              0,              1]]
    )
    rotation_matrix_2 = jnp.array([
        [1, 0, 0],
        [0, jnp.cos(phi), -jnp.sin(phi)],
        [0, jnp.sin(phi), jnp.cos(phi)],
         ])
    x = jax.vmap(jnp.matmul, in_axes=(None, 0))(rotation_matrix_1, x)
    x = jax.vmap(jnp.matmul, in_axes=(None, 0))(rotation_matrix_2, x)
    return x


def rotate_translate_3d(x, theta, phi, translation, rotate_first=True):
    chex.assert_shape(theta, ())
    chex.assert_shape(phi, ())
    chex.assert_shape(translation, x.shape[-1:])

    if rotate_first:
        return rotate_3d(x, theta, phi) + translation[None, :]
    else:
        return rotate_3d(x + translation[None, :], theta, phi)


def rotate_2d(x, theta):
    rotation_matrix = jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)],
         [jnp.sin(theta), jnp.cos(theta)]]
    )
    if len(x.shape) == 2:
        return jax.vmap(jnp.matmul, in_axes=(None, 0))(rotation_matrix, x)
    else:
        chex.assert_rank(x, 1)
        return jnp.matmul(rotation_matrix, x)


def rotate_translate_2d(x, theta, translation, rotate_first=True):
    chex.assert_shape(theta, ())
    chex.assert_shape(translation, x.shape[-1:])

    if rotate_first:
        return rotate_2d(x, theta) + translation[None, :]
    else:
        return rotate_2d(x + translation[None, :], theta)


def vector_rejection_single(a, b):
    chex.assert_rank(a, 1)
    chex.assert_equal_shape((a, b))
    vector_proj = b * jnp.dot(a, b) / jnp.dot(b, b)
    return a - vector_proj

def vector_rejection(a, b):
    if len(a.shape) == 1:
        return vector_rejection_single(a, b)
    elif len(a.shape) == 2:
        return jax.vmap(vector_rejection_single)(a, b)
    else:
        raise NotImplementedError
