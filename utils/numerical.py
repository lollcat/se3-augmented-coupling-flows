from typing import List

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


def rotate_translate_permute_3d(x, theta, phi, translation, rotate_first=True, permute=True):
    chex.assert_shape(theta, ())
    chex.assert_shape(phi, ())
    chex.assert_shape(translation, x.shape[-1:])

    if permute:
        x = jnp.roll(x, shift=1, axis=-2)

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


def rotate_translate_permute_2d(x, theta, translation, rotate_first=True, permute = True):
    chex.assert_shape(theta, ())
    chex.assert_shape(translation, x.shape[-1:])
    if permute:
        x = jnp.roll(x, shift=1, axis=-2)

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



def gram_schmidt_fn(vectors: List[chex.Array]):
    n_vectors = len(vectors)
    orthonormal_vectors = []
    u0 = vectors[0] / jnp.linalg.norm(vectors[0])
    orthonormal_vectors.append(u0)

    for i in range(n_vectors - 1):
        current_vector_indx = i + 1
        temp_vector = vectors[current_vector_indx]
        for j in range(current_vector_indx):
            temp_vector = vector_rejection_single(temp_vector, vectors[j])
        orthonormal_vectors.append(temp_vector / jnp.linalg.norm(temp_vector))

    return orthonormal_vectors


def rotate_translate_x_and_a_2d(x_and_a, theta, translation):
    x, a = jnp.split(x_and_a, axis=-1, indices_or_sections=2)
    x_rot = rotate_translate_permute_2d(x, theta, translation)
    a_rot = rotate_translate_permute_2d(a, theta, translation)
    return jnp.concatenate([x_rot, a_rot], axis=-1)

def rotate_translate_x_and_a_3d(x_and_a, theta, phi, translation):
    x, a = jnp.split(x_and_a, axis=-1, indices_or_sections=2)
    x_rot = rotate_translate_permute_3d(x, theta, phi, translation)
    a_rot = rotate_translate_permute_3d(a, theta, phi, translation)
    return jnp.concatenate([x_rot, a_rot], axis=-1)


