from typing import Optional

import chex
import jax
import jax.numpy as jnp

def safe_norm(x: jnp.ndarray, axis: int = None, keepdims=False) -> jnp.ndarray:
    """nan-safe norm. Copied from mace-jax"""
    x2 = jnp.sum(x**2, axis=axis, keepdims=keepdims)
    return jnp.where(x2 == 0.0, 0.0, jnp.where(x2 == 0, 1.0, x2) ** 0.5)


def rotate_3d(x: chex.Array, theta: chex.Array, phi: chex.Array) -> chex.Array:
    """Assumes x is of shape [n_nodes, 3]."""
    chex.assert_shape(theta, ())
    chex.assert_rank(x, 2)  # [n_nodes, dim]
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


def rotate_translate_permute_3d(x: chex.Array, theta: chex.Array, phi: chex.Array,
                                translation: chex.Array,
                                rotate_first: bool = True, permute: bool = False) -> chex.Array:
    """Assumes x is of shape [n_nodes, 3]."""
    chex.assert_shape(theta, ())
    chex.assert_shape(phi, ())
    chex.assert_shape(translation, x.shape[-1:])

    if permute:
        x = jnp.roll(x, shift=1, axis=-2)

    if rotate_first:
        return rotate_3d(x, theta, phi) + translation[None, :]
    else:
        return rotate_3d(x + translation[None, :], theta, phi)


def rotate_2d(x: chex.Array, theta: chex.Array) -> chex.Array:
    """Assumes x is of shape [n_nodes, 2]."""
    rotation_matrix = jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)],
         [jnp.sin(theta), jnp.cos(theta)]]
    )
    if len(x.shape) == 2:
        return jax.vmap(jnp.matmul, in_axes=(None, 0))(rotation_matrix, x)
    else:
        chex.assert_rank(x, 1)
        return jnp.matmul(rotation_matrix, x)


def rotate_translate_permute_2d(x: chex.Array, theta: chex.Array, translation: chex.Array,
                                rotate_first: bool = True, permute: bool = False):
    """Assumes x is of shape [n_nodes, 2]."""
    chex.assert_shape(theta, ())
    chex.assert_shape(translation, x.shape[-1:])
    if permute:
        x = jnp.roll(x, shift=1, axis=-2)

    if rotate_first:
        return rotate_2d(x, theta) + translation[None, :]
    else:
        return rotate_2d(x + translation[None, :], theta)


def rotate_translate_permute_general(x: chex.Array,
                                     translation: chex.Array,
                                     theta: chex.Array,
                                     phi: Optional[chex.Array] = None,
                                     rotate_first: bool = True, permute: bool = False):
    """Assumes x is of shape [n_nodes, dim]."""
    chex.assert_rank(x, 2)
    n_nodes, dim = x.shape
    chex.assert_shape(translation, (dim,))
    if dim == 2:
        return rotate_translate_permute_2d(x, theta, translation, rotate_first, permute)
    elif dim == 3:
        assert phi is not None
        chex.assert_shape(phi, ())
        return rotate_translate_permute_3d(x, theta, phi, translation, rotate_first, permute)
    else:
        raise NotImplementedError
