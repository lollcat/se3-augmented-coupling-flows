from typing import Tuple

import chex
import jax.numpy as jnp

from molboil.utils.numerical import safe_norm
from molboil.utils.numerical import rotate_2d

def to_spherical_and_log_det(x, reference) -> Tuple[chex.Array, chex.Array]:
    chex.assert_rank(x, 1)
    chex.assert_rank(reference, 2)
    dim = x.shape[0]
    if dim == 3:
        return _to_spherical_and_log_det(x, reference)
    else:
        assert dim == 2
        return _to_polar_and_log_det(x, reference)


def to_cartesian_and_log_det(sph_x, reference) -> \
        Tuple[chex.Array, chex.Array]:
    chex.assert_rank(sph_x, 1)
    chex.assert_rank(reference, 2)
    dim = sph_x.shape[0]
    if dim == 4:  # current hack
        # assert
        return _to_cartesian_and_log_det(sph_x, reference)
    else:
        assert dim == 2
        return polar_to_cartesian_and_log_det(sph_x, reference)



def _to_polar_and_log_det(x, reference):
    chex.assert_shape(x, (2,))
    origin, y = jnp.split(reference, (1,), axis=-2)
    y, origin = jnp.squeeze(y), jnp.squeeze(origin)
    chex.assert_equal_shape((origin, y, x))

    vector_x = x - origin
    vector_y = y - origin

    # Calculate radius.
    r = safe_norm(vector_x, axis=-1)
    unit_vector_x = vector_x / r


    # Calculate angle
    norm_y = safe_norm(y, axis=-1)
    unit_vector_y_axis = vector_y / norm_y
    x_proj_norm = jnp.dot(unit_vector_x, unit_vector_y_axis)
    perp_line = jnp.cross(unit_vector_y_axis, unit_vector_x)
    theta = jnp.arctan2(perp_line, x_proj_norm)
    log_det = - jnp.log(r)

    x_polar = jnp.stack([r, theta])
    return x_polar, log_det


def polar_to_cartesian_and_log_det(x_polar, reference):
    chex.assert_shape(x_polar, (2,))
    origin, y = jnp.split(reference, (1,), axis=-2)
    y, origin = jnp.squeeze(y), jnp.squeeze(origin)
    chex.assert_equal_shape((origin, y, x_polar))
    y_vector = y - origin

    r, theta = x_polar
    y_unit_vec = y_vector / safe_norm(y_vector)

    log_det = jnp.log(r)
    x_vector = rotate_2d(r * y_unit_vec, theta)
    x = origin + x_vector
    return x, log_det


def _to_spherical_and_log_det(x, reference) -> Tuple[chex.Array, chex.Array]:
    chex.assert_rank(x, 1)
    dim = x.shape[0]
    origin, _ = jnp.split(reference, (1,), axis=-2)
    origin = jnp.squeeze(origin, axis=-2)
    chex.assert_equal_shape([x, origin])

    # Only transform d for now.
    vector = x - origin
    norm = safe_norm(vector, axis=-1, keepdims=True)
    unit_vector = vector / norm
    x = jnp.concatenate([norm, unit_vector], axis=-1)
    log_det = - jnp.log(norm) * (dim - 1)
    return x, jnp.squeeze(log_det)


def _to_cartesian_and_log_det(sph_x, reference) -> \
        Tuple[chex.Array, chex.Array]:
    chex.assert_rank(sph_x, 1)
    dim = sph_x.shape[0] - 1  # TODO: Update when this is no longer hacky
    origin, _ = jnp.split(reference, (1,), axis=-2)
    origin = jnp.squeeze(origin, axis=-2)

    # Only transform d for now.
    norm, unit_vector = jnp.split(sph_x, (1,), axis=-1)
    x = origin + norm * unit_vector
    log_det = jnp.log(norm) * (dim - 1)
    return x, jnp.squeeze(log_det)

