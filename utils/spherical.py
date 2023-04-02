from typing import Tuple

import chex
import jax.numpy as jnp

from molboil.utils.numerical import safe_norm


def to_spherical_and_log_det(x, reference) -> Tuple[chex.Array, chex.Array]:
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


def to_cartesian_and_log_det(sph_x, reference) -> \
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