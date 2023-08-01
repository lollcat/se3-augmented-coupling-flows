from jax import numpy as jnp
import chex


def calc_bonds(ind1: chex.Array, ind2: chex.Array, coords: chex.Array):
    """Calculate bond lengths

    Parameters
    ----------
    ind1: Array
        A n_bond x 3 tensor of indices for the coordinates of particle 1
    ind2: Array
        A n_bond x 3 tensor of indices for the coordinates of particle 2
    coords: Array
        A n_coord or n_batch x n_coord array of flattened input coordinates
    """
    p1 = coords[..., ind1]
    p2 = coords[..., ind2]
    return jnp.linalg.norm(p2 - p1, axis=-1)


def calc_angles(ind1, ind2, ind3, coords):
    b = coords[..., ind1]
    c = coords[..., ind2]
    d = coords[..., ind3]
    bc = b - c
    bc = bc / jnp.linalg.norm(bc, axis=-1, keepdims=True)
    cd = d - c
    cd = cd / jnp.linalg.norm(cd, axis=-1, keepdims=True)
    cos_angle = jnp.sum(bc * cd, axis=-1)
    angle = jnp.arccos(cos_angle)
    return angle


def calc_dihedrals(ind1, ind2, ind3, ind4, coords):
    a = coords[..., ind1]
    b = coords[..., ind2]
    c = coords[..., ind3]
    d = coords[..., ind4]

    b0 = a - b
    b1 = c - b
    b1 = b1 / jnp.linalg.norm(b1, axis=-1, keepdims=True)
    b2 = d - c

    v = b0 - jnp.sum(b0 * b1, axis=-1, keepdims=True) * b1
    w = b2 - jnp.sum(b2 * b1, axis=-1, keepdims=True) * b1
    x = jnp.sum(v * w, axis=-1)
    b1xv = jnp.cross(b1, v)
    y = jnp.sum(b1xv * w, axis=-1)
    angle = jnp.arctan2(y, x)
    return -angle


def reconstruct_cart(cart, ref_atoms, bonds, angles, dihs):
    # Get the positions of the 4 reconstructing atoms
    p1 = cart[..., ref_atoms[..., 0], :]
    p2 = cart[..., ref_atoms[..., 1], :]
    p3 = cart[..., ref_atoms[..., 2], :]

    # Compute the log jacobian determinant.
    jac = jnp.sum(
        2 * jnp.log(jnp.abs(bonds))
        + jnp.log(jnp.abs(jnp.sin(angles))),
        axis=-1
    )

    bonds = jnp.expand_dims(bonds, -1)
    angles = jnp.expand_dims(angles, -1)
    dihs = jnp.expand_dims(dihs, -1)

    # Reconstruct the position of p4
    v1 = p1 - p2
    v2 = p1 - p3

    n = jnp.cross(v1, v2)
    n = n / jnp.linalg.norm(n, axis=-1, keepdims=True)
    nn = jnp.cross(v1, n)
    nn = nn / jnp.linalg.norm(nn, axis=-1, keepdims=True)

    n = n * jnp.sin(dihs)
    nn = nn * jnp.cos(dihs)

    v3 = n + nn
    v3 = v3 / jnp.linalg.norm(v3, axis=-1, keepdims=True)
    v3 = v3 * bonds * jnp.sin(angles)

    v1 = v1 / jnp.linalg.norm(v1, axis=-1, keepdims=True)
    v1 = v1 * bonds * jnp.cos(angles)

    # Store the final position in x
    new_cart = p1 + v3 - v1

    return new_cart, jac