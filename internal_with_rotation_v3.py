"""TODO
- test log det
- test log prob integrates to 1
- visualise rotations and make sure they are uniform
"""
from typing import Tuple

import chex
import distrax
import jax.numpy as jnp
import jax.random

def get_rotation_matrix(angle: chex.Array, fixed_axis: int) -> chex.Array:
    if fixed_axis == 0:
        rotation = jnp.array([
            [1., 0., 0.],
            [0., jnp.cos(angle), -jnp.sin(angle)],
            [0., jnp.sin(angle), jnp.cos(angle)]
        ])
    elif fixed_axis == 1:
        rotation = jnp.array([
            [jnp.cos(angle), 0., jnp.sin(angle)],
            [0., 1., 0],
            [-jnp.sin(angle), 0., jnp.cos(angle)]
        ])
    else:
        assert fixed_axis == 2
        rotation = jnp.array([
            [jnp.cos(angle), -jnp.sin(angle), 0],
            [jnp.sin(angle), jnp.cos(angle), 0],
            [0., 0., 1.]
        ])
    return rotation

def forward(x: chex.Array) -> chex.Array:
    """
    z1, t1, b1, a2, t2, b2 = x
    z1: z coordinate of atom 1 [-1, 1]
    t1: torsion of atom 1: [-jnp.pi, jnp.pi]
    b1: bond length of atom 1
    x2: x coord of atom 2
    y2: y coord of atom 2
    t2: torsion of angle 2
    """
    z1, t1, b1, x2, y2, t2 = x

    x1 = jnp.sqrt(1 - z1**2)
    angle_1 = jnp.arctan2(z1, x1)

    R1 = get_rotation_matrix(angle=t2, fixed_axis=0)
    R2 = get_rotation_matrix(angle=angle_1, fixed_axis=1)
    R3 = get_rotation_matrix(angle=t1, fixed_axis=2)

    atom_1_coords = jnp.array([jnp.squeeze(b1), 0, 0.])
    atom_2_coords = jnp.array([x2, y2, 0.0])

    # Apply rotation to y.
    y_un_rotated = jnp.concatenate([atom_1_coords[None], atom_2_coords[None]], axis=0)
    y = jax.vmap(lambda y_unrot: R3 @ R2 @ R1 @ y_unrot)(y_un_rotated)
    chex.assert_shape(y, (2, 3))

    return y.flatten()


def sample_and_log_prob_z1_t1_t2(key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
    key1, key2, key3 = jax.random.split(key, 3)
    z1, log_p_z1 = distrax.Uniform(low=-1., high=1.).sample_and_log_prob(seed=key1)
    t1, log_p_t1 = distrax.Uniform(low=-jnp.pi, high=jnp.pi).sample_and_log_prob(seed=key2)
    t2, log_p_t2 = distrax.Uniform(low=-jnp.pi, high=jnp.pi).sample_and_log_prob(seed=key2)
    return jnp.stack([z1, t1, t2]), log_p_z1 + log_p_t1 + log_p_t2


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    n_nodes = 8
    key1, key2, key3, key4, key5 = jax.random.split(key, 5)
    b1, a2, b2 = jax.random.uniform(key1, shape=(3,))
    (z1, t1, t2), log_prob = sample_and_log_prob_z1_t1_t2(key2)

    x2, y2 = jnp.cos(a2) * b2, jnp.sin(a2) * b2
    x = jnp.stack([z1, t1, b1, x2, y2, t2])
    y = forward(x)

    jac = jax.jacfwd(forward)(x)
    sign, log_det = jnp.linalg.slogdet(jac)


    expected_log_det = 2*jnp.log(b1) + jnp.log(b2) + jnp.log(jnp.sin(a2))

    print(f"log prob {log_prob}")
    print(f"log det {expected_log_det}")
    print(expected_log_det - log_det)


