"""TODO
- test log det
- test log prob integrates to 1
- visualise rotations and make sure they are uniform
"""


import chex
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
    a1, t1, b1, a2, t2, b2 = x

    R1 = get_rotation_matrix(angle=t2, fixed_axis=0)
    R2 = get_rotation_matrix(angle=jnp.pi/2 - a1, fixed_axis=1)
    R3 = get_rotation_matrix(angle=t1, fixed_axis=2)

    atom_1_coords = jnp.array([jnp.squeeze(b1), 0, 0.])
    atom_2_coords = jnp.array([jnp.cos(a2) * b2, jnp.sin(a2) * b2, 0.0])

    # Apply rotation to y.
    y_un_rotated = jnp.concatenate([atom_1_coords[None], atom_2_coords[None]], axis=0)
    y = jax.vmap(lambda y_unrot: R3 @ R2 @ R1 @ y_unrot)(y_un_rotated)
    chex.assert_shape(y, (2, 3))

    return y.flatten()


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    n_nodes = 8
    key1, key2, key3, key4, key5 = jax.random.split(key, 5)
    x = jax.random.uniform(key, shape=(6,))
    a1, t1, b1, a2, t2, b2 = x

    y = forward(x)

    jac = jax.jacfwd(forward)(x)
    sign, log_det = jnp.linalg.slogdet(jac)


    expected_log_det = 2*jnp.log(b1) + jnp.log(jnp.sin(a1)) + 2*jnp.log(b2) + jnp.log(jnp.sin(a2))

    print(expected_log_det, log_det)

