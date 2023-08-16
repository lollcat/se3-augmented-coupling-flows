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
    o1, o2, b1, x2, y2 = jnp.split(x, indices_or_sections=[2,3,4,5])

    theta_star = jnp.pi/2 - o1[0]
    phi = o1[1]

    R1 = get_rotation_matrix(angle=jnp.squeeze(o2), fixed_axis=0)
    R2 = get_rotation_matrix(angle=theta_star, fixed_axis=1)
    R3 = get_rotation_matrix(angle=phi, fixed_axis=2)

    atom_1_coords = jnp.array([jnp.squeeze(b1), 0, 0.])
    atom_2_coords = jnp.array([jnp.squeeze(x2), jnp.squeeze(y2), 0.0])

    # Apply rotation to y.
    y_un_rotated = jnp.concatenate([atom_1_coords[None], atom_2_coords[None]], axis=0)
    y = jax.vmap(lambda y_unrot: R3 @ R2 @ R1 @ y_unrot)(y_un_rotated)
    chex.assert_shape(y, (2, 3))

    return y.flatten()

def forward_with_extra_nodes(x: chex.Array) -> chex.Array:
    o1, o2, b1, x2, y2, x3_Nm1 = jnp.split(x, indices_or_sections=[2,3,4,5,6])
    x3_Nm1 = jnp.reshape(x3_Nm1, (-1, 3))
    n_nodes = x3_Nm1.shape[0] + 3

    theta_star = jnp.pi/2 - o1[0]
    phi = o1[1]

    R1 = get_rotation_matrix(angle=jnp.squeeze(o2), fixed_axis=0)
    R2 = get_rotation_matrix(angle=theta_star, fixed_axis=1)
    R3 = get_rotation_matrix(angle=phi, fixed_axis=2)

    atom_1_coords = jnp.array([jnp.squeeze(b1), 0, 0.])
    atom_2_coords = jnp.array([jnp.squeeze(x2), jnp.squeeze(y2), 0.0])

    # Apply rotation to y.
    y_un_rotated = jnp.concatenate([atom_1_coords[None], atom_2_coords[None], x3_Nm1], axis=0)
    y = jax.vmap(lambda y_unrot: R3 @ R2 @ R1 @ y_unrot)(y_un_rotated)
    chex.assert_shape(y, (n_nodes - 1, 3))

    return y.flatten()



if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    n_nodes = 8
    key1, key2, key3, key4, key5 = jax.random.split(key, 5)
    x3_Nm1 = jax.random.normal(key1, shape=(n_nodes-3, 3))
    o1 = jax.random.uniform(key=key2, shape=(2,))
    o2 = jax.random.uniform(key=key3, shape=(1,))
    b1 = jax.random.uniform(key4, shape=(1,)) + 1.0
    atom2_xy = jax.random.normal(key5, shape=(2,))
    x_full = jnp.concatenate([o1, o2, b1, atom2_xy[0, None], atom2_xy[1, None], x3_Nm1.flatten()])
    x = jnp.concatenate([o1, o2, b1, atom2_xy[0, None], atom2_xy[1, None]])

    y = forward(x)

    jac = jax.jacfwd(forward)(x)
    sign, log_det = jnp.linalg.slogdet(jac)

    theta = o1[0]
    b2 = jnp.linalg.norm(atom2_xy)
    expected_log_det = 2*jnp.log(b1) + jnp.log(jnp.sin(theta)) + 0.616796
    pass

    # o2_ = jax.random.uniform(key=key3, shape=(1,))-0.5
    # x_ = jnp.concatenate([o1, o2_, b1, atom2_xy[0, None], atom2_xy[1, None]])
    # o1_ = o1.at[1].set(0.1)
    # x_ = jnp.concatenate([o1_, o2, b1, atom2_xy[0, None], atom2_xy[1, None]])
    atom2_xy_ = atom2_xy + 10.
    x_ = jnp.concatenate([o1, o2, b1, atom2_xy_[0, None], atom2_xy_[1, None]])
    jac_ = jax.jacfwd(forward)(x_)
    sign, log_det_ = jnp.linalg.slogdet(jac_)


    # Check x3_Nm1 has no effect
    jac_full = jax.jacfwd(forward_with_extra_nodes)(x_full)
    sign_full, log_det_full = jnp.linalg.slogdet(jac_full)
    assert log_det_full == log_det




