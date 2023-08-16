from typing import Tuple

import chex
import distrax
import jax.numpy as jnp
import jax.random

jax.config.update("jax_enable_x64", True)

def endow_cartesian_coords_with_rotation(
        a1_z_component: chex.Array,
        a1_rot: chex.Array,
        a1_x: chex.Array,
        a2_x: chex.Array,
        a2_y: chex.Array,
        a2_rot: chex.Array,
        a3_n_minus_1_xyz: chex.Array
        ) -> chex.Array:
    """
    Function that takes in the Cartesian coordinates calculated
    from the internal coordinates (which is invariant to global
    rotation) and lifts it into a space that includes degrees
    of freedom for global rotation.

    Args:
        a1_z_component: z component of the direction of the
            unit vector u_1 (for atom 1). Between -1 and 1.
        a1_rot: Rotation of atom 1 coordinates about the Z-axis.
            Between -\pi and \pi.
        a1_x: x component of atom 1 (equal to bond length 1).
        a2_x: x component of atom 2 (equal to cos(a2) * b2
            where a2 is the angle between atom 1 and atom 2
            and b2 is the bond length between atom 0 and
            atom 2).
        a2_y: y component of atom 2 (equal to cos(a2) * b2).
        a2_rot: Rotation that determines which plane atom 2
            lies on.

    Returns:
        y: Cartesian coordinates of atoms 1 to N-1.
    """
    n_nodes = a3_n_minus_1_xyz.shape[0] + 3

    a1_angle = jnp.pi / 2 - jnp.arccos(a1_z_component)

    # Rotation about X-axis.
    # Effects plane that atom 2 lies on.
    R1 = jnp.array([
            [1., 0., 0.],
            [0., jnp.cos(a2_rot), -jnp.sin(a2_rot)],
            [0., jnp.sin(a2_rot), jnp.cos(a2_rot)]
        ])
    # Rotation about Y-Axis and then Z-Axis
    # Together these effect the direction that
    # the first atom is placed.
    R2 = jnp.array([
            [jnp.cos(a1_angle),  0., jnp.sin(a1_angle)],
            [0., 1., 0],
            [-jnp.sin(a1_angle), 0., jnp.cos(a1_angle)]
        ])  # Rotate about Y-axis.
    R3 = jnp.array([
            [jnp.cos(a1_rot), -jnp.sin(a1_rot), 0],
            [jnp.sin(a1_rot), jnp.cos(a1_rot), 0],
            [0., 0., 1]
        ])  # Rotate about Z-axis.

    atom_1_coords = jnp.array([jnp.squeeze(a1_x), 0, 0.])
    atom_2_coords = jnp.array([a2_x, a2_y, 0.0])

    # Apply rotation to x to lift it into the full space endowed with rotation.
    x = jnp.concatenate([atom_1_coords[None],
                         atom_2_coords[None],
                         a3_n_minus_1_xyz], axis=0)
    w = jax.vmap(lambda x_: R3 @ R2 @ R1 @ x_)(x)
    chex.assert_shape(w, (n_nodes-1, 3))

    return w


def forward(x: chex.Array) -> chex.Array:
    a1_z_component, a1_rot, a1_x, a2_x, a2_y, a2_rot, a3_n_minus_1_xyz = \
        jnp.split(x, indices_or_sections=[1,2,3,4,5,6])
    a3_n_minus_1_xyz = jnp.reshape(a3_n_minus_1_xyz, (-1, 3))
    a1_z_component, a1_rot, a1_x, a2_x, a2_y, a2_rot = jax.tree_map(
        jnp.squeeze,
        (a1_z_component, a1_rot, a1_x, a2_x, a2_y, a2_rot))
    y = endow_cartesian_coords_with_rotation(a1_z_component, a1_rot, a1_x, a2_x, a2_y, a2_rot, a3_n_minus_1_xyz)
    return y.flatten()



def sample_and_log_prob_z1_t1_t2(key: chex.PRNGKey) -> Tuple[chex.Array, chex.Array]:
    key1, key2, key3 = jax.random.split(key, 3)
    z1, log_p_z1 = distrax.Uniform(low=-1, high=1.).sample_and_log_prob(seed=key1)
    t1, log_p_t1 = distrax.Uniform(low=-jnp.pi, high=jnp.pi).sample_and_log_prob(seed=key2)
    t2, log_p_t2 = distrax.Uniform(low=-jnp.pi, high=jnp.pi).sample_and_log_prob(seed=key2)
    return jnp.stack([z1, t1, t2]), log_p_z1 + log_p_t1 + log_p_t2


if __name__ == '__main__':

    # https://mathworld.wolfram.com/SpherePointPicking.html

    key = jax.random.PRNGKey(0)
    n_nodes = 8
    key1, key2, key3, key4, key5 = jax.random.split(key, 5)
    b1, a2, b2 = jax.random.uniform(key1, shape=(3,))
    a3_n_minus_1_xyz = jax.random.normal(key2, shape=(n_nodes-3, 3)).flatten()
    (z1, t1, t2), log_prob = sample_and_log_prob_z1_t1_t2(key2)

    x2, y2 = jnp.cos(a2) * b2, jnp.sin(a2) * b2
    x = jnp.concatenate([jnp.stack([z1, t1, b1, x2, y2, t2]), a3_n_minus_1_xyz])
    y = forward(x)

    jac = jax.jacfwd(forward)(x)
    sign, log_det = jnp.linalg.slogdet(jac)


    expected_log_det = 2*jnp.log(b1) + jnp.log(b2) + jnp.log(jnp.sin(a2))

    print(f"log prob {log_prob}")
    print(f"log det {log_det}")
    print(expected_log_det - log_det)


