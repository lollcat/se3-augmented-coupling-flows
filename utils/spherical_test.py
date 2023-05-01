import jax.numpy as jnp
import chex
import jax

from molboil.utils.test import assert_is_invariant, assert_is_equivariant

from utils.spherical import to_spherical_and_log_det, to_cartesian_and_log_det, _to_polar_and_log_det, polar_to_cartesian_and_log_det



def understanding_cross_product_2d():
    dim = 2
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    reflector = jnp.array([1., -1.])
    x1 = jnp.array([0.1, 0.1])
    x2 = jnp.array([0.0, 1.])
    x1_reflect = reflector*x1
    x2_reflect = reflector*x2

    # We see x_reflect_cross == (-1) * x_cross.
    x_cross = jnp.cross(x1, x2)
    x_reflect_cross = jnp.cross(x1_reflect, x2_reflect)

    x1_alt = jnp.array([0.1, -0.1])
    x_alt_cross = jnp.cross(x1_alt, x2)
    # Changing x = [x1, -x2] does not change the cross product.


def test_to_polar_each_quadrant():
    """Manually check each quadrant maps to the expected (r, theta) combination."""
    x_options = jnp.array([[1., 1.], [1., -1.], [-1., 1.], [-1., -1.]])
    expected_thetas = jnp.array([jnp.pi/4, -jnp.pi/4, jnp.pi*3/4, -jnp.pi*3/4])
    origin = jnp.array([0., 0.])
    ref = jnp.array([1., 0])
    reference = jnp.stack([origin, ref], axis=0)

    for i in range(4):
        x, expected_theta = x_options[i], expected_thetas[i]
        x_sph, _ = _to_polar_and_log_det(x, reference)
        assert x_sph[0] == jnp.sqrt(2)  # radius
        assert x_sph[1] == expected_theta




# def checking_to_spherical():  # test_to_polar_each_octogrant():
#     """Manually check each octogrant maps to the expected (r, theta, phi) combination."""
#     # For z = 1.
#     x_options = jnp.array([[1., 1., 1.], [1., 1., -1.], [1., -1., 1.], [1., -1., -1.]])
#
#     expected_thetas = jnp.array([jnp.pi/4, -jnp.pi/4, jnp.pi*3/4, -jnp.pi*3/4])
#     origin = jnp.array([0., 0., 0])
#     ref = jnp.array([[1., 0., 0.], [0., 1., 0.]])
#     reference = jnp.concatenate([origin, ref], axis=1)
#
#     for i in range(4):
#         x, expected_theta = x_options[i], expected_thetas[i]
#         x_sph, _ = _to_polar_and_log_det(x, reference, enforce_parity_invariance=True)
#         assert x_sph[0] == jnp.sqrt(2)  # radius
#         assert x_sph[1] == expected_theta
#
#         # Check parity invariance.
#         x_reflect = x * jnp.array([-1, 1.])
#         reference_reflect = reference * jnp.array([[-1, 1]])
#         x_sph_reflect, _ = _to_polar_and_log_det(x_reflect, reference_reflect, enforce_parity_invariance=True)
#         assert x_sph_reflect[0] == jnp.sqrt(2)  # radius
#         assert x_sph_reflect[1] == expected_theta


def test_does_not_smoke_and_invertible():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    batch_size = 3
    for dim in [3, 2]:
        reference = jax.random.normal(key1, (batch_size, dim, dim))
        x = jax.random.normal(key2, (batch_size, dim))

        x_sph, log_det_fwd = jax.vmap(to_spherical_and_log_det)(x, reference)

        x_, log_det_rv = jax.vmap(to_cartesian_and_log_det)(x_sph, reference)

        x_sph_, _ = jax.vmap(to_spherical_and_log_det)(x_, reference)
        chex.assert_trees_all_close(log_det_fwd, -log_det_rv, rtol=1e-6)
        chex.assert_trees_all_close(x, x_, rtol=1e-6)


def test_polar_does_not_smoke_and_invertible():
    origin = jnp.array([0, 0])
    y_axis = jnp.array([1, 0])*2
    reference = jnp.stack([origin, y_axis], axis=-2)
    x = jnp.array([1., 1.])

    x_sph, log_det_fwd = _to_polar_and_log_det(x, reference)
    r, theta = x_sph
    chex.assert_trees_all_close((r, theta), (jnp.sqrt(2.), jnp.pi*0.25))

    x_, log_det_rv = polar_to_cartesian_and_log_det(x_sph, reference)

    chex.assert_trees_all_close(x, x_, rtol=1e-5)
    chex.assert_trees_all_close(log_det_fwd, -log_det_rv, rtol=1e-5)


def test_to_spherical_is_invariant():
    # Currently for 3D.
    key = jax.random.PRNGKey(0)

    def invariant_fn(x_and_ref):
        # Note: Invariant to rotations to both x and reference, hence package them as single input for this test.
        chex.assert_shape(x_and_ref, (4, 1, 3))  # Need multiplicity axis for `assert_is_invariant`.
        x_and_ref = jnp.squeeze(x_and_ref, axis=1)
        x, reference = jnp.split(x_and_ref, [1,], axis=0)
        x = jnp.squeeze(x, axis=0)
        return to_spherical_and_log_det(x, reference)[0]

    event_shape = (4, 1, 3)
    assert_is_invariant(invariant_fn = invariant_fn, key=key, event_shape=event_shape, translate=True)


def test_to_spherical_and_back_is_equivariant(reflection_invariant: bool = True):
    # Currently for 3D.
    key = jax.random.PRNGKey(0)
    event_shape = (4, 1, 3)

    def equivariant_fn(x_and_ref):
        # Note: Invariant to rotations to both x and reference, hence package them as single input for this test.
        chex.assert_shape(x_and_ref, event_shape)  # Need multiplicity axis for `assert_is_invariant`.
        x_and_ref = jnp.squeeze(x_and_ref, axis=1)
        x, reference = jnp.split(x_and_ref, [1,], axis=0)
        x = jnp.squeeze(x, axis=0)
        sph_x = to_spherical_and_log_det(x, reference, parity_invariant=reflection_invariant)[0]
        sph_x_new = sph_x + 0.01  # Perform some transform to the spherical cooridnates.
        x_new = to_cartesian_and_log_det(sph_x_new, reference, parity_invariant=reflection_invariant)[0]
        return x_new[None, None, :]  # Need to have same rank as input.

    assert_is_equivariant(equivariant_fn=equivariant_fn, key=key, event_shape=event_shape, translate=True,
                          reflect=reflection_invariant)



if __name__ == '__main__':
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)

    # test_polar_does_not_smoke_and_invertible()
    # test_does_not_smoke_and_invertible()
    # test_to_spherical_is_invariant()
    test_to_spherical_and_back_is_equivariant(True)
    # test_to_spherical_and_back_is_equivariant(False)
    # print("All tests passed")
