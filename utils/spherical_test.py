import jax.numpy as jnp
import chex
import jax

from molboil.utils.test import test_fn_is_invariant, test_fn_is_equivariant

from utils.spherical import to_spherical_and_log_det, to_cartesian_and_log_det, _to_polar_and_log_det, polar_to_cartesian_and_log_det


def tesst_does_not_smoke_and_invertible():
    key = jax.random.PRNGKey(0)
    key1, key2 = jax.random.split(key)
    batch_size = 100
    for dim in [2, 3]:
        reference = jax.random.normal(key1, (batch_size, dim, dim))
        x = jax.random.normal(key2, (batch_size, dim))

        x_sph, log_det_fwd = jax.vmap(to_spherical_and_log_det)(x, reference)

        x_, log_det_rv = jax.vmap(to_cartesian_and_log_det)(x_sph, reference)

        chex.assert_trees_all_close(log_det_fwd, -log_det_rv, rtol=1e-5)
        chex.assert_trees_all_close(x, x_, rtol=1e-4)



def tesst_polar_does_not_smoke_and_invertible():
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


# def tesst_to_spherical_is_invariant():
#     key = jax.random.PRNGKey(0)
#
#     def invariant_fn(x):
#         chex.assert_shape(x, (4, 1, 3))  # Need multiplicity axis for `test_fn_is_invariant`.
#         x = jnp.squeeze(x, axis=1)
#         x, a, b, c = jnp.split(x, 4, axis=0)
#         x, a, b, c = jnp.squeeze(x, axis=0), jnp.squeeze(a, axis=0), jnp.squeeze(b, axis=0), jnp.squeeze(c, axis=0)
#         return to_spherical_and_log_det(x, a, b, c)[0]
#
#
#     event_shape = (4, 1, 3)
#     test_fn_is_invariant(invariant_fn = invariant_fn, key=key, event_shape=event_shape, translate=True)
#
#
# def tesst_to_spherical_and_back_is_equivariant():
#     key = jax.random.PRNGKey(0)
#
#     def equivariant_fn(x):
#         chex.assert_shape(x, (4, 1, 3))  # Need multiplicity axis for `test_fn_is_invariant`.
#         x = jnp.squeeze(x, axis=1)
#         x, a, b, c = jnp.split(x, 4, axis=0)
#         x, a, b, c = jnp.squeeze(x, axis=0), jnp.squeeze(a, axis=0), jnp.squeeze(b, axis=0), jnp.squeeze(c, axis=0)
#         sph_x = to_spherical_and_log_det(x, a, b, c)
#         sph_x_new = sph_x + 0.01
#         x_new = to_cartesian(sph_x_new, a, b, c)
#         return x_new[None, None, :]  # Need to have same rank as input.
#
#
#     event_shape = (4, 1, 3)
#     test_fn_is_equivariant(equivariant_fn=equivariant_fn, key=key, event_shape=event_shape, translate=True)



if __name__ == '__main__':
    tesst_polar_does_not_smoke_and_invertible()
    tesst_does_not_smoke_and_invertible()
    # tesst_to_spherical_is_invariant()
    # tesst_to_spherical_and_back_is_equivariant()
    print("All tests passed")