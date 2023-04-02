import jax.numpy as jnp
import chex
import jax

from molboil.utils.test import test_fn_is_invariant, test_fn_is_equivariant

from utils.spherical import to_spherical_and_log_det, to_cartesian_and_log_det


def tesst_does_not_smoke_and_invertible():
    reference = jnp.eye(3)
    x = jnp.array([1., 1., 1.])

    x_sph, log_det_fwd = to_spherical_and_log_det(x, reference)

    x_, log_det_rv = to_cartesian_and_log_det(x_sph, reference)

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
    tesst_does_not_smoke_and_invertible()
    # tesst_to_spherical_is_invariant()
    # tesst_to_spherical_and_back_is_equivariant()
    print("All tests passed")