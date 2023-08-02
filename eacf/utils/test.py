from typing import Callable

from functools import partial

import chex
import jax
import jax.numpy as jnp

from eacf.utils.numerical import rotate_translate_permute_general


def random_rotate_translate_permute(x: chex.Array, key: chex.PRNGKey,
                                    reflect: bool = False,
                                    translate: bool = False, permute: bool = False) -> chex.Array:
    """Perform's random group action on x depending on the random key.
    Assumes the shape of x is [..., n_nodes, multiplicity, dim]."""
    n_nodes, multiplicity, dim = x.shape[-3:]
    leading_shape = x.shape[:-3]

    key1, key2, key3 = jax.random.split(key, 3)

    theta = jax.random.uniform(key1, shape=leading_shape) * 2*jnp.pi
    translation = jax.random.normal(key2, shape=(*leading_shape, dim,)) * (1 if translate else 0)
    phi = jax.random.uniform(key3, shape=leading_shape) * 2 * jnp.pi  # Only used if x is 3d.

    base_function = partial(rotate_translate_permute_general,  permute=permute)

    # We apply the same group action to all multiplicities.
    base_function = jax.vmap(base_function, in_axes=(-2, None, None, None), out_axes=-2)

    # We apply a different group action to each batch element (each dim in leading_shape).
    for i in range(len(leading_shape)):
        base_function = jax.vmap(base_function)
    x_g = base_function(x, translation, theta, phi)
    if reflect:
        flip = jnp.ones_like(x)
        flip = flip.at[..., 0].set(-1)
        x_g = x_g*flip
    return x_g


def assert_is_equivariant(equivariant_fn: Callable[[chex.Array], chex.Array], key: chex.PRNGKey,
                          event_shape: chex.Shape, translate: bool = False, permute: bool = False,
                          reflect: bool = False):
    """Test `equivariant_fn` is equivariant. Inputs random samples matrix of shape `event_shape`
    to the equivariant function."""
    dim = event_shape[-1]
    assert dim in (2, 3)

    # Setup
    key1, key2 = jax.random.split(key, 2)
    x = jax.random.normal(key1, shape=event_shape) * 0.1
    rtol = 1e-5 if x.dtype == jnp.float64 else 1e-3

    def group_action(x):
        return random_rotate_translate_permute(x, key2, permute=permute, translate=translate, reflect=reflect)


    x_rot = group_action(x)

    # Compute equivariant_fn of both the original and rotated matrices.
    x_new = equivariant_fn(x)
    x_new_rot = equivariant_fn(x_rot)
    chex.assert_equal_shape((x_new, x_new_rot))

    # Check that rotating x_and_a_new gives x_new_rot
    chex.assert_trees_all_close(x_new_rot, group_action(x_new), rtol=rtol)


def assert_is_invariant(invariant_fn: Callable[[chex.Array], chex.Array], key: chex.PRNGKey,
                        event_shape: chex.Shape, translate: bool = False, reflect: bool = False):
    """Test `invariant_fn` is invariant. Inputs random samples matrix of shape `event_shape`
    to the invariant function."""
    # TODO: Add permute option. This requires that we apply the group action to the invariant outputs, which
    #  is currently not supported.

    dim = event_shape[-1]
    assert dim in (2, 3)

    # Setup
    key1, key2 = jax.random.split(key, 2)
    x = jax.random.normal(key1, shape=event_shape) * 0.1
    rtol = 1e-5 if x.dtype == jnp.float64 else 1e-3

    def group_action(x):
        return random_rotate_translate_permute(x, key2, permute=False, translate=translate, reflect=reflect)


    x_rot = group_action(x)

    # Compute invariante_fn of both the original and rotated matrices.
    out = invariant_fn(x)
    out_rot = invariant_fn(x_rot)
    chex.assert_equal_shape((out, out_rot))

    # Check that rotating x_new_rot gives x_and_a_new
    chex.assert_trees_all_close(out, out_rot, rtol=rtol)
