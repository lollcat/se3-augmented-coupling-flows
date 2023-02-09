import chex
import jax.numpy as jnp
import jax

from utils.numerical import rotate_translate_2d, rotate_translate_3d

def rotate_translate_x_and_a_2d(x_and_a, theta, translation):
    x, a = jnp.split(x_and_a, axis=-1, indices_or_sections=2)
    x_rot = rotate_translate_2d(x, theta, translation)
    a_rot = rotate_translate_2d(a, theta, translation)
    return jnp.concatenate([x_rot, a_rot], axis=-1)

def rotate_translate_x_and_a_3d(x_and_a, theta, phi, translation):
    x, a = jnp.split(x_and_a, axis=-1, indices_or_sections=2)
    x_rot = rotate_translate_3d(x, theta, phi, translation)
    a_rot = rotate_translate_3d(a, theta, phi, translation)
    return jnp.concatenate([x_rot, a_rot], axis=-1)


def get_pairwise_distances(x):
    return jnp.linalg.norm(x - x[:, None], ord=2, axis=-1)

def test_fn_is_equivariant(equivariant_fn, key, dim, n_nodes=20):
    assert dim in (2, 3)

    # Setup
    key1, key2, key3, key4 = jax.random.split(key, 4)
    x_and_a = jnp.zeros((n_nodes, dim * 2))
    x_and_a = x_and_a + jax.random.normal(key1, shape=x_and_a.shape)

    rtol = 1e-5 if x_and_a.dtype == jnp.float64 else 1e-3

    # Get rotated version of x_and_a.
    theta = jax.random.uniform(key2) * 2*jnp.pi
    translation = jax.random.normal(key3, shape=(dim,))
    phi = jax.random.uniform(key4) * 2 * jnp.pi

    def group_action(x_and_a):
        if dim == 2:
            x_and_a_rot = rotate_translate_x_and_a_2d(x_and_a, theta, translation)
        else:  #  dim == 3:
            x_and_a_rot = rotate_translate_x_and_a_3d(x_and_a, theta, phi, translation)
        return x_and_a_rot

    x_and_a_rot = group_action(x_and_a)

    chex.assert_trees_all_close(get_pairwise_distances(x_and_a_rot),
                                get_pairwise_distances(x_and_a), rtol=rtol)

    # Compute equivariant_fn of both the original and rotated matrices.
    x_and_a_new = equivariant_fn(x_and_a)
    x_and_a_new_rot = equivariant_fn(x_and_a_rot)

    # Check that rotating x_and_a_new gives x_and_a_new_rot

    chex.assert_trees_all_close(x_and_a_new_rot, group_action(x_and_a_new), rtol=rtol)

    chex.assert_trees_all_close(get_pairwise_distances(x_and_a_new_rot),
                                get_pairwise_distances(x_and_a_new), rtol=rtol)


def test_fn_is_invariant(invariante_fn, key, dim, n_nodes=7):
    assert dim in (2, 3)

    # Setup
    key1, key2, key3, key4 = jax.random.split(key, 4)
    x_and_a = jnp.zeros((n_nodes, dim * 2))
    x_and_a = x_and_a + jax.random.normal(key1, shape=x_and_a.shape) * 0.1

    # Get rotated version of x_and_a.
    theta = jax.random.uniform(key2) * 2 * jnp.pi
    translation = jax.random.normal(key3, shape=(dim,))
    phi = jax.random.uniform(key4) * 2 * jnp.pi

    def group_action(x_and_a):
        if dim == 2:
            x_and_a_rot = rotate_translate_x_and_a_2d(x_and_a, theta, translation)
        else:  #  dim == 3:
            x_and_a_rot = rotate_translate_x_and_a_3d(x_and_a, theta, phi, translation)
        return x_and_a_rot


    x_and_a_rot = group_action(x_and_a)

    # Compute invariante_fn of both the original and rotated matrices.
    out = invariante_fn(x_and_a)
    out_rot = invariante_fn(x_and_a_rot)

    # Check that rotating x_and_a_new_rot gives x_and_a_new
    if x_and_a.dtype == jnp.float64:
        rtol = 1e-6
    else:
        rtol = 1e-3
    chex.assert_trees_all_close(out, out_rot, rtol=rtol)



def bijector_test(bijector_forward, bijector_backward,
                  dim: int, n_nodes: int):
    """Test that the bijector is equivariant, and that it's log determinant is invariant.
    Assumes bijectors are haiku transforms."""
    assert dim in (2, 3)

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    # Create dummy x and a.
    x_and_a = jnp.zeros((n_nodes, dim*2))
    x_and_a = x_and_a + jax.random.normal(subkey, shape=x_and_a.shape)*0.1

    if x_and_a.dtype == jnp.float64:
        rtol = 1e-4
    else:
        rtol = 1e-3

    # Initialise bijector parameters.
    params = bijector_forward.init(key, x_and_a)

    # Perform a forward pass, reverse and check the original `x_and_a` is recovered.
    x_and_a_new, log_det_fwd = bijector_forward.apply(params, x_and_a)
    x_and_a_old, log_det_rev = bijector_backward.apply(params, x_and_a_new)

    # Check inverse gives original `x_and_a`
    chex.assert_tree_all_finite(log_det_fwd)
    chex.assert_shape(log_det_fwd, ())
    chex.assert_trees_all_close(x_and_a, x_and_a_old, rtol=rtol)
    chex.assert_trees_all_close(log_det_rev, -log_det_fwd, rtol=rtol)

    # Test the transformation is equivariant.
    key, subkey = jax.random.split(key)
    test_fn_is_equivariant(lambda x_and_a: bijector_forward.apply(params, x_and_a)[0], subkey, dim)
    key, subkey = jax.random.split(key)
    test_fn_is_equivariant(lambda x_and_a: bijector_backward.apply(params, x_and_a)[0], subkey, dim)

    # Check the change to the log det is invariant.
    key, subkey = jax.random.split(key)
    test_fn_is_invariant(lambda x_and_a: bijector_forward.apply(params, x_and_a)[1], subkey, dim)
    key, subkey = jax.random.split(key)
    test_fn_is_invariant(lambda x_and_a: bijector_backward.apply(params, x_and_a)[1], subkey, dim)


    # Forward reverse test but with a batch.
    batch_size = 11
    x_and_a = jnp.zeros((batch_size, n_nodes, dim*2))
    x_and_a = x_and_a + jax.random.normal(subkey, shape=x_and_a.shape)*0.1
    x_and_a_new, log_det_fwd = bijector_forward.apply(params, x_and_a)
    x_and_a_old, log_det_rev = bijector_backward.apply(params, x_and_a_new)
    chex.assert_shape(log_det_fwd, (batch_size,))
    chex.assert_trees_all_close(x_and_a, x_and_a_old, rtol=rtol)
    chex.assert_trees_all_close(log_det_rev, -log_det_fwd, rtol=rtol)

    # Check single sample and batch behavior is the same
    i = 4
    x_and_a_new_0, log_det_fwd_0 = bijector_forward.apply(params, x_and_a[i])
    x_and_a_old_0, log_det_rev_0 = bijector_backward.apply(params, x_and_a_new[i])
    chex.assert_trees_all_close(x_and_a_new[i], x_and_a_new_0, rtol=rtol)
    chex.assert_trees_all_close(x_and_a_old[i], x_and_a_old_0, rtol=rtol)

    # Test we can take grad log prob
    grad = jax.grad(lambda params, x_and_a: bijector_forward.apply(params, x_and_a)[1])(params, x_and_a[0])
    chex.assert_tree_all_finite(grad)


def get_max_diff_log_prob_invariance_test(samples, log_prob_fn, seed=0):
    batch_size, n_nodes, joint_dim = samples.shape
    dim = joint_dim // 2

    key = jax.random.PRNGKey(seed)
    key1, key2, key3 = jax.random.split(key, 3)

    # Get rotated version of x_and_a.
    theta = jax.random.uniform(key1, shape=(batch_size,)) * 2 * jnp.pi
    translation = jax.random.normal(key2, shape=(batch_size, dim))
    phi = jax.random.uniform(key3, shape=(batch_size,)) * 2 * jnp.pi

    def group_action(x_and_a):
        if dim == 2:
            x_and_a_rot = jax.vmap(rotate_translate_x_and_a_2d)(x_and_a, theta, translation)
        else:  #  dim == 3:
            x_and_a_rot = jax.vmap(rotate_translate_x_and_a_3d)(x_and_a, theta, phi, translation)
        return x_and_a_rot


    samples_rot = group_action(samples)

    log_prob = log_prob_fn(samples)

    log_prob_alt = log_prob_fn(samples_rot)

    max_abs_diff = jnp.max(jnp.abs(log_prob_alt - log_prob))
    mean_abs_diff = jnp.mean(jnp.abs(log_prob_alt - log_prob))
    mean_diff_x_space = jnp.mean(jnp.abs(samples - samples_rot))
    return {"max_abs_diff_log_prob_after_group_action": max_abs_diff, "mean_abs_diff_after_group_action": mean_abs_diff,
            "mean_diff_x_space_after_group_action": mean_diff_x_space}



