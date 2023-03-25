import chex
import jax.numpy as jnp
import jax

from utils.numerical import rotate_translate_x_and_a_2d, rotate_translate_x_and_a_3d
from nets.base import NetsConfig, EgnnTorsoConfig, MLPHeadConfig, E3GNNTorsoConfig
from flow.aug_flow_dist import FullGraphSample, AugmentedFlow, AugmentedFlowParams


def get_minimal_nets_config(type = 'e3gnn'):
    nets_config = NetsConfig(type=type,
                             egnn_torso_config=EgnnTorsoConfig(mlp_units=(3,), h_embedding_dim=5),
                             e3gnn_torso_config=E3GNNTorsoConfig(
                                 n_blocks=2,
                                 mlp_units=(4,),
                                 n_vectors_hidden=2,
                                 n_invariant_feat_hidden=3,
                                 name='e3gnn_torso'
                             ),
                             mlp_head_config=MLPHeadConfig((4,)),
                             )
    return nets_config


def get_pairwise_distances(x):
    return jnp.linalg.norm(x - x[:, None], ord=2, axis=-1)


def test_fn_is_equivariant(equivariant_fn, key, event_shape):
    dim = event_shape[-1]
    assert dim in (2, 3)

    # Setup
    key1, key2, key3, key4 = jax.random.split(key, 4)
    x_and_a = jax.random.normal(key1, shape=event_shape) * 0.1

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

    # Compute equivariant_fn of both the original and rotated matrices.
    x_and_a_new = equivariant_fn(x_and_a)
    x_and_a_new_rot = equivariant_fn(x_and_a_rot)

    # Check that rotating x_and_a_new gives x_and_a_new_rot

    chex.assert_trees_all_close(x_and_a_new_rot, group_action(x_and_a_new), rtol=rtol)


def test_fn_is_invariant(invariante_fn, key, event_shape):
    dim = event_shape[-1]
    assert dim in (2, 3)

    # Setup
    key1, key2, key3, key4 = jax.random.split(key, 4)
    x_and_a = jax.random.normal(key1, shape=event_shape) * 0.1

    # Get rotated version of x_and_a.
    theta = jax.random.uniform(key2) * 2 * jnp.pi
    translation = jax.random.normal(key3, shape=(dim,)) * 10
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
                  dim: int, n_nodes: int, n_aux: int):
    """Test that the bijector is equivariant, and that it's log determinant is invariant.
    Assumes bijectors are haiku transforms."""
    assert dim in (2, 3)
    event_shape = (n_nodes, n_aux+1, dim)

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    x_and_a = jax.random.normal(subkey, shape=event_shape) * 0.1

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
    test_fn_is_equivariant(lambda x_and_a: bijector_forward.apply(params, x_and_a)[0], subkey, event_shape)
    key, subkey = jax.random.split(key)
    test_fn_is_equivariant(lambda x_and_a: bijector_backward.apply(params, x_and_a)[0], subkey, event_shape)

    # Check the change to the log det is invariant.
    key, subkey = jax.random.split(key)
    test_fn_is_invariant(lambda x_and_a: bijector_forward.apply(params, x_and_a)[1], subkey, event_shape)
    key, subkey = jax.random.split(key)
    test_fn_is_invariant(lambda x_and_a: bijector_backward.apply(params, x_and_a)[1], subkey, event_shape)


    # Forward reverse test but with a batch.
    batch_size = 11
    x_and_a = jax.random.normal(subkey, shape=(batch_size, *x_and_a.shape))*0.1
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


def get_max_diff_log_prob_invariance_test(samples: FullGraphSample,
                                          flow: AugmentedFlow,
                                          params: AugmentedFlowParams,
                                          key: chex.PRNGKey):
    log_prob_samples_only_fn = lambda x: flow.log_prob_apply(params, x)

    # Used in evaluation
    batch_size, n_nodes, n_var_groups, dim = samples.positions.shape
    n_aux = n_var_groups - 1

    key1, key2, key3 = jax.random.split(key, 3)

    # Get rotated version of x_and_a.
    theta = jax.random.uniform(key1, shape=(batch_size,)) * 2 * jnp.pi
    translation = jax.random.normal(key2, shape=(batch_size, dim))
    phi = jax.random.uniform(key3, shape=(batch_size,)) * 2 * jnp.pi

    def group_action(x_and_a):
        if dim == 2:
            x_and_a_rot = jax.vmap(rotate_translate_x_and_a_2d)(x_and_a, theta, translation)
        else:  # dim == 3:
            x_and_a_rot = jax.vmap(rotate_translate_x_and_a_3d)(x_and_a, theta, phi, translation)
        return x_and_a_rot


    positions_rot = group_action(samples.positions)
    samples_rot = FullGraphSample(features=samples.features, positions=positions_rot)

    chex.assert_trees_all_equal_shapes(samples, samples_rot)
    log_prob = log_prob_samples_only_fn(samples)
    log_prob_alt = log_prob_samples_only_fn(samples_rot)

    max_abs_diff = jnp.max(jnp.abs(log_prob_alt - log_prob))
    mean_abs_diff = jnp.mean(jnp.abs(log_prob_alt - log_prob))
    info = {"max_abs_diff_log_prob_after_group_action": max_abs_diff,
            "mean_abs_diff_log_prob_after_group_action": mean_abs_diff}
    return info



