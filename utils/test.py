import chex
import jax.numpy as jnp
import jax
import optax

from molboil.utils.test import test_fn_is_equivariant, test_fn_is_invariant, random_rotate_translate_perumute

from utils.numerical import param_count
from nets.base import NetsConfig, EGNNTorsoConfig, MLPHeadConfig, E3GNNTorsoConfig
from flow.aug_flow_dist import FullGraphSample, AugmentedFlow, AugmentedFlowParams


def get_minimal_nets_config(type = 'egnn'):
    nets_config = NetsConfig(type=type,
                             egnn_torso_config=EGNNTorsoConfig(
                                    n_blocks=2,
                                    mlp_units=(4,),
                                    n_vectors_hidden_per_vec_in=2,
                                    n_invariant_feat_hidden=3,
                                    name='e3gnn_v0_torso'),
                             e3gnn_torso_config=E3GNNTorsoConfig(
                                 n_blocks=2,
                                 mlp_units=(4,),
                                 n_vectors_hidden_per_vec_in=2,
                                 n_invariant_feat_hidden=3,
                                 name='e3gnn_torso'
                             ),
                             mlp_head_config=MLPHeadConfig((4,)),
                             )
    return nets_config

def bijector_test(bijector_forward, bijector_backward,
                  dim: int, n_nodes: int, n_aux: int):
    """Test that the bijector is equivariant, and that it's log determinant is invariant.
    Assumes bijectors are haiku transforms."""
    assert dim in (2, 3)
    event_shape = (n_nodes, n_aux+1, dim)

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    x_and_a = jax.random.normal(subkey, shape=event_shape) * 0.1
    centre_of_mass_original = jnp.mean(x_and_a, axis=-3)

    if x_and_a.dtype == jnp.float64:
        rtol = 1e-4
    else:
        rtol = 1e-3

    # Initialise bijector parameters.
    params = bijector_forward.init(key, x_and_a)
    print(f"bijector param count of {param_count(params)}")

    # Perform a forward pass, reverse and check the original `x_and_a` is recovered.
    x_and_a_new, log_det_fwd = bijector_forward.apply(params, x_and_a)
    chex.assert_trees_all_close(1 + jnp.mean(x_and_a_new, axis=-3), 1 + centre_of_mass_original,
                                rtol=rtol)  # Check subspace restriction.
    x_and_a_old, log_det_rev = bijector_backward.apply(params, x_and_a_new)

    # Check inverse gives original `x_and_a`
    magnitude_of_bijector = jnp.mean(jnp.linalg.norm(x_and_a_new - x_and_a, axis=-1))
    chex.assert_tree_all_finite(log_det_fwd)
    chex.assert_shape(log_det_fwd, ())
    chex.assert_trees_all_close((x_and_a - x_and_a_old)/magnitude_of_bijector,
                                jnp.zeros_like(x_and_a),
                                atol=rtol)
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
    centre_of_mass_original = jnp.mean(x_and_a, axis=-3)
    x_and_a_new, log_det_fwd = bijector_forward.apply(params, x_and_a)
    chex.assert_trees_all_close(1 + centre_of_mass_original, 1 + jnp.mean(x_and_a_new, axis=-3),
                                rtol=rtol)  # Check subspace restriction.
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
    def fake_loss_fn(params):
        x, log_det = bijector_forward.apply(params, x_and_a)
        return jnp.sum(log_det) + jnp.mean(x**2)
    grad = jax.grad(fake_loss_fn)(params)
    chex.assert_tree_all_finite(grad)
    assert optax.global_norm(grad) != 0.0


def get_max_diff_log_prob_invariance_test(samples: FullGraphSample,
                                          flow: AugmentedFlow,
                                          params: AugmentedFlowParams,
                                          key: chex.PRNGKey,
                                          permute: bool = False):

    log_prob_samples_only_fn = lambda x: flow.log_prob_apply(params, x)

    def group_action(x_and_a):
        return random_rotate_translate_perumute(x_and_a, key, permute=permute, translate=True)


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



