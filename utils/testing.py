from typing import Optional

import chex
import jax.numpy as jnp
import jax
import optax

from molboil.utils.test import assert_is_equivariant, assert_is_invariant, random_rotate_translate_permute
from molboil.train.base import maybe_masked_mean, maybe_masked_max

from utils.numerical import param_count
from nets.base import NetsConfig, EGNNTorsoConfig, MLPHeadConfig, TransformerConfig
from flow.aug_flow_dist import FullGraphSample, AugmentedFlow, AugmentedFlowParams


def get_minimal_nets_config(type = 'egnn'):
    nets_config = NetsConfig(type=type,
                            embedding_dim = 32,
                            num_discrete_feat=2,
                             egnn_torso_config=EGNNTorsoConfig(
                                    n_blocks=2,
                                    mlp_units=(4,),
                                    n_vectors_hidden_per_vec_in=2,
                                    n_invariant_feat_hidden=3,
                                    name='egnn_v0_torso'),
                             mlp_head_config=MLPHeadConfig((4,)),
                             non_equivariant_transformer_config=TransformerConfig(output_dim=6,
                                                                                  key_size_per_node_dim_in=2,
                                                                                  n_layers=2, mlp_units=(4,))

                             )
    return nets_config

def check_bijector_properties(bijector_forward, bijector_backward,
                              dim: int, n_nodes: int, n_aux: int, test_rotation_equivariance: bool = True):
    """Test that the bijector is equivariant, and that it's log determinant is invariant.
    Assumes bijectors are haiku transforms."""
    assert dim in (2, 3)
    event_shape = (n_nodes, n_aux+1, dim)

    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    x_and_a = jax.random.normal(subkey, shape=event_shape)
    x_and_a = x_and_a - jnp.mean(x_and_a[:, 0], axis=0, keepdims=True)[:, None]

    if x_and_a.dtype == jnp.float64:
        rtol = 1e-4
    else:
        rtol = 1e-3

    # Initialise bijector parameters.
    params = bijector_forward.init(key, x_and_a)
    print(f"bijector param count of {param_count(params)}")

    # Check parameter init not effected by batch or no-batch.
    params_batch_init = bijector_forward.init(key, jnp.repeat(x_and_a[None], 10, axis=0))
    chex.assert_trees_all_equal(params, params_batch_init)
    chex.assert_trees_all_equal_structs(params_batch_init, params)

    # Perform a forward pass, reverse and check the original `x_and_a` is recovered.
    x_and_a_new, log_det_fwd = bijector_forward.apply(params, x_and_a)
    centre_of_mass = jnp.mean(x_and_a_new[:, 0], axis=0)
    chex.assert_trees_all_close(1 + centre_of_mass, 1 + jnp.zeros_like(centre_of_mass),
                                rtol=rtol)  # Check subspace restriction.
    centre_of_mass_aug = jnp.mean(x_and_a_new[:, 1:], axis=0)
    assert (jnp.abs(centre_of_mass_aug) > 0.001).all()

    x_and_a_old, log_det_rev = bijector_backward.apply(params, x_and_a_new)

    # Check inverse gives original `x_and_a`
    magnitude_of_bijector = jnp.mean(jnp.linalg.norm(x_and_a_new - x_and_a, axis=-1))
    chex.assert_tree_all_finite(log_det_fwd)
    chex.assert_shape(log_det_fwd, ())
    chex.assert_trees_all_close(log_det_rev, -log_det_fwd, rtol=rtol)
    chex.assert_trees_all_close(1 + (x_and_a - x_and_a_old)/magnitude_of_bijector, jnp.ones_like(x_and_a),
                                atol=0.01)

    if test_rotation_equivariance:
        # Test the transformation is equivariant.
        key, subkey = jax.random.split(key)
        assert_is_equivariant(lambda x_and_a: bijector_forward.apply(params, x_and_a)[0], subkey, event_shape)
        key, subkey = jax.random.split(key)
        assert_is_equivariant(lambda x_and_a: bijector_backward.apply(params, x_and_a)[0], subkey, event_shape)

        # Check the change to the log det is invariant.
        key, subkey = jax.random.split(key)
        assert_is_invariant(lambda x_and_a: bijector_forward.apply(params, x_and_a)[1], subkey, event_shape)
        key, subkey = jax.random.split(key)
        assert_is_invariant(lambda x_and_a: bijector_backward.apply(params, x_and_a)[1], subkey, event_shape)


    # Forward reverse test but with a batch.
    batch_size = 101
    x_and_a = jax.random.normal(subkey, shape=(batch_size, *x_and_a.shape))
    x_and_a = x_and_a - jnp.mean(x_and_a[:, :, 0], axis=1, keepdims=True)[:, None]
    x_and_a_new, log_det_fwd = bijector_forward.apply(params, x_and_a)
    centre_of_mass = jnp.mean(x_and_a_new[:, :, 0], axis=1)
    chex.assert_trees_all_close(1 + centre_of_mass, 1 + jnp.zeros_like(centre_of_mass),
                                rtol=rtol)  # Check subspace restriction.
    centre_of_mass_aug = jnp.mean(x_and_a_new[:, :, 1:], axis=0)
    # (jnp.abs(jnp.mean(x_and_a[:, :, 1:], axis=0)) > 0.001).sum() is also interesting to eyeball.
    assert (jnp.abs(centre_of_mass_aug) > 0.0001).all()
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


def get_checks_for_flow_properties(samples: FullGraphSample,
                                   flow: AugmentedFlow,
                                   params: AugmentedFlowParams,
                                   key: chex.PRNGKey,
                                   permute: bool = False,
                                   mask: Optional[chex.Array] = None):
    """Tests invariance of the flow log prob. Also check that the
     forward and reverse of the bijector is consistent.
    """
    batch_size, n_nodes, mult, dim = samples.positions.shape

    log_prob_samples_only_fn = lambda x: flow.log_prob_apply(params, x)

    key1, key2, key3 = jax.random.split(key, 3)

    # Rotation.
    def group_action(x_and_a):
        return random_rotate_translate_permute(x_and_a, key1, permute=permute, translate=True)


    positions_rot = group_action(samples.positions)
    samples_rot = samples._replace(positions=positions_rot)

    chex.assert_trees_all_equal_shapes(samples, samples_rot)
    log_prob = log_prob_samples_only_fn(samples)
    log_prob_alt = log_prob_samples_only_fn(samples_rot)

    max_abs_diff = maybe_masked_max(jnp.abs(log_prob_alt - log_prob), mask)
    mean_abs_diff = maybe_masked_mean(jnp.abs(log_prob_alt - log_prob), mask)
    info = {"max_abs_diff_log_prob_after_group_action": max_abs_diff,
            "mean_abs_diff_log_prob_after_group_action": mean_abs_diff}


    # Reflection.
    flip = jnp.ones((1, 1, 1, dim))  # batch, n_nodes, mult, dim
    flip = flip.at[:, :, :, 0].set(-1.)
    samples_flip = FullGraphSample(features=samples.features, positions=samples.positions*flip)
    log_prob_flip = log_prob_samples_only_fn(samples_flip)
    abs_diff = jnp.abs(log_prob_flip - log_prob)
    max_abs_diff = maybe_masked_max(abs_diff, mask)
    mean_abs_diff = maybe_masked_mean(abs_diff, mask)
    info.update(max_abs_diff_log_prob_after_reflection=max_abs_diff,
            mean_abs_diff_log_prob_after_reflection=mean_abs_diff)


    # Reflection 2.
    flip = jnp.ones((1, 1, 1, dim))  # batch, n_nodes, mult, dim
    flip = flip.at[:, :, :, 1].set(-1.)
    samples_flip = FullGraphSample(features=samples.features, positions=samples.positions*flip)
    log_prob_flip_ = log_prob_samples_only_fn(samples_flip)
    abs_diff = jnp.abs(log_prob_flip - log_prob_flip_)
    max_abs_diff = maybe_masked_max(abs_diff, mask)
    mean_abs_diff = maybe_masked_mean(abs_diff, mask)
    info.update(max_abs_diff_log_prob_two_reflections=max_abs_diff,
            mean_abs_diff_log_prob_two_reflections=mean_abs_diff)


    # Test bijector forward vs reverse.
    # Recent test samples.
    samples = samples._replace(
        positions=samples.positions - jnp.mean(samples.positions[:, :, 0, :], axis=1, keepdims=True)[:, :, None])
    sample_latent, log_det_rv, extra_rv = flow.bijector_inverse_and_log_det_with_extra_apply(params.bijector, samples)
    samples_, log_det_fwd, extra_fwd = flow.bijector_forward_and_log_det_with_extra_apply(
        params.bijector, sample_latent)
    info.update(max_abs_diff_log_det_forward_reverse=maybe_masked_max(jnp.abs(log_det_fwd + log_det_rv), mask=mask))
    info.update(mean_abs_diff_log_det_forward_reverse=maybe_masked_mean(jnp.abs(log_det_fwd + log_det_rv), mask=mask))
    info.update(mean_diff_samples_flow_inverse_forward=maybe_masked_mean(
        jnp.mean(jnp.abs(samples_.positions - samples.positions), axis=(1,2,3)), mask=mask))

    # Test 0 mean subspace restriction.
    samples = flow.sample_apply(params, samples.features[0, :, 0], key3, samples.positions.shape[0:1])
    info.update(mean_abs_x_centre_of_mass=
                maybe_masked_mean(jnp.mean(jnp.abs(jnp.mean(samples.positions[:, :, 0], axis=1)), axis=1), mask=mask))
    info.update(latent_x_mean_abs_centre_of_mass=
                maybe_masked_mean(jnp.mean(jnp.abs(jnp.mean(sample_latent.positions[:, :, 0, :], axis=1)), axis=1),
                                  mask=mask))
    return info


