from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp

from molboil.utils.test import random_rotate_translate_permute

from utils.testing import get_checks_for_flow_properties
from flow.distrax_with_extra import Extra
from flow.aug_flow_dist import AugmentedFlow, FullGraphSample, AugmentedFlowParams

Params = chex.ArrayTree
X = chex.Array
LogProbWithExtraFn = Callable[[Params, X], Tuple[chex.Array, Extra]]


def general_ml_loss_fn(
        key: chex.PRNGKey,
        params: AugmentedFlowParams,
        x: FullGraphSample,
        verbose_info: bool,
        flow: AugmentedFlow,
        use_flow_aux_loss: bool,
        aux_loss_weight: float,
        apply_random_rotation: bool = False
) -> Tuple[chex.Array, dict]:
    if apply_random_rotation:
        key, subkey = jax.random.split(key)
        rotated_positions = jax.vmap(random_rotate_translate_permute)(
            x.positions, jax.random.split(subkey, x.positions.shape[0]))
        x = x._replace(positions=rotated_positions)
    aux_samples = flow.aux_target_sample_n_apply(params.aux_target, x, key)
    joint_samples = flow.separate_samples_to_joint(x.features, x.positions, aux_samples)
    log_q, extra = flow.log_prob_with_extra_apply(params, joint_samples)
    mean_log_prob_q = jnp.mean(log_q)
    # Train by maximum likelihood.
    loss = - mean_log_prob_q
    info = {"mean_log_prob_q_joint": mean_log_prob_q,
            }
    aux_loss = jnp.mean(extra.aux_loss)
    info.update(flow.get_base_and_target_info(params))
    if use_flow_aux_loss:
        loss = loss + aux_loss * aux_loss_weight
    if verbose_info:
        info.update({"layer_info/" + key: value for key, value in extra.aux_info.items()})
    info.update(aux_loss=aux_loss)
    return loss, info


def get_tree_leaf_norm_info(tree):
    norms = jax.tree_util.tree_map(jnp.linalg.norm, tree)
    norms = jnp.stack(jax.tree_util.tree_flatten(norms)[0])
    max_norm = jnp.max(norms)
    min_norm = jnp.min(norms)
    mean_norm = jnp.mean(norms)
    median_norm = jnp.median(norms)
    info = {}
    info.update(per_layer_max_norm=max_norm, per_layer_min_norm=min_norm,
                per_layer_mean_norm=mean_norm, per_layer_median_norm=median_norm)
    return info


def get_eval_on_test_batch(params: AugmentedFlowParams,
                           x_test: FullGraphSample,
                           key: chex.PRNGKey,
                           flow: AugmentedFlow,
                           K: int,
                           test_invariances: bool = True) -> dict:
    key, subkey = jax.random.split(key)
    x_augmented, log_p_a = flow.aux_target_sample_n_and_log_prob_apply(params.aux_target, x_test, subkey, K)
    x_test = jax.tree_map(lambda x: jnp.repeat(x[None, ...], K, axis=0), x_test)
    joint_sample = flow.separate_samples_to_joint(x_test.features, x_test.positions, x_augmented)

    log_q = jax.vmap(flow.log_prob_apply, in_axes=(None, 0))(params, joint_sample)
    chex.assert_equal_shape((log_p_a, log_q))
    log_w = log_q - log_p_a

    info = {}
    info.update(eval_log_lik = jnp.mean(log_q))
    marginal_log_lik = jnp.mean(jax.nn.logsumexp(log_w, axis=0) - jnp.log(jnp.array(K)), axis=0)
    info.update(marginal_log_lik=marginal_log_lik)

    info.update(var_log_w=jnp.mean(jnp.var(log_w, axis=0), axis=0))
    info.update(ess_marginal=jnp.mean(1 / jnp.sum(jax.nn.softmax(log_w, axis=0) ** 2, axis=0) / log_w.shape[0]))

    if test_invariances:
        key, subkey = jax.random.split(key)
        invariances_info = get_checks_for_flow_properties(joint_sample[0], flow=flow, params=params, key=subkey)
        info.update(invariances_info)
    return info


def eval_non_batched(params: AugmentedFlowParams, features: chex.Array, key: chex.PRNGKey, flow: AugmentedFlow,
                     batch_size: int, target_log_prob: Callable = None):
    info = {}
    joint_x_flow, log_prob_flow = flow.sample_and_log_prob_apply(params, features, key, (batch_size,))
    features, x_positions, a_positions = flow.joint_to_separate_samples(joint_x_flow)
    original_centre = jnp.mean(x_positions, axis=-2)
    aug_centre = jnp.mean(a_positions[:, :, 0, :], axis=-2)
    info.update(mean_aug_orig_norm=jnp.mean(jnp.linalg.norm(original_centre-aug_centre, axis=-1)))

    if target_log_prob is not None:
        # Calculate ESS
        log_w = target_log_prob(x_positions) + \
                flow.aux_target_log_prob_apply(params.aux_target,
                                       FullGraphSample(features=features, positions=x_positions),
                                       a_positions
                                       ) - log_prob_flow
        ess = 1 / jnp.sum(jax.nn.softmax(log_w) ** 2) / log_w.shape[0]
        info.update(
            {
             "ess": ess}
        )
        # TODO: Add KL to eval with batch info.
        # "eval_kl": jnp.mean(target_log_prob(x)) - info["eval_log_lik"],
    return info
