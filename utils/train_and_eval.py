from typing import Callable, Tuple

import chex
import numpy as np
import jax
import jax.numpy as jnp
from functools import partial
import optax

from flow.base_dist import get_conditional_gaussian_augmented_dist
from flow.test_utils import get_max_diff_log_prob_invariance_test
from utils.numerical import rotate_translate_x_and_a_2d, rotate_translate_x_and_a_3d
from flow.distrax_with_extra import Extra

Params = chex.ArrayTree
X = chex.Array
LogProbWithExtraFn = Callable[[Params, X], Tuple[chex.Array, Extra]]

def general_ml_loss_fn(params, x, log_prob_with_extra_fn: LogProbWithExtraFn, key, use_equivariance_loss, weight):
    if use_equivariance_loss:
        loss, info = ml_and_equivariance_loss_fn(params, x, log_prob_with_extra_fn, key, weight)
    else:
        loss, info = ml_loss_fn(params, x, log_prob_with_extra_fn)
    return loss, info


def ml_loss_fn(params, x, log_prob_with_extra_fn: LogProbWithExtraFn):
    log_prob, extra = log_prob_with_extra_fn(params, x)
    loss = - jnp.mean(log_prob)
    info = {"ml_loss": loss}
    info.update({"layer_info/" + key: value for key, value in extra.aux_info.items()})
    return loss, info

def ml_and_equivariance_loss_fn(params, x, log_prob_with_extra_fn: LogProbWithExtraFn, key, weight, batch_size_frac = 0.1):
    batch_size, n_nodes, dim = x.shape
    dim = dim // 2  # Account for augmentd coords.

    # ML loss.
    log_prob, extra = log_prob_with_extra_fn(params, x)
    ml_loss = - jnp.mean(log_prob)
    info = {"loss_ml": ml_loss}
    info.update({"layer_info/" + key: value for key, value in extra.aux_info.items()})

    # Equivariance loss.
    eq_batch_size = max(1, int(batch_size * batch_size_frac))
    key1, key2, key3 = jax.random.split(key, 3)
    theta = jax.random.uniform(key1, shape=(eq_batch_size,)) * 2 * jnp.pi
    translation = jax.random.normal(key2, shape=(eq_batch_size, dim))
    phi = jax.random.uniform(key3, shape=(eq_batch_size,)) * 2 * jnp.pi

    def group_action(x_and_a):
        if dim == 2:
            x_and_a_rot = jax.vmap(rotate_translate_x_and_a_2d)(x_and_a, theta, translation)
        else:
            x_and_a_rot = jax.vmap(rotate_translate_x_and_a_3d)(x_and_a, theta, phi, translation)
        return x_and_a_rot


    log_prob_g, _ = log_prob_with_extra_fn(params, group_action(x[:eq_batch_size]))
    eq_loss = jnp.mean((log_prob_g - log_prob[:eq_batch_size])**2)
    info.update(loss_equivariance=eq_loss)

    loss = eq_loss*weight + ml_loss
    info.update(loss=loss)
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


def ml_step(params, x, opt_state, log_prob_with_extra_fn: LogProbWithExtraFn,
            optimizer, key, use_equivariance_loss, eq_loss_weight):
    grad, info = jax.grad(general_ml_loss_fn, has_aux=True)(
        params, x, log_prob_with_extra_fn, key, use_equivariance_loss, eq_loss_weight)
    updates, new_opt_state = optimizer.update(grad, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    info.update(grad_norm=optax.global_norm(grad),
                update_norm=optax.global_norm(updates))
    info.update({"grad_" + key: value for key, value in get_tree_leaf_norm_info(grad).items()})
    info.update({"update_" + key: value for key, value in get_tree_leaf_norm_info(updates).items()})
    return new_params, new_opt_state, info


def original_dataset_to_joint_dataset(dataset, key, global_centering, aug_scale):
    augmented_dataset = get_target_augmented_variables(dataset, key, global_centering, aug_scale)
    dataset = jnp.concatenate((dataset, augmented_dataset), axis=-1)
    return dataset

def get_target_augmented_variables(x_original, key, global_centering, aug_scale):
    x_augmented = get_conditional_gaussian_augmented_dist(
        x=x_original, scale=aug_scale,
        global_centering=global_centering).sample(seed=key)
    return x_augmented


def get_augmented_sample_and_log_prob(x_original, global_centering, aug_scale, key, K):
    x_augmented, log_p_a = get_conditional_gaussian_augmented_dist(
        x=x_original, global_centering=global_centering, scale=aug_scale).sample_and_log_prob(seed=key, sample_shape=(K,))
    return x_augmented, log_p_a

def get_augmented_log_prob(x_original, x_augmented, global_centering, aug_scale):
    log_p_a = get_conditional_gaussian_augmented_dist(
        x=x_original, global_centering=global_centering, scale=aug_scale).log_prob(x_augmented)
    return log_p_a


def get_marginal_log_lik_info(log_prob_fn, x_original, key, global_centering, aug_scale, K: int) -> dict:
    x_augmented, log_p_a = get_augmented_sample_and_log_prob(x_original, global_centering, aug_scale,
                                                             key, K)
    x_original = jnp.repeat(x_original[None, ...], K, axis=0)
    log_q = jax.vmap(log_prob_fn)(jnp.concatenate((x_original, x_augmented), axis=-1))
    chex.assert_equal_shape((log_p_a, log_q))
    log_w = log_q - log_p_a

    info = {}
    marginal_log_lik = jnp.mean(jax.nn.logsumexp(log_w, axis=0) - jnp.log(jnp.array(K)), axis=0)
    info.update(marginal_log_lik=marginal_log_lik)

    info.update(var_log_w=jnp.mean(jnp.var(log_w, axis=0), axis=0))
    info.update(ess_marginal=jnp.mean(1 / jnp.sum(jax.nn.softmax(log_w, axis=0) ** 2, axis=0) / log_w.shape[0]))
    return info


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8, 9))
def eval_fn(params, x, key, flow_log_prob_apply_fn, flow_sample_and_log_prob_apply_fn,
            global_centering: bool,
            aug_scale: float,
            target_log_prob = None,
            batch_size=None,
            K: int = 20, test_invariances: bool = True):
    # TODO: Has excessive amount of forward passes currently, cut this down.
    if batch_size is None:
        batch_size = x.shape[0]
    else:
        batch_size = min(batch_size, x.shape[0])
        x = x[:x.shape[0] - x.shape[0] % batch_size]


    dim = x.shape[-1] // 2
    key1, key2 = jax.random.split(key)

    log_prob_samples_only_fn = lambda x: flow_log_prob_apply_fn(params, x)

    def scan_fn(carry, xs):
        x_batch, key = xs
        info = {}
        if test_invariances:
            invariances_info = get_max_diff_log_prob_invariance_test(
            x_batch,  log_prob_fn=log_prob_samples_only_fn, key=key)
            info.update(invariances_info)

        log_prob_batch = flow_log_prob_apply_fn(params, x_batch)
        marginal_log_lik_info = get_marginal_log_lik_info(log_prob_fn=lambda x: flow_log_prob_apply_fn(params, x),
                                                          x_original=x_batch[..., :dim], key=key, K=K,
                                                          global_centering=global_centering,
                                                          aug_scale=aug_scale)
        info.update(eval_log_lik = jnp.mean(log_prob_batch))
        info.update(marginal_log_lik_info)
        return None, info

    x_batched = jnp.reshape(x, (-1, batch_size, *x.shape[1:]))
    _, info = jax.lax.scan(
        scan_fn, None, (x_batched, jax.random.split(key1, x_batched.shape[0])))

    info = jax.tree_map(jnp.mean, info)

    x_flow, log_prob_flow = flow_sample_and_log_prob_apply_fn(params, key2, (batch_size,))
    x_flow_original, x_flow_aug = jnp.split(x_flow, axis=-1, indices_or_sections=2)
    original_centre = jnp.mean(x_flow_original, axis=-2)
    aug_centre = jnp.mean(x_flow_aug, axis=-2)
    info.update(mean_aug_orig_norm=jnp.mean(jnp.linalg.norm(original_centre-aug_centre, axis=-1)))

    if target_log_prob is not None:
        # Calculate ESS
        log_w = target_log_prob(x_flow_original) + get_augmented_log_prob(x_flow_original, x_flow_aug,
                                                                          global_centering, aug_scale) - log_prob_flow
        ess = 1 / jnp.sum(jax.nn.softmax(log_w) ** 2) / log_w.shape[0]
        info.update(
            {"eval_kl": jnp.mean(target_log_prob(x)) - info["eval_log_lik"],
             "ess": ess}
        )
    return info