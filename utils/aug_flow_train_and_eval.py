from typing import Callable, Tuple

import chex
import jax
import jax.numpy as jnp
from functools import partial
import optax

from flow.test_utils import get_max_diff_log_prob_invariance_test
from flow.distrax_with_extra import Extra
from flow.aug_flow_dist import AugmentedFlow, FullGraphSample, AugmentedFlowParams

Params = chex.ArrayTree
X = chex.Array
LogProbWithExtraFn = Callable[[Params, X], Tuple[chex.Array, Extra]]


def general_ml_loss_fn(params: AugmentedFlowParams,
                       x: FullGraphSample,
                       flow: AugmentedFlow,
                       key: chex.PRNGKey,
                       use_aux_loss: bool,
                       aux_loss_weight: float) -> Tuple[chex.Array, dict]:
    aux_samples = jnp.squeeze(flow.aux_target_sample_n_apply(params.aux_target, key, x, 1), axis=0)
    joint_samples = flow.separate_samples_to_joint(x.features, x.positions, aux_samples)
    log_prob, extra = flow.log_prob_with_extra_apply(params, joint_samples)
    loss = - jnp.mean(log_prob)
    info = {"ml_loss": loss}
    aux_loss = jnp.mean(extra.aux_loss)
    if use_aux_loss:
        loss = loss + aux_loss * aux_loss_weight
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


def ml_step(params: AugmentedFlowParams, x: FullGraphSample,
            opt_state: optax.OptState,
            flow: AugmentedFlow,
            optimizer: optax.GradientTransformation,
            key: chex.PRNGKey,
            use_aux_loss: bool,
            aux_loss_weight: float) -> Tuple[AugmentedFlowParams, optax.OptState, dict]:
    grad, info = jax.grad(general_ml_loss_fn, has_aux=True)(
        params, x, flow, key, use_aux_loss, aux_loss_weight)
    updates, new_opt_state = optimizer.update(grad, opt_state, params=params)
    new_params = optax.apply_updates(params, updates)
    info.update(grad_norm=optax.global_norm(grad),
                update_norm=optax.global_norm(updates))
    info.update({"grad_" + key: value for key, value in get_tree_leaf_norm_info(grad).items()})
    info.update({"update_" + key: value for key, value in get_tree_leaf_norm_info(updates).items()})
    return new_params, new_opt_state, info


def get_marginal_log_lik_info(flow: AugmentedFlow,
                              params: AugmentedFlowParams,
                              x: FullGraphSample,
                              key: chex.PRNGKey,
                              K: int) -> dict:
    x_augmented, log_p_a = flow.aux_target_sample_n_and_log_prob_apply(params.aux_target, x, key, K)
    x = jax.tree_map(lambda x: jnp.repeat(x[None, ...], K, axis=0), x)
    joint_sample = flow.separate_samples_to_joint(x.features, x.positions, x_augmented)

    log_q = jax.vmap(flow.log_prob_apply, in_axes=(None, 0))(params, joint_sample)
    chex.assert_equal_shape((log_p_a, log_q))
    log_w = log_q - log_p_a

    info = {}
    info.update(eval_log_lik = jnp.mean(log_q))
    marginal_log_lik = jnp.mean(jax.nn.logsumexp(log_w, axis=0) - jnp.log(jnp.array(K)), axis=0)
    info.update(marginal_log_lik=marginal_log_lik)

    info.update(var_log_w=jnp.mean(jnp.var(log_w, axis=0), axis=0))
    info.update(ess_marginal=jnp.mean(1 / jnp.sum(jax.nn.softmax(log_w, axis=0) ** 2, axis=0) / log_w.shape[0]))
    return info


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7, 8))
def eval_fn(params: AugmentedFlowParams,
            x: FullGraphSample,
            key: chex.PRNGKey,
            flow: AugmentedFlow,
            target_log_prob = None,
            batch_size=None,
            K: int = 20,
            test_invariances: bool = True):
    # TODO: Has excessive amount of forward passes currently, cut this down.
    if batch_size is None:
        batch_size = x.positions.shape[0]
    else:
        batch_size = min(batch_size, x.positions.shape[0])
        x = x[:x.positions.shape[0] - x.positions.shape[0] % batch_size]

    key1, key2 = jax.random.split(key)

    log_prob_samples_only_fn = lambda x: flow.log_prob_apply(params, x)

    def scan_fn(carry, xs):
        x_batch, key = xs
        info = {}
        if test_invariances:
            invariances_info = get_max_diff_log_prob_invariance_test(
            x_batch,  log_prob_fn=log_prob_samples_only_fn, key=key)
            info.update(invariances_info)

        marginal_log_lik_info = get_marginal_log_lik_info(flow,
                                                          params,
                                                          x=x_batch,
                                                          key=key, K=K
                                                          )
        info.update(marginal_log_lik_info)
        return None, info

    x_batched = jnp.reshape(x, (-1, batch_size, *x.shape[1:]))
    _, info = jax.lax.scan(
        scan_fn, None, (x_batched, jax.random.split(key1, x_batched.shape[0])))

    info = jax.tree_map(jnp.mean, info)


    joint_x_flow, log_prob_flow = flow.sample_and_log_prob_apply(params, x.features[0], key2, (batch_size,))
    x_flow_original, x_flow_aug = jnp.split(joint_x_flow.positions, axis=-2, indices_or_sections=jnp.array([1, ]))
    original_centre = jnp.mean(x_flow_original, axis=-2)
    aug_centre = jnp.mean(x_flow_aug[:, :, 0, :], axis=-2)
    info.update(mean_aug_orig_norm=jnp.mean(jnp.linalg.norm(original_centre-aug_centre, axis=-1)))

    if target_log_prob is not None:

        # Calculate ESS
        log_w = target_log_prob(x_flow_original) + flow.aux_target_log_prob_apply(params.aux_target, joint_x_flow) - \
                log_prob_flow
        ess = 1 / jnp.sum(jax.nn.softmax(log_w) ** 2) / log_w.shape[0]
        info.update(
            {"eval_kl": jnp.mean(target_log_prob(x)) - info["eval_log_lik"],
             "ess": ess}
        )
    return info
