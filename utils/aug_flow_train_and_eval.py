from typing import Callable, Tuple, Optional

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
    aux_samples, log_pi_a_given_x = flow.aux_target_sample_n_and_log_prob_apply(params.aux_target, x, key)
    joint_samples = flow.separate_samples_to_joint(x.features, x.positions, aux_samples)
    log_q, extra = flow.log_prob_with_extra_apply(params, joint_samples)
    mean_log_prob_q = jnp.mean(log_q)
    mean_log_p_a = jnp.mean(log_pi_a_given_x)
    # Train on lower bound of the marginal log likelihood. (Allows for log_pi_a_given_x to be parameterized)
    loss = mean_log_p_a - mean_log_prob_q
    info = {"mean_log_prob_q_joint": mean_log_prob_q,
            "mean_log_p_a": mean_log_p_a
            }
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


def get_eval_on_test_batch(flow: AugmentedFlow,
                           params: AugmentedFlowParams,
                           x_test: FullGraphSample,
                           key: chex.PRNGKey,
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
        invariances_info = get_max_diff_log_prob_invariance_test(joint_sample[0], flow=flow, params=params, key=subkey)
        info.update(invariances_info)
    return info


@partial(jax.jit, static_argnums=(3, 4, 5, 6, 7))
def eval_fn(params: AugmentedFlowParams,
            x: FullGraphSample,
            key: chex.PRNGKey,
            flow: AugmentedFlow,
            target_log_prob: Optional[Callable] = None,
            batch_size: Optional[int] = None,
            K: int = 20,
            test_invariances: bool = True):

    if batch_size is None:
        batch_size = x.positions.shape[0]
    else:
        batch_size = min(batch_size, x.positions.shape[0])
        x = x[:x.positions.shape[0] - x.positions.shape[0] % batch_size]

    key1, key2 = jax.random.split(key)

    def scan_fn(carry, xs):
        # Scan over data in the test set. Vmapping all at once causes memory issues I think?
        x_batch, key = xs
        info = {}
        test_eval_info = get_eval_on_test_batch(
            flow, params, x_test=x_batch, key=key, K=K, test_invariances=test_invariances)
        info.update(test_eval_info)
        return None, info

    x_batched = jax.tree_map(lambda x: jnp.reshape(x, (-1, batch_size, *x.shape[1:])), x)
    _, info = jax.lax.scan(
        scan_fn, None, (x_batched, jax.random.split(key1, x_batched.positions.shape[0])))
    # Aggregate test set info.
    info = jax.tree_map(jnp.mean, info)


    joint_x_flow, log_prob_flow = flow.sample_and_log_prob_apply(params, x.features[0], key2, (batch_size,))
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
            {"eval_kl": jnp.mean(target_log_prob(x)) - info["eval_log_lik"],
             "ess": ess}
        )
    return info
