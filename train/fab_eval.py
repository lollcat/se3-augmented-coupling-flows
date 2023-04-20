from typing import Union, Tuple

import chex
import jax
import jax.numpy as jnp
from fabjax.sampling.smc import SequentialMonteCarloSampler
from fabjax.sampling.resampling import log_effective_sample_size

from flow.aug_flow_dist import AugmentedFlow, GraphFeatures
from train.fab_train_no_buffer import TrainStateNoBuffer, flat_log_prob_components, get_joint_log_prob_target
from train.fab_train_with_buffer import TrainStateWithBuffer


def fab_eval_function(state: Union[TrainStateNoBuffer, TrainStateWithBuffer],
                      key: chex.PRNGKey,
                      flow: AugmentedFlow,
                      smc: SequentialMonteCarloSampler,
                      log_p_x,
                      features: GraphFeatures,
                      batch_size: int,
                      inner_batch_size: int) -> dict:
    """Evaluate the ESS of the flow, and AIS. """
    assert smc.alpha == 1.  # Make sure target is set to p.
    assert smc.use_resampling is False  # Make sure we are doing AIS, not SMC.

    # Setup scan function.
    features_with_multiplicity = features[:, None]
    n_nodes = features.shape[0]
    event_shape = (n_nodes, 1 + flow.n_augmented, flow.dim_x)
    features_with_multiplicity = features[:, None]
    flatten, unflatten, log_p_flat_fn, log_q_flat_fn, flow_log_prob_apply = flat_log_prob_components(
        log_p_x=log_p_x, flow=flow, params=state.params, features_with_multiplicity=features_with_multiplicity,
        event_shape=event_shape
    )
    joint_target_log_prob_fn = get_joint_log_prob_target(params=state.params, flow=flow, log_p_x=log_p_x,
                                                  features_with_multiplicity=features_with_multiplicity)

    def inner_fn(carry: None, xs: chex.PRNGKey) -> Tuple[None, Tuple[chex.Array, chex.Array]]:
        """Perform SMC forward pass and grab just the importance weights."""
        key = xs
        sample_flow, log_q_flow = flow.sample_and_log_prob_apply(state.params, features, key, (inner_batch_size,))
        x0 = flatten(sample_flow.positions)
        point, log_w, smc_state, smc_info = smc.step(x0, state.smc_state, log_q_flat_fn, log_p_flat_fn)
        log_w_flow = joint_target_log_prob_fn(sample_flow.positions) - log_q_flow
        return None, (log_w_flow, log_w)

    # Run scan function.
    n_batches = (batch_size // inner_batch_size) + 1
    _, (log_w_flow, log_w_ais) = jax.lax.scan(inner_fn, init=None, xs=jax.random.split(key, n_batches))

    # Compute metrics
    info = {}
    info.update(eval_ess_flow=jnp.exp(log_effective_sample_size(log_w_flow.flatten())),
                eval_ess_ais=jnp.exp(log_effective_sample_size(log_w_ais.flatten())))
    return info
