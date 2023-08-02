"""Training with FAB. Note assumes fixed conditioning info."""

from typing import Callable, NamedTuple, Tuple

import chex
import jax.numpy as jnp
import jax.random
import optax

from fabjax.sampling.smc import SequentialMonteCarloSampler, SMCState
from fabjax.utils.graph import setup_flat_log_prob

from eacf.flow.aug_flow_dist import AugmentedFlow, AugmentedFlowParams, GraphFeatures, FullGraphSample

Params = chex.ArrayTree
LogProbFn = Callable[[chex.Array], chex.Array]
ParameterizedLogProbFn = Callable[[chex.ArrayTree, chex.Array], chex.Array]
Info = dict

def fab_loss_smc_samples(params, x: chex.Array, log_w: chex.Array, log_q_fn_apply: ParameterizedLogProbFn):
    """Estimate FAB loss with a batch of samples from smc."""
    chex.assert_rank(log_w, 1)
    chex.assert_rank(x, 4)  # [batch_size, n_nodes, multiplicity, dim]

    log_q = log_q_fn_apply(params, x)
    chex.assert_equal_shape((log_q, log_w))
    return - jnp.mean(jax.nn.softmax(log_w) * log_q)


class TrainStateNoBuffer(NamedTuple):
    params: AugmentedFlowParams
    key: chex.PRNGKey
    opt_state: optax.OptState
    smc_state: SMCState


def get_joint_log_prob_target(params: AugmentedFlowParams, flow: AugmentedFlow, log_p_x: LogProbFn,
                              features_with_multiplicity: GraphFeatures) -> LogProbFn:
    def joint_log_prob(x: chex.Array):
        x, a = jnp.split(x, [1, ], axis=-2)
        x = jnp.squeeze(x, axis=-2)
        log_prob_x = log_p_x(x)
        log_prob_augmented = flow.aux_target_log_prob_apply(
            params.aux_target, FullGraphSample(features=features_with_multiplicity, positions=x), a)
        return log_prob_x + log_prob_augmented

    return joint_log_prob



def flat_log_prob_components(log_p_x: LogProbFn, flow: AugmentedFlow, params: AugmentedFlowParams,
                             features_with_multiplicity: chex.Array, event_shape: chex.Shape):
    def flow_log_prob_apply(params, x):
        return flow.log_prob_apply(params, FullGraphSample(positions=x, features=features_with_multiplicity))

    def flow_log_prob_apply_with_extra(params, x):
        return flow.log_prob_with_extra_apply(params, FullGraphSample(positions=x, features=features_with_multiplicity))

    def log_q_fn(x: chex.Array) -> chex.Array:
        return flow_log_prob_apply(params, x)

    flatten, unflatten, log_q_flat_fn = setup_flat_log_prob(log_q_fn, event_shape)

    joint_target_log_prob_fn = get_joint_log_prob_target(params, flow, log_p_x, features_with_multiplicity)

    def log_p_flat_fn(x: chex.Array):
        x = unflatten(x)
        return joint_target_log_prob_fn(x)

    return flatten, unflatten, log_p_flat_fn, log_q_flat_fn, flow_log_prob_apply, flow_log_prob_apply_with_extra


def build_smc_forward_pass(
        flow: AugmentedFlow,
        log_p_x: LogProbFn,
        features: GraphFeatures,
        smc: SequentialMonteCarloSampler,
        batch_size: int):

    features_with_multiplicity = features[:, None]
    n_nodes = features.shape[0]
    event_shape = (n_nodes, 1 + flow.n_augmented, flow.dim_x)

    def smc_forward_pass(params: AugmentedFlowParams, smc_state: SMCState, key: chex.PRNGKey,
                         unflatten_output: bool = True):
        flatten, unflatten, log_p_flat_fn, log_q_flat_fn, flow_log_prob_apply, flow_log_prob_apply_with_extra\
            = flat_log_prob_components(
            log_p_x=log_p_x, flow=flow, params=params, features_with_multiplicity=features_with_multiplicity,
            event_shape=event_shape
        )

        sample_flow = flow.sample_apply(params, features, key, (batch_size,))
        x0 = flatten(sample_flow.positions)
        point, log_w, smc_state, smc_info = smc.step(x0, smc_state, log_q_flat_fn, log_p_flat_fn)
        x_smc = point.x
        if unflatten_output:
            x_smc = unflatten(x_smc)
        return sample_flow.positions, x_smc, log_w, point.log_q, smc_state, smc_info

    return smc_forward_pass
                                                             



def build_fab_no_buffer_init_step_fns(
        flow: AugmentedFlow,
        log_p_x: LogProbFn,
        features: GraphFeatures,
        smc: SequentialMonteCarloSampler,
        optimizer: optax.GradientTransformation,
        batch_size: int):

    n_nodes = features.shape[0]
    # Setup smc forward pass.
    smc_forward = build_smc_forward_pass(flow, log_p_x, features, smc, batch_size)
    features_with_multiplicity = features[:, None]

    event_shape = (n_nodes, 1 + flow.n_augmented, flow.dim_x)

    def init(key: chex.PRNGKey) -> TrainStateNoBuffer:
        """Initialise the flow, optimizer and smc states."""
        key1, key2, key3 = jax.random.split(key, 3)
        dummy_sample = FullGraphSample(positions=jnp.zeros((n_nodes, flow.dim_x)), features=features)
        flow_params = flow.init(key1, dummy_sample)
        opt_state = optimizer.init(flow_params)
        smc_state = smc.init(key2)
        return TrainStateNoBuffer(params=flow_params, key=key3, opt_state=opt_state, smc_state=smc_state)

    @jax.jit
    @chex.assert_max_traces(4)
    def step(state: TrainStateNoBuffer) -> Tuple[TrainStateNoBuffer, Info]:
        flatten, unflatten, log_p_flat_fn, log_q_flat_fn, flow_log_prob_apply, flow_log_prob_apply_with_extra = \
            flat_log_prob_components(
            log_p_x=log_p_x, flow=flow, params=state.params, features_with_multiplicity=features_with_multiplicity,
            event_shape=event_shape
        )

        key, subkey = jax.random.split(state.key)
        info = {}

        # Run smc.
        sample_flow, x_smc, log_w, log_q, smc_state, smc_info = smc_forward(state.params, state.smc_state, subkey)
        info.update(smc_info)

        # Estimate loss and update flow params.
        loss, grad = jax.value_and_grad(fab_loss_smc_samples)(state.params, x_smc, log_w, flow_log_prob_apply)
        updates, new_opt_state = optimizer.update(grad, state.opt_state, params=state.params)
        new_params = optax.apply_updates(state.params, updates)
        info.update(loss=loss)

        new_state = TrainStateNoBuffer(params=new_params, key=key, opt_state=new_opt_state, smc_state=smc_state)
        return new_state, info

    return init, step
