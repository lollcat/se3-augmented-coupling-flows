"""Training with FAB. Note assumes fixed conditioning info."""

from typing import Callable, NamedTuple, Tuple

import chex
import jax.numpy as jnp
import jax.random
import optax

from fabjax.sampling.smc import SequentialMonteCarloSampler, SMCState
from fabjax.utils.graph import setup_flat_log_prob
from fabjax.buffer import PrioritisedBuffer, PrioritisedBufferState
from fabjax.utils.optimize import IgnoreNanOptState

from flow.aug_flow_dist import AugmentedFlow, AugmentedFlowParams, GraphFeatures, FullGraphSample

Params = chex.ArrayTree
LogProbFn = Callable[[chex.Array], chex.Array]
ParameterizedLogProbFn = Callable[[chex.ArrayTree, chex.Array], chex.Array]
Info = dict

def fab_loss_buffer_samples(
        params: AugmentedFlowParams,
        x: chex.Array,
        log_q_old: chex.Array,
        alpha: chex.Array,
        log_q_fn_apply: ParameterizedLogProbFn,
        w_adjust_clip: float,
) -> Tuple[chex.Array, Tuple[chex.Array, chex.Array]]:
    """Estimate FAB loss with a batch of samples from the prioritized replay buffer."""
    chex.assert_rank(x, 2)
    chex.assert_rank(log_q_old, 1)

    log_q = log_q_fn_apply(params, x)
    log_w_adjust = (1 - alpha) * (jax.lax.stop_gradient(log_q) - log_q_old)
    chex.assert_equal_shape((log_q, log_w_adjust))
    w_adjust = jnp.clip(jnp.exp(log_w_adjust), a_max=w_adjust_clip)
    return - jnp.mean(w_adjust * log_q), (log_w_adjust, log_q)



class TrainStateWithBuffer(NamedTuple):
    flow_params: AugmentedFlowParams
    key: chex.PRNGKey
    opt_state: optax.OptState
    smc_state: SMCState
    buffer_state: PrioritisedBufferState


def build_smc_forward_pass(
        flow: AugmentedFlow,
        log_p_x: LogProbFn,
        features: GraphFeatures,
        smc: SequentialMonteCarloSampler,
        batch_size: int):

    features_with_multiplicity = features[:, None]
    n_nodes = features.shape[0]

    def flow_log_prob_apply(params, x):
        return flow.log_prob_apply(params, FullGraphSample(positions=x, features=features_with_multiplicity))

    event_shape = (n_nodes, 1 + flow.n_augmented, flow.dim_x)

    def smc_forward_pass(state: TrainStateWithBuffer, key: chex.PRNGKey):
        def log_q_fn(x: chex.Array) -> chex.Array:
            return flow_log_prob_apply(state.params, x)

        flatten, unflatten, log_q_flat_fn = setup_flat_log_prob(log_q_fn, event_shape)

        def log_p_flat_fn(x: chex.Array):
            x = unflatten(x)
            x, a = jnp.split(x, [1, ], axis=-2)
            x = jnp.squeeze(x, axis=-2)
            log_prob_x = log_p_x(x)
            log_prob_augmented = flow.aux_target_log_prob_apply(
                state.params.aux_target, FullGraphSample(features=features_with_multiplicity, positions=x), a)
            return log_prob_x + log_prob_augmented

        sample_flow = flow.sample_apply(state.params, features, key, (batch_size,))
        x0 = flatten(sample_flow.positions)
        point, log_w, smc_state, smc_info = smc.step(x0, state.smc_state, log_q_flat_fn, log_p_flat_fn)
        x_smc = unflatten(point.x)
        return sample_flow.positions, x_smc, log_w, smc_state, smc_info

    return smc_forward_pass




def build_fab_with_buffer_init_step_fns(
        flow: AugmentedFlow,
        log_p_x: LogProbFn,
        features: GraphFeatures,
        smc: SequentialMonteCarloSampler,
        buffer: PrioritisedBuffer,
        optimizer: optax.GradientTransformation,
        batch_size: int,
        n_updates_per_smc_forward_pass: int,
        alpha: float = 2.,
        w_adjust_clip: float = 10.,
):
    """Create the `init` and `step` functions that define the FAB algorithm."""
    assert smc.alpha == alpha

    n_nodes = features.shape[0]
    # Setup smc forward pass.
    smc_forward = build_smc_forward_pass(flow, log_p_x, features, smc, batch_size)
    features_with_multiplicity = features[:, None]

    def flow_log_prob_apply(params, x):
        return flow.log_prob_apply(params, FullGraphSample(positions=x, features=features_with_multiplicity))


    def init(key: chex.PRNGKey) -> TrainStateWithBuffer:
        """Initialise the flow, optimizer and smc states."""
        key1, key2, key3 = jax.random.split(key, 3)
        dummy_sample = FullGraphSample(positions=jnp.zeros((n_nodes, flow.dim_x)), features=features)
        flow_params = flow.init(key1, dummy_sample)
        opt_state = optimizer.init(flow_params)
        smc_state = smc.init(key2)


        return TrainStateWithBuffer(params=flow_params, key=key3, opt_state=opt_state, smc_state=smc_state)

    @jax.jit
    @chex.assert_max_traces(4)
    def step(state: TrainStateNoBuffer) -> Tuple[TrainStateNoBuffer, Info]:
        key, subkey = jax.random.split(state.key)
        info = {}

        # Run smc.
        sample_flow, x_smc, log_w, smc_state, smc_info = smc_forward(state, subkey)
        info.update(smc_info)

        # Estimate loss and update flow params.
        loss, grad = jax.value_and_grad(fab_loss_smc_samples)(state.params, x_smc, log_w, flow_log_prob_apply)
        updates, new_opt_state = optimizer.update(grad, state.opt_state, params=state.params)
        new_params = optax.apply_updates(state.params, updates)
        info.update(loss=loss)

        new_state = TrainStateNoBuffer(params=new_params, key=key, opt_state=new_opt_state, smc_state=smc_state)
        return new_state, info

    return init, step
