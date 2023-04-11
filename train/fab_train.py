"""Training with FAB. Note assumes fixed conditioning info."""

from typing import Callable, NamedTuple, Tuple

import chex
import jax.numpy as jnp
import jax.random
import optax

from fabjax.sampling.ais import AnnealedImportanceSampler, AISState
from fabjax.utils.graph import setup_flat_log_prob

from flow.aug_flow_dist import AugmentedFlow, AugmentedFlowParams, GraphFeatures, FullGraphSample

Params = chex.ArrayTree
LogProbFn = Callable[[chex.Array], chex.Array]
ParameterizedLogProbFn = Callable[[chex.ArrayTree, chex.Array], chex.Array]
Info = dict

def fab_loss_ais_samples(params, x: chex.Array, log_w: chex.Array, log_q_fn_apply: ParameterizedLogProbFn):
    """Estimate FAB loss with a batch of samples from AIS."""
    chex.assert_rank(log_w, 1)
    chex.assert_rank(x, 4)  # [batch_size, n_nodes, multiplicity, dim]

    log_q = log_q_fn_apply(params, x)
    chex.assert_equal_shape((log_q, log_w))
    return - jnp.mean(jax.nn.softmax(log_w) * log_q)


class TrainStateNoBuffer(NamedTuple):
    params: AugmentedFlowParams
    key: chex.PRNGKey
    opt_state: optax.OptState
    ais_state: AISState


def build_ais_forward_pass(
        flow: AugmentedFlow,
        log_p_x: LogProbFn,
        features: GraphFeatures,
        ais: AnnealedImportanceSampler,
        batch_size: int):

    features_with_multiplicity = features[:, None]
    n_nodes = features.shape[0]

    def flow_log_prob_apply(params, x):
        return flow.log_prob_apply(params, FullGraphSample(positions=x, features=features_with_multiplicity))

    event_shape = (n_nodes, 1 + flow.n_augmented, flow.dim_x)

    def ais_forward_pass(state: TrainStateNoBuffer, key: chex.PRNGKey):
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
        point, log_w, ais_state, ais_info = ais.step(x0, state.ais_state, log_q_flat_fn, log_p_flat_fn)
        x_ais = unflatten(point.x)
        return sample_flow.positions, x_ais, log_w, ais_state, ais_info

    return ais_forward_pass




def build_fab_no_buffer_init_step_fns(
        flow: AugmentedFlow,
        log_p_x: LogProbFn,
        features: GraphFeatures,
        ais: AnnealedImportanceSampler,
        optimizer: optax.GradientTransformation,
        batch_size: int):

    n_nodes = features.shape[0]
    # Setup AIS forward pass.
    ais_forward = build_ais_forward_pass(flow, log_p_x, features, ais, batch_size)
    features_with_multiplicity = features[:, None]
    def flow_log_prob_apply(params, x):
        return flow.log_prob_apply(params, FullGraphSample(positions=x, features=features_with_multiplicity))


    def init(key: chex.PRNGKey) -> TrainStateNoBuffer:
        """Initialise the flow, optimizer and AIS states."""
        key1, key2, key3 = jax.random.split(key, 3)
        dummy_sample = FullGraphSample(positions=jnp.zeros((n_nodes, flow.dim_x)), features=features)
        flow_params = flow.init(key1, dummy_sample)
        opt_state = optimizer.init(flow_params)
        ais_state = ais.init(key2)
        return TrainStateNoBuffer(params=flow_params, key=key3, opt_state=opt_state, ais_state=ais_state)

    @jax.jit
    @chex.assert_max_traces(4)
    def step(state: TrainStateNoBuffer) -> Tuple[TrainStateNoBuffer, Info]:
        key, subkey = jax.random.split(state.key)
        info = {}

        # Run AIS.
        sample_flow, x_ais, log_w, ais_state, ais_info = ais_forward(state, subkey)
        info.update(ais_info)

        # Estimate loss and update flow params.
        loss, grad = jax.value_and_grad(fab_loss_ais_samples)(state.params, x_ais, log_w, flow_log_prob_apply)
        updates, new_opt_state = optimizer.update(grad, state.opt_state, params=state.params)
        new_params = optax.apply_updates(state.params, updates)
        info.update(loss=loss)

        new_state = TrainStateNoBuffer(params=new_params, key=key, opt_state=new_opt_state, ais_state=ais_state)
        return new_state, info

    return init, step
