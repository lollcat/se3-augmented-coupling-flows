from typing import NamedTuple, Callable, Tuple, Union, Any

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp

from flow.distrax_with_extra import Extra, BijectorWithExtra

Params = hk.Params
LogProb = chex.Array
LogDet = chex.Array
Sample = chex.Array



class FlowRecipe(NamedTuple):
    """Defines input needed to create an instance of the `Flow` callables."""
    make_base: Callable[[], distrax.Distribution]
    make_bijector: Callable[[], Union[distrax.Bijector, BijectorWithExtra]]
    n_layers: int
    config: Any

class FlowParams(NamedTuple):
    base: Params
    bijector: Params

class Flow(NamedTuple):
    init: Callable[[chex.PRNGKey, Sample], FlowParams]
    log_prob_apply: Callable[[FlowParams, Sample], LogProb]
    sample_and_log_prob_apply: Callable[[FlowParams, chex.PRNGKey, chex.Shape], Tuple[Sample, LogProb]]
    sample_apply: Callable[[FlowParams, chex.PRNGKey, chex.Shape], Sample]
    log_prob_with_extra_apply: Callable[[FlowParams, Sample], Tuple[LogProb, Extra]]
    sample_and_log_prob_with_extra_apply: Callable[[FlowParams, chex.PRNGKey, chex.Shape], Tuple[Sample, LogProb, Extra]]
    config: Any


def create_flow(recipe: FlowRecipe):
    """Create a `Flow` given the provided definition."""

    @hk.without_apply_rng
    @hk.transform
    def base_sample_fn(seed: chex.PRNGKey, sample_shape: chex.Shape) -> Sample:
        return recipe.make_base().sample(seed=seed, sample_shape=sample_shape)

    @hk.without_apply_rng
    @hk.transform
    def base_log_prob_fn(x: Sample) -> LogProb:
        return recipe.make_base().log_prob(value=x)

    @hk.without_apply_rng
    @hk.transform
    def bijector_forward_and_log_det(x: Sample) -> Tuple[Sample, LogDet]:
        return recipe.make_bijector().forward_and_log_det(x)

    @hk.without_apply_rng
    @hk.transform
    def bijector_inverse_and_log_det(x: Sample) -> Tuple[Sample, LogDet]:
        return recipe.make_bijector().inverse_and_log_det(x)

    @hk.without_apply_rng
    @hk.transform
    def bijector_forward_and_log_det_with_extra(x: Sample) -> Tuple[Sample, LogDet, Extra]:
        bijector = recipe.make_bijector()
        if isinstance(bijector, BijectorWithExtra):
            y, log_det, extra = bijector.forward_and_log_det_with_extra(x)
        else:
            y, log_det = bijector.forward_and_log_det(x)
            extra = Extra()
        return y, log_det, extra

    @hk.without_apply_rng
    @hk.transform
    def bijector_inverse_and_log_det_with_extra(y: Sample) -> Tuple[Sample, LogDet, Extra]:
        bijector = recipe.make_bijector()
        if isinstance(bijector, BijectorWithExtra):
            x, log_det, extra = bijector.inverse_and_log_det_with_extra(y)
        else:
            x, log_det = bijector.inverse_and_log_det(y)
            extra = Extra()
        return x, log_det, extra

    def init(seed: chex.PRNGKey, sample: Sample) -> FlowParams:
        params_base = base_log_prob_fn.init(seed, sample)
        params_bijector_single = bijector_forward_and_log_det.init(seed, sample)
        params_bijectors = jax.tree_map(lambda x: jnp.repeat(x[None, ...], recipe.n_layers, axis=0),
                                        params_bijector_single)
        return FlowParams(base=params_base, bijector=params_bijectors)


    def log_prob_apply(params: FlowParams, y: Sample) -> LogProb:
        def scan_fn(carry, bijector_params):
            y, log_det_prev = carry
            x, log_det = bijector_inverse_and_log_det.apply(bijector_params, y)
            return (x, log_det_prev + log_det), None

        (x, log_det), _ = jax.lax.scan(scan_fn, init=(y, jnp.zeros(y.shape[:-2])), xs=params.bijector, reverse=True)
        base_log_prob = base_log_prob_fn.apply(params.base, x)
        return base_log_prob + log_det

    def sample_and_log_prob_apply(params: FlowParams, seed: chex.PRNGKey, shape: chex.Shape) -> Tuple[Sample, LogProb]:
        def scan_fn(carry, bijector_params):
            x, log_det_prev = carry
            y, log_det = bijector_forward_and_log_det.apply(bijector_params, x)
            return (y, log_det_prev + log_det), None

        x = base_sample_fn.apply(params.base, seed, shape)
        base_log_prob = base_log_prob_fn.apply(params.base, x)
        (y, log_det), _ = jax.lax.scan(scan_fn, init=(x, jnp.zeros(x.shape[:-2])), xs=params.bijector)
        log_prob = base_log_prob - log_det
        return y, log_prob

    def log_prob_with_extra_apply(params: FlowParams, y: Sample) -> Tuple[LogProb, Extra]:
        def scan_fn(carry, bijector_params):
            y, log_det_prev = carry
            x, log_det, extra = bijector_inverse_and_log_det_with_extra.apply(bijector_params, y)
            return (x, log_det_prev + log_det), extra


        (x, log_det), extra = jax.lax.scan(scan_fn, init=(y, jnp.zeros(y.shape[:-2])), xs=params.bijector,
                                           reverse=True)
        base_log_prob = base_log_prob_fn.apply(params.base, x)

        info = {}
        aggregators = {}
        for i in reversed(range(recipe.n_layers)):
          info.update({f"block{i}_" + key: val[i] for key, val in extra.aux_info.items()})
          aggregators.update({f"block{i}_" + key: val for key, val in extra.info_aggregator.items()})
        extra = Extra(aux_loss=extra.aux_loss, aux_info=info, info_aggregator=aggregators)

        return base_log_prob + log_det, extra


    def sample_and_log_prob_with_extra_apply(params: FlowParams, key: chex.PRNGKey,
                                             shape: chex.Shape) -> Tuple[Sample, LogProb, Extra]:
        def scan_fn(carry, bijector_params):
            x, log_det_prev = carry
            y, log_det, extra = bijector_forward_and_log_det_with_extra.apply(bijector_params, x)
            return (y, log_det_prev + log_det), extra

        x = base_sample_fn.apply(params.base, key, shape)
        base_log_prob = base_log_prob_fn.apply(params.base, x)
        (y, log_det), extra = jax.lax.scan(scan_fn, init=(x, jnp.zeros(x.shape[:-2])), xs=params.bijector)
        log_prob = base_log_prob - log_det

        info = {}
        aggregators = {}
        for i in range(recipe.n_layers):
          info.update({f"block{i}_" + key: val[i] for key, val in extra.aux_info.items()})
          aggregators.update({f"block{i}_" + key: val for key, val in extra.info_aggregator.items()})
        extra = Extra(aux_loss=extra.aux_loss, aux_info=info, info_aggregator=aggregators)
        return y, log_prob, extra


    flow = Flow(init=init,
                log_prob_apply=log_prob_apply,
                sample_and_log_prob_apply=sample_and_log_prob_apply,
                log_prob_with_extra_apply=log_prob_with_extra_apply,
                sample_and_log_prob_with_extra_apply=sample_and_log_prob_with_extra_apply,
                sample_apply=lambda params, key, shape: sample_and_log_prob_apply(params, key, shape)[0],
                config=recipe.config
                )
    return flow