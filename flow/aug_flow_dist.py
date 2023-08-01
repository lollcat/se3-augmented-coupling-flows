from typing import NamedTuple, Callable, Tuple, Any, Optional

import chex
import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np

from flow.distrax_with_extra import Extra, BijectorWithExtra
from molboil.base import FullGraphSample

Params = hk.Params
LogProb = chex.Array
LogDet = chex.Array

GraphFeatures = chex.Array  # Non-positional information such as atom type.
Positions = chex.Array



class AugmentedFlowRecipe(NamedTuple):
    """Defines input needed to create an instance of the `AugmentedFlow` callables."""
    make_base: Callable[[], distrax.Distribution]
    make_bijector: Callable[[GraphFeatures], BijectorWithExtra]
    make_aug_target: Callable[[FullGraphSample], distrax.Distribution]
    n_layers: int
    config: Any
    dim_x: int
    n_augmented: int  # number of augmented variables, each of dimension dim_x.
    compile_n_unroll: int = 1


class AugmentedFlowParams(NamedTuple):
    """Container for the parameters of the augmented flow."""
    base: Params
    bijector: Params
    aux_target: Params


def separate_samples_to_full_joint(features: GraphFeatures, positions_x: Positions, positions_a: Positions) -> \
        FullGraphSample:
    """Put features, x and a positions into the `FullGraphSample` container with some checks."""
    features = jnp.expand_dims(features, axis=-2)
    assert len(features.shape) == len(positions_a.shape)
    assert len(positions_x.shape) == (len(positions_a.shape) - 1)
    positions = jnp.concatenate([jnp.expand_dims(positions_x, axis=-2), positions_a], axis=-2)
    return FullGraphSample(positions=positions, features=features)

def joint_to_separate_samples(joint_sample: FullGraphSample) -> Tuple[Positions, Positions, GraphFeatures]:
    """Pull out features, x and a positions from a joint `FullGraphSample`."""
    positions_x, positions_a = jnp.split(joint_sample.positions, indices_or_sections=[1],
                                         axis=-2)
    positions_x = jnp.squeeze(positions_x, axis=-2)
    features = jnp.squeeze(joint_sample.features, axis=-2)
    return features, positions_x, positions_a

def get_base_and_target_info(params: AugmentedFlowParams) -> dict:
    """Get info for logging that depends on the base and target params."""
    info = {}
    if params.base:
        if "x_log_scale" in params.base["~"]:
            info.update(base_x_scale=jnp.exp(params.base["~"]["x_log_scale"]))
        if 'augmented_log_scale' in params.base["~"]:
            scale = jnp.exp(params.base["~"]['augmented_log_scale'])
            for i in range(scale.shape[0]):
                info.update({f"base_augmented_scale{i}": scale[i]})
    if params.aux_target:
        target_scale = jnp.exp(params.aux_target['~']['aug_target_dist_augmented_scale_logit'])
        for i in range(target_scale.shape[0]):
            info.update({f"target_augmented_scale{i}": target_scale[i]})
    return info


class AugmentedFlow(NamedTuple):
    """Container that defines an augmented flow, mostly of pure jax functions."""
    init: Callable[[chex.PRNGKey, FullGraphSample], AugmentedFlowParams]
    log_prob_apply: Callable[[AugmentedFlowParams, FullGraphSample], LogProb]
    sample_and_log_prob_apply: Callable[[AugmentedFlowParams, GraphFeatures, chex.PRNGKey, chex.Shape], Tuple[FullGraphSample, LogProb]]
    sample_apply: Callable[[AugmentedFlowParams, GraphFeatures, chex.PRNGKey, chex.Shape], FullGraphSample]
    log_prob_with_extra_apply: Callable[[AugmentedFlowParams, FullGraphSample], Tuple[LogProb, Extra]]
    sample_and_log_prob_with_extra_apply: Callable[[AugmentedFlowParams, GraphFeatures, chex.PRNGKey, chex.Shape], Tuple[FullGraphSample, LogProb, Extra]]
    config: Any
    aux_target_log_prob_apply: Callable[[Params, FullGraphSample, Positions], LogProb]
    aux_target_sample_n_and_log_prob_apply: Callable[[Params, FullGraphSample, chex.PRNGKey, Optional[int]], Tuple[Positions, LogProb]]
    aux_target_sample_n_apply: Callable[[Params, FullGraphSample, chex.PRNGKey, Optional[int]], Positions]
    dim_x: int
    n_augmented: int  # number of augmented variables, each of dimension dim_x.

    # Also add access to the bijector forward and inverse, and base sample and log prob. These are useful for debugging.
    bijector_forward_and_log_det_with_extra_apply: Callable[[Params, FullGraphSample],
                    Tuple[FullGraphSample, LogDet, Extra]]
    bijector_inverse_and_log_det_with_extra_apply: Callable[[Params, FullGraphSample],
                    Tuple[FullGraphSample, LogDet, Extra]]
    base_sample: Callable[[Params, GraphFeatures, chex.PRNGKey, chex.Shape], FullGraphSample]
    base_log_prob: Callable[[Params, FullGraphSample], LogProb]

    separate_samples_to_joint: Callable[[GraphFeatures, Positions, Positions], FullGraphSample] = separate_samples_to_full_joint
    joint_to_separate_samples: Callable[[FullGraphSample], Tuple[GraphFeatures, Positions, Positions]] = joint_to_separate_samples
    get_base_and_target_info: Callable[[AugmentedFlowParams], dict] = get_base_and_target_info


def create_flow(recipe: AugmentedFlowRecipe) -> AugmentedFlow:
    """Create an `AugmentedFlow` given the provided definition.

    Make use of `jax.lax.scan` over flow blocks to keep compile time from being too long.
    """

    @hk.without_apply_rng
    @hk.transform
    def base_sample_fn(graph_features: GraphFeatures, seed: chex.PRNGKey, sample_shape: chex.Shape) -> FullGraphSample:
        positions = recipe.make_base().sample(seed=seed, sample_shape=sample_shape)
        graph_features = jnp.expand_dims(graph_features, axis=-2)  # add multiplicity
        assert len(sample_shape) in (0, 1)
        if len(sample_shape) == 1:
            # Broadcast graph features to batch.
            graph_features = jax.tree_map(lambda x: jnp.repeat(x[None, ...], sample_shape[0], axis=0),
                                          graph_features)
        return FullGraphSample(positions=positions, features=graph_features)

    @hk.without_apply_rng
    @hk.transform
    def base_log_prob_fn(sample: FullGraphSample) -> LogProb:
        return recipe.make_base().log_prob(value=sample.positions)

    @hk.without_apply_rng
    @hk.transform
    def bijector_forward_and_log_det_single(x: FullGraphSample) -> Tuple[FullGraphSample, LogDet]:
        y, logdet = recipe.make_bijector(x.features).forward_and_log_det(x.positions)
        return FullGraphSample(positions=y, features=x.features), logdet

    @hk.without_apply_rng
    @hk.transform
    def bijector_inverse_and_log_det_single(y: FullGraphSample) -> Tuple[FullGraphSample, LogDet]:
        x, logdet = recipe.make_bijector(y.features).inverse_and_log_det(y.positions)
        return FullGraphSample(positions=x, features=y.features), logdet

    @hk.without_apply_rng
    @hk.transform
    def bijector_forward_and_log_det_with_extra_single(x: FullGraphSample) -> \
            Tuple[FullGraphSample, LogDet, Extra]:
        bijector = recipe.make_bijector(x.features)
        if isinstance(bijector, BijectorWithExtra):
            y, log_det, extra = bijector.forward_and_log_det_with_extra(x.positions)
        else:
            y, log_det = bijector.forward_and_log_det(x.positions)
            extra = Extra()
        extra.aux_info.update(mean_log_det=jnp.mean(-log_det))
        extra.info_aggregator.update(mean_log_det=jnp.mean)
        return FullGraphSample(positions=y, features=x.features), log_det, extra


    @hk.without_apply_rng
    @hk.transform
    def bijector_inverse_and_log_det_with_extra_single(y: FullGraphSample) -> \
            Tuple[FullGraphSample, LogDet, Extra]:
        bijector = recipe.make_bijector(y.features)
        if isinstance(bijector, BijectorWithExtra):
            x, log_det, extra = bijector.inverse_and_log_det_with_extra(y.positions)
        else:
            x, log_det = bijector.inverse_and_log_det(y.positions)
            extra = Extra()
        extra.aux_info.update(mean_log_det=jnp.mean(log_det))
        extra.info_aggregator.update(mean_log_det=jnp.mean)
        return FullGraphSample(positions=x, features=y.features), log_det, extra


    def bijector_forward_and_log_det_with_extra_apply(
            params: Params,
            sample: FullGraphSample,
            layer_indices: Optional[Tuple[int, int]] = None  # [start, stop]
    ) -> Tuple[FullGraphSample, LogDet, Extra]:
        def scan_fn(carry, bijector_params):
            x, log_det_prev = carry
            y, log_det, extra = bijector_forward_and_log_det_with_extra_single.apply(bijector_params, x)
            chex.assert_equal_shape((log_det_prev, log_det))
            return (y, log_det_prev + log_det), extra

        if layer_indices is not None:
            params = jax.tree_map(lambda x: x[layer_indices[0]:layer_indices[1]], params)
        (y, log_det), extra = jax.lax.scan(scan_fn, init=(sample, jnp.zeros(sample.positions.shape[:-3])),
                                           xs=params,
                                           unroll=recipe.compile_n_unroll)
        info = {}
        aggregators = {}
        for i in range(recipe.n_layers):
          info.update({f"block{i}_" + key: val[i] for key, val in extra.aux_info.items()})
          aggregators.update({f"block{i}_" + key: val for key, val in extra.info_aggregator.items()})
        extra = Extra(aux_loss=extra.aux_loss, aux_info=info, info_aggregator=aggregators)
        return y, log_det, extra

    def bijector_inverse_and_log_det_with_extra_apply(
            params: Params,
            sample: FullGraphSample,
            layer_indices: Optional[Tuple[int, int]] = None,  # [start, stop]
    ) -> Tuple[FullGraphSample, LogDet, Extra]:

        def scan_fn(carry, bijector_params):
            y, log_det_prev = carry
            x, log_det, extra = bijector_inverse_and_log_det_with_extra_single.apply(bijector_params, y)
            chex.assert_equal_shape((log_det_prev, log_det))
            return (x, log_det_prev + log_det), extra

        # Restrict to zero-CoM subspace before passing through bijector.
        x = sample.positions[..., 0, :]
        centre_of_mass_x = jnp.mean(x, axis=-2, keepdims=True)
        sample = sample._replace(positions=sample.positions - jnp.expand_dims(centre_of_mass_x, axis=-2))
        log_prob_shape = sample.positions.shape[:-3]

        if layer_indices is not None:
            params = jax.tree_map(lambda x: x[layer_indices[0]:layer_indices[1]], params)
        (x, log_det), extra = jax.lax.scan(scan_fn, init=(sample, jnp.zeros(log_prob_shape)),
                                           xs=params,
                                           reverse=True, unroll=recipe.compile_n_unroll)

        info = {}
        aggregators = {}
        for i in reversed(range(recipe.n_layers)):
          info.update({f"block{i}_" + key: val[i] for key, val in extra.aux_info.items()})
          aggregators.update({f"block{i}_" + key: val for key, val in extra.info_aggregator.items()})
        extra = Extra(aux_loss=extra.aux_loss, aux_info=info, info_aggregator=aggregators)
        return x, log_det, extra


    def log_prob_apply(params: AugmentedFlowParams, sample: FullGraphSample) -> LogProb:
        def scan_fn(carry, bijector_params):
            y, log_det_prev = carry
            x, log_det = bijector_inverse_and_log_det_single.apply(bijector_params, y)
            chex.assert_equal_shape((log_det_prev, log_det))
            return (x, log_det_prev + log_det), None

        # Restrict to zero CoM subspace before passing through bijector.
        x = sample.positions[..., 0, :]   # Regular coordinates only (as we centre on this).
        centre_of_mass_x = jnp.mean(x, axis=-2, keepdims=True)
        sample = sample._replace(positions=sample.positions - jnp.expand_dims(centre_of_mass_x, axis=-2))

        log_prob_shape = sample.positions.shape[:-3]
        (x, log_det), _ = jax.lax.scan(scan_fn, init=(sample, jnp.zeros(log_prob_shape)),
                                       xs=params.bijector, reverse=True,
                                       unroll=recipe.compile_n_unroll)
        base_log_prob = base_log_prob_fn.apply(params.base, x)
        chex.assert_equal_shape((base_log_prob, log_det))
        return base_log_prob + log_det

    def log_prob_with_extra_apply(params: AugmentedFlowParams, sample: FullGraphSample) -> Tuple[LogProb, Extra]:
        x, log_det, extra = bijector_inverse_and_log_det_with_extra_apply(
            params.bijector, sample)
        base_log_prob = base_log_prob_fn.apply(params.base, x)
        chex.assert_equal_shape((base_log_prob, log_det))

        extra.aux_info.update(mean_base_log_prob=jnp.mean(base_log_prob))
        extra.info_aggregator.update(mean_base_log_prob=jnp.mean)

        return base_log_prob + log_det, extra

    def sample_and_log_prob_apply(params: AugmentedFlowParams, features: GraphFeatures,
                                  key: chex.PRNGKey, shape: chex.Shape) -> Tuple[FullGraphSample, LogProb]:
        def scan_fn(carry, bijector_params):
            x, log_det_prev = carry
            y, log_det = bijector_forward_and_log_det_single.apply(bijector_params, x)
            chex.assert_equal_shape((log_det_prev, log_det))
            return (y, log_det_prev + log_det), None

        x = base_sample_fn.apply(params.base, features, key, shape)
        base_log_prob = base_log_prob_fn.apply(params.base, x)
        (y, log_det), _ = jax.lax.scan(scan_fn, init=(x, jnp.zeros(x.positions.shape[:-3])), xs=params.bijector,
                                       unroll=recipe.compile_n_unroll)
        chex.assert_equal_shape((base_log_prob, log_det))
        log_prob = base_log_prob - log_det
        return y, log_prob


    def sample_and_log_prob_with_extra_apply(params: AugmentedFlowParams,
                                             features: GraphFeatures,
                                             key: chex.PRNGKey,
                                             shape: chex.Shape) -> Tuple[FullGraphSample, LogProb, Extra]:

        x = base_sample_fn.apply(params.base, features, key, shape)
        base_log_prob = base_log_prob_fn.apply(params.base, x)
        y, log_det, extra = bijector_forward_and_log_det_with_extra_apply(params.bijector, x)
        log_prob = base_log_prob - log_det

        extra.aux_info.update(mean_base_log_prob=jnp.mean(base_log_prob))
        extra.info_aggregator.update(mean_base_log_prob=jnp.mean)

        return y, log_prob, extra

    @hk.without_apply_rng
    @hk.transform
    def aux_target_log_prob(joint_sample: FullGraphSample) -> LogProb:
        features, positions_x, positions_a = joint_to_separate_samples(joint_sample)
        dist = recipe.make_aug_target(FullGraphSample(positions=positions_x, features=features))
        positions_a, log_prob = dist.log_prob(positions_a)
        return log_prob


    @hk.without_apply_rng
    @hk.transform
    def aux_target_sample_n_and_log_prob(sample_x: FullGraphSample, key: chex.PRNGKey, n: Optional[int] = None) -> \
            Tuple[Positions, LogProb]:
        dist = recipe.make_aug_target(sample_x)
        if n is None:
            positions_a, log_prob = dist._sample_n_and_log_prob(key, 1)
            positions_a, log_prob = jnp.squeeze(positions_a, axis=0), jnp.squeeze(log_prob, axis=0)
        else:
            positions_a, log_prob = dist._sample_n_and_log_prob(key, n)
        return positions_a, log_prob

    @hk.without_apply_rng
    @hk.transform
    def aux_target_sample_n(sample_x: FullGraphSample, key: chex.PRNGKey, n: Optional[int] = None) -> \
            Positions:
        dist = recipe.make_aug_target(sample_x)
        if n is None:
            sample = jnp.squeeze(dist._sample_n(key, 1), axis=0)
        else:
            sample = dist._sample_n(key, n)
        return sample

    @hk.without_apply_rng
    @hk.transform
    def aux_target_log_prob(sample_x: FullGraphSample, postions_a: Positions) -> LogProb:
        dist = recipe.make_aug_target(sample_x)
        chex.assert_tree_shape_suffix(postions_a, dist.event_shape)
        log_prob = dist.log_prob(postions_a)
        return log_prob


    def init(seed: chex.PRNGKey, sample: FullGraphSample) -> AugmentedFlowParams:
        key1, key2, key3, key4 = jax.random.split(seed, 4)
        params_aux_target = aux_target_log_prob.init(key1, sample,
                                                     jnp.repeat(jnp.expand_dims(sample.positions, axis=-2),
                                                        recipe.n_augmented, axis=-2))
        sample_a = aux_target_sample_n.apply(params_aux_target, sample, key2, n=1)
        sample_a = jnp.squeeze(sample_a, axis=0)

        # Check shapes.
        chex.assert_tree_shape_suffix(sample.positions, (recipe.dim_x, ))
        assert sample_a.shape[-2] == recipe.n_augmented

        sample_joint = separate_samples_to_full_joint(sample.features,
                                                      sample.positions, sample_a)
        params_base = base_log_prob_fn.init(key3, sample_joint)
        params_bijector_single = bijector_inverse_and_log_det_single.init(key4, sample_joint)
        params_bijectors = jax.tree_map(lambda x: jnp.repeat(x[None, ...], recipe.n_layers, axis=0),
                                        params_bijector_single)
        return AugmentedFlowParams(base=params_base, bijector=params_bijectors, aux_target=params_aux_target)

    def sample_apply(*args, **kwargs):
        return sample_and_log_prob_apply(*args, **kwargs)[0]


    flow = AugmentedFlow(
        dim_x=recipe.dim_x,
        n_augmented=recipe.n_augmented,
        init=init,
        log_prob_apply=log_prob_apply,
        sample_and_log_prob_apply=sample_and_log_prob_apply,
        log_prob_with_extra_apply=log_prob_with_extra_apply,
        sample_and_log_prob_with_extra_apply=sample_and_log_prob_with_extra_apply,
        bijector_forward_and_log_det_with_extra_apply=bijector_forward_and_log_det_with_extra_apply,
        bijector_inverse_and_log_det_with_extra_apply=bijector_inverse_and_log_det_with_extra_apply,
        base_sample=base_sample_fn.apply,
        base_log_prob=base_log_prob_fn.apply,
        sample_apply=sample_apply,
        config=recipe.config,
        aux_target_log_prob_apply=aux_target_log_prob.apply,
        aux_target_sample_n_apply=aux_target_sample_n.apply,
        aux_target_sample_n_and_log_prob_apply=aux_target_sample_n_and_log_prob.apply,
                        )
    return flow
