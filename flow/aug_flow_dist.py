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

GraphFeatures = chex.Array  # Non-positional information.
Positions = chex.Array


class FullGraphSample(NamedTuple):
    positions: Positions
    features: GraphFeatures
    def __getitem__(self, i):
        return FullGraphSample(self.positions[i], self.features[i])



class AugmentedFlowRecipe(NamedTuple):
    """Defines input needed to create an instance of the `Flow` callables."""
    make_base: Callable[[], distrax.Distribution]
    make_bijector: Callable[[GraphFeatures], BijectorWithExtra]
    make_aug_target: Callable[[FullGraphSample], distrax.Distribution]
    n_layers: int
    config: Any
    dim_x: int
    n_augmented: int  # number of augmented variables, each of dimension dim_x.
    compile_n_unroll: int = 2


class AugmentedFlowParams(NamedTuple):
    base: Params
    bijector: Params
    aux_target: Params


class AugmentedFlow(NamedTuple):
    init: Callable[[chex.PRNGKey, FullGraphSample], AugmentedFlowParams]
    log_prob_apply: Callable[[AugmentedFlowParams, FullGraphSample], LogProb]
    sample_and_log_prob_apply: Callable[[AugmentedFlowParams, GraphFeatures, chex.PRNGKey, chex.Shape], Tuple[FullGraphSample, LogProb]]
    sample_apply: Callable[[AugmentedFlowParams, GraphFeatures, chex.PRNGKey, chex.Shape], FullGraphSample]
    log_prob_with_extra_apply: Callable[[AugmentedFlowParams, FullGraphSample], Tuple[LogProb, Extra]]
    sample_and_log_prob_with_extra_apply: Callable[[AugmentedFlowParams, GraphFeatures, chex.PRNGKey, chex.Shape], Tuple[FullGraphSample, LogProb, Extra]]
    config: Any
    aux_target_log_prob_apply: Callable[[Params, FullGraphSample], LogProb]
    aux_target_sample_n_and_log_prob_apply: Callable[[Params, FullGraphSample, chex.PRNGKey, int], Tuple[Positions, LogProb]]
    aux_target_sample_n_apply: Callable[[Params, FullGraphSample, chex.PRNGKey, int], Positions]
    separate_samples_to_joint: Callable[[GraphFeatures, Positions, Positions], FullGraphSample]
    joint_to_separate_samples: Callable[[FullGraphSample], Tuple[GraphFeatures, Positions, Positions]]
    dim_x: int
    n_augmented: int  # number of augmented variables, each of dimension dim_x.


def create_flow(recipe: AugmentedFlowRecipe):
    """Create a `Flow` given the provided definition. \
    """

    # TODO: This has been made more general, towards being a flow on the full joint distribution, but currently
    #   still does conditional distribution of graph.
    # For this to be done we would make the passing of features to the sampling step optional, so the full joint
    # may be sampled from.

    def separate_samples_to_full_joint(features: GraphFeatures, positions_x: Positions, positions_a: Positions) -> \
            FullGraphSample:
        features = jnp.expand_dims(features, axis=-2)
        assert len(features.shape) == len(positions_a.shape)
        assert len(positions_x.shape) == (len(positions_a.shape) - 1)
        positions = jnp.concatenate([jnp.expand_dims(positions_x, axis=-2), positions_a], axis=-2)
        return FullGraphSample(positions=positions, features=features)

    def joint_to_separate_samples(joint_sample: FullGraphSample) -> Tuple[Positions, Positions, GraphFeatures]:
        positions_x, positions_a = jnp.split(joint_sample.positions, indices_or_sections=jnp.array([1]),
                                             axis=-2)
        features = jnp.squeeze(joint_sample.features, axis=-2)
        return features, positions_x, positions_a


    @hk.without_apply_rng
    @hk.transform
    def base_sample_fn(graph_features: GraphFeatures, seed: chex.PRNGKey, sample_shape: chex.Shape) -> FullGraphSample:
        # Currently the base does not condition on the graph features.
        # TODO: add this conditioning, could be simple linear layer for scaling.
        #  The base will then take in the graph features.
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
        # TODO: see base_sample_fn.
        return recipe.make_base().log_prob(value=sample.positions)

    @hk.without_apply_rng
    @hk.transform
    def bijector_forward_and_log_det(x: FullGraphSample) -> Tuple[FullGraphSample, LogDet]:
        y, logdet = recipe.make_bijector(x.features).forward_and_log_det(x.positions)
        return FullGraphSample(positions=y, features=x.features), logdet

    @hk.without_apply_rng
    @hk.transform
    def bijector_inverse_and_log_det(y: FullGraphSample) -> Tuple[FullGraphSample, LogDet]:
        x, logdet = recipe.make_bijector(y.features).inverse_and_log_det(y.positions)
        return FullGraphSample(positions=x, features=y.features), logdet

    @hk.without_apply_rng
    @hk.transform
    def bijector_forward_and_log_det_with_extra(x: FullGraphSample) -> \
            Tuple[FullGraphSample, LogDet, Extra]:
        bijector = recipe.make_bijector(x.features)
        if isinstance(bijector, BijectorWithExtra):
            y, log_det, extra = bijector.forward_and_log_det_with_extra(x.positions)
        else:
            y, log_det = bijector.forward_and_log_det(x.positions)
            extra = Extra()
        return FullGraphSample(positions=y, features=x.features), log_det, extra


    @hk.without_apply_rng
    @hk.transform
    def bijector_inverse_and_log_det_with_extra(y: FullGraphSample) -> \
            Tuple[FullGraphSample, LogDet, Extra]:
        bijector = recipe.make_bijector(y.features)
        if isinstance(bijector, BijectorWithExtra):
            x, log_det, extra = bijector.inverse_and_log_det_with_extra(y.positions)
        else:
            x, log_det = bijector.inverse_and_log_det(y.positions)
            extra = Extra()
        return FullGraphSample(positions=x, features=y.features), log_det, extra


    def log_prob_apply(params: AugmentedFlowParams, sample: FullGraphSample) -> LogProb:
        def scan_fn(carry, bijector_params):
            y, log_det_prev = carry
            x, log_det = bijector_inverse_and_log_det.apply(bijector_params, y)
            return (x, log_det_prev + log_det), None

        log_prob_shape = sample.positions.shape[:-3]
        (x, log_det), _ = jax.lax.scan(scan_fn, init=(sample, jnp.zeros(log_prob_shape)),
                                       xs=params.bijector, reverse=True,
                                       unroll=recipe.compile_n_unroll)
        base_log_prob = base_log_prob_fn.apply(params.base, x)
        return base_log_prob + log_det

    def log_prob_with_extra_apply(params: AugmentedFlowParams, sample: FullGraphSample) -> Tuple[LogProb, Extra]:
        def scan_fn(carry, bijector_params):
            y, log_det_prev = carry
            x, log_det, extra = bijector_inverse_and_log_det_with_extra.apply(bijector_params, y)
            return (x, log_det_prev + log_det), extra

        log_prob_shape = sample.positions.shape[:-3]
        (x, log_det), extra = jax.lax.scan(scan_fn, init=(sample, jnp.zeros(log_prob_shape)),
                                           xs=params.bijector,
                                           reverse=True, unroll=recipe.compile_n_unroll)
        base_log_prob = base_log_prob_fn.apply(params.base, x)

        info = {}
        aggregators = {}
        for i in reversed(range(recipe.n_layers)):
          info.update({f"block{i}_" + key: val[i] for key, val in extra.aux_info.items()})
          aggregators.update({f"block{i}_" + key: val for key, val in extra.info_aggregator.items()})
        extra = Extra(aux_loss=extra.aux_loss, aux_info=info, info_aggregator=aggregators)

        return base_log_prob + log_det, extra

    def sample_and_log_prob_apply(params: AugmentedFlowParams, features: GraphFeatures,
                                  key: chex.PRNGKey, shape: chex.Shape) -> Tuple[Positions, LogProb]:
        def scan_fn(carry, bijector_params):
            x, log_det_prev = carry
            y, log_det = bijector_forward_and_log_det.apply(bijector_params, x)
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
                                             shape: chex.Shape) -> Tuple[Positions, LogProb, Extra]:
        def scan_fn(carry, bijector_params):
            x, log_det_prev = carry
            y, log_det, extra = bijector_forward_and_log_det_with_extra.apply(bijector_params, x)
            return (y, log_det_prev + log_det), extra

        x = base_sample_fn.apply(params.base, features, key, shape)
        base_log_prob = base_log_prob_fn.apply(params.base, x)
        (y, log_det), extra = jax.lax.scan(scan_fn, init=(x, jnp.zeros(x.positions.shape[:-3])), xs=params.bijector,
                                           unroll=recipe.compile_n_unroll)
        chex.assert_equal_shape((base_log_prob, log_det))
        log_prob = base_log_prob - log_det

        info = {}
        aggregators = {}
        for i in range(recipe.n_layers):
          info.update({f"block{i}_" + key: val[i] for key, val in extra.aux_info.items()})
          aggregators.update({f"block{i}_" + key: val for key, val in extra.info_aggregator.items()})
        extra = Extra(aux_loss=extra.aux_loss, aux_info=info, info_aggregator=aggregators)
        return y, log_prob, extra

    @hk.without_apply_rng
    @hk.transform
    def aux_target_log_prob(joint_sample: FullGraphSample) -> Tuple[Positions, LogProb]:
        features, positions_x, positions_a = joint_to_separate_samples(joint_sample)
        dist = recipe.make_aug_target(FullGraphSample(positions=positions_x, features=features))
        positions_a, log_prob = dist.log_prob(positions_a)
        return positions_a, log_prob


    @hk.without_apply_rng
    @hk.transform
    def aux_target_sample_n_and_log_prob(sample_x: FullGraphSample, key: chex.PRNGKey, n: int) -> \
            Tuple[Positions, LogProb]:
        dist = recipe.make_aug_target(sample_x)
        positions_a, log_prob = dist._sample_n_and_log_prob(key, n)
        return positions_a, log_prob

    @hk.without_apply_rng
    @hk.transform
    def aux_target_sample_n(sample_x: FullGraphSample, key: chex.PRNGKey, n: int) -> \
            Positions:
        dist = recipe.make_aug_target(sample_x)
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
        params_bijector_single = bijector_inverse_and_log_det.init(key4, sample_joint)
        params_bijectors = jax.tree_map(lambda x: jnp.repeat(x[None, ...], recipe.n_layers, axis=0),
                                        params_bijector_single)
        return AugmentedFlowParams(base=params_base, bijector=params_bijectors, aux_target=params_aux_target)


    flow = AugmentedFlow(
        dim_x=recipe.dim_x,
        n_augmented=recipe.n_augmented,
        init=init,
        log_prob_apply=log_prob_apply,
        sample_and_log_prob_apply=sample_and_log_prob_apply,
        log_prob_with_extra_apply=log_prob_with_extra_apply,
        sample_and_log_prob_with_extra_apply=sample_and_log_prob_with_extra_apply,
        sample_apply=lambda params, key, shape: sample_and_log_prob_apply(params, key, shape)[0],
        config=recipe.config,
        aux_target_log_prob_apply=aux_target_log_prob.apply,
        aux_target_sample_n_apply=aux_target_sample_n.apply,
        aux_target_sample_n_and_log_prob_apply=aux_target_sample_n_and_log_prob.apply,
        separate_samples_to_joint=separate_samples_to_full_joint,
        joint_to_separate_samples=joint_to_separate_samples
                        )
    return flow
