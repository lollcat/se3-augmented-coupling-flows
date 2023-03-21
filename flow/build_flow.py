import chex

from flow.aug_flow_dist import AugmentedFlowRecipe, AugmentedFlow, create_flow, FullGraphSample

from typing import NamedTuple, Sequence, Union
import distrax

from flow.base_dist import CentreGravitryGaussianAndCondtionalGuassian
from flow.conditional_dist import build_aux_dist
from flow.bijectors.proj_real_nvp import make_se_equivariant_split_coupling_with_projection
from flow.bijectors.nice import make_se_equivariant_nice
from flow.bijectors.shrink_aug import make_shrink_aug_layer
from flow.bijectors.permute_aug import AugPermuteBijector
from nets.base import NetsConfig
from flow.distrax_with_extra import ChainWithExtra


class ConditionalAuxDistConfig(NamedTuple):
    global_centering: bool = False
    trainable_augmented_scale: bool = True
    scale_init: float = 1.0

class BaseConfig(NamedTuple):
    train_x_scale: bool = True
    x_scale_init: float = 1.0
    aug: ConditionalAuxDistConfig = ConditionalAuxDistConfig()


class FlowDistConfig(NamedTuple):
    dim: int
    n_aug: int
    nodes: int
    n_layers: int
    nets_config: NetsConfig
    type: Union[str, Sequence[str]]
    identity_init: bool = True
    compile_n_unroll: int = 2
    act_norm: bool = False
    kwargs: dict = {}
    base: BaseConfig = BaseConfig()
    target_aux_config: ConditionalAuxDistConfig = ConditionalAuxDistConfig()


def build_flow(config: FlowDistConfig) -> AugmentedFlow:
    recipe = create_flow_recipe(config)
    flow = create_flow(recipe)
    return flow


def create_flow_recipe(config: FlowDistConfig) -> AugmentedFlowRecipe:
    flow_type = [config.type] if isinstance(config.type, str) else config.type
    if 'nice' not in flow_type and 'proj' not in flow_type:
        raise Exception
    if not "proj" in config.kwargs.keys():
        if not config.kwargs == {}:
            raise NotImplementedError
    kwargs_proj = config.kwargs['proj'] if "proj" in config.kwargs.keys() else {}

    def make_base() -> distrax.Distribution:
        base = CentreGravitryGaussianAndCondtionalGuassian(
            dim=config.dim,
            n_nodes=config.nodes,
            global_centering=config.base.aug.global_centering,
            trainable_x_scale=config.base.train_x_scale,
            x_scale_init=config.base.x_scale_init,
            trainable_augmented_scale=config.base.aug.trainable_augmented_scale,
            augmented_scale_init=config.base.aug.scale_init,
            n_aux=config.n_aug
        )
        return base

    def make_bijector(graph_features: chex.Array):
        # Note that bijector.inverse moves through this forwards, and bijector.fowards reverses the bijector order
        bijectors = []
        layer_number = 0

        if config.act_norm:
            bijector = make_shrink_aug_layer(
                layer_number=layer_number,
                graph_features=graph_features,
                dim=config.dim,
                n_aug=config.n_aug,
                swap=False,
                identity_init=config.identity_init)
            bijectors.append(bijector)

        if config.n_aug > 1:
            bijectors.append(AugPermuteBijector())


        for swap in (False, True):  # For swap False we condition augmented on original.
            if "proj" in flow_type:
                bijector = make_se_equivariant_split_coupling_with_projection(
                    graph_features=graph_features, n_aug=config.n_aug,
                    layer_number=layer_number, dim=config.dim,
                    swap=swap,
                    identity_init=config.identity_init,
                    nets_config=config.nets_config)
                bijectors.append(bijector)

            if "nice" in flow_type:
                bijector = make_se_equivariant_nice(
                    layer_number=layer_number,
                    graph_features=graph_features,
                    dim=config.dim,
                    n_aug=config.n_aug,
                    swap=swap,
                    identity_init=config.identity_init,
                    nets_config=config.nets_config)
                bijectors.append(bijector)

        return ChainWithExtra(bijectors)

    make_aug_target = build_aux_dist(
        n_aug=config.n_aug,
        name='target',
        global_centering=config.target_aux_config.global_centering,
        augmented_scale_init=config.target_aux_config.scale_init,
        trainable_scale=config.target_aux_config.trainable_augmented_scale)


    definition = AugmentedFlowRecipe(make_base=make_base,
                                     make_bijector=make_bijector,
                                     make_aug_target=make_aug_target,
                                     n_layers=config.n_layers,
                                     config=config,
                                     dim_x=config.dim,
                                     n_augmented=config.n_aug,
                                     compile_n_unroll=config.compile_n_unroll,
                                     )
    return definition