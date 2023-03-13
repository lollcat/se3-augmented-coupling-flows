import chex

from flow.fast_flow_dist import AugmentedFlowRecipe, AugmentedFlow, create_flow, FullGraphSample

from typing import NamedTuple, Sequence, Union
import distrax

from flow.base_dist import CentreGravitryGaussianAndCondtionalGuassian
from flow.conditional_dist import build_aux_dist
from flow.bijectors.bijector_proj_real_nvp import make_se_equivariant_split_coupling_with_projection
from flow.bijectors.bijector_nice import make_se_equivariant_nice
from nets.base import NetsConfig
from flow.distrax_with_extra import ChainWithExtra


class ConditionalAuxDistConfig(NamedTuple):
    global_centering: bool = False
    trainable_augmented_scale: bool = True
    aug_scale_init: float = 1.0


class FlowDistConfig(NamedTuple):
    dim: int
    n_aux: int
    nodes: int
    n_layers: int
    nets_config: NetsConfig
    type: Union[str, Sequence[str]]
    identity_init: bool = True
    compile_n_unroll: int = 2
    act_norm: bool = False
    kwargs: dict = {}
    base_aux_config: ConditionalAuxDistConfig = ConditionalAuxDistConfig()
    target_aux_config: ConditionalAuxDistConfig = ConditionalAuxDistConfig()


def build_flow(config: FlowDistConfig) -> AugmentedFlow:
    recipe = create_flow_recipe(config)
    flow = create_flow(recipe)
    return flow


def create_flow_recipe(config: FlowDistConfig) -> AugmentedFlowRecipe:
    if config.type not in ['proj', 'nice'] or config.act_norm:
        raise NotImplementedError("WithInfo flow changes so far only applied to proj flow.")

    flow_type = [config.type] if isinstance(config.type, str) else config.type
    if not "proj" in config.kwargs.keys():
        if not config.kwargs == {}:
            raise NotImplementedError
    kwargs_proj = config.kwargs['proj'] if "proj" in config.kwargs.keys() else {}

    def make_base() -> distrax.Distribution:
        base = CentreGravitryGaussianAndCondtionalGuassian(
            dim=config.dim, n_nodes=config.nodes, global_centering=config.base_aux_config.global_centering,
            trainable_augmented_scale=config.base_aux_config.trainable_augmented_scale,
            augmented_scale_init=config.base_aux_config.aug_scale_init,
            n_aux=config.n_aux
        )
        return base
    def make_bijector(graph_features: chex.Array):
        bijectors = []
        layer_number = 0

        for swap in (True, False):  # For swap False we condition augmented on original.
            if "proj" in flow_type:
                bijector = make_se_equivariant_split_coupling_with_projection(layer_number=layer_number, dim=config.dim, swap=swap,
                                                                              identity_init=config.identity_init,
                                                                              nets_config=config.nets_config,
                                                                              **kwargs_proj)
                bijectors.append(bijector)

            if "nice" in flow_type:
                bijector = make_se_equivariant_nice(
                    layer_number=layer_number,
                    graph_features=graph_features,
                    dim=config.dim,
                    n_aux=config.n_aux,
                    swap=swap,
                    identity_init=config.identity_init,
                    nets_config=config.nets_config)
                bijectors.append(bijector)
        return ChainWithExtra(bijectors)

    make_aug_target = build_aux_dist(
        n_aux = config.n_aux,
        name = 'target',
        global_centering = config.target_aux_config.global_centering,
        augmented_scale_init = config.target_aux_config.aug_scale_init,
        trainable_scale = config.target_aux_config.trainable_augmented_scale)


    definition = AugmentedFlowRecipe(make_base=make_base,
                                     make_bijector=make_bijector,
                                     make_aug_target=make_aug_target,
                                     n_layers=config.n_layers,
                                     config=config,
                                     dim_x=config.dim,
                                     n_augmented=config.n_aux,
                                     compile_n_unroll=config.compile_n_unroll,
                                     )
    return definition