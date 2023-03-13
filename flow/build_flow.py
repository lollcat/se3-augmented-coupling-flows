from flow.fast_flow_dist import AugmentedFlowRecipe, AugmentedFlow, create_flow

from typing import NamedTuple, Sequence, Union
import distrax

from flow.bijectors.bijector_proj_real_nvp import make_se_equivariant_split_coupling_with_projection
from flow.bijectors.bijector_proj_real_nvp_v2 import make_se_equivariant_split_coupling_with_projection as proj_v2
from flow.bijectors.bijector_nice import make_se_equivariant_nice
from flow.bijectors.bijector_pseudo_act_norm import make_pseudo_act_norm_bijector
from flow.bijectors.bijector_scale_along_vector import make_se_equivariant_scale_along_vector
from flow.bijectors.bijector_real_nvp_non_equivariant import make_realnvp
from nets.base import NetsConfig
from flow.distrax_with_extra import ChainWithExtra


class ConditionalAuxDistConfig(NamedTuple):
    global_centering: bool = False
    trainable_augmented_scale: bool = True
    aug_scale_init: float = 1.0


class FlowDistConfig(NamedTuple):
    dim: int
    nodes: int
    n_layers: int
    nets_config: NetsConfig
    type: Union[str, Sequence[str]]
    identity_init: bool = True
    compile_n_unroll: int = 2
    act_norm: bool = False
    kwargs: dict = {}
    base_aux_config: ConditionalAuxDistConfig = ConditionalAuxDistConfig()
    target_aux_config: ConditionalAuxDistConfig = C


def build_flow(config: FlowDistConfig) -> AugmentedFlow:
    recipe = create_flow_recipe(config)
    flow = create_flow(recipe)
    return flow


def create_flow_recipe(config: FlowDistConfig) -> AugmentedFlowRecipe:
    if config.type not in ['proj', 'nice'] or config.act_norm:
        raise NotImplementedError("WithInfo flow changes so far only applied to proj flow.")

    flow_type = [config.type] if isinstance(config.type, str) else config.type
    if not ("proj" in config.kwargs.keys() or "proj_v2" in config.kwargs.keys()):
        if not config.kwargs == {}:
            raise NotImplementedError
    kwargs_proj = config.kwargs['proj'] if "proj" in config.kwargs.keys() else {}
    kwargs_proj_v2 = config.kwargs['proj_v2'] if "proj_v2" in config.kwargs.keys() else {}

    def make_base() -> distrax.Distribution:
        if config.base_config.double_centered_gaussian:
            base = DoubleCentreGravitryGaussian(dim=config.dim, n_nodes=config.nodes)
        else:
            base = CentreGravitryGaussianAndCondtionalGuassian(
                dim=config.dim, n_nodes=config.nodes, global_centering=config.base_config.global_centering,
                trainable_augmented_scale=config.base_config.trainable_augmented_scale,
                augmented_scale_init=config.base_config.aug_scale_init
            )
        return base
    def make_bijector():
        bijectors = []
        layer_number = 0
        if config.act_norm:
            bijectors.append(make_pseudo_act_norm_bijector(
                layer_number=layer_number, dim=config.dim, flow_identity_init=config.identity_init))

        for swap in (True, False):  # For swap False we condition augmented on original.
            if 'realnvp_non_eq' in flow_type:
                assert len(flow_type) == 1
                bijector = make_realnvp(layer_number=layer_number, dim=config.dim, swap=swap,
                                        nets_config=config.nets_config,
                                        identity_init=config.identity_init)
                bijectors.append(bijector)
            if "vector_scale" in flow_type:
                bijector = make_se_equivariant_scale_along_vector(layer_number=layer_number, dim=config.dim, swap=swap,
                                                                  identity_init=config.identity_init,
                                                                  nets_conifg=config.nets_config)
                bijectors.append(bijector)
            if "proj" in flow_type:
                bijector = make_se_equivariant_split_coupling_with_projection(layer_number=layer_number, dim=config.dim, swap=swap,
                                                                              identity_init=config.identity_init,
                                                                              nets_config=config.nets_config,
                                                                              **kwargs_proj)
                bijectors.append(bijector)

            if "nice" in flow_type:
                bijector = make_se_equivariant_nice(layer_number=layer_number, dim=config.dim, swap=swap,
                                                    identity_init=config.identity_init,
                                                    nets_config=config.nets_config)
                bijectors.append(bijector)
            if "proj_v2" in flow_type:
                bijector = proj_v2(layer_number=layer_number, dim=config.dim, swap=swap,
                                                                  identity_init=config.identity_init,
                                                                  nets_config=config.nets_config,
                                                                  **kwargs_proj_v2)
                bijectors.append(bijector)

        return ChainWithExtra(bijectors)

    definition = AugmentedFlowRecipe(make_base=make_base, make_bijector=make_bijector,
                                     n_layers=config.n_layers, config=config, compile_n_unroll=config.compile_n_unroll)
    return definition