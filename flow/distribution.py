from typing import NamedTuple, Optional, Sequence, Union
import distrax

from flow.base_dist import DoubleCentreGravitryGaussian, CentreGravitryGaussianAndCondtionalGuassian
from flow.bijectors.bijector_proj_real_nvp import make_se_equivariant_split_coupling_with_projection
from flow.bijectors.bijector_proj_real_nvp_v2 import make_se_equivariant_split_coupling_with_projection as proj_v2
from flow.bijectors.bijector_nice import make_se_equivariant_nice
from flow.bijectors.bijector_pseudo_act_norm import make_pseudo_act_norm_bijector
from flow.bijectors.bijector_scale_along_vector import make_se_equivariant_scale_along_vector
from flow.bijectors.bijector_real_nvp_non_equivariant import make_realnvp
from nets.base import NetsConfig
from flow.fast_hk_chain import Chain as FastChain
from flow.distrax_with_extra import TransformedWithExtra, ChainWithExtra


class BaseConfig(NamedTuple):
    double_centered_gaussian: bool = False
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
    fast_compile: bool = True
    compile_n_unroll: int = 1
    act_norm: bool = True
    kwargs: dict = {}
    base_config: BaseConfig = BaseConfig()



def make_equivariant_augmented_flow_dist(config: FlowDistConfig):
    if config.fast_compile:
        return make_equivariant_augmented_flow_dist_fast_compile(config)
    else:
        return make_equivariant_augmented_flow_dist_distrax_chain(config)


def make_equivariant_augmented_flow_dist_fast_compile(config: FlowDistConfig):
    if config.type != 'proj':
        raise NotImplementedError("WithInfo flow changes so far only applied to proj flow.")


    flow_type = [config.type] if isinstance(config.type, str) else config.type
    if not ("proj" in config.kwargs.keys() or "proj_v2" in config.kwargs.keys()):
        if not config.kwargs == {}:
            raise NotImplementedError
    kwargs_proj = config.kwargs['proj'] if "proj" in config.kwargs.keys() else {}
    kwargs_proj_v2 = config.kwargs['proj_v2'] if "proj_v2" in config.kwargs.keys() else {}

    if config.base_config.double_centered_gaussian:
        base = DoubleCentreGravitryGaussian(dim=config.dim, n_nodes=config.nodes)
    else:
        base = CentreGravitryGaussianAndCondtionalGuassian(
            dim=config.dim, n_nodes=config.nodes, global_centering=config.base_config.global_centering,
            trainable_augmented_scale=config.base_config.trainable_augmented_scale,
            augmented_scale_init=config.base_config.aug_scale_init
        )

    def bijector_fn():
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

    flow = FastChain(bijector_fn=bijector_fn, n_layers=config.n_layers, compile_n_unroll=config.compile_n_unroll)
    if config.act_norm:
        final_act_norm = make_pseudo_act_norm_bijector(layer_number=-1, dim=config.dim,
                                                       flow_identity_init=config.identity_init)
        flow = distrax.Chain([flow, final_act_norm])
    distribution = TransformedWithExtra(base, flow)
    return distribution




def make_equivariant_augmented_flow_dist_distrax_chain(config: FlowDistConfig):
    flow_type = [config.type] if isinstance(config.type, str) else config.type

    if not ("proj" in config.kwargs.keys() or "proj_v2" in config.kwargs.keys()):
        if not config.kwargs == {}:
            raise NotImplementedError
    kwargs_proj = config.kwargs['proj'] if "proj" in config.kwargs.keys() else {}
    kwargs_proj_v2 = config.kwargs['proj_v2'] if "proj_v2" in config.kwargs.keys() else {}

    if config.base_config.double_centered_gaussian:
        base = DoubleCentreGravitryGaussian(dim=config.dim, n_nodes=config.nodes)
    else:
        base = CentreGravitryGaussianAndCondtionalGuassian(
            dim=config.dim, n_nodes=config.nodes, global_centering=config.base_config.global_centering,
            trainable_augmented_scale=config.base_config.trainable_augmented_scale,
            augmented_scale_init=config.base_config.aug_scale_init
        )

    bijectors = []

    for i in range(config.n_layers):
        layer_number = i
        if config.act_norm:
            bijectors.append(make_pseudo_act_norm_bijector(
                layer_number=layer_number, dim=config.dim, flow_identity_init=config.identity_init))

        for swap in (True, False):  # For swap False we condition augmented on original.
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

    if config.act_norm:
        bijectors.append(make_pseudo_act_norm_bijector(layer_number=-1, dim=config.dim,
                                                       flow_identity_init=config.identity_init)
                         )

    flow = ChainWithExtra(bijectors)
    distribution = TransformedWithExtra(base, flow)
    return distribution

