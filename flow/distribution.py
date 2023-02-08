from typing import NamedTuple, Optional
import distrax

from flow.base import CentreGravityGaussian
from flow.bijector_proj_real_nvp import make_se_equivariant_split_coupling_with_projection
from flow.bijector_proj_real_nvp_v2 import make_se_equivariant_split_coupling_with_projection as proj_v2
from flow.bijector_nice import make_se_equivariant_nice
from flow.bijector_act_norm import make_global_scaling
from flow.bijector_scale_along_vector import make_se_equivariant_scale_along_vector
from flow.nets import EgnnConfig, TransformerConfig
from flow.fast_hk_chain import Chain


class EquivariantFlowDistConfig(NamedTuple):
    dim: int
    nodes: int
    n_layers: int
    type: str = "nice"
    identity_init: bool = True
    egnn_config: EgnnConfig = EgnnConfig(name="dummy_name")
    fast_compile: bool = True
    compile_n_unroll: int = 1
    transformer_config: Optional[TransformerConfig] = None
    kwargs: dict = {}


def make_equivariant_augmented_flow_dist(config: EquivariantFlowDistConfig):
    if config.fast_compile:
        return make_equivariant_augmented_flow_dist_fast_compile(
            config.dim, config.nodes, config.n_layers,
            config.type, config.identity_init,
            config.egnn_config, config.compile_n_unroll,
            transformer_config=config.transformer_config,
            kwargs=config.kwargs)
    else:
        return make_equivariant_augmented_flow_dist_distrax_chain(config.dim, config.nodes, config.n_layers,
                                                                  config.type, config.identity_init,
                                                                  config.egnn_config,
                                                                  transformer_config=config.transformer_config,
                                                                  kwargs=config.kwargs)


def make_equivariant_augmented_flow_dist_fast_compile(dim,
                                         nodes,
                                         n_layers,
                                         type="nice",
                                         flow_identity_init: bool = True,
                                         egnn_config: EgnnConfig= EgnnConfig(name="dummy_name"),
                                         compile_n_unroll: int = 2,
                                         transformer_config: Optional[TransformerConfig] = None,
                                         act_norm: bool = True,
                                         kwargs: dict = {}):
    if not "proj_v2" in kwargs.keys():
        if not kwargs == {}:
            raise NotImplementedError
    base = CentreGravityGaussian(dim=int(dim*2), n_nodes=nodes)

    def bijector_fn():
        bijectors = []
        for swap in (False, True):
            if act_norm:
                bijectors.append(make_global_scaling(layer_number=0, swap=swap, dim=dim))

            if type == "vector_scale_shift":
                # Append both the nice, and scale_along_vector bijectors
                bijector = make_se_equivariant_scale_along_vector(layer_number=0, dim=dim, swap=swap,
                                                                  identity_init=flow_identity_init,
                                                                  egnn_config=egnn_config)
                bijectors.append(bijector)

                bijector = make_se_equivariant_nice(layer_number=0, dim=dim, swap=swap,
                                                    identity_init=flow_identity_init,
                                                    egnn_config=egnn_config)
                bijectors.append(bijector)

            elif type == "vector_scale":
                bijector = make_se_equivariant_scale_along_vector(layer_number=0, dim=dim, swap=swap,
                                                                  identity_init=flow_identity_init,
                                                                  egnn_config=egnn_config)
                bijectors.append(bijector)

            elif type == "proj":
                bijector = make_se_equivariant_split_coupling_with_projection(layer_number=0, dim=dim, swap=swap,
                                                                              identity_init=flow_identity_init,
                                                                              egnn_config=egnn_config)
                bijectors.append(bijector)
            elif type == "proj_v2":
                kwargs_proj_v2 = kwargs['proj_v2'] if "proj_v2" in kwargs.keys() else {}
                bijector = proj_v2(layer_number=0, dim=dim, swap=swap,
                                  identity_init=flow_identity_init,
                                  egnn_config=egnn_config, transformer_config=transformer_config,
                                   **kwargs_proj_v2)
                bijectors.append(bijector)
            elif type == "nice":
                bijector = make_se_equivariant_nice(layer_number=0, dim=dim, swap=swap,
                                                    identity_init=flow_identity_init,
                                                    egnn_config=egnn_config)
                bijectors.append(bijector)
            else:
                raise NotImplemented
        return distrax.Chain(bijectors)
    flow = Chain(bijector_fn=bijector_fn, n_layers=n_layers, compile_n_unroll=compile_n_unroll)
    distribution = distrax.Transformed(base, flow)
    return distribution




def make_equivariant_augmented_flow_dist_distrax_chain(dim,
                                         nodes,
                                         n_layers,
                                         type="nice",
                                         flow_identity_init: bool = True,
                                         egnn_config: EgnnConfig= EgnnConfig(name="dummy_name"),
                                         transformer_config: Optional[TransformerConfig] = None,
                                         act_norm: bool = True,
                                         kwargs: dict = {}):
    if kwargs != {}:
        raise NotImplementedError
    base = CentreGravityGaussian(dim=int(dim*2), n_nodes=nodes)

    bijectors = []

    for i in range(n_layers):
        for swap in (False, True):
            if act_norm:
                bijectors.append(make_global_scaling(layer_number=0, swap=swap, dim=dim))
            if type == "vector_scale_shift":
                # Append both the nice, and scale_along_vector bijectors
                bijector = make_se_equivariant_scale_along_vector(layer_number=i, dim=dim, swap=swap,
                                                                  identity_init=flow_identity_init,
                                                                  egnn_config=egnn_config)
                bijectors.append(bijector)

                bijector = make_se_equivariant_nice(layer_number=i, dim=dim, swap=swap,
                                                    identity_init=flow_identity_init,
                                                    egnn_config=egnn_config)
                bijectors.append(bijector)

            elif type == "vector_scale":
                bijector = make_se_equivariant_scale_along_vector(layer_number=i, dim=dim, swap=swap,
                                                                  identity_init=flow_identity_init,
                                                                  egnn_config=egnn_config)
                bijectors.append(bijector)

            elif type == "proj":
                bijector = make_se_equivariant_split_coupling_with_projection(layer_number=i, dim=dim, swap=swap,
                                                                              identity_init=flow_identity_init,
                                                                              egnn_config=egnn_config)
                bijectors.append(bijector)
            elif type == "proj_v2":
                bijector = proj_v2(layer_number=i, dim=dim, swap=swap, identity_init=flow_identity_init,
                                                                              egnn_config=egnn_config,
                                   transformer_config=transformer_config)
                bijectors.append(bijector)
            elif type == "nice":
                bijector = make_se_equivariant_nice(layer_number=i, dim=dim, swap=swap,
                                                    identity_init=flow_identity_init,
                                                    egnn_config=egnn_config)
                bijectors.append(bijector)
            else:
                raise NotImplemented

    flow = distrax.Chain(bijectors)
    distribution = distrax.Transformed(base, flow)
    return distribution

