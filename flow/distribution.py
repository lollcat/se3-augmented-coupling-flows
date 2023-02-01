from typing import NamedTuple, Optional
import distrax

from flow.base import CentreGravityGaussian
from flow.bijector_proj_real_nvp import make_se_equivariant_split_coupling_with_projection
from flow.bijector_proj_real_nvp_v2 import make_se_equivariant_split_coupling_with_projection as proj_v2
from flow.bijector_nice import make_se_equivariant_nice
from flow.bijector_scale_along_vector import make_se_equivariant_scale_along_vector
from flow.nets import EgnnConfig, TransformerConfig
from flow.fast_hk_chain import Chain


class EquivariantFlowDistConfig(NamedTuple):
    dim: int
    nodes: int
    n_layers: int
    type: str = "nice"
    flow_identity_init: bool = True
    egnn_config: EgnnConfig = EgnnConfig(name="dummy_name")
    fast_compile: bool = True
    compile_n_unroll: int = 1
    transformer_config: Optional[TransformerConfig] = None


def make_equivariant_augmented_flow_dist(config: EquivariantFlowDistConfig):
    if config.fast_compile:
        return make_equivariant_augmented_flow_dist_fast_compile(config.dim, config.nodes, config.n_layers,
                                                                 config.type, config.flow_identity_init,
                                                                 config.egnn_config, config.compile_n_unroll)
    else:
        return make_equivariant_augmented_flow_dist_distrax_chain(config.dim, config.nodes, config.n_layers,
                                                          config.type, config.flow_identity_init,
                                                          config.egnn_config)


def make_equivariant_augmented_flow_dist_fast_compile(dim,
                                         nodes,
                                         n_layers,
                                         type="nice",
                                         flow_identity_init: bool = True,
                                         egnn_config: EgnnConfig= EgnnConfig(name="dummy_name"),
                                         compile_n_unroll: int = 2,
                                         transformer_config: Optional[TransformerConfig] = None):
    base = CentreGravityGaussian(dim=int(dim*2), n_nodes=nodes)

    def bijector_fn():
        bijectors = []
        for swap in (False, True):
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
                bijector = proj_v2(layer_number=0, dim=dim, swap=swap,
                                  identity_init=flow_identity_init,
                                  egnn_config=egnn_config, transformer_config=transformer_config)
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
                                         transformer_config: Optional[TransformerConfig] = None):
    base = CentreGravityGaussian(dim=int(dim*2), n_nodes=nodes)

    bijectors = []

    for i in range(n_layers):
        for swap in (False, True):
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

