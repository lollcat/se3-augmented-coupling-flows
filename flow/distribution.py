from typing import NamedTuple, Optional, Sequence, Union
import distrax

from flow.base import DoubleCentreGravitryGaussian, CentreGravitryGaussianAndCondtionalGuassian
from flow.bijector_proj_real_nvp import make_se_equivariant_split_coupling_with_projection
from flow.bijector_nice import make_se_equivariant_nice
from flow.bijector_act_norm import make_act_norm
from flow.bijector_scale_along_vector import make_se_equivariant_scale_along_vector
from flow.nets import EgnnConfig, TransformerConfig
from flow.fast_hk_chain import Chain


class BaseConfig(NamedTuple):
    double_centrered_gaussian: bool = False
    global_centering: bool = False
    trainable_augmented_scale: bool = True
    aug_scale_init: float = 1.0


class EquivariantFlowDistConfig(NamedTuple):
    dim: int
    nodes: int
    n_layers: int
    type: Union[str, Sequence[str]]
    identity_init: bool = True
    egnn_config: EgnnConfig = EgnnConfig(name="dummy_name")
    fast_compile: bool = True
    compile_n_unroll: int = 1
    transformer_config: Optional[TransformerConfig] = None
    act_norm: bool = True
    kwargs: dict = {}
    base_config: BaseConfig = BaseConfig()



def make_equivariant_augmented_flow_dist(config: EquivariantFlowDistConfig):
    if config.fast_compile:
        return make_equivariant_augmented_flow_dist_fast_compile(
            config.dim, config.nodes, config.n_layers,
            config.type, config.identity_init,
            config.egnn_config, config.compile_n_unroll,
            transformer_config=config.transformer_config,
            act_norm=config.act_norm,
            kwargs=config.kwargs,
            base_config=config.base_config
        )
    else:
        return make_equivariant_augmented_flow_dist_distrax_chain(
            config.dim, config.nodes, config.n_layers,
            config.type, config.identity_init,
            config.egnn_config,
            transformer_config=config.transformer_config,
            act_norm=config.act_norm,
            kwargs=config.kwargs,
            base_config=config.base_config
        )


def make_equivariant_augmented_flow_dist_fast_compile(dim,
                                         nodes,
                                         n_layers,
                                         type: Union[str, Sequence[str]],
                                         flow_identity_init: bool = True,
                                         egnn_config: EgnnConfig= EgnnConfig(name="dummy_name"),
                                         compile_n_unroll: int = 2,
                                         transformer_config: Optional[TransformerConfig] = None,
                                         act_norm: bool = True,
                                         base_config: BaseConfig = BaseConfig(),
                                         kwargs: dict = {}):
    if not "proj_v2" in kwargs.keys():
        if not kwargs == {}:
            raise NotImplementedError

    if base_config.double_centrered_gaussian:
        base = DoubleCentreGravitryGaussian(dim=dim, n_nodes=nodes)
    else:
        base = CentreGravitryGaussianAndCondtionalGuassian(
            dim=dim, n_nodes=nodes, global_centering=base_config.global_centering,
            trainable_augmented_scale=base_config.trainable_augmented_scale,
            augmented_scale_init=base_config.aug_scale_init
        )

    def bijector_fn():
        bijectors = []
        kwargs_proj_v2 = kwargs['proj_v2'] if "proj_v2" in kwargs.keys() else {}

        for swap in (True, False):  # For swap False we condition augmented on original.
            if act_norm:
                bijectors.append(make_act_norm(layer_number=0, swap=True, dim=dim,
                                               identity_init=flow_identity_init))
                bijectors.append(make_act_norm(layer_number=0, swap=False, dim=dim,
                                               identity_init=flow_identity_init))

            if "vector_scale" in type:
                bijector = make_se_equivariant_scale_along_vector(layer_number=0, dim=dim, swap=swap,
                                                                  identity_init=flow_identity_init,
                                                                  egnn_config=egnn_config)
                bijectors.append(bijector)
            if "proj" in type:
                bijector = make_se_equivariant_split_coupling_with_projection(layer_number=0, dim=dim, swap=swap,
                                                                              identity_init=flow_identity_init,
                                                                              egnn_config=egnn_config,
                                                                              transformer_config=transformer_config,
                                                                              **kwargs_proj_v2)
                bijectors.append(bijector)

            if "nice" in type:
                bijector = make_se_equivariant_nice(layer_number=0, dim=dim, swap=swap,
                                                    identity_init=flow_identity_init,
                                                    egnn_config=egnn_config)
                bijectors.append(bijector)
        return distrax.Chain(bijectors)
    flow = Chain(bijector_fn=bijector_fn, n_layers=n_layers, compile_n_unroll=compile_n_unroll)
    if act_norm:
        final_act_norm = distrax.Chain([make_act_norm(layer_number=-1, swap=True, dim=dim,
                                                      identity_init=flow_identity_init),
                                        make_act_norm(layer_number=-1, swap=False, dim=dim,
                                                      identity_init=flow_identity_init)])
        flow = distrax.Chain([flow, final_act_norm])
    distribution = distrax.Transformed(base, flow)
    return distribution




def make_equivariant_augmented_flow_dist_distrax_chain(dim,
                                         nodes,
                                         n_layers,
                                         type: Union[str, Sequence[str]],
                                         flow_identity_init: bool = True,
                                         egnn_config: EgnnConfig= EgnnConfig(name="dummy_name"),
                                         transformer_config: Optional[TransformerConfig] = None,
                                         act_norm: bool = True,
                                         kwargs: dict = {},
                                         base_config: BaseConfig = BaseConfig()):
    if not "proj_v2" in kwargs.keys():
        if not kwargs == {}:
            raise NotImplementedError

    if base_config.double_centrered_gaussian:
        base = DoubleCentreGravitryGaussian(dim=dim, n_nodes=nodes)
    else:
        base = CentreGravitryGaussianAndCondtionalGuassian(
            dim=dim, n_nodes=nodes, global_centering=base_config.global_centering,
            trainable_augmented_scale=base_config.trainable_augmented_scale,
        )

    bijectors = []
    kwargs_proj_v2 = kwargs['proj_v2'] if "proj_v2" in kwargs.keys() else {}


    for i in range(n_layers):
        for swap in (True, False):
            if act_norm:
                bijectors.append(make_act_norm(layer_number=0, swap=swap, dim=dim, identity_init=flow_identity_init))
            if "vector_scale" in type:
                bijector = make_se_equivariant_scale_along_vector(layer_number=i, dim=dim, swap=swap,
                                                                  identity_init=flow_identity_init,
                                                                  egnn_config=egnn_config)
                bijectors.append(bijector)

            if "proj" in type:
                bijector = make_se_equivariant_split_coupling_with_projection(layer_number=i, dim=dim, swap=swap,
                                                                              identity_init=flow_identity_init,
                                                                              egnn_config=egnn_config,
                                                                              transformer_config=transformer_config,
                                                                              **kwargs_proj_v2)
                bijectors.append(bijector)
            if "nice" in type:
                bijector = make_se_equivariant_nice(layer_number=i, dim=dim, swap=swap,
                                                    identity_init=flow_identity_init,
                                                    egnn_config=egnn_config)
                bijectors.append(bijector)

    if act_norm:
        bijectors.append(make_act_norm(layer_number=-1, swap=True, dim=dim, identity_init=flow_identity_init))
        bijectors.append(make_act_norm(layer_number=-1, swap=False, dim=dim, identity_init=flow_identity_init))

    flow = distrax.Chain(bijectors)
    distribution = distrax.Transformed(base, flow)
    return distribution

