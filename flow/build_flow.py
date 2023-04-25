import chex

from flow.aug_flow_dist import AugmentedFlowRecipe, AugmentedFlow, create_flow

from typing import NamedTuple, Sequence, Union, Iterable
import distrax

from flow.base_dist import JointBaseDistribution
from flow.x_base_dist import CentreGravityGaussian, HarmonicPotential
from flow.conditional_dist import build_aux_target_dist
from flow.bijectors.build_proj_coupling import make_proj_coupling_layer
from flow.bijectors.equi_nice import make_se_equivariant_nice
from flow.bijectors.pseudo_act_norm import make_act_norm
from flow.bijectors.build_spherical_coupling import make_spherical_coupling_layer
from flow.bijectors.build_along_vector_coupling import make_along_vector_coupling_layer
from flow.bijectors.build_centre_of_mass_invariant_coupling import make_centre_of_mass_invariant_coupling_layer
from flow.distrax_with_extra import ChainWithExtra

from nets.base import NetsConfig




class ConditionalAuxDistConfig(NamedTuple):
    conditioned_on_x: bool = False
    trainable_augmented_scale: bool = False
    scale_init: float = 1.0

class XDistConfig(NamedTuple):
    type: str = 'centre_gravity_gaussian'  # centre_gravity_gaussian or harmonic_potential
    a: float = 1.0
    edges: Iterable = []
    trainable_mode_scale: bool = False

class BaseConfig(NamedTuple):
    x_dist: XDistConfig = XDistConfig()
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
    for flow in flow_type:
        assert flow in ['nice', 'proj', 'spherical', 'along_vector', 'non_equivariant']
        if 'non_equivariant' in flow:
            assert len(flow_type) == 1
            assert config.act_norm is False

    def make_base() -> distrax.Distribution:
        assert config.base.x_dist.type in ['centre_gravity_gaussian', 'harmonic_potential']
        if config.base.x_dist.type == 'centre_gravity_gaussian':
            x_dist = CentreGravityGaussian(dim=config.dim, n_nodes=config.nodes)
        elif config.base.x_dist.type == 'harmonic_potential':
            x_dist = HarmonicPotential(dim=config.dim, n_nodes=config.nodes, a=config.base.x_dist.a,
                                       edges=config.base.x_dist.edges,
                                       trainable_mode_scale=config.base.x_dist.trainable_mode_scale)
        base = JointBaseDistribution(
            dim=config.dim,
            n_nodes=config.nodes,
            n_aux=config.n_aug,
            x_dist=x_dist,
            augmented_scale_init=config.base.aug.scale_init,
            augmented_conditioned=config.base.aug.conditioned_on_x,
            trainable_augmented_scale=config.base.aug.trainable_augmented_scale
        )
        return base

    def make_bijector(graph_features: chex.Array):
        # Note that bijector.inverse moves through this forwards, and bijector.fowards reverses the bijector order
        bijectors = []
        layer_number = 0

        if config.act_norm:
            bijector = make_act_norm(
                layer_number=layer_number,
                graph_features=graph_features,
                dim=config.dim,
                n_aug=config.n_aug,
                identity_init=config.identity_init)
            bijectors.append(bijector)

        for swap in (False, True):  # For swap False we condition augmented on original.
            if 'non_equivariant' in flow_type:
                kwargs_non_equivariant = config.kwargs['non_equivariant'] if \
                    'non_equivariant' in config.kwargs.keys() else {}
                bijector = make_centre_of_mass_invariant_coupling_layer(
                    graph_features=graph_features,
                    n_aug=config.n_aug,
                    layer_number=layer_number,
                    dim=config.dim,
                    swap=swap,
                    identity_init=config.identity_init,
                    mlp_head_config=config.nets_config.mlp_head_config,
                    transformer_config=config.nets_config.non_equivariant_transformer_config,
                    **kwargs_non_equivariant
                )
                bijectors.append(bijector)

            if 'along_vector' in flow_type:
                kwargs_along_vector = config.kwargs['along_vector'] if 'along_vector' in config.kwargs.keys() else {}
                bijector = make_along_vector_coupling_layer(
                    graph_features=graph_features,
                    n_aug=config.n_aug,
                    layer_number=layer_number,
                    dim=config.dim,
                    swap=swap,
                    identity_init=config.identity_init,
                    nets_config=config.nets_config,
                    **kwargs_along_vector
                )
                bijectors.append(bijector)
            if 'spherical' in flow_type:
                kwargs_spherical = config.kwargs['spherical'] if 'spherical' in config.kwargs.keys() else {}
                bijector = make_spherical_coupling_layer(
                    graph_features=graph_features,
                    n_aug=config.n_aug,
                    layer_number=layer_number,
                    dim=config.dim,
                    swap=swap,
                    identity_init=config.identity_init,
                    nets_config=config.nets_config,
                    **kwargs_spherical
                )
                bijectors.append(bijector)

            if "proj" in flow_type:
                kwargs_proj = config.kwargs["proj"] if "proj" in config.kwargs.keys() else {}
                bijector = make_proj_coupling_layer(
                    graph_features=graph_features, n_aug=config.n_aug,
                    layer_number=layer_number, dim=config.dim,
                    swap=swap,
                    identity_init=config.identity_init,
                    nets_config=config.nets_config,
                    **kwargs_proj
                )
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

    make_aug_target = build_aux_target_dist(
        n_aug=config.n_aug,
        augmented_scale_init=config.target_aux_config.scale_init,
        trainable_scale=config.target_aux_config.trainable_augmented_scale,
        conditioned=config.target_aux_config.conditioned_on_x
    )

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
