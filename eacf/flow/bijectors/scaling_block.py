import chex


from eacf.flow.bijectors.shrink import build_shrink_layer
from eacf.flow.bijectors.centre_of_mass_only_flow import build_unconditional_centre_of_mass_layer
from eacf.flow.distrax_with_extra import ChainWithExtra


def make_scaling_block(
        graph_features: chex.Array,
        layer_number: int,
        dim: int,
        n_aug: int,
        identity_init: bool,
        condition: bool
) -> ChainWithExtra:
    
    bijectors = []
    bijectors.append(
        build_unconditional_centre_of_mass_layer(
            graph_features=graph_features,
            layer_number=layer_number,
            dim=dim,
            identity_init=identity_init,
            n_aug=n_aug))

    bijectors.append(
        build_shrink_layer(
        graph_features=graph_features, layer_number=layer_number,
        dim=dim,
        identity_init=identity_init,
        n_aug=n_aug,
        condition=condition
        )
    )
    return ChainWithExtra(bijectors)
