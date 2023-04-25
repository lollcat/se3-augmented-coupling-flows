import distrax
import chex


from flow.bijectors.shrink import build_shrink_layer
from flow.bijectors.centre_of_mass_only_flow import build_unconditional_centre_of_mass_layer
from flow.distrax_with_extra import ChainWithExtra


def make_act_norm(
        graph_features: chex.Array,
        layer_number: int,
        dim: int,
        n_aug: int,
        identity_init: bool):
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
        n_aug=n_aug
        )
    )
    return ChainWithExtra(bijectors)
