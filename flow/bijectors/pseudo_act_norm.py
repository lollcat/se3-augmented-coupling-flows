import distrax
import chex


from flow.bijectors.shrink import make_shrink_aug_layer
from flow.distrax_with_extra import ChainWithExtra


def make_act_norm(
        graph_features: chex.Array,
        layer_number: int,
        dim: int,
        n_aug: int,
        identity_init: bool):
    bijectors = []
    # for swap in (True, False):
    swap = False
    bijectors.append(
        make_shrink_aug_layer(
        graph_features=graph_features, layer_number=layer_number,
        dim=dim,
        identity_init=identity_init,
        n_aug=n_aug, swap=swap)
    )
    return ChainWithExtra(bijectors)
