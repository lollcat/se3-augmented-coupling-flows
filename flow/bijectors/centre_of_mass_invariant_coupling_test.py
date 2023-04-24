import distrax
import haiku as hk
import jax.numpy as jnp

from utils.testing import check_bijector_properties
from flow.bijectors.build_centre_of_mass_invariant_coupling import make_centre_of_mass_invariant_coupling_layer
from utils.testing import get_minimal_nets_config

def tesst_bijector_centre_of_mass_only(dim: int = 3, n_layers: int = 1, n_nodes: int = 4, n_aux: int = 1,
                                      n_inner_transformer: int = 1):
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)

    nets_config = get_minimal_nets_config(type=type)

    graph_features = jnp.zeros((n_nodes, 1, 1))

    def make_flow():
        bijectors = []
        for i in range(n_layers):
            swap = i % 2 == 0
            bijector = make_centre_of_mass_invariant_coupling_layer(
                graph_features=graph_features,
                layer_number=i,
                dim=dim,
                n_aug=n_aux,
                swap=swap,
                identity_init=False,
                transformer_config=nets_config.non_equivariant_transformer_config,
                mlp_head_config=nets_config.mlp_head_config,
                n_inner_transforms=n_inner_transformer
            )
            bijectors.append(bijector)
        flow = distrax.Chain(bijectors)
        return flow

    @hk.without_apply_rng
    @hk.transform
    def bijector_forward(x):
        flow = make_flow()
        return flow.forward_and_log_det(x)

    @hk.without_apply_rng
    @hk.transform
    def bijector_backward(x):
        flow = make_flow()
        return flow.inverse_and_log_det(x)

    check_bijector_properties(bijector_forward, bijector_backward, dim=dim, n_nodes=n_nodes, n_aux=n_aux,
                              test_rotation_equivariance=False)


if __name__ == '__main__':
    tesst_bijector_centre_of_mass_only(dim=2)
    print('passed test in 2D')

    tesst_bijector_centre_of_mass_only(dim=3)
    print('passed test in 3D')
