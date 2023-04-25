import distrax
import haiku as hk
import jax.numpy as jnp

from utils.testing import check_bijector_properties
from flow.bijectors.build_proj_coupling import make_proj_coupling_layer
from utils.testing import get_minimal_nets_config

def test_bijector_with_proj(
        transform_type: str = 'spline',
        dim: int = 3, n_layers: int = 4, type='egnn',
                             n_nodes: int = 4, n_aux: int = 3):
    nets_config = get_minimal_nets_config(type=type)

    graph_features = jnp.zeros((n_nodes, 1, 1))

    orth_type = ['loewdin', 'gram-schmidt'][0]

    def make_flow():
        bijectors = []
        for i in range(n_layers):
            swap = i % 2 == 0
            bijector = make_proj_coupling_layer(
                transform_type=transform_type,
                graph_features=graph_features,
                layer_number=i,
                dim=dim,
                n_aug=n_aux,
                swap=swap,
                identity_init=False,
                nets_config=nets_config,
                add_small_identity=False,
                num_bins=4,
                n_inner_transforms=1,
                orthogonalization_method=orth_type
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

    check_bijector_properties(bijector_forward, bijector_backward, dim=dim, n_nodes=n_nodes, n_aux=n_aux)


if __name__ == '__main__':
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)

    for transform_type in ["spline", "real_nvp"]:
        test_bijector_with_proj(transform_type=transform_type, dim=3)
        print('passed test in 3D')

        test_bijector_with_proj(transform_type=transform_type, dim=2)
        print('passed test in 2D')









