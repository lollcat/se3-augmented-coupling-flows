import distrax
import haiku as hk
import jax.numpy as jnp

USE_64_BIT = True
if USE_64_BIT:
    from jax.config import config
    config.update("jax_enable_x64", True)


from eacf.utils.testing import check_bijector_properties
from eacf.flow.bijectors.build_spherical_coupling import make_spherical_coupling_layer
from eacf.utils.testing import get_minimal_nets_config

def test_bijector_spherical(dim: int = 3, n_layers: int = 1, type='egnn',
                            n_nodes: int = 19, n_aux: int = 1):
    nets_config = get_minimal_nets_config(type=type)

    graph_features = jnp.zeros((n_nodes, 1, 1), dtype=int)

    def make_flow():
        bijectors = []
        for i in range(n_layers):
            swap = i % 2 == 0
            bijector = make_spherical_coupling_layer(
                graph_features=graph_features,
                layer_number=i,
                dim=dim,
                n_aug=n_aux,
                swap=swap,
                identity_init=False,
                nets_config=nets_config,
                n_inner_transforms=2,
                reflection_invariant=True
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
    test_bijector_spherical(dim=2)
    print('passed test in 2D')

    test_bijector_spherical(dim=3)
    print('passed test in 3D')

