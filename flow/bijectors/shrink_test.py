import haiku as hk
import distrax

from utils.testing import check_bijector_properties
from flow.bijectors.shrink import make_shrink_aug_layer
import jax.numpy as jnp


def test_bijector_shrink(dim: int = 3, n_layers: int = 4, n_nodes: int = 4, n_aux=3):

    graph_features = jnp.zeros((n_nodes, 1, 1))

    def make_flow():
        bijectors = []
        for i in range(n_layers):
            bijector = make_shrink_aug_layer(
                graph_features=graph_features,
                layer_number=i,
                dim=dim,
                n_aug=n_aux,
                swap=False,
                identity_init=False)
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

    test_bijector_shrink(dim=3)
    print('passed test in 3D')
    test_bijector_shrink(dim=2)
    print('passed test in 2D')
