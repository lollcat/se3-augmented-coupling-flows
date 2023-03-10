import haiku as hk
import distrax

from flow.test_utils import bijector_test
from flow.bijectors.bijector_coupled_global_scaling import make_coupled_global_scaling


def test_bijector_with_proj(dim: int = 3, n_layers: int = 11):

    def make_flow():
        bijectors = []
        for i in range(n_layers):
            swap = i % 2 == 0
            bijector = make_coupled_global_scaling(layer_number=i, dim=dim, swap=swap, identity_init=False)
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

    bijector_test(bijector_forward, bijector_backward, dim=dim, n_nodes=4)


if __name__ == '__main__':
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)

    test_bijector_with_proj(dim=3)
    print('passed test in 3D')
    test_bijector_with_proj(dim=2)
    print('passed test in 2D')
