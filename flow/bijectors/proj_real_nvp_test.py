import distrax
import haiku as hk
import jax
import jax.numpy as jnp
import chex


from flow.test_utils import bijector_test
from flow.bijectors.proj_real_nvp import make_proj_realnvp
from flow.test_utils import get_minimal_nets_config

def test_bijector_with_proj(dim: int = 3, n_layers: int = 4, type='egnn',
                            n_nodes: int = 4, n_aux: int = 3):
    nets_config = get_minimal_nets_config(type=type)

    graph_features = jnp.zeros((n_nodes, 1, 1))

    def make_flow():
        bijectors = []
        for i in range(n_layers):
            swap = i % 2 == 0
            bijector = make_proj_realnvp(
                graph_features=graph_features,
                layer_number=i,
                dim=dim,
                n_aug=n_aux,
                swap=swap,
                identity_init=False,
                nets_config=nets_config,
                add_small_identity=False
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

    bijector_test(bijector_forward, bijector_backward, dim=dim, n_nodes=n_nodes, n_aux=n_aux)


if __name__ == '__main__':
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)


    test_bijector_with_proj(dim=2)
    print('passed test in 2D')

    test_bijector_with_proj(dim=3)
    print('passed test in 3D')







