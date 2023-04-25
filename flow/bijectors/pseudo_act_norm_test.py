import haiku as hk
import jax.numpy as jnp

from utils.test import bijector_test
from flow.bijectors.pseudo_act_norm import make_act_norm
from flow.distrax_with_extra import ChainWithExtra

def test_bijector_pseudo_act_norm(dim: int = 3, n_layers: int = 5,
                                   n_nodes: int = 4, n_aux: int = 3):

    graph_features = jnp.zeros((n_nodes, 1, 1))

    def make_flow():
        bijectors = []
        for i in range(n_layers):
            bijector = make_act_norm(
                graph_features=graph_features,
                layer_number=i,
                dim=dim,
                n_aug=n_aux,
                identity_init=False,
            )
            bijectors.append(bijector)
        flow = ChainWithExtra(bijectors)
        return flow

    @hk.without_apply_rng
    @hk.transform
    def bijector_forward(x):
        flow = make_flow()
        return flow.forward_and_log_det_with_extra(x)[:2]

    @hk.without_apply_rng
    @hk.transform
    def bijector_backward(x):
        flow = make_flow()
        return flow.inverse_and_log_det_with_extra(x)[:2]

    bijector_test(bijector_forward, bijector_backward, dim=dim, n_nodes=n_nodes, n_aux=n_aux)




if __name__ == '__main__':
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)


    test_bijector_pseudo_act_norm(dim=2)
    print('passed test in 2D')

    test_bijector_pseudo_act_norm(dim=3)
    print('passed test in 3D')







