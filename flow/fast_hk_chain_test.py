import haiku as hk

from flow.bijectors.bijector_scale_along_vector import make_se_equivariant_scale_along_vector
from flow.fast_hk_chain import Chain
from nets.egnn import EgnnConfig
from flow.test_utils import bijector_test

def test_chain(n_layers = 5):
    dim = 2
    swap = False
    flow_identity_init = False
    egnn_config = EgnnConfig("")


    def make_flow():
        bijector_fn = lambda: make_se_equivariant_scale_along_vector(layer_number=0, dim=dim, swap=swap,
                                                          identity_init=flow_identity_init,
                                                          egnn_config=egnn_config)
        flow = Chain(bijector_fn=bijector_fn, n_layers=n_layers)
        return flow

    @hk.without_apply_rng
    @hk.transform
    def bijector_forward(x):
        return make_flow().forward_and_log_det(x)

    @hk.without_apply_rng
    @hk.transform
    def bijector_backward(x):
        return make_flow().inverse_and_log_det(x)

    bijector_test(bijector_forward, bijector_backward, dim=dim, n_nodes=4)

if __name__ == '__main__':
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config

        config.update("jax_enable_x64", True)

    test_chain()

