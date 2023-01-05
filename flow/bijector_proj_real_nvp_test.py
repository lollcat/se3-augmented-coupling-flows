import jax
import jax.numpy as jnp
import haiku as hk
import chex

from flow.test_utils import bijector_test
from flow.bijector_proj_real_nvp import make_se_equivariant_split_coupling_with_projection


def test_bijector_with_proj():
    @hk.without_apply_rng
    @hk.transform
    def bijector_forward(x):
        bijector = make_se_equivariant_split_coupling_with_projection(dim, swap=False, identity_init=False)
        return bijector.forward_and_log_det(x)


    @hk.without_apply_rng
    @hk.transform
    def bijector_backward(x):
        bijector = make_se_equivariant_split_coupling_with_projection(dim, swap=False, identity_init=False)
        return bijector.inverse_and_log_det(x)

    dim = 2
    n_nodes = 4
    bijector_test(bijector_forward, bijector_backward, dim, n_nodes)


if __name__ == '__main__':
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config

        config.update("jax_enable_x64", True)

    test_bijector_with_proj()
