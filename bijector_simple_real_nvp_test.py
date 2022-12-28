import jax
import jax.numpy as jnp
import haiku as hk
import chex

from test_utils import test_fn_is_equivariant, test_fn_is_invariant
from bijector_simple_real_nvp import make_se_equivariant_split_coupling_simple


def test_bijector_with_proj():
    """Test that the bijector is equivariant, and that it's log determinant is invariant."""

    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config

        config.update("jax_enable_x64", True)

    if USE_64_BIT:
        r_tol = 1e-6
    else:
        r_tol = 1e-3

    @hk.without_apply_rng
    @hk.transform
    def bijector_forward(x):
        bijector = make_se_equivariant_split_coupling_simple(dim, swap=False)
        return bijector.forward_and_log_det(x)


    @hk.without_apply_rng
    @hk.transform
    def bijector_backward(x):
        bijector = make_se_equivariant_split_coupling_simple(dim, swap=False)
        return bijector.inverse_and_log_det(x)

    dim = 2
    n_nodes = 4
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    # Create dummy x and a.
    x_and_a = jnp.zeros((n_nodes, dim*2))
    x_and_a = x_and_a + jax.random.normal(subkey, shape=x_and_a.shape)*0.1

    # Initialise bijector parameters.
    params = bijector_forward.init(key, x_and_a)

    # Perform a forward pass.
    x_and_a_new, log_det_fwd = bijector_forward.apply(params, x_and_a)

    # Invert.
    x_and_a_old, log_det_rev = bijector_backward.apply(params, x_and_a_new)

    chex.assert_shape(log_det_fwd, ())
    chex.assert_trees_all_close(x_and_a, x_and_a_old, rtol=r_tol)

    # Test the transformation is equivariant.
    key, subkey = jax.random.split(key)
    test_fn_is_equivariant(lambda x_and_a: bijector_forward.apply(params, x_and_a)[0], subkey)
    key, subkey = jax.random.split(key)
    test_fn_is_equivariant(lambda x_and_a: bijector_backward.apply(params, x_and_a)[0], subkey)

    # Check the change to the log det is invariant
    key, subkey = jax.random.split(key)
    test_fn_is_invariant(lambda x_and_a: bijector_forward.apply(params, x_and_a)[1], subkey)
    key, subkey = jax.random.split(key)
    test_fn_is_invariant(lambda x_and_a: bijector_backward.apply(params, x_and_a)[1], subkey)


if __name__ == '__main__':
    test_bijector_with_proj()
