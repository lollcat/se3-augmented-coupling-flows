import distrax
import jax.numpy as jnp
import jax.random
import chex

from flow.bijectors.zero_mean_bijector import ZeroMeanBijector
from flow.centre_of_mass_gaussian import assert_mean_zero


def manual_forward(x, scale):
    chex.assert_rank(x, 2)
    n_nodes, dim = x.shape
    chex.assert_shape(scale, (n_nodes,))

    U = (jnp.eye(n_nodes) - 1 / n_nodes * jnp.ones(n_nodes))  # Zero-mean matrix
    S = jnp.diag(scale)

    x_out = U@S@x
    eigen_values, eigen_vectors = jnp.linalg.eig(U.T@S@U)
    nearest_to_zero = jnp.argmin(eigen_values.real**2)
    log_det = jnp.log(jnp.prod(eigen_values.real.at[nearest_to_zero].set(1.)))
    return x_out, log_det, eigen_values




def tesst_zero_mean_bijector(n_nodes: int = 5, dim: int = 2, n_aug: int = 3):
    key = jax.random.PRNGKey(0)
    event_shape = (n_nodes, n_aug, dim)

    key, subkey = jax.random.split(key)
    log_scales = jax.random.normal(key=subkey, shape=(n_nodes,))
    scales = jnp.exp(log_scales)
    x = jax.random.normal(key=key, shape=event_shape)
    x = x - jnp.mean(x, axis=0, keepdims=True)

    # Forward and log det with zero mean affine bijector.
    log_scale_affine_in = jnp.ones_like(x) * log_scales[:, None, None]
    affine_bijector = distrax.ScalarAffine(log_scale=log_scale_affine_in, shift=jnp.zeros(event_shape))
    zero_mean_affince_bijector = ZeroMeanBijector(affine_bijector)
    y, log_det = zero_mean_affince_bijector.forward_and_log_det(x)

    # Manually transform.
    y_, log_det_, eigen_values = jax.vmap(manual_forward, in_axes=(1, None), out_axes=(1, 0, 1))(x, scales)

    # Check zero mean bijector against manual calculation.
    assert_mean_zero(y_, node_axis=0)
    print("manual log det calculation", log_det_)
    print("log det automatic", log_det_)
    chex.assert_trees_all_close(log_det, jnp.sum(log_det_)*dim, rtol=1e-5)





if __name__ == '__main__':
    tesst_zero_mean_bijector()


