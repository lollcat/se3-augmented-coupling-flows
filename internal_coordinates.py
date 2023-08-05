import chex
import jax.numpy as jnp
import jax.random


def x0_centered_to_zero_CoM(x: chex.Array) -> chex.Array:
    n = x.shape[0] + 1
    return x - jnp.sum(x, axis=0, keepdims=True) / n

def zero_CoM_to_x0_centered(x: chex.Array):
    x0_coordinate = -jnp.sum(x, axis=0, keepdims=True)
    return x - x0_coordinate


if __name__ == '__main__':
    key = jax.random.PRNGKey(0)
    n_nodes = 8
    dim = 3
    x_0_centered = jax.random.normal(key, (n_nodes - 1, dim))
    x_zeroCom = x0_centered_to_zero_CoM(x_0_centered)
    x0_in_zeroCom = - jnp.sum(x_zeroCom, axis=0)

    x_0_centered_ = zero_CoM_to_x0_centered(x_zeroCom)

    chex.assert_trees_all_close(x_0_centered, x_0_centered_, atol=1e-6)

    jac = jax.jacfwd(x0_centered_to_zero_CoM)(x_0_centered)

    assert (jac[:, 1, :, 0] == 0).all()

    A = jnp.eye(n_nodes-1)
    u = 1 / n_nodes * jnp.ones(n_nodes-1)
    v = jnp.ones(n_nodes-1)
    u = u[:, None]
    v = v[:, None]
    expected_jac_dim_0 = A - u @ v.T


    chex.assert_trees_all_close(expected_jac_dim_0, jac[:, 0, :, 0])

    sign, log_det = jnp.linalg.slogdet(jac[:, 0, :, 0])

    expected_log_det_dim_0 = - jnp.log(n_nodes)

    log_det_ovall = - dim * jnp.log(n_nodes)
