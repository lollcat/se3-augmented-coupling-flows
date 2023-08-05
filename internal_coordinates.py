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
    x_0_centered = jax.random.normal(key, (n_nodes, dim))
    x_zeroCom = x0_centered_to_zero_CoM(x_0_centered)
    x0_in_zeroCom = - jnp.sum(x_zeroCom, axis=0)

    x_0_centered_ = zero_CoM_to_x0_centered(x_zeroCom)

    chex.assert_trees_all_close(x_0_centered, x_0_centered_, atol=1e-6)

