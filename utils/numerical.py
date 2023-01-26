import chex
import jax
import jax.numpy as jnp

def get_pairwise_distances(x):
    chex.assert_rank(x, 2)
    diff_combos = x - x[:, None]  # [n_nodes, n_nodes, dim]
    diff_combos = diff_combos.at[jnp.arange(x.shape[0]), jnp.arange(x.shape[0])].set(0.0)  # prevents nan grads
    norms = jnp.linalg.norm(diff_combos, ord=2, axis=-1)
    return norms


def set_diagonal_to_zero(x):
    chex.assert_rank(x, 2)
    return jnp.where(jnp.eye(x.shape[0]), jnp.zeros_like(x), x)

def rotate_translate_2d(x, theta, translation, rotate_first=True):
    chex.assert_shape(theta, ())
    chex.assert_shape(translation, x.shape[-1:])
    rotation_matrix = jnp.array(
        [[jnp.cos(theta), -jnp.sin(theta)],
         [jnp.sin(theta), jnp.cos(theta)]]
    )
    if rotate_first:
        return jax.vmap(jnp.matmul, in_axes=(None, 0))(rotation_matrix, x) + translation[None, :]
    else:
        return jax.vmap(jnp.matmul, in_axes=(None, 0))(rotation_matrix, x + translation[None, :])
