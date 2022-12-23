import chex
import jax
import jax.numpy as jnp
import haiku as hk

def equivariant_fn(x):
    chex.assert_rank(x, 2)
    diff_combos = x - x[:, None]  # [n_nodes, n_nodes, dim]
    norms = jnp.linalg.norm(diff_combos, ord=2, axis=-1)
    m = jnp.squeeze(hk.nets.MLP((5, 1), activation=jax.nn.elu)(norms[..., None]), axis=-1) * 3
    return x + jnp.einsum('ijd,ij->id', diff_combos, m)

def invariant_fn(x, n_vals):
    chex.assert_rank(x, 2)
    equivariant_x = jnp.stack([equivariant_fn(x) for _ in range(n_vals)], axis=-1)
    return jnp.linalg.norm(x[..., None] - equivariant_x, ord=2, axis=-2)
