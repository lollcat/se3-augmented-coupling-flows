from typing import List

import chex
import jax
import jax.numpy as jnp
import numpy as np

inverse_softplus = lambda x: jnp.log(jnp.exp(x) - 1.)

def param_count(x: chex.ArrayTree) -> int:
    """Count the number of parameters in a PyTree of parameters."""
    return sum(np.prod(p.shape) for p in jax.tree_util.tree_leaves(x))

def safe_norm(x: jnp.ndarray, axis: int = None, keepdims=False) -> jnp.ndarray:
    """nan-safe norm. Copied from mace-jax"""
    x2 = jnp.sum(x**2, axis=axis, keepdims=keepdims)
    return jnp.where(x2 == 0, 1, x2) ** 0.5

def vector_rejection_single(a, b):
    chex.assert_rank(a, 1)
    chex.assert_equal_shape((a, b))
    vector_proj = b * jnp.dot(a, b) / jnp.dot(b, b)
    return a - vector_proj

def vector_rejection(a, b):
    if len(a.shape) == 1:
        return vector_rejection_single(a, b)
    elif len(a.shape) == 2:
        return jax.vmap(vector_rejection_single)(a, b)
    else:
        raise NotImplementedError

def gram_schmidt_fn(vectors: List[chex.Array]):
    vectors.append(jnp.ones_like(vectors[0]))
    n_vectors = len(vectors)
    orthonormal_vectors = []
    u0 = vectors[0] / safe_norm(vectors[0])
    orthonormal_vectors.append(u0)

    for i in range(n_vectors - 1):
        current_vector_indx = i + 1
        temp_vector = vectors[current_vector_indx]
        for j in range(current_vector_indx):
            temp_vector = vector_rejection_single(temp_vector, vectors[j])
        orthonormal_vectors.append(temp_vector / safe_norm(temp_vector))

    return orthonormal_vectors

