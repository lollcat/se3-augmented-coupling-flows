import chex
import jax
import jax.numpy as jnp
from jax.flatten_util import ravel_pytree
import numpy as np

inverse_softplus = lambda x: jnp.log(jnp.exp(x) - 1.)

def param_count(x: chex.ArrayTree) -> int:
    """Count the number of parameters in a PyTree of parameters."""
    return ravel_pytree(x)[0].shape[0]

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

