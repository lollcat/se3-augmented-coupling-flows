"""Following https://github.com/vgsatorras/en_flows/blob/main/flows/utils.py."""
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from chex import PRNGKey, Array

import distrax


class DoubleCentreGravitryGaussian(distrax.Distribution):
    """Zero centre of gravity gaussian composed of two sets of coordinates, which together share
    a centre of gravity at zero."""
    def __init__(self, dim, n_nodes):
        self.dim = dim
        self.n_nodes = n_nodes

    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        key1, key2 = jax.random.split(key)
        shape = (n, self.n_nodes, self.dim)
        original_coords = sample_center_gravity_zero_gaussian(key1, shape)
        augmented_coords = sample_center_gravity_zero_gaussian(key2, shape)
        joint_coords = jnp.concatenate([original_coords, augmented_coords], axis=-1)
        return joint_coords

    def log_prob(self, value: Array) -> Array:
        value = remove_mean(value)
        original_coords, augmented_coords = jnp.split(value, 2, axis=-1)
        log_prob = center_gravity_zero_gaussian_log_likelihood(original_coords) +\
                   center_gravity_zero_gaussian_log_likelihood(augmented_coords)
        return log_prob

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return (self.n_nodes, self.dim)


class CentreGravityGaussian(distrax.Distribution):
    """Guassian distribution over nodes in space, with a zero centre of gravity.
    See https://arxiv.org/pdf/2105.09016.pdf."""
    def __init__(self, dim, n_nodes):
        self.dim = dim
        self.n_nodes = n_nodes

    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        shape = (n, self.n_nodes, self.dim)
        return sample_center_gravity_zero_gaussian(key, shape)

    def log_prob(self, value: Array) -> Array:
        value = remove_mean(value)
        return center_gravity_zero_gaussian_log_likelihood(value)

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return (self.n_nodes, self.dim)



def assert_mean_zero(x: chex.Array):
    mean = jnp.mean(x, axis=-2, keepdims=True)
    chex.assert_trees_all_close(mean, jnp.zeros_like(mean))

def remove_mean(x: chex.Array) -> chex.Array:
    mean = jnp.mean(x, axis=-2, keepdims=True)
    x = x - mean
    return x

def center_gravity_zero_gaussian_log_likelihood(x: chex.Array) -> chex.Array:
    try:
        chex.assert_rank(x, 3)  # [batch, nodes, x]
    except:
        chex.assert_rank(x, 2)  # [nodes, x]

    N, D = x.shape[-2:]

    # assert_mean_zero(x)

    # r is invariant to a basis change in the relevant hyperplane.
    r2 = jnp.sum(x**2, axis=(-1, -2))

    # The relevant hyperplane is (N-1) * D dimensional.
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * jnp.log(2*jnp.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px



def sample_center_gravity_zero_gaussian(key: chex.PRNGKey, shape: chex.Shape) -> chex.Array:
    x = jax.random.normal(key, shape)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean(x)
    return x_projected
