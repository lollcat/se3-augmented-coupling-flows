"""Following https://github.com/vgsatorras/en_flows/blob/main/flows/utils.py."""
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from chex import PRNGKey, Array

import distrax


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
    mean = jnp.mean(x, axis=1, keepdims=True)
    assert jnp.abs(mean).max().item() < 1e-4

def remove_mean(x: chex.Array) -> chex.Array:
    mean = jnp.mean(x, axis=1, keepdims=True)
    x = x - mean
    return x

def center_gravity_zero_gaussian_log_likelihood(x: chex.Array) -> chex.Array:
    chex.assert_rank(x, 3)  # [batch, nodes, x]
    B, N, D = x.shape
    assert_mean_zero(x)

    # r is invariant to a basis change in the relevant hyperplane.
    r2 = jnp.sum(x**2, axis=(-1, -2))

    # The relevant hyperplane is (N-1) * D dimensional.
    degrees_of_freedom = (N-1) * D

    # Normalizing constant and logpx are computed:
    log_normalizing_constant = -0.5 * degrees_of_freedom * jnp.log(2*jnp.pi)
    log_px = -0.5 * r2 + log_normalizing_constant

    return log_px



def sample_center_gravity_zero_gaussian(key: chex.PRNGKey, shape: chex.Shape) -> chex.Array:
    assert len(shape) == 3  # [batch, nodes, x]

    x = jax.random.normal(key, shape)

    # This projection only works because Gaussian is rotation invariant around
    # zero and samples are independent!
    x_projected = remove_mean(x)
    return x_projected



if __name__ == '__main__':
    from test_utils import test_fn_is_invariant
    key = jax.random.PRNGKey(0)
    dim = 2
    n_nodes = 3
    batch_size = 2
    shape = (batch_size, n_nodes, dim)

    # ************* Test distribution ******************8
    dist = CentreGravityGaussian(dim, n_nodes)

    # Sample
    sample = dist.sample(seed=key, sample_shape=batch_size)
    chex.assert_shape(sample, shape)
    assert_mean_zero(sample)

    # Log prob
    log_prob = dist.log_prob(sample)
    chex.assert_shape(log_prob, (batch_size,))
    test_fn_is_invariant(lambda x: dist.log_prob(x[None, ...]), key)  # add fake batch dimension.


    # *********** Test raw functions **********************
    samples = sample_center_gravity_zero_gaussian(key, shape)
    log_prob = center_gravity_zero_gaussian_log_likelihood(samples)
    print(log_prob)
