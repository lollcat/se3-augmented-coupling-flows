"""Following https://github.com/vgsatorras/en_flows/blob/main/flows/utils.py."""
from typing import Tuple, Iterable, Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey, Array
import distrax
import haiku as hk

class CentreGravityGaussian(distrax.Distribution):
    """Gaussian distribution over nodes in space, with a zero centre of gravity.
    See https://arxiv.org/pdf/2105.09016.pdf."""
    def __init__(self, dim: int, n_nodes: int):

        self.dim = dim
        self.n_nodes = n_nodes

    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        shape = (n, self.n_nodes, self.dim)
        return sample_center_gravity_zero_gaussian(key, shape)

    def log_prob(self, value: Array) -> Array:
        value = remove_mean(value)
        base_log_prob = center_gravity_zero_gaussian_log_likelihood(value)
        return base_log_prob

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return (self.n_nodes, self.dim)


class HarmonicPotential(distrax.Distribution):
    """Distribution having a harmonic potential based on a graph,
    similar to base distribution used in https://arxiv.org/abs/2304.02198"""
    def __init__(self, dim: int, n_nodes: int, edges: Iterable = [],
                 a: float = 1., mode_scale: Optional[Array] = None,
                 trainable_mode_scale: bool = False):
        self.dim = dim
        self.n_nodes = n_nodes
        H = np.zeros((n_nodes, n_nodes))
        for i, j in edges:
            H[i, j] = a
            H[j, i] = a
            H[i, i] += a
            H[j, j] += a
        self.D, self.P = jnp.linalg.eigh(jnp.array(H))
        self.std = jnp.concatenate([jnp.zeros(1), 1 / jnp.sqrt(self.D[1:])])
        self.log_det = dim * jnp.sum(jnp.log(self.std[1:]))
        self.log_Z = 0.5 * dim * (n_nodes - 1) * np.log(2 * np.pi)
        if mode_scale is None:
            mode_scale = jnp.ones(n_nodes - 1)
        if trainable_mode_scale:
            self.log_mode_scale = hk.get_parameter('x_base_dist_log_mode_scale', shape=(n_nodes - 1,),
                                                   init=hk.initializers.Constant(jnp.log(mode_scale)),
                                                   dtype=float)
        else:
            self.log_mode_scale = jnp.log(mode_scale)

    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        shape = (n, self.n_nodes, self.dim)
        eps = jax.random.normal(key, shape)
        scale = jnp.concatenate([jnp.zeros(1), jnp.exp(self.log_mode_scale)])
        x = scale[:, None] * eps
        x = self.P @ (self.std[:, None] * x)
        chex.assert_shape(x, shape)
        return remove_mean(x)

    def log_prob(self, value: Array) -> Array:
        value = remove_mean(value)
        value = self.P.T @ value
        value = value / self.std[:, None]
        value = value[..., 1:, :] / jnp.exp(self.log_mode_scale)[:, None]
        log_p = - 0.5 * jnp.sum(value**2, axis=(-2, -1)) - self.log_det \
                - self.log_Z - self.dim * jnp.sum(self.log_mode_scale)
        return log_p

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return (self.n_nodes, self.dim)



def assert_mean_zero(x: chex.Array, node_axis=-2):
    mean = jnp.mean(x, axis=node_axis, keepdims=True)
    chex.assert_trees_all_close(1 + jnp.zeros_like(mean), 1 + mean)

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
