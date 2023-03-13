"""Following https://github.com/vgsatorras/en_flows/blob/main/flows/utils.py."""
from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from chex import PRNGKey, Array
import haiku as hk

import distrax

from flow.conditional_dist import get_conditional_gaussian_augmented_dist

class CentreGravitryGaussianAndCondtionalGuassian(distrax.Distribution):
    """x ~ CentreGravityGaussian, a ~ Normal(x, I)"""
    def __init__(self, dim, n_nodes: int, n_aux: int, global_centering: bool = False,
                 augmented_scale_init: float = 1.0,
                 trainable_augmented_scale: bool = True):
        self.n_aux = n_aux
        self.dim = dim
        self.n_nodes = n_nodes
        self.global_centering = global_centering
        if trainable_augmented_scale:
            self.augmented_scale_logit = hk.get_parameter(name='augmented_scale_logit', shape=(n_aux,),
                                                          init=hk.initializers.Constant(jnp.log(augmented_scale_init)))
        else:
            self.augmented_scale_logit = jnp.log(jnp.ones(n_aux) * augmented_scale_init)

    def get_augmented_dist(self, x) -> distrax.Distribution:
        dist = get_conditional_gaussian_augmented_dist(x, n_aux=self.n_aux, scale=jnp.exp(self.augmented_scale_logit),
                                                       global_centering=self.global_centering)
        return dist


    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        key1, key2 = jax.random.split(key)
        shape = (n, self.n_nodes, self.dim)
        original_coords = sample_center_gravity_zero_gaussian(key1, shape)
        augmented_coords = self.get_augmented_dist(original_coords).sample(seed=key2)
        joint_coords = jnp.concatenate([original_coords[:, None, ...], augmented_coords], axis=1)
        return joint_coords

    def log_prob(self, value: Array) -> Array:
        original_coords, augmented_coords = jnp.split(value,
                                                      indices_or_sections=jnp.array([1,]),
                                                      axis=-3)
        assert original_coords.shape[-3] == 1
        assert augmented_coords.shape[-3] == self.n_aux
        original_coords = jnp.squeeze(original_coords, axis=-3)

        log_prob_a_given_x = self.get_augmented_dist(original_coords).log_prob(augmented_coords)
        log_p_x = center_gravity_zero_gaussian_log_likelihood(remove_mean(original_coords))
        chex.assert_equal_shape((log_p_x, log_prob_a_given_x))
        return log_p_x + log_prob_a_given_x

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return (self.n_aux+1, self.n_nodes, self.dim)


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
