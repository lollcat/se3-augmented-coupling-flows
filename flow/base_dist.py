"""Following https://github.com/vgsatorras/en_flows/blob/main/flows/utils.py."""
from typing import Tuple, Optional

import chex
import jax
import jax.numpy as jnp
import numpy as np
from chex import PRNGKey, Array
import distrax
import haiku as hk

from flow.conditional_dist import get_conditional_gaussian_augmented_dist
from flow.distrax_with_extra import DistributionWithExtra, Extra


class CentreGravitryGaussianAndCondtionalGuassian(DistributionWithExtra):
    """x ~ CentreGravityGaussian, a ~ Normal(x, I)"""
    def __init__(self, dim, n_nodes: int, n_aux: int,
                 global_centering: bool = False,
                 x_scale_init: float = 1.0,
                 trainable_x_scale: bool = False,
                 augmented_scale_init: float = 1.0,
                 trainable_augmented_scale: bool = True):
        self.n_aux = n_aux
        self.dim = dim
        self.n_nodes = n_nodes
        self.global_centering = global_centering
        if trainable_augmented_scale:
            self.augmented_log_scale = hk.get_parameter(name='augmented_log_scale', shape=(n_aux,),
                                                          init=hk.initializers.Constant(jnp.log(augmented_scale_init)))
        else:
            self.augmented_log_scale = jnp.zeros((n_aux,))
        if trainable_x_scale:
            self.x_log_scale = hk.get_parameter(name='x_log_scale', shape=(),
                                                          init=hk.initializers.Constant(jnp.log(x_scale_init)))
        else:
            self.x_log_scale = jnp.zeros(())
        self.x_dist = CentreGravityGaussian(dim=dim, n_nodes=n_nodes, log_scale=self.x_log_scale)


    def get_augmented_dist(self, x: chex.Array) -> distrax.Distribution:
        dist = get_conditional_gaussian_augmented_dist(
            x, n_aux=self.n_aux, scale=jnp.exp(self.augmented_log_scale), global_centering=self.global_centering)
        return dist


    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        key1, key2 = jax.random.split(key)
        original_coords = self.x_dist._sample_n(key1, n)
        augmented_coords = self.get_augmented_dist(original_coords).sample(seed=key2)
        joint_coords = jnp.concatenate([jnp.expand_dims(original_coords, axis=-2), augmented_coords], axis=-2)
        return joint_coords

    def log_prob(self, value: Array) -> Array:
        original_coords, augmented_coords = jnp.split(value, indices_or_sections=[1, ], axis=-2)
        assert original_coords.shape[-2] == 1
        assert augmented_coords.shape[-2] == self.n_aux
        original_coords = jnp.squeeze(original_coords, axis=-2)

        log_prob_a_given_x = self.get_augmented_dist(original_coords).log_prob(augmented_coords)
        log_p_x = self.x_dist.log_prob(original_coords)
        chex.assert_equal_shape((log_p_x, log_prob_a_given_x))
        return log_p_x + log_prob_a_given_x

    def get_extra(self) -> Extra:
        extra = Extra(aux_info={"base_x_scale": jnp.exp(self.x_log_scale)},
                      info_aggregator={"base_x_scale": jnp.mean})
        for i in range(self.n_aux):
            extra.aux_info[f"base_a{i}_scale"] = jnp.exp(self.augmented_log_scale[i])
            extra.info_aggregator[f"base_a{i}_scale"] = jnp.mean
        return extra

    def log_prob_with_extra(self, value: Array) -> Tuple[Array, Extra]:
        extra = self.get_extra()
        log_prob = self.log_prob(value)
        return log_prob, extra

    def sample_n_and_log_prob_with_extra(self, key: PRNGKey, n: int) -> Tuple[Array, Array, Extra]:
        x, log_prob = self._sample_n_and_log_prob(key, n)
        extra = self.get_extra()
        return x, log_prob, extra

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return (self.n_nodes, self.n_aux+1, self.dim)


class CentreGravityGaussian(distrax.Distribution):
    """Guassian distribution over nodes in space, with a zero centre of gravity.
    See https://arxiv.org/pdf/2105.09016.pdf."""
    def __init__(self, dim: int, n_nodes: int, log_scale: chex.Array = jnp.zeros(())):

        self.dim = dim
        self.n_nodes = n_nodes
        self.log_scale = log_scale

    @property
    def scale(self):
        return jnp.exp(self.log_scale)

    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        shape = (n, self.n_nodes, self.dim)
        return sample_center_gravity_zero_gaussian(key, shape)*self.scale

    def log_prob(self, value: Array) -> Array:
        value = remove_mean(value)
        value = value / self.scale
        inv_log_det = - self.log_scale*np.prod(self.event_shape)
        base_log_prob = center_gravity_zero_gaussian_log_likelihood(value)
        return base_log_prob + inv_log_det

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
