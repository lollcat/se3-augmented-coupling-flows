from typing import Tuple

import chex
import jax.numpy as jnp
import jax

from flow.distrax_with_extra import DistributionWithExtra, Extra
from flow.aug_flow_dist import FullGraphSample
from flow.centre_of_mass_gaussian import CentreGravityGaussian


class ConditionalCentreofMassGaussian(DistributionWithExtra):
    """a ~ x + CentreGravityGaussian"""
    def __init__(self,
                 dim: int,
                 n_nodes: int,
                 n_aux: int,
                 x: chex.Array,
                 log_scale: chex.Array = jnp.zeros(())):
        self.n_aux = n_aux
        self.dim = dim
        self.n_nodes = n_nodes
        self.centre_gravity_gaussian = CentreGravityGaussian(dim=dim, n_nodes=n_nodes, log_scale=log_scale)
        self.x = x

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return (self.n_nodes, self.n_aux, self.dim)

    def _sample_n(self, key: chex.PRNGKey, n: int) -> chex.Array:
        if self.x.ndim == 2:
            n_x = 1
            leading_x_shape = ()
        else:
            assert self.x.ndim == 3
            n_x = self.x.shape[0]
            leading_x_shape = (n_x,)
        leading_shape = (n, *leading_x_shape)
        momentum = self.centre_gravity_gaussian._sample_n(key, n_x*n*self.n_aux)
        if self.x.ndim == 3:
            momentum = jnp.expand_dims(momentum, 0)
        momentum = jnp.expand_dims(momentum, -3)  # expand for n_aux
        momentum = jnp.reshape(momentum, (*leading_shape, self.n_aux, self.n_nodes, self.dim))
        momentum = jnp.swapaxes(momentum, -2, -3)  # swap n_aux and n_nodes axes.
        augmented_coords = jnp.expand_dims(self.x, -2) + momentum
        chex.assert_shape(augmented_coords, (*leading_shape, self.n_nodes, self.n_aux, self.dim))
        return augmented_coords


    def log_prob(self, value: chex.Array) -> chex.Array:
        augmented_coords = value
        batch_shape = value.shape[:-3]
        assert value.shape[-3:] == (self.n_nodes, self.n_aux, self.dim)
        momentum = augmented_coords - jnp.expand_dims(self.x, -2)

        log_prob_momentum = jax.vmap(self.centre_gravity_gaussian.log_prob, in_axes=-2, out_axes=-1)(momentum)
        chex.assert_shape(log_prob_momentum, (*batch_shape, self.n_aux))
        log_prob_momentum = jnp.sum(log_prob_momentum, axis=-1)

        return log_prob_momentum

    def get_extra(self) -> Extra:
        return Extra()

    def log_prob_with_extra(self, value: chex.Array) -> Tuple[chex.Array, Extra]:
        extra = self.get_extra()
        log_prob = self.log_prob(value)
        return log_prob, extra

    def _sample_n_and_log_prob(self, key: chex.PRNGKey, n: int) -> Tuple[chex.Array, chex.Array]:
        samples = self._sample_n(key, n)
        log_prob = jax.vmap(self.log_prob)(samples)
        return samples, log_prob

    def sample_n_and_log_prob_with_extra(self, key: chex.PRNGKey, n: int) -> Tuple[chex.Array, chex.Array, Extra]:
        x, log_prob = self._sample_n_and_log_prob(key, n)
        extra = self.get_extra()
        return x, log_prob, extra

    @property
    def event_shape(self) -> Tuple[int, ...]:
        return (self.n_nodes, self.n_aux, self.dim)


def build_aux_dist(n_aug: int,
                   augmented_scale_init: float = 1.0):
    def make_aux_target(sample: FullGraphSample):
        x = sample.positions
        n_nodes, dim = x.shape[-2:]
        dist = ConditionalCentreofMassGaussian(dim=dim, n_nodes=n_nodes, n_aux=n_aug, x=x,
                                               log_scale=jnp.log(augmented_scale_init))
        return dist
    return make_aux_target
