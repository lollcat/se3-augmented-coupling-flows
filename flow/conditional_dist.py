from typing import Tuple, Optional

import chex
import distrax
import jax.numpy as jnp
import jax
import haiku as hk

from flow.distrax_with_extra import DistributionWithExtra, Extra
from flow.aug_flow_dist import FullGraphSample


class ConditionalGaussian(DistributionWithExtra):
    """
    Either:
        a ~ x + Normal
        or
        a ~ Normal
    depending on whether `conditioned` is True.
    """
    def __init__(self,
                 dim: int,
                 n_nodes: int,
                 n_aug: int,
                 x: chex.Array,
                 log_scale: Optional[chex.Array] = None,
                 conditioned: bool = True
                 ):
        self.n_aux = n_aug
        self.dim = dim
        self.n_nodes = n_nodes
        self.x = x
        if log_scale is not None:
            assert log_scale.shape == (n_aug,)
        else:
            log_scale = jnp.zeros(n_aug)
        self.log_scale = log_scale
        self.unit_gaussian = distrax.Independent(distrax.Normal(
            loc=jnp.zeros((n_nodes, n_aug, dim)), scale=jnp.ones((n_nodes, n_aug, dim))),
            reinterpreted_batch_ndims=3)
        self.conditioned = conditioned

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
        momentum = self.unit_gaussian.sample(seed=key, sample_shape=leading_shape)

        # Apply scaling
        if self.x.ndim == 2:
            momentum = jax.vmap(self.scaling_bijector().forward)(momentum)
        else:
            momentum = jax.vmap(jax.vmap(self.scaling_bijector().forward))(momentum)

        if self.conditioned:
            augmented_coords = jnp.expand_dims(self.x, -2) + momentum
        else:
            augmented_coords = momentum

        chex.assert_shape(augmented_coords, (*leading_shape, self.n_nodes, self.n_aux, self.dim))

        return augmented_coords

    def scaling_bijector(self):
        log_scale = jnp.ones((self.n_nodes, self.n_aux, self.dim)) * self.log_scale[None, :, None]
        affine_bijector = distrax.ScalarAffine(shift=jnp.zeros_like(log_scale), log_scale=log_scale)
        return distrax.Block(affine_bijector, ndims=3)

    def log_prob_single_sample(self, x: chex.Array, augmented_coords: chex.Array) -> chex.Array:
        chex.assert_rank(x, 2)  #  [n_nodes, dim]
        chex.assert_rank(augmented_coords, 3)  # [n_nodes, n_aux, dim]

        if self.conditioned:
            momentum = augmented_coords - jnp.expand_dims(x, -2)
        else:
            momentum = augmented_coords
        momentum, log_det_scaling = self.scaling_bijector().inverse_and_log_det(momentum)
        chex.assert_shape(log_det_scaling, ())

        log_prob_momentum_unit_scale = self.unit_gaussian.log_prob(momentum)
        chex.assert_shape(log_prob_momentum_unit_scale, ())

        return log_prob_momentum_unit_scale + log_det_scaling


    def log_prob(self, value: chex.Array) -> chex.Array:
        if self.x.ndim == 3:
            return jax.vmap(self.log_prob_single_sample)(self.x, value)
        else:
            assert self.x.ndim == 2
            return self.log_prob_single_sample(self.x, value)

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
                   augmented_scale_init: float = 1.0,
                   trainable_scale: bool = False,
                   conditioned: bool = True,
                   name: str = 'aux_dist'):

    def make_aux_target(sample: FullGraphSample):
        x = sample.positions
        n_nodes, dim = x.shape[-2:]
        if trainable_scale:
            log_scale = hk.get_parameter(name=name + '_augmented_scale_logit', shape=(n_aug,),
                                         init=hk.initializers.Constant(jnp.log(augmented_scale_init)), dtype=float)
        else:
            scale = jnp.ones(n_aug) * augmented_scale_init
            log_scale = jnp.log(scale)

        dist = ConditionalGaussian(dim=dim, n_nodes=n_nodes, n_aug=n_aug, x=x,
                                   log_scale=log_scale, conditioned=conditioned)
        return dist
    return make_aux_target
