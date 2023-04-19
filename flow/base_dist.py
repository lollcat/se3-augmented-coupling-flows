from typing import Tuple

import chex
import jax
import jax.numpy as jnp
from chex import PRNGKey, Array
import distrax

from flow.conditional_dist import ConditionalCentreofMassGaussian
from flow.distrax_with_extra import DistributionWithExtra, Extra


class JointBaseDistribution(DistributionWithExtra):
    """x ~ x_dist, a ~ x + CentreGravityGaussian."""
    def __init__(self,
                 dim,
                 n_nodes: int,
                 n_aux: int,
                 x_dist: distrax.Distribution,
                 augmented_scale_init: float = 1.0,
                 trainable_augmented_scale: bool = False,
                 augmented_conditioned: bool = True,
                 ):
        if trainable_augmented_scale:
            raise NotImplementedError
        self.n_aux = n_aux
        self.dim = dim
        self.n_nodes = n_nodes
        self.x_dist = x_dist
        self.augmented_log_scale = jnp.log(jnp.ones(self.n_aux)*augmented_scale_init)
        self.trainable_augmented_scale = trainable_augmented_scale
        self.augmented_conditioned = augmented_conditioned


    def get_augmented_dist(self, x) -> ConditionalCentreofMassGaussian:
        dist = ConditionalCentreofMassGaussian(
            self.dim, self.n_nodes, self.n_aux, x, log_scale=self.augmented_log_scale,
            conditioned=self.augmented_conditioned
        )
        return dist


    def _sample_n(self, key: PRNGKey, n: int) -> Array:
        key1, key2 = jax.random.split(key)
        original_coords = self.x_dist._sample_n(key1, n)
        aug_dist = self.get_augmented_dist(original_coords)
        augmented_coords = aug_dist.sample(seed=key2)
        joint_coords = jnp.concatenate([jnp.expand_dims(original_coords, axis=-2), augmented_coords], axis=-2)
        chex.assert_shape(joint_coords, (n, *self.event_shape))
        return joint_coords

    def log_prob(self, value: Array) -> Array:
        batch_shape = value.shape[:-3]
        assert value.shape[-3:] == (self.n_nodes, self.n_aux + 1, self.dim)
        original_coords, augmented_coords = jnp.split(value, indices_or_sections=[1, ], axis=-2)
        assert original_coords.shape[-2] == 1
        assert augmented_coords.shape[-2] == self.n_aux
        original_coords = jnp.squeeze(original_coords, axis=-2)

        log_prob_augmented = self.get_augmented_dist(original_coords).log_prob(augmented_coords)
        log_p_x = self.x_dist.log_prob(original_coords)
        chex.assert_equal_shape((log_p_x, log_prob_augmented))
        chex.assert_shape(log_p_x, batch_shape)
        return log_p_x + log_prob_augmented

    def get_extra(self) -> Extra:
        extra = Extra()
        if self.trainable_augmented_scale:
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

