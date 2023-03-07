from typing import Tuple, Union, Sequence

import distrax
import chex
import jax.numpy as jnp
from distrax._src.distributions.distribution import Array, PRNGKey, IntLike

Extra = chex.ArrayTree

class Bijector(distrax.Bijector):

    def forward_and_log_det_with_extra(self, x: Array) -> Tuple[Array, Array, Extra]:
        """Like forward_and_log det, but with additional info. Defaults to just returning an empty dict for extra."""
        y, log_det = self.forward_and_log_det(x)
        info = {}
        return y, log_det, info

    def inverse_and_log_det_with_extra(self, y: Array) -> Tuple[Array, Array, Extra]:
        """Like inverse_and_log det, but with additional info. Defaults to just returning an empty dict for extra."""
        x, log_det = self.inverse_and_log_det(y)
        info = {}
        return x, log_det, info


class Distribution(distrax.Distribution):
    def log_prob_with_info(self, x: chex.Array) -> Tuple[Array, Extra]:
        log_prob = self.log_prob(x)
        info = {}
        return log_prob, info

    def sample_with_info(self,
             *,
             seed: Union[IntLike, PRNGKey],
             sample_shape: Union[IntLike, Sequence[IntLike]] = ()) -> Tuple[Array, Extra]:
        sample = self.sample(seed=seed, sample_shape=sample_shape)
        info = {}
        return sample, info


    def sample_and_log_prob_with_info(self,
      *,
      seed: Union[IntLike, PRNGKey],
      sample_shape: Union[IntLike, Sequence[IntLike]] = ()
  ) -> Tuple[Array, Array, Extra]:
        sample, log_prob = self.sample_and_log_prob(seed=seed, sample_shape=sample_shape)
        info = {}
        return sample, log_prob, info



