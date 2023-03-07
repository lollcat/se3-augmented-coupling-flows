from typing import Tuple, Union, Sequence, Callable

import distrax
import chex
import jax
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
    def log_prob_with_extra(self, x: chex.Array) -> Tuple[Array, Extra]:
        log_prob = self.log_prob(x)
        info = {}
        return log_prob, info
    
    def sample_n_with_extra(self, key: PRNGKey, n: int) -> Tuple[Array, Extra]:
        sample = self._sample_n(key, n)
        info = {}
        return sample, info

    def sample_n_and_log_prob_with_extra(self, key: PRNGKey, n: int) -> Tuple[Array, Array, Extra]:
        sample, log_prob = self._sample_n_and_log_prob(key, n)
        info = {}
        return sample, log_prob, info


class Transformed(distrax.Transformed):
    def __init__(self, distribution: distrax.DistributionLike, bijector: Bijector):
        super().__init__(distribution=distribution, bijector=bijector)

    def log_prob_with_extra(self, value: chex.Array) -> Tuple[Array, Extra]:
        x, ildj_y, extra = self.bijector.inverse_and_log_det(value)
        lp_x = self.distribution.log_prob(x)
        lp_y = lp_x + ildj_y
        return lp_y, extra

    def sample_n_with_extra(self, key: PRNGKey, n: int) -> Tuple[Array, Extra]:
        x = self.distribution.sample(seed=key, sample_shape=n)
        y, fldj, extra = jax.vmap(self.bijector.forward_and_log_det_with_extra)(x)
        return y, extra

    def sample_n_and_log_prob_with_extra(self, key: PRNGKey, n: int) -> Tuple[Array, Array, Extra]:
        x, lp_x = self.distribution.sample_and_log_prob(seed=key, sample_shape=n)
        y, fldj, extra = jax.vmap(self.bijector.forward_and_log_det_with_extra)(x)
        lp_y = jax.vmap(jnp.subtract)(lp_x, fldj)
        return y, lp_y, extra


from distrax._src.bijectors.split_coupling import BijectorParams


class SplitCouplingWithInfo(distrax.SplitCoupling, Bijector):
    def __init__(self,
               split_index: int,
               event_ndims: int,
               conditioner: Callable[[Array], BijectorParams],
               bijector: Callable[[BijectorParams], Bijector],
               swap: bool = False,
               split_axis: int = -1):
        super().__init__(split_index, event_ndims, conditioner, bijector, swap, split_axis)

    def forward_and_log_det_with_extra(self, x: Array) -> Tuple[Array, Array, Extra]:
        """Like forward_and_log det, but with additional info. Defaults to just returning an empty dict for extra."""
        self._check_forward_input_shape(x)
        x1, x2 = self._split(x)
        params = self._conditioner(x1)
        inner_bijector = self._inner_bijector(params)
        y2, logdet, info = inner_bijector.forward_and_log_det_with_info(x2)
        return self._recombine(x1, y2), logdet

    def inverse_and_log_det_with_extra(self, y: Array) -> Tuple[Array, Array, Extra]:
        """Like inverse_and_log det, but with additional info. Defaults to just returning an empty dict for extra."""
        self._check_inverse_input_shape(y)
        y1, y2 = self._split(y)
        params = self._conditioner(y1)
        x2, logdet = self._inner_bijector(params).inverse_and_log_det(y2)
        return self._recombine(y1, x2), logdet


