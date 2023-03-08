from typing import Tuple, Union, Sequence, Callable, Optional, NamedTuple, List

import distrax
import chex
import jax
import jax.numpy as jnp
from distrax._src.distributions.distribution import Array, PRNGKey


@jax.tree_util.register_pytree_node_class
class Extra(NamedTuple):
    aux_loss: chex.Array = jnp.array(0.0)
    aux_info: Optional[dict] = None
    info_aggregator: Optional[dict] = jnp.mean

    def aggregate_info(self):
        """Aggregate info as specified, average loss."""
        new_info = {}
        for key, aggregator in self.info_aggregator.items():
            new_info[key] = aggregator(self.aux_info[key])
        return new_info

    def tree_flatten(self):
        return ((self.aux_loss, self.aux_info), self.info_aggregator)

    @classmethod
    def tree_unflatten(cls, aux_data, children):
        return cls(*children, aux_data)


class BijectorWithExtra(distrax.Bijector):

    def forward_and_log_det_with_extra(self, x: Array) -> Tuple[Array, Array, Extra]:
        """Like forward_and_log det, but with additional info. Defaults to just returning an empty dict for extra."""
        y, log_det = self.forward_and_log_det(x)
        info = Extra()
        return y, log_det, info

    def inverse_and_log_det_with_extra(self, y: Array) -> Tuple[Array, Array, Extra]:
        """Like inverse_and_log det, but with additional info. Defaults to just returning an empty dict for extra."""
        x, log_det = self.inverse_and_log_det(y)
        info = Extra()
        return x, log_det, info


class DistributionWithExtra(distrax.Distribution):
    def log_prob_with_extra(self, x: chex.Array) -> Tuple[Array, Extra]:
        log_prob = self.log_prob(x)
        info = Extra()
        return log_prob, info
    
    def sample_n_with_extra(self, key: PRNGKey, n: int) -> Tuple[Array, Extra]:
        sample = self._sample_n(key, n)
        info = Extra()
        return sample, info

    def sample_n_and_log_prob_with_extra(self, key: PRNGKey, n: int) -> Tuple[Array, Array, Extra]:
        sample, log_prob = self._sample_n_and_log_prob(key, n)
        info = Extra()
        return sample, log_prob, info


class TransformedWithExtra(distrax.Transformed):
    def __init__(self, distribution: distrax.DistributionLike, bijector: BijectorWithExtra):
        super().__init__(distribution=distribution, bijector=bijector)

    def log_prob_with_extra(self, value: chex.Array) -> Tuple[Array, Extra]:
        x, ildj_y, extra = self.bijector.inverse_and_log_det_with_extra(value)
        lp_x = self.distribution.log_prob(x)
        lp_y = lp_x + ildj_y
        return lp_y, extra

    def sample_n_with_extra(self, key: PRNGKey, n: int) -> Tuple[Array, Extra]:
        x = self.distribution.sample(seed=key, sample_shape=n)
        y, fldj, extra = jax.vmap(self.bijector.forward_and_log_det_with_extra)(x)
        extra = extra._replace(aux_info=extra.aggregate_info())  # Aggregate info after vmap.
        return y, extra

    def sample_n_and_log_prob_with_extra(self, key: PRNGKey, n: int) -> Tuple[Array, Array, Extra]:
        x, lp_x = self.distribution.sample_and_log_prob(seed=key, sample_shape=n)
        y, fldj, extra = jax.vmap(self.bijector.forward_and_log_det_with_extra)(x)
        lp_y = jax.vmap(jnp.subtract)(lp_x, fldj)
        extra = extra._replace(aux_info=extra.aggregate_info())  # Aggregate info after vmap.
        return y, lp_y, extra


class ChainWithExtra(distrax.Chain):

    def forward_and_log_det_with_extra(self, x: Array) -> Tuple[Array, Array, Extra]:
        """Like forward_and_log det, but with additional info."""
        n_layers = len(self._bijectors)
        losses = []
        info = {}
        info_aggregator = {}
        x, log_det, extra = self._bijectors[-1].forward_and_log_det_with_extra(x)
        losses.append(extra.aux_loss)
        info.update({f"lay_{n_layers}\{n_layers}" + key: value for key, value in extra.aux_info.items()})
        info_aggregator.update({f"lay_{n_layers}\{n_layers}" + key: value for key, value in extra.info_aggregator.items()})
        for i, bijector in enumerate(reversed(self._bijectors[:-1])):
            x, ld, extra = bijector.forward_and_log_det_with_extra(x)
            log_det += ld
            info.update({f"lay_{n_layers - 1 - i}\{n_layers}" + key: value for key, value in extra.aux_info.items()})
            info_aggregator.update({f"lay_{n_layers - 1 - i}\{n_layers}" + key: value for key, value in
                                    extra.info_aggregator.items()})
            losses.append(extra.aux_loss)
        extras = Extra(aux_loss = jnp.stack(losses), aux_info=info, info_aggregator=info_aggregator)
        return x, log_det, extras

    def inverse_and_log_det_with_extra(self, y: Array) -> Tuple[Array, Array, Extra]:
        """Like inverse_and_log det, but with additional extra. Defaults to just returning an empty dict for extra."""
        n_layers = len(self._bijectors)
        losses = []
        info = {}
        info_aggregator = {}
        y, log_det, extra = self._bijectors[0].inverse_and_log_det_with_extra(y)
        info.update({f"lay_{1}\{n_layers}" + key: value for key, value in extra.aux_info.items()})
        info_aggregator.update(
            {f"lay_{1}\{n_layers}" + key: value for key, value in extra.info_aggregator.items()})
        losses.append(extra.aux_loss)
        for i, bijector in enumerate(self._bijectors[1:]):
            y, ld, extra = bijector.inverse_and_log_det_with_extra(y)
            log_det += ld
            info.update({f"lay_{2 + i}\{n_layers}" + key: value for key, value in extra.aux_info.items()})
            info_aggregator.update({f"lay_{2 + i}\{n_layers}" + key: value for key, value in
                                    extra.info_aggregator.items()})
            losses.append(extra.aux_loss)
        extras = Extra(aux_loss = jnp.stack(losses), aux_info=info, info_aggregator=info_aggregator)
        return y, log_det, extras


from distrax._src.utils import math

class BlockWithExtra(distrax.Block):
    def __init__(self, bijector: BijectorWithExtra, ndims: int):
        super().__init__(bijector, ndims)

    def forward_and_log_det_with_extra(self, x: Array) -> Tuple[Array, Array, Extra]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        self._check_forward_input_shape(x)
        y, log_det, extra = self._bijector.forward_and_log_det_with_extra(x)
        return y, math.sum_last(log_det, self._ndims), extra

    def inverse_and_log_det_with_extra(self, y: Array) -> Tuple[Array, Array, Extra]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        self._check_inverse_input_shape(y)
        x, log_det, extra = self._bijector.inverse_and_log_det_with_extra(y)
        return x, math.sum_last(log_det, self._ndims), extra




from distrax._src.bijectors.split_coupling import BijectorParams


class SplitCouplingWithExtra(distrax.SplitCoupling, BijectorWithExtra):
    # TODO: make more clear that conditional can optionall take in whether or not we want info from it.
    def __init__(self,
                 split_index: int,
                 event_ndims: int,
                 conditioner: Callable[[Array, Optional[bool]], BijectorParams],
                 bijector: Callable[[BijectorParams], BijectorWithExtra],
                 swap: bool = False,
                 split_axis: int = -1):
        super().__init__(split_index, event_ndims, conditioner, bijector, swap, split_axis)

    def _inner_bijector(self, params: BijectorParams) -> BijectorWithExtra:
        """Returns an inner bijector for the passed params."""
        bijector = self._bijector(params)
        if bijector.event_ndims_in != bijector.event_ndims_out:
            raise ValueError(
                f'The inner bijector must have `event_ndims_in==event_ndims_out`. '
                f'Instead, it has `event_ndims_in=={bijector.event_ndims_in}` and '
                f'`event_ndims_out=={bijector.event_ndims_out}`.')
        extra_ndims = self.event_ndims_in - bijector.event_ndims_in
        if extra_ndims < 0:
            raise ValueError(
                f'The inner bijector can\'t have more event dimensions than the '
                f'coupling bijector. Got {bijector.event_ndims_in} for the inner '
                f'bijector and {self.event_ndims_in} for the coupling bijector.')
        elif extra_ndims > 0:
            bijector = BlockWithExtra(bijector, extra_ndims)
        return bijector

    def forward_and_log_det_with_extra(self, x: Array) -> Tuple[Array, Array, Extra]:
        """Like forward_and_log det, but with additional info. Defaults to just returning an empty dict for extra."""
        self._check_forward_input_shape(x)
        x1, x2 = self._split(x)
        params = self._conditioner(x1, return_info=True)
        inner_bijector = self._inner_bijector(params)
        y2, logdet, info = inner_bijector.forward_and_log_det_with_extra(x2)
        return self._recombine(x1, y2), logdet, info

    def inverse_and_log_det_with_extra(self, y: Array) -> Tuple[Array, Array, Extra]:
        """Like inverse_and_log det, but with additional info. Defaults to just returning an empty dict for extra."""
        self._check_inverse_input_shape(y)
        y1, y2 = self._split(y)
        params = self._conditioner(y1, return_info=True)
        x2, logdet, info = self._inner_bijector(params).inverse_and_log_det_with_extra(y2)
        return self._recombine(y1, x2), logdet, info