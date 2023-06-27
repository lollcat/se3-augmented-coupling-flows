from typing import Tuple, Union, Sequence, Callable, Optional, NamedTuple, List

import distrax
import chex
import jax
import jax.numpy as jnp
from distrax._src.distributions.distribution import Array, PRNGKey


@jax.tree_util.register_pytree_node_class
class Extra(NamedTuple):
    aux_loss: chex.Array = jnp.array(0.0)
    aux_info: Optional[dict] = {}
    info_aggregator: Optional[dict] = {}

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


class ChainWithExtra(distrax.Chain, BijectorWithExtra):

    def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        x, log_det = self._bijectors[-1].forward_and_log_det(x)
        for bijector in reversed(self._bijectors[:-1]):
            x, ld = bijector.forward_and_log_det(x)
            chex.assert_equal_shape((log_det, ld))
            log_det += ld
        return x, log_det

    def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        y, log_det = self._bijectors[0].inverse_and_log_det(y)
        for bijector in self._bijectors[1:]:
            y, ld = bijector.inverse_and_log_det(y)
            chex.assert_equal_shape((log_det, ld))
            log_det += ld
        return y, log_det

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
            chex.assert_equal_shape((log_det, ld))
            log_det += ld
            info.update({f"lay_{n_layers - 1 - i}\{n_layers}" + key: value for key, value in extra.aux_info.items()})
            info_aggregator.update({f"lay_{n_layers - 1 - i}\{n_layers}" + key: value for key, value in
                                    extra.info_aggregator.items()})
            losses.append(extra.aux_loss)
        extras = Extra(aux_loss=jnp.squeeze(jnp.mean(jnp.stack(losses))), aux_info=info, info_aggregator=info_aggregator)
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
            chex.assert_equal_shape((log_det, ld))
            log_det += ld
            info.update({f"lay_{2 + i}\{n_layers}" + key: value for key, value in extra.aux_info.items()})
            info_aggregator.update({f"lay_{2 + i}\{n_layers}" + key: value for key, value in
                                    extra.info_aggregator.items()})
            losses.append(extra.aux_loss)
        extras = Extra(aux_loss=jnp.mean(jnp.squeeze(jnp.stack(losses))), aux_info=info,
                       info_aggregator=info_aggregator)
        return y, log_det, extras


from distrax._src.utils import math

class BlockWithExtra(distrax.Block, BijectorWithExtra):
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


class DistributionWithExtra(distrax.Distribution):

    def sample_n_and_log_prob_with_extra(self, key: PRNGKey, n: int) -> Tuple[Array, Array, Extra]:
        sample, log_prob = self._sample_n_and_log_prob(key, n)
        return sample, log_prob, Extra()

    def log_prob_with_extra(self, value: Array) -> Tuple[Array, Extra]:
        log_prob = self.log_prob(value)
        return log_prob, Extra()

