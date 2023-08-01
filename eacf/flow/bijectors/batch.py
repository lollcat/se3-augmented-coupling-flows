from typing import Tuple

import distrax
import jax
import chex

from eacf.flow.distrax_with_extra import BijectorWithExtra, Extra

class BatchBijector(BijectorWithExtra):
    """Auto-detect batch and vmap. Only one batch axis allowed."""
    def __init__(self, inner_bijector: distrax.Bijector):
        super().__init__(event_ndims_in=inner_bijector.event_ndims_in,
                         event_ndims_out=inner_bijector.event_ndims_out)
        assert inner_bijector.event_ndims_in == inner_bijector.event_ndims_out
        self.inner_bijector = inner_bijector

    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        if x.ndim == self.event_ndims_out:
            y, log_det = self.inner_bijector.forward_and_log_det(x)
            chex.assert_shape(log_det, ())
            chex.assert_equal_shape((y, x))
        else:
            assert x.ndim == (self.event_ndims_in + 1)
            y, log_det = jax.vmap(self.inner_bijector.forward_and_log_det)(x)
            chex.assert_shape(log_det, (x.shape[0],))
            chex.assert_equal_shape((y, x))
        return y, log_det

    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        if y.ndim == self.event_ndims_out:
            x, log_det = self.inner_bijector.inverse_and_log_det(y)
            chex.assert_shape(log_det, ())
            chex.assert_equal_shape((y, x))
        else:
            assert y.ndim == (self.event_ndims_in + 1)
            x, log_det = jax.vmap(self.inner_bijector.inverse_and_log_det)(y)
            chex.assert_shape(log_det, (y.shape[0],))
            chex.assert_equal_shape((y, x))
        return x, log_det

    # TODO: Write the below functions. We can then apply this to essentially all of the bijectors, which
    # will greatly simplify all the auto-batch detection code.
    def forward_and_log_det_with_extra(self, x: chex.Array) -> Tuple[chex.Array, chex.Array, Extra]:
        if isinstance(self.inner_bijector, BijectorWithExtra):
            y, log_det = self.forward_and_log_det(x)
        else:
            y, log_det = self.forward_and_log_det(x)
        extra = Extra()
        return y, log_det, extra

    def inverse_and_log_det_with_extra(self, y: chex.Array) -> Tuple[chex.Array, chex.Array, Extra]:
        if isinstance(self.inner_bijector, BijectorWithExtra):
            x, log_det = self.inverse_and_log_det(y)
        else:
            x, log_det = self.inverse_and_log_det(y)
        extra = Extra()
        return x, log_det, extra

