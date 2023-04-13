from typing import Tuple

import chex
import distrax
import jax.numpy as jnp
import jax

class ZeroMeanBijector(distrax.Bijector):
    """Turn inner bijector into a zero mean bijector.
    This leaves the centre of mass the same in both inputs and outputs.

    This also turns the inner bijector into a block over
    [n_nodes, n_multiplicity, dim]."""
    def __init__(self,
                 inner_bijector: distrax.Bijector
                 ):
        super().__init__(event_ndims_in=0,
                         event_ndims_out=3)
        assert inner_bijector.event_ndims_in == 0
        zero_mean_axis: int = -3
        self.inner_bijector = inner_bijector
        self.zero_mean_axis = zero_mean_axis

    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        centre_of_mass_input = jnp.mean(x, axis=self.zero_mean_axis, keepdims=True)
        y, log_det = self.inner_bijector.forward_and_log_det(x)
        y = y - jnp.mean(y, axis=self.zero_mean_axis, keepdims=True) + centre_of_mass_input
        log_n_nodes = jnp.log(y.shape[self.zero_mean_axis])
        log_det_adjustment = jax.nn.logsumexp(-log_det, axis=self.zero_mean_axis) - log_n_nodes
        log_det_adjustment = jnp.sum(log_det_adjustment, axis=(-2, -1))
        log_det = jnp.sum(log_det, axis=(-3, -2, -1))
        chex.assert_equal_shape((log_det, log_det_adjustment))
        return y, (log_det + log_det_adjustment)

    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        centre_of_mass_input = jnp.mean(y, axis=self.zero_mean_axis, keepdims=True)
        x, log_det = self.inner_bijector.inverse_and_log_det(y)
        x = x - jnp.mean(x, axis=self.zero_mean_axis, keepdims=True) + centre_of_mass_input
        log_n_nodes = jnp.log(x.shape[self.zero_mean_axis])
        log_det_adjustment = jax.nn.logsumexp(-log_det, axis=self.zero_mean_axis) - log_n_nodes
        log_det_adjustment = jnp.sum(log_det_adjustment, axis=(-2, -1))
        log_det = jnp.sum(log_det, axis=(-3, -2, -1))
        chex.assert_equal_shape((log_det, log_det_adjustment))
        return x, (log_det + log_det_adjustment)
