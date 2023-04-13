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
                 inner_bijector: distrax.Bijector,
                 block_dim: Tuple[int, ...] = (-3, -2, -1)
                 ):
        """

        Args:
            inner_bijector: Bijector that may change the mean.
            block_dim: Dimensions with which to block over. Must be equal to 3 - inner_bijector_event_ndims_in.
                By block I mean consider as joint event_dimension (see distrax.Block).
        """
        super().__init__(event_ndims_in=inner_bijector.event_ndims_in,
                         event_ndims_out=3)
        assert inner_bijector.event_ndims_in < 3  # Node dimension must not yet be joint/block.
        assert (3 - inner_bijector.event_ndims_in) == len(block_dim)
        assert inner_bijector.event_ndims_in == inner_bijector.event_ndims_out

        # Zero mean axis is always -3, corresponding to the n_nodes dimension.
        zero_mean_axis: int = -3
        self.inner_bijector = inner_bijector
        self.zero_mean_axis = zero_mean_axis
        self.block_dim = block_dim

        # When calculating the log det adjustment, we sum over the n_nodes dimension, as well as
        # the n_multiplicity and dim groups, if these are not yet part of the event dimension.
        # Note: For the `proj` flow, the inner bijector transforms all dimensions independantly, hence
        # is diagonal within the [n_nodes, dim] dimensions.
        # On the other hand, the spherical flow transforms the [dim] dimension jointly. This effects
        # whether summing over the [dim] dimension for occurs inside of our outside of the `log_sum_exp`.
        # TODO: Think about this more and check it.
        self.log_det_adjust_sum_dim = block_dim[1:]
        self.log_det_adjust_zero_mean_axis = zero_mean_axis + self.inner_bijector.event_ndims_in

    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        centre_of_mass_input = jnp.mean(x, axis=self.zero_mean_axis, keepdims=True)
        y, log_det = self.inner_bijector.forward_and_log_det(x)
        y = y - jnp.mean(y, axis=self.zero_mean_axis, keepdims=True) + centre_of_mass_input
        log_n_nodes = jnp.log(y.shape[self.zero_mean_axis])
        log_det_adjustment = jax.nn.logsumexp(-log_det, axis=self.log_det_adjust_zero_mean_axis) - log_n_nodes
        log_det_adjustment = jnp.sum(log_det_adjustment, axis=self.log_det_adjust_sum_dim)
        log_det = jnp.sum(log_det, axis=self.block_dim)
        chex.assert_equal_shape((log_det, log_det_adjustment))
        return y, (log_det + log_det_adjustment)

    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        centre_of_mass_input = jnp.mean(y, axis=self.zero_mean_axis, keepdims=True)
        x, log_det = self.inner_bijector.inverse_and_log_det(y)
        x = x - jnp.mean(x, axis=self.zero_mean_axis, keepdims=True) + centre_of_mass_input
        log_n_nodes = jnp.log(x.shape[self.zero_mean_axis])
        log_det_adjustment = jax.nn.logsumexp(-log_det, axis=self.log_det_adjust_zero_mean_axis) - log_n_nodes
        log_det_adjustment = jnp.sum(log_det_adjustment, axis=self.log_det_adjust_sum_dim)
        log_det = jnp.sum(log_det, axis=self.block_dim)
        chex.assert_equal_shape((log_det, log_det_adjustment))
        return x, (log_det + log_det_adjustment)
