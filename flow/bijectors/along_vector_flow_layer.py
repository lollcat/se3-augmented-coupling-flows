from typing import Tuple

import chex
import distrax
import jax.numpy as jnp
import jax

from utils.numerical import safe_norm


class AlongVectorFlow(distrax.Bijector):
    def __init__(self,
                 inner_bijector: distrax.Bijector,
                 reference: chex.Array
                 ):
        super().__init__(event_ndims_in=1,
                         event_ndims_out=1)
        self.inner_bijector = inner_bijector
        self.reference = reference
        self.dim = reference.shape[-1]

    def to_radial_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
        chex.assert_rank(x, 3)
        chex.assert_equal_shape((x, self.reference))
        vector_of_change = x - self.reference
        r = safe_norm(vector_of_change, axis=-1, keepdims=True)
        unit_vector = vector_of_change / r
        log_det = - (self.dim - 1) * jnp.log(r)
        log_det = jnp.squeeze(log_det, axis=-1)
        return r, unit_vector, log_det

    def to_cartesian_and_log_det(self, r: chex.Array, unit_vector: chex.Array) -> \
            Tuple[chex.Array, chex.Array]:
        chex.assert_rank(r, 3)
        chex.assert_rank(unit_vector, 3)
        x_new = self.reference + r * unit_vector
        log_det = (self.dim - 1) * jnp.log(r)
        log_det = jnp.squeeze(log_det, axis=-1)
        return x_new, log_det

    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        n_nodes, multiplicity, dim = x.shape
        r, unit_vector, log_det_norm_fwd = self.to_radial_and_log_det(x)
        r_new, logdet_inner_bijector = self.inner_bijector.forward_and_log_det(r)
        chex.assert_shape(r_new, (n_nodes, multiplicity, 1))
        x, log_det_norm_rv = self.to_cartesian_and_log_det(r_new, unit_vector)

        # Compute log det and check shapes.
        logdet_inner_bijector = jnp.sum(logdet_inner_bijector, axis=-1)
        chex.assert_equal_shape((logdet_inner_bijector, log_det_norm_fwd, log_det_norm_rv))
        log_det = logdet_inner_bijector + log_det_norm_fwd + log_det_norm_rv
        chex.assert_shape(log_det, (n_nodes, multiplicity))

        return x, jnp.sum(log_det)

    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        n_nodes, multiplicity, dim = y.shape
        r, unit_vector, log_det_norm_fwd = self.to_radial_and_log_det(y)
        r_new, logdet_inner_bijector = self.inner_bijector.inverse_and_log_det(r)
        chex.assert_shape(r_new, (n_nodes, multiplicity, 1))
        x, log_det_norm_rv = self.to_cartesian_and_log_det(r_new, unit_vector)

        # Compute log det and check shapes.
        logdet_inner_bijector = jnp.sum(logdet_inner_bijector, axis=-1)
        chex.assert_equal_shape((logdet_inner_bijector, log_det_norm_fwd, log_det_norm_rv))
        log_det = logdet_inner_bijector + log_det_norm_fwd + log_det_norm_rv
        chex.assert_shape(log_det, (n_nodes, multiplicity))

        return x, jnp.sum(log_det)
