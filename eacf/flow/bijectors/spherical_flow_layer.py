from typing import Tuple

import chex
import distrax
import jax.numpy as jnp
import jax

from eacf.utils.spherical import to_spherical_and_log_det, to_cartesian_and_log_det



class SphericalFlow(distrax.Bijector):
    def __init__(self,
                 inner_bijector: distrax.Bijector,
                 reference: chex.Array,
                 reflection_invariant: bool
                 ):
        super().__init__(event_ndims_in=1,
                         event_ndims_out=1)
        self.inner_bijector = inner_bijector
        self.reference = reference
        self.reflection_invariant = reflection_invariant

    def to_spherical_and_log_det(self, x):
        chex.assert_rank(x, 3)
        chex.assert_rank(self.reference, 4)
        sph_x, log_det = jax.vmap(jax.vmap(to_spherical_and_log_det, in_axes=(0, 0, None))
                                  , in_axes=(0, 0, None))(
            x, self.reference, self.reflection_invariant)
        return sph_x, log_det


    def to_cartesian_and_log_det(self, x_sph):
        chex.assert_rank(x_sph, 3)
        chex.assert_rank(self.reference, 4)
        x, log_det = jax.vmap(jax.vmap(to_cartesian_and_log_det, in_axes=(0, 0, None))
                              , in_axes=(0, 0, None))(
            x_sph, self.reference, self.reflection_invariant)
        return x, log_det

    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        n_nodes, multiplicity, dim = x.shape
        sph_x_in, log_det_norm_fwd = self.to_spherical_and_log_det(x)
        sph_x_out, logdet_inner_bijector = self.inner_bijector.forward_and_log_det(sph_x_in)
        chex.assert_equal_shape((sph_x_out, sph_x_in))
        x, log_det_norm_rv = self.to_cartesian_and_log_det(sph_x_out)

        # Compute log det and check shapes.
        logdet_inner_bijector = jnp.sum(logdet_inner_bijector, axis=-1)
        chex.assert_equal_shape((logdet_inner_bijector, log_det_norm_fwd, log_det_norm_rv))
        log_det = logdet_inner_bijector + log_det_norm_fwd + log_det_norm_rv
        chex.assert_shape(log_det, (n_nodes, multiplicity))

        return x, jnp.sum(log_det)

    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        n_nodes, multiplicity, dim = y.shape
        sph_y_in, log_det_norm_fwd = self.to_spherical_and_log_det(y)
        ph_y_out, logdet_inner_bijector = self.inner_bijector.inverse_and_log_det(sph_y_in)
        y, log_det_norm_rv = self.to_cartesian_and_log_det(ph_y_out)

        # Compute log det and check shapes.
        logdet_inner_bijector = jnp.sum(logdet_inner_bijector, axis=-1)
        chex.assert_equal_shape((logdet_inner_bijector, log_det_norm_fwd, log_det_norm_rv))
        log_det = logdet_inner_bijector + log_det_norm_fwd + log_det_norm_rv
        chex.assert_shape(log_det, (n_nodes, multiplicity))

        return y, jnp.sum(log_det)
