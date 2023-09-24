from typing import Tuple

import chex
import distrax
import jax
import jax.numpy as jnp


def project(x: chex.Array, origin: chex.Array, u: chex.Array) -> Tuple[chex.Array, chex.Array]:
    chex.assert_rank(x, 1)
    chex.assert_equal_shape((x, origin, u))
    x_vector = x - origin
    x_proj = jnp.dot(x_vector, u)  # Scalar representing x position along line intersecting origin, in direction u.
    x_constant = x_vector - x_proj * u
    return x_proj[None], x_constant

def unproject(x_proj: chex.Array, x_constant, origin: chex.Array, u: chex.Array) -> chex.Array:
    chex.assert_shape(x_proj, (1,))
    chex.assert_equal_shape((x_constant, origin, u))
    x_proj = jnp.squeeze(x_proj)
    x = origin + x_proj * u + x_constant
    return x


class LineProjFlow(distrax.Bijector):
    def __init__(self,
                 inner_bijector: distrax.Bijector,
                 origin: chex.Array,
                 u: chex.Array
                 ):
        super().__init__(event_ndims_in=0,
                         event_ndims_out=0)
        self.inner_bijector = inner_bijector
        self.origin = origin
        self.u = u

    def to_invariant_space(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        x_proj, x_constant = jax.vmap(jax.vmap(project))(x, self.origin, self.u)
        return x_proj, x_constant

    def to_equivariant_space(self, x_proj: chex.Array, x_constant: chex.Array) -> chex.Array:
        x = jax.vmap(jax.vmap(unproject))(x_proj, x_constant, self.origin, self.u)
        return x

    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        chex.assert_rank(x, 3)
        n_nodes, multiplicity, dim = x.shape
        x_proj, x_constant = self.to_invariant_space(x)
        x_proj_out, log_det = self.inner_bijector.forward_and_log_det(x_proj)
        chex.assert_equal_shape((x_proj, x_proj_out))
        x_out = self.to_equivariant_space(x_proj_out, x_constant)

        chex.assert_equal_shape((x_out, x))
        chex.assert_shape(log_det, (n_nodes, multiplicity, 1))
        return x_out, jnp.sum(log_det)

    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        chex.assert_rank(y, 3)
        n_nodes, multiplicity, dim = y.shape
        y_proj, y_constant = self.to_invariant_space(y)
        y_proj_new, log_det = self.inner_bijector.inverse_and_log_det(y_proj)
        y_new = self.to_equivariant_space(y_proj_new, y_constant)

        chex.assert_equal_shape((y_new, y))
        chex.assert_shape(log_det, (n_nodes, multiplicity, 1))
        return y_new, jnp.sum(log_det)
