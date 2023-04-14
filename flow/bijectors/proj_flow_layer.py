from typing import Tuple

import chex
import distrax
import jax


def project(x: chex.Array, origin: chex.Array, change_of_basis_matrix: chex.Array) -> chex.Array:
    chex.assert_rank(x, 1)
    chex.assert_rank(change_of_basis_matrix, 2)
    chex.assert_equal_shape((x, origin, change_of_basis_matrix[0], change_of_basis_matrix[:, 0]))
    return change_of_basis_matrix.T @ (x - origin)

def unproject(x: chex.Array, origin: chex.Array, change_of_basis_matrix: chex.Array) -> chex.Array:
    chex.assert_rank(x, 1)
    chex.assert_rank(change_of_basis_matrix, 2)
    chex.assert_equal_shape((x, origin, change_of_basis_matrix[0], change_of_basis_matrix[:, 0]))
    return change_of_basis_matrix @ x + origin


class ProjFlow(distrax.Bijector):
    def __init__(self,
                 inner_bijector: distrax.Bijector,
                 origin: chex.Array,
                 change_of_basis_matrix: chex.Array
                 ):
        super().__init__(event_ndims_in=0,
                         event_ndims_out=0)
        self.inner_bijector = inner_bijector
        self.origin = origin
        self.change_of_basis_matrix = change_of_basis_matrix

    def to_invariant_space(self, x: chex.Array) -> chex.Array:
        x_proj = jax.vmap(jax.vmap(project))(x, self.origin, self.change_of_basis_matrix)
        return x_proj

    def to_equivariant_space(self, x_proj: chex.Array) -> chex.Array:
        x = jax.vmap(jax.vmap(unproject))(x_proj, self.origin, self.change_of_basis_matrix)
        return x

    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        chex.assert_rank(x, 3)
        n_nodes, multiplicity, dim = x.shape
        x_proj = self.to_invariant_space(x)
        x_proj_out, log_det = self.inner_bijector.forward_and_log_det(x_proj)
        chex.assert_equal_shape((x_proj, x_proj_out))
        x_out = self.to_equivariant_space(x_proj_out)

        chex.assert_equal_shape((x_out, x))
        chex.assert_equal_shape((x_out, log_det))
        return x_out, log_det

    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        chex.assert_rank(y, 3)
        n_nodes, multiplicity, dim = y.shape
        y_proj = self.to_invariant_space(y)
        y_proj_new, log_det = self.inner_bijector.inverse_and_log_det(y_proj)
        y_new = self.to_equivariant_space(y_proj_new)

        chex.assert_equal_shape((y_new, y))
        chex.assert_equal_shape((y_proj_new, log_det))
        return y_new, log_det
