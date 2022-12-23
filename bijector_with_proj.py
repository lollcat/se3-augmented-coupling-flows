from typing import Tuple

import chex
import haiku as hk
import distrax
import chex
import jax
import jax.numpy as jnp



def affine_transform_in_new_space(point, change_of_basis_matrix, origin, scale, shift):
    """Perform affine transformation in the space define by the `origin` and `change_of_basis_matrix`, and then
    go back into the original space."""
    chex.assert_rank(point, 1)
    point_in_new_space = jnp.linalg.inv(change_of_basis_matrix) @ (point - origin)
    transformed_point_in_new_space = point_in_new_space * scale + shift
    new_point_original_space = change_of_basis_matrix @ transformed_point_in_new_space + origin
    return new_point_original_space

def inverse_affine_transform_in_new_space(point, change_of_basis_matrix, origin, scale, shift):
    """Inverse of `affine_transform_in_new_space`."""
    chex.assert_rank(point, 1)
    point_in_new_space = jnp.linalg.inv(change_of_basis_matrix)  @ (point - origin)
    transformed_point_in_new_space = (point_in_new_space - shift) / scale
    new_point_original_space = change_of_basis_matrix @ transformed_point_in_new_space + origin
    return new_point_original_space


class ProjectedScalarAffine(distrax.Bijector):
    """Following style of `ScalarAffine` distrax Bijector.

    Note: Doesn't need to operate on batches, as it gets called with vmap."""
    def __init__(self, change_of_basis_matrix, origin, log_scale, shift):
        super().__init__(event_ndims_in=1, is_constant_jacobian=True)
        self._change_of_basis_matrix = change_of_basis_matrix
        self._origin = origin
        self._scale = jnp.exp(log_scale)
        self._inv_scale = jnp.exp(jnp.negative(log_scale))
        self._log_scale = log_scale
        self._shift = shift

    @property
    def shift(self) -> chex.Array:
        return self._shift

    @property
    def log_scale(self) -> chex.Array:
        return self._log_scale

    @property
    def scale(self) -> chex.Array:
        return self._scale

    @property
    def change_of_basis_matrix(self) -> chex.Array:
        return self._change_of_basis_matrix

    @property
    def origin(self) -> chex.Array:
        return self._origin

    def forward(self, x: chex.Array) -> chex.Array:
        """Computes y = f(x)."""
        return jax.vmap(affine_transform_in_new_space)(x, self._change_of_basis_matrix, self._origin, self._scale,
                                                       self._shift)

    def forward_log_det_jacobian(self, x: chex.Array) -> chex.Array:
        """Computes log|det J(f)(x)|."""
        return jnp.sum(self._log_scale, axis=-1)

    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self.forward(x), self.forward_log_det_jacobian(x)

    def inverse(self, y: chex.Array) -> chex.Array:
        """Computes x = f^{-1}(y)."""
        return jax.vmap(inverse_affine_transform_in_new_space)(y, self._change_of_basis_matrix, self._origin, self._scale,
                                                self._shift)

    def inverse_log_det_jacobian(self, y: chex.Array) -> chex.Array:
        """Computes log|det J(f^{-1})(y)|."""
        return jnp.sum(jnp.negative(self._log_scale), -1)

    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.inverse(y), self.inverse_log_det_jacobian(y)


def equivariant_fn(x):
    chex.assert_rank(x, 2)
    diff_combos = x - x[:, None]  # [n_nodes, n_nodes, dim]
    norms = jnp.linalg.norm(diff_combos, ord=2, axis=-1)
    m = jnp.squeeze(hk.nets.MLP((5, 1), activation=jax.nn.elu)(norms[..., None]), axis=-1) * 3
    return x + jnp.einsum('ijd,ij->id', diff_combos, m)

def invariant_fn(x, n_vals):
    chex.assert_rank(x, 2)
    equivariant_x = jnp.stack([equivariant_fn(x) for _ in range(n_vals)], axis=-1)
    return jnp.linalg.norm(x[..., None] - equivariant_x, ord=2, axis=-2)


def make_conditioner(equivariant_fn=equivariant_fn, invariant_fn=invariant_fn):

    def conditioner(x):
        dim = x.shape[-1]

        # Calculate new basis for the affine transform
        origin = equivariant_fn(x)
        y_basis_point = equivariant_fn(x)
        x_basis_point = equivariant_fn(x)

        y_basis_vector = y_basis_point - origin
        x_basis_vector = x_basis_point - origin
        change_of_basis_matrix = jnp.stack([x_basis_vector, y_basis_vector], axis=-1)

        # Get scale and shift, initialise to be small.
        log_scale = invariant_fn(x, dim)*0.001
        shift = invariant_fn(x, dim) * 0.001

        return change_of_basis_matrix, origin, log_scale, shift

    return conditioner


def make_se_equivariant_split_coupling(dim, swap):

    def bijector_fn(params):
        change_of_basis_matrix, origin, log_scale, shift = params
        return ProjectedScalarAffine(change_of_basis_matrix, origin, log_scale, shift)


    conditioner = make_conditioner()
    return distrax.SplitCoupling(
        split_index=dim,
        event_ndims=2,  # [nodes, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-1
    )


if __name__ == '__main__':
    from test_utils import test_fn_is_equivariant, test_fn_is_invariant
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config

        config.update("jax_enable_x64", True)

    if USE_64_BIT:
        r_tol = 1e-6
    else:
        r_tol = 1e-3

    @hk.without_apply_rng
    @hk.transform
    def bijector_forward(x):
        bijector = make_se_equivariant_split_coupling(dim, swap=False)
        return bijector.forward_and_log_det(x)


    @hk.without_apply_rng
    @hk.transform
    def bijector_backward(x):
        bijector = make_se_equivariant_split_coupling(dim, swap=False)
        return bijector.inverse_and_log_det(x)

    dim = 2
    batch_size = 3
    n_nodes = 4
    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)

    # Create dummy x and a.
    x_and_a = jnp.zeros((n_nodes, dim*2))
    x_and_a = x_and_a + jax.random.normal(subkey, shape=x_and_a.shape)*0.1

    # Initialise bijector parameters.
    params = bijector_forward.init(key, x_and_a)

    # Perform a forward pass.
    x_and_a_new, log_det_fwd = bijector_forward.apply(params, x_and_a)

    # Invert.
    x_and_a_old, log_det_rev = bijector_backward.apply(params, x_and_a_new)

    chex.assert_shape(log_det_fwd, ())
    chex.assert_trees_all_close(x_and_a, x_and_a_old, rtol=r_tol)

    # Test the transformation is equivariant.
    key, subkey = jax.random.split(key)
    test_fn_is_equivariant(lambda x_and_a: bijector_forward.apply(params, x_and_a)[0], subkey)
    key, subkey = jax.random.split(key)
    test_fn_is_equivariant(lambda x_and_a: bijector_backward.apply(params, x_and_a)[0], subkey)

    # Check the change to the log det is invariant
    key, subkey = jax.random.split(key)
    test_fn_is_invariant(lambda x_and_a: bijector_forward.apply(params, x_and_a)[1], subkey)
    key, subkey = jax.random.split(key)
    test_fn_is_invariant(lambda x_and_a: bijector_backward.apply(params, x_and_a)[1], subkey)
