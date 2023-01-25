from typing import Tuple

import distrax
import chex
import jax
import jax.numpy as jnp
import haiku as hk

from flow.nets import se_equivariant_net


# TODO: need to figure out how to push all transforms into a single NN, instead of having multiple.


def affine_transform_in_new_space(point, change_of_basis_matrix, origin, scale, shift):
    """Perform affine transformation in the space define by the `origin` and `change_of_basis_matrix`, and then
    go back into the original space."""
    chex.assert_rank(point, 1)
    chex.assert_equal_shape((point, scale, shift))
    point_in_new_space = jnp.linalg.inv(change_of_basis_matrix) @ (point - origin)
    # print(f"\n\n************************\n\n")
    # print(f"change_of_basis_matrix: {change_of_basis_matrix}")
    # print(f"point - origin: {point - origin}")
    # print(f"point in new space: {point_in_new_space}")
    transformed_point_in_new_space = point_in_new_space * scale + shift
    # print(f"transformed point in new space: {transformed_point_in_new_space}")
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
        if len(x.shape) == 2:
            return jax.vmap(affine_transform_in_new_space)(x, self._change_of_basis_matrix, self._origin, self._scale,
                                                       self._shift)
        elif len(x.shape) == 3:
            return jax.vmap(jax.vmap(affine_transform_in_new_space))(x, self._change_of_basis_matrix, self._origin, self._scale,
                                                           self._shift)
        else:
            raise Exception

    def forward_log_det_jacobian(self, x: chex.Array) -> chex.Array:
        """Computes log|det J(f)(x)|."""
        return jnp.sum(self._log_scale, axis=-1)

    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self.forward(x), self.forward_log_det_jacobian(x)

    def inverse(self, y: chex.Array) -> chex.Array:
        """Computes x = f^{-1}(y)."""
        if len(y.shape) == 2:
            return jax.vmap(inverse_affine_transform_in_new_space)(y, self._change_of_basis_matrix, self._origin, self._scale,
                                                    self._shift)
        elif len(y.shape) == 3:
            return jax.vmap(jax.vmap(inverse_affine_transform_in_new_space))(
                y, self._change_of_basis_matrix, self._origin, self._scale, self._shift)
        else:
            raise Exception

    def inverse_log_det_jacobian(self, y: chex.Array) -> chex.Array:
        """Computes log|det J(f^{-1})(y)|."""
        return jnp.sum(jnp.negative(self._log_scale), -1)

    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.inverse(y), self.inverse_log_det_jacobian(y)


def make_conditioner(origin_equivariant_fn,
                     y_equivariant_fn,
                     x_equivariant_fn
                     ):
    def _conditioner(x):
        chex.assert_rank(x, 2)
        dim = x.shape[-1]

        # Calculate new basis for the affine transform
        origin = origin_equivariant_fn(x)
        y_basis_point, y_scale_and_shift_params = y_equivariant_fn(x)
        x_basis_point, x_scale_and_shift_params = x_equivariant_fn(x)

        y_basis_vector = y_basis_point - origin
        # x_basis_vector = x_basis_point - origin
        theta = jnp.pi*0.4
        rotation_matrix = jnp.array(
            [[jnp.cos(theta), -jnp.sin(theta)],
             [jnp.sin(theta), jnp.cos(theta)]]
        )
        x_basis_vector = jax.vmap(jnp.matmul, in_axes=(None, 0))(rotation_matrix, y_basis_vector)

        change_of_basis_matrix = jnp.stack([x_basis_vector, y_basis_vector], axis=-1)

        log_scale = jnp.stack([y_scale_and_shift_params[..., 0], x_scale_and_shift_params[..., 0]], axis=-1)
        shift = jnp.stack([y_scale_and_shift_params[..., 1], x_scale_and_shift_params[..., 1]], axis=-1)

        return change_of_basis_matrix, origin, log_scale, shift

    def conditioner(x):
        if len(x.shape) == 2:
            return _conditioner(x)
        else:
            assert len(x.shape) == 3
            return jax.vmap(_conditioner)(x)

    return conditioner


def make_se_equivariant_split_coupling_with_projection(layer_number, dim, swap, egnn_config,
                                                       identity_init: bool = True):
    assert dim == 2  # Currently just written for 2D

    def bijector_fn(params):
        change_of_basis_matrix, origin, log_scale, shift = params
        return ProjectedScalarAffine(change_of_basis_matrix, origin, log_scale, shift)


    origin_equivariant_fn = se_equivariant_net(
        egnn_config._replace(name=f"layer_{layer_number}_origin",
                           identity_init_x=identity_init,
                           h_out=False))

    y_equivariant_fn = se_equivariant_net(
        egnn_config._replace(name=f"layer_{layer_number}_y",
                           identity_init_x=False,
                           zero_init_h=identity_init,
                           h_out_dim=1,
                           h_out=True
                           ))


    x_equivariant_fn = se_equivariant_net(
        egnn_config._replace(name=f"layer_{layer_number}_x",
                           identity_init_x=False,
                           zero_init_h=identity_init,
                           h_out_dim=1,
                           h_out=True
                           ))



    conditioner = make_conditioner(
        origin_equivariant_fn,
        y_equivariant_fn,
        x_equivariant_fn)

    return distrax.SplitCoupling(
        split_index=dim,
        event_ndims=2,  # [nodes, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-1
    )
