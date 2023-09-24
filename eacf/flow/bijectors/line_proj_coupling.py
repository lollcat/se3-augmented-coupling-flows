from typing import Tuple, Callable, Union

import chex
import distrax
import jax
import jax.numpy as jnp

from eacf.utils.numerical import safe_norm
from eacf.flow.distrax_with_extra import BijectorWithExtra, Array, Extra
from eacf.flow.bijectors.line_proj_flow_layer import LineProjFlow

BijectorParams = chex.Array

class LineSplitCoupling(BijectorWithExtra):
    def __init__(self,
                 split_index: int,
                 graph_features: chex.Array,
                 get_basis_vectors_and_invariant_vals: Callable,
                 bijector: Callable[[BijectorParams, int], Union[BijectorWithExtra, distrax.Bijector]],
                 origin_on_coupled_pair: bool = True,
                 swap: bool = False,
                 n_inner_transforms: int = 1,
                 event_ndims: int = 3,
                 split_axis: int = -2,
                 ):
        super().__init__(event_ndims_in=event_ndims, is_constant_jacobian=False)
        if event_ndims != 3:
            raise NotImplementedError("Only implemented for 3 event ndims")
        if split_axis != -2:
            raise NotImplementedError("Only implemented for split axis on the multiplicity axis (-2")
        if split_index < 0:
          raise ValueError(
              f'The split index must be non-negative; got {split_index}.')
        if split_axis >= 0:
          raise ValueError(f'The split axis must be negative; got {split_axis}.')
        if event_ndims < 0:
          raise ValueError(
              f'`event_ndims` must be non-negative; got {event_ndims}.')
        if split_axis < -event_ndims:
          raise ValueError(
              f'The split axis points to an axis outside the event. With '
              f'`event_ndims == {event_ndims}`, the split axis must be between -1 '
              f'and {-event_ndims}. Got `split_axis == {split_axis}`.')
        self._origin_on_aug = origin_on_coupled_pair
        self._split_index = split_index
        self._bijector = bijector
        self._swap = swap
        self._split_axis = split_axis
        self._get_basis_vectors_and_invariant_vals = get_basis_vectors_and_invariant_vals
        self._graph_features = graph_features
        self.n_inner_transforms = n_inner_transforms
        super().__init__(event_ndims_in=event_ndims)

    def adjust_centering_pre_proj(self, x: chex.Array) -> chex.Array:
        """x[:, 0, :] is constrained to ZeroMean Gaussian. But if `self._swap` is True, then we
        change this constraint to be with respect to x2[:, 0, :] instead."""
        chex.assert_rank(x, 3)  # [n_nodes, multiplicity, dim]
        if self._swap:
            centre_of_mass = jnp.mean(x[:, self._split_index], axis=0, keepdims=True)[:, None, :]
            return x - centre_of_mass
        else:
            return x

    def adjust_centering_post_proj(self, x: chex.Array) -> chex.Array:
        """Move centre of mass to be on the first multiplicity again if `self._swap` is True."""
        chex.assert_rank(x, 3)  # [n_nodes, multiplicity, dim]
        if self._swap:
            centre_of_mass = jnp.mean(x[:, 0], axis=0, keepdims=True)[:, None, :]
            return x - centre_of_mass
        else:
            return x


    def _split(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        x1, x2 = jnp.split(x, [self._split_index], self._split_axis)
        chex.assert_equal_shape((x1, x2))  # Currently assume split always in the middle.
        if self._swap:
            x1, x2 = x2, x1
        return x1, x2

    def _recombine(self, x1: chex.Array, x2: chex.Array) -> chex.Array:
        chex.assert_equal_shape((x1, x2))  # Currently assume split always in the middle.
        if self._swap:
          x1, x2 = x2, x1
        return jnp.concatenate([x1, x2], self._split_axis)

    def _inner_bijector(self, params: BijectorParams, transform_index: int,
                        origin: chex.Array, u: chex.Array) -> \
            Union[BijectorWithExtra, distrax.Bijector]:
      """Returns an inner bijector for the passed params."""
      inner_inner_bijector = self._bijector(params, transform_index)  # e.g. spline or rnvp
      bijector = LineProjFlow(
          inner_bijector=inner_inner_bijector, origin=origin, u=u)
      return bijector

    def get_basis_and_h(self, x: chex.Array, graph_features: chex.Array) ->\
            Tuple[chex.Array, chex.Array, chex.Array, Extra]:
        chex.assert_rank(x, 3)
        n_nodes, multiplicity, dim = x.shape

        # Calculate new basis for the affine transform
        vectors_out, h = self._get_basis_vectors_and_invariant_vals(x, graph_features)
        chex.assert_rank(vectors_out, 5)
        chex.assert_tree_shape_prefix(vectors_out, (n_nodes, multiplicity, self.n_inner_transforms))

        chex.assert_rank(h, 2)
        if self._origin_on_aug:
            chex.assert_shape(vectors_out, (n_nodes, multiplicity, self.n_inner_transforms, 1, dim))
            origin = jnp.repeat(x[:, :, None], self.n_inner_transforms, axis=2)
            vectors = vectors_out[:, :, :, 0]
        else:
            chex.assert_shape(vectors_out, (n_nodes, multiplicity, self.n_inner_transforms, 2, dim))
            origin = x[:, :, None] + vectors_out[:, :, :, 0]
            vectors = vectors_out[:, :, :, 1]

        unit_vectors = vectors / safe_norm(vectors, keepdims=True, axis=-1)

        chex.assert_shape(origin, (n_nodes, multiplicity, self.n_inner_transforms, dim))
        chex.assert_equal_shape((origin, unit_vectors))
        extra = Extra()
        return origin, unit_vectors, h, extra

    def forward_and_log_det_with_extra_single(self, x: Array, graph_features: chex.Array) -> Tuple[Array, Array, Extra]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        chex.assert_rank(x, 3)
        dim = x.shape[-1]

        self._check_forward_input_shape(x)
        x = self.adjust_centering_pre_proj(x)
        x1, x2 = self._split(x)
        origins, unit_vectors, bijector_feat_in, extra = self.get_basis_and_h(x1, graph_features)
        n_nodes, multiplicity, n_transforms, dim = origins.shape
        assert n_transforms == self.n_inner_transforms

        log_det_total = jnp.zeros(())
        for i in range(self.n_inner_transforms):
            origin = origins[:, :, i]
            unit_vector = unit_vectors[:, :, i]
            inner_bijector = self._inner_bijector(bijector_feat_in, i, origin, unit_vector)
            x2, log_det_inner = inner_bijector.forward_and_log_det(x2)
            log_det_total = log_det_total + log_det_inner

        y2 = x2
        y = self._recombine(x1, y2)
        y = self.adjust_centering_post_proj(y)
        return y, log_det_total, extra

    def inverse_and_log_det_with_extra_single(self, y: Array, graph_features: chex.Array) -> Tuple[Array, Array, Extra]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        self._check_inverse_input_shape(y)
        chex.assert_rank(y, 3)
        dim = y.shape[-1]

        y = self.adjust_centering_pre_proj(y)
        y1, y2 = self._split(y)
        origins, unit_vectors, bijector_feat_in, extra = self.get_basis_and_h(y1, graph_features)
        n_nodes, multiplicity, n_transforms, dim = unit_vectors.shape
        assert n_transforms == self.n_inner_transforms

        log_det_total = jnp.zeros(())
        for i in reversed(range(self.n_inner_transforms)):
            origin = origins[:, :, i]
            unit_vector = unit_vectors[:, :, i]
            inner_bijector = self._inner_bijector(bijector_feat_in, i, origin, unit_vector)
            y2, log_det_inner = inner_bijector.inverse_and_log_det(y2)
            log_det_total = log_det_total + log_det_inner

        x2 = y2
        x = self._recombine(y1, x2)
        x = self.adjust_centering_post_proj(x)
        return x, log_det_total, extra

    def forward_and_log_det_single(self, x: Array, graph_features: chex.Array) -> Tuple[Array, Array]:
        return self.forward_and_log_det_with_extra_single(x, graph_features)[:2]

    def inverse_and_log_det_single(self, y: Array, graph_features: chex.Array) -> Tuple[Array, Array]:
        return self.inverse_and_log_det_with_extra_single(y, graph_features)[:2]

    def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        if len(x.shape) == 3:
            return self.forward_and_log_det_single(x, self._graph_features)
        elif len(x.shape) == 4:
            if self._graph_features.shape[0] != x.shape[0]:
                print("graph features has no batch size")
                return jax.vmap(self.forward_and_log_det_single, in_axes=(0, None))(x, self._graph_features)
            else:
                return jax.vmap(self.forward_and_log_det_single)(x, self._graph_features)
        else:
            raise NotImplementedError

    def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        if len(y.shape) == 3:
            return self.inverse_and_log_det_single(y, self._graph_features)
        elif len(y.shape) == 4:
            if self._graph_features.shape[0] != y.shape[0]:
                print("graph features has no batch size")
                return jax.vmap(self.inverse_and_log_det_single, in_axes=(0, None))(y, self._graph_features)
            else:
                return jax.vmap(self.inverse_and_log_det_single)(y, self._graph_features)
        else:
            raise NotImplementedError


    def forward_and_log_det_with_extra(self, x: Array) -> Tuple[Array, Array, Extra]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        if len(x.shape) == 3:
            h, logdet, extra = self.forward_and_log_det_with_extra_single(x, self._graph_features)
        elif len(x.shape) == 4:
            if self._graph_features.shape[0] != x.shape[0]:
                print("graph features has no batch size")
                h, logdet, extra = jax.vmap(self.forward_and_log_det_with_extra_single, in_axes=(0, None))(
                    x, self._graph_features)
            else:
                h, logdet, extra = jax.vmap(self.forward_and_log_det_with_extra_single)(
                    x, self._graph_features)
            extra = extra._replace(aux_info=extra.aggregate_info(), aux_loss=jnp.mean(extra.aux_loss))
        else:
            raise NotImplementedError
        return h, logdet, extra

    def inverse_and_log_det_with_extra(self, y: Array) -> Tuple[Array, Array, Extra]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        if len(y.shape) == 3:
            x, logdet, extra = self.inverse_and_log_det_with_extra_single(y, self._graph_features)
        elif len(y.shape) == 4:
            if self._graph_features.shape[0] != y.shape[0]:
                print("graph features has no batch size")
                x, logdet, extra = jax.vmap(self.inverse_and_log_det_with_extra_single, in_axes=(0, None))(
                    y, self._graph_features)
            else:
                x, logdet, extra = jax.vmap(self.inverse_and_log_det_with_extra_single)(y, self._graph_features)
            extra = extra._replace(aux_info=extra.aggregate_info(), aux_loss=jnp.mean(extra.aux_loss))
        else:
            raise NotImplementedError
        return x, logdet, extra