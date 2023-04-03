from typing import Tuple, Callable, Union

import chex
import distrax
import jax
import jax.numpy as jnp

from molboil.utils.numerical import safe_norm
from utils.spherical import to_spherical_and_log_det, to_cartesian_and_log_det
from flow.distrax_with_extra import BijectorWithExtra, Array, BlockWithExtra, Extra

BijectorParams = chex.Array


class SphericalSplitCoupling(BijectorWithExtra):
    def __init__(self,
                 split_index: int,
                 event_ndims: int,
                 graph_features: chex.Array,
                 get_reference_vectors_and_invariant_vals: Callable,
                 bijector: Callable[[BijectorParams], Union[BijectorWithExtra, distrax.Bijector]],
                 swap: bool = False,
                 split_axis: int = -2,
                 use_aux_loss: bool = False,
                 condition_on_x_proj: bool = True
                 ):
        super().__init__(event_ndims_in=event_ndims, is_constant_jacobian=False)
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
        self._split_index = split_index
        self._bijector = bijector
        self._swap = swap
        self._split_axis = split_axis
        self._get_reference_points_and_invariant_vals = get_reference_vectors_and_invariant_vals
        self._graph_features = graph_features
        self.use_aux_loss = use_aux_loss
        self.condition_on_x_proj = condition_on_x_proj
        super().__init__(event_ndims_in=event_ndims)

    def _split(self, x: Array) -> Tuple[Array, Array]:
        x1, x2 = jnp.split(x, [self._split_index], self._split_axis)
        if self._swap:
          x1, x2 = x2, x1
        return x1, x2

    def _recombine(self, x1: Array, x2: Array) -> Array:
        if self._swap:
          x1, x2 = x2, x1
        return jnp.concatenate([x1, x2], self._split_axis)

    def _inner_bijector(self, params: BijectorParams, vector_index: int) -> Union[BijectorWithExtra, distrax.Bijector]:
      """Returns an inner bijector for the passed params."""
      bijector = self._bijector(params, vector_index)
      if bijector.event_ndims_in != bijector.event_ndims_out:
          raise ValueError(
              f'The inner bijector must have `event_ndims_in==event_ndims_out`. '
              f'Instead, it has `event_ndims_in=={bijector.event_ndims_in}` and '
              f'`event_ndims_out=={bijector.event_ndims_out}`.')
      extra_ndims = self.event_ndims_in - bijector.event_ndims_in
      if extra_ndims < 0:
          raise ValueError(
              f'The inner bijector can\'t have more event dimensions than the '
              f'coupling bijector. Got {bijector.event_ndims_in} for the inner '
              f'bijector and {self.event_ndims_in} for the coupling bijector.')
      elif extra_ndims > 0:
          if isinstance(bijector, BijectorWithExtra):
              bijector = BlockWithExtra(bijector, extra_ndims)
          else:
              bijector = distrax.Block(bijector, extra_ndims)
      return bijector

    def get_reference_points_and_h(self, x: chex.Array, graph_features: chex.Array) ->\
            Tuple[chex.Array, chex.Array, Extra]:
        chex.assert_rank(x, 3)
        n_nodes, multiplicity, dim = x.shape

        # Calculate new basis for the affine transform
        reference_vectors, h = self._get_reference_points_and_invariant_vals(x, graph_features)
        reference_points = x[:, :, None, :] + reference_vectors

        bijector_feat_in = jnp.repeat(h[:, None, :], multiplicity, axis=-2)
        if self.condition_on_x_proj:
            # For each set of reference points (for each multiplicity) project all points
            x_sh = jax.vmap(jax.vmap(jax.vmap(to_spherical_and_log_det, in_axes=(0, None)), in_axes=(None, 0))
                            )(x, reference_points)[0]
            x_sh = jnp.reshape(x_sh, (n_nodes, multiplicity, multiplicity*dim))
            bijector_feat_in = jnp.concatenate([bijector_feat_in, x_sh], axis=-1)

        extra = self.get_extra(reference_vectors)
        return reference_points, bijector_feat_in, extra

    def get_vector_info_single(self, basis_vectors: chex.Array) -> Tuple[chex.Array, chex.Array, chex.Array]:
        dim = basis_vectors.shape[-1]
        assert dim == 3
        basis_vectors = basis_vectors + jnp.eye(basis_vectors.shape[-1])[:basis_vectors.shape[1]][None, :, :] * 1e-30
        vec1 = basis_vectors[:, 1]
        vec2 = basis_vectors[:, 2]
        arccos_in = jax.vmap(jnp.dot)(vec1, vec2) / safe_norm(vec1, axis=-1) / safe_norm(vec2, axis=-1)
        theta = jax.vmap(jnp.arccos)(arccos_in)
        log_barrier_in = 1 - jnp.abs(arccos_in)
        log_barrier_in = jnp.where(log_barrier_in < 1e-6, log_barrier_in + 1e-6, log_barrier_in)
        aux_loss = - jnp.log(log_barrier_in)
        return theta, aux_loss, log_barrier_in

    def get_extra(self, various_x_points: chex.Array) -> Extra:
        dim = various_x_points.shape[-1]
        if dim == 2:
            return Extra()
        else:
            theta, aux_loss, log_barrier_in = jax.vmap(self.get_vector_info_single)(various_x_points)
            info = {}
            info_aggregator = {}
            info_aggregator.update(
                mean_abs_theta=jnp.mean, min_abs_theta=jnp.min,
                min_log_barrier_in=jnp.min
            )
            info.update(
                mean_abs_theta=jnp.mean(jnp.abs(theta)), min_abs_theta=jnp.min(jnp.abs(theta)),
                min_log_barrier_in=jnp.min(log_barrier_in)
            )
            if self.use_aux_loss:
                aux_loss = jnp.mean(aux_loss)
            else:
                aux_loss = jnp.array(0.)
            extra = Extra(aux_loss=aux_loss, aux_info=info, info_aggregator=info_aggregator)
            return extra

    def to_spherical_and_log_det(self, x, reference):
        chex.assert_rank(x, 3)
        chex.assert_rank(reference, 4)
        sph_x, log_det = jax.vmap(jax.vmap(to_spherical_and_log_det))(x, reference)
        return sph_x, jnp.sum(log_det)

    def to_cartesian_and_log_det(self, x_sph, reference):
        chex.assert_rank(x_sph, 3)
        chex.assert_rank(reference, 4)
        x, log_det = jax.vmap(jax.vmap(to_cartesian_and_log_det))(x_sph, reference)
        return x, jnp.sum(log_det)


    def forward_and_log_det_single_with_extra(self, x: Array, graph_features: chex.Array) -> Tuple[Array, Array, Extra]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        i = 0  # TODO: multiple reference frames
        self._check_forward_input_shape(x)
        dim = x.shape[-1]
        x1, x2 = self._split(x)
        reference_points, bijector_feat_in, extra = self.get_reference_points_and_h(x1, graph_features)
        n_nodes, multiplicity, n_vectors, dim_ = reference_points.shape
        sph_x_in, log_det_norm_fwd = self.to_spherical_and_log_det(x2, reference_points)
        sph_x_out, logdet_inner_bijector = \
            self._inner_bijector(bijector_feat_in, i).forward_and_log_det(sph_x_in)
        chex.assert_equal_shape((sph_x_out, sph_x_in))
        x2, log_det_norm_rv = self.to_cartesian_and_log_det(sph_x_out, reference_points)
        log_det_total = logdet_inner_bijector + log_det_norm_fwd + log_det_norm_rv
        chex.assert_shape(log_det_total, ())
        y2 = x2
        return self._recombine(x1, y2), log_det_total, extra

    def inverse_and_log_det_single_with_extra(self, y: Array, graph_features: chex.Array) -> Tuple[Array, Array, Extra]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        i = 0  # TODO: multiple reference frames
        self._check_inverse_input_shape(y)
        dim = y.shape[-1]
        y1, y2 = self._split(y)
        reference_points, bijector_feat_in, extra = self.get_reference_points_and_h(y1, graph_features)
        n_nodes, multiplicity, n_vectors, dim_ = reference_points.shape
        sph_y_in, log_det_norm_fwd = self.to_spherical_and_log_det(y2, reference_points)
        ph_y_out, logdet_inner_bijector = self._inner_bijector(bijector_feat_in, i).inverse_and_log_det(sph_y_in)
        y2, log_det_norm_rv = self.to_cartesian_and_log_det(ph_y_out, reference_points)
        log_det_total = logdet_inner_bijector + log_det_norm_fwd + log_det_norm_rv
        chex.assert_shape(log_det_total, ())
        x2 = y2
        return self._recombine(y1, x2), log_det_total, extra

    def forward_and_log_det_single(self, x: Array, graph_features: chex.Array) -> Tuple[Array, Array]:
        return self.forward_and_log_det_single_with_extra(x, graph_features)[:2]

    def inverse_and_log_det_single(self, y: Array, graph_features: chex.Array) -> Tuple[Array, Array]:
        return self.inverse_and_log_det_single_with_extra(y, graph_features)[:2]

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
            h, logdet, extra = self.forward_and_log_det_single_with_extra(x, self._graph_features)
        elif len(x.shape) == 4:
            if self._graph_features.shape[0] != x.shape[0]:
                print("graph features has no batch size")
                h, logdet, extra = jax.vmap(self.forward_and_log_det_single_with_extra, in_axes=(0, None))(
                    x, self._graph_features)
            else:
                h, logdet, extra = jax.vmap(self.forward_and_log_det_single_with_extra)(
                    x, self._graph_features)
            extra = extra._replace(aux_info=extra.aggregate_info(), aux_loss=jnp.mean(extra.aux_loss))
        else:
            raise NotImplementedError
        return h, logdet, extra

    def inverse_and_log_det_with_extra(self, y: Array) -> Tuple[Array, Array, Extra]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        if len(y.shape) == 3:
            x, logdet, extra = self.inverse_and_log_det_single_with_extra(y, self._graph_features)
        elif len(y.shape) == 4:
            if self._graph_features.shape[0] != y.shape[0]:
                print("graph features has no batch size")
                x, logdet, extra = jax.vmap(self.inverse_and_log_det_single_with_extra, in_axes=(0, None))(
                    y, self._graph_features)
            else:
                x, logdet, extra = jax.vmap(self.inverse_and_log_det_single_with_extra)(y, self._graph_features)
            extra = extra._replace(aux_info=extra.aggregate_info(), aux_loss=jnp.mean(extra.aux_loss))
        else:
            raise NotImplementedError
        return x, logdet, extra
