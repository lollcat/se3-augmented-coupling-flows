from typing import Tuple, Union, Callable

import chex
import distrax
import jax
from distrax._src.bijectors.split_coupling import BijectorParams
import jax.numpy as jnp

from flow.distrax_with_extra import BijectorWithExtra, BlockWithExtra, Extra
from flow.aug_flow_dist import GraphFeatures

Array = chex.Array

class SplitCouplingWithExtra(BijectorWithExtra):
    """Split coupling that obeys centre of mass subspace restriction for the original coordinates
    (non-augmented variables)."""

    def __init__(self,
                 split_index: int,
                 graph_features: GraphFeatures,
                 conditioner: Callable[[GraphFeatures, Array], BijectorParams],
                 bijector: Callable[[BijectorParams], Union[BijectorWithExtra, distrax.Bijector]],
                 swap: bool = False,
                 event_ndims: int = 3,
                 split_axis: int = -2,
                 ):
        super().__init__(event_ndims_in=event_ndims, is_constant_jacobian=False)
        if event_ndims != 3:
            raise NotImplementedError("Only implemented for 3 event ndims")
        if split_axis != -2:
            raise NotImplementedError("Only implemented for split axis on the multiplicity axis (-2")
        self._graph_features = graph_features
        self._conditioner = conditioner
        self._bijector = bijector
        self._swap = swap
        self._split_axis = split_axis
        self._split_index = split_index

    def _split(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        x1, x2 = jnp.split(x, [self._split_index], self._split_axis)
        if self._swap:
            x1, x2 = x2, x1
        return x1, x2

    def _recombine(self, x1: chex.Array, x2: chex.Array) -> chex.Array:
        if self._swap:
          x1, x2 = x2, x1
        return jnp.concatenate([x1, x2], self._split_axis)


    def adjust_centering_pre_proj(self, x: chex.Array) -> chex.Array:
        """x[:, 0, :] is constrained to ZeroMean Gaussian. But if `self._swap` is True, then we
        change this constraint to be with respect to x2[:, 0, :] instead."""
        chex.assert_rank(x, 3)  # [n_nodes, multiplicity, dim]
        if self._swap:
            centre_of_mass = jnp.mean(x[:, self._split_axis], axis=0, keepdims=True)[:, None, :]
        else:
            centre_of_mass = jnp.mean(x[:, 0], axis=0, keepdims=True)[:, None, :]
        return x - centre_of_mass

    def adjust_centering_post_proj(self, x: chex.Array) -> chex.Array:
        """Move centre of mass to be on the first multiplicity again."""
        chex.assert_rank(x, 3)  # [n_nodes, multiplicity, dim]
        centre_of_mass = jnp.mean(x[:, 0], axis=0, keepdims=True)[:, None, :]
        return x - centre_of_mass

    def _inner_bijector(self, params: BijectorParams) -> Union[BijectorWithExtra, distrax.Bijector]:
        """Returns an inner bijector for the passed params."""
        bijector = self._bijector(params)
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

    def forward_and_log_det_with_extra_single(self, x: Array, graph_features: GraphFeatures) -> Tuple[Array, Array, Extra]:
        """Like forward_and_log det, but with additional info. Defaults to just returning an empty dict for extra."""
        self._check_forward_input_shape(x)
        x = self.adjust_centering_post_proj(x)
        x1, x2 = self._split(x)
        params = self._conditioner(x1, graph_features)
        inner_bijector = self._inner_bijector(params)
        if isinstance(inner_bijector, BijectorWithExtra):
            y2, logdet, extra = inner_bijector.forward_and_log_det_with_extra(x2)
        else:
            y2, logdet = inner_bijector.forward_and_log_det(x2)
            extra = Extra()
        y = self._recombine(x1, y2)
        y = self.adjust_centering_post_proj(y)
        return y, logdet, extra

    def inverse_and_log_det_with_extra_single(self, y: Array, graph_features: GraphFeatures) -> Tuple[Array, Array, Extra]:
        """Like inverse_and_log det, but with additional extra. Defaults to just returning an empty dict for extra."""
        self._check_inverse_input_shape(y)
        y = self.adjust_centering_pre_proj(y)
        y1, y2 = self._split(y)
        params = self._conditioner(y1, graph_features)
        inner_bijector = self._inner_bijector(params)
        if isinstance(inner_bijector, BijectorWithExtra):
            x2, logdet, extra = inner_bijector.inverse_and_log_det_with_extra(y2)
        else:
            x2, logdet = inner_bijector.inverse_and_log_det(y2)
            extra = Extra()
        x = self._recombine(y1, x2)
        x = self.adjust_centering_post_proj(x)
        return x, logdet, extra

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