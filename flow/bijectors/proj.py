from typing import Tuple, Optional, Callable, Union

import chex
import distrax
import jax
import jax.numpy as jnp
import haiku as hk

from nets.transformer import Transformer
from nets.base import NetsConfig, build_egnn_fn
from utils.numerical import gram_schmidt_fn, rotate_2d, vector_rejection, safe_norm
from flow.distrax_with_extra import SplitCouplingWithExtra, BijectorWithExtra, Array, Extra, BlockWithExtra

BijectorParams = chex.Array


def project(x, origin, change_of_basis_matrix):
    return change_of_basis_matrix.T @ (x - origin)

def unproject(x, origin, change_of_basis_matrix):
    return change_of_basis_matrix @ x + origin


def get_new_space_basis(x: chex.Array, various_x_vectors: chex.Array, add_small_identity: bool = False):
    n_nodes, dim = x.shape
    # Calculate new basis for the affine transform
    various_x_vectors = jnp.swapaxes(various_x_vectors, 0, 1)

    origin = x + various_x_vectors[0]
    basis_vectors = various_x_vectors[1:]

    if add_small_identity:
        # Add independant vectors to try help improve numerical stability
        basis_vectors = basis_vectors + jnp.eye(x.shape[-1])[:basis_vectors.shape[0]][:, None, :]*1e-6
    chex.assert_tree_shape_suffix(various_x_vectors, (dim, n_nodes, dim))

    z_basis_vector = basis_vectors[0]
    if dim == 3:
        chex.assert_tree_shape_suffix(x, (3,))
        x_basis_vector = basis_vectors[1]
        # Compute reference axes.
        x_basis_vector = vector_rejection(x_basis_vector, z_basis_vector)
        y_basis_vector = jnp.cross(z_basis_vector, x_basis_vector)
        change_of_basis_matrix = jnp.stack([z_basis_vector, x_basis_vector, y_basis_vector], axis=-1)

    else:
        chex.assert_tree_shape_suffix(x, (2,))
        y_basis_vector = rotate_2d(z_basis_vector, theta=jnp.pi * 0.5)
        change_of_basis_matrix = jnp.stack([z_basis_vector, y_basis_vector], axis=-1)

    change_of_basis_matrix = change_of_basis_matrix / safe_norm(change_of_basis_matrix, axis=-2,
                                                                      keepdims=True)
    chex.assert_equal_shape((origin, x))
    chex.assert_shape(change_of_basis_matrix, (n_nodes, dim, dim))
    return origin, change_of_basis_matrix

class ProjSplitCoupling(BijectorWithExtra):
  def __init__(self,
               split_index: int,
               event_ndims: int,
               get_basis_vectors_and_invariant_vals: Callable,
               conditioner: Callable,
               bijector: Callable[[BijectorParams], Union[BijectorWithExtra, distrax.Bijector]],
               swap: bool = False,
               split_axis: int = -1):
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
    self._conditioner = conditioner
    self._bijector = bijector
    self._swap = swap
    self._split_axis = split_axis
    self._get_basis_vectors_and_invariant_vals = get_basis_vectors_and_invariant_vals
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

  def get_basis_and_h(self, x):
      chex.assert_rank(x, 3)
      n_nodes, multiplicity, dim = x.shape

      # Calculate new basis for the affine transform
      various_x_points, h = self._get_basis_vectors_and_invariant_vals(x)
      origin, change_of_basis_matrix = get_new_space_basis

      # Stack h, and x projected into the space.
      inv_change_of_basis = change_of_basis_matrix.T
      x_proj = jax.vmap(project, in_axes=(0, 0, 0))(x, inv_change_of_basis, origin)
      bijector_feat_in = jnp.concatenate([x_proj, h], axis=-1)



  def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
    """Computes y = f(x) and log|det J(f)(x)|."""
    self._check_forward_input_shape(x)
    x1, x2 = self._split(x)
    params = self._conditioner(x1)
    y2, logdet = self._inner_bijector(params).forward_and_log_det(x2)
    return self._recombine(x1, y2), logdet

  def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
    """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
    self._check_inverse_input_shape(y)
    y1, y2 = self._split(y)
    params = self._conditioner(y1)
    x2, logdet = self._inner_bijector(params).inverse_and_log_det(y2)
    return self._recombine(y1, x2), logdet


def make_se_equivariant_split_coupling_with_projection(
        graph_features: chex.Array,
        layer_number: int,
        dim: int,
        n_aug: int,
        swap: bool,
        nets_config: NetsConfig,
        identity_init: bool = True,
        gram_schmidt: bool = False,
        global_frame: bool = False,
        process_flow_params_jointly: bool = False,
        condition_on_x_proj: bool = True,
        add_small_identity: bool = False
        ) -> BijectorWithExtra:
    assert n_aug % 2 == 1
    assert dim in (2, 3)  # Currently just written for 2D and 3D

    def bijector_fn(params):
        change_of_basis_matrix, origin, log_scale, shift, info = params
        return ProjectedScalarAffine(change_of_basis_matrix, origin, log_scale, shift, info)

    n_heads = dim
    n_invariant_params = dim*2

    n_coupling_variable_groups = n_aug + 1
    if nets_config.type == "mace":
        n_invariant_feat_out = int(nets_config.mace_torso_config.n_invariant_feat_residual
                                       / (n_coupling_variable_groups / 2))
    elif nets_config.type == "egnn":
        n_invariant_feat_out = int(nets_config.egnn_torso_config.h_embedding_dim
                                       / (n_coupling_variable_groups / 2))
    elif nets_config.type == 'e3transformer':
        n_invariant_feat_out = int(nets_config.e3transformer_lay_config.n_invariant_feat_hidden
                                       / (n_coupling_variable_groups / 2))
    elif nets_config.type == "e3gnn":
        n_invariant_feat_out = int(nets_config.e3gnn_torso_config.n_invariant_feat_hidden
                                       / (n_coupling_variable_groups / 2))
    else:
        raise NotImplementedError
    equivariant_fn = build_egnn_fn(name=f"layer_{layer_number}_swap{swap}",
                                   nets_config=nets_config,
                                   n_equivariant_vectors_out=n_heads,
                                   n_invariant_feat_out=n_invariant_feat_out,
                                   zero_init_invariant_feat=False)

    if process_flow_params_jointly:
        transformer_config = nets_config.transformer_config._replace(output_dim=n_invariant_params, zero_init=identity_init)
        permutation_equivariant_fn = Transformer(name=f"layer_{layer_number}_swap{swap}_scale_shift",
                                                 config=transformer_config)
        mlp_function = None
    else:
        permutation_equivariant_fn = None
        mlp_function = hk.Sequential([
            hk.LayerNorm(axis=-1, create_offset=True, create_scale=True, param_axis=-1),
            hk.nets.MLP(nets_config.mlp_head_config.mlp_units, activate_final=True),
            hk.Linear(n_invariant_params, b_init=jnp.zeros, w_init=jnp.zeros) if identity_init else
            hk.Linear(n_invariant_params,
                      b_init=hk.initializers.VarianceScaling(0.01),
                      w_init=hk.initializers.VarianceScaling(0.01))
                                      ])

    conditioner = make_conditioner(
        graph_features=graph_features,
        global_frame=global_frame,
        process_flow_params_jointly=process_flow_params_jointly,
        mlp_function=mlp_function,
        multi_x_equivariant_fn=equivariant_fn,
        permutation_equivariant_fn=permutation_equivariant_fn,
        gram_schmidt=gram_schmidt,
        condition_on_x_proj=condition_on_x_proj,
        add_small_identity=add_small_identity
    )

    return SplitCouplingWithExtra(
        split_index=(n_aug + 1) // 2,
        event_ndims=3,  # [nodes, n_aug+1, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-2
    )
