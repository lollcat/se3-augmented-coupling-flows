from typing import NamedTuple, Optional, Callable, Sequence, Tuple
from functools import partial

import numpy as np
import haiku as hk
import jax.numpy as jnp
import jax
import e3nn_jax as e3nn
from mace_jax.modules.models import safe_norm
import chex

from utils.graph import get_senders_and_receivers_fully_connected, e3nn_apply_activation
from nets.e3gnn_blocks import MessagePassingConvolution, GeneralMLP
from nets.e3gnn_linear_haiku import Linear as e3nnLinear


class EGCL(hk.Module):
    def __init__(self,
                 name: str,
                 mlp_units: Sequence[int],
                 n_vec_hidden_per_vec_in: int,
                 n_invariant_feat_hidden: int,
                 activation_fn: Callable,
                 sh_irreps_max_ell: int,
                 residual_h: bool,
                 residual_x: bool,
                 normalization_constant: float,
                 get_shifts_via_tensor_product: bool,
                 variance_scaling_init: float,
                 vector_scaling_init: float,
                 use_e3nn_haiku: bool
                 ):
        super().__init__(name=name)
        self.vector_scaling_init = vector_scaling_init
        self.variance_scaling_init = variance_scaling_init
        self.mlp_units = mlp_units
        self.n_invariant_feat_hidden = n_invariant_feat_hidden
        self.n_vec_hidden_per_vec_in = n_vec_hidden_per_vec_in
        self.activation_fn = activation_fn
        self.residual_h = residual_h
        self.residual_x = residual_x
        self.normalization_constant = normalization_constant
        self.get_shifts_via_tensor_product = get_shifts_via_tensor_product
        self.use_e3nn_haiku = use_e3nn_haiku

        self.feature_irreps = e3nn.Irreps(f"{n_invariant_feat_hidden}x0e")
        self.sh_irreps = e3nn.Irreps.spherical_harmonics(sh_irreps_max_ell)[1:]
        self.sh_harms_fn = partial(
            e3nn.spherical_harmonics,
            self.sh_irreps,
            normalize=False,
            normalization="component")

        self.phi_e = GeneralMLP(use_e3nn=self.use_e3nn_haiku,
                                output_sizes=(*self.mlp_units, n_invariant_feat_hidden),
                                activation=self.activation_fn, activate_final=False,
                                variance_scaling_init=None)
        self.phi_inf = e3nnLinear(irreps_out=e3nn.Irreps("1x0e"), biases=True) if use_e3nn_haiku else \
            GeneralMLP(use_e3nn=False, output_sizes=(1,), activation=None, activate_final=False,
                       variance_scaling_init=None)
        self.phi_h = GeneralMLP(use_e3nn=use_e3nn_haiku,
                                output_sizes=self.mlp_units, activation=self.activation_fn,
                                activate_final=False, variance_scaling_init=None)
        if not get_shifts_via_tensor_product:
            self.phi_x = GeneralMLP(use_e3nn=use_e3nn_haiku,
                                    output_sizes=self.mlp_units,
                                    activation=self.activation_fn,
                                    activate_final=True,
                                    variance_scaling_init=None)

    def __call__(self, node_vectors, node_features):
        n_nodes, n_vectors, dim = node_vectors.shape
        vector_irreps = e3nn.Irreps(f"{n_vectors}x1o")
        avg_num_neighbours = n_nodes - 1

        chex.assert_tree_shape_suffix(node_features, (self.n_invariant_feat_hidden,))
        chex.assert_tree_shape_suffix(node_vectors, (3,))
        senders, receivers = get_senders_and_receivers_fully_connected(n_nodes)

        # Prepare the edge attributes.
        vectors = node_vectors[receivers] - node_vectors[senders]
        lengths = safe_norm(vectors, axis=-1)

        scalar_edge_features = e3nn.concatenate([
            e3nn.IrrepsArray(f"{n_vectors}x0e", lengths**2),
            e3nn.IrrepsArray(f"{self.n_invariant_feat_hidden}x0e", node_features[senders]),
            e3nn.IrrepsArray(f"{self.n_invariant_feat_hidden}x0e", node_features[receivers]),
        ]).simplify()

        m_ij = self.phi_e(scalar_edge_features)

        # Get positional output
        if self.get_shifts_via_tensor_product:
            sph_harmon = jax.vmap(self.sh_harms_fn, in_axes=-2, out_axes=-2)(
                vectors / (self.normalization_constant + lengths[..., None]))
            sph_harmon = sph_harmon.axis_to_mul()
            vector_per_node = MessagePassingConvolution(avg_num_neighbors=n_nodes-1, target_irreps=vector_irreps,
                                                        activation=self.activation_fn, mlp_units=self.mlp_units,
                                                        use_e3nn_haiku=self.use_e3nn_haiku,
                                                        variance_scaling_init=self.variance_scaling_init)(
                m_ij, sph_harmon, receivers, n_nodes)
            vector_per_node = e3nnLinear(vector_irreps, biases=True)(vector_per_node)
            vectors_out = vector_per_node.factor_mul_to_last_axis().array
        else:
            phi_x_out = self.phi_x(m_ij).array
            phi_x_out = hk.Linear(output_size=n_vectors,
                                  w_init=hk.initializers.VarianceScaling(self.variance_scaling_init))(phi_x_out)
            # Get shifts following the approach from the en gnn paper. Switch to plain haiku for this.
            shifts_ij = phi_x_out[:, :, None] * \
                        vectors / (self.normalization_constant + lengths[:, :, None])
            shifts_i = e3nn.scatter_sum(data=shifts_ij, dst=receivers, output_size=n_nodes)
            vectors_out = shifts_i / avg_num_neighbours
        chex.assert_equal_shape((vectors_out, node_vectors))

        # Get feature output
        e = self.phi_inf(m_ij)
        e = e3nn_apply_activation(e, jax.nn.sigmoid)
        m_i_to_sum = (m_ij.mul_to_axis() * e[:, :, None]).axis_to_mul()
        m_i = e3nn.scatter_sum(data=m_i_to_sum, dst=receivers, output_size=n_nodes) / jnp.sqrt(avg_num_neighbours)
        assert m_i.irreps == e3nn.Irreps(f"{self.n_invariant_feat_hidden}x0e")
        phi_h_in = e3nn.concatenate([m_i, e3nn.IrrepsArray(self.feature_irreps, node_features)]).simplify()
        phi_h_out = self.phi_h(phi_h_in)
        phi_h_out = e3nnLinear(irreps_out=self.feature_irreps, biases=True)(phi_h_out)
        features_out = phi_h_out.array
        chex.assert_equal_shape((features_out, node_features))

        # Final processing and conversion into plain arrays.

        if self.residual_h:
            features_out = features_out + node_features
        if self.residual_x:
            vectors_out = node_vectors + vectors_out
        return vectors_out, features_out



class E3GNNTorsoConfig(NamedTuple):
    n_blocks: int
    mlp_units: Sequence[int]
    n_vec_hidden_per_vec_in: int
    n_invariant_feat_hidden: int
    activation_fn: Callable = jax.nn.silu
    sh_irreps_max_ell: int = 2
    residual_h: bool = True
    residual_x: bool = True
    linear_softmax: bool = True
    get_shifts_via_tensor_product: bool = True
    normalization_constant: float = 1.
    variance_scaling_init: float = 0.001
    vector_scaling_init: float = 0.01
    use_e3nn_haiku: bool = False

    def get_EGCL_kwargs(self):
        kwargs = {}
        kwargs.update(mlp_units=self.mlp_units,
                      n_vec_hidden_per_vec_in=self.n_vec_hidden_per_vec_in,
                      n_invariant_feat_hidden=self.n_invariant_feat_hidden,
                      activation_fn=self.activation_fn,
                      sh_irreps_max_ell=self.sh_irreps_max_ell,
                      residual_h=self.residual_h,
                      residual_x=self.residual_x,
                      get_shifts_via_tensor_product=self.get_shifts_via_tensor_product,
                      normalization_constant=self.normalization_constant,
                      variance_scaling_init=self.variance_scaling_init,
                      vector_scaling_init=self.vector_scaling_init,
                      use_e3nn_haiku=self.use_e3nn_haiku
                      )
        return kwargs

class E3GNNConfig(NamedTuple):
    name: str
    n_invariant_feat_readout: int
    n_vectors_readout: int
    zero_init_invariant_feat: bool
    torso_config: E3GNNTorsoConfig


class E3Gnn(hk.Module):
    def __init__(self, config: E3GNNConfig):
        super().__init__(name=config.name)
        self.config = config
        self.egcl_blocks = [EGCL(config.name + str(i), **config.torso_config.get_EGCL_kwargs())
                                 for i in range(self.config.torso_config.n_blocks)]


    def __call__(self, x: chex.Array, h: chex.Array):
        if len(x.shape) == 3:
            return self.call_single(x, h)
        else:
            chex.assert_rank(x, 4)
            return hk.vmap(self.call_single, split_rng=False)(x, h)

    def call_single(self, x: chex.Array, h: chex.Array) -> Tuple[chex.Array, chex.Array]:
        assert x.shape[0] == h.shape[0]
        n_nodes, multiplicity_in = x.shape[:2]
        # Check that torso is at least as wide as outputs.
        assert self.config.torso_config.n_invariant_feat_hidden >= (multiplicity_in*self.config.n_invariant_feat_readout)
        output_irreps_vector = e3nn.Irreps(f"{self.config.n_vectors_readout*multiplicity_in}x1o")
        output_irreps_scalars = e3nn.Irreps(f"{self.config.n_invariant_feat_readout*multiplicity_in}x0e")

        vectors = x - jnp.mean(x, axis=0, keepdims=True)  # Centre mass

        # Project to number hidden vectors.
        vectors = jnp.repeat(vectors, repeats=self.config.torso_config.n_vec_hidden_per_vec_in, axis=1)
        vectors_in = vectors
        h = h.reshape(h.shape[0], np.prod(h.shape[1:]))  # flatten along last 2 axes.
        h = hk.Linear(self.config.torso_config.n_invariant_feat_hidden)(h)


        for egcl in self.egcl_blocks:
            vectors, h = egcl(vectors, h)

        chex.assert_shape(vectors, (n_nodes, self.config.torso_config.n_vec_hidden_per_vec_in*multiplicity_in, 3))

        # Get vector out.
        if self.config.torso_config.residual_x:
            vectors = vectors - vectors_in
        vectors = e3nn.IrrepsArray('1x1o', vectors)
        vectors = vectors.axis_to_mul(axis=-2)
        assert vectors.irreps == e3nn.Irreps(f"{self.config.torso_config.n_vec_hidden_per_vec_in*multiplicity_in}x1o")
        vectors = e3nnLinear(output_irreps_vector, biases=True)(vectors)
        vectors = vectors.factor_mul_to_last_axis()
        vectors_out = vectors.array
        chex.assert_shape(vectors_out, (n_nodes, self.config.n_vectors_readout*multiplicity_in, 3))
        vectors_out = jnp.reshape(vectors_out, (n_nodes, multiplicity_in, self.config.n_vectors_readout, 3))

        # Get scalar features.
        if self.config.torso_config.linear_softmax:
            h_out = jax.nn.softmax(h, axis=-1)
        else:
            h_out = h
        h_out = hk.Linear(self.config.n_invariant_feat_readout*multiplicity_in,
                          w_init=jnp.zeros if self.config.zero_init_invariant_feat else None)(
            h_out)
        h_out = jnp.reshape(h_out, (n_nodes, multiplicity_in, self.config.n_invariant_feat_readout))
        return vectors_out, h_out
