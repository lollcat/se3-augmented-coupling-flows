from typing import NamedTuple, Optional, Callable, Sequence
from functools import partial

import haiku as hk
import jax.numpy as jnp
import jax
from mace_jax import tools
import e3nn_jax as e3nn
from mace_jax.modules.models import LinearNodeEmbeddingBlock, RadialEmbeddingBlock, safe_norm
import chex

from utils.graph import get_senders_and_receivers_fully_connected, e3nn_apply_activation


class EGCL(hk.Module):
    def __init__(self,
                 mlp_units: Sequence[int],
                 n_vectors_hidden: int,
                 n_invariant_feat_hidden: int,
                 activation_fn: Callable = jax.nn.silu,
                 sh_irreps_max_ell: int = 2,
                 residual_h: bool = True
                 ):
        super().__init__()
        self.mlp_units = mlp_units
        self.n_invariant_feat_hidden = n_invariant_feat_hidden
        self.n_vectors_hidden = n_vectors_hidden
        self.activation_fn = activation_fn
        self.residual_h = residual_h

        self.feature_irreps = e3nn.Irreps(f"{n_invariant_feat_hidden}x0e")
        self.vector_irreps = e3nn.Irreps(f"{n_vectors_hidden}x1o")
        self.sh_irreps = e3nn.Irreps.spherical_harmonics(sh_irreps_max_ell)[1:]
        self.sh_harms_fn = partial(
            e3nn.spherical_harmonics,
            self.sh_irreps,
            normalize=False,
            normalization="component")

        self.phi_e = e3nn.haiku.MultiLayerPerceptron(list_neurons=list(self.mlp_units),
                                                               act=self.activation_fn,
                                                               output_activation=self.activation_fn)

        self.phi_inf = e3nn.haiku.Linear(irreps_out=e3nn.Irreps("1x0e"))
        self.phi_x = e3nn.haiku.MultiLayerPerceptron(list_neurons=list(self.mlp_units),
                                                     act=self.activation_fn,
                                                     output_activation=self.activation_fn
                                                     )
        self.phi_h = e3nn.haiku.MultiLayerPerceptron(list_neurons=list(self.mlp_units),
                                                     act=self.activation_fn,
                                                     output_activation=self.activation_fn
                                                     )


    def __call__(self, positions, features):
        n_nodes = positions.shape[0]
        chex.assert_tree_shape_suffix(features, (self.n_invariant_feat_hidden,))
        chex.assert_tree_shape_suffix(positions, (self.n_vectors_hidden, 3))
        senders, receivers = get_senders_and_receivers_fully_connected(positions.shape[0])

        # Prepare the edge attributes.
        vectors = positions[senders] - positions[receivers]
        lengths = safe_norm(vectors, axis=-1)

        scalar_edge_features = e3nn.concatenate([
            e3nn.IrrepsArray(f"{self.n_vectors_hidden}x0e", lengths),
            e3nn.IrrepsArray(f"{self.n_invariant_feat_hidden}x0e", features[senders]),
            e3nn.IrrepsArray(f"{self.n_invariant_feat_hidden}x0e", features[receivers]),
        ]).simplify()

        m_ij = self.phi_e(scalar_edge_features)

        # Get positional output
        phi_x_out = self.phi_x(m_ij)
        sph_harmon = jax.vmap(self.sh_harms_fn, in_axes=-2, out_axes=-2)(vectors / lengths[..., None])
        sph_harmon = sph_harmon.axis_to_mul()
        vector_irreps_array = e3nn.haiku.FullyConnectedTensorProduct(self.vector_irreps)(phi_x_out, sph_harmon)
        vector_per_node = e3nn.scatter_sum(data=vector_irreps_array, dst=senders, output_size=n_nodes)

        # Get feature output
        e = self.phi_inf(m_ij)
        e = e3nn_apply_activation(e, jax.nn.sigmoid)
        m_i = e3nn.scatter_sum(data=e3nn.tensor_product(m_ij, e), dst=senders, output_size=n_nodes)
        phi_h_in = e3nn.concatenate([m_i, features]).simplify()
        phi_h_out = self.phi_h(phi_h_in)
        phi_h_out = e3nn.haiku.Linear(irreps_out=self.feature_irreps)(phi_h_out)

        # Final processing and conversion into plain arrays.
        features_out = phi_h_out.array
        chex.assert_equal_shape((features_out, features))
        if self.residual_h:
            features_out = features_out + features
        shift = vector_per_node.factor_mul_to_last_axis().array
        chex.assert_equal_shape((shift, positions))
        positions_out = positions + shift
        return positions_out, features_out

