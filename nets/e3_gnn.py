from typing import NamedTuple, Optional, Callable, Sequence
from functools import partial

import haiku as hk
import jax.numpy as jnp
import jax
import e3nn_jax as e3nn
from mace_jax.modules.models import safe_norm
import chex

from utils.graph import get_senders_and_receivers_fully_connected, e3nn_apply_activation


class EGCL(hk.Module):
    def __init__(self,
                 mlp_units: Sequence[int],
                 n_vectors_hidden: int,
                 n_invariant_feat_hidden: int,
                 activation_fn: Callable = jax.nn.silu,
                 sh_irreps_max_ell: int = 2,
                 residual_h: bool = True,
                 normalization_constant: float = 1.0
                 ):
        super().__init__()
        self.mlp_units = mlp_units
        self.n_invariant_feat_hidden = n_invariant_feat_hidden
        self.n_vectors_hidden = n_vectors_hidden
        self.activation_fn = activation_fn
        self.residual_h = residual_h
        self.normalization_constant = normalization_constant

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


    def __call__(self, node_vectors, node_features):
        n_nodes = node_vectors.shape[0]
        chex.assert_tree_shape_suffix(node_features, (self.n_invariant_feat_hidden,))
        chex.assert_tree_shape_suffix(node_vectors, (self.n_vectors_hidden, 3))
        senders, receivers = get_senders_and_receivers_fully_connected(node_vectors.shape[0])

        # Prepare the edge attributes.
        vectors = node_vectors[senders] - node_vectors[receivers]
        lengths = safe_norm(vectors, axis=-1)

        scalar_edge_features = e3nn.concatenate([
            e3nn.IrrepsArray(f"{self.n_vectors_hidden}x0e", lengths),
            e3nn.IrrepsArray(f"{self.n_invariant_feat_hidden}x0e", node_features[senders]),
            e3nn.IrrepsArray(f"{self.n_invariant_feat_hidden}x0e", node_features[receivers]),
        ]).simplify()

        m_ij = self.phi_e(scalar_edge_features)

        # Get positional output
        phi_x_out = self.phi_x(m_ij)
        sph_harmon = jax.vmap(self.sh_harms_fn, in_axes=-2, out_axes=-2)(
            vectors / (self.normalization_constant + lengths[..., None]))
        sph_harmon = sph_harmon.axis_to_mul()
        vector_irreps_array = e3nn.haiku.FullyConnectedTensorProduct(self.vector_irreps)(phi_x_out, sph_harmon)
        vector_per_node = e3nn.scatter_sum(data=vector_irreps_array, dst=receivers, output_size=n_nodes) / (n_nodes - 1)

        # Get feature output
        e = self.phi_inf(m_ij)
        e = e3nn_apply_activation(e, jax.nn.sigmoid)
        m_i = e3nn.scatter_sum(data=e3nn.tensor_product(m_ij, e, irrep_normalization='norm'),
                               dst=receivers, output_size=n_nodes) / (n_nodes - 1)
        phi_h_in = e3nn.concatenate([m_i, node_features]).simplify()
        phi_h_out = self.phi_h(phi_h_in)
        phi_h_out = e3nn.haiku.Linear(irreps_out=self.feature_irreps)(phi_h_out)

        # Final processing and conversion into plain arrays.
        features_out = phi_h_out.array
        chex.assert_equal_shape((features_out, node_features))
        if self.residual_h:
            features_out = features_out + node_features
        vectors_out = vector_per_node.factor_mul_to_last_axis().array
        chex.assert_equal_shape((vectors_out, node_vectors))
        return vectors_out, features_out



class E3GNNTorsoConfig(NamedTuple):
    n_blocks: int
    mlp_units: Sequence[int]
    n_vectors_hidden: int
    n_invariant_feat_hidden: int
    activation_fn: Callable = jax.nn.silu
    sh_irreps_max_ell: int = 2
    residual_h: bool = True
    linear_softmax: bool = True
    layer_stack: bool = True

    def get_EGCL_kwargs(self):
        kwargs = {}
        kwargs.update(mlp_units=self.mlp_units,
                      n_vectors_hidden=self.n_vectors_hidden,
                      n_invariant_feat_hidden=self.n_invariant_feat_hidden,
                      activation_fn=self.activation_fn,
                      sh_irreps_max_ell=self.sh_irreps_max_ell,
                      residual_h=self.residual_h)
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
        self.egcl_fn = lambda x, h: EGCL(
            **config.torso_config.get_EGCL_kwargs())(x, h)
        self.output_irreps = e3nn.Irreps(f"{config.n_invariant_feat_readout}x0e+"
                                          f"{config.n_vectors_readout}x1o")


    def __call__(self, x):
        if len(x.shape) == 2:
            return self.call_single(x)
        else:
            chex.assert_rank(x, 3)
            return hk.vmap(self.call_single, split_rng=False)(x)

    def call_single(self, x):
        center_of_mass = jnp.mean(x, axis=0, keepdims=True)
        n_nodes = x.shape[0]
        h = jnp.ones((n_nodes, self.config.torso_config.n_invariant_feat_hidden))
        vectors = x - center_of_mass
        vectors = jnp.repeat(vectors[:, None, :], self.config.torso_config.n_vectors_hidden, axis=-2)

        chex.assert_shape(vectors, (n_nodes, self.config.torso_config.n_vectors_hidden, 3))
        if self.config.torso_config.layer_stack:
            stack = hk.experimental.layer_stack(self.config.torso_config.n_blocks, with_per_layer_inputs=False,
                                                name="EGCL_layer_stack")
            vectors, h = stack(self.egcl_fn)(vectors, h)
        else:
            for i in range(self.config.torso_config.n_blocks):
                vectors, h = self.egcl_fn(vectors, h)

        chex.assert_shape(vectors, (n_nodes, self.config.torso_config.n_vectors_hidden, 3))
        # Get vector features for final layer.

        vector_feat = e3nn.IrrepsArray('1x1o', vectors)
        vector_feat = vector_feat.axis_to_mul(axis=-2)
        assert vector_feat.irreps == e3nn.Irreps(f"{self.config.torso_config.n_vectors_hidden}x1o")

        # Get invariant features for final layer and concatenate.
        irreps_h = e3nn.Irreps(f"{self.config.torso_config.n_invariant_feat_hidden}x0e")
        h = e3nn.IrrepsArray(irreps_h, h)
        final_layer_in = e3nn.concatenate([h, vector_feat], axis=-1)

        # Pass through final layer.
        out = e3nn.haiku.Linear(self.output_irreps)(final_layer_in)

        # Get vector features.
        vector_features = out.filter(keep=f"{self.config.n_vectors_readout}x1o")
        vector_features = vector_features.factor_mul_to_last_axis()  # [n_nodes, n_vectors, dim]
        vector_features = vector_features.array
        chex.assert_shape(vector_features, (n_nodes, self.config.n_vectors_readout, 3))

        # Get scalar features.
        invariant_features = out.filter(keep=f"{self.config.n_vectors_readout}x0e")
        if self.config.torso_config.linear_softmax:
            invariant_features = e3nn_apply_activation(invariant_features, partial(jax.nn.softmax, axis=-1))
        invariant_features = hk.Linear(invariant_features.shape[-1],
                                       w_init=jnp.zeros if self.config.zero_init_invariant_feat else None,
                                       )(invariant_features.array)

        positions_out = vector_features + center_of_mass[:, None]
        return positions_out, invariant_features
