from typing import NamedTuple, Callable, Sequence, Tuple
import warnings

import haiku as hk
import jax.numpy as jnp
import jax
import e3nn_jax as e3nn
from mace_jax.modules.models import safe_norm
from e3nn_jax.experimental.transformer import Transformer as E3TransformerLayer
import chex

from molboil.models.e3gnn_blocks import GeneralMLP
from molboil.models.base import EquivariantForwardFunction


class E3TransformerBlock(hk.Module):
    def __init__(
            self,
            name: str,
            mlp_units: Sequence[int],
            num_heads: int,
            n_vectors_hidden: int,
            n_invariant_feat_hidden: int,
            node_feat_as_edge_feat: bool,
            activation_fn: Callable,
            residual_h: bool,
            residual_x: bool,
            variance_scaling_init: float
                 ):
        super().__init__(name=name)
        self.residual_x = residual_x
        self.residual_h = residual_h
        self.num_heads = num_heads
        self.mlp_units = mlp_units
        self.n_vectors_hidden = n_vectors_hidden
        self.n_invariant_feat_hidden = n_invariant_feat_hidden
        self.variance_scaling_init = variance_scaling_init

        # Allows node features to go into edge MLP, similar to egnn.
        self.node_feat_as_edge_feat = node_feat_as_edge_feat
        self.activation_fn = activation_fn
        # TODO: understand what is going on with the heads here.
        assert (n_vectors_hidden % num_heads) == 0
        assert (n_invariant_feat_hidden % num_heads) == 0
        self.feature_irreps = e3nn.Irreps(f"{n_invariant_feat_hidden}x0e")
        self.transformer_fn = E3TransformerLayer(
            irreps_node_output=e3nn.Irreps(f"{self.n_invariant_feat_hidden}x0e+{self.n_vectors_hidden}x1o"),
            list_neurons=list(mlp_units),
            act=activation_fn,
            num_heads=num_heads
        )
        # TODO: Go back to old implementation and add these back.
        # self.bessel_feat = False
        # self.spherical_harmon_feat = False

    def __call__(self, node_positions: chex.Array, node_features: chex.Array, senders: chex.Array,
                 receivers: chex.Array) -> Tuple[chex.Array, chex.Array]:
        chex.assert_rank(node_positions, 3)
        chex.assert_rank(node_features, 2)
        chex.assert_tree_shape_suffix(node_features, (self.n_invariant_feat_hidden,))
        chex.assert_tree_shape_suffix(node_positions, (self.n_vectors_hidden, 3))

        chex.assert_rank(senders, 1)
        chex.assert_equal_shape([senders, receivers])
        n_nodes, n_vectors, dim = node_positions.shape
        avg_num_neighbours = n_nodes - 1
        chex.assert_tree_shape_suffix(node_features, (self.n_invariant_feat_hidden,))


        # Prepare the edge attributes.
        edge_vectors = node_positions[receivers] - node_positions[senders]
        lengths = safe_norm(edge_vectors, axis=-1)
        sq_lengths = lengths ** 2
        scalar_edge_features = e3nn.IrrepsArray(f"{n_vectors}x0e", sq_lengths)

        if self.node_feat_as_edge_feat:
            scalar_edge_features = e3nn.concatenate([
                scalar_edge_features,
                e3nn.IrrepsArray(f"{self.n_invariant_feat_hidden}x0e", node_features[senders]),
                e3nn.IrrepsArray(f"{self.n_invariant_feat_hidden}x0e", node_features[receivers]),
            ]).simplify()
        scalar_edge_features = GeneralMLP(
            output_sizes=self.mlp_units,
            activate_final=False,
            activation=self.activation_fn,
            variance_scaling_init=self.variance_scaling_init,
            use_e3nn=True,
        )(scalar_edge_features)

        vector_irreps = e3nn.IrrepsArray(f"1x1o", edge_vectors)
        vector_irreps = vector_irreps.axis_to_mul()
        assert vector_irreps.irreps == f"{self.n_vectors_hidden}x1o"
        edge_attr = e3nn.concatenate([scalar_edge_features, vector_irreps]).simplify()

        # Setup as fully connected for now.
        edge_weight_cutoff = jnp.ones(edge_vectors.shape[:1])  # e3nn.soft_envelope(vectors, x_max=self.r_max)

        transformer_feat_in = e3nn.IrrepsArray(self.feature_irreps, node_features)
        transformer_feat_in = GeneralMLP(
            output_sizes=self.mlp_units,
            activate_final=False,
            activation=self.activation_fn,
            variance_scaling_init=self.variance_scaling_init,
            use_e3nn=False,
        )(transformer_feat_in)
        # Pass through transformer.
        transformer_out = self.transformer_fn(edge_src=senders, edge_dst=receivers,
                                              edge_weight_cutoff=edge_weight_cutoff,
                                              edge_attr=edge_attr,
                                              node_feat=transformer_feat_in)

        # Turn back into positions and features.
        vector_features = transformer_out.filter(keep=f"{self.n_vectors_hidden}x1o")
        vector_features = vector_features.factor_mul_to_last_axis()  # [n_nodes, n_vectors, dim]
        vectors_out = vector_features.array
        invariant_features = transformer_out.filter(keep=f"{self. n_invariant_feat_hidden}x0e")
        features_out = GeneralMLP(
            output_sizes=(*self.mlp_units, self.n_invariant_feat_hidden),
            activation=self.activation_fn, activate_final=False,
            variance_scaling_init=self.variance_scaling_init,
            use_e3nn=True,
            )(invariant_features)
        features_out = features_out.array

        if self.residual_h:
            features_out = features_out + node_features
        if self.residual_x:
            vectors_out = node_positions + vectors_out
        return vectors_out, features_out


class E3TransformerTorsoConfig(NamedTuple):
    name: str
    n_blocks: int  # number of layers
    mlp_units: Sequence[int]
    n_vectors_hidden_per_vec_in: int
    n_invariant_feat_hidden: int
    activation_fn: Callable = jax.nn.silu
    residual_h: bool = True
    residual_x: bool = True
    node_feat_as_edge_feat: bool = True
    num_heads: int = 1
    variance_scaling_init: float = 0.001
    # centre_mass: bool = True

    def get_transformer_layer_kwargs(self, i: int, vec_multpilicity_in: int):
        kwargs = self._asdict()
        del kwargs["n_blocks"]
        kwargs["name"] = kwargs["name"] + f"_{i}"
        kwargs["n_vectors_hidden"] = (
            vec_multpilicity_in * kwargs["n_vectors_hidden_per_vec_in"]
        )
        del kwargs["n_vectors_hidden_per_vec_in"]
        return kwargs



def make_e3transformer_torso_forward_fn(
    torso_config: E3TransformerTorsoConfig,
) -> EquivariantForwardFunction:
    # TODO
    # if torso_config.centre_mass is False:
    #     warnings.warn("""Note that`centre_mass=False` which will break translational invariance.
    #     This should only be used in special cases, such as when the data is already restricted
    #     to the 0 centre of mass subspace.""")

    def forward_fn(
        positions: chex.Array,
        node_features: chex.Array,
        senders: chex.Array,
        receivers: chex.Array,
    ) -> Tuple[chex.Array, chex.Array]:
        """Returns positions and node features transformed by E3GNN.
            positions: (n_nodes, vec_multiplicity_in * torso_config.n_vectors_hidden_per_vec_in, 3)
            node_features: (n_nodes, n_invariant_feat_hidden)
        """
        chex.assert_rank(positions, 3)
        chex.assert_rank(node_features, 2)
        chex.assert_rank(senders, 1)
        chex.assert_rank(receivers, 1)
        chex.assert_axis_dimension(positions, -1, 3)  # e3nn lib only works for 3D data.

        n_nodes, vec_multiplicity_in, dim = positions.shape

        vectors = positions - positions.mean(axis=0, keepdims=True)

        h = hk.Linear(torso_config.n_invariant_feat_hidden, with_bias=True)(
            node_features
        )

        vectors = jnp.repeat(vectors, torso_config.n_vectors_hidden_per_vec_in, axis=1)
        initial_vectors = vectors

        for i in range(torso_config.n_blocks):
            vectors, h = E3TransformerBlock(**torso_config.get_transformer_layer_kwargs(i, vec_multiplicity_in))(
                vectors, h, senders, receivers
            )

        chex.assert_shape(
            vectors,
            (
                n_nodes,
                vec_multiplicity_in * torso_config.n_vectors_hidden_per_vec_in,
                dim,
            ),
        )
        chex.assert_shape(h, (n_nodes, torso_config.n_invariant_feat_hidden))

        if torso_config.residual_x:
            vectors = vectors - initial_vectors

        return vectors, h

    return forward_fn