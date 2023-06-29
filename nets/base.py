from typing import NamedTuple, Optional, Sequence
import chex
import jax.numpy as jnp
import e3nn_jax as e3nn
import haiku as hk
import jax
import warnings

from molboil.utils.graph import get_senders_and_receivers_fully_connected
from molboil.models.base import EquivariantForwardFunction
from molboil.models.e3_gnn import E3GNNTorsoConfig, make_e3nn_torso_forward_fn
from molboil.models.e3gnn_linear_haiku import Linear as e3nnLinear
from molboil.models.en_gnn import make_egnn_torso_forward_fn, EGNNTorsoConfig


from nets.e3_transformer import make_e3transformer_torso_forward_fn, E3TransformerTorsoConfig
from nets.transformer import TransformerConfig

class MLPHeadConfig(NamedTuple):
    mlp_units: Sequence[int]
    stable: bool = True


class NetsConfig(NamedTuple):
    type: str
    egnn_torso_config: Optional[EGNNTorsoConfig] = None
    e3gnn_torso_config: Optional[E3GNNTorsoConfig] = None
    e3transformer_config: Optional[E3TransformerTorsoConfig] = None
    mlp_head_config: Optional[MLPHeadConfig] = None
    non_equivariant_transformer_config: Optional[TransformerConfig] = None
    softmax_layer_invariant_feat: bool = True
    embedding_dim: int = 32
    num_discrete_feat: Optional[int] = None  # E.g. number of atom types
    add_centre_node: bool = True  # Center sender only node.


def build_torso(name: str, config: NetsConfig,
                n_vectors_out: int,
                n_vectors_in: int) -> EquivariantForwardFunction:
    if config.type == 'e3gnn':
        torso = make_e3nn_torso_forward_fn(torso_config=config.e3gnn_torso_config._replace(
            name=name + config.e3gnn_torso_config.name),
        )
    elif config.type == 'egnn':
        assert n_vectors_out % n_vectors_in == 0
        torso = make_egnn_torso_forward_fn(config.egnn_torso_config._replace(
            name=name + config.egnn_torso_config.name,
            n_vectors_hidden_per_vec_in=n_vectors_out // n_vectors_in,
        ),
        )
    elif config.type == "e3transformer":
        torso = make_e3transformer_torso_forward_fn(torso_config=config.e3transformer_config._replace(
            name=name + config.e3transformer_config.name))
    else:
        raise NotImplementedError
    return torso


class EGNN(hk.Module):
    def __init__(
        self,
        name: str,
        nets_config: NetsConfig,
        zero_init_invariant_feat: bool,
        n_invariant_feat_out: int,
        n_equivariant_vectors_out: int,
                  ):
        super().__init__(name="EGNN_module" + name)
        self.name = name
        self.nets_config = nets_config
        self.zero_init_invariant_feat = zero_init_invariant_feat
        self.h_out = n_invariant_feat_out != 0
        self.n_invariant_feat_out = max(1, n_invariant_feat_out)
        self.n_equivariant_vectors_out = n_equivariant_vectors_out


    def call_single(
            self,
            x: chex.Array,
            h: chex.Array,
            senders: chex.Array,
            receivers: chex.Array):
        chex.assert_rank(x, 3)
        chex.assert_rank(h, 2)
        n_nodes, vec_multiplicity_in, dim = x.shape[-3:]
        assert h.shape[0] == x.shape[0]  # n_nodes

        # Create an embedding of the non-positional features.
        chex.assert_axis_dimension(h, 1, 1)
        h = jnp.squeeze(h, axis=-1)
        if self.nets_config.num_discrete_feat is None:
            warnings.warn("No number of discrete node features set.")
            h = hk.get_parameter("embedding", shape=(1,), dtype=float, init=hk.initializers.RandomNormal())
            h = jnp.repeat(h[None, :], n_nodes, axis=0)
        else:
            full_embedding = hk.get_parameter("embedding", shape=(self.nets_config.num_discrete_feat,
                                                                  self.nets_config.embedding_dim,), dtype=float,
                                              init=hk.initializers.RandomNormal())
            h = full_embedding[h]
            chex.assert_shape(h, (n_nodes, self.nets_config.embedding_dim))

        if self.nets_config.add_centre_node:
            h_centre = hk.get_parameter("centre_embedding", shape=(h.shape[-1],), dtype=float,
                                        init=hk.initializers.RandomNormal())
            x_centre = jnp.zeros_like(x[0])
            x = jnp.concatenate([x, x_centre[None]], axis=0)
            h = jnp.concatenate([h, h_centre[None]], axis=0)
            senders = jnp.concatenate([senders, jnp.ones(n_nodes)*n_nodes], axis=0, dtype=int)
            receivers = jnp.concatenate([receivers, jnp.arange(n_nodes)], axis=0, dtype=int)

        # Pass through torso network.
        torso = build_torso(self.name, self.nets_config, self.n_equivariant_vectors_out, vec_multiplicity_in)
        vectors, h = torso(x, h, senders, receivers)

        if self.nets_config.add_centre_node:
            vectors, h = vectors[:-1], h[:-1]

        chex.assert_rank(vectors, 3)
        chex.assert_rank(h, 2)
        n_vectors_torso_out = vectors.shape[-2]

        if n_vectors_torso_out != self.n_equivariant_vectors_out:
            if dim != 3:
                raise Exception("Uses e3nn so only works for dim=3")
            vectors = e3nn.IrrepsArray("1x1o", vectors)
            vectors = vectors.axis_to_mul(axis=-2)  # [n_nodes, n_vectors*dim]

            vectors = e3nnLinear(e3nn.Irreps(f"{self.n_equivariant_vectors_out}x1o"),
                                 biases=True)(vectors)  # [n_nodes, n_equivariant_vectors_out*dim]
            vectors = vectors.mul_to_axis().array

        if self.nets_config.softmax_layer_invariant_feat:
            h = jax.nn.softmax(h, axis=-1)
        final_layer_h = hk.Linear(self.n_invariant_feat_out, w_init=jnp.zeros, b_init=jnp.zeros) \
            if self.zero_init_invariant_feat else hk.Linear(self.n_invariant_feat_out)
        h = final_layer_h(h)
        return vectors, h

    def __call__(
            self,
            positions: chex.Array,
            features: chex.Array
    ):
        chex.assert_type(features, int)
        n_nodes, multiplicity, dim = positions.shape[-3:]
        chex.assert_axis_dimension(features, -2, 1)
        features = jnp.squeeze(features, axis=-2)
        chex.assert_axis_dimension(features, -2, n_nodes)
        senders, receivers = get_senders_and_receivers_fully_connected(n_nodes)
        if len(positions.shape) == 3:
            vectors, scalars = self.call_single(positions, features, senders, receivers)
        else:
            batch_size = positions.shape[0]
            if features.shape[0] != batch_size:
                chex.assert_rank(features, 2)
                features = jnp.repeat(features[None, ...], batch_size, axis=0)
            vectors, scalars = hk.vmap(self.call_single, split_rng=False, in_axes=(0, 0, None, None))(
                positions, features, senders, receivers)
        if self.h_out:
            return vectors, scalars
        else:
            return vectors
