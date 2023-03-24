from typing import NamedTuple, Optional, Sequence, Tuple

import chex
import jax.numpy as jnp
import e3nn_jax as e3nn
import haiku as hk
import jax

from molboil.utils.graph_utils import get_senders_and_receivers_fully_connected
from molboil.models.base import EquivariantForwardFunction
from molboil.models.e3_gnn import E3GNNTorsoConfig, make_e3nn_torso_forward_fn
from molboil.models.e3gnn_linear_haiku import Linear as e3nnLinear

from nets.en_gnn import en_gnn_net, MultiEgnnConfig, EgnnTorsoConfig


class MLPHeadConfig(NamedTuple):
    mlp_units: Sequence[int]


class NetsConfig(NamedTuple):
    type: str
    egnn_torso_config: Optional[EgnnTorsoConfig] = None
    e3gnn_torso_config: Optional[E3GNNTorsoConfig] = None
    mlp_head_config: Optional[MLPHeadConfig] = None


def build_torso(name: str, config: NetsConfig) -> EquivariantForwardFunction:
    if config.type == 'e3gnn':
        torso = make_e3nn_torso_forward_fn(torso_config=config.e3gnn_torso_config._replace(
            name=name + config.e3gnn_torso_config.name))
    else:
        raise NotImplementedError
    return torso


def build_egnn_fn(
        name: str,
        nets_config: NetsConfig,
        zero_init_invariant_feat: bool,
        n_invariant_feat_out: int,
        n_equivariant_vectors_out: int,
                  ):
    """Adds a head to the relevant EGNN to output the desired equivariant vectors & invariant scalars."""
    h_out = n_invariant_feat_out != 0
    n_invariant_feat_out = max(1, n_invariant_feat_out)

    def egnn_forward_single(
            x: chex.Array,
            h: chex.Array,
            senders: chex.Array,
            receivers: chex.Array):
        chex.assert_rank(x, 2)
        chex.assert_rank(h, 2)
        assert h.shape[0] == x.shape[0]  # n_nodes
        torso = build_torso(name, nets_config)
        vectors, h = torso(x, h, senders, receivers)

        if vectors.shape[1] != n_equivariant_vectors_out:
            vectors = e3nn.IrrepsArray("1x1o", vectors)
            vectors = vectors.axis_to_mul(axis=-2)  # [n_nodes, n_vectors*dim]

            vectors = e3nnLinear(f"{n_equivariant_vectors_out}x1o", biases=True)(vectors)  # [n_nodes, 1*dim]
            vectors = vectors.array

        h = hk.Linear(n_invariant_feat_out, w_init=jnp.zeros, b_init=jnp.zeros) \
            if zero_init_invariant_feat else hk.Linear(n_invariant_feat_out)(h)
        return vectors, h

    def egnn_forward(
            x: chex.Array,
            h: chex.Array
    ):
        n_nodes, multiplicity, dim = x.shape[-3:]
        if len(x.shape) == 3:
            vectors, h = egnn_forward_single(x, h, senders, receivers)
        else:
            vectors, h = jax.vmap(egnn_forward_single, in_axes=(0, 0, None, None))(x, h, senders, receivers)
        if h_out:
            return vectors, h
        else:
            return vectors


    return egnn_forward

