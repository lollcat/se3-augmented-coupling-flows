from typing import NamedTuple, Optional, Sequence, Tuple
import chex
import jax.numpy as jnp
import e3nn_jax as e3nn
import haiku as hk
import jax
import numpy as np

from molboil.models.base import EquivariantForwardFunction
from molboil.models.e3_gnn import E3GNNTorsoConfig, make_e3nn_torso_forward_fn
from molboil.models.e3gnn_linear_haiku import Linear as e3nnLinear

from nets.en_gnn import make_egnn_torso_forward_fn, EGNNTorsoConfig
from utils.graph import get_pos_feat_send_receive_flattened_over_multiplicity, unflatten_vectors_scalars


class MLPHeadConfig(NamedTuple):
    mlp_units: Sequence[int]


class NetsConfig(NamedTuple):
    type: str
    egnn_torso_config: Optional[EGNNTorsoConfig] = None
    e3gnn_torso_config: Optional[E3GNNTorsoConfig] = None
    mlp_head_config: Optional[MLPHeadConfig] = None


def build_torso(name: str, config: NetsConfig, n_vectors_out: int) -> EquivariantForwardFunction:
    if config.type == 'e3gnn':
        torso = make_e3nn_torso_forward_fn(torso_config=config.e3gnn_torso_config._replace(
            name=name + config.e3gnn_torso_config.name))
    elif config.type == 'egnn':
        torso = make_egnn_torso_forward_fn(config.egnn_torso_config._replace(
            name=name + config.egnn_torso_config.name,
            multiplicity=n_vectors_out
        ),
        )
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
        torso = build_torso(name, nets_config, n_equivariant_vectors_out)
        vectors, h = torso(x, h, senders, receivers)

        if vectors.shape[1] != n_equivariant_vectors_out:
            if nets_config.type == 'egnn':
                raise Exception("EGNN not configured for this to be different")
            vectors = e3nn.IrrepsArray("1x1o", vectors)
            vectors = vectors.axis_to_mul(axis=-2)  # [n_nodes, n_vectors*dim]

            vectors = e3nnLinear(e3nn.Irreps(f"{n_equivariant_vectors_out}x1o"),
                                 biases=True)(vectors)  # [n_nodes, n_equivariant_vectors_out*dim]
            vectors = vectors.mul_to_axis().array

        h = hk.Linear(n_invariant_feat_out, w_init=jnp.zeros, b_init=jnp.zeros) \
            if zero_init_invariant_feat else hk.Linear(n_invariant_feat_out)(h)
        return vectors, h

    def egnn_forward(
            positions: chex.Array,
            features: chex.Array
    ):
        n_nodes, multiplicity, dim = positions.shape[-3:]
        if features.shape[-2] != multiplicity:
            # Add multiplicity axis, and feature encoding.
            chex.assert_axis_dimension(features, 1, 1)
            multiplicity_encoding = hk.get_parameter(
                f'multiplicity_encoding', shape=(multiplicity,),
                init=hk.initializers.TruncatedNormal(stddev=1. / np.sqrt(multiplicity)))
            features = jnp.concatenate([jnp.concatenate([features,
                                                         jnp.ones((n_nodes, 1, 1))*multiplicity_encoding[i]], axis=-1)
                        for i in range(multiplicity)], axis=1)

        if len(positions.shape) == 3:
            positions_flat, features_flat, senders, receivers = \
                get_pos_feat_send_receive_flattened_over_multiplicity(positions, features)
            vectors, scalars = egnn_forward_single(positions_flat, features_flat, senders, receivers)
            vectors, scalars = unflatten_vectors_scalars(vectors, scalars, n_nodes, multiplicity, dim)
        else:
            batch_size = positions.shape[0]
            if features.shape[0] != batch_size:
                chex.assert_rank(features, 3)
                features = jnp.repeat(features[None, ...], batch_size)

            positions_flat, features_flat, senders, receivers = \
                jax.vmap(get_pos_feat_send_receive_flattened_over_multiplicity)(positions, features)
            vectors, scalars = jax.vmap(egnn_forward_single)(positions_flat, features_flat, senders, receivers)
            vectors, scalars = jax.vmap(unflatten_vectors_scalars, in_axes=(0, 0, None, None, None))(
                vectors, scalars, n_nodes, multiplicity, dim)
        if h_out:
            return vectors, scalars
        else:
            return vectors


    return egnn_forward

