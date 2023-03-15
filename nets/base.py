from typing import NamedTuple, Optional, Sequence

import jax.numpy as jnp

from nets.e3nn_transformer import EnTransformerTorsoConfig, EnTransformer, EnTransformerConfig
from nets.mace import MACETorsoConfig, MACEConfig, MaceNet
from nets.e3_gnn import E3GNNConfig, E3Gnn, E3GNNTorsoConfig
from nets.en_gnn import en_gnn_net, MultiEgnnConfig, EgnnTorsoConfig
from nets.transformer import TransformerConfig

class MLPHeadConfig(NamedTuple):
    mlp_units: Sequence[int]


class NetsConfig(NamedTuple):
    type: str
    mace_torso_config: Optional[MACETorsoConfig] = None
    egnn_torso_config: Optional[EgnnTorsoConfig] = None
    e3gnn_torso_config: Optional[E3GNNTorsoConfig] = None
    e3transformer_lay_config: Optional[EnTransformerTorsoConfig] = None
    transformer_config: Optional[TransformerConfig] = None
    mlp_head_config: Optional[MLPHeadConfig] = None


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

    def egnn_forward(x, h):
        assert h.shape[-3] == x.shape[-3]  # n_nodes
        assert h.shape[-2] == 1  # Currently only have single h multiplicity.
        if (len(x.shape) - 1) == len(h.shape):  # h needs batch size.
            h = jnp.repeat(h[None], x.shape[0], axis=0)

        assert len(x.shape) == len(h.shape)

        if nets_config.type == "mace":
            mace_config = MACEConfig(name=name+"_mace",
                                     torso_config=nets_config.mace_torso_config,
                                     n_vectors_out=n_equivariant_vectors_out,
                                     n_invariant_feat_out=n_invariant_feat_out,
                                     zero_init_invariant_feat=zero_init_invariant_feat)
            x, h = MaceNet(mace_config)(x, h)
            if n_equivariant_vectors_out == 1:
                x = jnp.squeeze(x, axis=-2)
        elif nets_config.type == "egnn":
            egnn_config = MultiEgnnConfig(name=name+"multi_x_egnn",
                                          torso_config=nets_config.egnn_torso_config,
                                          n_invariant_feat_out=n_invariant_feat_out,
                                          n_equivariant_vectors_out=n_equivariant_vectors_out,
                                          invariant_feat_zero_init=zero_init_invariant_feat
                                          )
            x, h = en_gnn_net(egnn_config)(x, h)
        elif nets_config.type == "e3transformer":
            config = EnTransformerConfig(name=name+"e3transformer",
                                         n_vectors_readout=n_equivariant_vectors_out,
                                         n_invariant_feat_readout=n_invariant_feat_out,
                                         zero_init_invariant_feat=zero_init_invariant_feat,
                                         torso_config=nets_config.e3transformer_lay_config)
            x, h = EnTransformer(config)(x, h)
            if n_equivariant_vectors_out == 1:
                x = jnp.squeeze(x, axis=-2)
        elif nets_config.type == "e3gnn":
            config = E3GNNConfig(name=name+"e3gnn",
                                         n_vectors_readout=n_equivariant_vectors_out,
                                         n_invariant_feat_readout=n_invariant_feat_out,
                                         zero_init_invariant_feat=zero_init_invariant_feat,
                                         torso_config=nets_config.e3gnn_torso_config)
            x, h = E3Gnn(config)(x, h)
            if n_equivariant_vectors_out == 1:
                x = jnp.squeeze(x, axis=-2)
        else:
            raise NotImplementedError

        if h_out:
            return x, h
        else:
            return x


    return egnn_forward

