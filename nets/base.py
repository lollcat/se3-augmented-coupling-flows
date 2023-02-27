from typing import NamedTuple, Optional, Sequence

import jax.numpy as jnp

from nets.e3nn_transformer import EnTransformerTorsoConfig, EnTransformer, EnTransformerConfig
from nets.mace import MACETorsoConfig, MACEConfig, MaceNet
from nets.en_gnn import EgnnTorsoConfig, EnGNN, EgnnConfig
from nets.en_gnn_multi_x import multi_se_equivariant_net, MultiEgnnConfig
from nets.transformer import TransformerConfig

class MLPHeadConfig(NamedTuple):
    mlp_units: Sequence[int]


class NetsConfig(NamedTuple):
    type: str
    mace_lay_config: Optional[MACETorsoConfig] = None
    egnn_lay_config: Optional[EgnnTorsoConfig] = None
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
    def egnn_forward(x):
        if nets_config.type == "mace":
            mace_config = MACEConfig(name=name+"_mace",
                                     layer_config=nets_config.mace_lay_config,
                                     n_vectors_readout=n_equivariant_vectors_out,
                                     n_invariant_feat_readout=n_invariant_feat_out,
                                     zero_init_invariant_feat=zero_init_invariant_feat)
            x, h = MaceNet(mace_config)(x)
            if n_equivariant_vectors_out == 1:
                x = jnp.squeeze(x, axis=-2)
        elif nets_config.type == "egnn":
            if n_equivariant_vectors_out == 1:
                egnn_config = EgnnConfig(name=name+"egnn",
                                         torso_config=nets_config.egnn_lay_config,
                                         n_invariant_feat_out=n_invariant_feat_out,
                                         invariant_feat_zero_init=zero_init_invariant_feat)
                x, h = EnGNN(egnn_config)(x)
            else:
                egnn_config = MultiEgnnConfig(name=name+"multi_x_egnn",
                                              torso_config=nets_config.egnn_lay_config,
                                              n_invariant_feat_out=n_invariant_feat_out,
                                              n_equivariant_vectors_out=n_equivariant_vectors_out,
                                              invariant_feat_zero_init=zero_init_invariant_feat
                                              )
                x, h = multi_se_equivariant_net(egnn_config)(x)
        elif nets_config.type == "e3transformer":
            config = EnTransformerConfig(name=name+"multi_x_egnn",
                                         n_vectors_readout=n_equivariant_vectors_out,
                                         n_invariant_feat_readout=n_invariant_feat_out,
                                         zero_init_invariant_feat=zero_init_invariant_feat,
                                         torso_config=nets_config.e3transformer_lay_config)
            x, h = EnTransformer(config)(x)
            if n_equivariant_vectors_out == 1:
                x = jnp.squeeze(x, axis=-2)
        else:
            raise NotImplementedError

        if h_out:
            return x, h
        else:
            return x


    return egnn_forward

