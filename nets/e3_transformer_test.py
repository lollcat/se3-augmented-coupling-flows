from nets.e3_transformer import make_e3transformer_torso_forward_fn, E3TransformerTorsoConfig
from molboil.models.torso_tests import tesst_net_does_not_smoke
from molboil.models.base import EquivariantForwardFunction


def make_e3_transformer_torso(
        n_invariant_feat_hidden: int = 5,
        n_vectors_hidden_per_vec_in: int = 2) -> EquivariantForwardFunction:
    config = E3TransformerTorsoConfig(
        n_blocks=2,
        mlp_units=(2,2),
        n_vectors_hidden_per_vec_in=n_vectors_hidden_per_vec_in,
        n_invariant_feat_hidden=n_invariant_feat_hidden,
        name='e3gnn_torso')
    egnn_torso = make_e3transformer_torso_forward_fn(config)
    return egnn_torso


if __name__ == '__main__':
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)
    tesst_net_does_not_smoke(dim=3, make_torso=make_e3_transformer_torso)
