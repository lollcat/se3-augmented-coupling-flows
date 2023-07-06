from typing import Callable

import jax
import jax.numpy as jnp
import haiku as hk
import chex

from molboil.utils.test import assert_is_invariant, assert_is_equivariant
from molboil.models.base import EquivariantForwardFunction
from molboil.models.en_gnn import make_egnn_torso_forward_fn, EGNNTorsoConfig
from molboil.utils.graph import get_senders_and_receivers_fully_connected


def make_egnn_torso(
        n_invariant_feat_hidden: int = 5,
        n_vectors_hidden_per_vec_in: int = 2) -> EquivariantForwardFunction:
    config = EGNNTorsoConfig(
        n_blocks=2,
        mlp_units=(2,2),
        n_vectors_hidden_per_vec_in=n_vectors_hidden_per_vec_in,
        n_invariant_feat_hidden=n_invariant_feat_hidden,
        name='e3gnn_torso')
    egnn_torso = make_egnn_torso_forward_fn(config)
    return egnn_torso


def tesst_net_does_not_smoke(
        make_torso: Callable[[int, int], EquivariantForwardFunction],
        n_invariant_feat_hidden: int = 5,
        n_vectors_hidden_per_vec_in: int = 2,
        dim: int = 2,
        n_nodes: int = 5,
        vec_multiplicity_in: int = 2) -> None:
    """Basis test that the egnn doesn't throw an error, and of invariance and equivariances."""

    @hk.without_apply_rng
    @hk.transform
    def forward(positions, features):
        egnn_torso = make_torso(n_invariant_feat_hidden, n_vectors_hidden_per_vec_in)
        senders, receivers = get_senders_and_receivers_fully_connected(n_nodes)
        vectors, scalars = egnn_torso(positions, features, senders, receivers)
        return vectors, scalars


    key = jax.random.PRNGKey(0)
    positions = jax.random.normal(key, (n_nodes, vec_multiplicity_in, dim))
    features = jnp.ones((n_nodes, 2))

    params = forward.init(key, positions, features)

    vectors, scalars = forward.apply(params, positions, features)
    chex.assert_shape(vectors, (n_nodes, vec_multiplicity_in*n_vectors_hidden_per_vec_in, dim))
    chex.assert_shape(scalars, (n_nodes, n_invariant_feat_hidden))

    def invariant_fn(positions: chex.Array) -> chex.Array:
        vectors, scalars = forward.apply(params, positions, features)
        chex.assert_shape(scalars, (n_nodes, n_invariant_feat_hidden))
        return scalars

    def equivariant_fn(positions: chex.Array) -> chex.Array:
        vectors, scalars = forward.apply(params, positions, features)
        chex.assert_shape(vectors, (n_nodes, vec_multiplicity_in * n_vectors_hidden_per_vec_in, dim))
        return vectors

    # Note: features are permutation invariant, so we the test can group action can include permutation.
    assert_is_invariant(invariant_fn=invariant_fn, key=key, event_shape=positions.shape,
                        translate=False)
    assert_is_equivariant(equivariant_fn=equivariant_fn, key=key, event_shape=positions.shape,
                          translate=False, permute=True)
    print('Test passed!')




if __name__ == '__main__':
    USE_64_BIT = True
    if USE_64_BIT:
        from jax.config import config
        config.update("jax_enable_x64", True)

    tesst_net_does_not_smoke(make_egnn_torso)
