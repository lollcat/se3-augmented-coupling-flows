import jax.numpy as jnp

from eacf.utils.graph import get_pos_feat_send_receive_flattened_over_multiplicity


def test_get_x_flattened_over_multiplicity_and_senders_receivers(
        dim: int = 3,
        n_nodes: int = 4,
        multiplicity: int = 5):

    positions = jnp.arange(dim*n_nodes*multiplicity)
    positions = jnp.reshape(positions, (n_nodes, multiplicity, dim))
    features = jnp.repeat(jnp.arange(multiplicity)[None], n_nodes, axis=0)[:, :, None]
    x_original = positions

    positions, features, senders, receivers = get_pos_feat_send_receive_flattened_over_multiplicity(positions,
                                                                                                    features)

    assert senders[receivers == 2].shape[0] == (n_nodes-1 + multiplicity-1)

    assert (jnp.reshape(positions, (n_nodes, multiplicity, dim)) == x_original).all()


if __name__ == '__main__':
    test_get_x_flattened_over_multiplicity_and_senders_receivers()

