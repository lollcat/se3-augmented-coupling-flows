import jax.numpy as jnp

from utils.graph import get_x_flattened_over_multiplicity_and_senders_recievers


def test_get_x_flattened_over_multiplicity_and_senders_receivers(
        dim: int = 3,
        n_nodes: int = 4,
        multiplicity: int = 5):

    x = jnp.arange(dim*n_nodes*multiplicity)
    x = jnp.reshape(x, (n_nodes, multiplicity, dim))
    x_original = x

    x, senders, receivers = get_x_flattened_over_multiplicity_and_senders_recievers(x)

    assert senders[receivers == 2].shape[0] == (n_nodes-1 + multiplicity-1)

    assert (jnp.reshape(x, (n_nodes, multiplicity, dim)) == x_original).all()


if __name__ == '__main__':
    test_get_x_flattened_over_multiplicity_and_senders_receivers()

