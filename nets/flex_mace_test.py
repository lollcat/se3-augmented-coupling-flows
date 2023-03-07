import haiku as hk
import e3nn_jax as e3nn
import jax.random

from flex_mace import FlexMACE
import jax.numpy as jnp
import jax

from utils.graph import get_senders_and_receivers_fully_connected


if __name__ == '__main__':

    @hk.without_apply_rng
    @hk.transform
    def forward_fn(positions):
        n_nodes = positions.shape[0]
        node_specie = jnp.zeros(n_nodes, dtype=int)
        shared_features = jnp.zeros(3)
        senders, receivers = get_senders_and_receivers_fully_connected(n_nodes)

        mace_net = FlexMACE(
        output_irreps = e3nn.Irreps("2x0e+4x1o"),
        mace_layer_output_irreps = e3nn.Irreps("4x0e+5x1o"),
        hidden_irreps = e3nn.Irreps("4x0e+5x1o"),
        readout_mlp_irreps = e3nn.Irreps("4x0e+5x1o"),
        num_features=2,
        avg_num_neighbors=(n_nodes-1)
        )

        mace_net_out = mace_net(
            positions=positions,
            node_specie=node_specie,
            shared_features=shared_features,
            senders=senders,
            receivers=receivers
        )
        return mace_net_out


    key = jax.random.PRNGKey(0)
    key, subkey = jax.random.split(key)
    n_nodes = 5
    dummy_pos = jax.random.normal(key=subkey, shape=(n_nodes, 3))
    params = forward_fn.init(subkey, dummy_pos)

    mace_net_out = forward_fn.apply(params, dummy_pos)