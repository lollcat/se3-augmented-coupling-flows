import haiku as hk
import e3nn_jax as e3nn
from flex_mace import FlexMACE
import jax.numpy as jnp


if __name__ == '__main__':


    def forward_fn(x):

        node_specie = jnp.ndarray()
        shared_features: jnp.ndarray, # [dim_shared_features] features shared among all nodes, like time, or total number of atoms in system
        senders: jnp.ndarray,  # [n_edges]
        receivers: jnp.ndarray,  # [n_edges]
        mace_net = FlexMACE(
        output_irreps = e3nn.Irreps("2+0e+4x1o"),
        mace_layer_output_irreps = e3nn.Irreps("4+0e+5x1o"),
        hidden_irreps = e3nn.Irreps("4+0e+5x1o"),
        readout_mlp_irreps = e3nn.Irreps("4+0e+5x1o")
        )
