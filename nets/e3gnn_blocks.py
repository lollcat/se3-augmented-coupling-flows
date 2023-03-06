from typing import Callable, Sequence

import haiku as hk
import e3nn_jax as e3nn
import jax.numpy as jnp

class GeneralMLP(hk.Module):
    def __init__(self, use_e3nn: bool, output_sizes, activation, activate_final):
        super().__init__()
        if use_e3nn:
            self.mlp = e3nn.haiku.MultiLayerPerceptron(list_neurons=list(output_sizes),
                                                       act=activation,
                                                       output_activation=activate_final
                                                       )
        else:
            self.mlp = HaikuMLP(output_sizes, activation, activate_final)
    def __call__(self, x: e3nn.IrrepsArray):
        assert x.irreps.is_scalar()
        return self.mlp(x)


class HaikuMLP(hk.Module):
    """Wrap haiku MLP to work on e3nn.IrrepsArray's.
    Note: Only works on scalars."""
    def __init__(self, output_sizes, activation, activate_final):
        super().__init__()
        self.mlp = hk.nets.MLP(output_sizes=output_sizes,
                               activation=activation,
                               activate_final=activate_final)

    def __call__(self, x: e3nn.IrrepsArray):
        assert x.irreps.is_scalar()
        x_out = self.mlp(x.array)
        irreps_array_out = e3nn.IrrepsArray(irreps=f"{x_out.shape[-1]}x0e", array=x_out)
        return irreps_array_out


class MessagePassingConvolution(hk.Module):
    def __init__(
        self,
        avg_num_neighbors: float,
        target_irreps: e3nn.Irreps,
        activation: Callable,
        mlp_units: Sequence[int],
    ):
        super().__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.activation = activation
        self.mlp_units = list(mlp_units)

    def __call__(
        self,
        edge_feats: e3nn.IrrepsArray,  # [n_edges, irreps]
        edge_attrs: e3nn.IrrepsArray,  # [n_edges, irreps]
        receivers: jnp.ndarray,  # [n_edges, ]
        n_nodes: int,
    ) -> e3nn.IrrepsArray:
        assert edge_feats.ndim == 2
        assert edge_attrs.ndim == 2

        messages = e3nn.tensor_product(
                    edge_feats,
                    edge_attrs,
                    filter_ir_out=self.target_irreps,
                )
        mix = e3nn.haiku.MultiLayerPerceptron(
            self.mlp_units + [messages.irreps.num_irreps],
            self.activation,
            output_activation=False,
        )(
            edge_feats.filter(keep="0e")
        )  # [n_edges, num_irreps]

        messages = messages * mix  # [n_edges, irreps]

        zeros = e3nn.IrrepsArray.zeros(
            messages.irreps, (n_nodes, ), messages.dtype
        )
        node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]

        return node_feats / jnp.sqrt(self.avg_num_neighbors)
