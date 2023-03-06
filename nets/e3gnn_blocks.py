from typing import Callable, Sequence, Optional

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

class HaikuLinear(hk.Module):
    """Wrap haiku Linear to work on e3nn.IrrepsArray's.
    Note: Only works on scalars."""
    def __init__(self, output_size, variance_scaling_init: Optional[float] = None):
        super().__init__()
        self.linear = hk.Linear(output_size,
                                w_init=hk.initializers.VarianceScaling(variance_scaling_init, "fan_avg", "uniform")
                                if variance_scaling_init else None)
    def __call__(self, x: e3nn.IrrepsArray):
        assert x.irreps.is_scalar()
        x_out = self.linear(x.array)
        irreps_array_out = e3nn.IrrepsArray(irreps=f"{x_out.shape[-1]}x0e", array=x_out)
        return irreps_array_out


class MessagePassingConvolution(hk.Module):
    def __init__(
        self,
        avg_num_neighbors: float,
        target_irreps: e3nn.Irreps,
        activation: Callable,
        mlp_units: Sequence[int],
        use_e3nn_haiku: bool,
        variance_scaling_init: Optional[float] = None
    ):
        super().__init__()
        self.avg_num_neighbors = avg_num_neighbors
        self.target_irreps = e3nn.Irreps(target_irreps)
        self.activation = activation
        self.mlp_units = mlp_units
        self.use_e3nn_haiku = use_e3nn_haiku
        self.variance_scaling = variance_scaling_init

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
        mix = GeneralMLP(
            use_e3nn=self.use_e3nn_haiku,
            output_sizes=self.mlp_units,
            activation=self.activation,
            activate_final=True,
        )(
            edge_feats.filter(keep="0e")
        )  # [n_edges, irreps]
        if self.use_e3nn_haiku or self.variance_scaling:
            mix = HaikuLinear(messages.irreps.num_irreps, self.variance_scaling)(mix)
        else:
            mix = e3nn.haiku.Linear(e3nn.Irreps(f"{messages.irreps.num_irreps}x0e"), biases=True)(mix)

        messages = messages * mix  # [n_edges, irreps]

        zeros = e3nn.IrrepsArray.zeros(
            messages.irreps, (n_nodes, ), messages.dtype
        )
        node_feats = zeros.at[receivers].add(messages)  # [n_nodes, irreps]

        return node_feats / jnp.sqrt(self.avg_num_neighbors)
