from typing import Callable, Sequence, Optional

import haiku as hk
import jax.nn
import jax.numpy as jnp

class NonLinearLayerWithResidualAndLayerNorm(hk.Module):
    def __init__(self, output_size: int, activation_fn: Callable = jax.nn.silu):
        super().__init__()
        self.output_size = output_size
        self.linear_layer = hk.Linear(self.output_size)
        self.layer_norm = hk.LayerNorm(axis=-1, create_offset=True, create_scale=True, param_axis=-1)
        self.activation_fn = activation_fn

    def __call__(self, x):
        out = self.activation_fn(self.linear_layer(self.layer_norm(x)))
        return out + x


class StableMLP(hk.Module):
    """MLP with layer norm and residual connections."""
    def __init__(self,
                 mlp_units: Sequence[int],
                 activate_final: bool = False,
                 zero_init_output: bool = False,
                 output_variance_scaling: Optional[float] = False,
                 stable_layer: bool = True,
                 activation: Callable = jax.nn.silu,
                 name: Optional[str] = None,
                 ):
        super().__init__(name=name)
        self.activate_final = activate_final
        if not activate_final:
            assert len(mlp_units) > 1, "MLP is single linear layer with no non-linearity"
            n_output_params = mlp_units[-1]
            mlp_units = mlp_units[:-1]
        for i in range(len(mlp_units) - 1):  # Make sure mlp_units have constant width.
            assert mlp_units[i] == mlp_units[i+1]
        if stable_layer:
            layers = [hk.Linear(mlp_units[0]), activation]
            layers.extend([NonLinearLayerWithResidualAndLayerNorm(layer_width, activation_fn=activation)
                           for layer_width in mlp_units[1:]])
            self.mlp_function = hk.Sequential(layers)
        else:
            self.mlp_function = hk.nets.MLP(mlp_units, activate_final=True, activation=activation)

        if zero_init_output or output_variance_scaling:
            assert activate_final is False
        if not activate_final:
            self.final_layer = hk.Linear(n_output_params, b_init=jnp.zeros, w_init=jnp.zeros) if zero_init_output else \
                hk.Linear(n_output_params,
                          w_init=hk.initializers.VarianceScaling(output_variance_scaling, "fan_avg", "uniform")
                          if output_variance_scaling else None)

    def __call__(self, params):
        out = self.mlp_function(params)
        if not self.activate_final:
            out = self.final_layer(out)
        return out
