from typing import Sequence

import haiku as hk
import jax.nn
import jax.numpy as jnp

from molboil.models.stable_mlp import StableMLP, NonLinearLayerWithResidualAndLayerNorm



class ConditionerHead(hk.Module):
    """Used for converting the invariant feat from the EGNN, into the parameters of the bijector transformation
    (e.g. scale and shift params for RealNVP)."""
    def __init__(self, name: str, mlp_units: Sequence[int], n_output_params: int, zero_init: bool,
                 output_variance_scaling: float = 0.1, stable_layer: bool = True, v2: bool = True):
        super().__init__(name=name)
        if v2:  # Variance init scaling on final layer of MLP.
            assert len(mlp_units) == 2
            mlp = []
            mlp.append(hk.Linear(mlp_units[0],
                                 w_init=hk.initializers.VarianceScaling(0.1, "fan_avg", "uniform")))
            mlp.append(jax.nn.silu)
            mlp.append(NonLinearLayerWithResidualAndLayerNorm(mlp_units[1]))
            final_layer = hk.Linear(n_output_params, b_init=jnp.zeros, w_init=jnp.zeros) if zero_init else \
                hk.Linear(n_output_params,
                          w_init=hk.initializers.VarianceScaling(output_variance_scaling, "fan_avg", "uniform")
                          if output_variance_scaling else None)
            mlp.append(final_layer)
            self.mlp = hk.Sequential(mlp)
        else:
            self.mlp = StableMLP(mlp_units=(*mlp_units, n_output_params), activate_final=False,
                                 output_variance_scaling=output_variance_scaling, stable_layer=stable_layer,
                                 zero_init_output=zero_init, layer_norm_inputs=False)

    def __call__(self, params):
        out = self.mlp(params)
        return out
