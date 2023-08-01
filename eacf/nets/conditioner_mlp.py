from typing import Sequence

import haiku as hk

from eacf.nets.stable_mlp import StableMLP



class ConditionerHead(hk.Module):
    """Used for converting the invariant feat from the EGNN, into the parameters of the bijector transformation
    (e.g. scale and shift params for RealNVP)."""
    def __init__(self, name: str, mlp_units: Sequence[int], n_output_params: int, zero_init: bool,
                 output_variance_scaling: float = 0.1, stable_layer: bool = True):
        super().__init__(name=name)
        self.mlp = StableMLP(mlp_units=(*mlp_units, n_output_params), activate_final=False,
                             output_variance_scaling=output_variance_scaling, stable_layer=stable_layer,
                             zero_init_output=zero_init, layer_norm_inputs=False)

    def __call__(self, params):
        out = self.mlp(params)
        return out
