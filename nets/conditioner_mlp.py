import haiku as hk
import jax.numpy as jnp

class ConditionerHead(hk.Module):
    """Used for converting the invariant feat from the EGNN, into the parameters of the bijector transformation
    (e.g. scale and shit params for RealNVP)."""
    def __init__(self, name, mlp_units, n_output_params, zero_init, layer_norm: bool = False):
        super().__init__(name=name)
        mlp_components = [hk.LayerNorm(axis=-1, create_offset=True, create_scale=True, param_axis=-1)]\
            if layer_norm else []
        if len(mlp_components) != 0:
            mlp_components.extend(hk.nets.MLP(mlp_units, activate_final=True))
        mlp_components.extend([
                hk.Linear(n_output_params, b_init=jnp.zeros, w_init=jnp.zeros) if zero_init else
                hk.Linear(n_output_params,
                          b_init=hk.initializers.VarianceScaling(0.01),
                          w_init=hk.initializers.VarianceScaling(0.01))
            ])

        self.mlp_function = hk.Sequential(mlp_components)

    def __call__(self, params):
        return self.mlp_function(params)
