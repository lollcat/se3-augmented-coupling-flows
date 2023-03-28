import haiku as hk
import jax.numpy as jnp

class ConditionerMLP(hk.Module):
    def __init__(self, name, mlp_units, n_output_params, identity_init):
        super().__init__(name=name)
        self.mlp_function = hk.Sequential(
            [
                hk.LayerNorm(axis=-1, create_offset=True, create_scale=True, param_axis=-1),
                hk.nets.MLP(mlp_units, activate_final=True),
                hk.Linear(n_output_params, b_init=jnp.zeros, w_init=jnp.zeros) if identity_init else
                hk.Linear(n_output_params,
                          b_init=hk.initializers.VarianceScaling(0.01),
                          w_init=hk.initializers.VarianceScaling(0.01))
            ])

    def __call__(self, params):
        return self.mlp_function(params)
