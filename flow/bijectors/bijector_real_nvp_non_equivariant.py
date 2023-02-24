import jax.numpy as jnp
import distrax

from nets.transformer import Transformer
from nets.base import NetsConfig


def make_realnvp(layer_number, dim, swap,
               nets_config: NetsConfig,
               identity_init: bool = True,
               ):
    def bijector_fn(params):
        log_scale, shift = params
        return distrax.ScalarAffine(log_scale=log_scale, shift=shift)

    def conditioner(x):
        transformer_config = nets_config.transformer_config._replace(output_dim=x.shape[-1]*2,
                                                                     zero_init=identity_init)
        permutation_equivariant_fn = Transformer(name=f"layer_{layer_number}_swap{swap}_scale_shift",
                                                 config=transformer_config)
        params = permutation_equivariant_fn(x)
        log_scale, shift = jnp.split(params, 2, axis=-1)
        return (log_scale, shift)

    return distrax.SplitCoupling(
        split_index=dim,
        event_ndims=2,  # [nodes, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-1
    )
