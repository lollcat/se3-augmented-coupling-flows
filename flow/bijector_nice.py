import distrax
import jax.numpy as jnp

from flow.nets import se_equivariant_net


def make_conditioner(equivariant_fn):
    def conditioner(x):
        shift = equivariant_fn(x) - x
        return shift
    return conditioner


def make_se_equivariant_nice(layer_number, dim, swap, identity_init: bool = True, mlp_units=(5, 5)):

    equivariant_fn = se_equivariant_net(name=f"layer_{layer_number}",
                                        zero_init=identity_init,
                                        mlp_units=mlp_units)
    def bijector_fn(shift):
        return distrax.ScalarAffine(log_scale=jnp.zeros_like(shift), shift=shift)

    conditioner = make_conditioner(equivariant_fn)
    return distrax.SplitCoupling(
        split_index=dim,
        event_ndims=2,  # [nodes, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-1
    )
