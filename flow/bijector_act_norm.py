import distrax
import jax.numpy as jnp
import haiku as hk


def make_conditioner(get_scale_fn):
    def conditioner(x):
        scale_logit = get_scale_fn() * jnp.ones_like(x)
        return scale_logit

    return conditioner


def make_global_scaling(layer_number, dim, swap):
    def bijector_fn(scale_logit):
        return distrax.ScalarAffine(log_scale=scale_logit, shift=jnp.zeros_like(scale_logit))
    get_scale_fn = lambda: hk.get_parameter(name=f'global_scaling_lay{layer_number}_swap{swap}',
                                            shape=(), init=jnp.zeros)

    conditioner = make_conditioner(get_scale_fn)
    return distrax.SplitCoupling(
        split_index=dim,
        event_ndims=2,  # [nodes, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-1
    )
