from typing import Optional, Tuple

import distrax
import jax.numpy as jnp
import haiku as hk
import chex
import jax

def global_scaling_forward(x, global_scaling):
    """
    Global scaling:
    x = (x - global_mean)*scaling + global_mean -> x = scaling*x + (1 - global_scaling) global_mean
    """
    chex.assert_rank(x, 2)
    global_mean = jnp.mean(x, axis=-2, keepdims=True)
    x = x * global_scaling + (1 - global_scaling)*global_mean
    return x

def global_inverse(x, global_scaling):
    chex.assert_rank(x, 2)
    global_mean = jnp.mean(x, axis=-2, keepdims=True)
    x = (x - (1 - global_scaling)*global_mean) / global_scaling
    return x


class GlobalScaling(distrax.Bijector):
    def __init__(self, log_global_scale,
                 activation = jax.nn.softplus):
        super().__init__(event_ndims_in=1, is_constant_jacobian=True)
        if activation == jnp.exp:
            self._global_scale = jnp.exp(log_global_scale)
            self._log_scale = log_global_scale
        else:
            assert activation == jax.nn.softplus
            inverse_softplus = lambda x: jnp.log(jnp.exp(x) - 1.)
            log_scale_global_param = log_global_scale + inverse_softplus(jnp.array(1.0))
            self._global_scale = jax.nn.softplus(log_scale_global_param)

            self._log_scale = jnp.log(jnp.abs(self._global_scale))

    def forward(self, x: chex.Array) -> chex.Array:
        """Computes y = f(x)."""
        if len(x.shape) == 2:
            return global_scaling_forward(x, self._global_scale)
        elif len(x.shape) == 3:
            return jax.vmap(global_scaling_forward)(x, self._global_scale)
        else:
            raise Exception

    def forward_log_det_jacobian(self, x: chex.Array) -> chex.Array:
        """Computes log|det J(f)(x)|."""
        return jnp.sum(self._log_scale, axis=-1)

    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self.forward(x), self.forward_log_det_jacobian(x)

    def inverse(self, y: chex.Array) -> chex.Array:
        """Computes x = f^{-1}(y)."""
        if len(y.shape) == 2:
            return global_inverse(y, self._global_scale)
        elif len(y.shape) == 3:
            return jax.vmap(global_inverse)(y, self._global_scale)
        else:
            raise Exception

    def inverse_log_det_jacobian(self, y: chex.Array) -> chex.Array:
        """Computes log|det J(f^{-1})(y)|."""
        return jnp.sum(jnp.negative(self._log_scale), -1)

    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.inverse(y), self.inverse_log_det_jacobian(y)


def make_conditioner(get_scale_fn):
    def conditioner(x):
        scale_logit = get_scale_fn() + jnp.zeros_like(x)
        return scale_logit

    return conditioner


def make_coupled_global_scaling(layer_number, dim, swap, identity_init):
    def bijector_fn(params):
        return GlobalScaling(log_global_scale=params)

    get_scale_fn = lambda: hk.get_parameter(name=f'global_scaling_lay{layer_number}_swap{swap}',
                                            shape=(),
                                            init=jnp.zeros if identity_init else jnp.ones)

    conditioner = make_conditioner(get_scale_fn)
    return distrax.SplitCoupling(
        split_index=dim,
        event_ndims=2,  # [nodes, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-1
    )
