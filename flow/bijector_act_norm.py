from typing import Optional, Tuple

import distrax
import jax.numpy as jnp
import haiku as hk
import chex
import jax

def global_scaling(x, scaling):
    chex.assert_rank(x, 2)
    global_mean = jnp.mean(x)
    return (x - global_mean)*scaling + global_mean

class GlobalScaling(distrax.Bijector):
    def __init__(self, log_scale):
        super().__init__(event_ndims_in=1, is_constant_jacobian=True)
        self._scale = jnp.exp(log_scale)
        self._inv_scale = jnp.exp(jnp.negative(log_scale))
        self._log_scale = log_scale

    @property
    def log_scale(self) -> chex.Array:
        return self._log_scale

    @property
    def scale(self) -> chex.Array:
        return self._scale

    def forward(self, x: chex.Array) -> chex.Array:
        """Computes y = f(x)."""
        if len(x.shape) == 2:
            return global_scaling(x, self._scale)
        elif len(x.shape) == 3:
            return jax.vmap(global_scaling)(x, self._scale)
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
            return global_scaling(y, 1 / self._scale)
        elif len(y.shape) == 3:
            return jax.vmap(global_scaling)(y,  1 / self._scale)
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
        scale_logit = get_scale_fn() * jnp.ones_like(x)
        return scale_logit

    return conditioner


def make_global_scaling(layer_number, dim, swap):
    def bijector_fn(scale_logit):
        return GlobalScaling(log_scale=scale_logit)
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
