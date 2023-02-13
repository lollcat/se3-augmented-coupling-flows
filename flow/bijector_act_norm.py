from typing import Optional, Tuple

import distrax
import jax.numpy as jnp
import haiku as hk
import chex
import jax

def global_scaling_and_shift_forward(x, global_scaling, centre_shift_interp, alt_coords_global_mean):
    """
    Global scaling:
    x = (x - global_mean)*scaling + global_mean -> x = scaling*x + (1 - global_scaling) global_mean
    Shift centre of mass along vector between aug-original centre of masses
    """
    chex.assert_rank(x, 2)
    global_mean = jnp.mean(x, axis=-2, keepdims=True)
    x = x * global_scaling + (1 - global_scaling)*global_mean
    x = x * centre_shift_interp + (1 - centre_shift_interp) * alt_coords_global_mean
    return x

def global_scaling_and_shift_inverse(x, global_scaling, centre_shift_interp, alt_coords_global_mean):
    chex.assert_rank(x, 2)
    x = (x - (1 - centre_shift_interp) * alt_coords_global_mean) / centre_shift_interp

    global_mean = jnp.mean(x, axis=-2, keepdims=True)
    x = (x - (1 - global_scaling)*global_mean) / global_scaling
    return x


class GlobalScaling(distrax.Bijector):
    def __init__(self, log_global_scale,
                 log_centre_shift_interp,
                 alt_coords_global_mean,
                 activation = jnp.exp):
        super().__init__(event_ndims_in=1, is_constant_jacobian=True)
        self.alt_coords_global_mean = alt_coords_global_mean
        if activation == jnp.exp:
            self._global_scale = jnp.exp(log_global_scale)
            self._centre_shift_interp = jnp.exp(log_centre_shift_interp)
            self._log_scale_overall = log_global_scale + log_centre_shift_interp
        else:
            assert activation == jax.nn.softplus
            inverse_softplus = lambda x: jnp.log(jnp.exp(x) - 1.)
            log_scale_global_param = log_global_scale + inverse_softplus(jnp.array(1.0))
            self._global_scale = jax.nn.softplus(log_scale_global_param)
            log_centre_shift_interp_param = log_centre_shift_interp + inverse_softplus(jnp.array(1.0))
            self._centre_shift_interp = jax.nn.softplus(log_centre_shift_interp_param)

            self._log_scale_overall = jnp.log(jnp.abs(self._global_scale)) + jnp.log(jnp.abs(self._centre_shift_interp))

    def forward(self, x: chex.Array) -> chex.Array:
        """Computes y = f(x)."""
        if len(x.shape) == 2:
            return global_scaling_and_shift_forward(x, self._global_scale, self._centre_shift_interp,
                                            self.alt_coords_global_mean)
        elif len(x.shape) == 3:
            return jax.vmap(global_scaling_and_shift_forward)(x, self._global_scale, self._centre_shift_interp,
                                            self.alt_coords_global_mean)
        else:
            raise Exception

    def forward_log_det_jacobian(self, x: chex.Array) -> chex.Array:
        """Computes log|det J(f)(x)|."""
        return jnp.sum(self._log_scale_overall, axis=-1)

    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self.forward(x), self.forward_log_det_jacobian(x)

    def inverse(self, y: chex.Array) -> chex.Array:
        """Computes x = f^{-1}(y)."""
        if len(y.shape) == 2:
            return global_scaling_and_shift_inverse(y, self._global_scale, self._centre_shift_interp,
                                            self.alt_coords_global_mean)
        elif len(y.shape) == 3:
            return jax.vmap(global_scaling_and_shift_inverse)(y, self._global_scale, self._centre_shift_interp,
                                            self.alt_coords_global_mean)
        else:
            raise Exception

    def inverse_log_det_jacobian(self, y: chex.Array) -> chex.Array:
        """Computes log|det J(f^{-1})(y)|."""
        return jnp.sum(jnp.negative(self._log_scale_overall), -1)

    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.inverse(y), self.inverse_log_det_jacobian(y)


def make_conditioner(get_scale_fn, get_shift_fn):
    def conditioner(x):
        scale_logit = get_scale_fn() + jnp.zeros_like(x)
        shift_interp_logit = get_shift_fn() + jnp.zeros_like(x)
        centre_of_mass = jnp.mean(x, axis=-2, keepdims=True)
        return scale_logit, shift_interp_logit, centre_of_mass

    return conditioner


def make_act_norm(layer_number, dim, swap, identity_init):
    def bijector_fn(params):
        global_scale_logit, global_shift_interp_logit, alt_global_mean = params
        return GlobalScaling(log_global_scale=global_scale_logit,
                             log_centre_shift_interp=global_shift_interp_logit,
                             alt_coords_global_mean=alt_global_mean)

    get_scale_fn = lambda: hk.get_parameter(name=f'global_scaling_lay{layer_number}_swap{swap}',
                                            shape=(),
                                            init=jnp.zeros if identity_init else jnp.ones)
    get_shift_fn = lambda: hk.get_parameter(name=f'global_shifting_lay{layer_number}_swap{swap}',
                                            shape=(),
                                            init=jnp.zeros if identity_init else jnp.ones)

    conditioner = make_conditioner(get_scale_fn, get_shift_fn)
    return distrax.SplitCoupling(
        split_index=dim,
        event_ndims=2,  # [nodes, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-1
    )
