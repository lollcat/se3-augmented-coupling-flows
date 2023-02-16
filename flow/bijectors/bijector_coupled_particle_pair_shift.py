from typing import Optional, Tuple

import distrax
import jax.numpy as jnp
import haiku as hk
import chex
import jax

def per_particle_shift_forward(x, shift_interp, alt_coords):
    """
    Shift centre of mass along vector between aug-original centre of masses:
    x = (x - alt_coords_global_mean)*centre_shift_interp + alt_coords_global_mean.
    """
    chex.assert_rank(x, 2)
    chex.assert_equal_shape((x, alt_coords))
    x = x * shift_interp + (1 - shift_interp) * alt_coords
    return x

def per_particle_shift_inverse(x, shift_interp, alt_coords):
    chex.assert_rank(x, 2)
    chex.assert_equal_shape((x, alt_coords))
    x = (x - (1 - shift_interp) * alt_coords) / shift_interp
    return x


class ParticlePairShift(distrax.Bijector):
    def __init__(self,
                 log_shift_interp,
                 alt_coords,
                 activation = jax.nn.softplus):
        super().__init__(event_ndims_in=1, is_constant_jacobian=True)
        self.alt_coords = alt_coords
        if activation == jnp.exp:
            self._centre_shift_interp = jnp.exp(log_shift_interp)
            self._log_scale = log_shift_interp
        else:
            assert activation == jax.nn.softplus
            inverse_softplus = lambda x: jnp.log(jnp.exp(x) - 1.)
            log_centre_shift_interp_param = log_shift_interp + inverse_softplus(jnp.array(1.0))
            self._centre_shift_interp = jax.nn.softplus(log_centre_shift_interp_param)

            self._log_scale = jnp.log(jnp.abs(self._centre_shift_interp))

    def forward(self, x: chex.Array) -> chex.Array:
        """Computes y = f(x)."""
        if len(x.shape) == 2:
            return per_particle_shift_forward(x, self._centre_shift_interp,
                                              self.alt_coords)
        elif len(x.shape) == 3:
            return jax.vmap(per_particle_shift_forward)(x, self._centre_shift_interp,
                                                        self.alt_coords)
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
            return per_particle_shift_inverse(y, self._centre_shift_interp,
                                              self.alt_coords)
        elif len(y.shape) == 3:
            return jax.vmap(per_particle_shift_inverse)(y, self._centre_shift_interp,
                                                        self.alt_coords)
        else:
            raise Exception

    def inverse_log_det_jacobian(self, y: chex.Array) -> chex.Array:
        """Computes log|det J(f^{-1})(y)|."""
        return jnp.sum(jnp.negative(self._log_scale), -1)

    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.inverse(y), self.inverse_log_det_jacobian(y)


def make_conditioner(get_shift_fn):
    def conditioner(x):
        shift_interp_logit = get_shift_fn() + jnp.zeros_like(x)
        return shift_interp_logit, x

    return conditioner


def make_per_particle_pair_shift_layer(layer_number, dim, swap, identity_init):
    def bijector_fn(params):
        shift_interp_logit, alt_global_mean = params
        return ParticlePairShift(log_shift_interp=shift_interp_logit,
                                 alt_coords=alt_global_mean)

    get_shift_fn = lambda: hk.get_parameter(name=f'global_shifting_lay{layer_number}_swap{swap}',
                                            shape=(),
                                            init=jnp.zeros if identity_init else hk.initializers.Constant(1.))

    conditioner = make_conditioner(get_shift_fn)
    return distrax.SplitCoupling(
        split_index=dim,
        event_ndims=2,  # [nodes, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=swap,
        split_axis=-1
    )
