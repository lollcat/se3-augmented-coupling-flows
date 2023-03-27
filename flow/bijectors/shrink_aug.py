from typing import Optional, Tuple, Callable

import jax.numpy as jnp
import haiku as hk
import chex
import jax

from flow.distrax_with_extra import SplitCouplingWithExtra, BijectorWithExtra

def per_particle_shift_forward(a: chex.Array, x: chex.Array, shift_interp: chex.Array) -> chex.Array:
    """
    Shift centre of mass along vector between aug-original centre of masses:
    a = (a - alt_coords_global_mean)*centre_shift_interp + alt_coords_global_mean.
    If shift interp is 1 then this is the identity transform. If shift interp is close to 0, then this moves
    a very close to x.

    The purpose of this flow is to allow the distribution of a (the augmented cooridnates)
    to be at the right scale relative to x (original coordinates).
    """
    chex.assert_rank(a, 1)
    chex.assert_equal_shape((a, x, shift_interp))
    a = a * shift_interp + (1 - shift_interp) * x
    return a

def per_particle_shift_inverse(a: chex.Array, x: chex.Array, shift_interp: chex.Array) -> chex.Array:
    chex.assert_rank(a, 1)
    chex.assert_equal_shape((a, x, shift_interp))
    a = (a - (1 - shift_interp) * x) / shift_interp
    return a


class ParticlePairShift(BijectorWithExtra):
    def __init__(self,
                 log_shift_interp: chex.Array,
                 alt_coords: chex.Array,
                 activation: Callable = jax.nn.softplus):
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
        chex.assert_equal_shape((x, self.alt_coords, self._centre_shift_interp))
        if len(x.shape) == 3:
            y = jax.vmap(jax.vmap(per_particle_shift_forward))(x, self.alt_coords, self._centre_shift_interp)
        elif len(x.shape) == 4:
            y = jax.vmap(jax.vmap(jax.vmap(per_particle_shift_forward)))(x, self.alt_coords, self._centre_shift_interp)
        else:
            raise Exception
        chex.assert_equal_shape((y, x))
        return y

    def forward_log_det_jacobian(self, x: chex.Array) -> chex.Array:
        """Computes log|det J(f)(x)|."""
        return jnp.sum(self._log_scale, axis=-1)

    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self.forward(x), self.forward_log_det_jacobian(x)

    def inverse(self, y: chex.Array) -> chex.Array:
        """Computes x = f^{-1}(y)."""
        chex.assert_equal_shape((y, self.alt_coords, self._centre_shift_interp))
        if len(y.shape) == 3:
            x = jax.vmap(jax.vmap(per_particle_shift_inverse))(y, self.alt_coords, self._centre_shift_interp)
        elif len(y.shape) == 4:
            x = jax.vmap(jax.vmap(jax.vmap(per_particle_shift_inverse)))(y, self.alt_coords, self._centre_shift_interp)
        else:
            raise Exception
        chex.assert_equal_shape((y, x))
        return x

    def inverse_log_det_jacobian(self, y: chex.Array) -> chex.Array:
        """Computes log|det J(f^{-1})(y)|."""
        return jnp.sum(jnp.negative(self._log_scale), -1)

    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.inverse(y), self.inverse_log_det_jacobian(y)


def make_conditioner(get_shift_fn, n_aug: int):
    def conditioner(x):
        shift_interp_logit = get_shift_fn()
        assert x.shape[-2] == 1, "Currently we assume x is passed into conditioner."
        shrinkage_graph_centre = jnp.repeat(x, n_aug, axis=-2)
        expand_dims = [i for i in range(len(x.shape) - 2)] + [-1, ]
        shift_interp_logit = jnp.expand_dims(shift_interp_logit, axis=expand_dims)
        assert len(shift_interp_logit.shape) == len(x.shape)
        shift_interp_logit = jnp.ones_like(x)*shift_interp_logit
        return shift_interp_logit, shrinkage_graph_centre

    return conditioner


def make_shrink_aug_layer(
        graph_features: chex.Array,
        layer_number: int,
        dim: int,
        n_aug: int,
        swap: bool,
        identity_init: bool) -> SplitCouplingWithExtra:
    # TODO: make option to have graph_features used, and for swap to be an option.

    def bijector_fn(params):
        shift_interp_logit, alt_global_mean = params
        return ParticlePairShift(log_shift_interp=shift_interp_logit,
                                 alt_coords=alt_global_mean)

    get_shift_fn = lambda: hk.get_parameter(name=f'global_shifting_lay{layer_number}_swap{swap}',
                                            shape=(n_aug,),
                                            init=jnp.zeros if identity_init else hk.initializers.Constant(1.))

    conditioner = make_conditioner(get_shift_fn, n_aug)
    return SplitCouplingWithExtra(
        split_index=1,
        event_ndims=3,  # [nodes, n_aug, dim]
        conditioner=conditioner,
        bijector=bijector_fn,
        swap=False,
        split_axis=-2
    )
