from typing import Tuple

import jax.numpy as jnp
import chex

from eacf.flow.distrax_with_extra import BijectorWithExtra


class AugPermuteBijector(BijectorWithExtra):
    def __init__(self, aug_only: bool = True):
        self.aug_only = aug_only
        super().__init__(event_ndims_in=3, is_constant_jacobian=True)

    def _split(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        x1, x2 = jnp.split(x, [1], -2)
        return x1, x2

    def _recombine(self, x1: chex.Array, x2: chex.Array) -> chex.Array:
        return jnp.concatenate([x1, x2], -2)


    def forward(self, x: chex.Array) -> chex.Array:
        """Computes y = f(x)."""
        if self.aug_only:
            orig, aug = self._split(x)
            aug = jnp.roll(aug, shift=1, axis=-2)
            y = self._recombine(orig, aug)
        else:
            y = jnp.roll(x, shift=1, axis=-2)
        return y
    def forward_log_det_jacobian(self, x: chex.Array) -> chex.Array:
        """Computes log|det J(f)(x)|."""
        n_dim_batch = len(x.shape) - 3
        return jnp.zeros(x.shape[:n_dim_batch])

    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        return self.forward(x), self.forward_log_det_jacobian(x)

    def inverse(self, y: chex.Array) -> chex.Array:
        """Computes x = f^{-1}(y)."""
        if self.aug_only:
            orig, aug = self._split(y)
            aug = jnp.roll(aug, shift=-1, axis=-2)
            x = self._recombine(orig, aug)
        else:
            x = jnp.roll(y, shift=-1, axis=-2)
        return x

    def inverse_log_det_jacobian(self, y: chex.Array) -> chex.Array:
        """Computes log|det J(f^{-1})(y)|."""
        n_dim_batch = len(y.shape) - 3
        return jnp.zeros(y.shape[:n_dim_batch])

    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        return self.inverse(y), self.inverse_log_det_jacobian(y)
