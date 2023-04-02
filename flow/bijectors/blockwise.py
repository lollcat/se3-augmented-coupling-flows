from typing import Optional, List, Tuple

import chex
import distrax
import jax.numpy as jnp


class Blockwise(distrax.Bijector):
    def __init__(self,
                 bijectors: List[distrax.Bijector],
                 split_indices: List[int],
                 event_ndims_in: int = 0):
        super().__init__(event_ndims_in=event_ndims_in)
        self.bijectors = bijectors
        self.split_indices = split_indices

    def forward_and_log_det(self, x: chex.Array) -> Tuple[chex.Array, chex.Array]:
        x_split = jnp.split(x, self.split_indices, axis=-1)
        ys = []
        lds = []
        for i, (bijector, x_block) in enumerate(zip(self.bijectors, x_split)):
            y, ld = bijector.forward_and_log_det(x_block)
            lds.append(ld)
            ys.append(y)
        y = jnp.concatenate(ys, axis=-1)
        log_det = jnp.concatenate(lds, axis=-1)
        chex.assert_equal_shape((y, x))
        return y, log_det

    def inverse_and_log_det(self, y: chex.Array) -> Tuple[chex.Array, chex.Array]:
        y_split = jnp.split(y, self.split_indices, axis=-1)
        log_det = jnp.zeros_like(y)
        xs = []
        lds = []
        for i, (bijector, y_block) in enumerate(zip(self.bijectors, y_split)):
            x, ld = bijector.inverse_and_log_det(y_block)
            lds.append(ld)
            xs.append(x)
        x = jnp.concatenate(xs, axis=-1)
        log_det = jnp.concatenate(lds, axis=-1)
        chex.assert_equal_shape((y, x))
        return x, log_det


